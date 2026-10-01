import numpy as np
import pytest
import torch
from transformers import BertConfig,BertModel
from nlp_lab.ie.data import convert_record,token_spans
from nlp_lab.ie.encoding import decode_joint
from nlp_lab.ie.metrics import evaluate,paired_interval
from nlp_lab.ie.model import EfficientPointer,pointer_loss,Extractor,joint_objective,pipeline_objective


def test_original_unicode_offsets_and_complex_slots():
    row={'text':'😀张三为电影甲配音，张三也参与制作。','spo_list':[{'subject':'张三','subject_type':'人物','predicate':'配音',
         'object':{'@value':'角色乙','inWork':'电影甲'},'object_type':{'@value':'角色','inWork':'影视作品'}}]}
    lookup={('配音','@value','人物','角色'):0,('配音','inWork','人物','影视作品'):1}
    with pytest.raises(ValueError,match='无法'):convert_record(row,lookup,'train',1)
    row['spo_list'][0]['object']['@value']='张三';row['spo_list'][0]['object_type']['@value']='人物'
    lookup[('配音','@value','人物','人物')]=0
    converted=convert_record(row,lookup,'train',1)
    assert converted['text']==row['text'] and len(converted['triples'])==2
    assert converted['triples'][0][0]==1 and converted['ambiguous_mentions']>=2
    offsets=[(0,0)]+[(i,i+1) for i in range(len(row['text']))]+[(0,0)]
    assert token_spans(offsets,converted)[(1,3,'人物')]==(2,3)


def test_pointer_loss_and_padding_gradient():
    x=torch.tensor([[[[2.,-1.],[-10000.,.4]]]],requires_grad=True)
    y=torch.tensor([[[[1.,0.],[0.,0.]]]])
    actual=pointer_loss(x,y)
    reference=torch.logsumexp(torch.tensor([0.,-2.]),0)+torch.logsumexp(torch.tensor([0.,-1.,.4]),0)
    torch.testing.assert_close(actual,reference)
    actual.backward();assert torch.isfinite(x.grad).all() and x.grad[0,0,1,0]==0
    module=EfficientPointer(16,3,size=8)
    hidden=torch.randn(1,5,16,requires_grad=True);mask=torch.tensor([[False,True,True,True,False]])
    logits=module(hidden,mask)
    assert (logits[:,:,0,:]==-10000).all() and (logits[:,:,3,1]==-10000).all()


def test_tail_links_and_direction_in_joint_decoding():
    offsets=[(0,0),(0,1),(1,2),(2,3),(0,0)]
    schema=[{'id':0,'subject_type':'人物','object_type':'机构'}]
    outputs={'entity':np.full((2,5,5),-10.),'head':np.full((1,5,5),-10.),'tail':np.full((1,5,5),-10.)}
    outputs['entity'][0,1,1]=3;outputs['entity'][1,2,3]=3;outputs['head'][0,1,2]=3
    assert not decode_joint(outputs,offsets,schema,['人物','机构'])['triples']
    assert len(decode_joint(outputs,offsets,schema,['人物','机构'],use_tail=False)['triples'])==1
    outputs['tail'][0,1,3]=3
    pred=decode_joint(outputs,offsets,schema,['人物','机构'])
    assert pred['triples'][0][:7]==(0,1,'人物',0,1,3,'机构')


@pytest.mark.parametrize('architecture',['pipeline','joint'])
def test_real_encoder_extraction_heads_backward_and_reload(tmp_path,architecture):
    from safetensors.torch import save_file,load_file
    torch.manual_seed(7)
    encoder=BertModel(BertConfig(vocab_size=30,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64),add_pooling_layer=False)
    model=Extractor(encoder,2,2,architecture)
    inputs={'input_ids':torch.tensor([[2,4,5,3,0]]),'attention_mask':torch.tensor([[1,1,1,1,0]])}
    mask=torch.tensor([[False,True,True,False,False]])
    pairs=torch.tensor([[[1,1,2,2]]]);outputs=model(inputs,mask,pairs)
    target={'entity':torch.zeros(1,2,5,5)};target['entity'][0,0,1,1]=1;target['entity'][0,1,2,2]=1
    if architecture=='joint':
        target.update(head=torch.zeros(1,2,5,5),tail=torch.zeros(1,2,5,5))
        target['head'][0,0,1,2]=1;target['tail'][0,0,1,2]=1
        loss,_=joint_objective(outputs,target)
    else:
        target.update(pair=torch.tensor([[[1.,0.]]]),pair_mask=torch.ones(1,1,2));loss,_=pipeline_objective(outputs,target)
    loss.backward();assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()>0
    torch.optim.AdamW(model.parameters(),lr=.001).step();model.eval()
    before=model(inputs,mask,pairs)
    save_file(model.state_dict(),str(tmp_path/'model.safetensors'))
    copy=Extractor(BertModel(encoder.config,add_pooling_layer=False),2,2,architecture).eval()
    copy.load_state_dict(load_file(str(tmp_path/'model.safetensors')))
    after=copy(inputs,mask,pairs)
    for name in ('entity','head','tail') if architecture=='joint' else ('entity','pair'):
        torch.testing.assert_close(before[name],after[name])


def test_typed_relation_metric_is_surface_based_and_bootstrap_paired():
    row={'text':'甲和乙，甲','triples':[[0,1,'人物',0,2,3,'机构']]}
    schema={'schemas':[{'id':0,'label':'任职'}]}
    pred={'entities':[(4,5,'人物',3),(2,3,'机构',3)],'triples':[(4,5,'人物',0,2,3,'机构',3)]}
    m,_=evaluate([row],[pred],schema)
    assert m['relation']['f1']==1 and m['entity']['f1']==.5
    counts=[{'correct':1,'predicted':2,'gold':2}]*5
    interval=paired_interval(counts,counts,20)
    assert interval['ci95']==[0.,0.]


def test_unicode_long_window_merge_keeps_global_offsets(tmp_path):
    from transformers import BertTokenizer
    from nlp_lab.ie.predict import Predictor
    vocabulary=tmp_path/'vocab.txt'
    vocabulary.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n字\n张\n三\n在\n北\n京\n工\n作\n。\n',encoding='utf-8')
    tokenizer=BertTokenizer(vocab=str(vocabulary))
    predictor=Predictor.__new__(Predictor)
    predictor.tokenizer=tokenizer;predictor.config={'max_length':64}
    predictor.thresholds={'entity':0.,'relation':0.}
    predictor.manifest={'schemas':[{'id':0,'predicate':'任职地','slot':'@value','label':'任职地'}]}
    def window_predictions(rows):
        predictions=[]
        for row in rows:
            text=row['text'];a=text.find('张三');b=text.find('北京')
            predictions.append({'entities':[(a,a+2,'人物',2),(b,b+2,'地点',3)] if a>=0 and b>=0 else [],
                                'triples':[(a,a+2,'人物',0,b,b+2,'地点',2)] if a>=0 and b>=0 else [],
                                'candidates_capped':False})
        return predictions
    predictor.predict_rows=window_predictions
    text='字'*90+'😀张三在北京工作。'+'字'*80
    result=predictor.extract(text)
    assert result['windows']>1 and len(result['entities'])==2 and len(result['relations'])==1
    entity=next(e for e in result['entities'] if e['text']=='张三')
    assert entity['start']==91 and text[entity['start']:entity['end']]=='张三'
    assert result['relations'][0]['subject']==entity['id']


def test_ie_split_hashes_and_overlap_are_rejected(tmp_path):
    import hashlib,json
    from nlp_lab.data import save_json
    from nlp_lab.ie.data import load,digest
    schema={'id':0,'subject_type':'人物','object_type':'机构'}
    for split,text in [('train','甲乙'),('validation','甲丙'),('test','甲丁')]:
        (tmp_path/(split+'.jsonl')).write_text(json.dumps({'text':text,'triples':[[0,1,'人物',0,1,2,'机构']]})+'\n',encoding='utf-8')
    def manifest():
        hashes={s:digest(tmp_path/(s+'.jsonl')) for s in ('train','validation','test')}
        save_json(tmp_path/'manifest.json',{'schemas':[schema],'split_sha256':hashes,
                  'dataset_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()})
    manifest();assert len(load(tmp_path)[1]['train'])==1
    (tmp_path/'test.jsonl').write_bytes((tmp_path/'train.jsonl').read_bytes())
    with pytest.raises(ValueError,match='指纹'):load(tmp_path)
    manifest()
    with pytest.raises(ValueError,match='重复'):load(tmp_path)
