"""Tiny offline encoder verifies training semantics, never task quality."""
import hashlib
import json
import pytest
from nlp_lab.data import read_json,save_json
from nlp_lab.ie.data import load,digest
from nlp_lab.ie.train import Config,allocate,fit
from nlp_lab.ie.predict import Predictor
from nlp_lab.ie.metrics import evaluate


@pytest.mark.parametrize('architecture',['pipeline','joint'])
def test_ie_fit_sample_accumulation_and_saved_best_reload(tmp_path,architecture):
    from transformers import BertConfig,BertModel,BertTokenizer
    model_dir=tmp_path/'pretrained';model_dir.mkdir()
    (model_dir/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n甲\n在\n乙\n工\n作\n0\n1\n2\n3\n4\n5\n6\n7\n',encoding='utf-8')
    tokenizer=BertTokenizer(vocab=str(model_dir/'vocab.txt'));tokenizer.save_pretrained(model_dir)
    BertModel(BertConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                        intermediate_size=32,max_position_embeddings=64),add_pooling_layer=False).save_pretrained(model_dir)
    dataset=tmp_path/'ie/datasets/tiny';dataset.mkdir(parents=True)
    for split,indices in [('train',range(5)),('validation',[5,6]),('test',[7])]:
        records=[{'text':'甲在乙工作'+str(i),'triples':[[0,1,'人物',0,2,3,'机构']]} for i in indices]
        (dataset/(split+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
    hashes={s:digest(dataset/(s+'.jsonl')) for s in ('train','validation','test')}
    save_json(dataset/'manifest.json',{'id':'tiny','split_sha256':hashes,'dataset_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
             'entity_types':['人物','机构'],'schemas':[{'id':0,'label':'任职','predicate':'任职','slot':'@value','subject_type':'人物','object_type':'机构'}]})
    run=allocate(tmp_path,dataset,Config(model=str(model_dir),architecture=architecture,epochs=2,batch_size=2,
                                       accumulation=2,max_length=64,mixed_precision='fp32'))
    fit(run,device='cpu');meta=read_json(run/'run.json')
    assert meta['status']=='completed' and meta['optimizer_steps']==4 and meta['amp_skipped_updates']==0
    assert len(read_json(run/'history.json'))==2 and not (tmp_path/'.training.lock').exists()
    assert not (run/'test_metrics.json').exists()
    manifest,rows=load(dataset);predictor=Predictor(run)
    measured,_=evaluate(rows['validation'],predictor.predict_rows(rows['validation']),manifest)
    saved=read_json(run/'validation_metrics.json')
    assert measured['relation']==saved['relation'] and measured['entity']==saved['entity']
    from nlp_lab.ie_worker import run_evaluation
    run_evaluation(run,'test')
    assert read_json(run/'evaluation_status.json')['status']=='completed'
    assert read_json(run/'test_metrics.json')['documents']==1
    original=digest(run/'test_metrics.json')
    meta['research_protocol']='frozen';save_json(run/'run.json',meta)
    with pytest.raises(ValueError,match='独立'):run_evaluation(run,'test')
    assert digest(run/'test_metrics.json')==original
