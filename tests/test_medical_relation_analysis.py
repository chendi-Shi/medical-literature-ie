import pytest

from nlp_lab.medical.relation_analysis import punctuation_variant,error_partition,slices,release_decision


def test_error_partition_separates_entity_and_link_failures():
    rows=[{'text':'甲乙丙','triples':[(0,1,'病',0,1,2,'药'),(0,1,'病',0,2,3,'药')]}]
    pred=[{'entities':[(0,1,'病'),(1,2,'药')],'triples':[]}]
    assert error_partition(rows,pred)=={'correct':0,'missing_with_entity_error':1,'missing_despite_both_entities':1,'extra':0}
    with pytest.raises(ValueError):error_partition(rows,pred+pred)


def test_typography_variant_preserves_unicode_spans_and_original():
    row={'text':'😀甲(乙),丙:丁;','triples':[(1,2,'病',0,3,4,'药')]}
    changed=punctuation_variant([row])[0]
    assert changed['text']=='😀甲（乙），丙：丁；'
    assert len(changed['text'])==len(row['text']) and row['text']=='😀甲(乙),丙:丁;'
    assert changed['text'][3:4]=='乙' and changed['triples']==row['triples']


def test_reused_holdout_can_veto_release_without_selecting_another_model():
    assert release_decision(.35,.37,.37,.39)['enabled']
    assert not release_decision(.35,.37,.37,.36)['enabled']
    assert not release_decision(.35,.351,.37,.39)['enabled']


def test_diagnostic_groups_are_based_on_gold_and_training_frequency():
    row={'text':'甲乙丙','triples':[(0,1,'病',0,1,2,'药'),(0,1,'病',0,2,3,'药')]}
    manifest={'schemas':[{'id':0,'subject_type':'病','object_type':'药'}]}
    result=slices([row],[row],[{'entities':[],'triples':[]}],manifest)
    assert result['shared_entity']['documents']==1 and result['rare_relation']['documents']==1
    assert result['long_text']['documents']==0 and result['short_text']['relation']['gold']==2


def test_active_relation_rejects_corrupted_release_artifacts(tmp_path):
    from nlp_lab.data import save_json,digest
    from nlp_lab.medical.runtime import relation_artifact
    root=tmp_path/'medical';run=tmp_path/'ie/runs/joint';run.mkdir(parents=True)
    save_json(root/'status.json',{'relation_run':'original'})
    save_json(root/'report.json',{'weights_sha256':{'relations':'old'},'relations':{'relation':{'f1':.3}}})
    assert relation_artifact(root)['version']=='relation-v1'
    (run/'model.safetensors').write_bytes(b'fixture')
    save_json(run/'run.json',{'status':'completed','config':{'architecture':'joint'}})
    save_json(run/'thresholds.json',{'entity':0.,'relation':0.})
    report={'release':{'enabled':True},'candidate_weights_sha256':digest(run/'model.safetensors'),'candidate':{'relation':{'f1':.4}}}
    save_json(root/'relation-v2/report.json',report)
    policy={'enabled':True,'report_sha256':digest(root/'relation-v2/report.json'),'thresholds_sha256':digest(run/'thresholds.json'),'run_relative_to_workspace':'ie/runs/joint'}
    save_json(root/'active_relation.json',policy)
    assert relation_artifact(root)['metrics']['relation']['f1']==.4
    (run/'model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='weights'):relation_artifact(root)
    (run/'model.safetensors').write_bytes(b'fixture')
    policy['run_relative_to_workspace']='../escape';save_json(root/'active_relation.json',policy)
    with pytest.raises(ValueError,match='path'):relation_artifact(root)


def test_activation_keeps_original_model_when_release_gate_fails(tmp_path):
    from nlp_lab.data import save_json,digest
    from scripts.activate_medical_relation import activate
    root=tmp_path/'medical';out=root/'relation-v2'
    save_json(root/'status.json',{'relation_run':'original'})
    save_json(root/'report.json',{'weights_sha256':{'relations':'old'},'relations':{'relation':{'f1':.3}}})
    save_json(out/'protocol.json',{'min_validation_gain':.005})
    save_json(out/'selection.json',{'chosen':{'entity':0.,'relation':0.}})
    release=release_decision(.35,.36,.37,.36)
    report={'baseline_validation_f1':.35,'candidate_validation_f1':.36,'baseline':{'relation':{'f1':.37}},
            'candidate':{'relation':{'f1':.36}},'release':release,'protocol_sha256':digest(out/'protocol.json'),
            'selection_sha256':digest(out/'selection.json')}
    save_json(out/'report.json',report)
    assert activate(root)['version']=='relation-v1'
    assert not __import__('json').loads((root/'active_relation.json').read_text())['enabled']
    save_json(out/'selection.json',{'chosen':{'entity':-.5,'relation':-.5}})
    with pytest.raises(ValueError,match='selection'):activate(root)


def test_checkpoint_restores_optimizer_and_all_random_streams(tmp_path):
    import random
    import numpy as np
    import torch
    from nlp_lab.medical.joint_train import rng_state,restore_rng,atomic_checkpoint
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step()
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state()}
    atomic_checkpoint(tmp_path/'resume.pt',state)
    expected=(random.random(),float(np.random.rand()),torch.rand(3))
    optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step()
    parameters=[p.detach().clone() for p in model.parameters()]
    loaded=torch.load(tmp_path/'resume.pt',weights_only=True,map_location='cpu')
    model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer']);restore_rng(loaded['rng'])
    assert random.random()==expected[0] and float(np.random.rand())==expected[1]
    torch.testing.assert_close(torch.rand(3),expected[2])
    optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step()
    for got,want in zip(model.parameters(),parameters):torch.testing.assert_close(got,want,rtol=0,atol=0)
    assert not (tmp_path/'resume.tmp').exists()

@pytest.mark.parametrize('interruption_point',['before_validation','within_epoch'])
def test_joint_training_resumes_without_changing_final_weights(tmp_path,monkeypatch,interruption_point):
    import hashlib,json,torch
    from transformers import BertConfig,BertModel,BertTokenizer
    from nlp_lab.data import save_json,read_json,digest
    from nlp_lab.ie.train import Config,allocate
    from nlp_lab.medical import joint_train
    model_dir=tmp_path/'encoder';model_dir.mkdir()
    (model_dir/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n甲\n在\n乙\n工\n作\n0\n1\n2\n3\n4\n5\n6\n7\n',encoding='utf-8')
    tokenizer=BertTokenizer(vocab=str(model_dir/'vocab.txt'));tokenizer.save_pretrained(model_dir)
    BertModel(BertConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
        intermediate_size=32,max_position_embeddings=64),add_pooling_layer=False).save_pretrained(model_dir)
    dataset=tmp_path/'dataset';dataset.mkdir()
    train_count=5 if interruption_point=='before_validation' else 257
    for split,indices in [('train',range(train_count)),('validation',[train_count,train_count+1]),('test',[train_count+2])]:
        rows=[{'text':'甲在乙工作'+str(i),'triples':[[0,1,'人物',0,2,3,'机构']]} for i in indices]
        (dataset/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    hashes={s:digest(dataset/f'{s}.jsonl') for s in ('train','validation','test')}
    save_json(dataset/'manifest.json',{'id':'resume-test','split_sha256':hashes,
        'dataset_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        'entity_types':['人物','机构'],'schemas':[{'id':0,'label':'任职','predicate':'任职','slot':'@value','subject_type':'人物','object_type':'机构'}]})
    config=Config(model=str(model_dir),architecture='joint',epochs=2,batch_size=2,accumulation=2,max_length=64,mixed_precision='fp32')
    reference=allocate(tmp_path,dataset,config);resumed=allocate(tmp_path,dataset,config)
    joint_train.fit_joint(reference,device='cpu')
    atomic=joint_train.atomic_checkpoint
    def interrupt(path,state):
        atomic(path,state)
        stop=(state['done']==(train_count+1)//2) if interruption_point=='before_validation' else state['steps']==64
        if state['epoch']==1 and stop:raise RuntimeError('injected interruption')
    monkeypatch.setattr(joint_train,'atomic_checkpoint',interrupt)
    with pytest.raises(RuntimeError,match='injected'):joint_train.fit_joint(resumed,device='cpu')
    assert (resumed/'resume.pt').exists() and not (tmp_path/'.training.lock').exists()
    monkeypatch.setattr(joint_train,'atomic_checkpoint',atomic)
    joint_train.fit_joint(resumed,device='cpu')
    assert read_json(resumed/'run.json')['optimizer_steps']==(4 if train_count==5 else 130)
    assert read_json(resumed/'run.json')['resume_count']==1
    assert read_json(reference/'history.json')==read_json(resumed/'history.json')
    a=torch.load(reference/'resume.pt',weights_only=True,map_location='cpu')['model']
    b=torch.load(resumed/'resume.pt',weights_only=True,map_location='cpu')['model']
    for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)


def test_upstream_head_sum_loss_matches_explicit_per_example_definition():
    import torch
    from nlp_lab.medical.gplinker_loss import sum_heads_objective
    outputs={name:torch.linspace(-2,2,2*heads*4).reshape(2,heads,2,2).requires_grad_()
             for name,heads in [('entity',2),('head',3),('tail',3)]}
    targets={k:torch.zeros_like(v,dtype=torch.bool) for k,v in outputs.items()}
    for value in targets.values():value[0,0,0,1]=True
    actual,components=sum_heads_objective(outputs,targets)
    expected=outputs['entity'].new_zeros(())
    for name,logits in outputs.items():
        for i in range(2):
            for j in range(logits.shape[1]):
                flat=logits[i,j].flatten();truth=targets[name][i,j].flatten();zero=flat.new_zeros(1)
                expected+=(torch.logsumexp(torch.cat([flat[~truth],zero]),0)
                           +torch.logsumexp(torch.cat([-flat[truth],zero]),0))/2
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert all(torch.isfinite(v.grad).all() for v in outputs.values())
    without_tail,_=sum_heads_objective(outputs,targets,use_tail=False)
    torch.testing.assert_close(without_tail,components['entity']+components['head'])
