"""Train a same-budget medical GPLinker baseline; preserve every v1 artifact."""
from pathlib import Path
import gc
import os
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nlp_lab.data import digest,read_json,save_json
from nlp_lab.training import now
from nlp_lab.ie.data import load
from nlp_lab.ie.train import Config,allocate
from nlp_lab.medical.joint_train import fit_joint
from nlp_lab.ie.metrics import evaluate,paired_interval
from nlp_lab.ie_context import predictor_for
from nlp_lab.medical.relation_analysis import punctuation_variant,error_partition,slices,release_decision


def compact(metrics):
    return {k:v for k,v in metrics.items() if k!='counts_per_document'}


def main():
    import torch
    root=Path('workspace/medical');out=root/'relation-v2';dataset=root/'datasets/cmeie-384-v1'
    baseline_run=Path(read_json(root/'status.json')['relation_run'])
    if (out/'report.json').exists():raise ValueError('Completed comparison is immutable')
    frozen=read_json(root/'protocol.json')
    for name,sha in frozen['source_hashes'].items():
        if digest(Path(name))!=sha:raise ValueError('Original training source changed: '+name)
    manifest,rows=load(dataset)
    config=Config(architecture='joint',epochs=6,batch_size=2,accumulation=16,max_length=384,seed=42)
    if not torch.cuda.is_available():raise ValueError('This full training protocol requires CUDA')
    if not (out/'protocol.json').exists():
        run=allocate(Path('workspace'),dataset,config)
        save_json(out/'protocol.json',{'created_at':now(),'config':vars(config),'run':str(run),
            'baseline_weights_sha256':digest(baseline_run/'model.safetensors'),
            'dataset_sha256':manifest['dataset_sha256'],'split_sha256':manifest['split_sha256'],
            'source_sha256':{p:digest(Path(p)) for p in ['scripts/compare_medical_relations.py','nlp_lab/medical/relation_analysis.py','nlp_lab/medical/joint_train.py',*frozen['source_hashes']]},
            'threshold_pairs':[[0.,0.],[-.5,-.5],[.5,.5]],'selection':'max validation relation F1; ties prefer zero',
            'checkpoint':'max validation relation F1 at zero thresholds, same as original trainer',
            'min_validation_gain':.005,'release_audit':'previously reported holdout may veto deployment; never choose another candidate using it',
            'ablation':'inference only: remove tail-link intersection using the same selected checkpoint and thresholds; not retraining',
            'robustness':'same-length ASCII to full-width punctuation, all test documents; original offsets preserved',
            'bootstrap':{'replicates':1000,'seed':20261007},
            'upstream':{'repo':'bojone/GPLinker','commit':'461fb1f5c9a828356daabd6d75d71a0cc1ff7f9a','path':'duie_v1.py','blob_sha':'6446e4f3171bd6efec5e619db1988e389324854a'},
            'limitations':['One training seed; bootstrap excludes seed variation.','Microbatch2/accumulation16 vs original4/8 due concurrent resource pressure; same effective batch32 and update budget, different dropout numerics.','Reuses previous dev-derived holdout; not pristine final testing.',
                           'Typed medical GPLinker adaptation, not exact reproduction of upstream 12-layer encoder/untyped roles.',
                           'CMeIE relations are not trained literature PICO or efficacy/adverse-event records.']})
    protocol=read_json(out/'protocol.json');run=Path(protocol['run'])
    for name,sha in protocol['source_sha256'].items():
        if digest(Path(name))!=sha:raise ValueError('Comparison source changed: '+name)
    if manifest['dataset_sha256']!=protocol['dataset_sha256']:raise ValueError('Comparison corpus changed')
    if read_json(run/'run.json')['status']!='completed':fit_joint(run,device='cuda')
    lock=Path('workspace/.training.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    try:
        meta=read_json(run/'run.json')
        if meta['status']!='completed' or meta['epochs_completed']!=6 or meta['amp_skipped_updates']:
            raise ValueError('Comparison did not complete six epochs without skipped updates')
        predictor=predictor_for(run,'cuda');options=[]
        for eth,rth in protocol['threshold_pairs']:
            metrics,_=evaluate(rows['validation'],predictor.predict_rows(rows['validation'],entity_threshold=eth,relation_threshold=rth,batch_size=1),manifest)
            options.append({'entity':eth,'relation':rth,'metrics':compact(metrics)})
        choice=max(options,key=lambda c:(c['metrics']['relation']['f1'],c['entity']==0.))
        save_json(run/'thresholds.json',{k:choice[k] for k in ('entity','relation')})
        predictor.thresholds={k:choice[k] for k in ('entity','relation')}
        selection={'selected_at':now(),'chosen':choice,'options':options}
        save_json(out/'selection.json',selection)
        test_opened=now();test=rows['test'];variant=punctuation_variant(test)
        candidate=predictor.predict_rows(test,batch_size=1);metrics,errors=evaluate(test,candidate,manifest)
        save_json(out/'errors.json',errors)
        no_tail=predictor.predict_rows(test,use_tail=False,batch_size=1);ablation,_=evaluate(test,no_tail,manifest)
        perturbed=predictor.predict_rows(variant,batch_size=1);robust,_=evaluate(variant,perturbed,manifest)
        candidate_slices=slices(rows['train'],test,candidate,manifest)
        del predictor;gc.collect();torch.cuda.empty_cache()
        baseline=predictor_for(baseline_run,'cuda')
        reference=baseline.predict_rows(test,batch_size=1);original,_=evaluate(test,reference,manifest)
        previous=read_json(baseline_run/'test_metrics.json')
        if original['relation']!=previous['relation']:raise ValueError('Original relation counts do not reproduce')
        bval=read_json(baseline_run/'tuned_validation_metrics.json')['relation']['f1']
        bperturbed=baseline.predict_rows(variant,batch_size=1);brobust,_=evaluate(variant,bperturbed,manifest)
        oracle=baseline.predict_rows(test,oracle_entities=True,batch_size=1);oracle_metrics,_=evaluate(test,oracle,manifest)
        report={'finished_at':now(),'test_opened_at':test_opened,'protocol_sha256':digest(out/'protocol.json'),
            'selection_sha256':digest(out/'selection.json'),'baseline':compact(original),'candidate':compact(metrics),
            'baseline_validation_f1':bval,'candidate_validation_f1':choice['metrics']['relation']['f1'],
            'paired_comparison':paired_interval(original['counts_per_document'],metrics['counts_per_document'],replicates=1000,seed=20261007),
            'inference_tail_ablation':compact(ablation),'punctuation_robustness':{'baseline':compact(brobust),'candidate':compact(robust)},
            'baseline_oracle_entities':compact(oracle_metrics),'error_partition':{'baseline':error_partition(test,reference),'candidate':error_partition(test,candidate)},
            'slices':{'baseline':slices(rows['train'],test,reference,manifest),'candidate':candidate_slices},
            'candidate_run':{k:v for k,v in meta.items() if k not in ('dataset_path','versions')},
            'candidate_weights_sha256':digest(run/'model.safetensors'),'limitations':protocol['limitations'],
            'release':release_decision(bval,choice['metrics']['relation']['f1'],original['relation']['f1'],metrics['relation']['f1'],protocol['min_validation_gain'])}
        save_json(out/'report.json',report)
        print({'candidate_f1':metrics['relation']['f1'],'baseline_f1':original['relation']['f1'],'release':report['release']},flush=True)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
