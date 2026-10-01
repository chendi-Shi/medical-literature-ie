"""Predeclared two-seed extension; same frozen trainer and dataset as the baseline."""
from pathlib import Path
from dataclasses import asdict
import gc,os,time,random,hashlib,json
import torch
from nlp_lab.data import read_json,save_json
from nlp_lab.training import now
from nlp_lab.ie.data import DATASET,digest,load
from nlp_lab.ie.train import Config,allocate,update
from nlp_lab.ie.suite import source_hashes,summary
from nlp_lab.ie.metrics import evaluate,paired_interval
from nlp_lab.ie_context import FEATURE_RECIPE,fit_context,ContextPredictor

EXTENSION='duie2-context-10k-v1'

def suite(workspace):
    workspace=Path(workspace).resolve();project=Path(__file__).resolve().parents[1]
    directory=workspace/'ie/research'/EXTENSION;directory.mkdir(parents=True,exist_ok=True)
    baseline=workspace/'ie/research'/DATASET;dataset=workspace/'ie/datasets'/DATASET
    manifest,rows=load(dataset)
    hashes={**{'frozen_core/'+k:v for k,v in source_hashes().items()},
            'nlp_lab/ie_context.py':digest(project/'nlp_lab/ie_context.py'),'scripts/run_ie_context.py':digest(Path(__file__))}
    protocol={'version':1,'created_at':now(),'dataset':DATASET,'dataset_sha256':manifest['dataset_sha256'],
              'split_sha256':manifest['split_sha256'],'source_sha256':hashes,'features':FEATURE_RECIPE,
              'recipes':[asdict(Config(architecture='pipeline',seed=s)) for s in (42,43)],'threshold_grid':[0.,-.5,.5],
              'selection':'zero-logit validation best epoch; common entity/relation threshold selected on validation; method by two-seed mean tuned validation; representative seed42',
              'test_policy':'both extension fits and validation tuning finish before extension test evaluation; extension designed using authored inference examples and training/validation only',
              'factory':'process-local frozen train.Extractor replacement with ContextExtractor; same optimizer, loss, negatives, encoder and budget',
              'stress':{'prefix':'信息：','sample_count':1000,'seed':2027},'smoke_disclosure':'first12 test rows previously used for metric plumbing, separately report remaining3988',
              'baseline_phase_when_frozen':read_json(baseline/'state.json')['phase']}
    if (directory/'protocol.json').exists():
        old=read_json(directory/'protocol.json')
        for key in ('dataset_sha256','source_sha256','features','recipes'):
            if old[key]!=protocol[key]:raise ValueError('Context frozen protocol mismatch: '+key)
        protocol=old
    else:save_json(directory/'protocol.json',protocol)
    fd=os.open(workspace/'ie/.context.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    state=read_json(directory/'state.json') if (directory/'state.json').exists() else {'entries':[]}
    def phase(value,**kwargs):
        state.update(phase=value,updated_at=now(),pid=os.getpid(),**kwargs);save_json(directory/'state.json',state)
    try:
        for config in protocol['recipes'][len(state['entries']):]:
            run=allocate(workspace,dataset,Config(**config))
            update(run,relation_features='context_position',research_protocol=str(directory/'protocol.json'))
            state['entries'].append({'name':'context_pipeline','seed':config['seed'],'run':run.name})
        phase('waiting_for_baseline')
        while True:
            prior=read_json(baseline/'state.json')
            if prior['phase']=='failed':raise ValueError('Baseline suite failed')
            if prior['phase']=='completed' and not (workspace/'ie/.suite.lock').exists():break
            time.sleep(30)
        phase('training')
        for entry in state['entries']:
            run=workspace/'ie/runs'/entry['run']
            if read_json(run/'run.json')['status']!='completed':fit_context(run)
        phase('validation_tuning')
        for entry in state['entries']:
            run=workspace/'ie/runs'/entry['run']
            if (run/'thresholds.json').exists():continue
            predictor=ContextPredictor(run,device='cuda');choices=[];best=-1.
            for value in protocol['threshold_grid']:
                metrics,_=evaluate(rows['validation'],predictor.predict_rows(rows['validation'],entity_threshold=value,relation_threshold=value),manifest)
                choices.append({'threshold':value,'relation_f1':metrics['relation']['f1'],'entity_f1':metrics['entity']['f1']})
                if metrics['relation']['f1']>best:best=metrics['relation']['f1'];selected=value;chosen=metrics
            save_json(run/'thresholds.json',{'entity':selected,'relation':selected,'selection_split':'validation','grid_results':choices})
            save_json(run/'tuned_validation_metrics.json',chosen)
            del predictor;gc.collect();torch.cuda.empty_cache()
        # Baseline test scores are read only after both extension fits and tuning finish.
        original_report=read_json(baseline/'report.json')
        validation_means={**original_report['validation_means'],'context_pipeline':summary([read_json(workspace/'ie/runs'/e['run']/'tuned_validation_metrics.json')['relation']['f1'] for e in state['entries']])['mean']}
        selected_method=max(validation_means,key=validation_means.get)
        selected_run=state['entries'][0]['run'] if selected_method=='context_pipeline' else original_report['selected_run']
        phase('test_evaluation',test_opened_at=now(),selected_architecture=selected_method,selected_run=selected_run)
        ids=list(range(len(rows['test'])));random.Random(2027).shuffle(ids);ids=ids[:1000]
        clean=[rows['test'][i] for i in ids];prefix=protocol['stress']['prefix'];shift=len(prefix)
        stress=[{**r,'text':prefix+r['text'],'triples':[[a+shift,b+shift,t,k,c+shift,d+shift,u] for a,b,t,k,c,d,u in r['triples']]} for r in clean]
        experiments=[]
        for entry in state['entries']:
            run=workspace/'ie/runs'/entry['run'];predictor=ContextPredictor(run,device='cuda')
            predictions=predictor.predict_rows(rows['test']);metrics,errors=evaluate(rows['test'],predictions,manifest)
            save_json(run/'test_metrics.json',metrics);save_json(run/'test_errors.json',[e for e in errors if e['missing'] or e['extra']])
            clean_metrics,_=evaluate(clean,[predictions[i] for i in ids],manifest)
            changed,_=evaluate(stress,predictor.predict_rows(stress),manifest)
            diagnostics={'stress':{'documents':1000,'clean':clean_metrics['relation'],'prefixed':changed['relation'],'drop_f1':clean_metrics['relation']['f1']-changed['relation']['f1'],
                                   'sample_indices_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}}
            save_json(run/'diagnostics.json',diagnostics);meta=read_json(run/'run.json')
            experiments.append({**entry,**{k:meta[k] for k in ('config','best_epoch','seconds','parameters','trainable_parameters','peak_cuda_memory_mb','model_revision')},
                                'checkpoint_sha256':digest(run/'model.safetensors'),'thresholds':read_json(run/'thresholds.json'),
                                'validation':read_json(run/'tuned_validation_metrics.json'),'test':metrics,'diagnostics':diagnostics})
            print({'test':entry,'relation':metrics['relation'],'entity':metrics['entity']},flush=True)
            del predictor,predictions;gc.collect();torch.cuda.empty_cache()
        reference=next(e for e in original_report['experiments'] if e['name']=='pipeline' and e['seed']==42)
        report={'protocol':protocol,'dataset':manifest,'experiments':experiments,
                'aggregates':{'context_pipeline':{key:summary([e['test'][metric]['f1'] for e in experiments]) for key,metric in [('relation_f1','relation'),('entity_f1','entity')]}},
                'paired_seed42':paired_interval(reference['test']['counts_per_document'],experiments[0]['test']['counts_per_document']),
                'selected_architecture':selected_method,'selected_run':selected_run,'validation_means':validation_means,'finished_at':now()}
        save_json(directory/'report.json',report);phase('completed');return report
    except Exception as e:phase('failed',error=f'{type(e).__name__}: {e}');raise
    finally:(workspace/'ie/.context.lock').unlink(missing_ok=True)

if __name__=='__main__':suite(Path('workspace'))
