"""Fixed-budget experiments. Test labels are opened only after all fits finish."""
from pathlib import Path
from dataclasses import asdict
import gc
import hashlib
import json
import os
import random
import time
import numpy as np
import torch
from ..data import read_json, save_json
from ..training import now
from .data import DATASET, digest, load
from .train import Config, allocate, fit, update
from .predict import Predictor
from .metrics import evaluate, paired_interval


def source_hashes():
    return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def summary(values):
    return {'mean':float(np.mean(values)), 'std':float(np.std(values,ddof=1)) if len(values)>1 else None,
            'values':values, 'seeds':len(values)}


def suite(workspace:Path):
    workspace=workspace.resolve();dataset=workspace/'ie/datasets'/DATASET
    manifest,rows=load(dataset)
    directory=workspace/'ie/research'/DATASET;directory.mkdir(parents=True,exist_ok=True)
    recipes=[{'name':arch,'config':asdict(Config(architecture=arch,seed=seed))}
             for arch in ('pipeline','joint') for seed in (42,43)]
    recipes.append({'name':'frozen_encoder','config':asdict(Config(freeze_encoder=True))})
    protocol={'version':1,'created_at':now(),'dataset':DATASET,'dataset_sha256':manifest['dataset_sha256'],
              'split_sha256':manifest['split_sha256'],'recipes':recipes,'source_sha256':source_hashes(),
              'selection':'epoch selected by validation relation F1 at zero logits; architecture by mean tuned validation F1',
              'threshold_grid':[0.,-.5,.5],'threshold_policy':'same entity/relation logit threshold; validation only, prefer zero on ties',
              'test_policy':'all fits and validation tuning finish before first heldout test evaluation',
              'stress':{'name':'neutral_prefix','prefix':'信息：','sample_count':1000,'seed':2027},
              'ablations':['frozen encoder, separately trained seed42','remove tail intersection, same joint seed42 weights'],
              'diagnostic':'pipeline seed42 with gold participant entities; not deployment performance',
              'interval':'600 paired text bootstrap replicates, seed31415, fixed seed42 models',
              'limitations':manifest['limitations']}
    if (directory/'protocol.json').exists():
        old=read_json(directory/'protocol.json')
        for k in ('dataset_sha256','recipes','source_sha256','threshold_grid'):
            if old[k]!=protocol[k]:raise ValueError('冻结 IE 协议与当前代码/配置不同：'+k)
        protocol=old
    else:save_json(directory/'protocol.json',protocol)
    lock=workspace/'ie/.suite.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    state=read_json(directory/'state.json') if (directory/'state.json').exists() else {'entries':[]}
    def phase(value,**kw):
        state.update(phase=value,updated_at=now(),pid=os.getpid(),**kw);save_json(directory/'state.json',state)
    def release(p):
        del p;gc.collect();torch.cuda.empty_cache()
    try:
        phase('training')
        for i,recipe in enumerate(recipes):
            if i>=len(state['entries']):
                run=allocate(workspace,dataset,Config(**recipe['config']))
                update(run,research_protocol=str(directory/'protocol.json'))
                state['entries'].append({'name':recipe['name'],'seed':recipe['config']['seed'],'run':run.name});phase('training')
            entry=state['entries'][i];run=workspace/'ie/runs'/entry['run'];meta=read_json(run/'run.json')
            if meta['status']!='completed':
                if meta['status'] not in ('queued','failed'):raise ValueError('有未清理的 IE 训练进程')
                fit(run)
        phase('validation_tuning')
        for entry in state['entries']:
            run=workspace/'ie/runs'/entry['run']
            if (run/'thresholds.json').exists():continue
            predictor=Predictor(run,device='cuda');best=-1.;choices=[]
            for value in protocol['threshold_grid']:
                predicted=predictor.predict_rows(rows['validation'],entity_threshold=value,relation_threshold=value)
                metrics,_=evaluate(rows['validation'],predicted,manifest)
                choices.append({'threshold':value,'relation_f1':metrics['relation']['f1'],'entity_f1':metrics['entity']['f1']})
                if metrics['relation']['f1']>best:
                    best=metrics['relation']['f1'];selected=value;chosen=metrics
            save_json(run/'thresholds.json',{'entity':selected,'relation':selected,'selection_split':'validation','grid_results':choices})
            save_json(run/'tuned_validation_metrics.json',chosen)
            print({'threshold_selection':entry,'value':selected,'f1':best},flush=True)
            del predictor;gc.collect();torch.cuda.empty_cache()
        validation_means={name:float(np.mean([read_json(workspace/'ie/runs'/e['run']/'tuned_validation_metrics.json')['relation']['f1']
                                             for e in state['entries'] if e['name']==name])) for name in ('pipeline','joint')}
        chosen=max(validation_means,key=validation_means.get)
        representative=next(e['run'] for e in state['entries'] if e['name']==chosen and e['seed']==42)
        phase('test_evaluation',test_opened_at=state.get('test_opened_at',now()),selected_architecture=chosen,selected_run=representative)
        stress_ids=list(range(len(rows['test'])));random.Random(2027).shuffle(stress_ids);stress_ids=stress_ids[:1000]
        original=[rows['test'][i] for i in stress_ids];prefix=protocol['stress']['prefix'];shift=len(prefix)
        stress=[{**r,'text':prefix+r['text'],'triples':[[a+shift,b+shift,t,k,c+shift,d+shift,u] for a,b,t,k,c,d,u in r['triples']]} for r in original]
        for entry in state['entries']:
            run=workspace/'ie/runs'/entry['run']
            if (run/'test_metrics.json').exists() and (run/'diagnostics.json').exists():continue
            predictor=Predictor(run,device='cuda')
            predictions=predictor.predict_rows(rows['test']);metrics,errors=evaluate(rows['test'],predictions,manifest)
            save_json(run/'test_metrics.json',metrics)
            save_json(run/'test_errors.json',[x for x in errors if x['missing'] or x['extra']])
            clean,_=evaluate(original,[predictions[i] for i in stress_ids],manifest)
            changed,_=evaluate(stress,predictor.predict_rows(stress),manifest)
            diagnostics={'stress':{'documents':1000,'clean':clean['relation'],'prefixed':changed['relation'],
                                    'drop_f1':clean['relation']['f1']-changed['relation']['f1'],'sample_indices_sha256':hashlib.sha256(json.dumps(stress_ids).encode()).hexdigest()}}
            if entry['name']=='pipeline' and entry['seed']==42:
                oracle,_=evaluate(rows['test'],predictor.predict_rows(rows['test'],oracle_entities=True),manifest)
                diagnostics['gold_entity_oracle']=oracle
            if entry['name']=='joint' and entry['seed']==42:
                no_tail,_=evaluate(rows['test'],predictor.predict_rows(rows['test'],use_tail=False),manifest)
                diagnostics['no_tail_decode']=no_tail
            save_json(run/'diagnostics.json',diagnostics)
            print({'test':entry,'relation':metrics['relation'],'entity':metrics['entity']},flush=True)
            del predictor,predictions;gc.collect();torch.cuda.empty_cache()
        phase('reporting')
        experiments=[]
        for entry in state['entries']:
            run=workspace/'ie/runs'/entry['run'];meta=read_json(run/'run.json')
            experiments.append({**entry,'config':meta['config'],'best_epoch':meta['best_epoch'],'seconds':meta['seconds'],
                                'parameters':meta['parameters'],'trainable_parameters':meta['trainable_parameters'],
                                'peak_cuda_memory_mb':meta['peak_cuda_memory_mb'],'model_revision':meta['model_revision'],
                                'checkpoint_sha256':digest(run/'model.safetensors'),'thresholds':read_json(run/'thresholds.json'),
                                'validation':read_json(run/'tuned_validation_metrics.json'),'test':read_json(run/'test_metrics.json'),
                                'diagnostics':read_json(run/'diagnostics.json')})
        aggregates={name:{'relation_f1':summary([e['test']['relation']['f1'] for e in experiments if e['name']==name]),
                          'entity_f1':summary([e['test']['entity']['f1'] for e in experiments if e['name']==name]),
                          'stress_drop':summary([e['diagnostics']['stress']['drop_f1'] for e in experiments if e['name']==name])}
                    for name in ('pipeline','joint','frozen_encoder')}
        reference=next(e for e in experiments if e['name']=='pipeline' and e['seed']==42)
        candidate=next(e for e in experiments if e['name']=='joint' and e['seed']==42)
        report={'protocol':protocol,'dataset':manifest,'experiments':experiments,'aggregates':aggregates,
                'paired_seed42':paired_interval(reference['test']['counts_per_document'],candidate['test']['counts_per_document']),
                'selected_architecture':chosen,'selected_run':representative,'validation_means':validation_means,'finished_at':now()}
        save_json(directory/'report.json',report);phase('completed')
        return report
    except Exception as e:
        phase('failed',error=f'{type(e).__name__}: {e}');raise
    finally:lock.unlink(missing_ok=True)
