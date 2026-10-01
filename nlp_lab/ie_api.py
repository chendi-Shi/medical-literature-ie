from pathlib import Path
from collections import OrderedDict
from threading import RLock
import os
import subprocess
import sys
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from .data import read_json,save_json
from .cli import safe_child
from .ie.data import DATASET


class ExtractRequest(BaseModel):
    text:str=Field(min_length=1,max_length=6000)
    run_id:str|None=None

class IETrainRequest(BaseModel):
    dataset_id:str=DATASET
    architecture:str='joint'
    epochs:int=Field(default=6,ge=1,le=30)
    seed:int=Field(default=42,ge=0,le=4294967295)
    freeze_encoder:bool=False

class IEImportRequest(BaseModel):
    name:str=Field(min_length=1,max_length=80)
    content:str=Field(min_length=1,max_length=1_000_000)
    schema_content:str=Field(min_length=1,max_length=100_000,alias='schema_json')
    provenance:str=Field(default='user-provided',max_length=1000)

def group_relations(result):
    """Group identical typed surface triples; preserve every mention link."""
    entities={e['id']:e for e in result['entities']};groups={}
    for r in result['relations']:
        subject,object_=entities[r['subject']],entities[r['object']]
        key=(r['subject_text'],subject['type'],r['predicate'],r['slot'],r['object_text'],object_['type'])
        if key not in groups:
            groups[key]={**r,'subject_type':subject['type'],'object_type':object_['type'],'mention_links':[]}
        groups[key]['confidence']=max(groups[key]['confidence'],r['confidence'])
        groups[key]['mention_links'].append({'subject':r['subject'],'object':r['object']})
    return list(groups.values())

def router(workspace:Path):
    routes=APIRouter(prefix='/api/ie');root=workspace/'ie';cache=OrderedDict();mutex=RLock()
    research=root/'research'/DATASET
    def checked(run_id):
        try:run=safe_child(root/'runs',run_id)
        except ValueError as e:raise HTTPException(400,str(e)) from e
        if not (run/'run.json').exists():raise HTTPException(404,'信息抽取实验不存在')
        return run
    def compact(metrics):
        return {k:v for k,v in metrics.items() if k!='counts_per_document'}
    @routes.get('')
    def overview():
        result={'state':{'phase':'not_prepared'},'experiments':[]}
        dataset=root/'datasets'/DATASET/'manifest.json'
        if dataset.exists():result['dataset']=read_json(dataset)
        for name in ('protocol','state'):
            if (research/(name+'.json')).exists():result[name]=read_json(research/(name+'.json'))
        for entry in result['state'].get('entries',[]):
            meta=read_json(checked(entry['run'])/'run.json')
            result['experiments'].append({**entry,**meta})
        if (research/'report.json').exists():
            report=read_json(research/'report.json')
            for entry in report['experiments']:
                entry['test']=compact(entry['test']);entry['validation']=compact(entry['validation'])
                for name in ('gold_entity_oracle','no_tail_decode'):
                    if name in entry['diagnostics']:entry['diagnostics'][name]=compact(entry['diagnostics'][name])
            result['report']=report
        if (research/'analysis.json').exists():result['analysis']=read_json(research/'analysis.json')
        extension=root/'research'/'duie2-context-10k-v1'
        if (extension/'state.json').exists():
            result['context_state']=read_json(extension/'state.json')
            for entry in result['context_state'].get('entries',[]):
                result['experiments'].append({**entry,**read_json(checked(entry['run'])/'run.json')})
            if (extension/'report.json').exists():
                context=read_json(extension/'report.json')
                for entry in context['experiments']:
                    entry['test']=compact(entry['test']);entry['validation']=compact(entry['validation'])
                result['context_report']=context
                result['report']['experiments']+=context['experiments']
                result['report']['aggregates'].update(context['aggregates'])
                result['report']['validation_means']=context['validation_means']
                result['report']['selected_architecture']=context['selected_architecture']
                result['report']['selected_run']=context['selected_run']
                result['state']['selected_run']=context['selected_run']
            elif result['state']['phase']=='completed':
                result['state']['phase']=result['context_state']['phase']
        return result
    @routes.get('/runs')
    def runs():
        return [m for p in sorted((root/'runs').glob('*/run.json'),reverse=True)
                for m in [read_json(p)] if m['dataset']==DATASET or m.get('task')=='information_extraction' and m['dataset']!='smoke']
    @routes.get('/datasets')
    def datasets():
        return [read_json(p) for p in sorted((root/'datasets').glob('*/manifest.json')) if p.parent.name!='smoke']
    @routes.post('/datasets',status_code=201)
    def import_dataset(body:IEImportRequest):
        import json
        from .ie_import import import_data
        try:return import_data(body.content,json.loads(body.schema_content),workspace,body.name,body.provenance)
        except (ValueError,KeyError,TypeError,OSError) as e:raise HTTPException(400,str(e)) from e
    @routes.get('/runs/{run_id}')
    def detail(run_id:str):
        run=checked(run_id);result=read_json(run/'run.json')
        for name in ('history','validation_metrics','tuned_validation_metrics','test_metrics','thresholds','diagnostics','test_errors','evaluation_status'):
            if (run/(name+'.json')).exists():
                value=read_json(run/(name+'.json'))
                if name=='test_errors':value=value[:50]
                elif name.endswith('metrics'):value=compact(value)
                result[name]=value
        return result
    @routes.post('/runs/{run_id}/evaluate',status_code=202)
    def evaluate_run(run_id:str,split:str='test'):
        run=checked(run_id);meta=read_json(run/'run.json')
        if meta.get('research_protocol'):raise HTTPException(409,'冻结研究的评测由套件统一生成，请查看已有结果')
        if meta['status']!='completed' or split not in ('test','validation'):raise HTTPException(400,'请指定已完成实验与有效划分')
        try:fd=os.open(run/'.evaluation.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError as e:raise HTTPException(409,'评测仍在运行') from e
        os.close(fd);save_json(run/'evaluation_status.json',{'status':'running','split':split})
        try:
            with (run/'evaluation.log').open('a',encoding='utf-8') as log:
                subprocess.Popen([sys.executable,'-m','nlp_lab.ie_worker',str(run),split],
                    cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                    env={**os.environ,'PYTHONIOENCODING':'utf-8','HF_HUB_OFFLINE':'1'})
        except OSError as e:
            (run/'.evaluation.lock').unlink(missing_ok=True);save_json(run/'evaluation_status.json',{'status':'failed','error':str(e)})
            raise HTTPException(500,'无法启动独立 IE 评测') from e
        return {'status':'running','split':split}
    @routes.post('/extract')
    def extract(body:ExtractRequest):
        state=read_json(research/'state.json') if (research/'state.json').exists() else {}
        extension=root/'research'/'duie2-context-10k-v1'/'report.json'
        if extension.exists():state['selected_run']=read_json(extension)['selected_run']
        run_id=body.run_id or state.get('selected_run')
        if not run_id:raise HTTPException(409,'请先完成训练，再选择一个可用模型')
        run=checked(run_id)
        try:
            with mutex:
                threshold_file=run/'thresholds.json'
                stamp=threshold_file.stat().st_mtime_ns if threshold_file.exists() else None
                if run_id not in cache or cache[run_id][1]!=stamp:
                    from .ie_context import predictor_for
                    cache[run_id]=(predictor_for(run,device='cpu'),stamp)
                    while len(cache)>2:cache.popitem(last=False)
                cache.move_to_end(run_id)
                result=cache[run_id][0].extract(body.text)
                return {**result,'relation_groups':group_relations(result),
                        'relation_grouping':'exact typed surface + predicate/slot; mention links preserved, no alias or coreference resolution',
                        'run_id':run_id,'model_architecture':'context_pipeline' if read_json(run/'run.json').get('relation_features')=='context_position' else read_json(run/'run.json')['config']['architecture']}
        except ValueError as e:raise HTTPException(400,str(e)) from e
    @routes.post('/train',status_code=202)
    def train(body:IETrainRequest):
        if (root/'.suite.lock').exists() or (root/'.context.lock').exists() or (workspace/'.training.lock').exists():
            raise HTTPException(409,'已有训练或冻结研究运行，请等待结束')
        from .ie.train import Config,allocate,update
        import torch
        config=Config(architecture='pipeline' if body.architecture=='context_pipeline' else body.architecture,epochs=body.epochs,seed=body.seed,freeze_encoder=body.freeze_encoder,
                      mixed_precision='fp16' if torch.cuda.is_available() else 'fp32')
        try:
            dataset=safe_child(root/'datasets',body.dataset_id)
            if not (dataset/'manifest.json').exists():raise ValueError('信息抽取数据版本不存在')
            run=allocate(workspace,dataset,config)
            if body.architecture=='context_pipeline':update(run,relation_features='context_position')
            with (run/'worker.log').open('a',encoding='utf-8') as log:
                subprocess.Popen([sys.executable,'-m','nlp_lab.ie_worker','train',str(run)],
                    cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                    env={**os.environ,'PYTHONIOENCODING':'utf-8','HF_HUB_OFFLINE':'1'})
        except (ValueError,OSError) as e:
            if 'run' in locals():update(run,status='failed',error=str(e))
            raise HTTPException(400,str(e)) from e
        return read_json(run/'run.json')
    return routes
