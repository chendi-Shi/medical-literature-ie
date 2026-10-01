from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from functools import partial
import gc
import math
import os
import random
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

from ..data import read_json, save_json
from ..training import now, versions
from .data import load, MODEL
from .encoding import encode_rows, collate
from .model import Extractor, joint_objective, pipeline_objective
from .predict import Predictor
from .metrics import evaluate


@dataclass
class Config:
    model:str=MODEL
    architecture:str='joint'
    seed:int=42
    epochs:int=6
    batch_size:int=8
    accumulation:int=4
    learning_rate:float=3e-5
    max_length:int=160
    freeze_encoder:bool=False
    use_tail:bool=True
    warmup_ratio:float=.06
    mixed_precision:str='fp16'

    def validate(self):
        if self.architecture not in ('joint','pipeline') or self.mixed_precision not in ('fp32','fp16'):
            raise ValueError('未知 IE 模型或精度')
        if min(self.epochs,self.batch_size,self.accumulation)<1 or not 16<=self.max_length<=512:
            raise ValueError('IE 训练预算无效')
        if not 0<self.learning_rate<1 or not 0<=self.warmup_ratio<1:
            raise ValueError('IE 学习率无效')


def update(run,**changes):
    meta=read_json(run/'run.json');meta.update(changes);save_json(run/'run.json',meta)


def allocate(workspace,dataset,config):
    config.validate()
    manifest,_=load(dataset)
    run=workspace/'ie'/'runs'/(time.strftime('%Y%m%d-%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    save_json(run/'run.json',{'id':run.name,'task':'information_extraction','status':'queued','created_at':now(),
                             'config':asdict(config),'dataset':manifest['id'],'dataset_path':str(dataset.resolve()),
                             'dataset_sha256':manifest['dataset_sha256'],'versions':versions()})
    save_json(run/'schema.json',manifest)
    return run


def fit(run,device=None):
    meta=read_json(run/'run.json');config=Config(**meta['config'])
    manifest,rows=load(Path(meta['dataset_path']))
    if manifest['dataset_sha256']!=meta['dataset_sha256']:raise ValueError('IE 排队后数据变化')
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device=='cpu' and config.mixed_precision!='fp32':raise ValueError('CPU 训练应指定 fp32')
    lock=run.parents[2]/'.training.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    started=time.perf_counter()
    try:
        torch.set_num_threads(4)
        random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(config.seed)
        update(run,status='running',started_at=now(),device=device)
        tokenizer=AutoTokenizer.from_pretrained(config.model,local_files_only=True,trust_remote_code=False,use_fast=True)
        encoder=AutoModel.from_pretrained(config.model,local_files_only=True,trust_remote_code=False,add_pooling_layer=False)
        model=Extractor(encoder,len(manifest['entity_types']),len(manifest['schemas']),config.architecture,config.use_tail)
        if config.freeze_encoder:
            for param in model.encoder.parameters():param.requires_grad=False
        model.to(device)
        encoder.config.save_pretrained(run)
        tokenizer.save_pretrained(run/'tokenizer')
        update(run,model_revision=getattr(encoder.config,'_commit_hash',None),
               parameters=sum(x.numel() for x in model.parameters()),trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad))
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        examples=encode_rows(rows['train'],tokenizer,manifest,config.max_length)
        generator=torch.Generator().manual_seed(config.seed)
        loader=DataLoader(examples,batch_size=config.batch_size,shuffle=True,generator=generator,num_workers=0,
                          collate_fn=partial(collate,manifest=manifest,architecture=config.architecture))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.learning_rate,weight_decay=.01)
        updates=math.ceil(len(loader)/config.accumulation)*config.epochs;warmup=int(updates*config.warmup_ratio)
        def factor(step):
            if step<warmup:return (step+1)/max(1,warmup)
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,updates-warmup))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
        amp=config.mixed_precision=='fp16' and device=='cuda'
        # Loss is accumulated as a sample sum, so use a conservative initial scale.
        scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=128.)
        best=-1.;history=[];steps=skips=0
        for epoch in range(1,config.epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True);window=0;total_loss=0.;total_samples=0
            for j,(inputs,content,pairs,targets) in enumerate(loader):
                inputs={k:v.to(device) for k,v in inputs.items()};content=content.to(device)
                pairs=pairs.to(device) if pairs is not None else None
                targets={k:v.to(device) for k,v in targets.items()}
                n=len(content)
                with torch.amp.autocast(device_type=device,dtype=torch.float16,enabled=amp):
                    outputs=model(inputs,content,pairs)
                    loss,components=(joint_objective(outputs,targets,config.use_tail) if config.architecture=='joint' else pipeline_objective(outputs,targets))
                if not torch.isfinite(loss):raise ValueError('IE 训练出现非有限 loss')
                scaler.scale(loss*n).backward();window+=n
                total_loss+=float(loss.detach())*n;total_samples+=n
                if (j+1)%config.accumulation==0 or j+1==len(loader):
                    scaler.unscale_(optimizer)
                    for param in model.parameters():
                        if param.grad is not None:param.grad.div_(window)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    old=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale()>=old:steps+=1;scheduler.step()
                    else:skips+=1
                    window=0
                if (j+1)%100==0:
                    update(run,current_epoch=epoch,batches_done=j+1,batches_per_epoch=len(loader),optimizer_steps=steps)
                    print({'run':run.name,'epoch':epoch,'batch':j+1,'loss':round(total_loss/total_samples,4)},flush=True)
            predictor=Predictor(device=device,model=model,tokenizer=tokenizer,manifest=manifest,config=asdict(config))
            predicted=predictor.predict_rows(rows['validation'],entity_threshold=0.,relation_threshold=0.)
            metrics,errors=evaluate(rows['validation'],predicted,manifest)
            history.append({'epoch':epoch,'train_loss':total_loss/total_samples,'entity_f1':metrics['entity']['f1'],
                            'relation_f1':metrics['relation']['f1'],'relation_precision':metrics['relation']['precision'],
                            'relation_recall':metrics['relation']['recall'],'optimizer_steps':steps,'amp_skipped_updates':skips})
            save_json(run/'history.json',history);print(history[-1],flush=True)
            if metrics['relation']['f1']>best:
                best=metrics['relation']['f1'];save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(run/'model.safetensors'))
                save_json(run/'validation_metrics.json',metrics);save_json(run/'validation_errors.json',[x for x in errors if x['missing'] or x['extra']])
                update(run,best_epoch=epoch,best_validation_relation_f1=best)
            del predictor,predicted,metrics,errors,outputs,targets
        update(run,status='completed',finished_at=now(),seconds=round(time.perf_counter()-started,3),
               optimizer_steps=steps,amp_skipped_updates=skips,epochs_completed=len(history),
               peak_cuda_memory_mb=round(torch.cuda.max_memory_allocated()/1024**2,1) if device=='cuda' else None)
        del model,encoder,optimizer,scaler
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        return run
    except Exception as e:
        update(run,status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:lock.unlink(missing_ok=True)
