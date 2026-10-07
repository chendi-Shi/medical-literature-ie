"""Resumable typed GPLinker training; original v1 trainer stays immutable."""
from pathlib import Path
from functools import partial
import gc,math,os,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel,AutoTokenizer
from safetensors.torch import save_file
from ..data import read_json,save_json
from ..training import now
from ..ie.data import load
from ..ie.train import Config,update
from ..ie.encoding import encode_rows,collate
from ..ie.model import Extractor,joint_objective
from ..ie.predict import Predictor
from ..ie.metrics import evaluate


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],torch.tensor(n[1].astype(np.int64)),n[2],n[3],float(n[4])],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],n[1].numpy().astype(np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def atomic_checkpoint(path,state):
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(path)


def fit_joint(run,device='cuda'):
    run=Path(run);meta=read_json(run/'run.json');config=Config(**meta['config'])
    if config.architecture!='joint':raise ValueError('Joint trainer requires joint architecture')
    manifest,rows=load(Path(meta['dataset_path']))
    if manifest['dataset_sha256']!=meta['dataset_sha256']:raise ValueError('Corpus changed')
    lock=run.parents[2]/'.training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    started=time.perf_counter();checkpoint=run/'resume.pt'
    try:
        torch.set_num_threads(4);random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
        if device=='cuda':torch.cuda.manual_seed_all(config.seed)
        tokenizer=AutoTokenizer.from_pretrained(config.model,local_files_only=True,use_fast=True)
        encoder=AutoModel.from_pretrained(config.model,local_files_only=True,add_pooling_layer=False)
        model=Extractor(encoder,len(manifest['entity_types']),len(manifest['schemas']),'joint',config.use_tail).to(device)
        encoder.config.save_pretrained(run);tokenizer.save_pretrained(run/'tokenizer')
        examples=encode_rows(rows['train'],tokenizer,manifest,config.max_length)
        generator=torch.Generator().manual_seed(config.seed)
        loader=DataLoader(examples,batch_size=config.batch_size,shuffle=True,generator=generator,num_workers=0,
                          collate_fn=partial(collate,manifest=manifest,architecture='joint'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.01)
        updates=math.ceil(len(loader)/config.accumulation)*config.epochs;warmup=int(updates*config.warmup_ratio)
        def factor(step):
            if step<warmup:return (step+1)/max(1,warmup)
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,updates-warmup))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
        amp=device=='cuda' and config.mixed_precision=='fp16'
        scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=128.)
        epoch=1;done=0;steps=skips=0;history=[];best=-1.;total_loss=0.;total_samples=0;elapsed=0.;resume_count=0
        epoch_generator=generator.get_state()
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location=device,weights_only=True)
            if saved['dataset_sha256']!=manifest['dataset_sha256'] or saved['config']!=meta['config']:raise ValueError('Resume recipe changed')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
            epoch,done,steps,skips=saved['epoch'],saved['done'],saved['steps'],saved['skips']
            history,best=saved['history'],saved['best'];total_loss,total_samples=saved['total_loss'],saved['total_samples']
            epoch_generator=saved['epoch_generator'].cpu();elapsed=saved['elapsed'];resume_count=saved['resume_count']+1
            saved_rng=saved['rng'];saved_rng['torch']=saved_rng['torch'].cpu();saved_rng['cuda']=[s.cpu() for s in saved_rng['cuda']]
            saved_rng['numpy'][1]=saved_rng['numpy'][1].cpu();restore_rng(saved_rng);del saved;gc.collect()
        update(run,status='running',started_at=meta.get('started_at',now()),device=device,
               model_revision=getattr(encoder.config,'_commit_hash',None),parameters=sum(p.numel() for p in model.parameters()),
               trainable_parameters=sum(p.numel() for p in model.parameters()),resume_count=resume_count,
               checkpoint_every_updates=64,validation_batch_size=1)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        def save_progress(e,b,gen):
            atomic_checkpoint(checkpoint,{'dataset_sha256':manifest['dataset_sha256'],'config':meta['config'],
                'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(),'epoch':e,'done':b,'steps':steps,'skips':skips,'history':history,'best':best,
                'total_loss':total_loss,'total_samples':total_samples,'epoch_generator':gen,'rng':rng_state(),
                'elapsed':elapsed+time.perf_counter()-started,'resume_count':resume_count})
            update(run,resumable_epoch=e,resumable_batches_done=b,optimizer_steps=steps,amp_skipped_updates=skips)
        for current in range(epoch,config.epochs+1):
            generator.set_state(epoch_generator);model.train();optimizer.zero_grad(set_to_none=True);window=0
            for j,(inputs,content,pairs,targets) in enumerate(loader):
                if current==epoch and j<done:continue
                inputs={k:v.to(device) for k,v in inputs.items()};content=content.to(device)
                targets={k:v.to(device) for k,v in targets.items()};n=len(content)
                with torch.amp.autocast(device_type=device,dtype=torch.float16,enabled=amp):
                    outputs=model(inputs,content);loss,_=joint_objective(outputs,targets,config.use_tail)
                if not torch.isfinite(loss):raise ValueError('Nonfinite joint loss')
                scaler.scale(loss*n).backward();window+=n;total_loss+=float(loss.detach())*n;total_samples+=n
                boundary=(j+1)%config.accumulation==0 or j+1==len(loader)
                if boundary:
                    scaler.unscale_(optimizer)
                    for param in model.parameters():
                        if param.grad is not None:param.grad.div_(window)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    old=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale()>=old:steps+=1;scheduler.step()
                    else:skips+=1
                    window=0
                    if steps%64==0:save_progress(current,j+1,epoch_generator)
                if (j+1)%100==0:
                    update(run,current_epoch=current,batches_done=j+1,batches_per_epoch=len(loader),optimizer_steps=steps)
            # Save the full optimizer/RNG state before memory-intensive validation.
            save_progress(current,len(loader),epoch_generator)
            predictor=Predictor(device=device,model=model,tokenizer=tokenizer,manifest=manifest,config=meta['config'])
            predicted=predictor.predict_rows(rows['validation'],entity_threshold=0.,relation_threshold=0.,batch_size=1)
            metrics,errors=evaluate(rows['validation'],predicted,manifest)
            history.append({'epoch':current,'train_loss':total_loss/total_samples,'entity_f1':metrics['entity']['f1'],
                            'relation_f1':metrics['relation']['f1'],'relation_precision':metrics['relation']['precision'],
                            'relation_recall':metrics['relation']['recall'],'optimizer_steps':steps,'amp_skipped_updates':skips})
            save_json(run/'history.json',history);print(history[-1],flush=True)
            if metrics['relation']['f1']>best:
                best=metrics['relation']['f1']
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(run/'model.safetensors'))
                save_json(run/'validation_metrics.json',metrics);update(run,best_epoch=current,best_validation_relation_f1=best)
            del predictor,predicted,metrics,errors
            total_loss=0.;total_samples=0;done=0;epoch_generator=generator.get_state()
            save_progress(current+1,0,epoch_generator)
        update(run,status='completed',finished_at=now(),seconds=round(elapsed+time.perf_counter()-started,3),
               optimizer_steps=steps,amp_skipped_updates=skips,epochs_completed=len(history),
               peak_cuda_memory_mb=round(torch.cuda.max_memory_allocated()/1024**2,1) if device=='cuda' else None)
        return run
    except Exception as error:
        update(run,status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        lock.unlink(missing_ok=True)

