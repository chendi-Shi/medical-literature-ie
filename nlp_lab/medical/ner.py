"""Nested medical NER; independent supervision rather than relation participants."""
from collections import Counter
from pathlib import Path
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..data import read_json, save_json, digest
from ..training import now
from ..ie.model import EfficientPointer, pointer_loss
from .data import TYPES, TYPE_NAMES, windows


class MedicalNER(nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.encoder = encoder
        self.dropout = nn.Dropout(.1); self.pointer = EfficientPointer(encoder.config.hidden_size, len(TYPES))

    def forward(self, inputs, mask):
        return self.pointer(self.dropout(self.encoder(**inputs).last_hidden_state), mask)


def examples(rows, tokenizer):
    result = []
    for row in rows:
        for a, b in windows(row['text'], tokenizer):
            encoded = tokenizer(row['text'][a:b], return_offsets_mapping=True)
            offsets = encoded.pop('offset_mapping')
            starts = {h: i for i, (h, t) in enumerate(offsets) if t > h}
            ends = {t: i for i, (h, t) in enumerate(offsets) if t > h}
            labels = [(TYPES.index(typ), starts[h-a], ends[t-a]) for h, t, typ in row['entities'] if a <= h < t <= b]
            result.append((encoded, offsets, labels))
    return result


def collate(batch):
    length = max(len(x[0]['input_ids']) for x in batch)
    inputs = {k: torch.zeros(len(batch), length, dtype=torch.long) for k in ('input_ids', 'attention_mask')}
    mask = torch.zeros(len(batch), length, dtype=torch.bool)
    target = torch.zeros(len(batch), len(TYPES), length, length)
    for i, (enc, offsets, labels) in enumerate(batch):
        n = len(offsets)
        for k in inputs: inputs[k][i, :n] = torch.tensor(enc[k])
        mask[i, :n] = torch.tensor([b > a for a, b in offsets])
        for typ, h, t in labels: target[i, typ, h, t] = 1
    return inputs, mask, target


def scores(rows, predictions):
    counts = {typ: Counter() for typ in TYPES}; errors = []
    for row, predicted in zip(rows, predictions):
        gold = {tuple(x) for x in row['entities']}; pred = {tuple(x[:3]) for x in predicted}
        for typ in TYPES:
            g = {x for x in gold if x[2] == typ}; p = {x for x in pred if x[2] == typ}
            counts[typ].update(correct=len(g & p), predicted=len(p), gold=len(g))
        if gold != pred: errors.append({'id': row['id'], 'text': row['text'], 'missing': sorted(gold-pred), 'extra': sorted(pred-gold)})
    def metric(c):
        p = c['correct']/max(1, c['predicted']); r = c['correct']/max(1, c['gold'])
        return {**c, 'precision': p, 'recall': r, 'f1': 2*p*r/max(1e-10, p+r)}
    total = Counter()
    for c in counts.values(): total.update(c)
    return {'documents': len(rows), 'micro': metric(total), 'per_type': {t: metric(c) for t, c in counts.items()}}, errors


class NERPredictor:
    def __init__(self, run=None, model=None, tokenizer=None, device='cpu', threshold=0.):
        self.device = device; self.threshold = threshold
        if run:
            run = Path(run); meta = read_json(run/'run.json')
            if meta['status'] != 'completed': raise ValueError('Medical NER training is incomplete')
            self.threshold = meta['threshold']
            model = MedicalNER(AutoModel.from_config(AutoConfig.from_pretrained(run), add_pooling_layer=False))
            model.load_state_dict(load_file(str(run/'model.safetensors')))
            tokenizer = AutoTokenizer.from_pretrained(run/'tokenizer', local_files_only=True)
        self.model = model.to(device).eval(); self.tokenizer = tokenizer

    @torch.inference_mode()
    def predict(self, texts, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        tasks = [(i, a, b, text[a:b]) for i, text in enumerate(texts) for a, b in windows(text, self.tokenizer)]
        results = [{} for _ in texts]
        for start in range(0, len(tasks), 8):
            batch = tasks[start:start+8]
            encoded = self.tokenizer([x[3] for x in batch], return_offsets_mapping=True, padding=True, return_tensors='pt')
            offsets = encoded.pop('offset_mapping').tolist()
            mask = torch.tensor([[b > a for a, b in off] for off in offsets], device=self.device)
            logits = self.model({k:v.to(self.device) for k,v in encoded.items()}, mask).cpu().numpy()
            for j, (i, a, b, _) in enumerate(batch):
                for typ, h, t in zip(*np.where(logits[j] > threshold)):
                    if h > t or t-h >= 128: continue
                    off = offsets[j]; ent = (a+off[h][0], a+off[t][1], TYPES[typ])
                    results[i][ent] = max(results[i].get(ent, -1e4), float(logits[j,typ,h,t]))
        return [[(*ent, value) for ent, value in sorted(r.items())] for r in results]


def dictionary(train, min_support=3):
    terms = Counter((row['text'][a:b], typ) for row in train for a,b,typ in row['entities'])
    return [(text,typ) for (text,typ), count in sorted(terms.items()) if count >= min_support and len(text) >= 2]


def dictionary_predict(texts, lexicon):
    result = []
    for text in texts:
        matches = set()
        for term, typ in lexicon:
            a = text.find(term)
            while a >= 0:
                matches.add((a,a+len(term),typ,0.)); a = text.find(term,a+1)
        result.append(sorted(matches))
    return result


def train(workspace, epochs=6):
    workspace = Path(workspace); root = workspace/'medical'; dataset = root/'datasets'
    manifest = read_json(dataset/'manifest.json')
    for name, sha in manifest['prepared_sha256'].items():
        if digest(dataset/name) != sha: raise ValueError('Medical data fingerprint mismatch')
    run = root/'runs'/'ner-v1'; run.mkdir(parents=True, exist_ok=False)
    lock = workspace/'.training.lock'
    import os
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f: f.write(str(os.getpid()))
    started = time.perf_counter()
    meta = {'status':'running','seed':42,'epochs':epochs,'batch_size':4,'accumulation':8,'learning_rate':3e-5,
            'encoder':manifest['encoder'],'started_at':now(),'max_length':384,'stride':96,'dataset_manifest_sha256':digest(dataset/'manifest.json')}
    save_json(run/'run.json', meta)
    try:
        torch.set_num_threads(4); random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'; meta['device'] = device
        tokenizer = AutoTokenizer.from_pretrained(meta['encoder'],local_files_only=True,use_fast=True)
        encoder = AutoModel.from_pretrained(meta['encoder'],local_files_only=True,add_pooling_layer=False)
        model = MedicalNER(encoder).to(device); encoder.config.save_pretrained(run); tokenizer.save_pretrained(run/'tokenizer')
        meta['parameters'] = sum(p.numel() for p in model.parameters()); meta['encoder_revision'] = getattr(encoder.config,'_commit_hash',None)
        rows = {s:read_json(dataset/f'ner-{s}.json') for s in ('train','validation')}
        train_examples = examples(rows['train'], tokenizer); meta['training_windows'] = len(train_examples)
        loader = DataLoader(train_examples,batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=collate)
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
        updates = math.ceil(len(loader)/8)*epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/max(1,int(.06*updates)))*.5*(1+math.cos(math.pi*min(1.,s/updates))))
        amp = device == 'cuda'; scaler = torch.amp.GradScaler('cuda',enabled=amp,init_scale=128.)
        best=-1.; history=[]; steps=skips=0
        if amp: torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,epochs+1):
            model.train(); optimizer.zero_grad(set_to_none=True); weight=total=seen=0
            for j,(inputs,mask,target) in enumerate(loader):
                inputs={k:v.to(device) for k,v in inputs.items()}; mask=mask.to(device); target=target.to(device); n=len(mask)
                with torch.amp.autocast(device_type=device,dtype=torch.float16,enabled=amp): loss=pointer_loss(model(inputs,mask),target)
                if not torch.isfinite(loss): raise ValueError('Nonfinite medical NER loss')
                scaler.scale(loss*n).backward(); weight+=n; seen+=n; total+=float(loss.detach())*n
                if (j+1)%8==0 or j+1==len(loader):
                    scaler.unscale_(optimizer)
                    for p in model.parameters():
                        if p.grad is not None: p.grad.div_(weight)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    old=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);weight=0
                    if scaler.get_scale()>=old: steps+=1;scheduler.step()
                    else: skips+=1
                if (j+1)%200==0:
                    meta.update(current_epoch=epoch,batches_done=j+1,batches_per_epoch=len(loader),optimizer_steps=steps)
                    save_json(run/'run.json',meta);print({'ner_epoch':epoch,'batch':j+1,'loss':total/seen},flush=True)
            pred=NERPredictor(model=model,tokenizer=tokenizer,device=device).predict([r['text'] for r in rows['validation']])
            metrics,_=scores(rows['validation'],pred)
            history.append({'epoch':epoch,'loss':total/seen,'validation':metrics,'optimizer_steps':steps,'amp_skipped_updates':skips})
            save_json(run/'history.json',history);print(history[-1]['validation']['micro'],flush=True)
            if metrics['micro']['f1']>best:
                best=metrics['micro']['f1'];meta['best_epoch']=epoch
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(run/'model.safetensors'))
            save_json(run/'run.json',meta)
        model.load_state_dict(load_file(str(run/'model.safetensors')))
        predictor=NERPredictor(model=model,tokenizer=tokenizer,device=device)
        candidates=[]
        for threshold in (0.,-.5,.5):
            metrics,_=scores(rows['validation'],predictor.predict([r['text'] for r in rows['validation']],threshold))
            candidates.append((metrics['micro']['f1'],threshold,metrics))
        chosen=max(candidates,key=lambda x:x[0]);meta['threshold']=chosen[1];save_json(run/'validation.json',chosen[2])
        # First holdout quality access is after checkpoint and threshold selection.
        meta['test_opened_at']=now();test=read_json(dataset/'ner-test.json')
        metrics,errors=scores(test,predictor.predict([r['text'] for r in test],chosen[1]))
        save_json(run/'test.json',metrics);save_json(run/'errors.json',errors)
        lexicon=dictionary(rows['train']); save_json(root/'lexicon.json',lexicon)
        baseline,_=scores(test,dictionary_predict([r['text'] for r in test],lexicon));save_json(run/'dictionary_test.json',baseline)
        meta.update(status='completed',finished_at=now(),optimizer_steps=steps,amp_skipped_updates=skips,
                    seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if amp else None,
                    weights_sha256=digest(run/'model.safetensors'))
        save_json(run/'run.json',meta);return run
    except Exception as e:
        meta.update(status='failed',error=f'{type(e).__name__}: {e}');save_json(run/'run.json',meta);raise
    finally: lock.unlink(missing_ok=True)
