from __future__ import annotations
import numpy as np
import torch
from .data import entities, token_spans


def encode_rows(rows, tokenizer, manifest, max_length=160):
    encoded=tokenizer([x['text'] for x in rows],return_offsets_mapping=True,truncation=True,max_length=max_length)
    type_id={x:i for i,x in enumerate(manifest['entity_types'])}
    examples=[]
    for row,ids,offsets,attention in zip(rows,encoded['input_ids'],encoded['offset_mapping'],encoded['attention_mask']):
        positions=token_spans(offsets,row)
        typed=[(*positions[ent],type_id[ent[2]]) for ent in entities(row)]
        triples=[]
        for a,b,typ,r,c,d,objtyp in row['triples']:
            sh,st=positions[(a,b,typ)];oh,ot=positions[(c,d,objtyp)]
            triples.append((sh,st,type_id[typ],r,oh,ot,type_id[objtyp]))
        pairs={}
        for sh,st,stype in typed:
            for oh,ot,otype in typed:
                allowed=[x['id'] for x in manifest['schemas'] if x['subject_type']==manifest['entity_types'][stype] and x['object_type']==manifest['entity_types'][otype]]
                if allowed:
                    pair=(sh,st,oh,ot)
                    values=pairs.setdefault(pair,{'allowed':set(),'positive':set()})
                    values['allowed'].update(allowed)
        for sh,st,_,r,oh,ot,_ in triples:
            pairs[(sh,st,oh,ot)]['positive'].add(r)
        # Keep every positive pair, then deterministic evenly spaced negatives.
        positive=[(p,v) for p,v in sorted(pairs.items()) if v['positive']]
        negative=[(p,v) for p,v in sorted(pairs.items()) if not v['positive']]
        cap=max(64,len(positive))
        if len(negative)>cap-len(positive):
            indices=np.linspace(0,len(negative)-1,cap-len(positive),dtype=int)
            negative=[negative[i] for i in indices]
        pairs=positive+negative
        examples.append({'input_ids':ids,'attention_mask':attention,'offsets':offsets,'typed':typed,
                         'triples':triples,'pairs':pairs,'row':row})
    return examples


def collate(examples, manifest, architecture):
    batch,length=len(examples),max(len(x['input_ids']) for x in examples)
    e,r=len(manifest['entity_types']),len(manifest['schemas'])
    inputs={k:torch.zeros(batch,length,dtype=torch.long) for k in ('input_ids','attention_mask')}
    content=torch.zeros(batch,length,dtype=torch.bool)
    target={'entity':torch.zeros(batch,e,length,length)}
    if architecture=='joint':
        target.update(head=torch.zeros(batch,r,length,length),tail=torch.zeros(batch,r,length,length))
        pair_indices=None
    else:
        count=max(1,max(len(x['pairs']) for x in examples))
        pair_indices=torch.zeros(batch,count,4,dtype=torch.long)
        target.update(pair=torch.zeros(batch,count,r),pair_mask=torch.zeros(batch,count,r))
    for i,x in enumerate(examples):
        n=len(x['input_ids'])
        for k in inputs:inputs[k][i,:n]=torch.tensor(x[k])
        content[i,:n]=torch.tensor([b>a for a,b in x['offsets']])
        for sh,st,typ in x['typed']:target['entity'][i,typ,sh,st]=1
        if architecture=='joint':
            for sh,st,_,rel,oh,ot,_ in x['triples']:
                target['head'][i,rel,sh,oh]=1;target['tail'][i,rel,st,ot]=1
        else:
            for j,(pair,values) in enumerate(x['pairs']):
                pair_indices[i,j]=torch.tensor(pair)
                target['pair'][i,j,list(values['positive'])]=1
                target['pair_mask'][i,j,list(values['allowed'])]=1
    return inputs,content,pair_indices,target


def candidate_entities(logits, offsets, threshold=0., cap=64, max_width=64):
    types,heads,tails=np.where(logits>threshold)
    candidates=[]
    for typ,h,t in zip(types,heads,tails):
        if h<=t and t-h+1<=max_width and offsets[h][1]>offsets[h][0] and offsets[t][1]>offsets[t][0]:
            candidates.append((int(h),int(t),int(typ),float(logits[typ,h,t])))
    candidates.sort(key=lambda x:(-x[3],x[:3]))
    return candidates[:cap],len(candidates)>cap


def decode_joint(outputs, offsets, schemas, entity_types, entity_threshold=0., relation_threshold=0., use_tail=True):
    candidates,capped=candidate_entities(outputs['entity'],offsets,entity_threshold)
    triples=[]
    by_types={}
    for rel in schemas:by_types.setdefault((rel['subject_type'],rel['object_type']),[]).append(rel)
    for sh,st,stype,sconfidence in candidates:
        for oh,ot,otype,oconfidence in candidates:
            for rel in by_types.get((entity_types[stype],entity_types[otype]),[]):
                head=float(outputs['head'][rel['id'],sh,oh])
                tail=float(outputs['tail'][rel['id'],st,ot])
                if head>relation_threshold and (not use_tail or tail>relation_threshold):
                    confidence=min(sconfidence,oconfidence,head,tail if use_tail else head)
                    triples.append((offsets[sh][0],offsets[st][1],rel['subject_type'],rel['id'],
                                    offsets[oh][0],offsets[ot][1],rel['object_type'],confidence))
    ent=[(offsets[h][0],offsets[t][1],entity_types[typ],score) for h,t,typ,score in candidates]
    return {'entities':ent,'triples':triples,'candidates_capped':capped}
