from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoConfig, AutoTokenizer
from ..data import read_json
from .data import token_spans, entities
from .model import Extractor
from .encoding import candidate_entities, decode_joint


class Predictor:
    def __init__(self,run=None,device='cpu',model=None,tokenizer=None,manifest=None,config=None):
        torch.set_num_threads(4)
        self.device=device
        if run is not None:
            run=Path(run)
            meta=read_json(run/'run.json')
            if meta['status']!='completed':raise ValueError('只能加载已完成的信息抽取实验')
            manifest=read_json(run/'schema.json');config=meta['config']
            encoder=AutoModel.from_config(AutoConfig.from_pretrained(run),add_pooling_layer=False)
            model=Extractor(encoder,len(manifest['entity_types']),len(manifest['schemas']),config['architecture'],config.get('use_tail',True))
            model.load_state_dict(load_file(str(run/'model.safetensors')))
            tokenizer=AutoTokenizer.from_pretrained(run/'tokenizer',local_files_only=True,trust_remote_code=False)
            self.thresholds=read_json(run/'thresholds.json') if (run/'thresholds.json').exists() else {'entity':0.,'relation':0.}
        else:self.thresholds={'entity':0.,'relation':0.}
        self.model=model.to(device).eval();self.tokenizer=tokenizer;self.manifest=manifest;self.config=config
        self.by_types={}
        for rel in manifest['schemas']:self.by_types.setdefault((rel['subject_type'],rel['object_type']),[]).append(rel['id'])

    @torch.inference_mode()
    def predict_rows(self,rows,entity_threshold=None,relation_threshold=None,oracle_entities=False,use_tail=None,batch_size=12):
        eth=self.thresholds['entity'] if entity_threshold is None else entity_threshold
        rth=self.thresholds['relation'] if relation_threshold is None else relation_threshold
        all_predictions=[]
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            encoded=self.tokenizer([x['text'] for x in batch],return_offsets_mapping=True,padding=True,truncation=True,
                                   max_length=self.config['max_length'],return_tensors='pt')
            offsets=encoded.pop('offset_mapping').tolist()
            content=torch.tensor([[b>a for a,b in off] for off in offsets],device=self.device)
            outputs=self.model({k:v.to(self.device) for k,v in encoded.items()},content)
            ent_logits=outputs['entity'].cpu().numpy()
            if self.config['architecture']=='joint':
                heads,tails=outputs['head'].cpu().numpy(),outputs['tail'].cpu().numpy()
                for i,row in enumerate(batch):
                    values={'entity':ent_logits[i],'head':heads[i],'tail':tails[i]}
                    pred=decode_joint(values,offsets[i],self.manifest['schemas'],self.manifest['entity_types'],eth,rth,
                                      self.config.get('use_tail',True) if use_tail is None else use_tail)
                    all_predictions.append(pred)
            else:
                candidate_rows,pair_rows,allowed_rows,caps=[],[],[],[]
                for i,row in enumerate(batch):
                    if oracle_entities:
                        mapped=token_spans(offsets[i],row)
                        candidates=[(*mapped[e],self.manifest['entity_types'].index(e[2]),100.) for e in entities(row)]
                        capped=False
                    else:candidates,capped=candidate_entities(ent_logits[i],offsets[i],eth)
                    pairs={}
                    for sh,st,stype,sc in candidates:
                        for oh,ot,otype,oc in candidates:
                            allowed=self.by_types.get((self.manifest['entity_types'][stype],self.manifest['entity_types'][otype]),[])
                            if allowed:
                                values=pairs.setdefault((sh,st,oh,ot),{})
                                for r in allowed:values[r]=max(values.get(r,-10000.),min(sc,oc))
                    candidate_rows.append(candidates);pair_rows.append(list(pairs));allowed_rows.append(list(pairs.values()));caps.append(capped)
                n=max(1,max(len(x) for x in pair_rows))
                tensor=torch.zeros(len(batch),n,4,dtype=torch.long,device=self.device)
                for i,pairs in enumerate(pair_rows):
                    if pairs:tensor[i,:len(pairs)]=torch.tensor(pairs,device=self.device)
                scores=self.model.pair_scores(outputs['hidden'],tensor).cpu().numpy()
                for i,row in enumerate(batch):
                    triples=[]
                    for j,(sh,st,oh,ot) in enumerate(pair_rows[i]):
                        for r,confidence in allowed_rows[i][j].items():
                            if scores[i,j,r]>rth:
                                s=self.manifest['schemas'][r]
                                triples.append((offsets[i][sh][0],offsets[i][st][1],s['subject_type'],r,
                                                offsets[i][oh][0],offsets[i][ot][1],s['object_type'],min(confidence,float(scores[i,j,r]))))
                    ent=[(offsets[i][h][0],offsets[i][t][1],self.manifest['entity_types'][typ],score) for h,t,typ,score in candidate_rows[i]]
                    all_predictions.append({'entities':ent,'triples':triples,'candidates_capped':caps[i]})
        return all_predictions

    def extract(self,text):
        if not isinstance(text,str) or not text.strip() or len(text)>6000:
            raise ValueError('输入必须是非空中文文本，最多 6,000 字符')
        # Offsets remain global to the original text, including overlapping windows.
        encoded=self.tokenizer(text,return_offsets_mapping=True,return_overflowing_tokens=True,truncation=True,
                               max_length=self.config['max_length'],stride=48)
        windows=[]
        for off in encoded['offset_mapping']:
            content=[(a,b) for a,b in off if b>a]
            if content:
                a,b=content[0][0],content[-1][1];windows.append((a,b))
        if len(windows)>64:raise ValueError('超过 64 个推理窗口，请按段落拆分')
        rows=[{'text':text[a:b]} for a,b in windows]
        predictions=self.predict_rows(rows)
        entity_map,triples={},{}
        for (a,b),pred in zip(windows,predictions):
            for h,t,typ,score in pred['entities']:
                key=(h+a,t+a,typ);entity_map[key]=max(entity_map.get(key,-10000.),score)
            for sh,st,typ,r,oh,ot,objtyp,score in pred['triples']:
                key=(sh+a,st+a,typ,r,oh+a,ot+a,objtyp);triples[key]=max(triples.get(key,-10000.),score)
        def confidence(x):return 1/(1+math.exp(-max(-50,min(50,x))))
        ent=[];ids={}
        for i,((a,b,typ),score) in enumerate(sorted(entity_map.items())):
            ids[(a,b,typ)]='e'+str(i)
            ent.append({'id':'e'+str(i),'text':text[a:b],'type':typ,'start':a,'end':b,'confidence':confidence(score)})
        relations=[]
        for (sh,st,typ,r,oh,ot,objtyp),score in sorted(triples.items()):
            s=self.manifest['schemas'][r]
            relations.append({'subject':ids[(sh,st,typ)],'object':ids[(oh,ot,objtyp)],'predicate':s['predicate'],
                              'slot':s['slot'],'label':s['label'],'subject_text':text[sh:st],'object_text':text[oh:ot],
                              'confidence':confidence(score)})
        return {'text':text,'entities':ent,'relations':relations,'windows':len(windows),'offset_unit':'Python Unicode code points, end exclusive',
                'thresholds':self.thresholds,'confidence_note':'sigmoid of minimum extraction logits, uncalibrated; not factual verification',
                'limitations':'跨窗口关系无法保证抽取；仅支持已训练 schema；这是文本中关系的预测，不是事实核验。',
                'candidate_limit_reached':any(x['candidates_capped'] for x in predictions)}
