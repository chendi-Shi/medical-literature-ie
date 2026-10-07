"""Medical models -> global source mentions -> evidence candidates -> review records."""
from pathlib import Path
import math
import re

from ..data import read_json,digest
from ..ie_context import predictor_for
from .ner import NERPredictor
from .data import windows,TYPE_NAMES
from .evidence import normalize,candidates,QUALIFIERS
from .documents import sentences

RE_TYPES={'疾病':'dis','药物':'dru','症状':'sym','部位':'bod','检查':'ite','手术治疗':'pro','其他治疗':'pro'}


class LiteratureExtractor:
    def __init__(self,workspace,device='cpu'):
        if device=='cpu':
            import torch
            torch.set_num_threads(4)
        root=Path(workspace)/'medical';status=read_json(root/'status.json')
        if status['phase']!='completed':raise ValueError('Medical training/evaluation is not completed')
        ner=root/'runs/ner-v1';relation=Path(status['relation_run']);report=read_json(root/'report.json')
        for name,path in [('ner',ner),('relations',relation)]:
            if digest(path/'model.safetensors')!=report['weights_sha256'][name]:raise ValueError('Medical model checksum changed')
        self.ner=NERPredictor(ner,device=device);self.relation=predictor_for(relation,device=device)
        self.fingerprints=report['weights_sha256'];self.glossary=read_json(root/'glossary.json') if (root/'glossary.json').exists() else []

    def extract(self,document):
        text=document['text'];tasks=[];skipped=[]
        for block in document['blocks']:
            if len(re.findall(r'[\u4e00-\u9fff]',block['text']))<max(3,len(block['text'])*.15):
                skipped.append(block['id']);continue
            for a,b in windows(block['text'],self.ner.tokenizer):tasks.append((block,block['start']+a,block['start']+b))
        if len(tasks)>256:raise ValueError('Document exceeds 256 model windows; split by chapters')
        if not tasks:raise ValueError('No Chinese medical text blocks found')
        texts=[text[a:b] for _,a,b in tasks]
        ner=self.ner.predict(texts)
        rel=self.relation.predict_rows([{'text':t} for t in texts],batch_size=4)
        mentions={};relations={};sentence_spans=sentences(document)
        def mention(a,b,typ,score,method):
            key=(a,b,typ)
            value=mentions.setdefault(key,{'start':a,'end':b,'text':text[a:b],'type':typ,'type_name':TYPE_NAMES.get(typ,typ),'methods':[],'score':score})
            value['score']=max(value['score'],score)
            if method not in value['methods']:value['methods'].append(method)
            return key
        for (block,base,end),ents,triples in zip(tasks,ner,rel):
            for a,b,typ,score in ents:mention(base+a,base+b,typ,score,'medical_nested_ner')
            for sh,st,typ,r,oh,ot,objtyp,score in triples['triples']:
                subject=mention(base+sh,base+st,RE_TYPES.get(typ,typ),score,'medical_relation_participant')
                object_=mention(base+oh,base+ot,RE_TYPES.get(objtyp,objtyp),score,'medical_relation_participant')
                schema=self.relation.manifest['schemas'][r];key=(subject,schema['predicate'],object_)
                lower,upper=min(subject[0],object_[0]),max(subject[1],object_[1])
                scope=next((s for s in sentence_spans if s['start']<=lower<upper<=s['end']),block)
                relation={'subject_key':subject,'object_key':object_,'predicate':schema['predicate'],'score':score,
                          'evidence':{'start':scope['start'],'end':scope['end'],'block':block['id'],
                                      'section':block['section'],'locator':block['locator']},'status':'model_candidate'}
                if key not in relations or score>relations[key]['score']:relations[key]=relation
        ordered=sorted(mentions);entities=[];ids={}
        for i,key in enumerate(ordered):
            entity=mentions[key];entity['id']=ids[key]=f'e{i}';entity['uncalibrated_score']=entity.pop('score')
            entities.append(entity)
        concepts,definitions=normalize(entities,text,self.glossary)
        result=[]
        for relation in relations.values():
            relation['subject']=ids[relation.pop('subject_key')];relation['object']=ids[relation.pop('object_key')]
            relation['uncalibrated_score']=relation.pop('score');a,b=relation['evidence']['start'],relation['evidence']['end']
            relation['evidence']['text']=text[a:b]
            relation['scope_flags']=[name for name,pattern in QUALIFIERS.items() if re.search(pattern,text[a:b],re.I)]
            result.append(relation)
        records=candidates({**document,'blocks':[b for b in document['blocks'] if b['id'] not in skipped or b['kind']=='table_row']},entities)
        return {'document_sha256':document['sha256'],'entities':entities,'concepts':concepts,'alias_definitions':definitions,
                'relations':result,'records':records,'models':self.fingerprints,'model_windows':len(tasks),
                'pipeline_source_sha256':{name:digest(Path(__file__).parent/name) for name in ('documents.py','evidence.py','bindings.py','pipeline.py','ner.py')},
                'skipped_non_chinese_blocks':skipped,'limitations':['CMeIE relation candidates are not drug causality or verified efficacy.',
                    'PICO/outcome/adverse-event evidence candidates use disclosed rules, pending human verification.',
                    'Literal same-clause endpoint/value/arm suggestions remain pending; no cross-block coreference or clinical inference.',
                    'Local exact/explicit-alias concept IDs are not MeSH, ICD or externally validated identifiers.',
                    'Scores are uncalibrated logits; evaluation covers medical corpora, not this full literature workflow.']}
