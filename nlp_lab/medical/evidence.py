"""Conservative evidence candidates; do not convert co-occurrence into causal claims."""
from collections import defaultdict
import hashlib
import re

from .documents import sentences

CUES={
    'population':r'纳入|入组|患者|受试者|病例|人群|participants|patients',
    'intervention':r'接受|给予|口服|服用|治疗组|干预组|联合|对照组|安慰剂|intervention|placebo',
    'efficacy':r'生存期|缓解率|有效率|无进展|终点|疗效|(?<![A-Za-z])(?:OS|PFS|ORR)(?![A-Za-z])|response rate|survival',
    'adverse_event':r'不良(?:事件|反应)|毒(?:副作用|性)|安全性|(?<![A-Za-z])(?:AEs?|TRAEs?|SAEs?)(?![A-Za-z])|adverse (?:event|reaction)|toxicity',
}
QUALIFIERS={
    'negation':r'未(?:能|见|达到|发生|提高|改善|接受|证明|提供|发现|观察)|没有|无(?:显著|统计学|明显|差异|需)|不能|不需要|不(?:接受|推荐)|未进一步获益|not (?:significant|improve)|no (?:significant|benefit)',
    'uncertainty':r'可能|提示|尚不|仍需|有待|may\b|might\b',
    'comparison':r'相比|对比|比较|对照|分别|vs\.?|versus|compared',
    'study_design':r'回顾性|随机|前瞻性|病例报告|Meta分析|倾向性评分|retrospective|randomized',
}
NUMBER=r'\d+(?:\.\d+)?'
INTERVAL=r'95%\s*(?:CI|置信区间)\s*[:：]?\s*[（(]?\s*(?:'+NUMBER+r'|NA|NR)\s*[-–~]\s*(?:'+NUMBER+r'|NA|NR)'
UNIT=r'(?:%|％|个月|例|岁|mg|μg|天|周|年)'
NUMBERS=re.compile(NUMBER+r'\s*[（(]\s*'+INTERVAL+r'[)）]\s*'+UNIT+'|'+INTERVAL+r'|(?:HR|OR|RR|P)\s*[=＜<＞>≤≥]\s*'+NUMBER+'|'+NUMBER+r'\s*'+UNIT,re.I)


def numeric_atom(value):
    """Normalize literal statistical notation, without assigning an arm or endpoint."""
    effect=re.fullmatch(r'(HR|OR|RR|P)\s*([=＜<＞>≤≥])\s*(\d+(?:\.\d+)?)',value,re.I)
    if effect:
        return {'kind':'p_value' if effect[1].upper()=='P' else 'effect_estimate','measure':effect[1].upper(),
                'operator':effect[2].replace('＜','<').replace('＞','>'),'value':float(effect[3]),'unit':None}
    interval=re.fullmatch(r'95%\s*(?:CI|置信区间)\s*[:：]?\s*[（(]?\s*('+NUMBER+r'|NA|NR)\s*[-–~]\s*('+NUMBER+r'|NA|NR)',value,re.I)
    if interval:return {'kind':'confidence_interval','level':.95,'lower':float(interval[1]) if interval[1].upper() not in ('NA','NR') else None,
                        'upper':float(interval[2]) if interval[2].upper() not in ('NA','NR') else None,'unit':None,'literal_bounds':[interval[1],interval[2]]}
    quantity_ci=re.fullmatch('('+NUMBER+r')\s*[（(]\s*('+INTERVAL+r')[)）]\s*('+UNIT+')',value,re.I)
    if quantity_ci:return {**numeric_atom(quantity_ci[1]+quantity_ci[3]),'confidence_interval':numeric_atom(quantity_ci[2])}
    quantity=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(%|％|个月|例|岁|mg|μg|天|周|年)',value,re.I)
    if quantity:
        unit=quantity[2];kind='percentage' if unit in ('%','％') else 'count' if unit=='例' else 'age' if unit=='岁' else 'dose' if unit.lower() in ('mg','μg') else 'duration'
        return {'kind':kind,'value':float(quantity[1]),'unit':'%' if unit=='％' else unit}
    return {'kind':'unparsed','literal':value}


def abbreviations(text,entities):
    """Document-local, explicit definitions only; retain collisions for review."""
    aliases=defaultdict(set);definitions=[]
    for m in re.finditer(r'[（(]([^()（）]{1,100})[)）]',text):
        # Typical medical definitions: 中文疾病（English expansion, NSCLC）.
        inner=m.group(1).strip()
        definition=re.fullmatch(r'(?:[A-Za-z][A-Za-z ,\-]+[,，]\s*|简称\s*)?([A-Z][A-Z0-9-]{1,13}s?)',inner)
        if not definition:continue
        tokens=[definition[1]] if definition[1] not in {'HR','OR','RR','CI','P'} else []
        preceding=[e for e in entities if e['end']==m.start()]
        if not preceding: continue
        canonical=max(preceding,key=lambda e:e['end']-e['start'])
        for alias in tokens:
            aliases[(alias,canonical['type'])].add(canonical['text'])
            definitions.append({'alias':alias,'canonical':canonical['text'],'type':canonical['type'],
                                'start':canonical['start'],'end':m.end(),'evidence':text[canonical['start']:m.end()]})
    return aliases,definitions


def normalize(entities,text,glossary=()):
    aliases,definitions=abbreviations(text,entities)
    for entry in glossary:
        if not entry.get('provenance'):raise ValueError('Normalization glossary requires provenance')
        for alias in [entry['canonical'],*entry.get('aliases',[])]: aliases[(alias,entry['type'])].add(entry['canonical'])
    concepts={}
    for e in entities:
        values=aliases.get((e['text'],e['type']),set())
        canonical=next(iter(values)) if len(values)==1 else e['text']
        cid='local:'+hashlib.sha256((e['type']+'\0'+canonical).encode()).hexdigest()[:16]
        e.update(canonical=canonical,concept_id=cid,normalization='ambiguous' if len(values)>1 else 'explicit_alias' if values else 'exact_surface',
                 normalization_candidates=sorted(values))
        concepts.setdefault(cid,{'id':cid,'canonical':canonical,'type':e['type'],'mentions':[]})['mentions'].append(e['id'])
    return list(concepts.values()),definitions


def candidates(document,entities):
    from .bindings import suggestions
    from .context import provenance_context
    records=[]
    for sentence in sentences(document):
        text=sentence['text']
        # Table rows are evidence, but lack reliable sentence-level arm/endpoint scope.
        fields=[name for name,pattern in CUES.items() if re.search(pattern,text,re.I)]
        if not fields:continue
        mentions=[e['id'] for e in entities if sentence['start']<=e['start']<e['end']<=sentence['end']]
        qualifiers={name:[{'text':m.group(),'start':sentence['start']+m.start(),'end':sentence['start']+m.end()}
                          for m in re.finditer(pattern,text,re.I)] for name,pattern in QUALIFIERS.items()}
        numbers=[{'text':m.group(),'start':sentence['start']+m.start(),'end':sentence['start']+m.end(),'normalized':numeric_atom(m.group()),
                  'arm':None,'endpoint':None,'binding_status':'unassigned'}
                 for m in NUMBERS.finditer(text)]
        for category in fields:
            bindings,abstentions=suggestions(sentence) if category in ('efficacy','adverse_event') else ([],[])
            bindings=[b for b in bindings if (b['endpoint'] in ('OS','PFS','ORR'))==(category=='efficacy')]
            records.append({'id':f'r{len(records)}','category':category,'evidence':sentence,'mentions':mentions,
                            'numbers':numbers,'qualifiers':qualifiers,'status':'pending',
                            'binding_suggestions':bindings,'binding_abstentions':abstentions,
                            'provenance_context':provenance_context(sentence),
                            'method':'evidence_rules_v1','arm':None,'endpoint':None,'normalized_value':None,
                            'reason':'Group, endpoint and value binding require review; candidate evidence is not a verified medical conclusion.'})
    return records
