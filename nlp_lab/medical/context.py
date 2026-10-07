"""Chinese character-scope modifiers inspired by ConText's direction/termination.

Original rules, not imported English medspaCy resources. A modifier is a
reviewable language cue, never proof of a patient's diagnosis or causality.
"""
import re

from .documents import sentences

RULES=[
    ('negated',r'未见|未发现|未发生|未出现|没有|否认|无','forward',{'dis','sym','mic'}),
    ('negated',r'阴性','backward',{'dis','mic','ite'}),
    ('uncertain',r'不能排除|不排除|疑似|可能|考虑','forward',{'dis','sym','mic'}),
    ('historical',r'既往患有|既往有|既往患|曾患','forward',{'dis','sym'}),
    ('historical',r'病史','backward',{'dis','sym'}),
    ('family',r'父亲|母亲|家族史|家族中','forward',{'dis','sym'}),
]
PSEUDO=re.compile(r'无进展|无论|有无|无创|无(?:明显|显著|统计学)?差异|无证据证明|没有(?:明显|显著|统计学)?差异')
TERMINATION=re.compile(r'[，,；;。！？!?]|但是|但|然而|不过|却')
MAX_SCOPE=32


def modifiers(sentence):
    text=sentence['text'];base=sentence['start'];boundaries=list(TERMINATION.finditer(text));pseudo=list(PSEUDO.finditer(text));result=[]
    for category,pattern,direction,allowed in RULES:
        for m in re.finditer(pattern,text):
            if any(p.start()<=m.start() and m.end()<=p.end() for p in pseudo):continue
            left=max([0,*[b.end() for b in boundaries if b.end()<=m.start()]])
            right=min([len(text),*[b.start() for b in boundaries if b.start()>=m.end()]])
            if direction=='forward':a,b=m.end(),min(right,m.end()+MAX_SCOPE)
            else:a,b=max(left,m.start()-MAX_SCOPE),m.start()
            result.append({'category':category,'direction':direction,'cue':{'text':m.group(),'start':base+m.start(),'end':base+m.end()},
                           'scope':{'text':text[a:b],'start':base+a,'end':base+b},'allowed_types':sorted(allowed)})
    return result


def entity_context(entity,sentence):
    relevant=[];blocked=[];text=sentence['text'];base=sentence['start']
    for m in modifiers(sentence):
        if entity['type'] not in m['allowed_types'] or not m['scope']['start']<=entity['start']<entity['end']<=m['scope']['end']:continue
        reason=None
        if entity['type']=='dis' and not re.search(r'[\u4e00-\u9fff]',entity.get('canonical',entity['text'])):
            reason='unverified_disease_surface'
        if m['direction']=='backward' and m['cue']['start']-entity['end']>4:
            reason='nonlocal_backward_modifier'
        if m['direction']=='forward' and m['category'] in ('negated','uncertain'):
            between=text[m['cue']['end']-base:entity['start']-base]
            suffix=text[entity['end']-base:min(entity['end']-base+5,m['scope']['end']-base)]
            if re.search(r'改善|提高|降低|延长|发生率|接受|治疗|驱动基因|批准',between) or re.match(r'(?:的)?(?:改善|进展|提高|降低|差异|优势)',suffix):
                reason='outcome_or_indirect_scope'
        if reason:blocked.append({**m,'reason':reason})
        else:relevant.append(m)
    labels={m['category'] for m in relevant}
    assertion='conflicting' if {'negated','uncertain'}<=labels else 'negated' if 'negated' in labels else 'uncertain' if 'uncertain' in labels else 'unmarked'
    return {'assertion':assertion,'temporality':'historical' if 'historical' in labels else 'unmarked',
            'experiencer':'family' if 'family' in labels else 'unmarked','modifiers':relevant,'blocked_modifiers':blocked,
            'method':'chinese_context_rules_v1','status':'rule_candidate'}


def annotate_entities(document,entities):
    spans=sentences(document)
    for entity in entities:
        sentence=next((s for s in spans if s['start']<=entity['start']<entity['end']<=s['end']),None)
        entity['context']=entity_context(entity,sentence) if sentence else {
            'assertion':'unassigned','temporality':'unmarked','experiencer':'unmarked','modifiers':[],
            'method':'chinese_context_rules_v1','status':'rule_candidate'}
    return entities


def provenance_context(sentence):
    text=sentence['text'];base=sentence['start'];matches=[]
    for category,pattern in [('current_study',r'本研究|我们(?:的)?研究'),('citation',r'\[\s*\d+(?:\s*[,，–-]\s*\d+)*\s*\]'),
                             ('named_trial',r'(?<![A-Za-z0-9])(?:KEYNOTE|CheckMate|RATIONALE|ORIENT|OAK|PROLUNG|TORG|LUME)[A-Za-z0-9-]*(?:\s+\d+)?(?![A-Za-z0-9])')]:
        matches.extend({'category':category,'text':m.group(),'start':base+m.start(),'end':base+m.end()} for m in re.finditer(pattern,text,re.I))
    current=any(m['category']=='current_study' for m in matches);external=any(m['category'] in ('citation','named_trial') for m in matches)
    return {'scope':'mixed_cues' if current and external else 'cited_study_cues' if external else 'current_study_cue' if current else 'unassigned',
            'cues':sorted(matches,key=lambda x:x['start']),'section':sentence['section'],'status':'rule_candidate',
            'method':'literal_study_cues_v1','requires_review':True}
