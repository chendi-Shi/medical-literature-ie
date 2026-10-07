import pytest
from nlp_lab.medical.documents import parse_document,sentences
from nlp_lab.medical.context import annotate_entities,provenance_context


@pytest.mark.parametrize('text,terms,expected',[
    ('😀未见肺炎，但存在咳嗽。',[('肺炎','dis'),('咳嗽','sym')],['negated','unmarked']),
    ('不能排除肺癌，但无肺炎。',[('肺癌','dis'),('肺炎','dis')],['uncertain','negated']),
    ('否认发热，无咳嗽。',[('发热','sym'),('咳嗽','sym')],['negated','negated']),
    ('无进展生存期增加，肺癌无显著差异。',[('无进展生存期','dis'),('肺癌','dis')],['unmarked','unmarked']),
    ('无论有无肺癌均需要检查。',[('肺癌','dis')],['unmarked']),
    ('无创检查发现肺癌。',[('肺癌','dis')],['unmarked']),
    ('肺炎筛查阴性。',[('肺炎','dis')],['negated']),
    ('未见肺炎、肺结核。',[('肺炎','dis'),('肺结核','dis')],['negated','negated']),
    ('未见肺炎。肺癌患者接受治疗。',[('肺癌','dis')],['unmarked']),
    ('未见疾病改善，给予阿司匹林。',[('阿司匹林','dru')],['unmarked']),
    ('没有提高患者的OS。',[('OS','dis')],['unmarked']),
    ('非小细胞肺癌专家委员会组织对驱动基因阴性讨论。',[('非小细胞肺癌','dis')],['unmarked']),
    ('未见肺癌改善。',[('肺癌','dis')],['unmarked']),
])
def test_scoped_cues_terminate_and_preserve_unicode(text,terms,expected):
    doc=parse_document(text);entities=[{'id':f'e{i}','text':term,'type':typ,'start':text.index(term),'end':text.index(term)+len(term)} for i,(term,typ) in enumerate(terms)]
    annotate_entities(doc,entities)
    assert [e['context']['assertion'] for e in entities]==expected
    for e in entities:
        assert e['context']['status']=='rule_candidate'
        for m in e['context']['modifiers']:
            for key in ('cue','scope'):
                r=m[key];assert doc['text'][r['start']:r['end']]==r['text']


def test_family_history_and_unknown_not_assumed_present():
    text='患者母亲既往有肺癌，本人出现咳嗽。';d=parse_document(text)
    entities=[{'text':x,'type':t,'start':text.index(x),'end':text.index(x)+len(x)} for x,t in [('肺癌','dis'),('咳嗽','sym')]]
    annotate_entities(d,entities)
    assert entities[0]['context']['temporality']=='historical' and entities[0]['context']['experiencer']=='family'
    assert entities[1]['context']['temporality']=='unmarked' and entities[1]['context']['experiencer']=='unmarked'
    assert entities[1]['context']['assertion']=='unmarked'


def test_provenance_flags_cited_studies_without_guessing_trial_assignment():
    d=parse_document('本研究与KEYNOTE-010研究[14]的结果一致。中位OS为10个月。')
    contexts=[provenance_context(s) for s in sentences(d)]
    assert contexts[0]['scope']=='mixed_cues' and contexts[1]['scope']=='unassigned'
    for c in contexts:
        for r in c['cues']:assert d['text'][r['start']:r['end']]==r['text']


def test_entity_type_review_recomputes_rule_scope_and_preserves_raw_model_type(tmp_path):
    from nlp_lab.medical.store import Store
    s=Store(tmp_path/'records.db');d=s.add_document(parse_document('未见阿司匹林。'))
    entities=[{'id':'e0','text':'阿司匹林','type':'sym','canonical':'阿司匹林','start':2,'end':6,'methods':[]}]
    annotate_entities(d,entities)
    assert entities[0]['context']['assertion']=='negated'
    extraction=s.add_extraction(d['id'],{'entities':entities,'records':[],'concepts':[],'relations':[]})
    reviewed=s.review_entity(extraction['id'],'e0',0,'approved','测试核验人','dru','阿司匹林')
    e=reviewed['entities'][0]
    assert e['model_type']=='sym' and e['type']=='dru'
    assert e['context']['assertion']=='unmarked' and e['context']['modifiers']==[]


def test_canonical_review_recomputes_abbreviation_gate(tmp_path):
    from nlp_lab.medical.store import Store
    s=Store(tmp_path/'records.db');d=s.add_document(parse_document('否认ALS。'))
    entities=[{'id':'e0','text':'ALS','type':'dis','canonical':'ALS','start':2,'end':5,'methods':[]}]
    annotate_entities(d,entities)
    assert entities[0]['context']['assertion']=='unmarked'
    assert entities[0]['context']['blocked_modifiers'][0]['reason']=='unverified_disease_surface'
    extraction=s.add_extraction(d['id'],{'entities':entities,'records':[],'concepts':[],'relations':[]})
    reviewed=s.review_entity(extraction['id'],'e0',0,'approved','测试核验人','dis','肌萎缩侧索硬化')
    e=reviewed['entities'][0]
    assert e['canonical']=='肌萎缩侧索硬化' and e['context']['assertion']=='negated'
    assert e['context']['blocked_modifiers']==[]
