"""Authored regression cases, not an expert-annotated clinical benchmark."""
import pytest

from nlp_lab.medical.bindings import suggestions
from nlp_lab.medical.documents import parse_document,sentences
from nlp_lab.medical.evidence import candidates
from nlp_lab.medical.exporting import export_payload
from nlp_lab.medical.store import Store


CASES=[
    ('中位OS为12.5个月。',[('OS',12.5,'个月',None)]),
    ('治疗组中位PFS达到8个月。',[('PFS',8.,'个月','治疗组')]),
    ('免疫联合化疗组的中位OS为27.87个月，中位PFS为11.50个月。',
     [('OS',27.87,'个月','免疫联合化疗组'),('PFS',11.5,'个月',None)]),
    ('治疗组和对照组的中位OS分别为12和8个月。',[('OS',12.,'个月','治疗组'),('OS',8.,'个月','对照组')]),
    ('单药组与联合组ORR分别为24%和40%。',[('ORR',24.,'%', '单药组'),('ORR',40.,'%', '联合组')]),
    ('两组ORR分别为29.63%和51.85%。',[('ORR',29.63,'%',None),('ORR',51.85,'%',None)]),
    ('ORR为22.6% vs 7.1%。',[('ORR',22.6,'%',None),('ORR',7.1,'%',None)]),
    ('中位OS为17.2 vs 11.9个月。',[('OS',17.2,'个月',None),('OS',11.9,'个月',None)]),
    ('中位总生存期为13个月。',[('OS',13.,'个月',None)]),
    ('无进展生存期为6周。',[('PFS',6.,'周',None)]),
    ('客观缓解率为20％。',[('ORR',20.,'%',None)]),
    ('治疗组总体不良事件发生率为62.86%。',[('总体不良事件发生率',62.86,'%','治疗组')]),
    ('对照组≥3级AEs发生率为14.29%。',[('≥3级AEs发生率',14.29,'%','对照组')]),
    ('治疗组总体不良反应（adverse events, AEs）发生率为62.86%。',[('总体不良反应',62.86,'%','治疗组')]),
    ('治疗组任意等级AEs为44例（62.86%）。',[('任意等级AEs',62.86,'%','治疗组')]),
    ('治疗组AEs为44例。',[]),
    ('未见显著OS优势，P=0.722。',[]),
    ('中位OS未达到。',[]),
    ('中位PFS未达到10个月。',[]),
    ('中位OS为10。',[]),
    ('中位OS为90%。',[]),
    ('ORR为12个月。',[]),
    ('中位OS为10个月和20天。',[]),
    ('中位PFS分别为1个月和2个月和3个月。',[]),
    ('治疗组有80例患者，ORR未报告。',[]),
    ('因OS改善而给予5mg药物。',[]),
    ('治疗组和对照组OS为12个月和8个月。',[('OS',12.,'个月',None),('OS',8.,'个月',None)]),
    ('OS为12个月，患者平均年龄75岁，PFS为8个月。',[('OS',12.,'个月',None),('PFS',8.,'个月',None)]),
]


@pytest.mark.parametrize('text,expected',CASES)
def test_literal_binding_and_abstention(text,expected):
    doc=parse_document('😀前言。'+text)
    sentence=sentences(doc)[-1];bindings,_=suggestions(sentence)
    actual=[(b['endpoint'],b['value'],b['unit'],b['arm']['text'] if b['arm'] else None) for b in bindings]
    assert actual==expected
    for b in bindings:
        assert b['status']=='pending'
        for ref in [b['endpoint_reference'],b['value_reference'],b['arm']]:
            if ref:assert doc['text'][ref['start']:ref['end']]==ref['text']


def test_ci_parallel_shared_unit_and_no_cross_sentence_arm():
    d=parse_document('治疗组已入组。中位OS达到27.87（95%CI: 16.41-39.33）个月。')
    bindings,_=suggestions(sentences(d)[1])
    assert bindings[0]['arm'] is None
    assert bindings[0]['confidence_interval']['upper']==39.33
    b,_=suggestions(sentences(parse_document('PFS为3.9（95%CI: 3.2-NR）个月。'))[0])
    assert b[0]['confidence_interval']['upper'] is None
    table={'text':'中位OS为10个月','start':20,'end':31,'kind':'table_row'}
    b,reasons=suggestions(table)
    assert b==[] and reasons[0]['reason']=='table_headers_not_bound'


def test_candidates_separate_suggestions_from_reviewed_fields():
    records=candidates(parse_document('中位OS为10个月，AEs发生率为20%。'),[])
    for r in records:
        assert r['arm'] is None and r['endpoint'] is None and r['normalized_value'] is None
        assert r['status']=='pending'
    efficacy=next(r for r in records if r['category']=='efficacy')
    safety=next(r for r in records if r['category']=='adverse_event')
    assert [b['endpoint'] for b in efficacy['binding_suggestions']]==['OS']
    assert [b['endpoint'] for b in safety['binding_suggestions']]==['AEs发生率']


def test_export_filters_rejected_mentions_and_trims_concepts(tmp_path):
    s=Store(tmp_path/'records.db');d=s.add_document(parse_document('肺癌患者服用药物。'))
    entities=[{'id':f'e{i}','text':'肺癌','type':'dis','canonical':'肺癌','start':0,'end':2,'methods':[]} for i in range(3)]
    records=candidates(d,entities);records[0]['mentions']=['e0','e1']
    r=s.add_extraction(d['id'],{'records':records,'entities':entities,'concepts':[],'relations':[]})
    s.review_entity(r['id'],'e0',0,'rejected','测试研究者','dis','肺癌')
    # Only e1 is retained as a pending supporting entity; e2 is neither cited nor approved.
    s.review(r['id'],r['records'][0]['id'],0,'approved','测试研究者')
    payload=export_payload(s,r['id'])
    assert [e['id'] for e in payload['entities']]==['e1']
    assert payload['records'][0]['mentions']==['e1']
    assert payload['records'][0]['omitted_rejected_mentions']==1
    assert payload['concepts'][0]['mentions']==['e1']
    assert export_payload(s,r['id'],'all')['review_history']['e0'][0]['status']=='rejected'


def test_evidence_revision_rebuilds_offsets_and_search_index(tmp_path):
    s=Store(tmp_path/'records.db');d=s.add_document(parse_document('肺癌患者中位OS为10个月；中位PFS为5个月。'))
    r=s.add_extraction(d['id'],{'records':candidates(d,[]),'entities':[],'concepts':[],'relations':[]})
    rid=next(x['id'] for x in r['records'] if x['category']=='efficacy')
    a=d['text'].index('中位PFS');b=len(d['text'])
    changed=s.review(r['id'],rid,0,'approved','核验人',{'evidence':{'start':a,'end':b,'text':d['text'][a:b]}})
    record=next(x for x in changed['records'] if x['id']==rid)
    assert [n['text'] for n in record['numbers']]==['5个月']
    assert [n['endpoint'] for n in record['binding_suggestions']]==['PFS']
    changed=s.review(r['id'],rid,1,'approved','核验人',{'note':'二次核验'})
    assert next(x for x in changed['records'] if x['id']==rid)['numbers']==record['numbers']
    assert all(hit['record_id']!=rid for hit in s.search('10个月'))
