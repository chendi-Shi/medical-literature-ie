import pytest
from fastapi.testclient import TestClient
from nlp_lab.api import create_app
from nlp_lab.medical.documents import parse_document,sentences
from nlp_lab.medical.evidence import candidates,normalize,numeric_atom
from nlp_lab.medical.store import Store,Conflict


def test_jats_nested_sections_tables_and_exact_unicode_evidence():
    xml='''<article><front><article-meta><article-id pub-id-type="pmc">PMC1</article-id><title-group><article-title>真实格式测试</article-title></title-group><abstract><sec><title>结果</title><p>😀ORR为24.39%，P=0.025。</p></sec></abstract></article-meta></front><body><sec><title>安全性</title><p>未见严重不良事件。</p><table-wrap><caption><p>疗效比较</p></caption><table><tr><th>组别</th><th>OS</th></tr><tr><td>单药</td><td>34.93个月</td></tr></table></table-wrap></sec></body><back><ref-list><ref><p>引用中的疗效不应混入本文结果。</p></ref></ref-list></back></article>'''
    d=parse_document(xml,'jats')
    assert d['metadata']['article_ids']['pmc']=='PMC1'
    assert '引用中的' not in d['text']
    assert any(b['kind']=='table_row' and b['text']=='单药 | 34.93个月' for b in d['blocks'])
    for b in d['blocks']:assert d['text'][b['start']:b['end']]==b['text']
    ss=sentences(d)
    assert any(s['text']=='😀ORR为24.39%，P=0.025。' for s in ss)
    for s in ss:assert d['text'][s['start']:s['end']]==s['text']
    records=candidates(d,[])
    adverse=next(r for r in records if r['category']=='adverse_event')
    assert adverse['qualifiers']['negation'][0]['text']=='未见'
    efficacy=next(r for r in records if r['category']=='efficacy')
    assert any(n['text']=='24.39%' for n in efficacy['numbers'])
    assert all(r['status']=='pending' and r['arm'] is None for r in records)
    assert numeric_atom('HR=1.11')=={'kind':'effect_estimate','measure':'HR','operator':'=','value':1.11,'unit':None}
    assert numeric_atom('95%CI: 0.62-2.02')['upper']==2.02
    assert numeric_atom('34.93个月')['kind']=='duration'
    assert numeric_atom('P＜0.05')['operator']=='<'
    assert numeric_atom('27.87（95%CI: 16.41-39.33）个月')['value']==27.87
    real=candidates(parse_document('中位OS达到27.87（95%CI: 16.41-39.33）个月，HR=1.11。两组≥3级AEs无差异。'),[])
    assert real[0]['numbers'][0]['normalized']['confidence_interval']['lower']==16.41
    assert real[1]['category']=='adverse_event' and real[1]['qualifiers']['negation']


@pytest.mark.parametrize('xml',['<!DOCTYPE article [<!ENTITY x "value">]><article/>','<!doctype article><article/>'])
def test_xml_dtd_rejected(xml):
    with pytest.raises(ValueError,match='DTD'):parse_document(xml,'jats')


def test_external_jats_declaration_is_stripped_without_network_fetch():
    raw='<!DOCTYPE article PUBLIC "-//NLM//DTD JATS//EN" "https://invalid.example/jats.dtd"><article><body><p>肺癌患者。</p></body></article>'
    assert parse_document(raw,'jats')['text']=='肺癌患者。'


def test_alias_explicit_definitions_and_collision_abstention():
    text='非小细胞肺癌（non-small cell lung cancer, NSCLC）。NSCLC患者。'
    start=text.index('NSCLC患者')
    entities=[{'id':'e0','text':'非小细胞肺癌','type':'dis','start':0,'end':6},
              {'id':'e1','text':'NSCLC','type':'dis','start':start,'end':start+5}]
    concepts,definitions=normalize(entities,text)
    assert entities[0]['concept_id']==entities[1]['concept_id']
    assert len(concepts)==1 and definitions[0]['alias']=='NSCLC'
    assert definitions[0]['evidence']==text[definitions[0]['start']:definitions[0]['end']]
    normalize(entities,text,[{'canonical':'另一疾病','type':'dis','aliases':['NSCLC'],'provenance':'test glossary'}])
    assert entities[1]['normalization']=='ambiguous' and entities[1]['canonical']=='NSCLC'
    with pytest.raises(ValueError,match='provenance'):normalize(entities,text,[{'canonical':'x','type':'dis'}])


def test_statistics_parentheses_are_not_medical_alias_definitions():
    text='OS（HR=1.11, 95%CI: 0.62-2.02, P=0.722），局部治疗（HR=0.29）。'
    entities=[{'id':'e0','text':'OS','type':'dis','start':0,'end':2}]
    _,definitions=normalize(entities,text)
    assert definitions==[]


def test_store_revision_conflict_source_validation_and_search(tmp_path):
    s=Store(tmp_path/'records.db');d=s.add_document(parse_document('😀肺癌患者未见严重不良事件。'))
    assert s.add_document(parse_document(d['text']))['id']==d['id']
    records=candidates(d,[]);result=s.add_extraction(d['id'],{'records':records})
    rid=records[0]['id'];eid=result['id']
    changed=s.review(eid,rid,0,'approved','研究者',{'endpoint':'严重不良事件','normalized_value':'未见'})
    assert changed['records'][0]['revision']==1
    with pytest.raises(Conflict):s.review(eid,rid,0,'rejected','研究者')
    with pytest.raises(ValueError,match='exact source'):s.review(eid,rid,1,'approved','研究者',{'evidence':{'start':0,'end':1,'text':'肺'}})
    changed=s.review(eid,rid,1,'rejected','第二核验人',{'note':'核验驳回'})
    assert changed['records'][0]['endpoint']=='严重不良事件' and changed['records'][0]['status']=='rejected'
    assert len(s.history(eid,rid))==2
    assert s.extraction(eid)['records'][0]['revision']==2
    assert s.search('肺癌') and s.search('不良事件')
    assert s.search('" OR 1=1 --')==[]


def test_api_medical_ingest_training_gate_and_approved_only_export(tmp_path):
    client=TestClient(create_app(tmp_path))
    d=client.post('/api/medical/documents',json={'content':'😀肺癌患者未见严重不良事件。','title':'测试文档'}).json()
    assert client.post(f"/api/medical/documents/{d['id']}/extract",json={}).status_code==409
    assert client.get('/').status_code==200 and '医学文献' in client.get('/').text
    assert client.get('/lab').status_code==200
    assert client.get('/assets/medical.js').status_code==200
    assert client.post('/api/medical/documents',json={'content':'<article>','format':'jats'}).status_code==400
    store=Store(tmp_path/'medical/literature.sqlite3')
    payload={'records':candidates(d,[]),'entities':[],'concepts':[],'relations':[{'predicate':'candidate'}]}
    e=store.add_extraction(d['id'],payload);eid=e['id'];rid=e['records'][0]['id']
    assert client.get(f'/api/medical/extractions/{eid}/export').json()['records']==[]
    response=client.post(f'/api/medical/extractions/{eid}/records/{rid}/review',json={'status':'approved','expected_revision':0,'reviewer':'研究者','endpoint':'人群'})
    assert response.status_code==200
    export=client.get(f'/api/medical/extractions/{eid}/export')
    assert 'attachment' in export.headers['content-disposition']
    assert len(export.json()['records'])==1 and export.json()['relations']==[]
    assert export.json()['source_text']==d['text']
    assert client.post(f'/api/medical/extractions/{eid}/records/{rid}/review',json={'status':'rejected','expected_revision':0,'reviewer':'研究者'}).status_code==409


def test_entity_correction_preserves_model_prediction_and_revision_history(tmp_path):
    store=Store(tmp_path/'records.db');d=store.add_document(parse_document('免疫单药。'))
    e=store.add_extraction(d['id'],{'records':[],'entities':[{'id':'e0','text':'免疫单药','type':'dru','canonical':'免疫单药',
        'start':0,'end':4,'concept_id':'initial','methods':['medical_nested_ner']}],'concepts':[],'relations':[]})
    changed=store.review_entity(e['id'],'e0',0,'approved','测试核验人','pro','免疫单药治疗','更正实体类型')
    entity=changed['entities'][0]
    assert entity['type']=='pro' and entity['model_type']=='dru'
    assert entity['revision']==1 and entity['canonical']=='免疫单药治疗'
    assert changed['concepts'][0]['type']=='pro'
    with pytest.raises(Conflict):store.review_entity(e['id'],'e0',0,'approved','测试核验人','pro','免疫单药治疗')
    rejected=store.review_entity(e['id'],'e0',1,'rejected','测试核验人','pro','免疫单药治疗')
    assert rejected['concepts']==[] and len(store.history(e['id'],'e0'))==2
