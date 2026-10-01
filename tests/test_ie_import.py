import json
import pytest
from nlp_lab.ie_import import import_data
from nlp_lab.ie.data import load


SCHEMA=[{'predicate':'任职','subject_type':'人物','object_type':{'@value':'机构'}}]
def records():
    return [{'text':f'😀张三在甲公司任职。记录{i}。','spo_list':[{'subject':'张三','subject_type':'人物','predicate':'任职',
             'object':{'@value':'甲公司'},'object_type':{'@value':'机构'}}]} for i in range(30)]
def content(rows):return '\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)
def tokenizer(text,**kwargs):return {'input_ids':list(range(len(text)+2)),'offset_mapping':[(0,0)]+[(i,i+1) for i in range(len(text))]+[(0,0)]}


def test_custom_ie_cleaning_preserves_offsets_negatives_and_fingerprints(tmp_path):
    rows=records();rows += [rows[0],{'text':'没有已知关系。','spo_list':[]},{'text':'未标注文本'}]
    m=import_data(content(rows),SCHEMA,tmp_path,'business-v1',tokenizer=tokenizer)
    assert m['source_statistics']['duplicates_removed']==1 and m['source_statistics']['rejected']==1
    manifest,splits=load(tmp_path/'ie/datasets/business-v1')
    assert sum(len(v) for v in splits.values())==31
    assert sum(c['negative_documents'] for c in manifest['counts'].values())==1
    row=next(r for values in splits.values() for r in values if r['triples'])
    assert row['triples'][0][:3]==[1,3,'人物'] and row['text'][1:3]=='张三'
    with pytest.raises(ValueError,match='已存在'):import_data(content(rows),SCHEMA,tmp_path,'business-v1',tokenizer=tokenizer)


def test_custom_ie_explicit_leakage_and_annotation_conflicts(tmp_path):
    rows=records()
    for i,row in enumerate(rows):row['split']='train' if i<24 else ('validation' if i<27 else 'test')
    duplicate={**rows[0],'split':'test'}
    with pytest.raises(ValueError,match='跨划分'):import_data(content(rows+[duplicate]),SCHEMA,tmp_path,'leaked',tokenizer=tokenizer)
    conflict={**rows[0],'spo_list':[]};m=import_data(content(rows+[conflict]),SCHEMA,tmp_path,'clean',tokenizer=tokenizer)
    assert m['source_statistics']['conflicting_rows']==2
    assert sum(c['documents'] for c in m['counts'].values())==29
    with pytest.raises(ValueError,match='至少'):import_data(content(rows[:5]),SCHEMA,tmp_path,'too-small',tokenizer=tokenizer)
