"""Local end-to-end check with authored examples, never a quality benchmark."""
from pathlib import Path
import json,time,urllib.request,urllib.error
from nlp_lab.data import read_json,save_json

URL='http://127.0.0.1:8778'

def request(path,body=None):
    payload=json.dumps(body,ensure_ascii=False).encode('utf-8') if body is not None else None
    req=urllib.request.Request(URL+path,data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)

def check(output=Path('docs/ie-results')):
    report=read_json(output/'report.json');overview=request('/api/ie')
    assert overview['state']['phase']=='completed' and len(overview['experiments'])==7
    assert overview['state']['selected_run']==report['selected_run']
    assert len(overview['report']['experiments'])==7
    examples=['😀刘慈欣是《三体》的作者。',
              '《霸王别姬》的导演是陈凯歌，主演为张国荣和巩俐。',
              '鲁迅于1881年9月25日出生于浙江绍兴，是中国作家。',
              '《三体》的作者是刘慈欣，出版社是重庆出版社。']
    checks=[]
    for text in examples:
        started=time.perf_counter();result=request('/api/ie/extract',{'text':text})
        assert result['run_id']==report['selected_run'] and result['text']==text
        ents={e['id']:e for e in result['entities']}
        for e in ents.values():
            assert 0<=e['start']<e['end']<=len(text) and text[e['start']:e['end']]==e['text']
            assert 0<=e['confidence']<=1
        for rel in result['relations']:
            assert ents[rel['subject']]['text']==rel['subject_text'] and ents[rel['object']]['text']==rel['object_text']
        assert sum(len(r['mention_links']) for r in result['relation_groups'])==len(result['relations'])
        downloaded=request(result['export_url'])
        assert downloaded['text']==text and downloaded['entities']==result['entities'] and downloaded['relations']==result['relations']
        assert downloaded['run_id']==result['run_id'] and downloaded['relation_groups']==result['relation_groups']
        result.pop('export_url')
        checks.append({'seconds':time.perf_counter()-started,'result':result})
    first=checks[0]['result']
    assert any(r['predicate']=='作者' and r['subject_text']=='三体' and r['object_text']=='刘慈欣' for r in first['relations'])
    assert any(e['text']=='刘慈欣' and e['start']==1 for e in first['entities'])
    for body,expected in [({'text':'甲','run_id':'../escape'},400),({'text':'甲'*6001},422)]:
        try:request('/api/ie/extract',body)
        except urllib.error.HTTPError as error:assert error.code==expected
        else:raise AssertionError('Invalid input accepted')
    artifact={'scope':'手写样例的真实HTTP推理核验，不能用作质量基准；时间含首次CPU加载，样本数不足以宣称生产延迟',
              'selected_run':report['selected_run'],'checks':checks,'passed':['completed7','validation-selected-default','unicode-span','entity-reference','mention-grouping','server-attachment-json-roundtrip','bad-path400','oversize422','author-relation']}
    save_json(output/'service_check.json',artifact)
    for entry in checks:
        print(entry['result']['text'],[(r['subject_text'],r['label'],r['object_text']) for r in entry['result']['relation_groups']])
    return artifact

if __name__=='__main__':check()
