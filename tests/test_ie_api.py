from pathlib import Path
from fastapi.testclient import TestClient
from nlp_lab.api import create_app
from nlp_lab.data import save_json
from nlp_lab.ie.data import DATASET


def test_ie_api_task_separation_and_protected_training(tmp_path):
    root=tmp_path/'ie';run=root/'runs'/'real-run'
    save_json(run/'run.json',{'id':'real-run','task':'information_extraction','dataset':DATASET,
                            'status':'completed','config':{'architecture':'joint','seed':42}})
    save_json(run/'test_metrics.json',{'relation':{'f1':.5},'counts_per_document':[{'gold':2}]})
    save_json(root/'runs'/'smoke'/'run.json',{'id':'smoke','dataset':'smoke','status':'completed'})
    save_json(root/'research'/DATASET/'state.json',{'phase':'training','entries':[{'run':'real-run','seed':42,'name':'joint'}]})
    client=TestClient(create_app(tmp_path))
    assert [x['id'] for x in client.get('/api/ie/runs').json()]==['real-run']
    assert client.get('/api/runs').json()==[]
    detail=client.get('/api/ie/runs/real-run').json()
    assert detail['test_metrics']['relation']['f1']==.5 and 'counts_per_document' not in detail['test_metrics']
    assert client.get('/api/ie').json()['experiments'][0]['task']=='information_extraction'
    (root/'.suite.lock').write_text('123',encoding='utf-8')
    assert client.post('/api/ie/train',json={}).status_code==409
    assert client.post('/api/runs',json={'dataset':'missing'}).status_code==409
    assert client.post('/api/ie/extract',json={'text':'原文','run_id':'../real-run'}).status_code==400
    assert client.post('/api/ie/extract',json={'text':'甲'*6001}).status_code==422
    assert client.post('/api/ie/extract',json={'text':'原文'}).status_code==409
    save_json(run/'run.json',{'id':'real-run','dataset':DATASET,'status':'completed','config':{'architecture':'joint'},'research_protocol':'fixed'})
    assert client.post('/api/ie/runs/real-run/evaluate',json={}).status_code==409


def test_ie_inference_preserves_neural_json_and_reuses_cpu_model(tmp_path,monkeypatch):
    import nlp_lab.ie.predict
    calls=[]
    class MockPredictor:
        def __init__(self,run,device):calls.append((Path(run).name,device))
        def extract(self,text):
            if not text.strip():raise ValueError('空白文本')
            return {'text':text,'entities':[{'id':'e0','text':'甲','type':'人物','start':1,'end':2}],
                    'relations':[],'offset_unit':'Python Unicode code points, end exclusive'}
    monkeypatch.setattr(nlp_lab.ie.predict,'Predictor',MockPredictor)
    save_json(tmp_path/'ie/runs/model-a/run.json',{'status':'completed','config':{'architecture':'joint'}})
    client=TestClient(create_app(tmp_path))
    for _ in range(2):
        result=client.post('/api/ie/extract',json={'text':'😀甲','run_id':'model-a'}).json()
        assert result['text']=='😀甲' and result['entities'][0]['start']==1 and result['run_id']=='model-a'
    assert calls==[('model-a','cpu')]
    save_json(tmp_path/'ie/runs/model-a/thresholds.json',{'entity':.5,'relation':.5})
    assert client.post('/api/ie/extract',json={'text':'😀甲','run_id':'model-a'}).status_code==200
    assert calls==[('model-a','cpu'),('model-a','cpu')]
    assert client.post('/api/ie/extract',json={'text':'   ','run_id':'model-a'}).status_code==400
    assert client.post('/api/ie/extract',json={'text':'甲','run_id':'model-a'},headers={'origin':'https://other.test'}).status_code==403
