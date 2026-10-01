import pytest
from nlp_lab.data import save_json
from nlp_lab.ie_commands import main

@pytest.mark.parametrize('lock',['ie/.suite.lock','ie/.context.lock','.training.lock'])
def test_cli_training_respects_research_and_shared_device_locks(tmp_path,monkeypatch,lock):
    target=tmp_path/lock;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('123')
    monkeypatch.setattr('sys.argv',['nlp-ie','--workspace',str(tmp_path),'train'])
    with pytest.raises(ValueError,match='已有训练'):main()
    assert not (tmp_path/'ie/runs').exists()

def test_cli_rejects_frozen_evaluation_before_loading_or_writing(tmp_path,monkeypatch):
    run=tmp_path/'ie/runs/real';save_json(run/'run.json',{'research_protocol':'frozen'})
    monkeypatch.setattr('sys.argv',['nlp-ie','--workspace',str(tmp_path),'evaluate','--run','real'])
    with pytest.raises(ValueError,match='不可覆写'):main()
    assert not (run/'evaluation_status.json').exists()
