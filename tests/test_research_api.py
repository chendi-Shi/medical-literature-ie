import numpy as np
from fastapi.testclient import TestClient
from scipy.special import softmax

from nlp_lab.api import create_app
from nlp_lab.cli import ROOT
from nlp_lab.data import prepare, save_json
from nlp_lab.training import Predictor, TrainConfig, allocate_run, execute_run, update_run


def test_svm_reload_and_validation_temperature(tmp_path):
    dataset = tmp_path / "datasets" / "demo"
    prepare(ROOT / "data/chinese-demo.jsonl", dataset)
    run = allocate_run(tmp_path, dataset, TrainConfig(baseline_model="linear_svm"))
    execute_run(run)
    texts = ["股票市场上涨", "球队赢得冠军", "芯片技术突破"]
    original = Predictor(run)
    scores = original.scores(texts)
    save_json(run / "calibration.json", {"temperature": .3})
    calibrated = Predictor(run)
    np.testing.assert_allclose(calibrated.probabilities(texts), softmax(scores / .3, axis=1))
    assert [x["label"] for x in original.predict(texts)] == [x["label"] for x in calibrated.predict(texts)]
    assert all(x["temperature_scaled"] for x in calibrated.predict(texts))


def test_research_api_blocks_mutation_and_serves_evidence(tmp_path):
    dataset = tmp_path / "datasets" / "demo"
    prepare(ROOT / "data/chinese-demo.jsonl", dataset)
    run = allocate_run(tmp_path, dataset, TrainConfig(baseline_model="linear_svm"))
    execute_run(run)
    update_run(run, research_protocol="thucnews-10k-v2")
    directory = tmp_path / "research" / "thucnews-10k-v2"
    save_json(directory / "protocol.json", {"dataset": "demo"})
    save_json(directory / "state.json", {"phase": "evaluating", "entries": [{"run": run.name, "variant": "linear_svm", "seed": 42}]})
    with TestClient(create_app(tmp_path)) as client:
        data = client.get("/api/research").json()
        assert data["experiments"][0]["status"] == "completed"
        assert 0 < data["experiments"][0]["validation_macro_f1"] <= 1
        assert client.post("/api/runs", json={"dataset": "demo"}).status_code == 409
        assert client.post(f"/api/runs/{run.name}/evaluate").status_code == 409
        save_json(directory / "state.json", {"phase": "completed", "entries": []})
        assert client.post(f"/api/runs/{run.name}/evaluate").status_code == 409
        assert client.get("/assets/research.js").status_code == 200
        assert client.get("/assets/research.css").status_code == 200
        assert client.get("/assets/private.json").status_code == 404
