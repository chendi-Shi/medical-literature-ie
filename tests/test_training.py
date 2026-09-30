import numpy as np
import pytest
from fastapi.testclient import TestClient

from nlp_lab.api import create_app
from nlp_lab.cli import ROOT, safe_child
from nlp_lab.data import load_dataset, prepare, read_json
from nlp_lab.metrics import evaluate_predictions
from nlp_lab.training import Predictor, TrainConfig, allocate_run, compare_runs, evaluate_run, execute_run


@pytest.fixture
def experiment(tmp_path):
    dataset = tmp_path / "datasets" / "demo"
    prepare(ROOT / "data" / "chinese-demo.jsonl", dataset, provenance="synthetic")
    run = allocate_run(tmp_path, dataset, TrainConfig())
    execute_run(run)
    return tmp_path, dataset, run


def test_baseline_heldout_metrics_and_reload(experiment):
    workspace, dataset, run = experiment
    meta = read_json(run / "run.json")
    assert meta["status"] == "completed"
    assert not (run / "test_metrics.json").exists()
    model = Predictor(run)
    results = model.predict(["球队在决赛中赢得冠军"])
    assert results[0]["label"] in meta["labels"]
    assert sum(results[0]["probabilities"].values()) == pytest.approx(1)
    _, rows = load_dataset(dataset)
    test_only = set(x["text"] for x in rows["test"]) - set(x["text"] for x in rows["train"])
    assert test_only
    test = evaluate_run(run)
    assert test["samples"] == len(rows["test"])
    assert len(read_json(run / "test_predictions.json")) == test["samples"]


def test_heldout_tokens_do_not_enter_fitted_vocabulary(tmp_path):
    import json
    records = [
        {"text": "运动比赛", "label": "体育", "split": "train"},
        {"text": "球员训练", "label": "体育", "split": "train"},
        {"text": "金融股票", "label": "财经", "split": "train"},
        {"text": "银行资金", "label": "财经", "split": "train"},
        {"text": "Ω验证球员", "label": "体育", "split": "validation"},
        {"text": "Ω验证银行", "label": "财经", "split": "validation"},
        {"text": "Ψ盲测比赛", "label": "体育", "split": "test"},
        {"text": "Ψ盲测股票", "label": "财经", "split": "test"},
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    dataset = tmp_path / "datasets" / "sentinel"
    prepare(source, dataset)
    run = allocate_run(tmp_path, dataset, TrainConfig())
    execute_run(run)
    vocabulary = Predictor(run).model.named_steps["tfidf"].vocabulary_
    assert "Ω" not in vocabulary and "Ψ" not in vocabulary
    assert "盲" not in vocabulary and "验" not in vocabulary


def test_reject_comparison_of_different_versions(experiment):
    workspace, dataset, run = experiment
    other = workspace / "datasets" / "other"
    prepare(ROOT / "data" / "chinese-demo.jsonl", other, seed=19)
    second = allocate_run(workspace, other, TrainConfig())
    execute_run(second)
    with pytest.raises(ValueError, match="同一数据版本"):
        compare_runs([run, second])


def test_single_training_lock_and_failure_state(experiment):
    workspace, dataset, _ = experiment
    lock = workspace / ".training.lock"
    lock.write_text("external training", encoding="utf-8")
    run = allocate_run(workspace, dataset, TrainConfig())
    with pytest.raises(FileExistsError):
        execute_run(run)
    assert lock.read_text(encoding="utf-8") == "external training"
    assert read_json(run / "run.json")["status"] == "failed"


def test_metrics_label_order_and_invalid_probabilities():
    rows = [{"text": "a", "label": "甲", "source_line": 1}, {"text": "b", "label": "乙", "source_line": 2}]
    m, details = evaluate_predictions(rows, [[.1, .9], [.9, .1]], ["乙", "甲"])
    assert m["macro_f1"] == 1
    assert all(x["correct"] for x in details)
    with pytest.raises(ValueError):
        evaluate_predictions(rows, [[.1, .1], [.9, .1]], ["乙", "甲"])


def test_local_api_predict_and_path_and_origin_checks(experiment):
    workspace, dataset, run = experiment
    with TestClient(create_app(workspace)) as client:
        assert client.get("/").status_code == 200
        assert len(client.get("/api/datasets").json()) == 1
        assert client.get("/api/runs/" + run.name).json()["status"] == "completed"
        p = client.post(f"/api/runs/{run.name}/predict", json={"texts": ["新款处理器支持人工智能算法"]})
        assert p.status_code == 200 and len(p.json()) == 1
        assert client.post(f"/api/runs/{run.name}/predict", json={"texts": [" "]}).status_code == 400
        assert client.post("/api/runs", json={"dataset": "../demo"}).status_code == 400
        assert client.post(f"/api/runs/{run.name}/predict", headers={"Origin": "https://evil.example"}, json={"texts": ["x"]}).status_code == 403
        assert client.get("/api/runs", headers={"Host": "evil.example"}).status_code == 403
    with pytest.raises(ValueError):
        safe_child(workspace, "../data")
