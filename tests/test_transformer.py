"""Offline tests with a locally initialized tiny model, not quality benchmarks."""
import json

import numpy as np
import pytest

from nlp_lab.data import load_dataset, prepare, read_json
from nlp_lab.training import Predictor, TrainConfig, allocate_run, evaluate_run, execute_run


@pytest.mark.parametrize("method,regularization", [("full", "ce"), ("lora", "ce"), ("head", "ce"),
                                                  ("full", "rdrop"), ("full", "fgm"), ("full", "rdrop_fgm")])
def test_neural_checkpoint_reload_and_trainable_parameters(tmp_path, method, regularization):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if method == "lora":
        pytest.importorskip("peft")
    torch.set_num_threads(2)
    pretrained = tmp_path / "tiny-pretrained"
    pretrained.mkdir()
    (pretrained / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n球\n队\n赢\n金\n融\n涨\n0\n1\n2\n3\n4\n5\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(vocab=str(pretrained / "vocab.txt"))
    tokenizer.save_pretrained(pretrained)
    model = transformers.BertForSequenceClassification(transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64, num_labels=2,
    ))
    model.save_pretrained(pretrained)
    records = [{"text": ("球队赢" if label == "体育" else "金融涨") + str(i), "label": label}
               for label in ("体育", "财经") for i in range(6)]
    source = tmp_path / "data.jsonl"
    source.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    dataset = tmp_path / "datasets" / "tiny"
    prepare(source, dataset)
    run = allocate_run(tmp_path, dataset, TrainConfig(backend="transformer", model=str(pretrained), method=method,
                                                     epochs=1, max_length=16, batch_size=3, accumulation=3, device="cpu",
                                                     regularization=regularization))
    execute_run(run)
    meta = read_json(run / "run.json")
    assert meta["status"] == "completed"
    # 8 train samples -> 3 batches -> one optimizer step, including the short batch.
    assert meta["optimizer_steps"] == 1
    if method == "full":
        assert meta["trainable_parameters"] == meta["parameters"]
    else:
        assert 0 < meta["trainable_parameters"] < meta["parameters"]
    saved_validation = read_json(run / "validation_metrics.json")
    reloaded_validation = evaluate_run(run, "validation")
    assert saved_validation["log_loss"] == pytest.approx(reloaded_validation["log_loss"], abs=1e-6)
    predictor = Predictor(run)
    p = predictor.probabilities(["球队赢"])
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1)
    assert not (run / "test_metrics.json").exists()
