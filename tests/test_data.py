import json
from pathlib import Path

import pytest

from nlp_lab.data import load_dataset, normalize, prepare


def source(tmp_path, extras=None, explicit=False):
    records = [{"text": f"类别 {label} 的独立样本 {i}", "label": label,
                **({"split": ["test", "validation", "train", "train", "train"][i]} if explicit else {})}
               for label in ["甲", "乙"] for i in range(5)]
    records += extras or []
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    return path


def test_cleaning_audit_and_conflict_quarantine(tmp_path):
    path = source(tmp_path, [{"text": "　类别 甲 的独立样本 ０　", "label": "甲"},
                             {"text": "", "label": "甲"}, {"text": "冲突", "label": "甲"}, {"text": "冲突", "label": "乙"}])
    manifest = prepare(path, tmp_path / "data")
    assert manifest["accepted_rows"] == 10
    assert manifest["rejected_rows"] == 4
    assert normalize(" Ａ\u200b  B\n") == "A B"
    _, rows = load_dataset(tmp_path / "data")
    assert len(rows["train"]) == 6
    assert len(rows["test"]) == 2


def test_reject_cross_split_leakage(tmp_path):
    path = source(tmp_path, [{"text": "类别 甲 的独立样本 0", "label": "甲", "split": "train"}], explicit=True)
    with pytest.raises(ValueError, match="跨划分"):
        prepare(path, tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_reproducible_split_and_tampering_detection(tmp_path):
    path = source(tmp_path)
    a = prepare(path, tmp_path / "a", seed=42)
    b = prepare(path, tmp_path / "b", seed=42)
    assert a["dataset_sha256"] == b["dataset_sha256"]
    with (tmp_path / "a" / "test.jsonl").open("a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="指纹"):
        load_dataset(tmp_path / "a")


def test_unseen_label_and_mixed_splits_rejected(tmp_path):
    path = source(tmp_path, [{"text": "额外文本", "label": "丙", "split": "test"}], explicit=True)
    with pytest.raises(ValueError, match="没有出现"):
        prepare(path, tmp_path / "data")
    path = source(tmp_path, [{"text": "未划分文本", "label": "甲"}], explicit=True)
    with pytest.raises(ValueError, match="混合"):
        prepare(path, tmp_path / "data")
