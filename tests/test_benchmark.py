import hashlib
import json

import pytest

from nlp_lab.benchmark import heldout_duplicates
from nlp_lab.data import digest, load_dataset, prepare, read_json, save_json


def test_benchmark_keeps_only_explicitly_permitted_heldout_duplicates(tmp_path):
    records = [{"text": f"类别 {label} 样本 {i}", "label": label} for label in ("甲", "乙") for i in range(5)]
    source = tmp_path / "data.jsonl"
    source.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    data = tmp_path / "prepared"
    prepare(source, data)
    test = data / "test.jsonl"
    line = test.read_text(encoding="utf-8").splitlines()[0]
    with test.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    manifest = read_json(data / "manifest.json")
    manifest["split_sha256"]["test"] = digest(test)
    manifest["dataset_sha256"] = hashlib.sha256(json.dumps(manifest["split_sha256"], sort_keys=True).encode()).hexdigest()
    save_json(data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="重复"):
        load_dataset(data)
    manifest["duplicate_policy"] = "retain_within_evaluation_only"
    save_json(data / "manifest.json", manifest)
    _, rows = load_dataset(data)
    assert len(rows["test"]) == 3
    assert len(heldout_duplicates(rows["test"])) == 1
