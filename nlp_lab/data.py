from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "validation", "test")


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def key(text: str) -> str:
    return normalize(text).casefold()


def read_records(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"text", "label"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV 必须包含 text、label 列")
            return [(i, row) for i, row in enumerate(reader, 2)]
    if path.suffix.lower() != ".jsonl":
        raise ValueError("仅支持 UTF-8 CSV / JSONL")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"第 {i} 行不是合法 JSON: {e.msg}") from e
        rows.append((i, row))
    return rows


def prepare(source: Path, destination: Path, seed=42, provenance="user-provided"):
    if destination.exists():
        raise ValueError("目标数据版本已存在；请使用新的数据集名称")
    groups = defaultdict(list)
    rejected = []
    raw = read_records(source)
    if not raw:
        raise ValueError("数据集为空")
    for line, row in raw:
        reason = None
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not isinstance(row.get("label"), str):
            reason = "text / label 必须是字符串"
        else:
            text, label = normalize(row["text"]), normalize(row["label"])
            if not text or not label:
                reason = "空文本或空标签"
            elif len(text) > 20000 or len(label) > 100:
                reason = "文本或标签过长"
            elif row.get("split") not in (None, "", *SPLITS):
                reason = "未知 split"
        if reason:
            rejected.append({"source_line": line, "reason": reason})
            continue
        groups[key(text)].append({"text": text, "label": label, "split": row.get("split") or None, "source_line": line})

    clean = []
    for group in groups.values():
        if len({x["label"] for x in group}) > 1:
            rejected.extend({"source_line": x["source_line"], "reason": "同一规范化文本存在冲突标签"} for x in group)
            continue
        if len({x["split"] for x in group if x["split"]}) > 1:
            raise ValueError(f"跨划分文本泄漏，来源行 {[x['source_line'] for x in group]}")
        clean.append(group[0])
        rejected.extend({"source_line": x["source_line"], "reason": "规范化重复"} for x in group[1:])
    if not clean:
        raise ValueError("清洗后没有有效样本")
    explicit = [x["split"] is not None for x in clean]
    if any(explicit) and not all(explicit):
        raise ValueError("不可混合指定 split 和未指定 split 的记录")
    labels = sorted({x["label"] for x in clean})
    if len(labels) < 2:
        raise ValueError("分类任务至少需要两个类别")
    if not any(explicit):
        by_label = defaultdict(list)
        for row in clean:
            by_label[row["label"]].append(row)
        rng = random.Random(seed)
        for label in labels:
            samples = sorted(by_label[label], key=lambda x: key(x["text"]))
            if len(samples) < 5:
                raise ValueError(f"类别 {label} 至少需要 5 条独立文本，以划分 train / validation / test")
            rng.shuffle(samples)
            n = max(1, int(len(samples) * .2))
            for i, row in enumerate(samples):
                row["split"] = "test" if i < n else "validation" if i < 2 * n else "train"
    partitions = {split: [x for x in clean if x["split"] == split] for split in SPLITS}
    for split, rows in partitions.items():
        if not rows:
            raise ValueError(f"{split} 划分不能为空")
    train_labels = {x["label"] for x in partitions["train"]}
    if train_labels != set(labels):
        raise ValueError("验证集或测试集含训练中没有出现的标签")
    destination.mkdir(parents=True)
    hashes = {}
    for split, rows in partitions.items():
        path = destination / f"{split}.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        hashes[split] = digest(path)
    manifest = {
        "schema_version": 1, "source": source.name, "source_sha256": digest(source),
        "provenance": provenance, "seed": seed, "split_policy": "supplied" if all(explicit) else "per-label 60/20/20 (floor, minimum 1)",
        "normalization": "NFKC + remove format/control characters + collapse whitespace; dedup key casefold",
        "input_rows": len(raw), "accepted_rows": len(clean), "rejected_rows": len(rejected), "labels": labels,
        "counts": {s: dict(Counter(x["label"] for x in rows)) for s, rows in partitions.items()},
        "split_sha256": hashes,
        "dataset_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "limitations": ["精确规范化去重不能排除语义近重复", "默认随机划分不等于按用户、文档或时间分组划分"],
    }
    save_json(destination / "manifest.json", manifest)
    save_json(destination / "audit.json", rejected)
    return manifest


def load_dataset(path: Path):
    manifest = read_json(path / "manifest.json")
    rows = {}
    seen = {}
    labels = set(manifest["labels"])
    for split in SPLITS:
        file = path / f"{split}.jsonl"
        if digest(file) != manifest["split_sha256"][split]:
            raise ValueError(f"{split} 文件已变化，与数据版本指纹不符")
        rows[split] = [json.loads(x) for x in file.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows[split]:
            raise ValueError(f"空划分: {split}")
        for row in rows[split]:
            k = key(row["text"])
            retained_eval_duplicate = (manifest.get("duplicate_policy") == "retain_within_evaluation_only"
                                       and split in ("validation", "test") and seen.get(k) == split)
            if (k in seen and not retained_eval_duplicate) or row["label"] not in labels or row["split"] != split:
                raise ValueError("数据重复、标签或 split 与数据版本不符")
            seen[k] = split
    expected = hashlib.sha256(json.dumps(manifest["split_sha256"], sort_keys=True).encode()).hexdigest()
    if expected != manifest["dataset_sha256"]:
        raise ValueError("数据版本总指纹不匹配")
    if {x["label"] for x in rows["train"]} != labels:
        raise ValueError("训练集标签不完整")
    return manifest, rows
