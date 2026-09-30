"""Pinned THUCNews-title benchmark, with source lineage and honest budget selection."""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import digest, key, normalize, read_json, save_json

REPOSITORY = "649453932/Chinese-Text-Classification-Pytorch"
COMMIT = "6cb26819af7b646275aff8a4693676f2849e67f6"
FILES = {
    "class.txt": ("THUCNews/data/class.txt", "4c20bac1a8a6be2c207f0e97be4c279651d76d21"),
    "train.txt": ("THUCNews/data/train.txt", "ffe8979a12879da0747847aca49e9a6c1d29898f"),
    "dev.txt": ("THUCNews/data/dev.txt", "cf7f8de6fc3e3c72284ab59e7ed850f49b7185c0"),
    "test.txt": ("THUCNews/data/test.txt", "9c216ace7d0cf2c89fec1b4c9d49a92dc1791d0a"),
    "LICENSE": ("LICENSE", "434bb7651a77791c223eee550b94201e334b4af6"),
}
CHINESE = dict(finance="财经", realty="房产", stocks="股票", education="教育", science="科技",
               society="社会", politics="时政", sports="体育", game="游戏", entertainment="娱乐")


def download_sources(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)

    def one(item):
        name, (relative, expected) = item
        target = destination / name
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{relative}"
        if not target.exists():
            error = None
            for _ in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "NLP-Training-Lab/0.2"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        content = response.read(15_000_001)
                    if len(content) > 15_000_000:
                        raise ValueError("来源文件超过预设大小")
                    actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                    if actual != expected:
                        raise ValueError(f"{name} 与 GitHub 核验的 blob SHA 不符")
                    temp = target.with_suffix(target.suffix + ".tmp")
                    temp.write_bytes(content)
                    temp.replace(target)
                    break
                except Exception as e:
                    error = e
            else:
                raise RuntimeError(f"无法下载 {url}: {error}")
        content = target.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != expected:
            raise ValueError(f"缓存 {name} 的 blob SHA 校验失败")
        return {"file": name, "url": url, "git_blob_sha": actual, "sha256": digest(target), "bytes": len(content)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, FILES.items()))
    save_json(destination / "sources.json", {"repository": REPOSITORY, "commit": COMMIT, "files": records,
              "data_origin": "THUCNews 的第三方 200,000 条新闻标题子集，并非原始 740,000 篇全文语料",
              "license_note": "保存镜像仓库 LICENSE；仓库代码许可不自动证明原始新闻文本可商业再分发；本项目不提交语料到 Git。"})
    return records


def heldout_duplicates(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row["text"])].append(row)
    return [{"lines": [r["source_line"] for r in group], "labels": sorted({r["label"] for r in group}),
             "conflicting": len({r["label"] for r in group}) > 1}
            for group in groups.values() if len(group) > 1]


def build_benchmark(workspace: Path, train_per_class=1000, selection_seed=1729):
    if train_per_class < 5:
        raise ValueError("每类预算至少 5 条")
    source = workspace / "sources" / "thucnews-titles"
    download_sources(source)
    source_labels = (source / "class.txt").read_text(encoding="utf-8").splitlines()
    labels = sorted(CHINESE.get(x, x) for x in source_labels)
    audit = []
    raw = {}
    for filename, split in (("train.txt", "train"), ("dev.txt", "validation"), ("test.txt", "test")):
        rows = []
        for line, text in enumerate((source / filename).read_text(encoding="utf-8").splitlines(), 1):
            original, label_id = text.rsplit("\t", 1)
            label_id = int(label_id)
            if not 0 <= label_id < len(source_labels) or not normalize(original):
                raise ValueError(f"{filename}:{line} 包含无效来源记录")
            rows.append({"text": normalize(original), "label": CHINESE.get(source_labels[label_id], source_labels[label_id]),
                         "split": split, "source_line": line, "source_file": filename})
        raw[split] = rows
    # Preserve every official mirror test record including duplicates/conflicts.
    test_keys = {key(x["text"]) for x in raw["test"]}
    validation = []
    for row in raw["validation"]:
        if key(row["text"]) in test_keys:
            audit.append({"file": "dev.txt", "line": row["source_line"], "reason": "validation 与 test 规范化重叠；保留 test"})
        else:
            validation.append(row)
    heldout_keys = test_keys | {key(x["text"]) for x in validation}
    train_groups = defaultdict(list)
    for row in raw["train"]:
        train_groups[key(row["text"])].append(row)
    clean_train = []
    for k, group in train_groups.items():
        if k in heldout_keys:
            reason = "train 与 held-out 文本重叠"
        elif len({x["label"] for x in group}) > 1:
            reason = "train 冲突标签"
        else:
            clean_train.append(group[0])
            group, reason = group[1:], "train 内部规范化重复"
        audit.extend({"file": "train.txt", "line": x["source_line"], "reason": reason} for x in group)
    by_label = defaultdict(list)
    for row in clean_train:
        by_label[row["label"]].append(row)
    rng = random.Random(selection_seed)
    selected = []
    for label in labels:
        pool = by_label[label]
        if len(pool) < train_per_class:
            raise ValueError(f"类别 {label} 清洗后不足 {train_per_class} 条")
        selected.extend(rng.sample(pool, train_per_class))
    selected.sort(key=lambda x: x["source_line"])
    destination = workspace / "datasets" / f"thucnews-{train_per_class * len(labels)}-s{selection_seed}"
    if destination.exists():
        manifest = read_json(destination / "manifest.json")
        if manifest.get("selection_seed") != selection_seed or manifest.get("train_per_class") != train_per_class:
            raise ValueError("现有预算数据版本配置不匹配")
        from .data import load_dataset
        load_dataset(destination)
        return destination
    destination.mkdir(parents=True)
    partitions = {"train": selected, "validation": validation, "test": raw["test"]}
    hashes = {}
    for split, rows in partitions.items():
        target = destination / f"{split}.jsonl"
        target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8", newline="\n")
        hashes[split] = digest(target)
    manifest = {
        "schema_version": 2, "labels": labels, "source": "THUCNews title subset", "source_commit": COMMIT,
        "source_sha256": digest(source / "train.txt"), "provenance": "公开真实新闻标题 / THUCNews 10类第三方子集 / 固定预算研究",
        "input_rows": sum(len(x) for x in raw.values()), "accepted_rows": sum(len(x) for x in partitions.values()),
        "rejected_rows": len(audit), "available_clean_train": len(clean_train), "budget_omitted_train": len(clean_train) - len(selected),
        "source_counts": {s: len(x) for s, x in raw.items()}, "train_per_class": train_per_class, "selection_seed": selection_seed,
        "seed": selection_seed, "split_policy": "保留来源 train/dev/test；只从 clean train 按类固定抽样；dev 与 test 重叠行移出 dev",
        "normalization": "NFKC + remove format/control + whitespace; dedup key casefold",
        "duplicate_policy": "retain_within_evaluation_only", "heldout_duplicate_groups": {s: heldout_duplicates(partitions[s]) for s in ("validation", "test")},
        "counts": {s: dict(Counter(x["label"] for x in rows)) for s, rows in partitions.items()}, "split_sha256": hashes,
        "dataset_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "limitations": ["随机预算子集，未使用全部 180,000 原始训练标题", "第三方标题子集，不等于官方 THUCNews 全文基准",
                         "规范化去重不能消除语义近重复；公开语料可能与预训练数据重叠", "原 test 重复或冲突标签保留并披露"],
    }
    save_json(destination / "manifest.json", manifest)
    save_json(destination / "audit.json", audit)
    save_json(destination / "sources.json", read_json(source / "sources.json"))
    return destination
