from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import urllib.request
import zipfile

from ..data import digest, key, read_json, save_json

DATA_URL = "https://dataset-bj.cdn.bcebos.com/qianyan/DuIE_2_0.zip"
ARCHIVE_SHA256 = "df223b18f3bd0fe19a61d18d7d59a22cd3f145c342c58470e93851bb4d1b40ed"
DATASET = "duie2-slots-10k-v1"
MODEL = "BAAI/bge-small-zh-v1.5"


def download(workspace):
    root = workspace / "sources" / "duie2"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "DuIE_2_0.zip"
    if not target.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(DATA_URL, timeout=60) as r, target.with_suffix(".part").open("wb") as f:
                    total = 0
                    while chunk := r.read(1024 * 1024):
                        total += len(chunk)
                        if total > 400_000_000:
                            raise ValueError("来源归档超过大小上限")
                        f.write(chunk)
                target.with_suffix(".part").replace(target)
                break
            except Exception:
                if attempt == 2:
                    raise
    if digest(target) != ARCHIVE_SHA256:
        raise ValueError("DuIE 归档与本轮固定 SHA-256 不同")
    with zipfile.ZipFile(target) as z:
        # Exact allowlist; no untrusted archive paths or AppleDouble metadata.
        for name in ("train.json", "dev.json", "schema.json", "License.pdf"):
            member = "DuIE_2_0/" + name
            if z.getinfo(member).file_size > 250_000_000:
                raise ValueError("来源文件过大")
            (root / name).write_bytes(z.read(member))
    save_json(root / "provenance.json", {"url": DATA_URL, "sha256": ARCHIVE_SHA256,
              "source_link": "https://github.com/PaddlePaddle/PaddleNLP/issues/447",
              "checksum_origin": "本轮实际下载字节的 SHA-256，非声称数据方发布了此校验值",
              "license": "数据包内 License.pdf；保留来源，语料不提交 Git、不再分发"})
    return root


def schema_items(root):
    raw = [json.loads(x) for x in (root / "schema.json").read_text(encoding="utf-8").splitlines()]
    schemas = []
    for item in raw:
        for slot, object_type in item["object_type"].items():
            schemas.append({"predicate": item["predicate"], "slot": slot,
                            "subject_type": item["subject_type"], "object_type": object_type})
    schemas.sort(key=lambda x: (x["predicate"], x["slot"], x["subject_type"], x["object_type"]))
    for i, item in enumerate(schemas):
        item["id"] = i
        item["label"] = item["predicate"] + ("" if item["slot"] == "@value" else " / " + item["slot"])
    types = sorted({x[t] for x in schemas for t in ("subject_type", "object_type")})
    return schemas, types


def convert_record(record, schema_lookup, source_file, source_line):
    """Derive first exact occurrence offsets, preserving original text byte-for-byte."""
    text = record["text"]
    triples, ambiguous = set(), 0
    if not isinstance(text, str) or not text.strip() or not record.get("spo_list"):
        raise ValueError("空文本或没有标注关系")
    for spo in record["spo_list"]:
        subject = str(spo["subject"])
        if not subject or subject not in text:
            raise ValueError("subject 无法与原文精确对齐")
        sh = text.index(subject)
        ambiguous += int(text.count(subject) > 1)
        for slot, value in spo["object"].items():
            obj = str(value)
            if not obj or obj not in text:
                raise ValueError("object 槽位无法与原文精确对齐")
            object_type = spo["object_type"][slot]
            schema_key = (spo["predicate"], slot, spo["subject_type"], object_type)
            if schema_key not in schema_lookup:
                raise ValueError("来源标注与 schema 不同")
            oh = text.index(obj)
            ambiguous += int(text.count(obj) > 1)
            triples.add((sh, sh + len(subject), spo["subject_type"], schema_lookup[schema_key], oh, oh + len(obj), object_type))
    result = {"text": text, "triples": [list(x) for x in sorted(triples)],
              "source_file": source_file, "source_line": source_line, "ambiguous_mentions": ambiguous}
    return result


def entities(row):
    return sorted({(sh, st, typ) for sh, st, typ, _, _, _, _ in row["triples"]} |
                  {(oh, ot, typ) for _, _, _, _, oh, ot, typ in row["triples"]})


def token_spans(offsets, row):
    starts = {a: i for i, (a, b) in enumerate(offsets) if b > a}
    ends = {b: i for i, (a, b) in enumerate(offsets) if b > a}
    mapped = {}
    for a, b, typ in entities(row):
        if a not in starts or b not in ends or starts[a] > ends[b]:
            raise ValueError("实体边界不与 tokenizer 边界对齐")
        mapped[(a, b, typ)] = (starts[a], ends[b])
    return mapped


def prepare(workspace: Path, model=MODEL, train_budget=10000, validation_budget=2000, test_budget=4000, seed=1729):
    from transformers import AutoTokenizer
    directory = workspace / "ie" / "datasets" / DATASET
    if directory.exists():
        manifest, _ = load(directory)
        if manifest["model"] != model or manifest["budgets"] != [train_budget, validation_budget, test_budget]:
            raise ValueError("已有 IE 版本预算不同，请建立新版本")
        return directory
    root = download(workspace)
    schemas, types = schema_items(root)
    lookup = {(x["predicate"], x["slot"], x["subject_type"], x["object_type"]): x["id"] for x in schemas}
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("信息抽取必须使用支持 offset_mapping 的 fast tokenizer")
    audit, pools, statistics = [], {}, {}
    for filename in ("dev.json", "train.json"):
        pool, seen, counts = [], {}, Counter()
        buffer = []
        def flush():
            if not buffer:
                return
            encoded = tokenizer([x["text"] for x in buffer], return_offsets_mapping=True, truncation=False)
            for row, ids, offsets in zip(buffer, encoded["input_ids"], encoded["offset_mapping"]):
                try:
                    if len(ids) > 128:
                        raise ValueError("超出本轮 128 token 数据预算")
                    token_spans(offsets, row)
                    k = key(row["text"])
                    if k in seen:
                        raise ValueError("同来源规范化重复，保留首条")
                    if filename == "train.json" and k in pools["dev_keys"]:
                        raise ValueError("train 与来源 dev 规范化重叠")
                    seen[k] = True
                    pool.append(row)
                    counts["accepted"] += 1
                    counts["ambiguous_mentions"] += row["ambiguous_mentions"]
                except ValueError as e:
                    counts[str(e)] += 1
                    audit.append({"file": filename, "line": row["source_line"], "reason": str(e)})
            buffer.clear()
        with (root / filename).open(encoding="utf-8") as f:
            for line, text in enumerate(f, 1):
                counts["input"] += 1
                try:
                    buffer.append(convert_record(json.loads(text), lookup, filename, line))
                except (KeyError, ValueError, TypeError) as e:
                    counts[str(e)] += 1
                    audit.append({"file": filename, "line": line, "reason": str(e)})
                if len(buffer) >= 1024:
                    flush()
        flush()
        pools[filename] = pool
        if filename == "dev.json":
            # Exclude overlap with every source dev text, not only our selected heldout rows.
            pools["dev_keys"] = {key(json.loads(x)["text"]) for x in (root / filename).read_text(encoding="utf-8").splitlines()}
        statistics[filename] = dict(counts)
        print(json.dumps({"file": filename, "audit": dict(counts)}, ensure_ascii=False), flush=True)
    rng = random.Random(seed)
    rng.shuffle(pools["dev.json"])
    rng.shuffle(pools["train.json"])
    if len(pools["dev.json"]) < validation_budget + test_budget or len(pools["train.json"]) < train_budget:
        raise ValueError("清洗后数据不足固定预算")
    partitions = {"train": pools["train.json"][:train_budget], "validation": pools["dev.json"][:validation_budget],
                  "test": pools["dev.json"][validation_budget:validation_budget + test_budget]}
    directory.mkdir(parents=True)
    hashes = {}
    for split, rows in partitions.items():
        target = directory / (split + ".jsonl")
        target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8", newline="\n")
        hashes[split] = digest(target)
    manifest = {"id": DATASET, "task": "typed entities + binary relation slots", "model": model, "schemas": schemas, "entity_types": types,
                "budgets": [train_budget, validation_budget, test_budget], "selection_seed": seed,
                "max_source_tokens": 128, "source_url": DATA_URL, "archive_sha256": ARCHIVE_SHA256, "source_statistics": statistics,
                "counts": {s: {"documents": len(rows), "entities": sum(len(entities(x)) for x in rows),
                              "triples": sum(len(x["triples"]) for x in rows), "ambiguous_mentions": sum(x["ambiguous_mentions"] for x in rows),
                              "relation_support": dict(Counter(str(t[3]) for x in rows for t in x["triples"]))} for s, rows in partitions.items()},
                "split_sha256": hashes, "dataset_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                "limitations": ["官方 test 无公开标签；本轮 validation/test 来自清洗后的官方 dev 的固定不重叠子集",
                                "仅保留原文完整对齐且不超过 128 tokens 的记录，不能称为全量 DuIE 基准",
                                "复杂 object 拆为 predicate/slot 二元关系；不等于官方完整多槽 SPO 指标",
                                "实体位置由原文首次精确匹配推导，重复提及的位置可能与实际语义提及不同",
                                "实体标注只覆盖关系参与者，不是独立完整 NER 语料；不做别名宽松匹配",
                                "精确去重不排除语义近重复或预训练语料重叠"]}
    save_json(directory / "manifest.json", manifest)
    save_json(directory / "audit.json", audit)
    return directory


def load(directory):
    manifest = read_json(directory / "manifest.json")
    rows, seen = {}, set()
    for split, expected in manifest["split_sha256"].items():
        path = directory / (split + ".jsonl")
        if digest(path) != expected:
            raise ValueError("IE 数据指纹不匹配")
        rows[split] = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        for row in rows[split]:
            if key(row["text"]) in seen:
                raise ValueError("IE 数据跨划分或内部重复")
            seen.add(key(row["text"]))
            for a,b,typ,r,c,d,objtyp in row["triples"]:
                schema=manifest["schemas"][r]
                if not (0<=a<b<=len(row["text"]) and 0<=c<d<=len(row["text"])) or (typ,objtyp)!=(schema["subject_type"],schema["object_type"]):
                    raise ValueError("实体位置或 schema 无效")
    expected = hashlib.sha256(json.dumps(manifest["split_sha256"],sort_keys=True).encode()).hexdigest()
    if expected != manifest["dataset_sha256"]:
        raise ValueError("IE 总指纹错误")
    return manifest, rows
