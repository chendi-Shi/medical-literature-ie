"""Frozen factorial experiments, multi-seed comparison and held-out stress tests."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from .data import digest, load_dataset, read_json, save_json
from .metrics import evaluate_predictions
from .training import Predictor, TrainConfig, allocate_run, execute_run, now, save_evaluation, update_run

SUITE_NAME = "thucnews-10k-v2"
VARIANTS = ("ce", "rdrop", "fgm", "rdrop_fgm")
SEEDS = (42, 43, 44)


def uncertainty(probabilities, truth, bins=15):
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum(axis=1, keepdims=True)
    truth = np.asarray(truth)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == truth
    ece, reliability = 0., []
    assignment = np.minimum((confidence * bins).astype(int), bins - 1)
    for i in range(bins):
        mask = assignment == i
        if mask.any():
            accuracy, average = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(accuracy - average)
            reliability.append({"lower": i / bins, "upper": (i + 1) / bins, "samples": int(mask.sum()),
                                "accuracy": accuracy, "confidence": average})
    onehot = np.eye(p.shape[1])[truth]
    return {"ece_15_bins": float(ece), "brier_score": float(np.square(p - onehot).sum(axis=1).mean()),
            "nll": float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-15, 1)).mean()),
            "reliability_bins": reliability}


def fit_temperature(scores, truth):
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=int)
    def nll(log_temperature):
        logits = scores / np.exp(log_temperature)
        return float((logsumexp(logits, axis=1) - logits[np.arange(len(truth)), truth]).mean())
    result = minimize_scalar(nll, bounds=(np.log(.1), np.log(100)), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("验证集温度校准失败")
    return float(np.exp(result.x))


def perturb_text(text, mode, rng):
    if mode == "punctuation":
        return "，".join(text[i:i + 6] for i in range(0, len(text), 6))
    if mode == "char_delete_5pct":
        if len(text) < 3:
            return text
        removed = set(rng.sample(range(len(text)), min(len(text) - 1, max(1, round(len(text) * .05)))))
        return "".join(c for i, c in enumerate(text) if i not in removed)
    if mode == "tail_truncate_30pct":
        return text[:max(1, int(len(text) * .7))]
    if mode == "clean_subset":
        return text
    raise ValueError("未知扰动模式")


def make_perturbations(rows, labels, per_class=200, seed=2027):
    rng = random.Random(seed)
    indices = []
    for label in labels:
        pool = [i for i, row in enumerate(rows) if row["label"] == label]
        indices.extend(rng.sample(pool, min(per_class, len(pool))))
    indices.sort()
    cases = {}
    for mode in ("clean_subset", "punctuation", "char_delete_5pct", "tail_truncate_30pct"):
        mode_rng = random.Random(seed)
        cases[mode] = [{**rows[i], "text": perturb_text(rows[i]["text"], mode, mode_rng), "original_text": rows[i]["text"], "test_index": i}
                       for i in indices]
    return {"selection_seed": seed, "per_class": per_class, "indices": indices, "cases": cases,
            "limitations": "删除和截断可能改变语义；沿用原标签，没有人工复标。这是合成扰动压力测试，不是自然分布外泛化评测。"}


def create_protocol(workspace: Path, dataset: Path):
    manifest, rows = load_dataset(dataset)
    directory = workspace / "research" / SUITE_NAME
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol["dataset_sha256"] != manifest["dataset_sha256"]:
            raise ValueError("冻结协议的数据版本与当前文件不同")
        return directory, protocol
    neural = TrainConfig(backend="transformer", method="full", epochs=2, batch_size=16, accumulation=2,
                         learning_rate=3e-5, max_length=48, seed=42, patience=2, schedule="cosine", warmup_ratio=.06,
                         mixed_precision="fp16", rdrop_alpha=.5, fgm_epsilon=.5, adversarial_weight=.5)
    code_dir = Path(__file__).parent
    protocol = {
        "schema_version": 2, "id": SUITE_NAME, "created_at": now(), "dataset": dataset.name,
        "dataset_sha256": manifest["dataset_sha256"], "split_sha256": manifest["split_sha256"], "labels": manifest["labels"],
        "question": "相同训练样本、优化器预算和小编码器下，R-Drop 与 FGM 能否改善分类与合成扰动稳定性？额外计算成本是否值得？",
        "source": {"repository": "649453932/Chinese-Text-Classification-Pytorch", "commit": manifest["source_commit"]},
        "baseline_models": ["logistic", "linear_svm"], "seeds": list(SEEDS), "variants": list(VARIANTS),
        "neural_config": asdict(neural), "selection": "每次实验按完整 validation Macro-F1 选 checkpoint；方法选择只看三种种子的 validation 平均值",
        "test_policy": "所有配方训练完成后才开始 test；不根据 test 调参；保留全部来源 test 行与标签",
        "robustness": {"per_class": 200, "seed": 2027, "modes": ["clean_subset", "punctuation", "char_delete_5pct", "tail_truncate_30pct"]},
        "calibration": {"method": "single temperature", "temperature_bounds": [.1, 100], "fit_split": "validation", "ece_bins": 15,
                        "selective_validation_coverage": .8, "note": "checkpoint 选择与温度拟合共用 validation；不宣称独立校准集"},
        "paired_bootstrap": {"replicates": 400, "seed": 31415, "ci": .95, "scope": "按类对测试样本配对重采样，固定三种训练种子；不覆盖训练随机性的总体不确定性"},
        "compute_policy": "相同样本和 optimizer-step 预算，不同方法的前向/反向次数不同；另报实际训练时间和显存，非相同 FLOPs 对照",
        "source_code_sha256": {name: digest(code_dir / name) for name in ("training.py", "algorithms.py", "benchmark.py", "metrics.py", "research.py")},
        "novelty": "对已有 R-Drop / FGM 的任务适配与组合实验，非原创算法声明",
    }
    save_json(protocol_path, protocol)
    save_json(directory / "perturbations.json", make_perturbations(rows["test"], manifest["labels"]))
    return directory, protocol


def evaluate_research_run(run, directory, protocol):
    meta = read_json(run / "run.json")
    manifest, rows = load_dataset(Path(meta["dataset_path"]))
    if manifest["dataset_sha256"] != protocol["dataset_sha256"]:
        raise ValueError("评测数据指纹不匹配")
    labels = manifest["labels"]
    predictor = Predictor(run, device="auto")
    y_val = np.array([labels.index(x["label"]) for x in rows["validation"]])
    val_scores = predictor.scores([x["text"] for x in rows["validation"]])
    temperature = fit_temperature(val_scores, y_val)
    val_raw = softmax(val_scores, axis=1)
    val_calibrated = softmax(val_scores / temperature, axis=1)
    threshold = float(np.quantile(val_calibrated.max(axis=1), .2))
    calibration = {"temperature": temperature, "validation_sha256": manifest["split_sha256"]["validation"],
                   "fit_samples": len(y_val), "raw": uncertainty(val_raw, y_val), "calibrated": uncertainty(val_calibrated, y_val),
                   "threshold_for_validation_80pct_coverage": threshold}
    save_json(run / "calibration.json", calibration)
    predictor.temperature = temperature
    start = time.perf_counter()
    test_scores = predictor.scores([x["text"] for x in rows["test"]])
    inference_seconds = time.perf_counter() - start
    test_p = softmax(test_scores / temperature, axis=1)
    y = np.array([labels.index(x["label"]) for x in rows["test"]])
    metrics = save_evaluation(run, "test", rows["test"], test_p, labels, manifest["split_sha256"]["test"])
    accepted = test_p.max(axis=1) >= threshold
    metrics.update({"uncalibrated": uncertainty(softmax(test_scores, axis=1), y), "calibrated": uncertainty(test_p, y),
                    "probability_kind": "validation_temperature_scaled", "temperature": temperature, "inference_seconds": inference_seconds,
                    "texts_per_second": len(y) / max(inference_seconds, 1e-9),
                    "selective": {"validation_threshold": threshold, "test_coverage": float(accepted.mean()), "accepted_samples": int(accepted.sum()),
                                  "test_accuracy": float((test_p.argmax(1)[accepted] == y[accepted]).mean()) if accepted.any() else None}})
    save_json(run / "test_metrics.json", metrics)
    np.savez_compressed(run / "test_arrays.npz", truth=y, prediction=test_p.argmax(1), scores=test_scores.astype(np.float32))
    perturbations = read_json(directory / "perturbations.json")
    robust = {"test_sha256": manifest["split_sha256"]["test"], "perturbations_sha256": digest(directory / "perturbations.json"),
              "limitations": perturbations["limitations"], "cases": {}}
    for mode, samples in perturbations["cases"].items():
        p = predictor.probabilities([x["text"] for x in samples])
        m, details = evaluate_predictions(samples, p, labels)
        robust["cases"][mode] = {"samples": m["samples"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
        save_json(run / f"stress_{mode}_errors.json", [x for x in details if not x["correct"]])
    for mode, values in robust["cases"].items():
        values["delta_macro_f1_vs_same_clean_subset"] = values["macro_f1"] - robust["cases"]["clean_subset"]["macro_f1"]
    save_json(run / "robustness.json", robust)
    update_run(run, research_evaluation="completed", calibration="validation_temperature_scaled")
    del predictor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def macro_f1_fast(truth, prediction, classes):
    matrix = np.bincount(truth * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    denominator = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2 * np.diag(matrix), denominator, out=np.zeros(classes, dtype=float), where=denominator > 0).mean())


def paired_bootstrap(reference_predictions, candidate_predictions, truth, classes, replicates=400, seed=31415):
    reference = np.asarray(reference_predictions)
    candidate = np.asarray(candidate_predictions)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != len(truth):
        raise ValueError("配对 bootstrap 的 seed / 样本顺序不匹配")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(truth == i) for i in range(classes)]
    observed = np.mean([macro_f1_fast(truth, c, classes) - macro_f1_fast(truth, r, classes) for r, c in zip(reference, candidate)])
    differences = []
    for _ in range(replicates):
        index = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups if len(group)])
        differences.append(np.mean([macro_f1_fast(truth[index], c[index], classes) - macro_f1_fast(truth[index], r[index], classes)
                                    for r, c in zip(reference, candidate)]))
    lower, upper = np.quantile(differences, [.025, .975])
    return {"delta_macro_f1": float(observed), "ci95": [float(lower), float(upper)], "replicates": replicates,
            "scope": "按类配对重采样测试行；固定已有训练种子；不含训练种子总体不确定性，也未做多重比较校正"}


def aggregate_results(workspace, directory, protocol, entries):
    details = []
    for entry in entries:
        run = workspace / "runs" / entry["run"]
        meta = read_json(run / "run.json")
        test = read_json(run / "test_metrics.json")
        val = read_json(run / "validation_metrics.json")
        robust = read_json(run / "robustness.json")
        details.append({"variant": entry["variant"], "seed": entry["seed"], "run": entry["run"],
                        "validation_macro_f1": val["macro_f1"], "test_macro_f1": test["macro_f1"], "test_accuracy": test["accuracy"],
                        "calibrated_ece": test["calibrated"]["ece_15_bins"], "uncalibrated_ece": test["uncalibrated"]["ece_15_bins"],
                        "nll": test["calibrated"]["nll"], "training_seconds": meta["seconds"], "test_texts_per_second": test["texts_per_second"],
                        "peak_cuda_memory_mb": meta.get("peak_cuda_memory_mb"), "selective": test["selective"], "robustness": robust["cases"]})
    groups = defaultdict(list)
    for entry in details:
        groups[entry["variant"]].append(entry)
    aggregates = []
    for variant, group in groups.items():
        item = {"variant": variant, "seeds": [x["seed"] for x in group], "runs": [x["run"] for x in group], "n_seeds": len(group)}
        for field in ("validation_macro_f1", "test_macro_f1", "test_accuracy", "calibrated_ece", "uncalibrated_ece", "nll", "training_seconds", "test_texts_per_second"):
            values = [x[field] for x in group]
            item[field] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        item["robustness"] = {mode: {"macro_f1_mean": float(np.mean([x["robustness"][mode]["macro_f1"] for x in group])),
                                          "delta_mean": float(np.mean([x["robustness"][mode]["delta_macro_f1_vs_same_clean_subset"] for x in group])),
                                          "samples": group[0]["robustness"][mode]["samples"]} for mode in group[0]["robustness"]}
        aggregates.append(item)
    comparisons = []
    reference = sorted(groups["ce"], key=lambda x: x["seed"])
    truth = None
    predictions = {}
    for entry in details:
        with np.load(workspace / "runs" / entry["run"] / "test_arrays.npz") as arrays:
            y = arrays["truth"]
            if truth is None:
                truth = y
            elif not np.array_equal(truth, y):
                raise ValueError("评测数组顺序不匹配")
            predictions[entry["run"]] = arrays["prediction"]
    for variant in VARIANTS[1:]:
        candidate = sorted(groups[variant], key=lambda x: x["seed"])
        if [x["seed"] for x in candidate] != [x["seed"] for x in reference]:
            raise ValueError("消融实验种子不匹配")
        interval = paired_bootstrap([predictions[x["run"]] for x in reference], [predictions[x["run"]] for x in candidate],
                                    truth, len(protocol["labels"]), **{k: protocol["paired_bootstrap"][k] for k in ("replicates", "seed")})
        comparisons.append({"reference": "ce", "candidate": variant, **interval})
    selected = max(aggregates, key=lambda x: x["validation_macro_f1"]["mean"])["variant"]
    report = {"protocol_id": protocol["id"], "protocol_sha256": digest(directory / "protocol.json"), "dataset_sha256": protocol["dataset_sha256"],
              "created_at": now(), "test_samples": len(truth), "status": "completed", "details": details, "aggregates": aggregates,
              "paired_comparisons": comparisons, "selected_by_validation": selected,
              "limitations": ["训练仅 10,000 条固定预算，不能与原仓库 180,000 条训练的论文式分数直接比较",
                               "相同 optimizer-step 而非相同 FLOPs；正则化增加计算成本", "仅一个公开新闻标题域，未测试真实业务分布外数据",
                               "合成删除/截断沿用原标签，未人工确认语义不变", "公开数据可能进入预训练；三种种子和条件 bootstrap 不足以证明普适优势"]}
    save_json(directory / "report.json", report)
    return report


def run_suite(workspace: Path, dataset: Path):
    directory, protocol = create_protocol(workspace, dataset)
    source_dir = Path(__file__).parent
    for name, expected in protocol["source_code_sha256"].items():
        if digest(source_dir / name) != expected:
            raise ValueError(f"冻结后算法源码 {name} 发生变化；需要创建新协议，不能混入旧套件")
    lock = directory / ".suite.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    state_path = directory / "state.json"
    previous = read_json(state_path) if state_path.exists() else {"entries": []}
    entries = previous["entries"]
    specs = [(name, 42, asdict(TrainConfig(backend="baseline", baseline_model=name))) for name in protocol["baseline_models"]]
    for seed in protocol["seeds"]:
        for variant in protocol["variants"]:
            config = {**protocol["neural_config"], "seed": seed, "regularization": variant}
            specs.append((variant, seed, config))
    def state(phase, **extra):
        save_json(state_path, {"phase": phase, "updated_at": now(), "entries": entries, "total_experiments": len(specs), **extra})
    try:
        state("training")
        for variant, seed, config in specs:
            existing = next((x for x in entries if x["variant"] == variant and x["seed"] == seed), None)
            run = workspace / "runs" / existing["run"] if existing else None
            if run and read_json(run / "run.json")["status"] != "completed":
                existing["previous_failed_run"] = run.name
                run = None
            if run is None:
                # Reuse only an exact matching completed config/data version.
                for candidate in sorted((workspace / "runs").glob("*"), reverse=True):
                    if not (candidate / "run.json").exists():
                        continue
                    meta = read_json(candidate / "run.json")
                    if meta["status"] == "completed" and meta["dataset_sha256"] == protocol["dataset_sha256"] and meta["config"] == config:
                        run = candidate
                        break
            if run is None:
                run = allocate_run(workspace, dataset, TrainConfig(**config))
            if existing is None:
                existing = {"variant": variant, "seed": seed, "run": run.name}
                entries.append(existing)
            else:
                existing["run"] = run.name
            update_run(run, research_protocol=protocol["id"], research_variant=variant, protocol_sha256=digest(directory / "protocol.json"))
            state("training", active_run=run.name, active_variant=variant, active_seed=seed)
            print(json.dumps({"phase": "training", "variant": variant, "seed": seed, "run": run.name}, ensure_ascii=False), flush=True)
            if read_json(run / "run.json")["status"] != "completed":
                execute_run(run)
            existing["training_status"] = "completed"
        state("evaluating", test_opened_at=now())
        for entry in entries:
            run = workspace / "runs" / entry["run"]
            state("evaluating", active_run=run.name, active_variant=entry["variant"], active_seed=entry["seed"])
            print(json.dumps({"phase": "evaluating", **entry}, ensure_ascii=False), flush=True)
            if read_json(run / "run.json").get("research_evaluation") != "completed":
                evaluate_research_run(run, directory, protocol)
            entry["evaluation_status"] = "completed"
        report = aggregate_results(workspace, directory, protocol, entries)
        state("completed", completed_at=now())
        return report
    except Exception as e:
        state("failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        lock.unlink(missing_ok=True)
