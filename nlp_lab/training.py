from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import load_dataset, normalize, read_json, save_json
from .metrics import evaluate_predictions


@dataclass
class TrainConfig:
    backend: str = "baseline"
    model: str = "BAAI/bge-small-zh-v1.5"
    method: str = "full"
    epochs: int = 3
    batch_size: int = 8
    accumulation: int = 1
    learning_rate: float = 2e-5
    max_length: int = 128
    seed: int = 42
    device: str = "auto"
    patience: int = 2
    allow_download: bool = False
    lora_rank: int = 8
    lora_targets: str = "query,value"

    def validate(self):
        if self.backend not in ("baseline", "transformer") or self.method not in ("full", "lora", "head"):
            raise ValueError("未知训练后端或微调方法")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 必须是 auto / cpu / cuda")
        for name in ("epochs", "batch_size", "accumulation", "patience", "lora_rank"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须为正整数")
        if not 8 <= self.max_length <= 4096 or not 0 < self.learning_rate < 1:
            raise ValueError("max_length 或 learning_rate 超出范围")


def now():
    return datetime.now(timezone.utc).isoformat()


def versions():
    values = {"python": platform.python_version()}
    for package in ("numpy", "scikit-learn", "torch", "transformers", "peft"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return values


def allocate_run(workspace: Path, dataset: Path, config: TrainConfig):
    config.validate()
    manifest, _ = load_dataset(dataset)
    run = workspace / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    save_json(run / "run.json", {
        "id": run.name, "status": "queued", "created_at": now(), "config": asdict(config),
        "dataset_path": str(dataset.resolve()), "dataset": dataset.name,
        "dataset_sha256": manifest["dataset_sha256"], "split_sha256": manifest["split_sha256"],
        "labels": manifest["labels"], "provenance": manifest["provenance"], "versions": versions(),
    })
    return run


def update_run(run: Path, **changes):
    meta = read_json(run / "run.json")
    meta.update(changes)
    save_json(run / "run.json", meta)


def save_evaluation(run, split, rows, probabilities, labels, split_sha256):
    metrics, details = evaluate_predictions(rows, probabilities, labels)
    metrics["split"] = split
    metrics["split_sha256"] = split_sha256
    save_json(run / f"{split}_metrics.json", metrics)
    save_json(run / f"{split}_predictions.json", details)
    save_json(run / f"{split}_errors.json", [x for x in details if not x["correct"]])
    return metrics


def execute_run(run: Path):
    meta = read_json(run / "run.json")
    workspace = run.parent.parent
    lock = workspace / ".training.lock"
    acquired = False
    started = time.perf_counter()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        acquired = True
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "run": run.name}))
        update_run(run, status="running", started_at=now())
        config = TrainConfig(**meta["config"])
        manifest, rows = load_dataset(Path(meta["dataset_path"]))
        if manifest["dataset_sha256"] != meta["dataset_sha256"] or manifest["labels"] != meta["labels"]:
            raise ValueError("排队后数据版本发生变化")
        random.seed(config.seed)
        np.random.seed(config.seed)
        if config.backend == "baseline":
            model = Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, max_features=60000)),
                ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=config.seed)),
            ])
            model.fit([x["text"] for x in rows["train"]], [x["label"] for x in rows["train"]])
            joblib.dump(model, run / "model.joblib")
            p = model.predict_proba([x["text"] for x in rows["validation"]])
            # sklearn class order is lexicographic; persist an explicit mapping.
            order = [list(model.classes_).index(label) for label in manifest["labels"]]
            metrics = save_evaluation(run, "validation", rows["validation"], p[:, order], manifest["labels"], manifest["split_sha256"]["validation"])
            save_json(run / "history.json", [{"epoch": 1, "validation_macro_f1": metrics["macro_f1"]}])
            update_run(run, device="cpu", best_epoch=1, best_validation_macro_f1=metrics["macro_f1"])
        else:
            train_transformer(run, config, manifest, rows)
        update_run(run, status="completed", finished_at=now(), seconds=round(time.perf_counter() - started, 3))
    except Exception as e:
        update_run(run, status="failed", error=f"{type(e).__name__}: {e}", finished_at=now())
        raise
    finally:
        if acquired:
            lock.unlink(missing_ok=True)


def torch_device(choice):
    import torch
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；选择 cpu 或 auto")
    return "cuda" if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()) else "cpu"


def train_transformer(run, config, manifest, rows):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch_device(config.device)
    labels = manifest["labels"]
    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=not config.allow_download, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={x: i for i, x in enumerate(labels)},
        local_files_only=not config.allow_download, trust_remote_code=False,
    )
    if config.max_length > model.config.max_position_embeddings:
        raise ValueError("max_length 超过模型的位置编码长度")
    if config.method == "head":
        for param in model.base_model.parameters():
            param.requires_grad = False
    elif config.method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=config.lora_rank, lora_alpha=2 * config.lora_rank,
            lora_dropout=.05, target_modules=[x.strip() for x in config.lora_targets.split(",") if x.strip()],
        ))
    model.to(device)
    encoded = {}
    lengths = {}
    for split in ("train", "validation"):
        texts = [x["text"] for x in rows[split]]
        # Count before truncation; record how many inputs lose content.
        raw_lengths = [len(x) for x in tokenizer(texts, truncation=False)["input_ids"]]
        lengths[split] = {"samples": len(texts), "truncated": sum(x > config.max_length for x in raw_lengths), "max_tokens": max(raw_lengths)}
        encoded[split] = tokenizer(texts, truncation=True, padding="max_length", max_length=config.max_length, return_tensors="pt")
    update_run(run, device=device, tokenization=lengths, model_revision=getattr(model.config, "_commit_hash", None),
               parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    target = torch.tensor([labels.index(x["label"]) for x in rows["train"]])
    samples = [{**{k: v[i] for k, v in encoded["train"].items()}, "labels": target[i]} for i in range(len(target))]
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(samples, batch_size=config.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=.01)
    history, best, stale = [], -1.0, 0
    optimizer_steps = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        # Accumulate summed sample losses then divide gradients by the actual
        # window sample count. This handles short last batches/windows correctly.
        window_samples = 0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            n = len(batch["labels"])
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("训练出现非有限 loss")
            (loss * n).backward()
            window_samples += n
            losses.append((float(loss.detach().cpu()), n))
            if (step + 1) % config.accumulation == 0 or step + 1 == len(loader):
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(window_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                window_samples = 0
        model.eval()
        probs = []
        with torch.inference_mode():
            for start in range(0, len(rows["validation"]), config.batch_size):
                batch = {k: v[start:start + config.batch_size].to(device) for k, v in encoded["validation"].items()}
                probs.extend(torch.softmax(model(**batch).logits.float(), dim=-1).cpu().tolist())
        metrics, _ = evaluate_predictions(rows["validation"], probs, labels)
        history.append({"epoch": epoch, "train_loss": sum(loss * n for loss, n in losses) / sum(n for _, n in losses),
                        "validation_macro_f1": metrics["macro_f1"], "validation_accuracy": metrics["accuracy"], "optimizer_steps": optimizer_steps})
        save_json(run / "history.json", history)
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            if config.method == "lora":
                # Embeddings are neither resized nor targeted here. Explicitly
                # disabling the auto check avoids PEFT querying the Hub on save.
                model.save_pretrained(run / "model", safe_serialization=True, save_embedding_layers=False)
            else:
                model.save_pretrained(run / "model", safe_serialization=True)
            tokenizer.save_pretrained(run / "model")
            save_evaluation(run, "validation", rows["validation"], probs, labels, manifest["split_sha256"]["validation"])
            update_run(run, best_epoch=epoch, best_validation_macro_f1=best)
        else:
            stale += 1
        if stale >= config.patience:
            break
    update_run(run, epochs_completed=len(history), optimizer_steps=optimizer_steps)


class Predictor:
    def __init__(self, run: Path, device="cpu"):
        self.meta = read_json(run / "run.json")
        if self.meta["status"] != "completed":
            raise ValueError("仅能加载已完成的实验")
        self.labels = self.meta["labels"]
        self.backend = self.meta["config"]["backend"]
        self.max_length = self.meta["config"]["max_length"]
        if self.backend == "baseline":
            self.model = joblib.load(run / "model.joblib")
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            self.device = torch_device(device)
            self.tokenizer = AutoTokenizer.from_pretrained(run / "model", local_files_only=True, trust_remote_code=False)
            if self.meta["config"]["method"] == "lora":
                from peft import PeftModel
                base = AutoModelForSequenceClassification.from_pretrained(
                    self.meta["config"]["model"], local_files_only=True, trust_remote_code=False,
                    revision=self.meta.get("model_revision") or "main",
                    num_labels=len(self.labels), id2label=dict(enumerate(self.labels)), label2id={x: i for i, x in enumerate(self.labels)},
                )
                self.model = PeftModel.from_pretrained(base, run / "model", local_files_only=True)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(run / "model", local_files_only=True, trust_remote_code=False)
            self.model.to(self.device).eval()

    def probabilities(self, texts, batch_size=16):
        if not texts or any(not isinstance(x, str) or not normalize(x) or len(x) > 20000 for x in texts):
            raise ValueError("每条文本必须非空且不超过 20000 字符")
        texts = [normalize(x) for x in texts]
        if self.backend == "baseline":
            p = self.model.predict_proba(texts)
            return p[:, [list(self.model.classes_).index(x) for x in self.labels]]
        import torch
        result = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
                result.extend(torch.softmax(logits.float(), dim=-1).cpu().tolist())
        return np.asarray(result)

    def predict(self, texts):
        probabilities = self.probabilities(texts)
        return [{"text": text, "label": self.labels[int(p.argmax())], "confidence": float(p.max()),
                 "probabilities": {label: float(p[i]) for i, label in enumerate(self.labels)}} for text, p in zip(texts, probabilities)]


def evaluate_run(run: Path, split="test", device="cpu"):
    if split not in ("validation", "test"):
        raise ValueError("仅可评测 validation / test")
    meta = read_json(run / "run.json")
    manifest, rows = load_dataset(Path(meta["dataset_path"]))
    if manifest["dataset_sha256"] != meta["dataset_sha256"] or manifest["labels"] != meta["labels"]:
        raise ValueError("评测数据与训练数据版本不符")
    predictor = Predictor(run, device)
    return save_evaluation(run, split, rows[split], predictor.probabilities([x["text"] for x in rows[split]]),
                           manifest["labels"], manifest["split_sha256"][split])


def compare_runs(runs: list[Path], split="validation"):
    if len(runs) < 2:
        raise ValueError("至少选择两个实验")
    metas = [read_json(run / "run.json") for run in runs]
    if any(x["status"] != "completed" for x in metas):
        raise ValueError("只能比较已完成实验")
    if len({x["dataset_sha256"] for x in metas}) != 1 or len({tuple(x["labels"]) for x in metas}) != 1:
        raise ValueError("对照实验必须使用同一数据版本和标签映射")
    results = []
    for run, meta in zip(runs, metas):
        metrics = read_json(run / f"{split}_metrics.json")
        if metrics["split_sha256"] != meta["split_sha256"][split]:
            raise ValueError("评测指纹不匹配")
        results.append({"run": run.name, "backend": meta["config"]["backend"], "method": meta["config"]["method"],
                        "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "samples": metrics["samples"], "split": split})
    return sorted(results, key=lambda x: x["macro_f1"], reverse=True)
