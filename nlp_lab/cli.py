import argparse
import json
import sys
from pathlib import Path

from .data import prepare, save_json
from .training import Predictor, TrainConfig, allocate_run, compare_runs, evaluate_run, execute_run

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description="NLP Lab 数据清洗 / 微调 / 推理 / 评测")
    p.add_argument("--workspace", type=Path, default=ROOT / "workspace")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("source", type=Path)
    a.add_argument("--name", required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--provenance", default="user-provided")
    a = sub.add_parser("train")
    a.add_argument("--dataset", required=True)
    a.add_argument("--backend", choices=["baseline", "transformer"], default="baseline")
    a.add_argument("--method", choices=["full", "lora", "head"], default="full")
    a.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    for name, default in [("epochs", 3), ("batch-size", 8), ("accumulation", 1), ("max-length", 128), ("seed", 42), ("patience", 2), ("lora-rank", 8)]:
        a.add_argument("--" + name, type=int, default=default)
    a.add_argument("--learning-rate", type=float, default=2e-5)
    a.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a.add_argument("--allow-download", action="store_true")
    a.add_argument("--lora-targets", default="query,value")
    a.add_argument("--baseline-model", choices=["logistic", "linear_svm"], default="logistic")
    a.add_argument("--regularization", choices=["ce", "rdrop", "fgm", "rdrop_fgm"], default="ce")
    a.add_argument("--rdrop-alpha", type=float, default=.5)
    a.add_argument("--fgm-epsilon", type=float, default=.5)
    a.add_argument("--adversarial-weight", type=float, default=.5)
    a.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    a.add_argument("--warmup-ratio", type=float, default=0.)
    a.add_argument("--mixed-precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    a = sub.add_parser("evaluate")
    a.add_argument("--run", required=True)
    a.add_argument("--split", choices=["validation", "test"], default="test")
    a.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    a = sub.add_parser("predict")
    a.add_argument("--run", required=True)
    a.add_argument("--text", required=True, action="append")
    a = sub.add_parser("compare")
    a.add_argument("runs", nargs="+")
    a.add_argument("--split", choices=["validation", "test"], default="validation")
    a = sub.add_parser("demo")
    a.add_argument("--transformer", action="store_true")
    a.add_argument("--method", choices=["full", "lora", "head"], default="full")
    a.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    a = sub.add_parser("serve")
    a.add_argument("--port", type=int, default=8778)
    a.add_argument("--production", action="store_true", help="开启单账号认证和生产访问限制；必须放在 TLS 反向代理之后")
    return p


def safe_child(parent: Path, name: str):
    if not name or name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
        raise ValueError("名称仅允许英文字母、数字、横线和下划线")
    result = (parent / name).resolve()
    if not result.is_relative_to(parent.resolve()):
        raise ValueError("路径越界")
    return result


def main():
    args = parser().parse_args()
    workspace = args.workspace.resolve()
    try:
        if args.command == "prepare":
            result = prepare(args.source, safe_child(workspace / "datasets", args.name), args.seed, args.provenance)
        elif args.command == "train":
            values = {name: getattr(args, name) for name in TrainConfig.__dataclass_fields__}
            run = allocate_run(workspace, safe_child(workspace / "datasets", args.dataset), TrainConfig(**values))
            execute_run(run)
            result = {"run": run.name, "path": str(run)}
        elif args.command == "evaluate":
            run = safe_child(workspace / "runs", args.run)
            from .data import read_json
            if read_json(run / "run.json").get("research_protocol"):
                raise ValueError("研究评测产物由冻结协议生成，请运行 scripts.run_research 或查看已有结果")
            result = evaluate_run(run, args.split, args.device)
        elif args.command == "predict":
            result = Predictor(safe_child(workspace / "runs", args.run)).predict(args.text)
        elif args.command == "compare":
            result = compare_runs([safe_child(workspace / "runs", name) for name in args.runs], args.split)
        elif args.command == "demo":
            dataset = workspace / "datasets" / "chinese-demo"
            if not dataset.exists():
                prepare(ROOT / "data" / "chinese-demo.jsonl", dataset, provenance="原创人工编写的合成流程演示；不可作业务效果结论")
            run = allocate_run(workspace, dataset, TrainConfig(backend="transformer" if args.transformer else "baseline", model=args.model,
                                                              method=args.method, epochs=4, learning_rate=3e-4 if args.method == "lora" else 5e-5,
                                                              max_length=64, batch_size=8))
            execute_run(run)
            # Demo's fixed recipe is not tuned against this test split.
            metrics = evaluate_run(run)
            result = {"run": run.name, "test": metrics, "warning": "仅合成演示数据，非真实业务或公开基准成绩"}
            save_json(workspace / "demo-latest.json", result)
        else:
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(workspace, production=args.production), host="127.0.0.1", port=args.port,
                         access_log=not args.production, proxy_headers=args.production,
                         forwarded_allow_ips="127.0.0.1")
            return
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
