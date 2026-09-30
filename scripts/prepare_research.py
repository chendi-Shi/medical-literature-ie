import argparse
import json
from pathlib import Path

from nlp_lab.benchmark import build_benchmark
from nlp_lab.data import read_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1] / "workspace")
    p.add_argument("--train-per-class", type=int, default=1000)
    args = p.parse_args()
    destination = build_benchmark(args.workspace, args.train_per_class)
    print(json.dumps({"path": str(destination), "manifest": read_json(destination / "manifest.json")}, ensure_ascii=False, indent=2))
