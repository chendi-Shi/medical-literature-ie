import argparse
import json
from pathlib import Path

from nlp_lab.research import run_suite

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1] / "workspace")
    p.add_argument("--dataset", default="thucnews-10000-s1729")
    args = p.parse_args()
    report = run_suite(args.workspace.resolve(), args.workspace.resolve() / "datasets" / args.dataset)
    print(json.dumps({"status": report["status"], "selected_by_validation": report["selected_by_validation"], "aggregates": report["aggregates"]}, ensure_ascii=False, indent=2))
