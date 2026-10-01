import argparse
from pathlib import Path
from nlp_lab.ie.data import prepare, MODEL

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--workspace",type=Path,default=Path(__file__).resolve().parents[1]/"workspace")
    p.add_argument("--model",default=MODEL)
    args=p.parse_args()
    print(prepare(args.workspace,args.model))
