"""Independent IE experiments and inference; suite artifacts stay immutable."""
import argparse
import json
from pathlib import Path
from ..cli import safe_child
from ..data import read_json,save_json
from .data import DATASET,prepare,load
from .train import Config,allocate,fit
from .predict import Predictor
from .metrics import evaluate


def main():
    p=argparse.ArgumentParser(description='中文实体与关系抽取：清洗 / 微调 / 推理 / 评测')
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2]/'workspace')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    a=sub.add_parser('train');a.add_argument('--dataset',default=DATASET)
    a.add_argument('--architecture',choices=['pipeline','joint'],default='joint')
    a.add_argument('--epochs',type=int,default=6);a.add_argument('--seed',type=int,default=42)
    a.add_argument('--freeze-encoder',action='store_true')
    a.add_argument('--precision',choices=['fp32','fp16'],default='fp16')
    a=sub.add_parser('fit');a.add_argument('--run',required=True)
    a=sub.add_parser('extract');a.add_argument('--run',required=True);a.add_argument('--text',required=True)
    a=sub.add_parser('evaluate');a.add_argument('--run',required=True)
    a.add_argument('--split',choices=['validation','test'],default='test')
    args=p.parse_args();w=args.workspace.resolve()
    if args.command=='prepare':print(prepare(w));return
    if args.command=='train':
        if (w/'ie/.suite.lock').exists():raise ValueError('冻结研究套件正在运行，请等待结束')
        run=allocate(w,safe_child(w/'ie/datasets',args.dataset),Config(architecture=args.architecture,epochs=args.epochs,
                     seed=args.seed,freeze_encoder=args.freeze_encoder,mixed_precision=args.precision))
        fit(run);print(run);return
    run=safe_child(w/'ie/runs',args.run)
    if args.command=='fit':fit(run);return
    predictor=Predictor(run)
    if args.command=='extract':result=predictor.extract(args.text)
    else:
        meta=read_json(run/'run.json')
        if meta.get('research_protocol'):raise ValueError('冻结套件产物不可覆写，请运行独立实验')
        manifest,rows=load(Path(meta['dataset_path']))
        result,errors=evaluate(rows[args.split],predictor.predict_rows(rows[args.split]),manifest)
        save_json(run/(args.split+'_metrics.json'),result);save_json(run/(args.split+'_errors.json'),errors)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
