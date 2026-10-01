"""Public IE commands with architecture-aware loading and protected research artifacts."""
import argparse,json
from pathlib import Path
from .cli import safe_child
from .data import read_json
from .ie.data import DATASET,prepare
from .ie.train import Config,allocate,update
from .ie_worker import run_training,run_evaluation
from .ie_context import predictor_for

def main():
    parser=argparse.ArgumentParser(description='中文实体与关系抽取：数据、微调、推理、评测')
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1]/'workspace')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare')
    train=commands.add_parser('train');train.add_argument('--dataset',default=DATASET)
    train.add_argument('--architecture',choices=['pipeline','joint','context_pipeline'],default='context_pipeline')
    train.add_argument('--epochs',type=int,default=6);train.add_argument('--seed',type=int,default=42)
    train.add_argument('--precision',choices=['fp16','fp32'],default='fp16');train.add_argument('--freeze-encoder',action='store_true')
    extract=commands.add_parser('extract');extract.add_argument('--run',required=True);extract.add_argument('--text',required=True)
    evaluate=commands.add_parser('evaluate');evaluate.add_argument('--run',required=True);evaluate.add_argument('--split',choices=['validation','test'],default='test')
    args=parser.parse_args();workspace=args.workspace.resolve()
    if args.command=='prepare':print(prepare(workspace));return
    if args.command=='train':
        if any(p.exists() for p in [workspace/'ie/.suite.lock',workspace/'ie/.context.lock',workspace/'.training.lock']):
            raise ValueError('已有训练或冻结研究运行，请等待结束')
        if not 0<=args.seed<=4294967295 or not 1<=args.epochs<=30:raise ValueError('种子或训练轮数无效')
        run=allocate(workspace,safe_child(workspace/'ie/datasets',args.dataset),
                     Config(architecture='pipeline' if args.architecture=='context_pipeline' else args.architecture,
                            epochs=args.epochs,seed=args.seed,mixed_precision=args.precision,freeze_encoder=args.freeze_encoder))
        if args.architecture=='context_pipeline':update(run,relation_features='context_position')
        run_training(run);print(run);return
    run=safe_child(workspace/'ie/runs',args.run)
    if args.command=='extract':result=predictor_for(run).extract(args.text)
    else:
        meta=read_json(run/'run.json')
        if meta.get('research_protocol'):raise ValueError('冻结研究评测不可覆写，请查看报告或训练独立模型')
        run_evaluation(run,args.split);result=read_json(run/(args.split+'_metrics.json'))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
