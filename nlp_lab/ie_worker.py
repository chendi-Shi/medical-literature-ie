"""Independent CPU evaluation worker. Never writes frozen suite artifacts."""
import sys
from pathlib import Path
from .data import read_json,save_json
from .ie.data import load
from .ie.predict import Predictor
from .ie.metrics import evaluate

def run_evaluation(run:Path,split:str):
    status=run/'evaluation_status.json'
    try:
        meta=read_json(run/'run.json')
        if split not in ('validation','test') or meta['status']!='completed' or meta.get('research_protocol'):
            raise ValueError('只允许评测已完成的独立 IE 实验')
        save_json(status,{'status':'running','split':split})
        manifest,rows=load(Path(meta['dataset_path']))
        if manifest['dataset_sha256']!=meta['dataset_sha256']:raise ValueError('实验数据指纹变化')
        predictor=Predictor(run,device='cpu')
        metrics,errors=evaluate(rows[split],predictor.predict_rows(rows[split]),manifest)
        save_json(run/(split+'_metrics.json'),metrics);save_json(run/(split+'_errors.json'),[e for e in errors if e['missing'] or e['extra']])
        save_json(status,{'status':'completed','split':split})
    except Exception as e:
        save_json(status,{'status':'failed','split':split,'error':f'{type(e).__name__}: {e}'})
        raise
    finally:(run/'.evaluation.lock').unlink(missing_ok=True)

def run_training(run:Path):
    from .ie.train import fit,update
    try:fit(run)
    except Exception as e:
        update(run,status='failed',error=f'{type(e).__name__}: {e}')
        raise

if __name__=='__main__':
    if sys.argv[1]=='train':run_training(Path(sys.argv[2]))
    else:run_evaluation(Path(sys.argv[1]),sys.argv[2])
