"""Predeclared validation-only decoding comparison; preserve frozen v1 reports."""
from pathlib import Path
import json
import os
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nlp_lab.data import read_json,save_json,digest
from nlp_lab.training import now
from nlp_lab.medical.data import TYPES
from nlp_lab.medical.decoding import decode,fit_thresholds,micro,counts,f1


def main():
    import numpy as np
    import torch
    from nlp_lab.medical.ner import NERPredictor,scores
    workspace=Path('workspace');root=workspace/'medical';out=root/'decoding-v2'
    if out.exists():raise ValueError('Decode experiment already exists; do not overwrite evaluated recipes')
    lock=workspace/'.training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    try:
        out.mkdir();run=root/'runs/ner-v1';manifest=read_json(root/'datasets/manifest.json')
        baseline=read_json(run/'run.json')['threshold'];grid=[i/4 for i in range(-8,9)]
        protocol={'created_at':now(),'weights_sha256':digest(run/'model.safetensors'),
                  'prepared_sha256':{n:s for n,s in manifest['prepared_sha256'].items() if n.startswith('ner-')},
                  'source_sha256':{p:digest(Path(p)) for p in ['nlp_lab/medical/decoding.py','scripts/improve_medical_decoding.py']},
                  'grid':grid,'baseline_threshold':baseline,'min_validation_type_support':50,
                  'policies':['raw','nested'],'strategies':['original_global','validation_global','validation_per_type'],
                  'selection':'max validation micro exact typed-span F1; ties prefer original then raw',
                  'promotion_min_validation_gain':.003,'bootstrap':{'unit':'document','samples':1000,'seed':20261007},
                  'scope':'Post-training decision comparison, not a new encoder or confidence probability calibration.',
                  'test_scope':'Reuses previously reported holdout; secondary comparison, not a pristine independently unseen final test.'}
        save_json(out/'protocol.json',protocol)
        for name,sha in protocol['prepared_sha256'].items():
            if digest(root/'datasets'/name)!=sha:raise ValueError('Prepared corpus changed')
        torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
        started=time.perf_counter();predictor=NERPredictor(run,device=device)
        validation=read_json(root/'datasets/ner-validation.json')
        low=predictor.predict([r['text'] for r in validation],threshold=min(grid))
        save_json(out/'validation_candidates.json',low)
        original={t:baseline for t in TYPES};typed,support=fit_thresholds(validation,low,grid,baseline)
        options=[]
        global_scores=[(micro(validation,decode(low,{t:v for t in TYPES})),v) for v in grid]
        best_global=max(global_scores,key=lambda x:(x[0],-abs(x[1]-baseline),-x[1]))[1]
        for strategy,thresholds in [('original_global',original),('validation_global',{t:best_global for t in TYPES}),('validation_per_type',typed)]:
            for policy in protocol['policies']:
                options.append({'strategy':strategy,'thresholds':thresholds,'policy':policy,'validation_f1':micro(validation,decode(low,thresholds,policy))})
        chosen=max(options,key=lambda x:(x['validation_f1'],x['strategy']=='original_global',x['policy']=='raw'))
        reference=options[0];promoted=chosen['validation_f1']-reference['validation_f1']>=protocol['promotion_min_validation_gain']
        deployment=chosen if promoted else reference
        # Seal the exact selection before loading reused holdout labels.
        selection={'chosen':chosen,'deployment':deployment,'promoted':promoted,'supports':support,'options':options,'selected_at':now()}
        save_json(out/'selection.json',selection);print(json.dumps(selection,ensure_ascii=False),flush=True)
        test_opened_at=now();test=read_json(root/'datasets/ner-test.json')
        low_test=predictor.predict([r['text'] for r in test],threshold=min(grid))
        old=decode(low_test,original);new=decode(low_test,chosen['thresholds'],chosen['policy'])
        old_metrics,_=scores(test,old);new_metrics,_=scores(test,new)
        saved=read_json(run/'test.json')['micro']
        if any(old_metrics['micro'][k]!=saved[k] for k in ('correct','predicted','gold')):
            raise ValueError('Original holdout counts not reproduced; investigate before reporting a gain')
        a=np.asarray(counts(test,old));b=np.asarray(counts(test,new));rng=np.random.default_rng(20261007);differences=[]
        for _ in range(1000):
            sample=rng.integers(0,len(test),len(test));differences.append(f1(b[sample].sum(0))-f1(a[sample].sum(0)))
        report={'protocol_sha256':digest(out/'protocol.json'),'selection_sha256':digest(out/'selection.json'),
                'baseline':old_metrics,'selected':new_metrics,'selection':selection,'test_opened_at':test_opened_at,
                'test_delta_f1':new_metrics['micro']['f1']-old_metrics['micro']['f1'],
                'paired_document_bootstrap_95_ci':np.quantile(differences,[.025,.975]).tolist(),
                'seconds':time.perf_counter()-started,'device':device,'finished_at':now(),
                'weights_sha256':protocol['weights_sha256'],'limitations':[protocol['test_scope'],protocol['scope'],
                    'Single checkpoint/seed; bootstrap reflects document sampling, not training-seed variation.',
                    'Performance measured on CMeEE, not full-document clinical outcome extraction.']}
        save_json(out/'report.json',report)
        if promoted:save_json(root/'active_decoder.json',{'version':'decoding-v2','weights_sha256':protocol['weights_sha256'],
                                                      **deployment,'selection_sha256':digest(out/'selection.json')})
        print(json.dumps({'baseline_f1':old_metrics['micro']['f1'],'selected_f1':new_metrics['micro']['f1'],'promoted':promoted,'seconds':report['seconds']}),flush=True)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
