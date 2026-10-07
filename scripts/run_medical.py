"""Predeclare and execute domain-specific NER + relation training on real corpora."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nlp_lab.data import read_json, save_json, digest
from nlp_lab.training import now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path('workspace'))
    args=parser.parse_args();workspace=args.workspace.resolve();root=workspace/'medical'
    from transformers import AutoTokenizer
    from nlp_lab.medical.data import prepare
    from nlp_lab.ie.data import MODEL,load
    from nlp_lab.medical.ner import train
    from nlp_lab.ie.train import Config,allocate,update
    from nlp_lab.ie_context import fit_context,predictor_for
    from nlp_lab.ie.metrics import evaluate
    code_root=Path(__file__).resolve().parents[1]
    files=[code_root/'nlp_lab/medical/data.py',code_root/'nlp_lab/medical/ner.py',Path(__file__).resolve(),
           code_root/'nlp_lab/ie_context.py',*sorted((code_root/'nlp_lab/ie').glob('*.py'))]
    hashes={str(p.relative_to(code_root)):digest(p) for p in files}
    if not (root/'datasets/manifest.json').exists():
        prepare(workspace,AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True))
    config=Config(architecture='pipeline',epochs=6,batch_size=4,accumulation=8,max_length=384)
    protocol={'created_at':now(),'source_hashes':hashes,'data_manifest_sha256':digest(root/'datasets/manifest.json'),
              'ner':{'epochs':6,'seed':42,'checkpoint':'max validation micro exact typed-span F1',
                     'thresholds':[0.,-.5,.5],'test':'official labeled dev minus first shuffled 1500 clean documents'},
              'relations':{'config':vars(config),'features':'context_position','checkpoint':'max validation relation F1',
                           'thresholds':[[0.,0.],[-.5,-.5],[.5,.5]],'test':'official labeled dev minus first shuffled 1200 clean documents'},
              'scope':'Medical NER / CMeIE relation scores only. PICO/outcome/adverse-event candidates are separate unvalidated rules.',
              'limitations':['Single seed; no training-seed uncertainty estimate.',
                             'Public mirror provenance recorded; no official leaderboard comparison.',
                             '4-layer general Chinese encoder, not a medical pretrained encoder.']}
    if (root/'protocol.json').exists():
        original=read_json(root/'protocol.json')
        if original['source_hashes']!=hashes or original['data_manifest_sha256']!=protocol['data_manifest_sha256']:
            raise ValueError('Frozen medical protocol/data changed')
    else:save_json(root/'protocol.json',protocol)
    status=read_json(root/'status.json') if (root/'status.json').exists() else {}
    def phase(value):status.update(phase=value,updated_at=now());save_json(root/'status.json',status)
    try:
        ner=root/'runs/ner-v1'
        if not ner.exists():phase('medical_ner_training');train(workspace)
        if read_json(ner/'run.json')['status']!='completed':raise ValueError('NER run failed or active; inspect run metadata')
        dataset=root/'datasets/cmeie-384-v1'
        if 'relation_run' not in status:
            run=allocate(workspace,dataset,config);update(run,relation_features='context_position',medical_domain=True)
            status['relation_run']=str(run);save_json(root/'status.json',status)
        else:run=Path(status['relation_run'])
        if read_json(run/'run.json')['status']=='queued':phase('medical_relation_training');fit_context(run)
        if read_json(run/'run.json')['status']!='completed':raise ValueError('Relation run failed or active')
        phase('validation_selection')
        predictor=predictor_for(run,'cuda');manifest,rows=load(dataset)
        candidates=[]
        for eth,rth in [(0.,0.),(-.5,-.5),(.5,.5)]:
            metrics,_=evaluate(rows['validation'],predictor.predict_rows(rows['validation'],entity_threshold=eth,relation_threshold=rth,batch_size=4),manifest)
            candidates.append((metrics['relation']['f1'],eth,rth,metrics))
        choice=max(candidates,key=lambda x:x[0]);save_json(run/'thresholds.json',{'entity':choice[1],'relation':choice[2]})
        save_json(run/'tuned_validation_metrics.json',choice[3]);status['relation_test_opened_at']=now();phase('heldout_evaluation')
        predictions=predictor.predict_rows(rows['test'],entity_threshold=choice[1],relation_threshold=choice[2],batch_size=4)
        metrics,errors=evaluate(rows['test'],predictions,manifest)
        save_json(run/'test_metrics.json',metrics);save_json(run/'test_errors.json',errors)
        report={'protocol':read_json(root/'protocol.json'),'ner':read_json(ner/'test.json'),
                'ner_dictionary_baseline':read_json(ner/'dictionary_test.json'),'ner_run':read_json(ner/'run.json'),
                'relations':metrics,'relation_run':read_json(run/'run.json'),'finished_at':now()}
        report['weights_sha256']={'ner':digest(ner/'model.safetensors'),'relations':digest(run/'model.safetensors')}
        save_json(root/'report.json',report);phase('completed')
        print(json.dumps({'ner_f1':report['ner']['micro']['f1'],'relation_f1':metrics['relation']['f1']},ensure_ascii=False),flush=True)
    except Exception as e:
        status['error']=f'{type(e).__name__}: {e}';phase('failed');raise


if __name__=='__main__':main()
