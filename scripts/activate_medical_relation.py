"""Release only the preselected, audited checkpoint; preserve original artifacts."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nlp_lab.data import digest,read_json,save_json
from nlp_lab.medical.relation_analysis import release_decision
from nlp_lab.medical.runtime import relation_artifact


def activate(root,version='relation-v2'):
    if version not in ('relation-v2','relation-v3'):raise ValueError('Invalid relation version')
    root=Path(root);out=root/version;report=read_json(out/'report.json');protocol=read_json(out/'protocol.json')
    selection=read_json(out/'selection.json')
    if digest(out/'protocol.json')!=report['protocol_sha256'] or digest(out/'selection.json')!=report['selection_sha256']:
        raise ValueError('Relation experiment protocol/selection changed')
    decision=release_decision(report['baseline_validation_f1'],report['candidate_validation_f1'],
        report['baseline']['relation']['f1'],report['candidate']['relation']['f1'],protocol['min_validation_gain'])
    if decision!=report['release']:raise ValueError('Release decision does not match predeclared gates')
    policy={'enabled':decision['enabled'],'version':version,'release':decision,'report_sha256':digest(out/'report.json')}
    if decision['enabled']:
        run=Path(protocol['run']).resolve();workspace=root.parent.resolve()
        if not run.is_relative_to(workspace/'ie/runs'):raise ValueError('Invalid run path')
        meta=read_json(run/'run.json')
        if meta['status']!='completed' or meta['epochs_completed']!=6 or meta['amp_skipped_updates']:
            raise ValueError('Incomplete or skipped optimization updates')
        if digest(run/'model.safetensors')!=report['candidate_weights_sha256']:raise ValueError('Candidate weights changed')
        thresholds=read_json(run/'thresholds.json')
        if thresholds!={k:selection['chosen'][k] for k in ('entity','relation')}:raise ValueError('Selected thresholds changed')
        policy.update(run_relative_to_workspace=run.relative_to(workspace).as_posix(),thresholds_sha256=digest(run/'thresholds.json'))
    save_json(root/'active_relation.json',policy)
    return relation_artifact(root)


if __name__=='__main__':
    root=Path('workspace/medical')
    winner=read_json(root/'relation-v3/selection.json')['winner_version'] if (root/'relation-v3/report.json').exists() else 'relation-v2'
    if winner=='relation-v1':
        save_json(root/'active_relation.json',{'enabled':False,'version':'relation-v1',
            'selection_sha256':digest(root/'relation-v3/selection.json'),
            'reason':'Neither validation candidate passed the predeclared gain gate'})
        active=relation_artifact(root)
    else:active=activate(root,winner)
    print({k:v for k,v in active.items() if k not in ('run','metrics')})
