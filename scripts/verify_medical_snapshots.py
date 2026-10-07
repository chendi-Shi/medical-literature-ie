"""Portable integrity check; does not need raw datasets or model checkpoints."""
from pathlib import Path
import hashlib
import json


def main():
    root=Path(__file__).resolve().parents[1]/'docs/medical-results'
    fingerprints=json.loads((root/'fingerprints.json').read_text(encoding='utf-8'))
    for name,expected in fingerprints.items():
        if Path(name).name!=name:raise ValueError('Invalid snapshot path')
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Snapshot hash mismatch: '+name)
    repository=root.parents[1]
    for version in ('relation_v2','relation_v3'):
        report_path=root/f'{version}_report.json'
        if not report_path.exists():continue
        report=json.loads(report_path.read_text(encoding='utf-8'))
        protocol_path=root/f'{version}_protocol.json';selection_path=root/f'{version}_selection.json'
        protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
        selection=json.loads(selection_path.read_text(encoding='utf-8'))
        for path,key in ((protocol_path,'protocol_sha256'),(selection_path,'selection_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=report[key]:raise ValueError('Experiment protocol changed')
        for source,expected in protocol['source_sha256'].items():
            path=(repository/source.replace('\\','/')).resolve()
            if not path.is_relative_to(repository.resolve()):raise ValueError('Invalid source path')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Experiment source changed: '+source)
        history=json.loads((root/f'{version}_history.json').read_text(encoding='utf-8'))
        if [row['epoch'] for row in history]!=list(range(1,7)):raise ValueError('Incomplete training history')
        if history[-1]['optimizer_steps']!=report['candidate_run']['optimizer_steps']:raise ValueError('Update counts mismatch')
        if any(row['amp_skipped_updates'] for row in history):raise ValueError('Skipped training updates')
        for key in ('baseline','candidate','inference_tail_ablation','baseline_oracle_entities'):
            metrics=report[key]['relation']
            if abs(metrics['f1']-2*metrics['correct']/max(1,metrics['predicted']+metrics['gold']))>1e-10:
                raise ValueError('F1 counts mismatch')
        if selection['chosen']['metrics']['relation']['f1']!=report['candidate_validation_f1']:
            raise ValueError('Validation selection mismatch')
        validation=report['candidate_validation_f1']-report['baseline_validation_f1']>=protocol['min_validation_gain']
        audit=report['candidate']['relation']['f1']>=report['baseline']['relation']['f1']
        if report['release']['enabled']!=(validation and audit):raise ValueError('Release gate mismatch')
    print(f'Verified {len(fingerprints)} medical report snapshots; this checks integrity, not model quality.')


if __name__=='__main__':main()
