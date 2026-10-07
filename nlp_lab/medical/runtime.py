"""Select audited relation artifacts without overwriting original experiment files."""
from pathlib import Path
from ..data import read_json,digest


def relation_training_status(root):
    protocol=next((Path(root)/v/'protocol.json' for v in ('relation-v3','relation-v2') if (Path(root)/v/'protocol.json').exists()),None)
    if protocol is None:return None
    run=Path(read_json(protocol)['run'])
    meta=read_json(run/'run.json')
    return {'version':protocol.parent.name,**{k:meta.get(k) for k in ('id','status','current_epoch','batches_done','batches_per_epoch','epochs_completed','best_epoch','best_validation_relation_f1','error','resume_count','resumable_epoch','resumable_batches_done')}}


def latest_comparison(root):
    return next((read_json(Path(root)/v/'report.json') for v in ('relation-v3','relation-v2') if (Path(root)/v/'report.json').exists()),None)


def relation_artifact(root):
    root=Path(root)
    original=read_json(root/'report.json')
    path=Path(read_json(root/'status.json')['relation_run'])
    result={'run':path,'sha256':original['weights_sha256']['relations'],
            'architecture':'context_position','metrics':original['relations'],'version':'relation-v1'}
    policy=root/'active_relation.json'
    if not policy.exists() or not read_json(policy).get('enabled'):return result
    policy=read_json(policy);version=policy.get('version','relation-v2')
    if version not in ('relation-v2','relation-v3'):raise ValueError('Invalid relation version')
    report_path=root/version/'report.json'
    if digest(report_path)!=policy['report_sha256']:raise ValueError('Active relation report changed')
    report=read_json(report_path)
    if not report['release']['enabled']:raise ValueError('Relation candidate did not pass release gates')
    run=(root.parent/policy['run_relative_to_workspace']).resolve()
    if not run.is_relative_to((root.parent/'ie/runs').resolve()):raise ValueError('Invalid relation artifact path')
    meta=read_json(run/'run.json')
    if meta['status']!='completed' or meta['config']['architecture']!='joint':raise ValueError('Active joint model is incomplete')
    if digest(run/'model.safetensors')!=report['candidate_weights_sha256']:raise ValueError('Active relation weights changed')
    if digest(run/'thresholds.json')!=policy['thresholds_sha256']:raise ValueError('Active relation thresholds changed')
    return {'run':run,'sha256':report['candidate_weights_sha256'],'architecture':'typed_gplinker_head_sum' if version=='relation-v3' else 'typed_gplinker',
            'metrics':report['candidate'],'version':version}
