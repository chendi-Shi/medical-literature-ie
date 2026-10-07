"""Verify the actual comparison and publish statistics without raw corpora."""
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nlp_lab.data import read_json,save_json,digest
from scripts.export_medical import main as export_original


def main():
    root=Path('workspace/medical');version='relation-v3' if (root/'relation-v3/report.json').exists() else 'relation-v2';out=root/version
    protocol=read_json(out/'protocol.json');report=read_json(out/'report.json')
    run=Path(protocol['run']);meta=read_json(run/'run.json');history=read_json(run/'history.json')
    if meta['status']!='completed' or len(history)!=6 or meta['amp_skipped_updates']:
        raise ValueError('Relation training is not complete')
    if digest(run/'model.safetensors')!=report['candidate_weights_sha256']:raise ValueError('Weights changed')
    if digest(out/'protocol.json')!=report['protocol_sha256'] or digest(out/'selection.json')!=report['selection_sha256']:
        raise ValueError('Protocol or selection changed')
    for source,sha in protocol['source_sha256'].items():
        if digest(Path(source))!=sha:raise ValueError('Comparison source changed: '+source)
    for split,sha in protocol['split_sha256'].items():
        if digest(root/f'datasets/cmeie-384-v1/{split}.jsonl')!=sha:raise ValueError('Corpus changed')
    for key in ('baseline','candidate','inference_tail_ablation','baseline_oracle_entities'):
        m=report[key]['relation']
        if abs(m['f1']-2*m['correct']/max(1,m['predicted']+m['gold']))>1e-10:raise ValueError('Metric counts mismatch')
    export_original()
    destination=Path('docs/medical-results')
    for arm in ('relation-v2','relation-v3'):
        if (root/arm/'report.json').exists():
            arm_run=Path(read_json(root/arm/'protocol.json')['run'])
            shutil.copyfile(arm_run/'history.json',destination/f'{arm.replace("-","_")}_history.json')
    b=report['baseline']['relation'];c=report['candidate']['relation'];pair=report['paired_comparison'];ci=pair['ci95']
    from nlp_lab.medical.runtime import relation_artifact
    active=relation_artifact(root);decision='启用'+active['version']+'联合模型' if active['version']!='relation-v1' else '保留原两阶段模型'
    save_json(destination/'relation_release_policy.json',read_json(root/'active_relation.json'))
    generated=f'''
## 实际运行结果

- 基线：P {b['precision']*100:.2f}% / R {b['recall']*100:.2f}% / F1 **{b['f1']*100:.2f}%**，正确 {b['correct']}、预测 {b['predicted']}、gold {b['gold']}。
- 联合：P {c['precision']*100:.2f}% / R {c['recall']*100:.2f}% / F1 **{c['f1']*100:.2f}%**，正确 {c['correct']}、预测 {c['predicted']}、gold {c['gold']}。
- 配对差值 {pair['delta_relation_f1']*100:+.2f} 个百分点；1,000 次文档 bootstrap 95% 区间 **[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]** 个百分点。此区间不包含训练种子波动。
- 验证集：基线 {report['baseline_validation_f1']*100:.2f}% → 联合 {report['candidate_validation_f1']*100:.2f}%；发布决定：**{decision}**。
- 实际六轮训练，最佳 epoch {meta['best_epoch']}，更新 {meta['optimizer_steps']}，FP16 跳过更新 {meta['amp_skipped_updates']}，耗时 {meta['seconds']:.1f} 秒，峰值 CUDA 分配 {meta['peak_cuda_memory_mb']:.1f} MiB。

尾链接推理消融 F1 **{report['inference_tail_ablation']['relation']['f1']*100:.2f}%**。仅移除解码约束，权重和阈值不变。

标点变化后，基线 F1 **{report['punctuation_robustness']['baseline']['relation']['f1']*100:.2f}%**，联合 **{report['punctuation_robustness']['candidate']['relation']['f1']*100:.2f}%**。该扰动不能代表全部临床输入鲁棒性。

基线用 gold 实体诊断 F1 **{report['baseline_oracle_entities']['relation']['f1']*100:.2f}%**，不能计入部署成绩。
'''
    if 'mean_head_control' in report:
        control=report['mean_head_control']
        generated+=f"\n独立均值损失对照：验证 F1 **{control['validation_f1']*100:.2f}%**，复用保留集 F1 **{control['relation']['f1']*100:.2f}%**。上面联合模型数字来自类别求和损失实验，两组均保留完整六轮历史。\n"
    for model,label in [('baseline','基线'),('candidate','联合')]:
        err=report['error_partition'][model]
        generated+=f"\n{label}漏关系中，原位置实体未检出 {err['missing_with_entity_error']} 条，参与实体已检出但链接漏检 {err['missing_despite_both_entities']} 条。\n"
    for name,label in [('short_text','短文本'),('long_text','长文本'),('shared_entity','共享实体'),('rare_relation','含稀有关系文本')]:
        before=report['slices']['baseline'][name];after=report['slices']['candidate'][name]
        generated+=f"\n- {label}：{after['documents']} 条，基线 {before['relation']['f1']*100:.2f}% → 联合 {after['relation']['f1']*100:.2f}%。\n"
    generated+='\n统计快照、预声明协议和六轮历史见 medical-results/relation_v3_report.json 或 relation_v2_report.json。原 v1 报告保留，NER 权重和 58.62% 结果不变。单种子稳定性、临床疗效抽取模型及专家验证仍待完成。\n'
    doc=Path('docs/MEDICAL_RELATION_UPGRADE.md')
    text=doc.read_text(encoding='utf-8').split('\n## 实际运行结果')[0].replace('本轮运行完成后追加实际分数和发布决定。','实际分数及发布决定见下节。')
    doc.write_text(text+generated,encoding='utf-8')
    results=Path('docs/MEDICAL_RESULTS.md')
    results.write_text(results.read_text(encoding='utf-8')+f'\n## v0.7 医学关系对照\n\n联合模型复用保留集 F1 {c["f1"]*100:.2f}%，原两阶段 {b["f1"]*100:.2f}%；当前{decision}。原始指标分别保留，不能称为新的独立未见测试。详见 [医学关系升级记录](MEDICAL_RELATION_UPGRADE.md)。\n',encoding='utf-8')
    save_json(destination/'fingerprints.json',{p.name:digest(p) for p in destination.glob('*.json') if p.name!='fingerprints.json'})
    print({'verified':True,'candidate_f1':c['f1'],'release':decision})


if __name__=='__main__':main()
