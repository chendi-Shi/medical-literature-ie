"""Verify real experiment artifacts and export the project's numerical report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from nlp_lab.data import read_json,save_json
from nlp_lab.ie.data import DATASET,digest,load
from nlp_lab.ie.suite import source_hashes

def export(workspace,output):
    report=read_json(workspace/'ie/research'/DATASET/'report.json')
    protocol=report['protocol'];manifest,rows=load(workspace/'ie/datasets'/DATASET)
    assert protocol['dataset_sha256']==manifest['dataset_sha256']
    assert protocol['source_sha256']==source_hashes(),'冻结代码指纹变化'
    for entry in report['experiments']:
        run=workspace/'ie/runs'/entry['run']
        assert digest(run/'model.safetensors')==entry['checkpoint_sha256']
        meta=read_json(run/'run.json')
        assert meta['status']=='completed' and meta['config']==entry['config']
        assert meta['epochs_completed']==6 and meta['optimizer_steps']>0
        assert entry['test']['documents']==4000 and entry['validation']['documents']==2000
        assert len(entry['test']['per_relation'])==55
        counts=entry['test']['counts_per_document'];totals={k:sum(x[k] for x in counts) for k in ('correct','predicted','gold')}
        assert abs(2*totals['correct']/(totals['predicted']+totals['gold'])-entry['test']['relation']['f1'])<1e-12
    assert len(report['experiments'])==5
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'protocol.json',protocol);save_json(output/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Pipeline','Joint GPLinker','Frozen encoder'];keys=['pipeline','joint','frozen_encoder']
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    colors=['#89a993','#24634d','#b8c8ba'];x=np.arange(3)
    for ax,metric,title in zip(axes,('relation_f1','entity_f1'),('Relation slot exact match','Typed participant entity exact span')):
        values=[100*report['aggregates'][k][metric]['mean'] for k in keys]
        std=[100*(report['aggregates'][k][metric]['std'] or 0) for k in keys]
        ax.bar(x,values,color=colors,yerr=std,capsize=4,width=.65)
        ax.set_xticks(x,names);ax.set_ylim(0,100);ax.set_ylabel('Test micro-F1 (%)');ax.set_title(title,fontsize=11)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(i,v+std[i]+2,f'{v:.2f}',ha='center',fontsize=10)
    fig.suptitle('DuIE2 binary-slot subset: 10k train / 2k validation / 4k test',fontsize=12)
    fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    selected=report['selected_architecture'];aggregate=report['aggregates'][selected]
    main=[e for e in report['experiments'] if e['name']==selected];representative=next(e for e in main if e['seed']==42)
    baseline=report['aggregates']['pipeline']['relation_f1']['mean'];joint=report['aggregates']['joint']['relation_f1']['mean']
    oracle=next(e for e in report['experiments'] if e['name']=='pipeline' and e['seed']==42)['diagnostics']['gold_entity_oracle']['relation']['f1']
    no_tail=next(e for e in report['experiments'] if e['name']=='joint' and e['seed']==42)['diagnostics']['no_tail_decode']['relation']['f1']
    def stat(a):return f"{100*a['mean']:.2f}%"+(f" ± {100*a['std']:.2f} pp" if a['std'] is not None else '（单次）')
    lines=['# 中文实体与关系抽取：本机实测结果','',
           f"完成时间：{report['finished_at']}。所有数值来自实际 checkpoint 重载与完整测试，不使用历史分类分数。",'',
           f"按验证均值选择 **{selected}** 架构。测试关系 Micro-F1 **{stat(aggregate['relation_f1'])}**，关系参与者实体 Micro-F1 **{stat(aggregate['entity_f1'])}**。主模型均为 seed42/43 两次，± 是样本标准差。部署固定使用该架构 seed42，不按测试集挑种子。",'',
           '![同预算抽取对照](ie-results/comparison.png)','',
           '## 数据与指标','',
           f"固定训练 10,000、验证 2,000、测试 4,000 条；训练二元槽位 18,243 个、测试槽位 7,453 个。55 槽位 schema、26 实体类型。训练覆盖全部 55 槽位，验证与测试各覆盖 53；固定 55 槽位 Macro-F1 将无真实支持槽位也计入。",'',
           '关系按带类型主客体表面字符串与 predicate/slot 精确匹配；实体按关系参与者的派生字符跨度匹配。官方复杂 object 拆为二元槽位，不等于完整多槽 SPO 评测。数据为清洗后的短文本固定子集，不是全量 DuIE 结果。', '',
           '早期流程自测曾用前 12 条测试文本核验小规模 checkpoint 重载，未用于正式选模或阈值；正式五个模型的完整测试均在训练和验证选择后统一执行。另见 [排除这 12 条及难度分组核验](IE_SUBGROUPS.md)，不能称测试文本从未在任何流程中读取。','',
           f"数据 SHA-256：`{manifest['dataset_sha256']}`。来源、清洗、开源代码阅读范围见 [IE_RESEARCH.md](IE_RESEARCH.md)。",'',
           '## 同预算对照','']
    for k in keys:
        a=report['aggregates'][k]
        lines += [f"- **{k}**：关系 F1 {stat(a['relation_f1'])}；实体 F1 {stat(a['entity_f1'])}；中性前缀压力下降 {100*a['stress_drop']['mean']:.2f} pp。"]
    interval=report['paired_seed42']
    lines+=['',f"两种子均值下，联合模型相对流水线变化 {(joint-baseline)*100:+.2f} pp。seed42 文本配对差值 {100*interval['delta_relation_f1']:+.2f} pp，条件 95% 区间 [{100*interval['ci95'][0]:+.2f}, {100*interval['ci95'][1]:+.2f}]。600 次配对文本 bootstrap，固定模型，不含训练种子总体不确定性，无多重比較校正。",'',
            '## 诊断、消融与部署','',
            f"流水线 seed42 输入真实实体时，关系 F1 {100*oracle:.2f}%；这是定位实体漏检/错边界影响的 oracle 诊断，不能当成部署表现。联合模型 seed42 去掉尾部交集后 F1 {100*no_tail:.2f}%，属于同权重解码消融。冻结编码器为单种子同轮数独立训练，不等于多种子因子实验。",'',
            '中性前缀压力使用同组 1,000 条测试文本、固定 seed2027，在原文前加“信息：”并平移所有跨度。它只说明前缀敏感性，不能代替真实业务外部测试。', '',
            f"默认 run：`{report['selected_run']}`。seed42 模型 {representative['parameters']:,} 参数，完整训练含验证和保存 {representative['seconds']:.1f} 秒，PyTorch 峰值分配显存 {representative['peak_cuda_memory_mb']:.1f} MiB。编码器版本 `{representative['model_revision']}`。",'',
            '## 可以写入简历的表述','',
            f"> 实现中文实体与关系联合抽取系统，基于 DuIE2 清洗构建 10k/2k/4k 固定划分，使用中文预训练编码器与带类型跨度 GPLinker，并完成流水线对照、两种子评测、实体误差级联诊断及解码/冻结消融；在当前二元槽位子集取得关系 Micro-F1 {100*aggregate['relation_f1']['mean']:.2f}%，提供原文定位和结构化 JSON 推理。",'',
            '上述成绩不能写成官方 DuIE 榜单、完整 NER 指标、原创算法或生产上线效果。模型仅支持有限 schema，长文本窗口不保证跨窗口关系；分数未校准。', '',
            '## 可核验证据','',
            '- [冻结协议](ie-results/protocol.json)：数据与代码指纹、全部配方、阈值/选模/测试政策。',
            '- [逐次数值报告](ie-results/report.json)：每次种子、checkpoint SHA、完整槽位指标、配对统计、压力测试与诊断。',
            '- [快照指纹](ie-results/fingerprints.json)。原始模型、逐轮历史和文本错误位于本地 workspace/ie/runs，不随 Git 分发。','']
    (output.parent/'IE_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    from scripts.analyze_ie import analyze
    analyze(workspace,output)
    save_json(output/'fingerprints.json',{p.name:digest(p) for p in output.glob('*.json') if p.name!='fingerprints.json'})
    print(output.parent/'IE_RESULTS.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('workspace'));p.add_argument('--output',type=Path,default=Path('docs/ie-results'))
    a=p.parse_args();export(a.workspace.resolve(),a.output.resolve())
