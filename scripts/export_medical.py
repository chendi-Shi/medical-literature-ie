"""Verify frozen experiment evidence; export small report snapshots without corpora/weights."""
from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nlp_lab.data import digest,read_json,save_json


def main():
    root=Path('workspace/medical');protocol=read_json(root/'protocol.json');report=read_json(root/'report.json')
    for name,sha in protocol['source_hashes'].items():
        if digest(Path(name))!=sha:raise ValueError('Frozen training source changed: '+name)
    if digest(root/'datasets/manifest.json')!=protocol['data_manifest_sha256']:raise ValueError('Data manifest changed')
    manifest=read_json(root/'datasets/manifest.json')
    for name,sha in manifest['prepared_sha256'].items():
        if digest(root/'datasets'/name)!=sha:raise ValueError('Prepared data changed: '+name)
    ner=root/'runs/ner-v1';re=Path(read_json(root/'status.json')['relation_run'])
    for name,path in [('ner',ner),('relations',re)]:
        if digest(path/'model.safetensors')!=report['weights_sha256'][name]:raise ValueError('Weights changed')
        meta=read_json(path/'run.json')
        if meta['status']!='completed' or meta['amp_skipped_updates']!=0:raise ValueError('Incomplete training')
        history=read_json(path/'history.json')
        if len(history)!=6:raise ValueError('Training epoch count mismatch')
    for metrics in (report['ner']['micro'],report['ner_dictionary_baseline']['micro'],report['relations']['relation']):
        if abs(metrics['f1']-2*metrics['correct']/max(1,metrics['predicted']+metrics['gold']))>1e-10:raise ValueError('Metric count mismatch')
    destination=Path('docs/medical-results');destination.mkdir(parents=True,exist_ok=True)
    for source,name in [(root/'protocol.json','protocol.json'),(root/'report.json','report.json'),
                        (root/'datasets/manifest.json','data_manifest.json'),(root/'paper_sources.json','paper_sources.json'),
                        (ner/'history.json','ner_history.json'),(re/'history.json','relation_history.json')]:
        shutil.copyfile(source,destination/name)
    if (root/'service_check.json').exists():shutil.copyfile(root/'service_check.json',destination/'service_check.json')
    n=report['ner']['micro'];b=report['ner_dictionary_baseline']['micro'];r=report['relations']['relation']
    types='\n'.join(f"- {typ}：F1 {m['f1']*100:.2f}%，gold {m['gold']}。" for typ,m in report['ner']['per_type'].items())
    text=f'''# 医学文献抽取：实际结果与边界

医学模型已完成独立训练和保留集评测。实体与关系分数衡量不同任务，不能拿它们当成文献疗效 / 不良事件抽取准确率。

- 医学嵌套实体：{report['ner']['documents']:,} 条文本，P {n['precision']*100:.2f}% / R {n['recall']*100:.2f}% / F1 **{n['f1']*100:.2f}%**；正确 {n['correct']:,}、预测 {n['predicted']:,}、gold {n['gold']:,}。
- 仅训练集构建的词典基线：F1 **{b['f1']*100:.2f}%**；同一保留集上提高 {(n['f1']-b['f1'])*100:.2f} 个百分点。单种子结果，不代表训练稳定性。
- 医学关系：{report['relations']['documents']:,} 条文本，P {r['precision']*100:.2f}% / R {r['recall']*100:.2f}% / F1 **{r['f1']*100:.2f}%**；正确 {r['correct']:,}、预测 {r['predicted']:,}、gold {r['gold']:,}。
- 关系 Macro F1：{report['relations']['macro_relation_f1']*100:.2f}%。关系模型没有强基线对照，不能声称提升或优于 CBLUE。

## 真实训练证据

两套训练均为 6 epochs、seed42、384 tokens、batch4 × 累积8、FP16。NER 最佳 checkpoint 是 epoch {report['ner_run']['best_epoch']}；验证阈值 {report['ner_run']['threshold']}；实际优化器更新 {report['ner_run']['optimizer_steps']:,}，跳过更新 0。关系实际更新 {report['relation_run']['optimizer_steps']:,}，跳过更新 0。协议先于训练及保留集质量评测保存；JSON 包含代码、数据、权重指纹。官方 test 无公开标签，本轮使用清洗后官方 dev 的固定不重叠划分。

编码器为 4 层 BAAI/bge-small-zh-v1.5，不是医学领域预训练编码器。NER 有独立九类嵌套跨度监督；关系使用实体跨度与上下文 / 距离特征，关系参与者实体另行预测，两条链路在文献输出中合并。原 DuIE 模型没有参与医学推理。

## 实体分项

{types}

## 文献业务能力

已导入 PMC11534547（21,031 字符，227 段落 / 表格行）和 PMC10918247（9,737 字符，67 段落 / 表格行）的实际 JATS 全文，来源及 CC BY 3.0 声明保留。模型窗口保留全文全局 Unicode 偏移，英文正文不进入中文模型；表格保留线性化行位置，未自动展开 rowspan / colspan 或绑定表头。

记录包含研究对象、干预 / 对照、疗效 / 终点、不良事件 / 安全性候选证据，原文数值、单位、HR / OR / RR、P 值、95% CI 及否定 / 比较线索。它们由公开的规则生成，不是单独训练过的 PICO 或事件模型。v0.5 新增同句明确终点 / 数值及显式组别候选，组别未知时留空，支持并列共享单位与部分发生率写法，全部仍需人工核验；不能声称已自动解决跨句共指、药物因果或完整临床证据综合。

实体可修改类型和标准名，证据记录可修改组别、终点、值及核验说明；原始预测不可覆盖，修订使用乐观并发控制并保留核验人和历史；证据范围变动会重算数字、绑定和实体引用。HTTP / CLI 共用导出契约，已核验导出剔除驳回实体并修剪概念引用，完整审计导出包含各次修订。local: 标准名 ID 不是 ICD / MeSH；仅原文明确缩写与有来源的术语表参与自动别名合并。

已观察到医学 NER 会将“免疫单药”误识为药物，以及症状 / 检验识别不足。规则曾把统计量括号误当别名，已经修正并添加回归测试。模型错误保留为候选，人工修订不计入模型分数。业务字段尚缺少独立专家标注文献保留集，不报告其准确率。

## v0.5 功能验证

新增 28 个手工编写的终点数值 / 放弃绑定回归样例，包括单位冲突、未达到终点、多组歧义、CI、明确组别、并列共享单位和不良事件发生率；这些用例不是医学专家标注测试集。真实全文的候选数量、原文偏移及引用检查见 `medical-results/service_check.json`。模型、训练协议和保留集分数不因这次规则更新而变化。

## 项目表述

可准确写作：实现中文医学文献抽取与证据核验系统，独立微调医学嵌套实体和关系模型，NER 在 {report['ner']['documents']:,} 条保留文本上 F1 {n['f1']*100:.2f}%，训练词典基线 F1 {b['f1']*100:.2f}%；完成长文档章节 / 表格定位、显式缩写归一、版本化人工修订与结构化导出。

不应写：生产可用、自动识别全部疗效 / 不良反应、医学推理准确率 58.62%、超过官方基线，或系统经过医学专家验证。
'''
    Path('docs/MEDICAL_RESULTS.md').write_text(text,encoding='utf-8')
    save_json(destination/'fingerprints.json',{p.name:digest(p) for p in destination.glob('*.json') if p.name!='fingerprints.json'})
    print(json.dumps({'ner_f1':n['f1'],'dictionary_f1':b['f1'],'relation_f1':r['f1'],'verified':True}))


if __name__=='__main__':main()
