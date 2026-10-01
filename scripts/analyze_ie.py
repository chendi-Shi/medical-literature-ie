"""Descriptive subgroups from saved predictions; no training or test tuning."""
import argparse
from pathlib import Path
import numpy as np
from nlp_lab.data import read_json,save_json
from nlp_lab.ie.data import DATASET,load,digest,entities
from nlp_lab.ie.metrics import prf


def analyze(workspace,output):
    root=workspace/'ie/research'/DATASET;report=read_json(root/'report.json')
    extension=workspace/'ie/research/duie2-context-10k-v1/report.json'
    if extension.exists():report['experiments']+=read_json(extension)['experiments']
    manifest,rows=load(workspace/'ie/datasets'/DATASET)
    assert report['protocol']['dataset_sha256']==manifest['dataset_sha256']
    known={(r['text'][a:b],typ) for r in rows['train'] for a,b,typ in entities(r)}
    groups={'one_gold_slot':[],'multiple_gold_slots':[],'distinct_overlapping_spans':[],
            'non_overlapping_spans':[],'all_typed_surfaces_seen_in_train':[],'some_typed_surface_unseen':[],
            'exclude_12_early_flow_smoke_rows':list(range(12,len(rows['test'])))}
    for i,row in enumerate(rows['test']):
        spans=entities(row)
        nested=any(max(a,c)<min(b,d) and (a,b)!=(c,d) for a,b,_ in spans for c,d,_ in spans)
        seen=all((row['text'][a:b],typ) in known for a,b,typ in spans)
        groups['one_gold_slot' if len(row['triples'])==1 else 'multiple_gold_slots'].append(i)
        groups['distinct_overlapping_spans' if nested else 'non_overlapping_spans'].append(i)
        groups['all_typed_surfaces_seen_in_train' if seen else 'some_typed_surface_unseen'].append(i)
    frequency={s['id']:manifest['counts']['train']['relation_support'].get(str(s['id']),0) for s in manifest['schemas']}
    bands={'train_support_1_49':[i for i,n in frequency.items() if n<50],
           'train_support_50_199':[i for i,n in frequency.items() if 50<=n<200],
           'train_support_200_plus':[i for i,n in frequency.items() if n>=200]}
    results={}
    for name in dict.fromkeys(e['name'] for e in report['experiments'] if e['name']!='frozen_encoder'):
        entries=[e for e in report['experiments'] if e['name']==name];values={}
        for group,indices in groups.items():
            metrics=[]
            for e in entries:
                counts=e['test']['counts_per_document'];total={k:sum(counts[i][k] for i in indices) for k in ('correct','predicted','gold')}
                metrics.append(prf(total['correct'],total['predicted'],total['gold']))
            values[group]={'documents':len(indices),'seed_metrics':metrics,'mean_relation_f1':float(np.mean([m['f1'] for m in metrics]))}
        for band,ids in bands.items():
            metrics=[]
            for e in entries:
                selected=[m for m in e['test']['per_relation'] if m['id'] in ids]
                total={k:sum(m[k] for m in selected) for k in ('correct','predicted','gold')}
                metrics.append(prf(total['correct'],total['predicted'],total['gold']))
            values[band]={'slots':len(ids),'seed_metrics':metrics,'mean_relation_f1':float(np.mean([m['f1'] for m in metrics]))}
        results[name]=values
    artifact={'dataset_sha256':manifest['dataset_sha256'],'source_sha256':digest(Path(__file__)),
              'scope':'主协议外的描述性分组；两种子固定模型均值，不做额外调参或显著性推断',
              'flow_smoke_disclosure':'正式套件前，前12条测试文本曾用于96训练/24验证的单轮流程自测和重载核验；未用于正式选模、阈值或质量超参选择。另报告排除这12条后的3988条关系指标。',
              'ambiguous_offset_documents':{s:sum(r['ambiguous_mentions']>0 for r in values) for s,values in rows.items()},
              'groups':{k:len(v) for k,v in groups.items()},'frequency_bands':bands,'results':results}
    save_json(root/'analysis.json',artifact);save_json(output/'subgroups.json',artifact)
    labels={'one_gold_slot':'单个金标槽位','multiple_gold_slots':'多个金标槽位','distinct_overlapping_spans':'不同边界跨度重叠',
            'non_overlapping_spans':'跨度不重叠','all_typed_surfaces_seen_in_train':'所有带类型表面字符串在 train 出现',
            'some_typed_surface_unseen':'包含 train 未出现的带类型表面字符串',
            'train_support_1_49':'训练支持 1–49 的关系槽位','train_support_50_199':'训练支持 50–199 的关系槽位','train_support_200_plus':'训练支持至少 200 的关系槽位',
            'exclude_12_early_flow_smoke_rows':'排除早期流程自测的 12 条'}
    text=['# 抽取难度分组分析','',artifact['scope']+'。不进入架构选择和阈值选择。各组有不同标签构成和文本难度，不能把组间差异归因于单一因素。','']
    for group,label in labels.items():
        value=results['pipeline'][group];unit=f"{value['documents']} 条文本" if 'documents' in value else f"{value['slots']} 个槽位"
        text += [f"- **{label}**（{unit}）：流水线 F1 {100*value['mean_relation_f1']:.2f}%；联合模型 F1 {100*results['joint'][group]['mean_relation_f1']:.2f}%。"]
    if 'context_pipeline' in results:
        text+=['','## 上下文与位置增强','']
        for group,label in labels.items():
            text += [f"- **{label}**：关系 F1 {100*results['context_pipeline'][group]['mean_relation_f1']:.2f}%。"]
    text += ['',artifact['flow_smoke_disclosure'],'','“未出现”只比较本轮训练标注中的精确表面字符串和实体类型，不代表编码器预训练未见过；没有做实体链接或近重复排除。长尾分组按训练金标槽位频数，全测试集该槽位的正确/预测/金标数聚合为 Micro-F1。','',
             '[原始分组数值](ie-results/subgroups.json) · [主结果](IE_RESULTS.md)','']
    (output.parent/'IE_SUBGROUPS.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    return artifact

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('workspace'));p.add_argument('--output',type=Path,default=Path('docs/ie-results'))
    a=p.parse_args();analyze(a.workspace.resolve(),a.output.resolve())
