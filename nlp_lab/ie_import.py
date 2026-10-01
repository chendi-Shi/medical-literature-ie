"""User annotated DuIE-style JSONL import, independent of the frozen benchmark."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import hashlib
import json
import random
import tempfile
from .cli import safe_child
from .data import key,save_json,digest
from .ie.data import MODEL,convert_record,schema_items,token_spans,entities


def import_data(content,schema,workspace,name,provenance='user-provided',seed=42,tokenizer=None):
    destination=safe_child(workspace/'ie/datasets',name)
    if destination.exists():raise ValueError('数据版本已存在，请使用新名称')
    if not isinstance(schema,list) or not 1<=len(schema)<=64:raise ValueError('schema 必须是 1–64 项的 JSON 数组')
    for s in schema:
        if not isinstance(s,dict) or not isinstance(s.get('object_type'),dict) or not s['object_type']:
            raise ValueError('schema object_type 必须是槽位到类型的字典')
        for value in [s.get('predicate'),s.get('subject_type'),*s['object_type'],*s['object_type'].values()]:
            if not isinstance(value,str) or not value.strip() or len(value)>80:raise ValueError('schema 名称/类型必须是 1–80 字符的字符串')
    with tempfile.TemporaryDirectory(prefix='ie-schema-') as temporary:
        root=Path(temporary);(root/'schema.json').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in schema),encoding='utf-8')
        schemas,types=schema_items(root)
    lookup={(s['predicate'],s['slot'],s['subject_type'],s['object_type']):s['id'] for s in schemas}
    if len(lookup)!=len(schemas) or len(schemas)>128:raise ValueError('schema 槽位重复或超过 128 项')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,use_fast=True)
    groups=defaultdict(list);audit=[];stats=Counter();split_flags=[]
    for line,raw in enumerate(content.splitlines(),1):
        if not raw.strip():continue
        stats['input']+=1
        try:
            record=json.loads(raw)
            if not isinstance(record,dict) or not isinstance(record.get('text'),str) or not record['text'].strip():raise ValueError('缺少非空 text')
            if not isinstance(record.get('spo_list'),list):raise ValueError('必须显式给出 spo_list；空数组表示已标注的无关系文本')
            if len(record['text'])>6000:raise ValueError('单条文本超过 6,000 字符')
            split=record.get('split')
            if split is not None and split not in ('train','validation','test'):raise ValueError('split 必须是 train / validation / test')
            row=convert_record(record,lookup,'user.jsonl',line) if record['spo_list'] else {'text':record['text'],'triples':[],
                   'source_file':'user.jsonl','source_line':line,'ambiguous_mentions':0}
            encoded=tokenizer(row['text'],return_offsets_mapping=True)
            if len(encoded['input_ids'])>128:raise ValueError('超出 128 tokens 短文本训练预算')
            token_spans(encoded['offset_mapping'],row)
            groups[key(row['text'])].append((row,split));split_flags.append(split is not None)
        except (ValueError,KeyError,TypeError,AttributeError) as e:
            reason=str(e);stats['rejected']+=1;audit.append({'source_line':line,'reason':reason})
    if split_flags and any(split_flags) and not all(split_flags):raise ValueError('有效记录的 split 必须全部指定或全部省略')
    explicit=bool(split_flags and all(split_flags));accepted=[]
    for group in groups.values():
        if explicit and len({split for _,split in group})>1:raise ValueError('自有数据存在跨划分规范化重叠')
        signatures={(r['text'],json.dumps(r['triples'],sort_keys=True)) for r,_ in group}
        if len(signatures)>1:
            stats['conflicting_rows']+=len(group)
            audit.extend({'source_line':r['source_line'],'reason':'同规范化文本的原文或关系标注冲突，整组隔离'} for r,_ in group);continue
        accepted.append(group[0]);stats['duplicates_removed']+=len(group)-1
    if len(accepted)<20:raise ValueError(f'清洗后只有 {len(accepted)} 条独立文本；至少需要 20 条')
    partitions={s:[] for s in ('train','validation','test')}
    if explicit:
        for row,split in accepted:partitions[split].append(row)
    else:
        random.Random(seed).shuffle(accepted);n=len(accepted);a=int(n*.8);b=a+max(1,int(n*.1))
        partitions={s:[r for r,_ in values] for s,values in zip(partitions,(accepted[:a],accepted[a:b],accepted[b:]))}
    if min(len(v) for v in partitions.values())<1:raise ValueError('train / validation / test 均需独立文本')
    covered={t[3] for row in partitions['train'] for t in row['triples']}
    if len(covered)!=len(schemas):raise ValueError('训练集未覆盖全部 schema 槽位，请增加标注或显式划分')
    destination.mkdir(parents=True)
    for split,records in partitions.items():
        (destination/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8',newline='\n')
    hashes={s:digest(destination/(s+'.jsonl')) for s in partitions}
    manifest={'id':name,'task':'typed entities + binary relation slots','model':MODEL,'schemas':schemas,'entity_types':types,
              'provenance':provenance,'source_content_sha256':hashlib.sha256(content.encode('utf-8')).hexdigest(),
              'selection_seed':seed,'split_policy':'explicit validated' if explicit else 'fixed seed 80/10/10 by document',
              'max_source_tokens':128,'source_statistics':dict(stats),'split_sha256':hashes,
              'dataset_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
              'counts':{s:{'documents':len(v),'entities':sum(len(entities(r)) for r in v),'triples':sum(len(r['triples']) for r in v),
                           'negative_documents':sum(not r['triples'] for r in v),'relation_support':dict(Counter(str(t[3]) for r in v for t in r['triples']))} for s,v in partitions.items()},
              'limitations':['仅关系参与者实体；首次精确出现派生位置，重复提及有歧义','精确去重不排除近重复；自动划分非多标签分层划分',
                             '复杂 object 展开二元槽位；不做别名匹配','显式空 spo_list 作为已标注无已知关系文本；未标注数据不能当负例']}
    save_json(destination/'manifest.json',manifest);save_json(destination/'schema-source.json',schema);save_json(destination/'audit.json',audit)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description='导入自有中文关系标注，不进入冻结研究')
    p.add_argument('source',type=Path);p.add_argument('--schema',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1]/'workspace')
    a=p.parse_args();m=import_data(a.source.read_text(encoding='utf-8'),json.loads(a.schema.read_text(encoding='utf-8')),a.workspace,a.name)
    print(json.dumps(m,ensure_ascii=False,indent=2))
