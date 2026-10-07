"""Prepare real CBLUE corpora; keep medical and general-domain protocols separate."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import random

from ..data import digest, key, save_json
from ..ie.data import MODEL, convert_record, token_spans

TYPES = ['bod', 'dep', 'dis', 'dru', 'equ', 'ite', 'mic', 'pro', 'sym']
TYPE_NAMES = dict(zip(TYPES, ['身体', '科室', '疾病', '药物', '医疗设备', '医学检验', '微生物', '医疗程序', '临床表现']))


def read_records(path):
    content = Path(path).read_text(encoding='utf-8-sig')
    return json.loads(content) if content.lstrip().startswith('[') else [json.loads(s) for s in content.splitlines() if s.strip()]


def windows(text, tokenizer, length=384, stride=96):
    encoded = tokenizer(text, return_offsets_mapping=True, return_overflowing_tokens=True,
                        truncation=True, max_length=length, stride=stride)
    return [(off[0][0], off[-1][1]) for offsets in encoded['offset_mapping']
            if (off := [(a, b) for a, b in offsets if b > a])]


def prepare(workspace, tokenizer):
    root = Path(workspace) / 'medical'; source = root / 'sources'
    destination = root / 'datasets'; destination.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': 20261001, 'encoder': MODEL, 'max_length': 384, 'stride': 96,
                'source_sha256': {p.name: digest(p) for p in source.glob('*.json')},
                'source_provenance': {'CMeEE_train': 'GitHub Z-MU-Z/cmeee mirror, master at retrieval; local SHA recorded',
                    'CMeEE_dev': 'HF Aunderline/CMeEE mirror, revision bea81a21dc077bb21f6df5a0385410f4768b97bd',
                    'CMeIE': 'GitHub yehqi/RelationExtraction mirror, main at retrieval; local SHA recorded',
                    'original': 'https://github.com/CBLUEbenchmark/CBLUE'},
                'limitations': ['Public mirrors are not publisher-authenticated checksums; original licensing must be checked before redistribution.',
                    'Official unlabeled test is not scored; labeled official dev split into validation and heldout.',
                    'Corpora are medical text, not a gold set of literature PICO/outcomes/adverse events.',
                    'Exact normalized deduplication does not remove all near duplicates.']}
    rng = random.Random(manifest['seed'])
    # Reject invalid span annotations; preserve nested entities and original Unicode offsets.
    pools = {}; audit = Counter(); seen = set()
    for split in ('dev', 'train'):
        pool = []
        for index, row in enumerate(read_records(source / f'CMeEE_{split}.json')):
            text = row['text']; k = key(text)
            if k in seen: audit[f'ner_{split}_duplicate'] += 1; continue
            seen.add(k)
            ents = {(e['start_idx'], e['end_idx'] + 1, e['type']) for e in row['entities']}
            if any(t not in TYPES or not 0 <= a < b <= len(text) for a, b, t in ents):
                audit[f'ner_{split}_invalid'] += 1; continue
            offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
            starts = {a for a, b in offsets if b > a}; ends = {b for a, b in offsets if b > a}
            if any(a not in starts or b not in ends for a, b, t in ents):
                audit[f'ner_{split}_unaligned'] += 1; continue
            spans = windows(text, tokenizer)
            if any(not any(l <= a < b <= r for l, r in spans) for a, b, t in ents):
                audit[f'ner_{split}_uncovered'] += 1; continue
            pool.append({'id': f'cmeee-{split}-{index}', 'text': text, 'entities': sorted(ents)})
        rng.shuffle(pool); pools[split] = pool
    ner = {'train': pools['train'], 'validation': pools['dev'][:1500], 'test': pools['dev'][1500:]}
    for split, rows in ner.items(): save_json(destination / f'ner-{split}.json', rows)
    manifest['ner_counts'] = {s: {'documents': len(v), 'entities': sum(len(x['entities']) for x in v)} for s, v in ner.items()}
    # Build exact CMeIE schema, with object_type converted to the public trainer format.
    schemas = []
    for i, s in enumerate(read_records(source / '53_schemas.json')):
        schemas.append({**s, 'id': i, 'slot': '@value', 'label': s['predicate'] + '/@value'})
    types = sorted({s[k] for s in schemas for k in ('subject_type', 'object_type')})
    lookup = {(s['predicate'], '@value', s['subject_type'], s['object_type']): s['id'] for s in schemas}
    pools = {}; seen = set()
    for split in ('dev', 'train'):
        pool = []
        for index, raw in enumerate(read_records(source / f'CMeIE_{split}.json')):
            k = key(raw['text'])
            if k in seen: audit[f're_{split}_duplicate'] += 1; continue
            seen.add(k)
            try:
                row = convert_record(raw, lookup, f'CMeIE_{split}.json', index + 1)
                encoded = tokenizer(row['text'], return_offsets_mapping=True)
                if len(encoded['input_ids']) > 384: raise ValueError('over_384_tokens')
                token_spans(encoded['offset_mapping'], row)
            except (KeyError, ValueError, TypeError) as e:
                audit[f're_{split}_{str(e)}'] += 1; continue
            pool.append(row)
        rng.shuffle(pool); pools[split] = pool
    relations = {'train': pools['train'], 'validation': pools['dev'][:1200], 'test': pools['dev'][1200:]}
    relation_dir = destination / 'cmeie-384-v1'; relation_dir.mkdir(exist_ok=True)
    for split, rows in relations.items():
        (relation_dir / f'{split}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
    hashes = {s: digest(relation_dir / f'{s}.jsonl') for s in relations}
    relation_manifest = {'id': 'medical-cmeie-384-v1', 'model': MODEL, 'schemas': schemas, 'entity_types': types,
        'max_source_tokens': 384, 'split_sha256': hashes,
        'dataset_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        'counts': {s: {'documents': len(v), 'triples': sum(len(x['triples']) for x in v)} for s, v in relations.items()},
        'limitations': ['First exact surface occurrence alignment; repeated mentions ambiguous.',
                        'Medical relation participants are not complete NER labels.', 'Independent dev-derived holdout, not official leaderboard score.']}
    save_json(relation_dir / 'manifest.json', relation_manifest)
    manifest['relation_counts'] = relation_manifest['counts']; manifest['audit'] = dict(audit)
    manifest['prepared_sha256'] = {str(p.relative_to(destination)): digest(p) for p in destination.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    save_json(destination / 'manifest.json', manifest)
    return manifest
