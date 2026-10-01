"""Descriptive role/participant error categories, without model or threshold tuning."""
from collections import Counter
from nlp_lab.data import read_json
from nlp_lab.ie.data import entities
from nlp_lab.ie.metrics import relation_key

def categories(rows,errors,manifest,exclude_first=12):
    lookup={(r.get('source_file'),r.get('source_line')):(i,r) for i,r in enumerate(rows)}
    counts=Counter();confusions=Counter()
    for error in errors:
        i,row=lookup[(error['source_file'],error['source_line'])]
        if i<exclude_first:continue
        gold={relation_key(row,t) for t in row['triples']}
        missing={tuple(t) for t in error['missing']};extra={tuple(t) for t in error['extra']}
        pair=lambda t:(t[0],t[1],t[3],t[4])
        pred_entities={(row['text'][a:b],typ) for a,b,typ in error['predicted_entities']}
        for triple in missing:
            if {(triple[0],triple[1]),(triple[3],triple[4])}<=pred_entities:
                counts['missing_despite_both_typed_surfaces_detected']+=1
            else:counts['missing_with_participant_surface_or_type_absent']+=1
            swapped=[t for t in extra if pair(t)==pair(triple)]
            if swapped:
                counts['missing_with_wrong_role_on_same_typed_pair']+=1
                for t in swapped:confusions[(triple[2],t[2])]+=1
        for triple in extra:
            counts['extra_wrong_role_on_gold_typed_pair' if any(pair(t)==pair(triple) for t in gold) else 'extra_other_pair_or_type']+=1
    return {'documents':len(rows)-exclude_first,'counts':dict(counts),
            'role_confusions':[{'gold':manifest['schemas'][a]['label'],'predicted':manifest['schemas'][b]['label'],'count':n}
                               for (a,b),n in confusions.most_common(15)],
            'interpretation':'描述性错误分类；同表面字符串且同类型实体对。错角色与漏关系类别可重叠；不是人工因果归因。'}

def analyze_runs(workspace,rows,manifest,experiments):
    return {e['name']:categories(rows,read_json(workspace/'ie/runs'/e['run']/'test_errors.json'),manifest)
            for e in experiments if e['seed']==42 and e['name']!='frozen_encoder'}
