"""Diagnostic relation slices; no threshold fitting or clinical conclusions."""
from collections import Counter
from ..ie.data import entities
from ..ie.metrics import evaluate, relation_key


def punctuation_variant(rows):
    """One-code-point typography changes keep all annotated offsets unchanged."""
    table=str.maketrans({',':'，',':':'：',';':'；','(':'（',')':'）'})
    return [{**r,'text':r['text'].translate(table)} for r in rows]


def error_partition(rows,predictions):
    counts=Counter()
    for row,pred in zip(rows,predictions,strict=True):
        gold={relation_key(row,t):t for t in row['triples']}
        predicted={relation_key(row,t) for t in pred['triples']}
        mentions={tuple(e[:3]) for e in pred['entities']}
        for key,t in gold.items():
            if key in predicted:counts['correct']+=1
            elif tuple(t[:3]) not in mentions or tuple(t[4:7]) not in mentions:
                counts['missing_with_entity_error']+=1
            else:counts['missing_despite_both_entities']+=1
        counts['extra']+=len(predicted-gold.keys())
    return {k:counts[k] for k in ('correct','missing_with_entity_error','missing_despite_both_entities','extra')}


def slices(train,rows,predictions,manifest):
    frequencies=Counter(t[3] for r in train for t in r['triples'])
    groups={'short_text':[],'long_text':[],'shared_entity':[],'rare_relation':[]}
    for i,row in enumerate(rows):
        groups['long_text' if len(row['text'])>=150 else 'short_text'].append(i)
        uses=Counter(tuple(t[:3]) for t in row['triples'])+Counter(tuple(t[4:7]) for t in row['triples'])
        if any(v>1 for v in uses.values()):groups['shared_entity'].append(i)
        if any(frequencies[t[3]]<100 for t in row['triples']):groups['rare_relation'].append(i)
    result={}
    for name,indices in groups.items():
        selected=[rows[i] for i in indices];pred=[predictions[i] for i in indices]
        metrics,_=evaluate(selected,pred,manifest)
        result[name]={'documents':len(indices),'relation':metrics['relation'],'errors':error_partition(selected,pred)}
    return result


def release_decision(baseline_validation,candidate_validation,baseline_test,candidate_test,min_gain=.005):
    """Validation selects; reused holdout can veto release, never tune thresholds."""
    validation_pass=candidate_validation-baseline_validation>=min_gain
    audit_pass=candidate_test>=baseline_test
    return {'enabled':validation_pass and audit_pass,'validation_gate_passed':validation_pass,
            'reused_holdout_release_audit_passed':audit_pass,'min_validation_gain':min_gain,
            'scope':'Secondary reused-holdout release audit, not an independently unseen final test.'}
