from collections import Counter
import numpy as np
from .data import entities


def prf(correct,predicted,gold):
    return {'precision':correct/predicted if predicted else 0.,'recall':correct/gold if gold else 0.,
            'f1':2*correct/(predicted+gold) if predicted+gold else 0.,'correct':correct,'predicted':predicted,'gold':gold}


def relation_key(row, triple):
    a,b,typ,r,c,d,objtyp=triple[:7]
    return row['text'][a:b],typ,r,row['text'][c:d],objtyp


def evaluate(rows,predictions,manifest):
    if len(rows)!=len(predictions):raise ValueError('信息抽取样本数不同')
    entity_counts=Counter();triple_counts=Counter();by_relation={r['id']:Counter() for r in manifest['schemas']}
    details=[]
    for row,pred in zip(rows,predictions):
        gold_e=set(entities(row));pred_e={tuple(x[:3]) for x in pred['entities']}
        gold_t={relation_key(row,x) for x in row['triples']};pred_t={relation_key(row,x) for x in pred['triples']}
        for counts,g,p in ((entity_counts,gold_e,pred_e),(triple_counts,gold_t,pred_t)):
            counts.update(correct=len(g&p),predicted=len(p),gold=len(g))
        for r,counts in by_relation.items():
            g={x for x in gold_t if x[2]==r};p={x for x in pred_t if x[2]==r}
            counts.update(correct=len(g&p),predicted=len(p),gold=len(g))
        details.append({'text':row['text'],'source_file':row.get('source_file'),'source_line':row.get('source_line'),
                        'correct':len(gold_t&pred_t),'gold':len(gold_t),'predicted':len(pred_t),
                        'missing':[list(x) for x in sorted(gold_t-pred_t)],'extra':[list(x) for x in sorted(pred_t-gold_t)],
                        'gold_entities':[list(x) for x in sorted(gold_e)],'predicted_entities':[list(x) for x in sorted(pred_e)]})
    per_relation=[{**s,**prf(c['correct'],c['predicted'],c['gold'])} for s in manifest['schemas'] for c in [by_relation[s['id']]]]
    result={'documents':len(rows),'entity':prf(entity_counts['correct'],entity_counts['predicted'],entity_counts['gold']),
            'relation':prf(triple_counts['correct'],triple_counts['predicted'],triple_counts['gold']),
            'per_relation':per_relation,'macro_relation_f1':float(np.mean([x['f1'] for x in per_relation])),
            'counts_per_document':[{k:x[k] for k in ('correct','gold','predicted')} for x in details],
            'entity_metric':'typed exact derived character span, relation participants only',
            'relation_metric':'typed exact subject/object surface + schema slot, not official DuIE multislot/alias metric'}
    return result,details


def paired_interval(reference,candidate,replicates=600,seed=31415):
    r=np.array([[x[k] for k in ('correct','predicted','gold')] for x in reference],dtype=float)
    c=np.array([[x[k] for k in ('correct','predicted','gold')] for x in candidate],dtype=float)
    if r.shape!=c.shape:raise ValueError('配对样本顺序不同')
    def f1(a):
        totals=a.sum(0);return 2*totals[0]/max(1,totals[1]+totals[2])
    rng=np.random.default_rng(seed)
    deltas=[]
    for _ in range(replicates):
        i=rng.integers(0,len(r),len(r));deltas.append(f1(c[i])-f1(r[i]))
    return {'delta_relation_f1':f1(c)-f1(r),'ci95':np.quantile(deltas,[.025,.975]).tolist(),'replicates':replicates,
            'scope':'按文本配对重采样，固定模型；不包含训练种子总体不确定性，无多重比较校正'}
