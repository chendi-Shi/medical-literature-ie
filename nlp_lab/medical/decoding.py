"""Validation-only span decision policies; scores remain uncalibrated logits."""
import math

from .data import TYPES


def decode(predictions, thresholds, policy='raw'):
    if set(thresholds)!=set(TYPES) or any(not math.isfinite(v) for v in thresholds.values()):
        raise ValueError('Finite threshold required for every medical type')
    if policy not in ('raw','nested'):raise ValueError('Unknown span policy')
    result=[]
    for spans in predictions:
        selected=[s for s in spans if s[3]>thresholds[s[2]]]
        if policy=='nested':
            kept=[]
            for s in sorted(selected,key=lambda x:(-x[3],x[0],x[1],x[2])):
                # Preserve containment, identical spans with different types and
                # adjacent spans. Only crossing boundaries are disallowed.
                if not any(s[0]<k[0]<s[1]<k[1] or k[0]<s[0]<k[1]<s[1] for k in kept):kept.append(s)
            selected=kept
        result.append(sorted(selected))
    return result


def counts(rows,predictions):
    if len(rows)!=len(predictions):raise ValueError('Rows and predictions must align')
    result=[]
    for row,pred in zip(rows,predictions):
        g={tuple(s) for s in row['entities']};p={tuple(s[:3]) for s in pred}
        result.append((len(g&p),len(p),len(g)))
    return result


def f1(total):
    correct,predicted,gold=total
    return 2*correct/max(1,predicted+gold)


def micro(rows,predictions):
    return f1(tuple(map(sum,zip(*counts(rows,predictions)))))


def fit_thresholds(rows,predictions,grid,baseline,min_support=50):
    """Optimize validation micro F1 over independent per-type threshold counts.

    Fractional optimization (Dinkelbach) uses no test data. Low-support labels
    keep the original global threshold to reduce unstable label decisions.
    """
    options={};supports={}
    for typ in TYPES:
        gold=[{(a,b,t) for a,b,t in r['entities'] if t==typ} for r in rows]
        support=sum(map(len,gold));supports[typ]=support;options[typ]=[]
        thresholds=grid if support>=min_support else [baseline]
        for threshold in thresholds:
            tp=predicted=0
            for g,spans in zip(gold,predictions):
                p={tuple(s[:3]) for s in spans if s[2]==typ and s[3]>threshold}
                tp+=len(p&g);predicted+=len(p)
            options[typ].append((threshold,tp,predicted))
    current={t:baseline for t in TYPES};q=micro(rows,decode(predictions,current))
    for _ in range(50):
        chosen={t:max(values,key=lambda v:(2*v[1]-q*v[2],-abs(v[0]-baseline),-v[0])) for t,values in options.items()}
        updated=f1((sum(v[1] for v in chosen.values()),sum(v[2] for v in chosen.values()),sum(supports.values())))
        current={t:v[0] for t,v in chosen.items()}
        if abs(updated-q)<1e-12:break
        q=updated
    return current,supports


class DecisionPredictor:
    """Use the original frozen encoder/checkpoint through a separate recipe."""
    def __init__(self,base,recipe):
        self.base=base;self.recipe=recipe;self.tokenizer=base.tokenizer
    def predict(self,texts):
        thresholds=self.recipe['thresholds']
        return decode(self.base.predict(texts,threshold=min(thresholds.values())),thresholds,self.recipe['policy'])
