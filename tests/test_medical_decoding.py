import itertools
import pytest

from nlp_lab.medical.data import TYPES
from nlp_lab.medical.decoding import decode,fit_thresholds,micro,DecisionPredictor


def test_nested_decode_preserves_nested_adjacent_and_multilabel_spans():
    spans=[(0,4,'dis',2.),(1,3,'bod',1.),(2,5,'sym',.9),(4,6,'dru',1.),(0,4,'sym',1.)]
    selected=decode([spans],{t:0. for t in TYPES},'nested')[0]
    assert (2,5,'sym',.9) not in selected
    assert {(s[0],s[1],s[2]) for s in selected}=={(0,4,'dis'),(1,3,'bod'),(4,6,'dru'),(0,4,'sym')}
    assert len(decode([spans],{t:0. for t in TYPES})[0])==5
    with pytest.raises(ValueError):decode([spans],{'dis':0.})


def test_fractional_micro_optimization_matches_exhaustive_threshold_search():
    rows=[{'entities':[(0,1,'dis'),(2,3,'sym')]},{'entities':[(0,1,'dis')]}]
    predicted=[[(0,1,'dis',.2),(2,3,'sym',-.2),(4,5,'dis',-.1)],[(0,1,'dis',.3),(1,2,'sym',.6)]]
    grid=[-.5,0.,.5];baseline=0.
    fitted,support=fit_thresholds(rows,predicted,grid,baseline,min_support=1)
    candidates=[]
    for dis,sym in itertools.product(grid,repeat=2):
        thresholds={t:baseline for t in TYPES};thresholds.update(dis=dis,sym=sym)
        candidates.append(micro(rows,decode(predicted,thresholds)))
    assert micro(rows,decode(predicted,fitted))==max(candidates)
    assert support['dep']==0 and fitted['dep']==baseline


def test_decision_predictor_uses_lowest_threshold_once():
    class Base:
        tokenizer='test'
        def predict(self,texts,threshold):
            assert texts==['肺癌'] and threshold==-.5
            return [[(0,2,'dis',.2),(0,1,'bod',.2)]]
    thresholds={t:.5 for t in TYPES};thresholds['dis']=-.5
    p=DecisionPredictor(Base(),{'thresholds':thresholds,'policy':'raw'})
    assert p.predict(['肺癌'])==[[(0,2,'dis',.2)]]
