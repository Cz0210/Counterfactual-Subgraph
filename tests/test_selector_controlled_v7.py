from types import SimpleNamespace
import json
import numpy as np
import pytest
from src.eval.selector_controlled_v7 import Objective, load_matrix, prefix_rows

def obj(d):
    d=np.array(d,dtype=float);n=d.shape[1]
    return Objective(d,SimpleNamespace(structural_similarity=np.eye(n),normalized_sizes=np.ones(n)))

def test_positive_common_weights_do_not_change_append_argmax():
    o=obj([[.03,.05,.2],[.2,.03,.02],[.07,.2,.05]])
    values=[o.value([0,j],multi=True,prefix=False) for j in (1,2)]
    weighted=[.3*o.value([0],multi=True,prefix=False)+.7*x for x in values]
    assert np.argmax(values)==np.argmax(weighted)

def test_unknown_is_not_infinity(tmp_path):
    p=tmp_path/'pairs.jsonl'
    p.write_text(json.dumps(dict(parent_id='p',candidate_id='c',pair_strict_flip=False,failure_reason='ERROR'))+'\n')
    with pytest.raises(ValueError,match='UNCLASSIFIED'):load_matrix(p,['c'])

def test_same_members_terminal_invariant_and_cost_separate():
    d=np.array([[.12,.2],[np.inf,.04]])
    a=prefix_rows(d,[0,1],.03)[-1];b=prefix_rows(d,[1,0],.03)[-1]
    assert a==b and a['covered']==1 and a['capped_mean']==.03
    assert a['conditional_median']==.08

def test_zero_gain_still_fills_distinct_and_stable_id_tie():
    o=obj([[np.inf]*4]);seq=o.greedy(['d','c','b','a'],multi=False)
    assert seq==[3,2,1,0]

def test_refine_never_changes_fixed_members_or_decreases_objective():
    o=obj([[.01,np.inf,.2],[.15,.02,np.inf]])
    seq,stats=o.refine([2,1,0],multi=True,prefix=True,replacement=False,max_proposals=40)
    assert set(seq)=={0,1,2}
    assert o.value(seq,multi=True,prefix=True)>=o.value([2,1,0],multi=True,prefix=True)
    assert stats['proposals']<=40
