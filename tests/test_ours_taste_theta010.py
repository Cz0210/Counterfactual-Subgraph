import numpy as np
import pytest
from src.eval.ours_taste_theta010 import metrics,raw_valid_distances
from src.eval.ours_taste_focus_selector import Selector

def test_coverage_raw_not_cap_probability_or_quantile():
    raw=np.array([[np.inf],[.12],[.08],[.001],[.1],[.01]])
    pred=np.array([1,1,1,0,1,1]);valid=np.array([[1],[1],[1],[1],[1],[0]],bool)
    old=metrics(raw,pred,[0],.008064485938494906,.03,valid)[-1]
    new=metrics(raw,pred,[0],.1,.03,valid)[-1]
    assert new['covered_count']==2 and old['covered_count']==0
    for key in ('reach','capped_mean','conditional_median'):assert old[key]==new[key]
    assert new['conditional_median']==.1

def test_unknown_never_becomes_failure():
    with pytest.raises(ValueError,match='UNKNOWN'):raw_valid_distances(np.array([[np.nan]]),[1])

def test_nested_prefix_and_live_selector_theta():
    d=np.array([[.08,.001],[.12,.1],[np.inf,np.inf]])
    rows=metrics(d,[1,1,1],[0,1],.1,.03)
    assert all(a['coverage']<=b['coverage'] and a['capped_mean']>=b['capped_mean'] for a,b in zip(rows,rows[1:]))
    s=Selector(d,['a','b'],.1,.03,[.008,.02],[1,1])
    assert s.theta==.1 and s.key([0])[0]==-1 and s.key([1])[0]==-2
