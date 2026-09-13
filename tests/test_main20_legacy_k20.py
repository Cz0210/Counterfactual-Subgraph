import pytest
from scripts.reduce_main20_legacy_k20 import reduce

def row(p,c,d):return dict(parent_id=p,candidate_id=c,distance=d,pred_before=1,pred_after=0,teacher_strict_flip='True',delete_valid='True',solver='exact_emd2',feature_cost='cosine',node_mass='uniform')

def test_original_prefix_and_cap_are_separate_from_conditional():
    p,d=reduce([row('p','a',.4),row('p','b',.1),row('q','a',.6),row('q','b',.9)],['a','b'],.2,.5)
    assert p[-1]['covered_count']==1 and p[-1]['denominator']==2
    assert p[-1]['fixed_capped_mean']==.3 and p[-1]['conditional_median']==.35
    assert p[-1]['k_effective']==2 and p[1]['coverage']==p[-1]['coverage']

def test_incomplete_pairs_are_not_zero():
    with pytest.raises(ValueError,match='Incomplete'):reduce([row('p','a',.1)],['a','b'],.2,.5)
