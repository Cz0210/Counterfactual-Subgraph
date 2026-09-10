import math
import pytest
from src.baselines.cm_crem_comparison import reduce_saved, legacy_records, require_distance_binding

def rows():
    return [dict(parent_id='a',strict_recourse_available='True',best_distance='.1',theta_star='.2',cost_cap='.5'),
            dict(parent_id='b',strict_recourse_available='False',best_distance='N/A',theta_star='.2',cost_cap='.5')]

def test_fixed_denominator_and_exact_ecdf():
    m,e=reduce_saved(rows(),['a','b'],.2,.5)
    assert m['coverage']==.5 and m['fixed_capped_mean_cost']==.3
    assert m['finite_recourse_count']==1 and m['conditional_median_cost']==.1
    assert e==[{'distance':0.,'coverage':0.},{'distance':.1,'coverage':.5}]

def test_distance_beyond_cap_is_not_replaced_in_ecdf():
    r=rows();r[0]['best_distance']='.9'
    m,e=reduce_saved(r,['a','b'],.2,.5)
    assert m['coverage']==0 and m['fixed_capped_mean_cost']==.5 and e[-1]['distance']==.9

@pytest.mark.parametrize('kind',['missing','duplicate','threshold','flag','nan'])
def test_rejects_bad_records(kind):
    r=rows()
    if kind=='missing': r.pop()
    if kind=='duplicate': r[1]['parent_id']='a'
    if kind=='threshold': r[0]['theta_star']='.3'
    if kind=='flag': r[1]['best_distance']='.1'
    if kind=='nan': r[0]['best_distance']='nan'
    with pytest.raises(ValueError):reduce_saved(r,['a','b'],.2,.5)

def test_undefined_median_not_zero():
    r=rows();r[0].update(strict_recourse_available='False',best_distance='inf')
    m,e=reduce_saved(r,['a','b'],.2,.5)
    assert m['conditional_median_cost'] is None and m['coverage']==0

def test_gin_aplus_rejected():
    with pytest.raises(ValueError,match='original BACE'):
        legacy_records({'resolved_parents':{'test':{'ordered_ids':['a']}},'resolved_oracle':{'backbone':'gin'},
                        'resolved_wnode':{},'resolved_evaluation':{}},[],[],{})

def test_shared_molclr_name_does_not_replace_numeric_producer():
    with pytest.raises(ValueError,match='SOURCE_PROVENANCE_GAP'):
        require_distance_binding(None,{})
