import math
import pytest
from src.baselines.main20_saved_view import reduce_distances,exact_curve,number

def test_exact_failed_recourses_are_not_missing_rows():
    rows=[dict(parent_id='a',best_distance='.01',strict_recourse_available='true'),
          dict(parent_id='b',best_distance='N/A',strict_recourse_available='false')]
    r,values=reduce_distances(rows,.02,.04)
    assert r['coverage']==.5 and r['finite_recourse_count']==1 and r['fixed_capped_mean_cost']==.025
    assert next(x for x in exact_curve(values,.02) if x['distance']==.02)['coverage']==.5
    assert r['conditional_median_cost']==.01

def test_missing_unknown_and_nan_rejected():
    with pytest.raises(ValueError):reduce_distances([dict(parent_id='N/A')],.02,.04)
    with pytest.raises(ValueError):number('nan')
    with pytest.raises(ValueError):reduce_distances([dict(parent_id='a',best_distance='N/A',strict_recourse_available='true')],.02,.04)

def test_k20_numeric_presence_distinct_from_registration_and_audit():
    from src.baselines.main20_saved_view import k20_numeric_complete
    r=dict(coverage=.5,cost=.03,N=10,finite_recourse_count=8,stage='SAVED_RECORDS')
    assert k20_numeric_complete(r)
    assert not k20_numeric_complete(dict(r,finite_recourse_count='N/A'))
    assert not k20_numeric_complete(dict(r,stage='UNDER_REPAIR'))
    assert not k20_numeric_complete(dict(r,cost='PENDING'))
    assert k20_numeric_complete(dict(r,coverage=0,finite_recourse_count=0,cost='N/A',cost_definition='ORIGINAL_CONDITIONAL'))
