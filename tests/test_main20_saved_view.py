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
