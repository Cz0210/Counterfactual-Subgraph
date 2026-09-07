import pytest
from src.eval.bace_reach_result_display import checked_control_rows


def sample():
    return dict(cohort_size=2, parent_rows=[r for k in range(1,21) for r in (
        dict(k=k,parent_id='a',best_distance=.2,capped_distance=.2,strict_recourse_available=True,theta_star_covered=True),
        dict(k=k,parent_id='b',best_distance=None,capped_distance=.4,strict_recourse_available=False,theta_star_covered=False))],
        prefix_rows=[dict(k=k,strict_flip_parent_count=1,num_theta_star_covered=1,ccrcov_theta_star=.5,
                         fixed_capped_mean_cost=.3,conditional_median_cost=.2) for k in range(1,21)])


def test_infinite_not_zero_and_costs_keep_distinct_scope():
    rows=checked_control_rows(sample(),parent_count=2,theta=.25,cap=.4)
    assert rows[9]['finite_reach']==1 and rows[9]['conditional_median']==.2
    assert rows[9]['capped_mean']==pytest.approx(.3)
    assert rows[9]['ecdf'][-1][1]==.5


def test_metric_conflict_is_not_silently_replotted():
    data=sample();data['prefix_rows'][9]['ccrcov_theta_star']=1.
    with pytest.raises(ValueError,match='NUMERICAL_SOURCE_CONFLICT'):
        checked_control_rows(data,parent_count=2,theta=.25,cap=.4)


def test_missing_parent_and_non_nested_rejected():
    data=sample();data['parent_rows'].pop()
    with pytest.raises(ValueError,match='PARTITION_CONFLICT'):
        checked_control_rows(data,parent_count=2,theta=.25,cap=.4)
    data=sample();data['parent_rows'][2]['best_distance']=.3
    with pytest.raises(ValueError,match='NONNESTED'):
        checked_control_rows(data,parent_count=2,theta=.25,cap=.4)
