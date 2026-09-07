import math
from pathlib import Path
import pytest
from src.experiments.bace_gin_fixed_pool import prefix_metrics, validate_spec, EXPERIMENT

def pairs():
    return [dict(parent_id=p,candidate_id=f'c{i}',pred_before=before,pred_after=0,
            pair_strict_flip=before==1,wnode_distance=.01+i*.001 if before==1 else None)
        for p,before in [('p1',1),('p2',0)] for i in range(15)]

def test_fixed_base_and_native_share_same_at_most_sequence():
    result=prefix_metrics(['p1','p2'],[f'c{i}' for i in range(15)],pairs(),theta=.02,cap=.03,endpoints=[.01,.03])
    fixed=[r for r in result['prefix_rows'] if r['cohort']=='fixed141']
    native=[r for r in result['prefix_rows'] if r['cohort']=='gin_native']
    assert fixed[9]['K_effective']==10 and fixed[19]['K_effective']==15
    assert all(r['coverage']==.5 and r['denominator']==2 for r in fixed)
    assert all(r['coverage']==1 and r['denominator']==1 for r in native)
    assert fixed[19]['fixed_capped_mean']==.02
    assert fixed[19]['conditional_median']==.01
    assert [r['K_effective'] for r in fixed[15:]]==[15]*5

def test_old_gine_flip_rejected():
    rows=pairs(); rows[-1]['pair_strict_flip']=True; rows[-1]['wnode_distance']=.001
    with pytest.raises(ValueError,match='WRONG_STRICT_FLIP'):
        prefix_metrics(['p1','p2'],[f'c{i}' for i in range(15)],rows,theta=.02,cap=.03,endpoints=[.03])

def test_missing_is_not_zero():
    with pytest.raises(ValueError,match='CARTESIAN'):
        prefix_metrics(['p1','p2'],[f'c{i}' for i in range(15)],pairs()[:-1],theta=.02,cap=.03,endpoints=[.03])

def test_duplicate_rule_rejected():
    with pytest.raises(ValueError,match='PREFIX'):
        prefix_metrics(['p1'],['a','a'],[],theta=.02,cap=.03,endpoints=[.03])

def spec():
    return dict(experiment_id=EXPERIMENT,training_rerun=False,temperature_refit=False,
        candidate_generation_repeated=False,reach_v2_candidates_used=False,main_matrix_write=False,
        pools={m:{} for m in ('ours','globalgce','gcfexplainer','comrecgc')},
        source_class=1,destination_class=0,base_counts={'calibration':66,'test':141},
        test_results_previously_observed=True,rule_budget_semantics='AT_MOST_K',output_root='/private/tmp/gin-fixture')

def test_independent_posthoc_scope():
    validate_spec(spec())
    for field in ('main_matrix_write','reach_v2_candidates_used','training_rerun','temperature_refit'):
        changed=spec(); changed[field]=True
        with pytest.raises(ValueError): validate_spec(changed)

def test_fixed141_cannot_be_old96_common():
    changed=spec(); changed['base_counts']['test']=96
    with pytest.raises(ValueError,match='COHORT'): validate_spec(changed)

def test_empty_native_is_na():
    rows=[dict(parent_id='p',candidate_id='c',pred_before=0,pred_after=0,pair_strict_flip=False,wnode_distance=None)]
    result=prefix_metrics(['p'],['c'],rows,theta=.02,cap=.03,endpoints=[.03])
    assert result['prefix_rows'][0]['coverage']==0
    assert result['prefix_rows'][1]['coverage'] is None
    assert result['prefix_rows'][0]['conditional_median'] is None

def test_script_has_real_isolated_bootstrap():
    script=(Path(__file__).parents[1]/'scripts/experiments/run_bace_gin_fixed_pool.py').read_text()
    assert 'sys.path.insert(0' in script and '--config' in script
