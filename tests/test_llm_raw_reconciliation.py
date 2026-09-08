import copy
from pathlib import Path
import pytest
from src.experiments import llm_raw_reconciliation as r
from src.experiments import bace_eval_migration as m


def test_old_conflicts_not_small_delta_acceptance():
    with pytest.raises(ValueError,match='CALIBRATION_CONFLICT_REDUCTION_REQUIRED'):
        r.classify({'calibration':{'conflicts':{'pair':{'values':[1.,1.+1e-12]}}},
                    'test':{'conflicts':{}}})


def test_test_only_scope_preserves_freeze():
    result=r.classify({'calibration':{'conflicts':{}},'test':{'conflicts':{'p':{}}}})
    assert result['pair_counts']=={'calibration':0,'test':1}
    assert result['calibration_values_changed'] is False
    assert result['historical_numerical_equality_claimed'] is False


def test_overlay_cannot_target_gnn_or_original_root(tmp_path):
    spec={'family':'llm_gin','output_root':str(tmp_path/'old')}
    assert r.overlay_path(spec,tmp_path/'fresh')==tmp_path/'fresh'
    with pytest.raises(ValueError):r.overlay_path(spec,tmp_path/'old')
    with pytest.raises(ValueError):r.overlay_path(dict(spec,family='gnn_a'),tmp_path/'fresh')


def test_numeric_context_changes_canonical_key():
    pair={'parent':'attributed graph','residual':'other graph'}
    first=m.stable_sha256(dict(pair_identity=pair,numerical_contract={'device':'cpu','dtype':'float32'}))
    second=m.stable_sha256(dict(pair_identity=pair,numerical_contract={'device':'cuda','dtype':'float32'}))
    assert first!=second


def test_array_proof_binds_dtype_shape_and_values():
    import numpy as np
    a=np.array([[1,2]],dtype=np.float32)
    assert r.array_proof(a)!=r.array_proof(a.astype(np.float64))
    assert r.array_proof(a)!=r.array_proof(a.reshape(2,1))
    assert r.array_proof(a)!=r.array_proof(a+1)


def test_no_calibration_or_gnn_mutation_in_repair():
    import inspect
    source=inspect.getsource(r.reconcile)
    assert 'CALIBRATION_FREEZE.json' not in source
    assert 'original_sources_modified=False' in source
    assert "result['test']['conflicts'].items()" in source
    assert 'result[\'calibration\'][\'index\']' not in source


def test_default_migration_still_rejects_conflict():
    import inspect
    source=inspect.getsource(m.load_raw)
    assert "if conflicts is None:" in source
    assert "raise ValueError('RAW_GRAPH_DISTANCE_CONFLICT:'+key)" in source


def test_recompute_uses_frozen_kernel_not_min_or_rounding():
    import inspect
    source=inspect.getsource(r.measure)
    assert 'compute_node_wasserstein_distance(left.H,right.H' in source
    assert 'round(' not in source and 'min(' not in source
    assert 'new_measurement_proves_old_producer_correct=False' in source
