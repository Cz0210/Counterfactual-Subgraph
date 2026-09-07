import copy
from pathlib import Path

import pytest

from src.experiments.bace_gin_audit import audit_parent_rows, compare_metrics, recompute_metrics
from src.experiments.bace_gin_fixed_pool import prefix_metrics


ORACLE = dict(model_sha256='model', temperature=1.2, temperature_sha256='temperature')


def fixture():
    common = dict(parent_id='p', parent_smiles='CC', candidate_id='c', split='test',
        oracle_checkpoint_hash='model', oracle_backend='gnn', oracle_backbone='gin',
        oracle_temperature=1.2, rf_oracle_used=False, pred_before=1, pred_after=0,
        p_before=[.2, .8], p_after=[.9, .1], p1_before=.8, p1_after=.1,
        cf_drop=.7, residual_smiles='C', canonical_fragment='C')
    apps = [dict(common, match_index=i, match_atom_indices=[i], delete_valid=True,
        sanitize_ok=True, residual_connected=True, residual_num_components=1,
        cf_flip=True, teacher_strict_flip=True, distance_ok=True, wnode_distance=value)
        for i, value in [(0, .02), (1, .01)]]
    pair = dict(common, applicable=True, num_matches=2, num_valid_residuals=2,
        num_strict_flip_matches=2, pair_strict_flip=True, best_match_index=1,
        best_match_atom_indices=[1], wnode_distance=.01, distance_for_selection=.01)
    return [pair], apps


def run(pairs, apps, method='ours'):
    return audit_parent_rows(pairs, apps, method=method, split='test', parent_id='p',
        parent_smiles='CC', candidate_ids=['c'], oracle=ORACLE)


def test_own_minimum_and_strict_flip_recomputed():
    pairs, apps = fixture()
    assert run(pairs, apps)['strict_flip_pairs'] == 1


@pytest.mark.parametrize('change, message', [
    ('missing', 'COUNT_EVIDENCE_GAP'), ('wrong_min', 'OWN_MINIMUM'),
    ('flip', 'STRICT_FLIP_MISMATCH'), ('distance', 'NOT_ZERO_COVERAGE'),
    ('model', 'GIN_BINDING'), ('temperature', 'TEMPERATURE_MISMATCH'),
    ('mapping', 'MINIMUM_MAPPING'), ('pred', 'ARGMAX'),
    ('duplicate', 'DUPLICATE_APPLICATION')])
def test_concrete_saved_record_conflicts_rejected(change, message):
    pairs, apps = fixture()
    if change == 'missing': apps.pop()
    if change == 'wrong_min': pairs[0]['wnode_distance'] = .02
    if change == 'flip': apps[0]['cf_flip'] = False
    if change == 'distance': apps[0]['distance_ok'] = False
    if change == 'model': apps[0]['oracle_checkpoint_hash'] = 'GINE'
    if change == 'temperature': apps[0]['oracle_temperature'] = 1.
    if change == 'mapping': pairs[0]['best_match_atom_indices'] = [0]
    if change == 'pred': apps[0]['pred_after'] = 1
    if change == 'duplicate': apps[0]['match_index'] = 1
    with pytest.raises(ValueError, match=message): run(pairs, apps)


def test_real_unmatched_pair_can_have_no_apps_not_fabricated():
    pairs, _ = fixture()
    pairs[0].update(applicable=False, num_matches=0, num_valid_residuals=0,
        num_strict_flip_matches=0, pair_strict_flip=False, wnode_distance=None, distance_for_selection='+inf')
    assert run(pairs, [])['application_count'] == 0


def test_native_fullgraph_does_not_require_deletion_mapping():
    pairs, apps = fixture()
    pair = pairs[0]
    pair.update(temperature_sha256='temperature')
    app = dict(pair, cf_flip=True, teacher_strict_flip=True, distance_ok=True,
        native_record_kind='complete_graph_intervention', operation_is_deletion=False,
        match_atom_indices=None, delete_valid=None)
    assert run([pair], [app], method='comrecgc')['application_count'] == 1
    app['operation_is_deletion'] = True
    with pytest.raises(ValueError, match='RELABELED'): run([pair], [app], method='comrecgc')


def test_independent_metrics_include_infinite_capped_and_native_same_order():
    ids = [f'c{i}' for i in range(15)]
    rows = [dict(parent_id=p, candidate_id=c, pred_before=int(p=='a'), pred_after=0,
                 pair_strict_flip=p=='a' and c=='c9', wnode_distance=.01 if p=='a' and c=='c9' else None)
            for p in ['a', 'b'] for c in ids]
    args = dict(theta=.02, cap=.03, endpoints=[0., .01, .02, .03])
    independent = recompute_metrics(['a', 'b'], ids, rows, **args)
    producer = prefix_metrics(['a', 'b'], ids, rows, **args)
    compare_metrics(producer, independent)
    fixed = [r for r in independent['prefix_rows'] if r['cohort']=='fixed141']
    assert fixed[0]['fixed_capped_mean'] == .03
    assert fixed[0]['conditional_median'] is None
    assert fixed[9]['fixed_capped_mean'] == .02 and fixed[9]['coverage'] == .5
    assert all(r['K_effective'] == 15 for r in fixed[15:])
    changed = copy.deepcopy(producer); changed['prefix_rows'][18]['coverage'] = 1.
    with pytest.raises(ValueError, match='METRIC_MISMATCH'): compare_metrics(changed, independent)


def test_empty_native_is_na_and_not_zero():
    rows = [dict(parent_id='p',candidate_id='c',pred_before=0,pred_after=0,pair_strict_flip=False,wnode_distance=None)]
    result = recompute_metrics(['p'], ['c'], rows, theta=.02, cap=.03, endpoints=[.03])
    assert result['prefix_rows'][0]['coverage'] == 0
    assert result['prefix_rows'][1]['coverage'] is None


def test_cli_has_paired_cpu_audit_no_science():
    root = Path(__file__).parents[1]
    script = (root/'scripts/experiments/audit_bace_gin_fixed_pool.py').read_text()
    slurm = (root/'scripts/slurm/audit_bace_gin_fixed_pool.sh').read_text()
    assert 'sys.path.insert(0' in script and '--config' in script
    assert '--config configs/hpc.yaml' in slurm and 'CUDA_VISIBLE_DEVICES=""' in slurm
    assert 'audit_bace_gin_fixed_pool.py' in slurm
