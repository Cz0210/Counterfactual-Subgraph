from dataclasses import dataclass
import pytest
from src.ablations.gnn.reach_v2_adapter import (
    validate_pool, parent_binding, require_global_freeze, SCOPE_NAME)
from src.eval.bace_frozen_gnn_contracts import stable_sha256


@dataclass
class Parent:
    parent_id: str
    smiles: str


def test_own_pool_retained_no_old_audit_relabel():
    old = [str(i) for i in range(66)]
    pool = [{'candidate_id': i} for i in old + ['new']]
    receipt = dict(candidate_universe_sha256='new-sha', test_opened=False,
                   calibration_opened_during_search=False)
    result = validate_pool(pool, receipt, current_pool_sha='new-sha', old_ids=old)
    assert result['candidate_count'] == 67
    assert not result['old_gnn_result_adopted_as_v2']
    with pytest.raises(ValueError):
        validate_pool(pool[1:], receipt, current_pool_sha='new-sha', old_ids=old)


def test_backbone_flip_cache_cannot_be_shared():
    common = dict(parent=Parent('train-id', 'CCO'), candidates=[{'candidate_id': 'a'}],
                  checkpoint_id='same-weight-name', temperature_sha='same-temp-name', pool_sha='new', split='calibration')
    assert parent_binding(**common, backbone='gin') != parent_binding(**common, backbone='gine')
    first = parent_binding(**common, backbone='gin')
    common['temperature_sha'] = 'new-temp'
    assert first != parent_binding(**common, backbone='gin')


def test_old_66_freeze_does_not_allow_new_test():
    with pytest.raises(ValueError, match='ALL_TEN'):
        require_global_freeze({'all_five_calibration_orders_frozen': True}, 'v2')


def test_all_ten_scope_bound_before_test():
    selectors = {}
    for name in ('gine', 'gin', 'gcn', 'gatv2', 'gatedgcn_plus'):
        for mode in ('native', 'common'):
            order = [str(i) for i in range(20)]
            row = dict(backbone=name, cohort_mode=mode, pool_sha256='v2', test_loaded=False,
                       global_selector_after_complete_merge=True, ordered_rule_ids=order,
                       prefixes={str(k): order[:k] for k in range(1, 21)})
            row['self_sha256'] = stable_sha256(row)
            selectors[f'{name}/{mode}'] = row
    freeze = dict(scope=SCOPE_NAME, pool_sha256='v2', test_loaded=False, selectors=selectors)
    assert require_global_freeze(freeze, 'v2')
    selectors['gin/common']['test_loaded'] = True
    with pytest.raises(ValueError):
        require_global_freeze(freeze, 'v2')


def calibration_fixture(root):
    from src.eval.bace_frozen_gnn_contracts import atomic_json
    from src.ablations.gnn.reach_v2_adapter import BACKBONES
    candidates = [{'candidate_id': f'r{i}', 'canonical_fragment': 'C'} for i in range(20)]
    for name in BACKBONES:
        for index, pid in enumerate(('cal-a', 'cal-b')):
            directory = root / name / 'calibration' / f'{index:04d}'
            rows = [dict(parent_id=pid, candidate_id=c['candidate_id'], applicable=True,
                         pair_strict_flip=True, wnode_distance=.05, cf_drop=.2) for c in candidates]
            science = dict(pair_rows=rows, match_rows=[])
            atomic_json(directory / 'parents' / f'{pid}.json', dict(scope=SCOPE_NAME,
                backbone=name, pool_sha256='v2', science= science, science_sha256=stable_sha256(science)))
            atomic_json(directory / 'terminal.json', dict(scope=SCOPE_NAME, backbone=name,
                spec_sha256='spec', pool_sha256='v2', split='calibration', index=index,
                state='PARENT_CHUNK_COMPLETE_NOT_CORE_PASS', global_selector_called=False,
                main_matrix_write=False, native_cohort_ids=['cal-a', 'cal-b'],
                parent_ids=[pid], pair_count=len(rows)))
    return candidates


def test_calibration_chunks_complete_before_one_global_merge(tmp_path):
    from src.ablations.gnn.reach_v2_adapter import merge_and_freeze_calibration, BACKBONES
    candidates = calibration_fixture(tmp_path)
    orders = {f'{name}/{mode}': [c['candidate_id'] for c in candidates]
              for name in BACKBONES for mode in ('native', 'common')}
    thresholds = dict(theta_star=.1, cost_cap=.5,
                      merged_thresholds=[dict(threshold=.1, weight=1)])
    result = merge_and_freeze_calibration(tmp_path, spec_sha='spec', pool_sha='v2', slots=2,
        candidates=candidates, thresholds=thresholds, old_orders=orders, solver_seconds=0)
    assert require_global_freeze(result, 'v2')
    assert not result['scientific_core_complete']
    assert result['common_calibration_parent_ids'] == ['cal-a', 'cal-b']
    assert not (tmp_path / 'test').exists()
    assert merge_and_freeze_calibration(tmp_path, spec_sha='spec', pool_sha='v2', slots=2,
        candidates=candidates, thresholds=thresholds, old_orders=orders, solver_seconds=0) == result


def test_calibration_missing_shard_is_not_complete(tmp_path):
    from src.ablations.gnn.reach_v2_adapter import collect_calibration_chunks
    candidates = calibration_fixture(tmp_path)
    # An existing complete shard cannot stand in for an absent array member.
    with pytest.raises(FileNotFoundError):
        collect_calibration_chunks(tmp_path, spec_sha='spec', pool_sha='v2', slots=3, candidates=candidates)


def test_calibration_parent_checkpoint_content_conflict_rejected(tmp_path):
    from src.eval.bace_frozen_gnn_contracts import read_json, atomic_json
    from src.ablations.gnn.reach_v2_adapter import collect_calibration_chunks
    candidates = calibration_fixture(tmp_path)
    path = tmp_path / 'gin/calibration/0000/parents/cal-a.json'
    row = read_json(path)
    row['science']['pair_rows'][0]['wnode_distance'] = .2
    atomic_json(path, row)
    with pytest.raises(ValueError, match='CONTENT_CONFLICT'):
        collect_calibration_chunks(tmp_path, spec_sha='spec', pool_sha='v2', slots=2, candidates=candidates)
