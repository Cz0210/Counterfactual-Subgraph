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
