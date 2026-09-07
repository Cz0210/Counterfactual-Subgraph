import hashlib
from pathlib import Path

import pytest

from src.ablations.gnn import reach_raw_distance_reuse as raw
from src.eval.bace_frozen_gnn_contracts import atomic_json, read_json, stable_sha256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, 'kernel_identity_proof', lambda *a: {'fixture': 'proof'})
    correction = tmp_path / 'corrected'
    evaluation = tmp_path / 'original'
    row = dict(parent_id='cal-1', candidate_id='old', match_index=0, match_atom_indices=[2],
        oracle_checkpoint_hash='old-gine', action_semantics_version='hard-delete',
        parent_smiles='CCO', residual_smiles='CC', distance_ok=True,
        delete_valid=True, sanitize_ok=True, residual_connected=True, wnode_distance=.2)
    science = dict(match_rows=[row], pair_rows=[])
    rel = 'gine/calibration/parents/' + 'a' * 64 + '.json'
    original = evaluation / rel
    atomic_json(original, dict(science=science, science_sha256=stable_sha256(science)))
    docs = dict(
        acceptance=dict(state='GNN_CORE_SEED7_CORRECTED_PASS', cache_provenance_gaps=[],
            raw_ot_recomputed_count=0, main_matrix_write=False, original_package_sha256='old-package'),
        original_inventory=dict(state='PASS', publication_driver_commit='a' * 40,
            files={'evaluation/' + rel: dict(sha256=sha(original))}),
        repair_contract=dict(original_package_sha256='old-package', original_exact_driver_commit='a' * 40,
            source_spec={'evaluation_root': str(evaluation)}, required_counts={'calibration_parent_units': 1}),
        raw_distance_contract=dict(wnode=dict(solver='exact_emd2', feature_cost='cosine',
            node_mass='uniform', size_penalty_beta=0., distance_namespace='same'), feature_schema={'sha256': 'schema'},
                                   molclr_checkpoint={'sha256': 'molclr'}))
    spec = {'correction_root': str(correction)}
    for name, payload in docs.items():
        path = correction / (name + '.json')
        atomic_json(path, payload)
        spec[name] = dict(relative_path=path.name, sha256=sha(path))
    return spec, original, docs['raw_distance_contract']


def test_one_time_split_migration_keeps_action_provenance(tmp_path, monkeypatch):
    spec, original, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / 'index.json'
    index = raw.build_index(spec, split='calibration', output=output, repo=tmp_path)
    assert index['source_parent_units'] == index['raw_cost_count'] == 1
    assert not index['source_flip_masks_reused'] and index['ot_recomputed'] == 0
    record = next(iter(index['graph_costs'].values()))
    assert record['source_records'][0]['original_action_context']['oracle_checkpoint_hash'] == 'old-gine'
    # Adopt the sealed compact index rather than re-read/re-hash old parents.
    original.unlink()
    assert raw.build_index(spec, split='calibration', output=output, repo=tmp_path) == index


def test_test_index_rejects_boolean_before_reading_source(tmp_path):
    with pytest.raises(ValueError, match='BEFORE_NEW_GLOBAL_FREEZE'):
        raw.build_index({}, split='test', output=tmp_path/'index', repo=tmp_path,
                        validate_test_freeze=True)


def test_incomplete_source_partition_rejected(tmp_path, monkeypatch):
    spec, _, _ = fixture(tmp_path, monkeypatch)
    path = Path(spec['correction_root'])/'repair_contract.json'
    data = read_json(path)
    data['required_counts']['calibration_parent_units'] = 2
    atomic_json(path, data)
    spec['repair_contract']['sha256'] = sha(path)
    with pytest.raises(ValueError, match='PARTITION_INCOMPLETE'):
        raw.build_index(spec, split='calibration', output=tmp_path/'index', repo=tmp_path)


def test_finite_distance_with_invalid_graph_not_adopted(tmp_path, monkeypatch):
    spec, original, _ = fixture(tmp_path, monkeypatch)
    row = read_json(original)
    row['science']['match_rows'][0]['residual_connected'] = False
    row['science_sha256'] = stable_sha256(row['science'])
    atomic_json(original, row)
    path = Path(spec['correction_root'])/'original_inventory.json'
    inventory = read_json(path)
    next(iter(inventory['files'].values()))['sha256'] = sha(original)
    atomic_json(path, inventory)
    spec['original_inventory']['sha256'] = sha(path)
    with pytest.raises(ValueError, match='INVALID_FINITE'):
        raw.build_index(spec, split='calibration', output=tmp_path/'index', repo=tmp_path)


def wrapper_fixture(tmp_path, monkeypatch):
    spec, _, contract = fixture(tmp_path, monkeypatch)
    index = raw.build_index(spec, split='calibration', output=tmp_path/'index', repo=tmp_path)
    for relative in raw.KERNELS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen')
    index['kernel_identity'] = {r: sha(tmp_path/r) for r in raw.KERNELS}
    index['self_sha256'] = stable_sha256({k:v for k,v in index.items() if k != 'self_sha256'})
    class Delegate:
        def __init__(self):
            from types import SimpleNamespace
            self.calls = 0
            self.config = SimpleNamespace(**contract['wnode'])
        def distance(self, p, r):
            self.calls += 1
            return dict(ok=True, distance=.3, cache_hit=False)
        def stats_dict(self): return {}
        def close(self): pass
    delegate = Delegate()
    return raw.VerifiedRawGraphDistance(delegate, index=index, current_raw_contract=contract,
                                       repo=tmp_path), delegate, index, contract


def context(teacher='new-gin'):
    return dict(candidate_id='new-attributed-id', match_atom_indices=[2], teacher_sha256=teacher,
        action_semantics_version='hard-delete', match_selection_policy='own-flip-min',
        distance_implementation_version='same-exact')


def test_cross_backbone_raw_content_reuse_not_mask_reuse(tmp_path, monkeypatch):
    provider, delegate, _, _ = wrapper_fixture(tmp_path, monkeypatch)
    assert provider.distance_for_action('OCC', 'CC', action_context=context())['distance'] == .2
    assert provider.distance_for_action('CCO', 'CC', action_context=context('new-gatv2'))['distance'] == .2
    assert delegate.calls == 0
    assert {r['current_action_context']['teacher_sha256'] for r in provider.used} == {'new-gin','new-gatv2'}


def test_new_residual_uses_raw_pair_api_once(tmp_path, monkeypatch):
    provider, delegate, _, _ = wrapper_fixture(tmp_path, monkeypatch)
    for teacher in ('gin', 'gcn'):
        assert provider.distance_for_action('CCO', 'CO', action_context=context(teacher))['distance'] == .3
    assert delegate.calls == 1


def test_changed_encoder_or_kernel_rejected(tmp_path, monkeypatch):
    _, delegate, index, contract = wrapper_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='CONTRACT_OR_INDEX'):
        raw.VerifiedRawGraphDistance(delegate, index=index, current_raw_contract={**contract, 'changed': True}, repo=tmp_path)
    (tmp_path / raw.KERNELS[0]).write_text('changed')
    with pytest.raises(ValueError, match='IMPLEMENTATION_DRIFT'):
        raw.VerifiedRawGraphDistance(delegate, index=index, current_raw_contract=contract, repo=tmp_path)


def test_delegate_runtime_cost_change_is_not_hidden_by_manifest(tmp_path, monkeypatch):
    _, delegate, index, contract = wrapper_fixture(tmp_path, monkeypatch)
    delegate.config.size_penalty_beta = .25
    with pytest.raises(ValueError, match='DELEGATE_NUMERICAL'):
        raw.VerifiedRawGraphDistance(delegate, index=index, current_raw_contract=contract, repo=tmp_path)


def ours_freeze_fixture(tmp_path):
    from src.eval.bace_reach_v2 import seal
    search = seal(tmp_path/'search_contract.json', {'science': 'unchanged'})
    pool = seal(tmp_path/'candidate_freeze.json', {'search_contract_sha256': search['self_sha256']})
    ids = [str(i) for i in range(20)]
    controls = dict(old_pool_old_selector=ids, new_pool_old_selector=ids)
    selector = seal(tmp_path/'selector_freeze.json', dict(test_opened=False,
        candidate_freeze_sha256=pool['self_sha256'], controls=controls,
        reach_first={'ordered_rule_ids': list(reversed(ids))}))
    gate = seal(tmp_path/'train_reach_gate.json', dict(
        state='NO_ADDITIONAL_PPO_REQUIRED_BY_TRAIN_GATE', candidate_freeze_sha256=pool['self_sha256']))
    value = dict(state='REACH_V2_FINAL_CONFIGURATION_FROZEN',
        selected_control='new_pool_reach_first', selected_using_test=False, test_opened=False,
        test_campaigns_max=1, main_matrix_write=False, claim_new_untouched_test=False,
        search_contract_sha256=search['self_sha256'], candidate_freeze_sha256=pool['self_sha256'],
        selector_freeze_sha256=selector['self_sha256'], train_gate_sha256=gate['self_sha256'],
        controls={**controls, 'new_pool_reach_first': list(reversed(ids))})
    return seal(tmp_path/'final_test_binding.json', value)


def test_ours_actual_three_control_freeze_permits_only_raw_adoption(tmp_path):
    freeze = ours_freeze_fixture(tmp_path)
    raw.validate_ours_final_freeze(freeze, tmp_path)


def test_ours_test_adoption_rejects_config_only_or_changed_selector(tmp_path):
    freeze = ours_freeze_fixture(tmp_path)
    with pytest.raises(ValueError, match='ACTUAL_FINAL_FREEZE'):
        raw.validate_ours_final_freeze({'state': 'PASS'}, tmp_path)
    # A fully re-sealed but different calibration choice is still not the bound one.
    from src.eval.bace_reach_v2 import seal
    path = tmp_path/'selector_freeze.json'
    prior = read_json(path)
    replacement = {k: v for k,v in prior.items() if k not in ('self_sha256', 'created_at')}
    replacement['reach_first']['ordered_rule_ids'] = list(reversed(replacement['reach_first']['ordered_rule_ids']))
    fresh = tmp_path/'changed-selector.json'
    seal(fresh, replacement)
    path.write_bytes(fresh.read_bytes())
    with pytest.raises(ValueError, match='DEPENDENCY_CHANGED'):
        raw.validate_ours_final_freeze(freeze, tmp_path)
