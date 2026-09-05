"""Portable correction acceptance: tiny sealed-row fixtures, no model or OT."""
from __future__ import annotations

import copy

import pytest

from src.ablations.gnn import temperature_repair as repair


@pytest.fixture
def portable(monkeypatch):
    names = repair.core.BACKBONES
    files, documents, csvs = {}, {}, {}
    def item(path, value):
        files[path] = {'sha256': 'sha:' + path, 'size': 1}
        documents[path] = value
        return files[path]['sha256']
    run = dict(execution_commit='science', bundle_sha256='sha:inputs/bundle_manifest.json',
        binding_sha256='binding', checkpoints={n: 'sha:classifiers/' + n + '/model.pt' for n in names},
        temperatures={n: 2.0 for n in names})
    item('evaluation/run_manifest.json', run)
    item('inputs/bundle_manifest.json', {'split_row_counts': {'calibration': 2, 'test': 2}})
    freeze = dict(binding_sha256='binding', all_five_calibration_orders_frozen=True,
        test_loaded=False, source_files={}, selections={})
    item('evaluation/CALIBRATION_FREEZE.json', freeze)
    evidence = dict(state='PASS', source_final_audit_sha256='sha:evaluation/gnn_seed7_final_audit.json',
        scientific_engine_commit='science', cohort_contract=dict(native_eligibility=repair.audit.ELIGIBILITY,
        bundle_manifest_sha256=run['bundle_sha256'], scientific_definition_changed=False), models={},
        calibration_prediction_sha256s={}, parent_scientific_checkpoint_sha256s={}, seed=7)
    evidence.update({k: True for k in ('global_calibration_selector_replayed',
        'all_five_validation_temperatures_fitted_and_input_bound', 'classifier_metrics_replayed',
        'native_common_metrics_replayed', 'per_match_flip_and_best_match_replayed', 'proposal_fixed')})
    evidence.update({k: False for k in ('test_used_for_selection', 'ot_recomputed',
        'classifier_inference_rerun', 'main_matrix_write', 'cross_seed_standard_deviation_claimed')})
    for n in names:
        evidence['models'][n] = {}
        for leaf, key in (('model.pt', 'model_sha256'), ('temperature_scaling.json', 'temperature_sha256'),
                          ('model_card.json', 'model_card_sha256')):
            evidence['models'][n][key] = item(f'classifiers/{n}/{leaf}', {'temperature': 2.0})
        freeze['selections'][n] = {}
        for mode in ('native', 'common'):
            freeze['source_files'][f'{n}/{mode}'] = item(f'evaluation/{n}/{mode}/selected_rules.json',
                dict(binding_sha256='binding', split='calibration', backbone=n, cohort=mode, candidate_ids=['r']))
            freeze['selections'][n][mode] = ['r']
        for split in ('calibration', 'test'):
            pred = f'evaluation/{n}/{split}_classifier_predictions.csv'
            records = [dict(parent_id=split + '-p' + str(i), label=i, predicted_label=i,
                logits=[2.0, 0.0] if i == 0 else [0.0, 2.0], checkpoint_id=run['checkpoints'][n],
                temperature=2.0, backbone=n, num_classes=2, source_label=1) for i in (0, 1)]
            for row in records:
                row['probabilities'] = repair.scaled_probabilities([row['logits']], 2.0)[0].tolist()
            csvs[pred] = records
            digest = item(pred, None)
            if split == 'calibration':
                evidence['calibration_prediction_sha256s'][n] = digest
            rel = f'{n}/{split}/parents/key.json'
            evidence['parent_scientific_checkpoint_sha256s'][rel] = item('evaluation/' + rel, {})
    for split in ('calibration', 'test'):
        item(f'evaluation/{split}_cohorts.json', dict(native={n: [split + '-p1'] for n in names},
            common=[split + '-p1'], source_label=1, definition='true_source_and_correctly_predicted_source', backbones=list(names)))
    required = {'run_manifest.json', 'CALIBRATION_FREEZE.json', 'calibration_cohorts.json', 'test_cohorts.json'}
    monkeypatch.setattr(repair.audit, 'required_files', lambda: required)
    producer = {'files': {r.removeprefix('evaluation/'): f['sha256'] for r, f in files.items() if r.startswith('evaluation/')}}
    item('evaluation/gnn_seed7_final_audit.json', producer)
    proof = {'independent_science_replay_sha256': item('publication/independent_science_replay.json', evidence)}
    manifest = {'files': files, 'scientific_engine_commit': 'science'}
    return manifest, documents, csvs, proof


def verify(fixture):
    manifest, documents, csvs, proof = fixture
    return repair._verify_corrective_science_bindings(manifest, documents.__getitem__, csvs.__getitem__, proof)


def test_portable_replays_five_model_prediction_cohorts_and_complete_parent_set(portable):
    units, paths = verify(portable)
    assert len(units) == len(paths) == 10
    assert ('gin', 'calibration', 'calibration-p1') in units


def _add_real_layout_progress_logs(portable):
    manifest, documents, _, _ = portable
    producer = documents['evaluation/gnn_seed7_final_audit.json']['files']
    for name in repair.core.BACKBONES:
        for split in ('calibration', 'test'):
            rel = f'{name}/{split}/parents/progress.json'
            manifest['files']['evaluation/' + rel] = {'sha256': 'progress:' + rel, 'size': 1}
            producer[rel] = 'progress:' + rel
            documents['evaluation/' + rel] = {'completed': 1}


def test_portable_real_five_backbone_two_split_layout_excludes_only_ten_progress_logs(portable):
    _add_real_layout_progress_logs(portable)
    units, paths = verify(portable)
    assert len(units) == len(paths) == 10
    assert not any(path.endswith('/progress.json') for path in paths)
    assert sum(path.endswith('/parents/progress.json') for path in portable[0]['files']) == 10


def test_portable_progress_logs_remain_integrity_bound(portable):
    _add_real_layout_progress_logs(portable)
    portable[0]['files']['evaluation/gin/test/parents/progress.json']['sha256'] = 'changed'
    with pytest.raises(ValueError, match='PORTABLE_PRODUCER_FILE_DRIFT'):
        verify(portable)


@pytest.mark.parametrize('leaf', ['checkpoint-progress.json', 'nested/progress.json'])
def test_portable_unknown_parent_file_is_not_silently_excluded(portable, leaf):
    _add_real_layout_progress_logs(portable)
    portable[0]['files']['evaluation/gin/test/parents/' + leaf] = {'sha256': 'unexpected', 'size': 1}
    with pytest.raises(ValueError, match='PORTABLE_PARENT_AUDIT_INVENTORY_DRIFT'):
        verify(portable)


@pytest.mark.parametrize('flag', ['state', 'all_five_validation_temperatures_fitted_and_input_bound',
    'global_calibration_selector_replayed', 'native_common_metrics_replayed', 'ot_recomputed'])
def test_portable_rejects_stale_or_non_scientific_replay_receipt(portable, flag):
    receipt = portable[1]['publication/independent_science_replay.json']
    receipt[flag] = 'FAILED' if flag == 'state' else not receipt[flag]
    with pytest.raises(ValueError, match='PORTABLE_INDEPENDENT_SCIENTIFIC_PASS_REQUIRED'):
        verify(portable)


def test_portable_rejects_replay_model_temperature_hash_drift(portable):
    portable[1]['publication/independent_science_replay.json']['models']['gin']['temperature_sha256'] = 'stale'
    with pytest.raises(ValueError, match='PORTABLE_REPLAY_MODEL_DRIFT'):
        verify(portable)


def test_portable_rejects_calibration_prediction_hash_drift(portable):
    portable[1]['publication/independent_science_replay.json']['calibration_prediction_sha256s']['gin'] = 'stale'
    with pytest.raises(ValueError, match='PORTABLE_CALIBRATION_PREDICTION_DRIFT'):
        verify(portable)


def test_portable_rejects_omitted_parent_despite_other_counts(portable):
    del portable[1]['publication/independent_science_replay.json']['parent_scientific_checkpoint_sha256s']['gin/test/parents/key.json']
    with pytest.raises(ValueError, match='PORTABLE_PARENT_AUDIT_INVENTORY_DRIFT'):
        verify(portable)


def test_portable_rejects_raw_parent_probability_double_scaling(portable):
    row = portable[2]['evaluation/gin/calibration_classifier_predictions.csv'][1]
    row['probabilities'] = repair.scaled_probabilities([row['logits']], 4.0)[0].tolist()
    with pytest.raises(ValueError, match='PORTABLE_PARENT_RAW_LOGIT_BINDING'):
        verify(portable)


def test_portable_rejects_different_native_or_common_cohort(portable):
    portable[1]['evaluation/test_cohorts.json']['common'] = []
    with pytest.raises(ValueError, match='PORTABLE_NATIVE_COMMON_COHORT_DRIFT'):
        verify(portable)


def test_portable_rejects_missing_frozen_selection_binding(portable):
    portable[1]['evaluation/gin/native/selected_rules.json']['candidate_ids'] = ['other']
    with pytest.raises(ValueError, match='PORTABLE_FROZEN_SELECTION_DRIFT'):
        verify(portable)


def test_portable_reuse_receipt_cannot_duplicate_parent_to_reach_total():
    receipt = {'backbone': 'gin', 'split': 'test', 'parent_id': 'p'}
    expected = {('gin', 'test', 'p')}
    paths = {'gin/test/parents/a.json', 'gin/test/parents/b.json'}
    seen, seen_paths = set(), set()
    repair._claim_portable_reuse_parent(receipt, 'a.json', expected, paths, seen, seen_paths)
    with pytest.raises(ValueError, match='PORTABLE_REUSE_PARENT_DUPLICATE_OR_OUTSIDE_COHORT'):
        repair._claim_portable_reuse_parent(receipt, 'b.json', expected, paths, seen, seen_paths)


def test_portable_reuse_receipt_cannot_reference_parent_outside_native_cohort():
    receipt = {'backbone': 'gin', 'split': 'test', 'parent_id': 'not-source'}
    with pytest.raises(ValueError, match='PORTABLE_REUSE_PARENT_DUPLICATE_OR_OUTSIDE_COHORT'):
        repair._claim_portable_reuse_parent(receipt, 'a.json', {('gin', 'test', 'p')},
            {'gin/test/parents/a.json'}, set(), set())
