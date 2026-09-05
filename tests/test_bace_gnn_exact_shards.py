from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from src.ablations.gnn import cpu_evaluation as core
from src.ablations.gnn import sharded_evaluation as shards
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file, stable_sha256, BACEParent


def test_partition_disjoint_complete_stable_and_empty_slots():
    ids = ['p5', 'p1', 'p2', 'p4', 'p3']
    result = shards.stable_partition(ids, chunk_size=2, slots=4)
    assert result == [['p1', 'p2'], ['p3', 'p4'], ['p5'], []]
    assert [p for part in result for p in part] == sorted(ids)
    with pytest.raises(ValueError):
        shards.stable_partition(['same', 'same'], chunk_size=2, slots=1)
    with pytest.raises(ValueError):
        shards.stable_partition(ids, chunk_size=2, slots=2)


def test_test_plan_cannot_parse_inputs_before_global_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, 'scoped', lambda p: Path(p))
    monkeypatch.setattr(shards, '_oracle_context', lambda *a: pytest.fail('Test input parsed before freeze'))
    with pytest.raises(FileNotFoundError):
        shards.prepare({'evaluation_root': str(tmp_path / 'evaluation')}, split='test')


def test_cached_closeout_never_silently_recomputes_a_missing_parent(tmp_path, monkeypatch):
    import src.eval.bace_frozen_gnn_verification as implementation
    monkeypatch.setattr(implementation, '_evaluate_rows', lambda *a, **kw: pytest.fail('Science rerun'))
    with pytest.raises(ValueError, match='MISSING_SEALED_PARENT_UNIT'):
        core._pairs([BACEParent('p', 'CCC', 1, 0)], [{'candidate_id': 'c', 'canonical_fragment': 'C'}],
            oracle=SimpleNamespace(checkpoint_id='model'), featurizer=None, distance=None,
            split='calibration', output=tmp_path, binding='binding', batch_size=256,
            predictions={}, require_cached=True)


def test_cached_parent_cannot_reuse_another_backbone_flip_mask(tmp_path, monkeypatch):
    import src.eval.bace_frozen_gnn_verification as implementation
    def evaluate(parents, candidates, **kwargs):
        return ([{'parent_id': 'p', 'candidate_id': 'c', 'canonical_fragment': 'C',
                  'pair_strict_flip': False, 'applicable': True, 'wnode_distance': None, 'cf_drop': None}], [])
    monkeypatch.setattr(implementation, '_evaluate_rows', evaluate)
    args = dict(featurizer=None, distance=None, split='calibration', output=tmp_path, binding='binding',
        batch_size=256, predictions={'p': {'predicted_label': 1, 'probabilities': [0.1, 0.9]}})
    p = [BACEParent('p', 'CCC', 1, 0)]
    c = [{'candidate_id': 'c', 'canonical_fragment': 'C'}]
    core._pairs(p, c, oracle=SimpleNamespace(checkpoint_id='gine'), **args)
    with pytest.raises(ValueError, match='MISSING_SEALED_PARENT_UNIT'):
        core._pairs(p, c, oracle=SimpleNamespace(checkpoint_id='gcn'), require_cached=True, **args)


def test_shard_merge_checks_scientific_identity_and_preserves_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, 'scoped', lambda p: Path(p))
    cohorts = {n: [] for n in core.BACKBONES}; cohorts['gin'] = ['p']
    task = {'index': 0, 'backbone': 'gin', 'split': 'calibration', 'parent_ids': ['p'], 'candidate_ids': ['c']}
    plan = {'shards': [task], 'cohorts': {'native': cohorts}, 'total_parent_units': 1, 'self_sha256': 'plan'}
    monkeypatch.setattr(shards, 'read_plan', lambda *a: (tmp_path, plan))
    root = tmp_path / 'shards/calibration/0000'
    value = {'binding': 'key', 'science': {'pair_rows': [{'parent_id': 'p'}], 'match_rows': []}}
    value['science_sha256'] = stable_sha256(value['science'])
    file = root / 'parents/key.json'; atomic_json(file, value)
    atomic_json(root / 'terminal.json', {'state': 'PASS', 'task': task, 'partition_sha256': 'plan', 'parent_files': {'key.json': sha256_file(file)}})
    assert shards.merge_parent_units({}, split='calibration')['completed_parent_units'] == 1
    assert (tmp_path / 'gin/calibration/parents/key.json').read_bytes() == file.read_bytes()
    assert shards.merge_parent_units({}, split='calibration')['completed_parent_units'] == 1


def test_two_stage_closeout_selects_once_and_parses_test_only_after_freeze(tmp_path, monkeypatch):
    from tests.test_bace_gnn_cpu_evaluation import candidates, selector, pairs
    bundle, output = tmp_path / 'bundle', tmp_path / 'result'
    bundle.mkdir(); output.mkdir()
    for s in ('calibration', 'test'):
        (bundle / (s + '.csv')).write_text('fixture\n')
    manifest = {'splits': {s: s + '.csv' for s in ('calibration', 'test')},
        'files': {p.name: {'size': p.stat().st_size, 'sha256': sha256_file(p)} for p in bundle.iterdir()}}
    loads, calls = [], []
    def parents(path):
        loads.append(path.stem)
        if path.stem == 'test':
            assert (output / 'CALIBRATION_FREEZE.json').is_file()
        return [BACEParent(path.stem + '_p', 'CCC', 1, 0)]
    monkeypatch.setattr(core, '_all_parents', parents)
    monkeypatch.setattr(core, '_predict', lambda *a: [{'predicted_label': 1, 'probabilities': [0.1, 0.9]}])
    monkeypatch.setattr(core, '_pairs', lambda p, c, **kw: pairs([x.parent_id for x in p], c))
    real_select = core.select_calibration
    monkeypatch.setattr(core, 'select_calibration', lambda *a: (calls.append(1) or real_select(*a)))
    models = {n: bundle / n for n in core.BACKBONES}
    oracles = {n: SimpleNamespace(model=SimpleNamespace(parameters=lambda: [])) for n in core.BACKBONES}
    args = (bundle, manifest, output, candidates(), selector(), oracles, models, None, None, 'binding', 256,
            lambda *a, **kw: {'roc_auc': None})
    result = core._run_phases(*args, phase='calibration', require_cached_pairs=True)
    assert result['core_pass'] is False and loads == ['calibration'] and len(calls) == 10
    frozen = (output / 'CALIBRATION_FREEZE.json').read_bytes()
    core._run_phases(*args, phase='finish', require_cached_pairs=True)
    assert len(calls) == 10
    assert loads == ['calibration', 'calibration', 'test']
    assert (output / 'CALIBRATION_FREEZE.json').read_bytes() == frozen
