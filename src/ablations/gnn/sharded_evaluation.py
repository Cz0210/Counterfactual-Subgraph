"""BACE-only exact parent partitions around the existing frozen evaluator.

No training, temperature fitting, candidate generation or per-shard selection.
Calibration merges and freezes globally before any test task is admissible.
"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from src.ablations.gnn import cpu_evaluation as science
from src.ablations.gnn.cpu_training import bundle_file, load_bundle
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_csv, read_json, sha256_file, stable_sha256

SCOPE = Path('/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn')


def scoped(path: str | Path) -> Path:
    result = Path(path).absolute()
    if result == SCOPE or not result.is_relative_to(SCOPE):
        raise ValueError('GNN persistent outputs must be within the dedicated /share scope')
    if result.resolve() != result:
        raise ValueError('No symlink or parent traversal in GNN output')
    return result


def stable_partition(ids: list[str], *, chunk_size: int, slots: int) -> list[list[str]]:
    if len(ids) != len(set(ids)) or chunk_size < 1 or slots < math.ceil(len(ids) / chunk_size):
        raise ValueError('Invalid unique parent universe or insufficient partition slots')
    ordered = sorted(ids)
    parts = [ordered[i * chunk_size:(i + 1) * chunk_size] for i in range(slots)]
    flattened = [x for part in parts for x in part]
    if flattened != ordered or len(flattened) != len(set(flattened)):
        raise ValueError('Parent partition is not disjoint and complete')
    return parts


def _inputs(spec: dict[str, Any]):
    root, manifest = load_bundle(spec['bundle_root'])
    if sha256_file(root / 'bundle_manifest.json') != spec['bundle_file_sha256']:
        raise ValueError('Original bundle file binding changed')
    for name, path in spec['models'].items():
        for filename, digest in spec['model_files'][name].items():
            if sha256_file(Path(path) / filename) != digest:
                raise ValueError(f'Sealed classifier/temperature drift:{name}:{filename}')
    return root, manifest


def _oracle_context(spec):
    import torch
    from src.oracles.gnn_oracle import GNNOracle
    torch.set_num_threads(spec['cpu_threads'])
    root, manifest = _inputs(spec)
    models = {n: Path(p) for n, p in spec['models'].items()}
    oracles = {n: GNNOracle.from_checkpoint(p, device='cpu', batch_size=spec['batch_size']) for n, p in models.items()}
    schema = read_json(bundle_file(root, manifest, manifest['feature_schema_path']))['schema_sha256']
    for name, oracle in oracles.items():
        card = read_json(models[name] / 'model_card.json')
        if (oracle.backbone != name or oracle.num_classes != 2 or oracle.source_label != 1
                or card['dataset'] != 'bace'
                or read_json(models[name] / 'feature_schema.json')['schema_sha256'] != schema):
            raise ValueError('Frozen model/schema/source contract differs')
    candidates = science._candidates(root, manifest)
    selector = science.frozen_selector(root, manifest)
    payload = {'schema': science.SCHEMA, 'execution_commit': manifest['execution_commit'],
        'bundle_sha256': sha256_file(root / 'bundle_manifest.json'),
        'oracle_batch_size': spec['batch_size'], 'cpu_threads': spec['cpu_threads'],
        'checkpoints': {n: o.checkpoint_id for n, o in oracles.items()},
        'temperatures': {n: o.temperature for n, o in oracles.items()},
        'selector_sha256': selector['input_sha256'],
        'candidate_universe_sha256': stable_sha256(candidates)}
    return root, manifest, oracles, candidates, payload


def prepare(spec: dict[str, Any], *, split: str) -> dict[str, Any]:
    output = scoped(spec['evaluation_root'])
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / f'{split}_partition.json'
    if plan_path.exists():
        return read_json(plan_path)
    if split == 'test':
        freeze = read_json(output / 'CALIBRATION_FREEZE.json')
        if freeze.get('all_five_calibration_orders_frozen') is not True or freeze.get('test_loaded') is not False:
            raise ValueError('Test plan requires all ten pre-test frozen selectors')
        for key, digest in freeze['source_files'].items():
            n, mode = key.split('/')
            if sha256_file(output / n / mode / 'selected_rules.json') != digest:
                raise ValueError('Frozen selection file drift before test planning')
    root, manifest, oracles, candidates, payload = _oracle_context(spec)
    binding = stable_sha256(payload)
    run_path = output / 'run_manifest.json'
    if run_path.exists() and read_json(run_path)['binding_sha256'] != binding:
        raise ValueError('Sharded science binding drift')
    atomic_json(run_path, {**payload, 'binding_sha256': binding, 'model_roots': spec['models'],
        'main_matrix_write': False, 'ChemLLM_loaded': False, 'PPO_rerun': False,
        'proposal_fixed': True, 'seed': 7, 'cpu_only': True,
        'evaluation_driver_commit': spec['driver_commit'],
        'original_scientific_commit': manifest['execution_commit'],
        'source_cohort_definition': 'true_source_and_correctly_predicted_source'})
    parents = science._all_parents(bundle_file(root, manifest, manifest['splits'][split]))
    if len(parents) != manifest['split_row_counts'][split]:
        raise ValueError('Split row count differs from frozen metadata')
    featurizer = science._featurizer(root, manifest)
    predictions = {n: science._predict(parents, o, featurizer, split, spec['batch_size']) for n, o in oracles.items()}
    cohorts = science.cohort_ids(parents, predictions)
    atomic_json(output / f'{split}_cohorts.json', cohorts)
    shards = []
    for name in science.BACKBONES:
        atomic_csv(output / name / f'{split}_classifier_predictions.csv', [
            {'parent_id': p.parent_id, 'label': p.label, **r} for p, r in zip(parents, predictions[name], strict=True)])
        chosen = candidates
        if split == 'test':
            ids = set(freeze['selections'][name]['native']) | set(freeze['selections'][name]['common'])
            chosen = [row for row in candidates if row['candidate_id'] in ids]
        for part in stable_partition(cohorts['native'][name], chunk_size=spec['chunk_size'], slots=spec['slots'][split]):
            shards.append({'index': len(shards), 'backbone': name, 'split': split,
                'parent_ids': part, 'candidate_ids': [row['candidate_id'] for row in chosen]})
    plan = {'state': 'SEALED', 'split': split, 'binding_sha256': binding,
        'spec_sha256': stable_sha256(spec), 'source_cohort_definition': cohorts['definition'],
        'parents': [asdict(p) for p in parents], 'predictions': predictions,
        'cohorts': cohorts, 'shards': shards, 'total_parent_units': sum(len(s['parent_ids']) for s in shards),
        'partition_disjoint': True, 'partition_complete': True,
        'test_loaded': split == 'test', 'global_selection_in_shard': False,
        'calibration_freeze_sha256': sha256_file(output / 'CALIBRATION_FREEZE.json') if split == 'test' else None}
    plan['self_sha256'] = stable_sha256(plan)
    atomic_json(plan_path, plan)
    return plan


def read_plan(spec, split):
    output = scoped(spec['evaluation_root'])
    plan = read_json(output / f'{split}_partition.json')
    if (plan['self_sha256'] != stable_sha256({k: v for k, v in plan.items() if k != 'self_sha256'})
            or plan['spec_sha256'] != stable_sha256(spec)):
        raise ValueError('Partition input binding changed')
    if split == 'test' and plan['calibration_freeze_sha256'] != sha256_file(output / 'CALIBRATION_FREEZE.json'):
        raise ValueError('Test plan freeze binding changed')
    for name in science.BACKBONES:
        actual = [s['parent_ids'] for s in plan['shards'] if s['backbone'] == name]
        expected = stable_partition(plan['cohorts']['native'][name], chunk_size=spec['chunk_size'], slots=spec['slots'][split])
        if actual != expected:
            raise ValueError('Partition does not cover the exact native cohort')
    return output, plan


def shard(spec, *, split: str, index: int):
    import torch
    from src.oracles.gnn_oracle import GNNOracle
    from src.eval.bace_frozen_gnn_contracts import BACEParent
    torch.set_num_threads(spec['cpu_threads'])
    root, manifest = _inputs(spec)
    output, plan = read_plan(spec, split)
    task = plan['shards'][index]
    if task['index'] != index:
        raise ValueError('Partition index drift')
    directory = output / 'shards' / split / f'{index:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        terminal = directory / 'terminal.json'
        if terminal.exists():
            receipt = read_json(terminal)
            if receipt['partition_sha256'] != plan['self_sha256'] or receipt['task'] != task:
                raise ValueError('Completed shard binding changed')
            for rel, digest in receipt['parent_files'].items():
                if sha256_file(directory / 'parents' / rel) != digest:
                    raise ValueError('Completed parent checkpoint changed')
            return receipt
        name = task['backbone']
        all_parents = [BACEParent(**p) for p in plan['parents']]
        by_id = {p.parent_id: p for p in all_parents}
        parents = [by_id[p] for p in task['parent_ids']]
        all_candidates = science._candidates(root, manifest)
        candidates = [r for r in all_candidates if r['candidate_id'] in set(task['candidate_ids'])]
        if [r['candidate_id'] for r in candidates] != task['candidate_ids']:
            raise ValueError('Candidate ordering differs from fixed universe')
        oracle = GNNOracle.from_checkpoint(spec['models'][name], device='cpu', batch_size=spec['batch_size'])
        before = {p.parent_id: r for p, r in zip(all_parents, plan['predictions'][name], strict=True)}
        featurizer = science._featurizer(root, manifest)
        start = time.monotonic()
        scratch_parent = os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR') or '/tmp'
        with tempfile.TemporaryDirectory(prefix='bace-gnn-exact-', dir=scratch_parent) as local:
            distance = science._distance(root, manifest, Path(local)) if parents else None
            try:
                rows = science._pairs(parents, candidates, oracle=oracle, featurizer=featurizer,
                    distance=distance, split=split, output=directory / 'parents',
                    binding=plan['binding_sha256'], batch_size=spec['batch_size'], predictions=before)
                statistics = distance.stats_dict() if distance else {}
            finally:
                if distance:
                    distance.close()
        files = {p.name: sha256_file(p) for p in (directory / 'parents').glob('*.json') if p.name != 'progress.json'}
        if len(files) != len(parents) or len(rows) != len(parents) * len(candidates):
            raise ValueError('Shard parent/rule Cartesian coverage is incomplete')
        receipt = {'state': 'PASS', 'task': task, 'partition_sha256': plan['self_sha256'],
            'parent_files': files, 'completed_parent_units': len(parents), 'pair_count': len(rows),
            'elapsed_seconds': time.monotonic() - start, 'distance_statistics': statistics,
            'node_local_scratch': True, 'global_selector_called': False,
            'model_sha256': sha256_file(Path(spec['models'][name]) / 'model.pt'),
            'temperature_sha256': sha256_file(Path(spec['models'][name]) / 'temperature_scaling.json'),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'main_matrix_write': False}
        atomic_json(terminal, receipt)
        return receipt


def merge_parent_units(spec, *, split):
    output, plan = read_plan(spec, split)
    completed, claimed = 0, set()
    for task in plan['shards']:
        directory = output / 'shards' / split / f'{task["index"]:04d}'
        receipt = read_json(directory / 'terminal.json')
        if receipt['state'] != 'PASS' or receipt['task'] != task or receipt['partition_sha256'] != plan['self_sha256']:
            raise ValueError('Missing or mismatched shard terminal')
        target = output / task['backbone'] / split / 'parents'
        target.mkdir(parents=True, exist_ok=True)
        for parent, (filename, digest) in zip(task['parent_ids'], sorted(receipt['parent_files'].items()), strict=True):
            # File names are content hashes, not ordered parent IDs; uniqueness is
            # checked by scientific row identity below rather than this zip order.
            source = directory / 'parents' / filename
            if Path(filename).name != filename or sha256_file(source) != digest:
                raise ValueError('Sealed parent file hash changed')
            value = read_json(source)
            if value['science_sha256'] != stable_sha256(value['science']):
                raise ValueError('Sealed parent scientific payload changed')
            ids = {r['parent_id'] for r in value['science']['pair_rows']}
            if len(ids) != 1 or not ids.issubset(set(task['parent_ids'])):
                raise ValueError('Parent checkpoint belongs to another partition')
            identity = (task['backbone'], next(iter(ids)))
            if identity in claimed:
                raise ValueError('Duplicate scientific parent unit across partitions')
            claimed.add(identity)
            destination = target / filename
            if destination.exists():
                if sha256_file(destination) != digest:
                    raise ValueError('Conflicting already-imported parent unit')
            else:
                atomic_json(destination, value)
                if sha256_file(destination) != digest:
                    raise ValueError('Parent checkpoint transport was not byte-exact')
            completed += 1
    expected = {(n, p) for n, ids in plan['cohorts']['native'].items() for p in ids}
    if claimed != expected or completed != plan['total_parent_units']:
        raise ValueError('Merged parent universe differs from the complete plan')
    result = {'state': 'PASS', 'split': split, 'completed_parent_units': completed,
        'partition_sha256': plan['self_sha256'], 'no_duplicate_or_missing_parent_units': True}
    atomic_json(output / f'{split}_merge.json', result)
    return result


def advance(spec, stage, index=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') not in {'', '-1'}:
        raise ValueError('HPC CPU evaluation must not see a GPU')
    if stage == 'regression':
        return train_regression(spec)
    if stage == 'prepare-calibration':
        return prepare(spec, split='calibration')
    if stage in {'calibration-shard', 'test-shard'}:
        return shard(spec, split=stage.split('-')[0], index=index)
    split = 'calibration' if stage == 'freeze-calibration' else 'test'
    if stage not in {'freeze-calibration', 'finish'}:
        raise ValueError('Unknown exact evaluation stage')
    merge_parent_units(spec, split=split)
    result = science.run_evaluation(bundle_root=spec['bundle_root'], model_roots=spec['models'],
        output_root=spec['evaluation_root'], resume=True, batch_size=spec['batch_size'],
        cpu_threads=spec['cpu_threads'], phase='calibration' if split == 'calibration' else 'finish',
        require_cached_pairs=True)
    if split == 'calibration':
        prepare(spec, split='test')
    return result


def train_regression(spec):
    """Compare independent parent units with sealed, real train-only probes."""
    from src.eval.bace_frozen_gnn_contracts import load_bace_parents
    root, manifest, oracles, candidates, _ = _oracle_context(spec)
    output = scoped(spec['evaluation_root']).parent / 'exact-regression'
    output.mkdir(parents=True, exist_ok=True)
    parents = load_bace_parents(bundle_file(root, manifest, manifest['splits']['train']))[:2]
    featurizer = science._featurizer(root, manifest)
    results = {}
    def scientific(value):
        if isinstance(value, dict):
            return {k: scientific(v) for k, v in value.items() if k != 'distance_cache_hit'}
        if isinstance(value, list):
            return [scientific(v) for v in value]
        return value
    for name, oracle in oracles.items():
        prior = Path(spec['source_admission']).parent / 'cpu_admission' / name / 'parents'
        predictions = science._predict(parents, oracle, featurizer, 'train', spec['batch_size'])
        before = {p.parent_id: r for p, r in zip(parents, predictions, strict=True)}
        checked = []
        for index, parent in enumerate(parents):
            target = output / name / str(index)
            scratch_parent = os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR') or '/tmp'
            with tempfile.TemporaryDirectory(prefix='bace-gnn-regression-', dir=scratch_parent) as local:
                distance = science._distance(root, manifest, Path(local))
                try:
                    science._pairs([parent], candidates, oracle=oracle, featurizer=featurizer,
                        distance=distance, split='train', output=target,
                        binding=sha256_file(root / 'bundle_manifest.json'), batch_size=spec['batch_size'], predictions=before)
                finally:
                    distance.close()
            files = [p for p in target.glob('*.json') if p.name != 'progress.json']
            if len(files) != 1:
                raise ValueError('Regression must produce one complete parent checkpoint')
            golden = prior / files[0].name
            expected = spec['reference_probe_parent_files'][name][golden.name]
            if sha256_file(golden) != expected:
                raise ValueError('Sealed train regression reference changed')
            actual, reference = read_json(files[0]), read_json(golden)
            if (actual['binding'] != reference['binding']
                    or scientific(actual['science']) != scientific(reference['science'])):
                raise ValueError(f'EXACT_PARENT_PARTITION_REGRESSION_FAILED:{name}:{parent.parent_id}')
            checked.append({'parent_id': parent.parent_id, 'original_sha256': expected,
                'actual_sha256': sha256_file(files[0]), 'scientific_sha256': stable_sha256(scientific(actual['science']))})
        results[name] = checked
    receipt = {'state': 'PASS', 'split': 'train', 'test_loaded': False, 'calibration_loaded': False,
        'all_five_backbones': True, 'parent_partition_scientific_equal': True,
        'only_ignored_observational_field': 'distance_cache_hit', 'results': results,
        'no_approximate_distance': True, 'no_model_or_temperature_change': True}
    atomic_json(output / 'regression.json', receipt)
    return receipt
