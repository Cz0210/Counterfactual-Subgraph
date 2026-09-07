"""Narrow new-pool adapter for the existing BACE parent-partition evaluator.

No training, temperature fit, selection in shards, or matrix writer. The old
66-rule sensitivity remains an independent scientific result, not a v2 audit.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.bace_frozen_gnn_contracts import atomic_json, read_json, stable_sha256, sha256_file

SCOPE_NAME = 'GINE_GUIDED_PROPOSAL_FIXED_BACKBONE_SENSITIVITY_REACH_V2'
BACKBONES = ('gine', 'gin', 'gcn', 'gatv2', 'gatedgcn_plus')


def collect_calibration_chunks(output: Path, *, spec_sha, pool_sha, slots, candidates,
                               model_files=None, chunk_size=None):
    """Read only completed calibration chunks, never heldout records.

    Each shard declares the same complete native cohort. The union of its
    committed parent units must equal that cohort exactly before selection.
    """
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
    if not isinstance(slots, int) or slots <= 0:
        raise ValueError('INVALID_CALIBRATION_PARTITION_SIZE')
    merged = {}
    for backbone in BACKBONES:
        native, seen, rows, source_files = None, set(), [], {}
        for index in range(slots):
            directory = output / backbone / 'calibration' / f'{index:04d}'
            terminal = read_json(directory / 'terminal.json')
            expected = dict(scope=SCOPE_NAME, spec_sha256=spec_sha,
                pool_sha256=pool_sha, backbone=backbone, split='calibration', index=index,
                state='PARENT_CHUNK_COMPLETE_NOT_CORE_PASS', global_selector_called=False,
                main_matrix_write=False)
            if any(terminal.get(key) != value for key, value in expected.items()):
                raise ValueError('CALIBRATION_SHARD_TERMINAL_CONFLICT')
            if type(terminal.get('index')) is not int:
                raise ValueError('CALIBRATION_CHUNK_INDEX_MUST_BE_INTEGER')
            if model_files is not None and terminal.get('model_files') != model_files[backbone]:
                raise ValueError('CALIBRATION_CHUNK_MODEL_BINDING_CONFLICT')
            declared = terminal['native_cohort_ids']
            if declared != sorted(set(declared)) or native is not None and native != declared:
                raise ValueError('CALIBRATION_NATIVE_COHORT_CONFLICT')
            native = declared
            ids = terminal['parent_ids']
            if len(ids) != len(set(ids)) or seen.intersection(ids) or not set(ids) <= set(native):
                raise ValueError('CALIBRATION_PARTITION_DUPLICATE_OR_OUTSIDE_COHORT')
            if chunk_size is not None and ids != native[index*chunk_size:(index+1)*chunk_size]:
                raise ValueError('CALIBRATION_STABLE_PARTITION_CONFLICT')
            source_files[str((directory/'terminal.json').relative_to(output))] = sha256_file(directory/'terminal.json')
            chunk_rows, checkpoint_parents = [], set()
            for checkpoint in sorted((directory / 'parents').glob('*.json')):
                saved = read_json(checkpoint)
                scientific = saved['science']
                if (saved.get('scope') != SCOPE_NAME or saved.get('backbone') != backbone
                        or saved.get('pool_sha256') != pool_sha
                        or saved.get('spec_sha256') != spec_sha
                        or saved.get('science_sha256') != stable_sha256(scientific)):
                    raise ValueError('CALIBRATION_PARENT_CONTENT_CONFLICT')
                parent_rows = scientific['pair_rows']
                if model_files is not None:
                    from src.ablations.gnn.reach_v2_closeout import verify_own_match_minima
                    verify_own_match_minima(scientific,
                        candidate_ids=[c['candidate_id'] for c in candidates],
                        model_sha=model_files[backbone]['model.pt'])
                actual_ids = {row['parent_id'] for row in parent_rows}
                if len(actual_ids) != 1 or checkpoint_parents.intersection(actual_ids):
                    raise ValueError('CALIBRATION_PARENT_CHECKPOINT_DUPLICATE_OR_EMPTY')
                checkpoint_parents.update(actual_ids)
                chunk_rows.extend(parent_rows)
                source_files[str(checkpoint.relative_to(output))] = sha256_file(checkpoint)
            if checkpoint_parents != set(ids) or len(chunk_rows) != terminal['pair_count']:
                raise ValueError('CALIBRATION_SHARD_MISSING_OR_EXTRA_PARENT_CHECKPOINT')
            matrix_from_pairs(ids, candidates, chunk_rows, root=directory, split='calibration')
            seen.update(ids)
            rows.extend(chunk_rows)
        if seen != set(native):
            raise ValueError('CALIBRATION_PARTITION_OMITS_PARENTS')
        merged[backbone] = {'parent_ids': native, 'pairs': rows, 'source_files':source_files}
    return merged


def merge_and_freeze_calibration(output: Path, *, spec_sha, pool_sha, slots,
                                 candidates, thresholds, old_orders, solver_seconds=120,
                                 model_files=None, chunk_size=None):
    """A single complete merge then ten global selectors, not shard top-K."""
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
    target = output / 'CALIBRATION_FREEZE.json'
    if target.exists():
        frozen = read_json(target)
        require_global_freeze(frozen, pool_sha)
        if frozen.get('spec_sha256') != spec_sha:
            raise ValueError('FROZEN_GNN_SPEC_CONFLICT')
        return frozen
    complete = collect_calibration_chunks(output, spec_sha=spec_sha, pool_sha=pool_sha,
                                         slots=slots, candidates=candidates,
                                         model_files=model_files, chunk_size=chunk_size)
    common = sorted(set.intersection(*(set(complete[name]['parent_ids']) for name in BACKBONES)))
    if not common or any(not complete[name]['parent_ids'] for name in BACKBONES):
        raise ValueError('BLOCKED_EMPTY_CALIBRATION_COHORT')
    expected_roles = {f'{name}/{mode}' for name in BACKBONES for mode in ('native', 'common')}
    if set(old_orders) != expected_roles:
        raise ValueError('ALL_TEN_ORIGINAL_CALIBRATION_ORDERS_REQUIRED')
    selectors = {}
    for backbone in BACKBONES:
        for mode in ('native', 'common'):
            ids = complete[backbone]['parent_ids'] if mode == 'native' else common
            included = set(ids)
            rows = [r for r in complete[backbone]['pairs'] if r['parent_id'] in included]
            matrix = matrix_from_pairs(ids, candidates, rows, root=output, split='calibration')
            role = f'{backbone}/{mode}'
            selectors[role] = freeze_backbone_orders(matrix, old_order=old_orders[role],
                thresholds=thresholds, backbone=backbone, cohort_mode=mode, pool_sha=pool_sha,
                expected_parent_ids=ids, output=output / backbone / mode / 'reach_selector.json',
                solver_seconds=solver_seconds)
    result = dict(scope=SCOPE_NAME, spec_sha256=spec_sha, pool_sha256=pool_sha,
        test_loaded=False, selectors=selectors, common_calibration_parent_ids=common,
        own_match_minimum_replayed=model_files is not None,
        calibration_files={k:v for n in BACKBONES for k,v in complete[n]['source_files'].items()},
        scientific_core_complete=False, main_matrix_write=False)
    require_global_freeze(result, pool_sha)
    atomic_json(target, result)
    return result


def validate_pool(pool: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any],
                  *, current_pool_sha: str, old_ids: Sequence[str]):
    ids = [str(row.get('candidate_id', '')) for row in pool]
    if not 66 <= len(ids) <= 4096 or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError('V2_POOL_EMPTY_DUPLICATE_OR_OUTSIDE_BUDGET')
    if freeze.get('candidate_universe_sha256') != current_pool_sha:
        raise ValueError('V2_FROZEN_POOL_BINDING_CONFLICT')
    if freeze.get('test_opened') is not False or freeze.get('calibration_opened_during_search') is not False:
        raise ValueError('V2_POOL_MUST_BE_TRAIN_ONLY')
    if len(old_ids) != 66 or len(set(old_ids)) != 66 or not set(old_ids) <= set(ids):
        raise ValueError('V2_MAIN_POOL_MUST_RETAIN_OWN_ORIGINAL_66')
    return {'scope': SCOPE_NAME, 'candidate_count': len(ids), 'candidate_universe_sha256': current_pool_sha,
            'original_66_retained': True, 'old_gnn_result_adopted_as_v2': False,
            'backbone_flip_and_match_min_recomputed': True, 'test_opened': False}


def parent_binding(parent, candidates, *, backbone, checkpoint_id, temperature_sha,
                   pool_sha, split):
    if backbone not in {'gine', 'gin', 'gcn', 'gatv2', 'gatedgcn_plus'} or split not in {'train', 'calibration', 'test'}:
        raise ValueError('V2_BACKBONE_OR_SPLIT_INVALID')
    return stable_sha256({'scope': SCOPE_NAME, 'parent': asdict(parent),
        'candidate_ids': [row['candidate_id'] for row in candidates],
        'pool_sha256': pool_sha, 'backbone': backbone, 'checkpoint_id': checkpoint_id,
        'temperature_sha256': temperature_sha, 'split': split,
        'enumeration': 'attributed_graph_all_matches_v1',
        'match_selection': 'min_wnode_among_own_strict_flip_matches'})


def evaluate_parent_chunk(parents, candidates, *, oracle, featurizer, distance_provider,
                          output: Path, split, pool_sha, temperature_sha, batch_size=64,
                          predictions=None, test_freeze=None, execution_spec_sha,
                          boundary_check=lambda: None):
    """Reuse one raw-distance provider, but never another backbone's flip/min.

    Called inside the existing CPU owner/Slurm partition. Each full parent is
    committed separately; a rerun reads that exact binding, not GINE's cache.
    """
    from src.eval.bace_reach_v2 import evaluate_pairs
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
    if not execution_spec_sha:
        raise ValueError('FROZEN_EXECUTION_SPEC_BINDING_REQUIRED')
    if split == 'test':
        require_global_freeze(test_freeze, pool_sha)
    elif test_freeze is not None:
        raise ValueError('Unexpected test dependency in non-test stage')
    output.mkdir(parents=True, exist_ok=True)
    combined = []
    for parent in parents:
        boundary_check()
        key = parent_binding(parent, candidates, backbone=oracle.backbone,
            checkpoint_id=oracle.checkpoint_id, temperature_sha=temperature_sha,
            pool_sha=pool_sha, split=split)
        path = output / f'{key}.json'
        if path.exists():
            saved = read_json(path)
            if (saved.get('binding') != key or saved.get('spec_sha256') != execution_spec_sha
                    or saved.get('science_sha256') != stable_sha256(saved['science'])):
                raise ValueError('V2_PARENT_CHECKPOINT_CONFLICT')
            pairs = saved['science']['pair_rows']
        else:
            cache = None
            if predictions is not None:
                before = predictions[parent.parent_id]
                cache = {parent.parent_id: {'parent_smiles': parent.smiles,
                    'pred_before': before['predicted_label'], 'p_before': before['probabilities']}}
            pairs, matches = evaluate_pairs([parent], candidates, oracle=oracle, featurizer=featurizer,
                distance_provider=distance_provider, split=split, oracle_checkpoint_id=oracle.checkpoint_id,
                oracle_batch_size=batch_size, parent_prediction_cache=cache)
            scientific = {'pair_rows': pairs, 'match_rows': matches}
            atomic_json(path, {'binding': key, 'science': scientific,
                'science_sha256': stable_sha256(scientific), 'scope': SCOPE_NAME,
                'backbone': oracle.backbone, 'pool_sha256': pool_sha,
                'spec_sha256': execution_spec_sha})
        # Reject missing/duplicate Cartesian rows even for existing checkpoints.
        matrix_from_pairs([parent.parent_id], candidates, pairs, root=output, split=split)
        combined.extend(pairs)
    return combined


def freeze_backbone_orders(matrix, *, old_order, thresholds, backbone, cohort_mode,
                           pool_sha, expected_parent_ids, output: Path, solver_seconds=120):
    """One global native/common selection, only after all calibration shards."""
    from src.eval.bace_reach_selector import ReachMasks, select_nested
    if matrix.manifest.get('split') != 'calibration' or matrix.manifest.get('test_loaded') is not False:
        raise ValueError('V2_SELECTOR_REQUIRES_CALIBRATION_ONLY')
    if cohort_mode not in {'native', 'common'} or not matrix.parent_ids:
        raise ValueError('V2_CALIBRATION_COHORT_INVALID_OR_EMPTY')
    if (tuple(matrix.parent_ids) != tuple(expected_parent_ids)
            or len(set(expected_parent_ids)) != len(expected_parent_ids)):
        raise ValueError('V2_SELECTOR_REJECTS_PARTIAL_OR_REORDERED_COHORT')
    masks = ReachMasks.from_distances(matrix.candidate_ids, matrix.distances, thresholds)
    selection = select_nested(masks, old_order, solver_seconds=solver_seconds)
    result = dict(selection, scope=SCOPE_NAME, backbone=backbone, cohort_mode=cohort_mode,
        pool_sha256=pool_sha, cohort_sha256=stable_sha256(list(matrix.parent_ids)),
        calibration_parent_ids=list(matrix.parent_ids),
        old_calibration_S10=list(old_order[:10]), test_loaded=False,
        global_selector_after_complete_merge=True)
    result['self_sha256'] = stable_sha256(result)
    if output.exists():
        if read_json(output) != result:
            raise ValueError('FROZEN_V2_SELECTOR_CANNOT_BE_REPLACED')
        return result
    atomic_json(output, result)
    return result


def require_global_freeze(freeze, pool_sha):
    expected = {f'{name}/{mode}' for name in ('gine', 'gin', 'gcn', 'gatv2', 'gatedgcn_plus')
                for mode in ('native', 'common')}
    if not isinstance(freeze, Mapping) or set(freeze.get('selectors', {})) != expected:
        raise ValueError('ALL_TEN_V2_SELECTORS_REQUIRED_BEFORE_TEST')
    if freeze.get('scope') != SCOPE_NAME or freeze.get('pool_sha256') != pool_sha or freeze.get('test_loaded') is not False:
        raise ValueError('V2_GLOBAL_FREEZE_SCOPE_CONFLICT')
    for role, selector in freeze['selectors'].items():
        if (selector.get('self_sha256') != stable_sha256({k: v for k, v in selector.items() if k != 'self_sha256'})
                or selector.get('pool_sha256') != pool_sha or selector.get('test_loaded') is not False
                or f"{selector['backbone']}/{selector['cohort_mode']}" != role
                or selector.get('global_selector_after_complete_merge') is not True):
            raise ValueError('V2_PER_SELECTOR_FREEZE_CONFLICT')
        order = selector.get('ordered_rule_ids', [])
        prefixes = selector.get('prefixes', {})
        if (len(order) != 20 or len(set(order)) != 20
                or any(prefixes.get(str(k)) != order[:k] for k in range(1, 21))):
            raise ValueError('V2_SELECTOR_MISSING_FIXED_NESTED_20_PREFIXES')
    return True
