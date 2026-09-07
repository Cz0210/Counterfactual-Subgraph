"""Narrow new-pool adapter for the existing BACE parent-partition evaluator.

No training, temperature fit, selection in shards, or matrix writer. The old
66-rule sensitivity remains an independent scientific result, not a v2 audit.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.bace_frozen_gnn_contracts import atomic_json, read_json, stable_sha256

SCOPE_NAME = 'GINE_GUIDED_PROPOSAL_FIXED_BACKBONE_SENSITIVITY_REACH_V2'


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
                          predictions=None, test_freeze=None, boundary_check=lambda: None):
    """Reuse one raw-distance provider, but never another backbone's flip/min.

    Called inside the existing CPU owner/Slurm partition. Each full parent is
    committed separately; a rerun reads that exact binding, not GINE's cache.
    """
    from src.eval.bace_reach_v2 import evaluate_pairs
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
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
            if saved.get('binding') != key or saved.get('science_sha256') != stable_sha256(saved['science']):
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
                'backbone': oracle.backbone, 'pool_sha256': pool_sha})
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
