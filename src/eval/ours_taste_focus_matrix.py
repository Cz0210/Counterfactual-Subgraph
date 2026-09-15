"""Read-only, bounded-memory adoption of sealed Taste Ours pair records.

This adapter never runs an oracle and never changes a published result. NaN is
uncomputed, +inf is a completed no-finite-recourse outcome in the old contract.
"""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_records(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact_pairs(path, parents, candidates, split, identity):
    pi = {p: i for i, p in enumerate(parents)}
    ci = {c: i for i, c in enumerate(candidates)}
    assert len(pi) == len(parents) and len(ci) == len(candidates)
    d = np.full((len(pi), len(ci)), np.nan, dtype=np.float64)
    predictions = np.full(len(pi), -1, dtype=np.int8)
    funnel = Counter()
    first = {}
    before = Path(path).stat()
    for row in load_records(path):
        assert row['split'] == split and row['source_label'] == 1
        assert row['cf_mode'] == 'strict_flip' and row['classifier_family'] == 'gine'
        assert row['rf_oracle_used'] is False
        for key in ('oracle_checkpoint_hash', 'temperature_calibration_hash',
                    'feature_schema_hash', 'molclr_checkpoint_hash', 'distance_namespace',
                    'action_semantics_version', 'match_selection_policy'):
            assert row[key] == identity[key], (key, row['parent_id'], row['candidate_id'])
        i, j = pi[row['parent_id']], ci[row['candidate_id']]
        assert np.isnan(d[i, j]), 'DUPLICATE_PAIR'
        pred = int(row['pred_before'])
        assert pred in (0, 1, 2)
        assert predictions[i] in (-1, pred), 'SOURCE_PREDICTION_CONFLICT'
        predictions[i] = pred
        funnel['computed_pairs'] += 1
        funnel['applicable_pairs'] += bool(row['applicable'])
        funnel['valid_residual_pairs'] += int(row['num_valid_residuals']) > 0
        funnel['match_count_at_1000'] += int(row['num_matches']) == 1000
        if row['pair_strict_flip'] is True:
            assert pred == 1 and row['pred_after'] in (0, 2) and row['destination_label'] in (0, 2)
            v = float(row['wnode_distance'])
            assert np.isfinite(v) and v >= 0 and row['residual_smiles']
            d[i, j] = v
            funnel['finite_strict_flip_pairs'] += 1
        else:
            # Accepted old evaluator explicitly recorded a complete negative.
            # Preserve its finite-WNode meaning, not a proof over all deletions.
            assert row['wnode_distance'] is None
            reason = row['failure_reason']
            assert reason in ('no_substructure_match', 'no_valid_strict_flip_with_finite_wnode'), reason
            d[i, j] = np.inf
            funnel[reason] += 1
            first.setdefault(reason, {'parent_id': row['parent_id'], 'candidate_id': row['candidate_id']})
    after = Path(path).stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), 'SOURCE_CHANGED'
    return d, predictions, {'counts': dict(funnel), 'first_failure': first,
                            'unknown_pairs': int(np.isnan(d).sum()),
                            'source_stat': {'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'inode': after.st_ino}}


def extract(source, output):
    source, output = Path(source), Path(output)
    assert not output.exists(), 'FRESH_OUTPUT_REQUIRED'
    run = read_json(source / 'run_manifest.json')
    freeze = read_json(source / 'freeze_manifest.json')
    original = read_json(source / 'input_identity.json')
    assert run['status'] == 'PASS' and run['independent_terminal_verification_passed'] is True
    assert run['selection_frozen_before_test'] is True
    pools = list(load_records(source / 'raw/candidate_universe.jsonl'))
    ids = [r['candidate_id'] for r in pools]
    selection = read_json(source / 'raw/selection_manifest.json')
    assert selection['selection_frozen'] and not selection['test_used_for_selection']
    output.mkdir(parents=True)
    for split in ('calibration', 'test'):
        manifest = read_json(source / f'raw/{split}_pair_manifest.json')
        with Path(original[f'{split}_path']).open() as f:
            parents = sorted(r['molecule_id'] for r in csv.DictReader(f)
                             if r['label'] == '1' and r['split'] == split and not r['exclusion_reason'])
        chosen = ids if split == 'calibration' else selection['ordered_rule_ids']
        path = source / f'raw/{split}_pair_details.jsonl'
        assert path.stat().st_size == freeze['files'][str(path.relative_to(source))]['bytes']
        assert len(parents) == manifest['parent_count'] and len(chosen) == manifest['candidate_count']
        d, pred, evidence = compact_pairs(path, parents, chosen, split, manifest['evaluation_identity'])
        assert evidence['counts']['computed_pairs'] == manifest['pair_count']
        assert not np.isnan(d).any(), 'SEALED_MATRIX_INCOMPLETE'
        np.savez_compressed(output / f'{split}.npz', distances=d, parents=np.asarray(parents),
                            candidates=np.asarray(chosen), predictions=pred)
        evidence.update({'source_root': str(source), 'source_pair_receipt_sha': manifest['pair_details_sha256'],
                         'input_identity': manifest['evaluation_identity'], 'full_pool': split == 'calibration',
                         'scope': 'ACCEPTED_SAVED_FINITE_WNODE_MATRIX_NOT_ALL_DELETION_SPACE'})
        dump_json(output / f'{split}_adoption.json', evidence)
    # Raw generation is train-only, not a complete train parent x rule matrix.
    train = {}
    generation_funnel = Counter()
    for mode in ('base', 'high_temp'):
        for row in load_records(source / f'raw/generation/{mode}/candidate_pool.jsonl'):
            assert row['split'] == 'train' and not row['test_loaded']
            pid = row['parent_id']
            train.setdefault(pid, {'parent_id': pid, 'smiles': row['parent_smiles'],
                                  'pred_before': row['pred_before'], 'p_before': row['p_before']})
            generation_funnel['attempt_records'] += 1
            for key in ('parse_ok', 'connected', 'direct_substructure', 'deletion_valid', 'cf_flip'):
                generation_funnel[key] += bool(row.get(key))
    dump_json(output / 'train_saved_predictions.json', list(train.values()))
    dump_json(output / 'candidate_pool.json', pools)
    dump_json(output / 'original_selection.json', selection)
    dump_json(output / 'contract.json', {**original, 'source_root': str(source), 'p0_count': len(ids),
        'train_matrix_state': 'NOT_SAVED_FULL_CARTESIAN', 'train_generation_funnel': dict(generation_funnel),
        'full_delete_space_ceiling': 'UNKNOWN', 'old_source_receipt_reused': True,
        'theta_grid': [float(r['threshold']) for r in csv.DictReader((source / 'figure4_coverage_vs_threshold.csv').open())],
        'new_oracle_queries': 0, 'new_ot_queries': 0, 'main_matrix_write': False})
    print(json.dumps({'state': 'COMPACT_SAVED_MATRIX_ADOPTED', 'p0': len(ids), 'train_source_records': len(train),
                      'output': str(output)}, ensure_ascii=False))
