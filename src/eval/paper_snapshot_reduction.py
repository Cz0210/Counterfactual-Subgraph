"""Offline numeric reduction of an already collected, partial paper snapshot.

This is not a source collector, scientific evaluator, or matrix verifier. Remote
source paths are descriptive strings and are never opened. The three local JSON
inputs are transport-bound in the output audit; historical identity waivers stay
open. Only existing strict-flip prefix distances are aggregated.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

DATASETS = ('AIDS', 'Mutagenicity', 'BACE', 'TasteMolNet')
METHODS = ('Ours', 'GlobalGCE', 'GCFExplainer', 'ComRecGC')
K_MAX = 20
TABLE_K = 10
COMPARISON_ATOL = 1e-12  # Compare CSV serialization, never modify distances.
INPUT_FILES = ('canonical/small_results_snapshot.json',
               'lineage/raw_reduction.json', 'lineage/v2_combined_manifest.json')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, *, nullable: bool = False) -> float | None:
    if nullable and (value is None or str(value).strip() in ('', 'N/A')):
        return None
    _require(not isinstance(value, bool), 'boolean is not a numeric metric')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid numeric metric: {value!r}') from exc
    _require(math.isfinite(result), 'nonfinite metric is not an N/A sentinel')
    return result


def _integer(value: Any) -> int:
    number = _number(value)
    _require(number == int(number), 'nonintegral count or K')
    return int(number)


def _truth(value: Any) -> bool:
    token = str(value).lower()
    _require(token in ('true', 'false', '1', '0', '1.0', '0.0'),
             'invalid strict_recourse_available boolean')
    return token in ('true', '1', '1.0')


def _rows(text: str) -> list[dict[str, str]]:
    result = list(csv.DictReader(io.StringIO(text)))
    _require(bool(result), 'empty numeric CSV')
    return result


def _metric(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row:
            return _number(row[name], nullable=True)
    return None


def _different(left: float | None, right: float | None) -> bool:
    return ((left is None) != (right is None) or
            (left is not None and right is not None and
             abs(left - right) > COMPARISON_ATOL))


def _coverage(best: list[float | None], theta: float) -> float:
    return sum(value is not None and value <= theta for value in best) / len(best)


def _median(best: list[float | None]) -> float | None:
    finite = [value for value in best if value is not None]
    return statistics.median(finite) if finite else None


def _validate_prefix(parent_ids: list[Any], prefix: list[list[Any]]) -> list[list[float | None]]:
    _require(bool(parent_ids) and len(set(parent_ids)) == len(parent_ids),
             'empty or duplicate parent cohort')
    _require(len(prefix) == K_MAX and all(len(row) == len(parent_ids) for row in prefix),
             'prefix must contain the complete K=1..20 parent rectangle')
    parsed = [[_number(value, nullable=True) for value in row] for row in prefix]
    for index, row in enumerate(parsed):
        _require(all(value is None or value >= 0 for value in row), 'negative distance')
        if index:
            _require(all(old is None or (new is not None and new <= old)
                         for old, new in zip(parsed[index - 1], row)),
                     'prefix best distance lost a recourse or increased')
    return parsed


def _legacy_binding(raw: dict[str, Any], manifest: dict[str, Any],
                    dataset: str, method: str, theta: float) -> None:
    _require(dataset in DATASETS[:2] and method != 'ComRecGC',
             'legacy reduction cannot replace a non-legacy cell or relabel CLEAR')
    sources = manifest['sources']
    if dataset == 'AIDS':
        record = sources['aids_table2'][method]
        pair_path, pair_sha = record['pair_details_path'], record['pair_details_sha256']
        expected_theta = manifest['aids']['figure3_theta']
        _require(raw['order_binding'].get('semantics') == 'stable_first_occurrence'
                 and raw['order_binding'].get('bound_by_pair_file') is True,
                 'AIDS candidate order is not bound to the original pair file')
    else:
        record = sources['mutagenicity'][method]
        pair_path, pair_sha = record['pair_path'], record['pair_sha256']
        expected_theta = manifest['mutagenicity']['theta']
        _require(raw['order_binding'].get('path') == record['path'] and
                 raw['order_binding'].get('sha256') == record['sha256'],
                 'Mutagenicity selected-order binding mismatch')
    _require(theta == _number(expected_theta), 'legacy frozen threshold mismatch')
    _require(raw['pair_binding'].get('path') == pair_path and
             raw['pair_binding'].get('sha256') == pair_sha and
             re.fullmatch(r'[0-9a-f]{64}', pair_sha) is not None,
             'legacy pair source path/SHA declaration differs from V2 manifest')


def _existing_parent_hash(files: dict[str, Any], summary: dict[str, Any]) -> tuple[str | None, str]:
    """Carry existing receipt identities, not hashes of normalized CSV text.

    Snapshot collectors may normalize CRLF to LF. The embedded text hash is a
    separate transport identity; it cannot replace the original byte SHA.
    """
    found = []
    documents = [('summary.json', summary)] + [
        (name, json.loads(files[name]['text']))
        for name in ('final_artifact_audit.json', 'run_manifest.json') if name in files]
    for name, document in documents:
        for section in ('file_sha256', 'files', 'audited_files', 'scientific_files'):
            values = document.get(section, {})
            if not isinstance(values, dict):
                continue
            value = values.get('parent_best_distances.csv')
            digest = value.get('sha256') if isinstance(value, dict) else value
            if digest:
                _require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest) is not None,
                         'invalid inherited parent CSV SHA declaration')
                found.append((digest, name + '/' + section))
    _require(len({digest for digest, _ in found}) <= 1, 'conflicting parent CSV SHA receipts')
    return found[0] if found else (None, 'PER_FILE_HASH_UNAVAILABLE_IN_COLLECTED_RECEIPTS')


def reduce_snapshot(snapshot: dict[str, Any], legacy: dict[str, Any],
                    manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate shape/semantics and return tables; perform no filesystem I/O."""
    raw_cells = legacy['cells']
    legacy_map = {(row['dataset'], row['method']): row for row in raw_cells}
    _require(len(legacy_map) == len(raw_cells), 'duplicate legacy reduction cell')
    cells = snapshot['cells']
    identities = [(item['cell']['dataset'], item['cell']['method']) for item in cells]
    _require(bool(cells) and len(set(identities)) == len(cells), 'empty/duplicate snapshot cell')
    _require(set(legacy_map) <= set(identities), 'orphan legacy reduction cell')
    _require(_integer(snapshot['pointer']['latest_count']) == len(cells),
             'pointer count differs from included cells')
    _require(set(snapshot['pointer']['applied_cells']) == {f'{d}/{m}' for d, m in identities}
             and len(snapshot['pointer']['applied_cells']) == len(cells),
             'pointer applied cell identities differ from snapshot')
    result: dict[str, Any] = {name: [] for name in
        ('figure3', 'figure4', 'table2', 'registered', 'differences', 'cross_checks',
         'lineage', 'keypoints', 'comparisons', 'comparison_scope')}
    cohorts = {}
    for item in cells:
        cell, files = item['cell'], item['files']
        dataset, method = cell['dataset'], cell['method']
        key = (dataset, method)
        _require(dataset in DATASETS and method in METHODS, 'unknown dataset/method (CLEAR forbidden)')
        _require(cell['status'] in ('FROZEN_PASS', 'ADOPTABLE_PASS'), 'nonaccepted cell in numeric snapshot')
        _require(_integer(cell['k_max']) == K_MAX and _integer(cell['table2_k']) == TABLE_K,
                 'snapshot does not match frozen K20/Table2-K10 protocol')
        summary = json.loads(files['summary.json']['text'])
        f3 = _rows(files['figure3_coverage_vs_k.csv']['text'])
        f4 = _rows(files['figure4_coverage_vs_threshold.csv']['text'])
        table_names = [name for name in files if name.startswith('table2_') and name.endswith('_k10.csv')]
        _require(len(table_names) == 1, 'ambiguous Table2 source')
        table_rows = _rows(files[table_names[0]]['text'])
        _require(len(table_rows) == 1, 'Table2 must have one cell row')
        table = table_rows[0]
        _require([_integer(row['k']) for row in f3] == list(range(1, K_MAX + 1)),
                 'Figure3 K rows missing, duplicated or reordered')
        theta = _number(summary.get('theta_star', f3[0].get('theta', f3[0].get('threshold'))))
        _require(theta >= 0, 'negative threshold')
        for row in f3 + [table]:
            for name in ('theta', 'threshold', 'theta_star'):
                if name in row:
                    _require(_number(row[name]) == theta, 'conflicting frozen threshold in numeric CSV')
        for name in ('k', 'requested_k', 'table2_k'):
            if name in table:
                _require(_integer(table[name]) == TABLE_K, 'Table2 K metadata mismatch')
        grid = [_number(row['threshold']) for row in f4]
        _require(grid == sorted(set(grid)) and grid[0] >= 0,
                 'Figure4 grid duplicated, unordered or negative')
        is_legacy = key in legacy_map
        _require(is_legacy == (cell.get('registry_exception') == 'USER_APPROVED_FROZEN_V4'),
                 'legacy waiver/reduction membership mismatch')
        if is_legacy:
            raw = legacy_map[key]
            _legacy_binding(raw, manifest, dataset, method, theta)
            parents, prefix = raw['parent_ids'], raw['prefix_best_distances']
            source, source_sha = raw['pair_binding']['path'], raw['pair_binding']['sha256']
            hash_receipt, embedded_sha = 'V2_DECLARATION_MATCH_ONLY', None
            scope = 'LEGACY_SOURCE_RECONCILED_EXPORT_ONLY'
        else:
            parent_file = files['parent_best_distances.csv']
            by_k: dict[int, list[dict[str, str]]] = {}
            for row in _rows(parent_file['text']):
                by_k.setdefault(_integer(row['k']), []).append(row)
            _require(set(by_k) == set(range(1, K_MAX + 1)), 'parent prefix K gap')
            parents = [row['parent_id'] for row in by_k[1]]
            prefix = []
            for k in range(1, K_MAX + 1):
                _require([row['parent_id'] for row in by_k[k]] == parents,
                         'parent membership/order differs across K')
                values = []
                for row in by_k[k]:
                    value = _number(row['best_distance'], nullable=True)
                    strict = _truth(row['strict_recourse_available'])
                    _require((strict and value is not None) or (not strict and value is None),
                             'strict recourse and best distance availability disagree')
                    values.append(value)
                prefix.append(values)
            source = parent_file['path']
            embedded_sha = hashlib.sha256(parent_file['text'].encode('utf-8')).hexdigest()
            source_sha, hash_receipt = _existing_parent_hash(files, summary)
            scope = ('DECLARED_HASH_BOUND_PARENT_SNAPSHOT_REDUCTION' if source_sha else
                     'ACCEPTED_CELL_PARENT_REDUCTION_PER_FILE_HASH_UNAVAILABLE')
        prefix = _validate_prefix(parents, prefix)
        cost_kind = ('conditional_median_cost' if dataset in DATASETS[:2]
                     else 'fixed_capped_mean_cost')
        cap = _number(summary.get('cost_cap'), nullable=True)
        _require(cap is None or cap >= 0, 'negative cost cap')
        _require(cost_kind != 'fixed_capped_mean_cost' or cap is not None, 'fixed capped mean requires frozen cap')
        old_cov = _metric(table, 'coverage', 'ccrcov')
        old_cost = _metric(table, 'cost', cost_kind)
        _require(old_cov is not None, 'registered Table2 coverage unavailable')
        cohorts[f'{dataset}/{method}'] = {'parent_count': len(parents), 'parent_ids': parents,
            'source': source, 'theta': theta, 'cost_semantics': cost_kind, 'cost_cap': cap}
        result['registered'].append({'dataset': dataset, 'method': method, 'matrix_state': cell['status'],
            'coverage_k10': old_cov, 'export_cost_original_column': old_cost,
            'cost_column_semantics': cost_kind, 'source': files[table_names[0]]['path']})
        for k, best in enumerate(prefix, 1):
            conditional = _median(best)
            capped = (sum(min(value, cap) if value is not None else cap for value in best) / len(best)
                      if cap is not None else None)
            cost = conditional if cost_kind == 'conditional_median_cost' else capped
            current = {'dataset': dataset, 'method': method, 'k': k, 'theta': theta,
                'coverage': _coverage(best, theta), 'cost': cost, 'cost_semantics': cost_kind,
                'conditional_median_cost': conditional, 'fixed_capped_mean_cost': capped,
                'num_parents': len(parents), 'strict_flip_rate': sum(v is not None for v in best) / len(best),
                'source_state': scope, 'source': source}
            result['figure3'].append(current)
            for metric, old, new in [('coverage', _metric(f3[k-1], 'coverage', 'close_cf_coverage'), current['coverage']),
                                     ('cost', _metric(f3[k-1], 'cost', cost_kind), cost)]:
                if _different(old, new):
                    _require(is_legacy, f'nonlegacy Figure3 mismatch: {dataset}/{method}/K{k}/{metric}')
                    result['differences'].append({'dataset': dataset, 'method': method,
                        'artifact': 'Figure3', 'k': k, 'metric': metric, 'old': old, 'source_reduced': new,
                        'delta_old_minus_source': old-new if old is not None and new is not None else None})
            if k == TABLE_K:
                result['table2'].append({**current, 'k20_coverage': _coverage(prefix[-1], theta),
                    'registered_coverage_k10': old_cov, 'registered_cost_k10': old_cost,
                    'CFDrop': _metric(table, 'CFDrop', 'mean_cf_drop'),
                    'applicable_rate': _metric(table, 'applicable_rate'),
                    'rank_scope': 'OBSERVED_DATASET_CELLS_ONLY'})
        for old, threshold in zip(f4, grid):
            cov = _coverage(prefix[TABLE_K-1], threshold)
            original_k = _integer(old.get('k', TABLE_K))
            _require(original_k == TABLE_K or (original_k == K_MAX and prefix[-1] == prefix[TABLE_K-1]),
                     'Figure4 source K differs without exact prefix equality')
            result['figure4'].append({'dataset': dataset, 'method': method, 'k': TABLE_K,
                'threshold': threshold, 'coverage': cov, 'num_parents': len(parents),
                'source_state': scope, 'source': source})
            old_value = _metric(old, 'coverage', 'close_cf_coverage')
            if _different(old_value, cov):
                _require(is_legacy, f'nonlegacy Figure4 mismatch: {dataset}/{method}')
                result['differences'].append({'dataset': dataset, 'method': method,
                    'artifact': 'Figure4', 'k': TABLE_K, 'metric': 'coverage', 'threshold': threshold,
                    'old': old_value, 'source_reduced': cov,
                    'delta_old_minus_source': old_value-cov if old_value is not None else None})
        exact = _coverage(prefix[TABLE_K-1], theta)
        current_table = result['table2'][-1]
        _require(is_legacy or (not _different(old_cov, exact) and not _different(old_cost, current_table['cost'])),
                 f'nonlegacy Table2 mismatch: {dataset}/{method}')
        result['keypoints'].append({'dataset': dataset, 'method': method, 'k': TABLE_K,
            'theta_exact': theta, 'coverage_exact': exact,
            'covered_count': sum(v is not None and v <= theta for v in prefix[TABLE_K-1]),
            'num_parents': len(parents), 'theta_on_original_grid': theta in grid,
            'grid_length_unchanged': len(grid), 'interpolation_used': False})
        result['cross_checks'].append({'dataset': dataset, 'method': method,
            'registered_table2': old_cov, 'reduced_table2': exact, 'reduced_figure3_k10': exact,
            'reduced_figure4_exact_keypoint': exact, 'table2_changed': _different(old_cov, exact),
            'original_Figure4_k': f4[0].get('k'), 'export_Figure4_k': TABLE_K,
            'state': 'SOURCE_RECONCILED' if is_legacy else 'CONSISTENT'})
        result['lineage'].append({'dataset': dataset, 'method': method,
            'matrix_root': cell['standardized_output_root'], 'raw_source': source,
            'raw_source_sha256': source_sha, 'state': scope, 'registry_unchanged': True,
            'legacy_identity_exception_retained': is_legacy, 'new_science_run': False,
            'original_source_payload_reopened': False,
            'source_sha_basis': hash_receipt, 'embedded_csv_utf8_sha256': embedded_sha})
    for dataset in DATASETS:
        rows = [row for row in result['table2'] if row['dataset'] == dataset]
        if not rows:
            continue
        contracts = [cohorts[f'{dataset}/{row["method"]}'] for row in rows]
        reference = contracts[0]
        comparable = all(set(current['parent_ids']) == set(reference['parent_ids'])
            and current['theta'] == reference['theta']
            and current['cost_semantics'] == reference['cost_semantics']
            and (current['cost_semantics'] != 'fixed_capped_mean_cost'
                 or current['cost_cap'] == reference['cost_cap']) for current in contracts)
        scope = ('OBSERVED_CELLS_SNAPSHOT_IDS_AND_CONTRACT_MATCH' if comparable else
                 'UNVERIFIED_COHORT_OR_CONTRACT_COMPARABILITY')
        result['comparison_scope'].append({'dataset': dataset, 'state': scope,
            'complete_dataset': len(rows) == len(METHODS),
            'cross_method_comparisons_enabled': comparable and len(rows) > 1,
            'identity_mapping_inferred': False})
        for row in rows:
            row['rank_scope'] = scope if len(rows) == len(METHODS) else 'PARTIAL_DATASET_RANK_UNAVAILABLE'
            ranked = comparable and len(rows) == len(METHODS)
            row['coverage_rank'] = (1 + sum(other['coverage'] > row['coverage'] + COMPARISON_ATOL for other in rows)
                                    if ranked else None)
            row['cost_rank'] = (None if row['cost'] is None or not ranked else 1 + sum(
                other['cost'] is not None and other['cost'] < row['cost'] - COMPARISON_ATOL for other in rows))
        if not comparable:
            continue
        ours = [row for row in result['figure3'] if row['dataset'] == dataset and row['method'] == 'Ours']
        for method in METHODS[1:]:
            other = [row for row in result['figure3'] if row['dataset'] == dataset and row['method'] == method]
            if not ours or not other:
                continue
            comparison = {'dataset': dataset, 'baseline': method}
            for metric, sign in [('coverage', 1), ('cost', -1)]:
                pairs = [(a['k'], sign * (a[metric] - b[metric])) for a, b in zip(ours, other)
                         if a[metric] is not None and b[metric] is not None]
                comparison[metric+'_wins'] = [k for k, delta in pairs if delta > COMPARISON_ATOL]
                comparison[metric+'_ties'] = [k for k, delta in pairs if abs(delta) <= COMPARISON_ATOL]
                comparison[metric+'_losses'] = [k for k, delta in pairs if delta < -COMPARISON_ATOL]
                comparison[metric+'_unavailable'] = [k for k in range(1, K_MAX+1) if k not in {p[0] for p in pairs}]
            result['comparisons'].append(comparison)
    result['pending'] = [{'dataset': dataset, 'method': method, 'source_state': 'PENDING'}
                         for dataset in DATASETS for method in METHODS if (dataset, method) not in identities]
    result['cohorts'] = cohorts
    result['pointer'] = snapshot['pointer']
    return result


def _csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline='')
    if rows:
        names = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows({key: 'N/A' if value is None else value for key, value in row.items()} for row in rows)
    return stream.getvalue()


def export_snapshot(*, snapshot_root: Path, output_root: Path, project_root: Path) -> dict[str, Any]:
    """Read three bounded snapshot files; write only a fresh external artifact root."""
    source = snapshot_root.resolve(strict=True)
    destination = output_root.resolve()
    _require(source.is_dir(), 'snapshot root is not a directory')
    _require(not destination.exists(), 'output root must be fresh')
    _require(not destination.is_relative_to(source) and not source.is_relative_to(destination),
             'output must not overlap the read-only snapshot root')
    _require(not destination.is_relative_to(project_root.resolve()), 'output must be outside source repository')
    documents, bindings = [], []
    for relative in INPUT_FILES:
        path = source / relative
        _require(path.resolve(strict=True).is_relative_to(source) and not path.is_symlink(),
                 'snapshot input escapes its root or is a symlink')
        _require(path.is_file() and path.stat().st_size <= 64 * 1024 * 1024,
                 'snapshot input missing, special or exceeds 64 MiB')
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 'snapshot input changed during read')
        documents.append(json.loads(content))
        bindings.append({'relative_path': relative, 'bytes': len(content),
                         'sha256': hashlib.sha256(content).hexdigest()})
    reduced = reduce_snapshot(*documents)
    csv_rows = {
        'current_registered_cells.csv': reduced['registered'],
        'current_three_dataset_comparison.csv': [r for r in reduced['table2'] if r['dataset'] in DATASETS[:3]],
        'table2_four_dataset_partial.csv': reduced['table2'] + reduced['pending'],
        'figure3_source_reconciled.csv': reduced['figure3'],
        'figure4_source_reconciled.csv': reduced['figure4'],
        'figure_cross_consistency.csv': reduced['cross_checks'],
        'canonical_vs_legacy_lineage.csv': reduced['lineage'],
        'legacy_source_differences.csv': reduced['differences'],
    }
    payloads = {'source_csv/' + name: _csv_text(rows) for name, rows in csv_rows.items()}
    for name, value in [('key_threshold_metrics.json', reduced['keypoints']),
                        ('cohort_source_binding.json', reduced['cohorts']),
                        ('ours_all_k_comparison.json', reduced['comparisons'])]:
        payloads[name] = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    audit = {'schema_version': 'paper_snapshot_numeric_audit_v1',
        'status': 'PARTIAL_SNAPSHOT_NUMERIC_AUDIT_COMPLETE',
        'main_registered_cells': len(reduced['table2']), 'pending_cells': reduced['pending'],
        'matrix_pointer_as_collected': reduced['pointer'], 'main_matrix_changed': False,
        'distance_recomputed': False, 'oracle_inference_run': False, 'selector_rerun': False,
        'scientific_verification': False, 'matrix_authority_reopened': False,
        'source_record_payloads_reopened': False, 'legacy_identity_waivers_remain_open': True,
        'f4_interpolation': False, 'frozen_k_max': K_MAX, 'table2_k': TABLE_K,
        'legacy_differing_metric_points': len(reduced['differences']),
        'cross_method_comparison_scope': reduced['comparison_scope'],
        'cross_figure_checks': reduced['cross_checks'], 'input_snapshot_files': bindings,
        'limitations': ['Numerical replay of collected prefix snapshots, not independent raw-source verification.',
                       'Partial staging only; never a final matrix PASS or publication authority.',
                       'Historical missing data/split/model identities are not repaired by arithmetic.'],
        'output_sha256': {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in payloads.items()}}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'source_csv').mkdir()
    for name, text in payloads.items():
        with (destination / name).open('x', encoding='utf-8', newline='') as stream:
            stream.write(text)
    with (destination / 'export_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return audit
