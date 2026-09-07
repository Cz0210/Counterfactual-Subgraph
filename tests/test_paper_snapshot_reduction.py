"""Tiny synthetic snapshots only; no science imports, models or registry writes."""
from __future__ import annotations

import copy
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.paper_snapshot_reduction import INPUT_FILES, _median, export_snapshot, reduce_snapshot


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def fixture(dataset='AIDS', best=(0.1, None), method='ComRecGC'):
    theta, cap = 0.1, 0.5
    finite = [v for v in best if v is not None]
    coverage = sum(v is not None and v <= theta for v in best) / 2
    conditional = (sum(finite) / len(finite)) if finite else None
    capped = sum(min(v, cap) if v is not None else cap for v in best) / 2
    cost = conditional if dataset in ('AIDS', 'Mutagenicity') else capped
    f3 = [{'k': k, 'theta': theta, 'coverage': coverage, 'cost': cost if cost is not None else 'N/A'}
          for k in range(1, 21)]
    f4 = [{'k': 10, 'threshold': threshold,
           'coverage': sum(v is not None and v <= threshold for v in best) / 2}
          for threshold in (0.0, 0.1, 0.2)]
    parents = [{'k': k, 'parent_id': f'p{i}', 'best_distance': v if v is not None else 'N/A',
                'strict_recourse_available': v is not None}
               for k in range(1, 21) for i, v in enumerate(best)]
    files = {name: {'path': '/not-opened/source/' + name, 'text': text} for name, text in {
        'summary.json': json.dumps({'theta_star': theta, 'cost_cap': cap}),
        'figure3_coverage_vs_k.csv': csv_text(f3),
        'figure4_coverage_vs_threshold.csv': csv_text(f4),
        'parent_best_distances.csv': csv_text(parents),
        'table2_method_k10.csv': csv_text([{'coverage': coverage, 'cost': cost if cost is not None else 'N/A'}]),
    }.items()}
    cell = {'dataset': dataset, 'method': method, 'status': 'FROZEN_PASS',
            'k_max': 20, 'table2_k': 10, 'standardized_output_root': '/not-opened/source'}
    snap = {'pointer': {'latest_count': 1, 'applied_cells': [dataset + '/' + method]},
            'cells': [{'cell': cell, 'files': files}]}
    return snap, {'cells': []}, {}


def change_csv(documents, name, update):
    descriptor = documents[0]['cells'][0]['files'][name]
    rows = list(csv.DictReader(io.StringIO(descriptor['text'])))
    update(rows)
    descriptor['text'] = csv_text(rows)


def test_inclusive_threshold_and_conditional_not_theta_conditioned():
    result = reduce_snapshot(*fixture(best=(0.1, 0.2)))
    row = result['table2'][0]
    assert row['coverage'] == 0.5
    assert row['cost'] == pytest.approx(0.15)  # Includes strict flip above theta.
    assert result['keypoints'][0]['covered_count'] == 1
    assert result['cross_checks'][0]['reduced_figure4_exact_keypoint'] == row['coverage']
    assert len(result['figure3']) == 20
    assert len(result['figure4']) == 3


def test_zero_is_not_na_or_pending():
    result = reduce_snapshot(*fixture(best=(None, None)))
    row = result['table2'][0]
    assert row['coverage'] == 0
    assert row['cost'] is None
    assert row['fixed_capped_mean_cost'] == 0.5
    assert row['cost_rank'] is None
    assert len(result['pending']) == 15
    assert all('coverage' not in row and 'cost' not in row for row in result['pending'])
    assert result['lineage'][0]['raw_source_sha256'] is None
    assert result['lineage'][0]['state'].endswith('PER_FILE_HASH_UNAVAILABLE')


def test_fixed_cap_uses_all_parents():
    row = reduce_snapshot(*fixture(dataset='BACE', best=(0.2, None)))['table2'][0]
    assert row['coverage'] == 0
    assert row['cost'] == 0.35
    assert row['conditional_median_cost'] == 0.2


def test_conditional_cost_is_median_not_mean():
    assert _median([0.1, 0.2, 9.0, None]) == 0.2
    assert _median([0.0, 0.1, 0.2, 9.0, None]) == pytest.approx(0.15)


@pytest.mark.parametrize('field', ['theta', 'k'])
def test_frozen_csv_metadata_disagreement(field):
    documents = fixture()
    if field == 'theta':
        change_csv(documents, 'figure3_coverage_vs_k.csv', lambda rows: rows[0].update(theta=999))
    else:
        table = documents[0]['cells'][0]['files']['table2_method_k10.csv']
        table['text'] = 'coverage,cost,k\n0.5,0.1,20\n'
    with pytest.raises(ValueError):
        reduce_snapshot(*documents)


def test_distinct_cohorts_do_not_get_comparisons_or_ranks():
    documents = fixture(dataset='BACE', method='Ours')
    other = fixture(dataset='BACE', method='GlobalGCE')
    change_csv(other, 'parent_best_distances.csv', lambda rows: [
        row.update(parent_id='different_' + row['parent_id']) for row in rows])
    documents[0]['cells'].append(other[0]['cells'][0])
    documents[0]['pointer'].update(latest_count=2, applied_cells=['BACE/Ours', 'BACE/GlobalGCE'])
    result = reduce_snapshot(*documents)
    assert not result['comparisons']
    assert result['comparison_scope'][0]['state'] == 'UNVERIFIED_COHORT_OR_CONTRACT_COMPARABILITY'
    assert all(row['coverage_rank'] is None and row['cost_rank'] is None for row in result['table2'])


def test_invalid_nonrecourse_distance_is_not_silently_missing():
    documents = fixture()
    change_csv(documents, 'parent_best_distances.csv', lambda rows: rows[1].update(best_distance='NOT_A_NUMBER'))
    with pytest.raises(ValueError):
        reduce_snapshot(*documents)


@pytest.mark.parametrize('failure', ['duplicate_cell', 'bad_count', 'unknown_method', 'unaccepted',
    'duplicate_k', 'parent_order', 'missing_k', 'bad_boolean', 'infinite', 'negative',
    'increasing_prefix', 'unordered_grid', 'native_figure3_drift', 'native_table2_drift'])
def test_malformed_inputs_fail_closed(failure):
    documents = fixture()
    snap = documents[0]
    if failure == 'duplicate_cell':
        snap['cells'].append(copy.deepcopy(snap['cells'][0]))
    elif failure == 'bad_count':
        snap['pointer']['latest_count'] = 2
    elif failure == 'unknown_method':
        snap['cells'][0]['cell']['method'] = 'CLEAR'
        snap['pointer']['applied_cells'] = ['AIDS/CLEAR']
    elif failure == 'unaccepted':
        snap['cells'][0]['cell']['status'] = 'RUNNING'
    elif failure == 'duplicate_k':
        change_csv(documents, 'figure3_coverage_vs_k.csv', lambda rows: rows[1].update(k=1))
    elif failure == 'parent_order':
        change_csv(documents, 'parent_best_distances.csv', lambda rows: rows[2].update(parent_id='p1'))
    elif failure == 'missing_k':
        change_csv(documents, 'parent_best_distances.csv', lambda rows: rows.pop())
    elif failure == 'bad_boolean':
        change_csv(documents, 'parent_best_distances.csv', lambda rows: rows[0].update(strict_recourse_available='maybe'))
    elif failure in ('infinite', 'negative', 'increasing_prefix'):
        value = {'infinite': 'inf', 'negative': -0.1, 'increasing_prefix': 0.2}[failure]
        change_csv(documents, 'parent_best_distances.csv', lambda rows: rows[2].update(best_distance=value))
    elif failure == 'unordered_grid':
        change_csv(documents, 'figure4_coverage_vs_threshold.csv', lambda rows: rows.reverse())
    elif failure == 'native_figure3_drift':
        change_csv(documents, 'figure3_coverage_vs_k.csv', lambda rows: rows[0].update(coverage=0.8))
    elif failure == 'native_table2_drift':
        change_csv(documents, 'table2_method_k10.csv', lambda rows: rows[0].update(cost=0.8))
    with pytest.raises(ValueError):
        reduce_snapshot(*documents)


def test_legacy_reconciliation_preserves_original_values_and_waiver():
    documents = fixture(method='Ours')
    snap, legacy, manifest = documents
    snap['cells'][0]['cell']['registry_exception'] = 'USER_APPROVED_FROZEN_V4'
    legacy['cells'] = [{'dataset': 'AIDS', 'method': 'Ours', 'parent_ids': ['p0', 'p1'],
        'prefix_best_distances': [[0.1, None] for _ in range(20)],
        'pair_binding': {'path': '/not-opened/pair.csv', 'sha256': 'a'*64},
        'order_binding': {'semantics': 'stable_first_occurrence', 'bound_by_pair_file': True}}]
    manifest.update(aids={'figure3_theta': 0.1}, sources={'aids_table2': {
        'Ours': {'pair_details_path': '/not-opened/pair.csv', 'pair_details_sha256': 'a'*64}}})
    change_csv(documents, 'figure3_coverage_vs_k.csv', lambda rows: rows[0].update(coverage=0.8))
    change_csv(documents, 'table2_method_k10.csv', lambda rows: rows[0].update(coverage=0.8))
    frozen = copy.deepcopy(documents)
    result = reduce_snapshot(*documents)
    assert documents == frozen
    assert result['registered'][0]['coverage_k10'] == 0.8
    assert result['table2'][0]['coverage'] == 0.5
    assert result['differences'][0]['old'] == 0.8
    assert result['lineage'][0]['legacy_identity_exception_retained'] is True
    legacy['cells'][0]['pair_binding']['sha256'] = 'b'*64
    with pytest.raises(ValueError, match='pair source'):
        reduce_snapshot(*documents)


def test_inherited_hash_is_not_normalized_transport_hash():
    documents = fixture()
    files = documents[0]['cells'][0]['files']
    files['final_artifact_audit.json'] = {'text': json.dumps({'audited_files': {
        'parent_best_distances.csv': {'sha256': 'c'*64}}})}
    lineage = reduce_snapshot(*documents)['lineage'][0]
    assert lineage['raw_source_sha256'] == 'c'*64
    assert lineage['embedded_csv_utf8_sha256'] != 'c'*64
    assert lineage['original_source_payload_reopened'] is False
    files['run_manifest.json'] = {'text': json.dumps({'scientific_files': {
        'parent_best_distances.csv': 'd'*64}})}
    with pytest.raises(ValueError, match='conflicting'):
        reduce_snapshot(*documents)


def write_inputs(root, documents):
    for relative, document in zip(INPUT_FILES, documents):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))


def test_fresh_deterministic_export_does_not_touch_inputs(tmp_path):
    source = tmp_path / 'source'
    write_inputs(source, fixture())
    before = {p: p.read_bytes() for p in source.rglob('*.json')}
    first, second = tmp_path / 'one', tmp_path / 'two'
    audit = export_snapshot(snapshot_root=source, output_root=first, project_root=tmp_path / 'repo')
    export_snapshot(snapshot_root=source, output_root=second, project_root=tmp_path / 'repo')
    assert {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()} == {
        p.relative_to(second): p.read_bytes() for p in second.rglob('*') if p.is_file()}
    assert all(path.read_bytes() == content for path, content in before.items())
    assert audit['scientific_verification'] is False
    assert audit['source_record_payloads_reopened'] is False
    assert audit['status'].startswith('PARTIAL_')
    with pytest.raises(ValueError, match='fresh'):
        export_snapshot(snapshot_root=source, output_root=first, project_root=tmp_path / 'repo')
    with pytest.raises(ValueError, match='overlap'):
        export_snapshot(snapshot_root=source, output_root=source / 'out', project_root=tmp_path / 'repo')


def test_cli_under_optimized_python_and_cpu_slurm(tmp_path):
    source = tmp_path / 'source'
    write_inputs(source, fixture())
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, '-I', '-O', '-B', str(repo / 'scripts/paper/reduce_result_snapshot.py'),
           '--snapshot-root', str(source), '--output-root', str(tmp_path / 'output')]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'PARTIAL_SNAPSHOT_NUMERIC_AUDIT_COMPLETE' in result.stdout
    wrapper = (repo / 'scripts/slurm/reduce_result_snapshot.sh').read_text()
    assert '#SBATCH --partition=intel' in wrapper and '#SBATCH --gres' not in wrapper
    assert 'export PYTHONPATH=$PWD' in wrapper
    assert '--config configs/hpc.yaml' in wrapper
    assert '--snapshot-root "$SNAPSHOT_ROOT"' in wrapper
