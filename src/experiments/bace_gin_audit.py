"""Independent, read-only result reconciliation for the BACE fixed-pool run.

No model, generation, fitting, OT, selector optimization or main publication.
Existing selector-verifier primitives audit the calibration decision; this
module independently reduces saved applications and recomputes heldout metrics.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.bace_frozen_gnn_contracts import atomic_json, read_json, sha256_file, stable_sha256, utc_now

TOLERANCE = 1e-12  # Existing selector FLOAT_TOLERANCE; never inferred from results.


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def equal(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=0., abs_tol=TOLERANCE), context)
    else:
        require(actual == expected, context)


def _rows_with_digest(path: Path) -> tuple[list[dict], str]:
    """One bounded per-parent read, not repeated whole-archive SHA scans."""
    digest, rows = hashlib.sha256(), []
    with path.open('rb') as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows.append(json.loads(line))
    return rows, digest.hexdigest()


def _stat(path: Path) -> tuple:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _prediction(row: Mapping[str, Any], side: str) -> None:
    probabilities, predicted = row.get('p_' + side), row.get('pred_' + side)
    require(isinstance(probabilities, list) and len(probabilities) == 2,
            'APPLICATION_PROBABILITY_EVIDENCE_GAP:' + side)
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in probabilities)
            and abs(sum(probabilities) - 1) <= TOLERANCE, 'INVALID_PROBABILITY_VECTOR:' + side)
    equal(predicted, max(range(2), key=lambda i: probabilities[i]), 'PREDICTION_ARGMAX_MISMATCH:' + side)
    if 'p1_' + side in row:
        equal(row['p1_' + side], probabilities[1], 'SOURCE_PROBABILITY_MISMATCH:' + side)


def audit_parent_rows(pairs: Sequence[Mapping[str, Any]], apps: Sequence[Mapping[str, Any]], *,
                      method: str, split: str, parent_id: str, parent_smiles: str,
                      candidate_ids: Sequence[str], oracle: Mapping[str, Any]) -> dict:
    """Recompute own strict flips and per-pair min; preserve native operations."""
    candidate_set = set(candidate_ids)
    require(len(pairs) == len(candidate_ids) and
            {r['candidate_id'] for r in pairs} == candidate_set, 'PAIR_CARTESIAN_EVIDENCE_GAP')
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    before, probability_before = None, None
    for row in [*pairs, *apps]:
        require(row.get('parent_id') == parent_id and row.get('parent_smiles') == parent_smiles
                and row.get('candidate_id') in candidate_set and row.get('split') == split,
                'APPLICATION_OR_PAIR_INPUT_IDENTITY_MISMATCH')
        require(row.get('oracle_checkpoint_hash') == oracle['model_sha256']
                and row.get('oracle_backend') == 'gnn' and row.get('rf_oracle_used') is False,
                'SAVED_RECORD_GIN_BINDING_MISMATCH')
        require(row.get('oracle_backbone', row.get('classifier_family')) == 'gin',
                'SAVED_RECORD_NOT_GIN')
        if method == 'ours':
            equal(row.get('oracle_temperature'), oracle['temperature'], 'SAVED_RECORD_TEMPERATURE_MISMATCH')
        else:
            require(row.get('temperature_sha256') == oracle['temperature_sha256'],
                    'SAVED_RECORD_TEMPERATURE_BINDING_MISMATCH')
        require(row.get('pred_before') in (0, 1), 'BEFORE_PREDICTION_MISSING')
        before = row['pred_before'] if before is None else before
        equal(row['pred_before'], before, 'INCONSISTENT_PARENT_PREDICTION')
        probability = row.get('p1_before')
        require(isinstance(probability, (float, int)) and math.isfinite(probability) and 0 <= probability <= 1,
                'PARENT_PROBABILITY_EVIDENCE_GAP')
        probability_before = probability if probability_before is None else probability_before
        equal(probability, probability_before, 'INCONSISTENT_PARENT_PROBABILITY')
        equal(row['pred_before'], int(probability > .5), 'PARENT_PROBABILITY_ARGMAX_MISMATCH')
    for app in apps:
        _prediction(app, 'before')
        valid = app.get('delete_valid') is True if method == 'ours' else app.get('applicable') is True
        if method == 'ours':
            require(isinstance(app.get('match_index'), int) and isinstance(app.get('match_atom_indices'), list),
                    'DELETION_MAPPING_EVIDENCE_GAP')
            if valid:
                require(app.get('sanitize_ok') is True and app.get('residual_connected') is True
                        and app.get('residual_num_components') == 1 and bool(app.get('residual_smiles')),
                        'VALID_DELETION_PRODUCT_EVIDENCE_CONFLICT')
        else:
            require(app.get('native_record_kind') == 'complete_graph_intervention'
                    and app.get('operation_is_deletion') is False
                    and app.get('match_atom_indices') is None, 'NATIVE_OPERATION_RELABELED_AS_DELETION')
        if valid:
            _prediction(app, 'after')
            equal(app.get('cf_drop'), app['p_before'][1] - app['p_after'][1], 'APPLICATION_CFDROP_MISMATCH')
        flip = valid and app['pred_before'] == 1 and app.get('pred_after') == 0
        equal(app.get('cf_flip'), flip, 'APPLICATION_STRICT_FLIP_MISMATCH')
        equal(app.get('teacher_strict_flip'), flip, 'APPLICATION_TEACHER_FLIP_MISMATCH')
        if flip:
            value = app.get('wnode_distance')
            require(app.get('distance_ok') is True and isinstance(value, (float, int))
                    and math.isfinite(value) and value >= 0,
                    'STRICT_FLIP_RAW_DISTANCE_FAILURE_NOT_ZERO_COVERAGE')
        else:
            require(app.get('distance_ok') is False and app.get('wnode_distance') is None,
                    'NONFLIP_DISTANCE_SELECTION_CONTAMINATION')
        grouped[app['candidate_id']].append(app)
    for pair in pairs:
        group = grouped[pair['candidate_id']]
        if method == 'ours':
            keys = [a['match_index'] for a in group]
            require(len(keys) == len(set(keys)), 'DUPLICATE_APPLICATION_MATCH')
            equal(pair.get('num_matches'), len(group), 'APPLICATION_COUNT_EVIDENCE_GAP')
            equal(pair.get('applicable'), bool(group), 'APPLICABILITY_DISAGREES_WITH_APPLICATIONS')
            equal(pair.get('num_valid_residuals'), sum(a['delete_valid'] is True for a in group), 'VALID_COUNT_MISMATCH')
            equal(pair.get('num_strict_flip_matches'), sum(a['cf_flip'] for a in group), 'FLIP_COUNT_MISMATCH')
        else:
            require(len(group) == 1, 'NATIVE_SINGLE_FULLGRAPH_APPLICATION_EVIDENCE_GAP')
            equal(pair.get('applicable'), group[0]['applicable'], 'NATIVE_APPLICABILITY_MISMATCH')
        choices = [a for a in group if a['cf_flip']]
        choices.sort(key=lambda a: (a['wnode_distance'], -a['cf_drop'], tuple(a['match_atom_indices'] or [])))
        best = choices[0] if choices else None
        equal(pair.get('pair_strict_flip'), best is not None, 'PAIR_FLIP_NOT_APPLICATION_EXISTENTIAL')
        if best:
            for key in ('wnode_distance', 'residual_smiles', 'pred_after', 'p1_after', 'cf_drop'):
                equal(pair.get(key), best.get(key), 'PAIR_NOT_OWN_MINIMUM:' + key)
            equal(pair.get('distance_for_selection'), best['wnode_distance'], 'PAIR_SELECTION_DISTANCE_MISMATCH')
            if method == 'ours':
                equal(pair.get('best_match_index'), best['match_index'], 'PAIR_MINIMUM_MATCH_MISMATCH')
                equal(pair.get('best_match_atom_indices'), best['match_atom_indices'], 'PAIR_MINIMUM_MAPPING_MISMATCH')
        else:
            require(pair.get('wnode_distance') is None and pair.get('distance_for_selection') == '+inf',
                    'NONFLIP_PAIR_COST_MUST_BE_UNDEFINED')
    return dict(pair_count=len(pairs), application_count=len(apps),
                strict_flip_applications=sum(a['cf_flip'] for a in apps),
                strict_flip_pairs=sum(p['pair_strict_flip'] for p in pairs), pred_before=before)


def recompute_metrics(parent_ids, ordered_ids, pairs, *, theta, cap, endpoints):
    """Independent implementation: no import/call to producer prefix_metrics."""
    require(0 < len(ordered_ids) <= 20 and len(ordered_ids) == len(set(ordered_ids)), 'FROZEN_PREFIX_INVALID')
    lookup = {(r['parent_id'], r['candidate_id']): r for r in pairs}
    require(len(lookup) == len(pairs) and set(lookup) == {(p, c) for p in parent_ids for c in ordered_ids},
            'HELDOUT_CARTESIAN_PRODUCT_GAP')
    before = {p: lookup[p, ordered_ids[0]]['pred_before'] for p in parent_ids}
    native = [p for p in parent_ids if before[p] == 1]
    rows, distances, ecdf = [], [], []
    for k in range(1, 21):
        prefix = ordered_ids[:k]
        best = {}
        for parent in parent_ids:
            values = [lookup[parent, c]['wnode_distance'] for c in prefix if lookup[parent, c]['pair_strict_flip']]
            best[parent] = min(values) if values else math.inf
            distances.append(dict(parent_id=parent, K_requested=k, K_effective=len(prefix),
                pred_before=before[parent], best_valid_distance=None if not values else best[parent]))
        for cohort, members in (('fixed141', parent_ids), ('gin_native', native)):
            values = [best[p] for p in members]
            finite = sorted(x for x in values if math.isfinite(x))
            n = len(members)
            covered = sum(x <= theta for x in values)
            rows.append(dict(cohort=cohort, denominator=n, K_requested=k, K_effective=len(prefix),
                covered_count=covered, coverage=covered/n if n else None,
                finite_strict_flip_count=len(finite), strict_flip_availability=len(finite)/n if n else None,
                fixed_capped_mean=math.fsum(cap if not math.isfinite(x) or x > cap else x for x in values)/n if n else None,
                conditional_median=(finite[(len(finite)-1)//2]+finite[len(finite)//2])/2 if finite else None,
                theta_star=theta, cost_cap=cap))
            if k in (10, 20):
                points = sorted(set([0., theta, *map(float, endpoints)] + [x for x in finite if x <= max(endpoints)]))
                for point in points:
                    count = sum(x <= point for x in values)
                    ecdf.append(dict(cohort=cohort, denominator=n, K_requested=k, K_effective=len(prefix),
                                     threshold=point, covered_count=count, coverage=count/n if n else None))
    return dict(prefix_rows=rows, parent_distances=distances, exact_ecdf=ecdf,
                parent_predictions=[dict(parent_id=p, pred_before=before[p], in_native=p in native) for p in parent_ids])


def compare_metrics(actual, expected):
    for field, rows in expected.items():
        require(isinstance(actual.get(field), list) and len(actual[field]) == len(rows), 'METRIC_ROW_COUNT:' + field)
        for index, (saved, recomputed) in enumerate(zip(actual[field], rows)):
            require(set(saved) == set(recomputed), 'METRIC_SCHEMA_MISMATCH:' + field)
            for key, value in recomputed.items():
                equal(saved[key], value, f'METRIC_MISMATCH:{field}:{index}:{key}')


def audit_method(spec, method, output_path):
    """Audit completed method only; output is a fresh independent receipt."""
    from src.experiments import bace_gin_fixed_pool as driver
    from src.experiments.bace_gin_ours import fixed_source_parents, validate_gin_adoption
    from src.eval.mutagenicity_wnode_selector import audit_mutagenicity_wnode_selector, threshold_bundle_from_dict
    driver.validate_spec(spec)
    root = Path(spec['output_root'])
    output = Path(output_path)
    require(output.is_absolute() and not output.exists(), 'FRESH_ABSOLUTE_AUDIT_OUTPUT_REQUIRED')
    require(output.name not in ('complete.json', 'selection_freeze.json', 'metrics.json'), 'AUDIT_CANNOT_OVERWRITE_SCIENCE')
    spec_sha = stable_sha256(spec)
    # Verify the method's NEW freeze before reading any saved heldout payload.
    frozen = driver.verify_freeze(spec, method)
    fp = driver.freeze_path(spec, method)
    freeze_sha = sha256_file(fp)
    manifests, candidates = driver.pool(spec, method)
    require(manifests.get('state') != 'BLOCKED_MATERIALIZATION', 'BLOCKED_MATERIALIZATION_IS_NOT_VALID_ZERO')
    candidates_by_id = {r['candidate_id']: r for r in candidates}
    require(set(frozen['ordered_rule_ids']) <= set(candidates_by_id), 'FROZEN_RULE_OUTSIDE_ORIGINAL_POOL')
    oracle = validate_gin_adoption(spec)  # Small accepted receipts; no weight/model load.
    equal(read_json(root/'manifests/oracle_contract.json'), oracle, 'ORACLE_ADOPTION_RECEIPT_DRIFT')
    contract = read_json(root/'manifests/experiment_contract.json')
    require(contract['spec_sha256'] == spec_sha and contract['test_payload_opened'] is False, 'EXPERIMENT_BINDING_MISMATCH')
    seals, heldout, counts = [], [], {}
    calibration_digest = hashlib.sha256()
    for split in ('calibration', 'test'):
        parents = fixed_source_parents(spec, split, test_authorized=split == 'test')
        require(len(parents) == spec['base_counts'][split], 'FIXED_COHORT_COUNT_MISMATCH')
        if split == 'calibration':
            equal([p.parent_id for p in parents], contract['calibration_parent_ids'], 'CALIBRATION_COHORT_CHANGED')
        ids = list(candidates_by_id) if split == 'calibration' else frozen['ordered_rule_ids']
        summary = dict(parent_count=len(parents), pair_count=0, application_count=0, strict_flip_applications=0, strict_flip_pairs=0)
        for index, parent in enumerate(parents):
            unit = root/method/split/f'parent-{index:05d}'
            paths = [unit/name for name in ('complete.json', 'pairs.jsonl', 'applications.jsonl')]
            require(all(p.is_file() for p in paths), 'PARENT_APPLICATION_EVIDENCE_GAP:' + str(unit))
            snapshot = [_stat(p) for p in paths]
            # Use the producer's existing lock read-only, never create a second lock.
            with (unit/'writer.lock').open('r') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                receipt = read_json(paths[0])
                expected_binding = stable_sha256(dict(spec=spec_sha, method=method, split=split,
                    parent_id=parent.parent_id, candidates=ids, freeze=frozen if split == 'test' else None))
                require(receipt.get('state') == 'COMPLETE' and receipt.get('parent_index') == index
                        and receipt.get('parent_id') == parent.parent_id and receipt.get('candidate_count') == len(ids)
                        and receipt.get('spec_sha256') == spec_sha and receipt.get('binding') == expected_binding
                        and receipt.get('oracle_backbone') == 'gin', 'PARENT_COMPLETION_BINDING_MISMATCH:' + str(unit))
                if split == 'test':
                    require(datetime.fromisoformat(receipt['completed_at']) >= datetime.fromisoformat(frozen['created_at']),
                            'HELDOUT_COMPLETION_PRECEDES_NEW_FREEZE')
                pair_rows, pair_sha = _rows_with_digest(paths[1])
                app_rows, app_sha = _rows_with_digest(paths[2])
                require(pair_sha == receipt['pairs_sha256'], 'PAIR_FILE_CHANGED:' + str(unit))
                if 'applications_sha256' in receipt:
                    require(app_sha == receipt['applications_sha256'], 'APPLICATION_FILE_CHANGED:' + str(unit))
                stats = audit_parent_rows(pair_rows, app_rows, method=method, split=split,
                    parent_id=parent.parent_id, parent_smiles=parent.smiles, candidate_ids=ids, oracle=oracle)
                for row in [*pair_rows, *app_rows]:
                    original = candidates_by_id[row['candidate_id']]
                    field = 'canonical_fragment' if method == 'ours' else 'canonical_smiles'
                    equal(row.get(field), original[field], 'APPLICATION_ORIGINAL_RULE_IDENTITY_MISMATCH')
                require(snapshot == [_stat(p) for p in paths], 'SOURCE_CHANGED_DURING_INDEPENDENT_AUDIT')
            for key in ('pair_count', 'application_count', 'strict_flip_applications', 'strict_flip_pairs'):
                summary[key] += stats[key]
            seals.append(dict(split=split, parent_id=parent.parent_id, parent_index=index, path=str(unit),
                pair_sha256=pair_sha, applications_sha256=app_sha, completion_binding=expected_binding,
                original_writer_bound_applications='applications_sha256' in receipt,
                application_binding_established_by_this_independent_audit=True))
            if split == 'test':
                heldout.extend(pair_rows)
            else:
                for row in pair_rows:
                    calibration_digest.update((json.dumps(row, sort_keys=True) + '\n').encode())
        counts[split] = summary
    matrix = root/method/'calibration_matrix'
    require(sha256_file(matrix/'pair_matrix.jsonl') == frozen['calibration_matrix_sha256']
            == calibration_digest.hexdigest(), 'FROZEN_CALIBRATION_MATRIX_DIFFERS_FROM_PARENT_RECORDS')
    # Existing audit rechecks all four original calibration variants and decision.
    selector = audit_mutagenicity_wnode_selector(run_dir=root/method/'selector', matrix_run_dir=matrix,
        expected_parent_count=66, expected_candidate_count=len(candidates), expected_top_k=20, expected_table_k=10,
        require_all_variants=True, require_nested_prefix=True, require_monotonic_coverage=True,
        require_nonincreasing_capped_cost=True, forbid_test=True)
    selected = read_json(root/method/'selector'/'variants'/selector['selected_variant']/'selected_top20.json')
    equal([r['candidate_id'] for r in selected['candidates']], frozen['ordered_rule_ids'], 'FROZEN_ORDER_DIFFERS_FROM_CALIBRATION_WINNER')
    thresholds = threshold_bundle_from_dict(driver.bound_json(spec['thresholds']))
    test_ids = [p.parent_id for p in parents]
    independently = recompute_metrics(test_ids, frozen['ordered_rule_ids'], heldout,
        theta=thresholds.theta_star, cap=thresholds.cost_cap, endpoints=thresholds.raw_thresholds)
    metric_path = root/method/'metrics.json'
    actual = read_json(metric_path)
    require(actual.get('state') == 'EVALUATED' and actual.get('spec_sha256') == spec_sha
            and actual.get('freeze_sha256') == freeze_sha and actual.get('main_matrix_write') is False, 'METRIC_INPUT_BINDING_MISMATCH')
    compare_metrics(actual, independently)
    funnel = actual['failure_funnel']
    for field, value in dict(base_parent_count=len(test_ids), selected_rule_count=len(frozen['ordered_rule_ids']),
            pair_count=len(heldout), gin_source_parents=sum(r['in_native'] for r in independently['parent_predictions']),
            applicable_pairs=sum(bool(r['applicable']) for r in heldout),
            strict_flip_pairs_with_exact_distance=sum(bool(r['pair_strict_flip']) for r in heldout)).items():
        equal(funnel.get(field), value, 'FAILURE_FUNNEL_MISMATCH:' + field)
    require(sha256_file(fp) == freeze_sha, 'FREEZE_CHANGED_DURING_AUDIT')
    result = dict(schema='bace_gin_fixed_pool_independent_result_audit_v1', state='RESULT_CONSISTENCY_PASS',
        method=method, spec_sha256=spec_sha, science_execution_commit=spec['execution_commit'],
        selection_freeze_sha256=freeze_sha, metrics_sha256=sha256_file(metric_path), oracle_contract=oracle,
        counts=counts, fixed_test_count=len(test_ids), gin_native_test_count=sum(r['in_native'] for r in independently['parent_predictions']),
        selector_audit=selector, application_seals=seals, new_application_seal_count=len(seals),
        historical_application_hash_gap_count=sum(not x['original_writer_bound_applications'] for x in seals),
        application_minima_independently_checked=True, strict_flip_probabilities_checked=True,
        independent_metrics_match=True, no_model_inference=True, ot_recomputed_count=0,
        original_model_training_scientifically_reexecuted=False, chemistry_reexecution_performed=False,
        scope='SAVED_RECORD_CONSISTENCY_NOT_NEW_MODEL_OR_CHEMISTRY_SCIENTIFIC_REEXECUTION',
        main_matrix_write=False, test_results_previously_observed=True,
        application_history_limitation='Original writer omitted application SHA; new hashes attest present unchanged files, not retroactive sealing.',
        k10=[r for r in independently['prefix_rows'] if r['K_requested'] == 10], created_at=utc_now())
    result['self_sha256'] = stable_sha256(result)
    atomic_json(output, result)
    return result
