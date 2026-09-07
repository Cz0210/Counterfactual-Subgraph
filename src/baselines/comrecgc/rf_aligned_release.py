"""Narrow native-summary bridge to the existing shared RF/WNode slot evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .rf_aligned_pool import atomic_json, digest, file_sha


def validate_lineage_records(lineage, summary):
    if [r['candidate_index'] for r in lineage.get('records', [])] != [r['original_candidate_index'] for r in summary]:
        raise ValueError('Selected native lineage candidate sequence changed')
    for record, selected in zip(lineage['records'], summary, strict=True):
        events = record['actions']
        if (record['replayed_compact_graph'] != selected['graph']
                or record['parent_id'] != selected['original_parent_id']
                or len(events) != record['action_count']):
            raise ValueError('Selected native event/graph mapping differs')
        previous = None
        for event in events:
            if (event.get('event') != 'selected_transition' or event.get('action_resolution') != 'exact'
                    or not event.get('action') or event.get('parent_id') != record['parent_id']
                    or (previous is not None and event['source_graph_sha256'] != previous)):
                raise ValueError('Selected native action chain is incomplete or crosses a parent')
            previous = event['target_graph_sha256']
        if events and previous != record['stable_graph_sha256']:
            raise ValueError('Selected native action chain ends at another graph')


def selected_lineage(config, *, recourse_root: Path, output_root: Path):
    """Preserve actual recorded native actions; never infer them from final graphs."""
    from .rf_aligned_pool import checked_events, predecessor_index, replay_candidate, compact_graph
    from .project_dataset import load_aids_generation_bundle
    summary_path = recourse_root / 'selected_native_recourses.json'
    target = output_root / 'selected_action_lineage.json'
    summary_sha = file_sha(summary_path)
    if target.exists():
        saved = json.loads(target.read_text())
        if saved.get('native_summary_sha256') != summary_sha or saved.get('state') != 'RECORDED_ACTION_REPLAY_PASS':
            raise ValueError('Selected native lineage differs from the frozen summary')
        validate_lineage_records(saved, json.loads(summary_path.read_text()))
        return target
    input_root = Path(config['native_lineage_root'])
    index_path = input_root / 'trace/candidate_action_lineage_index.jsonl'
    if file_sha(index_path) != config['candidate_lineage_sha256']:
        raise ValueError('Original selected-action index binding changed')
    selected = json.loads(summary_path.read_text())
    needed = {int(row['original_candidate_index']) for row in selected}
    original = {}
    with index_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row['candidate_index']) in needed:
                original[int(row['candidate_index'])] = row
    if set(original) != needed:
        raise ValueError('Selected candidate absent from original action index')
    # These sealed chunks were already completely replayed by the pool screen.
    # New selected recourses are independently replayed below, including every
    # source/target graph identity. No repeated full-package hash is necessary.
    predecessor, count = predecessor_index(checked_events(input_root, verify_hashes=False))
    pool = json.loads((Path(config['runtime_pool_root']) / 'terminal.json').read_text())
    if count != pool['trace_events']:
        raise ValueError('Original action stream differs from completed pool receipt')
    source = load_aids_generation_bundle(dataset_dir=config['dataset_dir'], source_csv=config['source_csv'])
    parents = dict(zip(source.parent_ids, source.graphs, strict=True))
    from .rf_aligned_pool import recorded_path
    records = []
    for selected_row in selected:
        row = original[int(selected_row['original_candidate_index'])]
        if row['parent_id'] != selected_row['original_parent_id']:
            raise ValueError('Selected native source parent changed')
        replayed = replay_candidate(row, predecessor, parents)
        if compact_graph(replayed) != selected_row['graph']:
            raise ValueError('Actual selected-action replay differs from selected medoid graph')
        records.append({'candidate_index': row['candidate_index'], 'parent_id': row['parent_id'],
                        'stable_graph_sha256': row['stable_graph_sha256'],
                        'official_graph_hash': row['official_graph_hash'],
                        'actions': recorded_path(row, predecessor),
                        'action_count': row['action_count'], 'replayed_compact_graph': selected_row['graph']})
    atomic_json(target, {'state': 'RECORDED_ACTION_REPLAY_PASS', 'native_summary_sha256': summary_sha,
                        'original_candidate_index_sha256': config['candidate_lineage_sha256'],
                        'original_trace_manifest': str(input_root / 'trace/selected_action_trace_manifest.json'),
                        'original_trace_manifest_sha256': file_sha(input_root / 'trace/selected_action_trace_manifest.json'),
                        'pool_contract_sha256': pool['contract_sha'], 'records': records,
                        'scientific_rng_used': False, 'generated_actions_inferred': False})
    return target


def freeze_summary(config, *, recourse_root: Path, output_root: Path):
    from .slot_evaluation import write_csv
    terminal = json.loads((recourse_root / 'terminal.json').read_text())
    if terminal.get('state') != 'RF_ALIGNED_NATIVE_SUMMARY_COMPLETE' or terminal.get('old_cluster_labels_reused') is not False:
        raise ValueError('Completed new-universe native common-recourse summary required')
    rows = json.loads((recourse_root / 'selected_native_recourses.json').read_text())
    if not rows or len(rows) != terminal['selected_count']:
        raise ValueError('Native medoid summary is absent or incomplete')
    if len({r['cluster_id'] for r in rows}) != len(rows):
        raise ValueError('Native summary repeats a cluster; no artificial rule padding')
    slots = []
    for rank, row in enumerate(rows, 1):
        if row['rf']['prediction'] != 0 or not row['canonical_smiles'] or not row.get('graph'):
            raise ValueError('Selected native medoid lacks chemistry/graph/RF target identity')
        slots.append({'official_cluster_rank': rank, 'cluster_id': str(row['cluster_id']), 'candidate_id': 'AIDS_NATIVE_' + str(row['original_candidate_index']), 'repair_success': True, 'repaired_smiles': row['canonical_smiles'], 'original_parent_id': row['original_parent_id'], 'native_graph': row['graph'], 'rf_target_prediction': 0, 'materialization': 'already_screened_native_graph_no_additional_chemical_repair', 'invalid_slot_backfill': False, 'rank_compaction': False})
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {'schema_version': 'aids_rf_aligned_summary_freeze_v1', 'method_variant': 'ComRecGC-RFAligned', 'run_complete': True, 'rf_pool_provenance_closed': True, 'summary_frozen_before_evaluation': True, 'native_summary_sha256': file_sha(recourse_root / 'selected_native_recourses.json'), 'recourse_terminal_sha256': file_sha(recourse_root / 'terminal.json'), 'native_recourse_root': str(recourse_root), 'source_scope': 'AIDS_HIV_EXISTING_1283_GENERATION_OVERLAP_NOT_UNSEEN_TEST', 'source_label': 1, 'target_label': 0, 'parent_count': 1283, 'source_eligible_count': terminal['source1_count'], 'benchmark_test_previously_seen': True, 'repair_selected_using_test': False, 'calibration_split_invented': False, 'selection_scope': 'original_native_train_source_common_recourse', 'candidate_order_unchanged': True, 'additional_chemical_repair': False, 'old_dbscan_labels_reused': False, 'project_commit': config['execution_commit'], 'upstream_commit': config['official_upstream_commit'], 'dataset_fingerprint': config['dataset_fingerprint'], 'generation_parent_ids_sha256': config['generation_parent_ids_sha256'], 'teacher_sha256': config['rf_sha256'], 'repair_policy_sha256': digest(config['evaluation_contract']), 'official_rank_count': len(slots)}
    target = output_root / 'run_manifest.json'
    if target.exists():
        if json.loads(target.read_text()) != identity:
            raise ValueError('Existing summary freeze differs; do not overwrite')
        return identity
    write_csv(output_root / 'medoid_validity.csv', slots)
    atomic_json(target, identity)
    return identity


def evaluate_frozen_summary(config, *, recourse_root: Path, output_root: Path):
    from scripts.baselines.comrecgc.run_slot_unified_eval import run
    frozen = output_root / 'summary_freeze'
    freeze_summary(config, recourse_root=recourse_root, output_root=frozen)
    contract = config['evaluation_contract']
    threshold_path = output_root / 'frozen_thresholds.json'
    if threshold_path.exists() and json.loads(threshold_path.read_text()) != contract['thresholds']:
        raise ValueError('Frozen evaluation threshold contract changed')
    atomic_json(threshold_path, contract['thresholds'])
    args = argparse.Namespace(mode='full', dataset='aids', chemistry_dir=str(frozen), dataset_csv=contract['source_csv'], teacher_path=contract['rf_path'], molclr_root=contract['molclr_root'], molclr_checkpoint=contract['molclr_checkpoint'], thresholds_json=str(threshold_path), theta_star=None, cost_cap=None, output_dir=str(output_root / 'unified_eval'), expected_parent_count=1283, max_k=20, device='cpu', resume=True, method_variant='ComRecGC-RFAligned')
    command = [sys.executable, 'scripts/baselines/comrecgc/run_slot_unified_eval.py', '--config', 'configs/hpc.yaml', '--set', 'inference.fallback_to_heuristic=false', '--mode', 'full', '--dataset', 'aids', '--chemistry-dir', str(frozen), '--dataset-csv', args.dataset_csv, '--teacher-path', args.teacher_path, '--molclr-root', args.molclr_root, '--molclr-checkpoint', args.molclr_checkpoint, '--thresholds-json', str(threshold_path), '--output-dir', args.output_dir, '--expected-parent-count', '1283', '--max-k', '20', '--device', 'cpu', '--resume', '--method-variant', 'ComRecGC-RFAligned']
    atomic_json(output_root / 'evaluation_dispatch.json', {'argv': command, 'selector_frozen_first': True, 'fullgraph_not_ours_deletion': True, 'same_1283_denominator': True, 'matrix_write_enabled': False})
    return run(args)


def validate_release(root_like, *, proc_root='/proc', require_writer_audit=True):
    """Reopen completed native science, frozen shared evaluation and provenance."""
    from src.eval.non_taste_matrix_append import _validate_rf_standardized, _writer_audit
    from .external_memory_dbscan import ExternalDBSCANContract, fit_external_memory_dbscan
    root = Path(root_like).resolve(strict=True)
    run = json.loads((root / 'run_manifest.json').read_text())
    expected = {'schema_version': 'aids_rf_aligned_release_v1', 'status': 'PASS',
                'method_variant': 'ComRecGC-RFAligned', 'dataset': 'aids',
                'source_label': 1, 'target_label': 0, 'source_denominator': 1283,
                'rf_source1_count': 1097, 'old_dbscan_labels_reused': False,
                'benchmark_test_previously_seen': True, 'repair_selected_using_test': False,
                'selection_frozen_before_evaluation': True, 'generation_rerun': False}
    if any(run.get(k) != v for k, v in expected.items()):
        raise ValueError('RFAligned release does not carry the frozen scientific scope')
    sources = run.get('bound_sources', {})
    required = {'native_terminal', 'native_summary', 'universe', 'count_manifest', 'dbscan_manifest', 'summary_freeze', 'source_pool_terminal', 'selected_action_lineage'}
    if set(sources) != required:
        raise ValueError('RFAligned scientific source closure is incomplete')
    payload = {}
    for name, identity in sources.items():
        path = Path(identity['path'])
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or file_sha(path) != identity['sha256']:
            raise ValueError('RFAligned changed scientific receipt: ' + name)
        payload[name] = json.loads(path.read_text())
    native = payload['native_terminal']
    universe = payload['universe']
    count = payload['count_manifest']
    cluster = payload['dbscan_manifest']
    summary = payload['native_summary']
    freeze = payload['summary_freeze']
    pool = payload['source_pool_terminal']
    if native.get('state') != 'RF_ALIGNED_NATIVE_SUMMARY_COMPLETE' or native.get('old_cluster_labels_reused') is not False:
        raise ValueError('New-universe native summary has not completed')
    if pool.get('state') != 'POOL_SCREEN_COMPLETE' or pool.get('counts', {}).get('CACHE_PROVENANCE_GAP', 0):
        raise ValueError('Full native pool screen is not closed')
    if universe.get('pool_contract') != pool['contract_sha'] or count.get('identity_sha') != digest(universe):
        raise ValueError('Native pool/count scientific universe differs')
    if count.get('state') != 'COUNT_COMPLETE' or native['pair_rows'] != count['pair_count'] or cluster.get('num_samples') != count['pair_count']:
        raise ValueError('Exact recourse pair count differs between stages')
    if (universe.get('source_denominator') != 1283 or len(universe.get('source_positions', [])) != 1097
            or universe.get('theta') != .1 or universe.get('eps') != .02 or universe.get('min_samples') != 3
            or universe.get('old_dbscan_labels_reused') is not False):
        raise ValueError('Native GREED/DBSCAN/source contract changed')
    if not summary or len(summary) != native['selected_count'] or any(r['rf']['prediction'] != 0 or not r.get('graph') for r in summary):
        raise ValueError('Native selected graph/RF result is incomplete')
    if freeze.get('native_summary_sha256') != sources['native_summary']['sha256'] or freeze.get('summary_frozen_before_evaluation') is not True:
        raise ValueError('Native summary was not bound before evaluation')
    lineage = payload['selected_action_lineage']
    if (lineage.get('state') != 'RECORDED_ACTION_REPLAY_PASS'
            or lineage.get('native_summary_sha256') != sources['native_summary']['sha256']
            or lineage.get('pool_contract_sha256') != pool['contract_sha']
            or lineage.get('generated_actions_inferred') is not False
            or [r['candidate_index'] for r in lineage.get('records', [])] != [r['original_candidate_index'] for r in summary]):
        raise ValueError('Actual selected-action lineage is not closed')
    validate_lineage_records(lineage, summary)
    dbscan_root = Path(sources['dbscan_manifest']['path']).parent
    if not (dbscan_root / 'run_manifest.json').is_file() or cluster.get('run_complete') is not True:
        raise ValueError('DBSCAN is not terminal; verifier must not run new clustering')
    contract = ExternalDBSCANContract(**cluster['scientific_identity']['contract'])
    # This API takes its terminal-only branch. It verifies the existing exact
    # certificates and immutable source stat/hash binding, never mines or fits.
    fit_external_memory_dbscan(vectors_path=cluster['scientific_identity']['vectors_path'], work_dir=dbscan_root, contract=contract, expected_vectors_sha256=cluster['scientific_identity']['vectors_sha256'], resume=True)
    standardized = _validate_rf_standardized(root, dataset='AIDS', dataset_key='aids', method_name='ComRecGC-RFAligned')
    if run.get('standardized_run_manifest_sha256') != standardized['run_manifest_sha256']:
        raise ValueError('Standardized evaluation does not bind to this release')
    evaluation = json.loads((Path(standardized['root']) / 'run_manifest.json').read_text())
    if (evaluation.get('native_summary_freeze_manifest_sha256') != sources['summary_freeze']['sha256']
            or evaluation.get('parent_count') != 1283 or evaluation.get('source_eligible_count') != 1097):
        raise ValueError('Evaluation changed summary order or denominator')
    verify_reference_threshold_identity(evaluation)
    return {'terminal_kind': 'AIDS_RFALIGNED_CORRECTIVE_RELEASE', 'root': str(root),
            'run_manifest_sha256': file_sha(root / 'run_manifest.json'), 'standardized': standardized,
            'identities': standardized['identities'], 'scientific_output_empty': False,
            'writer_audit': _writer_audit(root, proc_root=proc_root, required=require_writer_audit),
            'native_writer_audit': _writer_audit(Path(sources['native_terminal']['path']).parent, proc_root=proc_root, required=require_writer_audit)}


def verify_reference_threshold_identity(evaluation):
    import csv
    import hashlib
    reference = evaluation.get('threshold_reference')
    if not isinstance(reference, dict):
        raise ValueError('Exact frozen AIDS reference grid identity is required')
    path = Path(reference['figure4_path'])
    if file_sha(path) != reference['figure4_sha256']:
        raise ValueError('Frozen AIDS reference Figure4 changed')
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    strings = [r['threshold'] for r in rows]
    sequence_sha = hashlib.sha256(('\n'.join(strings) + '\n').encode()).hexdigest()
    if (sequence_sha != reference['threshold_raw_string_sha256']
            or [float(x) for x in strings] != evaluation['threshold_grid']
            or evaluation['threshold_config_hash'] != reference['threshold_raw_string_sha256']):
        raise ValueError('Exact threshold sequence or raw-string source binding changed')
    return reference


def complete_release(config, *, recourse_root: Path, output_root: Path):
    from .recovery_gate import gate_project_full
    from scripts.baselines.comrecgc.freeze_recovery_result import freeze
    release = output_root / 'release'
    if (release / 'run_manifest.json').exists():
        validate_release(release, require_writer_audit=False)
        return json.loads((release / 'run_manifest.json').read_text())
    lineage_path = selected_lineage(config, recourse_root=recourse_root, output_root=output_root)
    evaluate_frozen_summary(config, recourse_root=recourse_root, output_root=output_root)
    gate_project_full(output_root / 'unified_eval', output_root / 'gate', dataset='aids', expected_parent_count=1283, expected_teacher_sha256=config['rf_sha256'], expected_project_commit=config['execution_commit'])
    if not (release / 'standardized').exists():
        freeze(source_dir=output_root / 'unified_eval', gate_dir=output_root / 'gate', output_dir=release / 'standardized', dataset='aids')
    else:
        from src.eval.non_taste_matrix_append import _validate_rf_standardized
        _validate_rf_standardized(release, dataset='AIDS', dataset_key='aids', method_name='ComRecGC-RFAligned')
    paths = {'native_terminal': recourse_root / 'terminal.json', 'native_summary': recourse_root / 'selected_native_recourses.json', 'universe': recourse_root / 'universe_manifest.json', 'count_manifest': recourse_root / 'exact_count/manifest.json', 'dbscan_manifest': recourse_root / 'dbscan/run_manifest.json', 'summary_freeze': output_root / 'summary_freeze/run_manifest.json', 'source_pool_terminal': Path(config['runtime_pool_root']) / 'terminal.json', 'selected_action_lineage': lineage_path}
    receipt = {'schema_version': 'aids_rf_aligned_release_v1', 'status': 'PASS', 'method_variant': 'ComRecGC-RFAligned', 'dataset': 'aids', 'source_label': 1, 'target_label': 0, 'source_denominator': 1283, 'rf_source1_count': 1097, 'old_dbscan_labels_reused': False, 'benchmark_test_previously_seen': True, 'repair_selected_using_test': False, 'selection_frozen_before_evaluation': True, 'generation_rerun': False, 'bound_sources': {k: {'path': str(p), 'sha256': file_sha(p)} for k, p in paths.items()}, 'standardized_run_manifest_sha256': file_sha(release / 'standardized/run_manifest.json')}
    atomic_json(release / 'run_manifest.json', receipt)
    independent = validate_release(release, require_writer_audit=False)
    atomic_json(release / 'repair_scientific_audit.json', {**independent, 'audit_passed': True, 'independent_external_publisher_reopen_required': True, 'matrix_written': False})
    return receipt


def predecessor_state(recourse_root: Path, owner_terminal: Path):
    """No submitted/running/exit-zero-only state qualifies as native science."""
    if not owner_terminal.exists():
        return 'WAITING_NATIVE_OWNER'
    terminal = json.loads(owner_terminal.read_text())
    if terminal.get('returncode') != 0:
        return 'BLOCKED_NATIVE_OWNER_FAILED'
    native_path = recourse_root / 'terminal.json'
    if not native_path.exists():
        return 'BLOCKED_NATIVE_SCIENCE_INCOMPLETE'
    native = json.loads(native_path.read_text())
    return ('READY' if native.get('state') == 'RF_ALIGNED_NATIVE_SUMMARY_COMPLETE'
            and native.get('old_cluster_labels_reused') is False else 'BLOCKED_NATIVE_SCIENCE_INCOMPLETE')


def wait_and_release(config, *, recourse_root: Path, output_root: Path):
    """One AIDS stage successor, using the existing science/resource interfaces."""
    import fcntl
    import os
    import time
    from .rf_aligned_recourse import runtime_resource_gate
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / 'release_owner.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state = predecessor_state(recourse_root, Path(config['predecessor_owner_terminal']))
            atomic_json(output_root / 'release_heartbeat.json', {'state': state, 'pid': os.getpid(), 'sampled_at_unix': time.time(), 'recourse_root': str(recourse_root), 'GPU_requested': False})
            if state.startswith('BLOCKED_'):
                atomic_json(output_root / 'release_terminal.json', {'state': state, 'new_clustering_started': False, 'matrix_written': False})
                return {'state': state}
            if state == 'READY':
                # A terminal filename alone is not a released single-writer
                # boundary. The previous CPU lease must actually be available.
                with (recourse_root / 'writer.lock').open('a+') as predecessor:
                    try:
                        fcntl.flock(predecessor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(60)
                        continue
                    fcntl.flock(predecessor.fileno(), fcntl.LOCK_UN)
                runtime_resource_gate(config, output_root, 'NATIVE_SUMMARY_SHARED_EVALUATION')
                break
            time.sleep(60)
        import threading
        stop = threading.Event()
        def heartbeat():
            while not stop.is_set():
                atomic_json(output_root / 'release_heartbeat.json', {'state': 'SHARED_RF_WNODE_EVALUATION', 'pid': os.getpid(), 'sampled_at_unix': time.time(), 'GPU_requested': False})
                stop.wait(60)
        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        try:
            result = complete_release(config, recourse_root=recourse_root, output_root=output_root)
            runtime_resource_gate(config, output_root, 'FINAL_INDEPENDENT_PUBLICATION')
            published = publish_release(config, output_root=output_root) if config.get('publish_corrective_version') else None
            atomic_json(output_root / 'release_terminal.json', {'state': 'SHARED_RF_WNODE_RELEASE_PUBLISHED' if published else 'SHARED_RF_WNODE_RELEASE_COMPLETE', 'release_root': str(output_root / 'release'), 'matrix_written': published is not None, 'publication': published, 'audit_path': str(output_root / 'release/repair_scientific_audit.json')})
            return result
        except Exception as exc:
            atomic_json(output_root / 'release_terminal.json', {'state': 'FAILED_RELEASE_STAGE', 'error_type': type(exc).__name__, 'error': str(exc), 'source_artifacts_preserved': True, 'automatic_science_retry': False})
            raise
        finally:
            stop.set()
            monitor.join(timeout=5)


def publish_release(config, *, output_root: Path):
    """Call the original authority once, with at most one fresh stale-CAS retry."""
    import subprocess
    import uuid
    from src.eval.fast16_matrix_authority_pointer import read_authority_pointer
    authority = config['matrix_publication']
    worktree = Path(authority['worktree'])
    actual = subprocess.check_output(['git', '-C', str(worktree), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != authority['execution_commit']:
        raise ValueError('Publisher execution identity changed')
    for attempt in (1, 2):
        before = read_authority_pointer(state_path=authority['state_path'], lock_path=authority['lock_path'], initial_authority_root=None)
        prior = {'root': before['latest_authority_root'], 'matrix_sha256': before['latest_matrix_status_sha256'], 'complete': before['latest_count']}
        destination = Path(authority['output_parent']) / ('aids-rfaligned-' + str(uuid.uuid4()))
        command = [sys.executable, '-I', '-B', str(worktree / 'scripts/autodl/append_non_taste_matrix_authority.py'), '--dataset', 'AIDS', '--method', 'ComRecGC', '--cell-terminal-root', str(output_root / 'release'), '--output-root', str(destination), '--supersede-existing', '--expected-prior-authority-root', prior['root'], '--expected-prior-matrix-sha256', prior['matrix_sha256'], '--authority-state-path', authority['state_path'], '--authority-lock-path', authority['lock_path']]
        atomic_json(output_root / f'publication_dispatch_{attempt}.json', {'argv': command, 'expected_prior_complete': prior['complete'], 'operation': 'SAME_CELL_VERSION_SUPERSESSION'})
        result = subprocess.run(command, cwd=worktree, text=True, capture_output=True)
        log = output_root / f'publication_attempt_{attempt}.log'
        log.write_text(result.stdout + result.stderr)
        if result.returncode == 0:
            return {'state': 'PUBLISHED', 'authority_root': str(destination), 'count_before': prior['complete'], 'count_changed': False, 'dispatch': str(output_root / f'publication_dispatch_{attempt}.json')}
        if 'STALE_SUPERSESSION_CAS' not in result.stderr or attempt == 2:
            raise RuntimeError(f'Original authority refused corrective publication; see {log}')
    raise AssertionError('Unreachable publication state')
