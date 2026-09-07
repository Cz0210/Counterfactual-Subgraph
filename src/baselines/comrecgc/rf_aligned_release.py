"""Narrow native-summary bridge to the existing shared RF/WNode slot evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .rf_aligned_pool import atomic_json, digest, file_sha


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
