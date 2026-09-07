#!/usr/bin/env python3
"""One CPU parent chunk for a genuinely frozen Reach-v2 pool; no selection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ablations.gnn.reach_v2_adapter import (validate_pool, evaluate_parent_chunk,
    require_global_freeze, merge_and_freeze_calibration, SCOPE_NAME)
from src.eval.bace_frozen_gnn_contracts import read_json, read_jsonl, sha256_file, atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--spec', required=True, type=Path)
    p.add_argument('--split', required=True, choices=('calibration', 'test'))
    p.add_argument('--backbone', required=True, choices=('gine', 'gin', 'gcn', 'gatv2', 'gatedgcn_plus'))
    p.add_argument('--index', type=int)
    p.add_argument('--prepare-output-only', action='store_true',
                   help='Seal a fresh result-root binding only; no model inference')
    p.add_argument('--merge-calibration-only', action='store_true',
                   help='Merge complete calibration chunks, freeze ten global selectors; no test access')
    args = p.parse_args()
    if args.merge_calibration_only and (args.prepare_output_only or args.split != 'calibration'):
        p.error('Calibration merge is a distinct calibration-only compute-node stage')
    if not args.config.is_file() or (not args.prepare_output_only and not os.environ.get('SLURM_JOB_ID')):
        p.error('Existing config and HPC compute-node Slurm job required; no login-node inference')
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') not in ('', '-1'):
        p.error('CPU-only chunk must not expose GPU')
    spec = read_json(args.spec)
    if spec.get('scope') != SCOPE_NAME or spec.get('max_concurrent_jobs') != 2:
        raise ValueError('Require bounded new-pool GNN spec')
    if any(spec.get(key) is not False for key in ('training_rerun', 'temperature_refit', 'main_matrix_write')):
        raise ValueError('New-pool sensitivity cannot train, refit, or publish main cells')
    if not 1 <= int(spec['chunk_size']) <= 32 or not 1 <= int(spec['cpu_threads']) <= 8:
        raise ValueError('CPU parent chunk resource bound')
    output = Path(spec['output_root']).resolve()
    scope = Path('/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn')
    output.relative_to(scope)
    if output == scope or output.exists() and output.is_symlink():
        raise ValueError('Fresh dedicated evaluation root required')
    pool = read_jsonl(spec['candidate_universe'])
    pool_sha = sha256_file(spec['candidate_universe'])
    freeze = read_json(spec['candidate_freeze'])
    contract = read_json(spec['search_contract'])
    validate_pool(pool, freeze, current_pool_sha=pool_sha, old_ids=contract['old_candidate_ids'])
    root_binding = dict(scope=SCOPE_NAME, spec_sha256=sha256_file(args.spec),
                        pool_sha256=pool_sha, main_matrix_write=False)
    root_marker = output / 'version_scope.json'
    if args.prepare_output_only:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(root_marker, root_binding)
        print(json.dumps(dict(state='FRESH_OUTPUT_BOUND_NOT_SCIENCE', **root_binding)))
        return
    if not root_marker.is_file() or read_json(root_marker) != root_binding:
        raise ValueError('V2_EXPLICIT_FRESH_ROOT_PREPARATION_REQUIRED')
    if args.merge_calibration_only:
        # These small manifests bind old calibration orders and frozen thresholds;
        # no checkpoint/model/test payload is opened by the merge entry.
        old_orders = read_json(spec['old_calibration_orders'])
        thresholds = read_json(spec['thresholds'])
        if (sha256_file(spec['old_calibration_orders']) != spec['old_calibration_orders_sha256']
                or sha256_file(spec['thresholds']) != spec['thresholds_sha256']
                or old_orders.get('split') != 'calibration' or old_orders.get('test_loaded') is not False):
            raise ValueError('V2_CALIBRATION_MERGE_INPUT_BINDING_CONFLICT')
        with (output / 'calibration_merge.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = merge_and_freeze_calibration(output, spec_sha=root_binding['spec_sha256'],
                pool_sha=pool_sha, slots=spec['slots']['calibration'], candidates=pool,
                thresholds=thresholds, old_orders=old_orders['orders'],
                solver_seconds=spec['solver_seconds_per_k'])
        print(json.dumps(dict(state='ALL_TEN_V2_SELECTORS_FROZEN_NOT_CORE_PASS',
            pool_sha256=pool_sha, selector_count=len(result['selectors']), test_loaded=False)))
        return
    test_freeze = None
    if args.split == 'test':
        test_freeze = read_json(spec['global_selector_freeze'])
        require_global_freeze(test_freeze, pool_sha)
    # No heldout payload has been opened before all ten selectors above.
    from src.ablations.gnn import cpu_evaluation as old
    from src.ablations.gnn.cpu_training import load_bundle, bundle_file
    from src.oracles.gnn_oracle import GNNOracle
    from src.ablations.llm.compact_node_cache import install_compact_node_cache
    import torch
    torch.set_num_threads(spec['cpu_threads'])
    root, manifest = load_bundle(spec['bundle_root'])
    models = spec['models']
    if set(models) != set(old.BACKBONES):
        raise ValueError('Exactly the five frozen seed7 models required')
    model = Path(models[args.backbone])
    for leaf in ('model.pt', 'temperature_scaling.json', 'feature_schema.json'):
        if sha256_file(model / leaf) != spec['model_files'][args.backbone][leaf]:
            raise ValueError('Frozen model/temperature/schema binding changed')
    oracle = GNNOracle.from_checkpoint(model, device='cpu', batch_size=spec['batch_size'], verify_hashes=False)
    if oracle.backbone != args.backbone or oracle.source_label != 1 or oracle.num_classes != 2:
        raise ValueError('BACE frozen classifier role conflict')
    features = old._featurizer(root, manifest)
    parents = old._all_parents(bundle_file(root, manifest, manifest['splits'][args.split]))
    predictions = old._predict(parents, oracle, features, args.split, spec['batch_size'])
    all_records = {p.parent_id: r for p, r in zip(parents, predictions, strict=True)}
    cohort = sorted([p for p in parents if p.label == 1 and all_records[p.parent_id]['predicted_label'] == 1],
                    key=lambda row: row.parent_id)
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if index < 0 or index >= spec['slots'][args.split]:
        raise ValueError('Chunk index outside sealed array')
    if spec['slots'][args.split] * spec['chunk_size'] < len(cohort):
        raise ValueError('Array omits source parents')
    chosen = cohort[index * spec['chunk_size']:(index + 1) * spec['chunk_size']]
    if args.split == 'test':
        chosen_ids = set()
        for mode in ('native', 'common'):
            chosen_ids.update(test_freeze['selectors'][f'{args.backbone}/{mode}']['ordered_rule_ids'])
        pool = [r for r in pool if r['candidate_id'] in chosen_ids]
        if {r['candidate_id'] for r in pool} != chosen_ids:
            raise ValueError('Test selectors reference missing v2 candidates')
    directory = output / args.backbone / args.split / f'{index:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        terminal = directory / 'terminal.json'
        if terminal.exists():
            receipt = read_json(terminal)
            if receipt['spec_sha256'] != sha256_file(args.spec):
                raise ValueError('Completed chunk spec conflict')
            print(json.dumps(receipt))
            return
        # Same encoder/solver; compact raw-node cache. Never adopt backbone masks.
        distance = old._distance(root, manifest, directory)
        install_compact_node_cache(distance)
        try:
            rows = evaluate_parent_chunk(chosen, pool, oracle=oracle, featurizer=features,
                distance_provider=distance, output=directory / 'parents', split=args.split,
                pool_sha=pool_sha, temperature_sha=spec['model_files'][args.backbone]['temperature_scaling.json'],
                batch_size=spec['batch_size'], predictions=all_records, test_freeze=test_freeze)
        finally:
            distance.close()
        receipt = dict(state='PARENT_CHUNK_COMPLETE_NOT_CORE_PASS', scope=SCOPE_NAME,
            spec_sha256=sha256_file(args.spec), pool_sha256=pool_sha, backbone=args.backbone,
            split=args.split, index=index, parent_ids=[p.parent_id for p in chosen],
            native_cohort_ids=[p.parent_id for p in cohort], pair_count=len(rows),
            global_selector_called=False, main_matrix_write=False, slurm_job_id=os.environ['SLURM_JOB_ID'])
        atomic_json(terminal, receipt)
        print(json.dumps({k: v for k, v in receipt.items() if not k.endswith('_ids')}))


if __name__ == '__main__':
    main()
