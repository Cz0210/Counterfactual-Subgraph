#!/usr/bin/env python3
"""One CPU parent chunk for a genuinely frozen Reach-v2 pool; no selection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ablations.gnn.reach_v2_adapter import (validate_pool, evaluate_parent_chunk,
    require_global_freeze, merge_and_freeze_calibration, split_chunk_size, SCOPE_NAME)
from src.eval.bace_frozen_gnn_contracts import read_json, read_jsonl, sha256_file, atomic_json


def chunk_terminal(*, index, **fields):
    if type(index) is not int or index < 0:
        raise ValueError('CHUNK_TERMINAL_INDEX_MUST_BE_NONNEGATIVE_INTEGER')
    return dict(fields, index=index)


def main():
    started=time.monotonic()
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
    p.add_argument('--prepare-raw-reuse-only', action='store_true',
                   help='Once per split: index accepted old match costs without model/OT inference')
    p.add_argument('--merge-test-only', action='store_true', help='Complete held-out chunk merge and independent audit, no inference')
    p.add_argument('--verify-only', action='store_true', help='Reopen the actual new-pool core audit, not the old66 result')
    p.add_argument('--package-only', action='store_true', help='Package accepted new-pool results and unchanged-classifier adoption')
    args = p.parse_args()
    if sum((args.prepare_output_only, args.merge_calibration_only, args.prepare_raw_reuse_only,
            args.merge_test_only, args.verify_only, args.package_only)) > 1:
        p.error('Preparation, raw-cost adoption and selection are distinct stages')
    if args.merge_calibration_only and args.split != 'calibration':
        p.error('Calibration merge is a distinct calibration-only compute-node stage')
    if (args.merge_test_only or args.verify_only or args.package_only) and args.split != 'test':
        p.error('Held-out closeout requires split=test and all ten fresh selectors')
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
    chunk_size=split_chunk_size(spec,args.split)
    if args.prepare_raw_reuse_only:
        from src.ablations.gnn.reach_raw_distance_reuse import build_index
        kwargs = {}
        if args.split == 'test':
            pool_sha = sha256_file(spec['candidate_universe'])
            freeze_path = Path(spec['global_selector_freeze'])
            if freeze_path != Path(spec['output_root'])/'CALIBRATION_FREEZE.json':
                raise ValueError('TEST_FREEZE_MUST_BELONG_TO_CURRENT_ROOT')
            new_freeze = read_json(freeze_path)
            require_global_freeze(new_freeze, pool_sha)
            if new_freeze.get('spec_sha256') != sha256_file(args.spec):
                raise ValueError('TEST_FREEZE_SPEC_CHANGED')
            kwargs = dict(test_freeze_path=spec['global_selector_freeze'],
                test_freeze_sha=sha256_file(freeze_path),
                validate_test_freeze=lambda frozen: require_global_freeze(frozen, pool_sha))
        path = Path(spec['raw_cost_indexes'][args.split]['path']).resolve()
        path.relative_to('/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn')
        index = build_index(spec['raw_distance_source'], split=args.split, output=path,
                            repo=Path(__file__).resolve().parents[3], **kwargs)
        if args.split == 'test':
            from src.ablations.gnn.reach_v2_closeout import seal_test_dependencies
            seal_test_dependencies(spec, sha256_file(args.spec), pool_sha)
        print(json.dumps({k: index[k] for k in ('state', 'split', 'source_parent_units',
            'source_finite_match_records', 'raw_cost_count', 'ot_recomputed')}))
        return
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
                solver_seconds=spec['solver_seconds_per_k'],
                model_files=spec['model_files'], chunk_size=chunk_size)
        print(json.dumps(dict(state='ALL_TEN_V2_SELECTORS_FROZEN_NOT_CORE_PASS',
            pool_sha256=pool_sha, selector_count=len(result['selectors']), test_loaded=False)))
        return
    test_freeze = None
    dependencies = None
    if args.split == 'test':
        from src.ablations.gnn.reach_v2_closeout import require_test_dependencies, finish_test, verify_closeout, package_closeout
        dependencies, test_freeze = require_test_dependencies(spec, root_binding['spec_sha256'], pool_sha)
        if args.merge_test_only:
            with (output/'closeout.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result=finish_test(spec,spec_sha=root_binding['spec_sha256'],pool_sha=pool_sha,candidates=pool)
            print(json.dumps({k:v for k,v in result.items() if k!='files'})); return
        if args.verify_only:
            result=verify_closeout(output)
            print(json.dumps({k:v for k,v in result.items() if k!='files'})); return
        if args.package_only:
            print(json.dumps(package_closeout(spec))); return
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
    if spec['slots'][args.split] * chunk_size < len(cohort):
        raise ValueError('Array omits source parents')
    chosen = cohort[index * chunk_size:(index + 1) * chunk_size]
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
        from src.ablations.gnn.reach_raw_distance_reuse import VerifiedRawGraphDistance, raw_contract_from_bundle
        source_index = spec['raw_cost_indexes'][args.split]
        expected_index_sha = dependencies['test_index_sha256'] if dependencies else source_index['sha256']
        if sha256_file(source_index['path']) != expected_index_sha:
            raise ValueError('V2_RAW_COST_INDEX_NOT_BOUND')
        raw_index = read_json(source_index['path'])
        if raw_index['split'] != args.split:
            raise ValueError('V2_RAW_COST_WRONG_SPLIT')
        distance = VerifiedRawGraphDistance(distance, index=raw_index,
            current_raw_contract=raw_contract_from_bundle(manifest),
            repo=Path(__file__).resolve().parents[3])
        try:
            def boundary_check():
                free=os.statvfs(output)
                if free.f_bavail*free.f_frsize < int(spec.get('minimum_persistent_free_bytes', 4*1024**3)):
                    raise RuntimeError('RESOURCE_STOP_BEFORE_NEXT_COMPLETE_PARENT')
            rows = evaluate_parent_chunk(chosen, pool, oracle=oracle, featurizer=features,
                distance_provider=distance, output=directory / 'parents', split=args.split,
                pool_sha=pool_sha, temperature_sha=spec['model_files'][args.backbone]['temperature_scaling.json'],
                batch_size=spec['batch_size'], predictions=all_records, test_freeze=test_freeze,
                execution_spec_sha=root_binding['spec_sha256'],boundary_check=boundary_check)
            atomic_json(directory / 'raw_distance_reuse_receipt.json', dict(
                source_index_sha256=raw_index['self_sha256'], adopted_actions=distance.used,
                source_flip_masks_reused=False, source_match_minima_reused=False,
                stats=distance.stats_dict()))
        finally:
            distance.close()
        receipt = chunk_terminal(state='PARENT_CHUNK_COMPLETE_NOT_CORE_PASS', scope=SCOPE_NAME,
            spec_sha256=sha256_file(args.spec), pool_sha256=pool_sha, backbone=args.backbone,
            split=args.split, index=index, parent_ids=[p.parent_id for p in chosen],
            native_cohort_ids=[p.parent_id for p in cohort], pair_count=len(rows),
            model_files=spec['model_files'][args.backbone],
            global_freeze_sha256=dependencies['global_freeze_sha256'] if dependencies else None,
            elapsed_seconds=time.monotonic()-started,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            sealed_parent_bytes=sum(p.stat().st_size for p in (directory/'parents').glob('*.json')),
            total_chunk_persistent_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()),
            global_selector_called=False, main_matrix_write=False, slurm_job_id=os.environ['SLURM_JOB_ID'])
        atomic_json(terminal, receipt)
        print(json.dumps({k: v for k, v in receipt.items() if not k.endswith('_ids')}))


if __name__ == '__main__':
    main()
