#!/usr/bin/env python3
"""CPU-only native raw-cost migration; test requires the actual new GIN freeze."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-repo', help='Existing immutable import/kernel tree for a narrow code overlay')
    parser.add_argument('--base-commit', help='Actual base import-tree HEAD; required with --base-repo')
    parser.add_argument('--driver-commit', help='Committed overlay source; required with --base-repo')
    parser.add_argument('--split', choices=('calibration', 'test'), default='calibration')
    parser.add_argument('--experiment-spec', help='Unmodified original scheme-A spec copied from HPC')
    parser.add_argument('--experiment-spec-sha', help='Expected semantic stable_sha256 of that sealed spec')
    parser.add_argument('--test-freeze', help='Actual new per-method selection_freeze.json copied from HPC')
    parser.add_argument('--test-freeze-sha', help='Actual file SHA of the new HPC freeze receipt')
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error('Actual --config path must exist')
    repo = Path(__file__).resolve().parents[2]
    overlay_receipt = None
    if args.base_repo:
        if not args.base_commit or not args.driver_commit:
            parser.error('Overlay requires separate pinned --base-commit and --driver-commit')
        base = Path(args.base_repo).resolve(strict=True)
        actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=base, text=True).strip()
        if actual != args.base_commit:
            parser.error('Existing immutable base import HEAD changed')
        identities = {}
        for relative in ('scripts/experiments/migrate_bace_gin_native_raw.py',
                         'src/experiments/bace_gin_native_raw.py',
                         'src/ablations/gnn/reach_raw_distance_reuse.py'):
            data = (repo / relative).read_bytes()
            original = subprocess.check_output(['git','show', args.driver_commit + ':' + relative], cwd=base)
            if data != original:
                parser.error('Overlay file differs from its committed driver: ' + relative)
            identities[relative] = hashlib.sha256(data).hexdigest()
        sys.path.insert(0, str(base))
        loader = importlib.util.spec_from_file_location('_native_raw_overlay', repo / 'src/experiments/bace_gin_native_raw.py')
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        build_native_index = module.build_native_index
        validate_freeze = module.validate_portable_scheme_a_freeze
        overlay_receipt = dict(driver_commit=args.driver_commit, actual_base_import_commit=actual,
            base_import_root=str(base), driver_root=str(repo), files=identities,
            claim_full_driver_worktree_deployed=False, model_inference=False, ot_recomputed=0)
        repo = base
    else:
        if args.base_commit or args.driver_commit:
            parser.error('--base-repo is required for overlay pin arguments')
        from src.experiments.bace_gin_native_raw import build_native_index
        from src.experiments.bace_gin_native_raw import validate_portable_scheme_a_freeze as validate_freeze
    if overlay_receipt is not None:
        from src.eval.bace_frozen_gnn_contracts import atomic_json
        atomic_json(Path(args.output).parent / (Path(args.output).stem + '.execution_overlay.json'), overlay_receipt)
    binding = json.loads(Path(args.binding).read_text())
    test_options = {}
    if args.split == 'test':
        if not all((args.experiment_spec, args.experiment_spec_sha, args.test_freeze, args.test_freeze_sha)):
            parser.error('Test requires sealed scheme-A spec and actual new method freeze, each hash-bound')
        experiment = json.loads(Path(args.experiment_spec).read_text())
        test_options = dict(test_freeze_path=args.test_freeze, test_freeze_sha=args.test_freeze_sha,
            validate_test_freeze=lambda receipt: validate_freeze(receipt, experiment_spec=experiment,
                method=binding['method_id'], expected_spec_sha256=args.experiment_spec_sha))
    elif any((args.experiment_spec, args.experiment_spec_sha, args.test_freeze, args.test_freeze_sha)):
        parser.error('Do not attach future test receipts to a calibration migration')
    result = build_native_index(binding, split=args.split, output=Path(args.output), repo=repo, **test_options)
    print(json.dumps({k: result[k] for k in ('state','source_pair_rows','raw_cost_count',
        'source_missing_raw_distance_rows','ot_recomputed','self_sha256')}, sort_keys=True))


if __name__ == '__main__':
    main()
