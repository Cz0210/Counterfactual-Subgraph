#!/usr/bin/env python3
"""CPU-only one-pass native calibration raw-cost migration; no model/OT calls."""
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
    parser.add_argument('--split', choices=('calibration',), default='calibration',
                        help='Test migration is only callable through the actual new freeze validator API.')
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
        overlay_receipt = dict(driver_commit=args.driver_commit, actual_base_import_commit=actual,
            base_import_root=str(base), driver_root=str(repo), files=identities,
            claim_full_driver_worktree_deployed=False, model_inference=False, ot_recomputed=0)
        repo = base
    else:
        if args.base_commit or args.driver_commit:
            parser.error('--base-repo is required for overlay pin arguments')
        from src.experiments.bace_gin_native_raw import build_native_index
    if overlay_receipt is not None:
        from src.eval.bace_frozen_gnn_contracts import atomic_json
        atomic_json(Path(args.output).parent / (Path(args.output).stem + '.execution_overlay.json'), overlay_receipt)
    result = build_native_index(json.loads(Path(args.binding).read_text()), split=args.split,
        output=Path(args.output), repo=repo)
    print(json.dumps({k: result[k] for k in ('state','source_pair_rows','raw_cost_count',
        'source_missing_raw_distance_rows','ot_recomputed','self_sha256')}, sort_keys=True))


if __name__ == '__main__':
    main()
