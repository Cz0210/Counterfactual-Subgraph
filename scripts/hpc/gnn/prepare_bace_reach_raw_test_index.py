#!/usr/bin/env python3
"""Post-freeze metadata-only raw-cost adoption for Ours' descriptive test."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.reach_raw_distance_reuse import build_index, validate_ours_final_freeze
from src.eval.bace_frozen_gnn_contracts import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--source-spec', required=True, type=Path)
    p.add_argument('--freeze-evidence', required=True, type=Path)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if not args.config.is_file() or not os.environ.get('SLURM_JOB_ID'):
        p.error('Existing config and bounded compute-node job required')
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') not in ('', '-1'):
        p.error('No GPU is used by this metadata-only adoption')
    scope = Path('/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn')
    args.output.resolve().relative_to(scope)
    args.freeze_evidence.resolve().relative_to(scope)
    result = build_index(read_json(args.source_spec)['raw_distance_source'], split='test',
        output=args.output, repo=Path(__file__).resolve().parents[3],
        test_freeze_path=args.freeze_evidence/'final_test_binding.json',
        test_freeze_sha=args.freeze_sha256,
        validate_test_freeze=lambda frozen: validate_ours_final_freeze(frozen, args.freeze_evidence))
    print(json.dumps({k: result[k] for k in ('state', 'split', 'source_parent_units',
        'source_finite_match_records', 'raw_cost_count', 'ot_recomputed')}))


if __name__ == '__main__':
    main()
