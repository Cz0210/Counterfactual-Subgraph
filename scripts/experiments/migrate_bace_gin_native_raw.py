#!/usr/bin/env python3
"""CPU-only one-pass native calibration raw-cost migration; no model/OT calls."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('calibration',), default='calibration',
                        help='Test migration is only callable through the actual new freeze validator API.')
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error('Actual --config path must exist')
    from src.experiments.bace_gin_native_raw import build_native_index
    result = build_native_index(json.loads(Path(args.binding).read_text()), split=args.split,
        output=Path(args.output), repo=Path(__file__).resolve().parents[2])
    print(json.dumps({k: result[k] for k in ('state','source_pair_rows','raw_cost_count',
        'source_missing_raw_distance_rows','ot_recomputed','self_sha256')}, sort_keys=True))


if __name__ == '__main__':
    main()
