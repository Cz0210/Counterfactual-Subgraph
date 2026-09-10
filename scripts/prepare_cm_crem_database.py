#!/usr/bin/env python3
"""Verify this campaign's author-approved archive on an allocated CPU node."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cm_crem_database_prepare import prepare
from src.baselines.cm_crem_runtime import atomic_json, utc_now

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/hpc.yaml')
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    try:
        result = prepare(args.spec, args.archive, args.output_root, args.run_root)
    except Exception as exc:
        failure = dict(status='ASSET_PREPARATION_FAILED', error=str(exc),
                       stage=getattr(exc,'stage',None), detail=getattr(exc,'receipt',None), created_at=utc_now())
        atomic_json(args.output_root/'prepare_failure.json', failure, immutable=True)
        raise
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
