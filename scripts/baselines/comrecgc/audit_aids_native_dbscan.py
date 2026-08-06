#!/usr/bin/env python3
"""Audit frozen AIDS native COMRECGC recourse geometry without regeneration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.aids_dbscan_audit import (  # noqa: E402
    run_aids_native_dbscan_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--upstream-root", default="external/COMRECGC")
    parser.add_argument("--counterfactuals-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-limit", type=int, default=64)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-candidates", type=int, default=31)
    parser.add_argument("--expected-distance-pairs", type=int)
    parser.add_argument("--expected-eligible-pairs", type=int)
    parser.add_argument("--full-reject-parent-universe", action="store_true")
    parser.add_argument("--preregistration-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trusted-dataset-payload", required=True)
    parser.add_argument("--expected-cache-inventory-sha256", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project = Path(args.project_root).expanduser().resolve()
    upstream = Path(args.upstream_root)
    if not upstream.is_absolute():
        upstream = project / upstream
    result = run_aids_native_dbscan_audit(
        project_root=project,
        upstream_root=upstream,
        counterfactuals_path=args.counterfactuals_path,
        output_dir=args.output_dir,
        parent_limit=args.parent_limit,
        expected_sha256=args.expected_sha256,
        expected_candidates=args.expected_candidates,
        expected_distance_pairs=args.expected_distance_pairs,
        expected_eligible_pairs=args.expected_eligible_pairs,
        full_reject_parent_universe=args.full_reject_parent_universe,
        preregistration_path=args.preregistration_path,
        device=args.device,
        batch_size=args.batch_size,
        trusted_dataset_payload=args.trusted_dataset_payload,
        expected_cache_inventory_sha256=args.expected_cache_inventory_sha256,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
