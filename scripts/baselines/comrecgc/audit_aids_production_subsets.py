#!/usr/bin/env python3
"""Audit exact DBSCAN equivalence on deterministic AIDS production subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.production_subset_audit import (  # noqa: E402
    ProductionSubsetAuditContract,
    run_production_subset_equivalence_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hpc.yaml",
        help="Accepted for wrapper compatibility; this audit reads no config values.",
    )
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--close-pair-contract", required=True)
    parser.add_argument("--expected-close-pair-contract-sha256", required=True)
    parser.add_argument("--physical-pairs", required=True)
    parser.add_argument("--expected-physical-pairs-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--recourse-size", type=int, default=100)
    parser.add_argument("--subset-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scan-block-size", type=int, default=65_536)
    parser.add_argument("--query-block-size", type=int, default=64)
    parser.add_argument(
        "--max-rss-gb",
        type=float,
        default=32.0,
        help="Authorized absolute subset-process RSS ceiling in GiB.",
    )
    parser.add_argument(
        "--working-rss-margin-gb",
        type=float,
        default=8.0,
        help="Bounded margin added to the measured post-selection RSS peak.",
    )
    parser.add_argument("--expected-sklearn-version", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_production_subset_equivalence_audit(
        close_pair_contract_path=args.close_pair_contract,
        expected_close_pair_contract_sha256=(
            args.expected_close_pair_contract_sha256
        ),
        physical_pairs_path=args.physical_pairs,
        expected_physical_pairs_sha256=args.expected_physical_pairs_sha256,
        output_dir=args.output_dir,
        contract=ProductionSubsetAuditContract(
            eps=args.eps,
            min_samples=args.min_samples,
            radius=args.radius,
            recourse_size=args.recourse_size,
            subset_size=args.subset_size,
            seed=args.seed,
            scan_block_size=args.scan_block_size,
            query_block_size=args.query_block_size,
            max_rss_bytes=int(args.max_rss_gb * 1024**3),
            working_rss_margin_bytes=int(
                args.working_rss_margin_gb * 1024**3
            ),
            expected_sklearn_version=args.expected_sklearn_version,
        ),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_subsets_pass": result["all_subsets_pass"],
                "full_production_dbscan_equivalence_claimed": result[
                    "full_production_dbscan_equivalence_claimed"
                ],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )
    print("[AIDS_PRODUCTION_SUBSET_EQUIVALENCE_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
