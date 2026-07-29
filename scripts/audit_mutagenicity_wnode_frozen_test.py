#!/usr/bin/env python3
"""Independently audit a completed frozen Mutagenicity WNode test run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.mutagenicity_wnode_frozen_test import (  # noqa: E402
    audit_frozen_test_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frozen-selector-root", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--expected-parent-count", type=int, default=217)
    parser.add_argument("--expected-candidate-count", type=int, default=20)
    parser.add_argument("--expected-pair-count", type=int, default=4340)
    parser.add_argument("--expected-top-k", type=int, default=20)
    parser.add_argument("--expected-table-k", type=int, default=10)
    parser.add_argument("--require-complete-cartesian", action="store_true")
    parser.add_argument("--require-frozen-thresholds", action="store_true")
    parser.add_argument("--require-frozen-candidate-order", action="store_true")
    parser.add_argument("--require-monotonic-coverage", action="store_true")
    parser.add_argument("--require-nonincreasing-capped-cost", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    result = audit_frozen_test_run(
        run_dir,
        frozen_selector_root=args.frozen_selector_root,
        test_csv=args.test_csv,
        expected_parent_count=int(args.expected_parent_count),
        expected_candidate_count=int(args.expected_candidate_count),
        expected_pair_count=int(args.expected_pair_count),
        expected_top_k=int(args.expected_top_k),
        expected_table_k=int(args.expected_table_k),
        require_complete_cartesian=bool(args.require_complete_cartesian),
        require_frozen_thresholds=bool(args.require_frozen_thresholds),
        require_frozen_candidate_order=bool(args.require_frozen_candidate_order),
        require_monotonic_coverage=bool(args.require_monotonic_coverage),
        require_nonincreasing_capped_cost=bool(
            args.require_nonincreasing_capped_cost
        ),
    )
    (run_dir / "frozen_test_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    print("[MUTAGENICITY_WNODE_FROZEN_TEST_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
