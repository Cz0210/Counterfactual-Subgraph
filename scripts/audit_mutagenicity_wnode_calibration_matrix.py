#!/usr/bin/env python3
"""Audit a completed Mutagenicity WNode calibration action matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.mutagenicity_wnode_matrix import (  # noqa: E402
    audit_calibration_matrix_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-parent-count", type=int, default=0)
    parser.add_argument("--expected-candidate-count", type=int, default=0)
    parser.add_argument("--expected-pair-count", type=int, default=0)
    parser.add_argument("--expected-source-eligible-rows", type=int, default=0)
    parser.add_argument(
        "--expected-source-eligible-raw-unique",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--require-complete-cartesian",
        action="store_true",
    )
    parser.add_argument("--require-strict-flip-pair", action="store_true")
    parser.add_argument("--forbid-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    audit = audit_calibration_matrix_run(
        run_dir,
        expected_parent_count=int(args.expected_parent_count),
        expected_candidate_count=int(args.expected_candidate_count),
        expected_pair_count=int(args.expected_pair_count),
        expected_source_eligible_rows=int(args.expected_source_eligible_rows),
        expected_source_eligible_raw_unique=int(
            args.expected_source_eligible_raw_unique
        ),
        require_complete_cartesian=bool(args.require_complete_cartesian),
        require_strict_flip_pair=bool(args.require_strict_flip_pair),
        forbid_test=bool(args.forbid_test),
    )
    (run_dir / "matrix_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("[MUTAGENICITY_WNODE_CALIBRATION_MATRIX_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
