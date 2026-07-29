#!/usr/bin/env python3
"""Audit a Mutagenicity WNode-aware calibration prefix selector run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.mutagenicity_wnode_selector import (  # noqa: E402
    audit_mutagenicity_wnode_selector,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--matrix-run-dir", required=True)
    parser.add_argument("--expected-parent-count", type=int, default=0)
    parser.add_argument("--expected-candidate-count", type=int, default=0)
    parser.add_argument("--expected-top-k", type=int, default=20)
    parser.add_argument("--expected-table-k", type=int, default=10)
    parser.add_argument("--require-all-variants", action="store_true")
    parser.add_argument("--require-nested-prefix", action="store_true")
    parser.add_argument("--require-monotonic-coverage", action="store_true")
    parser.add_argument(
        "--require-nonincreasing-capped-cost",
        action="store_true",
    )
    parser.add_argument("--forbid-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_mutagenicity_wnode_selector(
        run_dir=args.run_dir,
        matrix_run_dir=args.matrix_run_dir,
        expected_parent_count=int(args.expected_parent_count),
        expected_candidate_count=int(args.expected_candidate_count),
        expected_top_k=int(args.expected_top_k),
        expected_table_k=int(args.expected_table_k),
        require_all_variants=bool(args.require_all_variants),
        require_nested_prefix=bool(args.require_nested_prefix),
        require_monotonic_coverage=bool(args.require_monotonic_coverage),
        require_nonincreasing_capped_cost=bool(
            args.require_nonincreasing_capped_cost
        ),
        forbid_test=bool(args.forbid_test),
    )
    run_dir = Path(args.run_dir).expanduser().resolve()
    (run_dir / "selector_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    print("[MUTAGENICITY_WNODE_PREFIX_SELECTOR_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
