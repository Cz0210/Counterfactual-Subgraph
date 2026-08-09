#!/usr/bin/env python3
"""Audit a completed BACE calibration-only WNode action matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.wnode_action_matrix import audit_bace_action_matrix  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-parent-count", type=int, default=0)
    parser.add_argument("--expected-candidate-count", type=int, default=0)
    parser.add_argument("--require-strict-flip-pair", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_bace_action_matrix(
        args.run_dir,
        expected_parent_count=int(args.expected_parent_count),
        expected_candidate_count=int(args.expected_candidate_count),
        require_strict_flip_pair=bool(args.require_strict_flip_pair),
    )
    destination = Path(args.run_dir).expanduser().resolve() / "matrix_audit.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[WNODE_ACTION_MATRIX_AUDIT_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
