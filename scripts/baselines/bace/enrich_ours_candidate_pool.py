#!/usr/bin/env python3
"""Attach processed BACE IDs to an unchanged Ours candidate sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bace_candidate_lineage import attach_bace_candidate_lineage  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--raw-pool-jsonl", required=True)
    parser.add_argument("--parent-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--expected-candidates-per-parent", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = attach_bace_candidate_lineage(
        raw_pool_jsonl=args.raw_pool_jsonl,
        parent_csv=args.parent_csv,
        output_jsonl=args.output_jsonl,
        manifest_path=args.manifest_path,
        expected_candidates_per_parent=args.expected_candidates_per_parent,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_OURS_CANDIDATE_LINEAGE_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
