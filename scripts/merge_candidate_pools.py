#!/usr/bin/env python3
"""Merge multiple candidate pools and keep the best row per dedup key."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.candidate_pool_merge import MergeConfig, merge_candidate_pools  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for HPC wrapper parity. The merge script uses explicit CLI values.",
    )
    parser.add_argument(
        "--pool-jsonl",
        action="append",
        required=True,
        help="Candidate pool JSONL path. Pass multiple times to merge multiple pools.",
    )
    parser.add_argument("--out-jsonl", required=True, help="Output merged candidate_pool.jsonl path.")
    parser.add_argument(
        "--out-summary-json",
        default="",
        help="Optional output merge_summary.json path. Defaults next to out-jsonl.",
    )
    parser.add_argument(
        "--dedup-key",
        default="final_fragment,parent_smiles",
        help="Comma-separated dedup key fields.",
    )
    parser.add_argument(
        "--keep-best-by",
        default="reward_total",
        help="Primary metric used to keep the best duplicate row.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = merge_candidate_pools(
        pool_jsonls=list(args.pool_jsonl),
        out_jsonl=args.out_jsonl,
        out_summary_json=args.out_summary_json or None,
        config=MergeConfig(
            dedup_key=tuple(part.strip() for part in str(args.dedup_key).split(",") if part.strip()),
            keep_best_by=str(args.keep_best_by),
        ),
    )
    print(f"out_jsonl: {Path(args.out_jsonl).expanduser().resolve()}")
    print(f"merged_count_after_dedup: {summary['merged_count_after_dedup']}")
    print(f"dedup_removed_count: {summary['dedup_removed_count']}")
    print(f"unique_parent_count: {summary['unique_parent_count']}")
    print(f"unique_final_fragment_count: {summary['unique_final_fragment_count']}")


if __name__ == "__main__":
    main()
