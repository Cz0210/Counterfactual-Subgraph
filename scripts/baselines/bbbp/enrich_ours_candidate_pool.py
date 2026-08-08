#!/usr/bin/env python3
"""Attach processed BBBP IDs to an unchanged Ours candidate sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bbbp_candidate_lineage import attach_bbbp_candidate_lineage  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--raw-pool-jsonl", required=True)
    parser.add_argument("--parent-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--expected-candidates-per-parent", type=int, default=4)
    parser.add_argument("--candidate-source", default="chemllm_ppo")
    parser.add_argument("--candidate-source-variant", default="stable300")
    parser.add_argument("--generation-seed", type=int, default=13)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-kind", default="ppo")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_pool = Path(args.raw_pool_jsonl).expanduser().resolve()
    parent_csv = Path(args.parent_csv).expanduser().resolve()
    missing = [str(path) for path in (raw_pool, parent_csv) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"BBBP lineage inputs are missing: {missing}")
    if args.validate_only or args.dry_run:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_RUN",
                    "raw_pool_jsonl": str(raw_pool),
                    "parent_csv": str(parent_csv),
                    "candidate_source": args.candidate_source,
                    "candidate_source_variant": args.candidate_source_variant,
                    "generation_seed": args.generation_seed,
                    "planned_output_jsonl": str(Path(args.output_jsonl).expanduser()),
                    "formal_output_written": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    result = attach_bbbp_candidate_lineage(
        raw_pool_jsonl=args.raw_pool_jsonl,
        parent_csv=args.parent_csv,
        output_jsonl=args.output_jsonl,
        manifest_path=args.manifest_path,
        expected_candidates_per_parent=args.expected_candidates_per_parent,
        candidate_source=args.candidate_source,
        candidate_source_variant=args.candidate_source_variant,
        generation_seed=args.generation_seed,
        checkpoint_path=args.checkpoint_path,
        checkpoint_kind=args.checkpoint_kind,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BBBP_OURS_CANDIDATE_LINEAGE_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
