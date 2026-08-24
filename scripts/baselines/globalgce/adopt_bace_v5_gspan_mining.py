#!/usr/bin/env python3
"""Build a deep, read-only adoption proof for completed BACE v5 gSpan mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_mining_adoption import (  # noqa: E402
    EXPECTED_OFFICIAL_COMMIT,
    build_globalgce_gspan_adoption,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--source-task-state", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--native-train-csv", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--gine-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-official-commit", default=EXPECTED_OFFICIAL_COMMIT
    )
    parser.add_argument("--expected-pattern-count", type=int, default=5_441_858)
    parser.add_argument("--expected-root-count", type=int, default=19)
    parser.add_argument("--min-freq", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_globalgce_gspan_adoption(
        source_run_manifest=args.source_run_manifest,
        source_task_state=args.source_task_state,
        source_checkpoint=args.source_checkpoint,
        source_sqlite=args.source_sqlite,
        official_root=args.official_root,
        native_train_csv=args.native_train_csv,
        source_manifest=args.source_manifest,
        gine_checkpoint=args.gine_checkpoint,
        output_dir=args.output_dir,
        expected_official_commit=args.expected_official_commit,
        expected_pattern_count=args.expected_pattern_count,
        expected_root_count=args.expected_root_count,
        min_freq=args.min_freq,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("[BACE_GLOBALGCE_V5_GSPAN_ADOPTION_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
