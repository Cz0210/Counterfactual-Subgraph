#!/usr/bin/env python3
"""Build one balanced unified label01 PPO prompt CSV from separate label CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.unified_ppo_prompts import (  # noqa: E402
    UnifiedPromptBuildConfig,
    build_unified_prompt_rows,
    write_prompt_csv_and_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper parity.")
    parser.add_argument("--label0-csv", required=True, help="Label=0 PPO prompt CSV.")
    parser.add_argument("--label1-csv", required=True, help="Label=1 PPO prompt CSV.")
    parser.add_argument("--out-csv", required=True, help="Output unified PPO prompt CSV.")
    parser.add_argument("--out-json", required=True, help="Output summary JSON.")
    parser.add_argument(
        "--balance-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance the two labels before interleaving.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Shuffle seed.")
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=0,
        help="Optional maximum number of rows per label before interleaving. 0 keeps the balanced default.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    header, rows, summary = build_unified_prompt_rows(
        args.label0_csv,
        args.label1_csv,
        config=UnifiedPromptBuildConfig(
            balance_labels=bool(args.balance_labels),
            seed=int(args.seed),
            max_per_label=int(args.max_per_label),
        ),
    )
    write_prompt_csv_and_summary(
        header=header,
        rows=rows,
        summary=summary,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )
    print(f"label0_csv: {Path(args.label0_csv).expanduser().resolve()}")
    print(f"label1_csv: {Path(args.label1_csv).expanduser().resolve()}")
    print(f"out_csv: {Path(args.out_csv).expanduser().resolve()}")
    print(f"out_json: {Path(args.out_json).expanduser().resolve()}")
    print(f"num_rows: {summary['num_rows']}")
    print(f"selected_label0_count: {summary['selected_label0_count']}")
    print(f"selected_label1_count: {summary['selected_label1_count']}")


if __name__ == "__main__":
    main()
