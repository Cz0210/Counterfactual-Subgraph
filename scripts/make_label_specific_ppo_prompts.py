#!/usr/bin/env python3
"""Build one label-specific PPO prompt CSV with explicit label-conditioned prompts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.unified_ppo_prompts import (  # noqa: E402
    PromptBuildConfig,
    build_label_specific_prompt_rows,
    write_prompt_csv_and_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper parity.")
    parser.add_argument("--input-csv", required=True, help="Input source CSV containing both labels.")
    parser.add_argument("--label", required=True, type=int, choices=(0, 1), help="Target label to keep.")
    parser.add_argument("--out-csv", required=True, help="Output label-specific PPO prompt CSV.")
    parser.add_argument("--out-json", required=True, help="Output summary JSON.")
    parser.add_argument("--label-col", default="label", help="Preferred label column.")
    parser.add_argument("--smiles-col", default="parent_smiles", help="Preferred SMILES column.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    header, rows, summary = build_label_specific_prompt_rows(
        args.input_csv,
        config=PromptBuildConfig(
            label_col=str(args.label_col),
            smiles_col=str(args.smiles_col),
            label=int(args.label),
        ),
    )
    write_prompt_csv_and_summary(
        header=header,
        rows=rows,
        summary=summary,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )
    print(f"input_csv: {Path(args.input_csv).expanduser().resolve()}")
    print(f"out_csv: {Path(args.out_csv).expanduser().resolve()}")
    print(f"out_json: {Path(args.out_json).expanduser().resolve()}")
    print(f"target_label: {args.label}")
    print(f"kept_count: {summary['kept_count']}")


if __name__ == "__main__":
    main()
