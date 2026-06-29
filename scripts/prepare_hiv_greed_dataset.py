#!/usr/bin/env python3
"""Prepare HIV graphs.jsonl for GREED-style distance training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.greed_distance.graph_conversion import prepare_hiv_graph_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument(
        "--dataset-csv",
        default="outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv",
    )
    parser.add_argument("--output-jsonl", default="outputs/hpc/greed_hiv/dataset/graphs.jsonl")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_hiv_graph_dataset(
        dataset_csv=args.dataset_csv,
        output_jsonl=args.output_jsonl,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        label=args.label,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
