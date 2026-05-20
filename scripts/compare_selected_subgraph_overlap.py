#!/usr/bin/env python3
"""Compare overlap between two selector outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.selected_subgraph_overlap import compare_selected_subgraph_overlap  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper parity.")
    parser.add_argument("--label0-selected-json", required=True, help="Selector output JSON for label=0.")
    parser.add_argument("--label1-selected-json", required=True, help="Selector output JSON for label=1.")
    parser.add_argument("--out-dir", required=True, help="Output directory for overlap artifacts.")
    parser.add_argument(
        "--sim-thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.85, 0.95],
        help="Similarity thresholds to summarize.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = compare_selected_subgraph_overlap(
        args.label0_selected_json,
        args.label1_selected_json,
        out_dir=args.out_dir,
        sim_thresholds=[float(value) for value in args.sim_thresholds],
    )
    exact = summary["exact_overlap"]
    soft = summary["soft_overlap"]
    print(f"label0_selected_json: {Path(args.label0_selected_json).expanduser().resolve()}")
    print(f"label1_selected_json: {Path(args.label1_selected_json).expanduser().resolve()}")
    print(f"out_dir: {Path(args.out_dir).expanduser().resolve()}")
    print(f"exact_intersection_count: {exact['exact_intersection_count']}")
    print(f"exact_jaccard: {exact['exact_jaccard']}")
    print(f"bidirectional_mean_max_sim: {soft['bidirectional_mean_max_sim']}")


if __name__ == "__main__":
    main()
