#!/usr/bin/env python3
"""Evaluate ours and GT-FullGraph CCRCov with GREED-predicted normalized GED."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.ccrcov_distance_eval import GreedDistanceProvider, evaluate_ccrcov_with_distance  # noqa: E402


def _thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--ours-selected-path", required=True)
    parser.add_argument("--gt-fullgraph-candidates-path", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--greed-checkpoint", required=True)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--thresholds", default="0.05,0.10,0.20")
    parser.add_argument("--output-root", default="outputs/hpc/eval/ccrcov_greed_hiv")
    parser.add_argument("--max-parents", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--require-flip-only", action="store_true")
    parser.add_argument("--min-cf-drop", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--partial-every", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = GreedDistanceProvider(args.greed_checkpoint, device=args.device)
    evaluate_ccrcov_with_distance(
        dataset_csv=args.dataset_csv,
        ours_selected_path=args.ours_selected_path,
        gt_fullgraph_candidates_path=args.gt_fullgraph_candidates_path,
        teacher_path=args.teacher_path,
        provider=provider,
        distance_type="ged",
        distance_name="greed_ged",
        label=args.label,
        thresholds=_thresholds(args.thresholds),
        output_root=args.output_root,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        max_parents=args.max_parents,
        max_candidates=args.max_candidates,
        require_flip_only=args.require_flip_only,
        min_cf_drop=args.min_cf_drop,
        partial_every=args.partial_every,
    )


if __name__ == "__main__":
    main()
