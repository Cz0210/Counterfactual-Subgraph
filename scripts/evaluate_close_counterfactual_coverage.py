#!/usr/bin/env python3
"""Evaluate GCFExplainer-style close counterfactual coverage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.close_counterfactual_coverage import (  # noqa: E402
    evaluate_gcf_counterfactual_graphs,
    evaluate_ours_selected_subgraphs,
)

DEFAULT_GED_THRESHOLDS = "0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.20"
DEFAULT_EMBEDDING_THRESHOLDS = "0.02,0.05,0.10,0.15,0.20,0.25,0.30"


def _parse_thresholds(value: str) -> list[float]:
    thresholds: list[float] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if not text:
            continue
        thresholds.append(float(text))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional config path kept for HPC wrapper parity.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept for Slurm wrapper parity.",
    )
    parser.add_argument("--mode", choices=["ours", "gcf"], required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--selected-subgraphs-path", default="")
    parser.add_argument("--gcf-candidates-path", default="")
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--distance-type", choices=["ged", "embedding"], required=True)
    parser.add_argument("--ged-mode", choices=["delete", "networkx"], default="delete")
    parser.add_argument("--thresholds", default=DEFAULT_GED_THRESHOLDS)
    parser.add_argument("--embedding-thresholds", default=DEFAULT_EMBEDDING_THRESHOLDS)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-flip-only", action="store_true")
    parser.add_argument("--min-cf-drop", type=float, default=0.0)
    parser.add_argument("--desired-label", type=int, default=None)
    parser.add_argument("--max-parents", type=int, default=None)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Reserved for future multiprocessing; current implementation is deterministic serial evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    thresholds = (
        _parse_thresholds(args.thresholds)
        if args.distance_type == "ged"
        else _parse_thresholds(args.embedding_thresholds)
    )
    print(
        f"[CLOSE_CF_CONFIG] mode={args.mode} distance_type={args.distance_type} "
        f"thresholds={thresholds} num_workers={args.num_workers}"
    )
    if int(args.num_workers) != 1:
        print("[CLOSE_CF_CONFIG] num_workers is currently accepted for CLI stability; evaluation runs serially.")

    if args.mode == "ours":
        if not args.selected_subgraphs_path:
            raise SystemExit("--selected-subgraphs-path is required when --mode ours")
        result = evaluate_ours_selected_subgraphs(
            dataset_csv=args.dataset_csv,
            selected_subgraphs_path=args.selected_subgraphs_path,
            teacher_path=args.teacher_path,
            label=int(args.label),
            distance_type=str(args.distance_type),
            thresholds=thresholds,
            output_dir=args.output_dir,
            smiles_col=str(args.smiles_col),
            label_col=str(args.label_col),
            ged_mode=str(args.ged_mode),
            require_flip_only=bool(args.require_flip_only),
            min_cf_drop=float(args.min_cf_drop),
            max_parents=args.max_parents,
            cache_path=args.cache_path,
        )
    else:
        if not args.gcf_candidates_path:
            raise SystemExit("--gcf-candidates-path is required when --mode gcf")
        result = evaluate_gcf_counterfactual_graphs(
            dataset_csv=args.dataset_csv,
            gcf_candidates_path=args.gcf_candidates_path,
            teacher_path=args.teacher_path,
            label=int(args.label),
            distance_type=str(args.distance_type),
            thresholds=thresholds,
            output_dir=args.output_dir,
            smiles_col=str(args.smiles_col),
            label_col=str(args.label_col),
            desired_label=args.desired_label,
            require_flip_only=bool(args.require_flip_only),
            min_cf_drop=float(args.min_cf_drop),
            max_parents=args.max_parents,
            cache_path=args.cache_path,
            ged_mode=str(args.ged_mode),
        )
    outputs = result["outputs"]
    print(f"details_csv: {outputs['details_csv']}")
    print(f"threshold_summary_csv: {outputs['threshold_summary_csv']}")
    print(f"threshold_summary_json: {outputs['threshold_summary_json']}")
    print(f"report_md: {outputs['report_md']}")


if __name__ == "__main__":
    main()
