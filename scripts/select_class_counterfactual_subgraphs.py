#!/usr/bin/env python3
"""Select a class-level low-redundancy counterfactual fragment set from a candidate pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.class_counterfactual_selector import (  # noqa: E402
    SelectorConfig,
    select_class_counterfactual_subgraphs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for HPC wrapper parity. The selector uses explicit CLI values.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept only for Slurm wrapper parity.",
    )
    parser.add_argument("--pool-jsonl", required=True, help="Candidate pool JSONL path.")
    parser.add_argument("--out-dir", required=True, help="Directory for selector artifacts.")
    parser.add_argument("--label", type=int, required=True, help="Target label to select from.")
    parser.add_argument("--top-k", type=int, default=20, help="How many fragments to select.")
    parser.add_argument("--alpha-cf", type=float, default=1.0, help="Weight for CF quality.")
    parser.add_argument(
        "--beta-coverage",
        type=float,
        default=1.0,
        help="Weight for parent coverage gain.",
    )
    parser.add_argument(
        "--gamma-redundancy",
        type=float,
        default=0.7,
        help="Penalty weight for maximum similarity to already selected fragments.",
    )
    parser.add_argument(
        "--eta-size",
        type=float,
        default=0.3,
        help="Penalty weight for atom-ratio size mismatch.",
    )
    parser.add_argument(
        "--min-cf-drop",
        type=float,
        default=0.2,
        help="Minimum per-candidate counterfactual drop to keep.",
    )
    parser.add_argument(
        "--require-cf-flip",
        action="store_true",
        help="Require each retained candidate to flip the teacher/oracle label.",
    )
    parser.add_argument(
        "--require-final-substructure",
        action="store_true",
        help="Require final_substructure=True for retained candidates.",
    )
    parser.add_argument(
        "--max-projection-used-rate",
        type=float,
        default=1.0,
        help="Drop aggregated fragments whose projection_used_rate exceeds this threshold.",
    )
    parser.add_argument(
        "--sim-metric",
        default="morgan",
        choices=["morgan"],
        help="Redundancy similarity metric.",
    )
    parser.add_argument(
        "--top-candidates-per-fragment",
        type=int,
        default=3,
        help="How many representative rows to preserve per selected fragment.",
    )
    parser.add_argument(
        "--dedup-by-final-fragment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Within each fragment, keep only the best row per parent before aggregation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = select_class_counterfactual_subgraphs(
        args.pool_jsonl,
        out_dir=args.out_dir,
        config=SelectorConfig(
            label=int(args.label),
            top_k=int(args.top_k),
            alpha_cf=float(args.alpha_cf),
            beta_coverage=float(args.beta_coverage),
            gamma_redundancy=float(args.gamma_redundancy),
            eta_size=float(args.eta_size),
            min_cf_drop=float(args.min_cf_drop),
            require_cf_flip=bool(args.require_cf_flip),
            require_final_substructure=bool(args.require_final_substructure),
            max_projection_used_rate=float(args.max_projection_used_rate),
            sim_metric=str(args.sim_metric),
            top_candidates_per_fragment=int(args.top_candidates_per_fragment),
            dedup_by_final_fragment=bool(args.dedup_by_final_fragment),
        ),
    )
    summary = result["summary"]
    outputs = result["outputs"]
    print(f"pool_jsonl: {Path(args.pool_jsonl).expanduser().resolve()}")
    print(f"out_dir: {Path(args.out_dir).expanduser().resolve()}")
    print(f"selected_count: {summary['selected_count']}")
    print(f"final_cumulative_coverage: {summary['final_cumulative_coverage']}")
    print(f"selected_mean_cf_drop: {summary['selected_mean_cf_drop']}")
    print(f"selected_pairwise_tanimoto_mean: {summary['selected_pairwise_tanimoto_mean']}")
    print(f"report_txt: {outputs['report_txt']}")


if __name__ == "__main__":
    main()
