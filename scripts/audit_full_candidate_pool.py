#!/usr/bin/env python3
"""Audit a full candidate_pool.jsonl for selector-facing pool quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.full_candidate_pool_audit import (  # noqa: E402
    FullPoolAuditConfig,
    audit_full_candidate_pool,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for HPC wrapper parity. The audit uses explicit CLI paths.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept only for Slurm wrapper parity.",
    )
    parser.add_argument("--pool-jsonl", required=True, help="Path to full candidate_pool JSONL.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Full label=1 PPO prompt dataset used for coverage and parent accounting.",
    )
    parser.add_argument(
        "--teacher-path",
        required=True,
        help="Teacher/oracle path kept in metadata for audit traceability.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for audit outputs.")
    parser.add_argument("--label-col", default="label", help="Preferred label column.")
    parser.add_argument("--smiles-col", default="parent_smiles", help="Preferred smiles column.")
    parser.add_argument("--target-label", type=int, default=1, help="Parent label to audit.")
    parser.add_argument(
        "--sim-sample-size",
        type=int,
        default=5000,
        help="Maximum sampled pair count for pairwise Tanimoto diversity.",
    )
    parser.add_argument(
        "--topk-show",
        type=int,
        default=10,
        help="How many top fragments to show in frequency and coverage outputs.",
    )
    parser.add_argument(
        "--coverage-parent-limit",
        type=int,
        default=0,
        help="Optional parent cap for expensive coverage matching. 0 means all parents.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = audit_full_candidate_pool(
        pool_jsonl=args.pool_jsonl,
        dataset_path=args.dataset_path,
        teacher_path=args.teacher_path,
        out_dir=args.out_dir,
        config=FullPoolAuditConfig(
            label_col=str(args.label_col),
            smiles_col=str(args.smiles_col),
            target_label=int(args.target_label),
            sim_sample_size=int(args.sim_sample_size),
            topk_show=int(args.topk_show),
            coverage_parent_limit=int(args.coverage_parent_limit),
        ),
    )
    overall = summary["overall"]
    selector_gate = summary["selector_gate"]
    print(f"pool_jsonl: {Path(args.pool_jsonl).expanduser().resolve()}")
    print(f"dataset_path: {Path(args.dataset_path).expanduser().resolve()}")
    print(f"out_dir: {Path(args.out_dir).expanduser().resolve()}")
    print(f"num_rows: {overall['num_rows']}")
    print(f"final_substructure_rate: {overall['final_substructure_rate']}")
    print(f"cf_flip_rate: {overall['cf_flip_rate']}")
    print(f"unique_final_fragment_rate: {overall['unique_final_fragment_rate']}")
    print(f"top5_final_fragment_ratio: {overall['top5_final_fragment_ratio']}")
    print(f"mean_pairwise_tanimoto: {overall['mean_pairwise_tanimoto']}")
    print(f"ready_for_selector: {selector_gate['ready_for_selector']}")
    print(f"recommendation: {selector_gate['recommendation']}")


if __name__ == "__main__":
    main()
