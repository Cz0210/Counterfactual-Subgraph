#!/usr/bin/env python3
"""Run official VRRW on the strict Mutagenicity train-source cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_adapter import write_failure_artifacts  # noqa: E402
from src.baselines.gcfexplainer_mutagenicity_runtime import run_official_vrrw  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--neurosed-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--parent-limit", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=0.05)
    parser.add_argument("--teleport", type=float, default=0.1)
    parser.add_argument("--candidate-capacity", type=int, default=100000)
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device1", default="cuda:0")
    parser.add_argument("--device2", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("VRRW requires --forbid-calibration-test.")
    try:
        summary = run_official_vrrw(
            dataset_dir=args.dataset_dir,
            official_root=args.official_root,
            gnn_checkpoint=args.gnn_checkpoint,
            neurosed_checkpoint=args.neurosed_checkpoint,
            output_dir=args.output_dir,
            profile=args.profile,
            parent_limit=args.parent_limit,
            max_steps=args.max_steps,
            alpha=args.alpha,
            theta=args.theta,
            teleport=args.teleport,
            candidate_capacity=args.candidate_capacity,
            sample=args.sample,
            sample_size=args.sample_size,
            seed=args.seed,
            device1=args.device1,
            device2=args.device2,
            resume=args.resume,
        )
    except Exception as exc:
        write_failure_artifacts(args.output_dir, error=exc, resolved_config=vars(args))
        raise
    marker = (
        "[MUTAGENICITY_GCFEXPLAINER_VRRW_SMOKE_OK]"
        if args.profile == "smoke"
        else "[MUTAGENICITY_GCFEXPLAINER_VRRW_FULL_OK]"
    )
    print(marker, flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
