#!/usr/bin/env python3
"""Train the official GCFExplainer GNN on strict Mutagenicity train/val."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_adapter import write_failure_artifacts  # noqa: E402
from src.baselines.gcfexplainer_mutagenicity_runtime import train_official_gnn  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--val-limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("GNN training requires --forbid-calibration-test.")
    try:
        summary = train_official_gnn(
            dataset_dir=args.dataset_dir,
            official_root=args.official_root,
            output_dir=args.output_dir,
            profile=args.profile,
            epochs=args.epochs,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
        )
    except Exception as exc:
        write_failure_artifacts(args.output_dir, error=exc, resolved_config=vars(args))
        raise
    marker = (
        "[MUTAGENICITY_GCFEXPLAINER_GNN_SMOKE_OK]"
        if args.profile == "smoke"
        else "[MUTAGENICITY_GCFEXPLAINER_GNN_FULL_OK]"
    )
    print(marker, flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
