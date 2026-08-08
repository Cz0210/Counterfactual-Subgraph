#!/usr/bin/env python3
"""Train the official GCFExplainer GNN on frozen BACE train/validation cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_bace_runtime import train_bace_official_gnn  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-limit", type=int, required=True)
    parser.add_argument("--val-limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    result = train_bace_official_gnn(
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
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GCFEXPLAINER_GNN_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
