#!/usr/bin/env python3
"""Train the GREED-style HIV graph distance model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.greed_distance.train import train_greed_distance_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--pairs-dir", default="outputs/hpc/greed_hiv/pairs")
    parser.add_argument("--checkpoint-path", default="outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt")
    parser.add_argument("--train-metrics-json", default="outputs/hpc/greed_hiv/reports/train_metrics.json")
    parser.add_argument("--test-metrics-csv", default="outputs/hpc/greed_hiv/reports/test_metrics.csv")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pair_dir = Path(args.pairs_dir).expanduser().resolve()
    summary = train_greed_distance_model(
        train_pairs_csv=pair_dir / "train_pairs_labeled.csv",
        val_pairs_csv=pair_dir / "val_pairs_labeled.csv",
        test_pairs_csv=pair_dir / "test_pairs_labeled.csv",
        checkpoint_path=args.checkpoint_path,
        train_metrics_json=args.train_metrics_json,
        test_metrics_csv=args.test_metrics_csv,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
