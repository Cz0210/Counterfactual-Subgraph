#!/usr/bin/env python3
"""Train one managed-worker TasteMolNet NeuroSED scientific bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.tastemolnet_neurosed import (  # noqa: E402
    TasteNeuroSEDTrainConfig,
    train_tastemolnet_neurosed,
)


def _read_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - AutoDL dependency gate.
        raise RuntimeError("Taste NeuroSED config requires PyYAML") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("training"), dict):
        raise ValueError("Taste NeuroSED config lacks a training mapping")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument(
        "--neurosed-config",
        default="configs/autodl/tastemolnet_neurosed_v1.yaml",
    )
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--preparation-split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution-git-commit", required=True)
    parser.add_argument("--execution-git-tree", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root or os.environ.get("MANAGED_ARTIFACT_ROOT")
    if not output_root:
        raise ValueError("--output-root or MANAGED_ARTIFACT_ROOT is required")
    raw = _read_config(Path(args.neurosed_config))
    training = dict(raw["training"])
    config = TasteNeuroSEDTrainConfig(
        seed=int(training["seed"]),
        train_pairs=int(training["train_pairs"]),
        validation_pairs=int(training["validation_pairs"]),
        batch_size=int(training["batch_size"]),
        max_epochs=int(training["max_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        cyclic_step_size_up=int(training["cyclic_step_size_up"]),
        cyclic_step_size_down=int(training["cyclic_step_size_down"]),
        max_grad_norm=float(training["max_grad_norm"]),
        num_workers=int(training["num_workers"]),
        require_cuda_health_gate=bool(training["require_cuda_health_gate"]),
    )
    result = train_tastemolnet_neurosed(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        preparation_split_manifest=args.preparation_split_manifest,
        output_root=Path(output_root),
        execution_git_commit=args.execution_git_commit,
        execution_git_tree=args.execution_git_tree,
        source_execution_config_sha256=hashlib.sha256(
            Path(args.neurosed_config).read_bytes()
        ).hexdigest(),
        config=config,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[TASTE_GCF_NEUROSED_WORKER_ARTIFACT_READY_FOR_SEAL]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
