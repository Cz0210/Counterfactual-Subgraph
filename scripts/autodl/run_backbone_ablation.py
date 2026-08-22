#!/usr/bin/env python3
"""Plan (but do not launch) reproducible molecular-GNN backbone ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


BACKBONES = ("gine", "gin", "gcn", "gatv2")


def _fresh_absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _existing_absolute(value: str) -> Path:
    path = _fresh_absolute(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _license_pass(path: Path | None) -> bool:
    if path is None:
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        isinstance(payload, dict)
        and payload.get("status") == "PASS"
        and payload.get("passed") is True
        and str(payload.get("license_basis") or "").strip()
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "tastemolnet" and not _license_pass(args.license_gate):
        enabled = False
        blocker = "BLOCKED_LICENSE_REVIEW"
    else:
        enabled = bool(args.enable)
        blocker = None if enabled else "NOT_STARTED_PRIMARY_RESULTS_FIRST"
    axes = {
        "dataset": args.dataset,
        "method": "frozen_molecular_classifier",
        "oracle_backend": "gnn",
        "gnn_backbone": list(args.backbone),
        "seed": args.seed,
        "candidate_pool_variant": args.candidate_pool_variant,
        "selector_variant": args.selector_variant,
        "reward_variant": args.reward_variant,
        "distance_variant": args.distance_variant,
    }
    tasks: list[dict[str, Any]] = []
    if enabled:
        for offset, backbone in enumerate(args.backbone):
            task_id = f"ablation_{args.dataset}_{backbone}_seed{args.seed}"
            output = str(
                args.output_root
                / args.dataset
                / "frozen_molecular_classifier"
                / backbone
                / f"seed-{args.seed}"
                / "attempt-{attempt}"
            )
            tasks.append(
                {
                    "id": task_id,
                    "dataset": f"{args.dataset}-gnn-ablation",
                    "stage": "GNN_BACKBONE_ABLATION",
                    "runner_dataset": f"{args.dataset}-gnn-ablation",
                    "runner_stage": "GNN_BACKBONE_ABLATION",
                    "depends_on": [],
                    "resource": "gpu",
                    "priority": 5000 + offset,
                    "data_splits": ["train", "validation"],
                    "command": [
                        "{python}",
                        "{project_root}/scripts/train_molecular_gnn.py",
                        "--config",
                        "{project_root}/configs/hpc.yaml",
                        "--config",
                        f"{{project_root}}/configs/gnn/{backbone}.yaml",
                        "--dataset",
                        args.dataset,
                        "--data-dir",
                        str(args.split_root),
                        "--output-dir",
                        "{task_output}",
                        "--profile",
                        "full",
                        "--device",
                        "cuda:0",
                        "--backbone",
                        backbone,
                        "--seed",
                        str(args.seed),
                    ],
                    "input_manifest": str(args.split_root / "split_manifest.json"),
                    "expected_output": output,
                    "required_output_files": [
                        "model.pt",
                        "model_card.json",
                        "training_metrics.json",
                        "test_evaluation_status.json",
                        "sha256sums.txt",
                    ],
                    "required_log_marker": "[MOLECULAR_GNN_TRAIN_OK]",
                    "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
                }
            )
    payload = {
        "schema_version": "gnn_backbone_ablation_plan_v1",
        "enabled": enabled,
        "blocker": blocker,
        "primary_results_required_first": True,
        "axes": axes,
        "tasks": tasks,
    }
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset", choices=("bace", "tastemolnet"), required=True)
    parser.add_argument("--backbone", action="append", choices=BACKBONES, default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-root", type=_existing_absolute, required=True)
    parser.add_argument("--output-root", type=_fresh_absolute, required=True)
    parser.add_argument("--output", type=_fresh_absolute, required=True)
    parser.add_argument("--license-gate", type=_existing_absolute)
    parser.add_argument("--candidate-pool-variant", default="primary")
    parser.add_argument("--selector-variant", default="primary")
    parser.add_argument("--reward-variant", default="primary")
    parser.add_argument("--distance-variant", default="wnode")
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    if not args.backbone:
        args.backbone = list(BACKBONES)
    result = build(args)
    print(json.dumps(result, sort_keys=True))
    print("[GNN_BACKBONE_ABLATION_PLAN_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
