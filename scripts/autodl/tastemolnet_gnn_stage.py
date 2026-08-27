#!/usr/bin/env python3
"""Run immutable Taste T3 adoption or calibration-cache-only T4 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from src.eval.tastemolnet_gnn_stages import (
    T3_MARKER,
    T4_MARKER,
    run_t3_existing_fit_adoption,
    run_t4_calibration_cache_smoke,
    verify_stage_output,
)

PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
HPC_CONFIG = PROJECT_ROOT / "configs/hpc.yaml"
HPC_CONFIG_SHA256 = "7d3fb9e5c42101ae4a2ee5c43f400710fad6227014c573b1550872c7005e0110"


def _validate_configs(values: list[str]) -> None:
    if len(values) != 1:
        raise ValueError("exactly one --config configs/hpc.yaml is required")
    path = Path(os.path.abspath(Path(values[0]).expanduser()))
    if path != HPC_CONFIG:
        raise ValueError("--config must be the tracked configs/hpc.yaml")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("--config may not contain symlink components")
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("--config must be one physical single-link file")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != HPC_CONFIG_SHA256:
        raise ValueError("tracked configs/hpc.yaml SHA-256 changed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    commands = parser.add_subparsers(dest="action", required=True)

    t3 = commands.add_parser(
        "t3-adopt", help="Adopt T2's existing validation-fitted temperature"
    )
    t3.add_argument("--checkpoint-dir", required=True)
    t3.add_argument("--graph-cache-root", required=True)
    t3.add_argument("--artifact-root", required=True)
    t3.add_argument("--output-dir", required=True)
    t3.add_argument("--downstream-policy", required=True)
    t3.add_argument("--base-policy", required=True)

    t4 = commands.add_parser(
        "t4-oracle-smoke", help="Run the bounded calibration-cache-only smoke"
    )
    t4.add_argument("--checkpoint-dir", required=True)
    t4.add_argument("--t3-gate", required=True)
    t4.add_argument("--graph-cache-root", required=True)
    t4.add_argument("--artifact-root", required=True)
    t4.add_argument("--output-dir", required=True)
    t4.add_argument("--downstream-policy", required=True)
    t4.add_argument("--base-policy", required=True)
    t4.add_argument("--physical-gpu-index", type=int, default=1)
    t4.add_argument("--gpu-uuid", required=True)
    t4.add_argument("--device", default="cuda:0")
    t4.add_argument("--batch-size", type=int, default=32)
    t4.add_argument("--source-count", type=int, default=16)
    t4.add_argument("--max-deletions-per-parent", type=int, default=4)

    verify = commands.add_parser("verify-output")
    verify.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_configs(args.config)
    if args.action == "t3-adopt":
        result = run_t3_existing_fit_adoption(
            checkpoint_dir=args.checkpoint_dir,
            graph_cache_root=args.graph_cache_root,
            artifact_root=args.artifact_root,
            output_dir=args.output_dir,
            downstream_policy_path=args.downstream_policy,
            base_policy_path=args.base_policy,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        print(f"[{T3_MARKER}]", flush=True)
        return 0
    if args.action == "t4-oracle-smoke":
        result = run_t4_calibration_cache_smoke(
            checkpoint_dir=args.checkpoint_dir,
            t3_gate_path=args.t3_gate,
            graph_cache_root=args.graph_cache_root,
            artifact_root=args.artifact_root,
            output_dir=args.output_dir,
            downstream_policy_path=args.downstream_policy,
            base_policy_path=args.base_policy,
            gpu_uuid=args.gpu_uuid,
            physical_gpu_index=args.physical_gpu_index,
            device=args.device,
            batch_size=args.batch_size,
            source_count=args.source_count,
            max_deletions_per_parent=args.max_deletions_per_parent,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        print(f"[{T4_MARKER}]", flush=True)
        return 0
    if args.action == "verify-output":
        print(json.dumps(verify_stage_output(args.output_dir), sort_keys=True), flush=True)
        return 0
    raise ValueError(f"unsupported action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
