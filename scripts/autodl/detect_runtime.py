#!/usr/bin/env python3
"""Detect the local AutoDL runtime without modifying Python packages."""

from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    build_runtime_layout,
    resolve_project_root,
    select_data_root,
    utc_now,
)


DEPENDENCIES = {
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "torch_scatter": "torch-scatter",
    "torch_sparse": "torch-sparse",
    "rdkit": "rdkit",
    "transformers": "transformers",
    "trl": "trl",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
}


def detect_dependency(module_name: str, distribution_name: str) -> dict[str, Any]:
    try:
        import_module(module_name)
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        package_version = version(distribution_name)
    except PackageNotFoundError:
        package_version = "UNKNOWN"
    return {"available": True, "version": package_version, "error": None}


def git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(project_root: Path, data_root: Path, *, prepare: bool) -> dict[str, Any]:
    layout = build_runtime_layout(project_root=project_root, data_root=data_root)
    if prepare:
        layout.ensure()
    disk = shutil.disk_usage(data_root)
    dependencies = {
        name: detect_dependency(name, distribution)
        for name, distribution in DEPENDENCIES.items()
    }
    torch_info: dict[str, Any] = {
        "cuda_available": None,
        "cuda_device_count": None,
        "torch_cuda_version": None,
    }
    if dependencies["torch"]["available"]:
        import torch

        torch_info = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "torch_cuda_version": torch.version.cuda,
        }
    return {
        "schema_version": 1,
        "detected_at": utc_now(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "git": {
            "branch": git_value(project_root, "branch", "--show-current"),
            "commit": git_value(project_root, "rev-parse", "HEAD"),
        },
        "layout": layout.as_json(),
        "data_root_disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "launchers": {
            "tmux": shutil.which("tmux"),
            "nohup": shutil.which("nohup"),
            "nvidia_smi": shutil.which("nvidia-smi"),
        },
        "dependencies": dependencies,
        "torch_runtime": torch_info,
        "prepared": prepare,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for CLI/Slurm parity; runtime detection does not load training config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project_root = resolve_project_root(args.project_root)
        data_root = select_data_root(
            project_root,
            explicit=args.data_root,
            min_free_bytes=max(1, int(args.min_free_gb * 1024**3)),
        )
        report = build_report(project_root, data_root, prepare=args.prepare)
    except AutoDLRuntimeError as exc:
        print(f"AUTODL_RUNTIME_DETECTION_FAILED: {exc}", file=sys.stderr)
        return 2
    body = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
