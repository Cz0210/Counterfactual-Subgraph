#!/usr/bin/env python3
"""Probe or hold a UUID-scoped AutoDL GPU advisory lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    GPUFileLock,
    build_runtime_layout,
    gpu_lock_available,
    resolve_project_root,
    sanitized_environment,
    select_data_root,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--project-root", type=Path)
    root.add_argument("--data-root", type=Path)
    root.add_argument("--config", action="append", default=[])
    commands = root.add_subparsers(dest="action", required=True)
    commands.add_parser("list")
    for name in ("probe", "run"):
        command = commands.add_parser(name)
        command.add_argument("--gpu-index", type=int, required=True)
        command.add_argument("--gpu-uuid", required=True)
        command.add_argument("--run-id")
        if name == "run":
            command.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        project_root = resolve_project_root(args.project_root)
        data_root = select_data_root(project_root, explicit=args.data_root)
        layout = build_runtime_layout(project_root=project_root, data_root=data_root).ensure()
        if args.action == "list":
            rows = []
            for path in sorted(layout.locks_dir.glob("gpu-*.lock")):
                try:
                    metadata = json.loads(path.read_text(encoding="utf-8") or "{}")
                except json.JSONDecodeError:
                    metadata = {"state": "CORRUPT"}
                rows.append({"path": str(path), "metadata": metadata})
            print(json.dumps({"locks": rows}, indent=2, sort_keys=True))
            return 0
        if args.action == "probe":
            available = gpu_lock_available(layout.locks_dir, args.gpu_uuid)
            print(
                json.dumps(
                    {
                        "gpu_index": args.gpu_index,
                        "gpu_uuid": args.gpu_uuid,
                        "lock_available": available,
                    },
                    sort_keys=True,
                )
            )
            return 0 if available else 3
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise AutoDLRuntimeError("gpu_lock.py run requires a command after --")
        owner = {"run_id": args.run_id, "command": command}
        with GPUFileLock(
            layout.locks_dir,
            gpu_index=args.gpu_index,
            gpu_uuid=args.gpu_uuid,
            owner=owner,
        ):
            environment = sanitized_environment()
            environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
            environment["AUTODL_PHYSICAL_GPU_INDEX"] = str(args.gpu_index)
            environment["AUTODL_PHYSICAL_GPU_UUID"] = str(args.gpu_uuid)
            completed = subprocess.run(command, env=environment, check=False)
            return int(completed.returncode)
    except AutoDLRuntimeError as exc:
        print(f"AUTODL_GPU_LOCK_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
