#!/usr/bin/env python3
"""Probe or hold a UUID-scoped AutoDL GPU advisory lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
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
            command.add_argument("--llm-dispatch-spec", type=Path)
            command.add_argument("--llm-dispatch-spec-sha256")
            command.add_argument("--owner-output-root", type=Path)
            command.add_argument("--wait-seconds", type=float, default=86400)
            command.add_argument("--refresh-seconds", type=float, default=30)
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
        if args.llm_dispatch_spec:
            if args.command or not args.llm_dispatch_spec_sha256 or not args.owner_output_root:
                raise AutoDLRuntimeError("LLM dispatch requires sealed SHA, fresh owner root, and no injected command")
            from src.ablations.llm.bace_native_runtime import verified_file
            from src.ablations.llm.contracts import canonical_json_sha256
            from src.ablations.llm.existing_gpu_owner import DISPATCH_SCHEMA, ResourceSampler, run_owned_child
            spec = json.loads(verified_file({"path": str(args.llm_dispatch_spec), "sha256": args.llm_dispatch_spec_sha256}).read_text())
            body = {k: v for k, v in spec.items() if k != "self_sha256"}
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
            expected_entry = str(PROJECT_ROOT / "scripts/ablations/llm/run_bace_llm_successor.py")
            if (spec.get("schema_version") != DISPATCH_SCHEMA or spec.get("self_sha256") != canonical_json_sha256(body)
                    or spec.get("execution_commit") != commit or spec.get("max_llm_gpus") != 1
                    or spec.get("borrow_enabled") is not False or spec["command"][1:4] != ["-I", "-B", expected_entry]):
                raise AutoDLRuntimeError("LLM dispatch binding/commit/entrypoint differs")
            config = json.loads(verified_file(spec["resource_config"]).read_text())
            if Path(config["gpu_lock_root"]).resolve() != layout.locks_dir.resolve():
                raise AutoDLRuntimeError("LLM must use the existing AutoDL project UUID lock root")
            sampler = ResourceSampler(config, args.gpu_index, args.gpu_uuid)
            return run_owned_child(command=spec["command"], environment=sanitized_environment(), sampler=sampler,
                       output_root=args.owner_output_root, lock_root=layout.locks_dir, run_id=args.run_id,
                       interval=args.refresh_seconds, max_wait_seconds=args.wait_seconds)
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
            environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_uuid)
            environment["AUTODL_PHYSICAL_GPU_INDEX"] = str(args.gpu_index)
            environment["AUTODL_PHYSICAL_GPU_UUID"] = str(args.gpu_uuid)
            completed = subprocess.run(command, env=environment, check=False)
            return int(completed.returncode)
    except (AutoDLRuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"AUTODL_GPU_LOCK_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
