#!/usr/bin/env python3
"""Read-only status view for AutoDL frozen-GNN runs and BACE B0--B14."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    BACE_STAGES,
    build_runtime_layout,
    latest_registry_events,
    query_gpu_inventory,
    read_bace_stage,
    read_registry,
    resolve_project_root,
    select_data_root,
    utc_now,
)


def collect_status(layout: Any, *, include_gpu: bool, limit: int) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for stage in BACE_STAGES:
        try:
            documents = read_bace_stage(layout, stage)
            state = documents["state"]
            gate = documents["gate"]
            stages.append(
                {
                    "stage": stage,
                    "state": state.get("state"),
                    "gate": gate.get("status"),
                    "run_id": state.get("run_id"),
                    "pid": state.get("pid"),
                    "child_pid": state.get("child_pid"),
                    "tmux_session": state.get("tmux_session"),
                    "gpu_index": state.get("gpu_index"),
                    "gpu_uuid": state.get("gpu_uuid"),
                    "updated_at": state.get("updated_at"),
                    "failures": state.get("failures", []),
                }
            )
        except AutoDLRuntimeError:
            stages.append(
                {
                    "stage": stage,
                    "state": "NOT_INITIALIZED",
                    "gate": "NOT_EVALUATED",
                }
            )
    runs = latest_registry_events(read_registry(layout.registry_path))[:limit]
    result: dict[str, Any] = {
        "schema_version": 1,
        "refreshed_at": utc_now(),
        "registry_path": str(layout.registry_path),
        "bace_stages": stages,
        "runs": runs,
    }
    if include_gpu:
        try:
            result["gpus"] = [gpu.as_json() for gpu in query_gpu_inventory()]
            result["gpu_error"] = None
        except AutoDLRuntimeError as exc:
            result["gpus"] = []
            result["gpu_error"] = str(exc)
    return result


def render_table(payload: dict[str, Any]) -> str:
    lines = [
        f"AutoDL frozen-GNN status @ {payload['refreshed_at']}",
        "",
        "BACE STAGES",
        f"{'STAGE':31} {'STATE':16} {'GATE':16} {'GPU':6} {'PID':10} RUN_ID",
    ]
    for row in payload["bace_stages"]:
        lines.append(
            f"{str(row.get('stage', '-')):31} "
            f"{str(row.get('state', '-')):16} "
            f"{str(row.get('gate', '-')):16} "
            f"{str(row.get('gpu_index', '-')):6} "
            f"{str(row.get('child_pid') or row.get('pid') or '-'):10} "
            f"{str(row.get('run_id') or '-')}"
        )
    lines.extend(
        [
            "",
            "LATEST RUNS",
            f"{'STATE':12} {'DATASET':16} {'STAGE':28} {'GPU':6} {'PID':10} RUN_ID",
        ]
    )
    for row in payload["runs"]:
        lines.append(
            f"{str(row.get('state', '-')):12} "
            f"{str(row.get('dataset', '-')):16} "
            f"{str(row.get('stage', '-')):28} "
            f"{str(row.get('gpu_index', '-')):6} "
            f"{str(row.get('pid') or '-'):10} "
            f"{str(row.get('run_id', '-'))}"
        )
    if "gpus" in payload:
        lines.extend(["", "GPU"])
        if payload.get("gpu_error"):
            lines.append(f"unavailable: {payload['gpu_error']}")
        for gpu in payload.get("gpus", []):
            lines.append(
                f"GPU {gpu['index']} {gpu['uuid']} util={gpu['utilization_gpu_percent']}% "
                f"free={gpu['memory_free_mb']}MiB processes={gpu['process_count']}"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--watch", type=float, default=0.0)
    parser.add_argument("--config", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project_root = resolve_project_root(args.project_root)
        data_root = select_data_root(project_root, explicit=args.data_root)
        layout = build_runtime_layout(project_root=project_root, data_root=data_root)
        while True:
            payload = collect_status(layout, include_gpu=args.gpu, limit=args.limit)
            body = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                if args.format == "json"
                else render_table(payload)
            )
            if args.watch > 0 and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(body, end="", flush=True)
            if args.watch <= 0:
                break
            time.sleep(max(1.0, args.watch))
        return 0
    except KeyboardInterrupt:
        return 130
    except AutoDLRuntimeError as exc:
        print(f"AUTODL_STATUS_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
