#!/usr/bin/env python3
"""Report physical GPUs and select only UUIDs idle for a stable window."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    build_runtime_layout,
    observe_stable_idle_gpus,
    resolve_project_root,
    select_data_root,
    validate_max_gpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--max-gpus", type=int, default=int(os.environ.get("AUTODL_MAX_GPUS", "2"))
    )
    parser.add_argument(
        "--gpu-hard-limit",
        type=int,
        default=int(os.environ.get("AUTODL_GPU_HARD_LIMIT", "2")),
        help=(
            "Explicit reviewed upper bound for --max-gpus. The default remains two; "
            "four-GPU workflows must opt in with --gpu-hard-limit 4."
        ),
    )
    parser.add_argument(
        "--min-free-memory-mb",
        type=int,
        default=int(os.environ.get("AUTODL_MIN_FREE_MEMORY_MB", "16000")),
    )
    parser.add_argument(
        "--idle-util-threshold",
        type=int,
        default=int(os.environ.get("AUTODL_IDLE_UTIL_THRESHOLD", "10")),
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=float(os.environ.get("AUTODL_IDLE_STABLE_SECONDS", "60")),
    )
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--format", choices=("json", "csv", "lines"), default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        max_gpus = validate_max_gpus(args.max_gpus, hard_limit=args.gpu_hard_limit)
        project_root = resolve_project_root(args.project_root)
        data_root = select_data_root(project_root, explicit=args.data_root)
        layout = build_runtime_layout(project_root=project_root, data_root=data_root).ensure()
        inventory = observe_stable_idle_gpus(
            stable_seconds=0.0 if args.once else args.stable_seconds,
            sample_interval_seconds=args.sample_interval,
            min_free_memory_mb=args.min_free_memory_mb,
            max_utilization_percent=args.idle_util_threshold,
        )
        selected = inventory.selected(
            max_gpus=max_gpus,
            lock_root=layout.locks_dir,
            hard_limit=args.gpu_hard_limit,
        )
    except AutoDLRuntimeError as exc:
        print(f"AUTODL_GPU_INVENTORY_FAILED: {exc}", file=sys.stderr)
        return 2

    selected_uuids = {gpu.uuid for gpu in selected}
    payload = {
        "schema_version": 1,
        "sampled_at": inventory.sampled_at,
        "stable_seconds": inventory.stable_seconds,
        "samples": inventory.samples,
        "constraints": {
            "max_gpus": max_gpus,
            "gpu_hard_limit": args.gpu_hard_limit,
            "min_free_memory_mb": args.min_free_memory_mb,
            "idle_util_threshold": args.idle_util_threshold,
        },
        "selected": [gpu.as_json() for gpu in selected],
        "gpus": [
            {
                **gpu.as_json(),
                "stable_idle": gpu.uuid in inventory.stable_idle_uuids,
                "selected": gpu.uuid in selected_uuids,
            }
            for gpu in inventory.observations
        ],
    }
    if args.format == "json":
        body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    elif args.format == "lines":
        body = "".join(f"{gpu.index}\t{gpu.uuid}\n" for gpu in selected)
    else:
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream)
        writer.writerow(
            [
                "index",
                "uuid",
                "name",
                "memory_total_mb",
                "memory_used_mb",
                "memory_free_mb",
                "utilization_gpu_percent",
                "process_count",
                "selected",
            ]
        )
        for gpu in inventory.observations:
            writer.writerow(
                [
                    gpu.index,
                    gpu.uuid,
                    gpu.name,
                    gpu.memory_total_mb,
                    gpu.memory_used_mb,
                    gpu.memory_free_mb,
                    gpu.utilization_gpu_percent,
                    gpu.process_count,
                    gpu.uuid in selected_uuids,
                ]
            )
        body = stream.getvalue()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body, end="")
    if args.require_idle and not selected:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
