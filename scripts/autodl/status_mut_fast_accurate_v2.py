#!/usr/bin/env python3
"""Read-only status view for the Mut historical-50k successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def snapshot(spec_path: Path) -> dict[str, Any]:
    spec = _json(spec_path)
    if spec is None:
        raise ValueError(f"invalid spec: {spec_path}")
    control = Path(str(spec["control_root"]))
    controller_id = str(spec["controller_id"])
    sidecar_root = control / "mut_fast_accurate_v2" / controller_id
    four_root = control / "four_gpu_recovery" / controller_id
    sidecar = _json(sidecar_root / "heartbeat.json")
    four = _json(four_root / "heartbeat.json")
    controller_state = _json(four_root / "controller_state.json")
    tasks: dict[str, Any] = {}
    for task_id in (
        "mut_fast_equivalence_500",
        "mut_fast_historical_binding",
        "mut_fast_threshold_freeze",
        "mut_fast_standardized",
    ):
        value = _json(four_root / "tasks" / task_id / "state.json")
        tasks[task_id] = value if value is not None else {"state": "NOT_CREATED"}
    matrix_receipt = _json(sidecar_root / "matrix_publisher_receipt.json")
    matrix_heartbeat = None
    if matrix_receipt is not None:
        matrix_heartbeat = _json(Path(str(matrix_receipt.get("heartbeat") or "")))
    observation = (sidecar or {}).get("observation") or {}
    return {
        "controller_id": controller_id,
        "spec": str(spec_path.resolve()),
        "successor_state": (sidecar or {}).get("state", "NOT_STARTED"),
        "successor_pid": (sidecar or {}).get("pid"),
        "successor_heartbeat": (sidecar or {}).get("heartbeat_at"),
        "trace_on_historical_adoption_authorized": spec.get(
            "allow_trace_on_historical_adoption", False
        ),
        "manual_intervention_required": (sidecar or {}).get(
            "manual_intervention_required", False
        ),
        "manual_intervention_reason": (sidecar or {}).get(
            "manual_intervention_reason"
        ),
        "cgroup_version": observation.get("cgroup_version", 1),
        "cgroup_mount_read_only": observation.get("cgroup_mount_read_only", True),
        "child_cgroup_created": observation.get("child_cgroup_created", False),
        "no_child_reason": observation.get(
            "no_child_reason", spec.get("no_child_fallback_reason")
        ),
        "parent_headroom_bytes": observation.get("parent_headroom_bytes"),
        "required_parent_headroom_bytes": spec["minimum_parent_headroom_bytes"],
        "eligible_idle_exclusive_gpus": observation.get(
            "eligible_idle_exclusive_gpus", []
        ),
        "stable_admission_seconds": (sidecar or {}).get(
            "stable_admission_seconds"
        ),
        "four_gpu_controller_state": (controller_state or {}).get("state"),
        "four_gpu_controller_heartbeat": (four or {}).get("heartbeat_at"),
        "tasks": tasks,
        "matrix_publisher_receipt": matrix_receipt,
        "matrix_publisher_heartbeat": matrix_heartbeat,
        "old_440_waiter_signaled_by_successor": False,
        "gnn_ablation_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--watch", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    while True:
        value = snapshot(args.spec)
        if args.json:
            print(json.dumps(value, indent=2, sort_keys=True))
        else:
            for key, item in value.items():
                encoded = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item
                print(f"{key}={encoded}")
        if args.watch <= 0:
            return 0
        time.sleep(max(5, args.watch))


if __name__ == "__main__":
    raise SystemExit(main())
