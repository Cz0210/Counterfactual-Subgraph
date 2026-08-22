#!/usr/bin/env python3
"""Read-only four-GPU recovery queue, worker, and lock dashboard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from scripts.autodl.run_four_gpu_recovery_controller import (
    CONTROLLER_NAME,
    RECOVERY_BACE_STAGES,
    audit_gpu_locks,
    load_controller_manifest,
)
from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    build_runtime_layout,
    query_gpu_inventory,
    read_json_object,
    resolve_project_root,
    select_data_root,
    utc_now,
)


def _heartbeat_age(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _elapsed_seconds(start: Any, end: Any = None) -> float | None:
    if not isinstance(start, str):
        return None
    try:
        start_at = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_at = (
            datetime.fromisoformat(end.replace("Z", "+00:00"))
            if isinstance(end, str)
            else datetime.now(timezone.utc)
        )
    except ValueError:
        return None
    if start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=timezone.utc)
    if end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=timezone.utc)
    return max(0.0, (end_at - start_at).total_seconds())


def _load_priorities(root: Path) -> dict[str, int]:
    snapshot = root / "controller_manifest.json"
    if not snapshot.is_file():
        return {}
    payload = read_json_object(snapshot)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return {}
    return {
        str(task.get("id")): int(task.get("priority", 100))
        for task in tasks
        if isinstance(task, dict) and task.get("id") is not None
    }


def _load_tasks(
    root: Path, layout: Any | None = None, priorities: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks_root = root / "tasks"
    if not tasks_root.is_dir():
        return rows
    for state_path in sorted(tasks_root.glob("*/state.json")):
        try:
            state = read_json_object(state_path)
            gate_path = state_path.with_name("gate.json")
            gate = read_json_object(gate_path) if gate_path.is_file() else {}
        except AutoDLRuntimeError as exc:
            rows.append(
                {
                    "task_id": state_path.parent.name,
                    "dataset": "unknown",
                    "stage": "unknown",
                    "state": "CORRUPT",
                    "gate": "CORRUPT",
                    "reason": str(exc),
                    "instances": [],
                }
            )
            continue
        instances = []
        for instance_id, instance in sorted((state.get("instances") or {}).items()):
            run_state: dict[str, Any] = {}
            launch_spec: dict[str, Any] = {}
            run_id = instance.get("run_id")
            if layout is not None and isinstance(run_id, str):
                run_state_path = layout.runs_root / run_id / "state.json"
                if run_state_path.is_file():
                    run_state = read_json_object(run_state_path)
                launch_spec_path = layout.runs_root / run_id / "launch_spec.json"
                if launch_spec_path.is_file():
                    launch_spec = read_json_object(launch_spec_path)
            started_at = instance.get("started_at") or launch_spec.get("created_at")
            completed_at = run_state.get("completed_at") or instance.get(
                "completed_at"
            )
            instances.append(
                {
                    "instance_id": instance_id,
                    "state": instance.get("state"),
                    "attempt": instance.get("attempt"),
                    "run_id": run_id,
                    "adopted": bool(instance.get("adopted", False)),
                    "gpu_index": instance.get("gpu_index"),
                    "gpu_uuid": instance.get("gpu_uuid"),
                    "worker_pid": instance.get("worker_pid")
                    or run_state.get("pid"),
                    "child_pid": instance.get("child_pid")
                    or run_state.get("child_pid"),
                    "tmux_session": instance.get("tmux_session")
                    or run_state.get("tmux_session"),
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": _elapsed_seconds(
                        started_at, completed_at
                    ),
                    "output_root": instance.get("expected_output")
                    or launch_spec.get("expected_output"),
                    "heartbeat_at": instance.get("heartbeat_at"),
                    "heartbeat_age_seconds": _heartbeat_age(
                        instance.get("heartbeat_at")
                    ),
                    "failure_class": instance.get("failure_class"),
                }
            )
        rows.append(
            {
                "task_id": state.get("task_id", state_path.parent.name),
                "dataset": state.get("dataset"),
                "stage": state.get("stage"),
                "state": state.get("state"),
                "gate": gate.get("status", "NOT_EVALUATED"),
                "reason": state.get("reason"),
                "updated_at": state.get("updated_at"),
                "priority": (priorities or {}).get(
                    str(state.get("task_id", state_path.parent.name)), 100
                ),
                "instances": instances,
            }
        )
    return rows


def collect_gpu_status(layout: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Collect one shared GPU/UUID-lock snapshot for read-only status clients."""

    try:
        observations = query_gpu_inventory()
        rows = audit_gpu_locks(
            layout.locks_dir, observations, probe_advisory_lock=False
        )
        by_index = {item.index: item for item in observations}
        for row in rows:
            observation = by_index.get(row.get("gpu_index"))
            if observation is not None:
                row["gpu_name"] = observation.name
        return (
            rows,
            None,
        )
    except AutoDLRuntimeError as exc:
        return [], str(exc)


def collect_status(
    layout: Any,
    *,
    controller_id: str,
    shared_gpu_status: tuple[list[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    root = layout.control_root / CONTROLLER_NAME / controller_id
    controller_state: dict[str, Any]
    if (root / "controller_state.json").is_file():
        controller_state = read_json_object(root / "controller_state.json")
    else:
        controller_state = {
            "controller_id": controller_id,
            "state": "NOT_INITIALIZED",
            "heartbeat_at": None,
            "task_counts": {},
        }
    priorities = _load_priorities(root)
    tasks = _load_tasks(root, layout, priorities)
    by_stage = {str(row.get("stage")): row for row in tasks}
    by_dataset = {str(row.get("dataset")): row for row in tasks}
    if shared_gpu_status is None:
        gpus, gpu_error = collect_gpu_status(layout)
    else:
        gpus, gpu_error = shared_gpu_status
    bace = [
        by_stage.get(
            stage,
            {
                "task_id": stage,
                "dataset": "bace",
                "stage": stage,
                "state": "NOT_CONFIGURED",
                "gate": "NOT_EVALUATED",
                "instances": [],
            },
        )
        for stage in RECOVERY_BACE_STAGES
    ]
    queue_order = {
        "RUNNING": 0,
        "STARTING": 1,
        "READY": 2,
        "WAITING_RESOURCE": 3,
        "WAITING_DEPENDENCY": 4,
        "NOT_STARTED": 5,
        "FAILED": 6,
        "BLOCKED": 7,
        "SKIPPED": 8,
        "PASS": 9,
    }
    queue = sorted(
        tasks,
        key=lambda row: (
            queue_order.get(str(row.get("state")), 99),
            int(row.get("priority", 100)),
            str(row.get("task_id")),
        ),
    )
    ready = sorted(
        (
            row
            for row in tasks
            if row.get("state") in {"READY", "WAITING_RESOURCE"}
        ),
        key=lambda row: (int(row.get("priority", 100)), str(row.get("task_id"))),
    )
    return {
        "schema_version": 1,
        "refreshed_at": utc_now(),
        "controller_root": str(root),
        "controller": {
            **controller_state,
            "heartbeat_age_seconds": _heartbeat_age(
                controller_state.get("heartbeat_at")
            ),
        },
        "gpus": gpus,
        "gpu_error": gpu_error,
        "queue": queue,
        "next_queued": ready[0] if ready else None,
        "mut": by_dataset.get(
            "mutagenicity",
            {"state": "NOT_CONFIGURED", "task_id": "mut_recovery"},
        ),
        "aids": by_dataset.get(
            "aids", {"state": "NOT_CONFIGURED", "task_id": "aids_recovery"}
        ),
        "bace_b6_b14": bace,
        "tastemolnet": {
            "state": "BLOCKED",
            "reason": "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW",
        },
        "paper": {"state": "FROZEN", "mutations_allowed": False},
    }


def _instance_summary(row: dict[str, Any]) -> str:
    instances = row.get("instances") or []
    if not instances:
        return "-"
    values = []
    for instance in instances:
        gpu = instance.get("gpu_index")
        worker = instance.get("worker_pid")
        child = instance.get("child_pid")
        tmux = instance.get("tmux_session") or "-"
        duration = instance.get("duration_seconds")
        duration_text = f"{duration:.0f}s" if isinstance(duration, float) else "-"
        heartbeat = instance.get("heartbeat_age_seconds")
        heartbeat_text = f"{heartbeat:.0f}s" if isinstance(heartbeat, float) else "-"
        output = str(instance.get("output_root") or "-")
        if len(output) > 48:
            output = "..." + output[-45:]
        values.append(
            f"{instance.get('instance_id')}={instance.get('state')}"
            f"/a{instance.get('attempt')}@gpu{gpu if gpu is not None else '-'} "
            f"pid={worker or '-'}/{child or '-'} tmux={tmux} "
            f"duration={duration_text} heartbeat={heartbeat_text} output={output}"
            f"{' [adopted]' if instance.get('adopted') else ''}"
        )
    return ",".join(values)


def render_table(payload: dict[str, Any]) -> str:
    controller = payload["controller"]
    lines = [
        f"AutoDL four-GPU recovery @ {payload['refreshed_at']}",
        f"CONTROLLER {controller.get('controller_id')} "
        f"state={controller.get('state')} pid={controller.get('pid', '-')} "
        f"workload_state={controller.get('workload_state', controller.get('state'))} "
        f"heartbeat_age={controller.get('heartbeat_age_seconds', '-')}s",
        "",
        "GPU / UUID LOCK AUDIT",
        f"{'GPU':4} {'UUID':22} {'UTIL':7} {'USED/FREE/TOTAL_MIB':24} {'LOCK':18} {'AUDIT'}",
    ]
    if payload.get("gpu_error"):
        lines.append(f"GPU inventory unavailable: {payload['gpu_error']}")
    for row in payload.get("gpus", []):
        lines.append(
            f"{str(row.get('gpu_index')):4} "
            f"{str(row.get('gpu_uuid'))[:22]:22} "
            f"{str(row.get('utilization_gpu_percent')) + '%':7} "
            f"{str(row.get('memory_used_mb')) + '/' + str(row.get('memory_free_mb')) + '/' + str(row.get('memory_total_mb')):24} "
            f"{str(row.get('lock_state')):18} "
            f"{str(row.get('audit'))}"
        )
    lines.extend(
        [
            "",
            "NEXT QUEUED",
            (
                f"{payload['next_queued'].get('task_id')} "
                f"priority={payload['next_queued'].get('priority')} "
                f"state={payload['next_queued'].get('state')}"
                if payload.get("next_queued")
                else "none"
            ),
            "",
            "WORK-CONSERVING QUEUE",
            f"{'TASK':32} {'DATASET':14} {'STAGE':29} {'STATE':20} INSTANCES",
        ]
    )
    for row in payload.get("queue", []):
        lines.append(
            f"{str(row.get('task_id')):32} "
            f"{str(row.get('dataset')):14} "
            f"{str(row.get('stage')):29} "
            f"{str(row.get('state')):20} "
            f"{_instance_summary(row)}"
        )
    lines.extend(
        [
            "",
            "REQUIRED LINES",
            f"MUT   {payload['mut'].get('state')} ({payload['mut'].get('task_id')})",
            f"AIDS  {payload['aids'].get('state')} ({payload['aids'].get('task_id')})",
        ]
    )
    for row in payload["bace_b6_b14"]:
        lines.append(
            f"BACE  {str(row.get('stage')):29} {str(row.get('state')):20} "
            f"gate={row.get('gate')}"
        )
    lines.extend(
        [
            "TASTE BLOCKED  TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW",
            "PAPER FROZEN   mutations_allowed=false",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--controller-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--watch", type=float, default=0)
    parser.add_argument("--config", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        controller_id = args.controller_id
        if args.manifest is not None:
            manifest = load_controller_manifest(args.manifest)
            if controller_id is not None and controller_id != manifest.controller_id:
                raise AutoDLRuntimeError("--controller-id conflicts with --manifest")
            controller_id = manifest.controller_id
        if not controller_id:
            raise AutoDLRuntimeError("Provide --controller-id or --manifest")
        project_root = resolve_project_root(args.project_root)
        data_root = select_data_root(project_root, explicit=args.data_root)
        layout = build_runtime_layout(
            project_root=project_root,
            data_root=data_root,
            control_root=args.control_root,
        )
        while True:
            payload = collect_status(layout, controller_id=controller_id)
            body = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
                if args.format == "json"
                else render_table(payload)
            )
            if args.watch > 0 and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(body, end="", flush=True)
            if args.watch <= 0:
                return 0
            time.sleep(max(1, args.watch))
    except KeyboardInterrupt:
        return 130
    except (AutoDLRuntimeError, OSError, ValueError) as exc:
        print(f"FOUR_GPU_RECOVERY_STATUS_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
