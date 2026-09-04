"""Read-only orchestration snapshot for the final four main-table cells.

This module deliberately does not dispatch subprocesses. Science ownership is
provided by sealed one-shot binders and recorded in the canonical owner
registry. The controller only reopens that registry, checks live PID
generations, and reports which already-owned successor is responsible for each
remaining cell.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from .final16_owner_registry_v1 import (
    process_start_ticks,
    validate_owner_registry,
)
from .final_four_cells_observer import (
    REMAINING_CELLS,
    read_matrix_authority,
    stable_sha256,
)


SNAPSHOT_SCHEMA = "final16_successors_snapshot_v1"
HEARTBEAT_SCHEMA = "final16_successors_heartbeat_v1"
OWNER_SCHEMA = "final16_successors_controller_owner_v1"
TERMINAL_SCHEMA = "final16_successors_controller_terminal_v1"


class Final16SuccessorsError(RuntimeError):
    """The owner registry or matrix authority cannot be reopened safely."""


def _physical_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise Final16SuccessorsError(f"{label} must be one non-empty physical file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Final16SuccessorsError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Final16SuccessorsError(f"{label} must contain one JSON object")
    return value


def _heartbeat_pid(value: Mapping[str, Any]) -> int | None:
    for field in ("owner_pid", "controller_pid", "pid"):
        candidate = value.get(field)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
    return None


def _heartbeat_ticks(value: Mapping[str, Any]) -> int | None:
    for field in ("owner_start_ticks", "controller_start_ticks", "start_ticks"):
        candidate = value.get(field)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
    return None


def _observe_identity(
    row: Mapping[str, Any], *, proc_root: Path, label: str
) -> dict[str, Any]:
    declared_state = str(row.get("owner_state") or "")
    pid = row.get("owner_pid")
    ticks = row.get("owner_start_ticks")
    heartbeat_raw = row.get("heartbeat")
    result: dict[str, Any] = {
        "declared_state": declared_state,
        "owner_pid": pid,
        "owner_start_ticks": ticks,
        "heartbeat": heartbeat_raw,
        "process_live": False,
        "heartbeat_identity_match": None,
        "heartbeat_state": None,
        "observation": "NOT_RUNNING",
    }
    if declared_state not in {"ADOPTED_RUNNING", "RUNNING"}:
        return result
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(ticks, int)
        or isinstance(ticks, bool)
    ):
        result["observation"] = "STALE_REGISTRY_OWNER"
        return result
    observed_ticks = process_start_ticks(proc_root, pid)
    result["observed_start_ticks"] = observed_ticks
    result["process_live"] = observed_ticks == ticks
    if not isinstance(heartbeat_raw, str) or not heartbeat_raw:
        result["observation"] = "STALE_REGISTRY_OWNER"
        return result
    heartbeat_path = Path(heartbeat_raw)
    try:
        heartbeat = _physical_json(heartbeat_path, label=f"{label} heartbeat")
    except Final16SuccessorsError as exc:
        result["heartbeat_error"] = str(exc)
        result["observation"] = "STALE_REGISTRY_OWNER"
        return result
    heartbeat_pid = _heartbeat_pid(heartbeat)
    heartbeat_ticks = _heartbeat_ticks(heartbeat)
    identity_match = heartbeat_pid == pid and heartbeat_ticks in (None, ticks)
    result["heartbeat_identity_match"] = identity_match
    result["heartbeat_state"] = heartbeat.get(
        "state", heartbeat.get("phase", heartbeat.get("status"))
    )
    if result["process_live"] and identity_match:
        result["observation"] = "ADOPT_EXISTING_OWNER"
    else:
        result["observation"] = "STALE_REGISTRY_OWNER"
    return result


def _resolve_authority_state(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink():
        raise Final16SuccessorsError(
            "matrix authority root must be an absolute non-symlink directory"
        )
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise Final16SuccessorsError("matrix authority root is not a directory")
    state = resolved / "state.json"
    if state.is_symlink() or not state.is_file():
        raise Final16SuccessorsError("matrix authority state.json is absent or indirect")
    return state


def build_snapshot(
    *,
    matrix_authority_root: Path,
    owner_registry_path: Path,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Return one evidence-only snapshot without launching or signalling work."""

    authority_state = _resolve_authority_state(matrix_authority_root)
    matrix = read_matrix_authority(authority_state)
    registry_raw = _physical_json(owner_registry_path, label="owner registry")
    try:
        registry = validate_owner_registry(
            registry_raw, proc_root=proc_root, check_processes=False
        )
    except Exception as exc:
        raise Final16SuccessorsError(f"owner registry validation failed: {exc}") from exc
    expected_authority = matrix_authority_root.resolve(strict=True)
    registered_authority = Path(str(registry["matrix_authority_root"])).resolve(
        strict=True
    )
    if registered_authority != expected_authority:
        raise Final16SuccessorsError(
            "owner registry references a different matrix authority"
        )

    task_observations: dict[str, dict[str, Any]] = {}
    stale_owners: list[str] = []
    ready_without_live_owner: list[str] = []
    for row in registry["tasks"]:
        task_id = str(row["task_id"])
        observed = _observe_identity(row, proc_root=proc_root, label=task_id)
        task_observations[task_id] = {
            **observed,
            "dataset": row["dataset"],
            "method": row["method"],
            "stage": row["stage"],
            "gpu": row["gpu"],
            "output_root": row["output_root"],
            "successor_task_id": row["successor_task_id"],
            "publisher_id": row["publisher_id"],
        }
        if observed["observation"] == "STALE_REGISTRY_OWNER":
            stale_owners.append(task_id)
        if row["owner_state"] in {"PREDEPLOYED", "READY"}:
            ready_without_live_owner.append(task_id)

    publisher_observations: dict[str, dict[str, Any]] = {}
    stale_publishers: list[str] = []
    for row in registry["publishers"]:
        publisher_id = str(row["publisher_id"])
        observed = _observe_identity(row, proc_root=proc_root, label=publisher_id)
        publisher_observations[publisher_id] = {
            **observed,
            "cell_id": row["cell_id"],
            "claim_enabled": row["claim_enabled"],
            "active_writer_count": row["active_writer_count"],
            "locator": row["locator"],
            "lease_path": row["lease_path"],
        }
        if observed["observation"] == "STALE_REGISTRY_OWNER":
            stale_publishers.append(publisher_id)

    missing_cells = [
        cell for cell in REMAINING_CELLS if cell not in matrix["applied_cells"]
    ]
    main_ready_waiting_gpu = sorted(
        task_id
        for task_id, row in task_observations.items()
        if row["declared_state"] == "READY" and row["gpu"] is not None
    )
    blockers: list[str] = []
    if stale_owners:
        blockers.append("STALE_REGISTRY_OWNER")
    if stale_publishers:
        blockers.append("STALE_REGISTRY_PUBLISHER")

    matrix_count = int(matrix["latest_count"])
    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA,
        "state": (
            "BLOCKED_STALE_REGISTRY"
            if blockers
            else "MAIN_MATRIX_COMPLETE"
            if matrix_count == 16
            else "RUNNING_LONG_EXPERIMENTS"
        ),
        "matrix_authority_root": str(expected_authority),
        "matrix_authority_state": str(authority_state),
        "matrix_authority_state_sha256": stable_sha256(matrix),
        "matrix_complete_cells": matrix_count,
        "matrix_total_cells": 16,
        "missing_cells": missing_cells,
        "owner_registry": str(owner_registry_path.resolve(strict=True)),
        "owner_registry_sha256": registry["self_sha256"],
        "registry_id": registry["registry_id"],
        "tasks": task_observations,
        "publishers": publisher_observations,
        "gpu_leases": registry["gpu_leases"],
        "stale_owners": sorted(stale_owners),
        "stale_publishers": sorted(stale_publishers),
        "predeployed_or_ready_tasks": sorted(ready_without_live_owner),
        "main_ready_waiting_gpu": main_ready_waiting_gpu,
        "blockers": blockers,
        "dispatch_mode": "SEALED_ONE_SHOT_BINDERS_ONLY",
        "controller_launches_science": False,
        "controller_launches_publishers": False,
        "controller_consumes_next_action": False,
        "science_restart_performed": False,
        "signal_sent": False,
        "matrix_write_performed": False,
        "secondary_matrix_authority_created": False,
        "gpu_lock_acquired": False,
        "llm_ablation_gate": (
            "DELEGATED_TO_EXISTING_STRICT_GATE"
            if matrix_count >= 13
            else "BLOCKED_WAITING_MATRIX_13"
        ),
        "gnn_ablation_gate": (
            "WAITING_FINAL_EXPORT_RECEIPTS"
            if matrix_count == 16
            else "BLOCKED_WAITING_MATRIX_16"
        ),
        "llm_ablation_started": False,
        "gnn_ablation_started": False,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    snapshot["snapshot_sha256"] = stable_sha256(snapshot)
    return snapshot


__all__ = [
    "Final16SuccessorsError",
    "HEARTBEAT_SCHEMA",
    "OWNER_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "TERMINAL_SCHEMA",
    "build_snapshot",
]
