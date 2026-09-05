"""Canonical owner and publisher registry for the final four main-table cells.

The registry is deliberately a *description* of already running or predeployed
owners.  It never starts science.  This lets the closeout controller adopt a
healthy process without turning a stale controller snapshot into runtime truth,
and gives every matrix cell exactly one canonical publisher claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .autodl_mut_first_divergence_v1 import stable_sha256


SCHEMA = "final16_canonical_owner_registry_v1"
TASK_FIELDS = {
    "task_id",
    "dataset",
    "method",
    "stage",
    "owner_state",
    "owner_pid",
    "owner_start_ticks",
    "heartbeat",
    "input_root",
    "output_root",
    "execution_commit",
    "task_spec_sha",
    "gpu",
    "successor_task_id",
    "publisher_id",
}
RUNNING_STATES = {"ADOPTED_RUNNING", "RUNNING"}
NONRUNNING_STATES = {
    "PREDEPLOYED",
    "READY",
    "PASS",
    "BLOCKED",
    "MISSING",
    "TERMINAL_FAILED_ENGINEERING",
}
PUBLISHER_ACTIVE_STATES = {"ADOPTED_RUNNING", "RUNNING", "PREDEPLOYED", "READY"}
PUBLISHER_INACTIVE_STATES = {"PASS", "SUPERSEDED_DUPLICATE_CLAIM", "BLOCKED"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class Final16OwnerRegistryError(RuntimeError):
    """Registry input is ambiguous, stale, or permits duplicate ownership."""


def _absolute(value: Any, *, field: str, allow_absent: bool = True) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise Final16OwnerRegistryError(f"{field} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise Final16OwnerRegistryError(
            f"{field} must be an absolute non-symlink path"
        )
    resolved = path.resolve(strict=not allow_absent)
    if not allow_absent and not resolved.exists():  # pragma: no cover - resolve raises
        raise Final16OwnerRegistryError(f"{field} is absent")
    return resolved


def process_start_ticks(proc_root: str | Path, pid: int) -> int | None:
    """Return Linux start ticks while handling spaces/parentheses in comm."""

    try:
        raw = (Path(proc_root) / str(pid) / "stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        if closing < 0:
            return None
        return int(raw[closing + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise Final16OwnerRegistryError(f"{field} is absent or indirect: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Final16OwnerRegistryError(f"{field} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Final16OwnerRegistryError(f"{field} must contain one JSON object")
    return value


def _heartbeat_identity(value: Mapping[str, Any]) -> tuple[int | None, int | None]:
    pid = value.get("owner_pid", value.get("pid"))
    ticks = value.get("owner_start_ticks", value.get("start_ticks"))
    return (
        pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        ticks if isinstance(ticks, int) and not isinstance(ticks, bool) else None,
    )


def _validate_live_identity(
    row: Mapping[str, Any], *, proc_root: Path, label: str
) -> dict[str, Any]:
    pid = row.get("owner_pid")
    ticks = row.get("owner_start_ticks")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(ticks, int)
        or isinstance(ticks, bool)
        or ticks <= 0
    ):
        raise Final16OwnerRegistryError(f"{label} lacks an exact process identity")
    observed = process_start_ticks(proc_root, pid)
    if observed != ticks:
        raise Final16OwnerRegistryError(
            f"{label} process identity is not live: expected={ticks}, observed={observed}"
        )
    heartbeat = _absolute(row.get("heartbeat"), field=f"{label}.heartbeat", allow_absent=False)
    payload = _read_json(heartbeat, field=f"{label}.heartbeat")
    heartbeat_pid, heartbeat_ticks = _heartbeat_identity(payload)
    if heartbeat_pid != pid:
        raise Final16OwnerRegistryError(f"{label} heartbeat binds another PID")
    if heartbeat_ticks is not None and heartbeat_ticks != ticks:
        raise Final16OwnerRegistryError(f"{label} heartbeat binds another process")
    return {
        "pid": pid,
        "start_ticks": ticks,
        "heartbeat": str(heartbeat),
        "heartbeat_state": payload.get("state", payload.get("phase")),
    }


def _normalize_task(
    raw: Mapping[str, Any], *, proc_root: Path, check_processes: bool
) -> dict[str, Any]:
    if set(raw) != TASK_FIELDS:
        raise Final16OwnerRegistryError(
            "task fields changed: "
            f"missing={sorted(TASK_FIELDS - set(raw))}, "
            f"extra={sorted(set(raw) - TASK_FIELDS)}"
        )
    row = dict(raw)
    task_id = str(row.get("task_id") or "")
    if not task_id or "/" in task_id:
        raise Final16OwnerRegistryError("task_id is empty or unsafe")
    for field in ("dataset", "method", "stage"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise Final16OwnerRegistryError(f"{task_id}.{field} is empty")
    state = str(row.get("owner_state") or "")
    if state not in RUNNING_STATES | NONRUNNING_STATES:
        raise Final16OwnerRegistryError(f"{task_id}.owner_state is unsupported")
    for field in ("input_root", "output_root"):
        row[field] = str(_absolute(row[field], field=f"{task_id}.{field}"))
    commit = row.get("execution_commit")
    if not isinstance(commit, str) or _GIT_SHA.fullmatch(commit) is None:
        raise Final16OwnerRegistryError(f"{task_id}.execution_commit is not a Git SHA")
    spec_sha = row.get("task_spec_sha")
    if not isinstance(spec_sha, str) or _SHA256.fullmatch(spec_sha) is None:
        raise Final16OwnerRegistryError(f"{task_id}.task_spec_sha is not SHA-256")
    gpu = row.get("gpu")
    if gpu is not None and (
        not isinstance(gpu, int) or isinstance(gpu, bool) or not 0 <= gpu <= 15
    ):
        raise Final16OwnerRegistryError(f"{task_id}.gpu is invalid")
    for field in ("successor_task_id", "publisher_id"):
        if row.get(field) is not None and (
            not isinstance(row[field], str) or not row[field]
        ):
            raise Final16OwnerRegistryError(f"{task_id}.{field} is invalid")
    if state in RUNNING_STATES:
        if check_processes:
            _validate_live_identity(row, proc_root=proc_root, label=task_id)
        elif row.get("owner_pid") is None or row.get("owner_start_ticks") is None:
            raise Final16OwnerRegistryError(f"{task_id} running identity is incomplete")
    else:
        if row.get("owner_pid") is not None or row.get("owner_start_ticks") is not None:
            raise Final16OwnerRegistryError(
                f"{task_id} non-running owner may not claim a process"
            )
        if row.get("heartbeat") is not None:
            row["heartbeat"] = str(
                _absolute(row["heartbeat"], field=f"{task_id}.heartbeat")
            )
    return row


def _normalize_publisher(
    raw: Mapping[str, Any], *, proc_root: Path, check_processes: bool
) -> dict[str, Any]:
    required = {
        "publisher_id",
        "cell_id",
        "owner_state",
        "owner_pid",
        "owner_start_ticks",
        "heartbeat",
        "locator",
        "lease_path",
        "execution_commit",
        "claim_enabled",
        "active_writer_count",
    }
    if set(raw) != required:
        raise Final16OwnerRegistryError("publisher fields changed")
    row = dict(raw)
    publisher_id = str(row.get("publisher_id") or "")
    cell_id = str(row.get("cell_id") or "")
    if not publisher_id or cell_id.count("/") != 1:
        raise Final16OwnerRegistryError("publisher identity/cell is invalid")
    state = str(row.get("owner_state") or "")
    if state not in PUBLISHER_ACTIVE_STATES | PUBLISHER_INACTIVE_STATES:
        raise Final16OwnerRegistryError(f"{publisher_id}.owner_state is unsupported")
    for field in ("locator", "lease_path"):
        row[field] = str(_absolute(row[field], field=f"{publisher_id}.{field}"))
    commit = row.get("execution_commit")
    if not isinstance(commit, str) or _GIT_SHA.fullmatch(commit) is None:
        raise Final16OwnerRegistryError(f"{publisher_id}.execution_commit is invalid")
    claim_enabled = row.get("claim_enabled")
    writer_count = row.get("active_writer_count")
    if claim_enabled not in (True, False) or (
        not isinstance(writer_count, int)
        or isinstance(writer_count, bool)
        or writer_count < 0
    ):
        raise Final16OwnerRegistryError(
            f"{publisher_id} claim/writer evidence is invalid"
        )
    if state in PUBLISHER_ACTIVE_STATES and claim_enabled is not True:
        raise Final16OwnerRegistryError(f"{publisher_id} active claim is disabled")
    if state in PUBLISHER_INACTIVE_STATES and claim_enabled is not False:
        raise Final16OwnerRegistryError(f"{publisher_id} inactive claim is enabled")
    if state == "SUPERSEDED_DUPLICATE_CLAIM" and writer_count != 0:
        raise Final16OwnerRegistryError(
            f"{publisher_id} superseded publisher still has an active writer"
        )
    has_identity = row.get("owner_pid") is not None or row.get("owner_start_ticks") is not None
    if state in RUNNING_STATES or (
        state == "SUPERSEDED_DUPLICATE_CLAIM" and has_identity
    ):
        if check_processes:
            _validate_live_identity(row, proc_root=proc_root, label=publisher_id)
    elif has_identity:
        raise Final16OwnerRegistryError(
            f"{publisher_id} inactive/predeployed claim may not own a PID"
        )
    elif row.get("heartbeat") is not None:
        row["heartbeat"] = str(
            _absolute(row["heartbeat"], field=f"{publisher_id}.heartbeat")
        )
    return row


def build_owner_registry(
    *,
    registry_id: str,
    matrix_authority_root: str | Path,
    tasks: Sequence[Mapping[str, Any]],
    publishers: Sequence[Mapping[str, Any]],
    gpu_leases: Sequence[Mapping[str, Any]],
    proc_root: str | Path = "/proc",
    check_processes: bool = True,
) -> dict[str, Any]:
    """Normalize live ownership and reject duplicate writers/publishers."""

    if not registry_id or "/" in registry_id:
        raise Final16OwnerRegistryError("registry_id is empty or unsafe")
    authority = _absolute(
        matrix_authority_root,
        field="matrix_authority_root",
        allow_absent=not check_processes,
    )
    proc = Path(proc_root)
    normalized_tasks = [
        _normalize_task(row, proc_root=proc, check_processes=check_processes)
        for row in tasks
    ]
    ids = [str(row["task_id"]) for row in normalized_tasks]
    if len(ids) != len(set(ids)):
        raise Final16OwnerRegistryError("duplicate task_id")
    active_outputs: dict[str, str] = {}
    for row in normalized_tasks:
        if row["owner_state"] not in RUNNING_STATES:
            continue
        output = str(row["output_root"])
        if output in active_outputs:
            raise Final16OwnerRegistryError(
                f"duplicate active output writer: {active_outputs[output]} and {row['task_id']}"
            )
        active_outputs[output] = str(row["task_id"])
    normalized_publishers = [
        _normalize_publisher(row, proc_root=proc, check_processes=check_processes)
        for row in publishers
    ]
    publisher_ids = [str(row["publisher_id"]) for row in normalized_publishers]
    if len(publisher_ids) != len(set(publisher_ids)):
        raise Final16OwnerRegistryError("duplicate publisher_id")
    canonical_by_cell: dict[str, str] = {}
    for row in normalized_publishers:
        if row["claim_enabled"] is not True:
            continue
        cell = str(row["cell_id"])
        if cell in canonical_by_cell:
            raise Final16OwnerRegistryError(
                f"multiple canonical publishers for {cell}: "
                f"{canonical_by_cell[cell]} and {row['publisher_id']}"
            )
        canonical_by_cell[cell] = str(row["publisher_id"])
    known_publishers = set(publisher_ids)
    known_tasks = set(ids)
    for row in normalized_tasks:
        successor = row.get("successor_task_id")
        publisher = row.get("publisher_id")
        if successor is not None and successor not in known_tasks:
            raise Final16OwnerRegistryError(
                f"{row['task_id']} references unknown successor {successor}"
            )
        if publisher is not None and publisher not in known_publishers:
            raise Final16OwnerRegistryError(
                f"{row['task_id']} references unknown publisher {publisher}"
            )
    normalized_leases: list[dict[str, Any]] = []
    active_gpu_owners: dict[int, str] = {}
    for raw in gpu_leases:
        if set(raw) != {"gpu", "task_id", "state", "lease_path"}:
            raise Final16OwnerRegistryError("gpu lease fields changed")
        gpu = raw.get("gpu")
        task_id = str(raw.get("task_id") or "")
        state = str(raw.get("state") or "")
        if (
            not isinstance(gpu, int)
            or isinstance(gpu, bool)
            or not 0 <= gpu <= 15
            or task_id not in known_tasks
            or state not in {"HELD", "PREDEPLOYED", "RELEASED"}
        ):
            raise Final16OwnerRegistryError("gpu lease is invalid")
        if state == "HELD" and gpu in active_gpu_owners:
            raise Final16OwnerRegistryError(f"multiple active owners for GPU {gpu}")
        if state == "HELD":
            active_gpu_owners[gpu] = task_id
        normalized_leases.append(
            {
                "gpu": gpu,
                "task_id": task_id,
                "state": state,
                "lease_path": str(
                    _absolute(raw["lease_path"], field=f"gpu{gpu}.lease_path")
                ),
            }
        )
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "registry_id": registry_id,
        "matrix_authority_root": str(authority),
        "tasks": sorted(normalized_tasks, key=lambda row: str(row["task_id"])),
        "publishers": sorted(
            normalized_publishers, key=lambda row: str(row["publisher_id"])
        ),
        "gpu_leases": sorted(
            normalized_leases, key=lambda row: (int(row["gpu"]), str(row["task_id"]))
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    value["self_sha256"] = stable_sha256(value)
    return value


def validate_owner_registry(
    raw: Mapping[str, Any], *, proc_root: str | Path = "/proc", check_processes: bool = True
) -> dict[str, Any]:
    value = dict(raw)
    if value.get("schema_version") != SCHEMA:
        raise Final16OwnerRegistryError("owner registry schema changed")
    observed = value.pop("self_sha256", None)
    if observed != stable_sha256(value):
        raise Final16OwnerRegistryError("owner registry self hash changed")
    rebuilt = build_owner_registry(
        registry_id=str(value.get("registry_id") or ""),
        matrix_authority_root=str(value.get("matrix_authority_root") or ""),
        tasks=value.get("tasks") if isinstance(value.get("tasks"), list) else [],
        publishers=(
            value.get("publishers") if isinstance(value.get("publishers"), list) else []
        ),
        gpu_leases=(
            value.get("gpu_leases") if isinstance(value.get("gpu_leases"), list) else []
        ),
        proc_root=proc_root,
        check_processes=check_processes,
    )
    # Preserve the signed observation timestamp rather than generating a new one.
    rebuilt["updated_at"] = value.get("updated_at")
    rebuilt["self_sha256"] = observed
    if rebuilt != {**value, "self_sha256": observed}:
        raise Final16OwnerRegistryError("owner registry canonical content changed")
    return rebuilt


def atomic_write_owner_registry(path: str | Path, value: Mapping[str, Any]) -> None:
    """Durably replace ``current.json`` and fsync its parent directory."""

    target = _absolute(path, field="owner_registry_output")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise Final16OwnerRegistryError("owner registry output may not be a symlink")
    encoded = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "Final16OwnerRegistryError",
    "SCHEMA",
    "atomic_write_owner_registry",
    "build_owner_registry",
    "process_start_ticks",
    "validate_owner_registry",
]
