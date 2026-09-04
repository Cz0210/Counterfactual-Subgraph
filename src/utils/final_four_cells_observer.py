"""Read-only state aggregation for the four remaining main-table cells."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from src.utils.main_ready_task_specs import load_spec, probe_owner, probe_terminal


SCHEMA_VERSION = "final_four_cells_observer_v1"
REMAINING_CELLS = (
    "Mutagenicity/ComRecGC",
    "TasteMolNet/GCFExplainer",
    "TasteMolNet/GlobalGCE",
    "TasteMolNet/ComRecGC",
)


class FinalFourObserverError(RuntimeError):
    """Observer input evidence is missing or inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(dict(payload)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_matrix_authority(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalFourObserverError("matrix authority must be one physical file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalFourObserverError("matrix authority is unreadable") from exc
    if type(payload) is not dict:
        raise FinalFourObserverError("matrix authority must contain one object")
    if payload.get("schema_version") != "fast16_matrix_authority_pointer_v1":
        raise FinalFourObserverError("matrix authority schema changed")
    count = payload.get("latest_count")
    cells = payload.get("applied_cells")
    if type(count) is not int or not 0 <= count <= 16:
        raise FinalFourObserverError("matrix authority count is invalid")
    if type(cells) is not list or len(cells) != count or len(set(cells)) != count:
        raise FinalFourObserverError("matrix authority cell set is invalid")
    return dict(payload)


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise FinalFourObserverError(f"status input is not one physical file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalFourObserverError(f"status input is unreadable: {path}") from exc
    if type(payload) is not dict:
        raise FinalFourObserverError(f"status input is not one object: {path}")
    return dict(payload)


def snapshot(
    *,
    matrix_authority: Path,
    task_specs: Sequence[Path] = (),
    hpc_t8_pointer: Path | None = None,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    matrix = read_matrix_authority(matrix_authority)
    components: dict[str, Any] = {}
    for path in task_specs:
        spec = load_spec(path)
        task_id = str(spec["task_id"])
        if task_id in components:
            raise FinalFourObserverError(f"duplicate task spec: {task_id}")
        components[task_id] = {
            "task_kind": spec["task_kind"],
            "spec": str(path),
            "spec_sha256": spec["spec_sha256"],
            "owner": probe_owner(spec, proc_root=proc_root),
            "terminal": probe_terminal(spec),
        }
    t8 = _read_optional_json(hpc_t8_pointer)
    missing = [cell for cell in REMAINING_CELLS if cell not in matrix["applied_cells"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS" if not missing else "RUNNING_LONG_EXPERIMENTS",
        "matrix_authority": str(matrix_authority),
        "matrix_complete_cells": matrix["latest_count"],
        "matrix_total_cells": 16,
        "missing_cells": missing,
        "components": components,
        "hpc_t8_pointer": t8,
        "llm_ablation_started": False,
        "gnn_ablation_started": False,
        "science_restart_performed": False,
        "matrix_write_performed": False,
    }


__all__ = [
    "FinalFourObserverError",
    "REMAINING_CELLS",
    "SCHEMA_VERSION",
    "atomic_json",
    "read_matrix_authority",
    "snapshot",
    "stable_sha256",
    "utc_now",
]
