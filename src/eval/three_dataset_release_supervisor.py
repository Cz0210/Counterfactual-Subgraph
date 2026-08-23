"""Persistent, fail-closed release supervisor for the 12-cell staging result.

The supervisor is deliberately outside every scientific controller.  It owns
no GPU slot and never writes a scientific cell root.  Its only mutable state is
its private control directory.  Once all twelve AIDS/Mutagenicity/BACE
standardized cells close, it creates one canonical registry in a temporary
directory, publishes that directory atomically, and invokes the existing
staging-only exporter through its public CLI.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.eval.four_by_four_main_results import (
    METHOD_SLUGS,
    REQUIRED_CELL_FILES,
    MainResultsError,
    audit_cell,
)
from src.eval.four_by_four_registry import (
    AuditConfig,
    PASS_STATUSES,
    SCHEMA_VERSION as MATRIX_SCHEMA_VERSION,
    audit_registry,
    sha256_file,
    stable_json_sha256,
    write_registry_outputs,
)
from src.eval.three_dataset_main_results import (
    DATASET_ORDER,
    TASTE_BLOCKED_REASON,
    TASTE_BLOCKED_STATUS,
    TASTE_DATASET,
    _validate_three_dataset_boundary,
)


CATALOG_SCHEMA_VERSION = "three_dataset_release_cell_catalog_v1"
SPEC_SCHEMA_VERSION = "three_dataset_release_supervisor_spec_v1"
CONTROL_SCHEMA_VERSION = "three_dataset_release_supervisor_control_v1"
STATE_SCHEMA_VERSION = "three_dataset_release_supervisor_state_v1"
SNAPSHOT_SCHEMA_VERSION = "three_dataset_release_cell_snapshot_v1"
REGISTRY_RELEASE_SCHEMA_VERSION = "three_dataset_release_registry_v1"
TRANSACTION_SCHEMA_VERSION = "three_dataset_release_transaction_v1"
POLL_INTERVAL_SECONDS = 60
MAX_EXPORT_ATTEMPTS = 2
METHOD_ORDER = ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC")
EXPECTED_CELLS = tuple(
    (dataset, method) for dataset in DATASET_ORDER for method in METHOD_ORDER
)
FROZEN_V4_CELLS = frozenset(
    (dataset, method)
    for dataset in ("AIDS", "Mutagenicity")
    for method in ("Ours", "GCFExplainer", "GlobalGCE")
)
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


class ReleaseSupervisorError(RuntimeError):
    """A release-control invariant failed."""


class ReleaseSpecError(ReleaseSupervisorError):
    """The immutable release specification is invalid."""


class ReleaseBlocked(ReleaseSupervisorError):
    """The release cannot advance without a new external closure."""


@dataclass(frozen=True)
class CellProbe:
    dataset: str
    method: str
    state: str
    root: str | None
    missing_files: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class TickResult:
    state: str
    complete_cells: int
    probes: tuple[CellProbe, ...]
    reason: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) is atomic and, unlike replace(2), fails if another writer
        # occupied the supposedly fresh destination after our initial check.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise FileExistsError(f"Fresh output required: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing entry."""

    if platform.system() == "Linux":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            status = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if status == 0:
                return
            observed_errno = ctypes.get_errno()
            if observed_errno == errno.EEXIST:
                raise FileExistsError(f"Fresh output required: {destination}")
            if observed_errno not in {errno.ENOSYS, errno.EINVAL}:
                raise OSError(
                    observed_errno,
                    os.strerror(observed_errno),
                    str(destination),
                )
    # Development-only portability path (the persistent controller itself is
    # Linux/procfs-bound). The immediately preceding existence check is still
    # guarded by this process's exclusive supervisor lock.
    if destination.exists():
        raise FileExistsError(f"Fresh output required: {destination}")
    os.rename(source, destination)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseSpecError(f"Invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseSpecError(f"{label} must contain one JSON object: {path}")
    return dict(payload)


def _absolute(value: Any, *, label: str, existing: bool = False) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise ReleaseSpecError(f"{label} must be absolute: {value!r}")
    try:
        return path.resolve(strict=existing)
    except FileNotFoundError as exc:
        raise ReleaseSpecError(f"{label} does not exist: {path}") from exc


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if SAFE_ID_PATTERN.fullmatch(text) is None:
        raise ReleaseSpecError(f"{label} is not a safe ID: {value!r}")
    return text


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ReleaseSpecError(f"Identity input must be a physical file: {path}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_file_identity(identity: Mapping[str, Any], *, label: str) -> Path:
    path = _absolute(identity.get("path"), label=f"{label}.path", existing=True)
    if path.is_symlink() or not path.is_file():
        raise ReleaseSpecError(f"{label} is not a physical file: {path}")
    expected_size = int(identity.get("size", -1))
    expected_hash = str(identity.get("sha256") or "").lower()
    if HASH_PATTERN.fullmatch(expected_hash) is None:
        raise ReleaseSpecError(f"{label}.sha256 is invalid")
    stat = path.stat()
    if stat.st_size != expected_size or sha256_file(path) != expected_hash:
        raise ReleaseSpecError(f"{label} identity drift: {path}")
    return path


def _expand_catalog_path(value: Any, *, matrix_root: Path) -> str:
    return str(value or "").replace("{matrix_root}", str(matrix_root))


def _parse_assignments(
    values: Sequence[str], *, label: str
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for declaration in values:
        if "=" not in declaration or "/" not in declaration.split("=", 1)[0]:
            raise ReleaseSpecError(f"{label} must use DATASET/METHOD=VALUE")
        identity, value = declaration.split("=", 1)
        dataset, method = (item.strip() for item in identity.split("/", 1))
        key = (dataset, method)
        if key not in EXPECTED_CELLS:
            raise ReleaseSpecError(f"{label} has unsupported cell: {identity}")
        if key in result or not value.strip():
            raise ReleaseSpecError(f"Duplicate or empty {label}: {declaration}")
        result[key] = value.strip()
    return result


def _walk_json(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_json(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_json(item, (*path, str(index)))


def _root_matches_template(root: Path, template: str) -> bool:
    escaped = re.escape(template)
    escaped = escaped.replace(re.escape("{attempt}"), r"[0-9]+")
    return re.fullmatch(escaped, str(root)) is not None


def _find_owner_binding(
    payload: Mapping[str, Any], *, root: Path, task_id: str | None
) -> dict[str, Any] | None:
    exact_root_paths = [
        path for path, value in _walk_json(payload) if str(value) == str(root)
    ]
    task_objects: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    if task_id:
        for path, value in _walk_json(payload):
            if not isinstance(value, Mapping):
                continue
            observed = value.get("id") or value.get("task_id")
            if str(observed or "") == task_id:
                task_objects.append((path, value))
        for path, task in task_objects:
            for key in (
                "expected_output",
                "output_root",
                "standardized_output_root",
                "root",
            ):
                raw = task.get(key)
                if isinstance(raw, str) and _root_matches_template(root, raw):
                    return {
                        "kind": "task_output_template",
                        "json_path": "/".join((*path, key)),
                        "task_id": task_id,
                        "value": raw,
                    }
        if task_objects and exact_root_paths:
            return {
                "kind": "task_id_and_exact_root",
                "json_path": "/".join(exact_root_paths[0]),
                "task_id": task_id,
                "value": str(root),
            }
        return None
    if exact_root_paths:
        return {
            "kind": "exact_root",
            "json_path": "/".join(exact_root_paths[0]),
            "value": str(root),
        }
    return None


def load_cell_catalog(path: str | Path) -> dict[str, Any]:
    catalog_path = _absolute(path, label="catalog", existing=True)
    payload = _read_json(catalog_path, label="cell catalog")
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ReleaseSpecError("Unsupported three-dataset cell catalog schema")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 12:
        raise ReleaseSpecError("Cell catalog must contain exactly 12 entries")
    observed: set[tuple[str, str]] = set()
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise ReleaseSpecError("Every cell catalog entry must be an object")
        key = (str(raw.get("dataset") or ""), str(raw.get("method") or ""))
        if key not in EXPECTED_CELLS or key in observed:
            raise ReleaseSpecError(f"Invalid or duplicate catalog cell: {key}")
        observed.add(key)
        if raw.get("root_kind") not in {"fixed", "placeholder"}:
            raise ReleaseSpecError(f"Invalid root_kind for {key}")
        if raw.get("root_layout", "direct") not in {
            "direct",
            "nested_standardized",
        }:
            raise ReleaseSpecError(f"Invalid root_layout for {key}")
    if observed != set(EXPECTED_CELLS):
        raise ReleaseSpecError("Cell catalog identity set is incomplete")
    return payload


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip().lower()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ReleaseSpecError("Unable to bind the execution Git commit")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_release_spec(
    *,
    catalog_path: str | Path,
    controller_id: str,
    project_root: str | Path,
    runtime_root: str | Path,
    python: str | Path,
    state_root: str | Path,
    registry_root: str | Path,
    output_root: str | Path,
    paper_staging_root: str | Path,
    expectations_json: str | Path,
    taste_license_gate_json: str | Path,
    adoption_manifest: str | Path | None = None,
    cell_root_overrides: Sequence[str] = (),
    owner_manifest_overrides: Sequence[str] = (),
    owner_task_overrides: Sequence[str] = (),
    require_runnable: bool = False,
) -> dict[str, Any]:
    """Build one immutable release spec without creating result destinations."""

    catalog_file = _absolute(catalog_path, label="catalog", existing=True)
    catalog = load_cell_catalog(catalog_file)
    project = _absolute(project_root, label="project_root", existing=True)
    runtime = _absolute(runtime_root, label="runtime_root", existing=True)
    interpreter = _absolute(python, label="python", existing=True)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ReleaseSpecError(f"Python is not executable: {interpreter}")
    state = _absolute(state_root, label="state_root")
    registry = _absolute(registry_root, label="registry_root")
    output = _absolute(output_root, label="output_root")
    staging = _absolute(paper_staging_root, label="paper_staging_root")
    paper = (project / "paper").resolve(strict=False)
    for label, destination in (
        ("registry_root", registry),
        ("output_root", output),
        ("paper_staging_root", staging),
    ):
        if _inside(destination, paper):
            raise ReleaseSpecError(f"{label} may not be under paper/: {destination}")
    if len({registry, output, staging}) != 3:
        raise ReleaseSpecError("Registry, result, and paper-staging roots must differ")
    for label, destination in (
        ("registry_root", registry),
        ("output_root", output),
        ("paper_staging_root", staging),
    ):
        if destination.exists():
            raise FileExistsError(f"{label} must be fresh at spec build: {destination}")

    matrix_root = _absolute(catalog.get("matrix_root"), label="catalog.matrix_root")
    if output != matrix_root / "three_datasets_complete_v1":
        raise ReleaseSpecError(
            "output_root must be MATRIX_ROOT/three_datasets_complete_v1"
        )
    for label, destination in (
        ("state_root", state),
        ("registry_root", registry),
        ("output_root", output),
        ("paper_staging_root", staging),
    ):
        if not _inside(destination, runtime):
            raise ReleaseSpecError(f"{label} must remain under runtime_root")
    roots = _parse_assignments(cell_root_overrides, label="--cell-root")
    owners = _parse_assignments(
        owner_manifest_overrides, label="--cell-owner-manifest"
    )
    owner_tasks = _parse_assignments(owner_task_overrides, label="--cell-owner-task")
    adoption_value = adoption_manifest or catalog.get("adoption_manifest")
    adoption_path = _absolute(adoption_value, label="adoption_manifest")
    adoption_identity: dict[str, Any] = {
        "path": str(adoption_path),
        "status": "PLACEHOLDER_MISSING",
    }
    if adoption_path.is_file() and not adoption_path.is_symlink():
        adoption_identity = {**_file_identity(adoption_path), "status": "FIXED"}

    cells: list[dict[str, Any]] = []
    unresolved: list[str] = []
    raw_by_key = {
        (str(raw["dataset"]), str(raw["method"])): raw
        for raw in catalog["cells"]
    }
    for dataset, method in EXPECTED_CELLS:
        raw = dict(raw_by_key[(dataset, method)])
        root_layout = str(raw.get("root_layout") or "direct")
        if root_layout not in {"direct", "nested_standardized"}:
            raise ReleaseSpecError(
                f"Unsupported root layout for {dataset}/{method}: {root_layout}"
            )
        override = roots.get((dataset, method))
        if override is not None:
            root = _absolute(override, label=f"cell root {dataset}/{method}")
            root_kind = "fixed"
            placeholder = ""
        elif raw.get("root_kind") == "fixed":
            root = _absolute(
                _expand_catalog_path(raw.get("standardized_root"), matrix_root=matrix_root),
                label=f"catalog root {dataset}/{method}",
            )
            root_kind = "fixed"
            placeholder = ""
        else:
            root = None
            root_kind = "placeholder"
            placeholder = str(raw.get("placeholder") or "UNRESOLVED_CELL_ROOT")
            unresolved.append(f"{dataset}/{method}:root:{placeholder}")

        task_id = owner_tasks.get((dataset, method)) or str(
            raw.get("owner_task_id") or ""
        )
        if task_id:
            task_id = _safe_id(task_id, label=f"owner task {dataset}/{method}")
        required_task_id = str(raw.get("required_owner_task_id") or "")
        if required_task_id:
            required_task_id = _safe_id(
                required_task_id,
                label=f"required owner task {dataset}/{method}",
            )
            if task_id != required_task_id:
                raise ReleaseSpecError(
                    f"{dataset}/{method} requires owner task {required_task_id}"
                )
        if (dataset, method) in FROZEN_V4_CELLS:
            owner: dict[str, Any] = {
                "kind": "user_approved_frozen_v4",
                "manifest": dict(adoption_identity),
                "task_id": None,
                "binding": None,
            }
            if adoption_identity["status"] != "FIXED":
                unresolved.append(f"{dataset}/{method}:adoption_manifest_missing")
        else:
            owner_value = owners.get((dataset, method)) or raw.get(
                "owner_manifest_hint"
            )
            owner_path = (
                _absolute(owner_value, label=f"owner manifest {dataset}/{method}")
                if owner_value
                else None
            )
            owner = {
                "kind": "controller_manifest",
                "manifest": {
                    "path": str(owner_path) if owner_path else "",
                    "status": "PLACEHOLDER_MISSING",
                },
                "task_id": task_id or None,
                "binding": None,
            }
            if (
                root is not None
                and owner_path is not None
                and owner_path.is_file()
                and not owner_path.is_symlink()
            ):
                owner_payload = _read_json(
                    owner_path, label=f"owner manifest {dataset}/{method}"
                )
                binding = _find_owner_binding(
                    owner_payload, root=root, task_id=task_id or None
                )
                if binding is not None:
                    owner = {
                        **owner,
                        "manifest": {**_file_identity(owner_path), "status": "FIXED"},
                        "binding": binding,
                    }
            if owner["manifest"]["status"] != "FIXED" or owner["binding"] is None:
                unresolved.append(f"{dataset}/{method}:owner_manifest_binding_unresolved")

        binding_fixed = (
            root_kind == "fixed" and owner["manifest"]["status"] == "FIXED"
        )
        inactive_root_hint = (
            str(root)
            if root is not None and not binding_fixed
            else _expand_catalog_path(
                raw.get("candidate_root_hint"), matrix_root=matrix_root
            )
            or None
        )
        cells.append(
            {
                "dataset": dataset,
                "method": method,
                "root_layout": root_layout,
                "required_owner_task_id": required_task_id or None,
                "binding_state": "FIXED" if binding_fixed else "PLACEHOLDER",
                "owner_output_root": str(root) if binding_fixed else None,
                "standardized_root": (
                    str(root / "standardized")
                    if binding_fixed and root_layout == "nested_standardized"
                    else str(root)
                    if binding_fixed
                    else None
                ),
                "placeholder": (
                    None
                    if binding_fixed
                    else placeholder or "OWNER_MANIFEST_BINDING_UNRESOLVED"
                ),
                "candidate_root_hint": inactive_root_hint,
                "owner": owner,
            }
        )

    if require_runnable and unresolved:
        raise ReleaseSpecError(
            "Runnable spec still has unresolved bindings: " + "; ".join(unresolved)
        )

    expectations_identity = _file_identity(
        _absolute(expectations_json, label="expectations_json", existing=True)
    )
    taste_identity = _file_identity(
        _absolute(
            taste_license_gate_json,
            label="taste_license_gate_json",
            existing=True,
        )
    )
    taste_payload = _read_json(Path(taste_identity["path"]), label="Taste license gate")
    if taste_payload.get("passed") is True or str(
        taste_payload.get("status") or ""
    ).upper() == "PASS":
        raise ReleaseSpecError("Taste license gate must remain fail-closed for this release")

    code_paths = {
        "supervisor_module": project / "src/eval/three_dataset_release_supervisor.py",
        "supervisor_cli": project / "scripts/autodl/run_three_dataset_release_supervisor.py",
        "exporter_cli": project / "scripts/autodl/export_three_dataset_main_results.py",
        "exporter_module": project / "src/eval/three_dataset_main_results.py",
        "registry_module": project / "src/eval/four_by_four_registry.py",
        "config": project / "configs/hpc.yaml",
    }
    code = {
        name: _file_identity(path.resolve(strict=True))
        for name, path in code_paths.items()
    }
    controller = _safe_id(controller_id, label="controller_id")
    spec = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "controller_id": controller,
        "created_at": utc_now(),
        "execution_commit": _git_head(project),
        "runnable": not unresolved,
        "unresolved_bindings": unresolved,
        "resource": "cpu",
        "gpu_required": False,
        "gpu_lock_acquired": False,
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        "max_export_attempts": MAX_EXPORT_ATTEMPTS,
        "run_tastemolnet": 0,
        "paths": {
            "project_root": str(project),
            "runtime_root": str(runtime),
            "matrix_root": str(matrix_root),
            "python": str(interpreter),
            "state_root": str(state),
            "registry_root": str(registry),
            "output_root": str(output),
            "paper_staging_root": str(staging),
        },
        "inputs": {
            "catalog": _file_identity(catalog_file),
            "expectations": expectations_identity,
            "taste_license_gate": taste_identity,
            "adoption_manifest": adoption_identity,
            "code": code,
        },
        "cells": cells,
        "taste_cells": [
            {
                "dataset": TASTE_DATASET,
                "method": method,
                "status": TASTE_BLOCKED_STATUS,
                "reason": TASTE_BLOCKED_REASON,
            }
            for method in METHOD_ORDER
        ],
        "release_policy": {
            "required_complete_cells": 12,
            "matrix_total_cells": 16,
            "six_frozen_v4_cells_recomputed": False,
            "numeric_imputation_allowed": False,
            "paper_directory_writes_allowed": False,
            "existing_controller_mutation_allowed": False,
            "scientific_cell_writes_allowed": False,
        },
    }
    spec["content_sha256"] = stable_json_sha256(
        {key: value for key, value in spec.items() if key != "content_sha256"}
    )
    return spec


def write_release_spec(path: str | Path, spec: Mapping[str, Any]) -> Path:
    destination = _absolute(path, label="spec_output")
    _fresh_json(destination, spec)
    return destination


def load_release_spec(path: str | Path) -> tuple[Path, str, dict[str, Any]]:
    spec_path = _absolute(path, label="release spec", existing=True)
    if spec_path.is_symlink() or not spec_path.is_file():
        raise ReleaseSpecError("Release spec must be one physical file")
    raw = spec_path.read_bytes()
    digest = _digest_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseSpecError(f"Invalid release spec: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ReleaseSpecError("Unsupported release spec schema")
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != stable_json_sha256(content):
        raise ReleaseSpecError("Release spec content hash is invalid")
    if int(payload.get("poll_interval_seconds", 0)) != POLL_INTERVAL_SECONDS:
        raise ReleaseSpecError("Release supervisor poll interval must be exactly 60 seconds")
    if payload.get("resource") != "cpu" or payload.get("gpu_required") is not False:
        raise ReleaseSpecError("Release supervisor must be CPU-only")
    if int(payload.get("run_tastemolnet", -1)) != 0:
        raise ReleaseSpecError("RUN_TASTEMOLNET must remain zero")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 12:
        raise ReleaseSpecError("Release spec must bind exactly 12 cells")
    observed = {
        (cell.get("dataset"), cell.get("method"))
        for cell in raw_cells
        if isinstance(cell, Mapping)
    }
    if observed != set(EXPECTED_CELLS):
        raise ReleaseSpecError("Release spec cell identity set is invalid")
    unresolved_cells = 0
    for cell in raw_cells:
        if not isinstance(cell, Mapping):
            raise ReleaseSpecError("Release spec cell entry is not an object")
        key = (str(cell.get("dataset") or ""), str(cell.get("method") or ""))
        binding_state = cell.get("binding_state")
        root_layout = cell.get("root_layout")
        if root_layout not in {"direct", "nested_standardized"}:
            raise ReleaseSpecError(f"Invalid cell root layout: {key}")
        if binding_state not in {"FIXED", "PLACEHOLDER"}:
            raise ReleaseSpecError(f"Invalid cell binding state: {key}")
        if binding_state == "PLACEHOLDER":
            unresolved_cells += 1
            if (
                cell.get("standardized_root") is not None
                or cell.get("owner_output_root") is not None
            ):
                raise ReleaseSpecError(f"Placeholder cell has a writable root: {key}")
            continue
        standardized_root = _absolute(
            cell.get("standardized_root"),
            label=f"standardized root {key[0]}/{key[1]}",
        )
        owner_output_root = _absolute(
            cell.get("owner_output_root"),
            label=f"owner output root {key[0]}/{key[1]}",
        )
        expected_standardized = (
            owner_output_root / "standardized"
            if root_layout == "nested_standardized"
            else owner_output_root
        )
        if standardized_root != expected_standardized:
            raise ReleaseSpecError(f"Cell root layout does not match its owner: {key}")
        owner = cell.get("owner")
        if not isinstance(owner, Mapping):
            raise ReleaseSpecError(f"Fixed cell owner is missing: {key}")
        manifest = owner.get("manifest")
        if not isinstance(manifest, Mapping) or manifest.get("status") != "FIXED":
            raise ReleaseSpecError(f"Fixed cell owner manifest is not bound: {key}")
        if key in FROZEN_V4_CELLS:
            if owner.get("kind") != "user_approved_frozen_v4":
                raise ReleaseSpecError(f"Frozen-v4 owner authority changed: {key}")
        elif (
            owner.get("kind") != "controller_manifest"
            or not owner.get("task_id")
            or not isinstance(owner.get("binding"), Mapping)
        ):
            raise ReleaseSpecError(
                f"External owner manifest/task/output binding is incomplete: {key}"
            )
        required_task_id = cell.get("required_owner_task_id")
        if required_task_id and owner.get("task_id") != required_task_id:
            raise ReleaseSpecError(f"Required owner task changed: {key}")
    if bool(payload.get("runnable")) != (unresolved_cells == 0):
        raise ReleaseSpecError("Release spec runnable flag disagrees with cell bindings")
    paths = payload.get("paths")
    if not isinstance(paths, Mapping):
        raise ReleaseSpecError("Release spec paths are missing")
    project = _absolute(paths.get("project_root"), label="project_root", existing=True)
    paper = (project / "paper").resolve(strict=False)
    runtime = _absolute(paths.get("runtime_root"), label="runtime_root", existing=True)
    matrix_root = _absolute(paths.get("matrix_root"), label="matrix_root")
    state_root = _absolute(paths.get("state_root"), label="state_root")
    destinations = [
        _absolute(paths.get(name), label=name)
        for name in ("registry_root", "output_root", "paper_staging_root")
    ]
    if len(set(destinations)) != 3 or any(_inside(path, paper) for path in destinations):
        raise ReleaseSpecError("Release destinations overlap or enter paper/")
    if any(not _inside(path, runtime) for path in (*destinations, state_root)):
        raise ReleaseSpecError("Release destinations must remain under runtime_root")
    if destinations[1] != matrix_root / "three_datasets_complete_v1":
        raise ReleaseSpecError(
            "output_root must be MATRIX_ROOT/three_datasets_complete_v1"
        )
    taste_cells = payload.get("taste_cells")
    expected_taste = {
        (TASTE_DATASET, method, TASTE_BLOCKED_STATUS, TASTE_BLOCKED_REASON)
        for method in METHOD_ORDER
    }
    observed_taste = {
        (
            str(cell.get("dataset") or ""),
            str(cell.get("method") or ""),
            str(cell.get("status") or ""),
            str(cell.get("reason") or ""),
        )
        for cell in taste_cells or []
        if isinstance(cell, Mapping)
    }
    if observed_taste != expected_taste:
        raise ReleaseSpecError("TasteMolNet fail-closed rows changed")
    for name, identity in (payload.get("inputs", {}).get("code") or {}).items():
        if not isinstance(identity, Mapping):
            raise ReleaseSpecError(f"Invalid code identity: {name}")
        _verify_file_identity(identity, label=f"code.{name}")
    return spec_path, digest, payload


def _cell_required_names(method: str, *, direct: bool) -> tuple[str, ...]:
    return (
        *REQUIRED_CELL_FILES,
        f"table2_{METHOD_SLUGS[method]}_k10.csv",
        "_FINALIZED.json",
        *(("PASS",) if direct else ()),
    )


def probe_cells(spec: Mapping[str, Any]) -> tuple[CellProbe, ...]:
    probes: list[CellProbe] = []
    for binding in spec["cells"]:
        dataset = str(binding["dataset"])
        method = str(binding["method"])
        raw_root = binding.get("standardized_root")
        raw_owner_root = binding.get("owner_output_root")
        if (
            binding.get("binding_state") != "FIXED"
            or not raw_root
            or not raw_owner_root
        ):
            probes.append(
                CellProbe(
                    dataset,
                    method,
                    "WAITING_PLACEHOLDER",
                    None,
                    reason=str(binding.get("placeholder") or "UNRESOLVED_BINDING"),
                )
            )
            continue
        unresolved = Path(str(raw_root)).expanduser()
        unresolved_owner = Path(str(raw_owner_root)).expanduser()
        if unresolved.is_symlink():
            probes.append(
                CellProbe(dataset, method, "BLOCKED_SYMLINK_ROOT", str(unresolved))
            )
            continue
        if unresolved_owner.is_symlink():
            probes.append(
                CellProbe(
                    dataset,
                    method,
                    "BLOCKED_SYMLINK_OWNER_ROOT",
                    str(unresolved_owner),
                )
            )
            continue
        try:
            root = unresolved.resolve(strict=True)
        except FileNotFoundError:
            probes.append(
                CellProbe(dataset, method, "WAITING_MISSING_ROOT", str(unresolved))
            )
            continue
        try:
            owner_root = unresolved_owner.resolve(strict=True)
        except FileNotFoundError:
            probes.append(
                CellProbe(
                    dataset,
                    method,
                    "WAITING_MISSING_OWNER_ROOT",
                    str(unresolved_owner),
                )
            )
            continue
        if not root.is_dir():
            probes.append(CellProbe(dataset, method, "BLOCKED_NOT_DIRECTORY", str(root)))
            continue
        if not owner_root.is_dir():
            probes.append(
                CellProbe(
                    dataset,
                    method,
                    "BLOCKED_OWNER_NOT_DIRECTORY",
                    str(owner_root),
                )
            )
            continue
        direct = binding.get("root_layout") == "direct"
        missing = tuple(
            name
            for name in _cell_required_names(method, direct=direct)
            if not (root / name).is_file() or (root / name).stat().st_size <= 0
        )
        if not direct:
            missing += tuple(
                f"owner:{name}"
                for name in (
                    "run_manifest.json",
                    "final_gate.json",
                    "_RUN_COMPLETE.json",
                    "PASS",
                )
                if not (owner_root / name).is_file()
                or (owner_root / name).stat().st_size <= 0
            )
        probes.append(
            CellProbe(
                dataset,
                method,
                "READY_FOR_FULL_AUDIT" if not missing else "WAITING_CLOSURE_FILES",
                str(root),
                missing_files=missing,
            )
        )
    return tuple(probes)


def _verify_bound_inputs(spec: Mapping[str, Any]) -> None:
    inputs = spec.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ReleaseSpecError("Spec input identities are missing")
    for name in ("catalog", "expectations", "taste_license_gate"):
        identity = inputs.get(name)
        if not isinstance(identity, Mapping):
            raise ReleaseSpecError(f"Missing bound input: {name}")
        _verify_file_identity(identity, label=name)
    adoption = inputs.get("adoption_manifest")
    if not isinstance(adoption, Mapping) or adoption.get("status") != "FIXED":
        raise ReleaseBlocked("Frozen-v4 adoption manifest is not hash-bound")
    _verify_file_identity(adoption, label="adoption_manifest")
    for binding in spec["cells"]:
        owner = binding.get("owner")
        if binding.get("binding_state") != "FIXED":
            continue
        if not isinstance(owner, Mapping):
            raise ReleaseSpecError("Fixed cell owner binding is missing")
        identity = owner.get("manifest")
        if not isinstance(identity, Mapping) or identity.get("status") != "FIXED":
            raise ReleaseBlocked(
                f"{binding['dataset']}/{binding['method']} owner is not hash-bound"
            )
        _verify_file_identity(
            identity, label=f"owner.{binding['dataset']}.{binding['method']}"
        )


def _matrix_payload(result: Any) -> dict[str, Any]:
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "audit_complete": True,
        "matrix_complete_cells": result.matrix_complete_cells,
        "matrix_total_cells": result.matrix_total_cells,
        "all_cells_complete": result.matrix_complete_cells == result.matrix_total_cells,
        "cells": list(result.matrix_rows),
        "pass_statuses": sorted(status.value for status in PASS_STATUSES),
        "no_numeric_imputation": True,
    }


def _audit_registry(spec: Mapping[str, Any]) -> Any:
    explicit = {
        f"{binding['dataset']}/{binding['method']}": binding["owner_output_root"]
        for binding in spec["cells"]
    }
    expectations_path = _verify_file_identity(
        spec["inputs"]["expectations"], label="expectations"
    )
    taste_path = _verify_file_identity(
        spec["inputs"]["taste_license_gate"], label="taste_license_gate"
    )
    expectations = _read_json(expectations_path, label="expectations")
    taste = _read_json(taste_path, label="Taste license gate")
    return audit_registry(
        AuditConfig(
            scan_roots=(),
            output_root=Path(spec["paths"]["registry_root"]),
            expectations=expectations,
            explicit_cells=explicit,
            taste_license_gate=taste,
            max_hash_bytes=64 * 1024 * 1024,
        )
    )


def _closure_snapshot(
    spec: Mapping[str, Any], matrix_payload: Mapping[str, Any]
) -> dict[str, Any]:
    selected, _taste = _validate_three_dataset_boundary(matrix_payload)
    cells: list[dict[str, Any]] = []
    for row in selected:
        audited = audit_cell(row)
        binding = next(
            item
            for item in spec["cells"]
            if item["dataset"] == audited.dataset and item["method"] == audited.method
        )
        bound_standardized = Path(str(binding["standardized_root"])).resolve(
            strict=True
        )
        owner_root = Path(str(binding["owner_output_root"])).resolve(strict=True)
        if audited.root != bound_standardized:
            raise ReleaseBlocked(
                f"{audited.dataset}/{audited.method}: registry selected a root "
                "outside the immutable cell binding"
            )
        files: dict[str, dict[str, Any]] = {}
        for name, digest in sorted(audited.source_hashes.items()):
            path = audited.root / name
            stat = path.stat()
            files[name] = {
                "size": stat.st_size,
                "sha256": digest,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        # The presentation auditor deliberately hashes only numeric/provenance
        # inputs.  The release controller additionally binds the standardized
        # closure marker and its publication record so a later restart cannot
        # silently promote a directory that merely happens to retain the CSVs.
        for name in ("_FINALIZED.json",):
            path = audited.root / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise ReleaseBlocked(
                    f"{audited.dataset}/{audited.method}: closure file is missing: {path}"
                )
            stat = path.stat()
            files[name] = {
                "size": stat.st_size,
                "sha256": sha256_file(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        owner_names = (
            ("PASS",)
            if binding["root_layout"] == "direct"
            else ("run_manifest.json", "final_gate.json", "_RUN_COMPLETE.json", "PASS")
        )
        owner_files: dict[str, dict[str, Any]] = {}
        for name in owner_names:
            path = owner_root / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise ReleaseBlocked(
                    f"{audited.dataset}/{audited.method}: owner closure file is "
                    f"missing: {path}"
                )
            stat = path.stat()
            owner_files[name] = {
                "size": stat.st_size,
                "sha256": sha256_file(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        if (owner_root / "PASS").read_text(encoding="utf-8") != "PASS\n":
            raise ReleaseBlocked(
                f"{audited.dataset}/{audited.method}: PASS marker is malformed"
            )
        cells.append(
            {
                "dataset": audited.dataset,
                "method": audited.method,
                "status": row["status"],
                "standardized_root": str(audited.root),
                "root_device": audited.root.stat().st_dev,
                "root_inode": audited.root.stat().st_ino,
                "root_layout": binding["root_layout"],
                "owner_output_root": str(owner_root),
                "owner_root_device": owner_root.stat().st_dev,
                "owner_root_inode": owner_root.stat().st_ino,
                "owner_manifest_sha256": binding["owner"]["manifest"]["sha256"],
                "owner_task_id": binding["owner"].get("task_id"),
                "owner_output_binding": binding["owner"].get("binding"),
                "pass_sha256": owner_files["PASS"]["sha256"],
                "files": files,
                "owner_files": owner_files,
            }
        )
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "cells": cells,
        "adoption_manifest": dict(spec["inputs"]["adoption_manifest"]),
        "scientific_metrics_recomputed": False,
        "six_frozen_v4_cells_recomputed": False,
    }
    snapshot["snapshot_sha256"] = stable_json_sha256(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    return snapshot


def verify_closure_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ReleaseBlocked("Unsupported closure snapshot schema")
    content = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    if snapshot.get("snapshot_sha256") != stable_json_sha256(content):
        raise ReleaseBlocked("Closure snapshot self-hash mismatch")
    cells = snapshot.get("cells")
    if not isinstance(cells, list) or len(cells) != 12:
        raise ReleaseBlocked("Closure snapshot does not bind twelve cells")
    for cell in cells:
        root = _absolute(
            cell.get("standardized_root"),
            label=f"snapshot root {cell.get('dataset')}/{cell.get('method')}",
            existing=True,
        )
        if root.is_symlink() or not root.is_dir():
            raise ReleaseBlocked(f"Snapshot root changed: {root}")
        stat = root.stat()
        if stat.st_dev != int(cell.get("root_device", -1)) or stat.st_ino != int(
            cell.get("root_inode", -1)
        ):
            raise ReleaseBlocked(f"Snapshot root inode changed: {root}")
        files = cell.get("files")
        if not isinstance(files, Mapping):
            raise ReleaseBlocked(f"Snapshot file map missing: {root}")
        for name, identity in files.items():
            path = root / str(name)
            if path.is_symlink() or not path.is_file():
                raise ReleaseBlocked(f"Snapshot file missing or symlinked: {path}")
            observed = path.stat()
            expected_hash = str(identity.get("sha256") or "")
            if (
                observed.st_size != int(identity.get("size", -1))
                or observed.st_dev != int(identity.get("device", -1))
                or observed.st_ino != int(identity.get("inode", -1))
                or observed.st_mtime_ns != int(identity.get("mtime_ns", -1))
                or observed.st_ctime_ns != int(identity.get("ctime_ns", -1))
                or sha256_file(path) != expected_hash
            ):
                raise ReleaseBlocked(f"Snapshot file identity changed: {path}")
        owner_root = _absolute(
            cell.get("owner_output_root"),
            label=f"snapshot owner root {cell.get('dataset')}/{cell.get('method')}",
            existing=True,
        )
        if owner_root.is_symlink() or not owner_root.is_dir():
            raise ReleaseBlocked(f"Snapshot owner root changed: {owner_root}")
        owner_stat = owner_root.stat()
        if owner_stat.st_dev != int(
            cell.get("owner_root_device", -1)
        ) or owner_stat.st_ino != int(cell.get("owner_root_inode", -1)):
            raise ReleaseBlocked(f"Snapshot owner root inode changed: {owner_root}")
        expected_standardized = (
            owner_root / "standardized"
            if cell.get("root_layout") == "nested_standardized"
            else owner_root
        )
        if root != expected_standardized:
            raise ReleaseBlocked("Snapshot root layout changed")
        owner_files = cell.get("owner_files")
        if not isinstance(owner_files, Mapping) or "PASS" not in owner_files:
            raise ReleaseBlocked(f"Snapshot owner file map missing: {owner_root}")
        for name, identity in owner_files.items():
            path = owner_root / str(name)
            if path.is_symlink() or not path.is_file():
                raise ReleaseBlocked(f"Snapshot owner file is missing: {path}")
            observed = path.stat()
            expected_hash = str(identity.get("sha256") or "")
            if (
                observed.st_size != int(identity.get("size", -1))
                or observed.st_dev != int(identity.get("device", -1))
                or observed.st_ino != int(identity.get("inode", -1))
                or observed.st_mtime_ns != int(identity.get("mtime_ns", -1))
                or observed.st_ctime_ns != int(identity.get("ctime_ns", -1))
                or sha256_file(path) != expected_hash
            ):
                raise ReleaseBlocked(
                    f"Snapshot owner file identity changed: {path}"
                )
        if cell.get("pass_sha256") != owner_files["PASS"].get("sha256"):
            raise ReleaseBlocked("Snapshot PASS identity is inconsistent")


def _publish_registry(
    spec: Mapping[str, Any], spec_sha256: str, result: Any
) -> Path:
    final = Path(spec["paths"]["registry_root"])
    if final.exists():
        raise FileExistsError(f"Registry root already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.release-", dir=final.parent)
    )
    try:
        write_registry_outputs(result, temporary)
        matrix_path = temporary / "matrix_status.json"
        matrix = _read_json(matrix_path, label="staged matrix_status")
        _validate_three_dataset_boundary(matrix)
        snapshot = _closure_snapshot(spec, matrix)
        _fresh_json(temporary / "cell_closure_snapshot.json", snapshot)
        release_manifest = {
            "schema_version": REGISTRY_RELEASE_SCHEMA_VERSION,
            "status": "PASS",
            "created_at": utc_now(),
            "controller_id": spec["controller_id"],
            "spec_sha256": spec_sha256,
            "execution_commit": spec["execution_commit"],
            "matrix_status_sha256": sha256_file(matrix_path),
            "cell_closure_snapshot_sha256": sha256_file(
                temporary / "cell_closure_snapshot.json"
            ),
            "matrix_complete_cells": 12,
            "matrix_total_cells": 16,
            "taste_status": TASTE_BLOCKED_REASON,
            "scientific_metrics_recomputed": False,
            "paper_directory_written": False,
        }
        _fresh_json(temporary / "release_registry_manifest.json", release_manifest)
        _fresh_json(
            temporary / "RELEASE_REGISTRY_PASS.json",
            {
                "schema_version": REGISTRY_RELEASE_SCHEMA_VERSION,
                "status": "PASS",
                "manifest_sha256": sha256_file(
                    temporary / "release_registry_manifest.json"
                ),
            },
        )
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, final)
        _fsync_directory(final.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def verify_published_registry(
    spec: Mapping[str, Any], spec_sha256: str
) -> tuple[Path, dict[str, Any]]:
    root = _absolute(
        spec["paths"]["registry_root"], label="registry_root", existing=True
    )
    if root.is_symlink() or not root.is_dir():
        raise ReleaseBlocked("Published registry root is not a physical directory")
    manifest_path = root / "release_registry_manifest.json"
    marker_path = root / "RELEASE_REGISTRY_PASS.json"
    snapshot_path = root / "cell_closure_snapshot.json"
    for path in (manifest_path, marker_path, snapshot_path, root / "matrix_status.json"):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ReleaseBlocked(f"Published registry closure is incomplete: {path}")
    manifest = _read_json(manifest_path, label="release registry manifest")
    marker = _read_json(marker_path, label="release registry marker")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("spec_sha256") != spec_sha256
        or marker.get("status") != "PASS"
        or marker.get("manifest_sha256") != sha256_file(manifest_path)
        or manifest.get("matrix_status_sha256") != sha256_file(root / "matrix_status.json")
        or manifest.get("cell_closure_snapshot_sha256") != sha256_file(snapshot_path)
    ):
        raise ReleaseBlocked("Published registry identity is invalid")
    matrix = _read_json(root / "matrix_status.json", label="published matrix_status")
    _validate_three_dataset_boundary(matrix)
    snapshot = _read_json(snapshot_path, label="cell closure snapshot")
    verify_closure_snapshot(snapshot)
    return root, manifest


def _tree_inventory(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReleaseBlocked(f"Release tree contains a symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def _verify_export_tree(root: Path, matrix_sha256: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ReleaseBlocked(f"Export root is not physical: {root}")
    required = (
        "PASS",
        "THREE_DATASET_EXPORT_PASS.json",
        "three_dataset_export_manifest.json",
        "three_dataset_export_audit.json",
        "paper_figure3_three_datasets.pdf",
        "paper_figure4_three_datasets.pdf",
        "paper_table2_three_datasets.tex",
    )
    for name in required:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ReleaseBlocked(f"Incomplete three-dataset export: {path}")
    if (root / "PASS").read_text(encoding="utf-8") != "PASS\n":
        raise ReleaseBlocked("Three-dataset PASS marker is malformed")
    manifest = _read_json(
        root / "three_dataset_export_manifest.json", label="export manifest"
    )
    audit = _read_json(root / "three_dataset_export_audit.json", label="export audit")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("matrix_complete_cells") != 12
        or manifest.get("matrix_total_cells") != 16
        or manifest.get("matrix_status_sha256") != matrix_sha256
        or manifest.get("paper_status") != "PAPER_FROZEN_PARTIAL"
        or audit.get("passed") is not True
        or audit.get("all_12_three_dataset_cells_verified") is not True
        or audit.get("taste_license_block_preserved") is not True
    ):
        raise ReleaseBlocked("Three-dataset export manifest/audit gate is invalid")


def _transaction_paths(spec: Mapping[str, Any], transaction_id: str) -> tuple[Path, Path]:
    output = Path(spec["paths"]["output_root"])
    staging = Path(spec["paths"]["paper_staging_root"])
    return (
        output.parent / f".{output.name}.{transaction_id}.partial",
        staging.parent / f".{staging.name}.{transaction_id}.partial",
    )


def _run_exporter_process(
    *,
    spec: Mapping[str, Any],
    registry_root: Path,
    temporary_output: Path,
    temporary_staging: Path,
    log_path: Path,
    heartbeat: Callable[[], None],
    process_started: Callable[[Mapping[str, Any]], None],
) -> int:
    paths = spec["paths"]
    project = Path(paths["project_root"])
    python = Path(paths["python"])
    exporter = Path(spec["inputs"]["code"]["exporter_cli"]["path"])
    config = Path(spec["inputs"]["code"]["config"]["path"])
    argv = [
        str(python),
        str(exporter),
        "--config",
        str(config),
        "--set",
        "inference.fallback_to_heuristic=false",
        "--matrix-status",
        str(registry_root / "matrix_status.json"),
        "--output-root",
        str(temporary_output),
        "--paper-staging-root",
        str(temporary_staging),
        "--project-root",
        str(project),
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": str(project),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "RUN_TASTEMOLNET": "0",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            argv,
            cwd=project,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_started(_linux_process_identity(process.pid))
        last_heartbeat = time.monotonic()
        while True:
            return_code = process.poll()
            if return_code is not None:
                return return_code
            now = time.monotonic()
            if now - last_heartbeat >= POLL_INTERVAL_SECONDS:
                heartbeat()
                last_heartbeat = now
            time.sleep(2.0)


def _promote_export_transaction(
    *,
    spec: Mapping[str, Any],
    matrix_sha256: str,
    temporary_output: Path,
    temporary_staging: Path,
    transaction_path: Path,
    transaction: dict[str, Any],
) -> None:
    final_output = Path(spec["paths"]["output_root"])
    final_staging = Path(spec["paths"]["paper_staging_root"])
    _verify_export_tree(temporary_output, matrix_sha256)
    _verify_export_tree(temporary_staging, matrix_sha256)
    if _tree_inventory(temporary_output) != _tree_inventory(temporary_staging):
        raise ReleaseBlocked("Runtime paper staging is not byte-identical to export")
    if final_output.exists() or final_staging.exists():
        raise ReleaseBlocked("Fresh/no-clobber destination was occupied before promotion")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_staging.parent.mkdir(parents=True, exist_ok=True)
    transaction["status"] = "TEMP_EXPORT_VERIFIED"
    transaction["updated_at"] = utc_now()
    _atomic_json(transaction_path, transaction)
    _rename_directory_noreplace(temporary_output, final_output)
    _fsync_directory(final_output.parent)
    transaction["status"] = "OUTPUT_PROMOTED"
    transaction["updated_at"] = utc_now()
    _atomic_json(transaction_path, transaction)
    _rename_directory_noreplace(temporary_staging, final_staging)
    _fsync_directory(final_staging.parent)
    transaction["status"] = "PASS"
    transaction["updated_at"] = utc_now()
    _atomic_json(transaction_path, transaction)


def _reconcile_export(
    *, spec: Mapping[str, Any], state_root: Path, matrix_sha256: str
) -> bool:
    final_output = Path(spec["paths"]["output_root"])
    final_staging = Path(spec["paths"]["paper_staging_root"])
    transaction_path = state_root / "release_transaction.json"
    if not transaction_path.is_file():
        if not final_output.exists() and not final_staging.exists():
            return False
        raise ReleaseBlocked("Release destination exists without owned transaction")
    transaction = _read_json(transaction_path, label="release transaction")
    if (
        transaction.get("schema_version") != TRANSACTION_SCHEMA_VERSION
        or transaction.get("spec_sha256") != spec.get("_runtime_spec_sha256")
    ):
        raise ReleaseBlocked("Release destination transaction identity differs")
    temporary_output = Path(str(transaction["temporary_output"]))
    temporary_staging = Path(str(transaction["temporary_staging"]))
    status = str(transaction.get("status") or "")
    if not final_output.exists() and not final_staging.exists():
        if status == "EXPORTING":
            exporter_process = transaction.get("exporter_process")
            if isinstance(exporter_process, Mapping) and _pid_identity_alive(
                exporter_process
            ):
                raise ReleaseBlocked("The owned exporter process remains live")
            try:
                _verify_export_tree(temporary_output, matrix_sha256)
                _verify_export_tree(temporary_staging, matrix_sha256)
            except (OSError, ReleaseSupervisorError, ValueError):
                transaction["status"] = "FAILED_PROCESS_LOSS"
                transaction["updated_at"] = utc_now()
                transaction["preserved_partial_output"] = str(temporary_output)
                transaction["preserved_partial_staging"] = str(temporary_staging)
                _atomic_json(transaction_path, transaction)
                return False
            _promote_export_transaction(
                spec=spec,
                matrix_sha256=matrix_sha256,
                temporary_output=temporary_output,
                temporary_staging=temporary_staging,
                transaction_path=transaction_path,
                transaction=transaction,
            )
            return True
        if status == "TEMP_EXPORT_VERIFIED":
            _promote_export_transaction(
                spec=spec,
                matrix_sha256=matrix_sha256,
                temporary_output=temporary_output,
                temporary_staging=temporary_staging,
                transaction_path=transaction_path,
                transaction=transaction,
            )
            return True
        if status == "PASS":
            raise ReleaseBlocked("PASS transaction lost both published destinations")
        return False
    if final_output.exists() and not final_staging.exists():
        _verify_export_tree(final_output, matrix_sha256)
        _verify_export_tree(temporary_staging, matrix_sha256)
        if _tree_inventory(final_output) != _tree_inventory(temporary_staging):
            raise ReleaseBlocked("Half-promoted staging tree differs from output")
        _rename_directory_noreplace(temporary_staging, final_staging)
        _fsync_directory(final_staging.parent)
    elif final_staging.exists() and not final_output.exists():
        raise ReleaseBlocked("Paper staging was promoted before the canonical output")
    _verify_export_tree(final_output, matrix_sha256)
    _verify_export_tree(final_staging, matrix_sha256)
    if _tree_inventory(final_output) != _tree_inventory(final_staging):
        raise ReleaseBlocked("Published output and staging trees differ")
    transaction["status"] = "PASS"
    transaction["updated_at"] = utc_now()
    _atomic_json(transaction_path, transaction)
    return True


def _export_once(
    *,
    spec: Mapping[str, Any],
    spec_sha256: str,
    state_root: Path,
    registry_root: Path,
    heartbeat: Callable[[], None],
    runner: Callable[..., int] = _run_exporter_process,
) -> None:
    matrix_path = registry_root / "matrix_status.json"
    matrix_sha = sha256_file(matrix_path)
    runtime_spec = dict(spec)
    runtime_spec["_runtime_spec_sha256"] = spec_sha256
    if _reconcile_export(
        spec=runtime_spec, state_root=state_root, matrix_sha256=matrix_sha
    ):
        return
    transaction_path = state_root / "release_transaction.json"
    previous_attempt = 0
    if transaction_path.is_file():
        previous = _read_json(transaction_path, label="release transaction")
        if previous.get("spec_sha256") != spec_sha256:
            raise ReleaseBlocked("Existing release transaction belongs to another spec")
        previous_attempt = int(previous.get("attempt", 0))
        if previous.get("status") not in {"FAILED_PROCESS_LOSS", "FAILED_EXPORT"}:
            raise ReleaseBlocked(
                f"Unreconciled release transaction: {previous.get('status')}"
            )
    attempt = previous_attempt + 1
    if attempt > int(spec.get("max_export_attempts", MAX_EXPORT_ATTEMPTS)):
        raise ReleaseBlocked("Bounded exporter attempts are exhausted")
    transaction_id = f"{spec_sha256[:12]}-a{attempt}"
    temporary_output, temporary_staging = _transaction_paths(spec, transaction_id)
    if temporary_output.exists() or temporary_staging.exists():
        raise ReleaseBlocked("Owned fresh export transaction root already exists")
    transaction = {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "status": "EXPORTING",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "controller_id": spec["controller_id"],
        "spec_sha256": spec_sha256,
        "attempt": attempt,
        "matrix_status_sha256": matrix_sha,
        "temporary_output": str(temporary_output),
        "temporary_staging": str(temporary_staging),
        "final_output": spec["paths"]["output_root"],
        "final_staging": spec["paths"]["paper_staging_root"],
    }
    _atomic_json(transaction_path, transaction)
    def process_started(identity: Mapping[str, Any]) -> None:
        transaction["exporter_process"] = dict(identity)
        transaction["updated_at"] = utc_now()
        _atomic_json(transaction_path, transaction)

    try:
        return_code = runner(
            spec=spec,
            registry_root=registry_root,
            temporary_output=temporary_output,
            temporary_staging=temporary_staging,
            log_path=state_root / f"export-attempt-{attempt}.log",
            heartbeat=heartbeat,
            process_started=process_started,
        )
    except Exception as exc:
        identity = transaction.get("exporter_process")
        if not isinstance(identity, Mapping) or not _pid_identity_alive(identity):
            transaction["status"] = "FAILED_EXPORT"
        transaction["exception"] = f"{type(exc).__name__}: {exc}"
        transaction["updated_at"] = utc_now()
        _atomic_json(transaction_path, transaction)
        raise
    transaction["return_code"] = return_code
    transaction["updated_at"] = utc_now()
    if return_code != 0:
        transaction["status"] = "FAILED_EXPORT"
        _atomic_json(transaction_path, transaction)
        raise ReleaseBlocked(f"Three-dataset exporter exited {return_code}")
    _promote_export_transaction(
        spec=spec,
        matrix_sha256=matrix_sha,
        temporary_output=temporary_output,
        temporary_staging=temporary_staging,
        transaction_path=transaction_path,
        transaction=transaction,
    )


def _linux_process_identity(pid: int) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise ReleaseSpecError("Persistent release supervision requires Linux procfs")
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    raw_stat = stat_path.read_text(encoding="utf-8")
    close = raw_stat.rfind(")")
    if close < 0:
        raise ReleaseSpecError("Malformed procfs stat record")
    remainder = raw_stat[close + 2 :].split()
    if len(remainder) < 20:
        raise ReleaseSpecError("Incomplete procfs stat record")
    return {
        "pid": pid,
        "boot_id": boot_path.read_text(encoding="utf-8").strip(),
        "start_time_ticks": int(remainder[19]),
        "cmdline_sha256": _digest_bytes(cmdline_path.read_bytes()),
    }


def _pid_identity_alive(identity: Mapping[str, Any]) -> bool:
    try:
        current = _linux_process_identity(int(identity.get("pid", -1)))
    except (OSError, ValueError, ReleaseSpecError):
        return False
    return all(
        current.get(key) == identity.get(key)
        for key in ("pid", "boot_id", "start_time_ticks", "cmdline_sha256")
    )


class ReleaseSupervisor:
    """One process-bound, restartable controller for the staging release."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        process_identity: Mapping[str, Any] | None = None,
        exporter_runner: Callable[..., int] = _run_exporter_process,
    ) -> None:
        path, digest, payload = load_release_spec(spec_path)
        self.spec_path = path
        self.spec_sha256 = digest
        self.spec = payload
        self.state_root = Path(payload["paths"]["state_root"])
        self.process_identity = dict(
            process_identity or _linux_process_identity(os.getpid())
        )
        self.exporter_runner = exporter_runner
        self._lock_handle: Any | None = None
        self._stop_requested = False
        self._heartbeat_sequence = 0
        self._last_result = TickResult("STARTING", 0, ())

    def _acquire_lock(self) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_root / "supervisor.lock"
        handle = lock_path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ReleaseBlocked("Another release supervisor owns the control lock") from exc
        self._lock_handle = handle

    def _initialize_control(self) -> None:
        controller_path = self.state_root / "controller.json"
        control = {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "controller_id": self.spec["controller_id"],
            "spec_path": str(self.spec_path),
            "spec_sha256": self.spec_sha256,
            "execution_commit": self.spec["execution_commit"],
            "resource": "cpu",
            "gpu_required": False,
        }
        restart_count = 0
        if controller_path.exists():
            observed = _read_json(controller_path, label="controller identity")
            for key, value in control.items():
                if observed.get(key) != value:
                    raise ReleaseBlocked(f"Controller identity drift for {key}")
            restart_count = int(observed.get("restart_count", 0)) + 1
            previous_pid_path = self.state_root / "pid.json"
            if previous_pid_path.is_file():
                previous = _read_json(previous_pid_path, label="previous PID")
                if _pid_identity_alive(previous.get("process") or {}):
                    raise ReleaseBlocked("A prior PID identity remains live")
            previous_heartbeat = self.state_root / "heartbeat.json"
            if previous_heartbeat.is_file():
                heartbeat = _read_json(previous_heartbeat, label="previous heartbeat")
                if heartbeat.get("spec_sha256") != self.spec_sha256:
                    raise ReleaseBlocked("Previous heartbeat belongs to another spec")
                self._heartbeat_sequence = int(heartbeat.get("sequence", 0))
        else:
            control["created_at"] = utc_now()
        control["restart_count"] = restart_count
        control["last_started_at"] = utc_now()
        _atomic_json(controller_path, control)
        _atomic_json(
            self.state_root / "pid.json",
            {
                "schema_version": CONTROL_SCHEMA_VERSION,
                "controller_id": self.spec["controller_id"],
                "spec_sha256": self.spec_sha256,
                "process": self.process_identity,
                "started_at": utc_now(),
            },
        )

    def _heartbeat(self) -> None:
        self._heartbeat_sequence += 1
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "controller_id": self.spec["controller_id"],
            "spec_sha256": self.spec_sha256,
            "timestamp": utc_now(),
            "sequence": self._heartbeat_sequence,
            "state": self._last_result.state,
            "complete_cells": self._last_result.complete_cells,
            "matrix_total_cells": 16,
            "process": self.process_identity,
        }
        _atomic_json(self.state_root / "heartbeat.json", payload)

    def _publish_state(self, result: TickResult) -> None:
        self._last_result = result
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "controller_id": self.spec["controller_id"],
            "spec_sha256": self.spec_sha256,
            "updated_at": utc_now(),
            "state": result.state,
            "complete_cells": result.complete_cells,
            "matrix_total_cells": 16,
            "reason": result.reason,
            "gpu_required": False,
            "gpu_lock_acquired": False,
            "run_tastemolnet": 0,
            "paper_status": "PAPER_FROZEN_PARTIAL",
            "probes": [probe.__dict__ for probe in result.probes],
            "registry_root": self.spec["paths"]["registry_root"],
            "output_root": self.spec["paths"]["output_root"],
            "paper_staging_root": self.spec["paths"]["paper_staging_root"],
        }
        _atomic_json(self.state_root / "state.json", payload)
        self._heartbeat()

    def _install_signal_handlers(self) -> None:
        def stop(_signum: int, _frame: Any) -> None:
            self._stop_requested = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def __enter__(self) -> "ReleaseSupervisor":
        self._acquire_lock()
        try:
            self._initialize_control()
            self._install_signal_handlers()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def tick(self) -> TickResult:
        _path, digest, current = load_release_spec(self.spec_path)
        if digest != self.spec_sha256 or current != self.spec:
            raise ReleaseBlocked("Release spec bytes changed after startup")
        probes = probe_cells(self.spec)
        ready = sum(probe.state == "READY_FOR_FULL_AUDIT" for probe in probes)
        try:
            _verify_bound_inputs(self.spec)
        except (OSError, ReleaseSupervisorError, ValueError) as exc:
            result = TickResult(
                "BLOCKED_BOUND_INPUT_DRIFT",
                ready,
                probes,
                reason=f"{type(exc).__name__}: {exc}",
            )
            self._publish_state(result)
            return result
        blocked = [probe for probe in probes if probe.state.startswith("BLOCKED_")]
        if blocked:
            result = TickResult(
                "BLOCKED_INVALID_CELL_ROOT",
                ready,
                probes,
                reason="; ".join(
                    f"{item.dataset}/{item.method}:{item.state}" for item in blocked
                ),
            )
            self._publish_state(result)
            return result
        if ready < 12:
            result = TickResult(
                "WAITING_DEPENDENCY",
                ready,
                probes,
                reason="No matrix or numeric release output is created before 12/16",
            )
            self._publish_state(result)
            return result
        try:
            registry_path = Path(self.spec["paths"]["registry_root"])
            if not registry_path.exists():
                result = _audit_registry(self.spec)
                if result.matrix_complete_cells != 12:
                    reasons = [
                        f"{row['dataset']}/{row['method']}={row['status']}:"
                        f"{row.get('rerun_reason') or ''}"
                        for row in result.matrix_rows
                        if row["dataset"] in DATASET_ORDER
                        and row["status"] not in {status.value for status in PASS_STATUSES}
                    ]
                    waiting = TickResult(
                        "WAITING_DEPENDENCY",
                        result.matrix_complete_cells,
                        probes,
                        reason=(
                            "; ".join(reasons)
                            or "Full registry audit has fewer than 12 PASS cells"
                        ),
                    )
                    self._publish_state(waiting)
                    return waiting
                matrix = _matrix_payload(result)
                _validate_three_dataset_boundary(matrix)
                _publish_registry(self.spec, self.spec_sha256, result)
            registry_root, _manifest = verify_published_registry(
                self.spec, self.spec_sha256
            )
            exporting = TickResult("EXPORTING", 12, probes)
            self._publish_state(exporting)
            _export_once(
                spec=self.spec,
                spec_sha256=self.spec_sha256,
                state_root=self.state_root,
                registry_root=registry_root,
                heartbeat=self._heartbeat,
                runner=self.exporter_runner,
            )
            matrix_sha = sha256_file(registry_root / "matrix_status.json")
            _verify_export_tree(Path(self.spec["paths"]["output_root"]), matrix_sha)
            _verify_export_tree(
                Path(self.spec["paths"]["paper_staging_root"]), matrix_sha
            )
            passed = TickResult("PASS", 12, probes)
            self._publish_state(passed)
            pass_payload = {
                "schema_version": STATE_SCHEMA_VERSION,
                "status": "PASS",
                "controller_id": self.spec["controller_id"],
                "spec_sha256": self.spec_sha256,
                "matrix_complete_cells": 12,
                "matrix_total_cells": 16,
                "taste_status": TASTE_BLOCKED_REASON,
                "paper_status": "PAPER_FROZEN_PARTIAL",
                "registry_manifest_sha256": sha256_file(
                    registry_root / "release_registry_manifest.json"
                ),
                "output_manifest_sha256": sha256_file(
                    Path(self.spec["paths"]["output_root"])
                    / "three_dataset_export_manifest.json"
                ),
            }
            release_pass = self.state_root / "RELEASE_PASS.json"
            if release_pass.is_file():
                if _read_json(release_pass, label="release PASS") != pass_payload:
                    raise ReleaseBlocked("Immutable release PASS marker drifted")
            else:
                _fresh_json(release_pass, pass_payload)
            return passed
        except (MainResultsError, OSError, ReleaseSupervisorError, ValueError) as exc:
            blocked_result = TickResult(
                "BLOCKED_RELEASE_GATE",
                ready,
                probes,
                reason=f"{type(exc).__name__}: {exc}",
            )
            self._publish_state(blocked_result)
            return blocked_result

    def run(self, *, once: bool = False) -> int:
        while True:
            if self._stop_requested:
                stopped = TickResult(
                    "STOPPED",
                    self._last_result.complete_cells,
                    self._last_result.probes,
                    reason="Graceful signal received; restart with the same immutable spec",
                )
                self._publish_state(stopped)
                return 0
            self.tick()
            if once:
                return 0 if self._last_result.state in {"PASS", "WAITING_DEPENDENCY"} else 3
            if self._stop_requested:
                stopped = TickResult(
                    "STOPPED",
                    self._last_result.complete_cells,
                    self._last_result.probes,
                    reason="Graceful signal received; restart with the same immutable spec",
                )
                self._publish_state(stopped)
                return 0
            deadline = time.monotonic() + POLL_INTERVAL_SECONDS
            while time.monotonic() < deadline and not self._stop_requested:
                time.sleep(min(2.0, deadline - time.monotonic()))


def read_supervisor_status(state_root: str | Path) -> dict[str, Any]:
    root = _absolute(state_root, label="state_root", existing=True)
    result: dict[str, Any] = {"state_root": str(root)}
    for name in (
        "controller.json",
        "pid.json",
        "heartbeat.json",
        "state.json",
        "release_transaction.json",
        "RELEASE_PASS.json",
    ):
        path = root / name
        if path.is_file() and not path.is_symlink():
            result[name] = _read_json(path, label=name)
    pid_record = result.get("pid.json")
    result["pid_identity_live"] = bool(
        isinstance(pid_record, Mapping)
        and _pid_identity_alive(pid_record.get("process") or {})
    )
    return result


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "EXPECTED_CELLS",
    "FROZEN_V4_CELLS",
    "POLL_INTERVAL_SECONDS",
    "ReleaseBlocked",
    "ReleaseSpecError",
    "ReleaseSupervisor",
    "TickResult",
    "build_release_spec",
    "load_cell_catalog",
    "load_release_spec",
    "probe_cells",
    "read_supervisor_status",
    "verify_closure_snapshot",
    "write_release_spec",
]
