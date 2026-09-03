"""Immutable task specifications for one-shot AutoDL main-table dispatch.

This module is intentionally small.  It validates a fully resolved command and
its scientific inputs; it is not a scheduler and it never mutates the matrix.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence
from uuid import UUID


SCHEMA = "main_ready_task_spec_v1"
MANIFEST_SCHEMA = "main_ready_task_specs_manifest_v1"
POINTER_SCHEMA = "main_ready_task_specs_pointer_v1"
OWNER_SCHEMA = "main_ready_task_owner_evidence_v1"

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "task_kind",
        "attempt_uuid",
        "repo_root",
        "execution_commit",
        "python",
        "entrypoint",
        "config_path",
        "config_sha256",
        "manifest_path",
        "manifest_sha256",
        "input_roots",
        "input_hashes",
        "output_root",
        "gpu_request",
        "cpu_request",
        "memory_request",
        "required_environment",
        "matrix_authority_root",
        "expected_owner_command_sha256",
        "expected_heartbeat_path",
        "expected_pid_file",
        "resume_policy",
        "single_writer_policy",
        "created_at",
        "spec_sha256",
    }
)
OPTIONAL_FIELDS = frozenset(
    {
        "arguments",
        "science_contract",
        "owner_timeout_seconds",
        "owner_probe",
        "expected_terminal_path",
    }
)
TASK_SPEC_PATH_TOKEN = "{task_spec_path}"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class MainReadyTaskSpecError(RuntimeError):
    """A bound main-table task is incomplete or ambiguous."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(dict(value)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _absolute(value: Any, *, field: str) -> Path:
    if not isinstance(value, str):
        raise MainReadyTaskSpecError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise MainReadyTaskSpecError(f"{field} must be normalized and absolute")
    return path


def materialize_task_spec_path(raw: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Replace the one descriptor-only spec-path token before self hashing.

    A descriptor cannot contain its final file hash, but it can safely contain
    its already-determined final path.  Materializing that path before sealing
    prevents T12-style owners from accidentally receiving the descriptor path
    or an unresolved token in ``/proc/<pid>/cmdline``.
    """

    published = _absolute(str(path), field="published task spec")

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(TASK_SPEC_PATH_TOKEN, str(published))
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, tuple):
            return [replace(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): replace(item) for key, item in value.items()}
        return value

    result = replace(dict(raw))
    if TASK_SPEC_PATH_TOKEN.encode("ascii") in canonical_bytes(result):
        raise MainReadyTaskSpecError("task-spec path token was not fully materialized")
    return result


def command_from_spec(spec: Mapping[str, Any]) -> list[str]:
    arguments = spec.get("arguments", [])
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise MainReadyTaskSpecError("arguments must be a string list")
    entrypoint = str(_absolute(spec.get("entrypoint"), field="entrypoint"))
    python = str(_absolute(spec.get("python"), field="python"))
    command = [python, "-I", "-B", entrypoint]
    config = str(_absolute(spec.get("config_path"), field="config_path"))
    if not any(item == "--config" or item.startswith("--config=") for item in arguments):
        command.extend(["--config", config])
    command.extend(arguments)
    return command


def owner_command_sha256(spec: Mapping[str, Any]) -> str:
    return stable_sha256(command_from_spec(spec))


def validate_spec(raw: Mapping[str, Any], *, check_files: bool = True) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MainReadyTaskSpecError("task spec must be a JSON object")
    value = dict(raw)
    keys = set(value)
    if not REQUIRED_FIELDS.issubset(keys) or keys - REQUIRED_FIELDS - OPTIONAL_FIELDS:
        missing = sorted(REQUIRED_FIELDS - keys)
        extra = sorted(keys - REQUIRED_FIELDS - OPTIONAL_FIELDS)
        raise MainReadyTaskSpecError(f"task spec keys differ: missing={missing}, extra={extra}")
    if value.get("schema_version") != SCHEMA:
        raise MainReadyTaskSpecError("task spec schema changed")
    for field in ("task_id", "task_kind", "attempt_uuid", "resume_policy", "single_writer_policy"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise MainReadyTaskSpecError(f"{field} must be a non-empty string")
    if _TASK_ID.fullmatch(value["task_id"]) is None:
        raise MainReadyTaskSpecError("task_id is not a safe path component")
    try:
        attempt_uuid = UUID(value["attempt_uuid"])
    except (ValueError, TypeError, AttributeError) as exc:
        raise MainReadyTaskSpecError("attempt_uuid must be one canonical UUIDv4") from exc
    if attempt_uuid.version != 4 or str(attempt_uuid) != value["attempt_uuid"]:
        raise MainReadyTaskSpecError("attempt_uuid must be one canonical UUIDv4")
    if _GIT_SHA.fullmatch(str(value.get("execution_commit") or "")) is None:
        raise MainReadyTaskSpecError("execution_commit must be one full Git SHA")
    for field in (
        "repo_root",
        "python",
        "entrypoint",
        "config_path",
        "manifest_path",
        "output_root",
        "matrix_authority_root",
        "expected_heartbeat_path",
        "expected_pid_file",
    ):
        _absolute(value.get(field), field=field)
    roots = value.get("input_roots")
    hashes = value.get("input_hashes")
    if not isinstance(roots, Mapping) or not roots:
        raise MainReadyTaskSpecError("input_roots must be a non-empty mapping")
    if not isinstance(hashes, Mapping) or set(hashes) != set(roots):
        raise MainReadyTaskSpecError("input hashes must exactly bind input roots")
    for key, path in roots.items():
        if not isinstance(key, str) or not key:
            raise MainReadyTaskSpecError("input role is invalid")
        _absolute(path, field=f"input_roots.{key}")
        if _SHA256.fullmatch(str(hashes.get(key) or "")) is None:
            raise MainReadyTaskSpecError(f"input_hashes.{key} is not SHA-256")
    for field in ("config_sha256", "manifest_sha256", "expected_owner_command_sha256", "spec_sha256"):
        if _SHA256.fullmatch(str(value.get(field) or "")) is None:
            raise MainReadyTaskSpecError(f"{field} is not SHA-256")
    environment = value.get("required_environment")
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(item, str)
        for key, item in environment.items()
    ):
        raise MainReadyTaskSpecError("required_environment must be a string mapping")
    for field in ("gpu_request", "cpu_request", "memory_request"):
        if not isinstance(value.get(field), Mapping):
            raise MainReadyTaskSpecError(f"{field} must be a mapping")
    if "science_contract" in value and not isinstance(value["science_contract"], Mapping):
        raise MainReadyTaskSpecError("science_contract must be a mapping")
    timeout = value.get("owner_timeout_seconds", 120)
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise MainReadyTaskSpecError("owner_timeout_seconds must be positive")
    if TASK_SPEC_PATH_TOKEN.encode("ascii") in canonical_bytes(value):
        raise MainReadyTaskSpecError("task spec contains an unresolved descriptor token")
    owner_probe = value.get("owner_probe", {})
    if not isinstance(owner_probe, Mapping):
        raise MainReadyTaskSpecError("owner_probe must be a mapping")
    allowed_probe = {
        "heartbeat_pid_field",
        "heartbeat_start_ticks_field",
        "heartbeat_task_id_field",
        "heartbeat_output_root_field",
        "heartbeat_timestamp_field",
        "max_age_seconds",
        "expected_cwd",
        "heartbeat_bindings",
    }
    if set(owner_probe) - allowed_probe:
        raise MainReadyTaskSpecError("owner_probe contains unsupported fields")
    for field in (
        "heartbeat_pid_field",
        "heartbeat_start_ticks_field",
        "heartbeat_timestamp_field",
    ):
        if field in owner_probe and (
            not isinstance(owner_probe[field], str) or not owner_probe[field]
        ):
            raise MainReadyTaskSpecError(f"owner_probe.{field} must be a non-empty string")
    for field in ("heartbeat_task_id_field", "heartbeat_output_root_field"):
        if field in owner_probe and owner_probe[field] is not None and (
            not isinstance(owner_probe[field], str) or not owner_probe[field]
        ):
            raise MainReadyTaskSpecError(f"owner_probe.{field} must be a string or null")
    if "expected_cwd" in owner_probe:
        _absolute(owner_probe["expected_cwd"], field="owner_probe.expected_cwd")
    max_age = owner_probe.get("max_age_seconds", value.get("owner_timeout_seconds", 120))
    if not isinstance(max_age, (int, float)) or isinstance(max_age, bool) or max_age <= 0:
        raise MainReadyTaskSpecError("owner heartbeat max age must be positive")
    bindings = owner_probe.get("heartbeat_bindings", {})
    if not isinstance(bindings, Mapping) or any(
        not isinstance(key, str) or not key for key in bindings
    ):
        raise MainReadyTaskSpecError("owner heartbeat bindings are invalid")
    terminal = value.get("expected_terminal_path")
    if terminal is not None:
        _absolute(terminal, field="expected_terminal_path")
    if value.get("single_writer_policy") != "fail_if_live_owner_or_output_writer":
        raise MainReadyTaskSpecError("single-writer policy is not fail closed")
    unsigned = {key: item for key, item in value.items() if key != "spec_sha256"}
    if value["spec_sha256"] != stable_sha256(unsigned):
        raise MainReadyTaskSpecError("task spec self hash changed")
    if value["expected_owner_command_sha256"] != owner_command_sha256(value):
        raise MainReadyTaskSpecError("bound owner command changed")
    if check_files:
        repo = _absolute(value["repo_root"], field="repo_root")
        python = _absolute(value["python"], field="python")
        entrypoint = _absolute(value["entrypoint"], field="entrypoint")
        config = _absolute(value["config_path"], field="config_path")
        manifest = _absolute(value["manifest_path"], field="manifest_path")
        if not repo.is_dir() or repo.is_symlink() or repo.resolve(strict=True) != repo:
            raise MainReadyTaskSpecError(f"repo input is absent or indirect: {repo}")
        # Conda intentionally exposes ``bin/python`` as a symlink to its exact
        # environment interpreter.  Rejecting that standard path made every
        # otherwise valid AutoDL descriptor impossible to seal.
        try:
            resolved_python = python.resolve(strict=True)
        except OSError as exc:
            raise MainReadyTaskSpecError(f"python input is absent: {python}") from exc
        if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
            raise MainReadyTaskSpecError(f"python input is not executable: {python}")
        for path, label in (
            (entrypoint, "entrypoint"),
            (config, "config"),
            (manifest, "manifest"),
        ):
            if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
                raise MainReadyTaskSpecError(f"{label} input is absent or indirect: {path}")
        try:
            entrypoint.relative_to(repo)
            config.relative_to(repo)
        except ValueError as exc:
            raise MainReadyTaskSpecError(
                "entrypoint and config must belong to the execution worktree"
            ) from exc
        try:
            head = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise MainReadyTaskSpecError("cannot verify execution worktree HEAD") from exc
        if head != value["execution_commit"]:
            raise MainReadyTaskSpecError("execution worktree HEAD differs from spec")
        if file_sha256(config) != value["config_sha256"]:
            raise MainReadyTaskSpecError("config bytes changed")
        if file_sha256(manifest) != value["manifest_sha256"]:
            raise MainReadyTaskSpecError("manifest bytes changed")
        for role, path in roots.items():
            physical = _absolute(path, field=f"input_roots.{role}")
            if not physical.exists() or physical.is_symlink():
                raise MainReadyTaskSpecError(f"input root is absent or indirect: {role}")
    return value


def load_spec(path: Path, *, check_files: bool = True) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        raise MainReadyTaskSpecError("task spec path must be absolute and non-symlink")
    try:
        if path.resolve(strict=True) != path or not path.is_file():
            raise MainReadyTaskSpecError("task spec path is absent or indirect")
    except OSError as exc:
        raise MainReadyTaskSpecError(f"cannot resolve task spec: {path}") from exc
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MainReadyTaskSpecError(f"cannot read task spec: {path}") from exc
    return validate_spec(value, check_files=check_files)


def seal_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    value["schema_version"] = SCHEMA
    value["expected_owner_command_sha256"] = "0" * 64
    value["spec_sha256"] = "0" * 64
    value["expected_owner_command_sha256"] = owner_command_sha256(value)
    value["spec_sha256"] = stable_sha256(
        {key: item for key, item in value.items() if key != "spec_sha256"}
    )
    return validate_spec(value, check_files=False)


def manifest_for_specs(
    paths: Sequence[Path], *, published_paths: Sequence[Path] | None = None
) -> dict[str, Any]:
    if not paths:
        raise MainReadyTaskSpecError("task-spec manifest cannot be empty")
    published = list(paths if published_paths is None else published_paths)
    if len(published) != len(paths):
        raise MainReadyTaskSpecError("published task-spec path count changed")
    rows = []
    matrix_authority = None
    task_ids: set[str] = set()
    for path, published_path in zip(paths, published, strict=True):
        spec = load_spec(path)
        if spec["task_id"] in task_ids:
            raise MainReadyTaskSpecError("task-spec manifest has duplicate task IDs")
        task_ids.add(spec["task_id"])
        normalized_published = _absolute(
            str(published_path), field="published task-spec path"
        )
        if matrix_authority is None:
            matrix_authority = spec["matrix_authority_root"]
        elif matrix_authority != spec["matrix_authority_root"]:
            raise MainReadyTaskSpecError("specs do not share one matrix authority")
        rows.append(
            {
                "task_id": spec["task_id"],
                "task_kind": spec["task_kind"],
                "path": str(normalized_published),
                "file_sha256": file_sha256(path),
                "spec_sha256": spec["spec_sha256"],
            }
        )
    value = {
        "schema_version": MANIFEST_SCHEMA,
        "matrix_authority_root": matrix_authority,
        "task_specs": rows,
    }
    value["manifest_sha256"] = stable_sha256(value)
    return value


def validate_manifest(
    raw: Mapping[str, Any], *, spec_paths: Sequence[Path] | None = None
) -> dict[str, Any]:
    """Validate one immutable task-spec inventory and, optionally, its files."""

    if not isinstance(raw, Mapping):
        raise MainReadyTaskSpecError("task-spec manifest must be a JSON object")
    value = dict(raw)
    required = {
        "schema_version",
        "matrix_authority_root",
        "task_specs",
        "manifest_sha256",
    }
    if set(value) != required or value.get("schema_version") != MANIFEST_SCHEMA:
        raise MainReadyTaskSpecError("task-spec manifest schema or keys changed")
    _absolute(value.get("matrix_authority_root"), field="matrix_authority_root")
    rows = value.get("task_specs")
    if not isinstance(rows, list) or not rows:
        raise MainReadyTaskSpecError("task-spec manifest rows are absent")
    task_ids: set[str] = set()
    expected_paths = None if spec_paths is None else [str(path) for path in spec_paths]
    if expected_paths is not None and len(expected_paths) != len(rows):
        raise MainReadyTaskSpecError("task-spec manifest path count changed")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "task_id",
            "task_kind",
            "path",
            "file_sha256",
            "spec_sha256",
        }:
            raise MainReadyTaskSpecError("task-spec manifest row schema changed")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
            raise MainReadyTaskSpecError("task-spec manifest task ID is invalid")
        if task_id in task_ids:
            raise MainReadyTaskSpecError("task-spec manifest has duplicate task IDs")
        task_ids.add(task_id)
        path = _absolute(row.get("path"), field="task-spec manifest path")
        for field in ("file_sha256", "spec_sha256"):
            if _SHA256.fullmatch(str(row.get(field) or "")) is None:
                raise MainReadyTaskSpecError(f"task-spec manifest {field} is invalid")
        if expected_paths is not None and str(path) != expected_paths[index]:
            raise MainReadyTaskSpecError("task-spec manifest path ordering changed")
        if spec_paths is not None:
            spec = load_spec(path)
            if (
                file_sha256(path) != row["file_sha256"]
                or spec["spec_sha256"] != row["spec_sha256"]
                or spec["task_id"] != task_id
                or spec["task_kind"] != row["task_kind"]
                or spec["matrix_authority_root"] != value["matrix_authority_root"]
            ):
                raise MainReadyTaskSpecError("task-spec manifest binding changed")
    claimed = value.get("manifest_sha256")
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if claimed != stable_sha256(unsigned):
        raise MainReadyTaskSpecError("task-spec manifest self hash changed")
    return value


def load_manifest(
    path: Path, *, spec_paths: Sequence[Path] | None = None
) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MainReadyTaskSpecError("task-spec manifest path is absent or indirect")
    if path.resolve(strict=True) != path:
        raise MainReadyTaskSpecError("task-spec manifest path is indirect")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MainReadyTaskSpecError("cannot read task-spec manifest") from exc
    return validate_manifest(raw, spec_paths=spec_paths)


def seal_pointer(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    value["schema_version"] = POINTER_SCHEMA
    value.pop("pointer_sha256", None)
    value["pointer_sha256"] = stable_sha256(value)
    return validate_pointer(value, check_files=False)


def validate_pointer(raw: Mapping[str, Any], *, check_files: bool = True) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MainReadyTaskSpecError("task-spec pointer must be a JSON object")
    value = dict(raw)
    required = {
        "schema_version",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_sha256",
        "sidecar_control_root",
        "sidecar_pid",
        "sidecar_start_ticks",
        "sidecar_command_sha256",
        "published_at",
        "pointer_sha256",
    }
    if set(value) != required or value.get("schema_version") != POINTER_SCHEMA:
        raise MainReadyTaskSpecError("task-spec pointer schema or keys changed")
    manifest = _absolute(value.get("manifest_path"), field="pointer manifest_path")
    _absolute(value.get("sidecar_control_root"), field="pointer sidecar_control_root")
    for field in (
        "manifest_file_sha256",
        "manifest_sha256",
        "sidecar_command_sha256",
        "pointer_sha256",
    ):
        if _SHA256.fullmatch(str(value.get(field) or "")) is None:
            raise MainReadyTaskSpecError(f"pointer {field} is invalid")
    for field in ("sidecar_pid", "sidecar_start_ticks"):
        if (
            not isinstance(value.get(field), int)
            or isinstance(value[field], bool)
            or value[field] <= 0
        ):
            raise MainReadyTaskSpecError(f"pointer {field} is invalid")
    try:
        age = _heartbeat_age(value.get("published_at"), now_epoch=time.time())
    except MainReadyTaskSpecError as exc:
        raise MainReadyTaskSpecError("pointer publication timestamp is invalid") from exc
    if age < -30:
        raise MainReadyTaskSpecError("pointer publication timestamp is in the future")
    unsigned = {key: item for key, item in value.items() if key != "pointer_sha256"}
    if value["pointer_sha256"] != stable_sha256(unsigned):
        raise MainReadyTaskSpecError("task-spec pointer self hash changed")
    if check_files:
        if (
            manifest.is_symlink()
            or not manifest.is_file()
            or manifest.resolve(strict=True) != manifest
        ):
            raise MainReadyTaskSpecError("pointer manifest is absent or indirect")
        if file_sha256(manifest) != value["manifest_file_sha256"]:
            raise MainReadyTaskSpecError("pointer manifest bytes changed")
        loaded = load_manifest(manifest)
        if loaded["manifest_sha256"] != value["manifest_sha256"]:
            raise MainReadyTaskSpecError("pointer manifest semantic hash changed")
    return value


def load_pointer(path: Path, *, check_files: bool = True) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MainReadyTaskSpecError("task-spec pointer path is absent or indirect")
    if path.resolve(strict=True) != path:
        raise MainReadyTaskSpecError("task-spec pointer path is indirect")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MainReadyTaskSpecError("cannot read task-spec pointer") from exc
    return validate_pointer(raw, check_files=check_files)


def process_identity(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, Any] | None:
    """Return a PID-reuse-safe Linux identity with robust ``stat`` parsing."""

    normalized = int(pid)
    if normalized <= 0:
        return None
    process = proc_root / str(normalized)
    try:
        raw_stat = (process / "stat").read_text(encoding="utf-8").strip()
        closing = raw_stat.rfind(")")
        tail = raw_stat[closing + 2 :].split() if closing >= 0 else []
        if len(tail) <= 19 or len(tail[0]) != 1 or not tail[19].isdigit():
            return None
        raw_command = (process / "cmdline").read_bytes()
        argv = [
            item.decode("utf-8", errors="surrogateescape")
            for item in raw_command.rstrip(b"\0").split(b"\0")
            if item
        ]
        if not argv:
            return None
        cwd = os.readlink(process / "cwd")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return {
        "pid": normalized,
        "state": tail[0],
        "alive": tail[0] != "Z",
        "start_ticks": int(tail[19]),
        "argv": argv,
        "command": " ".join(argv),
        "command_sha256": stable_sha256(argv),
        "cwd": cwd,
    }


def read_pid_file(path: Path) -> int | None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        return None
    try:
        if path.suffix == ".json":
            value: Any = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = path.read_text(encoding="utf-8").strip()
        if isinstance(value, Mapping):
            value = value.get("owner_pid") or value.get("pid") or value.get(
                "controller_pid"
            )
        return int(value) if int(value) > 0 else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def _heartbeat_age(value: Any, *, now_epoch: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return now_epoch - float(value)
    if not isinstance(value, str) or not value:
        raise MainReadyTaskSpecError("owner heartbeat timestamp is absent")
    from datetime import datetime

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MainReadyTaskSpecError("owner heartbeat timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise MainReadyTaskSpecError("owner heartbeat timestamp lacks a timezone")
    return now_epoch - parsed.timestamp()


def probe_owner(
    spec: Mapping[str, Any],
    *,
    proc_root: Path = Path("/proc"),
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Validate heartbeat, PID generation, cwd and exact owner command."""

    value = validate_spec(spec, check_files=False)
    pid_path = Path(value["expected_pid_file"])
    heartbeat_path = Path(value["expected_heartbeat_path"])
    pid = read_pid_file(pid_path)
    if pid is None:
        return {"state": "ABSENT", "reason": "OWNER_PID_ABSENT"}
    if (
        not heartbeat_path.is_absolute()
        or heartbeat_path.is_symlink()
        or not heartbeat_path.is_file()
    ):
        return {"state": "ABSENT", "reason": "OWNER_HEARTBEAT_ABSENT", "pid": pid}
    try:
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"state": "INVALID", "reason": f"HEARTBEAT_UNREADABLE: {exc}"}
    if not isinstance(heartbeat, Mapping):
        return {"state": "INVALID", "reason": "HEARTBEAT_NOT_OBJECT"}
    probe = dict(value.get("owner_probe") or {})
    pid_field = str(probe.get("heartbeat_pid_field", "owner_pid"))
    ticks_field = str(probe.get("heartbeat_start_ticks_field", "owner_start_ticks"))
    task_field = probe.get("heartbeat_task_id_field", "task_id")
    output_field = probe.get("heartbeat_output_root_field", "output_root")
    timestamp_field = str(probe.get("heartbeat_timestamp_field", "written_at"))
    try:
        heartbeat_pid = _nested(heartbeat, pid_field)
        heartbeat_ticks = _nested(heartbeat, ticks_field)
        if heartbeat_pid != pid:
            raise MainReadyTaskSpecError("heartbeat PID differs from PID file")
        if task_field is not None and _nested(heartbeat, str(task_field)) != value["task_id"]:
            raise MainReadyTaskSpecError("heartbeat task ID changed")
        if output_field is not None and _nested(
            heartbeat, str(output_field)
        ) != value["output_root"]:
            raise MainReadyTaskSpecError("heartbeat output root changed")
        for field, expected in dict(probe.get("heartbeat_bindings") or {}).items():
            if _nested(heartbeat, field) != expected:
                raise MainReadyTaskSpecError(f"heartbeat binding changed: {field}")
        age = _heartbeat_age(
            _nested(heartbeat, timestamp_field),
            now_epoch=time.time() if now_epoch is None else float(now_epoch),
        )
        max_age = float(
            probe.get("max_age_seconds", value.get("owner_timeout_seconds", 120))
        )
        if age < -30 or age > max_age:
            raise MainReadyTaskSpecError(f"owner heartbeat is stale: {age:.1f}s")
        identity = process_identity(pid, proc_root=proc_root)
        if identity is None or not identity["alive"]:
            return {"state": "ABSENT", "reason": "OWNER_PROCESS_EXITED", "pid": pid}
        if not isinstance(heartbeat_ticks, int) or isinstance(heartbeat_ticks, bool):
            raise MainReadyTaskSpecError("heartbeat start ticks are invalid")
        if identity["start_ticks"] != heartbeat_ticks:
            raise MainReadyTaskSpecError("owner PID start ticks changed")
        expected_cwd = str(probe.get("expected_cwd", value["repo_root"]))
        if identity["cwd"] != expected_cwd:
            raise MainReadyTaskSpecError("owner cwd changed")
        if identity["command_sha256"] != value["expected_owner_command_sha256"]:
            raise MainReadyTaskSpecError("owner command SHA256 changed")
    except (KeyError, TypeError, ValueError, MainReadyTaskSpecError) as exc:
        return {"state": "INVALID", "reason": str(exc), "pid": pid}
    return {
        "state": "OWNER_CONFIRMED",
        "pid": pid,
        "start_ticks": identity["start_ticks"],
        "command_sha256": identity["command_sha256"],
        "cwd": identity["cwd"],
        "heartbeat": str(heartbeat_path),
        "heartbeat_age_seconds": age,
    }


def probe_terminal(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_path = spec.get("expected_terminal_path")
    if raw_path is None:
        raw_path = str(Path(str(spec["expected_heartbeat_path"])).parent / "terminal.json")
    path = _absolute(raw_path, field="expected_terminal_path")
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"state": "INVALID", "reason": f"TERMINAL_UNREADABLE: {exc}"}
    if not isinstance(value, Mapping):
        return {"state": "INVALID", "reason": "TERMINAL_NOT_OBJECT"}
    if "task_id" in value and value["task_id"] != spec["task_id"]:
        return {"state": "INVALID", "reason": "TERMINAL_TASK_ID_CHANGED"}
    if "output_root" in value and value["output_root"] != spec["output_root"]:
        return {"state": "INVALID", "reason": "TERMINAL_OUTPUT_ROOT_CHANGED"}
    if "terminal_sha256" in value:
        claimed = value["terminal_sha256"]
        unsigned = {key: item for key, item in value.items() if key != "terminal_sha256"}
        canonical_newline = hashlib.sha256(canonical_bytes(unsigned) + b"\n").hexdigest()
        # Existing typed terminals use either the compact semantic hash or the
        # exact bytes emitted by ``atomic_json`` (which include one newline).
        if claimed not in {stable_sha256(unsigned), canonical_newline}:
            return {"state": "INVALID", "reason": "TERMINAL_SELF_HASH_CHANGED"}
    status = value.get("status", value.get("state"))
    if not isinstance(status, str) or not status:
        return {"state": "INVALID", "reason": "TERMINAL_STATUS_ABSENT"}
    return {"state": "TERMINAL", "status": status, "path": str(path), "payload": value}


def conflicting_output_writers(
    output_root: str | Path, *, proc_root: Path = Path("/proc")
) -> list[dict[str, Any]]:
    """Find processes whose argv, cwd, or open descriptors bind one output root."""

    root = _absolute(str(output_root), field="output_root")
    root_text = str(root)

    def inside(text: str) -> bool:
        return text == root_text or text.startswith(root_text + os.sep)

    conflicts: list[dict[str, Any]] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return conflicts
    for entry in entries:
        if not entry.name.isdigit():
            continue
        identity = process_identity(int(entry.name), proc_root=proc_root)
        if identity is None or not identity["alive"]:
            continue
        argv_match = any(
            inside(argument)
            or ("=" in argument and inside(argument.split("=", 1)[1]))
            for argument in identity["argv"]
        )
        cwd_match = inside(identity["cwd"])
        fd_match = False
        fd_root = entry / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except OSError:
            descriptors = []
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if inside(target.removesuffix(" (deleted)")):
                fd_match = True
                break
        if argv_match or cwd_match or fd_match:
            conflicts.append(
                {
                    **identity,
                    "argv_match": argv_match,
                    "cwd_match": cwd_match,
                    "fd_match": fd_match,
                }
            )
    return conflicts


__all__ = [
    "MANIFEST_SCHEMA",
    "MainReadyTaskSpecError",
    "OWNER_SCHEMA",
    "POINTER_SCHEMA",
    "REQUIRED_FIELDS",
    "SCHEMA",
    "atomic_json",
    "command_from_spec",
    "conflicting_output_writers",
    "file_sha256",
    "load_manifest",
    "load_pointer",
    "load_spec",
    "materialize_task_spec_path",
    "manifest_for_specs",
    "owner_command_sha256",
    "probe_owner",
    "probe_terminal",
    "process_identity",
    "read_pid_file",
    "seal_spec",
    "seal_pointer",
    "stable_sha256",
    "validate_manifest",
    "validate_pointer",
    "validate_spec",
]
