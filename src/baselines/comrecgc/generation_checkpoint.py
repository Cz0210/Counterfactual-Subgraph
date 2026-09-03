"""Atomic, fail-closed checkpoints for completed COMRECGC walk steps.

This module deliberately does not hook the pinned upstream random-walk loop.
It provides the persistence boundary that the runtime can call only after one
step, including its trace event, has completed successfully.
"""

from __future__ import annotations

import inspect
import json
import os
import random
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .contracts import atomic_write_bytes, sha256_file, stable_json_sha256, write_json


CHECKPOINT_SCHEMA_VERSION = "comrecgc_generation_checkpoint_v2"
CHECKPOINT_STATE_SCHEMA_VERSION = "comrecgc_generation_state_v2"
CHECKPOINT_BOUNDARY = "after_fully_completed_step_v1"
LATEST_SCHEMA_VERSION = "comrecgc_generation_checkpoint_latest_v1"
STATE_FILENAME = "generation_state.pt"
SQLITE_FILENAME = "authoritative_graph_store.sqlite3"
MANIFEST_FILENAME = "checkpoint_manifest.json"
COMPLETE_FILENAME = "_CHECKPOINT_COMPLETE.json"
MIRRORED_FILENAME = "_CHECKPOINT_MIRRORED.json"
LATEST_FILENAME = "LATEST"
PENDING_LATEST_FILENAME = "PENDING_LATEST.json"
RETENTION_HISTORY_DIRNAME = "retention_history"
_CHECKPOINT_NAME = re.compile(r"^step-(?P<step>[0-9]{12})$")


class GenerationCheckpointError(RuntimeError):
    """Raised when a generation checkpoint cannot be trusted or restored."""


@dataclass(frozen=True, slots=True)
class GenerationCheckpointValidation:
    checkpoint_dir: Path
    completed_step: int
    checkpoint_digest: str
    provenance_fingerprints: dict[str, str]
    scientific_argv: tuple[str, ...]
    command_sha256: str
    total_steps: int
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedGenerationCheckpoint:
    validation: GenerationCheckpointValidation
    algorithm_state: dict[str, Any]
    trace_state: dict[str, Any]
    rng_state: dict[str, Any]
    sqlite_snapshot_path: Path

    @property
    def completed_step(self) -> int:
        return self.validation.completed_step


def _torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - required HPC dependency
        raise GenerationCheckpointError(
            "COMRECGC exact checkpoints require PyTorch."
        ) from exc
    return torch


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GenerationCheckpointError(f"Invalid checkpoint JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GenerationCheckpointError(f"Expected checkpoint JSON object: {path}")
    return value


def _normalize_provenance(provenance: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(provenance, Mapping) or not provenance:
        raise GenerationCheckpointError(
            "Checkpoint provenance fingerprints must be a non-empty mapping."
        )
    normalized: dict[str, str] = {}
    for raw_key, raw_value in provenance.items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key or not value:
            raise GenerationCheckpointError(
                "Checkpoint provenance fingerprint keys and values must be non-empty."
            )
        if key in normalized:
            raise GenerationCheckpointError(
                f"Duplicate normalized checkpoint provenance key: {key!r}."
            )
        normalized[key] = value
    return dict(sorted(normalized.items()))


def _normalize_scientific_argv(scientific_argv: Any) -> tuple[str, ...]:
    if not isinstance(scientific_argv, (list, tuple)) or not scientific_argv:
        raise GenerationCheckpointError(
            "Checkpoint scientific argv must be a non-empty list of strings."
        )
    normalized = tuple(str(value) for value in scientific_argv)
    if any(not value or "\x00" in value for value in normalized):
        raise GenerationCheckpointError(
            "Checkpoint scientific argv contains an empty or invalid argument."
        )
    if any(value == "--resume" or value.startswith("--resume=") for value in normalized):
        raise GenerationCheckpointError(
            "Checkpoint scientific argv must exclude only the transport flag --resume."
        )
    return normalized


def scientific_command_sha256(scientific_argv: Any) -> str:
    """Hash the canonical, already-redacted scientific command contract."""

    normalized = _normalize_scientific_argv(scientific_argv)
    return stable_json_sha256(
        {
            "schema_version": "comrecgc_scientific_command_v1",
            "argv": list(normalized),
        }
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture global Python, NumPy, and PyTorch CPU/CUDA RNG state."""

    torch = _torch()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in cuda_states],
        "torch_cuda_device_count": len(cuda_states),
    }


def _validate_rng_state(
    rng_state: Mapping[str, Any], *, require_current_cuda_match: bool = False
) -> None:
    torch = _torch()
    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "torch_cuda_device_count",
    }
    missing = sorted(required - set(rng_state))
    if missing:
        raise GenerationCheckpointError(
            f"Checkpoint RNG state is incomplete: missing={missing}."
        )
    if not isinstance(rng_state["python"], tuple):
        raise GenerationCheckpointError("Checkpoint Python RNG state is malformed.")
    numpy_state = rng_state["numpy"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise GenerationCheckpointError("Checkpoint NumPy RNG state is malformed.")
    cpu_state = rng_state["torch_cpu"]
    if not torch.is_tensor(cpu_state) or cpu_state.dtype != torch.uint8:
        raise GenerationCheckpointError("Checkpoint torch CPU RNG state is malformed.")
    cuda_states = rng_state["torch_cuda"]
    if not isinstance(cuda_states, list) or any(
        not torch.is_tensor(state) or state.dtype != torch.uint8
        for state in cuda_states
    ):
        raise GenerationCheckpointError("Checkpoint torch CUDA RNG state is malformed.")
    stored_cuda_count = int(rng_state["torch_cuda_device_count"])
    if stored_cuda_count < 0 or stored_cuda_count != len(cuda_states):
        raise GenerationCheckpointError(
            "Checkpoint torch CUDA RNG state count is inconsistent."
        )
    try:
        random.Random().setstate(rng_state["python"])
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(numpy_state)
        torch.Generator(device="cpu").set_state(cpu_state)
    except Exception as exc:
        raise GenerationCheckpointError(
            "Checkpoint CPU RNG state cannot be restored exactly."
        ) from exc
    if require_current_cuda_match:
        current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if current_cuda_count != stored_cuda_count:
            raise GenerationCheckpointError(
                "Checkpoint CUDA RNG device count differs from the current process: "
                f"stored={stored_cuda_count}, current={current_cuda_count}."
            )
        try:
            for index, state in enumerate(cuda_states):
                torch.Generator(device=f"cuda:{index}").set_state(state)
        except Exception as exc:
            raise GenerationCheckpointError(
                "Checkpoint CUDA RNG state cannot be restored exactly."
            ) from exc


def restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    """Restore all captured RNG streams, rejecting a CUDA topology mismatch."""

    _validate_rng_state(rng_state, require_current_cuda_match=True)
    torch = _torch()
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if int(rng_state["torch_cuda_device_count"]):
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


@contextmanager
def _sqlite_source(
    source: sqlite3.Connection | str | Path,
) -> Iterator[sqlite3.Connection]:
    if isinstance(source, sqlite3.Connection):
        if source.in_transaction:
            raise GenerationCheckpointError(
                "SQLite checkpoint source has an open transaction; checkpoint only "
                "after the completed step has committed."
            )
        yield source
        return
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise GenerationCheckpointError(f"SQLite checkpoint source is missing: {path}")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=60.0)
    try:
        yield connection
    finally:
        connection.close()


def _inspect_sqlite(path: Path) -> dict[str, Any]:
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro&immutable=1",
            uri=True,
            timeout=60.0,
        )
        try:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            return {
                "integrity_check": integrity,
                "page_count": int(
                    connection.execute("PRAGMA page_count").fetchone()[0]
                ),
                "freelist_count": int(
                    connection.execute("PRAGMA freelist_count").fetchone()[0]
                ),
                "schema_version": int(
                    connection.execute("PRAGMA schema_version").fetchone()[0]
                ),
                "user_version": int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                ),
            }
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise GenerationCheckpointError(
            f"SQLite checkpoint cannot be audited: {path}"
        ) from exc


def _backup_sqlite(
    source: sqlite3.Connection | str | Path, destination: Path
) -> dict[str, Any]:
    if destination.exists():
        raise GenerationCheckpointError(
            f"SQLite checkpoint destination already exists: {destination}"
        )
    with _sqlite_source(source) as source_connection:
        destination_connection = sqlite3.connect(str(destination), timeout=60.0)
        try:
            source_connection.backup(destination_connection)
            destination_connection.commit()
            integrity = str(
                destination_connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
            if integrity != "ok":
                raise GenerationCheckpointError(
                    f"SQLite backup failed integrity_check: {integrity!r}."
                )
        finally:
            destination_connection.close()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{destination}{suffix}")
        if sidecar.exists():
            sidecar.unlink()
    _fsync_file(destination)
    return _inspect_sqlite(destination)


def _torch_save(payload: Mapping[str, Any], path: Path) -> None:
    _torch().save(dict(payload), path)
    _fsync_file(path)


def _copy_file_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _write_latest(root: Path, validation: GenerationCheckpointValidation) -> None:
    atomic_write_bytes(
        root / LATEST_FILENAME,
        (
            json.dumps(
                {
                    "schema_version": LATEST_SCHEMA_VERSION,
                    "checkpoint_dir": validation.checkpoint_dir.name,
                    "completed_step": validation.completed_step,
                    "checkpoint_digest": validation.checkpoint_digest,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_pending_latest(
    root: Path, validation: GenerationCheckpointValidation
) -> None:
    atomic_write_bytes(
        root / PENDING_LATEST_FILENAME,
        (
            json.dumps(
                {
                    "schema_version": "comrecgc_generation_checkpoint_pending_v1",
                    "checkpoint_dir": validation.checkpoint_dir.name,
                    "completed_step": validation.completed_step,
                    "checkpoint_digest": validation.checkpoint_digest,
                    "payload_reload_state": "PENDING_INDEPENDENT_RELOAD",
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _torch_load(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    torch = _torch()
    try:
        parameters = inspect.signature(torch.load).parameters
        options: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in parameters:
            options["weights_only"] = False
        if mmap and "mmap" in parameters:
            options["mmap"] = True
        value = torch.load(path, **options)
    except Exception as exc:
        raise GenerationCheckpointError(
            f"Checkpoint state payload cannot be loaded: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise GenerationCheckpointError("Checkpoint state payload must be a dictionary.")
    return value


def _checkpoint_digest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "checkpoint_digest"}


def _resolve_reference(
    checkpoint_root_or_dir: str | Path,
    *,
    validate_state_payload: bool,
) -> tuple[Path, dict[str, Any] | None]:
    raw = Path(checkpoint_root_or_dir).expanduser()
    if raw.is_symlink():
        raise GenerationCheckpointError(
            f"Checkpoint reference must not be a symbolic link: {raw}"
        )
    path = raw.resolve()
    if (path / MANIFEST_FILENAME).is_file():
        return path, None
    latest_path = path / LATEST_FILENAME
    latest: dict[str, Any] | None = None
    ignored: list[dict[str, str]] = []
    if latest_path.is_file() and not latest_path.is_symlink():
        try:
            latest = _json_object(latest_path)
        except GenerationCheckpointError as exc:
            ignored.append({"entry": LATEST_FILENAME, "reason": str(exc)})
    elif latest_path.exists() or latest_path.is_symlink():
        ignored.append(
            {"entry": LATEST_FILENAME, "reason": "pointer is not a physical file"}
        )

    valid: list[GenerationCheckpointValidation] = []
    if not path.is_dir():
        raise GenerationCheckpointError(f"Checkpoint root is missing: {path}")
    for candidate in sorted(path.iterdir(), key=lambda item: item.name):
        if candidate.name.startswith(".step-") and candidate.name.endswith(".tmp"):
            ignored.append(
                {
                    "entry": candidate.name,
                    "reason": "incomplete atomic staging directory ignored",
                }
            )
            continue
        if _CHECKPOINT_NAME.fullmatch(candidate.name) is None:
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            ignored.append(
                {"entry": candidate.name, "reason": "not a physical directory"}
            )
            continue
        try:
            valid.append(
                validate_generation_checkpoint(
                    candidate,
                    _validate_state_payload=validate_state_payload,
                )
            )
        except (GenerationCheckpointError, OSError, ValueError) as exc:
            ignored.append({"entry": candidate.name, "reason": str(exc)})
    if not valid:
        raise GenerationCheckpointError(
            "Checkpoint root has no complete valid checkpoint: "
            f"root={path}, ignored={ignored}"
        )
    selected = max(valid, key=lambda item: item.completed_step)
    latest_matches = bool(
        latest
        and latest.get("schema_version") == LATEST_SCHEMA_VERSION
        and latest.get("checkpoint_dir") == selected.checkpoint_dir.name
        and int(latest.get("completed_step", -1)) == selected.completed_step
        and latest.get("checkpoint_digest") == selected.checkpoint_digest
    )
    if not latest_matches:
        _write_latest(path, selected)
        latest = _json_object(latest_path)
    if ignored or not latest_matches:
        write_json(
            path / "checkpoint_recovery_audit.json",
            {
                "schema_version": "comrecgc_checkpoint_recovery_audit_v1",
                "selected_checkpoint": selected.checkpoint_dir.name,
                "selected_completed_step": selected.completed_step,
                "selected_checkpoint_digest": selected.checkpoint_digest,
                "latest_repaired": not latest_matches,
                "ignored_entries": ignored,
                "audited_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return selected.checkpoint_dir, latest


def validate_generation_checkpoint(
    checkpoint_root_or_dir: str | Path,
    *,
    expected_provenance: Mapping[str, str] | None = None,
    expected_scientific_argv: Any | None = None,
    expected_command_sha256: str | None = None,
    expected_total_steps: int | None = None,
    expected_completed_step: int | None = None,
    _validate_state_payload: bool = True,
) -> GenerationCheckpointValidation:
    """Validate every checkpoint component and return only on exact success."""

    checkpoint_dir, latest = _resolve_reference(
        checkpoint_root_or_dir,
        validate_state_payload=bool(_validate_state_payload),
    )
    if checkpoint_dir.is_symlink():
        raise GenerationCheckpointError("Checkpoint directory must not be a symlink.")
    match = _CHECKPOINT_NAME.fullmatch(checkpoint_dir.name)
    if match is None:
        raise GenerationCheckpointError(
            f"Checkpoint directory name is invalid: {checkpoint_dir.name!r}."
        )
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    complete_path = checkpoint_dir / COMPLETE_FILENAME
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not complete_path.is_file()
        or complete_path.is_symlink()
    ):
        raise GenerationCheckpointError("Checkpoint manifest/completion marker is missing.")
    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise GenerationCheckpointError("Checkpoint schema version is unsupported.")
    if (
        manifest.get("file_digest_algorithm") != "sha256"
        or manifest.get("checkpoint_digest_scheme") != "stable_json_sha256_v1"
    ):
        raise GenerationCheckpointError("Checkpoint digest scheme is unsupported.")
    if manifest.get("boundary") != CHECKPOINT_BOUNDARY:
        raise GenerationCheckpointError("Checkpoint is not at a completed-step boundary.")
    if manifest.get("state_schema_version") != CHECKPOINT_STATE_SCHEMA_VERSION:
        raise GenerationCheckpointError("Checkpoint state schema declaration is unsupported.")
    if manifest.get("atomic_complete") is not True:
        raise GenerationCheckpointError("Checkpoint is not atomically complete.")
    if manifest.get("rng_components") != [
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    ]:
        raise GenerationCheckpointError("Checkpoint RNG component declaration is incomplete.")
    completed_step = int(manifest.get("completed_step", -1))
    if completed_step <= 0 or completed_step != int(match.group("step")):
        raise GenerationCheckpointError(
            "Checkpoint completed step does not match its directory name."
        )
    if int(manifest.get("next_step", -1)) != completed_step + 1:
        raise GenerationCheckpointError("Checkpoint next-step boundary is inconsistent.")
    if manifest.get("checkpoint_dir") != checkpoint_dir.name:
        raise GenerationCheckpointError(
            "Checkpoint manifest directory identity does not match its location."
        )
    if expected_completed_step is not None and completed_step != int(
        expected_completed_step
    ):
        raise GenerationCheckpointError(
            "Checkpoint completed step differs from the requested step: "
            f"actual={completed_step}, expected={int(expected_completed_step)}."
        )
    provenance = _normalize_provenance(manifest.get("provenance_fingerprints") or {})
    if stable_json_sha256(provenance) != manifest.get("provenance_sha256"):
        raise GenerationCheckpointError("Checkpoint provenance digest mismatch.")
    if expected_provenance is not None:
        expected = _normalize_provenance(expected_provenance)
        if provenance != expected:
            raise GenerationCheckpointError(
                "Checkpoint provenance differs from the current runtime."
            )
    scientific_argv = _normalize_scientific_argv(manifest.get("scientific_argv"))
    command_sha256 = str(manifest.get("command_sha256") or "")
    if command_sha256 != scientific_command_sha256(scientific_argv):
        raise GenerationCheckpointError(
            "Checkpoint scientific command SHA256 does not match canonical argv."
        )
    if expected_scientific_argv is not None:
        expected_argv = _normalize_scientific_argv(expected_scientific_argv)
        if scientific_argv != expected_argv:
            raise GenerationCheckpointError(
                "Checkpoint scientific argv differs from the current runtime."
            )
    if expected_command_sha256 is not None and command_sha256 != str(
        expected_command_sha256
    ):
        raise GenerationCheckpointError(
            "Checkpoint command SHA256 differs from the current runtime."
        )
    total_steps = int(manifest.get("total_steps", -1))
    if total_steps <= 0 or completed_step > total_steps:
        raise GenerationCheckpointError(
            "Checkpoint total_steps is invalid or precedes completed_step."
        )
    if expected_total_steps is not None and total_steps != int(expected_total_steps):
        raise GenerationCheckpointError(
            "Checkpoint total_steps differs from the current runtime."
        )
    if provenance.get("scientific_command_sha256") != command_sha256 or provenance.get(
        "total_steps"
    ) != str(total_steps):
        raise GenerationCheckpointError(
            "Checkpoint provenance does not bind command SHA256 and total_steps."
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {STATE_FILENAME, SQLITE_FILENAME}:
        raise GenerationCheckpointError("Checkpoint file inventory is incomplete.")
    for name, identity in files.items():
        if not isinstance(identity, dict):
            raise GenerationCheckpointError(f"Checkpoint file identity is invalid: {name}")
        file_path = checkpoint_dir / name
        if not file_path.is_file() or file_path.is_symlink():
            raise GenerationCheckpointError(f"Checkpoint file is missing: {file_path}")
        if file_path.stat().st_size != int(identity.get("bytes", -1)):
            raise GenerationCheckpointError(f"Checkpoint file size mismatch: {file_path}")
        if sha256_file(file_path) != str(identity.get("sha256")):
            raise GenerationCheckpointError(f"Checkpoint file SHA256 mismatch: {file_path}")
    checkpoint_digest = str(manifest.get("checkpoint_digest") or "")
    if checkpoint_digest != stable_json_sha256(_checkpoint_digest_payload(manifest)):
        raise GenerationCheckpointError("Checkpoint manifest digest mismatch.")
    complete = _json_object(complete_path)
    if complete != {
        "checkpoint_digest": checkpoint_digest,
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
    }:
        raise GenerationCheckpointError("Checkpoint completion marker mismatch.")
    if latest is not None and (
        latest.get("checkpoint_digest") != checkpoint_digest
        or int(latest.get("completed_step", -1)) != completed_step
        or latest.get("checkpoint_dir") != checkpoint_dir.name
    ):
        raise GenerationCheckpointError("Checkpoint LATEST pointer digest mismatch.")
    sqlite_audit = _inspect_sqlite(checkpoint_dir / SQLITE_FILENAME)
    if sqlite_audit.get("integrity_check") != "ok" or sqlite_audit != manifest.get(
        "sqlite_snapshot"
    ):
        raise GenerationCheckpointError("Checkpoint SQLite snapshot audit mismatch.")
    if _validate_state_payload:
        state = _torch_load(checkpoint_dir / STATE_FILENAME, mmap=True)
        _validate_checkpoint_state_payload(
            state,
            manifest=manifest,
            scientific_argv=scientific_argv,
            command_sha256=command_sha256,
            total_steps=total_steps,
            completed_step=completed_step,
        )
    return GenerationCheckpointValidation(
        checkpoint_dir=checkpoint_dir,
        completed_step=completed_step,
        checkpoint_digest=checkpoint_digest,
        provenance_fingerprints=provenance,
        scientific_argv=scientific_argv,
        command_sha256=command_sha256,
        total_steps=total_steps,
        manifest=manifest,
    )


def _validate_checkpoint_state_payload(
    state: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    scientific_argv: tuple[str, ...],
    command_sha256: str,
    total_steps: int,
    completed_step: int,
) -> None:
    """Validate one already-loaded state without causing a second load.

    Large T14 checkpoints contain tens of GiB of Python/NumPy state.  The old
    save path reloaded that state while the live walk and its serialization
    projection were still resident, and the old load path deserialized it once
    in ``validate_generation_checkpoint`` and a second time immediately after.
    Keeping payload validation separate preserves every semantic check while
    allowing the writer to validate its in-memory payload and the reader to
    validate the single object it actually restores.
    """

    if not isinstance(state, Mapping):
        raise GenerationCheckpointError("Checkpoint state payload must be a dictionary.")
    if state.get("schema_version") != CHECKPOINT_STATE_SCHEMA_VERSION:
        raise GenerationCheckpointError("Checkpoint state schema version is unsupported.")
    if state.get("boundary") != CHECKPOINT_BOUNDARY or state.get(
        "fully_completed_step"
    ) is not True:
        raise GenerationCheckpointError("Checkpoint state is not a completed-step state.")
    if int(state.get("completed_step", -1)) != completed_step:
        raise GenerationCheckpointError("Checkpoint state step differs from its manifest.")
    if int(state.get("next_step", -1)) != completed_step + 1:
        raise GenerationCheckpointError("Checkpoint state next step is inconsistent.")
    if state.get("provenance_sha256") != manifest.get("provenance_sha256"):
        raise GenerationCheckpointError("Checkpoint state provenance digest mismatch.")
    if state.get("scientific_argv") != list(scientific_argv):
        raise GenerationCheckpointError(
            "Checkpoint state scientific argv differs from its manifest."
        )
    if state.get("command_sha256") != command_sha256:
        raise GenerationCheckpointError(
            "Checkpoint state command SHA256 differs from its manifest."
        )
    if int(state.get("total_steps", -1)) != total_steps:
        raise GenerationCheckpointError(
            "Checkpoint state total_steps differs from its manifest."
        )
    if not isinstance(state.get("algorithm_state"), dict) or not state[
        "algorithm_state"
    ]:
        raise GenerationCheckpointError("Checkpoint algorithm state is missing.")
    if not isinstance(state.get("trace_state"), dict) or not state["trace_state"]:
        raise GenerationCheckpointError("Checkpoint trace state is missing.")
    if not isinstance(state.get("rng_state"), dict):
        raise GenerationCheckpointError("Checkpoint RNG state is missing.")
    _validate_rng_state(state["rng_state"])
    if int(manifest.get("torch_cuda_device_count", -1)) != int(
        state["rng_state"]["torch_cuda_device_count"]
    ):
        raise GenerationCheckpointError(
            "Checkpoint manifest/state CUDA RNG device counts differ."
        )


def validate_generation_checkpoint_envelope(
    checkpoint_root_or_dir: str | Path,
    *,
    expected_provenance: Mapping[str, str] | None = None,
    expected_scientific_argv: Any | None = None,
    expected_command_sha256: str | None = None,
    expected_total_steps: int | None = None,
    expected_completed_step: int | None = None,
) -> GenerationCheckpointValidation:
    """Validate atomicity, exact file hashes and SQLite without deserializing state.

    This is intentionally an envelope check, not an independent reload PASS.
    Resume and terminal validation continue to use the full validator.  Its
    narrow purpose is to keep a live high-memory writer from loading a second
    copy of the state it has just serialized.
    """

    return validate_generation_checkpoint(
        checkpoint_root_or_dir,
        expected_provenance=expected_provenance,
        expected_scientific_argv=expected_scientific_argv,
        expected_command_sha256=expected_command_sha256,
        expected_total_steps=expected_total_steps,
        expected_completed_step=expected_completed_step,
        _validate_state_payload=False,
    )


def save_generation_checkpoint(
    checkpoint_root: str | Path,
    *,
    completed_step: int,
    step_complete: bool,
    algorithm_state: Mapping[str, Any],
    trace_state: Mapping[str, Any],
    sqlite_source: sqlite3.Connection | str | Path,
    provenance_fingerprints: Mapping[str, str],
    scientific_argv: Any,
    command_sha256: str,
    total_steps: int,
    rng_state: Mapping[str, Any] | None = None,
    reload_after_write: bool = True,
) -> GenerationCheckpointValidation:
    """Publish one exact checkpoint with directory and LATEST atomicity."""

    step = int(completed_step)
    if not step_complete or step <= 0 or step > 999_999_999_999:
        raise GenerationCheckpointError(
            "Generation checkpoints may only be written after a positive, fully "
            "completed step."
        )
    if not isinstance(algorithm_state, Mapping) or not algorithm_state:
        raise GenerationCheckpointError("Checkpoint algorithm state must be non-empty.")
    if not isinstance(trace_state, Mapping) or not trace_state:
        raise GenerationCheckpointError("Checkpoint trace state must be non-empty.")
    provenance = _normalize_provenance(provenance_fingerprints)
    normalized_argv = _normalize_scientific_argv(scientific_argv)
    normalized_command_sha256 = str(command_sha256).strip()
    if normalized_command_sha256 != scientific_command_sha256(normalized_argv):
        raise GenerationCheckpointError(
            "Checkpoint command SHA256 does not match canonical scientific argv."
        )
    normalized_total_steps = int(total_steps)
    if normalized_total_steps <= 0 or step > normalized_total_steps:
        raise GenerationCheckpointError(
            "Checkpoint total_steps must be positive and at least completed_step."
        )
    if provenance.get("scientific_command_sha256") != normalized_command_sha256:
        raise GenerationCheckpointError(
            "Checkpoint provenance must bind the scientific command SHA256."
        )
    if provenance.get("total_steps") != str(normalized_total_steps):
        raise GenerationCheckpointError(
            "Checkpoint provenance must bind total_steps."
        )
    captured_rng = dict(capture_rng_state() if rng_state is None else rng_state)
    _validate_rng_state(captured_rng)
    raw_root = Path(checkpoint_root).expanduser()
    if raw_root.is_symlink():
        raise GenerationCheckpointError("Generation checkpoint root must not be a symlink.")
    root = raw_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(root.parent)
    _fsync_directory(root)
    checkpoint_name = f"step-{step:012d}"
    final = root / checkpoint_name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"Generation checkpoint already exists: {final}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{checkpoint_name}-", suffix=".tmp", dir=root)
    )
    published = False
    try:
        state_path = temporary / STATE_FILENAME
        sqlite_path = temporary / SQLITE_FILENAME
        state = {
            "schema_version": CHECKPOINT_STATE_SCHEMA_VERSION,
            "boundary": CHECKPOINT_BOUNDARY,
            "fully_completed_step": True,
            "completed_step": step,
            "next_step": step + 1,
            "provenance_sha256": stable_json_sha256(provenance),
            "scientific_argv": list(normalized_argv),
            "command_sha256": normalized_command_sha256,
            "total_steps": normalized_total_steps,
            "rng_state": captured_rng,
            "algorithm_state": dict(algorithm_state),
            "trace_state": dict(trace_state),
        }
        _torch_save(state, state_path)
        sqlite_audit = _backup_sqlite(sqlite_source, sqlite_path)
        files = {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in (
                (STATE_FILENAME, state_path),
                (SQLITE_FILENAME, sqlite_path),
            )
        }
        manifest: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_schema_version": CHECKPOINT_STATE_SCHEMA_VERSION,
            "file_digest_algorithm": "sha256",
            "checkpoint_digest_scheme": "stable_json_sha256_v1",
            "boundary": CHECKPOINT_BOUNDARY,
            "atomic_complete": True,
            "checkpoint_dir": checkpoint_name,
            "completed_step": step,
            "next_step": step + 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provenance_fingerprints": provenance,
            "provenance_sha256": stable_json_sha256(provenance),
            "scientific_argv": list(normalized_argv),
            "command_sha256": normalized_command_sha256,
            "total_steps": normalized_total_steps,
            "rng_components": [
                "python",
                "numpy",
                "torch_cpu",
                "torch_cuda",
            ],
            "torch_cuda_device_count": int(
                captured_rng["torch_cuda_device_count"]
            ),
            "sqlite_snapshot_method": "sqlite_connection_backup_api_v1",
            "sqlite_snapshot": sqlite_audit,
            "files": files,
        }
        manifest["checkpoint_digest"] = stable_json_sha256(
            _checkpoint_digest_payload(manifest)
        )
        # Validate the exact in-memory payload before publication.  Reopening
        # the just-written archive here used to deserialize a second copy of a
        # 40+ GiB T14 state while the live walk was still resident.  Exact file
        # hashes below still close the bytes; an independent resume/terminal
        # verifier performs the ordinary full deserialize check.
        _validate_checkpoint_state_payload(
            state,
            manifest=manifest,
            scientific_argv=normalized_argv,
            command_sha256=normalized_command_sha256,
            total_steps=normalized_total_steps,
            completed_step=step,
        )
        manifest_path = temporary / MANIFEST_FILENAME
        write_json(manifest_path, manifest)
        write_json(
            temporary / COMPLETE_FILENAME,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_digest": manifest["checkpoint_digest"],
                "manifest_sha256": sha256_file(manifest_path),
            },
        )
        _fsync_directory(temporary)
        os.rename(temporary, final)
        published = True
        _fsync_directory(root)
        validation = validate_generation_checkpoint_envelope(
            final,
            expected_provenance=provenance,
            expected_scientific_argv=normalized_argv,
            expected_command_sha256=normalized_command_sha256,
            expected_total_steps=normalized_total_steps,
            expected_completed_step=step,
        )
        if reload_after_write:
            validation = validate_generation_checkpoint(
                final,
                expected_provenance=provenance,
                expected_scientific_argv=normalized_argv,
                expected_command_sha256=normalized_command_sha256,
                expected_total_steps=normalized_total_steps,
                expected_completed_step=step,
            )
            _write_latest(root, validation)
        else:
            _write_pending_latest(root, validation)
        return validation
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def promote_generation_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_provenance: Mapping[str, str] | None = None,
    expected_scientific_argv: Any | None = None,
    expected_command_sha256: str | None = None,
    expected_total_steps: int | None = None,
    expected_completed_step: int | None = None,
) -> GenerationCheckpointValidation:
    """Independently reload one exact pending checkpoint, then promote LATEST."""

    raw = Path(checkpoint_dir).expanduser()
    if raw.is_symlink() or not (raw / MANIFEST_FILENAME).is_file():
        raise GenerationCheckpointError(
            "Checkpoint promotion requires one exact physical checkpoint directory."
        )
    validation = validate_generation_checkpoint(
        raw,
        expected_provenance=expected_provenance,
        expected_scientific_argv=expected_scientific_argv,
        expected_command_sha256=expected_command_sha256,
        expected_total_steps=expected_total_steps,
        expected_completed_step=expected_completed_step,
    )
    root = validation.checkpoint_dir.parent
    pending_path = root / PENDING_LATEST_FILENAME
    if not pending_path.is_file() or pending_path.is_symlink():
        raise GenerationCheckpointError("Checkpoint promotion has no physical pending pointer.")
    pending = _json_object(pending_path)
    if pending != {
        "schema_version": "comrecgc_generation_checkpoint_pending_v1",
        "checkpoint_dir": validation.checkpoint_dir.name,
        "completed_step": validation.completed_step,
        "checkpoint_digest": validation.checkpoint_digest,
        "payload_reload_state": "PENDING_INDEPENDENT_RELOAD",
    }:
        raise GenerationCheckpointError("Checkpoint pending pointer differs from payload.")
    _write_latest(root, validation)
    write_json(
        root / f"{validation.checkpoint_dir.name}.promotion.json",
        {
            "schema_version": "comrecgc_generation_checkpoint_promotion_v1",
            "status": "PASS",
            "checkpoint_dir": validation.checkpoint_dir.name,
            "completed_step": validation.completed_step,
            "checkpoint_digest": validation.checkpoint_digest,
            "payload_reload_pass": True,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return validation


def load_generation_checkpoint(
    checkpoint_root_or_dir: str | Path,
    *,
    expected_provenance: Mapping[str, str] | None = None,
    expected_scientific_argv: Any | None = None,
    expected_command_sha256: str | None = None,
    expected_total_steps: int | None = None,
    expected_completed_step: int | None = None,
    single_pass: bool = False,
) -> LoadedGenerationCheckpoint:
    """Validate and load checkpoint payloads without mutating runtime state."""

    # Envelope validation closes every persisted byte and the SQLite snapshot
    # without materializing algorithm state.  The single mmap-capable load is
    # then validated in-place, eliminating the historical double deserialize.
    validator = (
        validate_generation_checkpoint_envelope
        if single_pass
        else validate_generation_checkpoint
    )
    if single_pass:
        candidate = Path(checkpoint_root_or_dir).expanduser()
        if not (candidate / MANIFEST_FILENAME).is_file():
            raise GenerationCheckpointError(
                "Single-pass checkpoint load requires one exact checkpoint directory."
            )
    validation = validator(
        checkpoint_root_or_dir,
        expected_provenance=expected_provenance,
        expected_scientific_argv=expected_scientific_argv,
        expected_command_sha256=expected_command_sha256,
        expected_total_steps=expected_total_steps,
        expected_completed_step=expected_completed_step,
    )
    state = _torch_load(validation.checkpoint_dir / STATE_FILENAME, mmap=True)
    _validate_checkpoint_state_payload(
        state,
        manifest=validation.manifest,
        scientific_argv=validation.scientific_argv,
        command_sha256=validation.command_sha256,
        total_steps=validation.total_steps,
        completed_step=validation.completed_step,
    )
    return LoadedGenerationCheckpoint(
        validation=validation,
        algorithm_state=state["algorithm_state"],
        trace_state=state["trace_state"],
        rng_state=state["rng_state"],
        sqlite_snapshot_path=validation.checkpoint_dir / SQLITE_FILENAME,
    )


def restore_sqlite_snapshot(
    sqlite_snapshot_path: str | Path, destination_path: str | Path
) -> Path:
    """Atomically restore a validated standalone SQLite snapshot."""

    raw_source = Path(sqlite_snapshot_path).expanduser()
    if raw_source.is_symlink():
        raise GenerationCheckpointError(
            f"SQLite snapshot must not be a symbolic link: {raw_source}"
        )
    source = raw_source.resolve()
    if not source.is_file():
        raise GenerationCheckpointError(f"SQLite snapshot is missing: {source}")
    if _inspect_sqlite(source).get("integrity_check") != "ok":
        raise GenerationCheckpointError("SQLite source snapshot failed integrity_check.")
    raw_destination = Path(destination_path).expanduser()
    if raw_destination.is_symlink():
        raise GenerationCheckpointError("SQLite restore destination must not be a symlink.")
    destination = raw_destination.resolve()
    for suffix in ("-wal", "-shm"):
        if Path(f"{destination}{suffix}").exists():
            raise GenerationCheckpointError(
                "SQLite restore destination has a live WAL/SHM sidecar; close the "
                "runtime connection before restore."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _backup_sqlite(source, temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(f"{temporary}{suffix}").unlink(missing_ok=True)
    if _inspect_sqlite(destination).get("integrity_check") != "ok":
        raise GenerationCheckpointError("Restored SQLite database failed integrity_check.")
    return destination


def restore_generation_checkpoint(
    checkpoint_root_or_dir: str | Path,
    *,
    destination_sqlite_path: str | Path,
    expected_provenance: Mapping[str, str] | None = None,
    expected_scientific_argv: Any | None = None,
    expected_command_sha256: str | None = None,
    expected_total_steps: int | None = None,
    expected_completed_step: int | None = None,
    restore_rng: bool = True,
) -> LoadedGenerationCheckpoint:
    """Validate, restore SQLite atomically, and optionally restore global RNGs.

    The caller remains responsible for applying ``algorithm_state`` and
    ``trace_state`` to live objects only after this function succeeds.
    """

    loaded = load_generation_checkpoint(
        checkpoint_root_or_dir,
        expected_provenance=expected_provenance,
        expected_scientific_argv=expected_scientific_argv,
        expected_command_sha256=expected_command_sha256,
        expected_total_steps=expected_total_steps,
        expected_completed_step=expected_completed_step,
    )
    if restore_rng:
        _validate_rng_state(loaded.rng_state, require_current_cuda_match=True)
    restore_sqlite_snapshot(loaded.sqlite_snapshot_path, destination_sqlite_path)
    if restore_rng:
        restore_rng_state(loaded.rng_state)
    return loaded


def list_generation_checkpoints(checkpoint_root: str | Path) -> list[Path]:
    """List valid published checkpoints; crash-left ``*.tmp`` dirs are ignored."""

    root = Path(checkpoint_root).expanduser().resolve()
    if not root.is_dir():
        return []
    checkpoints: list[Path] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if _CHECKPOINT_NAME.fullmatch(path.name) is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise GenerationCheckpointError(
                f"Published checkpoint entry is not a physical directory: {path}"
            )
        validate_generation_checkpoint(path)
        checkpoints.append(path.resolve())
    return checkpoints


def mirror_generation_checkpoint(
    checkpoint_root_or_dir: str | Path,
    mirror_root: str | Path,
    *,
    expected_provenance: Mapping[str, str] | None = None,
) -> GenerationCheckpointValidation:
    """Atomically mirror one complete checkpoint and publish proof on both roots."""

    source = validate_generation_checkpoint(
        checkpoint_root_or_dir, expected_provenance=expected_provenance
    )
    source_root = source.checkpoint_dir.parent.resolve()
    raw_mirror = Path(mirror_root).expanduser()
    if raw_mirror.is_symlink():
        raise GenerationCheckpointError("Checkpoint mirror root must not be a symlink.")
    mirror = raw_mirror.resolve()
    if mirror == source_root or mirror in source_root.parents or source_root in mirror.parents:
        raise GenerationCheckpointError(
            "Checkpoint source and mirror roots must be distinct and non-nested."
        )
    mirror.mkdir(parents=True, exist_ok=True)
    _fsync_directory(mirror.parent)
    _fsync_directory(mirror)
    final = mirror / source.checkpoint_dir.name
    if final.exists() or final.is_symlink():
        mirrored = validate_generation_checkpoint(
            final,
            expected_provenance=source.provenance_fingerprints,
            expected_completed_step=source.completed_step,
        )
        if mirrored.checkpoint_digest != source.checkpoint_digest:
            raise GenerationCheckpointError(
                "Existing mirrored checkpoint differs from its source."
            )
    else:
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{source.checkpoint_dir.name}-mirror-",
                suffix=".tmp",
                dir=mirror,
            )
        )
        published = False
        try:
            for name in (
                STATE_FILENAME,
                SQLITE_FILENAME,
                MANIFEST_FILENAME,
                COMPLETE_FILENAME,
            ):
                source_path = source.checkpoint_dir / name
                if not source_path.is_file() or source_path.is_symlink():
                    raise GenerationCheckpointError(
                        f"Checkpoint mirror source is unsafe: {source_path}"
                    )
                _copy_file_fsync(source_path, temporary / name)
            _fsync_directory(temporary)
            os.rename(temporary, final)
            published = True
            _fsync_directory(mirror)
        finally:
            if not published and temporary.exists():
                shutil.rmtree(temporary)
        mirrored = validate_generation_checkpoint(
            final,
            expected_provenance=source.provenance_fingerprints,
            expected_completed_step=source.completed_step,
        )
        if mirrored.checkpoint_digest != source.checkpoint_digest:
            raise GenerationCheckpointError(
                "Mirrored checkpoint digest differs after atomic publication."
            )
    marker = {
        "schema_version": "comrecgc_generation_checkpoint_mirror_v1",
        "checkpoint_mirrored": True,
        "completed_step": source.completed_step,
        "checkpoint_digest": source.checkpoint_digest,
        "source_checkpoint": str(source.checkpoint_dir),
        "mirror_checkpoint": str(mirrored.checkpoint_dir),
        "mirrored_at": datetime.now(timezone.utc).isoformat(),
    }
    # A mirror is fully committed only after its proof marker exists.  Publish
    # LATEST last so a power loss can leave at worst an unreferenced, unmarked
    # checkpoint directory that recovery will ignore.
    write_json(mirrored.checkpoint_dir / MIRRORED_FILENAME, marker)
    write_json(source.checkpoint_dir / MIRRORED_FILENAME, marker)
    _write_latest(mirror, mirrored)
    return mirrored


def prune_mirrored_generation_checkpoints(
    checkpoint_root: str | Path,
    mirror_root: str | Path,
    *,
    keep_last: int = 2,
    expected_provenance: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Retain the newest mirrored checkpoints and audit every exact deletion.

    The complete preflight happens before the first deletion.  Consequently a
    missing/corrupt mirror or mirror marker leaves every checkpoint untouched.
    """

    keep = int(keep_last)
    if keep < 2:
        raise ValueError("Generation checkpoint retention must keep at least two.")
    local_root = Path(checkpoint_root).expanduser().resolve()
    resolved_mirror_root = Path(mirror_root).expanduser().resolve()
    local = list_generation_checkpoints(local_root)
    mirror = list_generation_checkpoints(resolved_mirror_root)
    mirror_by_name = {path.name: path for path in mirror}
    candidates = local[:-keep] if len(local) > keep else []
    preflight: list[tuple[GenerationCheckpointValidation, Path, dict[str, Any]]] = []
    for local_path in candidates:
        local_validation = validate_generation_checkpoint(
            local_path, expected_provenance=expected_provenance
        )
        marker_path = local_path / MIRRORED_FILENAME
        if not marker_path.is_file() or marker_path.is_symlink():
            raise GenerationCheckpointError(
                f"Refusing to prune an unmirrored checkpoint: {local_path}"
            )
        marker = _json_object(marker_path)
        mirror_path = mirror_by_name.get(local_path.name)
        if (
            marker.get("checkpoint_mirrored") is not True
            or marker.get("checkpoint_digest") != local_validation.checkpoint_digest
            or mirror_path is None
        ):
            raise GenerationCheckpointError(
                f"Checkpoint mirror proof is incomplete: {local_path}"
            )
        mirror_validation = validate_generation_checkpoint(
            mirror_path,
            expected_provenance=local_validation.provenance_fingerprints,
            expected_completed_step=local_validation.completed_step,
        )
        if mirror_validation.checkpoint_digest != local_validation.checkpoint_digest:
            raise GenerationCheckpointError(
                f"Checkpoint mirror proof has a digest mismatch: {local_path}"
            )
        preflight.append((local_validation, mirror_path, marker))

    removed: list[dict[str, Any]] = []
    local_history = local_root / RETENTION_HISTORY_DIRNAME
    mirror_history = resolved_mirror_root / RETENTION_HISTORY_DIRNAME
    local_history.mkdir(parents=True, exist_ok=True)
    mirror_history.mkdir(parents=True, exist_ok=True)
    for validation, mirror_path, marker in preflight:
        audit = {
            "schema_version": "comrecgc_generation_checkpoint_retention_v1",
            "checkpoint_mirrored": True,
            "completed_step": validation.completed_step,
            "checkpoint_digest": validation.checkpoint_digest,
            "local_checkpoint": str(validation.checkpoint_dir),
            "mirror_checkpoint": str(mirror_path),
            "mirror_marker_sha256": stable_json_sha256(marker),
            "retention_keep_last": keep,
            "pruned_at": datetime.now(timezone.utc).isoformat(),
        }
        history_name = f"step-{validation.completed_step:012d}.json"
        write_json(local_history / history_name, audit)
        write_json(mirror_history / history_name, audit)
        shutil.rmtree(validation.checkpoint_dir)
        _fsync_directory(local_root)
        shutil.rmtree(mirror_path)
        _fsync_directory(resolved_mirror_root)
        removed.append(audit)

    # Recover the only destructive crash window: local removal completed but
    # the paired mirror removal did not.  A prewritten retention audit with the
    # exact digest is mandatory; arbitrary unpaired mirror directories remain.
    remaining_local_names = {
        path.name for path in list_generation_checkpoints(local_root)
    }
    for mirror_path in list_generation_checkpoints(resolved_mirror_root):
        if mirror_path.name in remaining_local_names:
            continue
        history_path = mirror_history / f"{mirror_path.name}.json"
        if not history_path.is_file() or history_path.is_symlink():
            continue
        history = _json_object(history_path)
        mirror_validation = validate_generation_checkpoint(mirror_path)
        if (
            history.get("checkpoint_mirrored") is not True
            or history.get("checkpoint_digest") != mirror_validation.checkpoint_digest
            or int(history.get("completed_step", -1))
            != mirror_validation.completed_step
        ):
            raise GenerationCheckpointError(
                f"Orphan mirror retention proof is invalid: {mirror_path}"
            )
        shutil.rmtree(mirror_path)
        _fsync_directory(resolved_mirror_root)
    return removed
