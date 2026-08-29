"""Typed, CPU-only controller contract for the AIDS exact-DBSCAN recovery.

The failed c766 task is *evidence*, not a successful DBSCAN dependency.  This
module deliberately uses a private typed DAG and never converts the failed
selection receipt into a generic task PASS.  Production execution remains
disabled until every independently reviewed release pin is populated.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from src.utils.autodl_exec_startup_barrier import (
    ArmedExecStartupBarrier,
    MAX_RECORD_BYTES as EXEC_STARTUP_BARRIER_MAX_RECORD_BYTES,
    StartupBarrierRecord,
    arm_exec_startup_barrier,
    reconcile_interrupted_startup_barrier_publication,
    validate_reopenable_unreleased_barrier,
    validate_startup_barrier_record,
)


SPEC_SCHEMA = "aids_comrecgc_exact_recovery_controller_v1_spec_v1"
MANIFEST_SCHEMA = "aids_comrecgc_exact_recovery_controller_v1_manifest_v1"
OWNER_SCHEMA = "aids_comrecgc_exact_recovery_controller_owner_v3"
STATE_SCHEMA = "aids_comrecgc_exact_recovery_controller_state_v2"
STAGE_GATE_SCHEMA = "aids_comrecgc_exact_recovery_typed_stage_gate_v1"
TERMINAL_SCHEMA = "aids_comrecgc_exact_recovery_controller_terminal_v2"
EXACT_STAGE_RECEIPT_SCHEMA = "aids_comrecgc_exact_component_recovery_stage_v1"
SUBSET_STAGE_RECEIPT_SCHEMA = "aids_comrecgc_production_subset_stage_v1"
FINAL_STAGE_RECEIPT_SCHEMA = "aids_comrecgc_recovered_standardized_freeze_v1"
COEXISTENCE_SCHEMA = "aids_comrecgc_exact_recovery_coexistence_probe_v2"
RESOURCE_SCHEMA = "aids_comrecgc_exact_recovery_resource_budget_v1"
HANDOVER_SCHEMA = "aids_comrecgc_old_brute_handover_eligibility_v3"
EXACT_PROGRESS_MONITOR_SCHEMA = (
    "aids_comrecgc_exact_recovery_progress_monitor_v1"
)
HANDOVER_CONTROLLER_GENERATION_SCHEMA = (
    "aids_comrecgc_handover_controller_generation_v2"
)
HANDOVER_RESUME_RECEIPT_SCHEMA = "aids_comrecgc_handover_resume_receipt_v2"
HANDOVER_PROGRESS_OBSERVATION_SCHEMA = (
    "aids_comrecgc_handover_progress_observation_v2"
)
HANDOVER_EVIDENCE_MAX_GENERATIONS = 32
HANDOVER_EVIDENCE_MAX_OBSERVATIONS = 4096
HANDOVER_EVIDENCE_MAX_BYTES = 256 * 1024
HANDOVER_EVIDENCE_CLOCK_SKEW_SECONDS = 5.0
EXACT_MONOTONIC_PROGRESS_FIELD = "component_recovery_monotonic_rows"
CLOSURE_INVENTORY_SCHEMA = "aids_comrecgc_exact_recovery_stage_closure_inventory_v1"
CLOSURE_REHASH_MAX_BYTES = 16 * 1024**2
ROOT_CLAIM_SUFFIX = ".controller-root-claim.lock"
ROOT_CLAIM_SCHEMA = "aids_comrecgc_exact_recovery_controller_root_claim_v2"
ROOT_CLAIM_MAX_BYTES = 4096
STARTUP_BARRIER_BINDING_SCHEMA = (
    "aids_comrecgc_exact_recovery_exec_startup_binding_v1"
)
STARTUP_BARRIER_MAX_GENERATIONS = 32
STARTUP_BARRIER_RECORD_MAX_BYTES = EXEC_STARTUP_BARRIER_MAX_RECORD_BYTES
STARTUP_BARRIER_PUBLICATION_FILE_MULTIPLIER = 2
OUTER_STARTUP_BARRIER_STAGE_COUNT = 4
INNER_CONTINUATION_STAGE_COUNT = 5
SUBSET_MAX_ATTEMPTS = 8
PARTIAL_STAGE_ARCHIVE_COUNT = 4
PARTIAL_STAGE_ARCHIVE_MAX_BYTES = 1024**3
HANDOVER_MIN_PROGRESS_SECONDS = 10 * 60
HANDOVER_MAX_OBSERVATION_AGE_SECONDS = 2 * 60
HANDOVER_MAX_ETA_SECONDS = 48 * 60 * 60
HANDOVER_MIN_RELATIVE_SPEEDUP = 100.0
OLD_BRUTE_PID = 273939
OLD_BRUTE_START_TICKS = 687141119
OLD_BRUTE_CMDLINE_SHA256 = (
    "792679fed417737f85462d940243153e5081d8b80c7dab663591131c5bbd51b8"
)

CONTROLLER_ID = "aids_comrecgc_exact_component_recovery_v1"
CID_PATTERN = re.compile(
    r"^aids_comrecgc_exact_recovery_v1_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$"
)
SCIENCE_RELEASE_COMMIT = "d8912ccb0901840ee1f0458ef66f630312024b0b"

ADOPTION_STAGE = "failed_selection_adoption"
SUBSET_STAGE = "production_subset_equivalence"
EXACT_STAGE = "exact_component_recovery"
DOWNSTREAM_STAGE = "component_downstream_radius_ab"
FINAL_STAGE = "standardized_freeze_terminal"
STAGE_ORDER = (
    ADOPTION_STAGE,
    SUBSET_STAGE,
    EXACT_STAGE,
    DOWNSTREAM_STAGE,
    FINAL_STAGE,
)
STAGE_KINDS = {
    ADOPTION_STAGE: "failed_selection_recovery_evidence",
    SUBSET_STAGE: "production_derived_subset_preflight",
    EXACT_STAGE: "exact_component_partition",
    DOWNSTREAM_STAGE: "streaming_component_downstream_boundary_replay",
    FINAL_STAGE: "standardized_freeze",
}
DEPENDENCIES = {
    ADOPTION_STAGE: (),
    SUBSET_STAGE: (ADOPTION_STAGE,),
    EXACT_STAGE: (ADOPTION_STAGE, SUBSET_STAGE),
    DOWNSTREAM_STAGE: (EXACT_STAGE,),
    FINAL_STAGE: (ADOPTION_STAGE, SUBSET_STAGE, EXACT_STAGE, DOWNSTREAM_STAGE),
}
REQUIRED_ARGV_BINDING_ROLES = {
    # The reviewed adoption CLI owns a fixed production authority profile; it
    # intentionally accepts no caller-selected source paths.
    ADOPTION_STAGE: {"output"},
    SUBSET_STAGE: {"output", "controller_manifest", "adoption_gate"},
    EXACT_STAGE: {
        "output",
        "controller_manifest",
        "adoption_gate",
        "subset_gate",
    },
    DOWNSTREAM_STAGE: {
        "output",
        "controller_manifest",
        "exact_gate",
    },
    FINAL_STAGE: {
        "output",
        "controller_manifest",
        "adoption_gate",
        "subset_gate",
        "exact_gate",
        "downstream_gate",
    },
}

EXPECTED_ROWS = 91_916_686
EXPECTED_VECTOR_DIM = 64
EXPECTED_PARENT_COUNT = 1_283
EXPECTED_CANDIDATE_COUNT = 71_642
HANDOVER_CONSERVATIVE_WORK_ROWS = 2 * EXPECTED_ROWS
EXPECTED_SUBSET_NAMES = ("first", "random", "dense", "sparse", "theta_boundary")
DEFAULT_SUBSET_SIZE = 2_000
DEFAULT_BLOCK_SIZE = 65_536
MIN_THREAD_COUNT = 8
MAX_THREAD_COUNT = 12
DEFAULT_THREAD_COUNT = MIN_THREAD_COUNT
DEFAULT_MAX_RSS_BYTES = 96 * 1024**3
DEFAULT_SUBSET_MAX_RSS_BYTES = 32 * 1024**3
DEFAULT_SUBSET_RSS_WORKING_MARGIN_BYTES = 8 * 1024**3
SUBSET_MAX_RSS_SCOPE = (
    "production_subset_process_peak_including_full_source_selection_scans"
)
SUBSET_RSS_BUDGET_SEMANTICS = (
    "measured_pre_dbscan_peak_plus_bounded_working_margin"
)
DEFAULT_SAFETY_FLOOR_BYTES = 8 * 1024**3

REQUIRED_RELEASE_PINS = (
    "science_commit",
    "adoption_commit",
    "controller_commit",
    "exact_runner_commit",
    "subset_runner_commit",
    "downstream_runner_commit",
    "standardization_runner_commit",
)
ADOPTION_VALIDATOR_API = "fixed_production_output_dir_recovery_evidence_validator_v1"
EXPECTED_ADOPTION_VALIDATOR_MODULE = (
    "src.baselines.comrecgc.failed_selection_adoption"
)
EXPECTED_ADOPTION_VALIDATOR_CALLABLE = (
    "verify_aids_c766_failed_selection_recovery_evidence"
)
EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256 = {
    "close": "f2bcde0b4cf8b86082abb3bc9b7499c8a9459f1a1df92d8eada28996e332a780",
    "final": "b455b618d29ac807eecead64b3aa8f47bfdee67344dab9cfb566337d148c12ab",
}


class RecoveryControllerError(RuntimeError):
    """Fail-closed controller contract error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _frozen_stage_environment(manifest: Mapping[str, Any]) -> dict[str, str]:
    threads = str(manifest["resources"]["thread_count"])
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
    }


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    source = Path(path)
    before = _physical_file(source, label="SHA256 source")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise RecoveryControllerError("SHA256 source changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                block = handle.read(block_size)
                if not block:
                    break
                digest.update(block)
        after_fd = os.fstat(descriptor)
        after_path = source.stat()
        expected = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            (
                after_fd.st_dev,
                after_fd.st_ino,
                after_fd.st_size,
                after_fd.st_mtime_ns,
                after_fd.st_ctime_ns,
            )
            != expected
            or (
                after_path.st_dev,
                after_path.st_ino,
                after_path.st_size,
                after_path.st_mtime_ns,
                after_path.st_ctime_ns,
            )
            != expected
        ):
            raise RecoveryControllerError("SHA256 source changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_absolute(value: Any, *, label: str, existing: str | None = None) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise RecoveryControllerError(f"{label} must be absolute")
    if existing is None:
        return path.resolve(strict=False)
    if path.is_symlink():
        raise RecoveryControllerError(f"{label} may not be a symlink")
    resolved = path.resolve(strict=True)
    if existing == "file" and (
        not resolved.is_file() or resolved.stat().st_size <= 0
    ):
        raise RecoveryControllerError(f"{label} must be a nonempty file")
    if existing == "dir" and not resolved.is_dir():
        raise RecoveryControllerError(f"{label} must be a directory")
    return resolved


def _physical_file(path: Path, *, label: str) -> os.stat_result:
    if path.is_symlink():
        raise RecoveryControllerError(f"{label} may not be a symlink")
    try:
        value = path.stat()
    except FileNotFoundError as exc:
        raise RecoveryControllerError(f"{label} is absent: {path}") from exc
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise RecoveryControllerError(f"{label} must be a physical nonempty file")
    return value


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    before = _physical_file(source, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RecoveryControllerError(f"{label} changed while opening")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            payload = json.load(handle)
        after_fd = os.fstat(descriptor)
        after = source.stat()
        expected = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            (
                after_fd.st_dev,
                after_fd.st_ino,
                after_fd.st_size,
                after_fd.st_mtime_ns,
                after_fd.st_ctime_ns,
            )
            != expected
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != expected
        ):
            raise RecoveryControllerError(f"{label} path identity changed")
    except Exception:
        os.close(descriptor)
        raise
    os.close(descriptor)
    if not isinstance(payload, dict):
        raise RecoveryControllerError(f"{label} must contain one JSON object")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_bytes(path: Path, payload: bytes) -> None:
    """Crash-reconcilable immutable publication with a fixed private temp."""

    parent = path.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
    ):
        raise RecoveryControllerError(
            f"immutable output parent must be a physical directory: {parent}"
        )
    temporary = parent / f".{path.name}.publish.tmp"
    if (path.exists() or path.is_symlink()) and not (
        temporary.exists() or temporary.is_symlink()
    ):
        raise RecoveryControllerError(f"immutable output already exists: {path}")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except FileExistsError:
        descriptor = os.open(
            temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        current = temporary.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink not in {1, 2}
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RecoveryControllerError(
                f"immutable publication temp identity changed: {temporary}"
            )
        if opened.st_nlink == 2:
            final = path.lstat()
            if (
                (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino)
                or path.read_bytes() != payload
            ):
                raise RecoveryControllerError(
                    f"immutable linked publication changed: {path}"
                )
            temporary.unlink()
            _fsync_directory(parent)
            return
        if path.exists() or path.is_symlink():
            raise RecoveryControllerError(f"immutable output already exists: {path}")
        os.ftruncate(descriptor, 0)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RecoveryControllerError("immutable output write made no progress")
            offset += written
        os.ftruncate(descriptor, len(payload))
        os.fsync(descriptor)
        _fsync_directory(parent)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise RecoveryControllerError(f"immutable output already exists: {path}") from exc
        _fsync_directory(parent)
        final = path.lstat()
        linked = os.fstat(descriptor)
        if (
            (final.st_dev, final.st_ino) != (linked.st_dev, linked.st_ino)
            or path.read_bytes() != payload
        ):
            raise RecoveryControllerError(f"immutable output identity changed: {path}")
        temporary.unlink()
        _fsync_directory(parent)
    finally:
        os.close(descriptor)


def _reconcile_immutable_link_publication(path: Path) -> bool:
    """Remove only the fixed same-inode temp left after a successful link."""

    temporary = path.parent / f".{path.name}.publish.tmp"
    if not temporary.exists() and not temporary.is_symlink():
        return False
    if not path.exists() or path.is_symlink():
        return False
    descriptor = os.open(
        temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        final = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 2
            or (opened.st_dev, opened.st_ino) != (final.st_dev, final.st_ino)
        ):
            raise RecoveryControllerError(
                f"immutable publication temp cannot be reconciled: {path}"
            )
        temporary.unlink()
        _fsync_directory(path.parent)
        return True
    finally:
        os.close(descriptor)


def _inspect_immutable_publication(path: Path) -> bool:
    """Read-only check; return true only for an authentic link-crash window."""

    temporary = path.parent / f".{path.name}.publish.tmp"
    temp_present = temporary.exists() or temporary.is_symlink()
    final_present = path.exists() or path.is_symlink()
    if not temp_present:
        if final_present:
            final = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(final.st_mode)
                or stat.S_IMODE(final.st_mode) != 0o600
                or final.st_uid != os.getuid()
                or final.st_nlink != 1
            ):
                raise RecoveryControllerError(
                    f"immutable publication final identity changed: {path}"
                )
        return False
    if not final_present or path.is_symlink() or temporary.is_symlink():
        raise RecoveryControllerError(
            f"immutable publication requires locked reconciliation: {path}"
        )
    temp = temporary.lstat()
    final = path.lstat()
    if (
        not stat.S_ISREG(temp.st_mode)
        or not stat.S_ISREG(final.st_mode)
        or stat.S_IMODE(temp.st_mode) != 0o600
        or temp.st_uid != os.getuid()
        or temp.st_nlink != 2
        or final.st_nlink != 2
        or (temp.st_dev, temp.st_ino) != (final.st_dev, final.st_ino)
    ):
        raise RecoveryControllerError(
            f"immutable publication temp identity changed: {path}"
        )
    return True


def _discard_or_reconcile_immutable_publication(path: Path) -> bool:
    """Resolve one controller-owned unpublished/link-complete temp under lock."""

    temporary = path.parent / f".{path.name}.publish.tmp"
    if not temporary.exists() and not temporary.is_symlink():
        return False
    if path.exists() and not path.is_symlink():
        return _reconcile_immutable_link_publication(path)
    descriptor = os.open(
        temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        current = temporary.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RecoveryControllerError(
                f"unpublished immutable temp identity changed: {temporary}"
            )
        temporary.unlink()
        _fsync_directory(path.parent)
        return True
    finally:
        os.close(descriptor)


def _reconcile_controller_owned_publications(
    manifest: Mapping[str, Any], root: Path
) -> None:
    """Mutate crash temps only while the caller holds ``.controller.lock``."""

    paths = [
        *(_gate_path(manifest, stage_id) for stage_id in STAGE_ORDER),
        root / "coexistence_probe.json",
        root / "terminal.json",
        root / "PASS",
    ]
    for path in paths:
        _discard_or_reconcile_immutable_publication(path)
    for temporary in sorted((root / "logs").glob(".prelaunch.*.json.publish.tmp")):
        final_name = temporary.name[1 : -len(".publish.tmp")]
        _discard_or_reconcile_immutable_publication(temporary.parent / final_name)
    state_temporary = root / ".state.json.replace.tmp"
    if state_temporary.exists() or state_temporary.is_symlink():
        descriptor = os.open(
            state_temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            current = state_temporary.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_uid != os.getuid()
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)
            ):
                raise RecoveryControllerError("mutable state temp identity changed")
            state_temporary.unlink()
            _fsync_directory(root)
        finally:
            os.close(descriptor)


def _reconcile_completed_stage_publication(
    *, stage_id: str, stage: Mapping[str, Any]
) -> None:
    """Normalize a completed stage terminal only under that stage's own lock."""

    if stage_id in {ADOPTION_STAGE, DOWNSTREAM_STAGE}:
        return
    terminal = Path(stage["terminal_path"])
    temporary = terminal.parent / f".{terminal.name}.publish.tmp"
    if not temporary.exists() and not temporary.is_symlink():
        return
    if not terminal.exists() or terminal.is_symlink():
        # The stage owns a temp-only retry and must reopen it through its
        # normal resume command; the controller must not discard it here.
        return
    from src.utils.autodl_aids_comrecgc_exact_recovery_stages_v1 import (
        _reconcile_immutable_stage_publication,
        _stage_writer,
    )

    output_root = Path(stage["output_dir"])
    lock_root = (
        output_root / ".exact-recovery-final-owner"
        if stage_id == FINAL_STAGE
        else output_root
    )
    with _stage_writer(lock_root, resume=True) as held:
        if not _reconcile_immutable_stage_publication(terminal):
            raise RecoveryControllerError(
                f"stage terminal publication disappeared: {stage_id}"
            )
        held.verify()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        _json_payload_bytes(payload),
    )


def _json_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _atomic_state(path: Path, payload: Mapping[str, Any]) -> None:
    if (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise RecoveryControllerError("mutable state parent is not physical")
    temporary = path.parent / f".{path.name}.replace.tmp"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        current = temporary.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RecoveryControllerError("mutable state temp identity changed")
        encoded = _json_payload_bytes(payload)
        os.ftruncate(descriptor, 0)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RecoveryControllerError("mutable state write made no progress")
            offset += written
        os.ftruncate(descriptor, len(encoded))
        os.fsync(descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        os.close(descriptor)


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "nlink": int(value.st_nlink),
    }


def _directory_identity(path: Path) -> dict[str, int]:
    if path.is_symlink():
        raise RecoveryControllerError(f"directory may not be a symlink: {path}")
    value = path.stat()
    if not stat.S_ISDIR(value.st_mode):
        raise RecoveryControllerError(f"not a physical directory: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
    }


def derive_output_budget(
    *,
    row_count: int,
    vector_dim: int,
    subset_size: int,
    subset_count: int,
    block_size: int,
    safety_floor_bytes: int = DEFAULT_SAFETY_FLOOR_BYTES,
) -> dict[str, Any]:
    """Derive a conservative output bound without counting adopted 25GB data."""

    if (
        row_count <= 0
        or vector_dim <= 0
        or subset_size <= 0
        or subset_count <= 0
        or block_size <= 0
        or safety_floor_bytes < 4 * 1024**3
    ):
        raise RecoveryControllerError("invalid output-budget inputs")
    # Recovery can promote fresh uint32 lower bounds/attachments plus int64
    # labels and boolean core/strict-radius masks.  Rename-based promotion has
    # at most one largest-array transient at a time.
    arrays = {
        "component_core_lower_bounds_uint32": row_count * 4,
        "component_attachments_uint32": row_count * 4,
        "labels_int64": row_count * 8,
        "core_mask_bool": row_count,
        "retained_mask_bool": row_count,
    }
    transient = max(arrays.values())
    # Each subset's worst-case exact edge ledger is bounded as a dense uint64
    # square.  This intentionally overbounds the real blockwise fixtures.
    subset_attempt_bound = subset_count * subset_size * subset_size * 8
    subset_bound = SUBSET_MAX_ATTEMPTS * subset_attempt_bound
    checkpoint_blocks = (row_count + block_size - 1) // block_size
    checkpoint_bound = checkpoint_blocks * 16 * 1024
    startup_barrier_bound = (
        (OUTER_STARTUP_BARRIER_STAGE_COUNT + INNER_CONTINUATION_STAGE_COUNT)
        * STARTUP_BARRIER_MAX_GENERATIONS
        * STARTUP_BARRIER_RECORD_MAX_BYTES
        * STARTUP_BARRIER_PUBLICATION_FILE_MULTIPLIER
    )
    handover_evidence_bound = (
        2 * HANDOVER_EVIDENCE_MAX_GENERATIONS
        + HANDOVER_EVIDENCE_MAX_OBSERVATIONS
    ) * HANDOVER_EVIDENCE_MAX_BYTES
    fixed = {
        "standardized_exports_and_freeze": 1024**3,
        # One preserved interrupted attempt for each deterministic stage after
        # common recourse. The continuation enforces both count and byte caps.
        "one_partial_archive_per_downstream_stage": (
            PARTIAL_STAGE_ARCHIVE_COUNT * PARTIAL_STAGE_ARCHIVE_MAX_BYTES
        ),
        "logs_manifests_and_certificates": 512 * 1024**2,
        "atomic_publication_transient": transient,
        "all_subset_attempts_dense_edge_upper_bound": subset_bound,
        "progress_ledger_upper_bound": checkpoint_bound,
        "startup_barrier_record_and_temp_upper_bound": startup_barrier_bound,
        "durable_handover_evidence_upper_bound": handover_evidence_bound,
    }
    max_output = sum(arrays.values()) + sum(fixed.values())
    return {
        "schema_version": RESOURCE_SCHEMA,
        "row_count": row_count,
        "vector_dim": vector_dim,
        "subset_count": subset_count,
        "subset_size": subset_size,
        "subset_max_attempts": SUBSET_MAX_ATTEMPTS,
        "block_size": block_size,
        "partial_stage_archive_count": PARTIAL_STAGE_ARCHIVE_COUNT,
        "partial_stage_archive_max_bytes_each": PARTIAL_STAGE_ARCHIVE_MAX_BYTES,
        "startup_barrier_max_generations": STARTUP_BARRIER_MAX_GENERATIONS,
        "startup_barrier_record_max_bytes": STARTUP_BARRIER_RECORD_MAX_BYTES,
        "startup_barrier_publication_file_multiplier": (
            STARTUP_BARRIER_PUBLICATION_FILE_MULTIPLIER
        ),
        "handover_evidence_max_generations": HANDOVER_EVIDENCE_MAX_GENERATIONS,
        "handover_evidence_max_observations": HANDOVER_EVIDENCE_MAX_OBSERVATIONS,
        "handover_evidence_max_bytes_each": HANDOVER_EVIDENCE_MAX_BYTES,
        "zero_copy_source_bytes_excluded": True,
        "source_pair_store_regenerated": False,
        "arrays": arrays,
        "diagnostic_bounds": {
            "subset_dense_edge_upper_bound_per_attempt": subset_attempt_bound,
        },
        "fixed_bounds": fixed,
        "max_output_bytes": max_output,
        "safety_floor_bytes": safety_floor_bytes,
        "minimum_free_bytes_before_launch": max_output + safety_floor_bytes,
        "formula": (
            "sum(2*N*uint32 + N*int64 + 2*N*bool) + max(row_array) + "
            "8*5*S^2*uint64 + ceil(N/B)*16KiB + 1GiB final + "
            "4*1GiB interrupted-stage archives + 512MiB + "
            "2*(4+5)*32*64KiB startup records + "
            "(2*32+4096)*256KiB durable handover evidence"
        ),
    }


def handover_contract() -> dict[str, Any]:
    """Return the immutable, no-signal old-brute handover policy."""

    return {
        "schema_version": HANDOVER_SCHEMA,
        "old_brute_process": {
            "pid": OLD_BRUTE_PID,
            "start_ticks": OLD_BRUTE_START_TICKS,
            "cmdline_sha256": OLD_BRUTE_CMDLINE_SHA256,
        },
        "minimum_continuous_progress_seconds": HANDOVER_MIN_PROGRESS_SECONDS,
        "maximum_observation_age_seconds": (
            HANDOVER_MAX_OBSERVATION_AGE_SECONDS
        ),
        "maximum_new_route_eta_seconds": HANDOVER_MAX_ETA_SECONDS,
        "minimum_relative_speedup": HANDOVER_MIN_RELATIVE_SPEEDUP,
        "conservative_exact_work_rows": HANDOVER_CONSERVATIVE_WORK_ROWS,
        "eta_or_relative_speedup_gate_required": False,
        "eligibility_policy": (
            "durable_checkpoint_independent_reload_and_10m_positive_progress"
        ),
        "durable_evidence": {
            "controller_generation_schema": (
                HANDOVER_CONTROLLER_GENERATION_SCHEMA
            ),
            "resume_receipt_schema": HANDOVER_RESUME_RECEIPT_SCHEMA,
            "progress_observation_schema": (
                HANDOVER_PROGRESS_OBSERVATION_SCHEMA
            ),
            "publication": "O_EXCL_0600_file_fsync_parent_fsync",
            "elapsed_time_authority": "physical_file_ctime_ns",
            "mutable_state_is_authority": False,
            "max_generations": HANDOVER_EVIDENCE_MAX_GENERATIONS,
            "max_observations": HANDOVER_EVIDENCE_MAX_OBSERVATIONS,
        },
        "old_route_signal_authorized_here": False,
    }


def _validate_argv(
    value: Any,
    *,
    label: str,
    project_root: Path,
    expected_entrypoint_sha256: str,
) -> list[str]:
    if not isinstance(value, list) or len(value) < 2 or not all(
        isinstance(item, str) and item for item in value
    ):
        raise RecoveryControllerError(f"{label} must be a nonempty argv list")
    if value[1] in {"-c", "-m"} or value[0] in {"sh", "bash", "zsh"}:
        raise RecoveryControllerError(f"{label} may not use shell/code-string execution")
    executable = _require_absolute(value[0], label=f"{label}[0]", existing="file")
    entrypoint = _require_absolute(value[1], label=f"{label}[1]", existing="file")
    try:
        entrypoint.relative_to(project_root)
    except ValueError as exc:
        raise RecoveryControllerError(f"{label} entrypoint escaped project root") from exc
    if not _is_sha256(expected_entrypoint_sha256):
        raise RecoveryControllerError(f"{label} entrypoint SHA is invalid")
    if sha256_file(entrypoint) != expected_entrypoint_sha256:
        raise RecoveryControllerError(f"{label} entrypoint SHA mismatch")
    return [str(executable), str(entrypoint), *value[2:]]


def _release_pin_state(pins: Mapping[str, Any]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for field in REQUIRED_RELEASE_PINS:
        value = pins.get(field)
        if not _is_git_sha(value):
            missing.append(field)
    if pins.get("science_commit") not in {None, SCIENCE_RELEASE_COMMIT}:
        raise RecoveryControllerError("science release pin changed")
    return not missing, missing


def _release_requirement_state(
    pins: Mapping[str, Any], adoption: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    _pins_ready, missing = _release_pin_state(pins)
    if not _is_sha256(adoption.get("validator_module_sha256")):
        missing.append("adoption_validator_module_sha256")
    if not _is_sha256(adoption.get("authority_profile_sha256")):
        missing.append("adoption_authority_profile_sha256")
    projections = adoption.get("expected_task_state_projection_sha256")
    if (
        not isinstance(projections, Mapping)
        or set(projections) != {"close", "final"}
        or not all(_is_sha256(value) for value in projections.values())
    ):
        missing.append("adoption_expected_task_state_projection_sha256")
    if adoption.get("validator_api") != ADOPTION_VALIDATOR_API:
        missing.append("adoption_validator_api")
    return not missing, missing


def _git_head(project_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RecoveryControllerError("cannot resolve execution commit") from exc
    if not _is_git_sha(value):
        raise RecoveryControllerError("execution commit is invalid")
    return value


def _release_commits_are_ancestors(
    project_root: Path, *, execution_commit: str, pins: Mapping[str, Any]
) -> bool:
    for value in pins.values():
        if not _is_git_sha(value):
            continue
        result = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "merge-base",
                "--is-ancestor",
                str(value),
                execution_commit,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        if result.returncode != 0:
            return False
    return True


def _execution_tree_clean(project_root: Path) -> bool:
    tracked = subprocess.run(
        ["git", "-C", str(project_root), "diff", "--quiet", "HEAD", "--"],
        check=False,
        timeout=30,
    )
    untracked = subprocess.check_output(
        ["git", "-C", str(project_root), "ls-files", "--others", "--exclude-standard"],
        text=True,
        timeout=30,
    ).strip()
    return tracked.returncode == 0 and not untracked


def _validate_source_authority(source: Mapping[str, Any]) -> dict[str, Any]:
    required_sha_fields = (
        "source_controller_manifest_sha256",
        "close_pass_gate_sha256",
        "failed_final_gate_sha256",
        "failed_shortcut_artifact_sha256",
        "failed_checkpoint_sha256",
        "adaptive_selection_sha256",
        "anchor_indices_sha256",
        "anchor_rows_sha256",
        "failure_indices_sha256",
        "anchor_edges_sha256",
        "close_pair_manifest_sha256",
        "pair_semantics_receipt_sha256",
        "pair_store_manifest_sha256",
        "physical_pairs_sha256",
        "normalized_distances_sha256",
        "close_bitmap_sha256",
        "source_vectors_sha256",
    )
    required_path_fields = tuple(
        field[: -len("_sha256")] + "_path" for field in required_sha_fields
    )
    for field in required_sha_fields:
        if not _is_sha256(source.get(field)):
            raise RecoveryControllerError(f"source authority SHA is absent: {field}")
    for field in required_path_fields:
        _require_absolute(source.get(field), label=f"source_authority.{field}")
    if (
        int(source.get("physical_pair_count", -1)) != EXPECTED_ROWS
        or source.get("pair_store_regenerated") is not False
        or source.get("seed_failure_scan_reexecuted") is not False
        or source.get("source_pair_store_access") != "read_only_zero_copy"
        or source.get("failed_final_gate_status") != "FAILED"
        or source.get("failed_final_reason") != "anchor_epsilon_graph_disconnected"
        or source.get("failed_final_gate_ordinary_pass_eligible") is not False
    ):
        raise RecoveryControllerError("recovery source authority would rerun/copy or bless failed data")
    return dict(source)


def _validate_runtime_inputs(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze every path needed by the existing standardized continuation.

    These are execution inputs, not an invitation to rediscover production
    paths at runtime.  Files and directories must already exist when the
    controller manifest is built; the continuation performs its own stronger
    source-generation closure before consuming them.
    """

    directory_fields = (
        "source_generation_root",
        "upstream_root",
        "dataset_dir",
        "molclr_root",
        "pair_store_owner_root",
    )
    file_fields = (
        "source_csv",
        "distance_checkpoint",
        "dataset_csv",
        "teacher_path",
        "molclr_checkpoint",
        "thresholds_path",
    )
    allowed = {
        *directory_fields,
        *file_fields,
        "expected_sklearn_version",
        "theta_star",
        "cost_cap",
    }
    if set(runtime) != allowed:
        raise RecoveryControllerError("runtime input field set changed")
    result: dict[str, Any] = {}
    for field in directory_fields:
        result[field] = str(
            _require_absolute(
                runtime.get(field), label=f"runtime_inputs.{field}", existing="dir"
            )
        )
    for field in file_fields:
        result[field] = str(
            _require_absolute(
                runtime.get(field), label=f"runtime_inputs.{field}", existing="file"
            )
        )
    sklearn_version = str(runtime.get("expected_sklearn_version") or "")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9.+-]*)?", sklearn_version):
        raise RecoveryControllerError("runtime sklearn version is invalid")
    result["expected_sklearn_version"] = sklearn_version
    for field in ("theta_star", "cost_cap"):
        raw = runtime.get(field)
        if raw is None:
            result[field] = None
            continue
        value = float(raw)
        if not (value >= 0.0 and value < float("inf")):
            raise RecoveryControllerError(f"runtime_inputs.{field} is invalid")
        result[field] = value
    return result


def _validate_stage_spec(
    raw: Mapping[str, Any],
    *,
    stage_id: str,
    project_root: Path,
    controller_root: Path,
) -> dict[str, Any]:
    if raw.get("stage_id") != stage_id or raw.get("kind") != STAGE_KINDS[stage_id]:
        raise RecoveryControllerError(f"stage identity mismatch: {stage_id}")
    if tuple(raw.get("dependencies", ())) != DEPENDENCIES[stage_id]:
        raise RecoveryControllerError(f"stage dependency mismatch: {stage_id}")
    output_dir = _require_absolute(raw.get("output_dir"), label=f"{stage_id}.output_dir")
    terminal_path = _require_absolute(
        raw.get("terminal_path"), label=f"{stage_id}.terminal_path"
    )
    if stage_id != ADOPTION_STAGE:
        try:
            output_dir.relative_to(controller_root)
            terminal_path.relative_to(output_dir)
        except ValueError as exc:
            raise RecoveryControllerError(f"{stage_id} output escaped controller root") from exc
    commands = raw.get("commands")
    if not isinstance(commands, Mapping):
        raise RecoveryControllerError(f"{stage_id}.commands is absent")
    entrypoint_sha = str(raw.get("entrypoint_sha256") or "")
    fresh = _validate_argv(
        commands.get("fresh"),
        label=f"{stage_id}.fresh",
        project_root=project_root,
        expected_entrypoint_sha256=entrypoint_sha,
    )
    resume_raw = commands.get("resume")
    resume = None
    if resume_raw is not None:
        resume = _validate_argv(
            resume_raw,
            label=f"{stage_id}.resume",
            project_root=project_root,
            expected_entrypoint_sha256=entrypoint_sha,
        )
    if stage_id in {SUBSET_STAGE, EXACT_STAGE, DOWNSTREAM_STAGE, FINAL_STAGE} and resume is None:
        raise RecoveryControllerError(
            f"long recovery stage requires an explicit resume argv: {stage_id}"
        )
    bindings_raw = raw.get("argv_bindings")
    if not isinstance(bindings_raw, Mapping) or set(bindings_raw) != REQUIRED_ARGV_BINDING_ROLES[stage_id]:
        raise RecoveryControllerError(f"{stage_id} argv binding roles are incomplete")
    bindings: dict[str, dict[str, str]] = {}
    for role, binding_raw in bindings_raw.items():
        if not isinstance(binding_raw, Mapping):
            raise RecoveryControllerError(f"{stage_id} argv binding is invalid: {role}")
        flag = str(binding_raw.get("flag") or "")
        value = str(binding_raw.get("value") or "")
        if not flag.startswith("--") or not value:
            raise RecoveryControllerError(f"{stage_id} argv binding is empty: {role}")
        for command_name, command in (("fresh", fresh), ("resume", resume)):
            if command is None:
                continue
            matches = sum(
                1
                for index in range(len(command) - 1)
                if command[index] == flag and command[index + 1] == value
            )
            if matches != 1:
                raise RecoveryControllerError(
                    f"{stage_id} {command_name} argv binding mismatch: {role}"
                )
        bindings[str(role)] = {"flag": flag, "value": value}
    if bindings["output"]["value"] != str(output_dir):
        raise RecoveryControllerError(f"{stage_id} output argv binding changed")
    return {
        "stage_id": stage_id,
        "kind": STAGE_KINDS[stage_id],
        "dependencies": list(DEPENDENCIES[stage_id]),
        "output_dir": str(output_dir),
        "terminal_path": str(terminal_path),
        "terminal_schema": str(raw.get("terminal_schema") or ""),
        "entrypoint_sha256": entrypoint_sha,
        "commands": {
            "fresh": fresh,
            "resume": resume,
            "fresh_sha256": stable_json_sha256(fresh),
            "resume_sha256": None if resume is None else stable_json_sha256(resume),
        },
        "argv_bindings": bindings,
        "progress_checkpoint_path": (
            None
            if raw.get("progress_checkpoint_path") is None
            else str(
                _require_absolute(
                    raw["progress_checkpoint_path"],
                    label=f"{stage_id}.progress_checkpoint_path",
                )
            )
        ),
        "progress_field": raw.get("progress_field"),
    }


def build_controller_payload(spec_path: str | Path) -> dict[str, Any]:
    spec_file = _require_absolute(spec_path, label="spec", existing="file")
    spec = _read_json(spec_file, label="recovery controller spec")
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise RecoveryControllerError("recovery controller spec schema mismatch")
    if spec.get("controller_id") != CONTROLLER_ID:
        raise RecoveryControllerError("recovery controller ID mismatch")
    project_root = _require_absolute(
        spec.get("project_root"), label="project_root", existing="dir"
    )
    execution_commit = _git_head(project_root)
    controller_root = _require_absolute(spec.get("controller_root"), label="controller_root")
    cid = str(spec.get("cid") or "")
    if not CID_PATTERN.fullmatch(cid) or controller_root.name != cid:
        raise RecoveryControllerError("fresh recovery CID/root identity mismatch")
    if controller_root.exists():
        raise RecoveryControllerError("fresh controller root already exists")
    controller_manifest_path = _require_absolute(
        spec.get("controller_manifest_path"), label="controller_manifest_path"
    )
    authority_parent = _require_absolute(
        spec.get("adoption_authority_parent"),
        label="adoption_authority_parent",
        existing="dir",
    )
    if controller_root.parent.resolve(strict=True).stat().st_dev != authority_parent.stat().st_dev:
        raise RecoveryControllerError("controller/adoption roots must share the budgeted filesystem")
    try:
        controller_root.relative_to(authority_parent)
    except ValueError:
        pass
    else:
        raise RecoveryControllerError("adoption authority parent may not own science output")
    stages_raw = spec.get("stages")
    if not isinstance(stages_raw, list) or len(stages_raw) != len(STAGE_ORDER):
        raise RecoveryControllerError("exactly five typed stages are required")
    indexed = {str(row.get("stage_id")): row for row in stages_raw if isinstance(row, Mapping)}
    if set(indexed) != set(STAGE_ORDER):
        raise RecoveryControllerError("typed stage set mismatch")
    stages = [
        _validate_stage_spec(
            indexed[stage_id],
            stage_id=stage_id,
            project_root=project_root,
            controller_root=controller_root,
        )
        for stage_id in STAGE_ORDER
    ]
    adoption_output = Path(stages[0]["output_dir"])
    try:
        adoption_parent = adoption_output.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RecoveryControllerError(
            "adoption output must be a direct authority-parent child"
        ) from exc
    if adoption_parent != authority_parent:
        raise RecoveryControllerError("adoption output must be a direct authority-parent child")
    adoption_contract = spec.get("adoption_contract")
    if not isinstance(adoption_contract, Mapping):
        raise RecoveryControllerError("typed adoption contract is absent")
    required_adoption = {
        "receipt_schema",
        "artifact_kind",
        "projection_profile",
        "validator_module",
        "validator_callable",
        "validator_module_sha256",
        "validator_api",
        "receipt_name",
        "ready_marker_name",
        "receipt_status",
        "authority_profile_sha256",
        "expected_task_state_projection_sha256",
    }
    if set(adoption_contract) < required_adoption:
        raise RecoveryControllerError("typed adoption contract is incomplete")
    if adoption_contract.get("recovery_only") is not True:
        raise RecoveryControllerError("adoption must be recovery-only")
    if adoption_contract.get("ordinary_pass_dependency_eligible") is not False:
        raise RecoveryControllerError("adoption cannot be an ordinary PASS dependency")
    if adoption_contract.get("dbscan_partition_proven") is not False:
        raise RecoveryControllerError("adoption may not claim a DBSCAN partition")
    if (
        adoption_contract.get("validator_module")
        != EXPECTED_ADOPTION_VALIDATOR_MODULE
        or adoption_contract.get("validator_callable")
        != EXPECTED_ADOPTION_VALIDATOR_CALLABLE
    ):
        raise RecoveryControllerError("adoption typed validator interface changed")
    if (
        adoption_contract.get("receipt_name")
        != "failed_selection_adoption_receipt.json"
        or adoption_contract.get("ready_marker_name")
        != "RECOVERY_EVIDENCE_READY"
        or adoption_contract.get("receipt_status") != "RECOVERY_ONLY_READY"
        or adoption_contract.get("expected_task_state_projection_sha256")
        != EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256
    ):
        raise RecoveryControllerError("adoption production receipt contract changed")
    resources = spec.get("resources")
    if not isinstance(resources, Mapping):
        raise RecoveryControllerError("resource contract is absent")
    row_count = int(resources.get("row_count", -1))
    vector_dim = int(resources.get("vector_dim", -1))
    subset_size = int(resources.get("subset_size", -1))
    block_size = int(resources.get("block_size", -1))
    if row_count != EXPECTED_ROWS or vector_dim != EXPECTED_VECTOR_DIM:
        raise RecoveryControllerError("production row/vector contract changed")
    derived = derive_output_budget(
        row_count=row_count,
        vector_dim=vector_dim,
        subset_size=subset_size,
        subset_count=len(EXPECTED_SUBSET_NAMES),
        block_size=block_size,
        safety_floor_bytes=int(resources.get("safety_floor_bytes", 0)),
    )
    if resources.get("budget") != derived:
        raise RecoveryControllerError("resource budget is not formula-derived")
    if (
        int(resources.get("subset_max_attempts", -1)) != SUBSET_MAX_ATTEMPTS
        or int(resources.get("partial_stage_archive_count", -1))
        != PARTIAL_STAGE_ARCHIVE_COUNT
        or int(resources.get("partial_stage_archive_max_bytes_each", -1))
        != PARTIAL_STAGE_ARCHIVE_MAX_BYTES
        or int(resources.get("startup_barrier_max_generations", -1))
        != STARTUP_BARRIER_MAX_GENERATIONS
        or int(resources.get("startup_barrier_record_max_bytes", -1))
        != STARTUP_BARRIER_RECORD_MAX_BYTES
        or int(resources.get("startup_barrier_publication_file_multiplier", -1))
        != STARTUP_BARRIER_PUBLICATION_FILE_MULTIPLIER
    ):
        raise RecoveryControllerError("resource retention contract changed")
    if int(resources.get("max_rss_bytes", 0)) != DEFAULT_MAX_RSS_BYTES:
        raise RecoveryControllerError("recovery RSS budget must remain 96GiB")
    if (
        int(resources.get("subset_max_rss_bytes", 0))
        != DEFAULT_SUBSET_MAX_RSS_BYTES
        or int(resources.get("subset_rss_working_margin_bytes", 0))
        != DEFAULT_SUBSET_RSS_WORKING_MARGIN_BYTES
        or resources.get("subset_max_rss_scope") != SUBSET_MAX_RSS_SCOPE
        or resources.get("subset_rss_budget_semantics")
        != SUBSET_RSS_BUDGET_SEMANTICS
    ):
        raise RecoveryControllerError("production subset RSS contract changed")
    if resources.get("old_brute_handover") != handover_contract():
        raise RecoveryControllerError("old-brute handover contract changed")
    if (
        resources.get("max_rss_scope")
        != "exact_dbscan_process_with_native_peak_certificate"
    ):
        raise RecoveryControllerError("recovery RSS budget scope changed")
    if resources.get("proc_root", "/proc") != "/proc":
        raise RecoveryControllerError("production process authority must be /proc")
    thread_count = int(resources.get("thread_count", 0))
    if not MIN_THREAD_COUNT <= thread_count <= MAX_THREAD_COUNT:
        raise RecoveryControllerError(
            "CPU coexistence thread count must remain between 8 and 12"
        )
    if resources.get("cpu_only") is not True or resources.get("gpu_lock_required") is not False:
        raise RecoveryControllerError("recovery route must be CPU-only and GPU-lock-free")
    probe = resources.get("coexistence_probe")
    if (
        not isinstance(probe, Mapping)
        or int(probe.get("min_progress_rows", 0)) <= 0
        or float(probe.get("max_load_per_cpu", 0)) <= 0
        or not 0 < float(probe.get("max_iowait_fraction", 0)) < 1
        or int(probe.get("timeout_seconds", 0)) <= 0
    ):
        raise RecoveryControllerError("CPU coexistence probe is incomplete")
    pins = spec.get("release_pins")
    if not isinstance(pins, Mapping):
        raise RecoveryControllerError("release pins are absent")
    deployment_authorized = spec.get("production_deployment_authorized")
    if deployment_authorized not in {True, False}:
        raise RecoveryControllerError(
            "production deployment authorization must be explicit"
        )
    if (
        deployment_authorized is True
        and pins.get("controller_commit") != execution_commit
    ):
        raise RecoveryControllerError(
            "authorized controller release pin must equal execution commit"
        )
    ready, missing_pins = _release_requirement_state(pins, adoption_contract)
    if not _release_commits_are_ancestors(
        project_root, execution_commit=execution_commit, pins=pins
    ):
        raise RecoveryControllerError("release commit is not an execution ancestor")
    if ready and not _execution_tree_clean(project_root):
        raise RecoveryControllerError("release-ready execution worktree is dirty")
    source_authority = spec.get("source_authority")
    if not isinstance(source_authority, Mapping):
        raise RecoveryControllerError("source authority contract is absent")
    source_authority = _validate_source_authority(source_authority)
    runtime_inputs_raw = spec.get("runtime_inputs")
    if not isinstance(runtime_inputs_raw, Mapping):
        raise RecoveryControllerError("runtime input contract is absent")
    runtime_inputs = _validate_runtime_inputs(runtime_inputs_raw)
    stage_map = {row["stage_id"]: row for row in stages}
    exact_progress_path = Path(stage_map[EXACT_STAGE]["output_dir"]) / "dbscan/checkpoint.json"
    if (
        stage_map[EXACT_STAGE].get("progress_checkpoint_path")
        != str(exact_progress_path)
        or stage_map[EXACT_STAGE].get("progress_field")
        != EXACT_MONOTONIC_PROGRESS_FIELD
    ):
        raise RecoveryControllerError("exact monotonic progress contract changed")

    def bound(stage_id: str, role: str) -> str:
        return str(stage_map[stage_id]["argv_bindings"][role]["value"])

    expected_bindings = {
        (ADOPTION_STAGE, "output"): stage_map[ADOPTION_STAGE]["output_dir"],
        (SUBSET_STAGE, "output"): stage_map[SUBSET_STAGE]["output_dir"],
        (SUBSET_STAGE, "controller_manifest"): str(controller_manifest_path),
        (SUBSET_STAGE, "adoption_gate"): str(
            controller_root / "gates/01_failed_selection_adoption.json"
        ),
        (EXACT_STAGE, "output"): stage_map[EXACT_STAGE]["output_dir"],
        (EXACT_STAGE, "controller_manifest"): str(controller_manifest_path),
        (EXACT_STAGE, "adoption_gate"): str(
            controller_root / "gates/01_failed_selection_adoption.json"
        ),
        (EXACT_STAGE, "subset_gate"): str(
            controller_root / "gates/02_production_subset_equivalence.json"
        ),
        (DOWNSTREAM_STAGE, "output"): stage_map[DOWNSTREAM_STAGE]["output_dir"],
        (DOWNSTREAM_STAGE, "controller_manifest"): str(controller_manifest_path),
        (DOWNSTREAM_STAGE, "exact_gate"): str(
            controller_root / "gates/03_exact_component_recovery.json"
        ),
        (FINAL_STAGE, "output"): stage_map[FINAL_STAGE]["output_dir"],
        (FINAL_STAGE, "controller_manifest"): str(controller_manifest_path),
        (FINAL_STAGE, "adoption_gate"): str(
            controller_root / "gates/01_failed_selection_adoption.json"
        ),
        (FINAL_STAGE, "subset_gate"): str(
            controller_root / "gates/02_production_subset_equivalence.json"
        ),
        (FINAL_STAGE, "exact_gate"): str(
            controller_root / "gates/03_exact_component_recovery.json"
        ),
        (FINAL_STAGE, "downstream_gate"): str(
            controller_root / "gates/04_component_downstream_radius_ab.json"
        ),
    }
    for (stage_id, role), expected in expected_bindings.items():
        if bound(stage_id, role) != str(expected):
            raise RecoveryControllerError(
                f"typed stage argv authority mismatch: {stage_id}.{role}"
            )
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "cid": cid,
        "project_root": str(project_root),
        "execution_commit": execution_commit,
        "controller_root": str(controller_root),
        "controller_manifest_path": str(controller_manifest_path),
        "adoption_authority_parent": str(authority_parent),
        "spec_path": str(spec_file),
        "spec_sha256": sha256_file(spec_file),
        "stages": stages,
        "stage_order": list(STAGE_ORDER),
        "adoption_contract": dict(adoption_contract),
        "source_authority": source_authority,
        "runtime_inputs": runtime_inputs,
        "resources": dict(resources),
        "release_pins": dict(pins),
        "release_ready": ready,
        "missing_release_pins": missing_pins,
        "failed_evidence_is_ordinary_pass": False,
        "ordinary_dependencies_may_consume_adoption": False,
        "matrix_may_consume_adoption": False,
        "mut_may_consume_adoption": False,
        "production_deployment_authorized": deployment_authorized,
        "hpc_used": False,
        "gpu_used": False,
    }
    payload["scientific_identity_sha256"] = stable_json_sha256(payload)
    return payload


def validate_controller_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    identity_sha = payload.get("scientific_identity_sha256")
    projected = dict(payload)
    projected.pop("scientific_identity_sha256", None)
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA
        or payload.get("controller_id") != CONTROLLER_ID
        or not _is_git_sha(payload.get("execution_commit"))
        or payload.get("execution_commit") != _git_head(Path(payload["project_root"]))
        or payload.get("stage_order") != list(STAGE_ORDER)
        or identity_sha != stable_json_sha256(projected)
        or payload.get("failed_evidence_is_ordinary_pass") is not False
        or payload.get("ordinary_dependencies_may_consume_adoption") is not False
        or payload.get("matrix_may_consume_adoption") is not False
        or payload.get("mut_may_consume_adoption") is not False
        or payload.get("production_deployment_authorized") not in {True, False}
        or payload.get("hpc_used") is not False
        or payload.get("gpu_used") is not False
        or not isinstance(payload.get("runtime_inputs"), Mapping)
    ):
        raise RecoveryControllerError("controller manifest closure mismatch")
    stages = payload.get("stages")
    if not isinstance(stages, list) or [row.get("stage_id") for row in stages] != list(STAGE_ORDER):
        raise RecoveryControllerError("controller typed DAG order mismatch")
    for row in stages:
        stage_id = str(row["stage_id"])
        if row.get("kind") != STAGE_KINDS[stage_id] or tuple(row.get("dependencies", ())) != DEPENDENCIES[stage_id]:
            raise RecoveryControllerError(f"controller dependency closure mismatch: {stage_id}")
    runtime_inputs = payload.get("runtime_inputs")
    if (
        not isinstance(runtime_inputs, Mapping)
        or dict(runtime_inputs) != _validate_runtime_inputs(runtime_inputs)
    ):
        raise RecoveryControllerError("controller runtime input closure mismatch")
    ready, missing = _release_requirement_state(
        payload.get("release_pins", {}), payload.get("adoption_contract", {})
    )
    if (
        payload.get("production_deployment_authorized") is True
        and payload.get("release_pins", {}).get("controller_commit")
        != payload.get("execution_commit")
    ):
        raise RecoveryControllerError(
            "authorized controller release pin differs from execution commit"
        )
    if payload.get("release_ready") is not ready or payload.get("missing_release_pins") != missing:
        raise RecoveryControllerError("release pin state mismatch")
    return {"status": "PASS", "release_ready": ready, "missing_release_pins": missing}


def load_bound_controller_manifest(path: str | Path) -> dict[str, Any]:
    """Open one launchable manifest at its frozen path with runtime bindings.

    Stage entrypoints use this public loader too; invoking a stage CLI directly
    therefore cannot bypass release pins, worktree ancestry/cleanliness, or the
    explicit production-deployment authorization owned by the controller.
    """

    source = _require_absolute(path, label="controller manifest", existing="file")
    manifest = _read_json(source, label="controller manifest")
    validate_controller_manifest(manifest)
    if str(source) != manifest.get("controller_manifest_path"):
        raise RecoveryControllerError("controller manifest was copied to an unbound path")
    result = dict(manifest)
    result["manifest_path"] = str(source)
    result["manifest_sha256"] = sha256_file(source)
    if result.get("release_ready") is not True:
        raise RecoveryControllerError(
            "RELEASE_PINS_UNSET:"
            + ",".join(result.get("missing_release_pins", []))
        )
    if result.get("production_deployment_authorized") is not True:
        raise RecoveryControllerError("PRODUCTION_DEPLOYMENT_NOT_AUTHORIZED")
    if (
        result.get("release_pins", {}).get("controller_commit")
        != result.get("execution_commit")
    ):
        raise RecoveryControllerError(
            "authorized controller release pin changed before launch"
        )
    project_root = Path(result["project_root"])
    if not _execution_tree_clean(project_root):
        raise RecoveryControllerError("release execution worktree became dirty")
    if not _release_commits_are_ancestors(
        project_root,
        execution_commit=str(result["execution_commit"]),
        pins=result["release_pins"],
    ):
        raise RecoveryControllerError("release ancestry changed before launch")
    for stage in result["stages"]:
        entrypoint = Path(stage["commands"]["fresh"][1]).resolve(strict=True)
        if sha256_file(entrypoint) != stage["entrypoint_sha256"]:
            raise RecoveryControllerError(
                f"stage entrypoint changed before launch: {stage['stage_id']}"
            )
    return result


def build_controller_manifest(*, spec_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    output = _require_absolute(output_path, label="output_path")
    payload = build_controller_payload(spec_path)
    if str(output) != payload.get("controller_manifest_path"):
        raise RecoveryControllerError("build output changed controller manifest identity")
    validate_controller_manifest(payload)
    _write_new_json(output, payload)
    reopened = _read_json(output, label="built controller manifest")
    validate_controller_manifest(reopened)
    return reopened


def _stage(payload: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    for row in payload["stages"]:
        if row.get("stage_id") == stage_id:
            return row
    raise RecoveryControllerError(f"unknown stage: {stage_id}")


def _artifact_binding(path: Path) -> dict[str, Any]:
    _physical_file(path, label="stage terminal")
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def validate_ordinary_pass_dependency(path: str | Path) -> dict[str, Any]:
    """Reject recovery-only evidence at every Matrix/Mut/generic boundary."""

    payload = _read_json(path, label="ordinary dependency")
    if (
        payload.get("schema_version") == STAGE_GATE_SCHEMA
        and payload.get("stage_id") == ADOPTION_STAGE
    ) or payload.get("recovery_only") is True or payload.get(
        "failed_evidence_adopted_for_recovery_only"
    ) is True:
        raise RecoveryControllerError("RECOVERY_ONLY_EVIDENCE_IS_NOT_ORDINARY_PASS")
    if payload.get("ordinary_pass_dependency_eligible") is not True:
        raise RecoveryControllerError("dependency is not ordinary-PASS eligible")
    if payload.get("schema_version") == TERMINAL_SCHEMA:
        manifest_path = _require_absolute(
            payload.get("controller_manifest_path"),
            label="terminal controller manifest",
            existing="file",
        )
        if sha256_file(manifest_path) != payload.get("controller_manifest_sha256"):
            raise RecoveryControllerError("terminal controller manifest changed")
        manifest = _read_json(manifest_path, label="terminal controller manifest")
        validate_controller_manifest(manifest)
        manifest = dict(manifest)
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        validate_controller_terminal(manifest)
    return payload


def _load_adoption_validator(contract: Mapping[str, Any]) -> Callable[..., Mapping[str, Any]]:
    module_name = str(contract.get("validator_module") or "")
    callable_name = str(contract.get("validator_callable") or "")
    module_sha = contract.get("validator_module_sha256")
    if (
        module_name != EXPECTED_ADOPTION_VALIDATOR_MODULE
        or callable_name != EXPECTED_ADOPTION_VALIDATOR_CALLABLE
    ):
        raise RecoveryControllerError("adoption canonical validator is not pinned")
    module = importlib.import_module(module_name)
    module_file = Path(str(getattr(module, "__file__", "") or "")).resolve(strict=True)
    if not _is_sha256(module_sha) or sha256_file(module_file) != module_sha:
        raise RecoveryControllerError("adoption canonical validator module SHA mismatch")
    value = getattr(module, callable_name, None)
    if not callable(value):
        raise RecoveryControllerError("adoption canonical validator callable is absent")
    return value


def validate_typed_adoption_receipt(
    *,
    manifest: Mapping[str, Any],
    validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    stage = _stage(manifest, ADOPTION_STAGE)
    receipt_path = Path(stage["terminal_path"])
    receipt_before = _read_json(receipt_path, label="typed failed-selection adoption receipt")
    receipt_sha_before = sha256_file(receipt_path)
    contract = manifest["adoption_contract"]
    if validator is None:
        validator = _load_adoption_validator(contract)
    validated = validator(output_dir=receipt_path.parent)
    if not isinstance(validated, Mapping):
        raise RecoveryControllerError("adoption canonical validator returned no mapping")
    receipt_after = _read_json(receipt_path, label="typed failed-selection adoption receipt")
    receipt_sha_after = sha256_file(receipt_path)
    if receipt_before != receipt_after or receipt_sha_before != receipt_sha_after:
        raise RecoveryControllerError("adoption receipt changed around canonical validation")
    required = {
        "schema_version": contract["receipt_schema"],
        "artifact_kind": contract["artifact_kind"],
        "status": contract["receipt_status"],
        "failed_evidence_adopted_for_recovery_only": True,
        "ordinary_pass_dependency_eligible": False,
        "generic_pass_marker_created": False,
        "scientific_result_pass": False,
        "dbscan_partition_pass": False,
        "source_final_status": "FAILED",
        "source_recomputed": False,
        "source_copied": False,
        "large_payload_copied": False,
    }
    for field, expected in required.items():
        if receipt_after.get(field) != expected:
            raise RecoveryControllerError(f"typed adoption field mismatch: {field}")
    if dict(validated) != receipt_after:
        raise RecoveryControllerError("adoption typed validator did not return the receipt")
    failed = receipt_after.get("failed_selection")
    if not isinstance(failed, Mapping) or failed.get("dbscan_partition_proven") is not False:
        raise RecoveryControllerError("adoption receipt could be mistaken for a partition")
    if receipt_after.get("authority_profile_sha256") != contract["authority_profile_sha256"]:
        raise RecoveryControllerError("adoption authority profile mismatch")
    task_authority = receipt_after.get("task_state_authority")
    expected_projections = contract["expected_task_state_projection_sha256"]
    if not isinstance(task_authority, Mapping):
        raise RecoveryControllerError("adoption task-state projection authority is absent")
    for task_name in ("close", "final"):
        projection = task_authority.get(task_name)
        projection_sha = task_authority.get(f"{task_name}_projection_sha256")
        if (
            not isinstance(projection, Mapping)
            or projection_sha != expected_projections[task_name]
            or stable_json_sha256(projection) != projection_sha
        ):
            raise RecoveryControllerError(
                f"adoption canonical task-state projection mismatch: {task_name}"
            )
    for observation_field in (
        "task_state_observations",
        "terminal_reopen_task_state_observations",
    ):
        observations = receipt_after.get(observation_field)
        if not isinstance(observations, Mapping) or set(observations) != {"close", "final"}:
            raise RecoveryControllerError(
                f"adoption mutable-state observations are absent: {observation_field}"
            )
        for task_name in ("close", "final"):
            rows = observations.get(task_name)
            if not isinstance(rows, list) or len(rows) != 2:
                raise RecoveryControllerError(
                    f"adoption mutable-state double read changed: {task_name}"
                )
            for row in rows:
                if (
                    not isinstance(row, Mapping)
                    or not _is_sha256(row.get("observed_sha256"))
                    or row.get("projection_sha256")
                    != expected_projections[task_name]
                    or row.get("projection") != task_authority[task_name]
                ):
                    raise RecoveryControllerError(
                        f"adoption mutable-state projection observation changed: {task_name}"
                    )
    if receipt_after.get("terminal_marker") != contract["ready_marker_name"]:
        raise RecoveryControllerError("adoption terminal marker contract changed")
    source_rows = receipt_after.get("source_artifacts")
    if not isinstance(source_rows, list):
        raise RecoveryControllerError("adoption source-artifact closure is absent")
    source_bindings = {
        (str(row.get("path") or ""), str(row.get("sha256") or ""))
        for row in source_rows
        if isinstance(row, Mapping)
    }
    for sha_field, expected_sha in manifest["source_authority"].items():
        if not sha_field.endswith("_sha256"):
            continue
        path_field = sha_field[: -len("_sha256")] + "_path"
        expected_path = manifest["source_authority"].get(path_field)
        if (str(expected_path or ""), str(expected_sha or "")) not in source_bindings:
            raise RecoveryControllerError(
                f"adoption receipt did not bind source authority: {sha_field}"
            )
    ready_path = receipt_path.parent / str(contract["ready_marker_name"])
    _physical_file(ready_path, label="recovery evidence ready marker")
    if (receipt_path.parent / "PASS").exists():
        raise RecoveryControllerError("adoption output illegally contains generic PASS")
    output = Path(stage["output_dir"]).resolve(strict=True)
    parent = Path(manifest["adoption_authority_parent"]).resolve(strict=True)
    if (
        output.parent != parent
        or receipt_path.resolve(strict=True).parent != output
        or receipt_path.name != contract["receipt_name"]
    ):
        raise RecoveryControllerError("adoption receipt is not in the fixed direct child")
    allowed = set(contract.get("authority_parent_allowed_entries", [])) | {output.name}
    observed = {entry.name for entry in parent.iterdir()}
    if observed != allowed:
        raise RecoveryControllerError("adoption authority parent is not a unique-child authority")
    return {
        "receipt": receipt_after,
        "receipt_path": str(receipt_path.resolve(strict=True)),
        "receipt_sha256": receipt_sha_after,
        "canonical_validation": {
            "authority_profile_sha256": receipt_after["authority_profile_sha256"],
            "task_state_projection_sha256": dict(expected_projections),
            "double_read_observation_sets": 2,
            "ready_marker_path": str(ready_path),
            "ready_marker_sha256": sha256_file(ready_path),
        },
    }


def _validate_subset_terminal(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stage = _stage(manifest, SUBSET_STAGE)
    path = Path(stage["terminal_path"]).resolve(strict=True)
    receipt = _read_json(path, label="production subset stage receipt")
    subset_path = _require_absolute(
        receipt.get("subset_manifest_path"),
        label="production subset equivalence",
        existing="file",
    )
    value = _read_json(subset_path, label="production subset equivalence")
    from src.baselines.comrecgc.production_subset_audit import (
        ProductionSubsetAuditContract,
        SCHEMA_VERSION as SUBSET_SCHEMA,
    )

    expected_subset_contract = asdict(
        ProductionSubsetAuditContract(
            eps=0.02,
            min_samples=3,
            radius=0.02,
            recourse_size=100,
            subset_size=int(manifest["resources"]["subset_size"]),
            seed=0,
            scan_block_size=int(manifest["resources"]["block_size"]),
            query_block_size=64,
            max_rss_bytes=int(
                manifest["resources"]["subset_max_rss_bytes"]
            ),
            working_rss_margin_bytes=int(
                manifest["resources"]["subset_rss_working_margin_bytes"]
            ),
            expected_sklearn_version=str(
                manifest["runtime_inputs"]["expected_sklearn_version"]
            ),
            expected_theta=0.1,
            expected_parent_count=EXPECTED_PARENT_COUNT,
            expected_candidate_count=EXPECTED_CANDIDATE_COUNT,
            expected_physical_pair_count=EXPECTED_ROWS,
        )
    )
    rss_budget = value.get("rss_budget")
    if not isinstance(rss_budget, Mapping):
        raise RecoveryControllerError("production subset RSS evidence is absent")
    try:
        baseline_rss = int(rss_budget["baseline_rss_bytes"])
        working_margin = int(rss_budget["working_rss_margin_bytes"])
        effective_max_rss = int(rss_budget["effective_max_rss_bytes"])
        authorized_max_rss = int(rss_budget["authorized_max_rss_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryControllerError(
            "production subset RSS evidence is malformed"
        ) from exc
    expected_margin = int(
        manifest["resources"]["subset_rss_working_margin_bytes"]
    )
    expected_authorized = int(manifest["resources"]["subset_max_rss_bytes"])
    if (
        baseline_rss < 0
        or working_margin != expected_margin
        or authorized_max_rss != expected_authorized
        or effective_max_rss != baseline_rss + expected_margin
        or effective_max_rss > expected_authorized
        or rss_budget.get("mode") != SUBSET_RSS_BUDGET_SEMANTICS
        or rss_budget.get("measurement")
        != "max_of_process_VmRSS_and_VmHWM_with_ru_maxrss_fallback"
        or rss_budget.get("measured_after_full_source_selection_scans") is not True
    ):
        raise RecoveryControllerError("production subset RSS evidence changed")

    if (
        receipt.get("schema_version") != SUBSET_STAGE_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("run_complete") is not True
        or receipt.get("controller_manifest_path") != manifest.get("manifest_path")
        or receipt.get("controller_manifest_sha256")
        != manifest.get("manifest_sha256")
        or receipt.get("subset_manifest_path") != str(subset_path)
        or receipt.get("subset_manifest_sha256") != sha256_file(subset_path)
        or receipt.get("ordinary_pass_dependency_eligible") is not False
        or receipt.get("recovery_only") is not True
        or receipt.get("observed_environment") != _frozen_stage_environment(manifest)
        or value.get("schema_version") != SUBSET_SCHEMA
        or value.get("status") != "PASS"
        or value.get("run_complete") is not True
        or value.get("all_subsets_pass") is not True
        or value.get("full_production_dbscan_equivalence_claimed") is not False
        or value.get("scope_warning") != "subset PASS is not full-production DBSCAN PASS"
        or value.get("approximation_used") is not False
        or value.get("contract") != expected_subset_contract
        or set(value.get("subsets", {})) != set(EXPECTED_SUBSET_NAMES)
    ):
        raise RecoveryControllerError("production subset preflight contract mismatch")
    # PASS-last is part of the upstream subset authority, but it is never used
    # as a generic controller dependency.
    pass_path = subset_path.parent / "PASS"
    if pass_path.read_bytes() != b"PASS\n":
        raise RecoveryControllerError("production subset PASS-last marker is absent")
    for name in EXPECTED_SUBSET_NAMES:
        row = value["subsets"][name]
        audit_path = Path(str(row.get("audit_path") or "")).resolve(strict=True)
        if audit_path.parent.parent != subset_path.parent or sha256_file(audit_path) != row.get("audit_sha256"):
            raise RecoveryControllerError(f"production subset artifact mismatch: {name}")
        audit = _read_json(audit_path, label=f"production subset audit: {name}")
        if audit.get("rss_budget") != rss_budget:
            raise RecoveryControllerError(
                f"production subset RSS evidence mismatch: {name}"
            )
    if value.get("result_sha256") != stable_json_sha256(
        {key: item for key, item in value.items() if key != "result_sha256"}
    ):
        raise RecoveryControllerError("production subset result hash mismatch")
    source = manifest["source_authority"]
    subset_source = value.get("source_authority")
    if (
        value.get("close_pair_contract_sha256")
        != source.get("close_pair_manifest_sha256")
        or value.get("physical_vectors_sha256")
        != source.get("source_vectors_sha256")
        or value.get("physical_pairs_sha256")
        != source.get("physical_pairs_sha256")
        or value.get("close_bitmap_sha256") != source.get("close_bitmap_sha256")
        or not isinstance(subset_source, Mapping)
        or subset_source.get("pair_semantics_contract_sha256")
        != source.get("pair_semantics_receipt_sha256")
        or subset_source.get("pair_store_manifest_sha256")
        != source.get("pair_store_manifest_sha256")
    ):
        raise RecoveryControllerError("production subset source authority mismatch")
    return {"manifest": value, **_artifact_binding(path)}


def _validate_exact_terminal(manifest: Mapping[str, Any], adoption_gate: Mapping[str, Any]) -> dict[str, Any]:
    stage = _stage(manifest, EXACT_STAGE)
    path = Path(stage["terminal_path"]).resolve(strict=True)
    receipt = _read_json(path, label="exact component recovery stage terminal")
    from src.baselines.comrecgc.external_memory_dbscan import (
        ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
        _validate_component_recovery_closure,
    )

    dbscan_path = _require_absolute(
        receipt.get("dbscan_manifest_path"),
        label="exact component DBSCAN manifest",
        existing="file",
    )
    value = _read_json(dbscan_path, label="exact component DBSCAN terminal")
    if (
        receipt.get("schema_version") != EXACT_STAGE_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("run_complete") is not True
        or receipt.get("ordinary_pass_dependency_eligible") is not False
        or receipt.get("recovery_only") is not True
        or receipt.get("observed_environment") != _frozen_stage_environment(manifest)
        or receipt.get("dbscan_manifest_sha256") != sha256_file(dbscan_path)
        or receipt.get("dbscan_manifest_path") != str(dbscan_path)
        or value.get("run_complete") is not True
        or value.get("clustering_path") != ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
        or value.get("approximation_used") is not False
        or int(value.get("num_samples", -1)) != EXPECTED_ROWS
        or int(value.get("core_count", -1)) != EXPECTED_ROWS
        or int(value.get("noise_count", -1)) != 0
    ):
        raise RecoveryControllerError("exact component recovery terminal mismatch")
    _validate_component_recovery_closure(manifest=value, root=dbscan_path.parent)
    recovery_source = receipt.get("recovery_source_authority")
    adoption_artifact = adoption_gate.get("artifact")
    if (
        not isinstance(recovery_source, Mapping)
        or not isinstance(adoption_artifact, Mapping)
        or recovery_source.get("adoption_receipt_sha256") != adoption_artifact.get("sha256")
        or recovery_source.get("task_state_projection_sha256")
        != manifest["adoption_contract"][
            "expected_task_state_projection_sha256"
        ]
        or recovery_source.get("source_authority_sha256")
        != stable_json_sha256(manifest["source_authority"])
        or recovery_source.get("seed_failure_scan_reexecuted") is not False
        or recovery_source.get("source_seed_failure_ledgers_adopted_read_only")
        is not True
        or recovery_source.get("fresh_component_ledger_derived") is not True
        or recovery_source.get("source_vectors_zero_copy") is not True
        or recovery_source.get("failed_dbscan_terminal_adopted_as_pass") is not False
        or recovery_source.get("source_access") != "read_only"
    ):
        raise RecoveryControllerError("exact recovery did not bind typed adopted selection")
    promotion_path = _require_absolute(
        receipt.get("promotion_manifest_path"),
        label="failed-selection promotion manifest",
        existing="file",
    )
    if sha256_file(promotion_path) != receipt.get("promotion_manifest_sha256"):
        raise RecoveryControllerError("exact recovery promotion binding changed")
    from src.baselines.comrecgc.failed_selection_recovery import (
        PROMOTION_CLAIM_SCHEMA_VERSION,
        PROMOTION_MANIFEST_NAME,
        PROMOTION_SCHEMA_VERSION,
    )

    source = manifest["source_authority"]
    promotion = _read_json(promotion_path, label="failed-selection promotion")
    promotion_root = dbscan_path.parent
    claim_path = _require_absolute(
        promotion.get("promotion_claim_path"),
        label="failed-selection promotion claim",
        existing="file",
    )
    selection_path = _require_absolute(
        promotion.get("selection_manifest_path"),
        label="promoted adaptive selection",
        existing="file",
    )
    if (
        promotion_path != promotion_root / PROMOTION_MANIFEST_NAME
        or promotion.get("schema_version") != PROMOTION_SCHEMA_VERSION
        or promotion.get("status") != "READY_FOR_EXACT_COMPONENT_RECOVERY"
        or promotion.get("work_dir") != str(promotion_root)
        or promotion.get("vectors_path") != source["source_vectors_path"]
        or promotion.get("vectors_sha256") != source["source_vectors_sha256"]
        or promotion.get("source_checkpoint_path")
        != source["failed_checkpoint_path"]
        or promotion.get("source_checkpoint_sha256")
        != source["failed_checkpoint_sha256"]
        or promotion.get("source_selection_manifest_path")
        != source["adaptive_selection_path"]
        or promotion.get("source_selection_manifest_sha256")
        != source["adaptive_selection_sha256"]
        or promotion.get("source_failure_artifact_path")
        != source["failed_shortcut_artifact_path"]
        or promotion.get("source_failure_artifact_sha256")
        != source["failed_shortcut_artifact_sha256"]
        or promotion.get("adoption_receipt_sha256")
        != adoption_artifact.get("sha256")
        or promotion.get("source_authority_sha256")
        != stable_json_sha256(source)
        or promotion.get("seed_failure_scan_reexecuted") is not False
        or promotion.get("source_root_written") is not False
        or promotion.get("source_large_arrays_copied") is not False
        or promotion.get("fresh_checkpoint_rebuilt") is not True
        or promotion.get("failed_terminal_adopted_as_pass") is not False
        or promotion.get("recovery_only") is not True
        or promotion.get("approximation_used") is not False
        or claim_path != promotion_root / "failed_selection_fresh_promotion_claim.json"
        or sha256_file(claim_path) != promotion.get("promotion_claim_sha256")
        or selection_path != promotion_root / "adaptive_anchor_selection.json"
        or sha256_file(selection_path) != promotion.get("selection_manifest_sha256")
    ):
        raise RecoveryControllerError("exact recovery promotion contract changed")
    claim = _read_json(claim_path, label="failed-selection promotion claim")
    from src.baselines.comrecgc.external_memory_dbscan import _load_checkpoint

    source_checkpoint = _load_checkpoint(Path(source["failed_checkpoint_path"]))
    if (
        claim.get("schema_version") != PROMOTION_CLAIM_SCHEMA_VERSION
        or claim.get("work_dir") != str(promotion_root)
        or claim.get("vectors_path") != source["source_vectors_path"]
        or claim.get("vectors_sha256") != source["source_vectors_sha256"]
        or claim.get("source_checkpoint_sha256")
        != source["failed_checkpoint_sha256"]
        or claim.get("source_checkpoint_path")
        != source["failed_checkpoint_path"]
        or claim.get("source_selection_manifest_sha256")
        != source["adaptive_selection_sha256"]
        or claim.get("source_selection_manifest_path")
        != source["adaptive_selection_path"]
        or claim.get("source_failure_artifact_sha256")
        != source["failed_shortcut_artifact_sha256"]
        or claim.get("source_failure_artifact_path")
        != source["failed_shortcut_artifact_path"]
        or claim.get("adoption_receipt_sha256")
        != adoption_artifact.get("sha256")
        or claim.get("adoption_receipt_path") != adoption_artifact.get("path")
        or claim.get("source_authority_sha256") != stable_json_sha256(source)
        or claim.get("contract")
        != source_checkpoint.get("identity", {}).get("contract")
        or claim.get("created_at") != promotion.get("created_at")
    ):
        raise RecoveryControllerError("exact recovery promotion claim changed")
    evidence_path = _require_absolute(
        receipt.get("source_evidence_receipt_path"),
        label="failed-tree evidence receipt",
        existing="file",
    )
    evidence = _read_json(evidence_path, label="failed-tree evidence receipt")
    evidence_unsigned = dict(evidence)
    evidence_identity_sha = evidence_unsigned.pop("receipt_sha256", None)
    adoption_receipt = _read_json(
        adoption_artifact["path"], label="typed adoption receipt for exact recovery"
    )
    evidence_root = promotion_root.parent / "source_evidence"
    artifacts = evidence.get("artifacts")
    if (
        evidence_path != evidence_root / "source_evidence_receipt.json"
        or sha256_file(evidence_path)
        != receipt.get("source_evidence_receipt_sha256")
        or receipt.get("promoted_source_artifact_count") != 13
        or evidence.get("schema_version")
        != "aids_c766_failed_tree_small_evidence_copy_v1"
        or evidence.get("status") != "RECOVERY_ONLY_EVIDENCE_COPIED"
        or evidence.get("promoted_artifact_count") != 13
        or evidence.get("failed_marker_excluded") is not True
        or evidence.get("source_large_arrays_copied") is not False
        or evidence_identity_sha != stable_json_sha256(evidence_unsigned)
        or evidence.get("source_root")
        != adoption_receipt.get("final_task", {}).get("expected_output")
        or evidence.get("target_root") != str(evidence_root)
        or not isinstance(artifacts, list)
        or len(artifacts) != 13
        or (evidence_root / "FAILED.json").exists()
    ):
        raise RecoveryControllerError("failed-tree promoted evidence changed")
    observed_evidence_paths: set[Path] = set()
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise RecoveryControllerError(
                "promoted failed-tree artifact row changed"
            )
        artifact_path = _require_absolute(
            row.get("path"),
            label="promoted failed-tree artifact",
            existing="file",
        )
        try:
            artifact_path.relative_to(evidence_root)
        except ValueError as exc:
            raise RecoveryControllerError(
                "promoted failed-tree artifact escaped"
            ) from exc
        relative = Path(str(row.get("relative_path") or ""))
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative == Path("FAILED.json")
            or artifact_path != (evidence_root / relative).resolve(strict=True)
            or artifact_path in observed_evidence_paths
            or sha256_file(artifact_path) != row.get("sha256")
        ):
            raise RecoveryControllerError("promoted failed-tree artifact changed")
        observed_evidence_paths.add(artifact_path)
    bootstrap_path = _require_absolute(
        receipt.get("continuation_bootstrap_path"),
        label="exact recovery continuation bootstrap",
        existing="file",
    )
    final_root = Path(_stage(manifest, FINAL_STAGE)["output_dir"])
    bootstrap = _read_json(
        bootstrap_path, label="exact recovery continuation bootstrap"
    )
    if (
        bootstrap_path
        != final_root / "exact_recovery_continuation_bootstrap.json"
        or sha256_file(bootstrap_path)
        != receipt.get("continuation_bootstrap_sha256")
        or receipt.get("continuation_bootstrap") != bootstrap
        or bootstrap.get("status") != "READY_FOR_EXTERNAL_COMMON_RECOVERY"
        or bootstrap.get("output_root") != str(final_root)
        or bootstrap.get("common_recourse_started") is not False
        or bootstrap.get("downstream_started") is not False
    ):
        raise RecoveryControllerError("exact recovery continuation bootstrap changed")
    for name, field in (
        ("generation_adoption_manifest.json", "generation_adoption_manifest_sha256"),
        ("upstream_checkout_audit.json", "upstream_checkout_audit_sha256"),
        ("continuation_resume_contract.json", "continuation_resume_contract_sha256"),
    ):
        if sha256_file(final_root / name) != bootstrap.get(field):
            raise RecoveryControllerError(
                f"exact recovery continuation bootstrap artifact changed: {name}"
            )
    proof = _read_json(value["shortcut_proof_path"], label="exact recovery proof")
    if (
        proof.get("unique_seed_component_proven") is not True
        or int(proof.get("seed_component_count", -1)) != 1
        or proof.get("all_points_core_proven") is not True
        or proof.get("exact_multicomponent_partition_proven") is not True
        or proof.get("all_progress_prefixes_complete") is not True
    ):
        raise RecoveryControllerError("exact component proof theorem is incomplete")
    return {
        "manifest": value,
        "stage_receipt": receipt,
        "proof": proof,
        **_artifact_binding(path),
    }


def _validate_downstream_terminal(manifest: Mapping[str, Any], exact_gate: Mapping[str, Any]) -> dict[str, Any]:
    stage = _stage(manifest, DOWNSTREAM_STAGE)
    path = Path(stage["terminal_path"]).resolve(strict=True)
    from src.baselines.comrecgc.external_component_summary import (
        validate_proven_all_core_component_summary,
    )

    result = validate_proven_all_core_component_summary(
        path, pair_indices=None, full_replay=True
    )
    value = _read_json(path, label="streaming component downstream terminal")
    exact_artifact = exact_gate.get("artifact")
    exact_receipt = (
        _read_json(exact_artifact["path"], label="exact stage receipt")
        if isinstance(exact_artifact, Mapping)
        else None
    )
    identity = value.get("scientific_identity")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(exact_artifact, Mapping)
        or not isinstance(exact_receipt, Mapping)
        or identity.get("dbscan_manifest_sha256")
        != exact_receipt.get("dbscan_manifest_sha256")
        or value.get("centroid_reduction_classification") != "PROJECT_EXTENSION"
        or value.get("numeric_decision_disagreement_policy") != "FAIL_CLOSED"
        or int(value.get("float64_radius_decision_disagreement_count", -1)) != 0
        or int(value.get("float64_theta_decision_disagreement_cluster_count", -1)) != 0
        or value.get("strict_radius_comparison_preserved") is not True
        or value.get("radius_filter_operator") != "<"
        or value.get("centroid_norm_filter_operator") != "<"
        or value.get("no_cluster_duplicated_to_fill_recourse_size") is not True
    ):
        raise RecoveryControllerError("multi-component radius A/B terminal mismatch")
    return {
        "manifest": value,
        "selected_count": len(result.selected),
        "official_result": [list(row) for row in result.official_result],
        **_artifact_binding(path),
    }


def _validate_final_terminal(manifest: Mapping[str, Any], dependency_gates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage = _stage(manifest, FINAL_STAGE)
    path = Path(stage["terminal_path"]).resolve(strict=True)
    value = _read_json(path, label="standardized recovery terminal")
    process_group = value.get("observed_process_group")
    if (
        value.get("schema_version") != FINAL_STAGE_RECEIPT_SCHEMA
        or value.get("status") != "PASS"
        or value.get("run_complete") is not True
        or value.get("dataset") != "aids"
        or value.get("method") != "COMRECGC"
        or value.get("controller_manifest_path") != manifest.get("manifest_path")
        or value.get("controller_manifest_sha256")
        != manifest.get("manifest_sha256")
        or value.get("failed_evidence_adopted_as_pass") is not False
        or value.get("seed_failure_scan_reexecuted") is not False
        or value.get("component_downstream_full_replay_pass") is not True
        or value.get("gpu_used") is not False
        or value.get("observed_environment") != _frozen_stage_environment(manifest)
        or not isinstance(process_group, Mapping)
        or int(process_group.get("runner_pid", -1)) <= 0
        or process_group.get("runner_pid") != process_group.get("process_group_id")
    ):
        raise RecoveryControllerError("standardized terminal contract mismatch")
    standard_root = path.parent
    pass_path = standard_root / "PASS"
    if pass_path.read_bytes() != b"PASS\n":
        raise RecoveryControllerError("standardized PASS-last marker is absent")
    # Reopen the common-recourse terminal with the production continuation's
    # strongest component-summary validator.
    from scripts.autodl.run_comrecgc_standardized_continuation import (
        _validate_common_recourse_completion,
    )

    continuation_terminal_path = _require_absolute(
        value.get("continuation_terminal_path"),
        label="standardized continuation terminal",
        existing="file",
    )
    common_terminal_path = _require_absolute(
        value.get("common_terminal_path"),
        label="common recourse terminal",
        existing="file",
    )
    freeze_path = _require_absolute(
        value.get("freeze_manifest_path"),
        label="standardized freeze manifest",
        existing="file",
    )
    if (
        continuation_terminal_path != standard_root / "_RUN_COMPLETE.json"
        or common_terminal_path
        != standard_root / "common_recourse" / "_RUN_COMPLETE.json"
        or freeze_path != standard_root / "standardized" / "freeze_manifest.json"
        or sha256_file(continuation_terminal_path)
        != value.get("continuation_terminal_sha256")
        or sha256_file(common_terminal_path) != value.get("common_terminal_sha256")
        or sha256_file(freeze_path) != value.get("freeze_manifest_sha256")
    ):
        raise RecoveryControllerError("standardized recovery receipt binding changed")
    continuation_terminal = _read_json(
        continuation_terminal_path, label="standardized continuation terminal"
    )
    if (
        continuation_terminal.get("status") != "PASS"
        or continuation_terminal.get("run_complete") is not True
    ):
        raise RecoveryControllerError("standardized continuation did not pass")
    common_terminal = _read_json(common_terminal_path, label="common recourse terminal")
    _validate_common_recourse_completion(marker=common_terminal_path, terminal=common_terminal)
    expected_dependency_shas = [gate["gate_sha256"] for gate in dependency_gates]
    if value.get("typed_dependency_gate_sha256") != expected_dependency_shas:
        raise RecoveryControllerError("standardized receipt lacks typed recovery closure")
    return {
        "manifest": value,
        "common_terminal_sha256": sha256_file(common_terminal_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        **_artifact_binding(path),
    }


def _gate_path(manifest: Mapping[str, Any], stage_id: str) -> Path:
    return Path(manifest["controller_root"]) / "gates" / f"{STAGE_ORDER.index(stage_id) + 1:02d}_{stage_id}.json"


def _current_lock_identity(root: Path) -> dict[str, Any]:
    path = root / ".controller.lock"
    if path.is_symlink():
        raise RecoveryControllerError("controller lock may not be a symlink")
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise RecoveryControllerError("controller lock is not a physical file")
    return {
        "path": str(path),
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "root_identity": _directory_identity(root),
        "gates_identity": _directory_identity(root / "gates"),
        "logs_identity": _directory_identity(root / "logs"),
    }


def _hash_closure_file(path: Path) -> tuple[str, dict[str, int]]:
    if path.is_symlink():
        raise RecoveryControllerError(f"stage closure may not contain a symlink: {path}")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RecoveryControllerError(f"stage closure path is not regular: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RecoveryControllerError("stage closure file changed while opening")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    expected = {
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mode": int(before.st_mode),
        "size": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "ctime_ns": int(before.st_ctime_ns),
        "nlink": int(before.st_nlink),
    }
    if (
        _stat_identity(path) != expected
        or {
            "device": int(after_fd.st_dev),
            "inode": int(after_fd.st_ino),
            "mode": int(after_fd.st_mode),
            "size": int(after_fd.st_size),
            "mtime_ns": int(after_fd.st_mtime_ns),
            "ctime_ns": int(after_fd.st_ctime_ns),
            "nlink": int(after_fd.st_nlink),
        }
        != expected
        or {
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
            "mode": int(after.st_mode),
            "size": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
            "ctime_ns": int(after.st_ctime_ns),
            "nlink": int(after.st_nlink),
        }
        != expected
    ):
        raise RecoveryControllerError("stage closure file changed while hashing")
    return digest.hexdigest(), expected


def _build_stage_closure_inventory(
    manifest: Mapping[str, Any], stage_id: str
) -> dict[str, Any]:
    stage = _stage(manifest, stage_id)
    root = Path(stage["output_dir"])
    if root.is_symlink():
        raise RecoveryControllerError("stage closure root may not be a symlink")
    root = root.resolve(strict=True)
    terminal = Path(stage["terminal_path"]).resolve(strict=True)
    allowed_future_descendants: list[Path] = []
    if stage_id == EXACT_STAGE:
        downstream = Path(_stage(manifest, DOWNSTREAM_STAGE)["output_dir"])
        if downstream.is_absolute():
            downstream = downstream.resolve(strict=False)
        try:
            downstream.relative_to(root)
        except ValueError as exc:
            raise RecoveryControllerError(
                "downstream output is not nested under exact output"
            ) from exc
        if downstream == root:
            raise RecoveryControllerError("downstream output aliases exact output")
        allowed_future_descendants.append(downstream)

    def is_allowed_future(candidate: Path) -> bool:
        logical = candidate.absolute()
        return any(
            logical == allowed or allowed in logical.parents
            for allowed in allowed_future_descendants
        )

    rows: list[dict[str, Any]] = []
    directories: list[str] = []
    for candidate in sorted(root.rglob("*"), key=lambda value: str(value)):
        if is_allowed_future(candidate):
            continue
        if candidate.is_symlink():
            raise RecoveryControllerError(
                f"stage closure contains a symlink: {candidate}"
            )
        if candidate.is_dir():
            directories.append(candidate.resolve(strict=True).relative_to(root).as_posix())
            continue
        if candidate.name.endswith(
            (".publish.tmp", ".copy.tmp", ".replace.tmp")
        ):
            raise RecoveryControllerError(
                f"stage closure contains an unfinished publication: {candidate}"
            )
        if not candidate.is_file():
            raise RecoveryControllerError(
                f"stage closure contains a non-regular path: {candidate}"
            )
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise RecoveryControllerError("stage closure path escaped") from exc
        digest, identity = _hash_closure_file(resolved)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "path": str(resolved),
                "sha256": digest,
                "stat_identity": identity,
                "rehash_on_reopen": int(identity["size"])
                <= CLOSURE_REHASH_MAX_BYTES,
            }
        )
    if not rows or str(terminal) not in {row["path"] for row in rows}:
        raise RecoveryControllerError("stage terminal is absent from closure inventory")
    inventory: dict[str, Any] = {
        "schema_version": CLOSURE_INVENTORY_SCHEMA,
        "stage_id": stage_id,
        "root": str(root),
        "root_identity": _directory_identity(root),
        "terminal_path": str(terminal),
        "artifact_count": len(rows),
        "directories": directories,
        "allowed_future_descendant_roots": [
            str(value) for value in allowed_future_descendants
        ],
        "large_file_policy": "sha256_at_publish_plus_exact_inode_ctime_stat_on_reopen",
        "small_file_rehash_max_bytes": CLOSURE_REHASH_MAX_BYTES,
        "artifacts": rows,
    }
    inventory["inventory_sha256"] = stable_json_sha256(inventory)
    return inventory


def _validate_stage_closure_inventory(
    manifest: Mapping[str, Any], stage_id: str, inventory: Any
) -> None:
    if not isinstance(inventory, Mapping):
        raise RecoveryControllerError(f"stage closure inventory is absent: {stage_id}")
    projected = dict(inventory)
    inventory_sha = projected.pop("inventory_sha256", None)
    stage = _stage(manifest, stage_id)
    root = Path(stage["output_dir"]).resolve(strict=True)
    terminal = Path(stage["terminal_path"]).resolve(strict=True)
    rows = inventory.get("artifacts")
    expected_allowed: list[str] = []
    if stage_id == EXACT_STAGE:
        downstream = Path(_stage(manifest, DOWNSTREAM_STAGE)["output_dir"]).resolve(
            strict=False
        )
        try:
            downstream.relative_to(root)
        except ValueError as exc:
            raise RecoveryControllerError(
                "downstream output escaped exact closure root"
            ) from exc
        expected_allowed = [str(downstream)]
    directories = inventory.get("directories")
    if (
        inventory.get("schema_version") != CLOSURE_INVENTORY_SCHEMA
        or inventory.get("stage_id") != stage_id
        or inventory.get("root") != str(root)
        or inventory.get("root_identity") != _directory_identity(root)
        or inventory.get("terminal_path") != str(terminal)
        or inventory.get("large_file_policy")
        != "sha256_at_publish_plus_exact_inode_ctime_stat_on_reopen"
        or inventory.get("small_file_rehash_max_bytes")
        != CLOSURE_REHASH_MAX_BYTES
        or not isinstance(rows, list)
        or not isinstance(directories, list)
        or inventory.get("allowed_future_descendant_roots") != expected_allowed
        or int(inventory.get("artifact_count", -1)) != len(rows)
        or inventory_sha != stable_json_sha256(projected)
    ):
        raise RecoveryControllerError(f"stage closure inventory changed: {stage_id}")
    allowed_paths = [Path(value) for value in expected_allowed]
    for allowed in allowed_paths:
        if allowed.exists() and (allowed.is_symlink() or not allowed.is_dir()):
            raise RecoveryControllerError(
                f"allowed downstream closure root is not physical: {stage_id}"
            )

    def is_allowed_future(candidate: Path) -> bool:
        logical = candidate.absolute()
        return any(
            logical == allowed or allowed in logical.parents for allowed in allowed_paths
        )

    current_files: set[Path] = set()
    current_directories: set[str] = set()
    for candidate in root.rglob("*"):
        if is_allowed_future(candidate):
            continue
        if candidate.is_symlink():
            raise RecoveryControllerError(
                f"stage closure gained a symlink: {stage_id}"
            )
        if candidate.is_dir():
            current_directories.add(
                candidate.resolve(strict=True).relative_to(root).as_posix()
            )
            continue
        if not candidate.is_file():
            raise RecoveryControllerError(
                f"stage closure gained a non-regular path: {stage_id}"
            )
        current_files.add(candidate.resolve(strict=True))
    if current_directories != set(str(value) for value in directories):
        raise RecoveryControllerError(
            f"stage closure directory set changed: {stage_id}"
        )
    observed: set[Path] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise RecoveryControllerError(f"stage closure row changed: {stage_id}")
        relative = Path(str(row.get("relative_path") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RecoveryControllerError(f"stage closure relative path escaped: {stage_id}")
        path = Path(str(row.get("path") or ""))
        if path != (root / relative).resolve(strict=True) or path in observed:
            raise RecoveryControllerError(f"stage closure path changed: {stage_id}")
        if path.is_symlink() or _stat_identity(path) != row.get("stat_identity"):
            raise RecoveryControllerError(f"stage closure stat changed: {stage_id}")
        size = int(row["stat_identity"]["size"])
        should_rehash = size <= CLOSURE_REHASH_MAX_BYTES
        if row.get("rehash_on_reopen") is not should_rehash:
            raise RecoveryControllerError(f"stage closure rehash policy changed: {stage_id}")
        if should_rehash:
            digest, identity = _hash_closure_file(path)
            if digest != row.get("sha256") or identity != row.get("stat_identity"):
                raise RecoveryControllerError(f"stage closure content changed: {stage_id}")
        elif not _is_sha256(row.get("sha256")):
            raise RecoveryControllerError(f"stage closure SHA is invalid: {stage_id}")
        observed.add(path)
    if terminal not in observed:
        raise RecoveryControllerError(f"stage terminal left closure: {stage_id}")
    if current_files != observed:
        raise RecoveryControllerError(f"stage closure file set changed: {stage_id}")


def _open_gate(manifest: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    # A gate written immediately before a crash is never trusted until the
    # whole controller root is remeasured against the frozen hard cap.
    _disk_preflight(manifest)
    path = _gate_path(manifest, stage_id)
    if _inspect_immutable_publication(path):
        raise RecoveryControllerError(
            f"IMMUTABLE_PUBLICATION_RECONCILIATION_REQUIRED:{path}"
        )
    gate = _read_json(path, label=f"typed stage gate {stage_id}")
    projected = dict(gate)
    gate_sha = projected.pop("gate_sha256", None)
    if (
        gate.get("schema_version") != STAGE_GATE_SCHEMA
        or gate.get("status") != "TYPED_RECOVERY_STAGE_COMPLETE"
        or gate.get("stage_id") != stage_id
        or gate.get("kind") != STAGE_KINDS[stage_id]
        or gate.get("controller_manifest_path") != manifest.get("manifest_path")
        or gate.get("controller_manifest_sha256")
        != manifest.get("manifest_sha256")
        or gate_sha != stable_json_sha256(projected)
        or gate.get("ordinary_pass_dependency_eligible") is not False
        or gate.get("evidence_projection_sha256")
        != stable_json_sha256(gate.get("validation_projection"))
        or gate.get("writer_lock_identity")
        != _current_lock_identity(Path(manifest["controller_root"]))
    ):
        raise RecoveryControllerError(f"typed stage gate mismatch: {stage_id}")
    expected_dependencies = [
        _open_gate(manifest, dependency)["gate_sha256"]
        for dependency in DEPENDENCIES[stage_id]
    ]
    if gate.get("dependency_gate_sha256") != expected_dependencies:
        raise RecoveryControllerError(f"typed dependency hash mismatch: {stage_id}")
    artifact = gate.get("artifact")
    if not isinstance(artifact, Mapping) or sha256_file(artifact["path"]) != artifact.get("sha256"):
        raise RecoveryControllerError(f"typed stage artifact changed: {stage_id}")
    _validate_stage_closure_inventory(
        manifest, stage_id, gate.get("closure_inventory")
    )
    if stage_id == EXACT_STAGE:
        probe_path = Path(str(gate.get("coexistence_probe_path") or ""))
        _validate_coexistence_receipt(manifest, probe_path)
        if sha256_file(probe_path) != gate.get("coexistence_probe_sha256"):
            raise RecoveryControllerError("exact coexistence probe gate binding changed")
    return gate


def open_typed_recovery_gate(
    manifest: Mapping[str, Any], stage_id: str
) -> dict[str, Any]:
    """Public read-only gate reopen for a typed stage runner."""

    if stage_id not in STAGE_ORDER:
        raise RecoveryControllerError(f"unknown typed stage: {stage_id}")
    return _open_gate(manifest, stage_id)


def _publish_stage_gate(
    manifest: Mapping[str, Any],
    *,
    stage_id: str,
    evidence: Mapping[str, Any],
    held: HeldControllerLock,
) -> dict[str, Any]:
    _disk_preflight(manifest)
    dependencies = [_open_gate(manifest, value) for value in DEPENDENCIES[stage_id]]
    validation_projection = json.loads(
        json.dumps(dict(evidence), sort_keys=True, ensure_ascii=True)
    )
    closure_inventory = _build_stage_closure_inventory(manifest, stage_id)
    payload: dict[str, Any] = {
        "schema_version": STAGE_GATE_SCHEMA,
        "status": "TYPED_RECOVERY_STAGE_COMPLETE",
        "controller_id": CONTROLLER_ID,
        "controller_manifest_path": manifest["manifest_path"],
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "stage_id": stage_id,
        "kind": STAGE_KINDS[stage_id],
        "dependency_gate_sha256": [value["gate_sha256"] for value in dependencies],
        "artifact": {"path": evidence["path"], "sha256": evidence["sha256"]},
        "writer_lock_identity": held.identity,
        "recovery_only": True,
        "ordinary_pass_dependency_eligible": False,
        "matrix_dependency_eligible": False,
        "mut_dependency_eligible": False,
        "dbscan_partition_proven": stage_id in {EXACT_STAGE, DOWNSTREAM_STAGE, FINAL_STAGE},
        "validation_projection": validation_projection,
        "evidence_projection_sha256": stable_json_sha256(validation_projection),
        "closure_inventory": closure_inventory,
        "completed_at": _utc_now(),
    }
    if stage_id == EXACT_STAGE:
        probe_path = Path(manifest["controller_root"]) / "coexistence_probe.json"
        _validate_coexistence_receipt(manifest, probe_path)
        payload["coexistence_probe_path"] = str(probe_path)
        payload["coexistence_probe_sha256"] = sha256_file(probe_path)
    payload["gate_sha256"] = stable_json_sha256(payload)
    path = _gate_path(manifest, stage_id)
    encoded = _json_payload_bytes(payload)
    _reserve_output_growth(manifest, [(path, len(encoded), False)])
    held.verify()
    _write_new_bytes(path, encoded)
    held.verify()
    _disk_preflight(manifest)
    return _open_gate(manifest, stage_id)


def validate_stage_terminal(
    manifest: Mapping[str, Any],
    *,
    stage_id: str,
    adoption_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if stage_id == ADOPTION_STAGE:
        result = validate_typed_adoption_receipt(
            manifest=manifest, validator=adoption_validator
        )
        return {
            **result,
            "path": result["receipt_path"],
            "sha256": result["receipt_sha256"],
        }
    if stage_id in {SUBSET_STAGE, EXACT_STAGE, FINAL_STAGE}:
        terminal_path = Path(_stage(manifest, stage_id)["terminal_path"])
        if _inspect_immutable_publication(terminal_path):
            raise RecoveryControllerError(
                f"IMMUTABLE_PUBLICATION_RECONCILIATION_REQUIRED:{terminal_path}"
            )
    if stage_id == SUBSET_STAGE:
        _open_gate(manifest, ADOPTION_STAGE)
        return _validate_subset_terminal(manifest)
    if stage_id == EXACT_STAGE:
        return _validate_exact_terminal(manifest, _open_gate(manifest, ADOPTION_STAGE))
    if stage_id == DOWNSTREAM_STAGE:
        return _validate_downstream_terminal(manifest, _open_gate(manifest, EXACT_STAGE))
    if stage_id == FINAL_STAGE:
        dependencies = [_open_gate(manifest, value) for value in DEPENDENCIES[FINAL_STAGE]]
        return _validate_final_terminal(manifest, dependencies)
    raise RecoveryControllerError(f"unknown typed stage: {stage_id}")


@dataclass(frozen=True)
class HeldControllerLock:
    path: Path
    descriptor: int
    device: int
    inode: int
    root: Path
    root_identity: Mapping[str, int]
    gates_identity: Mapping[str, int]
    logs_identity: Mapping[str, int]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "device": int(self.device),
            "inode": int(self.inode),
            "root_identity": dict(self.root_identity),
            "gates_identity": dict(self.gates_identity),
            "logs_identity": dict(self.logs_identity),
        }

    def verify(self) -> None:
        opened = os.fstat(self.descriptor)
        current = self.path.stat()
        if (
            self.path.is_symlink()
            or not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (self.device, self.inode)
            or (current.st_dev, current.st_ino) != (self.device, self.inode)
            or _directory_identity(self.root) != dict(self.root_identity)
            or _directory_identity(self.root / "gates")
            != dict(self.gates_identity)
            or _directory_identity(self.root / "logs") != dict(self.logs_identity)
        ):
            raise RecoveryControllerError("controller writer lock identity changed")


@contextmanager
def _controller_lock(root: Path) -> Iterator[HeldControllerLock]:
    lock_path = root / ".controller.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        os.close(descriptor)
        raise RecoveryControllerError("another recovery controller owns this root") from exc
    opened = os.fstat(descriptor)
    held = HeldControllerLock(
        lock_path,
        descriptor,
        opened.st_dev,
        opened.st_ino,
        root,
        _directory_identity(root),
        _directory_identity(root / "gates"),
        _directory_identity(root / "logs"),
    )
    held.verify()
    try:
        yield held
    finally:
        try:
            held.verify()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _root_claim_path(manifest: Mapping[str, Any]) -> Path:
    root = Path(manifest["controller_root"])
    return root.parent / (
        f".{root.name}.{manifest['manifest_sha256']}{ROOT_CLAIM_SUFFIX}"
    )


def _root_claim_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ROOT_CLAIM_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "controller_root": str(Path(manifest["controller_root"])),
        "claim_attempt_id": secrets.token_hex(16),
        "claim_nonce": secrets.token_hex(32),
        "created_at": _utc_now(),
    }


def _validate_root_claim_payload(
    manifest: Mapping[str, Any], payload: Any
) -> dict[str, Any]:
    required = {
        "schema_version",
        "controller_id",
        "controller_manifest_sha256",
        "controller_root",
        "claim_attempt_id",
        "claim_nonce",
        "created_at",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RecoveryControllerError("controller root preclaim identity changed")
    attempt = payload.get("claim_attempt_id")
    nonce = payload.get("claim_nonce")
    if (
        payload.get("schema_version") != ROOT_CLAIM_SCHEMA
        or payload.get("controller_id") != CONTROLLER_ID
        or payload.get("controller_manifest_sha256")
        != manifest["manifest_sha256"]
        or payload.get("controller_root")
        != str(Path(manifest["controller_root"]))
        or not isinstance(attempt, str)
        or re.fullmatch(r"[0-9a-f]{32}", attempt) is None
        or not isinstance(nonce, str)
        or re.fullmatch(r"[0-9a-f]{64}", nonce) is None
        or not isinstance(payload.get("created_at"), str)
        or not payload["created_at"]
    ):
        raise RecoveryControllerError("controller root preclaim identity changed")
    return dict(payload)


def _root_claim_stat_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_root_claim_binding(
    manifest: Mapping[str, Any],
    *,
    descriptor: int | None = None,
) -> dict[str, Any]:
    """Reopen one physical claim and bind content plus ABA-sensitive stat."""

    path = _root_claim_path(manifest)
    if path.is_symlink():
        raise RecoveryControllerError("controller root preclaim is not physical")
    owned_descriptor = descriptor is None
    if descriptor is None:
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
        except FileNotFoundError as exc:
            raise RecoveryControllerError("controller root preclaim is absent") from exc
        except OSError as exc:
            raise RecoveryControllerError(
                "controller root preclaim is not physical"
            ) from exc
    try:
        opened = os.fstat(descriptor)
        try:
            named_before = path.lstat()
        except FileNotFoundError as exc:
            raise RecoveryControllerError(
                "controller root preclaim is absent"
            ) from exc
        expected = _root_claim_stat_tuple(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or opened.st_size <= 0
            or opened.st_size > ROOT_CLAIM_MAX_BYTES
            or _root_claim_stat_tuple(named_before) != expected
        ):
            raise RecoveryControllerError(
                "controller root preclaim identity changed"
            )
        chunks: list[bytes] = []
        offset = 0
        while offset < opened.st_size:
            block = os.pread(descriptor, opened.st_size - offset, offset)
            if not block:
                raise RecoveryControllerError(
                    "controller root preclaim identity changed"
                )
            chunks.append(block)
            offset += len(block)
        if os.pread(descriptor, 1, opened.st_size):
            raise RecoveryControllerError("controller root preclaim identity changed")
        encoded = b"".join(chunks)
        try:
            decoded = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryControllerError(
                "controller root preclaim identity changed"
            ) from exc
        payload = _validate_root_claim_payload(manifest, decoded)
        after_fd = os.fstat(descriptor)
        try:
            named_after = path.lstat()
        except FileNotFoundError as exc:
            raise RecoveryControllerError(
                "controller root preclaim identity changed"
            ) from exc
        if (
            _root_claim_stat_tuple(after_fd) != expected
            or _root_claim_stat_tuple(named_after) != expected
        ):
            raise RecoveryControllerError(
                "controller root preclaim identity changed"
            )
        return {
            "path": str(path),
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "mode": int(opened.st_mode),
            "uid": int(opened.st_uid),
            "gid": int(opened.st_gid),
            "nlink": int(opened.st_nlink),
            "size": int(opened.st_size),
            "mtime_ns": int(opened.st_mtime_ns),
            "ctime_ns": int(opened.st_ctime_ns),
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
            "claim_attempt_id": payload["claim_attempt_id"],
            "claim_nonce_sha256": hashlib.sha256(
                payload["claim_nonce"].encode("ascii")
            ).hexdigest(),
        }
    finally:
        if owned_descriptor:
            os.close(descriptor)


def _validate_controller_owner_claim(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(manifest["controller_root"]).resolve(strict=True)
    claim_path = _root_claim_path(manifest)
    conflicting = [
        entry
        for entry in claim_path.parent.iterdir()
        if entry.name.startswith(f".{root.name}.")
        and entry.name.endswith(ROOT_CLAIM_SUFFIX)
        and entry != claim_path
    ]
    if conflicting:
        raise RecoveryControllerError("controller CID has a conflicting root claim")
    preclaim = _read_root_claim_binding(manifest)
    owner_path = root / "owner_claim.json"
    if _inspect_immutable_publication(owner_path):
        raise RecoveryControllerError(
            f"IMMUTABLE_PUBLICATION_RECONCILIATION_REQUIRED:{owner_path}"
        )
    owner = _read_json(owner_path, label="controller owner claim")
    if (
        owner.get("schema_version") != OWNER_SCHEMA
        or owner.get("controller_id") != CONTROLLER_ID
        or owner.get("root") != str(root)
        or owner.get("root_stat_identity") != _directory_identity(root)
        or owner.get("gates_stat_identity") != _directory_identity(root / "gates")
        or owner.get("logs_stat_identity") != _directory_identity(root / "logs")
        or owner.get("root_preclaim") != preclaim
        or owner.get("controller_manifest_sha256") != manifest["manifest_sha256"]
        or not isinstance(owner.get("claimed_at"), str)
    ):
        raise RecoveryControllerError("controller owner claim mismatch")
    return {
        "owner_claim_path": str(owner_path),
        "owner_claim_sha256": sha256_file(owner_path),
        "root_preclaim": preclaim,
    }


def _handover_owner_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable root authority used by every handover record."""

    root = Path(manifest["controller_root"])
    owner = _validate_controller_owner_claim(manifest)
    return {
        "owner_claim_path": owner["owner_claim_path"],
        "owner_claim_sha256": owner["owner_claim_sha256"],
        "root_preclaim": dict(owner["root_preclaim"]),
        "gates_stat_identity": _directory_identity(root / "gates"),
    }


def _handover_stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "nlink": int(value.st_nlink),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _read_handover_artifact(
    path: Path, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Descriptor-reopen one immutable record and return content/stat binding."""

    if path.is_symlink():
        raise RecoveryControllerError(f"{label} is not a physical file")
    try:
        named_before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except (FileNotFoundError, OSError) as exc:
        raise RecoveryControllerError(f"{label} is absent or not physical") from exc
    try:
        opened = os.fstat(descriptor)
        expected = _handover_stat_identity(opened)
        if (
            _handover_stat_identity(named_before) != expected
            or not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or opened.st_size <= 0
            or opened.st_size > HANDOVER_EVIDENCE_MAX_BYTES
        ):
            raise RecoveryControllerError(f"{label} stat closure changed")
        chunks: list[bytes] = []
        offset = 0
        while offset < opened.st_size:
            block = os.pread(descriptor, opened.st_size - offset, offset)
            if not block:
                raise RecoveryControllerError(f"{label} read was truncated")
            chunks.append(block)
            offset += len(block)
        if os.pread(descriptor, 1, opened.st_size):
            raise RecoveryControllerError(f"{label} grew during read")
        encoded = b"".join(chunks)
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryControllerError(f"{label} JSON changed") from exc
        after_fd = os.fstat(descriptor)
        try:
            named_after = path.lstat()
        except FileNotFoundError as exc:
            raise RecoveryControllerError(f"{label} path disappeared") from exc
        if (
            _handover_stat_identity(after_fd) != expected
            or _handover_stat_identity(named_after) != expected
        ):
            raise RecoveryControllerError(f"{label} changed during reopen")
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict):
        raise RecoveryControllerError(f"{label} must contain one JSON object")
    return payload, {
        "path": str(path),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "stat_identity": expected,
    }


def _publish_handover_artifact(
    manifest: Mapping[str, Any], path: Path, payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish once with O_EXCL, complete write, fsync, and parent fsync."""

    root = Path(manifest["controller_root"])
    gates = root / "gates"
    if (
        path.parent != gates
        or gates.is_symlink()
        or gates.resolve(strict=True) != gates
    ):
        raise RecoveryControllerError("handover evidence escaped bound gates directory")
    encoded = _json_payload_bytes(payload)
    if len(encoded) > HANDOVER_EVIDENCE_MAX_BYTES:
        raise RecoveryControllerError("handover evidence exceeds its size bound")
    _reserve_output_growth(manifest, [(path, len(encoded), False)])
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RecoveryControllerError(
            f"handover evidence already exists: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or opened.st_size != 0
        ):
            raise RecoveryControllerError("handover O_EXCL claim identity changed")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RecoveryControllerError(
                    "handover evidence write made no progress"
                )
            offset += written
        os.ftruncate(descriptor, len(encoded))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(gates)
    reopened, binding = _read_handover_artifact(
        path, label="published handover evidence"
    )
    if reopened != dict(payload):
        raise RecoveryControllerError("published handover evidence content changed")
    _disk_preflight(manifest)
    return reopened, binding


def _handover_generation_path(root: Path, generation: int) -> Path:
    if not 0 <= generation < HANDOVER_EVIDENCE_MAX_GENERATIONS:
        raise RecoveryControllerError("handover controller generation is invalid")
    return root / "gates" / f"90_handover_controller_{generation:02d}.json"


def _handover_resume_path(root: Path, generation: int) -> Path:
    if not 1 <= generation < HANDOVER_EVIDENCE_MAX_GENERATIONS:
        raise RecoveryControllerError("handover resume generation is invalid")
    return root / "gates" / f"91_handover_resume_{generation:02d}.json"


def _handover_observation_path(root: Path, generation: int, sequence: int) -> Path:
    if (
        not 1 <= generation < HANDOVER_EVIDENCE_MAX_GENERATIONS
        or not 0 <= sequence < HANDOVER_EVIDENCE_MAX_OBSERVATIONS
    ):
        raise RecoveryControllerError("handover observation coordinate is invalid")
    return (
        root
        / "gates"
        / f"92_handover_progress_{generation:02d}_{sequence:04d}.json"
    )


def _artifact_observed_epoch(binding: Mapping[str, Any]) -> float:
    stat_identity = binding.get("stat_identity")
    if not isinstance(stat_identity, Mapping):
        raise RecoveryControllerError("handover artifact stat binding is absent")
    return int(stat_identity["ctime_ns"]) / 1_000_000_000.0


def _validate_artifact_clock(
    payload: Mapping[str, Any], binding: Mapping[str, Any]
) -> None:
    observed = payload.get("observed_epoch")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or abs(float(observed) - _artifact_observed_epoch(binding))
        > HANDOVER_EVIDENCE_CLOCK_SKEW_SECONDS
    ):
        raise RecoveryControllerError(
            "handover payload timestamp does not match durable ctime"
        )


@contextmanager
def _controller_root_claim_lock(
    manifest: Mapping[str, Any], *, fresh: bool
) -> Iterator[dict[str, Any]]:
    """Own an atomic parent-side claim while finalizing the fresh root.

    The O_EXCL file carries a fresh attempt id and unpredictable nonce bound to
    the CID, root, and immutable controller-manifest SHA.  Resume accepts it
    only when its content hash and complete ctime/mtime/stat receipt still
    match the immutable owner claim.
    """

    path = _root_claim_path(manifest)
    parent = path.parent
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise RecoveryControllerError("controller parent is not physical")
    root = Path(manifest["controller_root"])
    prefix = f".{root.name}."
    collisions = [
        entry
        for entry in parent.iterdir()
        if entry.name.startswith(prefix)
        and entry.name.endswith(ROOT_CLAIM_SUFFIX)
        and entry != path
    ]
    if collisions:
        raise RecoveryControllerError("controller CID has a conflicting root claim")
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    created = False
    if fresh:
        try:
            descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError as exc:
            raise RecoveryControllerError("fresh controller root already exists") from exc
    else:
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as exc:
            raise RecoveryControllerError("controller root preclaim is absent") from exc
        except OSError as exc:
            raise RecoveryControllerError("controller root preclaim is not physical") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RecoveryControllerError(
                "another controller initializer owns this CID"
            ) from exc
        if created:
            payload = _json_payload_bytes(_root_claim_payload(manifest))
            if len(payload) > ROOT_CLAIM_MAX_BYTES:
                raise RecoveryControllerError(
                    "controller root preclaim payload exceeds its bound"
                )
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise RecoveryControllerError(
                        "controller root preclaim write made no progress"
                    )
                offset += written
            os.ftruncate(descriptor, len(payload))
            os.fsync(descriptor)
            _fsync_directory(parent)
        identity = _read_root_claim_binding(
            manifest,
            descriptor=descriptor,
        )
        try:
            yield identity
        finally:
            if _read_root_claim_binding(
                manifest,
                descriptor=descriptor,
            ) != identity:
                raise RecoveryControllerError(
                    "controller root preclaim changed while held"
                )
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _claim_controller_root(manifest: Mapping[str, Any], *, resume: bool) -> Path:
    root = Path(manifest["controller_root"])
    owner_path = root / "owner_claim.json"
    with _controller_root_claim_lock(manifest, fresh=not resume) as preclaim:
        if not resume and (root.exists() or root.is_symlink()):
            raise RecoveryControllerError("fresh controller root already exists")
        if not root.exists():
            root.mkdir(mode=0o755)
            _fsync_directory(root.parent)
        if root.is_symlink() or root.resolve(strict=True) != root:
            raise RecoveryControllerError("controller root is not physical")
        owner_temp = root / ".owner_claim.json.publish.tmp"
        if owner_path.exists() and owner_temp.exists():
            _reconcile_immutable_link_publication(owner_path)
        if not owner_path.exists():
            allowed = {"gates", "logs", owner_temp.name}
            unexpected = {entry.name for entry in root.iterdir()} - allowed
            if unexpected:
                raise RecoveryControllerError(
                    "ownerless controller root contains unexpected artifacts"
                )
            for name in ("gates", "logs"):
                target = root / name
                if not target.exists():
                    target.mkdir(mode=0o755)
                    _fsync_directory(root)
                elif target.is_symlink() or not target.is_dir():
                    raise RecoveryControllerError(
                        f"controller {name} authority is not physical"
                    )
            owner = {
                "schema_version": OWNER_SCHEMA,
                "controller_id": CONTROLLER_ID,
                "root": str(root),
                "root_stat_identity": _directory_identity(root),
                "gates_stat_identity": _directory_identity(root / "gates"),
                "logs_stat_identity": _directory_identity(root / "logs"),
                "root_preclaim": dict(preclaim),
                "controller_manifest_sha256": manifest["manifest_sha256"],
                "claimed_at": _utc_now(),
            }
            _write_new_json(owner_path, owner)
        owner_binding = _validate_controller_owner_claim(manifest)
        if owner_binding["root_preclaim"] != preclaim:
            raise RecoveryControllerError("controller owner preclaim changed while held")
        return root


def _disk_preflight(manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(manifest["controller_root"])
    probe_path = root if root.exists() else root.parent
    usage = shutil.disk_usage(probe_path)
    budget = manifest["resources"]["budget"]
    existing_bytes = 0
    if root.exists():
        for path in root.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    existing_bytes += path.stat().st_size
            except FileNotFoundError:
                pass
    max_output_bytes = int(budget["max_output_bytes"])
    if existing_bytes > max_output_bytes:
        raise RecoveryControllerError(
            "RECOVERY_OUTPUT_BUDGET_EXCEEDED:"
            f"existing={existing_bytes}:maximum={max_output_bytes}"
        )
    remaining = max_output_bytes - existing_bytes
    required = remaining + int(budget["safety_floor_bytes"])
    if usage.free < required:
        raise RecoveryControllerError(
            f"RECOVERY_DISK_HEADROOM_INSUFFICIENT:free={usage.free}:required={required}"
        )
    return {
        "free_bytes": usage.free,
        "existing_output_bytes": existing_bytes,
        "remaining_output_budget_bytes": remaining,
        "required_free_bytes": required,
        "checked_at": _utc_now(),
    }


def _reserve_output_growth(
    manifest: Mapping[str, Any],
    writes: Sequence[tuple[Path, int, bool]],
) -> dict[str, Any]:
    """Reserve exact serialized growth before trusted controller publication.

    Each row is ``(path, new_size, replaces_existing)``. Mutable state uses
    its exact net retained-size delta; immutable gates/terminals/PASS reserve
    their complete encoded length. Post-write preflight remains mandatory for
    race/crash reconciliation.
    """

    snapshot = _disk_preflight(manifest)
    root = Path(manifest["controller_root"]).resolve(strict=True)
    delta = 0
    rows: list[dict[str, Any]] = []
    observed_paths: set[Path] = set()
    for raw_path, raw_size, replaces in writes:
        path = Path(raw_path)
        if path in observed_paths:
            raise RecoveryControllerError("duplicate output-growth reservation path")
        observed_paths.add(path)
        size = int(raw_size)
        if size < 0:
            raise RecoveryControllerError("negative output-growth reservation")
        try:
            path.parent.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise RecoveryControllerError(
                f"output-growth reservation escaped controller root: {path}"
            ) from exc
        old_size = 0
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise RecoveryControllerError(
                    f"reserved output path is not a physical file: {path}"
                )
            old_size = int(path.stat().st_size)
            if not replaces:
                raise RecoveryControllerError(
                    f"immutable reserved output already exists: {path}"
                )
        elif replaces:
            old_size = 0
        growth = max(0, size - old_size) if replaces else size
        delta += growth
        rows.append(
            {
                "path": str(path),
                "new_size": size,
                "old_size": old_size,
                "net_growth": growth,
                "replaces_existing": bool(replaces),
            }
        )
    maximum = int(manifest["resources"]["budget"]["max_output_bytes"])
    projected = int(snapshot["existing_output_bytes"]) + delta
    if projected > maximum:
        raise RecoveryControllerError(
            "RECOVERY_OUTPUT_BUDGET_RESERVATION_EXCEEDED:"
            f"existing={snapshot['existing_output_bytes']}:growth={delta}:"
            f"projected={projected}:maximum={maximum}"
        )
    return {
        "existing_output_bytes": int(snapshot["existing_output_bytes"]),
        "reserved_growth_bytes": delta,
        "projected_output_bytes": projected,
        "maximum_output_bytes": maximum,
        "writes": rows,
    }


def _read_proc_start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> int | None:
    identity = _read_proc_identity(pid, proc_root=proc_root)
    return None if identity is None else identity[1]


def _read_proc_identity(
    pid: int, *, proc_root: Path = Path("/proc")
) -> tuple[str, int] | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        state = fields[0]
        start_ticks = int(fields[19])
        if close < 0 or len(state) != 1 or start_ticks <= 0:
            return None
        return state, start_ticks
    except (FileNotFoundError, PermissionError, OSError, UnicodeError, ValueError, IndexError):
        return None


def _pid_alive(pid: int, start_ticks: int, *, proc_root: Path = Path("/proc")) -> bool:
    identity = _read_proc_identity(pid, proc_root=proc_root)
    return (
        identity is not None
        and identity[1] == start_ticks
        and identity[0] not in {"Z", "X", "x"}
    )


def _proc_argv(pid: int, *, proc_root: Path = Path("/proc")) -> list[str] | None:
    try:
        before = (proc_root / str(pid) / "stat").stat()
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
        after = (proc_root / str(pid) / "stat").stat()
    except (FileNotFoundError, PermissionError):
        return None
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise RecoveryControllerError("worker process changed around cmdline read")
    return [part.decode("utf-8", errors="strict") for part in raw.split(b"\0") if part]


def _proc_cmdline_sha256(
    pid: int,
    start_ticks: int,
    *,
    proc_root: Path = Path("/proc"),
) -> str | None:
    """Hash the raw cmdline only while the exact PID generation stays live."""

    if not _pid_alive(pid, start_ticks, proc_root=proc_root):
        return None
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError):
        return None
    if not raw or not _pid_alive(pid, start_ticks, proc_root=proc_root):
        return None
    return hashlib.sha256(raw).hexdigest()


def _process_group_member_pids(
    process_group_id: int, *, proc_root: Path = Path("/proc")
) -> tuple[int, ...]:
    if process_group_id <= 0:
        raise RecoveryControllerError("invalid worker process-group identity")
    if not proc_root.is_dir():
        raise RecoveryControllerError(
            "worker process-group quiescence requires a physical procfs"
        )
    members: list[int] = []
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            raw = (pid_dir / "stat").read_text(encoding="utf-8")
            close = raw.rfind(")")
            fields = raw[close + 2 :].split()
            process_state = fields[0]
            observed_group = int(fields[2])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, OSError, UnicodeError, ValueError, IndexError) as exc:
            raise RecoveryControllerError(
                f"cannot prove worker process-group quiescence: pid={pid_dir.name}"
            ) from exc
        if observed_group == process_group_id and process_state not in {"Z", "X", "x"}:
            members.append(int(pid_dir.name))
    return tuple(sorted(members))


def _wait_for_process_group_quiescence(
    process_group_id: int,
    *,
    proc_root: Path,
    timeout_seconds: float | None = 30.0,
    poll_seconds: float = 0.05,
) -> None:
    deadline: float | None = None
    while True:
        members = _process_group_member_pids(process_group_id, proc_root=proc_root)
        if not members:
            return
        now = time.monotonic()
        if deadline is None and timeout_seconds is not None:
            deadline = now + timeout_seconds
        if deadline is not None and now >= deadline:
            raise RecoveryControllerError(
                "WORKER_PROCESS_GROUP_NOT_QUIESCENT:"
                f"pgid={process_group_id}:members={list(members)[:16]}"
            )
        time.sleep(poll_seconds)


def _startup_barrier_paths(
    root: Path, *, stage_id: str, generation: int
) -> tuple[Path, Path]:
    if stage_id not in STAGE_ORDER or not 0 <= generation < STARTUP_BARRIER_MAX_GENERATIONS:
        raise RecoveryControllerError("startup barrier generation is invalid")
    logs = root / "logs"
    return (
        logs / f".{stage_id}.exec-startup.lock",
        logs / f".{stage_id}.exec-startup.{generation:02d}.json",
    )


def _stage_target_argv_for_sha(
    stage: Mapping[str, Any], target_sha256: str
) -> list[str]:
    for role in ("fresh", "resume"):
        argv = stage["commands"].get(role)
        if argv is not None and stable_json_sha256(argv) == target_sha256:
            return list(argv)
    raise RecoveryControllerError("startup barrier target is not a frozen stage command")


def _validate_startup_barrier_binding(
    *,
    root: Path,
    stage: Mapping[str, Any],
    binding: Any,
    allowed_phases: set[str],
) -> tuple[list[str], StartupBarrierRecord | None]:
    if not isinstance(binding, Mapping):
        raise RecoveryControllerError("startup barrier binding is absent")
    phase = binding.get("phase")
    common = {
        "schema_version",
        "stage_id",
        "generation",
        "phase",
        "record_path",
        "lock_path",
        "target_argv_sha256",
    }
    armed = common | {"record_sha256", "launcher_argv_sha256"}
    expected_fields = common if phase == "PRE_ARM" else armed
    if (
        phase not in allowed_phases
        or set(binding) != expected_fields
        or binding.get("schema_version") != STARTUP_BARRIER_BINDING_SCHEMA
        or binding.get("stage_id") != stage.get("stage_id")
        or isinstance(binding.get("generation"), bool)
        or not isinstance(binding.get("generation"), int)
    ):
        raise RecoveryControllerError("startup barrier binding schema changed")
    generation = int(binding["generation"])
    lock_path, record_path = _startup_barrier_paths(
        root, stage_id=str(stage["stage_id"]), generation=generation
    )
    if (
        binding.get("lock_path") != str(lock_path)
        or binding.get("record_path") != str(record_path)
    ):
        raise RecoveryControllerError("startup barrier path binding changed")
    target = _stage_target_argv_for_sha(
        stage, str(binding.get("target_argv_sha256") or "")
    )
    if phase == "PRE_ARM":
        try:
            reconcile_interrupted_startup_barrier_publication(
                lock_path=lock_path,
                record_path=record_path,
                timeout_seconds=30.0,
            )
        except Exception as exc:
            raise RecoveryControllerError(
                f"startup barrier PRE_ARM reconciliation failed: {stage['stage_id']}"
            ) from exc
    if phase == "PRE_ARM" and not record_path.exists():
        return target, None
    try:
        record = validate_startup_barrier_record(
            record_path,
            expected_target_argv=target,
            validate_lock_path=True,
        )
    except Exception as exc:
        raise RecoveryControllerError(
            f"startup barrier record validation failed: {stage['stage_id']}"
        ) from exc
    if (
        record.lock_path != str(lock_path)
        or (phase != "PRE_ARM" and binding.get("record_sha256") != sha256_file(record_path))
        or (
            phase != "PRE_ARM"
            and binding.get("launcher_argv_sha256")
            != stable_json_sha256(record.launcher_argv)
        )
    ):
        raise RecoveryControllerError("startup barrier durable record binding changed")
    return target, record


def _worker_actual_argv_is_bound(
    *,
    root: Path,
    stage: Mapping[str, Any],
    worker: Mapping[str, Any],
    actual_argv: Sequence[str] | None,
) -> bool:
    if actual_argv is None:
        return False
    target_sha = worker.get("argv_sha256")
    try:
        target = _stage_target_argv_for_sha(stage, str(target_sha or ""))
    except RecoveryControllerError:
        return False
    actual_sha = stable_json_sha256(list(actual_argv))
    if actual_sha == target_sha:
        return True
    try:
        bound_target, record = _validate_startup_barrier_binding(
            root=root,
            stage=stage,
            binding=worker.get("startup_barrier"),
            allowed_phases={"BOUND"},
        )
    except RecoveryControllerError:
        return False
    return (
        bound_target == target
        and record is not None
        and actual_sha == stable_json_sha256(record.launcher_argv)
    )


def _validated_bound_worker_pid_for_signal(
    *,
    root: Path,
    stage: Mapping[str, Any],
    worker: Any,
    proc_root: Path,
) -> int | None:
    """Return a PID only when generation, stage, and argv are all still bound."""

    if not isinstance(worker, Mapping) or worker.get("stage_id") != stage.get("stage_id"):
        return None
    try:
        pid = int(worker["pid"])
        start_ticks = int(worker["start_ticks"])
    except (KeyError, TypeError, ValueError):
        return None
    worker_argv_sha = worker.get("argv_sha256")
    if pid <= 0:
        return None
    try:
        if not _pid_alive(pid, start_ticks, proc_root=proc_root):
            return None
        actual = _proc_argv(pid, proc_root=proc_root)
        if not _worker_actual_argv_is_bound(
            root=root, stage=stage, worker=worker, actual_argv=actual
        ):
            return None
        if not _pid_alive(pid, start_ticks, proc_root=proc_root):
            return None
        actual_after = _proc_argv(pid, proc_root=proc_root)
        if not _worker_actual_argv_is_bound(
            root=root, stage=stage, worker=worker, actual_argv=actual_after
        ):
            return None
        if not _pid_alive(pid, start_ticks, proc_root=proc_root):
            return None
    except RecoveryControllerError:
        return None
    return pid


def _terminate_bound_worker_group(
    *,
    root: Path,
    stage: Mapping[str, Any],
    worker: Any,
    proc_root: Path,
) -> bool:
    pid = _validated_bound_worker_pid_for_signal(
        root=root, stage=stage, worker=worker, proc_root=proc_root
    )
    if pid is None:
        return False
    process_group_id = int(worker.get("process_group_id", pid))
    if process_group_id != pid:
        return False
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        return False
    _wait_for_process_group_quiescence(process_group_id, proc_root=proc_root)
    return True


def _proc_rss_bytes(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    try:
        for line in (proc_root / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def _host_sample(*, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    load1 = float(os.getloadavg()[0])
    values = (proc_root / "stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    ticks = [int(value) for value in values]
    return {
        "sampled_at": _utc_now(),
        "load1": load1,
        "cpu_count": cpu_count,
        "load_per_cpu": load1 / cpu_count,
        "cpu_total_ticks": sum(ticks),
        "cpu_iowait_ticks": ticks[4] if len(ticks) > 4 else 0,
    }


def _progress_value(stage: Mapping[str, Any]) -> int | None:
    path_value = stage.get("progress_checkpoint_path")
    field = stage.get("progress_field")
    if not path_value or not field or not Path(path_value).is_file():
        return None
    value: Any = _read_json(path_value, label="exact progress checkpoint")
    if field == EXACT_MONOTONIC_PROGRESS_FIELD:
        ledgers = value.get("progress_ledgers") if isinstance(value, Mapping) else None
        if not isinstance(ledgers, Mapping):
            return None
        # The adopted seed/failure ledgers already each cover N and are not
        # new recovery work. Primary remains complete when expansion starts,
        # so their committed offsets form one monotonic component-work count.
        total = 0
        observed = False
        for phase in (
            "shortcut_anchor_scan",
            "adaptive_component_expansion_scan",
        ):
            ledger = ledgers.get(phase)
            if isinstance(ledger, Mapping):
                try:
                    committed = int(ledger.get("committed_offset", -1))
                except (TypeError, ValueError):
                    return None
                if committed < 0:
                    return None
                total += committed
                observed = True
        return total if observed else None
    for part in str(field).split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validated_exact_checkpoint_snapshot(
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly reopen the hash-chained exact checkpoint for handover use."""

    from src.baselines.comrecgc.external_memory_dbscan import (
        SCHEMA_VERSION as DBSCAN_SCHEMA_VERSION,
        _load_checkpoint,
        _load_progress_ledgers,
        _stable_hash,
    )

    path = _require_absolute(
        stage.get("progress_checkpoint_path"),
        label="exact handover checkpoint",
        existing="file",
    )
    checkpoint_sha_before = sha256_file(path)
    try:
        checkpoint = _load_checkpoint(path)
    except Exception as exc:
        raise RecoveryControllerError(
            "exact handover checkpoint authentication changed"
        ) from exc
    checkpoint_sha_after = sha256_file(path)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RecoveryControllerError(
            "exact handover checkpoint changed during observation"
        )
    identity = checkpoint.get("identity")
    if (
        checkpoint.get("schema_version") != DBSCAN_SCHEMA_VERSION
        or not isinstance(identity, Mapping)
        or checkpoint.get("identity_sha256") != _stable_hash(identity)
    ):
        raise RecoveryControllerError("exact handover checkpoint identity changed")
    try:
        ledgers = _load_progress_ledgers(
            checkpoint,
            identity=identity,
            num_samples=EXPECTED_ROWS,
        )
    except Exception as exc:
        raise RecoveryControllerError(
            "exact handover checkpoint ledger closure changed"
        ) from exc
    progress = 0
    observed = False
    for phase in (
        "shortcut_anchor_scan",
        "adaptive_component_expansion_scan",
    ):
        ledger = ledgers.get(phase)
        if isinstance(ledger, Mapping):
            committed = int(ledger.get("committed_offset", -1))
            if committed < 0:
                raise RecoveryControllerError(
                    "exact handover checkpoint progress changed"
                )
            progress += committed
            observed = True
    if not observed:
        raise RecoveryControllerError("exact handover checkpoint has no recovery ledger")
    vectors_sha = identity.get("vectors_sha256")
    if not _is_sha256(vectors_sha):
        raise RecoveryControllerError("exact handover checkpoint vector SHA is absent")
    return {
        "path": str(path),
        "sha256_at_observation": checkpoint_sha_after,
        "checkpoint_payload_sha256": checkpoint["checkpoint_payload_sha256"],
        "identity_sha256": checkpoint["identity_sha256"],
        "progress_ledgers_sha256": checkpoint["progress_ledgers_sha256"],
        "progress_rows": progress,
        "vectors_sha256": vectors_sha,
    }


def _live_process_observation(
    pid: int, start_ticks: int, *, proc_root: Path
) -> dict[str, Any]:
    """Triple-reopen one live PID generation and its raw command line."""

    if pid <= 0 or start_ticks <= 0 or not _pid_alive(
        pid, start_ticks, proc_root=proc_root
    ):
        raise RecoveryControllerError("handover process generation is not live")
    argv = _proc_argv(pid, proc_root=proc_root)
    cmdline_sha = _proc_cmdline_sha256(pid, start_ticks, proc_root=proc_root)
    if (
        not isinstance(argv, list)
        or not argv
        or not _is_sha256(cmdline_sha)
        or not _pid_alive(pid, start_ticks, proc_root=proc_root)
    ):
        raise RecoveryControllerError("handover process procfs closure changed")
    argv_after = _proc_argv(pid, proc_root=proc_root)
    cmdline_sha_after = _proc_cmdline_sha256(
        pid, start_ticks, proc_root=proc_root
    )
    if (
        argv_after != argv
        or cmdline_sha_after != cmdline_sha
        or not _pid_alive(pid, start_ticks, proc_root=proc_root)
    ):
        raise RecoveryControllerError("handover process changed during reopen")
    return {
        "pid": int(pid),
        "start_ticks": int(start_ticks),
        "cmdline_sha256": str(cmdline_sha),
        "argv_sha256": stable_json_sha256(argv),
    }


def _validate_controller_generation_payload(
    *,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    binding: Mapping[str, Any],
    generation: int,
    previous_artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "controller_manifest_sha256",
        "owner_binding",
        "generation",
        "previous_generation_artifact",
        "controller_process",
        "generation_attempt_id",
        "generation_nonce",
        "signals_sent",
        "observed_epoch",
        "observed_at",
        "receipt_sha256",
    }
    projected = dict(payload)
    receipt_sha = projected.pop("receipt_sha256", None)
    process = payload.get("controller_process")
    process_keys = {"pid", "start_ticks", "cmdline_sha256", "argv_sha256"}
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != HANDOVER_CONTROLLER_GENERATION_SCHEMA
        or payload.get("controller_manifest_sha256")
        != manifest["manifest_sha256"]
        or payload.get("owner_binding") != _handover_owner_binding(manifest)
        or payload.get("generation") != generation
        or payload.get("previous_generation_artifact") != previous_artifact
        or not isinstance(process, Mapping)
        or set(process) != process_keys
        or isinstance(process.get("pid"), bool)
        or int(process.get("pid", 0)) <= 0
        or isinstance(process.get("start_ticks"), bool)
        or int(process.get("start_ticks", 0)) <= 0
        or not _is_sha256(process.get("cmdline_sha256"))
        or not _is_sha256(process.get("argv_sha256"))
        or re.fullmatch(
            r"[0-9a-f]{32}", str(payload.get("generation_attempt_id", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(payload.get("generation_nonce", ""))
        )
        is None
        or payload.get("signals_sent") != []
        or not isinstance(payload.get("observed_at"), str)
        or not payload.get("observed_at")
        or receipt_sha != stable_json_sha256(projected)
    ):
        raise RecoveryControllerError(
            "handover controller generation closure changed"
        )
    _validate_artifact_clock(payload, binding)
    return dict(payload)


def _load_controller_generation_chain(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = Path(manifest["controller_root"])
    gates = root / "gates"
    prefix = "90_handover_controller_"
    pattern = re.compile(r"^90_handover_controller_([0-9]{2})\.json$")
    indexed: dict[int, Path] = {}
    for path in gates.iterdir():
        if not path.name.startswith(prefix):
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            raise RecoveryControllerError(
                "malformed handover controller generation artifact"
            )
        generation = int(match.group(1))
        if generation in indexed:
            raise RecoveryControllerError("duplicate handover controller generation")
        indexed[generation] = path
    if indexed and sorted(indexed) != list(range(max(indexed) + 1)):
        raise RecoveryControllerError("handover controller generation chain has gaps")
    if len(indexed) > HANDOVER_EVIDENCE_MAX_GENERATIONS:
        raise RecoveryControllerError("handover controller generation budget exceeded")
    rows: list[dict[str, Any]] = []
    previous_artifact: Mapping[str, Any] | None = None
    for generation in sorted(indexed):
        expected_path = _handover_generation_path(root, generation)
        if indexed[generation] != expected_path:
            raise RecoveryControllerError("handover generation path changed")
        payload, artifact = _read_handover_artifact(
            expected_path, label="handover controller generation"
        )
        payload = _validate_controller_generation_payload(
            manifest=manifest,
            payload=payload,
            binding=artifact,
            generation=generation,
            previous_artifact=previous_artifact,
        )
        rows.append({"payload": payload, "artifact": artifact})
        previous_artifact = artifact
    return rows


def _ensure_current_controller_generation(
    *,
    manifest: Mapping[str, Any],
    current_controller: Mapping[str, Any],
) -> dict[str, Any]:
    proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
    try:
        pid = int(current_controller["pid"])
        start_ticks = int(current_controller["start_ticks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryControllerError("current controller identity is invalid") from exc
    if (pid, start_ticks) != (
        os.getpid(),
        _read_proc_start_ticks(os.getpid(), proc_root=proc_root),
    ):
        raise RecoveryControllerError("current controller procfs identity changed")
    process = _live_process_observation(pid, start_ticks, proc_root=proc_root)
    rows = _load_controller_generation_chain(manifest)
    if rows:
        latest = rows[-1]
        if latest["payload"]["controller_process"] == process:
            return latest
        old = latest["payload"]["controller_process"]
        if _pid_alive(
            int(old["pid"]), int(old["start_ticks"]), proc_root=proc_root
        ):
            raise RecoveryControllerError(
                "prior handover controller generation is still live"
            )
    generation = len(rows)
    path = _handover_generation_path(
        Path(manifest["controller_root"]), generation
    )
    payload: dict[str, Any] = {
        "schema_version": HANDOVER_CONTROLLER_GENERATION_SCHEMA,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "owner_binding": _handover_owner_binding(manifest),
        "generation": generation,
        "previous_generation_artifact": (
            rows[-1]["artifact"] if rows else None
        ),
        "controller_process": process,
        "generation_attempt_id": secrets.token_hex(16),
        "generation_nonce": secrets.token_hex(32),
        "signals_sent": [],
        "observed_epoch": time.time(),
        "observed_at": _utc_now(),
    }
    payload["receipt_sha256"] = stable_json_sha256(payload)
    _publish_handover_artifact(manifest, path, payload)
    return _load_controller_generation_chain(manifest)[-1]


def _exact_checkpoint_observation(stage: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(stage["progress_checkpoint_path"]))
    before = _stat_identity(path)
    snapshot = _validated_exact_checkpoint_snapshot(stage)
    after = _stat_identity(path)
    if before != after:
        raise RecoveryControllerError(
            "exact handover checkpoint stat changed during observation"
        )
    return {**snapshot, "stat_identity_at_observation": after}


def _validate_checkpoint_observation_shape(
    *, manifest: Mapping[str, Any], checkpoint: Any, require_positive: bool
) -> dict[str, Any]:
    expected_keys = {
        "path",
        "sha256_at_observation",
        "checkpoint_payload_sha256",
        "identity_sha256",
        "progress_ledgers_sha256",
        "progress_rows",
        "vectors_sha256",
        "stat_identity_at_observation",
    }
    expected_stat_keys = {
        "device",
        "inode",
        "mode",
        "size",
        "mtime_ns",
        "ctime_ns",
        "nlink",
    }
    stat_identity = (
        checkpoint.get("stat_identity_at_observation")
        if isinstance(checkpoint, Mapping)
        else None
    )
    progress = checkpoint.get("progress_rows") if isinstance(checkpoint, Mapping) else None
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != expected_keys
        or checkpoint.get("path")
        != _stage(manifest, EXACT_STAGE)["progress_checkpoint_path"]
        or isinstance(progress, bool)
        or not isinstance(progress, int)
        or int(progress) < (1 if require_positive else 0)
        or int(progress) > HANDOVER_CONSERVATIVE_WORK_ROWS
        or checkpoint.get("vectors_sha256")
        != manifest["source_authority"]["source_vectors_sha256"]
        or not all(
            _is_sha256(checkpoint.get(name))
            for name in (
                "sha256_at_observation",
                "checkpoint_payload_sha256",
                "identity_sha256",
                "progress_ledgers_sha256",
            )
        )
        or not isinstance(stat_identity, Mapping)
        or set(stat_identity) != expected_stat_keys
        or any(
            isinstance(stat_identity.get(name), bool)
            or not isinstance(stat_identity.get(name), int)
            for name in expected_stat_keys
        )
        or int(stat_identity.get("device", -1)) < 0
        or int(stat_identity.get("inode", 0)) <= 0
        or not stat.S_ISREG(int(stat_identity.get("mode", 0)))
        or int(stat_identity.get("size", 0)) <= 0
        or int(stat_identity.get("mtime_ns", -1)) < 0
        or int(stat_identity.get("ctime_ns", -1)) < 0
        or int(stat_identity.get("nlink", 0)) <= 0
    ):
        raise RecoveryControllerError("handover checkpoint observation shape changed")
    return dict(checkpoint)


def _live_exact_worker_observation(
    *, manifest: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(manifest["controller_root"])
    stage = _stage(manifest, EXACT_STAGE)
    proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
    worker = state.get("worker")
    pid = _validated_bound_worker_pid_for_signal(
        root=root, stage=stage, worker=worker, proc_root=proc_root
    )
    if pid is None or not isinstance(worker, Mapping):
        raise RecoveryControllerError("exact handover worker is not live and bound")
    process = _live_process_observation(
        pid, int(worker["start_ticks"]), proc_root=proc_root
    )
    return {
        "stage_id": EXACT_STAGE,
        **process,
        "worker_argv_sha256": str(worker["argv_sha256"]),
    }


def _reopen_receipted_exact_worker(
    *, manifest: Mapping[str, Any], worker: Mapping[str, Any]
) -> dict[str, Any]:
    """Reopen a receipted exact worker without trusting mutable state.json."""

    stage = _stage(manifest, EXACT_STAGE)
    allowed_worker_argv = {
        value
        for value in (
            stage["commands"].get("fresh_sha256"),
            stage["commands"].get("resume_sha256"),
        )
        if value is not None
    }
    if (
        worker.get("stage_id") != EXACT_STAGE
        or worker.get("worker_argv_sha256") not in allowed_worker_argv
    ):
        raise RecoveryControllerError("receipted exact worker command is not frozen")
    proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
    process = _live_process_observation(
        int(worker["pid"]), int(worker["start_ticks"]), proc_root=proc_root
    )
    reopened = {
        "stage_id": EXACT_STAGE,
        **process,
        "worker_argv_sha256": worker["worker_argv_sha256"],
    }
    if reopened != dict(worker):
        raise RecoveryControllerError("receipted exact worker procfs identity changed")
    return reopened


def _validate_resume_receipt(
    *,
    manifest: Mapping[str, Any],
    generations: Sequence[Mapping[str, Any]],
    require_live: bool,
) -> dict[str, Any]:
    if len(generations) < 2:
        raise RecoveryControllerError("durable resume requires two generations")
    current = generations[-1]
    previous = generations[-2]
    generation = int(current["payload"]["generation"])
    path = _handover_resume_path(Path(manifest["controller_root"]), generation)
    payload, artifact = _read_handover_artifact(
        path, label="handover resume receipt"
    )
    expected_keys = {
        "schema_version",
        "controller_manifest_sha256",
        "owner_binding",
        "previous_controller_artifact",
        "resumed_controller_artifact",
        "reattached_exact_worker",
        "checkpoint_snapshot",
        "resume_attempt_id",
        "resume_nonce",
        "science_worker_reattached",
        "signals_sent",
        "observed_epoch",
        "observed_at",
        "receipt_sha256",
    }
    projected = dict(payload)
    receipt_sha = projected.pop("receipt_sha256", None)
    worker = payload.get("reattached_exact_worker")
    checkpoint = payload.get("checkpoint_snapshot")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != HANDOVER_RESUME_RECEIPT_SCHEMA
        or payload.get("controller_manifest_sha256")
        != manifest["manifest_sha256"]
        or payload.get("owner_binding") != _handover_owner_binding(manifest)
        or payload.get("previous_controller_artifact") != previous["artifact"]
        or payload.get("resumed_controller_artifact") != current["artifact"]
        or not isinstance(worker, Mapping)
        or set(worker)
        != {
            "stage_id",
            "pid",
            "start_ticks",
            "cmdline_sha256",
            "argv_sha256",
            "worker_argv_sha256",
        }
        or worker.get("stage_id") != EXACT_STAGE
        or int(worker.get("pid", 0)) <= 0
        or int(worker.get("start_ticks", 0)) <= 0
        or not _is_sha256(worker.get("cmdline_sha256"))
        or not _is_sha256(worker.get("argv_sha256"))
        or not _is_sha256(worker.get("worker_argv_sha256"))
        or not isinstance(checkpoint, Mapping)
        or re.fullmatch(
            r"[0-9a-f]{32}", str(payload.get("resume_attempt_id", ""))
        )
        is None
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("resume_nonce", "")))
        is None
        or payload.get("science_worker_reattached") is not True
        or payload.get("signals_sent") != []
        or not isinstance(payload.get("observed_at"), str)
        or not payload.get("observed_at")
        or receipt_sha != stable_json_sha256(projected)
    ):
        raise RecoveryControllerError("durable handover resume closure changed")
    _validate_checkpoint_observation_shape(
        manifest=manifest, checkpoint=checkpoint, require_positive=True
    )
    _validate_artifact_clock(payload, artifact)
    previous_process = previous["payload"]["controller_process"]
    resumed_process = current["payload"]["controller_process"]
    if previous_process == resumed_process:
        raise RecoveryControllerError("durable resume did not change controller generation")
    if require_live:
        proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
        if _pid_alive(
            int(previous_process["pid"]),
            int(previous_process["start_ticks"]),
            proc_root=proc_root,
        ):
            raise RecoveryControllerError("previous controller is still live")
        if (
            _live_process_observation(
                int(resumed_process["pid"]),
                int(resumed_process["start_ticks"]),
                proc_root=proc_root,
            )
            != resumed_process
        ):
            raise RecoveryControllerError("resumed controller procfs identity changed")
        _reopen_receipted_exact_worker(manifest=manifest, worker=worker)
    return {"payload": dict(payload), "artifact": artifact}


def _publish_resume_receipt_if_reattached(
    *,
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any] | None:
    generations = _load_controller_generation_chain(manifest)
    if len(generations) < 2 or generations[-1]["artifact"] != generation["artifact"]:
        return None
    path = _handover_resume_path(
        Path(manifest["controller_root"]),
        int(generation["payload"]["generation"]),
    )
    if path.exists() or path.is_symlink():
        return _validate_resume_receipt(
            manifest=manifest,
            generations=generations,
            require_live=True,
        )
    previous_process = generations[-2]["payload"]["controller_process"]
    proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
    if _pid_alive(
        int(previous_process["pid"]),
        int(previous_process["start_ticks"]),
        proc_root=proc_root,
    ):
        raise RecoveryControllerError("previous controller is live during resume")
    try:
        worker = _live_exact_worker_observation(manifest=manifest, state=state)
        checkpoint = _exact_checkpoint_observation(_stage(manifest, EXACT_STAGE))
    except RecoveryControllerError:
        return None
    if int(checkpoint["progress_rows"]) <= 0:
        return None
    payload: dict[str, Any] = {
        "schema_version": HANDOVER_RESUME_RECEIPT_SCHEMA,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "owner_binding": _handover_owner_binding(manifest),
        "previous_controller_artifact": generations[-2]["artifact"],
        "resumed_controller_artifact": generations[-1]["artifact"],
        "reattached_exact_worker": worker,
        "checkpoint_snapshot": checkpoint,
        "resume_attempt_id": secrets.token_hex(16),
        "resume_nonce": secrets.token_hex(32),
        "science_worker_reattached": True,
        "signals_sent": [],
        "observed_epoch": time.time(),
        "observed_at": _utc_now(),
    }
    payload["receipt_sha256"] = stable_json_sha256(payload)
    _publish_handover_artifact(manifest, path, payload)
    return _validate_resume_receipt(
        manifest=manifest,
        generations=generations,
        require_live=True,
    )


def _validate_progress_observation_payload(
    *,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    binding: Mapping[str, Any],
    generation: Mapping[str, Any],
    resume: Mapping[str, Any],
    sequence: int,
    previous_artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "controller_manifest_sha256",
        "owner_binding",
        "generation_artifact",
        "resume_receipt_artifact",
        "sequence",
        "previous_observation_artifact",
        "controller_process",
        "exact_worker_process",
        "checkpoint_snapshot",
        "observation_attempt_id",
        "observation_nonce",
        "observed_epoch",
        "observed_at",
        "receipt_sha256",
    }
    projected = dict(payload)
    receipt_sha = projected.pop("receipt_sha256", None)
    checkpoint = payload.get("checkpoint_snapshot")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != HANDOVER_PROGRESS_OBSERVATION_SCHEMA
        or payload.get("controller_manifest_sha256")
        != manifest["manifest_sha256"]
        or payload.get("owner_binding") != _handover_owner_binding(manifest)
        or payload.get("generation_artifact") != generation["artifact"]
        or payload.get("resume_receipt_artifact") != resume["artifact"]
        or payload.get("sequence") != sequence
        or payload.get("previous_observation_artifact") != previous_artifact
        or payload.get("controller_process")
        != generation["payload"]["controller_process"]
        or payload.get("exact_worker_process")
        != resume["payload"]["reattached_exact_worker"]
        or not isinstance(checkpoint, Mapping)
        or re.fullmatch(
            r"[0-9a-f]{32}", str(payload.get("observation_attempt_id", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(payload.get("observation_nonce", ""))
        )
        is None
        or not isinstance(payload.get("observed_at"), str)
        or not payload.get("observed_at")
        or receipt_sha != stable_json_sha256(projected)
    ):
        raise RecoveryControllerError("handover progress observation changed")
    _validate_checkpoint_observation_shape(
        manifest=manifest, checkpoint=checkpoint, require_positive=False
    )
    _validate_artifact_clock(payload, binding)
    return dict(payload)


def _load_progress_observation_chain(
    *,
    manifest: Mapping[str, Any],
    generation: Mapping[str, Any],
    resume: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = Path(manifest["controller_root"])
    gates = root / "gates"
    generation_index = int(generation["payload"]["generation"])
    prefix = f"92_handover_progress_{generation_index:02d}_"
    pattern = re.compile(
        rf"^92_handover_progress_{generation_index:02d}_([0-9]{{4}})\.json$"
    )
    indexed: dict[int, Path] = {}
    for path in gates.iterdir():
        if not path.name.startswith("92_handover_progress_"):
            continue
        if not path.name.startswith(prefix):
            # Historical generations may retain their own sealed chains.
            if re.fullmatch(
                r"92_handover_progress_[0-9]{2}_[0-9]{4}\.json",
                path.name,
            ) is None:
                raise RecoveryControllerError(
                    "malformed handover progress observation artifact"
                )
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            raise RecoveryControllerError(
                "malformed current handover progress observation"
            )
        sequence = int(match.group(1))
        if sequence in indexed:
            raise RecoveryControllerError("duplicate handover progress observation")
        indexed[sequence] = path
    if indexed and sorted(indexed) != list(range(max(indexed) + 1)):
        raise RecoveryControllerError("handover progress observation chain has gaps")
    if len(indexed) > HANDOVER_EVIDENCE_MAX_OBSERVATIONS:
        raise RecoveryControllerError("handover progress observation budget exceeded")
    rows: list[dict[str, Any]] = []
    previous_artifact: Mapping[str, Any] | None = None
    previous_progress: int | None = None
    checkpoint_identity: str | None = None
    for sequence in sorted(indexed):
        expected_path = _handover_observation_path(
            root, generation_index, sequence
        )
        if indexed[sequence] != expected_path:
            raise RecoveryControllerError("handover progress path changed")
        payload, artifact = _read_handover_artifact(
            expected_path, label="handover progress observation"
        )
        payload = _validate_progress_observation_payload(
            manifest=manifest,
            payload=payload,
            binding=artifact,
            generation=generation,
            resume=resume,
            sequence=sequence,
            previous_artifact=previous_artifact,
        )
        progress = int(payload["checkpoint_snapshot"]["progress_rows"])
        identity = str(payload["checkpoint_snapshot"]["identity_sha256"])
        if previous_progress is not None and progress <= previous_progress:
            raise RecoveryControllerError(
                "handover progress observations are not strictly increasing"
            )
        if checkpoint_identity is not None and identity != checkpoint_identity:
            raise RecoveryControllerError(
                "handover checkpoint identity changed across observations"
            )
        rows.append({"payload": payload, "artifact": artifact})
        previous_artifact = artifact
        previous_progress = progress
        checkpoint_identity = identity
    return rows


def _record_durable_progress_observation(
    *, manifest: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Append one real procfs/checkpoint observation after durable resume."""

    generations = _load_controller_generation_chain(manifest)
    if len(generations) < 2:
        return None
    generation = generations[-1]
    try:
        resume = _validate_resume_receipt(
            manifest=manifest,
            generations=generations,
            require_live=True,
        )
    except RecoveryControllerError as exc:
        if "absent" in str(exc):
            return None
        raise
    worker = _live_exact_worker_observation(manifest=manifest, state=state)
    if worker != resume["payload"]["reattached_exact_worker"]:
        raise RecoveryControllerError("handover exact worker generation changed")
    checkpoint = _exact_checkpoint_observation(_stage(manifest, EXACT_STAGE))
    rows = _load_progress_observation_chain(
        manifest=manifest, generation=generation, resume=resume
    )
    if rows:
        previous_checkpoint = rows[-1]["payload"]["checkpoint_snapshot"]
        if int(checkpoint["progress_rows"]) < int(previous_checkpoint["progress_rows"]):
            raise RecoveryControllerError("handover checkpoint progress regressed")
        if int(checkpoint["progress_rows"]) == int(previous_checkpoint["progress_rows"]):
            return rows[-1]
        if checkpoint["identity_sha256"] != previous_checkpoint["identity_sha256"]:
            raise RecoveryControllerError("handover checkpoint identity changed")
    sequence = len(rows)
    if sequence >= HANDOVER_EVIDENCE_MAX_OBSERVATIONS:
        raise RecoveryControllerError("handover progress observation budget exhausted")
    process = generation["payload"]["controller_process"]
    proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
    if (
        _live_process_observation(
            int(process["pid"]), int(process["start_ticks"]), proc_root=proc_root
        )
        != process
    ):
        raise RecoveryControllerError("handover controller generation changed")
    payload: dict[str, Any] = {
        "schema_version": HANDOVER_PROGRESS_OBSERVATION_SCHEMA,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "owner_binding": _handover_owner_binding(manifest),
        "generation_artifact": generation["artifact"],
        "resume_receipt_artifact": resume["artifact"],
        "sequence": sequence,
        "previous_observation_artifact": rows[-1]["artifact"] if rows else None,
        "controller_process": process,
        "exact_worker_process": worker,
        "checkpoint_snapshot": checkpoint,
        "observation_attempt_id": secrets.token_hex(16),
        "observation_nonce": secrets.token_hex(32),
        "observed_epoch": time.time(),
        "observed_at": _utc_now(),
    }
    payload["receipt_sha256"] = stable_json_sha256(payload)
    path = _handover_observation_path(
        Path(manifest["controller_root"]),
        int(generation["payload"]["generation"]),
        sequence,
    )
    _publish_handover_artifact(manifest, path, payload)
    return _load_progress_observation_chain(
        manifest=manifest, generation=generation, resume=resume
    )[-1]


def _verify_durable_handover_evidence(
    *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Independent read-only verifier; mutable monitor fields are not inputs."""

    generations = _load_controller_generation_chain(manifest)
    if len(generations) < 2:
        raise RecoveryControllerError("handover has fewer than two generations")
    generation = generations[-1]
    resume = _validate_resume_receipt(
        manifest=manifest,
        generations=generations,
        require_live=True,
    )
    rows = _load_progress_observation_chain(
        manifest=manifest, generation=generation, resume=resume
    )
    if len(rows) < 3:
        raise RecoveryControllerError(
            "handover progress ledger lacks a successor seal"
        )
    live_worker = _reopen_receipted_exact_worker(
        manifest=manifest,
        worker=resume["payload"]["reattached_exact_worker"],
    )
    if live_worker != resume["payload"]["reattached_exact_worker"]:
        raise RecoveryControllerError("handover live worker no longer matches resume")
    for row in rows:
        if row["payload"]["exact_worker_process"] != live_worker:
            raise RecoveryControllerError(
                "handover observation worker generation changed"
            )
    current_checkpoint = _exact_checkpoint_observation(
        _stage(manifest, EXACT_STAGE)
    )
    resume_checkpoint = resume["payload"]["checkpoint_snapshot"]
    seal_checkpoint = rows[-1]["payload"]["checkpoint_snapshot"]
    if (
        current_checkpoint["path"] != resume_checkpoint["path"]
        or current_checkpoint["identity_sha256"]
        != resume_checkpoint["identity_sha256"]
        or current_checkpoint["identity_sha256"]
        != seal_checkpoint["identity_sha256"]
        or current_checkpoint["vectors_sha256"]
        != resume_checkpoint["vectors_sha256"]
        or int(current_checkpoint["progress_rows"])
        < int(seal_checkpoint["progress_rows"])
        or int(resume_checkpoint["progress_rows"]) <= 0
    ):
        raise RecoveryControllerError(
            "current checkpoint no longer extends durable handover evidence"
        )

    # The chain head is a seal for the last decision-bearing observation. It
    # is verified above but never contributes progress or elapsed time, so
    # every record used by the decision has a successor binding its complete
    # content/stat identity. Filesystem ctime, not a payload timestamp,
    # supplies cadence, elapsed-time, and freshness authority.
    sealed_rows = rows[:-1]
    seal_epoch = _artifact_observed_epoch(rows[-1]["artifact"])
    last_sealed_epoch = _artifact_observed_epoch(sealed_rows[-1]["artifact"])
    if (
        seal_epoch < last_sealed_epoch
        or seal_epoch - last_sealed_epoch > HANDOVER_MAX_OBSERVATION_AGE_SECONDS
    ):
        raise RecoveryControllerError("handover successor seal cadence changed")
    suffix_start = len(sealed_rows) - 1
    while suffix_start > 0:
        newer = _artifact_observed_epoch(sealed_rows[suffix_start]["artifact"])
        older = _artifact_observed_epoch(
            sealed_rows[suffix_start - 1]["artifact"]
        )
        gap = newer - older
        if gap < 0:
            raise RecoveryControllerError("handover ledger ctime regressed")
        if gap > HANDOVER_MAX_OBSERVATION_AGE_SECONDS:
            break
        suffix_start -= 1
    first = sealed_rows[suffix_start]
    last = sealed_rows[-1]
    first_epoch = _artifact_observed_epoch(first["artifact"])
    last_epoch = _artifact_observed_epoch(last["artifact"])
    now_epoch = time.time()
    if seal_epoch > now_epoch + HANDOVER_EVIDENCE_CLOCK_SKEW_SECONDS:
        raise RecoveryControllerError("handover ledger ctime is in the future")
    duration = max(0.0, last_epoch - first_epoch)
    observation_age = max(0.0, now_epoch - last_epoch)
    first_progress = int(first["payload"]["checkpoint_snapshot"]["progress_rows"])
    last_progress = int(last["payload"]["checkpoint_snapshot"]["progress_rows"])
    progress_delta = last_progress - first_progress
    throughput = progress_delta / duration if duration > 0 and progress_delta > 0 else None
    remaining = max(0, HANDOVER_CONSERVATIVE_WORK_ROWS - last_progress)
    eta = remaining / throughput if throughput is not None and throughput > 0 else None
    continuous_pass = bool(
        duration >= HANDOVER_MIN_PROGRESS_SECONDS
        and observation_age <= HANDOVER_MAX_OBSERVATION_AGE_SECONDS
        and progress_delta > 0
    )
    positive_throughput_pass = bool(throughput is not None and throughput > 0)
    return {
        "resume_smoke_pass": True,
        "first_durable_checkpoint_pass": True,
        "continuous_progress_pass": continuous_pass,
        "positive_throughput_pass": positive_throughput_pass,
        "new_exact_worker_generation_bound_pass": True,
        "progress_rows": last_progress,
        "progress_delta_rows": progress_delta,
        "continuous_progress_seconds": duration,
        "observation_age_seconds": observation_age,
        "last_change_age_seconds": observation_age,
        "throughput_rows_per_second": throughput,
        "conservative_remaining_rows": remaining,
        "conservative_eta_seconds": eta,
        "checkpoint_snapshot": current_checkpoint,
        "resume_smoke_receipt": resume["payload"],
        "controller_generation_count": len(generations),
        "progress_observation_count": len(rows),
        "continuous_suffix_start_sequence": int(first["payload"]["sequence"]),
        "successor_seal_sequence": int(rows[-1]["payload"]["sequence"]),
    }


def _validate_exact_progress_monitor(
    *, controller_manifest_sha256: str, value: Any
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryControllerError("exact progress monitor is absent")
    expected_keys = {
        "schema_version",
        "controller_manifest_sha256",
        "stage_id",
        "progress",
        "last_change_epoch",
        "continuous_progress_since_epoch",
        "continuous_start_progress",
        "baseline_progress",
        "observed_epoch",
        "observed_at",
        "monitor_sha256",
    }
    projected = dict(value)
    monitor_sha = projected.pop("monitor_sha256", None)
    continuous_since = value.get("continuous_progress_since_epoch")
    continuous_start = value.get("continuous_start_progress")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != EXACT_PROGRESS_MONITOR_SCHEMA
        or value.get("controller_manifest_sha256")
        != controller_manifest_sha256
        or value.get("stage_id") != EXACT_STAGE
        or not isinstance(value.get("progress"), int)
        or int(value.get("progress", -1)) < 0
        or not isinstance(value.get("baseline_progress"), int)
        or int(value.get("baseline_progress", -1)) < 0
        or not isinstance(value.get("last_change_epoch"), (int, float))
        or not isinstance(value.get("observed_epoch"), (int, float))
        or (
            continuous_since is not None
            and not isinstance(continuous_since, (int, float))
        )
        or (continuous_start is not None and not isinstance(continuous_start, int))
        or (continuous_since is None) != (continuous_start is None)
        or not isinstance(value.get("observed_at"), str)
        or not value.get("observed_at")
        or monitor_sha != stable_json_sha256(projected)
    ):
        raise RecoveryControllerError("exact progress monitor closure changed")
    return dict(value)


def _old_brute_generation_alive(
    manifest: Mapping[str, Any],
) -> bool:
    """Reopen the one frozen old PID generation without signal authority."""

    contract = manifest["resources"]["old_brute_handover"]
    identity = contract["old_brute_process"]
    pid = int(identity["pid"])
    start_ticks = int(identity["start_ticks"])
    expected_cmdline_sha256 = str(identity["cmdline_sha256"])
    proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
    try:
        return (
            _is_sha256(expected_cmdline_sha256)
            and _proc_cmdline_sha256(
                pid,
                start_ticks,
                proc_root=proc_root,
            )
            == expected_cmdline_sha256
        )
    except (KeyError, TypeError, ValueError, RecoveryControllerError):
        return False


def _stage_environment(manifest: Mapping[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(_frozen_stage_environment(manifest))
    return env


def _initial_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "status": "RUNNING",
        "current_stage": None,
        "stages": {stage: "PENDING" for stage in STAGE_ORDER},
        "controller_process": None,
        "worker": None,
        "startup_barrier": None,
        "durable_handover_generation": None,
        "durable_resume_receipt_path": None,
        "exact_coexistence_baseline": None,
        "exact_progress_monitor": None,
        "updated_at": _utc_now(),
    }


def _load_state(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "state.json"
    if not path.exists():
        return _initial_state(manifest)
    value = _read_json(path, label="controller mutable state")
    if (
        value.get("schema_version") != STATE_SCHEMA
        or value.get("controller_id") != CONTROLLER_ID
        or value.get("controller_manifest_sha256") != manifest["manifest_sha256"]
        or set(value.get("stages", {})) != set(STAGE_ORDER)
    ):
        raise RecoveryControllerError("controller mutable state identity mismatch")
    # c082 roots may carry the superseded self-signed projection. It is not
    # validated, copied forward, or consulted by the durable handover gate.
    value.pop("resume_smoke", None)
    if value.get("exact_progress_monitor") is not None:
        _validate_exact_progress_monitor(
            controller_manifest_sha256=manifest["manifest_sha256"],
            value=value["exact_progress_monitor"],
        )
    return value


def _save_state(
    manifest: Mapping[str, Any],
    root: Path,
    state: dict[str, Any],
    guard: Callable[[], None],
    *,
    refresh_timestamp: bool = True,
) -> None:
    guard()
    if refresh_timestamp:
        state["updated_at"] = _utc_now()
    encoded = _json_payload_bytes(state)
    _reserve_output_growth(
        manifest, [(root / "state.json", len(encoded), True)]
    )
    _atomic_state(root / "state.json", state)
    guard()
    _disk_preflight(manifest)


def _reconcile_previous_startup_barrier(
    *,
    root: Path,
    stage: Mapping[str, Any],
    state: dict[str, Any],
) -> int:
    binding = state.get("startup_barrier")
    if binding is None:
        return 0
    target, record = _validate_startup_barrier_binding(
        root=root,
        stage=stage,
        binding=binding,
        allowed_phases={"PRE_ARM", "ARMED", "QUIESCENT"},
    )
    if record is not None:
        try:
            validate_reopenable_unreleased_barrier(
                record.record_path,
                expected_target_argv=target,
                timeout_seconds=30.0,
            )
        except Exception as exc:
            raise RecoveryControllerError(
                f"previous startup barrier is not quiescent: {stage['stage_id']}"
            ) from exc
    generation = int(binding["generation"]) + 1
    if generation >= STARTUP_BARRIER_MAX_GENERATIONS:
        raise RecoveryControllerError("startup barrier generation budget exhausted")
    return generation


def _prepare_exec_startup_barrier(
    *,
    manifest: Mapping[str, Any],
    root: Path,
    stage: Mapping[str, Any],
    state: dict[str, Any],
    target_argv: Sequence[str],
    guard: Callable[[], None],
) -> tuple[ArmedExecStartupBarrier, dict[str, Any]]:
    generation = _reconcile_previous_startup_barrier(
        root=root, stage=stage, state=state
    )
    lock_path, record_path = _startup_barrier_paths(
        root, stage_id=str(stage["stage_id"]), generation=generation
    )
    target_sha = stable_json_sha256(list(target_argv))
    pre_arm = {
        "schema_version": STARTUP_BARRIER_BINDING_SCHEMA,
        "stage_id": stage["stage_id"],
        "generation": generation,
        "phase": "PRE_ARM",
        "record_path": str(record_path),
        "lock_path": str(lock_path),
        "target_argv_sha256": target_sha,
    }
    state["startup_barrier"] = pre_arm
    _save_state(manifest, root, state, guard)
    barrier = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=list(target_argv),
        python_executable=sys.executable,
        record_policy="fresh",
    )
    armed = {
        **pre_arm,
        "phase": "ARMED",
        "record_sha256": sha256_file(record_path),
        "launcher_argv_sha256": stable_json_sha256(barrier.launcher_argv),
    }
    state["startup_barrier"] = armed
    try:
        _save_state(manifest, root, state, guard)
    except BaseException:
        barrier.abort()
        raise
    return barrier, armed


def _ensure_exact_coexistence_baseline(
    *,
    manifest: Mapping[str, Any],
    stage: Mapping[str, Any],
    state: dict[str, Any],
    worker_argv_sha256: str,
    root: Path,
    guard: Callable[[], None],
    proc_root: Path,
) -> Mapping[str, Any]:
    """Persist the first exact-work resource/progress observation.

    The baseline belongs to the controller-level exact stage, not to one OS
    process generation.  A resumed worker can therefore finish within one
    block of the terminal without losing the already-observed recovery work.
    It is deliberately written before ``Popen`` so a worker that creates its
    terminal before PID binding still has an authenticated start sample.
    """

    if stage.get("stage_id") != EXACT_STAGE:
        raise RecoveryControllerError("coexistence baseline requested outside exact stage")
    allowed_argv = {
        stage["commands"]["fresh_sha256"],
        stage["commands"]["resume_sha256"],
    } - {None}
    if worker_argv_sha256 not in allowed_argv:
        raise RecoveryControllerError("coexistence baseline command is not release-bound")
    existing = state.get("exact_coexistence_baseline")
    if existing is not None:
        if (
            not isinstance(existing, Mapping)
            or existing.get("stage_id") != EXACT_STAGE
            or existing.get("controller_manifest_sha256")
            != manifest["manifest_sha256"]
            or existing.get("worker_argv_sha256") not in allowed_argv
            or not isinstance(existing.get("start_host"), Mapping)
            or int(existing.get("start_progress", -1)) < 0
        ):
            raise RecoveryControllerError("persisted exact coexistence baseline is invalid")
        return existing
    start_progress = _progress_value(stage)
    payload = {
        "stage_id": EXACT_STAGE,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "worker_argv_sha256": worker_argv_sha256,
        "start_progress": 0 if start_progress is None else int(start_progress),
        "start_host": _host_sample(proc_root=proc_root),
        "recorded_before_worker_spawn": True,
        "recorded_at": _utc_now(),
    }
    state["exact_coexistence_baseline"] = payload
    _save_state(manifest, root, state, guard)
    return payload


def _update_exact_progress_monitor(
    *, stage: Mapping[str, Any], state: dict[str, Any]
) -> Mapping[str, Any] | None:
    if stage.get("stage_id") != EXACT_STAGE:
        return None
    progress = _progress_value(stage)
    if progress is None:
        return None
    now = time.time()
    existing = state.get("exact_progress_monitor")
    if existing is not None:
        existing = _validate_exact_progress_monitor(
            controller_manifest_sha256=state["controller_manifest_sha256"],
            value=existing,
        )
    baseline = state.get("exact_coexistence_baseline")
    baseline_progress = (
        int(baseline.get("start_progress", 0))
        if isinstance(baseline, Mapping)
        else 0
    )
    if isinstance(existing, Mapping):
        previous = int(existing.get("progress", -1))
        if progress < previous:
            raise RecoveryControllerError("exact monitored progress regressed")
        changed_at = (
            now if progress > previous else float(existing.get("last_change_epoch", now))
        )
        previous_observed = float(existing.get("observed_epoch", now))
        observation_gap = max(0.0, now - previous_observed)
        continuous_since = existing.get("continuous_progress_since_epoch")
        continuous_start_progress = existing.get("continuous_start_progress")
        if (
            observation_gap > HANDOVER_MAX_OBSERVATION_AGE_SECONDS
            or not isinstance(continuous_since, (int, float))
            or not isinstance(continuous_start_progress, int)
        ):
            if progress > baseline_progress:
                continuous_since = now
                continuous_start_progress = (
                    previous
                    if observation_gap <= HANDOVER_MAX_OBSERVATION_AGE_SECONDS
                    and previous >= baseline_progress
                    else progress
                )
            else:
                continuous_since = None
                continuous_start_progress = None
    else:
        changed_at = now
        continuous_since = now if progress > baseline_progress else None
        continuous_start_progress = (
            baseline_progress if progress > baseline_progress else None
        )
    monitor = {
        "schema_version": EXACT_PROGRESS_MONITOR_SCHEMA,
        "controller_manifest_sha256": state["controller_manifest_sha256"],
        "stage_id": EXACT_STAGE,
        "progress": int(progress),
        "last_change_epoch": changed_at,
        "continuous_progress_since_epoch": continuous_since,
        "continuous_start_progress": continuous_start_progress,
        "baseline_progress": baseline_progress,
        "observed_epoch": now,
        "observed_at": _utc_now(),
    }
    monitor["monitor_sha256"] = stable_json_sha256(monitor)
    state["exact_progress_monitor"] = monitor
    return monitor


def _coexistence_probe(
    *,
    manifest: Mapping[str, Any],
    stage: Mapping[str, Any],
    pid: int,
    start_sample: Mapping[str, Any],
    start_progress: int,
    start_time: float,
    worker_argv_sha256: str,
    proc_root: Path,
) -> dict[str, Any] | None:
    if stage["stage_id"] != EXACT_STAGE:
        return {"not_applicable": True}
    progress = _progress_value(stage)
    if progress is None or progress - start_progress < int(
        manifest["resources"]["coexistence_probe"]["min_progress_rows"]
    ):
        if time.monotonic() - start_time > int(
            manifest["resources"]["coexistence_probe"]["timeout_seconds"]
        ):
            raise RecoveryControllerError("exact coexistence checkpoint probe timed out")
        return None
    end = _host_sample(proc_root=proc_root)
    total_delta = int(end["cpu_total_ticks"]) - int(start_sample["cpu_total_ticks"])
    iowait_delta = int(end["cpu_iowait_ticks"]) - int(start_sample["cpu_iowait_ticks"])
    iowait_fraction = 0.0 if total_delta <= 0 else max(0.0, iowait_delta / total_delta)
    rss = _proc_rss_bytes(pid)
    contract = manifest["resources"]["coexistence_probe"]
    passed = (
        float(end["load_per_cpu"]) <= float(contract["max_load_per_cpu"])
        and iowait_fraction <= float(contract["max_iowait_fraction"])
        and rss <= int(manifest["resources"]["max_rss_bytes"])
    )
    result = {
        "schema_version": COEXISTENCE_SCHEMA,
        "status": "PASS" if passed else "BLOCKED",
        "controller_id": CONTROLLER_ID,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "worker_argv_sha256": worker_argv_sha256,
        "thread_count": int(manifest["resources"]["thread_count"]),
        "cuda_visible_devices": "",
        "gpu_lock_acquired": False,
        "start_progress": start_progress,
        "end_progress": progress,
        "start_host": dict(start_sample),
        "end_host": end,
        "iowait_fraction": iowait_fraction,
        "worker_rss_bytes": rss,
        "contract": dict(contract),
        "checked_at": _utc_now(),
    }
    if not passed:
        raise RecoveryControllerError("exact CPU coexistence probe failed")
    return result


def _publish_terminal_exact_coexistence_probe(
    *,
    manifest: Mapping[str, Any],
    stage: Mapping[str, Any],
    state: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Close a fast/naturally-finished exact worker from persisted evidence."""

    if path.exists():
        return _validate_coexistence_receipt(manifest, path)
    baseline = state.get("exact_coexistence_baseline")
    if not isinstance(baseline, Mapping) or baseline.get("stage_id") != EXACT_STAGE:
        raise RecoveryControllerError(
            "exact terminal lacks its pre-spawn coexistence baseline"
        )
    start = baseline.get("start_host")
    if not isinstance(start, Mapping):
        raise RecoveryControllerError("exact coexistence start sample is absent")
    progress = _progress_value(stage)
    start_progress = int(baseline.get("start_progress", -1))
    contract = manifest["resources"]["coexistence_probe"]
    if progress is None or start_progress < 0 or progress < start_progress:
        raise RecoveryControllerError("exact terminal checkpoint progress regressed")
    terminal_path = _require_absolute(
        stage["terminal_path"], label="exact terminal coexistence receipt", existing="file"
    )
    terminal = _read_json(terminal_path, label="exact terminal coexistence evidence")
    terminal_sha256 = sha256_file(terminal_path)
    dbscan_path = _require_absolute(
        terminal.get("dbscan_manifest_path"),
        label="exact terminal DBSCAN coexistence evidence",
        existing="file",
    )
    if sha256_file(dbscan_path) != terminal.get("dbscan_manifest_sha256"):
        raise RecoveryControllerError("exact terminal DBSCAN changed before probe")
    dbscan = _read_json(dbscan_path, label="exact terminal DBSCAN coexistence")
    rss = int(dbscan.get("peak_rss_bytes_observed", -1))
    end = _host_sample(
        proc_root=Path(str(manifest["resources"].get("proc_root", "/proc")))
    )
    total_delta = int(end["cpu_total_ticks"]) - int(start.get("cpu_total_ticks", 0))
    iowait_delta = int(end["cpu_iowait_ticks"]) - int(
        start.get("cpu_iowait_ticks", 0)
    )
    iowait_fraction = (
        0.0 if total_delta <= 0 else max(0.0, iowait_delta / total_delta)
    )
    passed = (
        float(end["load_per_cpu"]) <= float(contract["max_load_per_cpu"])
        and iowait_fraction <= float(contract["max_iowait_fraction"])
        and 0 <= rss <= int(manifest["resources"]["max_rss_bytes"])
    )
    result = {
        "schema_version": COEXISTENCE_SCHEMA,
        "status": "PASS" if passed else "BLOCKED",
        "controller_id": CONTROLLER_ID,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "worker_argv_sha256": baseline.get("worker_argv_sha256"),
        "thread_count": int(manifest["resources"]["thread_count"]),
        "cuda_visible_devices": "",
        "gpu_lock_acquired": False,
        "start_progress": start_progress,
        "end_progress": progress,
        "start_host": dict(start),
        "end_host": end,
        "iowait_fraction": iowait_fraction,
        "worker_rss_bytes": rss,
        "worker_rss_source": "terminal_dbscan_peak_rss_bytes_observed",
        "terminal_fast_completion_reconciled": True,
        "terminal_completed_before_min_progress": (
            progress - start_progress < int(contract["min_progress_rows"])
        ),
        "terminal_path": str(terminal_path),
        "terminal_sha256": terminal_sha256,
        "dbscan_manifest_path": str(dbscan_path),
        "dbscan_manifest_sha256": terminal["dbscan_manifest_sha256"],
        "contract": dict(contract),
        "checked_at": _utc_now(),
    }
    if not passed:
        raise RecoveryControllerError("exact terminal CPU coexistence probe failed")
    encoded = _json_payload_bytes(result)
    _reserve_output_growth(manifest, [(path, len(encoded), False)])
    _write_new_bytes(path, encoded)
    _disk_preflight(manifest)
    return _validate_coexistence_receipt(manifest, path)


def _validate_coexistence_receipt(
    manifest: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    if _inspect_immutable_publication(path):
        raise RecoveryControllerError(
            f"IMMUTABLE_PUBLICATION_RECONCILIATION_REQUIRED:{path}"
        )
    value = _read_json(path, label="exact CPU coexistence receipt")
    contract = manifest["resources"]["coexistence_probe"]
    exact = _stage(manifest, EXACT_STAGE)
    allowed_argv = {
        exact["commands"]["fresh_sha256"],
        exact["commands"]["resume_sha256"],
    } - {None}
    progress_delta = int(value.get("end_progress", -1)) - int(
        value.get("start_progress", -1)
    )
    terminal_reconciled = value.get("terminal_fast_completion_reconciled") is True
    progress_requirement_passed = progress_delta >= int(contract["min_progress_rows"])
    if terminal_reconciled:
        terminal_path = _require_absolute(
            value.get("terminal_path"),
            label="terminal-reconciled coexistence receipt",
            existing="file",
        )
        expected_terminal = Path(exact["terminal_path"]).resolve(strict=True)
        dbscan_path = _require_absolute(
            value.get("dbscan_manifest_path"),
            label="terminal-reconciled DBSCAN manifest",
            existing="file",
        )
        terminal = _read_json(terminal_path, label="terminal-reconciled exact receipt")
        dbscan = _read_json(dbscan_path, label="terminal-reconciled DBSCAN manifest")
        terminal_closure_passed = (
            terminal_path == expected_terminal
            and sha256_file(terminal_path) == value.get("terminal_sha256")
            and terminal.get("dbscan_manifest_path") == str(dbscan_path)
            and terminal.get("dbscan_manifest_sha256")
            == value.get("dbscan_manifest_sha256")
            and sha256_file(dbscan_path) == value.get("dbscan_manifest_sha256")
            and int(dbscan.get("peak_rss_bytes_observed", -1))
            == int(value.get("worker_rss_bytes", -2))
        )
        progress_requirement_passed = progress_delta >= 0
    else:
        terminal_closure_passed = (
            value.get("terminal_completed_before_min_progress") in {None, False}
            and value.get("terminal_path") is None
            and value.get("terminal_sha256") is None
            and value.get("dbscan_manifest_path") is None
            and value.get("dbscan_manifest_sha256") is None
        )
    start_host = value.get("start_host")
    end_host = value.get("end_host")
    host_counters_monotonic = (
        isinstance(start_host, Mapping)
        and isinstance(end_host, Mapping)
        and int(end_host.get("cpu_total_ticks", -1))
        >= int(start_host.get("cpu_total_ticks", 0))
        and int(end_host.get("cpu_iowait_ticks", -1))
        >= int(start_host.get("cpu_iowait_ticks", 0))
    )
    if (
        value.get("schema_version") != COEXISTENCE_SCHEMA
        or value.get("status") != "PASS"
        or value.get("controller_id") != CONTROLLER_ID
        or value.get("controller_manifest_sha256") != manifest["manifest_sha256"]
        or value.get("worker_argv_sha256") not in allowed_argv
        or value.get("thread_count") != manifest["resources"]["thread_count"]
        or value.get("cuda_visible_devices") != ""
        or value.get("gpu_lock_acquired") is not False
        or value.get("contract") != contract
        or not progress_requirement_passed
        or not terminal_closure_passed
        or not host_counters_monotonic
        or not isinstance(end_host, Mapping)
        or float(end_host.get("load_per_cpu", float("inf")))
        > float(contract["max_load_per_cpu"])
        or float(value.get("iowait_fraction", float("inf")))
        > float(contract["max_iowait_fraction"])
        or int(value.get("worker_rss_bytes", -1)) < 0
        or int(value.get("worker_rss_bytes", -1))
        > int(manifest["resources"]["max_rss_bytes"])
    ):
        raise RecoveryControllerError("exact CPU coexistence receipt is invalid")
    return value


def _run_or_attach_stage(
    *,
    manifest: Mapping[str, Any],
    stage: Mapping[str, Any],
    root: Path,
    state: dict[str, Any],
    guard: Callable[[], None],
    poll_seconds: float,
) -> None:
    stage_id = str(stage["stage_id"])
    terminal = Path(stage["terminal_path"])
    worker = state.get("worker")
    proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
    process: subprocess.Popen[bytes] | None = None
    start_sample: Mapping[str, Any]
    start_progress: int
    probe_argv_sha256: str
    start_time = time.monotonic()
    try:
        state["resource_preflight"] = _disk_preflight(manifest)
    except Exception:
        _terminate_bound_worker_group(
            root=root, stage=stage, worker=worker, proc_root=proc_root
        )
        raise
    _update_exact_progress_monitor(stage=stage, state=state)
    last_resource_check = time.monotonic()
    if isinstance(worker, Mapping) and worker.get("stage_id") == stage_id:
        pid = int(worker["pid"])
        start_ticks = int(worker["start_ticks"])
        if _pid_alive(pid, start_ticks, proc_root=proc_root):
            actual_argv = _proc_argv(pid, proc_root=proc_root)
            if not _worker_actual_argv_is_bound(
                root=root,
                stage=stage,
                worker=worker,
                actual_argv=actual_argv,
            ):
                raise RecoveryControllerError(
                    f"live worker command identity mismatch: {stage_id}"
                )
            if stage_id == EXACT_STAGE:
                baseline = _ensure_exact_coexistence_baseline(
                    manifest=manifest,
                    stage=stage,
                    state=state,
                    worker_argv_sha256=str(worker["argv_sha256"]),
                    root=root,
                    guard=guard,
                    proc_root=proc_root,
                )
                start_sample = baseline["start_host"]
                start_progress = int(baseline["start_progress"])
                probe_argv_sha256 = str(baseline["worker_argv_sha256"])
            else:
                start_sample = worker["coexistence_start"]
                start_progress = int(worker["start_progress"])
                probe_argv_sha256 = str(worker["argv_sha256"])
            start_time = time.monotonic() - float(worker.get("elapsed_seconds", 0.0))
            if stage_id == EXACT_STAGE:
                _record_durable_progress_observation(
                    manifest=manifest, state=state
                )
        elif terminal.is_file():
            _wait_for_process_group_quiescence(
                int(worker.get("process_group_id", pid)),
                proc_root=proc_root,
                timeout_seconds=30.0,
            )
            if (
                stage_id == EXACT_STAGE
                and not (root / "coexistence_probe.json").exists()
            ):
                _publish_terminal_exact_coexistence_probe(
                    manifest=manifest,
                    stage=stage,
                    state=state,
                    path=root / "coexistence_probe.json",
                )
                _update_exact_progress_monitor(stage=stage, state=state)
            state["worker"] = None
            state["startup_barrier"] = None
            return
        else:
            process_group_id = int(worker.get("process_group_id", pid))
            members = _process_group_member_pids(
                process_group_id, proc_root=proc_root
            )
            if members:
                raise RecoveryControllerError(
                    "DEAD_WORKER_HAS_LIVE_PROCESS_GROUP:"
                    f"stage={stage_id}:pgid={process_group_id}:"
                    f"members={list(members)[:16]}"
                )
            resume_argv = stage["commands"]["resume"]
            if resume_argv is None:
                raise RecoveryControllerError(f"dead stage has no safe resume command: {stage_id}")
            startup = state.get("startup_barrier")
            if isinstance(startup, Mapping):
                startup = dict(startup)
                if startup.get("phase") != "BOUND":
                    raise RecoveryControllerError(
                        f"dead worker startup binding is not BOUND: {stage_id}"
                    )
                startup["phase"] = "QUIESCENT"
                state["startup_barrier"] = startup
            else:
                raise RecoveryControllerError(
                    f"dead worker startup binding is absent: {stage_id}"
                )
            state["worker"] = None
            _save_state(manifest, root, state, guard)
            worker = None
    else:
        worker = None
    if worker is None:
        output = Path(stage["output_dir"])
        fresh = not output.exists()
        argv = stage["commands"]["fresh" if fresh else "resume"]
        if argv is None:
            raise RecoveryControllerError(f"stage output exists without a resume command: {stage_id}")
        argv_sha256 = stable_json_sha256(argv)
        baseline: Mapping[str, Any] | None = None
        if stage_id == EXACT_STAGE:
            # Persist before Popen.  A scientifically complete worker can exit
            # before /proc exposes a bindable PID generation.
            baseline = _ensure_exact_coexistence_baseline(
                manifest=manifest,
                stage=stage,
                state=state,
                worker_argv_sha256=argv_sha256,
                root=root,
                guard=guard,
                proc_root=proc_root,
            )
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stdout = None
        stderr = None
        barrier: ArmedExecStartupBarrier | None = None
        ownership_bound = False
        science_released = False
        fast_terminal_complete = False
        try:
            stdout = (logs / f"{stage_id}.stdout.log").open("ab")
            stderr = (logs / f"{stage_id}.stderr.log").open("ab")
            guard()
            barrier, barrier_binding = _prepare_exec_startup_barrier(
                manifest=manifest,
                root=root,
                stage=stage,
                state=state,
                target_argv=argv,
                guard=guard,
            )
            process = barrier.launch(
                cwd=manifest["project_root"],
                env=_stage_environment(manifest),
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            stdout.close()
            stdout = None
            stderr.close()
            stderr = None
            deadline = time.monotonic() + 10.0
            start_ticks = None
            while start_ticks is None and time.monotonic() < deadline:
                start_ticks = _read_proc_start_ticks(process.pid, proc_root=proc_root)
                if start_ticks is None:
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
            if start_ticks is None:
                process.wait(timeout=30)
                raise RecoveryControllerError(
                    f"cannot bind worker PID generation: {stage_id}"
                )
            actual_argv = _proc_argv(process.pid, proc_root=proc_root)
            if (
                actual_argv is None
                or stable_json_sha256(actual_argv)
                != barrier_binding["launcher_argv_sha256"]
            ):
                raise RecoveryControllerError(
                    f"spawned startup wrapper identity mismatch: {stage_id}"
                )
            if baseline is not None:
                start_sample = baseline["start_host"]
                start_progress = int(baseline["start_progress"])
                probe_argv_sha256 = str(baseline["worker_argv_sha256"])
            else:
                start_sample = _host_sample(proc_root=proc_root)
                start_progress = _progress_value(stage) or 0
                probe_argv_sha256 = argv_sha256
            bound_barrier = {**barrier_binding, "phase": "BOUND"}
            state["startup_barrier"] = bound_barrier
            state["worker"] = {
                "stage_id": stage_id,
                "pid": process.pid,
                "process_group_id": process.pid,
                "start_ticks": start_ticks,
                "argv_sha256": argv_sha256,
                "coexistence_start": start_sample,
                "start_progress": start_progress,
                "elapsed_seconds": 0.0,
                "startup_barrier": bound_barrier,
            }
            _save_state(manifest, root, state, guard)
            ownership_bound = True
            barrier.release()
            science_released = True
            pid = process.pid
        except BaseException:
            if barrier is not None and not science_released:
                barrier.abort()
            if (
                process is not None
                and not fast_terminal_complete
                and process.poll() is None
            ):
                if science_released:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                _wait_for_process_group_quiescence(
                    process.pid,
                    proc_root=proc_root,
                    timeout_seconds=None,
                )
            raise
        finally:
            if stdout is not None:
                stdout.close()
            if stderr is not None:
                stderr.close()
    probe_path = root / "coexistence_probe.json"
    probe_complete = probe_path.exists() or stage_id != EXACT_STAGE
    try:
        while (
            process.poll() is None
            if process is not None
            else _pid_alive(pid, start_ticks, proc_root=proc_root)
        ):
            guard()
            now = time.monotonic()
            if now - last_resource_check >= 60.0:
                state["resource_preflight"] = _disk_preflight(manifest)
                last_resource_check = now
            if not probe_complete:
                result = _coexistence_probe(
                    manifest=manifest,
                    stage=stage,
                    pid=pid,
                    start_sample=start_sample,
                    start_progress=start_progress,
                    start_time=start_time,
                    worker_argv_sha256=probe_argv_sha256,
                    proc_root=proc_root,
                )
                if result is not None:
                    encoded = _json_payload_bytes(result)
                    _reserve_output_growth(
                        manifest, [(probe_path, len(encoded), False)]
                    )
                    _write_new_bytes(probe_path, encoded)
                    _disk_preflight(manifest)
                    probe_complete = True
            _update_exact_progress_monitor(stage=stage, state=state)
            if stage_id == EXACT_STAGE:
                _record_durable_progress_observation(
                    manifest=manifest, state=state
                )
            state["worker"]["elapsed_seconds"] = time.monotonic() - start_time
            _save_state(manifest, root, state, guard)
            time.sleep(poll_seconds)
    except Exception:
        _terminate_bound_worker_group(
            root=root,
            stage=stage,
            worker=state.get("worker"),
            proc_root=proc_root,
        )
        raise
    if process is not None:
        return_code = process.wait()
        _wait_for_process_group_quiescence(
            process.pid, proc_root=proc_root, timeout_seconds=30.0
        )
        if return_code != 0 and not terminal.is_file():
            raise RecoveryControllerError(f"stage worker failed: {stage_id}:exit={return_code}")
    if not terminal.is_file():
        raise RecoveryControllerError(f"stage worker exited without terminal: {stage_id}")
    if stage_id == EXACT_STAGE and not probe_complete:
        _publish_terminal_exact_coexistence_probe(
            manifest=manifest,
            stage=stage,
            state=state,
            path=probe_path,
        )
        probe_complete = True
    _update_exact_progress_monitor(stage=stage, state=state)
    state["worker"] = None
    state["startup_barrier"] = None


def _publish_final_terminal(
    manifest: Mapping[str, Any],
    root: Path,
    held: HeldControllerLock,
    state: dict[str, Any],
) -> dict[str, Any]:
    _disk_preflight(manifest)
    gates = [_open_gate(manifest, stage) for stage in STAGE_ORDER]
    owner_binding = _validate_controller_owner_claim(manifest)
    payload: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": "PASS",
        "run_complete": True,
        "controller_id": CONTROLLER_ID,
        "controller_manifest_path": manifest["manifest_path"],
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "typed_stage_gate_sha256": [gate["gate_sha256"] for gate in gates],
        "writer_lock_identity": held.identity,
        **owner_binding,
        "stage_order": list(STAGE_ORDER),
        "adoption_was_recovery_only": True,
        "failed_evidence_was_not_ordinary_pass": True,
        "subset_preflight_preceded_full_recovery": True,
        "exact_partition_proven": True,
        "downstream_boundary_full_replay_pass": True,
        "recovery_only": False,
        "ordinary_pass_dependency_eligible": True,
        "dbscan_partition_proven": True,
        "gpu_used": False,
        "hpc_used": False,
        "completed_at": _utc_now(),
    }
    payload["terminal_sha256"] = stable_json_sha256(payload)
    terminal_path = root / "terminal.json"
    state["status"] = "PASS"
    state["current_stage"] = None
    state["worker"] = None
    state["updated_at"] = _utc_now()
    terminal_bytes = _json_payload_bytes(payload)
    state_bytes = _json_payload_bytes(state)
    _reserve_output_growth(
        manifest,
        [
            (terminal_path, len(terminal_bytes), False),
            (root / "state.json", len(state_bytes), True),
            (root / "PASS", len(b"PASS\n"), False),
        ],
    )
    held.verify()
    _write_new_bytes(terminal_path, terminal_bytes)
    held.verify()
    _atomic_state(root / "state.json", state)
    held.verify()
    _write_new_bytes(root / "PASS", b"PASS\n")
    held.verify()
    _disk_preflight(manifest)
    return payload


def validate_controller_terminal(
    manifest: Mapping[str, Any], *, require_pass_marker: bool = True
) -> dict[str, Any]:
    _disk_preflight(manifest)
    root = Path(manifest["controller_root"]).resolve(strict=True)
    for path in (root / "terminal.json", root / "PASS"):
        if (path.exists() or path.is_symlink()) and _inspect_immutable_publication(path):
            raise RecoveryControllerError(
                f"IMMUTABLE_PUBLICATION_RECONCILIATION_REQUIRED:{path}"
            )
    terminal = _read_json(root / "terminal.json", label="recovery controller terminal")
    projected = dict(terminal)
    terminal_sha = projected.pop("terminal_sha256", None)
    gates = [_open_gate(manifest, stage) for stage in STAGE_ORDER]
    owner_binding = _validate_controller_owner_claim(manifest)
    if (
        terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("status") != "PASS"
        or terminal.get("run_complete") is not True
        or terminal.get("controller_manifest_path") != manifest.get("manifest_path")
        or terminal.get("controller_manifest_sha256")
        != manifest.get("manifest_sha256")
        or terminal_sha != stable_json_sha256(projected)
        or terminal.get("typed_stage_gate_sha256") != [gate["gate_sha256"] for gate in gates]
        or terminal.get("writer_lock_identity") != _current_lock_identity(root)
        or terminal.get("owner_claim_path")
        != owner_binding["owner_claim_path"]
        or terminal.get("owner_claim_sha256")
        != owner_binding["owner_claim_sha256"]
        or terminal.get("root_preclaim") != owner_binding["root_preclaim"]
        or terminal.get("failed_evidence_was_not_ordinary_pass") is not True
        or terminal.get("recovery_only") is not False
        or terminal.get("ordinary_pass_dependency_eligible") is not True
        or terminal.get("dbscan_partition_proven") is not True
        or terminal.get("gpu_used") is not False
        or terminal.get("hpc_used") is not False
    ):
        raise RecoveryControllerError("recovery controller terminal closure mismatch")
    if require_pass_marker and (
        not (root / "PASS").is_file() or (root / "PASS").read_bytes() != b"PASS\n"
    ):
        raise RecoveryControllerError("recovery controller PASS-last marker is absent")
    return terminal


def _load_launchable_manifest(manifest_path: str | Path) -> dict[str, Any]:
    return load_bound_controller_manifest(manifest_path)


def prepare_controller_launch(
    manifest_path: str | Path, *, resume: bool
) -> dict[str, Any]:
    """Preclaim one fresh CID and publish an immutable launch-local receipt.

    The shell launcher never writes a fixed filename in the generic control
    root.  Fresh preparation claims the controller root once; the actual
    controller therefore always starts with ``--resume``.
    """

    manifest = _load_launchable_manifest(manifest_path)
    root = _claim_controller_root(manifest, resume=resume)
    with _controller_lock(root) as held:
        _reconcile_controller_owned_publications(manifest, root)
        resource = _disk_preflight(manifest)
        launch_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "_"
            + manifest["manifest_sha256"][:8]
        )
        log_path = root / "logs" / f"controller.{launch_id}.log"
        pid_path = root / "logs" / f"controller.{launch_id}.pid"
        receipt_path = root / "logs" / f"prelaunch.{launch_id}.json"
        session = f"aids_exact_{manifest['cid'][-8:]}_{launch_id[:15]}"
        payload = {
            "schema_version": "aids_comrecgc_exact_recovery_prelaunch_v1",
            "status": "READY",
            "controller_id": CONTROLLER_ID,
            "cid": manifest["cid"],
            "controller_root": str(root),
            "controller_manifest_path": manifest["manifest_path"],
            "controller_manifest_sha256": manifest["manifest_sha256"],
            "requested_mode": "resume" if resume else "fresh",
            "controller_invocation_requires_resume": True,
            "launch_id": launch_id,
            "log_path": str(log_path),
            "pid_path": str(pid_path),
            "tmux_session": session,
            "thread_count": int(manifest["resources"]["thread_count"]),
            "resource_preflight": resource,
            "writer_lock_identity": held.identity,
            "prepared_at": _utc_now(),
        }
        encoded = _json_payload_bytes(payload)
        _reserve_output_growth(
            manifest, [(receipt_path, len(encoded), False)]
        )
        held.verify()
        _write_new_bytes(receipt_path, encoded)
        held.verify()
        _disk_preflight(manifest)
        return {**payload, "prelaunch_receipt_path": str(receipt_path)}


def run_controller(
    manifest_path: str | Path,
    *,
    resume: bool,
    poll_seconds: float = 5.0,
    adoption_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = _load_launchable_manifest(manifest_path)
    root = _claim_controller_root(manifest, resume=resume)
    with _controller_lock(root) as held:
        _reconcile_controller_owned_publications(manifest, root)
        if (root / "terminal.json").exists():
            terminal = validate_controller_terminal(
                manifest, require_pass_marker=False
            )
            if not (root / "PASS").exists():
                # A crash may occur after terminal.json but before PASS-last.
                # Reopen every typed scientific terminal (including the full
                # component-summary replay) before reconstructing authority.
                for stage_id in STAGE_ORDER:
                    validate_stage_terminal(
                        manifest,
                        stage_id=stage_id,
                        adoption_validator=adoption_validator,
                    )
                state = _load_state(root, manifest)
                state["status"] = "PASS"
                state["current_stage"] = None
                state["worker"] = None
                state["updated_at"] = _utc_now()
                state_bytes = _json_payload_bytes(state)
                _reserve_output_growth(
                    manifest,
                    [
                        (root / "state.json", len(state_bytes), True),
                        (root / "PASS", len(b"PASS\n"), False),
                    ],
                )
                held.verify()
                _atomic_state(root / "state.json", state)
                held.verify()
                _write_new_bytes(root / "PASS", b"PASS\n")
                held.verify()
                _disk_preflight(manifest)
            else:
                state = _load_state(root, manifest)
                if state.get("status") != "PASS" or state.get("current_stage") is not None:
                    raise RecoveryControllerError(
                        "controller PASS exists before terminal state closure"
                    )
            validate_controller_terminal(manifest)
            return terminal
        state = _load_state(root, manifest)
        try:
            state["status"] = "RUNNING"
            controller_start_ticks = _read_proc_start_ticks(
                os.getpid(),
                proc_root=Path(str(manifest["resources"].get("proc_root", "/proc"))),
            )
            current_controller = {
                "pid": os.getpid(),
                "start_ticks": controller_start_ticks,
                "argv_sha256": stable_json_sha256(sys.argv),
                "registered_at": _utc_now(),
            }
            handover_generation = _ensure_current_controller_generation(
                manifest=manifest,
                current_controller=current_controller,
            )
            durable_resume = _publish_resume_receipt_if_reattached(
                manifest=manifest,
                state=state,
                generation=handover_generation,
            )
            # Mutable state is diagnostic only. Handover eligibility reopens
            # the O_EXCL generation/resume artifacts and never trusts this
            # projection or its publicly recomputable hash.
            state["durable_handover_generation"] = int(
                handover_generation["payload"]["generation"]
            )
            state["durable_resume_receipt_path"] = (
                durable_resume["artifact"]["path"]
                if durable_resume is not None
                else None
            )
            state["controller_process"] = current_controller
            _save_state(manifest, root, state, held.verify)
            for stage_id in STAGE_ORDER:
                held.verify()
                stage = _stage(manifest, stage_id)
                gate_path = _gate_path(manifest, stage_id)
                tracked_worker = state.get("worker")
                if (
                    isinstance(tracked_worker, Mapping)
                    and tracked_worker.get("stage_id") == stage_id
                    and (gate_path.exists() or Path(stage["terminal_path"]).is_file())
                ):
                    # A terminal or even an already-published gate is not a
                    # substitute for writer quiescence.  Reattach until the
                    # bound leader and every non-zombie process-group member
                    # are gone, then reopen the scientific closure below.
                    _run_or_attach_stage(
                        manifest=manifest,
                        stage=stage,
                        root=root,
                        state=state,
                        guard=held.verify,
                        poll_seconds=poll_seconds,
                    )
                startup = state.get("startup_barrier")
                if (
                    not isinstance(state.get("worker"), Mapping)
                    and isinstance(startup, Mapping)
                    and startup.get("stage_id") == stage_id
                    and startup.get("phase") == "BOUND"
                ):
                    raise RecoveryControllerError(
                        f"BOUND startup barrier lost its worker binding: {stage_id}"
                    )
                _reconcile_completed_stage_publication(
                    stage_id=stage_id, stage=stage
                )
                if gate_path.exists():
                    _open_gate(manifest, stage_id)
                    state["stages"][stage_id] = "PASS"
                    continue
                if Path(stage["terminal_path"]).is_file():
                    if (
                        stage_id == EXACT_STAGE
                        and not (root / "coexistence_probe.json").exists()
                    ):
                        _publish_terminal_exact_coexistence_probe(
                            manifest=manifest,
                            stage=stage,
                            state=state,
                            path=root / "coexistence_probe.json",
                        )
                    evidence = validate_stage_terminal(
                        manifest,
                        stage_id=stage_id,
                        adoption_validator=adoption_validator,
                    )
                    _publish_stage_gate(
                        manifest,
                        stage_id=stage_id,
                        evidence=evidence,
                        held=held,
                    )
                    state["stages"][stage_id] = "PASS"
                    _save_state(manifest, root, state, held.verify)
                    continue
                state["current_stage"] = stage_id
                state["stages"][stage_id] = "RUNNING"
                _save_state(manifest, root, state, held.verify)
                _run_or_attach_stage(
                    manifest=manifest,
                    stage=stage,
                    root=root,
                    state=state,
                    guard=held.verify,
                    poll_seconds=poll_seconds,
                )
                _reconcile_completed_stage_publication(
                    stage_id=stage_id, stage=stage
                )
                evidence = validate_stage_terminal(
                    manifest,
                    stage_id=stage_id,
                    adoption_validator=adoption_validator,
                )
                held.verify()
                _publish_stage_gate(
                    manifest,
                    stage_id=stage_id,
                    evidence=evidence,
                    held=held,
                )
                state["stages"][stage_id] = "PASS"
                state["worker"] = None
                state["startup_barrier"] = None
                _save_state(manifest, root, state, held.verify)
            final = _publish_final_terminal(manifest, root, held, state)
            validate_controller_terminal(manifest)
            return final
        except Exception as exc:
            state["status"] = "BLOCKED"
            current = state.get("current_stage")
            if current in state["stages"]:
                state["stages"][current] = "BLOCKED"
            state["last_error"] = {
                "error_class": type(exc).__name__,
                "message": str(exc),
                "recorded_at": _utc_now(),
            }
            _save_state(manifest, root, state, held.verify)
            raise


def _old_brute_handover_status(
    *,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a read-only, fail-closed old-route handover certificate.

    This controller never signals the legacy process. A separate reviewed
    executor may consume ``eligible_to_request_old_brute_stop`` only after it
    independently reopens the exact old PID generation.
    """

    contract = handover_contract()
    errors: dict[str, str] = {}
    project_root = Path(str(manifest["project_root"]))
    try:
        release_ancestry_pass = _release_commits_are_ancestors(
            project_root,
            execution_commit=str(manifest["execution_commit"]),
            pins=manifest["release_pins"],
        )
        execution_tree_clean = _execution_tree_clean(project_root)
    except Exception as exc:
        release_ancestry_pass = False
        execution_tree_clean = False
        errors["review_release"] = f"{type(exc).__name__}:{exc}"
    review_release_pass = bool(
        manifest.get("release_ready") is True
        and manifest.get("production_deployment_authorized") is True
        and manifest.get("release_pins", {}).get("controller_commit")
        == manifest.get("execution_commit")
        and release_ancestry_pass
        and execution_tree_clean
    )

    adoption_gate_pass = result.get("stages", {}).get(ADOPTION_STAGE) == "PASS"
    source = manifest.get("source_authority")
    source_manifest_pass = False
    pair_store_checksum_pass = False
    if adoption_gate_pass and isinstance(source, Mapping):
        try:
            validated_source = _validate_source_authority(source)
        except Exception as exc:
            errors["source_manifest"] = f"{type(exc).__name__}:{exc}"
        else:
            source_manifest_pass = True
            pair_store_checksum_pass = bool(
                int(validated_source.get("physical_pair_count", -1))
                == EXPECTED_ROWS
                and validated_source.get("pair_store_regenerated") is False
                and validated_source.get("source_pair_store_access")
                == "read_only_zero_copy"
                and _is_sha256(validated_source.get("pair_store_manifest_sha256"))
                and _is_sha256(validated_source.get("physical_pairs_sha256"))
                and _is_sha256(
                    validated_source.get("normalized_distances_sha256")
                )
                and _is_sha256(validated_source.get("close_bitmap_sha256"))
                and _is_sha256(validated_source.get("source_vectors_sha256"))
            )

    resume_smoke_pass = False
    first_durable_checkpoint_pass = False
    observation_age_seconds: float | None = None
    last_change_age_seconds: float | None = None
    continuous_progress_seconds: float | None = None
    progress_rows: int | None = None
    progress_delta_rows: int | None = None
    throughput_rows_per_second: float | None = None
    conservative_remaining_rows: int | None = None
    conservative_eta_seconds: float | None = None
    continuous_progress_pass = False
    positive_throughput_pass = False
    checkpoint: dict[str, Any] | None = None
    resume_smoke: dict[str, Any] | None = None
    new_exact_worker_generation_bound_pass = False
    try:
        durable = _verify_durable_handover_evidence(
            manifest=manifest
        )
    except Exception as exc:
        errors["durable_handover_evidence"] = f"{type(exc).__name__}:{exc}"
    else:
        resume_smoke_pass = bool(durable["resume_smoke_pass"])
        first_durable_checkpoint_pass = bool(
            durable["first_durable_checkpoint_pass"]
        )
        continuous_progress_pass = bool(durable["continuous_progress_pass"])
        positive_throughput_pass = bool(durable["positive_throughput_pass"])
        new_exact_worker_generation_bound_pass = bool(
            durable["new_exact_worker_generation_bound_pass"]
        )
        observation_age_seconds = durable["observation_age_seconds"]
        last_change_age_seconds = durable["last_change_age_seconds"]
        continuous_progress_seconds = durable["continuous_progress_seconds"]
        progress_rows = durable["progress_rows"]
        progress_delta_rows = durable["progress_delta_rows"]
        throughput_rows_per_second = durable["throughput_rows_per_second"]
        conservative_remaining_rows = durable["conservative_remaining_rows"]
        conservative_eta_seconds = durable["conservative_eta_seconds"]
        checkpoint = durable["checkpoint_snapshot"]
        resume_smoke = durable["resume_smoke_receipt"]

    eta_within_48h_pass = bool(
        positive_throughput_pass
        and conservative_eta_seconds is not None
        and conservative_eta_seconds <= HANDOVER_MAX_ETA_SECONDS
    )
    # A caller-supplied old-route denominator is not evidence. The distinct
    # stop executor must provide a separately reviewed, exact-generation live
    # throughput receipt if the conservative ETA branch is unavailable.
    relative_speedup_100x_pass = False
    eta_or_100x_pass = eta_within_48h_pass or relative_speedup_100x_pass
    old_brute_generation_alive_pass = _old_brute_generation_alive(manifest)
    controller_status_allows_handover = bool(
        result.get("status") not in {
            "BLOCKED",
            "FILESYSTEM_RECONCILIATION_REQUIRED",
            "OUTPUT_BUDGET_EXCEEDED",
            "PRELAUNCH_HISTORY_INVALID",
            "TERMINAL_RECONCILIATION_REQUIRED",
        }
        and result.get("route_viability") not in {
            "BLOCKED",
            "BLOCKED_RECOVERABLE",
            "RUNNING_STALLED",
        }
    )
    conditions = {
        "review_release_pass": review_release_pass,
        "source_manifest_pass": source_manifest_pass,
        "pair_store_checksum_pass": pair_store_checksum_pass,
        "resume_smoke_pass": resume_smoke_pass,
        "first_durable_checkpoint_pass": first_durable_checkpoint_pass,
        "continuous_progress_10m_pass": continuous_progress_pass,
        "positive_throughput_pass": positive_throughput_pass,
        "eta_within_48h_pass": eta_within_48h_pass,
        "relative_speedup_100x_pass": relative_speedup_100x_pass,
        "eta_or_100x_pass": eta_or_100x_pass,
        "eta_or_100x_gate_required": False,
        "old_brute_exact_generation_alive_pass": (
            old_brute_generation_alive_pass
        ),
        "new_exact_worker_generation_bound_pass": (
            new_exact_worker_generation_bound_pass
        ),
        "controller_status_allows_handover": controller_status_allows_handover,
    }
    eligible = all(
        conditions[name]
        for name in (
            "review_release_pass",
            "source_manifest_pass",
            "pair_store_checksum_pass",
            "resume_smoke_pass",
            "first_durable_checkpoint_pass",
            "continuous_progress_10m_pass",
            "positive_throughput_pass",
            "old_brute_exact_generation_alive_pass",
            "new_exact_worker_generation_bound_pass",
            "controller_status_allows_handover",
        )
    )
    return {
        "schema_version": HANDOVER_SCHEMA,
        "status": "ELIGIBLE" if eligible else "NOT_ELIGIBLE",
        "conditions": conditions,
        "old_brute_process": dict(contract["old_brute_process"]),
        "minimum_continuous_progress_seconds": HANDOVER_MIN_PROGRESS_SECONDS,
        "maximum_observation_age_seconds": (
            HANDOVER_MAX_OBSERVATION_AGE_SECONDS
        ),
        "maximum_new_route_eta_seconds": HANDOVER_MAX_ETA_SECONDS,
        "minimum_relative_speedup": HANDOVER_MIN_RELATIVE_SPEEDUP,
        "conservative_exact_work_rows": HANDOVER_CONSERVATIVE_WORK_ROWS,
        "eta_or_relative_speedup_gate_required": False,
        "progress_rows": progress_rows,
        "progress_delta_rows": progress_delta_rows,
        "continuous_progress_seconds": continuous_progress_seconds,
        "observation_age_seconds": observation_age_seconds,
        "last_change_age_seconds": last_change_age_seconds,
        "throughput_rows_per_second": throughput_rows_per_second,
        "conservative_remaining_rows": conservative_remaining_rows,
        "conservative_eta_seconds": conservative_eta_seconds,
        "checkpoint_snapshot": checkpoint,
        "resume_smoke_receipt": resume_smoke,
        "external_old_route_speedup_evidence_required": False,
        "external_old_route_speedup_evidence_accepted_here": False,
        "eligible_to_request_old_brute_stop": eligible,
        "old_route_signal_authorized_here": False,
        "old_route_signal_sent": False,
        "errors": errors,
        "observed_at": _utc_now(),
    }


def controller_status(manifest_path: str | Path) -> dict[str, Any]:
    path = _require_absolute(manifest_path, label="controller manifest", existing="file")
    manifest = _read_json(path, label="controller manifest")
    validate_controller_manifest(manifest)
    if str(path) != manifest.get("controller_manifest_path"):
        raise RecoveryControllerError("controller manifest was copied to an unbound path")
    manifest = dict(manifest)
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = sha256_file(path)
    root = Path(manifest["controller_root"])
    result: dict[str, Any] = {
        "controller_id": CONTROLLER_ID,
        "controller_root": str(root),
        "release_ready": manifest["release_ready"],
        "missing_release_pins": manifest["missing_release_pins"],
        "production_deployment_authorized": manifest[
            "production_deployment_authorized"
        ],
        "root_exists": root.is_dir(),
        "status": "NOT_STARTED",
        "stages": {stage: "PENDING" for stage in STAGE_ORDER},
        "controller_process_alive": False,
        "scientific_worker_alive": False,
        "scientific_progress_state": "NOT_STARTED",
        "route_viability": "NOT_STARTED",
    }
    if not root.is_dir():
        result["old_brute_handover"] = _old_brute_handover_status(
            manifest=manifest,
            result=result,
            state=None,
        )
        return result
    pending_publications = sorted(
        str(candidate)
        for pattern in ("*.publish.tmp", "*.replace.tmp")
        for candidate in root.rglob(pattern)
    )
    if pending_publications:
        result["filesystem_reconciliation_required"] = pending_publications
    state_path = root / "state.json"
    state: Mapping[str, Any] | None = None
    if state_path.is_file():
        state = _load_state(root, manifest)
        result["status"] = state["status"]
        result["current_stage"] = state["current_stage"]
        result["stages"] = state["stages"]
        worker = state.get("worker")
        if isinstance(worker, Mapping):
            proc_root = Path(str(manifest["resources"].get("proc_root", "/proc")))
            result["worker"] = {
                **dict(worker),
                "alive": _pid_alive(
                    int(worker["pid"]), int(worker["start_ticks"]), proc_root=proc_root
                ),
            }
        controller_process = state.get("controller_process")
        if isinstance(controller_process, Mapping):
            ticks = controller_process.get("start_ticks")
            controller_alive = (
                _pid_alive(
                    int(controller_process["pid"]),
                    int(ticks),
                    proc_root=Path(
                        str(manifest["resources"].get("proc_root", "/proc"))
                    ),
                )
                if isinstance(ticks, int)
                else None
            )
            result["controller_process"] = {
                **dict(controller_process),
                "alive": controller_alive,
            }
    for stage_id in STAGE_ORDER:
        if _gate_path(manifest, stage_id).is_file():
            try:
                _open_gate(manifest, stage_id)
            except Exception as exc:
                result["stages"][stage_id] = (
                    "RECONCILIATION_REQUIRED"
                    if "RECONCILIATION_REQUIRED" in str(exc)
                    else "TAMPERED"
                )
                result.setdefault("gate_errors", {})[stage_id] = str(exc)
            else:
                result["stages"][stage_id] = "PASS"
    worker_status = result.get("worker")
    result["scientific_worker_alive"] = (
        bool(worker_status.get("alive"))
        if isinstance(worker_status, Mapping)
        else False
    )
    controller_process = result.get("controller_process")
    if isinstance(controller_process, Mapping):
        result["controller_process_alive"] = controller_process.get("alive")
    if (root / "terminal.json").is_file():
        if pending_publications:
            result["status"] = "TERMINAL_RECONCILIATION_REQUIRED"
            result["scientific_progress_state"] = "BLOCKED_RECOVERABLE"
            result["route_viability"] = "BLOCKED_RECOVERABLE"
        else:
            validate_controller_terminal(manifest, require_pass_marker=False)
            if (root / "PASS").is_file():
                validate_controller_terminal(manifest)
                result["status"] = "PASS"
                result["scientific_progress_state"] = "PASS"
                result["route_viability"] = "PASS"
            else:
                # The controller can safely reconstruct PASS-last only after it
                # reopens every typed stage terminal.  Status must remain useful
                # during that recoverable crash window without calling it PASS.
                result["status"] = "TERMINAL_RECONCILIATION_REQUIRED"
                result["scientific_progress_state"] = "BLOCKED_RECOVERABLE"
                result["route_viability"] = "BLOCKED_RECOVERABLE"
    elif result.get("status") == "BLOCKED":
        result["scientific_progress_state"] = "BLOCKED"
        result["route_viability"] = "BLOCKED"
    elif result["scientific_worker_alive"]:
        worker = result.get("worker")
        current_stage = result.get("current_stage")
        measurable_progress = False
        progress_stalled = False
        if isinstance(worker, Mapping) and current_stage == EXACT_STAGE:
            progress = _progress_value(_stage(manifest, EXACT_STAGE))
            start_progress = int(worker.get("start_progress", 0))
            result["scientific_progress_rows"] = progress
            result["scientific_progress_delta"] = (
                None if progress is None else int(progress) - start_progress
            )
            monitor = state.get("exact_progress_monitor") if state is not None else None
            if isinstance(monitor, Mapping) and progress is not None:
                monitored_progress = int(monitor.get("progress", -1))
                changed_epoch = float(monitor.get("last_change_epoch", 0.0))
                now_epoch = time.time()
                age_seconds = max(0.0, now_epoch - changed_epoch)
                timeout_seconds = float(
                    manifest["resources"]["coexistence_probe"]["timeout_seconds"]
                )
                result["scientific_progress_monitor_rows"] = monitored_progress
                result["scientific_progress_age_seconds"] = age_seconds
                if int(progress) < monitored_progress:
                    progress_stalled = True
                    result["scientific_progress_error"] = (
                        "exact checkpoint progress regressed below persisted monitor"
                    )
                elif int(progress) > monitored_progress:
                    checkpoint_path = Path(
                        str(_stage(manifest, EXACT_STAGE)["progress_checkpoint_path"])
                    )
                    checkpoint_stat = checkpoint_path.lstat()
                    if (
                        checkpoint_path.is_symlink()
                        or not stat.S_ISREG(checkpoint_stat.st_mode)
                    ):
                        progress_stalled = True
                        result["scientific_progress_error"] = (
                            "exact progress checkpoint is not a physical regular file"
                        )
                    else:
                        checkpoint_age = max(
                            0.0, now_epoch - float(checkpoint_stat.st_mtime)
                        )
                        result["scientific_checkpoint_age_seconds"] = checkpoint_age
                        if checkpoint_age <= timeout_seconds:
                            measurable_progress = True
                            result["scientific_progress_observed_after_state"] = True
                        else:
                            progress_stalled = True
                elif int(progress) > start_progress and age_seconds <= timeout_seconds:
                    measurable_progress = True
                elif age_seconds > timeout_seconds:
                    progress_stalled = True
            else:
                measurable_progress = (
                    progress is not None and int(progress) > start_progress
                )
        if progress_stalled:
            result["scientific_progress_state"] = "RUNNING_STALLED"
        elif measurable_progress:
            result["scientific_progress_state"] = "RUNNING_PROGRESSING"
        else:
            result["scientific_progress_state"] = "RUNNING_SLOW"
        probe = root / "coexistence_probe.json"
        if probe.is_file():
            _validate_coexistence_receipt(manifest, probe)
            result["coexistence_probe_status"] = "PASS"
        # Resource viability and scientific liveness are separate fields, but
        # an exact worker that has exceeded the frozen no-progress timeout is
        # not reported as viable merely because an older probe once passed.
        result["route_viability"] = result["scientific_progress_state"]
    elif isinstance(worker_status, Mapping):
        result["scientific_progress_state"] = "RUNNING_STALLED"
        result["route_viability"] = "RUNNING_STALLED"
    elif result.get("controller_process_alive") is True:
        result["scientific_progress_state"] = "RUNNING_SLOW"
        result["route_viability"] = "RUNNING_SLOW"
    elif result.get("status") == "NOT_STARTED":
        result["scientific_progress_state"] = "NOT_STARTED"
        result["route_viability"] = "NOT_STARTED"
    else:
        result["scientific_progress_state"] = "RUNNING_STALLED"
        result["route_viability"] = "RUNNING_STALLED"
    if pending_publications and not result["scientific_worker_alive"]:
        result["status"] = "FILESYSTEM_RECONCILIATION_REQUIRED"
        result["scientific_progress_state"] = "BLOCKED_RECOVERABLE"
        result["route_viability"] = "BLOCKED_RECOVERABLE"
    result["old_brute_handover"] = _old_brute_handover_status(
        manifest=manifest,
        result=result,
        state=state,
    )
    return result


__all__ = [
    "ADOPTION_STAGE",
    "CONTROLLER_ID",
    "DOWNSTREAM_STAGE",
    "DEFAULT_SUBSET_MAX_RSS_BYTES",
    "DEFAULT_SUBSET_RSS_WORKING_MARGIN_BYTES",
    "EXACT_STAGE",
    "FINAL_STAGE",
    "MANIFEST_SCHEMA",
    "RecoveryControllerError",
    "SCIENCE_RELEASE_COMMIT",
    "SPEC_SCHEMA",
    "SUBSET_MAX_RSS_SCOPE",
    "SUBSET_RSS_BUDGET_SEMANTICS",
    "STAGE_GATE_SCHEMA",
    "STAGE_ORDER",
    "SUBSET_STAGE",
    "build_controller_manifest",
    "build_controller_payload",
    "controller_status",
    "derive_output_budget",
    "run_controller",
    "sha256_file",
    "stable_json_sha256",
    "validate_controller_manifest",
    "validate_controller_terminal",
    "validate_ordinary_pass_dependency",
    "validate_stage_terminal",
    "validate_typed_adoption_receipt",
    "load_bound_controller_manifest",
    "open_typed_recovery_gate",
    "prepare_controller_launch",
]
