"""Fail-closed reconciliation for the completed T12 250->500->510 diagnostic.

The accelerated branch is diagnostic evidence only.  In particular, a 510
checkpoint must never be treated as the 20k production terminal and this
module never writes into the accelerated science root.  It independently
reopens the sealed 500/510 checkpoints, records the known post-commit
engineering failure, and can retire the dead owner in the canonical registry.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping

from .final16_owner_registry_v1 import (
    Final16OwnerRegistryError,
    atomic_write_owner_registry,
    build_owner_registry,
    process_start_ticks,
    validate_owner_registry,
)
from .main_ready_task_specs import atomic_json, stable_sha256


SCHEMA = "tastemolnet_t12_diagnostic_510_reconciliation_v1"
TERMINAL_STATUS = "DIAGNOSTIC_510_RECONCILED_ENGINEERING_FAILURE"
ENGINEERING_ERROR = "T12 native candidates may be materialized only at 20k"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "stage",
    "payload_file",
    "payload_sha256",
    "payload_bytes",
    "checkpoint_cursor",
    "total_steps",
    "purpose",
    "attempt_id",
    "generation_token",
    "identity_sha256",
    "state_sha256",
    "rng_sha256",
    "written_at",
    "immutable_no_replace",
}


class T12DiagnosticReconcileError(RuntimeError):
    """The diagnostic evidence or owner retirement is ambiguous."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise T12DiagnosticReconcileError(f"{label} is not one physical file")
    try:
        resolved = path.resolve(strict=True)
        info = path.stat()
    except OSError as exc:
        raise T12DiagnosticReconcileError(f"{label} is unavailable") from exc
    if resolved != path or not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise T12DiagnosticReconcileError(f"{label} is not one physical file")
    return path


def _json(path: Path, *, label: str) -> dict[str, Any]:
    _physical_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T12DiagnosticReconcileError(f"{label} is invalid JSON") from exc
    if type(value) is not dict:
        raise T12DiagnosticReconcileError(f"{label} is not one JSON object")
    return value


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_no_live_writer(
    *, output_root: Path, expected_dead_pid: int, expected_start_ticks: int, proc_root: Path
) -> dict[str, Any]:
    """Prove the former owner is dead and no process can write the root."""

    root = output_root.resolve(strict=True)
    observed_ticks = process_start_ticks(proc_root, expected_dead_pid)
    if observed_ticks is not None:
        relation = "same" if observed_ticks == expected_start_ticks else "reused"
        raise T12DiagnosticReconcileError(
            f"former T12 owner PID is still present ({relation} process identity)"
        )
    scanned = 0
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or not process_dir.is_dir():
            continue
        scanned += 1
        try:
            cwd = (process_dir / "cwd").resolve(strict=True)
        except OSError:
            cwd = None
        if cwd is not None and _within(cwd, root):
            raise T12DiagnosticReconcileError(
                f"process {process_dir.name} has cwd inside diagnostic root"
            )
        fd_root = process_dir / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = descriptor.resolve(strict=True)
                flags_text = (process_dir / "fdinfo" / descriptor.name).read_text(
                    encoding="utf-8"
                )
            except OSError:
                continue
            flags_line = next(
                (line for line in flags_text.splitlines() if line.startswith("flags:")),
                None,
            )
            if flags_line is None:
                continue
            try:
                flags = int(flags_line.split(":", 1)[1].strip(), 8)
            except ValueError:
                continue
            writable = flags & os.O_ACCMODE in {os.O_WRONLY, os.O_RDWR}
            if writable and _within(target, root):
                raise T12DiagnosticReconcileError(
                    f"process {process_dir.name} has writable fd in diagnostic root"
                )
    return {
        "former_owner_pid": expected_dead_pid,
        "former_owner_start_ticks": expected_start_ticks,
        "former_owner_alive": False,
        "writable_open_fd_count": 0,
        "cwd_inside_root_count": 0,
        "processes_scanned": scanned,
        "output_root": str(root),
    }


def _validate_manifest(path: Path, *, cursor: int) -> dict[str, Any]:
    value = _json(path, label=f"checkpoint {cursor} manifest")
    payload_name = f"checkpoint-{cursor:08d}.pt"
    if (
        set(value) != _MANIFEST_KEYS
        or value.get("status") != "COMMITTED"
        or value.get("purpose") != "production"
        or value.get("checkpoint_cursor") != cursor
        or value.get("total_steps") != 510
        or value.get("payload_file") != payload_name
        or value.get("immutable_no_replace") is not True
        or type(value.get("payload_bytes")) is not int
        or value["payload_bytes"] <= 0
        or any(_SHA256.fullmatch(str(value.get(field) or "")) is None for field in (
            "payload_sha256", "identity_sha256", "state_sha256", "rng_sha256"
        ))
    ):
        raise T12DiagnosticReconcileError(
            f"checkpoint {cursor} is not a sealed diagnostic commit"
        )
    payload = _physical_file(path.parent / payload_name, label=f"checkpoint {cursor} payload")
    if payload.stat().st_size != value["payload_bytes"] or file_sha256(payload) != value[
        "payload_sha256"
    ]:
        raise T12DiagnosticReconcileError(f"checkpoint {cursor} payload bytes changed")
    return value


def _validate_generation_receipt(
    path: Path, *, cursor: int, allow_absent: bool
) -> dict[str, Any] | None:
    if not path.exists():
        if allow_absent:
            return None
        raise T12DiagnosticReconcileError(f"generation receipt {cursor} is absent")
    value = _json(path, label=f"generation receipt {cursor}")
    if (
        value.get("status") != "GENERATION_CHECKPOINT_COMMITTED"
        or value.get("checkpoint_cursor") != cursor
        or value.get("candidate_manifest") is not None
        or value.get("candidate_manifest_sha256") is not None
        or value.get("calibration_loaded") is not False
        or value.get("test_loaded") is not False
        or value.get("paper_cell_pass") is not False
    ):
        raise T12DiagnosticReconcileError(
            f"generation receipt {cursor} is not diagnostic-only"
        )
    if cursor == 510 and value.get("terminal_candidate_materialization_requested") is not False:
        raise T12DiagnosticReconcileError(
            "repaired 510 receipt does not explicitly disable candidate materialization"
        )
    return value


def reconcile_diagnostic_510(
    *,
    task_spec_path: Path,
    source_terminal_path: Path,
    segment_510_log_path: Path,
    overlay_root: Path,
    expected_owner_pid: int,
    expected_owner_start_ticks: int,
    proc_root: Path,
    checkpoint_reopener: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Reopen 500/510 and publish a fresh diagnostic-only terminal overlay."""

    if overlay_root.exists() or overlay_root.is_symlink():
        raise T12DiagnosticReconcileError("diagnostic overlay root must be fresh")
    spec = _json(task_spec_path, label="accelerated task spec")
    unsigned = {key: value for key, value in spec.items() if key != "spec_sha256"}
    if (
        spec.get("task_kind") != "T12_ACCELERATED_FROM_CHECKPOINT250"
        or spec.get("spec_sha256") != stable_sha256(unsigned)
        or type(spec.get("science_contract")) is not dict
    ):
        raise T12DiagnosticReconcileError("accelerated task spec binding changed")
    contract = spec["science_contract"]
    root = Path(str(spec.get("output_root")))
    if not root.is_absolute() or root.is_symlink() or root.resolve(strict=True) != root:
        raise T12DiagnosticReconcileError("accelerated output root is invalid")
    terminal = _json(source_terminal_path, label="accelerated owner terminal")
    if (
        terminal.get("task_id") != spec.get("task_id")
        or terminal.get("status") != "FAILED_AT_510"
        or terminal.get("exit_code") != 1
        or terminal.get("completed_step") != 500
        or terminal.get("owner_pid") != expected_owner_pid
        or terminal.get("owner_start_ticks") != expected_owner_start_ticks
        or terminal.get("output_root") != str(root)
        or terminal.get("gpu_lock_held") is not False
        or terminal.get("reference_signaled") is not False
    ):
        raise T12DiagnosticReconcileError("accelerated owner terminal changed")
    log = _physical_file(segment_510_log_path, label="segment 510 log")
    if ENGINEERING_ERROR not in log.read_text(encoding="utf-8", errors="strict"):
        raise T12DiagnosticReconcileError("segment 510 failure is not the known engineering error")
    writer_evidence = validate_no_live_writer(
        output_root=root,
        expected_dead_pid=expected_owner_pid,
        expected_start_ticks=expected_owner_start_ticks,
        proc_root=proc_root,
    )
    run_identity_path = _physical_file(root / "run_identity.json", label="run identity")
    run_identity = _json(run_identity_path, label="run identity")
    template = run_identity.get("identity_template")
    if type(template) is not dict:
        raise T12DiagnosticReconcileError("run identity template is absent")
    checkpoints: dict[str, Any] = {}
    for cursor in (500, 510):
        manifest_path = Path(str(contract.get(f"accelerated_checkpoint_{cursor}")))
        expected_path = root / "checkpoints" / f"checkpoint-{cursor:08d}.manifest.json"
        if manifest_path != expected_path:
            raise T12DiagnosticReconcileError(f"checkpoint {cursor} locator changed")
        manifest = _validate_manifest(manifest_path, cursor=cursor)
        identity = dict(template)
        identity.update({"purpose": "production", "total_steps": 510, "checkpoint_cursor": cursor})
        reopened = dict(
            checkpoint_reopener(
                manifest_path,
                expected_identity=identity,
            )
        )
        if (
            reopened.get("identity_sha256") != manifest["identity_sha256"]
            or reopened.get("state_sha256") != manifest["state_sha256"]
            or reopened.get("rng_sha256") != manifest["rng_sha256"]
        ):
            raise T12DiagnosticReconcileError(f"checkpoint {cursor} reload digest changed")
        receipt_path = root / f"generation_receipt_{cursor:08d}.json"
        receipt = _validate_generation_receipt(
            receipt_path, cursor=cursor, allow_absent=cursor == 510
        )
        checkpoints[str(cursor)] = {
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "payload": str(manifest_path.parent / manifest["payload_file"]),
            "payload_sha256": manifest["payload_sha256"],
            "identity_sha256": manifest["identity_sha256"],
            "state_sha256": manifest["state_sha256"],
            "rng_sha256": manifest["rng_sha256"],
            "independent_reload": "PASS",
            "generation_receipt": str(receipt_path) if receipt else None,
            "generation_receipt_sha256": file_sha256(receipt_path) if receipt else None,
        }
    receipt = {
        "schema_version": SCHEMA,
        "status": TERMINAL_STATUS,
        "task_id": spec["task_id"],
        "task_spec": str(task_spec_path),
        "task_spec_sha256": file_sha256(task_spec_path),
        "source_terminal": str(source_terminal_path),
        "source_terminal_sha256": file_sha256(source_terminal_path),
        "segment_510_log": str(log),
        "segment_510_log_sha256": file_sha256(log),
        "failure_class": "POST_CHECKPOINT_DIAGNOSTIC_MATERIALIZATION_BUG",
        "failure_message": ENGINEERING_ERROR,
        "checkpoints": checkpoints,
        "checkpoint_500_complete": True,
        "checkpoint_510_complete": True,
        "checkpoint_510_rerun": False,
        "native_candidates_materialized": False,
        "diagnostic_only": True,
        "diagnostic_total_steps": 510,
        "production_total_steps": 20_000,
        "promotion_allowed": False,
        "paper_cell_pass": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "reference_signaled": False,
        "writer_evidence": writer_evidence,
    }
    receipt["receipt_sha256"] = stable_sha256(receipt)
    overlay_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt_path = overlay_root / "reconciliation_receipt.json"
    atomic_json(receipt_path, receipt)
    terminal_overlay = {
        "schema_version": "tastemolnet_t12_diagnostic_terminal_overlay_v1",
        "status": TERMINAL_STATUS,
        "task_id": spec["task_id"],
        "source_terminal": str(source_terminal_path),
        "reconciliation_receipt": str(receipt_path),
        "reconciliation_receipt_sha256": file_sha256(receipt_path),
        "completed_step": 510,
        "science_rerun": False,
        "candidate_materialization_attempt_excluded": True,
        "native_candidates_materialized": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "paper_cell_pass": False,
    }
    terminal_overlay["terminal_sha256"] = stable_sha256(terminal_overlay)
    terminal_path = overlay_root / "diagnostic_terminal.json"
    atomic_json(terminal_path, terminal_overlay)
    checksums = overlay_root / "SHA256SUMS"
    checksums.write_text(
        f"{file_sha256(terminal_path)}  diagnostic_terminal.json\n"
        f"{file_sha256(receipt_path)}  reconciliation_receipt.json\n",
        encoding="ascii",
    )
    return {
        "status": TERMINAL_STATUS,
        "overlay_root": str(overlay_root),
        "terminal": str(terminal_path),
        "receipt": str(receipt_path),
        "checkpoint_510_rerun": False,
    }


def reconcile_registry_after_diagnostic(
    *,
    registry: Mapping[str, Any],
    expected_registry_sha256: str,
    task_id: str,
    expected_owner_pid: int,
    expected_owner_start_ticks: int,
    diagnostic_terminal: Mapping[str, Any],
    proc_root: Path,
) -> dict[str, Any]:
    """Retire exactly one dead owner and release only its stale GPU lease."""

    if registry.get("self_sha256") != expected_registry_sha256:
        raise T12DiagnosticReconcileError("canonical owner registry changed")
    try:
        validated = validate_owner_registry(registry, proc_root=proc_root, check_processes=False)
    except Final16OwnerRegistryError as exc:
        raise T12DiagnosticReconcileError("canonical owner registry is invalid") from exc
    if (
        diagnostic_terminal.get("status") != TERMINAL_STATUS
        or diagnostic_terminal.get("task_id") != task_id
        or diagnostic_terminal.get("diagnostic_only") is not True
        or diagnostic_terminal.get("native_candidates_materialized") is not False
        or diagnostic_terminal.get("promotion_allowed") is not False
        or diagnostic_terminal.get("paper_cell_pass") is not False
        or diagnostic_terminal.get("terminal_sha256")
        != stable_sha256(
            {
                key: value
                for key, value in diagnostic_terminal.items()
                if key != "terminal_sha256"
            }
        )
    ):
        raise T12DiagnosticReconcileError("diagnostic terminal overlay is invalid")
    tasks = [dict(row) for row in validated["tasks"]]
    matches = [row for row in tasks if row["task_id"] == task_id]
    if len(matches) != 1:
        raise T12DiagnosticReconcileError("accelerated task is not unique in registry")
    task = matches[0]
    if (
        task.get("owner_state") not in {"ADOPTED_RUNNING", "RUNNING"}
        or task.get("owner_pid") != expected_owner_pid
        or task.get("owner_start_ticks") != expected_owner_start_ticks
    ):
        raise T12DiagnosticReconcileError("registry no longer binds the dead accelerated owner")
    validate_no_live_writer(
        output_root=Path(task["output_root"]),
        expected_dead_pid=expected_owner_pid,
        expected_start_ticks=expected_owner_start_ticks,
        proc_root=proc_root,
    )
    task.update(
        {
            "stage": TERMINAL_STATUS,
            "owner_state": "TERMINAL_FAILED_ENGINEERING",
            "owner_pid": None,
            "owner_start_ticks": None,
            "heartbeat": None,
            "gpu": None,
        }
    )
    leases = [dict(row) for row in validated["gpu_leases"]]
    lease_matches = [row for row in leases if row["task_id"] == task_id and row["state"] == "HELD"]
    if len(lease_matches) != 1:
        raise T12DiagnosticReconcileError("accelerated stale HELD lease is not unique")
    lease_row = lease_matches[0]
    lease_path = Path(lease_row["lease_path"])
    if lease_path.exists():
        if lease_path.is_symlink() or not lease_path.is_file():
            raise T12DiagnosticReconcileError("accelerated lease path is indirect")
        with lease_path.open("rb") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise T12DiagnosticReconcileError("accelerated GPU lease is still held") from exc
            finally:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    lease_row["state"] = "RELEASED"
    try:
        return build_owner_registry(
            registry_id=validated["registry_id"],
            matrix_authority_root=validated["matrix_authority_root"],
            tasks=tasks,
            publishers=validated["publishers"],
            gpu_leases=leases,
            proc_root=proc_root,
            check_processes=True,
        )
    except Final16OwnerRegistryError as exc:
        raise T12DiagnosticReconcileError("post-reconcile registry validation failed") from exc


def reconcile_registry_file_after_diagnostic(
    *,
    registry_path: Path,
    expected_registry_file_sha256: str,
    expected_registry_sha256: str,
    task_id: str,
    expected_owner_pid: int,
    expected_owner_start_ticks: int,
    diagnostic_terminal: Mapping[str, Any],
    proc_root: Path,
) -> dict[str, Any]:
    """CAS-update ``current.json`` while holding the matrix publication lock.

    Locking a replaceable JSON inode is insufficient: another process could
    lock the old inode while this function atomically installs a new one.  The
    stable matrix ``publish.lock`` is therefore shared with cell publishers.
    Both the file hash and embedded registry hash are reopened inside that
    lock before the replacement is built and installed.
    """

    registry_file = _physical_file(registry_path, label="canonical owner registry")
    if _SHA256.fullmatch(expected_registry_file_sha256) is None:
        raise T12DiagnosticReconcileError("expected registry file SHA is invalid")
    if _SHA256.fullmatch(expected_registry_sha256) is None:
        raise T12DiagnosticReconcileError("expected registry self SHA is invalid")
    # The initial read is used only to locate the stable publication lock.  It
    # must already match both caller-provided digests so a raced/forged file
    # cannot make this process create or lock an unrelated path.
    initial_raw = registry_file.read_bytes()
    if hashlib.sha256(initial_raw).hexdigest() != expected_registry_file_sha256:
        raise T12DiagnosticReconcileError("canonical registry file changed before lock")
    try:
        initial = json.loads(initial_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T12DiagnosticReconcileError("canonical registry is invalid JSON") from exc
    if type(initial) is not dict or initial.get("self_sha256") != expected_registry_sha256:
        raise T12DiagnosticReconcileError("canonical registry self hash changed before lock")
    matrix_root = Path(str(initial.get("matrix_authority_root") or ""))
    if (
        not matrix_root.is_absolute()
        or matrix_root.is_symlink()
        or matrix_root.resolve(strict=True) != matrix_root
        or not matrix_root.is_dir()
    ):
        raise T12DiagnosticReconcileError("registry matrix authority is invalid")
    lock_path = matrix_root / "publish.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise T12DiagnosticReconcileError("matrix publication lock is indirect")
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        opened = os.fstat(descriptor)
        named = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise T12DiagnosticReconcileError("matrix publication lock identity changed")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        raw = registry_file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_registry_file_sha256:
            raise T12DiagnosticReconcileError("canonical registry file changed before CAS")
        try:
            current = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise T12DiagnosticReconcileError("canonical registry is invalid JSON") from exc
        if type(current) is not dict or current.get("self_sha256") != expected_registry_sha256:
            raise T12DiagnosticReconcileError("canonical registry self hash changed before CAS")
        if Path(str(current.get("matrix_authority_root"))) / "publish.lock" != lock_path:
            raise T12DiagnosticReconcileError("registry changed its publication lock")
        updated = reconcile_registry_after_diagnostic(
            registry=current,
            expected_registry_sha256=expected_registry_sha256,
            task_id=task_id,
            expected_owner_pid=expected_owner_pid,
            expected_owner_start_ticks=expected_owner_start_ticks,
            diagnostic_terminal=diagnostic_terminal,
            proc_root=proc_root,
        )
        atomic_write_owner_registry(registry_file, updated)
        installed = _json(registry_file, label="installed canonical owner registry")
        if installed != updated or installed.get("self_sha256") != updated["self_sha256"]:
            raise T12DiagnosticReconcileError("canonical registry CAS did not install exactly")
        return updated
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "ENGINEERING_ERROR",
    "SCHEMA",
    "TERMINAL_STATUS",
    "T12DiagnosticReconcileError",
    "file_sha256",
    "reconcile_diagnostic_510",
    "reconcile_registry_after_diagnostic",
    "reconcile_registry_file_after_diagnostic",
    "validate_no_live_writer",
]
