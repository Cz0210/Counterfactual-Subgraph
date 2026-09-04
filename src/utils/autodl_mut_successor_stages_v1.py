"""Dataset-specific terminal stages for the Mut final16 successor.

The generic post-A/B executor deliberately knows nothing about the large Mut
artifact schemas.  These small adapters give it three explicit operations:

* reopen the completed trace-on-adoption standardization and seal an export
  receipt without recomputing any scientific value;
* publish that same terminal through the unique matrix authority and then
  write the canonical cell locator;
* fail closed on Route B until a real generation-to-DBSCAN continuation
  exists.  In particular, the Route-B adapter never launches generation by
  itself and cannot turn a generation-only terminal into a completed cell.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping

from scripts.autodl.append_bace_gcf_matrix_authority import _git_identity
from src.eval.fast16_matrix_authority_pointer import append_under_authority_pointer
from src.eval.non_taste_matrix_append import (
    _validate_mut_fast_accurate_terminal,
    append_non_taste_matrix_cell,
)
from src.train.molecular_gnn_resume import atomic_rename_directory_noreplace
from src.utils.autodl_mut_first_divergence_v1 import file_sha256, stable_sha256
from src.utils.autodl_mut_route_b_v1 import validate_route_b_evidence
from src.utils.final16_owner_registry_v1 import (
    process_start_ticks,
    validate_owner_registry,
)


EXPORT_SCHEMA = "mut_successor_strict_export_terminal_v1"
PUBLISH_SCHEMA = "mut_successor_canonical_publish_terminal_v1"
ROUTE_B_BLOCKED_SCHEMA = "mut_route_b_closeout_blocked_v1"
LOCATOR_SCHEMA = "fast16_matrix_cell_root_locator_v1"
MUT_CELL_ID = "Mutagenicity/ComRecGC"
REQUIRED_EXPORTS = (
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "table2_comrecgc_k10.csv",
    "summary.json",
    "run_manifest.json",
    "final_artifact_audit.json",
    "freeze_manifest.json",
    "_FINALIZED.json",
)
ROUTE_B_MISSING_ADAPTERS = (
    "ROUTE_B_GENERATION_TERMINAL_TO_PAIR_STORE",
    "ROUTE_B_PAIR_STORE_TO_EXACT_DBSCAN",
    "ROUTE_B_DBSCAN_TO_STANDARDIZED_EVALUATION",
    "ROUTE_B_STANDARDIZED_TERMINAL_TO_CANONICAL_PUBLICATION",
)


class MutSuccessorStageError(RuntimeError):
    """A Mut successor stage is incomplete, stale, or ambiguously owned."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(
    value: str | os.PathLike[str], *, label: str, must_exist: bool = False
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise MutSuccessorStageError(
            f"{label} must be an absolute non-symlink path"
        )
    try:
        return path.resolve(strict=must_exist)
    except OSError as exc:
        raise MutSuccessorStageError(f"{label} is absent: {path}") from exc


def _physical_file(value: str | os.PathLike[str], *, label: str) -> Path:
    path = _absolute(value, label=label, must_exist=True)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise MutSuccessorStageError(f"{label} is not a physical nonempty file: {path}")
    return path


def _physical_directory(value: str | os.PathLike[str], *, label: str) -> Path:
    path = _absolute(value, label=label, must_exist=True)
    if path.is_symlink() or not path.is_dir():
        raise MutSuccessorStageError(f"{label} is not a physical directory: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    physical = _physical_file(path, label=label)
    try:
        value = json.loads(physical.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutSuccessorStageError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MutSuccessorStageError(f"{label} must contain one JSON object")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not replace):
        raise MutSuccessorStageError(f"refusing to replace output: {path}")
    encoded = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise MutSuccessorStageError(
                    f"refusing to replace output: {path}"
                ) from exc
            temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes(path: Path, value: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise MutSuccessorStageError(f"refusing to replace output: {path}")
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _seal_root(
    destination: Path,
    *,
    terminal: Mapping[str, Any],
    marker: str,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise MutSuccessorStageError(f"stage output root must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.staging-{secrets.token_hex(16)}"
    )
    staging.mkdir(mode=0o700)
    try:
        _atomic_json(staging / "terminal.json", terminal, replace=False)
        _write_bytes(staging / marker, (marker + "\n").encode("ascii"))
        atomic_rename_directory_noreplace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    reopened = _read_json(destination / "terminal.json", label="sealed stage terminal")
    if reopened != dict(terminal):
        raise MutSuccessorStageError("sealed stage terminal changed on reopen")
    return reopened


def _file_identity(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    reopened = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity != reopened:
        raise MutSuccessorStageError(f"artifact changed while hashing: {path}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest}


def _export_identities(standardized: Path) -> dict[str, dict[str, Any]]:
    return {
        name: _file_identity(_physical_file(standardized / name, label=name))
        for name in REQUIRED_EXPORTS
    }


def validate_export_receipt(
    receipt_path: str | os.PathLike[str], *, terminal_root: Path | None = None
) -> dict[str, Any]:
    """Reopen the stage-A receipt and all exported bytes it names."""

    path = _physical_file(receipt_path, label="Mut export receipt")
    receipt = _read_json(path, label="Mut export receipt")
    observed = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("schema_version") != EXPORT_SCHEMA or receipt.get("status") != "PASS":
        raise MutSuccessorStageError("Mut export receipt terminal contract changed")
    if observed != stable_sha256(unsigned):
        raise MutSuccessorStageError("Mut export receipt self hash changed")
    source = _physical_directory(
        receipt.get("source_terminal_root", ""), label="Mut export source terminal"
    )
    if terminal_root is not None and source != terminal_root:
        raise MutSuccessorStageError("Mut export receipt binds another terminal root")
    standardized = _physical_directory(
        receipt.get("standardized_root", ""), label="Mut standardized output"
    )
    if standardized != source / "standardized":
        raise MutSuccessorStageError("Mut export standardized-root binding changed")
    expected = _export_identities(standardized)
    if receipt.get("exports") != expected:
        raise MutSuccessorStageError("Mut exported artifacts changed after sealing")
    if (
        receipt.get("scientific_metrics_recomputed") is not False
        or receipt.get("figure_table_recomputed") is not False
        or receipt.get("test_used_for_selection") is not False
    ):
        raise MutSuccessorStageError("Mut export receipt claims forbidden recomputation")
    return receipt


def reopen_completed_export(
    *,
    terminal_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    proc_root: str | os.PathLike[str] = "/proc",
    terminal_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Strictly reopen one completed Mut terminal and seal existing exports."""

    source = _physical_directory(terminal_root, label="Mut terminal root")
    destination = _absolute(output_root, label="Mut export stage output")
    proc = _absolute(proc_root, label="proc root", must_exist=True)
    validator = terminal_validator or _validate_mut_fast_accurate_terminal
    try:
        evidence = dict(
            validator(source, proc_root=proc, require_writer_audit=True)
        )
    except Exception as exc:
        raise MutSuccessorStageError(f"strict Mut terminal reopen failed: {exc}") from exc
    if (
        evidence.get("terminal_kind") != "MUT_FAST_ACCURATE_STANDARDIZATION_FINAL"
        or _physical_directory(evidence.get("root", ""), label="reopened Mut root")
        != source
    ):
        raise MutSuccessorStageError("strict Mut terminal reopened another contract/root")
    standardized = _physical_directory(
        evidence.get("standardized", {}).get("root", ""),
        label="reopened Mut standardized root",
    )
    if standardized != source / "standardized":
        raise MutSuccessorStageError("strict Mut standardized output escaped its root")
    terminal: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA,
        "status": "PASS",
        "dataset": "Mutagenicity",
        "method": "ComRecGC",
        "source_terminal_root": str(source),
        "source_terminal_evidence": evidence,
        "source_terminal_evidence_sha256": stable_sha256(evidence),
        "standardized_root": str(standardized),
        "exports": _export_identities(standardized),
        "scientific_metrics_recomputed": False,
        "figure_table_recomputed": False,
        "test_used_for_selection": False,
        "sealed_at": _utc_now(),
    }
    terminal["receipt_sha256"] = stable_sha256(terminal)
    return _seal_root(destination, terminal=terminal, marker="PASS")


def _load_registry(path: Path) -> tuple[dict[str, Any], str]:
    physical = _physical_file(path, label="canonical final16 owner registry")
    before = physical.stat()
    raw_bytes = physical.read_bytes()
    after = physical.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise MutSuccessorStageError("canonical owner registry changed while reading")
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutSuccessorStageError("canonical owner registry is invalid JSON") from exc
    if not isinstance(raw, dict):
        raise MutSuccessorStageError("canonical owner registry must be one JSON object")
    try:
        value = validate_owner_registry(raw, check_processes=False)
    except Exception as exc:
        raise MutSuccessorStageError(f"canonical owner registry failed validation: {exc}") from exc
    return value, hashlib.sha256(raw_bytes).hexdigest()


def _publisher_row(
    registry: Mapping[str, Any],
    *,
    publisher_id: str,
    locator: Path,
    lease_path: Path,
    execution_commit: str,
    proc_root: Path,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in registry.get("publishers", [])
        if row.get("claim_enabled") is True and row.get("cell_id") == MUT_CELL_ID
    ]
    if len(rows) != 1:
        raise MutSuccessorStageError("Mut must have exactly one canonical publisher claim")
    row = rows[0]
    expected = {
        "publisher_id": publisher_id,
        "locator": str(locator),
        "lease_path": str(lease_path),
        "execution_commit": execution_commit,
    }
    changed = [key for key, value in expected.items() if row.get(key) != value]
    if changed:
        raise MutSuccessorStageError(
            "canonical Mut publisher binding changed: " + ", ".join(changed)
        )
    state = row.get("owner_state")
    if state not in {"PREDEPLOYED", "READY", "RUNNING", "ADOPTED_RUNNING"}:
        raise MutSuccessorStageError("canonical Mut publisher is not launchable")
    if state in {"PREDEPLOYED", "READY"}:
        if row.get("owner_pid") is not None or row.get("active_writer_count") != 0:
            raise MutSuccessorStageError("predeployed Mut publisher already claims a writer")
    else:
        if row.get("owner_pid") != os.getppid() or row.get("active_writer_count") != 1:
            raise MutSuccessorStageError(
                "running canonical publisher is not this wrapper's exact parent owner"
            )
        if process_start_ticks(proc_root, int(row["owner_pid"])) != row.get(
            "owner_start_ticks"
        ):
            raise MutSuccessorStageError("running canonical publisher PID was reused")
        heartbeat = _read_json(
            _physical_file(row.get("heartbeat", ""), label="publisher heartbeat"),
            label="publisher heartbeat",
        )
        heartbeat_pid = heartbeat.get("owner_pid", heartbeat.get("pid"))
        heartbeat_ticks = heartbeat.get(
            "owner_start_ticks", heartbeat.get("start_ticks")
        )
        if heartbeat_pid != row["owner_pid"] or (
            heartbeat_ticks is not None
            and heartbeat_ticks != row["owner_start_ticks"]
        ):
            raise MutSuccessorStageError("canonical publisher heartbeat binds another PID")
    return row


def _acquire_exact_lease(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MutSuccessorStageError("canonical publisher lease may not be a symlink")
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    opened = os.fstat(descriptor)
    named = path.lstat()
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        os.close(descriptor)
        raise MutSuccessorStageError("canonical publisher lease identity changed")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def publish_canonical_mut_cell(
    *,
    terminal_root: str | os.PathLike[str],
    export_receipt: str | os.PathLike[str],
    owner_registry: str | os.PathLike[str],
    publisher_id: str,
    publisher_locator: str | os.PathLike[str],
    publisher_lease_path: str | os.PathLike[str],
    matrix_authority_root: str | os.PathLike[str],
    matrix_output_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    proc_root: str | os.PathLike[str] = "/proc",
    git_identity: Mapping[str, str] | None = None,
    terminal_validator: Callable[..., Mapping[str, Any]] | None = None,
    append_cell: Callable[..., Mapping[str, Any]] | None = None,
    append_pointer: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Append Mut once under its canonical claim, then publish its locator."""

    source = _physical_directory(terminal_root, label="Mut terminal root")
    receipt_path = _physical_file(export_receipt, label="Mut export receipt")
    validate_export_receipt(receipt_path, terminal_root=source)
    registry_path = _physical_file(owner_registry, label="owner registry")
    locator = _absolute(publisher_locator, label="canonical Mut locator")
    lease_path = _absolute(publisher_lease_path, label="canonical Mut publisher lease")
    authority = _physical_directory(matrix_authority_root, label="matrix authority root")
    state_path = authority / "state.json"
    lock_path = authority / "publish.lock"
    matrix_output = _absolute(matrix_output_root, label="new matrix authority output")
    destination = _absolute(output_root, label="Mut publish stage output")
    proc = _absolute(proc_root, label="proc root", must_exist=True)
    for fresh, label in (
        (locator, "canonical Mut locator"),
        (matrix_output, "new matrix authority output"),
        (destination, "Mut publish stage output"),
    ):
        if fresh.exists() or fresh.is_symlink():
            raise MutSuccessorStageError(f"{label} must be fresh: {fresh}")
    identity = dict(git_identity or _git_identity())
    if set(identity) != {"commit", "tree"}:
        raise MutSuccessorStageError("execution Git identity is incomplete")
    initial_registry, _initial_registry_file_sha = _load_registry(registry_path)
    if Path(str(initial_registry.get("matrix_authority_root"))).resolve() != authority:
        raise MutSuccessorStageError("owner registry binds another matrix authority")
    _publisher_row(
        initial_registry,
        publisher_id=publisher_id,
        locator=locator,
        lease_path=lease_path,
        execution_commit=str(identity["commit"]),
        proc_root=proc,
    )
    lease = _acquire_exact_lease(lease_path)
    try:
        # Reopen after taking the canonical lease so a replaced registry cannot
        # silently transfer the claim between validation and publication.
        current_registry, current_registry_file_sha = _load_registry(registry_path)
        if Path(str(current_registry.get("matrix_authority_root"))).resolve() != authority:
            raise MutSuccessorStageError(
                "owner registry changed its matrix authority while taking the lease"
            )
        _publisher_row(
            current_registry,
            publisher_id=publisher_id,
            locator=locator,
            lease_path=lease_path,
            execution_commit=str(identity["commit"]),
            proc_root=proc,
        )
        validator = terminal_validator or _validate_mut_fast_accurate_terminal
        try:
            reopened = dict(
                validator(source, proc_root=proc, require_writer_audit=True)
            )
        except Exception as exc:
            raise MutSuccessorStageError(
                f"strict Mut terminal reopen before append failed: {exc}"
            ) from exc
        if reopened.get("terminal_kind") != "MUT_FAST_ACCURATE_STANDARDIZATION_FINAL":
            raise MutSuccessorStageError("Mut publish received the wrong terminal kind")

        cell_fn = append_cell or append_non_taste_matrix_cell
        pointer_fn = append_pointer or append_under_authority_pointer

        def _append(prior: Path) -> Mapping[str, Any]:
            return cell_fn(
                prior_authority_root=prior,
                dataset="Mutagenicity",
                method="ComRecGC",
                cell_terminal_root=source,
                output_root=matrix_output,
                proc_root=proc,
                require_writer_audit=True,
                git_identity=identity,
            )

        result = dict(
            pointer_fn(
                state_path=state_path,
                lock_path=lock_path,
                initial_authority_root=None,
                requested_cells=(MUT_CELL_ID,),
                append=_append,
            )
        )
        if (
            result.get("status") != "PASS"
            or result.get("appended_cell") != MUT_CELL_ID
            or Path(str(result.get("output_root") or "")).resolve() != matrix_output
        ):
            raise MutSuccessorStageError("matrix append did not publish exactly Mut/ComRecGC")
        locator_payload = {
            "schema_version": LOCATOR_SCHEMA,
            "status": "READY",
            "dataset": "Mutagenicity",
            "method": "ComRecGC",
            "terminal_root": str(source),
        }
        _atomic_json(locator, locator_payload, replace=False)
        if _read_json(locator, label="canonical Mut locator") != locator_payload:
            raise MutSuccessorStageError("canonical Mut locator changed on reopen")
        terminal: dict[str, Any] = {
            "schema_version": PUBLISH_SCHEMA,
            "status": "PASS",
            "dataset": "Mutagenicity",
            "method": "ComRecGC",
            "publisher_id": publisher_id,
            "owner_registry": str(registry_path),
            "owner_registry_sha256": current_registry_file_sha,
            "owner_registry_self_sha256": current_registry["self_sha256"],
            "export_receipt": str(receipt_path),
            "export_receipt_sha256": file_sha256(receipt_path),
            "source_terminal_root": str(source),
            "matrix_authority_root": str(authority),
            "matrix_output_root": str(matrix_output),
            "matrix_append": result,
            "publisher_locator": str(locator),
            "publisher_locator_sha256": file_sha256(locator),
            "scientific_metrics_recomputed": False,
            "test_used_for_selection": False,
            "published_at": _utc_now(),
        }
        terminal["receipt_sha256"] = stable_sha256(terminal)
        return _seal_root(destination, terminal=terminal, marker="PASS")
    finally:
        try:
            fcntl.flock(lease, fcntl.LOCK_UN)
        finally:
            os.close(lease)


def write_route_b_adapter_blocker(
    *,
    decision_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    check_files: bool = True,
) -> dict[str, Any]:
    """Seal a non-launching Route-B blocker until the closeout adapters exist."""

    decision_file = _physical_file(decision_path, label="consumed Route-B decision")
    decision = _read_json(decision_file, label="consumed Route-B decision")
    try:
        validate_route_b_evidence(decision, check_files=check_files)
    except Exception as exc:
        raise MutSuccessorStageError(f"Route-B decision failed validation: {exc}") from exc
    destination = _absolute(output_root, label="Route-B blocker output")
    terminal: dict[str, Any] = {
        "schema_version": ROUTE_B_BLOCKED_SCHEMA,
        "status": "BLOCKED_ADAPTER_MISSING",
        "dataset": "Mutagenicity",
        "method": "ComRecGC",
        "decision_path": str(decision_file),
        "decision_file_sha256": file_sha256(decision_file),
        "decision_content_sha256": decision.get("decision_sha256"),
        "classification": "SCIENTIFIC_STATE_DIVERGENCE",
        "route_b_selected": True,
        "route_b_started": False,
        "fresh_50k_started": False,
        "generation_started": False,
        "pair_store_recomputed": False,
        "dbscan_recomputed": False,
        "missing_adapters": list(ROUTE_B_MISSING_ADAPTERS),
        "reason": (
            "The sealed Route-B owner stops at generation; no reviewed entrypoint "
            "currently binds fresh generation to pair-store, exact DBSCAN, "
            "standardization, and canonical publication."
        ),
        "automatic_fallback_started": False,
        "written_at": _utc_now(),
    }
    terminal["receipt_sha256"] = stable_sha256(terminal)
    return _seal_root(
        destination, terminal=terminal, marker="BLOCKED_ADAPTER_MISSING"
    )


__all__ = [
    "EXPORT_SCHEMA",
    "LOCATOR_SCHEMA",
    "MUT_CELL_ID",
    "MutSuccessorStageError",
    "PUBLISH_SCHEMA",
    "REQUIRED_EXPORTS",
    "ROUTE_B_BLOCKED_SCHEMA",
    "ROUTE_B_MISSING_ADAPTERS",
    "publish_canonical_mut_cell",
    "reopen_completed_export",
    "validate_export_receipt",
    "write_route_b_adapter_blocker",
]
