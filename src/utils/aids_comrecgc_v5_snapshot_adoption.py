"""Hash-bound, read-only adoption of a completed AIDS v5 pair-store snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from src.baselines.comrecgc.external_memory_recourse import _file_stat_identity
from src.utils.aids_comrecgc_v5_snapshot import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_PARENT_COUNT,
    EXPECTED_ROWS,
    EXPECTED_VECTOR_DIM,
    PairStoreSnapshotError,
    validate_promoted_pair_store_snapshot,
)


ADOPTION_SCHEMA = "comrecgc_promoted_pair_store_snapshot_adoption_v1"
SOURCE_CONTROLLER_ID = (
    "four_methods_four_datasets_aids_comrecgc_exact_route_v5_pair_order_v1"
)
SOURCE_SNAPSHOT_TASK_ID = (
    "aids_comrecgc_pair_store_physical_snapshot_v5_pair_order_v1"
)


class SnapshotAdoptionError(RuntimeError):
    """Raised when the frozen snapshot owner or artifacts no longer close."""


def _sha256(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _physical_file(path: str | Path, *, label: str) -> Path:
    logical = Path(path).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise SnapshotAdoptionError(f"{label} must be an absolute physical file")
    try:
        resolved = logical.resolve(strict=True)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise SnapshotAdoptionError(f"{label} is unavailable") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SnapshotAdoptionError(f"{label} must be a nonempty regular file")
    return resolved


def _physical_dir(path: str | Path, *, label: str) -> Path:
    logical = Path(path).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise SnapshotAdoptionError(f"{label} must be an absolute physical directory")
    try:
        resolved = logical.resolve(strict=True)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise SnapshotAdoptionError(f"{label} is unavailable") from exc
    if not resolved.is_dir():
        raise SnapshotAdoptionError(f"{label} must be a directory")
    return resolved


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SnapshotAdoptionError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SnapshotAdoptionError(f"{label} must be one JSON object")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    if partial.is_symlink():
        raise SnapshotAdoptionError("snapshot adoption partial may not be a symlink")
    if partial.exists():
        if not partial.is_file():
            raise SnapshotAdoptionError("snapshot adoption partial is not regular")
        partial.unlink()
        _fsync_directory(path.parent)
    descriptor = os.open(
        partial,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(partial, path)
    _fsync_directory(path.parent)


def _publish_pass(path: Path) -> None:
    if path.is_symlink():
        raise SnapshotAdoptionError("snapshot adoption PASS may not be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != b"PASS\n":
            raise SnapshotAdoptionError("snapshot adoption PASS changed")
        return
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _validate_owner(
    *,
    owner_manifest_path: Path,
    owner_manifest_sha256: str,
    owner_task_gate_path: Path,
    owner_task_gate_sha256: str,
    snapshot_root: Path,
) -> dict[str, Any]:
    if _sha256(owner_manifest_path) != owner_manifest_sha256:
        raise SnapshotAdoptionError("snapshot owner manifest SHA256 mismatch")
    owner = load_controller_manifest(owner_manifest_path)
    if owner.controller_id != SOURCE_CONTROLLER_ID:
        raise SnapshotAdoptionError("snapshot owner controller identity changed")
    if SOURCE_SNAPSHOT_TASK_ID not in owner.by_id:
        raise SnapshotAdoptionError("snapshot owner task is absent")
    namespace_root = owner_manifest_path.parent.parent
    expected_manifest = (
        namespace_root / "manifests" / f"{SOURCE_CONTROLLER_ID}.json"
    ).resolve(strict=False)
    expected_gate = (
        namespace_root
        / SOURCE_CONTROLLER_ID
        / "tasks"
        / SOURCE_SNAPSHOT_TASK_ID
        / "gate.json"
    ).resolve(strict=False)
    if owner_manifest_path != expected_manifest or owner_task_gate_path != expected_gate:
        raise SnapshotAdoptionError("snapshot owner authority path changed")
    task = owner.by_id[SOURCE_SNAPSHOT_TASK_ID]
    expected_attempt0 = task.expected_output.replace("{attempt}", "0")
    if Path(expected_attempt0).resolve(strict=False) != snapshot_root:
        raise SnapshotAdoptionError("snapshot owner task output changed")
    if _sha256(owner_task_gate_path) != owner_task_gate_sha256:
        raise SnapshotAdoptionError("snapshot owner task gate SHA256 mismatch")
    gate = _read_object(owner_task_gate_path, label="snapshot owner task gate")
    runs = gate.get("runs")
    if (
        gate.get("status") != "PASS"
        or gate.get("task_id") != SOURCE_SNAPSHOT_TASK_ID
        or not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(runs[0], Mapping)
        or runs[0].get("state") != "PASS"
        or int(runs[0].get("attempt", -1)) != 0
        or Path(str(runs[0].get("expected_output") or "")).resolve(strict=False)
        != snapshot_root
    ):
        raise SnapshotAdoptionError("snapshot owner task is not exact PASS attempt-0")
    return {
        "controller_id": owner.controller_id,
        "manifest": str(owner_manifest_path),
        "manifest_sha256": owner_manifest_sha256,
        "task_id": SOURCE_SNAPSHOT_TASK_ID,
        "task_gate": str(owner_task_gate_path),
        "task_gate_sha256": owner_task_gate_sha256,
        "task_status": "PASS",
        "attempt": 0,
        "expected_output": str(snapshot_root),
    }


def _identity(
    *,
    owner_manifest_path: Path,
    owner_manifest_sha256: str,
    owner_task_gate_path: Path,
    owner_task_gate_sha256: str,
    snapshot_root: Path,
    snapshot_manifest_sha256: str,
    dbscan_contract_sha256: str,
    pair_store_manifest_sha256: str,
    pairs_sha256: str,
    vectors_sha256: str,
    source_root: Path,
    source_manifest_sha256: str,
    allowed_pid: int,
    allowed_start_ticks: int,
    allowed_cmdline_sha256: str,
    allowed_output_root: Path,
    allowed_project_root: Path,
    expected_row_count: int,
    expected_vector_dim: int,
    expected_parent_count: int,
    expected_candidate_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": ADOPTION_SCHEMA,
        "owner_controller_id": SOURCE_CONTROLLER_ID,
        "owner_task_id": SOURCE_SNAPSHOT_TASK_ID,
        "owner_manifest": str(owner_manifest_path),
        "owner_manifest_sha256": owner_manifest_sha256,
        "owner_task_gate": str(owner_task_gate_path),
        "owner_task_gate_sha256": owner_task_gate_sha256,
        "snapshot_root": str(snapshot_root),
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "dbscan_contract_sha256": dbscan_contract_sha256,
        "pair_store_manifest_sha256": pair_store_manifest_sha256,
        "pairs_sha256": pairs_sha256,
        "vectors_sha256": vectors_sha256,
        "source_root": str(source_root),
        "source_manifest_sha256": source_manifest_sha256,
        "allowed_old_generation": {
            "pid": int(allowed_pid),
            "start_ticks": int(allowed_start_ticks),
            "cmdline_sha256": allowed_cmdline_sha256,
            "output_root": str(allowed_output_root),
            "project_root": str(allowed_project_root),
        },
        "expected_row_count": int(expected_row_count),
        "expected_vector_dim": int(expected_vector_dim),
        "expected_parent_count": int(expected_parent_count),
        "expected_candidate_count": int(expected_candidate_count),
        "copy_or_hardlink_performed": False,
        "snapshot_read_only": True,
    }


def _reopen_snapshot(
    *, identity: Mapping[str, Any], proc_root: Path
) -> dict[str, Any]:
    old = identity["allowed_old_generation"]
    try:
        terminal = validate_promoted_pair_store_snapshot(
            source_root=identity["source_root"],
            expected_source_manifest_sha256=identity["source_manifest_sha256"],
            output_dir=identity["snapshot_root"],
            proc_root=proc_root,
            allowed_pid=int(old["pid"]),
            allowed_start_ticks=int(old["start_ticks"]),
            allowed_cmdline_sha256=str(old["cmdline_sha256"]),
            allowed_output_root=old["output_root"],
            allowed_project_root=old["project_root"],
            expected_row_count=int(identity["expected_row_count"]),
            expected_vector_dim=int(identity["expected_vector_dim"]),
            expected_parent_count=int(identity["expected_parent_count"]),
            expected_candidate_count=int(identity["expected_candidate_count"]),
            require_pass=True,
        )
    except PairStoreSnapshotError as exc:
        raise SnapshotAdoptionError("snapshot full closure validation failed") from exc
    root = Path(str(identity["snapshot_root"]))
    artifacts = {
        "snapshot_manifest.json": identity["snapshot_manifest_sha256"],
        "dbscan_contract.json": identity["dbscan_contract_sha256"],
        "pair_store/run_manifest.json": identity["pair_store_manifest_sha256"],
    }
    for relative, expected in artifacts.items():
        if _sha256(_physical_file(root / relative, label=relative)) != expected:
            raise SnapshotAdoptionError(f"adopted snapshot hash changed: {relative}")
    pair_root = root / "pair_store"
    pair_manifest = _read_object(
        _physical_file(pair_root / "run_manifest.json", label="snapshot pair manifest"),
        label="snapshot pair manifest",
    )
    if (
        pair_manifest.get("pairs_sha256") != identity["pairs_sha256"]
        or pair_manifest.get("vectors_sha256") != identity["vectors_sha256"]
        or terminal.get("pair_store_root") != str(pair_root)
    ):
        raise SnapshotAdoptionError("adopted snapshot array identity changed")
    return {
        "snapshot_manifest_sha256": artifacts["snapshot_manifest.json"],
        "dbscan_contract_sha256": artifacts["dbscan_contract.json"],
        "pair_store_manifest_sha256": artifacts["pair_store/run_manifest.json"],
        "pairs_sha256": identity["pairs_sha256"],
        "vectors_sha256": identity["vectors_sha256"],
        "pairs_stat": _file_stat_identity(pair_root / "pair_indices.npy"),
        "vectors_stat": _file_stat_identity(pair_root / "recourse_vectors.npy"),
        "destination_writable_reference_count": 0,
        "destination_partial_artifacts": [],
    }


def validate_snapshot_adoption(
    *,
    output_dir: str | Path,
    proc_root: str | Path,
    identity: Mapping[str, Any],
    require_pass: bool = True,
) -> dict[str, Any]:
    output = _physical_dir(output_dir, label="snapshot adoption output")
    proc = _physical_dir(proc_root, label="proc root")
    manifest_path = _physical_file(
        output / "snapshot_adoption_manifest.json",
        label="snapshot adoption manifest",
    )
    manifest = _read_object(manifest_path, label="snapshot adoption manifest")
    if (
        manifest.get("schema_version") != ADOPTION_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("identity") != identity
        or manifest.get("identity_sha256") != _stable_hash(identity)
    ):
        raise SnapshotAdoptionError("snapshot adoption terminal identity changed")
    owner = _validate_owner(
        owner_manifest_path=_physical_file(
            identity["owner_manifest"], label="snapshot owner manifest"
        ),
        owner_manifest_sha256=str(identity["owner_manifest_sha256"]),
        owner_task_gate_path=_physical_file(
            identity["owner_task_gate"], label="snapshot owner task gate"
        ),
        owner_task_gate_sha256=str(identity["owner_task_gate_sha256"]),
        snapshot_root=_physical_dir(identity["snapshot_root"], label="snapshot root"),
    )
    closure = _reopen_snapshot(identity=identity, proc_root=proc)
    if manifest.get("owner") != owner or manifest.get("snapshot_closure") != closure:
        raise SnapshotAdoptionError("snapshot adoption evidence changed")
    if require_pass:
        marker = _physical_file(output / "PASS", label="snapshot adoption PASS")
        if marker.read_bytes() != b"PASS\n":
            raise SnapshotAdoptionError("snapshot adoption PASS changed")
    return manifest


def create_snapshot_adoption(
    *,
    output_dir: str | Path,
    proc_root: str | Path,
    owner_manifest: str | Path,
    owner_manifest_sha256: str,
    owner_task_gate: str | Path,
    owner_task_gate_sha256: str,
    snapshot_root: str | Path,
    snapshot_manifest_sha256: str,
    dbscan_contract_sha256: str,
    pair_store_manifest_sha256: str,
    pairs_sha256: str,
    vectors_sha256: str,
    source_root: str | Path,
    source_manifest_sha256: str,
    allowed_pid: int,
    allowed_start_ticks: int,
    allowed_cmdline_sha256: str,
    allowed_output_root: str | Path,
    allowed_project_root: str | Path,
    expected_row_count: int = EXPECTED_ROWS,
    expected_vector_dim: int = EXPECTED_VECTOR_DIM,
    expected_parent_count: int = EXPECTED_PARENT_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    resume: bool = False,
) -> dict[str, Any]:
    if (
        int(expected_row_count) <= 0
        or int(expected_vector_dim) <= 0
        or int(expected_parent_count) <= 0
        or int(expected_candidate_count) <= 0
        or int(expected_row_count)
        != int(expected_parent_count) * int(expected_candidate_count)
    ):
        raise SnapshotAdoptionError("invalid snapshot adoption dimensions")
    output_logical = Path(output_dir).expanduser()
    if not output_logical.is_absolute() or output_logical.is_symlink():
        raise SnapshotAdoptionError("snapshot adoption output must be fresh physical")
    if output_logical.exists():
        if not resume:
            raise SnapshotAdoptionError("snapshot adoption output already exists")
        output = _physical_dir(output_logical, label="snapshot adoption output")
        terminal = output / "snapshot_adoption_manifest.json"
        if terminal.exists():
            raise SnapshotAdoptionError(
                "snapshot adoption is already terminal; use validate-only"
            )
        partial = output / ".snapshot_adoption_manifest.json.partial"
        if partial.exists() or partial.is_symlink():
            if partial.is_symlink() or not partial.is_file():
                raise SnapshotAdoptionError("snapshot adoption partial is unsafe")
            partial.unlink()
            _fsync_directory(output)
        if any(output.iterdir()):
            raise SnapshotAdoptionError("snapshot adoption resume root is not empty")
    else:
        output_logical.mkdir(parents=True, exist_ok=False)
        output = output_logical.resolve(strict=True)
        _fsync_directory(output.parent)
    proc = _physical_dir(proc_root, label="proc root")
    owner_manifest_path = _physical_file(owner_manifest, label="snapshot owner manifest")
    owner_gate_path = _physical_file(owner_task_gate, label="snapshot owner task gate")
    snapshot = _physical_dir(snapshot_root, label="snapshot root")
    source = _physical_dir(source_root, label="snapshot source root")
    old_output = _physical_dir(allowed_output_root, label="old output root")
    old_project = _physical_dir(allowed_project_root, label="old project root")
    identity = _identity(
        owner_manifest_path=owner_manifest_path,
        owner_manifest_sha256=owner_manifest_sha256,
        owner_task_gate_path=owner_gate_path,
        owner_task_gate_sha256=owner_task_gate_sha256,
        snapshot_root=snapshot,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        dbscan_contract_sha256=dbscan_contract_sha256,
        pair_store_manifest_sha256=pair_store_manifest_sha256,
        pairs_sha256=pairs_sha256,
        vectors_sha256=vectors_sha256,
        source_root=source,
        source_manifest_sha256=source_manifest_sha256,
        allowed_pid=allowed_pid,
        allowed_start_ticks=allowed_start_ticks,
        allowed_cmdline_sha256=allowed_cmdline_sha256,
        allowed_output_root=old_output,
        allowed_project_root=old_project,
        expected_row_count=expected_row_count,
        expected_vector_dim=expected_vector_dim,
        expected_parent_count=expected_parent_count,
        expected_candidate_count=expected_candidate_count,
    )
    owner = _validate_owner(
        owner_manifest_path=owner_manifest_path,
        owner_manifest_sha256=owner_manifest_sha256,
        owner_task_gate_path=owner_gate_path,
        owner_task_gate_sha256=owner_task_gate_sha256,
        snapshot_root=snapshot,
    )
    closure = _reopen_snapshot(identity=identity, proc_root=proc)
    terminal_payload = {
        "schema_version": ADOPTION_SCHEMA,
        "status": "PASS",
        "identity": identity,
        "identity_sha256": _stable_hash(identity),
        "owner": owner,
        "snapshot_closure": closure,
        "copy_or_hardlink_performed": False,
        "old_snapshot_mutated": False,
    }
    _atomic_json(output / "snapshot_adoption_manifest.json", terminal_payload)
    # PASS is published only after the terminal manifest is reopenable and its
    # complete owner/snapshot identity is present.  Full source/destination
    # hashes were closed immediately above; science performs the same full
    # validator again before consuming the snapshot.
    reopened = _read_object(
        _physical_file(
            output / "snapshot_adoption_manifest.json",
            label="snapshot adoption manifest",
        ),
        label="snapshot adoption manifest",
    )
    if reopened != terminal_payload:
        raise SnapshotAdoptionError("snapshot adoption terminal write changed")
    _publish_pass(output / "PASS")
    return validate_snapshot_adoption(
        output_dir=output,
        proc_root=proc,
        identity=identity,
        require_pass=True,
    )


__all__ = [
    "ADOPTION_SCHEMA",
    "SOURCE_CONTROLLER_ID",
    "SOURCE_SNAPSHOT_TASK_ID",
    "SnapshotAdoptionError",
    "create_snapshot_adoption",
    "validate_snapshot_adoption",
]
