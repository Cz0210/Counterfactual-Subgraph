"""Crash-safe physical snapshot of a promoted ComRecGC pair store.

The old repair-v4 process keeps its vector memmap open read/write even though
its DBSCAN stage treats the array as an input.  A v5 consumer must not adopt
that inode directly.  This module copies the two terminal arrays into fresh
physical inodes, brackets the copy with full source hashes/stat identities,
and publishes a destination closure only after every byte and schema has been
revalidated.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from scripts.autodl.verify_aids_comrecgc_v5_process_set import verify_process_set
from src.baselines.comrecgc.external_memory_recourse import (
    PAIR_STORE_SCHEMA,
    _file_stat_identity,
    _find_writable_process_references,
    _validate_pair_store_manifest,
)


SNAPSHOT_SCHEMA = "comrecgc_promoted_pair_store_physical_snapshot_v1"
CHECKPOINT_SCHEMA = "comrecgc_promoted_pair_store_snapshot_checkpoint_v1"
DBSCAN_CONTRACT_SCHEMA = "aids_comrecgc_exact_dbscan_contract_v1"
COPY_BUFFER_BYTES = 16 * 1024 * 1024
MIN_FREE_AFTER_BYTES = 40 * 1024**3
EXPECTED_ROWS = 91_916_686
EXPECTED_PARENT_COUNT = 1_283
EXPECTED_CANDIDATE_COUNT = 71_642
EXPECTED_VECTOR_DIM = 64


class PairStoreSnapshotError(RuntimeError):
    """Raised when the physical snapshot cannot be proven exact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for chunk in iter(lambda: handle.read(COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PairStoreSnapshotError(f"physical nonempty JSON required: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairStoreSnapshotError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PairStoreSnapshotError(f"JSON object required: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _physical_file(path: Path, *, label: str) -> Path:
    logical = path.expanduser()
    if logical.is_symlink():
        raise PairStoreSnapshotError(f"{label} may not be a symlink")
    resolved = logical.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise PairStoreSnapshotError(f"{label} must be a physical nonempty file")
    return resolved


def _validate_array(
    path: Path, *, shape: tuple[int, ...], dtype: np.dtype[Any], label: str
) -> None:
    if path.is_symlink() or not path.is_file():
        raise PairStoreSnapshotError(f"{label} is not a physical file")
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise PairStoreSnapshotError(f"{label} is not a valid NPY array") from exc
    if value.shape != shape or value.dtype != dtype:
        raise PairStoreSnapshotError(
            f"{label} schema mismatch: {value.shape}/{value.dtype}"
        )


def _validate_cartesian_pair_order(
    path: Path,
    *,
    row_count: int,
    parent_count: int,
    candidate_count: int,
    block_rows: int = 262_144,
) -> dict[str, Any]:
    """Prove the stored row provenance is the frozen Cartesian order."""

    if row_count != parent_count * candidate_count:
        raise PairStoreSnapshotError("Cartesian pair-count identity mismatch")
    pairs = np.load(path, mmap_mode="r", allow_pickle=False)
    if pairs.shape != (row_count, 2) or pairs.dtype != np.dtype(np.int64):
        raise PairStoreSnapshotError("pair-index array schema mismatch")
    for offset in range(0, row_count, block_rows):
        stop = min(row_count, offset + block_rows)
        positions = np.arange(offset, stop, dtype=np.int64)
        block = pairs[offset:stop]
        if not np.array_equal(block[:, 0], positions // parent_count) or not np.array_equal(
            block[:, 1], positions % parent_count
        ):
            raise PairStoreSnapshotError(
                f"candidate-major/parent-minor pair order changed at rows {offset}:{stop}"
            )
    return {
        "status": "PASS",
        "row_count": row_count,
        "parent_count": parent_count,
        "candidate_count": candidate_count,
        "formula": "candidate_index=row//parent_count;parent_index=row%parent_count",
        "all_rows_checked": True,
        "pair_indices_are_row_provenance_not_adjacency": True,
    }


def _tree_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *names]:
            entry = current_path / name
            if entry.is_symlink():
                raise PairStoreSnapshotError(f"snapshot tree contains symlink: {entry}")
            if name.endswith(".partial") or ".partial." in name:
                raise PairStoreSnapshotError(f"snapshot tree contains partial: {entry}")
        files.extend(
            current_path / name
            for name in names
            if (current_path / name).is_file()
        )
    return sorted(files)


def _source_closure(
    *,
    source_root: Path,
    expected_manifest_sha256: str,
    proc_root: Path,
    allowed_pid: int,
    allowed_start_ticks: int,
    allowed_cmdline_sha256: str,
    allowed_output_root: Path,
    allowed_project_root: Path,
    expected_row_count: int,
    expected_vector_dim: int,
    expected_parent_count: int,
    expected_candidate_count: int,
    require_old_alive: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source_root.is_symlink() or not source_root.resolve(strict=True).is_dir():
        raise PairStoreSnapshotError("source pair store must be a physical directory")
    root = source_root.resolve(strict=True)
    for entry in root.iterdir():
        if entry.is_symlink():
            raise PairStoreSnapshotError(f"source pair store has symlink: {entry}")
        if entry.name.endswith(".partial") or ".partial." in entry.name:
            raise PairStoreSnapshotError(f"source pair store has partial: {entry}")
    manifest_path = _physical_file(root / "run_manifest.json", label="source manifest")
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise PairStoreSnapshotError("source manifest SHA256 mismatch")
    manifest = _load_object(manifest_path)
    if (
        manifest.get("schema_version") != PAIR_STORE_SCHEMA
        or manifest.get("run_complete") is not True
        or manifest.get("candidate_major_parent_minor_order") is not True
    ):
        raise PairStoreSnapshotError("source pair store is not terminal/order-frozen")
    pairs = _physical_file(Path(str(manifest["pairs_path"])), label="source pairs")
    vectors = _physical_file(
        Path(str(manifest["vectors_path"])), label="source vectors"
    )
    if pairs.parent != root or vectors.parent != root:
        raise PairStoreSnapshotError("source arrays escaped the terminal root")
    process = verify_process_set(
        proc_root=proc_root,
        allowed_pid=allowed_pid,
        allowed_start_ticks=allowed_start_ticks,
        allowed_cmdline_sha256=allowed_cmdline_sha256,
        allowed_output_root=allowed_output_root,
        allowed_project_root=allowed_project_root,
    )
    allowed_count = int(process.get("allowed_old_process_count", -1))
    if require_old_alive and allowed_count != 1:
        raise PairStoreSnapshotError("the exact old source generation must be alive")
    if allowed_count not in {0, 1}:
        raise PairStoreSnapshotError("invalid exact old source process count")
    authoritative = [manifest_path, pairs, vectors]
    writers = _find_writable_process_references(authoritative, proc_root=proc_root)
    if any(int(row.get("pid", -1)) != int(allowed_pid) for row in writers):
        raise PairStoreSnapshotError(
            "unexpected source writer: " + json.dumps(writers, sort_keys=True)
        )
    if (
        int(manifest.get("row_count", -1)) != int(expected_row_count)
        or int(manifest.get("vector_dim", -1)) != int(expected_vector_dim)
        or manifest.get("vectors_dtype") != "float32"
    ):
        raise PairStoreSnapshotError("source AIDS pair-store shape/dtype changed")
    _validate_array(
        pairs,
        shape=(expected_row_count, 2),
        dtype=np.dtype(np.int64),
        label="source pairs",
    )
    _validate_array(
        vectors,
        shape=(expected_row_count, expected_vector_dim),
        dtype=np.dtype(np.float32),
        label="source vectors",
    )
    pair_order = _validate_cartesian_pair_order(
        pairs,
        row_count=expected_row_count,
        parent_count=expected_parent_count,
        candidate_count=expected_candidate_count,
    )
    stats = {str(path): _file_stat_identity(path) for path in authoritative}
    hashes = {str(path): _sha256(path) for path in authoritative}
    if (
        hashes[str(manifest_path)] != expected_manifest_sha256
        or hashes[str(pairs)] != manifest.get("pairs_sha256")
        or hashes[str(vectors)] != manifest.get("vectors_sha256")
    ):
        raise PairStoreSnapshotError("source full-hash closure mismatch")
    return manifest, {
        "manifest_path": str(manifest_path),
        "root": str(root),
        "files": {path: {"stat": stats[path], "sha256": hashes[path]} for path in stats},
        "allowed_writer_generation": process,
        "writable_references": writers,
        "writable_reference_count": len(writers),
        "pair_order_proof": pair_order,
    }


def _validate_final_copy(
    *, source: Path, destination: Path, expected_sha256: str, expected_size: int
) -> dict[str, Any]:
    destination = _physical_file(destination, label="snapshot destination")
    if destination.stat().st_size != expected_size:
        raise PairStoreSnapshotError("snapshot destination size mismatch")
    if _sha256(destination) != expected_sha256:
        raise PairStoreSnapshotError("snapshot destination SHA256 mismatch")
    source_stat = _file_stat_identity(source)
    destination_stat = _file_stat_identity(destination)
    if (source_stat["device"], source_stat["inode"]) == (
        destination_stat["device"],
        destination_stat["inode"],
    ):
        raise PairStoreSnapshotError("snapshot destination is a hardlink/source inode")
    return {
        "source": str(source),
        "destination": str(destination),
        "size": expected_size,
        "sha256": expected_sha256,
        "source_stat": source_stat,
        "destination_stat": destination_stat,
        "physical_inode_distinct": True,
        "hardlinked": False,
    }


def _copy_one(
    *, source: Path, destination: Path, expected_sha256: str, expected_size: int
) -> dict[str, Any]:
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.exists() or destination.is_symlink():
        if partial.exists() or partial.is_symlink():
            raise PairStoreSnapshotError("partial exists beside promoted destination")
        return _validate_final_copy(
            source=source,
            destination=destination,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
    if partial.is_symlink():
        raise PairStoreSnapshotError("snapshot partial may not be a symlink")
    # A partial is never authority.  Same-root recovery restarts this one file
    # from byte zero, while already promoted/verified earlier files are kept.
    partial.unlink(missing_ok=True)
    source_descriptor = -1
    destination_descriptor = -1
    digest = hashlib.sha256()
    copied = 0
    try:
        source_descriptor = os.open(
            source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        destination_descriptor = os.open(
            partial,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            chunk = os.read(source_descriptor, COPY_BUFFER_BYTES)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise PairStoreSnapshotError("zero-length snapshot write")
                view = view[written:]
            digest.update(chunk)
            copied += len(chunk)
        # Linux production uses fdatasync; fsync is the portable durability
        # equivalent used by the macOS development fixtures.
        getattr(os, "fdatasync", os.fsync)(destination_descriptor)
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
    if copied != expected_size or digest.hexdigest() != expected_sha256:
        raise PairStoreSnapshotError("copied snapshot bytes differ from source authority")
    # Publish without clobbering a concurrently created destination.  The
    # temporary and final names briefly reference the newly copied inode (never
    # the source inode); unlinking the partial after the directory fsync leaves
    # exactly one physical destination name.  A crash in this two-name window
    # is reconciled below only when both names prove the same inode.
    os.link(partial, destination, follow_symlinks=False)
    _fsync_directory(destination.parent)
    partial.unlink()
    _fsync_directory(destination.parent)
    return _validate_final_copy(
        source=source,
        destination=destination,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )


def _discard_restartable_partial(*, destination: Path) -> bool:
    partial = destination.with_name(f".{destination.name}.partial")
    if not partial.exists() and not partial.is_symlink():
        return False
    if destination.exists() or destination.is_symlink():
        if (
            not destination.is_symlink()
            and destination.is_file()
            and not partial.is_symlink()
            and partial.is_file()
            and (destination.stat().st_dev, destination.stat().st_ino)
            == (partial.stat().st_dev, partial.stat().st_ino)
        ):
            partial.unlink()
            _fsync_directory(destination.parent)
            return True
        raise PairStoreSnapshotError("partial exists beside promoted destination")
    if partial.is_symlink() or not partial.is_file():
        raise PairStoreSnapshotError("snapshot partial must be a physical regular file")
    partial.unlink()
    _fsync_directory(destination.parent)
    return True


def _dbscan_contract(*, manifest: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": DBSCAN_CONTRACT_SCHEMA,
        "status": "PASS",
        "pair_store_manifest_sha256": manifest_sha256,
        "row_count": int(manifest["row_count"]),
        "vector_dim": int(manifest["vector_dim"]),
        "row_semantics": "candidate_by_parent_recourse_embedding_vector",
        "row_order": "candidate_major_parent_minor",
        "pair_indices_role": "row_provenance_only_not_adjacency_or_distance_edges",
        "precomputed_pairwise_distance_edges": False,
        "metric": "euclidean",
        "eps": 0.02,
        "min_samples": 3,
        "self_neighbor_included": True,
        "sklearn_version": "1.7.2",
        "fit_algorithm": "brute",
        "border_assignment": "sklearn_first_reachable_core_component_in_sample_order",
        "label_order": "sklearn_component_discovery_order_over_input_rows",
        "adaptive_anchor_certificate": {
            "mode": "all_core_one_component_adaptive_anchor_v1",
            "status": "REQUIRED_DOWNSTREAM_NOT_PRECOMPUTED_BY_SNAPSHOT",
            "exactness": (
                "a sufficient lower-bound certificate proves every row core and "
                "the anchor epsilon graph connected; it never treats pair_indices "
                "as an adjacency graph"
            ),
            "fallback_max_samples": 0,
            "failure_cap": 4096,
            "seed_count": 3,
        },
    }


def _publish_pass(path: Path) -> None:
    if path.is_symlink():
        raise PairStoreSnapshotError("snapshot PASS may not be a symlink")
    if path.exists():
        if path.read_bytes() != b"PASS\n":
            raise PairStoreSnapshotError("snapshot PASS marker changed")
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


def validate_promoted_pair_store_snapshot(
    *,
    source_root: str | Path,
    expected_source_manifest_sha256: str,
    output_dir: str | Path,
    proc_root: str | Path,
    allowed_pid: int,
    allowed_start_ticks: int,
    allowed_cmdline_sha256: str,
    allowed_output_root: str | Path,
    allowed_project_root: str | Path,
    expected_row_count: int = EXPECTED_ROWS,
    expected_vector_dim: int = EXPECTED_VECTOR_DIM,
    expected_parent_count: int = EXPECTED_PARENT_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    require_pass: bool = True,
) -> dict[str, Any]:
    """Reopen and prove the entire terminal snapshot closure."""

    source_logical = Path(source_root).expanduser()
    output_logical = Path(output_dir).expanduser()
    if source_logical.is_symlink() or output_logical.is_symlink():
        raise PairStoreSnapshotError("snapshot source/output may not be a symlink")
    source = source_logical.resolve(strict=True)
    output = output_logical.resolve(strict=True)
    terminal_path = _physical_file(
        output / "snapshot_manifest.json", label="snapshot terminal manifest"
    )
    terminal = _load_object(terminal_path)
    if terminal.get("schema_version") != SNAPSHOT_SCHEMA or terminal.get("status") != "PASS":
        raise PairStoreSnapshotError("snapshot terminal schema/status mismatch")
    if require_pass:
        pass_path = _physical_file(output / "PASS", label="snapshot PASS")
        if pass_path.read_bytes() != b"PASS\n":
            raise PairStoreSnapshotError("snapshot PASS marker changed")

    source_manifest, current_source = _source_closure(
        source_root=source,
        expected_manifest_sha256=expected_source_manifest_sha256,
        proc_root=Path(proc_root),
        allowed_pid=allowed_pid,
        allowed_start_ticks=allowed_start_ticks,
        allowed_cmdline_sha256=allowed_cmdline_sha256,
        allowed_output_root=Path(allowed_output_root),
        allowed_project_root=Path(allowed_project_root),
        expected_row_count=expected_row_count,
        expected_vector_dim=expected_vector_dim,
        expected_parent_count=expected_parent_count,
        expected_candidate_count=expected_candidate_count,
        require_old_alive=False,
    )
    for recorded_key in ("source", "source_post"):
        recorded = terminal.get(recorded_key)
        if not isinstance(recorded, Mapping) or recorded.get("files") != current_source["files"]:
            raise PairStoreSnapshotError("snapshot source content/stat closure drift")

    pair_store = output / "pair_store"
    destination_manifest_path = _physical_file(
        pair_store / "run_manifest.json", label="snapshot pair manifest"
    )
    destination_manifest = _load_object(destination_manifest_path)
    try:
        _validate_pair_store_manifest(destination_manifest_path, destination_manifest)
    except Exception as exc:
        raise PairStoreSnapshotError("snapshot pair-store closure invalid") from exc
    destination_manifest_sha = _sha256(destination_manifest_path)
    if (
        destination_manifest.get("physical_snapshot") is not True
        or destination_manifest.get("physical_snapshot_schema") != SNAPSHOT_SCHEMA
        or destination_manifest.get("source_manifest_sha256")
        != expected_source_manifest_sha256
        or int(destination_manifest.get("row_count", -1)) != expected_row_count
        or int(destination_manifest.get("vector_dim", -1)) != expected_vector_dim
        or terminal.get("pair_store_manifest_sha256") != destination_manifest_sha
    ):
        raise PairStoreSnapshotError("snapshot destination manifest identity drift")
    source_pairs = Path(str(source_manifest["pairs_path"])).resolve(strict=True)
    source_vectors = Path(str(source_manifest["vectors_path"])).resolve(strict=True)
    expected_copies = [
        _validate_final_copy(
            source=source_pairs,
            destination=pair_store / "pair_indices.npy",
            expected_sha256=str(source_manifest["pairs_sha256"]),
            expected_size=source_pairs.stat().st_size,
        ),
        _validate_final_copy(
            source=source_vectors,
            destination=pair_store / "recourse_vectors.npy",
            expected_sha256=str(source_manifest["vectors_sha256"]),
            expected_size=source_vectors.stat().st_size,
        ),
    ]
    if terminal.get("copy_records") != expected_copies:
        raise PairStoreSnapshotError("snapshot copy-record closure drift")

    dbscan_path = _physical_file(
        output / "dbscan_contract.json", label="snapshot DBSCAN contract"
    )
    expected_dbscan = _dbscan_contract(
        manifest=destination_manifest,
        manifest_sha256=destination_manifest_sha,
    )
    if _load_object(dbscan_path) != expected_dbscan:
        raise PairStoreSnapshotError("snapshot DBSCAN contract changed")
    if terminal.get("dbscan_contract_sha256") != _sha256(dbscan_path):
        raise PairStoreSnapshotError("snapshot DBSCAN contract hash changed")

    files = _tree_files(output)
    writers = _find_writable_process_references(files, proc_root=Path(proc_root))
    if writers:
        raise PairStoreSnapshotError(
            "snapshot destination has a live writer: " + json.dumps(writers, sort_keys=True)
        )
    if (
        terminal.get("source_mutated") is not False
        or terminal.get("hardlinked") is not False
        or terminal.get("destination_writable_reference_count") != 0
        or terminal.get("destination_partial_artifacts") != []
        or terminal.get("expected_row_count") != expected_row_count
        or terminal.get("expected_vector_dim") != expected_vector_dim
    ):
        raise PairStoreSnapshotError("snapshot terminal safety contract changed")
    return terminal


def create_promoted_pair_store_snapshot(
    *,
    source_root: str | Path,
    expected_source_manifest_sha256: str,
    output_dir: str | Path,
    proc_root: str | Path,
    allowed_pid: int,
    allowed_start_ticks: int,
    allowed_cmdline_sha256: str,
    allowed_output_root: str | Path,
    allowed_project_root: str | Path,
    min_free_after_bytes: int = MIN_FREE_AFTER_BYTES,
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
        raise PairStoreSnapshotError("invalid frozen Cartesian snapshot dimensions")
    source_logical = Path(source_root).expanduser()
    proc_logical = Path(proc_root).expanduser()
    output_logical = Path(output_dir).expanduser()
    if source_logical.is_symlink() or proc_logical.is_symlink():
        raise PairStoreSnapshotError("snapshot source/proc root may not be a symlink")
    if output_logical.is_symlink():
        raise PairStoreSnapshotError("snapshot output may not be a symlink")
    source = source_logical.resolve(strict=True)
    proc = proc_logical.resolve(strict=True)
    output = output_logical.resolve(strict=False)
    if output.exists() and not resume:
        raise FileExistsError(f"snapshot output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.snapshot.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        output.mkdir(mode=0o755, exist_ok=resume)
        pass_path = output / "PASS"
        snapshot_manifest_path = output / "snapshot_manifest.json"
        if pass_path.exists() and not snapshot_manifest_path.exists():
            raise PairStoreSnapshotError("PASS exists without snapshot terminal manifest")
        if snapshot_manifest_path.exists():
            terminal = validate_promoted_pair_store_snapshot(
                source_root=source,
                expected_source_manifest_sha256=expected_source_manifest_sha256,
                output_dir=output,
                proc_root=proc,
                allowed_pid=allowed_pid,
                allowed_start_ticks=allowed_start_ticks,
                allowed_cmdline_sha256=allowed_cmdline_sha256,
                allowed_output_root=allowed_output_root,
                allowed_project_root=allowed_project_root,
                expected_row_count=expected_row_count,
                expected_vector_dim=expected_vector_dim,
                expected_parent_count=expected_parent_count,
                expected_candidate_count=expected_candidate_count,
                require_pass=pass_path.exists(),
            )
            # Crash reconciliation: terminal JSON was atomically published and
            # fully revalidated, but PASS creation had not completed.
            _publish_pass(pass_path)
            return validate_promoted_pair_store_snapshot(
                source_root=source,
                expected_source_manifest_sha256=expected_source_manifest_sha256,
                output_dir=output,
                proc_root=proc,
                allowed_pid=allowed_pid,
                allowed_start_ticks=allowed_start_ticks,
                allowed_cmdline_sha256=allowed_cmdline_sha256,
                allowed_output_root=allowed_output_root,
                allowed_project_root=allowed_project_root,
                expected_row_count=expected_row_count,
                expected_vector_dim=expected_vector_dim,
                expected_parent_count=expected_parent_count,
                expected_candidate_count=expected_candidate_count,
                require_pass=True,
            )

        source_manifest, before = _source_closure(
            source_root=source,
            expected_manifest_sha256=expected_source_manifest_sha256,
            proc_root=proc,
            allowed_pid=allowed_pid,
            allowed_start_ticks=allowed_start_ticks,
            allowed_cmdline_sha256=allowed_cmdline_sha256,
            allowed_output_root=Path(allowed_output_root),
            allowed_project_root=Path(allowed_project_root),
            expected_row_count=expected_row_count,
            expected_vector_dim=expected_vector_dim,
            expected_parent_count=expected_parent_count,
            expected_candidate_count=expected_candidate_count,
            # The builder requires the frozen generation to be live, but it
            # may finish naturally before this queued CPU task starts.  A
            # missing generation is safer (zero writers); verify_process_set
            # still rejects PID reuse and every unexpected process.
            require_old_alive=False,
        )
        source_pairs = Path(str(source_manifest["pairs_path"])).resolve(strict=True)
        source_vectors = Path(str(source_manifest["vectors_path"])).resolve(strict=True)
        required = source_pairs.stat().st_size + source_vectors.stat().st_size
        free = shutil.disk_usage(output.parent).free
        pair_store = output / "pair_store"
        if pair_store.is_symlink():
            raise PairStoreSnapshotError("snapshot pair-store root may not be symlink")
        pair_store.mkdir(mode=0o755, exist_ok=resume)
        identity = {
            "schema_version": CHECKPOINT_SCHEMA,
            "source_manifest_path": str(source / "run_manifest.json"),
            "source_manifest_sha256": expected_source_manifest_sha256,
            "source_scientific_identity_sha256": source_manifest.get(
                "scientific_identity_sha256"
            ),
            "output_root": str(output),
            "row_count": int(source_manifest["row_count"]),
            "vector_dim": int(source_manifest["vector_dim"]),
            "parent_count": int(expected_parent_count),
            "candidate_count": int(expected_candidate_count),
            "vectors_dtype": str(source_manifest["vectors_dtype"]),
            "copy_order": ["pair_indices.npy", "recourse_vectors.npy"],
        }
        checkpoint_path = output / "snapshot_checkpoint.json"
        if checkpoint_path.exists():
            checkpoint = _load_object(checkpoint_path)
            if checkpoint.get("identity") != identity or checkpoint.get(
                "identity_sha256"
            ) != _stable_hash(identity):
                raise PairStoreSnapshotError("snapshot checkpoint identity mismatch")
        else:
            _atomic_json(
                checkpoint_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "phase": "copying",
                    "identity": identity,
                    "identity_sha256": _stable_hash(identity),
                    "promoted": [],
                },
            )

        discarded_partials: list[str] = []
        for name in ("pair_indices.npy", "recourse_vectors.npy"):
            destination = pair_store / name
            if _discard_restartable_partial(destination=destination):
                discarded_partials.append(str(destination.with_name(f".{name}.partial")))

        remaining = 0
        for source_path, name, expected_hash in (
            (source_pairs, "pair_indices.npy", str(source_manifest["pairs_sha256"])),
            (
                source_vectors,
                "recourse_vectors.npy",
                str(source_manifest["vectors_sha256"]),
            ),
        ):
            destination = pair_store / name
            if destination.exists() or destination.is_symlink():
                _validate_final_copy(
                    source=source_path,
                    destination=destination,
                    expected_sha256=expected_hash,
                    expected_size=source_path.stat().st_size,
                )
            else:
                remaining += source_path.stat().st_size
        free = shutil.disk_usage(output.parent).free
        if free - remaining < int(min_free_after_bytes):
            raise PairStoreSnapshotError(
                f"insufficient persistent snapshot headroom: {free}-{remaining}"
            )

        copies: list[dict[str, Any]] = []
        for source_path, name, expected_hash in (
            (source_pairs, "pair_indices.npy", str(source_manifest["pairs_sha256"])),
            (
                source_vectors,
                "recourse_vectors.npy",
                str(source_manifest["vectors_sha256"]),
            ),
        ):
            copies.append(
                _copy_one(
                    source=source_path,
                    destination=pair_store / name,
                    expected_sha256=expected_hash,
                    expected_size=source_path.stat().st_size,
                )
            )
            _atomic_json(
                checkpoint_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "phase": "copying",
                    "identity": identity,
                    "identity_sha256": _stable_hash(identity),
                    "promoted": [row["destination"] for row in copies],
                    "copies": copies,
                },
            )

        _validate_array(
            pair_store / "pair_indices.npy",
            shape=(int(source_manifest["row_count"]), 2),
            dtype=np.dtype(np.int64),
            label="snapshot pairs",
        )
        _validate_array(
            pair_store / "recourse_vectors.npy",
            shape=(
                int(source_manifest["row_count"]),
                int(source_manifest["vector_dim"]),
            ),
            dtype=np.dtype(str(source_manifest["vectors_dtype"])),
            label="snapshot vectors",
        )
        _validate_cartesian_pair_order(
            pair_store / "pair_indices.npy",
            row_count=expected_row_count,
            parent_count=expected_parent_count,
            candidate_count=expected_candidate_count,
        )

        destination_manifest = dict(source_manifest)
        destination_manifest.update(
            {
                "pairs_path": str(pair_store / "pair_indices.npy"),
                "vectors_path": str(pair_store / "recourse_vectors.npy"),
                "chunk_count": 0,
                "chunks": [],
                "physical_snapshot": True,
                "physical_snapshot_schema": SNAPSHOT_SCHEMA,
                "source_manifest_path": str(source / "run_manifest.json"),
                "source_manifest_sha256": expected_source_manifest_sha256,
                "source_chunk_count": int(source_manifest.get("chunk_count", 0)),
                "source_chunks_sha256": _stable_hash(source_manifest.get("chunks", [])),
            }
        )
        destination_manifest_path = pair_store / "run_manifest.json"
        _atomic_json(destination_manifest_path, destination_manifest)
        _validate_pair_store_manifest(destination_manifest_path, destination_manifest)
        destination_manifest_sha = _sha256(destination_manifest_path)
        dbscan_contract = _dbscan_contract(
            manifest=destination_manifest,
            manifest_sha256=destination_manifest_sha,
        )
        _atomic_json(output / "dbscan_contract.json", dbscan_contract)

        source_manifest_after, after = _source_closure(
            source_root=source,
            expected_manifest_sha256=expected_source_manifest_sha256,
            proc_root=proc,
            allowed_pid=allowed_pid,
            allowed_start_ticks=allowed_start_ticks,
            allowed_cmdline_sha256=allowed_cmdline_sha256,
            allowed_output_root=Path(allowed_output_root),
            allowed_project_root=Path(allowed_project_root),
            expected_row_count=expected_row_count,
            expected_vector_dim=expected_vector_dim,
            expected_parent_count=expected_parent_count,
            expected_candidate_count=expected_candidate_count,
            require_old_alive=False,
        )
        if source_manifest_after != source_manifest or after["files"] != before["files"]:
            raise PairStoreSnapshotError("source pair store drifted during snapshot")
        destination_files = [
            destination_manifest_path,
            pair_store / "pair_indices.npy",
            pair_store / "recourse_vectors.npy",
            output / "dbscan_contract.json",
        ]
        if any(path.is_symlink() for path in destination_files):
            raise PairStoreSnapshotError("snapshot closure contains a symlink")
        partials = sorted(str(path) for path in output.rglob("*.partial"))
        writers = _find_writable_process_references(destination_files, proc_root=proc)
        if partials or writers:
            raise PairStoreSnapshotError(
                "snapshot destination is not immutable: "
                + json.dumps({"partials": partials, "writers": writers}, sort_keys=True)
            )
        terminal = {
            "schema_version": SNAPSHOT_SCHEMA,
            "status": "PASS",
            "source_mutated": False,
            "hardlinked": False,
            "source": before,
            "source_post": after,
            "copy_records": copies,
            "copy_order": ["pair_indices.npy", "recourse_vectors.npy"],
            "required_bytes": required,
            "free_bytes_before": free,
            "minimum_free_after_bytes": int(min_free_after_bytes),
            "remaining_bytes_at_headroom_gate": remaining,
            "discarded_non_authoritative_partials": discarded_partials,
            "expected_row_count": int(expected_row_count),
            "expected_vector_dim": int(expected_vector_dim),
            "expected_parent_count": int(expected_parent_count),
            "expected_candidate_count": int(expected_candidate_count),
            "pair_store_root": str(pair_store),
            "pair_store_manifest": str(destination_manifest_path),
            "pair_store_manifest_sha256": destination_manifest_sha,
            "dbscan_contract": str(output / "dbscan_contract.json"),
            "dbscan_contract_sha256": _sha256(output / "dbscan_contract.json"),
            "destination_writable_reference_count": 0,
            "destination_partial_artifacts": [],
            "resume_supported": True,
            "partial_is_never_authority": True,
        }
        _atomic_json(snapshot_manifest_path, terminal)
        # Never publish PASS from locally accumulated state.  Reopen every
        # source/destination artifact and enforce the same terminal validator
        # used by restart reconciliation.
        validate_promoted_pair_store_snapshot(
            source_root=source,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            output_dir=output,
            proc_root=proc,
            allowed_pid=allowed_pid,
            allowed_start_ticks=allowed_start_ticks,
            allowed_cmdline_sha256=allowed_cmdline_sha256,
            allowed_output_root=allowed_output_root,
            allowed_project_root=allowed_project_root,
            expected_row_count=expected_row_count,
            expected_vector_dim=expected_vector_dim,
            expected_parent_count=expected_parent_count,
            expected_candidate_count=expected_candidate_count,
            require_pass=False,
        )
        _publish_pass(pass_path)
        return validate_promoted_pair_store_snapshot(
            source_root=source,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            output_dir=output,
            proc_root=proc,
            allowed_pid=allowed_pid,
            allowed_start_ticks=allowed_start_ticks,
            allowed_cmdline_sha256=allowed_cmdline_sha256,
            allowed_output_root=allowed_output_root,
            allowed_project_root=allowed_project_root,
            expected_row_count=expected_row_count,
            expected_vector_dim=expected_vector_dim,
            expected_parent_count=expected_parent_count,
            expected_candidate_count=expected_candidate_count,
            require_pass=True,
        )
    finally:
        os.close(lock_descriptor)


__all__ = [
    "DBSCAN_CONTRACT_SCHEMA",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_PARENT_COUNT",
    "EXPECTED_ROWS",
    "EXPECTED_VECTOR_DIM",
    "MIN_FREE_AFTER_BYTES",
    "PairStoreSnapshotError",
    "SNAPSHOT_SCHEMA",
    "create_promoted_pair_store_snapshot",
    "validate_promoted_pair_store_snapshot",
]
