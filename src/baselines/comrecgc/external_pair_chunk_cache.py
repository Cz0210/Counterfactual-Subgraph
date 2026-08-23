"""Audited immutable pair chunks and a reconstructible local vector cache.

The AIDS repair-v4 pair store can have every candidate chunk durably closed
before its network-filesystem consolidation finishes.  This module treats
those persistent chunks—not an active partial—as the authority, proves the
full Cartesian pair order, and reconstructs only the contiguous vector array
on local XFS.  The local array is a cache: its bytes are deterministically
derivable from the persistent chunk closure and are never the sole scientific
authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat as stat_module
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .external_memory_dbscan import ExternalMemoryDBSCANError
from .external_memory_recourse import (
    PAIR_STORE_SCHEMA,
    _atomic_json,
    _file_stat_identity,
    _find_writable_process_references,
    _fsync_directory,
    _sha256_file,
    _stable_hash,
)


CHUNK_CACHE_SCHEMA = "comrecgc_cartesian_chunk_vector_cache_v1"
DEFAULT_LOCAL_FREE_FLOOR_BYTES = 3 * 1024**3


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalMemoryDBSCANError(f"invalid chunk-cache JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExternalMemoryDBSCANError(f"expected chunk-cache JSON object: {path}")
    return value


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("checkpoint_payload_sha256", None)
    return _stable_hash(value)


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["checkpoint_payload_sha256"] = _payload_sha256(value)
    _atomic_json(path, value)
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    value = _load_object(path)
    if value.get("checkpoint_payload_sha256") != _payload_sha256(value):
        raise ExternalMemoryDBSCANError("chunk-cache checkpoint authentication failed")
    return value


def _npy_header_bytes(*, dtype: np.dtype[Any], shape: tuple[int, ...]) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": tuple(int(value) for value in shape),
        },
    )
    return stream.getvalue()


def _npy_data_offset_and_schema(
    path: Path,
) -> tuple[int, tuple[int, ...], np.dtype[Any], bool]:
    with path.open("rb") as handle:
        major, minor = np.lib.format.read_magic(handle)
        if (major, minor) == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif (major, minor) == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ExternalMemoryDBSCANError(
                f"unsupported chunk npy version {(major, minor)}: {path}"
            )
        return int(handle.tell()), tuple(map(int, shape)), np.dtype(dtype), bool(fortran)


def _stream_npy_data(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> Iterator[bytes]:
    offset, _shape, _dtype, _fortran = _npy_data_offset_and_schema(path)
    with path.open("rb") as handle:
        handle.seek(offset)
        for block in iter(lambda: handle.read(int(chunk_bytes)), b""):
            yield block


def _statvfs_free_bytes(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail) * int(value.f_frsize)


def _preallocate_file(path: Path, *, size: int) -> None:
    if not hasattr(os, "posix_fallocate"):
        raise ExternalMemoryDBSCANError("POSIX_FALLOCATE_REQUIRED_FOR_VECTOR_CACHE")
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.posix_fallocate(descriptor, 0, int(size))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_no_symlink(value: str | Path, *, label: str) -> Path:
    logical = Path(value).expanduser()
    if logical.is_symlink():
        raise ExternalMemoryDBSCANError(f"{label} is a symlink")
    resolved = logical.resolve(strict=True)
    current = resolved.stat()
    if not stat_module.S_ISREG(current.st_mode) or current.st_size <= 0:
        raise ExternalMemoryDBSCANError(f"{label} is not a nonempty regular file")
    return resolved


def _scan_active_owner_processes(
    owner_root: Path, *, proc_root: Path, exclude_pids: set[int]
) -> list[dict[str, Any]]:
    """Find processes whose argv names the old owner root.

    Writable-FD scans protect exact source inodes.  This second gate prevents
    a still-live repair-v4 child from atomically replacing its checkpoint just
    after the adoption scan, even when it currently writes only a sibling
    consolidation partial.
    """

    target = str(owner_root.resolve(strict=True))
    found: list[dict[str, Any]] = []
    for process in proc_root.resolve(strict=True).iterdir():
        if not process.name.isdigit() or int(process.name) in exclude_pids:
            continue
        try:
            raw = (process / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        if target in command:
            found.append({"pid": int(process.name), "command": command})
    return sorted(found, key=lambda row: int(row["pid"]))


@dataclass(frozen=True)
class CartesianPairIndexView:
    """Read-only implicit ``[parent, candidate]`` Cartesian pair matrix."""

    parent_count: int
    candidate_count: int
    logical_npy_sha256: str

    @property
    def row_count(self) -> int:
        return int(self.parent_count) * int(self.candidate_count)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_count, 2)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(np.int64)

    def __len__(self) -> int:
        return self.row_count

    def _rows(self, indices: np.ndarray) -> np.ndarray:
        if np.any(indices < 0) or np.any(indices >= self.row_count):
            raise IndexError("Cartesian pair row is out of bounds")
        return np.column_stack(
            (
                indices % int(self.parent_count),
                indices // int(self.parent_count),
            )
        ).astype(np.int64, copy=False)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            rows, columns = key
            return self[rows][..., columns]
        if isinstance(key, slice):
            start, stop, step = key.indices(self.row_count)
            return self._rows(np.arange(start, stop, step, dtype=np.int64))
        values = np.asarray(key)
        if values.ndim == 0:
            index = int(values)
            if index < 0:
                index += self.row_count
            return self._rows(np.asarray([index], dtype=np.int64))[0]
        if values.dtype == np.dtype(np.bool_):
            if values.shape != (self.row_count,):
                raise IndexError("Cartesian pair boolean mask has the wrong shape")
            indices = np.flatnonzero(values).astype(np.int64, copy=False)
        else:
            indices = values.astype(np.int64, copy=False)
            indices = np.where(indices < 0, indices + self.row_count, indices)
        return self._rows(indices)


@dataclass(frozen=True)
class ChunkVectorCacheResult:
    vectors_path: Path
    vectors_sha256: str
    row_count: int
    vector_dim: int
    vectors_dtype: str
    pairs: CartesianPairIndexView
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    local_free_bytes_after: int


@contextmanager
def exclusive_scratch_lock(path: str | Path) -> Iterator[dict[str, Any]]:
    lock_path = Path(path).expanduser().resolve(strict=False)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExternalMemoryDBSCANError("VECTOR_CACHE_SCRATCH_LOCK_BUSY") from exc
        yield {
            "path": str(lock_path),
            "device": int(os.fstat(descriptor).st_dev),
            "inode": int(os.fstat(descriptor).st_ino),
            "pid": os.getpid(),
        }
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _expected_pair_block(
    *, start: int, stop: int, parent_count: int
) -> np.ndarray:
    rows = np.arange(int(start), int(stop), dtype=np.int64)
    return np.column_stack((rows % int(parent_count), rows // int(parent_count)))


def _logical_pair_sha256(*, row_count: int, parent_count: int, block: int = 1_000_000) -> str:
    digest = hashlib.sha256()
    digest.update(_npy_header_bytes(dtype=np.dtype(np.int64), shape=(int(row_count), 2)))
    for start in range(0, int(row_count), int(block)):
        stop = min(int(row_count), start + int(block))
        digest.update(
            _expected_pair_block(
                start=start, stop=stop, parent_count=int(parent_count)
            ).tobytes(order="C")
        )
    return digest.hexdigest()


def _derived_vector_sha256(
    *, chunks: Sequence[Mapping[str, Any]], row_count: int, vector_dim: int, dtype: np.dtype[Any]
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _npy_header_bytes(
            dtype=dtype,
            shape=(int(row_count), int(vector_dim)),
        )
    )
    for row in chunks:
        path = Path(str(row["vectors_path"])).resolve(strict=True)
        for block in _stream_npy_data(path):
            digest.update(block)
    return digest.hexdigest()


def _validate_chunk_source(
    *,
    source_checkpoint_path: Path,
    expected_scientific_identity: Mapping[str, Any],
    expected_chunk_identities: Sequence[Mapping[str, Any]],
    parent_count: int,
    candidate_count: int,
    proc_root: Path,
    source_owner_root: Path,
    require_owner_inactive: bool = True,
) -> dict[str, Any]:
    source = _regular_no_symlink(source_checkpoint_path, label="chunk checkpoint")
    source_stat_before = _file_stat_identity(source)
    source_sha_before = _sha256_file(source)
    checkpoint = _load_object(source)
    identity = dict(expected_scientific_identity)
    if (
        checkpoint.get("schema_version") != PAIR_STORE_SCHEMA
        or checkpoint.get("phase") != "chunks"
        or checkpoint.get("scientific_identity") != identity
        or checkpoint.get("scientific_identity_sha256") != _stable_hash(identity)
    ):
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_SCIENTIFIC_IDENTITY_MISMATCH")
    chunks = checkpoint.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != len(expected_chunk_identities):
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_COUNT_MISMATCH")
    if int(checkpoint.get("next_chunk_index", -1)) != len(chunks):
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_NOT_AT_COMPLETE_BOUNDARY")
    expected_rows = int(parent_count) * int(candidate_count)
    if int(checkpoint.get("row_count", -1)) != expected_rows:
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_CARTESIAN_ROW_COUNT_MISMATCH")

    owner_root = source_owner_root.resolve(strict=True)
    if not chunks or not isinstance(chunks[0], Mapping):
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_EMPTY")
    first_pair_path = _regular_no_symlink(
        chunks[0].get("pairs_path", ""), label="first pair chunk"
    )
    chunk_root = first_pair_path.parent
    if chunk_root.name != "chunks":
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_ROOT_SCHEMA_MISMATCH")
    try:
        chunk_root.relative_to(owner_root)
    except ValueError as exc:
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_OUTSIDE_OWNER_ROOT") from exc
    files: list[Path] = [source]
    resolved_rows: list[tuple[Mapping[str, Any], Mapping[str, Any], Path, Path]] = []
    for row, expected_chunk in zip(chunks, expected_chunk_identities):
        if not isinstance(row, Mapping):
            raise ExternalMemoryDBSCANError("CHUNK_SOURCE_ROW_SCHEMA_MISMATCH")
        pair_path = _regular_no_symlink(row.get("pairs_path", ""), label="pair chunk")
        vector_path = _regular_no_symlink(
            row.get("vectors_path", ""), label="vector chunk"
        )
        if pair_path.parent != chunk_root or vector_path.parent != chunk_root:
            raise ExternalMemoryDBSCANError("CHUNK_SOURCE_PATH_ESCAPE")
        files.extend((pair_path, vector_path))
        resolved_rows.append((row, expected_chunk, pair_path, vector_path))
    if len({str(path) for path in files}) != len(files):
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_DUPLICATE_ARTIFACT_PATH")
    before = {str(path): _file_stat_identity(path) for path in files}
    if before[str(source)] != source_stat_before or _sha256_file(source) != source_sha_before:
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_CHECKPOINT_DRIFT")

    cursor = 0
    frozen_rows: list[dict[str, Any]] = []
    for index, (row, expected_chunk, pair_path, vector_path) in enumerate(resolved_rows):
        if (
            int(row.get("chunk_index", -1)) != index
            or row.get("scientific_identity") != dict(expected_chunk)
            or row.get("scientific_identity_sha256") != _stable_hash(expected_chunk)
        ):
            raise ExternalMemoryDBSCANError("CHUNK_SOURCE_ROW_IDENTITY_MISMATCH")
        pair_hash = _sha256_file(pair_path)
        vector_hash = _sha256_file(vector_path)
        if pair_hash != row.get("pairs_sha256") or vector_hash != row.get(
            "vectors_sha256"
        ):
            raise ExternalMemoryDBSCANError("CHUNK_SOURCE_HASH_MISMATCH")
        pairs = np.load(pair_path, mmap_mode="r", allow_pickle=False)
        vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
        count = int(row.get("row_count", -1))
        if (
            pairs.shape != (count, 2)
            or pairs.dtype != np.dtype(np.int64)
            or vectors.shape != (count, int(row.get("vector_dim", -1)))
            or str(vectors.dtype) != row.get("vectors_dtype")
            or bool(vectors.flags.f_contiguous and not vectors.flags.c_contiguous)
        ):
            raise ExternalMemoryDBSCANError("CHUNK_SOURCE_ARRAY_SCHEMA_MISMATCH")
        expected_pairs = _expected_pair_block(
            start=cursor,
            stop=cursor + count,
            parent_count=int(parent_count),
        )
        if not np.array_equal(pairs, expected_pairs):
            raise ExternalMemoryDBSCANError("CHUNK_SOURCE_CARTESIAN_ORDER_MISMATCH")
        cursor += count
        frozen_rows.append(
            {
                "chunk_index": index,
                "row_count": count,
                "pairs_path": str(pair_path),
                "pairs_sha256": pair_hash,
                "vectors_path": str(vector_path),
                "vectors_sha256": vector_hash,
                "vector_dim": int(vectors.shape[1]),
                "vectors_dtype": str(vectors.dtype),
                "first_pair": row.get("first_pair"),
                "last_pair": row.get("last_pair"),
            }
        )
        del pairs, vectors, expected_pairs
    if cursor != expected_rows:
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_CARTESIAN_UNION_MISMATCH")
    dimensions = {int(row["vector_dim"]) for row in frozen_rows}
    dtypes = {str(row["vectors_dtype"]) for row in frozen_rows}
    if len(dimensions) != 1 or len(dtypes) != 1:
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_VECTOR_SCHEMA_DISAGREEMENT")

    checkpoint_sha = source_sha_before
    writers = _find_writable_process_references(files, proc_root=proc_root)
    active = _scan_active_owner_processes(
        source_owner_root,
        proc_root=proc_root,
        exclude_pids={os.getpid()},
    )
    if writers:
        raise ExternalMemoryDBSCANError(
            "CHUNK_SOURCE_HAS_LIVE_WRITER:" + json.dumps(writers, sort_keys=True)
        )
    if active and require_owner_inactive:
        raise ExternalMemoryDBSCANError(
            "CHUNK_SOURCE_OWNER_PROCESS_ACTIVE:" + json.dumps(active, sort_keys=True)
        )
    after = {str(path): _file_stat_identity(path) for path in files}
    if before != after or checkpoint_sha != _sha256_file(source):
        raise ExternalMemoryDBSCANError("CHUNK_SOURCE_STAT_DRIFT")
    file_hashes: dict[str, str] = {str(source): checkpoint_sha}
    for row in frozen_rows:
        file_hashes[str(row["pairs_path"])] = str(row["pairs_sha256"])
        file_hashes[str(row["vectors_path"])] = str(row["vectors_sha256"])
    return {
        "source_checkpoint_path": str(source),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_stat": after[str(source)],
        "source_files": {
            str(path): {
                "stat": after[str(path)],
                "sha256": file_hashes[str(path)],
            }
            for path in files
        },
        "chunks": frozen_rows,
        "chunk_count": len(frozen_rows),
        "row_count": expected_rows,
        "parent_count": int(parent_count),
        "candidate_count": int(candidate_count),
        "vector_dim": dimensions.pop(),
        "vectors_dtype": dtypes.pop(),
        "scientific_identity": identity,
        "scientific_identity_sha256": _stable_hash(identity),
        "source_owner_root": str(source_owner_root.resolve(strict=True)),
        "source_pair_store_root": str(chunk_root.parent),
        "writable_reference_count": 0,
        "active_owner_process_count": len(active),
        "active_owner_processes": active,
    }


def audit_cartesian_chunk_source(
    *,
    source_checkpoint_path: str | Path,
    source_owner_root: str | Path,
    output_path: str | Path,
    expected_scientific_identity: Mapping[str, Any],
    expected_chunk_identities: Sequence[Mapping[str, Any]],
    parent_count: int,
    candidate_count: int,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Publish a read-only source audit without authorizing live adoption.

    A still-running old owner is recorded and makes ``eligible_for_adoption``
    false, but does not erase a complete physical/hash/order audit.  Writable
    references to any checkpoint/chunk remain a hard failure.
    """

    source = _validate_chunk_source(
        source_checkpoint_path=Path(source_checkpoint_path),
        expected_scientific_identity=expected_scientific_identity,
        expected_chunk_identities=expected_chunk_identities,
        parent_count=int(parent_count),
        candidate_count=int(candidate_count),
        proc_root=Path(proc_root),
        source_owner_root=Path(source_owner_root),
        require_owner_inactive=False,
    )
    payload = {
        "schema_version": "comrecgc_cartesian_chunk_source_audit_v1",
        "status": "PASS",
        "scientific_source_closure_pass": True,
        "diagnostic_only": source["active_owner_process_count"] != 0,
        "eligible_for_adoption": source["active_owner_process_count"] == 0,
        "source": source,
        "source_closure_sha256": _stable_hash(source),
        "approximation_used": False,
    }
    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.exists():
        existing = _load_object(destination)
        if existing != payload:
            raise ExternalMemoryDBSCANError("CHUNK_SOURCE_AUDIT_ALREADY_DIFFERS")
    else:
        _atomic_json(destination, payload)
    return payload


def _validate_cache_prefix(
    *, cache: np.ndarray, chunks: Sequence[Mapping[str, Any]], stop_chunk: int
) -> int:
    cursor = 0
    for row in chunks[: int(stop_chunk)]:
        source = np.load(row["vectors_path"], mmap_mode="r", allow_pickle=False)
        stop = cursor + int(row["row_count"])
        if not np.array_equal(cache[cursor:stop], source):
            raise ExternalMemoryDBSCANError("VECTOR_CACHE_COMMITTED_PREFIX_MISMATCH")
        cursor = stop
    return cursor


def materialize_cartesian_chunk_vector_cache(
    *,
    source_checkpoint_path: str | Path,
    source_owner_root: str | Path,
    persistent_root: str | Path,
    local_cache_root: str | Path,
    scratch_lock_path: str | Path,
    expected_scientific_identity: Mapping[str, Any],
    expected_chunk_identities: Sequence[Mapping[str, Any]],
    parent_count: int,
    candidate_count: int,
    min_local_free_bytes: int = DEFAULT_LOCAL_FREE_FLOOR_BYTES,
    proc_root: str | Path = "/proc",
    resume: bool = False,
) -> ChunkVectorCacheResult:
    """Validate immutable chunks and build one local contiguous vector cache."""

    root = Path(persistent_root).expanduser().resolve(strict=False)
    local = Path(local_cache_root).expanduser().resolve(strict=False)
    state_path = root / "checkpoint.json"
    manifest_path = root / "run_manifest.json"
    cache_partial = local / "recourse_vectors.partial.npy"
    cache_final = local / "recourse_vectors.npy"
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"chunk-cache persistent root is non-empty: {root}")
    if local.exists() and any(local.iterdir()) and not resume:
        raise FileExistsError(f"chunk-cache local root is non-empty: {local}")
    root.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)

    with exclusive_scratch_lock(scratch_lock_path) as lock:
        source = _validate_chunk_source(
            source_checkpoint_path=Path(source_checkpoint_path),
            expected_scientific_identity=expected_scientific_identity,
            expected_chunk_identities=expected_chunk_identities,
            parent_count=int(parent_count),
            candidate_count=int(candidate_count),
            proc_root=Path(proc_root),
            source_owner_root=Path(source_owner_root),
        )
        dtype = np.dtype(source["vectors_dtype"])
        shape = (int(source["row_count"]), int(source["vector_dim"]))
        header = _npy_header_bytes(dtype=dtype, shape=shape)
        expected_size = len(header) + int(np.prod(shape)) * int(dtype.itemsize)
        pair_sha = _logical_pair_sha256(
            row_count=int(source["row_count"]),
            parent_count=int(parent_count),
        )
        identity = {
            "schema_version": CHUNK_CACHE_SCHEMA,
            "source_checkpoint_path": source["source_checkpoint_path"],
            "source_checkpoint_sha256": source["source_checkpoint_sha256"],
            "source_pair_scientific_identity_sha256": source[
                "scientific_identity_sha256"
            ],
            "row_count": int(source["row_count"]),
            "parent_count": int(parent_count),
            "candidate_count": int(candidate_count),
            "pair_formula": "parent=row%parent_count;candidate=row//parent_count",
            "logical_pair_indices_npy_sha256": pair_sha,
            "vectors_shape": list(shape),
            "vectors_dtype": str(dtype),
            "target_npy_header_sha256": hashlib.sha256(header).hexdigest(),
            "target_npy_size_bytes": expected_size,
            "local_vectors_path": str(cache_final),
            "min_local_free_bytes": int(min_local_free_bytes),
        }
        identity_sha = _stable_hash(identity)
        terminal_manifest_exists = manifest_path.exists()
        if terminal_manifest_exists:
            terminal = validate_cartesian_chunk_vector_cache(
                manifest_path,
                require_cache=False,
                proc_root=proc_root,
            )
            terminal_payload = _load_object(manifest_path)
            if terminal_payload.get("scientific_identity") != identity:
                raise ExternalMemoryDBSCANError(
                    "VECTOR_CACHE_TERMINAL_IDENTITY_MISMATCH"
                )
            if cache_final.exists():
                return validate_cartesian_chunk_vector_cache(
                    manifest_path,
                    require_cache=True,
                    proc_root=proc_root,
                )
            if not resume:
                raise ExternalMemoryDBSCANError(
                    "VECTOR_CACHE_LOCAL_ARTIFACT_MISSING_RESUME_REQUIRED"
                )
            # The local file is explicitly reconstructible and not scientific
            # authority.  Re-enter copy_chunks at zero under the same terminal
            # identity; the immutable chunk closure is revalidated above and
            # again before the existing manifest is trusted.
            state = _write_checkpoint(
                state_path,
                {
                    "schema_version": CHUNK_CACHE_SCHEMA,
                    "phase": "copy_chunks",
                    "scientific_identity": identity,
                    "scientific_identity_sha256": identity_sha,
                    "next_chunk_index": 0,
                    "next_row_offset": 0,
                    "terminal_manifest_sha256": terminal.manifest_sha256,
                },
            )
        elif state_path.exists():
            state = _load_checkpoint(state_path)
            if (
                state.get("schema_version") != CHUNK_CACHE_SCHEMA
                or state.get("scientific_identity") != identity
                or state.get("scientific_identity_sha256") != identity_sha
            ):
                raise ExternalMemoryDBSCANError("VECTOR_CACHE_RESUME_IDENTITY_MISMATCH")
        else:
            state = _write_checkpoint(
                state_path,
                {
                    "schema_version": CHUNK_CACHE_SCHEMA,
                    "phase": "copy_chunks",
                    "scientific_identity": identity,
                    "scientific_identity_sha256": identity_sha,
                    "next_chunk_index": 0,
                    "next_row_offset": 0,
                },
            )
        phase = str(state.get("phase"))
        if phase == "copy_chunks":
            start_chunk = int(state.get("next_chunk_index", 0))
            if start_chunk < 0 or start_chunk > len(source["chunks"]):
                raise ExternalMemoryDBSCANError("VECTOR_CACHE_CHECKPOINT_OFFSET_INVALID")
            if cache_partial.exists():
                cache = np.load(cache_partial, mmap_mode="r+", allow_pickle=False)
                if cache.shape != shape or cache.dtype != dtype:
                    raise ExternalMemoryDBSCANError("VECTOR_CACHE_PARTIAL_SCHEMA_MISMATCH")
            else:
                if start_chunk != 0:
                    raise ExternalMemoryDBSCANError("VECTOR_CACHE_PARTIAL_MISSING")
                free_before = _statvfs_free_bytes(local)
                if free_before < expected_size + int(min_local_free_bytes):
                    raise ExternalMemoryDBSCANError(
                        "VECTOR_CACHE_LOCAL_HEADROOM_BLOCKED:"
                        f"free={free_before}:required={expected_size + int(min_local_free_bytes)}"
                    )
                cache = np.lib.format.open_memmap(
                    cache_partial, mode="w+", dtype=dtype, shape=shape
                )
                cache.flush()
                _preallocate_file(cache_partial, size=expected_size)
                free_after_reservation = _statvfs_free_bytes(local)
                if free_after_reservation < int(min_local_free_bytes):
                    raise ExternalMemoryDBSCANError(
                        "VECTOR_CACHE_LOCAL_FLOOR_VIOLATED_AFTER_RESERVATION"
                    )
            replayed_offset = _validate_cache_prefix(
                cache=cache,
                chunks=source["chunks"],
                stop_chunk=start_chunk,
            )
            if replayed_offset != int(state.get("next_row_offset", -1)):
                raise ExternalMemoryDBSCANError("VECTOR_CACHE_CHECKPOINT_CURSOR_MISMATCH")
            cursor = replayed_offset
            for index in range(start_chunk, len(source["chunks"])):
                row = source["chunks"][index]
                values = np.load(row["vectors_path"], mmap_mode="r", allow_pickle=False)
                stop = cursor + int(row["row_count"])
                cache[cursor:stop] = values
                cache.flush()
                with cache_partial.open("rb") as handle:
                    os.fsync(handle.fileno())
                cursor = stop
                state = _write_checkpoint(
                    state_path,
                    {
                        "schema_version": CHUNK_CACHE_SCHEMA,
                        "phase": "copy_chunks",
                        "scientific_identity": identity,
                        "scientific_identity_sha256": identity_sha,
                        "next_chunk_index": index + 1,
                        "next_row_offset": cursor,
                    },
                )
            if cursor != shape[0]:
                raise ExternalMemoryDBSCANError("VECTOR_CACHE_FINAL_CURSOR_MISMATCH")
            cache.flush()
            del cache
            cache_sha = _sha256_file(cache_partial)
            logical_sha = _derived_vector_sha256(
                chunks=source["chunks"],
                row_count=shape[0],
                vector_dim=shape[1],
                dtype=dtype,
            )
            if cache_sha != logical_sha:
                raise ExternalMemoryDBSCANError("VECTOR_CACHE_DERIVED_HASH_MISMATCH")
            state = _write_checkpoint(
                state_path,
                {
                    "schema_version": CHUNK_CACHE_SCHEMA,
                    "phase": "cache_ready",
                    "scientific_identity": identity,
                    "scientific_identity_sha256": identity_sha,
                    "next_chunk_index": len(source["chunks"]),
                    "next_row_offset": shape[0],
                    "vectors_sha256": cache_sha,
                },
            )
            phase = "cache_ready"
        if phase != "cache_ready":
            raise ExternalMemoryDBSCANError(f"unknown vector-cache phase: {phase}")
        expected_sha = str(state.get("vectors_sha256") or "")
        candidates = [path for path in (cache_partial, cache_final) if path.exists()]
        if len(candidates) != 1:
            raise ExternalMemoryDBSCANError("VECTOR_CACHE_PROMOTION_STATE_AMBIGUOUS")
        current = candidates[0]
        values = np.load(current, mmap_mode="r", allow_pickle=False)
        if values.shape != shape or values.dtype != dtype or _sha256_file(current) != expected_sha:
            raise ExternalMemoryDBSCANError("VECTOR_CACHE_PROMOTION_CLOSURE_MISMATCH")
        del values
        if current == cache_partial:
            os.replace(cache_partial, cache_final)
            _fsync_directory(local)

        # Revalidate all persistent sources after the long copy and close the
        # full cache content before publishing the terminal manifest.
        source_after = _validate_chunk_source(
            source_checkpoint_path=Path(source_checkpoint_path),
            expected_scientific_identity=expected_scientific_identity,
            expected_chunk_identities=expected_chunk_identities,
            parent_count=int(parent_count),
            candidate_count=int(candidate_count),
            proc_root=Path(proc_root),
            source_owner_root=Path(source_owner_root),
        )
        if source_after != source:
            raise ExternalMemoryDBSCANError("CHUNK_SOURCE_CHANGED_DURING_CACHE_BUILD")
        logical_sha = _derived_vector_sha256(
            chunks=source["chunks"],
            row_count=shape[0],
            vector_dim=shape[1],
            dtype=dtype,
        )
        if logical_sha != expected_sha or _sha256_file(cache_final) != expected_sha:
            raise ExternalMemoryDBSCANError("VECTOR_CACHE_TERMINAL_HASH_MISMATCH")
        free_after = _statvfs_free_bytes(local)
        if free_after < int(min_local_free_bytes):
            raise ExternalMemoryDBSCANError("VECTOR_CACHE_LOCAL_FLOOR_VIOLATED")
        manifest = {
            "schema_version": CHUNK_CACHE_SCHEMA,
            "run_complete": True,
            "scientific_identity": identity,
            "scientific_identity_sha256": identity_sha,
            "source": source,
            "logical_pair_indices_npy_sha256": pair_sha,
            "pair_indices_materialized": False,
            "pair_formula_proven_elementwise": True,
            "vectors_path": str(cache_final),
            "vectors_sha256": expected_sha,
            "vectors_stat_at_initial_publication": _file_stat_identity(cache_final),
            "vectors_reconstructible_from_persistent_chunks": True,
            "vectors_cache_is_scientific_authority": False,
            "persistent_chunks_are_scientific_authority": True,
            "local_scratch_lock": lock,
            "proc_root": str(Path(proc_root).resolve(strict=True)),
            "local_free_bytes_after": free_after,
            "min_local_free_bytes": int(min_local_free_bytes),
            "approximation_used": False,
        }
        if not terminal_manifest_exists:
            _atomic_json(manifest_path, manifest)
        return validate_cartesian_chunk_vector_cache(
            manifest_path,
            require_cache=True,
            proc_root=proc_root,
        )


def validate_cartesian_chunk_vector_cache(
    manifest_path: str | Path,
    *,
    require_cache: bool,
    proc_root: str | Path | None = None,
) -> ChunkVectorCacheResult:
    """Validate persistent authority and, optionally, the local cache file."""

    path = _regular_no_symlink(manifest_path, label="chunk-cache manifest")
    manifest = _load_object(path)
    identity = manifest.get("scientific_identity")
    source = manifest.get("source")
    if (
        manifest.get("schema_version") != CHUNK_CACHE_SCHEMA
        or manifest.get("run_complete") is not True
        or not isinstance(identity, Mapping)
        or manifest.get("scientific_identity_sha256") != _stable_hash(identity)
        or not isinstance(source, Mapping)
        or manifest.get("pair_indices_materialized") is not False
        or manifest.get("pair_formula_proven_elementwise") is not True
        or manifest.get("vectors_reconstructible_from_persistent_chunks") is not True
        or manifest.get("vectors_cache_is_scientific_authority") is not False
        or manifest.get("persistent_chunks_are_scientific_authority") is not True
        or manifest.get("approximation_used") is not False
    ):
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_TERMINAL_CONTRACT_MISMATCH")
    effective_proc_root = Path(
        manifest.get("proc_root") if proc_root is None else proc_root
    ).resolve(strict=True)
    source_checkpoint = _regular_no_symlink(
        source.get("source_checkpoint_path", ""), label="chunk checkpoint"
    )
    if (
        _sha256_file(source_checkpoint) != source.get("source_checkpoint_sha256")
        or _file_stat_identity(source_checkpoint) != source.get("source_checkpoint_stat")
    ):
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_SOURCE_CHECKPOINT_DRIFT")
    chunks = source.get("chunks")
    if not isinstance(chunks, list) or int(source.get("chunk_count", -1)) != len(chunks):
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_SOURCE_CHUNK_SCHEMA_MISMATCH")
    all_files: list[Path] = [source_checkpoint]
    for row in chunks:
        for field, hash_field in (
            ("pairs_path", "pairs_sha256"),
            ("vectors_path", "vectors_sha256"),
        ):
            artifact = _regular_no_symlink(row[field], label=field)
            all_files.append(artifact)
            recorded = source["source_files"].get(str(artifact))
            if (
                not isinstance(recorded, Mapping)
                or _file_stat_identity(artifact) != recorded.get("stat")
                or _sha256_file(artifact) != row[hash_field]
                or recorded.get("sha256") != row[hash_field]
            ):
                raise ExternalMemoryDBSCANError("VECTOR_CACHE_SOURCE_ARTIFACT_DRIFT")
    writers = _find_writable_process_references(
        all_files, proc_root=effective_proc_root
    )
    if writers:
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_SOURCE_HAS_WRITER")
    owner_root = Path(str(source.get("source_owner_root") or "")).resolve(strict=True)
    active = _scan_active_owner_processes(
        owner_root,
        proc_root=effective_proc_root,
        exclude_pids={os.getpid()},
    )
    if active:
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_SOURCE_OWNER_PROCESS_ACTIVE")
    row_count = int(source["row_count"])
    parent_count = int(source["parent_count"])
    candidate_count = int(source["candidate_count"])
    pair_sha = _logical_pair_sha256(
        row_count=row_count,
        parent_count=parent_count,
    )
    if (
        row_count != parent_count * candidate_count
        or pair_sha != manifest.get("logical_pair_indices_npy_sha256")
        or pair_sha != identity.get("logical_pair_indices_npy_sha256")
    ):
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_LOGICAL_PAIR_CLOSURE_MISMATCH")
    dtype = np.dtype(source["vectors_dtype"])
    derived_sha = _derived_vector_sha256(
        chunks=chunks,
        row_count=row_count,
        vector_dim=int(source["vector_dim"]),
        dtype=dtype,
    )
    if derived_sha != manifest.get("vectors_sha256"):
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_LOGICAL_VECTOR_CLOSURE_MISMATCH")
    cache = Path(str(manifest.get("vectors_path") or "")).expanduser()
    if str(cache.resolve(strict=False)) != identity.get("local_vectors_path"):
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_LOCAL_PATH_MISMATCH")
    if cache.exists():
        cache = _regular_no_symlink(cache, label="local vector cache")
        values = np.load(cache, mmap_mode="r", allow_pickle=False)
        if (
            values.shape != (row_count, int(source["vector_dim"]))
            or values.dtype != dtype
            or _sha256_file(cache) != derived_sha
        ):
            raise ExternalMemoryDBSCANError("VECTOR_CACHE_LOCAL_ARTIFACT_MISMATCH")
    elif require_cache:
        raise ExternalMemoryDBSCANError("VECTOR_CACHE_LOCAL_ARTIFACT_MISSING")
    return ChunkVectorCacheResult(
        vectors_path=cache,
        vectors_sha256=derived_sha,
        row_count=row_count,
        vector_dim=int(source["vector_dim"]),
        vectors_dtype=str(dtype),
        pairs=CartesianPairIndexView(
            parent_count=parent_count,
            candidate_count=candidate_count,
            logical_npy_sha256=pair_sha,
        ),
        source_checkpoint_path=source_checkpoint,
        source_checkpoint_sha256=str(source["source_checkpoint_sha256"]),
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        local_free_bytes_after=int(manifest["local_free_bytes_after"]),
    )


__all__ = [
    "CHUNK_CACHE_SCHEMA",
    "DEFAULT_LOCAL_FREE_FLOOR_BYTES",
    "CartesianPairIndexView",
    "ChunkVectorCacheResult",
    "audit_cartesian_chunk_source",
    "exclusive_scratch_lock",
    "materialize_cartesian_chunk_vector_cache",
    "validate_cartesian_chunk_vector_cache",
]
