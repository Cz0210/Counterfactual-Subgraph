"""Disk-backed exact pair materialization and common-recourse summarization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ExternalMemoryDBSCANError,
    _check_rss,
    _rss_bytes,
    _validate_component_recovery_closure,
    _validate_shortcut_proof_closure,
)


PAIR_STORE_SCHEMA = "comrecgc_external_pair_store_v1"
PAIR_STORE_ADOPTION_SCHEMA = "comrecgc_read_only_pair_store_adoption_v1"
SUMMARY_SCHEMA = "comrecgc_external_cluster_summary_v1"
ONE_CLUSTER_SUMMARY_SCHEMA = "comrecgc_exact_one_cluster_summary_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, values: np.ndarray) -> str:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return _sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reconcile_promoted_pair_array(
    *,
    partial: Path,
    final: Path,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    expected_sha256: str,
    label: str,
) -> None:
    if partial.exists() and final.exists():
        raise ExternalMemoryDBSCANError(
            f"{label} has both partial and final consolidation artifacts"
        )
    source = final if final.exists() else partial
    if not source.exists() or source.is_symlink():
        raise ExternalMemoryDBSCANError(
            f"{label} checkpointed consolidation artifact is missing"
        )
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    if values.shape != shape or values.dtype != dtype:
        raise ExternalMemoryDBSCANError(
            f"{label} consolidation schema mismatch: {values.shape}/{values.dtype}"
        )
    del values
    if _sha256_file(source) != expected_sha256:
        raise ExternalMemoryDBSCANError(
            f"{label} consolidation checksum mismatch"
        )
    if source == partial:
        os.replace(partial, final)
        _fsync_directory(final.parent)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalMemoryDBSCANError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ExternalMemoryDBSCANError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class PairStoreResult:
    pairs_path: Path
    vectors_path: Path
    manifest_path: Path
    row_count: int
    vector_dim: int
    vectors_dtype: str
    pairs_sha256: str
    vectors_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class ExactOneClusterSummaryResult:
    """Terminal products of the proven all-core/one-component summary path."""

    official_result: tuple[list[int], list[float], list[int]]
    selected: list[dict[str, Any]]
    manifest_path: Path
    manifest_sha256: str
    retained_mask_path: Path
    retained_positions_path: Path | None
    retained_vectors_path: Path | None


@dataclass(frozen=True)
class AdoptedPairStoreResult:
    """Read-only source pair store plus its fresh adoption evidence."""

    pair_store: PairStoreResult
    adoption_manifest_path: Path
    adoption_manifest_sha256: str


class ExternalPairStore:
    """Append bounded pair/vector chunks and atomically consolidate them."""

    def __init__(
        self,
        *,
        root: str | Path,
        scientific_identity: Mapping[str, Any],
        max_rss_bytes: int,
        resume: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.chunks = self.root / "chunks"
        self.state_path = self.root / "checkpoint.json"
        self.manifest_path = self.root / "run_manifest.json"
        self.identity = dict(scientific_identity)
        self.identity_hash = _stable_hash(self.identity)
        self.max_rss_bytes = int(max_rss_bytes)
        if self.max_rss_bytes <= 0:
            raise ExternalMemoryDBSCANError("pair-store max RSS must be positive")
        if self.manifest_path.exists():
            self.state = _load_object(self.manifest_path)
            if (
                self.state.get("run_complete") is not True
                or self.state.get("scientific_identity") != self.identity
                or self.state.get("scientific_identity_sha256") != self.identity_hash
            ):
                raise ExternalMemoryDBSCANError("completed pair-store identity mismatch")
            _validate_pair_store_manifest(self.manifest_path, self.state)
            self.complete = True
            return
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise FileExistsError(f"pair-store root is non-empty: {self.root}")
        self.chunks.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self.state = _load_object(self.state_path)
            if (
                self.state.get("schema_version") != PAIR_STORE_SCHEMA
                or self.state.get("scientific_identity") != self.identity
                or self.state.get("scientific_identity_sha256") != self.identity_hash
            ):
                raise ExternalMemoryDBSCANError("pair-store checkpoint identity mismatch")
        else:
            self.state = {
                "schema_version": PAIR_STORE_SCHEMA,
                "phase": "chunks",
                "scientific_identity": self.identity,
                "scientific_identity_sha256": self.identity_hash,
                "next_chunk_index": 0,
                "row_count": 0,
                "chunks": [],
                "peak_rss_bytes": _rss_bytes(),
                "updated_at": _utc_now(),
            }
            _atomic_json(self.state_path, self.state)
        self.complete = False

    @property
    def next_chunk_index(self) -> int:
        return int(self.state.get("next_chunk_index", 0))

    @property
    def completed_chunk_count(self) -> int:
        return len(self.state.get("chunks") or [])

    def append(
        self,
        *,
        chunk_index: int,
        pairs: np.ndarray,
        vectors: np.ndarray,
        chunk_identity: Mapping[str, Any],
    ) -> None:
        if self.complete:
            raise ExternalMemoryDBSCANError("cannot append to a completed pair store")
        if int(chunk_index) != self.next_chunk_index:
            raise ExternalMemoryDBSCANError(
                f"pair chunk order mismatch: {chunk_index} != {self.next_chunk_index}"
            )
        pair_values = np.asarray(pairs)
        vector_values = np.asarray(vectors)
        if (
            pair_values.ndim != 2
            or pair_values.shape[1] != 2
            or pair_values.dtype != np.dtype(np.int64)
        ):
            raise ExternalMemoryDBSCANError("pair chunks must be int64 [N,2]")
        if (
            vector_values.ndim != 2
            or vector_values.shape[0] != pair_values.shape[0]
            or vector_values.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        ):
            raise ExternalMemoryDBSCANError(
                "vector chunks must be aligned float32/float64 [N,D]"
            )
        if vector_values.size and not np.isfinite(vector_values).all():
            raise ExternalMemoryDBSCANError("pair-store vectors contain NaN/Inf")
        if pair_values.shape[0] > 1:
            previous = pair_values[:-1]
            following = pair_values[1:]
            if np.any(
                (following[:, 1] < previous[:, 1])
                | (
                    (following[:, 1] == previous[:, 1])
                    & (following[:, 0] < previous[:, 0])
                )
            ):
                raise ExternalMemoryDBSCANError(
                    "pair chunk does not preserve candidate-major/parent-minor order"
                )
        chunks_before = list(self.state.get("chunks") or [])
        if pair_values.shape[0] and chunks_before:
            previous_last = next(
                (
                    row.get("last_pair")
                    for row in reversed(chunks_before)
                    if row.get("last_pair") is not None
                ),
                None,
            )
            if previous_last is not None:
                if not isinstance(previous_last, list) or len(previous_last) != 2:
                    raise ExternalMemoryDBSCANError("previous pair boundary is invalid")
                current_first = [int(pair_values[0, 0]), int(pair_values[0, 1])]
                if (current_first[1], current_first[0]) < (
                    int(previous_last[1]),
                    int(previous_last[0]),
                ):
                    raise ExternalMemoryDBSCANError(
                        "pair chunks do not preserve global candidate/parent order"
                    )
        _check_rss(self.max_rss_bytes, phase="pair_store.append")
        stem = f"chunk-{int(chunk_index):08d}"
        pair_path = self.chunks / f"{stem}.pairs.npy"
        vector_path = self.chunks / f"{stem}.vectors.npy"
        pair_sha = _atomic_npy(pair_path, pair_values)
        vector_sha = _atomic_npy(vector_path, vector_values)
        row = {
            "chunk_index": int(chunk_index),
            "scientific_identity": dict(chunk_identity),
            "scientific_identity_sha256": _stable_hash(chunk_identity),
            "row_count": int(pair_values.shape[0]),
            "vector_dim": int(vector_values.shape[1]),
            "vectors_dtype": str(vector_values.dtype),
            "pairs_path": str(pair_path),
            "pairs_sha256": pair_sha,
            "vectors_path": str(vector_path),
            "vectors_sha256": vector_sha,
            "first_pair": (
                None
                if pair_values.shape[0] == 0
                else [int(pair_values[0, 0]), int(pair_values[0, 1])]
            ),
            "last_pair": (
                None
                if pair_values.shape[0] == 0
                else [int(pair_values[-1, 0]), int(pair_values[-1, 1])]
            ),
        }
        chunks = [*chunks_before, row]
        self.state = {
            **self.state,
            "next_chunk_index": int(chunk_index) + 1,
            "row_count": int(self.state.get("row_count", 0)) + int(pair_values.shape[0]),
            "chunks": chunks,
            "peak_rss_bytes": max(
                int(self.state.get("peak_rss_bytes", 0)), _rss_bytes()
            ),
            "updated_at": _utc_now(),
        }
        _atomic_json(self.state_path, self.state)

    def verify_completed_chunk(
        self, *, chunk_index: int, chunk_identity: Mapping[str, Any]
    ) -> int:
        chunks = list(self.state.get("chunks") or [])
        if int(chunk_index) >= len(chunks):
            raise ExternalMemoryDBSCANError("requested pair chunk is not complete")
        row = chunks[int(chunk_index)]
        if (
            int(row.get("chunk_index", -1)) != int(chunk_index)
            or row.get("scientific_identity") != dict(chunk_identity)
            or row.get("scientific_identity_sha256") != _stable_hash(chunk_identity)
        ):
            raise ExternalMemoryDBSCANError("completed pair chunk identity mismatch")
        for field, hash_field in (
            ("pairs_path", "pairs_sha256"),
            ("vectors_path", "vectors_sha256"),
        ):
            path = Path(str(row[field])).resolve(strict=True)
            if path.parent != self.chunks or _sha256_file(path) != row[hash_field]:
                raise ExternalMemoryDBSCANError("completed pair chunk checksum mismatch")
        return int(row["row_count"])

    def finalize(self) -> PairStoreResult:
        if self.complete:
            return _pair_store_result(self.manifest_path, self.state)
        chunks = list(self.state.get("chunks") or [])
        if not chunks:
            raise ExternalMemoryDBSCANError("cannot finalize an empty pair store")
        dimensions = {int(row["vector_dim"]) for row in chunks}
        dtypes = {str(row["vectors_dtype"]) for row in chunks}
        if len(dimensions) != 1 or len(dtypes) != 1:
            raise ExternalMemoryDBSCANError("pair chunks disagree on vector schema")
        vector_dim = dimensions.pop()
        dtype = np.dtype(dtypes.pop())
        total = sum(int(row["row_count"]) for row in chunks)
        if total < 0:
            raise ExternalMemoryDBSCANError("negative pair-store row count")
        pairs_partial = self.root / "pair_indices.partial.npy"
        vectors_partial = self.root / "recourse_vectors.partial.npy"
        pairs_final = self.root / "pair_indices.npy"
        vectors_final = self.root / "recourse_vectors.npy"
        phase = str(self.state.get("phase") or "chunks")
        if phase == "chunks":
            if pairs_final.exists() or vectors_final.exists():
                raise ExternalMemoryDBSCANError(
                    "uncheckpointed final consolidation artifact exists"
                )
            # A crash while filling a partial file precedes the ready
            # checkpoint.  Chunks are the immutable source of truth, so these
            # exact scratch names are safely rebuilt from them.
            for partial in (pairs_partial, vectors_partial):
                if partial.exists():
                    if partial.is_symlink():
                        raise ExternalMemoryDBSCANError(
                            "pair-store partial consolidation is a symlink"
                        )
                    partial.unlink()
            pairs_out = np.lib.format.open_memmap(
                pairs_partial, mode="w+", dtype=np.int64, shape=(total, 2)
            )
            vectors_out = np.lib.format.open_memmap(
                vectors_partial, mode="w+", dtype=dtype, shape=(total, vector_dim)
            )
            cursor = 0
            for row in chunks:
                pair_path = Path(str(row["pairs_path"])).resolve(strict=True)
                vector_path = Path(str(row["vectors_path"])).resolve(strict=True)
                if (
                    _sha256_file(pair_path) != row["pairs_sha256"]
                    or _sha256_file(vector_path) != row["vectors_sha256"]
                ):
                    raise ExternalMemoryDBSCANError(
                        "pair chunk changed before consolidation"
                    )
                pair_chunk = np.load(pair_path, mmap_mode="r", allow_pickle=False)
                vector_chunk = np.load(vector_path, mmap_mode="r", allow_pickle=False)
                stop = cursor + int(row["row_count"])
                pairs_out[cursor:stop] = pair_chunk
                vectors_out[cursor:stop] = vector_chunk
                cursor = stop
                _check_rss(self.max_rss_bytes, phase="pair_store.consolidate")
            pairs_out.flush()
            vectors_out.flush()
            del pairs_out, vectors_out
            for path in (pairs_partial, vectors_partial):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            ready = {
                **self.state,
                "phase": "consolidation_ready",
                "consolidation_row_count": total,
                "consolidation_vector_dim": vector_dim,
                "consolidation_vectors_dtype": str(dtype),
                "consolidation_pairs_sha256": _sha256_file(pairs_partial),
                "consolidation_vectors_sha256": _sha256_file(vectors_partial),
                "updated_at": _utc_now(),
            }
            _atomic_json(self.state_path, ready)
            self.state = ready
            phase = "consolidation_ready"
        if phase != "consolidation_ready":
            raise ExternalMemoryDBSCANError(
                f"unknown pair-store consolidation phase: {phase}"
            )
        if (
            int(self.state.get("consolidation_row_count", -1)) != total
            or int(self.state.get("consolidation_vector_dim", -1)) != vector_dim
            or self.state.get("consolidation_vectors_dtype") != str(dtype)
        ):
            raise ExternalMemoryDBSCANError(
                "pair-store consolidation checkpoint schema mismatch"
            )
        pairs_sha256 = str(self.state.get("consolidation_pairs_sha256") or "")
        vectors_sha256 = str(self.state.get("consolidation_vectors_sha256") or "")
        _reconcile_promoted_pair_array(
            partial=pairs_partial,
            final=pairs_final,
            shape=(total, 2),
            dtype=np.dtype(np.int64),
            expected_sha256=pairs_sha256,
            label="pair indices",
        )
        _reconcile_promoted_pair_array(
            partial=vectors_partial,
            final=vectors_final,
            shape=(total, vector_dim),
            dtype=dtype,
            expected_sha256=vectors_sha256,
            label="recourse vectors",
        )
        _fsync_directory(self.root)
        manifest = {
            "schema_version": PAIR_STORE_SCHEMA,
            "run_complete": True,
            "scientific_identity": self.identity,
            "scientific_identity_sha256": self.identity_hash,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "row_count": total,
            "vector_dim": vector_dim,
            "vectors_dtype": str(dtype),
            "pairs_path": str(pairs_final),
            "pairs_sha256": pairs_sha256,
            "vectors_path": str(vectors_final),
            "vectors_sha256": vectors_sha256,
            "candidate_major_parent_minor_order": True,
            "peak_rss_bytes_observed": max(
                int(self.state.get("peak_rss_bytes", 0)), _rss_bytes()
            ),
            "max_rss_bytes": self.max_rss_bytes,
            "completed_at": _utc_now(),
        }
        if int(manifest["peak_rss_bytes_observed"]) > self.max_rss_bytes:
            raise ExternalMemoryDBSCANError("pair-store peak RSS exceeded budget")
        _atomic_json(self.manifest_path, manifest)
        self.state = manifest
        self.complete = True
        return _pair_store_result(self.manifest_path, manifest)


def _pair_store_result(path: Path, manifest: Mapping[str, Any]) -> PairStoreResult:
    return PairStoreResult(
        pairs_path=Path(str(manifest["pairs_path"])).resolve(strict=True),
        vectors_path=Path(str(manifest["vectors_path"])).resolve(strict=True),
        manifest_path=path,
        row_count=int(manifest["row_count"]),
        vector_dim=int(manifest["vector_dim"]),
        vectors_dtype=str(manifest["vectors_dtype"]),
        pairs_sha256=str(manifest["pairs_sha256"]),
        vectors_sha256=str(manifest["vectors_sha256"]),
        manifest_sha256=_sha256_file(path),
    )


def _validate_pair_store_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    root = path.parent.resolve(strict=True)
    pairs_path = Path(str(manifest.get("pairs_path") or "")).resolve(strict=True)
    vectors_path = Path(str(manifest.get("vectors_path") or "")).resolve(strict=True)
    if pairs_path.parent != root or vectors_path.parent != root:
        raise ExternalMemoryDBSCANError("pair-store terminal paths escaped their root")
    if (
        _sha256_file(pairs_path) != manifest.get("pairs_sha256")
        or _sha256_file(vectors_path) != manifest.get("vectors_sha256")
    ):
        raise ExternalMemoryDBSCANError("pair-store terminal checksum mismatch")
    pairs = np.load(pairs_path, mmap_mode="r", allow_pickle=False)
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if (
        pairs.shape != (int(manifest.get("row_count", -1)), 2)
        or pairs.dtype != np.dtype(np.int64)
        or vectors.shape
        != (
            int(manifest.get("row_count", -1)),
            int(manifest.get("vector_dim", -1)),
        )
        or str(vectors.dtype) != manifest.get("vectors_dtype")
    ):
        raise ExternalMemoryDBSCANError("pair-store terminal array schema mismatch")


def _file_stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _find_writable_process_references(
    paths: Sequence[Path], *, proc_root: Path
) -> list[dict[str, Any]]:
    """Return writable FDs/maps referring to exact source inodes."""

    root = proc_root.expanduser().resolve(strict=True)
    targets = {
        (int(path.stat().st_dev), int(path.stat().st_ino)): str(path)
        for path in paths
    }
    writers: list[dict[str, Any]] = []
    for process in root.iterdir():
        if not process.name.isdigit() or not process.is_dir():
            continue
        pid = int(process.name)
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            descriptors = []
        for descriptor in descriptors:
            try:
                value = descriptor.stat()
                target = targets.get((int(value.st_dev), int(value.st_ino)))
                if target is None:
                    continue
                flags_text = (process / "fdinfo" / descriptor.name).read_text(
                    encoding="utf-8"
                )
                flags_line = next(
                    line for line in flags_text.splitlines() if line.startswith("flags:")
                )
                flags = int(flags_line.split(":", 1)[1].strip(), 8)
                if (flags & os.O_ACCMODE) in {os.O_WRONLY, os.O_RDWR}:
                    writers.append(
                        {
                            "pid": pid,
                            "kind": "fd",
                            "fd": int(descriptor.name),
                            "path": target,
                            "flags_octal": oct(flags),
                        }
                    )
            except (
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                StopIteration,
                ValueError,
            ):
                continue
        maps_path = process / "maps"
        try:
            mappings = maps_path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in mappings:
            parts = line.split(maxsplit=5)
            if len(parts) < 5 or "w" not in parts[1] or int(parts[4]) == 0:
                continue
            try:
                major_hex, minor_hex = parts[3].split(":", 1)
                device = os.makedev(int(major_hex, 16), int(minor_hex, 16))
                target = targets.get((int(device), int(parts[4])))
            except (ValueError, OSError):
                continue
            if target is not None:
                writers.append(
                    {
                        "pid": pid,
                        "kind": "mapping",
                        "path": target,
                        "permissions": parts[1],
                    }
                )
    return sorted(
        writers,
        key=lambda row: (
            int(row["pid"]),
            str(row["kind"]),
            int(row.get("fd", -1)),
        ),
    )


def _active_process_commands_referencing(
    target_root: Path, *, proc_root: Path, exclude_pids: set[int]
) -> list[dict[str, Any]]:
    target = str(target_root.resolve(strict=True))
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


def _pair_store_source_root_guard(
    *,
    pair_store_root: Path,
    source_owner_root: Path,
    proc_root: Path,
    reject_partial_files: bool,
) -> dict[str, Any]:
    """Close sibling partials, writable inodes, and old owner processes."""

    pair_root = pair_store_root.resolve(strict=True)
    owner_logical = source_owner_root.expanduser()
    if owner_logical.is_symlink():
        raise ExternalMemoryDBSCANError("PAIR_STORE_SOURCE_OWNER_ROOT_IS_SYMLINK")
    owner_root = owner_logical.resolve(strict=True)
    try:
        pair_root.relative_to(owner_root)
    except ValueError as exc:
        raise ExternalMemoryDBSCANError(
            "PAIR_STORE_SOURCE_OUTSIDE_OWNER_ROOT"
        ) from exc
    files: list[Path] = []
    partials: list[str] = []
    for entry in pair_root.rglob("*"):
        if entry.is_symlink():
            raise ExternalMemoryDBSCANError(
                f"PAIR_STORE_SOURCE_TREE_HAS_SYMLINK:{entry}"
            )
        if entry.is_file():
            resolved = entry.resolve(strict=True)
            files.append(resolved)
            if ".partial." in resolved.name or resolved.name.endswith(".partial"):
                partials.append(str(resolved))
    if reject_partial_files and partials:
        raise ExternalMemoryDBSCANError(
            "PAIR_STORE_SOURCE_HAS_PARTIAL_ARTIFACTS:"
            + json.dumps(sorted(partials))
        )
    writers = _find_writable_process_references(files, proc_root=proc_root)
    if writers:
        raise ExternalMemoryDBSCANError(
            "PAIR_STORE_SOURCE_TREE_HAS_LIVE_WRITER:"
            + json.dumps(writers, sort_keys=True)
        )
    active = _active_process_commands_referencing(
        owner_root,
        proc_root=proc_root,
        # The fresh continuation parent carries the read-only owner path in
        # its own argv and must not be confused with the old scientific
        # producer.  The old repair is a separate controller process, never
        # our parent.
        exclude_pids={os.getpid(), os.getppid()},
    )
    if active:
        raise ExternalMemoryDBSCANError(
            "PAIR_STORE_SOURCE_OWNER_PROCESS_ACTIVE:"
            + json.dumps(active, sort_keys=True)
        )
    return {
        "source_owner_root": str(owner_root),
        "pair_store_root": str(pair_root),
        "guarded_regular_file_count": len(files),
        "partial_artifact_paths": sorted(partials),
        "partial_artifacts_rejected": bool(reject_partial_files),
        "writable_reference_count": 0,
        "active_owner_process_count": 0,
    }


def validate_adopted_pair_store_read_only(
    manifest_path: str | Path,
    *,
    expected_scientific_identity: Mapping[str, Any] | None = None,
    proc_root: str | Path = "/proc",
) -> AdoptedPairStoreResult:
    """Revalidate a physical-reference adoption without trusting its claims.

    Every invocation reopens and hashes the source pair-store closure, checks
    the stat identity frozen by the adoption manifest, and brackets that work
    with writable-FD/mapping scans.  This intentionally costs one full source
    read at terminal reconciliation: downstream stages must never consume a
    stale, replaced, or concurrently-written 25 GiB source merely because a
    small adoption JSON still exists.
    """

    adoption_logical = Path(manifest_path).expanduser()
    if adoption_logical.is_symlink():
        raise ExternalMemoryDBSCANError("pair-store adoption manifest is a symlink")
    adoption_path = adoption_logical.resolve(strict=True)
    if not adoption_path.is_file():
        raise ExternalMemoryDBSCANError("pair-store adoption manifest is not regular")
    adoption = _load_object(adoption_path)
    expected_identity = (
        None
        if expected_scientific_identity is None
        else dict(expected_scientific_identity)
    )
    if (
        adoption.get("schema_version") != PAIR_STORE_ADOPTION_SCHEMA
        or adoption.get("run_complete") is not True
        or adoption.get("adoption_mode") != "physical_read_only_reference"
        or adoption.get("source_open_mode_required") != "read_only"
        or adoption.get("source_mutated") is not False
        or adoption.get("copied") is not False
        or adoption.get("hardlinked") is not False
    ):
        raise ExternalMemoryDBSCANError("pair-store adoption contract mismatch")
    identity = adoption.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise ExternalMemoryDBSCANError("pair-store adoption identity missing")
    identity = dict(identity)
    identity_sha = _stable_hash(identity)
    if (
        adoption.get("scientific_identity_sha256") != identity_sha
        or (expected_identity is not None and identity != expected_identity)
    ):
        raise ExternalMemoryDBSCANError("pair-store adoption identity mismatch")

    source_logical = Path(
        str(adoption.get("source_manifest_path") or "")
    ).expanduser()
    if source_logical.is_symlink():
        raise ExternalMemoryDBSCANError("adopted source manifest is a symlink")
    source_path = source_logical.resolve(strict=True)
    if not source_path.is_file():
        raise ExternalMemoryDBSCANError("adopted source manifest is not regular")
    source_manifest = _load_object(source_path)
    if (
        source_manifest.get("schema_version") != PAIR_STORE_SCHEMA
        or source_manifest.get("run_complete") is not True
        or source_manifest.get("scientific_identity") != identity
        or source_manifest.get("scientific_identity_sha256") != identity_sha
    ):
        raise ExternalMemoryDBSCANError("adopted source scientific identity mismatch")
    pairs_logical = Path(str(source_manifest.get("pairs_path") or ""))
    vectors_logical = Path(str(source_manifest.get("vectors_path") or ""))
    if pairs_logical.is_symlink() or vectors_logical.is_symlink():
        raise ExternalMemoryDBSCANError("adopted pair-store array is a symlink")
    pairs_path = pairs_logical.resolve(strict=True)
    vectors_path = vectors_logical.resolve(strict=True)
    sources = [source_path, pairs_path, vectors_path]
    if any(path.is_symlink() or not path.is_file() for path in sources):
        raise ExternalMemoryDBSCANError("adopted pair-store source is not regular")
    source_files = adoption.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != {
        str(path) for path in sources
    }:
        raise ExternalMemoryDBSCANError("pair-store adoption source-file set mismatch")
    recorded_stats: dict[str, dict[str, int]] = {}
    for path in sources:
        entry = source_files.get(str(path))
        if not isinstance(entry, Mapping) or not isinstance(entry.get("stat"), Mapping):
            raise ExternalMemoryDBSCANError("pair-store adoption stat identity missing")
        recorded_stats[str(path)] = {
            str(key): int(value) for key, value in entry["stat"].items()
        }
    before = {str(path): _file_stat_identity(path) for path in sources}
    if before != recorded_stats:
        raise ExternalMemoryDBSCANError("pair-store adopted source stat drift")

    proc = Path(proc_root).expanduser().resolve(strict=True)
    writer_claim = adoption.get("source_writer_scan")
    if (
        not isinstance(writer_claim, Mapping)
        or writer_claim.get("proc_root") != str(proc)
        or int(writer_claim.get("writable_reference_count", -1)) != 0
        or writer_claim.get("writers") != []
    ):
        raise ExternalMemoryDBSCANError("pair-store adoption writer-scan claim mismatch")
    writers_before = _find_writable_process_references(sources, proc_root=proc)
    if writers_before:
        raise ExternalMemoryDBSCANError(
            "PAIR_STORE_SOURCE_HAS_LIVE_WRITER:"
            + json.dumps(writers_before, sort_keys=True)
        )
    source_guard_claim = adoption.get("source_root_guard")
    if not isinstance(source_guard_claim, Mapping):
        raise ExternalMemoryDBSCANError("pair-store source-root guard missing")
    source_guard = _pair_store_source_root_guard(
        pair_store_root=source_path.parent,
        source_owner_root=Path(
            str(source_guard_claim.get("source_owner_root") or "")
        ),
        proc_root=proc,
        reject_partial_files=True,
    )
    if source_guard != source_guard_claim:
        raise ExternalMemoryDBSCANError("pair-store source-root guard drift")
    _validate_pair_store_manifest(source_path, source_manifest)
    actual_hashes = {
        str(source_path): _sha256_file(source_path),
        str(pairs_path): str(source_manifest["pairs_sha256"]),
        str(vectors_path): str(source_manifest["vectors_sha256"]),
    }
    if adoption.get("source_manifest_sha256") != actual_hashes[str(source_path)]:
        raise ExternalMemoryDBSCANError("pair-store source manifest checksum mismatch")
    for path in sources:
        entry = source_files[str(path)]
        if entry.get("sha256") != actual_hashes[str(path)]:
            raise ExternalMemoryDBSCANError("pair-store adopted source checksum mismatch")
    writers_after = _find_writable_process_references(sources, proc_root=proc)
    if writers_after:
        raise ExternalMemoryDBSCANError(
            "PAIR_STORE_SOURCE_HAS_LIVE_WRITER:"
            + json.dumps(writers_after, sort_keys=True)
        )
    after = {str(path): _file_stat_identity(path) for path in sources}
    if after != before:
        raise ExternalMemoryDBSCANError("pair-store source drift during validation")
    return AdoptedPairStoreResult(
        pair_store=_pair_store_result(source_path, source_manifest),
        adoption_manifest_path=adoption_path,
        adoption_manifest_sha256=_sha256_file(adoption_path),
    )


def adopt_external_pair_store_read_only(
    *,
    source_manifest_path: str | Path,
    source_owner_root: str | Path | None = None,
    adoption_root: str | Path,
    expected_scientific_identity: Mapping[str, Any],
    proc_root: str | Path = "/proc",
    resume: bool = False,
) -> AdoptedPairStoreResult:
    """Adopt a completed pair store by physical read-only reference.

    No source file is copied, linked, chmodded, or opened writable.  The fresh
    adoption manifest binds source content/stat identity and a live Linux
    writable-FD/mapping scan.  The caller must continue to open the returned
    arrays with ``mmap_mode='r'``.
    """

    source_logical = Path(source_manifest_path).expanduser()
    if source_logical.is_symlink():
        raise ExternalMemoryDBSCANError("pair-store source manifest is a symlink")
    source_path = source_logical.resolve(strict=True)
    root = Path(adoption_root).expanduser().resolve(strict=False)
    manifest_path = root / "run_manifest.json"
    expected_identity = dict(expected_scientific_identity)
    expected_identity_sha = _stable_hash(expected_identity)
    if not source_path.is_file():
        raise ExternalMemoryDBSCANError("pair-store source manifest is not regular")
    source_manifest = _load_object(source_path)
    if (
        source_manifest.get("schema_version") != PAIR_STORE_SCHEMA
        or source_manifest.get("run_complete") is not True
    ):
        raise ExternalMemoryDBSCANError(
            "PAIR_STORE_SOURCE_NOT_TERMINALLY_PROMOTED"
        )
    if (
        source_manifest.get("scientific_identity") != expected_identity
        or source_manifest.get("scientific_identity_sha256")
        != expected_identity_sha
    ):
        raise ExternalMemoryDBSCANError("adopted pair-store identity mismatch")
    pairs_logical = Path(str(source_manifest.get("pairs_path") or ""))
    vectors_logical = Path(str(source_manifest.get("vectors_path") or ""))
    if pairs_logical.is_symlink() or vectors_logical.is_symlink():
        raise ExternalMemoryDBSCANError("pair-store source array is a symlink")
    pairs_path = pairs_logical.resolve(strict=True)
    vectors_path = vectors_logical.resolve(strict=True)
    sources = [source_path, pairs_path, vectors_path]
    if any(path.is_symlink() or not path.is_file() for path in sources):
        raise ExternalMemoryDBSCANError("adopted pair-store source is not regular")
    before = {str(path): _file_stat_identity(path) for path in sources}
    _validate_pair_store_manifest(source_path, source_manifest)
    owner_logical = (
        source_path.parent
        if source_owner_root is None
        else Path(source_owner_root).expanduser()
    )
    if owner_logical.is_symlink():
        raise ExternalMemoryDBSCANError("PAIR_STORE_SOURCE_OWNER_ROOT_IS_SYMLINK")
    owner_root = owner_logical.resolve(strict=True)
    source_guard = _pair_store_source_root_guard(
        pair_store_root=source_path.parent,
        source_owner_root=owner_root,
        proc_root=Path(proc_root),
        reject_partial_files=True,
    )
    writers = _find_writable_process_references(
        sources, proc_root=Path(proc_root)
    )
    if writers:
        raise ExternalMemoryDBSCANError(
            "PAIR_STORE_SOURCE_HAS_LIVE_WRITER:" + json.dumps(writers, sort_keys=True)
        )
    after = {str(path): _file_stat_identity(path) for path in sources}
    if before != after:
        raise ExternalMemoryDBSCANError("pair-store source stat drift during adoption")
    source_manifest_sha = _sha256_file(source_path)
    final_after = {str(path): _file_stat_identity(path) for path in sources}
    if after != final_after:
        raise ExternalMemoryDBSCANError("pair-store source drift after writer scan")
    after = final_after
    payload = {
        "schema_version": PAIR_STORE_ADOPTION_SCHEMA,
        "run_complete": True,
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": source_manifest_sha,
        "scientific_identity": expected_identity,
        "scientific_identity_sha256": expected_identity_sha,
        "source_files": {
            str(path): {
                "stat": after[str(path)],
                "sha256": (
                    source_manifest_sha
                    if path == source_path
                    else str(
                        source_manifest[
                            "pairs_sha256" if path == pairs_path else "vectors_sha256"
                        ]
                    )
                ),
            }
            for path in sources
        },
        "source_writer_scan": {
            "proc_root": str(Path(proc_root).expanduser().resolve(strict=True)),
            "writable_reference_count": 0,
            "writers": [],
        },
        "source_root_guard": source_guard,
        "adoption_mode": "physical_read_only_reference",
        "source_open_mode_required": "read_only",
        "source_mutated": False,
        "copied": False,
        "hardlinked": False,
        "adopted_at": _utc_now(),
    }
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"pair-store adoption already exists: {root}")
        existing = _load_object(manifest_path)
        frozen = dict(existing)
        frozen.pop("adopted_at", None)
        current = dict(payload)
        current.pop("adopted_at", None)
        if frozen != current:
            raise ExternalMemoryDBSCANError("pair-store adoption resume mismatch")
        payload = existing
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"pair-store adoption root is non-empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest_path, payload)
    return validate_adopted_pair_store_read_only(
        manifest_path,
        expected_scientific_identity=expected_identity,
        proc_root=proc_root,
    )


def _validate_exact_one_cluster_source(
    *,
    dbscan_manifest_path: Path,
    dbscan_manifest_sha256: str,
    recourse_vectors: np.ndarray,
) -> dict[str, Any]:
    """Close the all-core proof before any specialized summary is attempted."""

    manifest_path = dbscan_manifest_path.expanduser().resolve(strict=True)
    if _sha256_file(manifest_path) != str(dbscan_manifest_sha256):
        raise ExternalMemoryDBSCANError("one-cluster DBSCAN manifest checksum mismatch")
    manifest = _load_object(manifest_path)
    if (
        manifest.get("run_complete") is not True
        or manifest.get("clustering_path")
        not in {
            ALL_CORE_ONE_COMPONENT_SHORTCUT,
            ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
            ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
        }
        or manifest.get("shortcut_proof_path") is None
        or manifest.get("shortcut_proof_sha256") is None
        or int(manifest.get("cluster_count", -1)) != 1
        or int(manifest.get("noise_count", -1)) != 0
        or int(manifest.get("core_count", -1)) != len(recourse_vectors)
        or int(manifest.get("num_samples", -1)) != len(recourse_vectors)
        or manifest.get("neighbor_counts_available") is not False
        or manifest.get("approximation_used") is not False
    ):
        raise ExternalMemoryDBSCANError(
            "one-cluster summary requires a complete exact anchor proof"
        )
    if manifest["clustering_path"] == ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY:
        _validate_component_recovery_closure(
            manifest=manifest, root=manifest_path.parent
        )
    else:
        _validate_shortcut_proof_closure(
            manifest=manifest, root=manifest_path.parent
        )
    proof_path = Path(str(manifest["shortcut_proof_path"])).resolve(strict=True)
    if (
        proof_path.parent != manifest_path.parent
        or _sha256_file(proof_path) != manifest["shortcut_proof_sha256"]
    ):
        raise ExternalMemoryDBSCANError("one-cluster shortcut proof checksum mismatch")
    proof = _load_object(proof_path)
    if (
        proof.get("status") != "PASS"
        or proof.get("all_points_core_proven") is not True
        or proof.get("single_epsilon_component_proven") is not True
        or proof.get("labels_are_exact_sklearn_order") is not True
        or (
            manifest["clustering_path"] != ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
            and proof.get("label_value") != 0
        )
        or (
            manifest["clustering_path"] != ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
            and proof.get("core_mask_value") is not True
        )
        or proof.get("approximation_used") is not False
    ):
        raise ExternalMemoryDBSCANError("one-cluster shortcut proof is incomplete")
    identity = manifest.get("scientific_identity")
    if not isinstance(identity, dict):
        raise ExternalMemoryDBSCANError("DBSCAN scientific identity is missing")
    if manifest.get("scientific_identity_sha256") != _stable_hash(identity):
        raise ExternalMemoryDBSCANError("DBSCAN scientific identity hash mismatch")
    if (
        identity.get("vectors_shape")
        != [int(recourse_vectors.shape[0]), int(recourse_vectors.shape[1])]
        or identity.get("vectors_dtype") != str(recourse_vectors.dtype)
    ):
        raise ExternalMemoryDBSCANError("one-cluster vector schema drifted")
    return manifest


def _open_or_create_memmap(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    resume: bool,
) -> np.memmap:
    if path.exists():
        if path.is_symlink() or not resume:
            raise ExternalMemoryDBSCANError(f"unexpected summary partial: {path}")
        values = np.load(path, mmap_mode="r+", allow_pickle=False)
        if values.shape != shape or values.dtype != dtype:
            raise ExternalMemoryDBSCANError(
                f"summary partial schema mismatch: {path.name}"
            )
        return values
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _fsync_memmap(values: np.memmap) -> None:
    values.flush()
    filename = getattr(values, "filename", None)
    if filename is None:
        raise ExternalMemoryDBSCANError("summary memmap lost its backing file")
    with Path(filename).open("rb") as handle:
        os.fsync(handle.fileno())


def _summary_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    phase: str,
    next_offset: int,
    retained_count: int,
    peak_rss_bytes: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": ONE_CLUSTER_SUMMARY_SCHEMA,
        "scientific_identity": dict(identity),
        "scientific_identity_sha256": _stable_hash(identity),
        "phase": str(phase),
        "next_offset": int(next_offset),
        "retained_count": int(retained_count),
        "peak_rss_bytes": int(peak_rss_bytes),
        "updated_at": _utc_now(),
        **dict(extra or {}),
    }
    _atomic_json(path, payload)
    return payload


def _promote_summary_array(
    *,
    partial: Path,
    final: Path,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    expected_sha256: str,
    label: str,
) -> Path:
    _reconcile_promoted_pair_array(
        partial=partial,
        final=final,
        shape=shape,
        dtype=dtype,
        expected_sha256=expected_sha256,
        label=label,
    )
    return final


def _validate_one_cluster_summary_manifest(
    *, manifest_path: Path, manifest: Mapping[str, Any], identity: Mapping[str, Any]
) -> ExactOneClusterSummaryResult:
    root = manifest_path.parent.resolve(strict=True)
    if (
        manifest.get("schema_version") != ONE_CLUSTER_SUMMARY_SCHEMA
        or manifest.get("run_complete") is not True
        or manifest.get("scientific_identity") != dict(identity)
        or manifest.get("scientific_identity_sha256") != _stable_hash(identity)
        or manifest.get("exact_one_cluster_semantics_replayed") is not True
        or manifest.get("approximation_used") is not False
    ):
        raise ExternalMemoryDBSCANError("completed one-cluster summary mismatch")
    pairs_storage = str(identity.get("pairs_storage") or "physical_npy")
    if pairs_storage == "physical_npy":
        pair_source = Path(str(identity.get("pairs_path") or "")).resolve(strict=True)
        if _sha256_file(pair_source) != identity.get("pairs_sha256"):
            raise ExternalMemoryDBSCANError(
                "completed one-cluster physical pair source changed"
            )
    elif pairs_storage == "implicit_cartesian_v1":
        authority = Path(
            str(identity.get("pair_authority_manifest_path") or "")
        ).resolve(strict=True)
        if _sha256_file(authority) != identity.get("pair_authority_manifest_sha256"):
            raise ExternalMemoryDBSCANError(
                "completed one-cluster implicit pair authority changed"
            )
    else:
        raise ExternalMemoryDBSCANError(
            "completed one-cluster pair storage contract is invalid"
        )
    artifact_paths: dict[str, Path | None] = {}
    for name in (
        "retained_mask",
        "retained_positions",
        "retained_vectors",
        "torch_centroid",
        "stable_float64_centroid",
        "numpy_centroid",
        "retained_centroid",
    ):
        path_value = manifest.get(f"{name}_path")
        sha_value = manifest.get(f"{name}_sha256")
        if path_value is None:
            if sha_value is not None or name in {
                "retained_mask",
                "torch_centroid",
                "stable_float64_centroid",
                "numpy_centroid",
            }:
                raise ExternalMemoryDBSCANError(
                    f"completed one-cluster {name} closure mismatch"
                )
            artifact_paths[name] = None
            continue
        path = Path(str(path_value)).resolve(strict=True)
        if path.parent != root or _sha256_file(path) != sha_value:
            raise ExternalMemoryDBSCANError(
                f"completed one-cluster {name} checksum mismatch"
            )
        artifact_paths[name] = path
    vector_shape = manifest.get("scientific_identity", {}).get("vectors_shape")
    vector_dtype = manifest.get("scientific_identity", {}).get("vectors_dtype")
    if (
        not isinstance(vector_shape, list)
        or len(vector_shape) != 2
        or int(vector_shape[0]) <= 0
        or int(vector_shape[1]) <= 0
    ):
        raise ExternalMemoryDBSCANError("completed one-cluster vector schema is invalid")
    n_samples, n_features = map(int, vector_shape)
    retained_count = int(manifest.get("retained_count", -1))
    mask = np.load(artifact_paths["retained_mask"], mmap_mode="r", allow_pickle=False)
    if mask.shape != (n_samples,) or mask.dtype != np.dtype(np.bool_):
        raise ExternalMemoryDBSCANError("completed one-cluster mask schema mismatch")
    if int(np.count_nonzero(mask)) != retained_count:
        raise ExternalMemoryDBSCANError("completed one-cluster retained count mismatch")
    for name in ("torch_centroid", "numpy_centroid"):
        center = np.load(artifact_paths[name], allow_pickle=False)
        if center.shape != (n_features,) or str(center.dtype) != vector_dtype:
            raise ExternalMemoryDBSCANError(
                f"completed one-cluster {name} schema mismatch"
            )
    stable_center = np.load(
        artifact_paths["stable_float64_centroid"], allow_pickle=False
    )
    if stable_center.shape != (n_features,) or stable_center.dtype != np.dtype(
        np.float64
    ):
        raise ExternalMemoryDBSCANError(
            "completed one-cluster stable float64 centroid schema mismatch"
        )
    if (
        artifact_paths["retained_positions"] is not None
        or artifact_paths["retained_vectors"] is not None
        or manifest.get("large_retained_arrays_materialized") is not False
        or int(manifest.get("retained_vector_bytes_materialized", -1)) != 0
    ):
        raise ExternalMemoryDBSCANError(
            "completed one-cluster summary unexpectedly materialized retained rows"
        )
    if retained_count:
        retained_center = np.load(
            artifact_paths["retained_centroid"], allow_pickle=False
        )
        if (
            retained_center.shape != (n_features,)
            or str(retained_center.dtype) != vector_dtype
        ):
            raise ExternalMemoryDBSCANError(
                "completed one-cluster retained artifact schema mismatch"
            )
    official = manifest.get("official_result")
    selected = manifest.get("selected")
    official_parents = manifest.get("official_covered_parent_indices")
    official_first_candidates = manifest.get(
        "official_first_counterfactual_indices"
    )
    official_radius_candidates = manifest.get(
        "official_radius_counterfactual_indices"
    )
    if (
        not isinstance(official, list)
        or len(official) != 3
        or not all(isinstance(part, list) for part in official)
        or not isinstance(selected, list)
        or not isinstance(official_parents, list)
        or not isinstance(official_first_candidates, list)
        or not isinstance(official_radius_candidates, list)
        or official_parents != sorted(set(map(int, official_parents)))
        or official_first_candidates
        != sorted(set(map(int, official_first_candidates)))
        or official_radius_candidates
        != sorted(set(map(int, official_radius_candidates)))
        or not set(official_first_candidates).issubset(
            set(official_radius_candidates)
        )
    ):
        raise ExternalMemoryDBSCANError("completed one-cluster result schema mismatch")
    return ExactOneClusterSummaryResult(
        official_result=(list(official[0]), list(official[1]), list(official[2])),
        selected=[dict(row) for row in selected],
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        retained_mask_path=artifact_paths["retained_mask"],  # type: ignore[arg-type]
        retained_positions_path=None,
        retained_vectors_path=None,
    )


def validate_proven_one_cluster_summary(
    manifest_path: str | Path,
) -> ExactOneClusterSummaryResult:
    """Validate a terminal summary and every hash-bound scientific artifact."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = _load_object(path)
    identity = manifest.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise ExternalMemoryDBSCANError(
            "completed one-cluster scientific identity is absent"
        )
    return _validate_one_cluster_summary_manifest(
        manifest_path=path, manifest=manifest, identity=identity
    )


def _validate_scan_offset(offset: int, *, total: int, block_size: int, phase: str) -> None:
    if (
        int(offset) < 0
        or int(offset) > int(total)
        or (int(offset) != int(total) and int(offset) % int(block_size) != 0)
    ):
        raise ExternalMemoryDBSCANError(
            f"{phase} checkpoint offset is not a committed block boundary"
        )


def _scan_torch_coverage_audit_prefix(
    *,
    recourse_vectors: np.ndarray,
    pair_indices: Any,
    center: np.ndarray,
    stable_float64_center: np.ndarray,
    radius: float,
    stop_offset: int,
    block_size: int,
    torch_module: Any,
) -> tuple[dict[int, int], set[int], int, int, int]:
    """Replay official strict-radius coverage plus a float64 boundary audit."""

    tensor = torch_module.from_numpy(recourse_vectors)
    center_tensor = torch_module.from_numpy(center)
    first_cf_by_parent: dict[int, int] = {}
    all_counterfactuals: set[int] = set()
    within = 0
    exactly_at_radius = 0
    float64_membership_disagreements = 0
    for offset in range(0, int(stop_offset), int(block_size)):
        stop = min(int(stop_offset), offset + int(block_size))
        distances = torch_module.norm(tensor[offset:stop] - center_tensor, dim=-1)
        retained = (distances < float(radius)).detach().cpu().numpy()
        exactly = (distances == float(radius)).detach().cpu().numpy()
        stable_distances = np.linalg.norm(
            np.asarray(recourse_vectors[offset:stop], dtype=np.float64)
            - stable_float64_center,
            axis=1,
        )
        stable_retained = stable_distances < float(radius)
        within += int(np.count_nonzero(retained))
        exactly_at_radius += int(np.count_nonzero(exactly))
        float64_membership_disagreements += int(
            np.count_nonzero(retained != stable_retained)
        )
        if bool(np.any(retained)):
            block_pairs = pair_indices[offset:stop][retained]
            all_counterfactuals.update(map(int, block_pairs[:, 1].tolist()))
            parents, first = np.unique(block_pairs[:, 0], return_index=True)
            for parent, local_index in zip(parents.tolist(), first.tolist()):
                first_cf_by_parent.setdefault(
                    int(parent), int(block_pairs[int(local_index), 1])
                )
    return (
        first_cf_by_parent,
        all_counterfactuals,
        within,
        exactly_at_radius,
        float64_membership_disagreements,
    )


def _numpy_trace_membership(
    values: np.ndarray, *, center: np.ndarray, radius: float
) -> np.ndarray:
    distances = np.linalg.norm(values - center, axis=1)
    return distances.astype(np.float64, copy=False) < float(radius)


def _validate_trace_mask_prefix(
    *,
    mask: np.ndarray,
    recourse_vectors: np.ndarray,
    center: np.ndarray,
    radius: float,
    stop_offset: int,
    block_size: int,
) -> int:
    count = 0
    for offset in range(0, int(stop_offset), int(block_size)):
        stop = min(int(stop_offset), offset + int(block_size))
        expected = _numpy_trace_membership(
            recourse_vectors[offset:stop], center=center, radius=radius
        )
        if not np.array_equal(mask[offset:stop], expected):
            raise ExternalMemoryDBSCANError(
                "trace-mask committed prefix content mismatch"
            )
        count += int(np.count_nonzero(expected))
    return count


def _array_hex(values: np.ndarray) -> list[str]:
    return [float(value).hex() for value in np.asarray(values).tolist()]


def _array_from_hex(values: Sequence[str], *, dtype: np.dtype[Any]) -> np.ndarray:
    return np.asarray([float.fromhex(str(value)) for value in values], dtype=dtype)


def _stream_masked_sum_prefix(
    *,
    mask: np.ndarray,
    recourse_vectors: np.ndarray,
    stop_offset: int,
    block_size: int,
    dtype: np.dtype[Any],
) -> tuple[np.ndarray, int]:
    total = np.zeros(int(recourse_vectors.shape[1]), dtype=dtype)
    count = 0
    for offset in range(0, int(stop_offset), int(block_size)):
        stop = min(int(stop_offset), offset + int(block_size))
        selected = np.asarray(recourse_vectors[offset:stop][mask[offset:stop]])
        if len(selected):
            block_sum = np.sum(selected, axis=0, dtype=dtype)
            total = np.add(total, block_sum, dtype=dtype)
            count += int(len(selected))
    return total, count


def _stream_float64_centroid(
    values: np.ndarray, *, block_size: int
) -> np.ndarray:
    total = np.zeros(int(values.shape[1]), dtype=np.float64)
    for offset in range(0, len(values), int(block_size)):
        stop = min(len(values), offset + int(block_size))
        total += np.sum(
            np.asarray(values[offset:stop], dtype=np.float64),
            axis=0,
            dtype=np.float64,
        )
    return total / np.float64(len(values))


def _scan_retained_medoid_prefix(
    *,
    mask: np.ndarray,
    recourse_vectors: np.ndarray,
    pair_indices: Any,
    center: np.ndarray,
    stop_offset: int,
    block_size: int,
) -> tuple[int, float, set[int], set[int]]:
    """Replay a first-argmin medoid and coverage sets without retained copies."""

    winner_position = -1
    winner_distance = float("inf")
    parents: set[int] = set()
    candidates: set[int] = set()
    for offset in range(0, int(stop_offset), int(block_size)):
        stop = min(int(stop_offset), offset + int(block_size))
        local = np.flatnonzero(mask[offset:stop])
        if not len(local):
            continue
        physical = local.astype(np.int64, copy=False) + offset
        selected = np.asarray(recourse_vectors[offset:stop][local])
        distances = np.linalg.norm(selected - center, axis=1)
        local_winner = int(np.argmin(distances))
        local_distance = float(distances[local_winner])
        if local_distance < winner_distance:
            winner_distance = local_distance
            winner_position = int(physical[local_winner])
        pairs = pair_indices[physical]
        parents.update(map(int, np.unique(pairs[:, 0]).tolist()))
        candidates.update(map(int, np.unique(pairs[:, 1]).tolist()))
    return winner_position, winner_distance, parents, candidates


def summarize_proven_one_cluster_external(
    *,
    work_dir: str | Path,
    dbscan_manifest_path: str | Path,
    dbscan_manifest_sha256: str,
    recourse_vectors: np.ndarray,
    pair_indices: Any,
    pairs_sha256: str,
    pair_authority_manifest_path: str | Path | None = None,
    pair_authority_manifest_sha256: str | None = None,
    radius: float,
    theta: float,
    recourse_size: int,
    official_greedy: Callable[..., Any],
    torch_module: Any,
    max_rss_bytes: int,
    block_size: int = 65_536,
    resume: bool = False,
) -> ExactOneClusterSummaryResult:
    """Exactly replay the pinned one-cluster coverage/trace without Python O(N).

    The entrypoint is legal only after a hash-closed DBSCAN certificate proves
    that every row is core and sklearn's sole label is zero.  Torch and NumPy
    centroids remain separate because the two pinned legacy consumers use
    different reduction implementations.  Per-row norms are blockwise (there
    is no cross-row reduction).  Radius membership is stored as a one-byte
    bitmap; retained rows are then reduced and searched in global row order.
    No retained position/vector matrix is materialized.
    """

    root = Path(work_dir).expanduser().resolve(strict=False)
    state_path = root / "checkpoint.json"
    manifest_path = root / "run_manifest.json"
    if (
        recourse_vectors.ndim != 2
        or pair_indices.ndim != 2
        or pair_indices.shape != (len(recourse_vectors), 2)
        or pair_indices.dtype != np.dtype(np.int64)
        or recourse_vectors.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or len(recourse_vectors) <= 0
    ):
        raise ExternalMemoryDBSCANError("one-cluster arrays are not exactly aligned")
    if not np.isfinite(float(radius)) or float(radius) <= 0:
        raise ExternalMemoryDBSCANError("one-cluster radius must be positive")
    if not np.isfinite(float(theta)) or float(theta) < 0:
        raise ExternalMemoryDBSCANError("one-cluster theta must be nonnegative")
    if int(recourse_size) < 0 or int(block_size) <= 0 or int(max_rss_bytes) <= 0:
        raise ExternalMemoryDBSCANError("invalid one-cluster execution bound")
    dbscan_path = Path(dbscan_manifest_path).expanduser().resolve(strict=True)
    dbscan_manifest = _validate_exact_one_cluster_source(
        dbscan_manifest_path=dbscan_path,
        dbscan_manifest_sha256=dbscan_manifest_sha256,
        recourse_vectors=recourse_vectors,
    )
    pair_filename = str(getattr(pair_indices, "filename", "") or "")
    pair_path: Path | None = None
    pair_authority_path: Path | None = None
    if pair_filename:
        pair_path = Path(pair_filename).resolve(strict=True)
        if _sha256_file(pair_path) != str(pairs_sha256):
            raise ExternalMemoryDBSCANError("one-cluster pair checksum mismatch")
        pairs_storage = "physical_npy"
    else:
        if (
            getattr(pair_indices, "logical_npy_sha256", None) != str(pairs_sha256)
            or pair_authority_manifest_path is None
            or pair_authority_manifest_sha256 is None
        ):
            raise ExternalMemoryDBSCANError(
                "one-cluster implicit pair authority is incomplete"
            )
        pair_authority_path = Path(pair_authority_manifest_path).expanduser().resolve(
            strict=True
        )
        if _sha256_file(pair_authority_path) != str(pair_authority_manifest_sha256):
            raise ExternalMemoryDBSCANError(
                "one-cluster implicit pair authority checksum mismatch"
            )
        pairs_storage = "implicit_cartesian_v1"
    vector_path = Path(
        str(getattr(recourse_vectors, "filename", "") or "")
    ).resolve(strict=True)
    dbscan_identity = dbscan_manifest["scientific_identity"]
    if (
        Path(str(dbscan_identity.get("vectors_path") or "")).resolve(strict=True)
        != vector_path
        or _sha256_file(vector_path) != dbscan_identity.get("vectors_sha256")
    ):
        raise ExternalMemoryDBSCANError("one-cluster vector checksum mismatch")
    identity = {
        "schema_version": ONE_CLUSTER_SUMMARY_SCHEMA,
        "dbscan_manifest_path": str(dbscan_path),
        "dbscan_manifest_sha256": str(dbscan_manifest_sha256),
        "shortcut_proof_path": dbscan_manifest["shortcut_proof_path"],
        "shortcut_proof_sha256": dbscan_manifest["shortcut_proof_sha256"],
        "vectors_path": str(vector_path),
        "vectors_sha256": dbscan_identity["vectors_sha256"],
        "vectors_shape": [int(value) for value in recourse_vectors.shape],
        "vectors_dtype": str(recourse_vectors.dtype),
        "pairs_storage": pairs_storage,
        "pairs_path": None if pair_path is None else str(pair_path),
        "pairs_sha256": str(pairs_sha256),
        "pair_authority_manifest_path": (
            None if pair_authority_path is None else str(pair_authority_path)
        ),
        "pair_authority_manifest_sha256": (
            None
            if pair_authority_path is None
            else str(pair_authority_manifest_sha256)
        ),
        "radius": float(radius),
        "theta": float(theta),
        "recourse_size": int(recourse_size),
        "block_size": int(block_size),
        "torch_version": str(torch_module.__version__),
        "numpy_version": str(np.__version__),
        "pair_order": "candidate_major_parent_minor",
        "coverage_comparison": "torch.norm(row-torch.mean(all_rows)) < radius",
        "trace_comparison": "numpy.linalg.norm(row-numpy.mean(all_rows)) < radius",
        "retained_summary_storage": "bitmap_plus_streaming_reductions_v1",
        "retained_medoid": "first_argmin_in_global_pair_order",
        "large_retained_arrays_materialized": False,
    }
    identity_hash = _stable_hash(identity)
    if manifest_path.exists():
        manifest = _load_object(manifest_path)
        return _validate_one_cluster_summary_manifest(
            manifest_path=manifest_path, manifest=manifest, identity=identity
        )
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"one-cluster summary root is non-empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = _load_object(state_path)
        if (
            state.get("schema_version") != ONE_CLUSTER_SUMMARY_SCHEMA
            or state.get("scientific_identity") != identity
            or state.get("scientific_identity_sha256") != identity_hash
        ):
            raise ExternalMemoryDBSCANError("one-cluster checkpoint identity mismatch")
    else:
        state = _summary_checkpoint(
            state_path,
            identity=identity,
            phase="torch_coverage",
            next_offset=0,
            retained_count=0,
            peak_rss_bytes=_rss_bytes(),
        )
    peak = max(int(state.get("peak_rss_bytes", 0)), _rss_bytes())
    n_samples, n_features = map(int, recourse_vectors.shape)
    row_bytes = n_features * int(recourse_vectors.dtype.itemsize)
    reservation = int(block_size) * (row_bytes * 2 + 32) + 128 * 1024**2
    _check_rss(
        int(max_rss_bytes), phase="one_cluster.block", reserved_bytes=reservation
    )

    torch_center_path = root / "torch_centroid.npy"
    checkpointed_torch_center_sha = state.get("torch_centroid_sha256")
    if torch_center_path.exists() and checkpointed_torch_center_sha:
        if _sha256_file(torch_center_path) != checkpointed_torch_center_sha:
            raise ExternalMemoryDBSCANError(
                "torch centroid checkpoint checksum mismatch"
            )
        torch_center = np.load(torch_center_path, allow_pickle=False)
        if torch_center.shape != (n_features,) or torch_center.dtype != recourse_vectors.dtype:
            raise ExternalMemoryDBSCANError("torch centroid checkpoint schema mismatch")
    else:
        tensor = torch_module.from_numpy(recourse_vectors)
        torch_center_tensor = torch_module.mean(tensor, dim=0)
        torch_center = torch_center_tensor.detach().cpu().numpy().copy()
        _atomic_npy(torch_center_path, torch_center)
        del torch_center_tensor, tensor
    torch_center_sha = _sha256_file(torch_center_path)
    torch_center_norm = float(
        torch_module.norm(torch_module.from_numpy(torch_center)).item()
    )
    float64_center_path = root / "stable_float64_centroid.npy"
    if float64_center_path.exists():
        float64_center = np.load(float64_center_path, allow_pickle=False)
        if float64_center.shape != (n_features,) or float64_center.dtype != np.dtype(
            np.float64
        ):
            raise ExternalMemoryDBSCANError(
                "stable float64 centroid checkpoint schema mismatch"
            )
    else:
        float64_center = _stream_float64_centroid(
            recourse_vectors, block_size=int(block_size)
        )
        _atomic_npy(float64_center_path, float64_center)
    float64_center_sha = _sha256_file(float64_center_path)
    float64_center_norm = float(np.linalg.norm(float64_center))
    centroid_max_abs_difference = float(
        np.max(np.abs(np.asarray(torch_center, dtype=np.float64) - float64_center))
    )
    centroid_norm_decision_disagreement = (
        (torch_center_norm < float(theta))
        != (float64_center_norm < float(theta))
    )
    # The pinned official decision is the fixed-order float32/Torch result.
    # Float64 is an audit value, not a silent replacement for that decision.

    phase = str(state.get("phase"))
    if phase == "torch_coverage":
        first_cf_by_parent = {
            int(parent): int(candidate)
            for parent, candidate in (state.get("first_cf_by_parent") or [])
        }
        official_counterfactuals = set(
            map(int, state.get("official_counterfactuals") or [])
        )
        start = int(state.get("next_offset", 0))
        _validate_scan_offset(
            start,
            total=n_samples,
            block_size=int(block_size),
            phase="torch coverage",
        )
        (
            replayed_first,
            replayed_counterfactuals,
            replayed_within,
            replayed_exact,
            replayed_disagreements,
        ) = _scan_torch_coverage_audit_prefix(
            recourse_vectors=recourse_vectors,
            pair_indices=pair_indices,
            center=torch_center,
            stable_float64_center=float64_center,
            radius=float(radius),
            stop_offset=start,
            block_size=int(block_size),
            torch_module=torch_module,
        )
        if (
            replayed_first != first_cf_by_parent
            or replayed_counterfactuals != official_counterfactuals
            or replayed_within != int(state.get("official_within_radius_count", 0))
            or replayed_exact != int(state.get("count_exactly_at_delta", 0))
            or replayed_disagreements
            != int(state.get("float64_radius_membership_disagreement_count", 0))
        ):
            raise ExternalMemoryDBSCANError(
                "torch-coverage checkpoint prefix state mismatch"
            )
        official_within_radius_count = replayed_within
        count_exactly_at_delta = replayed_exact
        float64_radius_membership_disagreement_count = replayed_disagreements
        tensor = torch_module.from_numpy(recourse_vectors)
        center_tensor = torch_module.from_numpy(torch_center)
        for offset in range(start, n_samples, int(block_size)):
            stop = min(n_samples, offset + int(block_size))
            distances = torch_module.norm(
                tensor[offset:stop] - center_tensor, dim=-1
            )
            retained = distances < float(radius)
            retained_numpy = retained.detach().cpu().numpy()
            exactly_numpy = (distances == float(radius)).detach().cpu().numpy()
            stable_distances = np.linalg.norm(
                np.asarray(recourse_vectors[offset:stop], dtype=np.float64)
                - float64_center,
                axis=1,
            )
            stable_retained = stable_distances < float(radius)
            official_within_radius_count += int(np.count_nonzero(retained_numpy))
            count_exactly_at_delta += int(np.count_nonzero(exactly_numpy))
            float64_radius_membership_disagreement_count += int(
                np.count_nonzero(retained_numpy != stable_retained)
            )
            if bool(np.any(retained_numpy)):
                block_pairs = pair_indices[offset:stop][retained_numpy]
                official_counterfactuals.update(
                    map(int, block_pairs[:, 1].tolist())
                )
                parents, first = np.unique(block_pairs[:, 0], return_index=True)
                for parent, local_index in zip(parents.tolist(), first.tolist()):
                    first_cf_by_parent.setdefault(
                        int(parent), int(block_pairs[int(local_index), 1])
                    )
            del (
                distances,
                retained,
                retained_numpy,
                exactly_numpy,
                stable_distances,
                stable_retained,
            )
            peak = max(peak, _check_rss(int(max_rss_bytes), phase="one_cluster.torch"))
            state = _summary_checkpoint(
                state_path,
                identity=identity,
                phase="torch_coverage",
                next_offset=stop,
                retained_count=0,
                peak_rss_bytes=peak,
                extra={
                    "torch_centroid_sha256": torch_center_sha,
                    "torch_centroid_norm": torch_center_norm,
                    "stable_float64_centroid_sha256": float64_center_sha,
                    "stable_float64_centroid_norm": float64_center_norm,
                    "centroid_max_abs_difference": centroid_max_abs_difference,
                    "official_within_radius_count": official_within_radius_count,
                    "count_exactly_at_delta": count_exactly_at_delta,
                    "float64_radius_membership_disagreement_count": (
                        float64_radius_membership_disagreement_count
                    ),
                    "first_cf_by_parent": sorted(first_cf_by_parent.items()),
                    "official_counterfactuals": sorted(official_counterfactuals),
                },
            )
        state = _summary_checkpoint(
            state_path,
            identity=identity,
            phase="trace_mask",
            next_offset=0,
            retained_count=0,
            peak_rss_bytes=peak,
            extra={
                "torch_centroid_sha256": torch_center_sha,
                "torch_centroid_norm": torch_center_norm,
                "stable_float64_centroid_sha256": float64_center_sha,
                "stable_float64_centroid_norm": float64_center_norm,
                "centroid_max_abs_difference": centroid_max_abs_difference,
                "official_within_radius_count": official_within_radius_count,
                "count_exactly_at_delta": count_exactly_at_delta,
                "float64_radius_membership_disagreement_count": (
                    float64_radius_membership_disagreement_count
                ),
                "first_cf_by_parent": sorted(first_cf_by_parent.items()),
                "official_counterfactuals": sorted(official_counterfactuals),
            },
        )
        del center_tensor, tensor
        phase = "trace_mask"

    if state.get("torch_centroid_sha256") != torch_center_sha:
        raise ExternalMemoryDBSCANError("torch centroid checkpoint checksum mismatch")
    if state.get("stable_float64_centroid_sha256") != float64_center_sha:
        raise ExternalMemoryDBSCANError(
            "stable float64 centroid checkpoint checksum mismatch"
        )
    first_cf_by_parent = {
        int(parent): int(candidate)
        for parent, candidate in (state.get("first_cf_by_parent") or [])
    }
    official_counterfactuals = set(
        map(int, state.get("official_counterfactuals") or [])
    )
    official_result: tuple[list[int], list[float], list[int]]
    if torch_center_norm < float(theta) and int(recourse_size) > 0:
        official_result = (
            [len(first_cf_by_parent)],
            [torch_center_norm],
            [len(set(first_cf_by_parent.values()))],
        )
    else:
        official_result = ([], [], [])

    numpy_center_path = root / "numpy_centroid.npy"
    checkpointed_numpy_center_sha = state.get("numpy_centroid_sha256")
    if numpy_center_path.exists() and checkpointed_numpy_center_sha:
        if _sha256_file(numpy_center_path) != checkpointed_numpy_center_sha:
            raise ExternalMemoryDBSCANError(
                "NumPy centroid checkpoint checksum mismatch"
            )
        numpy_center = np.load(numpy_center_path, allow_pickle=False)
        if numpy_center.shape != (n_features,) or numpy_center.dtype != recourse_vectors.dtype:
            raise ExternalMemoryDBSCANError("NumPy centroid checkpoint schema mismatch")
    else:
        # The all-zero label mask selects every row in original order.  The
        # direct contiguous memmap view has the same dtype/shape/strides as the
        # legacy advanced-index copy and therefore the same NumPy reduction.
        numpy_center = np.mean(recourse_vectors, axis=0)
        _atomic_npy(numpy_center_path, numpy_center)
    numpy_center_sha = _sha256_file(numpy_center_path)
    numpy_center_norm = float(np.linalg.norm(numpy_center))

    mask_partial = root / "retained_mask.partial.npy"
    mask_final = root / "retained_mask.npy"
    phase = str(state.get("phase"))
    if phase == "trace_mask":
        mask = _open_or_create_memmap(
            mask_partial,
            shape=(n_samples,),
            dtype=np.dtype(np.bool_),
            resume=bool(resume),
        )
        retained_count = int(state.get("retained_count", 0))
        start = int(state.get("next_offset", 0))
        _validate_scan_offset(
            start,
            total=n_samples,
            block_size=int(block_size),
            phase="trace mask",
        )
        replayed_count = _validate_trace_mask_prefix(
            mask=mask,
            recourse_vectors=recourse_vectors,
            center=numpy_center,
            radius=float(radius),
            stop_offset=start,
            block_size=int(block_size),
        )
        if replayed_count != retained_count:
            raise ExternalMemoryDBSCANError(
                "trace-mask checkpoint retained count mismatch"
            )
        for offset in range(start, n_samples, int(block_size)):
            stop = min(n_samples, offset + int(block_size))
            # The legacy trace executes ``float(np_scalar) < float(radius)``;
            # the helper preserves that comparison instead of NumPy 2's
            # value-based Python-scalar promotion.
            retained = _numpy_trace_membership(
                recourse_vectors[offset:stop],
                center=numpy_center,
                radius=float(radius),
            )
            mask[offset:stop] = retained
            retained_count += int(np.count_nonzero(retained))
            _fsync_memmap(mask)
            peak = max(peak, _check_rss(int(max_rss_bytes), phase="one_cluster.trace"))
            state = _summary_checkpoint(
                state_path,
                identity=identity,
                phase="trace_mask",
                next_offset=stop,
                retained_count=retained_count,
                peak_rss_bytes=peak,
                extra={
                    "torch_centroid_sha256": torch_center_sha,
                    "torch_centroid_norm": torch_center_norm,
                    "stable_float64_centroid_sha256": float64_center_sha,
                    "stable_float64_centroid_norm": float64_center_norm,
                    "centroid_max_abs_difference": centroid_max_abs_difference,
                    "official_within_radius_count": int(
                        state["official_within_radius_count"]
                    ),
                    "count_exactly_at_delta": int(state["count_exactly_at_delta"]),
                    "float64_radius_membership_disagreement_count": int(
                        state["float64_radius_membership_disagreement_count"]
                    ),
                    "first_cf_by_parent": sorted(first_cf_by_parent.items()),
                    "official_counterfactuals": sorted(official_counterfactuals),
                    "numpy_centroid_sha256": numpy_center_sha,
                    "numpy_centroid_norm": numpy_center_norm,
                },
            )
        _fsync_memmap(mask)
        mask_sha = _sha256_file(mask_partial)
        del mask
        state = _summary_checkpoint(
            state_path,
            identity=identity,
            phase="trace_mask_ready",
            next_offset=n_samples,
            retained_count=retained_count,
            peak_rss_bytes=peak,
            extra={
                "torch_centroid_sha256": torch_center_sha,
                "torch_centroid_norm": torch_center_norm,
                "stable_float64_centroid_sha256": float64_center_sha,
                "stable_float64_centroid_norm": float64_center_norm,
                "centroid_max_abs_difference": centroid_max_abs_difference,
                "official_within_radius_count": int(
                    state["official_within_radius_count"]
                ),
                "count_exactly_at_delta": int(state["count_exactly_at_delta"]),
                "float64_radius_membership_disagreement_count": int(
                    state["float64_radius_membership_disagreement_count"]
                ),
                "first_cf_by_parent": sorted(first_cf_by_parent.items()),
                "official_counterfactuals": sorted(official_counterfactuals),
                "numpy_centroid_sha256": numpy_center_sha,
                "numpy_centroid_norm": numpy_center_norm,
                "retained_mask_sha256": mask_sha,
            },
        )
        phase = "trace_mask_ready"
    if phase == "trace_mask_ready":
        _promote_summary_array(
            partial=mask_partial,
            final=mask_final,
            shape=(n_samples,),
            dtype=np.dtype(np.bool_),
            expected_sha256=str(state.get("retained_mask_sha256") or ""),
            label="retained mask",
        )
        state = _summary_checkpoint(
            state_path,
            identity=identity,
            phase="retained_centroid",
            next_offset=0,
            retained_count=int(state["retained_count"]),
            peak_rss_bytes=peak,
            extra={key: value for key, value in state.items() if key not in {
                "schema_version", "scientific_identity", "scientific_identity_sha256",
                "phase", "next_offset", "retained_count", "peak_rss_bytes", "updated_at"
            }},
        )
        phase = "retained_centroid"

    if state.get("numpy_centroid_sha256") != numpy_center_sha:
        raise ExternalMemoryDBSCANError("NumPy centroid checkpoint checksum mismatch")
    mask = np.load(mask_final, mmap_mode="r", allow_pickle=False)
    retained_count = int(state.get("retained_count", 0))
    retained_center_path = root / "retained_centroid.npy"
    phase = str(state.get("phase"))
    if phase == "retained_centroid":
        start = int(state.get("next_offset", 0))
        _validate_scan_offset(
            start,
            total=n_samples,
            block_size=int(block_size),
            phase="retained centroid",
        )
        replay_sum, replay_count = _stream_masked_sum_prefix(
            mask=mask,
            recourse_vectors=recourse_vectors,
            stop_offset=start,
            block_size=int(block_size),
            dtype=recourse_vectors.dtype,
        )
        checkpoint_sum = _array_from_hex(
            state.get("retained_sum_hex") or _array_hex(
                np.zeros(n_features, dtype=recourse_vectors.dtype)
            ),
            dtype=recourse_vectors.dtype,
        )
        if (
            not np.array_equal(replay_sum, checkpoint_sum)
            or replay_count != int(state.get("retained_sum_count", 0))
        ):
            raise ExternalMemoryDBSCANError(
                "retained-centroid checkpoint prefix does not replay"
            )
        retained_sum = replay_sum
        summed_count = replay_count
        for offset in range(start, n_samples, int(block_size)):
            stop = min(n_samples, offset + int(block_size))
            selected = np.asarray(
                recourse_vectors[offset:stop][mask[offset:stop]]
            )
            if len(selected):
                retained_sum = np.add(
                    retained_sum,
                    np.sum(selected, axis=0, dtype=recourse_vectors.dtype),
                    dtype=recourse_vectors.dtype,
                )
                summed_count += int(len(selected))
            peak = max(
                peak,
                _check_rss(int(max_rss_bytes), phase="one_cluster.retained_centroid"),
            )
            state = _summary_checkpoint(
                state_path,
                identity=identity,
                phase="retained_centroid",
                next_offset=stop,
                retained_count=retained_count,
                peak_rss_bytes=peak,
                extra={
                    **{key: value for key, value in state.items() if key not in {
                        "schema_version", "scientific_identity", "scientific_identity_sha256",
                        "phase", "next_offset", "retained_count", "peak_rss_bytes", "updated_at",
                        "retained_sum_hex", "retained_sum_count"
                    }},
                    "retained_sum_hex": _array_hex(retained_sum),
                    "retained_sum_count": summed_count,
                },
            )
        if summed_count != retained_count:
            raise ExternalMemoryDBSCANError("retained-centroid count drift")
        retained_center_sha: str | None = None
        if retained_count:
            retained_center = np.asarray(
                retained_sum / np.asarray(retained_count, dtype=recourse_vectors.dtype),
                dtype=recourse_vectors.dtype,
            )
            retained_center_sha = _atomic_npy(retained_center_path, retained_center)
        state = _summary_checkpoint(
            state_path,
            identity=identity,
            phase="retained_medoid",
            next_offset=0,
            retained_count=retained_count,
            peak_rss_bytes=peak,
            extra={
                **{key: value for key, value in state.items() if key not in {
                    "schema_version", "scientific_identity", "scientific_identity_sha256",
                    "phase", "next_offset", "retained_count", "peak_rss_bytes", "updated_at"
                }},
                "retained_centroid_sha256": retained_center_sha,
                "medoid_position": -1,
                "medoid_distance_hex": float("inf").hex(),
                "covered_parents": [],
                "member_counterfactuals": [],
            },
        )
        phase = "retained_medoid"
    if phase == "retained_medoid":
        start = int(state.get("next_offset", 0))
        _validate_scan_offset(
            start,
            total=n_samples,
            block_size=int(block_size),
            phase="retained medoid",
        )
        if retained_count:
            if (
                not retained_center_path.is_file()
                or _sha256_file(retained_center_path)
                != state.get("retained_centroid_sha256")
            ):
                raise ExternalMemoryDBSCANError(
                    "retained streaming centroid checksum mismatch"
                )
            retained_center = np.load(retained_center_path, allow_pickle=False)
            (
                replay_winner,
                replay_distance,
                replay_parents,
                replay_candidates,
            ) = _scan_retained_medoid_prefix(
                mask=mask,
                recourse_vectors=recourse_vectors,
                pair_indices=pair_indices,
                center=retained_center,
                stop_offset=start,
                block_size=int(block_size),
            )
            if (
                replay_winner != int(state.get("medoid_position", -1))
                or replay_distance.hex() != state.get("medoid_distance_hex")
                or sorted(replay_parents) != state.get("covered_parents")
                or sorted(replay_candidates) != state.get("member_counterfactuals")
            ):
                raise ExternalMemoryDBSCANError(
                    "retained-medoid checkpoint prefix does not replay"
                )
            winner = replay_winner
            winner_distance = replay_distance
            covered_parents = replay_parents
            member_counterfactuals = replay_candidates
            for offset in range(start, n_samples, int(block_size)):
                stop = min(n_samples, offset + int(block_size))
                local = np.flatnonzero(mask[offset:stop])
                if len(local):
                    physical = local.astype(np.int64, copy=False) + offset
                    selected_vectors = np.asarray(
                        recourse_vectors[offset:stop][local]
                    )
                    distances = np.linalg.norm(
                        selected_vectors - retained_center, axis=1
                    )
                    local_winner = int(np.argmin(distances))
                    local_distance = float(distances[local_winner])
                    if local_distance < winner_distance:
                        winner = int(physical[local_winner])
                        winner_distance = local_distance
                    pairs = pair_indices[physical]
                    covered_parents.update(
                        map(int, np.unique(pairs[:, 0]).tolist())
                    )
                    member_counterfactuals.update(
                        map(int, np.unique(pairs[:, 1]).tolist())
                    )
                peak = max(
                    peak,
                    _check_rss(int(max_rss_bytes), phase="one_cluster.retained_medoid"),
                )
                state = _summary_checkpoint(
                    state_path,
                    identity=identity,
                    phase="retained_medoid",
                    next_offset=stop,
                    retained_count=retained_count,
                    peak_rss_bytes=peak,
                    extra={
                        **{key: value for key, value in state.items() if key not in {
                            "schema_version", "scientific_identity", "scientific_identity_sha256",
                            "phase", "next_offset", "retained_count", "peak_rss_bytes", "updated_at",
                            "medoid_position", "medoid_distance_hex", "covered_parents",
                            "member_counterfactuals"
                        }},
                        "medoid_position": winner,
                        "medoid_distance_hex": winner_distance.hex(),
                        "covered_parents": sorted(covered_parents),
                        "member_counterfactuals": sorted(member_counterfactuals),
                    },
                )
        else:
            winner = -1
            winner_distance = float("inf")
            covered_parents = set()
            member_counterfactuals = set()
        state = _summary_checkpoint(
            state_path,
            identity=identity,
            phase="finalize",
            next_offset=n_samples,
            retained_count=retained_count,
            peak_rss_bytes=peak,
            extra={key: value for key, value in state.items() if key not in {
                "schema_version", "scientific_identity", "scientific_identity_sha256",
                "phase", "next_offset", "retained_count", "peak_rss_bytes", "updated_at"
            }},
        )
        phase = "finalize"
    if phase == "finalize":
        state = _summary_checkpoint(
            state_path,
            identity=identity,
            phase="finalize",
            next_offset=n_samples,
            retained_count=retained_count,
            peak_rss_bytes=peak,
            extra={key: value for key, value in state.items() if key not in {
                "schema_version", "scientific_identity", "scientific_identity_sha256",
                "phase", "next_offset", "retained_count", "peak_rss_bytes", "updated_at"
            }},
        )
        phase = "finalize"
    if phase != "finalize":
        raise ExternalMemoryDBSCANError(f"unknown one-cluster phase: {phase}")

    # A checkpoint is only a progress hint.  Recompute every scientific prefix
    # before publishing PASS so a skipped/tampered offset can never turn an
    # unwritten zero-filled suffix into a valid result.
    if (
        (
            pair_path is not None
            and _sha256_file(pair_path) != str(pairs_sha256)
        )
        or (
            pair_authority_path is not None
            and _sha256_file(pair_authority_path)
            != str(pair_authority_manifest_sha256)
        )
        or _sha256_file(vector_path) != dbscan_identity.get("vectors_sha256")
        or _sha256_file(dbscan_path) != str(dbscan_manifest_sha256)
    ):
        raise ExternalMemoryDBSCANError(
            "one-cluster source changed during summary replay"
        )
    _validate_exact_one_cluster_source(
        dbscan_manifest_path=dbscan_path,
        dbscan_manifest_sha256=dbscan_manifest_sha256,
        recourse_vectors=recourse_vectors,
    )
    (
        replayed_first,
        replayed_counterfactuals,
        replayed_official_within,
        replayed_exact_delta,
        replayed_radius_disagreements,
    ) = _scan_torch_coverage_audit_prefix(
        recourse_vectors=recourse_vectors,
        pair_indices=pair_indices,
        center=torch_center,
        stable_float64_center=float64_center,
        radius=float(radius),
        stop_offset=n_samples,
        block_size=int(block_size),
        torch_module=torch_module,
    )
    if (
        replayed_first != first_cf_by_parent
        or replayed_counterfactuals != official_counterfactuals
        or replayed_official_within
        != int(state.get("official_within_radius_count", -1))
        or replayed_exact_delta != int(state.get("count_exactly_at_delta", -1))
        or replayed_radius_disagreements
        != int(state.get("float64_radius_membership_disagreement_count", -1))
    ):
        raise ExternalMemoryDBSCANError(
            "terminal torch-coverage replay mismatch"
        )
    replayed_retained_count = _validate_trace_mask_prefix(
        mask=mask,
        recourse_vectors=recourse_vectors,
        center=numpy_center,
        radius=float(radius),
        stop_offset=n_samples,
        block_size=int(block_size),
    )
    if replayed_retained_count != retained_count:
        raise ExternalMemoryDBSCANError("terminal trace-mask coverage mismatch")
    terminal_sum, terminal_sum_count = _stream_masked_sum_prefix(
        mask=mask,
        recourse_vectors=recourse_vectors,
        stop_offset=n_samples,
        block_size=int(block_size),
        dtype=recourse_vectors.dtype,
    )
    if terminal_sum_count != retained_count:
        raise ExternalMemoryDBSCANError("terminal retained-centroid count mismatch")

    selected: list[dict[str, Any]] = []
    retained_center_path = root / "retained_centroid.npy"
    covered_parents: set[int] = set()
    member_counterfactuals: set[int] = set()
    winner = -1
    winner_distance = float("inf")
    if retained_count:
        retained_center = np.asarray(
            terminal_sum
            / np.asarray(retained_count, dtype=recourse_vectors.dtype),
            dtype=recourse_vectors.dtype,
        )
        if (
            not retained_center_path.is_file()
            or _sha256_file(retained_center_path)
            != state.get("retained_centroid_sha256")
            or not np.array_equal(
                np.load(retained_center_path, allow_pickle=False), retained_center
            )
        ):
            raise ExternalMemoryDBSCANError(
                "terminal retained streaming centroid mismatch"
            )
        (
            winner,
            winner_distance,
            covered_parents,
            member_counterfactuals,
        ) = _scan_retained_medoid_prefix(
            mask=mask,
            recourse_vectors=recourse_vectors,
            pair_indices=pair_indices,
            center=retained_center,
            stop_offset=n_samples,
            block_size=int(block_size),
        )
        if (
            winner < 0
            or winner != int(state.get("medoid_position", -1))
            or winner_distance.hex() != state.get("medoid_distance_hex")
            or sorted(covered_parents) != state.get("covered_parents")
            or sorted(member_counterfactuals) != state.get("member_counterfactuals")
        ):
            raise ExternalMemoryDBSCANError(
                "terminal retained-medoid replay mismatch"
            )
        filtered = numpy_center_norm < float(theta) and bool(covered_parents)
        if filtered and int(recourse_size) > 0:
            selection = official_greedy(
                counterfactual_covering={0: set(covered_parents)},
                graphs_covered_by={parent: {0} for parent in covered_parents},
                k=1,
            )
            if selection != {1: (0, len(covered_parents))}:
                raise ExternalMemoryDBSCANError(
                    "official one-cluster greedy result changed"
                )
            source_index, counterfactual_index = pair_indices[int(winner)]
            selected = [
                {
                    "rank": 1,
                    "selected_rank": 1,
                    "cluster_label": 0,
                    "cluster_id": 0,
                    "cluster_center_norm": numpy_center_norm,
                    "centroid_norm": numpy_center_norm,
                    "cluster_radius": float(radius),
                    "cluster_size": n_samples,
                    "representative_source_index": int(source_index),
                    "representative_counterfactual_index": int(counterfactual_index),
                    "representative_distance_to_center": winner_distance,
                    "covered_parent_indices_native": sorted(covered_parents),
                    "native_cumulative_covered_count": len(covered_parents),
                    "cumulative_covered_count": len(covered_parents),
                    "native_cumulative_cost": numpy_center_norm,
                    "member_counterfactual_indices": sorted(member_counterfactuals),
                    "representative_candidate_ids": [
                        int(counterfactual_index)
                    ],
                }
            ]
    artifacts = {
        "retained_mask_path": str(mask_final),
        "retained_mask_sha256": _sha256_file(mask_final),
        "retained_positions_path": None,
        "retained_positions_sha256": None,
        "retained_vectors_path": None,
        "retained_vectors_sha256": None,
        "torch_centroid_path": str(torch_center_path),
        "torch_centroid_sha256": torch_center_sha,
        "stable_float64_centroid_path": str(float64_center_path),
        "stable_float64_centroid_sha256": float64_center_sha,
        "numpy_centroid_path": str(numpy_center_path),
        "numpy_centroid_sha256": numpy_center_sha,
        "retained_centroid_path": (
            str(retained_center_path) if retained_count else None
        ),
        "retained_centroid_sha256": (
            _sha256_file(retained_center_path) if retained_count else None
        ),
    }
    manifest = {
        "schema_version": ONE_CLUSTER_SUMMARY_SCHEMA,
        "run_complete": True,
        "scientific_identity": identity,
        "scientific_identity_sha256": identity_hash,
        "exact_one_cluster_semantics_replayed": True,
        "official_coverage_function_invoked": False,
        "official_coverage_semantics_derived_for_single_label_zero": True,
        "official_greedy_invoked_for_trace": bool(selected),
        "legacy_torch_reduction_order_preserved": True,
        "legacy_numpy_reduction_order_preserved": True,
        "retained_streaming_reduction_order": (
            "fixed_global_row_order_with_fixed_block_size"
        ),
        "strict_radius_comparison_preserved": True,
        "radius_filter_operator": "<",
        "centroid_norm_filter_operator": "<",
        "candidate_major_parent_minor_order_preserved": True,
        "medoid_first_argmin_tie_order_preserved": True,
        "greedy_tie_break": "ascending_canonical_cluster_id",
        "approximation_used": False,
        "num_samples": n_samples,
        "cluster_member_count": n_samples,
        "retained_count": retained_count,
        "within_centroid_radius_count": replayed_official_within,
        "outside_centroid_radius_count": n_samples - replayed_official_within,
        "count_exactly_at_delta": replayed_exact_delta,
        "float64_radius_membership_disagreement_count": (
            replayed_radius_disagreements
        ),
        "torch_centroid_norm": torch_center_norm,
        "centroid_norm": torch_center_norm,
        "stable_float64_centroid_norm": float64_center_norm,
        "centroid_max_abs_difference": centroid_max_abs_difference,
        "centroid_norm_decision_disagreement": centroid_norm_decision_disagreement,
        "numpy_centroid_norm": numpy_center_norm,
        "centroid_norm_lt_theta": torch_center_norm < float(theta),
        "count_exactly_at_theta": int(torch_center_norm == float(theta)),
        "trace_numpy_centroid_norm_lt_theta": numpy_center_norm < float(theta),
        "trace_numpy_count_exactly_at_theta": int(
            numpy_center_norm == float(theta)
        ),
        "coverage_pair_orientation": "col0_parent_col1_candidate",
        "official_covered_parent_indices": sorted(first_cf_by_parent),
        "official_first_counterfactual_indices": sorted(
            set(first_cf_by_parent.values())
        ),
        "official_radius_counterfactual_indices": sorted(
            official_counterfactuals
        ),
        "official_parent_to_covering_clusters": {
            str(parent): [0] for parent in sorted(first_cf_by_parent)
        },
        "covered_parent_indices": sorted(covered_parents),
        "counterfactual_indices": sorted(member_counterfactuals),
        "parent_to_covering_clusters": {
            str(parent): [0] for parent in sorted(covered_parents)
        },
        "selected_common_recourse_count": len(selected),
        "large_retained_arrays_materialized": False,
        "retained_vector_bytes_materialized": 0,
        "retained_position_bytes_materialized": 0,
        "storage_bytes_avoided": retained_count * (row_bytes + 8),
        "official_result": [list(part) for part in official_result],
        "selected": selected,
        "peak_rss_bytes_observed": max(peak, _rss_bytes()),
        "max_rss_bytes": int(max_rss_bytes),
        **artifacts,
        "completed_at": _utc_now(),
    }
    if int(manifest["peak_rss_bytes_observed"]) > int(max_rss_bytes):
        raise ExternalMemoryDBSCANError("one-cluster summary peak RSS exceeded budget")
    _atomic_json(manifest_path, manifest)
    return _validate_one_cluster_summary_manifest(
        manifest_path=manifest_path, manifest=manifest, identity=identity
    )


def invoke_official_coverage_summary_external(
    *,
    labels: np.ndarray,
    recourse_vectors: np.ndarray,
    pair_indices: np.ndarray,
    radius: float,
    theta: float,
    recourse_size: int,
    official_coverage_summary: Callable[..., Any],
    torch_module: Any,
    max_rss_bytes: int,
) -> tuple[Any, dict[str, Any]]:
    """Invoke the pinned upstream function without copying the full vector matrix."""

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest = int(counts.max()) if counts.size else 0
    row_bytes = int(recourse_vectors.shape[1]) * int(recourse_vectors.dtype.itemsize)
    estimate = largest * row_bytes * 3 + len(labels) * 24 + 128 * 1024**2
    _check_rss(max_rss_bytes, phase="official_coverage", reserved_bytes=estimate)
    proxy = SimpleNamespace(labels_=labels)
    # ``from_numpy`` is a zero-copy view.  The official function performs the
    # same per-cluster boolean indexing/torch reductions as the legacy route.
    tensor = torch_module.from_numpy(recourse_vectors)
    result = official_coverage_summary(
        db_2=proxy,
        rec=tensor,
        idxs=pair_indices,
        radius=float(radius),
        threshold_theta=float(theta),
        recourse_size=int(recourse_size),
    )
    peak = _check_rss(max_rss_bytes, phase="official_coverage.complete")
    return result, {
        "schema_version": SUMMARY_SCHEMA,
        "official_coverage_summary_invoked": True,
        "full_vector_tensor_copy_created": False,
        "largest_cluster_size": largest,
        "cluster_count": int(unique.size),
        "peak_rss_bytes_observed": peak,
    }


def trace_external_cluster_order(
    *,
    labels: np.ndarray,
    recourse_vectors: np.ndarray,
    pair_indices: np.ndarray,
    radius: float,
    theta: float,
    recourse_size: int,
    official_greedy: Callable[..., Any],
    max_rss_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """External-array equivalent of ``trace_official_cluster_order``."""

    if (
        labels.ndim != 1
        or recourse_vectors.ndim != 2
        or pair_indices.ndim != 2
        or pair_indices.shape[1] != 2
        or len(labels) != len(recourse_vectors)
        or len(labels) != len(pair_indices)
    ):
        raise ExternalMemoryDBSCANError("cluster arrays are not exactly aligned")
    common_recourse: dict[int, set[int]] = {}
    centroid_norms: dict[int, float] = {}
    cluster_has_retained_pairs: dict[int, bool] = {}
    cluster_sizes: dict[int, int] = {}
    peak = _rss_bytes()
    max_label = int(labels.max()) if labels.size else -1
    for cluster_label in range(max_label + 1):
        positions = np.flatnonzero(labels == cluster_label)
        cluster_sizes[cluster_label] = int(len(positions))
        if positions.size == 0:
            continue
        estimate = (
            positions.nbytes
            + int(positions.size)
            * int(recourse_vectors.shape[1])
            * int(recourse_vectors.dtype.itemsize)
            * 3
            + 64 * 1024**2
        )
        _check_rss(
            max_rss_bytes,
            phase=f"trace_cluster_{cluster_label}",
            reserved_bytes=estimate,
        )
        # This is intentionally the same advanced-index copy and NumPy
        # reduction used by the legacy implementation, but only one cluster is
        # resident at a time.
        points = recourse_vectors[positions]
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        covered: set[int] = set()
        has_retained = False
        for local_index, distance in enumerate(distances):
            if float(distance) < float(radius):
                pair = pair_indices[int(positions[local_index])]
                parent = int(pair[0])
                covered.add(parent)
                has_retained = True
        common_recourse[cluster_label] = covered
        centroid_norms[cluster_label] = float(np.linalg.norm(centroid))
        cluster_has_retained_pairs[cluster_label] = has_retained
        peak = max(peak, _check_rss(max_rss_bytes, phase="trace_cluster.complete"))

    filtered = {
        label: set(parents)
        for label, parents in common_recourse.items()
        if centroid_norms[label] < float(theta)
        and parents
        and cluster_has_retained_pairs[label]
    }
    covered_by: dict[int, set[int]] = defaultdict(set)
    for label, parents in filtered.items():
        for parent in parents:
            covered_by[parent].add(label)
    if not filtered:
        return [], {
            "schema_version": SUMMARY_SCHEMA,
            "cluster_count": max_label + 1,
            "selected_count": 0,
            "peak_rss_bytes_observed": peak,
            "legacy_numpy_reduction_order_preserved": True,
        }
    selection = official_greedy(
        counterfactual_covering={label: set(values) for label, values in filtered.items()},
        graphs_covered_by={key: set(values) for key, values in covered_by.items()},
        k=min(int(recourse_size), len(filtered)),
    )
    ordered: list[dict[str, Any]] = []
    cumulative_cost = 0.0
    covered: set[int] = set()
    for rank, value in selection.items():
        cluster_label = int(value[0])
        positions = np.flatnonzero(labels == cluster_label)
        points = recourse_vectors[positions]
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        retained_positions = np.flatnonzero(distances < float(radius))
        if retained_positions.size == 0:
            raise ExternalMemoryDBSCANError(
                "selected cluster lost all strict-radius members"
            )
        retained_vectors = points[retained_positions]
        retained_pairs = pair_indices[positions[retained_positions]]
        retained_centroid = np.mean(retained_vectors, axis=0)
        retained_distances = np.linalg.norm(
            retained_vectors - retained_centroid, axis=1
        )
        winner = int(np.argmin(retained_distances))
        source_index, counterfactual_index = retained_pairs[winner]
        medoid_distance = float(retained_distances[winner])
        member_counterfactuals = sorted(
            {int(value) for value in retained_pairs[:, 1].tolist()}
        )
        covered.update(filtered[cluster_label])
        cumulative_cost += centroid_norms[cluster_label]
        ordered.append(
            {
                "rank": int(rank),
                "selected_rank": int(rank),
                "cluster_label": cluster_label,
                "cluster_id": cluster_label,
                "cluster_center_norm": centroid_norms[cluster_label],
                "centroid_norm": centroid_norms[cluster_label],
                "cluster_radius": float(radius),
                "cluster_size": cluster_sizes[cluster_label],
                "representative_source_index": int(source_index),
                "representative_counterfactual_index": int(counterfactual_index),
                "representative_distance_to_center": medoid_distance,
                "covered_parent_indices_native": sorted(filtered[cluster_label]),
                "native_cumulative_covered_count": len(covered),
                "cumulative_covered_count": len(covered),
                "native_cumulative_cost": cumulative_cost,
                "member_counterfactual_indices": member_counterfactuals,
                "representative_candidate_ids": [int(counterfactual_index)],
            }
        )
    return ordered, {
        "schema_version": SUMMARY_SCHEMA,
        "cluster_count": max_label + 1,
        "selected_count": len(ordered),
        "peak_rss_bytes_observed": max(peak, _rss_bytes()),
        "legacy_numpy_reduction_order_preserved": True,
        "official_greedy_invoked": True,
        "cluster_order": "ascending_sklearn_label",
    }


__all__ = [
    "AdoptedPairStoreResult",
    "ExactOneClusterSummaryResult",
    "ExternalPairStore",
    "ONE_CLUSTER_SUMMARY_SCHEMA",
    "PAIR_STORE_ADOPTION_SCHEMA",
    "PAIR_STORE_SCHEMA",
    "PairStoreResult",
    "SUMMARY_SCHEMA",
    "invoke_official_coverage_summary_external",
    "adopt_external_pair_store_read_only",
    "summarize_proven_one_cluster_external",
    "trace_external_cluster_order",
    "validate_adopted_pair_store_read_only",
    "validate_proven_one_cluster_summary",
]
