"""Bounded-memory downstream replay for exact all-core DBSCAN components.

This module is intentionally separate from :mod:`external_memory_dbscan`.
The latter proves the partition; this module consumes only a hash-closed
all-core component-recovery result and reproduces COMRECGC's centroid,
strict-radius coverage, greedy selection, and representative lineage without
advanced-indexing an entire production cluster.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
    ExternalMemoryDBSCANError,
    _check_rss,
    _rss_bytes,
    _validate_component_recovery_closure,
)


ALL_CORE_COMPONENT_SUMMARY_SCHEMA = (
    "comrecgc_exact_all_core_component_streaming_summary_v1"
)
_CHECKPOINT_SCHEMA = "comrecgc_exact_all_core_component_summary_checkpoint_v1"
_OWNER_CLAIM_SCHEMA = "comrecgc_exact_component_summary_owner_claim_v1"
_WRITER_LOCK_SCHEMA = "comrecgc_exact_component_summary_writer_lock_v1"


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


def _atomic_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    writer_guard: Callable[[], None] | None = None,
) -> None:
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
        if writer_guard is not None:
            writer_guard()
        os.replace(temporary, path)
        if writer_guard is not None:
            writer_guard()
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(
    path: Path,
    values: np.ndarray,
    *,
    writer_guard: Callable[[], None] | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        if writer_guard is not None:
            writer_guard()
        os.replace(temporary, path)
        result = _sha256_file(path)
        if writer_guard is not None:
            writer_guard()
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalMemoryDBSCANError(
            f"invalid all-core summary JSON artifact: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ExternalMemoryDBSCANError(
            f"expected all-core summary JSON object: {path}"
        )
    return value


def _load_regular_json_nofollow(path: Path) -> dict[str, Any]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ExternalMemoryDBSCANError(
            "all-core summary requires O_NOFOLLOW artifact reads"
        )
    if path.is_symlink():
        raise ExternalMemoryDBSCANError(
            f"all-core summary JSON artifact is a symlink: {path}"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | int(nofollow))
    except OSError as exc:
        raise ExternalMemoryDBSCANError(
            f"all-core summary JSON artifact cannot be opened: {path}"
        ) from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
            or descriptor_stat.st_size > 4 * 1024 * 1024
        ):
            raise ExternalMemoryDBSCANError(
                f"all-core summary JSON artifact inode mismatch: {path}"
            )
        chunks: list[bytes] = []
        remaining = int(descriptor_stat.st_size)
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = json.loads(b"".join(chunks))
    except ExternalMemoryDBSCANError:
        raise
    except Exception as exc:
        raise ExternalMemoryDBSCANError(
            f"all-core summary JSON artifact is invalid: {path}"
        ) from exc
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise ExternalMemoryDBSCANError(
            f"expected all-core summary JSON object: {path}"
        )
    return value


def _root_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
    }


def _current_writer_lock_identity(root: Path) -> dict[str, Any]:
    lock_path = root / ".writer.lock"
    receipt = _load_regular_json_nofollow(lock_path)
    value = os.stat(lock_path, follow_symlinks=False)
    expected = {
        "schema_version": _WRITER_LOCK_SCHEMA,
        "root": str(root),
        "root_stat_identity": _root_stat(root),
        "lock_stat_identity": {
            "device": int(value.st_dev),
            "inode": int(value.st_ino),
            "mode": int(value.st_mode),
        },
    }
    if receipt != expected:
        raise ExternalMemoryDBSCANError(
            "all-core summary writer lock terminal identity mismatch"
        )
    return expected


def _claim_summary_root(
    root: Path, *, identity_sha256: str, resume: bool
) -> dict[str, Any]:
    """Atomically claim a fresh writer root or reopen its inode-bound claim."""

    claim_path = root / "owner_claim.json"
    if resume:
        if (
            root.is_symlink()
            or not root.is_dir()
            or claim_path.is_symlink()
            or not claim_path.exists()
        ):
            raise ExternalMemoryDBSCANError(
                "all-core summary resume root has no owner claim"
            )
        claim = _load_regular_json_nofollow(claim_path)
        if (
            claim.get("schema_version") != _OWNER_CLAIM_SCHEMA
            or claim.get("root") != str(root)
            or claim.get("root_stat_identity") != _root_stat(root)
            or claim.get("scientific_identity_sha256") != identity_sha256
        ):
            raise ExternalMemoryDBSCANError(
                "all-core summary owner claim mismatch"
            )
        return claim
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"all-core component summary root is already claimed: {root}"
        ) from exc
    claim = {
        "schema_version": _OWNER_CLAIM_SCHEMA,
        "root": str(root),
        "root_stat_identity": _root_stat(root),
        "scientific_identity_sha256": str(identity_sha256),
        "claimed_at": _utc_now(),
    }
    _atomic_json(claim_path, claim)
    return claim


def _validate_owner_claim(root: Path, *, identity_sha256: str) -> dict[str, Any]:
    return _claim_summary_root(
        root, identity_sha256=identity_sha256, resume=True
    )


def _revoke_terminal_manifest(root: Path) -> None:
    """Remove a PASS marker if writer-lock continuity can no longer be proven."""

    manifest = root / "run_manifest.json"
    if manifest.is_file() and not manifest.is_symlink():
        manifest.unlink()
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class _HeldWriterLock:
    descriptor: int
    lock_path: Path
    root: Path
    root_stat_identity: Mapping[str, int]
    lock_stat_identity: Mapping[str, int]

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": _WRITER_LOCK_SCHEMA,
            "root": str(self.root),
            "root_stat_identity": dict(self.root_stat_identity),
            "lock_stat_identity": dict(self.lock_stat_identity),
        }

    def verify(self) -> None:
        """Prove the held descriptor is still the published lock inode."""

        try:
            descriptor_stat = os.fstat(self.descriptor)
            path_stat = os.stat(self.lock_path, follow_symlinks=False)
        except OSError as exc:
            raise ExternalMemoryDBSCANError(
                "all-core summary writer lock disappeared while held"
            ) from exc
        observed_lock = {
            "device": int(descriptor_stat.st_dev),
            "inode": int(descriptor_stat.st_ino),
            "mode": int(descriptor_stat.st_mode),
        }
        path_lock = {
            "device": int(path_stat.st_dev),
            "inode": int(path_stat.st_ino),
            "mode": int(path_stat.st_mode),
        }
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or observed_lock != dict(self.lock_stat_identity)
            or path_lock != dict(self.lock_stat_identity)
            or _root_stat(self.root) != dict(self.root_stat_identity)
        ):
            raise ExternalMemoryDBSCANError(
                "all-core summary writer lock inode changed while held"
            )
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        try:
            receipt = json.loads(os.read(self.descriptor, 64 * 1024))
        except Exception as exc:
            raise ExternalMemoryDBSCANError(
                "all-core summary writer lock receipt changed while held"
            ) from exc
        expected = self.identity()
        if receipt != expected:
            raise ExternalMemoryDBSCANError(
                "all-core summary writer lock receipt changed while held"
            )


@contextmanager
def _exclusive_summary_writer_lock(
    root: Path,
) -> Iterator[tuple[bool, _HeldWriterLock]]:
    """Hold an inode-bound, nonblocking writer lock for the full invocation.

    The directory is exclusively created for a fresh invocation.  Resumes
    must reopen the already-created regular lock inode with ``O_NOFOLLOW``;
    they never create a replacement inode.  This closes both fresh-writer and
    concurrent-resume races around checkpoint/artifact publication.
    """

    root.parent.mkdir(parents=True, exist_ok=True)
    fresh = False
    try:
        root.mkdir()
        fresh = True
    except FileExistsError:
        if root.is_symlink() or not root.is_dir():
            raise ExternalMemoryDBSCANError(
                "all-core component summary root is not a regular directory"
            )
    lock_path = root / ".writer.lock"
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ExternalMemoryDBSCANError(
            "all-core component summary requires O_NOFOLLOW writer locks"
        )
    flags = os.O_RDWR | int(nofollow)
    if fresh:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileNotFoundError as exc:
        raise ExternalMemoryDBSCANError(
            "all-core summary resume root has no writer lock"
        ) from exc
    except FileExistsError as exc:
        raise ExternalMemoryDBSCANError(
            "all-core summary fresh writer lock already exists"
        ) from exc
    except OSError as exc:
        raise ExternalMemoryDBSCANError(
            "all-core summary writer lock cannot be opened safely"
        ) from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(lock_path, follow_symlinks=False)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
        ):
            raise ExternalMemoryDBSCANError(
                "all-core summary writer lock inode mismatch"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExternalMemoryDBSCANError(
                "all-core summary writer lock is already held"
            ) from exc
        lock_stat_identity = {
            "device": int(descriptor_stat.st_dev),
            "inode": int(descriptor_stat.st_ino),
            "mode": int(descriptor_stat.st_mode),
        }
        lock_identity = {
            "schema_version": _WRITER_LOCK_SCHEMA,
            "root": str(root),
            "root_stat_identity": _root_stat(root),
            "lock_stat_identity": lock_stat_identity,
        }
        if fresh:
            encoded = (
                json.dumps(lock_identity, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        else:
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                observed = json.loads(os.read(descriptor, 64 * 1024))
            except Exception as exc:
                raise ExternalMemoryDBSCANError(
                    "all-core summary writer lock receipt is invalid"
                ) from exc
            if observed != lock_identity:
                raise ExternalMemoryDBSCANError(
                    "all-core summary writer lock receipt mismatch"
                )
        held = _HeldWriterLock(
            descriptor=descriptor,
            lock_path=lock_path,
            root=root,
            root_stat_identity=lock_identity["root_stat_identity"],
            lock_stat_identity=lock_stat_identity,
        )
        held.verify()
        yield fresh, held
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _array_hex(values: np.ndarray) -> list[list[str]]:
    matrix = np.asarray(values)
    return [[float(value).hex() for value in row] for row in matrix.tolist()]


def _array_from_hex(
    values: Sequence[Sequence[str]], *, dtype: np.dtype[Any]
) -> np.ndarray:
    return np.asarray(
        [[float.fromhex(str(value)) for value in row] for row in values],
        dtype=dtype,
    )


def _float_hex(values: Sequence[float]) -> list[str]:
    return [float(value).hex() for value in values]


def _float_from_hex(values: Sequence[str]) -> list[float]:
    return [float.fromhex(str(value)) for value in values]


def _phase_genesis(*, phase: str, identity_sha256: str) -> str:
    return _stable_hash(
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "phase": str(phase),
            "scientific_identity_sha256": str(identity_sha256),
            "entry": "GENESIS",
        }
    )


def _new_ledger(*, phase: str, identity_sha256: str) -> dict[str, Any]:
    return {
        "phase": str(phase),
        "entries": [],
        "committed_offset": 0,
        "head_sha256": _phase_genesis(
            phase=phase, identity_sha256=identity_sha256
        ),
    }


def _append_ledger(
    ledger: dict[str, Any],
    *,
    start: int,
    stop: int,
    payload: Mapping[str, Any],
    identity_sha256: str,
) -> None:
    if int(ledger.get("committed_offset", -1)) != int(start) or stop <= start:
        raise ExternalMemoryDBSCANError(
            "all-core summary progress ledger is noncontiguous"
        )
    core = {
        "phase": str(ledger["phase"]),
        "start": int(start),
        "stop": int(stop),
        "previous_sha256": str(ledger["head_sha256"]),
        "payload_sha256": _stable_hash(payload),
        "scientific_identity_sha256": str(identity_sha256),
    }
    entry = {**core, "entry_sha256": _stable_hash(core)}
    ledger["entries"].append(entry)
    ledger["committed_offset"] = int(stop)
    ledger["head_sha256"] = entry["entry_sha256"]


def _validate_ledger(
    ledger: Mapping[str, Any],
    *,
    phase: str,
    identity_sha256: str,
    total: int,
    block_size: int,
    payload: Mapping[str, Any],
) -> None:
    if (
        ledger.get("phase") != phase
        or not isinstance(ledger.get("entries"), list)
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary progress ledger schema mismatch"
        )
    previous = _phase_genesis(
        phase=phase, identity_sha256=identity_sha256
    )
    offset = 0
    for entry in ledger["entries"]:
        if not isinstance(entry, Mapping):
            raise ExternalMemoryDBSCANError(
                "all-core summary progress ledger entry is invalid"
            )
        core = dict(entry)
        observed = core.pop("entry_sha256", None)
        if (
            int(core.get("start", -1)) != offset
            or int(core.get("stop", -1)) <= offset
            or int(core.get("stop", -1)) > int(total)
            or (
                int(core.get("stop", -1)) != int(total)
                and int(core.get("stop", -1)) % int(block_size) != 0
            )
            or core.get("phase") != phase
            or core.get("previous_sha256") != previous
            or core.get("scientific_identity_sha256") != identity_sha256
            or observed != _stable_hash(core)
        ):
            raise ExternalMemoryDBSCANError(
                "all-core summary progress ledger chain mismatch"
            )
        offset = int(core["stop"])
        previous = str(observed)
    if (
        int(ledger.get("committed_offset", -1)) != offset
        or ledger.get("head_sha256") != previous
        or offset < 0
        or offset > int(total)
        or (
            ledger["entries"]
            and ledger["entries"][-1].get("payload_sha256")
            != _stable_hash(payload)
        )
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary progress ledger closure mismatch"
        )


def _write_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    phase: str,
    next_offset: int,
    phase_payload: Mapping[str, Any],
    ledger: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    peak_rss_bytes: int,
    writer_guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if writer_guard is not None:
        writer_guard()
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "scientific_identity": dict(identity),
        "scientific_identity_sha256": _stable_hash(identity),
        "phase": str(phase),
        "next_offset": int(next_offset),
        "phase_payload": dict(phase_payload),
        "progress_ledger": dict(ledger),
        "progress_ledger_sha256": _stable_hash(ledger),
        "artifacts": dict(artifacts),
        "peak_rss_bytes": int(peak_rss_bytes),
        "updated_at": _utc_now(),
    }
    unsigned = dict(payload)
    payload["checkpoint_payload_sha256"] = _stable_hash(unsigned)
    _atomic_json(path, payload, writer_guard=writer_guard)
    if writer_guard is not None:
        writer_guard()
    return payload


def _load_checkpoint(
    path: Path, *, identity: Mapping[str, Any], total: int
) -> dict[str, Any]:
    state = _load_object(path)
    unsigned = dict(state)
    observed = unsigned.pop("checkpoint_payload_sha256", None)
    if (
        state.get("schema_version") != _CHECKPOINT_SCHEMA
        or state.get("scientific_identity") != dict(identity)
        or state.get("scientific_identity_sha256") != _stable_hash(identity)
        or state.get("progress_ledger_sha256")
        != _stable_hash(state.get("progress_ledger"))
        or observed != _stable_hash(unsigned)
        or int(state.get("next_offset", -1)) < 0
        or int(state.get("next_offset", -1)) > int(total)
        or int(state.get("next_offset", -1))
        != int(state.get("progress_ledger", {}).get("committed_offset", -2))
        or (
            int(state.get("next_offset", -1)) != int(total)
            and int(state.get("next_offset", -1))
            % int(identity.get("block_size", 0))
            != 0
        )
        or not isinstance(state.get("phase_payload"), Mapping)
        or not isinstance(state.get("artifacts"), Mapping)
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary checkpoint identity/closure mismatch"
        )
    return state


def _fsync_memmap(values: np.memmap) -> None:
    values.flush()
    filename = getattr(values, "filename", None)
    if filename is None:
        raise ExternalMemoryDBSCANError(
            "all-core summary memmap lost its backing file"
        )
    with Path(filename).open("rb") as handle:
        os.fsync(handle.fileno())


def _reconcile_array(
    *,
    partial: Path,
    final: Path,
    expected_sha256: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
    writer_guard: Callable[[], None] | None = None,
) -> Path:
    if partial.exists() and final.exists():
        raise ExternalMemoryDBSCANError(
            f"all-core summary {label} has partial and final artifacts"
        )
    source = final if final.exists() else partial
    if not source.is_file() or source.is_symlink():
        raise ExternalMemoryDBSCANError(
            f"all-core summary {label} artifact is absent"
        )
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    if values.shape != shape or values.dtype != dtype:
        raise ExternalMemoryDBSCANError(
            f"all-core summary {label} schema mismatch"
        )
    del values
    if _sha256_file(source) != str(expected_sha256):
        raise ExternalMemoryDBSCANError(
            f"all-core summary {label} checksum mismatch"
        )
    if source == partial:
        if writer_guard is not None:
            writer_guard()
        os.replace(partial, final)
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if writer_guard is not None:
            writer_guard()
    return final


def _stream_label_counts_and_minima(
    labels: np.ndarray,
    *,
    cluster_count: int,
    block_size: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Count canonical labels without materializing the 92M-row label array."""

    counts = np.zeros(int(cluster_count), dtype=np.int64)
    minima = [-1] * int(cluster_count)
    for offset in range(0, len(labels), int(block_size)):
        stop = min(len(labels), offset + int(block_size))
        block_labels = np.asarray(labels[offset:stop])
        if (
            bool(np.any(block_labels < 0))
            or bool(np.any(block_labels >= int(cluster_count)))
        ):
            raise ExternalMemoryDBSCANError(
                "all-core summary label escaped canonical components"
            )
        if mask is None:
            selected_labels = block_labels
        else:
            block_mask = np.asarray(mask[offset:stop])
            if block_mask.shape != (stop - offset,) or block_mask.dtype != np.bool_:
                raise ExternalMemoryDBSCANError(
                    "all-core summary streaming mask schema mismatch"
                )
            selected_labels = block_labels[block_mask]
        if len(selected_labels):
            counts += np.bincount(
                selected_labels, minlength=int(cluster_count)
            ).astype(np.int64, copy=False)
        if mask is None:
            for raw_label in np.unique(block_labels).tolist():
                label = int(raw_label)
                if minima[label] < 0:
                    minima[label] = offset + int(
                        np.flatnonzero(block_labels == label)[0]
                    )
    return counts, minima


def _validate_dbscan_source(
    *,
    dbscan_manifest_path: Path,
    dbscan_manifest_sha256: str,
    labels: np.ndarray,
    recourse_vectors: np.ndarray,
) -> dict[str, Any]:
    path = dbscan_manifest_path.expanduser().resolve(strict=True)
    if _sha256_file(path) != str(dbscan_manifest_sha256):
        raise ExternalMemoryDBSCANError(
            "all-core summary DBSCAN manifest checksum mismatch"
        )
    manifest = _load_object(path)
    if (
        manifest.get("run_complete") is not True
        or manifest.get("clustering_path")
        != ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
        or int(manifest.get("num_samples", -1)) != len(recourse_vectors)
        or int(manifest.get("core_count", -1)) != len(recourse_vectors)
        or int(manifest.get("noise_count", -1)) != 0
        or int(manifest.get("cluster_count", -1)) <= 0
        or manifest.get("neighbor_counts_available") is not False
        or manifest.get("approximation_used") is not False
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary requires a complete exact recovery partition"
        )
    _validate_component_recovery_closure(manifest=manifest, root=path.parent)
    label_path = Path(str(manifest.get("labels_path") or "")).resolve(strict=True)
    vector_identity = manifest.get("scientific_identity")
    if not isinstance(vector_identity, Mapping):
        raise ExternalMemoryDBSCANError(
            "all-core summary DBSCAN scientific identity is absent"
        )
    vector_path = Path(
        str(getattr(recourse_vectors, "filename", "") or "")
    ).resolve(strict=True)
    supplied_label_path = Path(
        str(getattr(labels, "filename", "") or "")
    ).resolve(strict=True)
    if (
        supplied_label_path != label_path
        or _sha256_file(label_path) != manifest.get("labels_sha256")
        or vector_path
        != Path(str(vector_identity.get("vectors_path") or "")).resolve(
            strict=True
        )
        or _sha256_file(vector_path) != vector_identity.get("vectors_sha256")
        or labels.shape != (len(recourse_vectors),)
        or labels.dtype != np.dtype(np.intp)
        or recourse_vectors.ndim != 2
        or recourse_vectors.dtype != np.dtype(np.float32)
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary label/vector source binding mismatch"
        )
    cluster_count = int(manifest["cluster_count"])
    counts, minima = _stream_label_counts_and_minima(
        labels,
        cluster_count=cluster_count,
        block_size=1_000_000,
    )
    if bool(np.any(counts <= 0)):
        raise ExternalMemoryDBSCANError(
            "all-core summary labels are not canonical contiguous components"
        )
    if minima != sorted(minima):
        raise ExternalMemoryDBSCANError(
            "all-core summary labels violate sklearn visitation order"
        )
    return manifest


def _pair_authority(
    *,
    pair_indices: Any,
    pairs_sha256: str,
    pair_authority_manifest_path: str | Path | None,
    pair_authority_manifest_sha256: str | None,
) -> tuple[str, Path | None, Path | None, Any, Any | None]:
    filename = str(getattr(pair_indices, "filename", "") or "")
    authority_requested = (
        pair_authority_manifest_path is not None
        or pair_authority_manifest_sha256 is not None
    )
    if not authority_requested and filename:
        path = Path(filename).resolve(strict=True)
        if _sha256_file(path) != str(pairs_sha256):
            raise ExternalMemoryDBSCANError(
                "all-core summary pair checksum mismatch"
        )
        return "physical_npy", path, None, pair_indices, None
    if (
        pair_authority_manifest_path is None
        or pair_authority_manifest_sha256 is None
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary close-view pair authority is incomplete"
        )
    authority = Path(pair_authority_manifest_path).expanduser().resolve(
        strict=True
    )
    if _sha256_file(authority) != str(pair_authority_manifest_sha256):
        raise ExternalMemoryDBSCANError(
            "all-core summary close-view pair authority checksum mismatch"
        )
    from .close_pair_view import validate_theta_close_pair_view

    try:
        close_view = validate_theta_close_pair_view(
            authority,
            require_dbscan_eligible=True,
            require_pair_semantics_authority=True,
        )
        authoritative_pairs = close_view.open_pairs()
    except Exception as exc:
        raise ExternalMemoryDBSCANError(
            "all-core summary pair close-view authority is invalid"
        ) from exc
    authoritative_path_raw = str(
        getattr(authoritative_pairs, "filename", "") or ""
    )
    authoritative_path = (
        None
        if not authoritative_path_raw
        else Path(authoritative_path_raw).resolve(strict=True)
    )
    if (
        int(close_view.logical_close_rows) != int(pair_indices.shape[0])
        or close_view.pairs_sha256 != str(pairs_sha256)
        or authoritative_pairs.shape != pair_indices.shape
        or authoritative_pairs.dtype != np.dtype(np.int64)
        or (
            authoritative_path is None
            and getattr(authoritative_pairs, "logical_npy_sha256", None)
            != str(pairs_sha256)
        )
        or (
            authoritative_path is not None
            and (
                close_view.pairs_path != authoritative_path
                or _sha256_file(authoritative_path) != str(pairs_sha256)
            )
        )
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary pair close-view contract mismatch"
        )
    return (
        "theta_close_view_v1",
        authoritative_path,
        authority,
        authoritative_pairs,
        close_view,
    )


@dataclass(frozen=True)
class ExactAllCoreComponentSummaryResult:
    official_result: tuple[list[int], list[float], list[int]]
    selected: list[dict[str, Any]]
    manifest_path: Path
    manifest_sha256: str
    retained_mask_path: Path


def _empty_centroid_payload(
    *, cluster_count: int, feature_count: int, dtype: np.dtype[Any]
) -> dict[str, Any]:
    zeros = np.zeros((cluster_count, feature_count), dtype=dtype)
    stable = np.zeros((cluster_count, feature_count), dtype=np.float64)
    return {
        "cluster_counts": [0] * cluster_count,
        "official_sums_hex": _array_hex(zeros),
        "numpy_sums_hex": _array_hex(zeros),
        "stable_sums_hex": _array_hex(stable),
    }


def _centroid_arrays(
    payload: Mapping[str, Any], *, dtype: np.dtype[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(payload["cluster_counts"], dtype=np.int64)
    official = _array_from_hex(payload["official_sums_hex"], dtype=dtype)
    numpy_sums = _array_from_hex(payload["numpy_sums_hex"], dtype=dtype)
    stable = _array_from_hex(payload["stable_sums_hex"], dtype=np.dtype(np.float64))
    return counts, official, numpy_sums, stable


def _centroid_payload(
    counts: np.ndarray,
    official: np.ndarray,
    numpy_sums: np.ndarray,
    stable: np.ndarray,
) -> dict[str, Any]:
    return {
        "cluster_counts": [int(value) for value in counts.tolist()],
        "official_sums_hex": _array_hex(official),
        "numpy_sums_hex": _array_hex(numpy_sums),
        "stable_sums_hex": _array_hex(stable),
    }


def _scan_centroid_range(
    *,
    labels: np.ndarray,
    vectors: np.ndarray,
    start: int,
    stop: int,
    block_size: int,
    counts: np.ndarray,
    official_sums: np.ndarray,
    numpy_sums: np.ndarray,
    stable_sums: np.ndarray,
    torch_module: Any,
) -> None:
    dtype = vectors.dtype
    for offset in range(int(start), int(stop), int(block_size)):
        end = min(int(stop), offset + int(block_size))
        block_labels = np.asarray(labels[offset:end])
        block_vectors = np.asarray(vectors[offset:end])
        for cluster in np.unique(block_labels).tolist():
            cluster_id = int(cluster)
            selected = np.asarray(block_vectors[block_labels == cluster_id])
            if not len(selected):
                continue
            torch_sum = (
                torch_module.sum(torch_module.from_numpy(selected), dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            official_sums[cluster_id] = np.asarray(
                torch_module.add(
                    torch_module.from_numpy(official_sums[cluster_id]),
                    torch_module.from_numpy(torch_sum),
                ).numpy(),
                dtype=dtype,
            )
            numpy_sums[cluster_id] = np.add(
                numpy_sums[cluster_id],
                np.sum(selected, axis=0, dtype=dtype),
                dtype=dtype,
            )
            stable_sums[cluster_id] += np.sum(
                np.asarray(selected, dtype=np.float64),
                axis=0,
                dtype=np.float64,
            )
            counts[cluster_id] += int(len(selected))


def _centroids_from_sums(
    *,
    counts: np.ndarray,
    official_sums: np.ndarray,
    numpy_sums: np.ndarray,
    stable_sums: np.ndarray,
    torch_module: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bool(np.any(counts <= 0)):
        raise ExternalMemoryDBSCANError(
            "all-core summary contains an empty canonical cluster"
        )
    dtype = official_sums.dtype
    official = np.empty_like(official_sums)
    for cluster in range(len(counts)):
        official[cluster] = (
            torch_module.true_divide(
                torch_module.from_numpy(official_sums[cluster]),
                int(counts[cluster]),
            )
            .detach()
            .cpu()
            .numpy()
        )
    numpy_centroids = np.asarray(
        numpy_sums / counts[:, None].astype(dtype), dtype=dtype
    )
    stable = stable_sums / counts[:, None].astype(np.float64)
    return official, numpy_centroids, stable


def _empty_membership_payload(cluster_count: int) -> dict[str, Any]:
    return {
        "official_first_by_parent": [[] for _ in range(cluster_count)],
        "official_all_candidates": [[] for _ in range(cluster_count)],
        "trace_parents": [[] for _ in range(cluster_count)],
        "trace_candidates": [[] for _ in range(cluster_count)],
        "official_within_counts": [0] * cluster_count,
        "trace_retained_counts": [0] * cluster_count,
        "count_exactly_at_delta": [0] * cluster_count,
        "float64_membership_disagreements": [0] * cluster_count,
    }


def _membership_state(
    payload: Mapping[str, Any], *, cluster_count: int
) -> tuple[
    list[dict[int, int]],
    list[set[int]],
    list[set[int]],
    list[set[int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    first = [
        {int(parent): int(candidate) for parent, candidate in rows}
        for rows in payload["official_first_by_parent"]
    ]
    official_candidates = [set(map(int, rows)) for rows in payload["official_all_candidates"]]
    trace_parents = [set(map(int, rows)) for rows in payload["trace_parents"]]
    trace_candidates = [set(map(int, rows)) for rows in payload["trace_candidates"]]
    arrays = [
        np.asarray(payload[name], dtype=np.int64)
        for name in (
            "official_within_counts",
            "trace_retained_counts",
            "count_exactly_at_delta",
            "float64_membership_disagreements",
        )
    ]
    if not (
        len(first)
        == len(official_candidates)
        == len(trace_parents)
        == len(trace_candidates)
        == cluster_count
        and all(array.shape == (cluster_count,) for array in arrays)
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary membership checkpoint schema mismatch"
        )
    return first, official_candidates, trace_parents, trace_candidates, *arrays


def _membership_payload(
    first: Sequence[Mapping[int, int]],
    official_candidates: Sequence[set[int]],
    trace_parents: Sequence[set[int]],
    trace_candidates: Sequence[set[int]],
    official_within: np.ndarray,
    trace_retained: np.ndarray,
    exactly_delta: np.ndarray,
    disagreements: np.ndarray,
) -> dict[str, Any]:
    return {
        "official_first_by_parent": [
            [[int(parent), int(candidate)] for parent, candidate in sorted(row.items())]
            for row in first
        ],
        "official_all_candidates": [sorted(values) for values in official_candidates],
        "trace_parents": [sorted(values) for values in trace_parents],
        "trace_candidates": [sorted(values) for values in trace_candidates],
        "official_within_counts": [int(value) for value in official_within.tolist()],
        "trace_retained_counts": [int(value) for value in trace_retained.tolist()],
        "count_exactly_at_delta": [int(value) for value in exactly_delta.tolist()],
        "float64_membership_disagreements": [
            int(value) for value in disagreements.tolist()
        ],
    }


def _numpy_trace_membership(
    values: np.ndarray, *, center: np.ndarray, radius: float
) -> np.ndarray:
    distances = np.linalg.norm(values - center, axis=1)
    radius_in_distance_dtype = np.asarray(float(radius), dtype=distances.dtype)
    return distances < radius_in_distance_dtype


def _scan_membership_range(
    *,
    labels: np.ndarray,
    vectors: np.ndarray,
    pairs: Any,
    official_centroids: np.ndarray,
    numpy_centroids: np.ndarray,
    stable_centroids: np.ndarray,
    radius: float,
    start: int,
    stop: int,
    block_size: int,
    trace_mask: np.ndarray | None,
    validate_mask: bool,
    first: list[dict[int, int]],
    official_candidates: list[set[int]],
    trace_parents: list[set[int]],
    trace_candidates: list[set[int]],
    official_within: np.ndarray,
    trace_retained: np.ndarray,
    exactly_delta: np.ndarray,
    disagreements: np.ndarray,
    torch_module: Any,
) -> None:
    for offset in range(int(start), int(stop), int(block_size)):
        end = min(int(stop), offset + int(block_size))
        block_labels = np.asarray(labels[offset:end])
        block_vectors = np.asarray(vectors[offset:end])
        expected_mask = np.zeros(end - offset, dtype=np.bool_)
        for raw_cluster in np.unique(block_labels).tolist():
            cluster = int(raw_cluster)
            local = np.flatnonzero(block_labels == cluster)
            selected = np.asarray(block_vectors[local])
            torch_distances = (
                torch_module.norm(
                    torch_module.from_numpy(selected)
                    - torch_module.from_numpy(official_centroids[cluster]),
                    dim=-1,
                )
                .detach()
                .cpu()
            )
            official_member = (torch_distances < float(radius)).numpy()
            exactly = (torch_distances == float(radius)).numpy()
            stable_distances = np.linalg.norm(
                np.asarray(selected, dtype=np.float64)
                - stable_centroids[cluster],
                axis=1,
            )
            stable_member = stable_distances < float(radius)
            trace_member = _numpy_trace_membership(
                selected, center=numpy_centroids[cluster], radius=radius
            )
            # The production mask is the paper-faithful Torch/official mask.
            # NumPy and float64 memberships remain audit values only.
            expected_mask[local] = official_member
            official_within[cluster] += int(np.count_nonzero(official_member))
            trace_retained[cluster] += int(np.count_nonzero(trace_member))
            exactly_delta[cluster] += int(np.count_nonzero(exactly))
            disagreements[cluster] += int(
                np.count_nonzero(official_member != stable_member)
            )
            if bool(np.any(official_member)):
                physical = local[official_member].astype(np.int64) + offset
                selected_pairs = np.asarray(pairs[physical])
                for parent, candidate in selected_pairs.tolist():
                    first[cluster].setdefault(int(parent), int(candidate))
                    official_candidates[cluster].add(int(candidate))
            if bool(np.any(trace_member)):
                physical = local[trace_member].astype(np.int64) + offset
                selected_pairs = np.asarray(pairs[physical])
                trace_parents[cluster].update(
                    map(int, selected_pairs[:, 0].tolist())
                )
                trace_candidates[cluster].update(
                    map(int, selected_pairs[:, 1].tolist())
                )
        if trace_mask is not None:
            if validate_mask:
                if not np.array_equal(trace_mask[offset:end], expected_mask):
                    raise ExternalMemoryDBSCANError(
                        "all-core summary retained-mask committed prefix mismatch"
                    )
            else:
                trace_mask[offset:end] = expected_mask


def _empty_retained_payload(
    *, cluster_count: int, feature_count: int, dtype: np.dtype[Any]
) -> dict[str, Any]:
    return {
        "retained_counts": [0] * cluster_count,
        "retained_sums_hex": _array_hex(
            np.zeros((cluster_count, feature_count), dtype=dtype)
        ),
    }


def _retained_state(
    payload: Mapping[str, Any], *, dtype: np.dtype[Any]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(payload["retained_counts"], dtype=np.int64),
        _array_from_hex(payload["retained_sums_hex"], dtype=dtype),
    )


def _retained_payload(counts: np.ndarray, sums: np.ndarray) -> dict[str, Any]:
    return {
        "retained_counts": [int(value) for value in counts.tolist()],
        "retained_sums_hex": _array_hex(sums),
    }


def _scan_retained_range(
    *,
    labels: np.ndarray,
    vectors: np.ndarray,
    mask: np.ndarray,
    start: int,
    stop: int,
    block_size: int,
    counts: np.ndarray,
    sums: np.ndarray,
) -> None:
    for offset in range(int(start), int(stop), int(block_size)):
        end = min(int(stop), offset + int(block_size))
        local_mask = np.asarray(mask[offset:end])
        if not bool(np.any(local_mask)):
            continue
        block_labels = np.asarray(labels[offset:end])
        block_vectors = np.asarray(vectors[offset:end])
        for raw_cluster in np.unique(block_labels[local_mask]).tolist():
            cluster = int(raw_cluster)
            selected = np.asarray(
                block_vectors[local_mask & (block_labels == cluster)]
            )
            sums[cluster] = np.add(
                sums[cluster],
                np.sum(selected, axis=0, dtype=vectors.dtype),
                dtype=vectors.dtype,
            )
            counts[cluster] += int(len(selected))


def _empty_medoid_payload(cluster_count: int) -> dict[str, Any]:
    return {
        "positions": [-1] * cluster_count,
        "distance_hex": [float("inf").hex()] * cluster_count,
    }


def _medoid_state(payload: Mapping[str, Any]) -> tuple[np.ndarray, list[float]]:
    return (
        np.asarray(payload["positions"], dtype=np.int64),
        _float_from_hex(payload["distance_hex"]),
    )


def _medoid_payload(positions: np.ndarray, distances: Sequence[float]) -> dict[str, Any]:
    return {
        "positions": [int(value) for value in positions.tolist()],
        "distance_hex": _float_hex(distances),
    }


def _scan_medoid_range(
    *,
    labels: np.ndarray,
    vectors: np.ndarray,
    mask: np.ndarray,
    retained_centroids: np.ndarray,
    start: int,
    stop: int,
    block_size: int,
    positions: np.ndarray,
    distances: list[float],
) -> None:
    for offset in range(int(start), int(stop), int(block_size)):
        end = min(int(stop), offset + int(block_size))
        local_mask = np.asarray(mask[offset:end])
        if not bool(np.any(local_mask)):
            continue
        block_labels = np.asarray(labels[offset:end])
        block_vectors = np.asarray(vectors[offset:end])
        for raw_cluster in np.unique(block_labels[local_mask]).tolist():
            cluster = int(raw_cluster)
            local = np.flatnonzero(local_mask & (block_labels == cluster))
            selected = np.asarray(block_vectors[local])
            current = np.linalg.norm(
                selected - retained_centroids[cluster], axis=1
            )
            winner = int(np.argmin(current))
            candidate_distance = float(current[winner])
            candidate_position = int(offset + local[winner])
            if candidate_distance < distances[cluster]:
                distances[cluster] = candidate_distance
                positions[cluster] = candidate_position


def _deterministic_greedy(
    covering: Mapping[int, set[int]], *, k: int
) -> dict[int, tuple[int, int]]:
    remaining = {int(label): set(values) for label, values in sorted(covering.items())}
    covered: set[int] = set()
    result: dict[int, tuple[int, int]] = {}
    for rank in range(1, min(int(k), len(remaining)) + 1):
        label = min(remaining, key=lambda value: (-len(remaining[value]), value))
        gains = set(remaining.pop(label))
        covered.update(gains)
        for values in remaining.values():
            values.difference_update(gains)
        result[rank] = (label, len(covered))
    return result


def _invoke_and_validate_greedy(
    *,
    covering: Mapping[int, set[int]],
    k: int,
    official_greedy: Callable[..., Any],
) -> dict[int, tuple[int, int]]:
    covered_by: dict[int, set[int]] = defaultdict(set)
    for label, parents in sorted(covering.items()):
        for parent in parents:
            covered_by[int(parent)].add(int(label))
    official = official_greedy(
        counterfactual_covering={
            int(label): set(values) for label, values in sorted(covering.items())
        },
        graphs_covered_by={
            int(parent): set(values) for parent, values in sorted(covered_by.items())
        },
        k=min(int(k), len(covering)),
    )
    normalized = {
        int(rank): (int(value[0]), int(value[1]))
        for rank, value in official.items()
    }
    expected = _deterministic_greedy(covering, k=int(k))
    if normalized != expected:
        raise ExternalMemoryDBSCANError(
            "official greedy no longer preserves canonical cluster tie order"
        )
    return normalized


def _results_from_replay(
    *,
    cluster_counts: np.ndarray,
    official_centroids: np.ndarray,
    numpy_centroids: np.ndarray,
    stable_centroids: np.ndarray,
    membership: tuple[
        list[dict[int, int]],
        list[set[int]],
        list[set[int]],
        list[set[int]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    retained_counts: np.ndarray,
    medoid_positions: np.ndarray,
    medoid_distances: Sequence[float],
    pairs: Any,
    radius: float,
    theta: float,
    recourse_size: int,
    official_greedy: Callable[..., Any],
    torch_module: Any,
) -> tuple[
    tuple[list[int], list[float], list[int]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    (
        first,
        official_candidates,
        trace_parents,
        trace_candidates,
        official_within,
        trace_retained,
        exactly_delta,
        disagreements,
    ) = membership
    cluster_count = len(cluster_counts)
    official_norms = [
        float(torch_module.norm(torch_module.from_numpy(official_centroids[value])).item())
        for value in range(cluster_count)
    ]
    numpy_norms = [
        float(np.linalg.norm(numpy_centroids[value]))
        for value in range(cluster_count)
    ]
    stable_norms = [
        float(np.linalg.norm(stable_centroids[value]))
        for value in range(cluster_count)
    ]
    theta_decision_disagreements = [
        label
        for label in range(cluster_count)
        if (official_norms[label] < float(theta))
        != (stable_norms[label] < float(theta))
    ]
    radius_decision_disagreement_count = int(np.sum(disagreements))
    if radius_decision_disagreement_count or theta_decision_disagreements:
        raise ExternalMemoryDBSCANError(
            "PROJECT_EXTENSION_NUMERIC_DECISION_DISAGREEMENT:"
            f"radius_rows={radius_decision_disagreement_count}:"
            f"theta_clusters={theta_decision_disagreements}"
        )
    official_covering = {
        label: set(first[label])
        for label in range(cluster_count)
        if official_norms[label] < float(theta)
    }
    official_selection = _invoke_and_validate_greedy(
        covering=official_covering,
        k=int(recourse_size),
        official_greedy=official_greedy,
    )
    covering_values: list[int] = []
    cumulative_costs: list[float] = []
    unique_first_candidates: set[int] = set()
    sizes: list[int] = []
    cumulative_cost = 0.0
    for rank in range(1, len(official_selection) + 1):
        label, cumulative_covered = official_selection[rank]
        covering_values.append(int(cumulative_covered))
        cumulative_cost += official_norms[label]
        cumulative_costs.append(cumulative_cost)
        unique_first_candidates.update(first[label].values())
        sizes.append(len(unique_first_candidates))
    official_result = (covering_values, cumulative_costs, sizes)

    selected_covering = {
        label: set(first[label])
        for label in range(cluster_count)
        if official_norms[label] < float(theta)
        and first[label]
        and int(retained_counts[label]) > 0
    }
    selected_selection = _invoke_and_validate_greedy(
        covering=selected_covering,
        k=int(recourse_size),
        official_greedy=official_greedy,
    )
    official_nonempty_prefix = [
        value
        for _rank, value in sorted(official_selection.items())
        if int(value[0]) in selected_covering
    ][: len(selected_selection)]
    if official_nonempty_prefix != [
        selected_selection[rank]
        for rank in range(1, len(selected_selection) + 1)
    ]:
        raise ExternalMemoryDBSCANError(
            "official coverage and standardized selection order diverged"
        )
    selected: list[dict[str, Any]] = []
    covered: set[int] = set()
    cumulative_trace_cost = 0.0
    for rank in range(1, len(selected_selection) + 1):
        label, cumulative_covered = selected_selection[rank]
        position = int(medoid_positions[label])
        if position < 0 or not np.isfinite(float(medoid_distances[label])):
            raise ExternalMemoryDBSCANError(
                "selected all-core component has no retained medoid"
            )
        source, candidate = pairs[position]
        covered.update(selected_covering[label])
        cumulative_trace_cost += official_norms[label]
        if len(covered) != int(cumulative_covered):
            raise ExternalMemoryDBSCANError(
                "all-core component greedy cumulative coverage drifted"
            )
        selected.append(
            {
                "rank": rank,
                "selected_rank": rank,
                "cluster_label": label,
                "cluster_id": label,
                "cluster_center_norm": official_norms[label],
                "centroid_norm": official_norms[label],
                "cluster_radius": float(radius),
                "cluster_size": int(cluster_counts[label]),
                "representative_source_index": int(source),
                "representative_counterfactual_index": int(candidate),
                "representative_distance_to_center": float(
                    medoid_distances[label]
                ),
                "covered_parent_indices_native": sorted(selected_covering[label]),
                "native_cumulative_covered_count": len(covered),
                "cumulative_covered_count": len(covered),
                "native_cumulative_cost": cumulative_trace_cost,
                "member_counterfactual_indices": sorted(
                    official_candidates[label]
                ),
                "representative_candidate_ids": [int(candidate)],
            }
        )
    cluster_summaries = [
        {
            "cluster_id": label,
            "cluster_member_count": int(cluster_counts[label]),
            "official_within_centroid_radius_count": int(official_within[label]),
            "trace_within_centroid_radius_count": int(trace_retained[label]),
            "outside_official_centroid_radius_count": int(
                cluster_counts[label] - official_within[label]
            ),
            "count_exactly_at_delta": int(exactly_delta[label]),
            "float64_radius_membership_disagreement_count": int(
                disagreements[label]
            ),
            "official_float32_centroid_norm": official_norms[label],
            "trace_numpy_centroid_norm": numpy_norms[label],
            "stable_float64_centroid_norm": stable_norms[label],
            "centroid_max_abs_difference": float(
                np.max(
                    np.abs(
                        np.asarray(official_centroids[label], dtype=np.float64)
                        - stable_centroids[label]
                    )
                )
            ),
            "centroid_norm_lt_theta": official_norms[label] < float(theta),
            "count_exactly_at_theta": int(official_norms[label] == float(theta)),
            "float64_centroid_norm_lt_theta": stable_norms[label]
            < float(theta),
            "centroid_norm_decision_disagreement": (
                (official_norms[label] < float(theta))
                != (stable_norms[label] < float(theta))
            ),
            "official_covered_parent_indices": sorted(first[label]),
            "official_first_counterfactual_indices": sorted(
                set(first[label].values())
            ),
            "official_radius_counterfactual_indices": sorted(
                official_candidates[label]
            ),
            "trace_covered_parent_indices": sorted(trace_parents[label]),
            "trace_counterfactual_indices": sorted(trace_candidates[label]),
            "retained_count": int(retained_counts[label]),
            "medoid_global_row": int(medoid_positions[label]),
        }
        for label in range(cluster_count)
    ]
    return official_result, selected, cluster_summaries


def _artifact_array(
    path: Path, *, expected_sha256: str, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    if path.is_symlink() or _sha256_file(path) != str(expected_sha256):
        raise ExternalMemoryDBSCANError(
            f"all-core summary artifact checksum mismatch: {path.name}"
        )
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != shape or values.dtype != dtype:
        raise ExternalMemoryDBSCANError(
            f"all-core summary artifact schema mismatch: {path.name}"
        )
    return values


def _summarize_proven_all_core_components_external_locked(
    *,
    work_dir: str | Path,
    dbscan_manifest_path: str | Path,
    dbscan_manifest_sha256: str,
    labels: np.ndarray,
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
    _fresh_root_claimed: bool = False,
    _writer_guard: Callable[[], None] | None = None,
    _writer_lock_identity: Mapping[str, Any] | None = None,
) -> ExactAllCoreComponentSummaryResult:
    """Replay every exact all-core component with bounded streaming scans."""

    root = Path(work_dir).expanduser().resolve(strict=False)
    checkpoint_path = root / "checkpoint.json"
    manifest_path = root / "run_manifest.json"
    if (
        pair_indices.ndim != 2
        or pair_indices.shape != (len(recourse_vectors), 2)
        or pair_indices.dtype != np.dtype(np.int64)
        or len(recourse_vectors) <= 0
        or not np.isfinite(float(radius))
        or float(radius) <= 0
        or not np.isfinite(float(theta))
        or float(theta) < 0
        or int(recourse_size) < 0
        or int(block_size) <= 0
        or int(max_rss_bytes) <= 0
        or not isinstance(_writer_lock_identity, Mapping)
        or not _writer_lock_identity
    ):
        raise ExternalMemoryDBSCANError(
            "all-core component summary execution contract is invalid"
        )
    dbscan_path = Path(dbscan_manifest_path).expanduser().resolve(strict=True)
    dbscan = _validate_dbscan_source(
        dbscan_manifest_path=dbscan_path,
        dbscan_manifest_sha256=dbscan_manifest_sha256,
        labels=labels,
        recourse_vectors=recourse_vectors,
    )
    (
        storage,
        pair_path,
        pair_authority_path,
        pair_indices,
        close_pair_view,
    ) = _pair_authority(
        pair_indices=pair_indices,
        pairs_sha256=pairs_sha256,
        pair_authority_manifest_path=pair_authority_manifest_path,
        pair_authority_manifest_sha256=pair_authority_manifest_sha256,
    )
    vector_path = Path(
        str(getattr(recourse_vectors, "filename", "") or "")
    ).resolve(strict=True)
    label_path = Path(str(getattr(labels, "filename", "") or "")).resolve(
        strict=True
    )
    if close_pair_view is not None and (
        close_pair_view.vectors_path != vector_path
        or close_pair_view.vectors_sha256
        != dbscan["scientific_identity"]["vectors_sha256"]
    ):
        raise ExternalMemoryDBSCANError(
            "all-core summary close-view/vector authority mismatch"
        )
    n_samples, n_features = map(int, recourse_vectors.shape)
    cluster_count = int(dbscan["cluster_count"])
    identity = {
        "schema_version": ALL_CORE_COMPONENT_SUMMARY_SCHEMA,
        "dbscan_manifest_path": str(dbscan_path),
        "dbscan_manifest_sha256": str(dbscan_manifest_sha256),
        "dbscan_clustering_path": dbscan["clustering_path"],
        "dbscan_cluster_count": cluster_count,
        "labels_path": str(label_path),
        "labels_sha256": dbscan["labels_sha256"],
        "vectors_path": str(vector_path),
        "vectors_sha256": dbscan["scientific_identity"]["vectors_sha256"],
        "vectors_shape": [n_samples, n_features],
        "vectors_dtype": str(recourse_vectors.dtype),
        "pairs_storage": storage,
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
        "max_rss_bytes": int(max_rss_bytes),
        "torch_version": str(torch_module.__version__),
        "numpy_version": str(np.__version__),
        "pair_order": "candidate_major_parent_minor",
        "centroid_reduction": (
            "PROJECT_EXTENSION:fixed_global_row_order_fixed_block_torch_float32_"
            "plus_float64_audit_v1:not_upstream_one_shot_bit_identical"
        ),
        "radius_filter_operator": "<",
        "centroid_norm_filter_operator": "<",
        "large_cluster_advanced_index_copy": False,
    }
    identity_sha = _stable_hash(identity)
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                "completed all-core summary requires explicit resume adoption"
            )
        completed = _load_object(manifest_path)
        if completed.get("scientific_identity") != identity:
            raise ExternalMemoryDBSCANError(
                "completed all-core summary invocation identity drift"
            )
        return validate_proven_all_core_component_summary(
            manifest_path,
            torch_module=torch_module,
            pair_indices=pair_indices,
        )
    if _fresh_root_claimed:
        if (root / "owner_claim.json").exists():
            raise ExternalMemoryDBSCANError(
                "fresh all-core summary root already has an owner claim"
            )
        claim = {
            "schema_version": _OWNER_CLAIM_SCHEMA,
            "root": str(root),
            "root_stat_identity": _root_stat(root),
            "scientific_identity_sha256": str(identity_sha),
            "claimed_at": _utc_now(),
        }
        if _writer_guard is not None:
            _writer_guard()
        _atomic_json(
            root / "owner_claim.json", claim, writer_guard=_writer_guard
        )
        if _writer_guard is not None:
            _writer_guard()
    elif root.exists():
        if not resume:
            raise FileExistsError(
                f"all-core component summary root is already claimed: {root}"
            )
        _validate_owner_claim(root, identity_sha256=identity_sha)
        unexpected = {
            path.name
            for path in root.iterdir()
            if path.name
            not in {".writer.lock", "owner_claim.json", "checkpoint.json"}
        }
        if not checkpoint_path.exists() and unexpected:
            raise ExternalMemoryDBSCANError(
                "all-core summary owner-only resume root has uncheckpointed artifacts"
            )
    else:
        raise ExternalMemoryDBSCANError(
            "all-core summary writer lock did not preclaim a fresh root"
        )
    dtype = recourse_vectors.dtype
    artifacts: dict[str, Any] = {}
    if checkpoint_path.exists():
        state = _load_checkpoint(
            checkpoint_path, identity=identity, total=n_samples
        )
    else:
        payload = _empty_centroid_payload(
            cluster_count=cluster_count,
            feature_count=n_features,
            dtype=dtype,
        )
        ledger = _new_ledger(phase="centroid_scan", identity_sha256=identity_sha)
        state = _write_checkpoint(
            checkpoint_path,
            identity=identity,
            phase="centroid_scan",
            next_offset=0,
            phase_payload=payload,
            ledger=ledger,
            artifacts=artifacts,
            peak_rss_bytes=_rss_bytes(),
            writer_guard=_writer_guard,
        )
    peak = max(int(state["peak_rss_bytes"]), _rss_bytes())
    reservation = int(block_size) * (n_features * dtype.itemsize * 3 + 64)
    _check_rss(
        int(max_rss_bytes),
        phase="all_core_component_summary.block",
        reserved_bytes=reservation + 128 * 1024**2,
    )

    phase = str(state["phase"])
    artifacts = dict(state["artifacts"])
    if phase == "centroid_scan":
        counts, official_sums, numpy_sums, stable_sums = _centroid_arrays(
            state["phase_payload"], dtype=dtype
        )
        start = int(state["next_offset"])
        replay_counts = np.zeros(cluster_count, dtype=np.int64)
        replay_official = np.zeros((cluster_count, n_features), dtype=dtype)
        replay_numpy = np.zeros((cluster_count, n_features), dtype=dtype)
        replay_stable = np.zeros((cluster_count, n_features), dtype=np.float64)
        _scan_centroid_range(
            labels=labels,
            vectors=recourse_vectors,
            start=0,
            stop=start,
            block_size=int(block_size),
            counts=replay_counts,
            official_sums=replay_official,
            numpy_sums=replay_numpy,
            stable_sums=replay_stable,
            torch_module=torch_module,
        )
        replay_payload = _centroid_payload(
            replay_counts, replay_official, replay_numpy, replay_stable
        )
        _validate_ledger(
            state["progress_ledger"],
            phase=phase,
            identity_sha256=identity_sha,
            total=n_samples,
            block_size=int(block_size),
            payload=state["phase_payload"],
        )
        if replay_payload != state["phase_payload"]:
            raise ExternalMemoryDBSCANError(
                "all-core centroid committed prefix does not replay"
            )
        ledger = dict(state["progress_ledger"])
        ledger["entries"] = list(ledger["entries"])
        for offset in range(start, n_samples, int(block_size)):
            stop = min(n_samples, offset + int(block_size))
            _scan_centroid_range(
                labels=labels,
                vectors=recourse_vectors,
                start=offset,
                stop=stop,
                block_size=int(block_size),
                counts=counts,
                official_sums=official_sums,
                numpy_sums=numpy_sums,
                stable_sums=stable_sums,
                torch_module=torch_module,
            )
            payload = _centroid_payload(
                counts, official_sums, numpy_sums, stable_sums
            )
            _append_ledger(
                ledger,
                start=offset,
                stop=stop,
                payload=payload,
                identity_sha256=identity_sha,
            )
            peak = max(
                peak,
                _check_rss(int(max_rss_bytes), phase="all_core.centroid"),
            )
            state = _write_checkpoint(
                checkpoint_path,
                identity=identity,
                phase=phase,
                next_offset=stop,
                phase_payload=payload,
                ledger=ledger,
                artifacts=artifacts,
                peak_rss_bytes=peak,
                writer_guard=_writer_guard,
            )
        if int(np.sum(counts)) != n_samples:
            raise ExternalMemoryDBSCANError(
                "all-core centroid counts do not close the partition"
            )
        official_centroids, numpy_centroids, stable_centroids = (
            _centroids_from_sums(
                counts=counts,
                official_sums=official_sums,
                numpy_sums=numpy_sums,
                stable_sums=stable_sums,
                torch_module=torch_module,
            )
        )
        for name, values in (
            ("cluster_counts", counts),
            ("official_float32_centroids", official_centroids),
            ("trace_numpy_centroids", numpy_centroids),
            ("stable_float64_centroids", stable_centroids),
        ):
            path = root / f"{name}.npy"
            artifacts[f"{name}_path"] = str(path)
            artifacts[f"{name}_sha256"] = _atomic_npy(
                path, values, writer_guard=_writer_guard
            )
        payload = _empty_membership_payload(cluster_count)
        ledger = _new_ledger(
            phase="membership_scan", identity_sha256=identity_sha
        )
        state = _write_checkpoint(
            checkpoint_path,
            identity=identity,
            phase="membership_scan",
            next_offset=0,
            phase_payload=payload,
            ledger=ledger,
            artifacts=artifacts,
            peak_rss_bytes=peak,
            writer_guard=_writer_guard,
        )
        phase = "membership_scan"

    counts = np.asarray(
        _artifact_array(
            Path(artifacts["cluster_counts_path"]),
            expected_sha256=artifacts["cluster_counts_sha256"],
            shape=(cluster_count,),
            dtype=np.dtype(np.int64),
        )
    )
    official_centroids = np.asarray(
        _artifact_array(
            Path(artifacts["official_float32_centroids_path"]),
            expected_sha256=artifacts["official_float32_centroids_sha256"],
            shape=(cluster_count, n_features),
            dtype=dtype,
        )
    )
    numpy_centroids = np.asarray(
        _artifact_array(
            Path(artifacts["trace_numpy_centroids_path"]),
            expected_sha256=artifacts["trace_numpy_centroids_sha256"],
            shape=(cluster_count, n_features),
            dtype=dtype,
        )
    )
    stable_centroids = np.asarray(
        _artifact_array(
            Path(artifacts["stable_float64_centroids_path"]),
            expected_sha256=artifacts["stable_float64_centroids_sha256"],
            shape=(cluster_count, n_features),
            dtype=np.dtype(np.float64),
        )
    )

    mask_partial = root / "retained_mask.partial.npy"
    mask_final = root / "retained_mask.npy"
    if phase == "membership_scan":
        membership = _membership_state(
            state["phase_payload"], cluster_count=cluster_count
        )
        start = int(state["next_offset"])
        if mask_partial.exists():
            mask = np.load(mask_partial, mmap_mode="r+", allow_pickle=False)
            if mask.shape != (n_samples,) or mask.dtype != np.dtype(np.bool_):
                raise ExternalMemoryDBSCANError(
                    "all-core retained-mask partial schema mismatch"
                )
        elif start == 0 and not mask_final.exists():
            mask = np.lib.format.open_memmap(
                mask_partial, mode="w+", dtype=np.bool_, shape=(n_samples,)
            )
            mask[:] = False
        else:
            raise ExternalMemoryDBSCANError(
                "all-core retained-mask partial/checkpoint mismatch"
            )
        replay_membership = _membership_state(
            _empty_membership_payload(cluster_count), cluster_count=cluster_count
        )
        _scan_membership_range(
            labels=labels,
            vectors=recourse_vectors,
            pairs=pair_indices,
            official_centroids=official_centroids,
            numpy_centroids=numpy_centroids,
            stable_centroids=stable_centroids,
            radius=float(radius),
            start=0,
            stop=start,
            block_size=int(block_size),
            trace_mask=mask,
            validate_mask=True,
            first=replay_membership[0],
            official_candidates=replay_membership[1],
            trace_parents=replay_membership[2],
            trace_candidates=replay_membership[3],
            official_within=replay_membership[4],
            trace_retained=replay_membership[5],
            exactly_delta=replay_membership[6],
            disagreements=replay_membership[7],
            torch_module=torch_module,
        )
        replay_payload = _membership_payload(*replay_membership)
        _validate_ledger(
            state["progress_ledger"],
            phase=phase,
            identity_sha256=identity_sha,
            total=n_samples,
            block_size=int(block_size),
            payload=state["phase_payload"],
        )
        if replay_payload != state["phase_payload"]:
            raise ExternalMemoryDBSCANError(
                "all-core membership committed prefix does not replay"
            )
        ledger = dict(state["progress_ledger"])
        ledger["entries"] = list(ledger["entries"])
        for offset in range(start, n_samples, int(block_size)):
            stop = min(n_samples, offset + int(block_size))
            _scan_membership_range(
                labels=labels,
                vectors=recourse_vectors,
                pairs=pair_indices,
                official_centroids=official_centroids,
                numpy_centroids=numpy_centroids,
                stable_centroids=stable_centroids,
                radius=float(radius),
                start=offset,
                stop=stop,
                block_size=int(block_size),
                trace_mask=mask,
                validate_mask=False,
                first=membership[0],
                official_candidates=membership[1],
                trace_parents=membership[2],
                trace_candidates=membership[3],
                official_within=membership[4],
                trace_retained=membership[5],
                exactly_delta=membership[6],
                disagreements=membership[7],
                torch_module=torch_module,
            )
            _fsync_memmap(mask)
            payload = _membership_payload(*membership)
            _append_ledger(
                ledger,
                start=offset,
                stop=stop,
                payload=payload,
                identity_sha256=identity_sha,
            )
            peak = max(
                peak,
                _check_rss(int(max_rss_bytes), phase="all_core.membership"),
            )
            state = _write_checkpoint(
                checkpoint_path,
                identity=identity,
                phase=phase,
                next_offset=stop,
                phase_payload=payload,
                ledger=ledger,
                artifacts=artifacts,
                peak_rss_bytes=peak,
                writer_guard=_writer_guard,
            )
        _fsync_memmap(mask)
        mask_sha = _sha256_file(mask_partial)
        del mask
        artifacts["retained_mask_path"] = str(mask_final)
        artifacts["retained_mask_sha256"] = mask_sha
        payload = _empty_retained_payload(
            cluster_count=cluster_count,
            feature_count=n_features,
            dtype=dtype,
        )
        ledger = _new_ledger(
            phase="retained_centroid_scan", identity_sha256=identity_sha
        )
        state = _write_checkpoint(
            checkpoint_path,
            identity=identity,
            phase="retained_centroid_scan",
            next_offset=0,
            phase_payload=payload,
            ledger=ledger,
            artifacts=artifacts,
            peak_rss_bytes=peak,
            writer_guard=_writer_guard,
        )
        if _writer_guard is not None:
            _writer_guard()
        os.replace(mask_partial, mask_final)
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _writer_guard is not None:
            _writer_guard()
        phase = "retained_centroid_scan"

    mask_path = _reconcile_array(
        partial=mask_partial,
        final=mask_final,
        expected_sha256=str(artifacts["retained_mask_sha256"]),
        shape=(n_samples,),
        dtype=np.dtype(np.bool_),
        label="retained mask",
        writer_guard=_writer_guard,
    )
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if phase == "retained_centroid_scan":
        retained_counts, retained_sums = _retained_state(
            state["phase_payload"], dtype=dtype
        )
        start = int(state["next_offset"])
        replay_counts = np.zeros(cluster_count, dtype=np.int64)
        replay_sums = np.zeros((cluster_count, n_features), dtype=dtype)
        _scan_retained_range(
            labels=labels,
            vectors=recourse_vectors,
            mask=mask,
            start=0,
            stop=start,
            block_size=int(block_size),
            counts=replay_counts,
            sums=replay_sums,
        )
        replay_payload = _retained_payload(replay_counts, replay_sums)
        _validate_ledger(
            state["progress_ledger"],
            phase=phase,
            identity_sha256=identity_sha,
            total=n_samples,
            block_size=int(block_size),
            payload=state["phase_payload"],
        )
        if replay_payload != state["phase_payload"]:
            raise ExternalMemoryDBSCANError(
                "all-core retained-centroid committed prefix does not replay"
            )
        ledger = dict(state["progress_ledger"])
        ledger["entries"] = list(ledger["entries"])
        for offset in range(start, n_samples, int(block_size)):
            stop = min(n_samples, offset + int(block_size))
            _scan_retained_range(
                labels=labels,
                vectors=recourse_vectors,
                mask=mask,
                start=offset,
                stop=stop,
                block_size=int(block_size),
                counts=retained_counts,
                sums=retained_sums,
            )
            payload = _retained_payload(retained_counts, retained_sums)
            _append_ledger(
                ledger,
                start=offset,
                stop=stop,
                payload=payload,
                identity_sha256=identity_sha,
            )
            state = _write_checkpoint(
                checkpoint_path,
                identity=identity,
                phase=phase,
                next_offset=stop,
                phase_payload=payload,
                ledger=ledger,
                artifacts=artifacts,
                peak_rss_bytes=peak,
                writer_guard=_writer_guard,
            )
        retained_centroids = np.full(
            (cluster_count, n_features), np.nan, dtype=dtype
        )
        nonempty = retained_counts > 0
        retained_centroids[nonempty] = np.asarray(
            retained_sums[nonempty]
            / retained_counts[nonempty, None].astype(dtype),
            dtype=dtype,
        )
        for name, values in (
            ("retained_counts", retained_counts),
            ("retained_centroids", retained_centroids),
        ):
            path = root / f"{name}.npy"
            artifacts[f"{name}_path"] = str(path)
            artifacts[f"{name}_sha256"] = _atomic_npy(
                path, values, writer_guard=_writer_guard
            )
        payload = _empty_medoid_payload(cluster_count)
        ledger = _new_ledger(
            phase="medoid_scan", identity_sha256=identity_sha
        )
        state = _write_checkpoint(
            checkpoint_path,
            identity=identity,
            phase="medoid_scan",
            next_offset=0,
            phase_payload=payload,
            ledger=ledger,
            artifacts=artifacts,
            peak_rss_bytes=peak,
            writer_guard=_writer_guard,
        )
        phase = "medoid_scan"

    retained_counts = np.asarray(
        _artifact_array(
            Path(artifacts["retained_counts_path"]),
            expected_sha256=artifacts["retained_counts_sha256"],
            shape=(cluster_count,),
            dtype=np.dtype(np.int64),
        )
    )
    retained_centroids = np.asarray(
        _artifact_array(
            Path(artifacts["retained_centroids_path"]),
            expected_sha256=artifacts["retained_centroids_sha256"],
            shape=(cluster_count, n_features),
            dtype=dtype,
        )
    )
    if phase == "medoid_scan":
        positions, distances = _medoid_state(state["phase_payload"])
        start = int(state["next_offset"])
        replay_positions = np.full(cluster_count, -1, dtype=np.int64)
        replay_distances = [float("inf")] * cluster_count
        _scan_medoid_range(
            labels=labels,
            vectors=recourse_vectors,
            mask=mask,
            retained_centroids=retained_centroids,
            start=0,
            stop=start,
            block_size=int(block_size),
            positions=replay_positions,
            distances=replay_distances,
        )
        replay_payload = _medoid_payload(replay_positions, replay_distances)
        _validate_ledger(
            state["progress_ledger"],
            phase=phase,
            identity_sha256=identity_sha,
            total=n_samples,
            block_size=int(block_size),
            payload=state["phase_payload"],
        )
        if replay_payload != state["phase_payload"]:
            raise ExternalMemoryDBSCANError(
                "all-core medoid committed prefix does not replay"
            )
        ledger = dict(state["progress_ledger"])
        ledger["entries"] = list(ledger["entries"])
        for offset in range(start, n_samples, int(block_size)):
            stop = min(n_samples, offset + int(block_size))
            _scan_medoid_range(
                labels=labels,
                vectors=recourse_vectors,
                mask=mask,
                retained_centroids=retained_centroids,
                start=offset,
                stop=stop,
                block_size=int(block_size),
                positions=positions,
                distances=distances,
            )
            payload = _medoid_payload(positions, distances)
            _append_ledger(
                ledger,
                start=offset,
                stop=stop,
                payload=payload,
                identity_sha256=identity_sha,
            )
            state = _write_checkpoint(
                checkpoint_path,
                identity=identity,
                phase=phase,
                next_offset=stop,
                phase_payload=payload,
                ledger=ledger,
                artifacts=artifacts,
                peak_rss_bytes=peak,
                writer_guard=_writer_guard,
            )
        state = _write_checkpoint(
            checkpoint_path,
            identity=identity,
            phase="finalize",
            next_offset=n_samples,
            phase_payload=_medoid_payload(positions, distances),
            ledger=ledger,
            artifacts=artifacts,
            peak_rss_bytes=peak,
            writer_guard=_writer_guard,
        )
        phase = "finalize"
    if phase != "finalize":
        raise ExternalMemoryDBSCANError(
            f"unknown all-core component summary phase: {phase}"
        )
    positions, distances = _medoid_state(state["phase_payload"])

    # Recompute every scientific scan before publishing PASS.  The checkpoint
    # is a progress hint; it is never accepted as terminal scientific truth.
    if _writer_guard is not None:
        _writer_guard()
    replay_counts = np.zeros(cluster_count, dtype=np.int64)
    replay_official_sums = np.zeros((cluster_count, n_features), dtype=dtype)
    replay_numpy_sums = np.zeros((cluster_count, n_features), dtype=dtype)
    replay_stable_sums = np.zeros((cluster_count, n_features), dtype=np.float64)
    _scan_centroid_range(
        labels=labels,
        vectors=recourse_vectors,
        start=0,
        stop=n_samples,
        block_size=int(block_size),
        counts=replay_counts,
        official_sums=replay_official_sums,
        numpy_sums=replay_numpy_sums,
        stable_sums=replay_stable_sums,
        torch_module=torch_module,
    )
    replay_official, replay_numpy, replay_stable = _centroids_from_sums(
        counts=replay_counts,
        official_sums=replay_official_sums,
        numpy_sums=replay_numpy_sums,
        stable_sums=replay_stable_sums,
        torch_module=torch_module,
    )
    if not (
        np.array_equal(replay_counts, counts)
        and np.array_equal(replay_official, official_centroids)
        and np.array_equal(replay_numpy, numpy_centroids)
        and np.array_equal(replay_stable, stable_centroids)
    ):
        raise ExternalMemoryDBSCANError(
            "terminal all-core centroid replay mismatch"
        )
    replay_membership = _membership_state(
        _empty_membership_payload(cluster_count), cluster_count=cluster_count
    )
    _scan_membership_range(
        labels=labels,
        vectors=recourse_vectors,
        pairs=pair_indices,
        official_centroids=official_centroids,
        numpy_centroids=numpy_centroids,
        stable_centroids=stable_centroids,
        radius=float(radius),
        start=0,
        stop=n_samples,
        block_size=int(block_size),
        trace_mask=mask,
        validate_mask=True,
        first=replay_membership[0],
        official_candidates=replay_membership[1],
        trace_parents=replay_membership[2],
        trace_candidates=replay_membership[3],
        official_within=replay_membership[4],
        trace_retained=replay_membership[5],
        exactly_delta=replay_membership[6],
        disagreements=replay_membership[7],
        torch_module=torch_module,
    )
    replay_retained_counts = np.zeros(cluster_count, dtype=np.int64)
    replay_retained_sums = np.zeros((cluster_count, n_features), dtype=dtype)
    _scan_retained_range(
        labels=labels,
        vectors=recourse_vectors,
        mask=mask,
        start=0,
        stop=n_samples,
        block_size=int(block_size),
        counts=replay_retained_counts,
        sums=replay_retained_sums,
    )
    replay_retained_centroids = np.full_like(retained_centroids, np.nan)
    nonempty = replay_retained_counts > 0
    replay_retained_centroids[nonempty] = np.asarray(
        replay_retained_sums[nonempty]
        / replay_retained_counts[nonempty, None].astype(dtype),
        dtype=dtype,
    )
    if not (
        np.array_equal(replay_retained_counts, retained_counts)
        and np.array_equal(
            replay_retained_centroids, retained_centroids, equal_nan=True
        )
    ):
        raise ExternalMemoryDBSCANError(
            "terminal all-core retained-centroid replay mismatch"
        )
    replay_positions = np.full(cluster_count, -1, dtype=np.int64)
    replay_distances = [float("inf")] * cluster_count
    _scan_medoid_range(
        labels=labels,
        vectors=recourse_vectors,
        mask=mask,
        retained_centroids=retained_centroids,
        start=0,
        stop=n_samples,
        block_size=int(block_size),
        positions=replay_positions,
        distances=replay_distances,
    )
    if (
        not np.array_equal(replay_positions, positions)
        or _float_hex(replay_distances) != _float_hex(distances)
    ):
        raise ExternalMemoryDBSCANError(
            "terminal all-core medoid replay mismatch"
        )
    official_result, selected, cluster_summaries = _results_from_replay(
        cluster_counts=counts,
        official_centroids=official_centroids,
        numpy_centroids=numpy_centroids,
        stable_centroids=stable_centroids,
        membership=replay_membership,
        retained_counts=retained_counts,
        medoid_positions=positions,
        medoid_distances=distances,
        pairs=pair_indices,
        radius=float(radius),
        theta=float(theta),
        recourse_size=int(recourse_size),
        official_greedy=official_greedy,
        torch_module=torch_module,
    )
    if storage == "theta_close_view_v1":
        (
            terminal_storage,
            _terminal_pair_path,
            terminal_authority,
            _terminal_pairs,
            terminal_close_view,
        ) = _pair_authority(
            pair_indices=pair_indices,
            pairs_sha256=pairs_sha256,
            pair_authority_manifest_path=pair_authority_path,
            pair_authority_manifest_sha256=pair_authority_manifest_sha256,
        )
        if (
            terminal_storage != storage
            or terminal_authority != pair_authority_path
            or terminal_close_view is None
            or terminal_close_view.vectors_path != vector_path
            or terminal_close_view.vectors_sha256
            != dbscan["scientific_identity"]["vectors_sha256"]
        ):
            raise ExternalMemoryDBSCANError(
                "all-core summary implicit pair authority drifted before PASS"
            )
    if (
        _sha256_file(dbscan_path) != str(dbscan_manifest_sha256)
        or _sha256_file(vector_path)
        != dbscan["scientific_identity"]["vectors_sha256"]
        or _sha256_file(label_path) != dbscan["labels_sha256"]
        or (pair_path is not None and _sha256_file(pair_path) != str(pairs_sha256))
        or (
            pair_authority_path is not None
            and _sha256_file(pair_authority_path)
            != str(pair_authority_manifest_sha256)
        )
    ):
        raise ExternalMemoryDBSCANError(
            "all-core component summary source changed before PASS"
        )
    _validate_dbscan_source(
        dbscan_manifest_path=dbscan_path,
        dbscan_manifest_sha256=dbscan_manifest_sha256,
        labels=labels,
        recourse_vectors=recourse_vectors,
    )
    official_parent_to_clusters: dict[int, set[int]] = defaultdict(set)
    trace_parent_to_clusters: dict[int, set[int]] = defaultdict(set)
    for row in cluster_summaries:
        for parent in row["official_covered_parent_indices"]:
            official_parent_to_clusters[int(parent)].add(int(row["cluster_id"]))
        for parent in row["trace_covered_parent_indices"]:
            trace_parent_to_clusters[int(parent)].add(int(row["cluster_id"]))
    manifest = {
        "schema_version": ALL_CORE_COMPONENT_SUMMARY_SCHEMA,
        "status": "PASS",
        "run_complete": True,
        "scientific_identity": identity,
        "scientific_identity_sha256": identity_sha,
        "writer_lock_identity": dict(_writer_lock_identity or {}),
        "exact_all_core_component_semantics_replayed": True,
        "cluster_count": cluster_count,
        "noise_count": 0,
        "num_samples": n_samples,
        "cluster_summaries": cluster_summaries,
        "official_result": [list(value) for value in official_result],
        "selected": selected,
        "selected_common_recourse_count": len(selected),
        "official_coverage_function_invoked": False,
        "official_coverage_semantics_streamed_for_all_components": True,
        "official_greedy_invoked": True,
        "greedy_tie_break": "ascending_canonical_cluster_id",
        "no_cluster_duplicated_to_fill_recourse_size": True,
        "strict_radius_comparison_preserved": True,
        "radius_filter_operator": "<",
        "centroid_norm_filter_operator": "<",
        "coverage_pair_orientation": "col0_parent_col1_candidate",
        "official_parent_to_covering_clusters": {
            str(parent): sorted(values)
            for parent, values in sorted(official_parent_to_clusters.items())
        },
        "trace_numpy_parent_to_covering_clusters": {
            str(parent): sorted(values)
            for parent, values in sorted(trace_parent_to_clusters.items())
        },
        "large_cluster_advanced_index_copy": False,
        "largest_materialized_vector_block_rows": min(
            int(block_size), n_samples
        ),
        "full_cluster_vector_bytes_materialized": 0,
        "deterministic_streaming_reduction_order": (
            "fixed_global_row_order_with_frozen_block_size"
        ),
        "centroid_reduction_classification": "PROJECT_EXTENSION",
        "upstream_one_shot_torch_mean_bit_identical": False,
        "paper_control_flow_reproduced_with_streaming_reduction": True,
        "streaming_block_size": int(block_size),
        "float64_radius_decision_disagreement_count": 0,
        "float64_theta_decision_disagreement_cluster_count": 0,
        "numeric_decision_disagreement_policy": "FAIL_CLOSED",
        "official_float32_and_stable_float64_centroids_recorded": True,
        "terminal_full_replay_complete": True,
        "checkpoint_is_progress_hint_only": True,
        "approximation_used": False,
        "peak_rss_bytes_observed": max(peak, _rss_bytes()),
        "max_rss_bytes": int(max_rss_bytes),
        **artifacts,
        "completed_at": _utc_now(),
    }
    if int(manifest["peak_rss_bytes_observed"]) > int(max_rss_bytes):
        raise ExternalMemoryDBSCANError(
            "all-core component summary peak RSS exceeded budget"
        )
    if _writer_guard is not None:
        _writer_guard()
    _atomic_json(
        manifest_path, manifest, writer_guard=_writer_guard
    )  # PASS is published last.
    if _writer_guard is not None:
        try:
            _writer_guard()
        except Exception:
            _revoke_terminal_manifest(root)
            raise
    return validate_proven_all_core_component_summary(
        manifest_path,
        torch_module=torch_module,
        pair_indices=pair_indices,
        full_replay=False,
    )


def summarize_proven_all_core_components_external(
    *,
    work_dir: str | Path,
    dbscan_manifest_path: str | Path,
    dbscan_manifest_sha256: str,
    labels: np.ndarray,
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
) -> ExactAllCoreComponentSummaryResult:
    """Run or resume one summary under an invocation-wide writer lock."""

    requested_root = Path(work_dir).expanduser()
    if requested_root.is_symlink():
        raise ExternalMemoryDBSCANError(
            "all-core component summary root may not be a symlink"
        )
    root = requested_root.resolve(strict=False)
    with _exclusive_summary_writer_lock(root) as (fresh, held):
        try:
            result = _summarize_proven_all_core_components_external_locked(
                work_dir=root,
                dbscan_manifest_path=dbscan_manifest_path,
                dbscan_manifest_sha256=dbscan_manifest_sha256,
                labels=labels,
                recourse_vectors=recourse_vectors,
                pair_indices=pair_indices,
                pairs_sha256=pairs_sha256,
                pair_authority_manifest_path=pair_authority_manifest_path,
                pair_authority_manifest_sha256=pair_authority_manifest_sha256,
                radius=radius,
                theta=theta,
                recourse_size=recourse_size,
                official_greedy=official_greedy,
                torch_module=torch_module,
                max_rss_bytes=max_rss_bytes,
                block_size=block_size,
                resume=resume,
                _fresh_root_claimed=fresh,
                _writer_guard=held.verify,
                _writer_lock_identity=held.identity(),
            )
            held.verify()
            return result
        except Exception:
            try:
                held.verify()
            except Exception:
                _revoke_terminal_manifest(root)
            raise


def validate_proven_all_core_component_summary(
    manifest_path: str | Path,
    *,
    torch_module: Any | None = None,
    pair_indices: Any | None = None,
    full_replay: bool = True,
) -> ExactAllCoreComponentSummaryResult:
    """Reopen the terminal manifest and its complete hash closure.

    ``full_replay`` is reserved for the fresh final gate.  Creation has just
    completed a full terminal replay and therefore reopens hashes with the
    flag disabled to avoid a fourth redundant scan.
    """

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = _load_object(path)
    identity = manifest.get("scientific_identity")
    if (
        manifest.get("schema_version") != ALL_CORE_COMPONENT_SUMMARY_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("run_complete") is not True
        or not isinstance(identity, Mapping)
        or manifest.get("scientific_identity_sha256") != _stable_hash(identity)
        or manifest.get("exact_all_core_component_semantics_replayed") is not True
        or manifest.get("terminal_full_replay_complete") is not True
        or manifest.get("large_cluster_advanced_index_copy") is not False
        or manifest.get("approximation_used") is not False
        or identity.get("centroid_reduction")
        != (
            "PROJECT_EXTENSION:fixed_global_row_order_fixed_block_torch_float32_"
            "plus_float64_audit_v1:not_upstream_one_shot_bit_identical"
        )
        or manifest.get("centroid_reduction_classification")
        != "PROJECT_EXTENSION"
        or manifest.get("upstream_one_shot_torch_mean_bit_identical") is not False
        or manifest.get("numeric_decision_disagreement_policy") != "FAIL_CLOSED"
        or int(manifest.get("float64_radius_decision_disagreement_count", -1))
        != 0
        or int(
            manifest.get("float64_theta_decision_disagreement_cluster_count", -1)
        )
        != 0
    ):
        raise ExternalMemoryDBSCANError(
            "completed all-core component summary manifest is invalid"
        )
    root = path.parent.resolve(strict=True)
    terminal_lock_identity = manifest.get("writer_lock_identity")
    if (
        not isinstance(terminal_lock_identity, Mapping)
        or dict(terminal_lock_identity) != _current_writer_lock_identity(root)
    ):
        raise ExternalMemoryDBSCANError(
            "completed all-core writer lock identity mismatch"
        )
    n_samples, n_features = map(int, identity["vectors_shape"])
    cluster_count = int(identity["dbscan_cluster_count"])
    dtype = np.dtype(identity["vectors_dtype"])
    _validate_owner_claim(
        root, identity_sha256=str(manifest["scientific_identity_sha256"])
    )
    dbscan_path = Path(identity["dbscan_manifest_path"]).resolve(strict=True)
    label_path = Path(identity["labels_path"]).resolve(strict=True)
    vector_path = Path(identity["vectors_path"]).resolve(strict=True)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    _validate_dbscan_source(
        dbscan_manifest_path=dbscan_path,
        dbscan_manifest_sha256=identity["dbscan_manifest_sha256"],
        labels=labels,
        recourse_vectors=vectors,
    )
    if identity["pairs_storage"] not in {
        "physical_npy",
        "theta_close_view_v1",
    }:
        raise ExternalMemoryDBSCANError(
            "completed all-core pair storage contract is invalid"
        )
    if identity["pairs_storage"] == "physical_npy":
        pairs_path = Path(identity["pairs_path"]).resolve(strict=True)
        if _sha256_file(pairs_path) != identity["pairs_sha256"]:
            raise ExternalMemoryDBSCANError(
                "completed all-core physical pair source changed"
            )
        pairs: Any = (
            np.load(pairs_path, mmap_mode="r", allow_pickle=False)
            if pair_indices is None
            else pair_indices
        )
        supplied_path = str(getattr(pairs, "filename", "") or "")
        if (
            pairs.shape != (n_samples, 2)
            or pairs.dtype != np.dtype(np.int64)
            or (
                supplied_path
                and Path(supplied_path).resolve(strict=True) != pairs_path
            )
        ):
            raise ExternalMemoryDBSCANError(
                "completed all-core physical pair view mismatch"
            )
    else:
        authority = Path(identity["pair_authority_manifest_path"]).resolve(
            strict=True
        )
        if _sha256_file(authority) != identity["pair_authority_manifest_sha256"]:
            raise ExternalMemoryDBSCANError(
                "completed all-core implicit pair authority changed"
            )
        from .close_pair_view import validate_theta_close_pair_view

        try:
            close_view = validate_theta_close_pair_view(
                authority,
                require_dbscan_eligible=True,
                require_pair_semantics_authority=True,
            )
            pairs = close_view.open_pairs()
        except Exception as exc:
            raise ExternalMemoryDBSCANError(
                "completed all-core implicit pair authority cannot be reopened"
            ) from exc
        authoritative_filename = str(getattr(pairs, "filename", "") or "")
        authoritative_path = (
            None
            if not authoritative_filename
            else Path(authoritative_filename).resolve(strict=True)
        )
        identity_pair_path = (
            None
            if identity.get("pairs_path") is None
            else Path(identity["pairs_path"]).resolve(strict=True)
        )
        if (
            close_view.logical_close_rows != n_samples
            or close_view.pairs_sha256 != identity["pairs_sha256"]
            or close_view.vectors_path != vector_path
            or close_view.vectors_sha256 != identity["vectors_sha256"]
            or pairs.shape != (n_samples, 2)
            or pairs.dtype != np.dtype(np.int64)
            or authoritative_path != identity_pair_path
            or (
                authoritative_path is None
                and getattr(pairs, "logical_npy_sha256", None)
                != identity["pairs_sha256"]
            )
            or (
                authoritative_path is not None
                and _sha256_file(authoritative_path)
                != identity["pairs_sha256"]
            )
            or (
                pair_indices is not None
                and (
                    pair_indices.shape != (n_samples, 2)
                    or pair_indices.dtype != np.dtype(np.int64)
                )
            )
        ):
            raise ExternalMemoryDBSCANError(
                "completed all-core implicit pair view mismatch"
            )
    artifact_specs = {
        "cluster_counts": ((cluster_count,), np.dtype(np.int64)),
        "official_float32_centroids": ((cluster_count, n_features), dtype),
        "trace_numpy_centroids": ((cluster_count, n_features), dtype),
        "stable_float64_centroids": (
            (cluster_count, n_features),
            np.dtype(np.float64),
        ),
        "retained_mask": ((n_samples,), np.dtype(np.bool_)),
        "retained_counts": ((cluster_count,), np.dtype(np.int64)),
        "retained_centroids": ((cluster_count, n_features), dtype),
    }
    opened: dict[str, np.ndarray] = {}
    for name, (shape, expected_dtype) in artifact_specs.items():
        artifact_path = Path(str(manifest.get(f"{name}_path") or "")).resolve(
            strict=True
        )
        if artifact_path.parent != root:
            raise ExternalMemoryDBSCANError(
                f"completed all-core {name} escaped summary root"
            )
        opened[name] = _artifact_array(
            artifact_path,
            expected_sha256=str(manifest.get(f"{name}_sha256") or ""),
            shape=shape,
            dtype=expected_dtype,
        )
    streamed_counts, _minima = _stream_label_counts_and_minima(
        labels,
        cluster_count=cluster_count,
        block_size=int(identity["block_size"]),
    )
    streamed_retained_counts, _ = _stream_label_counts_and_minima(
        labels,
        cluster_count=cluster_count,
        block_size=int(identity["block_size"]),
        mask=opened["retained_mask"],
    )
    if (
        int(np.sum(opened["cluster_counts"])) != n_samples
        or not np.array_equal(streamed_counts, opened["cluster_counts"])
        or not np.array_equal(
            streamed_retained_counts, opened["retained_counts"]
        )
    ):
        raise ExternalMemoryDBSCANError(
            "completed all-core cluster/mask counts do not close"
        )
    official = manifest.get("official_result")
    selected = manifest.get("selected")
    if (
        not isinstance(official, list)
        or len(official) != 3
        or not all(isinstance(value, list) for value in official)
        or not isinstance(selected, list)
        or not isinstance(manifest.get("cluster_summaries"), list)
        or len(manifest["cluster_summaries"]) != cluster_count
    ):
        raise ExternalMemoryDBSCANError(
            "completed all-core result schema mismatch"
        )
    if full_replay:
        if pairs is None:
            raise ExternalMemoryDBSCANError(
                "full all-core replay requires the live implicit pair view"
            )
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        counts = np.zeros(cluster_count, dtype=np.int64)
        official_sums = np.zeros((cluster_count, n_features), dtype=dtype)
        numpy_sums = np.zeros((cluster_count, n_features), dtype=dtype)
        stable_sums = np.zeros((cluster_count, n_features), dtype=np.float64)
        _scan_centroid_range(
            labels=labels,
            vectors=vectors,
            start=0,
            stop=n_samples,
            block_size=int(identity["block_size"]),
            counts=counts,
            official_sums=official_sums,
            numpy_sums=numpy_sums,
            stable_sums=stable_sums,
            torch_module=torch_module,
        )
        official_centroids, numpy_centroids, stable_centroids = (
            _centroids_from_sums(
                counts=counts,
                official_sums=official_sums,
                numpy_sums=numpy_sums,
                stable_sums=stable_sums,
                torch_module=torch_module,
            )
        )
        if not (
            np.array_equal(official_centroids, opened["official_float32_centroids"])
            and np.array_equal(numpy_centroids, opened["trace_numpy_centroids"])
            and np.array_equal(stable_centroids, opened["stable_float64_centroids"])
        ):
            raise ExternalMemoryDBSCANError(
                "completed all-core centroid terminal replay mismatch"
            )
        replay_membership = _membership_state(
            _empty_membership_payload(cluster_count),
            cluster_count=cluster_count,
        )
        _scan_membership_range(
            labels=labels,
            vectors=vectors,
            pairs=pairs,
            official_centroids=official_centroids,
            numpy_centroids=numpy_centroids,
            stable_centroids=stable_centroids,
            radius=float(identity["radius"]),
            start=0,
            stop=n_samples,
            block_size=int(identity["block_size"]),
            trace_mask=opened["retained_mask"],
            validate_mask=True,
            first=replay_membership[0],
            official_candidates=replay_membership[1],
            trace_parents=replay_membership[2],
            trace_candidates=replay_membership[3],
            official_within=replay_membership[4],
            trace_retained=replay_membership[5],
            exactly_delta=replay_membership[6],
            disagreements=replay_membership[7],
            torch_module=torch_module,
        )
        replay_retained_counts = np.zeros(cluster_count, dtype=np.int64)
        replay_retained_sums = np.zeros((cluster_count, n_features), dtype=dtype)
        _scan_retained_range(
            labels=labels,
            vectors=vectors,
            mask=opened["retained_mask"],
            start=0,
            stop=n_samples,
            block_size=int(identity["block_size"]),
            counts=replay_retained_counts,
            sums=replay_retained_sums,
        )
        replay_retained_centroids = np.full(
            (cluster_count, n_features), np.nan, dtype=dtype
        )
        nonempty = replay_retained_counts > 0
        replay_retained_centroids[nonempty] = np.asarray(
            replay_retained_sums[nonempty]
            / replay_retained_counts[nonempty, None].astype(dtype),
            dtype=dtype,
        )
        if not (
            np.array_equal(replay_retained_counts, opened["retained_counts"])
            and np.array_equal(
                replay_retained_centroids,
                opened["retained_centroids"],
                equal_nan=True,
            )
        ):
            raise ExternalMemoryDBSCANError(
                "completed all-core retained centroid terminal replay mismatch"
            )
        replay_positions = np.full(cluster_count, -1, dtype=np.int64)
        replay_distances = [float("inf")] * cluster_count
        _scan_medoid_range(
            labels=labels,
            vectors=vectors,
            mask=opened["retained_mask"],
            retained_centroids=opened["retained_centroids"],
            start=0,
            stop=n_samples,
            block_size=int(identity["block_size"]),
            positions=replay_positions,
            distances=replay_distances,
        )

        def deterministic_greedy(
            counterfactual_covering: Mapping[int, set[int]],
            graphs_covered_by: Mapping[int, set[int]],
            k: int,
        ) -> dict[int, tuple[int, int]]:
            del graphs_covered_by
            return _deterministic_greedy(counterfactual_covering, k=int(k))

        replay_official, replay_selected, replay_clusters = _results_from_replay(
            cluster_counts=counts,
            official_centroids=official_centroids,
            numpy_centroids=numpy_centroids,
            stable_centroids=stable_centroids,
            membership=replay_membership,
            retained_counts=replay_retained_counts,
            medoid_positions=replay_positions,
            medoid_distances=replay_distances,
            pairs=pairs,
            radius=float(identity["radius"]),
            theta=float(identity["theta"]),
            recourse_size=int(identity["recourse_size"]),
            official_greedy=deterministic_greedy,
            torch_module=torch_module,
        )
        if (
            [list(value) for value in replay_official] != official
            or replay_selected != selected
            or replay_clusters != manifest["cluster_summaries"]
        ):
            raise ExternalMemoryDBSCANError(
                "completed all-core coverage/greedy terminal replay mismatch"
            )
    return ExactAllCoreComponentSummaryResult(
        official_result=(list(official[0]), list(official[1]), list(official[2])),
        selected=[dict(value) for value in selected],
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        retained_mask_path=Path(manifest["retained_mask_path"]).resolve(
            strict=True
        ),
    )


__all__ = [
    "ALL_CORE_COMPONENT_SUMMARY_SCHEMA",
    "ExactAllCoreComponentSummaryResult",
    "summarize_proven_all_core_components_external",
    "validate_proven_all_core_component_summary",
]
