"""Exact, resumable DBSCAN without materializing all epsilon neighborhoods.

``sklearn.cluster.DBSCAN`` first asks ``NearestNeighbors`` for every radius
neighborhood and retains the resulting object array for the entire fit.  That
is scientifically convenient but has quadratic peak memory for dense radius
graphs.  This module uses the same sklearn radius query, in bounded blocks,
and reconstructs the exact DBSCAN labelling from its graph definition:

* core status is the radius-neighbor count (including self);
* core points in the same epsilon-connected component share a cluster;
* components are numbered by their first core sample; and
* an ambiguous border point belongs to the earliest numbered adjacent core
  component.

Those rules are equivalent to sklearn's ordered ``dbscan_inner`` traversal.
The three passes are checkpointed independently.  Partial arrays live only in
the caller-owned fresh work directory and are fsynced before an atomic state
update, so replaying the last incomplete block is idempotent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "comrecgc_external_memory_dbscan_v1"


class ExternalMemoryDBSCANError(RuntimeError):
    """Raised when exact external-memory clustering cannot be proven."""


@dataclass(frozen=True)
class ExternalDBSCANContract:
    eps: float
    min_samples: int
    query_block_size: int
    checkpoint_interval_blocks: int
    max_rss_bytes: int
    expected_sklearn_version: str

    def validate(self) -> None:
        if not np.isfinite(float(self.eps)) or float(self.eps) <= 0:
            raise ExternalMemoryDBSCANError("eps must be finite and positive")
        if int(self.min_samples) <= 0:
            raise ExternalMemoryDBSCANError("min_samples must be positive")
        if int(self.query_block_size) <= 0:
            raise ExternalMemoryDBSCANError("query_block_size must be positive")
        if int(self.checkpoint_interval_blocks) <= 0:
            raise ExternalMemoryDBSCANError(
                "checkpoint_interval_blocks must be positive"
            )
        if int(self.max_rss_bytes) <= 0:
            raise ExternalMemoryDBSCANError("max_rss_bytes must be positive")
        if not str(self.expected_sklearn_version):
            raise ExternalMemoryDBSCANError(
                "the sklearn version must be explicitly frozen"
            )


@dataclass(frozen=True)
class ExternalDBSCANResult:
    labels_path: Path
    core_mask_path: Path
    neighbor_counts_path: Path
    manifest_path: Path
    num_samples: int
    num_features: int
    cluster_count: int
    noise_count: int
    core_count: int
    manifest_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
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


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalMemoryDBSCANError(f"invalid checkpoint JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExternalMemoryDBSCANError(f"checkpoint is not an object: {path}")
    return value


def _fsync_memmap(value: np.memmap) -> None:
    value.flush()
    with Path(value.filename).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        fields: dict[str, int] = {}
        for line in status.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in {"VmRSS", "VmHWM"}:
                fields[parts[0].rstrip(":")] = int(parts[1]) * 1024
        if fields:
            return max(fields.values())
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return raw if sys.platform == "darwin" else raw * 1024


def _check_rss(limit: int, *, phase: str, reserved_bytes: int = 0) -> int:
    current = _rss_bytes()
    if current + int(reserved_bytes) > int(limit):
        raise ExternalMemoryDBSCANError(
            "RSS_BUDGET_EXCEEDED:"
            f"phase={phase}:rss={current}:reserved={reserved_bytes}:limit={limit}"
        )
    return current


def _open_npy_memmap(path: Path, *, mode: str) -> np.memmap:
    try:
        value = np.load(path, mmap_mode=mode, allow_pickle=False)
    except Exception as exc:
        raise ExternalMemoryDBSCANError(f"cannot open ndarray artifact: {path}") from exc
    if not isinstance(value, np.memmap):
        raise ExternalMemoryDBSCANError(f"array is not memory mapped: {path}")
    return value


def _new_memmap(path: Path, *, dtype: Any, shape: Sequence[int]) -> np.memmap:
    if path.exists() or path.is_symlink():
        raise ExternalMemoryDBSCANError(f"fresh partial array already exists: {path}")
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=tuple(int(value) for value in shape),
    )


def _checkpoint(
    state_path: Path,
    *,
    identity: Mapping[str, Any],
    phase: str,
    next_offset: int,
    peak_rss_bytes: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(identity),
        "identity_sha256": _stable_hash(identity),
        "phase": phase,
        "next_offset": int(next_offset),
        "peak_rss_bytes": int(peak_rss_bytes),
        "updated_at": _utc_now(),
    }
    if extra:
        payload.update(dict(extra))
    _atomic_json(state_path, payload)


def _query_reservation_bytes(num_samples: int, rows: int) -> int:
    # Worst case: one intp neighbor id per query/sample plus the object array,
    # temporary masks, and one equally large safety allowance.
    ids = int(num_samples) * int(rows) * np.dtype(np.intp).itemsize
    return 2 * ids + int(rows) * 4096 + 64 * 1024**2


def _bounded_block_size(
    *, requested: int, num_samples: int, max_rss_bytes: int, phase: str
) -> int:
    current = _check_rss(max_rss_bytes, phase=phase)
    available = int(max_rss_bytes) - current - 64 * 1024**2
    per_row = max(1, 2 * int(num_samples) * np.dtype(np.intp).itemsize + 4096)
    safe = available // per_row
    if safe < 1:
        raise ExternalMemoryDBSCANError(
            f"RSS budget cannot hold one worst-case neighborhood row during {phase}"
        )
    return max(1, min(int(requested), int(safe)))


def _fit_neighbors(vectors: np.ndarray, *, eps: float) -> tuple[Any, str, str]:
    try:
        import sklearn
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover - dependency gate
        raise ExternalMemoryDBSCANError(
            "external-memory DBSCAN requires scikit-learn"
        ) from exc
    model = NearestNeighbors(
        radius=float(eps),
        algorithm="auto",
        leaf_size=30,
        metric="euclidean",
        metric_params=None,
        p=None,
        n_jobs=None,
    )
    model.fit(vectors)
    method = str(getattr(model, "_fit_method", "UNKNOWN"))
    return model, str(sklearn.__version__), method


def _roots(parent: np.ndarray, values: np.ndarray) -> np.ndarray:
    roots = np.asarray(values, dtype=np.intp).copy()
    while roots.size:
        next_roots = np.asarray(parent[roots], dtype=np.intp)
        if np.array_equal(next_roots, roots):
            break
        roots = next_roots
    return roots


def _validate_partial(
    value: np.ndarray, *, shape: Sequence[int], dtype: Any, label: str
) -> None:
    if value.shape != tuple(shape) or value.dtype != np.dtype(dtype):
        raise ExternalMemoryDBSCANError(
            f"{label} shape/dtype mismatch: {value.shape}/{value.dtype}"
        )


def _reconcile_promoted_array(
    *,
    partial: Path,
    final: Path,
    shape: Sequence[int],
    dtype: Any,
    expected_sha256: str,
    label: str,
) -> Path:
    """Finish one checkpointed rename without trusting either filename."""

    if partial.exists() and final.exists():
        raise ExternalMemoryDBSCANError(
            f"{label} has both partial and final artifacts"
        )
    source = final if final.exists() else partial
    if not source.exists() or source.is_symlink():
        raise ExternalMemoryDBSCANError(
            f"{label} checkpointed artifact is missing"
        )
    value = _open_npy_memmap(source, mode="r")
    _validate_partial(value, shape=shape, dtype=dtype, label=label)
    del value
    if _sha256_file(source) != str(expected_sha256):
        raise ExternalMemoryDBSCANError(f"{label} checkpoint hash mismatch")
    if source == partial:
        os.replace(partial, final)
        _fsync_directory(final.parent)
    return final


def fit_external_memory_dbscan(
    *,
    vectors_path: str | Path,
    work_dir: str | Path,
    contract: ExternalDBSCANContract,
    expected_vectors_sha256: str | None = None,
    resume: bool = False,
) -> ExternalDBSCANResult:
    """Fit exact sklearn-compatible DBSCAN using three bounded passes."""

    contract.validate()
    source = Path(vectors_path).expanduser().resolve(strict=True)
    root = Path(work_dir).expanduser().resolve(strict=False)
    state_path = root / "checkpoint.json"
    final_manifest_path = root / "run_manifest.json"
    if final_manifest_path.exists():
        manifest = _load_object(final_manifest_path)
        if manifest.get("run_complete") is not True:
            raise ExternalMemoryDBSCANError("terminal DBSCAN manifest is incomplete")
        scientific_identity = manifest.get("scientific_identity")
        if not isinstance(scientific_identity, Mapping):
            raise ExternalMemoryDBSCANError("terminal DBSCAN identity is absent")
        if scientific_identity.get("vectors_path") != str(source):
            raise ExternalMemoryDBSCANError("terminal DBSCAN vector path mismatch")
        if scientific_identity.get("contract") != asdict(contract):
            raise ExternalMemoryDBSCANError("terminal DBSCAN contract mismatch")
        actual_source_sha = _sha256_file(source)
        if (
            scientific_identity.get("vectors_sha256") != actual_source_sha
            or (
                expected_vectors_sha256 is not None
                and actual_source_sha != expected_vectors_sha256
            )
        ):
            raise ExternalMemoryDBSCANError("terminal DBSCAN vector hash mismatch")
        if manifest.get("scientific_identity_sha256") != _stable_hash(
            scientific_identity
        ):
            raise ExternalMemoryDBSCANError("terminal DBSCAN identity hash mismatch")
        labels_path = Path(str(manifest["labels_path"])).resolve(strict=True)
        core_path = Path(str(manifest["core_mask_path"])).resolve(strict=True)
        counts_path = Path(str(manifest["neighbor_counts_path"])).resolve(strict=True)
        for path, field in (
            (labels_path, "labels_sha256"),
            (core_path, "core_mask_sha256"),
            (counts_path, "neighbor_counts_sha256"),
        ):
            if path.parent != root or _sha256_file(path) != manifest.get(field):
                raise ExternalMemoryDBSCANError(
                    f"terminal DBSCAN artifact closure mismatch: {field}"
                )
        return ExternalDBSCANResult(
            labels_path=labels_path,
            core_mask_path=core_path,
            neighbor_counts_path=counts_path,
            manifest_path=final_manifest_path,
            num_samples=int(manifest["num_samples"]),
            num_features=int(manifest["num_features"]),
            cluster_count=int(manifest["cluster_count"]),
            noise_count=int(manifest["noise_count"]),
            core_count=int(manifest["core_count"]),
            manifest_sha256=_sha256_file(final_manifest_path),
        )
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"external DBSCAN work directory is non-empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    vectors = _open_npy_memmap(source, mode="r")
    if vectors.ndim != 2 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ExternalMemoryDBSCANError("vectors must be a nonempty 2-D ndarray")
    if vectors.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ExternalMemoryDBSCANError(
            f"vectors must preserve float32/float64 semantics, got {vectors.dtype}"
        )
    n_samples, n_features = (int(vectors.shape[0]), int(vectors.shape[1]))
    actual_vectors_sha = _sha256_file(source)
    if expected_vectors_sha256 and actual_vectors_sha != expected_vectors_sha256:
        raise ExternalMemoryDBSCANError("recourse vector SHA256 mismatch")
    neighbors, sklearn_version, fit_method = _fit_neighbors(
        vectors, eps=float(contract.eps)
    )
    if sklearn_version != contract.expected_sklearn_version:
        raise ExternalMemoryDBSCANError(
            "SKLEARN_VERSION_MISMATCH:"
            f"actual={sklearn_version}:expected={contract.expected_sklearn_version}"
        )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "vectors_path": str(source),
        "vectors_sha256": actual_vectors_sha,
        "vectors_dtype": str(vectors.dtype),
        "vectors_shape": [n_samples, n_features],
        "contract": asdict(contract),
        "sklearn_version": sklearn_version,
        "nearest_neighbors_fit_method": fit_method,
        "nearest_neighbors_metric": "euclidean",
        "nearest_neighbors_algorithm": "auto",
        "border_assignment": "minimum_cluster_label_of_adjacent_core_component",
    }
    state: dict[str, Any]
    if state_path.exists():
        state = _load_object(state_path)
        if not resume:
            raise ExternalMemoryDBSCANError("checkpoint exists but resume=false")
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("identity_sha256") != _stable_hash(identity)
            or state.get("identity") != identity
        ):
            raise ExternalMemoryDBSCANError("checkpoint scientific identity mismatch")
    else:
        state = {
            "phase": "neighbor_counts",
            "next_offset": 0,
            "peak_rss_bytes": _rss_bytes(),
        }
        _checkpoint(
            state_path,
            identity=identity,
            phase="neighbor_counts",
            next_offset=0,
            peak_rss_bytes=int(state["peak_rss_bytes"]),
        )

    peak = max(int(state.get("peak_rss_bytes", 0)), _rss_bytes())
    counts_partial = root / "neighbor_counts.partial.npy"
    counts_final = root / "neighbor_counts.npy"
    core_path = root / "core_mask.npy"
    parent_partial = root / "core_union_parent.partial.npy"
    labels_partial = root / "labels.partial.npy"
    labels_final = root / "labels.npy"

    phase = str(state.get("phase"))
    if phase == "neighbor_counts_finalize":
        _reconcile_promoted_array(
            partial=counts_partial,
            final=counts_final,
            shape=(n_samples,),
            dtype=np.intp,
            expected_sha256=str(state.get("neighbor_counts_sha256") or ""),
            label="neighbor counts",
        )
        _reconcile_promoted_array(
            partial=root / "core_mask.partial.npy",
            final=core_path,
            shape=(n_samples,),
            dtype=np.bool_,
            expected_sha256=str(state.get("core_mask_sha256") or ""),
            label="core mask",
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase="core_union",
            next_offset=0,
            peak_rss_bytes=peak,
            extra={
                "effective_query_block_size": int(
                    state.get("effective_query_block_size", contract.query_block_size)
                )
            },
        )
        phase = "core_union"
        state = _load_object(state_path)
    if phase == "neighbor_counts":
        if counts_partial.exists():
            counts = _open_npy_memmap(counts_partial, mode="r+")
            _validate_partial(
                counts, shape=(n_samples,), dtype=np.intp, label="neighbor counts"
            )
        else:
            if int(state.get("next_offset", 0)) != 0:
                raise ExternalMemoryDBSCANError("missing neighbor-count partial")
            counts = _new_memmap(
                counts_partial, dtype=np.intp, shape=(n_samples,)
            )
        start = int(state.get("next_offset", 0))
        block = _bounded_block_size(
            requested=contract.query_block_size,
            num_samples=n_samples,
            max_rss_bytes=contract.max_rss_bytes,
            phase="neighbor_counts",
        )
        blocks_since_checkpoint = 0
        for offset in range(start, n_samples, block):
            stop = min(n_samples, offset + block)
            reserved = _query_reservation_bytes(n_samples, stop - offset)
            _check_rss(
                contract.max_rss_bytes,
                phase="neighbor_counts.query",
                reserved_bytes=reserved,
            )
            neighborhoods = neighbors.radius_neighbors(
                vectors[offset:stop], return_distance=False
            )
            counts[offset:stop] = np.fromiter(
                (len(row) for row in neighborhoods),
                dtype=np.intp,
                count=stop - offset,
            )
            if np.any(counts[offset:stop] < 1):
                raise ExternalMemoryDBSCANError("radius query omitted a self neighbor")
            del neighborhoods
            peak = max(peak, _check_rss(contract.max_rss_bytes, phase="neighbor_counts"))
            blocks_since_checkpoint += 1
            if (
                blocks_since_checkpoint >= contract.checkpoint_interval_blocks
                or stop == n_samples
            ):
                _fsync_memmap(counts)
                _checkpoint(
                    state_path,
                    identity=identity,
                    phase="neighbor_counts",
                    next_offset=stop,
                    peak_rss_bytes=peak,
                    extra={"effective_query_block_size": block},
                )
                blocks_since_checkpoint = 0
        _fsync_memmap(counts)
        core = np.asarray(counts >= int(contract.min_samples), dtype=np.bool_)
        temporary_core = core_path.with_name("core_mask.partial.npy")
        with temporary_core.open("wb") as handle:
            np.save(handle, core, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        counts_sha256 = _sha256_file(counts_partial)
        core_sha256 = _sha256_file(temporary_core)
        _checkpoint(
            state_path,
            identity=identity,
            phase="neighbor_counts_finalize",
            next_offset=n_samples,
            peak_rss_bytes=peak,
            extra={
                "effective_query_block_size": block,
                "neighbor_counts_sha256": counts_sha256,
                "core_mask_sha256": core_sha256,
            },
        )
        del counts
        _reconcile_promoted_array(
            partial=counts_partial,
            final=counts_final,
            shape=(n_samples,),
            dtype=np.intp,
            expected_sha256=counts_sha256,
            label="neighbor counts",
        )
        _reconcile_promoted_array(
            partial=temporary_core,
            final=core_path,
            shape=(n_samples,),
            dtype=np.bool_,
            expected_sha256=core_sha256,
            label="core mask",
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase="core_union",
            next_offset=0,
            peak_rss_bytes=peak,
            extra={"effective_query_block_size": block},
        )
        phase = "core_union"
        state = _load_object(state_path)

    counts = _open_npy_memmap(counts_final, mode="r")
    core = np.load(core_path, mmap_mode="r", allow_pickle=False)
    _validate_partial(core, shape=(n_samples,), dtype=np.bool_, label="core mask")
    core_indices = np.flatnonzero(core).astype(np.intp, copy=False)

    if phase == "core_union":
        if parent_partial.exists():
            parent = _open_npy_memmap(parent_partial, mode="r+")
            _validate_partial(parent, shape=(n_samples,), dtype=np.intp, label="union parent")
        else:
            if int(state.get("next_offset", 0)) != 0:
                raise ExternalMemoryDBSCANError("missing union partial")
            parent = _new_memmap(parent_partial, dtype=np.intp, shape=(n_samples,))
            parent[:] = np.arange(n_samples, dtype=np.intp)
            _fsync_memmap(parent)
        start = int(state.get("next_offset", 0))
        block = _bounded_block_size(
            requested=contract.query_block_size,
            num_samples=n_samples,
            max_rss_bytes=contract.max_rss_bytes,
            phase="core_union",
        )
        blocks_since_checkpoint = 0
        for offset in range(start, len(core_indices), block):
            stop = min(len(core_indices), offset + block)
            query_indices = core_indices[offset:stop]
            reserved = _query_reservation_bytes(n_samples, stop - offset)
            _check_rss(
                contract.max_rss_bytes,
                phase="core_union.query",
                reserved_bytes=reserved,
            )
            neighborhoods = neighbors.radius_neighbors(
                vectors[query_indices], return_distance=False
            )
            for point, neighborhood in zip(query_indices.tolist(), neighborhoods):
                neighbor_values = np.asarray(neighborhood, dtype=np.intp)
                core_neighbors = neighbor_values[np.asarray(core[neighbor_values])]
                roots = np.unique(
                    _roots(parent, np.concatenate((np.asarray([point]), core_neighbors)))
                )
                if roots.size > 1:
                    parent[roots] = int(roots.min())
            del neighborhoods
            peak = max(peak, _check_rss(contract.max_rss_bytes, phase="core_union"))
            blocks_since_checkpoint += 1
            if (
                blocks_since_checkpoint >= contract.checkpoint_interval_blocks
                or stop == len(core_indices)
            ):
                _fsync_memmap(parent)
                _checkpoint(
                    state_path,
                    identity=identity,
                    phase="core_union",
                    next_offset=stop,
                    peak_rss_bytes=peak,
                    extra={"effective_query_block_size": block},
                )
                blocks_since_checkpoint = 0
        flatten_block = max(1, min(1_000_000, n_samples))
        for offset in range(0, n_samples, flatten_block):
            stop = min(n_samples, offset + flatten_block)
            parent[offset:stop] = _roots(
                parent, np.arange(offset, stop, dtype=np.intp)
            )
        _fsync_memmap(parent)
        _checkpoint(
            state_path,
            identity=identity,
            phase="labels",
            next_offset=0,
            peak_rss_bytes=peak,
            extra={"effective_query_block_size": block},
        )
        phase = "labels"
        state = _load_object(state_path)

    parent = _open_npy_memmap(parent_partial, mode="r+")
    component_roots = np.unique(np.asarray(parent[core_indices], dtype=np.intp))
    if phase == "labels_finalize":
        _reconcile_promoted_array(
            partial=labels_partial,
            final=labels_final,
            shape=(n_samples,),
            dtype=np.intp,
            expected_sha256=str(state.get("labels_sha256") or ""),
            label="labels",
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase="complete",
            next_offset=n_samples,
            peak_rss_bytes=peak,
            extra={
                "effective_query_block_size": int(
                    state.get("effective_query_block_size", contract.query_block_size)
                )
            },
        )
        phase = "complete"
        state = _load_object(state_path)
    if phase == "labels":
        if labels_partial.exists():
            labels = _open_npy_memmap(labels_partial, mode="r+")
            _validate_partial(labels, shape=(n_samples,), dtype=np.intp, label="labels")
        else:
            if int(state.get("next_offset", 0)) != 0:
                raise ExternalMemoryDBSCANError("missing labels partial")
            labels = _new_memmap(labels_partial, dtype=np.intp, shape=(n_samples,))
            labels[:] = -1
            if core_indices.size:
                labels[core_indices] = np.searchsorted(
                    component_roots, np.asarray(parent[core_indices], dtype=np.intp)
                ).astype(np.intp, copy=False)
            _fsync_memmap(labels)
        border_indices = np.flatnonzero(~np.asarray(core)).astype(np.intp, copy=False)
        start = int(state.get("next_offset", 0))
        block = _bounded_block_size(
            requested=contract.query_block_size,
            num_samples=n_samples,
            max_rss_bytes=contract.max_rss_bytes,
            phase="labels",
        )
        blocks_since_checkpoint = 0
        for offset in range(start, len(border_indices), block):
            stop = min(len(border_indices), offset + block)
            query_indices = border_indices[offset:stop]
            reserved = _query_reservation_bytes(n_samples, stop - offset)
            _check_rss(
                contract.max_rss_bytes,
                phase="labels.query",
                reserved_bytes=reserved,
            )
            neighborhoods = neighbors.radius_neighbors(
                vectors[query_indices], return_distance=False
            )
            for point, neighborhood in zip(query_indices.tolist(), neighborhoods):
                neighbor_values = np.asarray(neighborhood, dtype=np.intp)
                adjacent_core = neighbor_values[np.asarray(core[neighbor_values])]
                if adjacent_core.size:
                    adjacent_roots = np.unique(
                        np.asarray(parent[adjacent_core], dtype=np.intp)
                    )
                    candidate_labels = np.searchsorted(
                        component_roots, adjacent_roots
                    )
                    labels[point] = int(candidate_labels.min())
            del neighborhoods
            peak = max(peak, _check_rss(contract.max_rss_bytes, phase="labels"))
            blocks_since_checkpoint += 1
            if (
                blocks_since_checkpoint >= contract.checkpoint_interval_blocks
                or stop == len(border_indices)
            ):
                _fsync_memmap(labels)
                _checkpoint(
                    state_path,
                    identity=identity,
                    phase="labels",
                    next_offset=stop,
                    peak_rss_bytes=peak,
                    extra={"effective_query_block_size": block},
                )
                blocks_since_checkpoint = 0
        _fsync_memmap(labels)
        labels_sha256 = _sha256_file(labels_partial)
        _checkpoint(
            state_path,
            identity=identity,
            phase="labels_finalize",
            next_offset=n_samples,
            peak_rss_bytes=peak,
            extra={
                "effective_query_block_size": block,
                "labels_sha256": labels_sha256,
            },
        )
        del labels
        _reconcile_promoted_array(
            partial=labels_partial,
            final=labels_final,
            shape=(n_samples,),
            dtype=np.intp,
            expected_sha256=labels_sha256,
            label="labels",
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase="complete",
            next_offset=n_samples,
            peak_rss_bytes=peak,
            extra={"effective_query_block_size": block},
        )

    labels = _open_npy_memmap(labels_final, mode="r")
    cluster_labels = np.unique(labels[labels >= 0])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_complete": True,
        "scientific_identity": identity,
        "scientific_identity_sha256": _stable_hash(identity),
        "num_samples": n_samples,
        "num_features": n_features,
        "core_count": int(core_indices.size),
        "cluster_count": int(cluster_labels.size),
        "noise_count": int(np.count_nonzero(labels < 0)),
        "neighbor_counts_path": str(counts_final),
        "neighbor_counts_sha256": _sha256_file(counts_final),
        "core_mask_path": str(core_path),
        "core_mask_sha256": _sha256_file(core_path),
        "labels_path": str(labels_final),
        "labels_sha256": _sha256_file(labels_final),
        "peak_rss_bytes_observed": max(peak, _rss_bytes()),
        "max_rss_bytes": int(contract.max_rss_bytes),
        "all_neighborhoods_materialized_simultaneously": False,
        "passes": ["neighbor_counts", "core_union", "border_assignment"],
        "sklearn_dbscan_label_semantics_preserved": True,
        "completed_at": _utc_now(),
    }
    if manifest["peak_rss_bytes_observed"] > int(contract.max_rss_bytes):
        raise ExternalMemoryDBSCANError("recorded peak RSS exceeded the frozen budget")
    _atomic_json(final_manifest_path, manifest)
    return ExternalDBSCANResult(
        labels_path=labels_final,
        core_mask_path=core_path,
        neighbor_counts_path=counts_final,
        manifest_path=final_manifest_path,
        num_samples=n_samples,
        num_features=n_features,
        cluster_count=int(cluster_labels.size),
        noise_count=int(manifest["noise_count"]),
        core_count=int(core_indices.size),
        manifest_sha256=_sha256_file(final_manifest_path),
    )


__all__ = [
    "ExternalDBSCANContract",
    "ExternalDBSCANResult",
    "ExternalMemoryDBSCANError",
    "SCHEMA_VERSION",
    "fit_external_memory_dbscan",
]
