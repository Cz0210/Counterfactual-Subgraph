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

from .external_memory_dbscan import ExternalMemoryDBSCANError, _check_rss, _rss_bytes


PAIR_STORE_SCHEMA = "comrecgc_external_pair_store_v1"
SUMMARY_SCHEMA = "comrecgc_external_cluster_summary_v1"


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
        if total <= 0:
            raise ExternalMemoryDBSCANError("no theta-eligible recourse pairs")
        pairs_partial = self.root / "pair_indices.partial.npy"
        vectors_partial = self.root / "recourse_vectors.partial.npy"
        pairs_final = self.root / "pair_indices.npy"
        vectors_final = self.root / "recourse_vectors.npy"
        if any(path.exists() for path in (pairs_partial, vectors_partial, pairs_final, vectors_final)):
            raise ExternalMemoryDBSCANError("pair-store consolidation target already exists")
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
                raise ExternalMemoryDBSCANError("pair chunk changed before consolidation")
            pair_chunk = np.load(pair_path, mmap_mode="r", allow_pickle=False)
            vector_chunk = np.load(vector_path, mmap_mode="r", allow_pickle=False)
            stop = cursor + int(row["row_count"])
            pairs_out[cursor:stop] = pair_chunk
            vectors_out[cursor:stop] = vector_chunk
            cursor = stop
            _check_rss(self.max_rss_bytes, phase="pair_store.consolidate")
        pairs_out.flush()
        vectors_out.flush()
        for path in (pairs_partial, vectors_partial):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(pairs_partial, pairs_final)
        os.replace(vectors_partial, vectors_final)
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
            "pairs_sha256": _sha256_file(pairs_final),
            "vectors_path": str(vectors_final),
            "vectors_sha256": _sha256_file(vectors_final),
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
                "cluster_label": cluster_label,
                "cluster_center_norm": centroid_norms[cluster_label],
                "cluster_radius": float(radius),
                "cluster_size": cluster_sizes[cluster_label],
                "representative_source_index": int(source_index),
                "representative_counterfactual_index": int(counterfactual_index),
                "representative_distance_to_center": medoid_distance,
                "covered_parent_indices_native": sorted(filtered[cluster_label]),
                "native_cumulative_covered_count": len(covered),
                "native_cumulative_cost": cumulative_cost,
                "member_counterfactual_indices": member_counterfactuals,
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
    "ExternalPairStore",
    "PAIR_STORE_SCHEMA",
    "PairStoreResult",
    "SUMMARY_SCHEMA",
    "invoke_official_coverage_summary_external",
    "trace_external_cluster_order",
]
