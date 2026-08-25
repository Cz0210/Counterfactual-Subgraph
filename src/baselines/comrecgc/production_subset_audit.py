"""Fail-closed exact-equivalence audits on hash-closed AIDS induced inputs.

The audit never treats a passing subset as proof about the full production
partition.  It validates a terminal theta-close view, deterministically
materializes five small induced inputs, and compares sklearn DBSCAN with the
project's general external-memory implementation.  The all-core certificate
route is also attempted, but is compared only when its exact proof applies.
"Hash-closed" describes artifact provenance, not closure under epsilon edges
to rows outside an induced input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import numpy as np

from .close_pair_view import ThetaClosePairView, validate_theta_close_pair_view
from .external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ExternalDBSCANContract,
    ExternalMemoryDBSCANError,
    fit_external_memory_dbscan,
)
from .external_memory_recourse import trace_external_cluster_order


SCHEMA_VERSION = "aids_comrecgc_production_subset_equivalence_v1"
SUBSET_NAMES = ("first", "random", "dense", "sparse", "theta_boundary")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalMemoryDBSCANError(f"invalid audit JSON authority: {path}") from exc
    if not isinstance(value, dict):
        raise ExternalMemoryDBSCANError(f"audit JSON authority is not an object: {path}")
    return value


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise ExternalMemoryDBSCANError(f"audit source is not a physical regular file: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
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
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _write_pass_last(root: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".PASS.", suffix=".tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("PASS\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / "PASS")
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ProductionSubsetAuditContract:
    eps: float = 0.02
    min_samples: int = 3
    radius: float = 0.02
    recourse_size: int = 100
    subset_size: int = 2_000
    seed: int = 0
    scan_block_size: int = 65_536
    query_block_size: int = 64
    max_rss_bytes: int = 8 * 1024**3
    expected_sklearn_version: str = ""
    expected_theta: float = 0.1
    expected_parent_count: int = 1_283
    expected_candidate_count: int = 71_642
    expected_physical_pair_count: int = 91_916_686

    def validate(self) -> None:
        if (
            not np.isfinite(self.eps)
            or self.eps != 0.02
            or not np.isfinite(self.radius)
            or self.radius != 0.02
            or self.min_samples != 3
            or self.subset_size < self.min_samples
            or self.recourse_size != 100
            or self.scan_block_size <= 0
            or self.query_block_size <= 0
            or self.max_rss_bytes <= 0
            or not self.expected_sklearn_version
            or self.expected_theta != 0.1
            or self.expected_parent_count <= 0
            or self.expected_candidate_count <= 0
            or self.expected_physical_pair_count
            != self.expected_parent_count * self.expected_candidate_count
        ):
            raise ExternalMemoryDBSCANError("invalid production-subset audit contract")


def _logical_to_physical(view: ThetaClosePairView, logical: np.ndarray) -> np.ndarray:
    if view.all_pairs_close:
        return np.asarray(logical, dtype=np.int64)
    if view.physical_row_indices_path is None:
        raise ExternalMemoryDBSCANError("partial close view has no physical-row mapping")
    rows = np.load(view.physical_row_indices_path, mmap_mode="r", allow_pickle=False)
    return np.asarray(rows[logical], dtype=np.int64)


def _take_smallest(
    scores: np.ndarray, indices: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact lexicographic top-k by (score, index), including ties."""

    scores = np.asarray(scores, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if len(scores) <= count:
        order = np.lexsort((indices, scores))
        return scores[order], indices[order]
    cutoff = float(np.partition(scores, count - 1)[count - 1])
    lower = np.flatnonzero(scores < cutoff)
    equal = np.flatnonzero(scores == cutoff)
    equal = equal[np.argsort(indices[equal], kind="stable")]
    keep = np.concatenate((lower, equal[: count - len(lower)]))
    order = np.lexsort((indices[keep], scores[keep]))
    keep = keep[order]
    return scores[keep], indices[keep]


def _sample_unique_indices(
    *, population: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform rejection sampler with O(count) memory for huge populations."""

    selected: set[int] = set()
    while len(selected) < count:
        selected.add(int(rng.integers(0, population)))
    return np.asarray(sorted(selected), dtype=np.int64)


def _stream_extreme_vector_rows(
    vectors: Any,
    *,
    pivot: np.ndarray,
    count: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    smallest_scores = np.empty(0, dtype=np.float64)
    smallest_indices = np.empty(0, dtype=np.int64)
    largest_scores = np.empty(0, dtype=np.float64)
    largest_indices = np.empty(0, dtype=np.int64)
    for start in range(0, len(vectors), block_size):
        stop = min(len(vectors), start + block_size)
        block = np.asarray(vectors[start:stop], dtype=np.float64)
        scores = np.einsum("ij,ij->i", block - pivot, block - pivot)
        indices = np.arange(start, stop, dtype=np.int64)
        smallest_scores, smallest_indices = _take_smallest(
            np.concatenate((smallest_scores, scores)),
            np.concatenate((smallest_indices, indices)),
            count,
        )
        largest_scores, largest_indices = _take_smallest(
            np.concatenate((largest_scores, -scores)),
            np.concatenate((largest_indices, indices)),
            count,
        )
    return np.sort(smallest_indices), np.sort(largest_indices)


def _stream_theta_boundary_rows(
    *,
    view: ThetaClosePairView,
    distances: np.ndarray,
    theta: float,
    count: int,
    block_size: int,
    physical_row_map: np.ndarray | None,
) -> np.ndarray:
    retained_scores = np.empty(0, dtype=np.float64)
    retained_indices = np.empty(0, dtype=np.int64)
    for start in range(0, view.logical_close_rows, block_size):
        stop = min(view.logical_close_rows, start + block_size)
        logical = np.arange(start, stop, dtype=np.int64)
        physical = (
            logical
            if physical_row_map is None
            else np.asarray(physical_row_map[logical], dtype=np.int64)
        )
        scores = np.abs(np.asarray(distances[physical], dtype=np.float64) - theta)
        retained_scores, retained_indices = _take_smallest(
            np.concatenate((retained_scores, scores)),
            np.concatenate((retained_indices, logical)),
            count,
        )
    return np.sort(retained_indices)


def _canonicalize(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    canonical = np.full(labels.shape, -1, dtype=np.int64)
    components = []
    for label in np.unique(labels[labels >= 0]):
        positions = np.flatnonzero(labels == label)
        components.append((int(positions[0]), positions))
    for canonical_label, (_first, positions) in enumerate(sorted(components)):
        canonical[positions] = canonical_label
    return canonical


def _partition_summary(
    *,
    labels: np.ndarray,
    core_mask: np.ndarray,
    vectors: np.ndarray,
    pairs: np.ndarray,
    radius: float,
    theta: float,
    recourse_size: int,
) -> dict[str, Any]:
    canonical = _canonicalize(labels)
    clusters: list[dict[str, Any]] = []
    covering: dict[int, set[int]] = {}
    for label in sorted(int(value) for value in np.unique(canonical[canonical >= 0])):
        positions = np.flatnonzero(canonical == label)
        points = np.asarray(vectors[positions])
        centroid = np.mean(points, axis=0)
        stable_centroid = np.mean(points.astype(np.float64), axis=0, dtype=np.float64)
        distances = np.linalg.norm(points - centroid, axis=1)
        radius_in_distance_dtype = np.asarray(float(radius), dtype=distances.dtype)
        retained = positions[np.flatnonzero(distances < radius_in_distance_dtype)]
        parents = sorted({int(value) for value in pairs[retained, 0].tolist()})
        candidates = sorted({int(value) for value in pairs[retained, 1].tolist()})
        norm = float(np.linalg.norm(centroid))
        if norm < float(theta) and parents:
            covering[label] = set(parents)
        clusters.append(
            {
                "cluster_id": label,
                "member_count": int(len(positions)),
                "core_member_count": int(np.count_nonzero(core_mask[positions])),
                "centroid_sha256": hashlib.sha256(centroid.tobytes()).hexdigest(),
                "stable_float64_centroid_sha256": hashlib.sha256(
                    stable_centroid.tobytes()
                ).hexdigest(),
                "centroid_max_absolute_difference": float(
                    np.max(np.abs(centroid.astype(np.float64) - stable_centroid))
                ),
                "centroid_norm": norm,
                "centroid_norm_lt_theta": bool(norm < float(theta)),
                "count_exactly_at_theta": int(norm == float(theta)),
                "within_centroid_radius_count": int(len(retained)),
                "outside_centroid_radius_count": int(len(positions) - len(retained)),
                "count_exactly_at_delta": int(
                    np.count_nonzero(distances == radius_in_distance_dtype)
                ),
                "covered_parent_ids": parents,
                "counterfactual_ids": candidates,
            }
        )
    remaining = {label: set(parents) for label, parents in covering.items()}
    selected: list[dict[str, Any]] = []
    clusters_by_id = {int(row["cluster_id"]): row for row in clusters}
    covered: set[int] = set()
    for rank in range(1, min(int(recourse_size), len(remaining)) + 1):
        label = min(
            remaining,
            key=lambda value: (-len(remaining[value] - covered), int(value)),
        )
        covered.update(remaining.pop(label))
        cluster = clusters_by_id[int(label)]
        selected.append(
            {
                "selected_rank": rank,
                "cluster_id": int(label),
                "cumulative_covered_count": len(covered),
                "centroid_norm": cluster["centroid_norm"],
                "covered_parent_ids": cluster["covered_parent_ids"],
                "counterfactual_ids": cluster["counterfactual_ids"],
            }
        )
    summary = {
        "canonical_labels_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "core_mask_sha256": hashlib.sha256(
            np.asarray(core_mask, dtype=np.bool_).tobytes()
        ).hexdigest(),
        "noise_mask_sha256": hashlib.sha256(
            np.asarray(canonical < 0, dtype=np.bool_).tobytes()
        ).hexdigest(),
        "cluster_count": len(clusters),
        "noise_count": int(np.count_nonzero(canonical < 0)),
        "clusters": clusters,
        "selected": selected,
        "selected_common_recourse_count": len(selected),
        "radius_filter_operator": "<",
        "centroid_norm_filter_operator": "<",
        "greedy_tie_break": "ascending_canonical_cluster_id",
    }
    summary["result_sha256"] = _stable_hash(summary)
    return summary


def _stable_greedy(
    counterfactual_covering: dict[int, set[int]],
    graphs_covered_by: dict[int, set[int]],
    k: int,
) -> dict[int, tuple[int, int]]:
    """Official destructive marginal-coverage loop with an explicit stable tie."""

    del graphs_covered_by  # The equivalent set subtraction below is explicit.
    remaining = {int(label): set(parents) for label, parents in counterfactual_covering.items()}
    covered: set[int] = set()
    selected: dict[int, tuple[int, int]] = {}
    for rank in range(1, min(int(k), len(remaining)) + 1):
        label = min(
            remaining,
            key=lambda value: (-len(remaining[value] - covered), int(value)),
        )
        covered.update(remaining.pop(label))
        selected[rank] = (int(label), len(covered))
    return selected


def _production_downstream_summary(
    *,
    labels: np.ndarray,
    vectors: np.ndarray,
    pairs: np.ndarray,
    radius: float,
    theta: float,
    recourse_size: int,
    max_rss_bytes: int,
) -> dict[str, Any]:
    """Normalize the existing production downstream implementation for comparison."""

    ordered, audit = trace_external_cluster_order(
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=pairs,
        radius=radius,
        theta=theta,
        recourse_size=recourse_size,
        official_greedy=_stable_greedy,
        max_rss_bytes=max_rss_bytes,
    )
    selected = [
        {
            "selected_rank": int(row["selected_rank"]),
            "cluster_id": int(row["cluster_id"]),
            "cumulative_covered_count": int(row["cumulative_covered_count"]),
            "centroid_norm": float(row["centroid_norm"]),
            "covered_parent_ids": list(row["covered_parent_indices_native"]),
            "counterfactual_ids": list(row["member_counterfactual_indices"]),
        }
        for row in ordered
    ]
    result = {
        "selected": selected,
        "selected_common_recourse_count": len(selected),
        "implementation_schema_version": audit["schema_version"],
        "legacy_numpy_reduction_order_preserved": audit[
            "legacy_numpy_reduction_order_preserved"
        ],
    }
    result["result_sha256"] = _stable_hash(result)
    return result


def _verify_pair_authority(
    *,
    view: ThetaClosePairView,
    physical_pairs_path: Path,
    expected_physical_pairs_sha256: str,
) -> dict[str, Any]:
    authority_path = view.pair_semantics_contract_path
    if authority_path is None:
        raise ExternalMemoryDBSCANError("PAIR_SEMANTICS_AUTHORITY_NOT_BOUND")
    authority = _load_object(authority_path)
    source_hashes = authority.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ExternalMemoryDBSCANError("pair-semantics source hashes are absent")
    if source_hashes.get("pair_indices_direct_sha256") != expected_physical_pairs_sha256:
        raise ExternalMemoryDBSCANError("physical pair SHA is not pair-authority bound")
    source_manifest_path = Path(
        str(authority.get("source_pair_store_manifest") or "")
    ).resolve(strict=True)
    _stat_identity(source_manifest_path)
    source_manifest_sha = sha256_file(source_manifest_path)
    if source_manifest_sha != authority.get("source_pair_store_manifest_sha256"):
        raise ExternalMemoryDBSCANError("pair-store manifest authority drift")
    source_manifest = _load_object(source_manifest_path)
    if (
        Path(str(source_manifest.get("pairs_path") or "")).resolve(strict=True)
        != physical_pairs_path
        or source_manifest.get("pairs_sha256") != expected_physical_pairs_sha256
    ):
        raise ExternalMemoryDBSCANError("physical pair path/hash is not store-bound")
    return {
        "pair_semantics_contract_path": str(authority_path),
        "pair_semantics_contract_sha256": sha256_file(authority_path),
        "pair_store_manifest_path": str(source_manifest_path),
        "pair_store_manifest_sha256": source_manifest_sha,
    }


def run_production_subset_equivalence_audit(
    *,
    close_pair_contract_path: str | Path,
    expected_close_pair_contract_sha256: str,
    physical_pairs_path: str | Path,
    expected_physical_pairs_sha256: str,
    output_dir: str | Path,
    contract: ProductionSubsetAuditContract,
) -> dict[str, Any]:
    """Run the five production-derived audits and publish PASS last."""

    contract.validate()
    close_path = Path(close_pair_contract_path).expanduser().resolve(strict=True)
    pair_path = Path(physical_pairs_path).expanduser().resolve(strict=True)
    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"production-subset audit output already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    close_sha = sha256_file(close_path)
    pair_sha = sha256_file(pair_path)
    if close_sha != expected_close_pair_contract_sha256:
        raise ExternalMemoryDBSCANError("close-pair contract SHA256 mismatch")
    if pair_sha != expected_physical_pairs_sha256:
        raise ExternalMemoryDBSCANError("physical pair SHA256 mismatch")
    source_stats_before = {
        "close_pair_contract": _stat_identity(close_path),
        "physical_pairs": _stat_identity(pair_path),
    }
    view = validate_theta_close_pair_view(
        close_path,
        require_dbscan_eligible=True,
        require_pair_semantics_authority=True,
    )
    if (
        view.theta != contract.expected_theta
        or view.parent_count != contract.expected_parent_count
        or view.candidate_count != contract.expected_candidate_count
        or view.physical_store_rows != contract.expected_physical_pair_count
    ):
        raise ExternalMemoryDBSCANError("AIDS production close-view identity mismatch")
    authority = _verify_pair_authority(
        view=view,
        physical_pairs_path=pair_path,
        expected_physical_pairs_sha256=pair_sha,
    )
    close_manifest = _load_object(close_path)
    identity = close_manifest["scientific_identity"]
    physical_vectors_path = Path(identity["physical_vectors_path"]).resolve(strict=True)
    normalized_distances_path = Path(
        identity["normalized_distances_path"]
    ).resolve(strict=True)
    bitmap_path = view.bitmap_path.resolve(strict=True)
    bitmap_sha = sha256_file(bitmap_path)
    if bitmap_sha != close_manifest.get("close_bitmap_hash"):
        raise ExternalMemoryDBSCANError("close bitmap lost its manifest binding")
    authority_path = Path(authority["pair_semantics_contract_path"])
    pair_store_manifest_path = Path(authority["pair_store_manifest_path"])
    vectors_sha = str(identity["physical_vectors_sha256"])
    distances_sha = str(identity["normalized_distances_sha256"])
    source_stats_before.update(
        {
            "physical_vectors": _stat_identity(physical_vectors_path),
            "normalized_distances": _stat_identity(normalized_distances_path),
            "close_bitmap": _stat_identity(bitmap_path),
            "pair_semantics_contract": _stat_identity(authority_path),
            "pair_store_manifest": _stat_identity(pair_store_manifest_path),
        }
    )
    vectors = view.open_vectors()
    logical_pairs = view.open_pairs()
    physical_pairs = np.load(pair_path, mmap_mode="r", allow_pickle=False)
    distances = np.load(normalized_distances_path, mmap_mode="r", allow_pickle=False)
    if physical_pairs.shape != (view.physical_store_rows, 2):
        raise ExternalMemoryDBSCANError("physical pair array shape mismatch")
    count = min(int(contract.subset_size), int(view.logical_close_rows))
    if count < contract.min_samples:
        raise ExternalMemoryDBSCANError("logical close view is too small for subset audit")
    rng = np.random.default_rng(int(contract.seed))
    pivot_index = int(rng.integers(0, view.logical_close_rows))
    pivot = np.asarray(vectors[pivot_index], dtype=np.float64)
    dense_indices, sparse_indices = _stream_extreme_vector_rows(
        vectors,
        pivot=pivot,
        count=count,
        block_size=contract.scan_block_size,
    )
    physical_row_map = (
        None
        if view.all_pairs_close
        else np.load(
            view.physical_row_indices_path, mmap_mode="r", allow_pickle=False
        )
    )
    selections = {
        "first": np.arange(count, dtype=np.int64),
        "random": _sample_unique_indices(
            population=view.logical_close_rows,
            count=count,
            rng=rng,
        ),
        "dense": dense_indices,
        "sparse": sparse_indices,
        "theta_boundary": _stream_theta_boundary_rows(
            view=view,
            distances=distances,
            theta=view.theta,
            count=count,
            block_size=contract.scan_block_size,
            physical_row_map=physical_row_map,
        ),
    }
    from sklearn import __version__ as sklearn_version
    from sklearn.cluster import DBSCAN

    if sklearn_version != contract.expected_sklearn_version:
        raise ExternalMemoryDBSCANError(
            f"SKLEARN_VERSION_MISMATCH:actual={sklearn_version}:"
            f"expected={contract.expected_sklearn_version}"
        )
    results: dict[str, Any] = {}
    for name in SUBSET_NAMES:
        indices = selections[name]
        if len(indices) != count or len(np.unique(indices)) != count:
            raise ExternalMemoryDBSCANError(f"{name} selection is not an exact set")
        subset_root = root / name
        subset_root.mkdir()
        subset_vectors = np.asarray(vectors[indices])
        subset_pairs = np.asarray(logical_pairs[indices], dtype=np.int64)
        physical_rows = _logical_to_physical(view, indices)
        if not np.array_equal(subset_pairs, physical_pairs[physical_rows]):
            raise ExternalMemoryDBSCANError(f"{name} pair orientation/mapping mismatch")
        if (
            np.any(subset_pairs[:, 0] < 0)
            or np.any(subset_pairs[:, 0] >= view.parent_count)
            or np.any(subset_pairs[:, 1] < 0)
            or np.any(subset_pairs[:, 1] >= view.candidate_count)
        ):
            raise ExternalMemoryDBSCANError(f"{name} pair axis values are invalid")
        indices_sha = _atomic_npy(subset_root / "logical_indices.npy", indices)
        physical_rows_sha = _atomic_npy(
            subset_root / "physical_rows.npy", physical_rows
        )
        vectors_subset_sha = _atomic_npy(
            subset_root / "recourse_vectors.npy", subset_vectors
        )
        pairs_subset_sha = _atomic_npy(subset_root / "pair_indices.npy", subset_pairs)
        expected = DBSCAN(
            eps=contract.eps,
            min_samples=contract.min_samples,
            metric="euclidean",
        ).fit(subset_vectors)
        sklearn_labels = _canonicalize(expected.labels_)
        sklearn_core = np.zeros(count, dtype=np.bool_)
        sklearn_core[expected.core_sample_indices_] = True
        general = fit_external_memory_dbscan(
            vectors_path=subset_root / "recourse_vectors.npy",
            work_dir=subset_root / "external_general",
            contract=ExternalDBSCANContract(
                eps=contract.eps,
                min_samples=contract.min_samples,
                query_block_size=min(contract.query_block_size, count),
                checkpoint_interval_blocks=1,
                max_rss_bytes=contract.max_rss_bytes,
                expected_sklearn_version=contract.expected_sklearn_version,
            ),
            expected_vectors_sha256=vectors_subset_sha,
        )
        external_labels = _canonicalize(np.load(general.labels_path, allow_pickle=False))
        external_core = np.load(general.core_mask_path, allow_pickle=False)
        if not np.array_equal(sklearn_labels, external_labels):
            raise ExternalMemoryDBSCANError(f"{name} exact partition mismatch")
        if not np.array_equal(sklearn_core, external_core):
            raise ExternalMemoryDBSCANError(f"{name} exact core-mask mismatch")
        sklearn_summary = _partition_summary(
            labels=sklearn_labels,
            core_mask=sklearn_core,
            vectors=subset_vectors,
            pairs=subset_pairs,
            radius=contract.radius,
            theta=view.theta,
            recourse_size=contract.recourse_size,
        )
        external_summary = _partition_summary(
            labels=external_labels,
            core_mask=external_core,
            vectors=subset_vectors,
            pairs=subset_pairs,
            radius=contract.radius,
            theta=view.theta,
            recourse_size=contract.recourse_size,
        )
        if sklearn_summary != external_summary:
            raise ExternalMemoryDBSCANError(f"{name} downstream summary mismatch")
        production_downstream = _production_downstream_summary(
            labels=external_labels,
            vectors=subset_vectors,
            pairs=subset_pairs,
            radius=contract.radius,
            theta=view.theta,
            recourse_size=contract.recourse_size,
            max_rss_bytes=contract.max_rss_bytes,
        )
        if production_downstream["selected"] != sklearn_summary["selected"]:
            raise ExternalMemoryDBSCANError(
                f"{name} production/reference coverage-greedy mismatch"
            )
        certificate: dict[str, Any]
        certificate_root = subset_root / "certificate"
        try:
            certified = fit_external_memory_dbscan(
                vectors_path=subset_root / "recourse_vectors.npy",
                work_dir=certificate_root,
                contract=ExternalDBSCANContract(
                    eps=contract.eps,
                    min_samples=contract.min_samples,
                    query_block_size=min(contract.query_block_size, count),
                    checkpoint_interval_blocks=1,
                    max_rss_bytes=contract.max_rss_bytes,
                    expected_sklearn_version=contract.expected_sklearn_version,
                    shortcut_mode=ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
                    shortcut_seed_count=min(3, count),
                    shortcut_failure_cap=min(4096, count),
                    shortcut_query_block_size=min(contract.query_block_size, count),
                    exact_fallback_max_samples=0,
                ),
                expected_vectors_sha256=vectors_subset_sha,
            )
        except ExternalMemoryDBSCANError as exc:
            failure_reason = str(exc)
            if not failure_reason.startswith(
                (
                    "EXACT_DBSCAN_COMPLEXITY_BLOCKED:",
                    "EXACT_DBSCAN_GENERAL_EXTERNAL_REQUIRED:",
                )
            ):
                raise
            failure_path = certificate_root / "shortcut_failure.json"
            certificate = {
                "applicable": False,
                "reason": failure_reason,
                "failure_path": str(failure_path),
                "failure_sha256": sha256_file(failure_path),
            }
        else:
            certified_labels = _canonicalize(
                np.load(certified.labels_path, allow_pickle=False)
            )
            certified_core = np.load(certified.core_mask_path, allow_pickle=False)
            if (
                not np.array_equal(sklearn_labels, certified_labels)
                or not np.array_equal(sklearn_core, certified_core)
            ):
                raise ExternalMemoryDBSCANError(f"{name} certificate partition mismatch")
            certified_summary = _partition_summary(
                labels=certified_labels,
                core_mask=certified_core,
                vectors=subset_vectors,
                pairs=subset_pairs,
                radius=contract.radius,
                theta=view.theta,
                recourse_size=contract.recourse_size,
            )
            if certified_summary != sklearn_summary:
                raise ExternalMemoryDBSCANError(f"{name} certificate summary mismatch")
            certified_downstream = _production_downstream_summary(
                labels=certified_labels,
                vectors=subset_vectors,
                pairs=subset_pairs,
                radius=contract.radius,
                theta=view.theta,
                recourse_size=contract.recourse_size,
                max_rss_bytes=contract.max_rss_bytes,
            )
            if certified_downstream["selected"] != sklearn_summary["selected"]:
                raise ExternalMemoryDBSCANError(
                    f"{name} certificate downstream production mismatch"
                )
            certified_manifest = _load_object(certified.manifest_path)
            certificate = {
                "applicable": True,
                "manifest_path": str(certified.manifest_path),
                "manifest_sha256": certified.manifest_sha256,
                "shortcut_proof_path": str(certified.shortcut_proof_path),
                "shortcut_proof_sha256": sha256_file(certified.shortcut_proof_path),
                "downstream_result_sha256": certified_summary["result_sha256"],
                "production_downstream_result_sha256": certified_downstream[
                    "result_sha256"
                ],
                "all_core_certificate_path": certified_manifest[
                    "all_core_certificate_path"
                ],
                "all_core_certificate_sha256": certified_manifest[
                    "all_core_certificate_sha256"
                ],
                "connectivity_certificate_path": certified_manifest[
                    "connectivity_certificate_path"
                ],
                "connectivity_certificate_sha256": certified_manifest[
                    "connectivity_certificate_sha256"
                ],
                "boundary_certificate_path": certified_manifest[
                    "boundary_certificate_path"
                ],
                "boundary_certificate_sha256": certified_manifest[
                    "boundary_certificate_sha256"
                ],
                "cluster_partition_path": certified_manifest[
                    "cluster_partition_path"
                ],
                "cluster_partition_sha256": certified_manifest[
                    "cluster_partition_sha256"
                ],
            }
        subset_manifest = {
            "schema_version": SCHEMA_VERSION,
            "subset_name": name,
            "selection_semantics": {
                "first": "first rows in logical close-view order",
                "random": "seeded without-replacement rows; restored logical order",
                "dense": "smallest squared Euclidean distance to seeded pivot",
                "sparse": "largest squared Euclidean distance to seeded pivot",
                "theta_boundary": "smallest abs(normalized_distance-theta)",
            }[name],
            "seed": contract.seed,
            "pivot_logical_index": pivot_index,
            "sample_count": count,
            "logical_indices_path": str(subset_root / "logical_indices.npy"),
            "logical_indices_sha256": indices_sha,
            "physical_rows_path": str(subset_root / "physical_rows.npy"),
            "physical_rows_sha256": physical_rows_sha,
            "vectors_path": str(subset_root / "recourse_vectors.npy"),
            "vectors_sha256": vectors_subset_sha,
            "pairs_path": str(subset_root / "pair_indices.npy"),
            "pairs_sha256": pairs_subset_sha,
            "partition_canonicalization": "components ordered by minimum subset row; noise=-1",
            "sklearn_result": sklearn_summary,
            "external_result": external_summary,
            "production_downstream_result": production_downstream,
            "external_manifest_path": str(general.manifest_path),
            "external_manifest_sha256": general.manifest_sha256,
            "certificate_route": certificate,
            "status": "PASS",
            "subset_scope_only": True,
        }
        _atomic_json(subset_root / "audit.json", subset_manifest)
        results[name] = {
            "audit_path": str(subset_root / "audit.json"),
            "audit_sha256": sha256_file(subset_root / "audit.json"),
            "result_sha256": sklearn_summary["result_sha256"],
            "certificate_applicable": certificate["applicable"],
        }
    source_stats_after = {
        "close_pair_contract": _stat_identity(close_path),
        "physical_pairs": _stat_identity(pair_path),
        "physical_vectors": _stat_identity(physical_vectors_path),
        "normalized_distances": _stat_identity(normalized_distances_path),
        "close_bitmap": _stat_identity(bitmap_path),
        "pair_semantics_contract": _stat_identity(authority_path),
        "pair_store_manifest": _stat_identity(pair_store_manifest_path),
    }
    if source_stats_before != source_stats_after:
        raise ExternalMemoryDBSCANError("production authority changed during subset audit")
    if (
        sha256_file(close_path) != close_sha
        or sha256_file(pair_path) != pair_sha
        or sha256_file(bitmap_path) != bitmap_sha
        or sha256_file(authority_path)
        != authority["pair_semantics_contract_sha256"]
        or sha256_file(pair_store_manifest_path)
        != authority["pair_store_manifest_sha256"]
    ):
        raise ExternalMemoryDBSCANError("small production authority changed during audit")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "run_complete": True,
        "contract": asdict(contract),
        "close_pair_contract_path": str(close_path),
        "close_pair_contract_sha256": close_sha,
        "physical_vectors_path": str(physical_vectors_path),
        "physical_vectors_sha256": vectors_sha,
        "physical_pairs_path": str(pair_path),
        "physical_pairs_sha256": pair_sha,
        "normalized_distances_path": str(normalized_distances_path),
        "normalized_distances_sha256": distances_sha,
        "close_bitmap_path": str(bitmap_path),
        "close_bitmap_sha256": bitmap_sha,
        "logical_close_pair_count": view.logical_close_rows,
        "physical_pair_count": view.physical_store_rows,
        "theta": view.theta,
        "pair_axis": "col0=parent;col1=candidate",
        "source_authority": authority,
        "source_stats_before": source_stats_before,
        "source_stats_after": source_stats_after,
        "selection_seed": contract.seed,
        "selection_pivot_logical_index": pivot_index,
        "partition_canonicalization": "components ordered by minimum subset row; noise=-1",
        "subsets": results,
        "all_subsets_pass": True,
        "full_production_dbscan_equivalence_claimed": False,
        "scope_warning": "subset PASS is not full-production DBSCAN PASS",
        "approximation_used": False,
        "completed_at": _utc_now(),
    }
    manifest["result_sha256"] = _stable_hash(manifest)
    _atomic_json(root / "production_subset_equivalence.json", manifest)
    _write_pass_last(root)
    return manifest


__all__ = [
    "ProductionSubsetAuditContract",
    "SCHEMA_VERSION",
    "SUBSET_NAMES",
    "run_production_subset_equivalence_audit",
    "sha256_file",
]
