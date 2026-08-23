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


SCHEMA_VERSION = "comrecgc_external_memory_dbscan_v2"
SHORTCUT_DISABLED = "disabled"
ALL_CORE_ONE_COMPONENT_SHORTCUT = "all_core_one_component_anchor_v1"
SHORTCUT_PROOF_SCHEMA_VERSION = "comrecgc_dbscan_anchor_proof_v1"


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
    shortcut_mode: str = SHORTCUT_DISABLED
    shortcut_anchor_count: int = 64
    shortcut_query_block_size: int = 65_536
    exact_fallback_max_samples: int = 100_000

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
        if self.shortcut_mode not in {
            SHORTCUT_DISABLED,
            ALL_CORE_ONE_COMPONENT_SHORTCUT,
        }:
            raise ExternalMemoryDBSCANError(
                f"unsupported exact DBSCAN shortcut: {self.shortcut_mode}"
            )
        if int(self.shortcut_anchor_count) <= 0:
            raise ExternalMemoryDBSCANError("shortcut_anchor_count must be positive")
        if int(self.shortcut_anchor_count) > 65_535:
            raise ExternalMemoryDBSCANError(
                "shortcut_anchor_count exceeds the auditable finite-anchor limit"
            )
        if int(self.shortcut_query_block_size) <= 0:
            raise ExternalMemoryDBSCANError(
                "shortcut_query_block_size must be positive"
            )
        if int(self.exact_fallback_max_samples) < 0:
            raise ExternalMemoryDBSCANError(
                "exact_fallback_max_samples must be nonnegative"
            )


@dataclass(frozen=True)
class ExternalDBSCANResult:
    labels_path: Path
    core_mask_path: Path
    neighbor_counts_path: Path | None
    shortcut_proof_path: Path | None
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


def _fit_anchor_neighbors(vectors: np.ndarray, *, eps: float) -> tuple[Any, str]:
    """Fit the frozen sklearn brute Euclidean radius implementation on anchors.

    The production AIDS vectors have 64 features, for which sklearn's
    ``algorithm=auto`` DBSCAN route resolves to ``brute``.  The shortcut is
    deliberately unavailable for any other resolved full-data method.  Using
    an explicit brute anchor model therefore preserves the same floating-point
    distance kernel and inclusive-radius comparison while making work linear
    in the number of input rows.
    """

    try:
        import sklearn
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover - dependency gate
        raise ExternalMemoryDBSCANError(
            "external-memory DBSCAN requires scikit-learn"
        ) from exc
    model = NearestNeighbors(
        radius=float(eps),
        algorithm="brute",
        leaf_size=30,
        metric="euclidean",
        metric_params=None,
        p=None,
        n_jobs=None,
    )
    model.fit(vectors)
    method = str(getattr(model, "_fit_method", "UNKNOWN"))
    if method != "brute":
        raise ExternalMemoryDBSCANError(
            f"anchor radius model did not resolve to brute: {method}"
        )
    return model, str(sklearn.__version__)


def _deterministic_anchor_indices(num_samples: int, requested: int) -> np.ndarray:
    """Return unique, input-order-independent-of-RNG sample-index anchors."""

    count = min(int(num_samples), int(requested))
    if count <= 0:
        raise ExternalMemoryDBSCANError("the anchor witness requires samples")
    if count == 1:
        return np.asarray([0], dtype=np.intp)
    # Integer arithmetic includes both endpoints and cannot duplicate indices
    # when ``count <= num_samples``.
    indices = (
        np.arange(count, dtype=np.int64) * (int(num_samples) - 1) // (count - 1)
    ).astype(np.intp, copy=False)
    if len(np.unique(indices)) != count:
        raise ExternalMemoryDBSCANError("deterministic anchors are not distinct")
    return indices


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    """Write one ndarray through a same-directory fsync + atomic rename."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _constant_npy(
    path: Path,
    *,
    shape: Sequence[int],
    dtype: Any,
    fill_value: int | bool,
    write_block_size: int = 1_000_000,
) -> None:
    """Atomically materialize a deterministic constant array with bounded RSS."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial.npy")
    temporary.unlink(missing_ok=True)
    try:
        result = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=tuple(int(part) for part in shape),
        )
        total = int(shape[0])
        for offset in range(0, total, max(1, int(write_block_size))):
            result[offset : min(total, offset + write_block_size)] = fill_value
        _fsync_memmap(result)
        del result
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _shortcut_query_reservation_bytes(rows: int, anchor_count: int) -> int:
    # sklearn returns one Python-object entry and, in the dense witness case,
    # one intp id per query/anchor pair.  Reserve a second copy plus fixed
    # pairwise-distance workspace rather than using the full-N quadratic bound.
    ids = int(rows) * int(anchor_count) * np.dtype(np.intp).itemsize
    distances = int(rows) * int(anchor_count) * np.dtype(np.float64).itemsize
    return 2 * ids + distances + int(rows) * 4096 + 64 * 1024**2


def _bounded_shortcut_block_size(
    *, requested: int, anchor_count: int, max_rss_bytes: int, phase: str
) -> int:
    current = _check_rss(max_rss_bytes, phase=phase)
    available = int(max_rss_bytes) - current - 64 * 1024**2
    per_row = max(
        1,
        2 * int(anchor_count) * np.dtype(np.intp).itemsize
        + int(anchor_count) * np.dtype(np.float64).itemsize
        + 4096,
    )
    safe = available // per_row
    if safe < 1:
        raise ExternalMemoryDBSCANError(
            f"RSS budget cannot hold one anchor-witness row during {phase}"
        )
    return max(1, min(int(requested), int(safe)))


def _anchor_graph(
    *, model: Any, anchor_vectors: np.ndarray
) -> tuple[np.ndarray, bool, int, list[list[int]]]:
    """Return canonical undirected anchor edges and exact connectivity."""

    neighborhoods = model.radius_neighbors(
        anchor_vectors, return_distance=False
    )
    rows: list[list[int]] = []
    adjacency: list[set[int]] = []
    anchor_count = int(anchor_vectors.shape[0])
    for local_index, raw in enumerate(neighborhoods):
        values = np.asarray(raw, dtype=np.intp)
        if len(values) != len(np.unique(values)):
            raise ExternalMemoryDBSCANError(
                "anchor radius query returned duplicate sample indices"
            )
        if local_index not in values:
            raise ExternalMemoryDBSCANError(
                "anchor radius query omitted its explicit self sample"
            )
        members = {int(value) for value in values.tolist()}
        if any(value < 0 or value >= anchor_count for value in members):
            raise ExternalMemoryDBSCANError("anchor radius query escaped anchor set")
        adjacency.append(members)
        rows.append(sorted(members))
    for source, members in enumerate(adjacency):
        for target in members:
            if source not in adjacency[target]:
                raise ExternalMemoryDBSCANError(
                    "anchor epsilon graph is numerically asymmetric"
                )
    edges = np.asarray(
        sorted(
            (source, target)
            for source, members in enumerate(adjacency)
            for target in members
            if source < target
        ),
        dtype=np.intp,
    ).reshape(-1, 2)
    visited = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for target in adjacency[source]:
            if target not in visited:
                visited.add(target)
                frontier.append(target)
    return edges, len(visited) == anchor_count, len(visited), rows


def _shortcut_failure(
    *,
    root: Path,
    identity: Mapping[str, Any],
    reason: str,
    num_samples: int,
    fallback_limit: int,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "comrecgc_dbscan_anchor_proof_failure_v1",
        "status": "INCONCLUSIVE",
        "scientific_identity_sha256": _stable_hash(identity),
        "reason": str(reason),
        "details": dict(details),
        "num_samples": int(num_samples),
        "exact_fallback_max_samples": int(fallback_limit),
        "fallback_allowed": int(num_samples) <= int(fallback_limit),
        "approximation_used": False,
        "created_at": _utc_now(),
    }
    path = root / "shortcut_failure.json"
    _atomic_json(path, payload)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "payload": payload,
    }


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


def _ensure_exact_npy(path: Path, expected: np.ndarray, *, label: str) -> str:
    """Create or verify one small deterministic witness array."""

    if path.exists():
        if path.is_symlink():
            raise ExternalMemoryDBSCANError(f"{label} must not be a symlink")
        actual = np.load(path, allow_pickle=False)
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ExternalMemoryDBSCANError(f"{label} resume content mismatch")
    else:
        _atomic_npy(path, expected)
    return _sha256_file(path)


def _validate_shortcut_proof_closure(
    *, manifest: Mapping[str, Any], root: Path
) -> tuple[Path, Path, Path, Path, Path]:
    """Validate a terminal shortcut without inventing exact neighbor counts."""

    if (
        manifest.get("clustering_path") != ALL_CORE_ONE_COMPONENT_SHORTCUT
        or manifest.get("neighbor_counts_available") is not False
        or manifest.get("neighbor_counts_path") is not None
        or manifest.get("neighbor_counts_sha256") is not None
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut contract is invalid")
    proof_path = Path(str(manifest.get("shortcut_proof_path") or "")).resolve(
        strict=True
    )
    if (
        proof_path.parent != root
        or _sha256_file(proof_path) != manifest.get("shortcut_proof_sha256")
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut proof closure mismatch")
    proof = _load_object(proof_path)
    scientific_identity = manifest.get("scientific_identity")
    if not isinstance(scientific_identity, Mapping):
        raise ExternalMemoryDBSCANError("terminal shortcut identity is absent")
    shortcut_contract = scientific_identity.get("shortcut_contract")
    if not isinstance(shortcut_contract, Mapping):
        raise ExternalMemoryDBSCANError("terminal shortcut contract is absent")
    if (
        proof.get("schema_version") != SHORTCUT_PROOF_SCHEMA_VERSION
        or proof.get("status") != "PASS"
        or proof.get("scientific_identity_sha256")
        != manifest.get("scientific_identity_sha256")
        or proof.get("all_points_core_proven") is not True
        or proof.get("single_epsilon_component_proven") is not True
        or proof.get("labels_are_exact_sklearn_order") is not True
        or proof.get("exact_neighbor_counts_materialized") is not False
        or proof.get("approximation_used") is not False
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut proof is incomplete")
    artifacts: list[Path] = []
    for path_field, hash_field in (
        ("anchor_indices_path", "anchor_indices_sha256"),
        ("anchor_edges_path", "anchor_edges_sha256"),
        ("anchor_neighbor_lower_bounds_path", "anchor_neighbor_lower_bounds_sha256"),
    ):
        path = Path(str(proof.get(path_field) or "")).resolve(strict=True)
        if path.parent != root or _sha256_file(path) != proof.get(hash_field):
            raise ExternalMemoryDBSCANError(
                f"terminal shortcut proof artifact mismatch: {path_field}"
            )
        artifacts.append(path)
    anchor_indices = np.load(artifacts[0], allow_pickle=False)
    anchor_edges = np.load(artifacts[1], allow_pickle=False)
    lower = _open_npy_memmap(artifacts[2], mode="r")
    n_samples = int(manifest.get("num_samples", -1))
    anchor_count = int(proof.get("anchor_count", -1))
    min_samples = int(proof.get("min_samples", -1))
    recorded_minimum = int(
        proof.get("minimum_distinct_anchor_neighbors_excluding_self", -1)
    )
    if (
        anchor_indices.dtype != np.dtype(np.intp)
        or anchor_indices.shape != (anchor_count,)
        or len(np.unique(anchor_indices)) != anchor_count
        or np.any(anchor_indices < 0)
        or np.any(anchor_indices >= n_samples)
        or anchor_edges.dtype != np.dtype(np.intp)
        or anchor_edges.ndim != 2
        or anchor_edges.shape[1:] != (2,)
        or lower.dtype != np.dtype(np.uint32)
        or lower.shape != (n_samples,)
        or recorded_minimum < min_samples - 1
        or (
            n_samples > anchor_count
            and int(proof.get("minimum_non_anchor_anchor_neighbors", -1)) < 1
        )
        or proof.get("selected_anchor_indices_sha256")
        != shortcut_contract.get("selected_anchor_indices_sha256")
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut witness schema mismatch")
    if int(np.min(lower)) != recorded_minimum:
        raise ExternalMemoryDBSCANError("terminal shortcut lower-bound minimum mismatch")
    expected_indices_sha = hashlib.sha256(
        json.dumps(anchor_indices.tolist(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected_indices_sha != shortcut_contract.get("selected_anchor_indices_sha256"):
        raise ExternalMemoryDBSCANError("terminal shortcut anchor identity mismatch")
    adjacency = [{index} for index in range(anchor_count)]
    for source, target in anchor_edges.tolist():
        source_index, target_index = int(source), int(target)
        if not (0 <= source_index < target_index < anchor_count):
            raise ExternalMemoryDBSCANError("terminal shortcut edge is noncanonical")
        adjacency[source_index].add(target_index)
        adjacency[target_index].add(source_index)
    reached = {0}
    frontier = [0]
    while frontier:
        source_index = frontier.pop()
        for target_index in adjacency[source_index]:
            if target_index not in reached:
                reached.add(target_index)
                frontier.append(target_index)
    if len(reached) != anchor_count:
        raise ExternalMemoryDBSCANError("terminal shortcut anchor graph is disconnected")
    del lower
    labels = Path(str(manifest.get("labels_path") or "")).resolve(strict=True)
    core = Path(str(manifest.get("core_mask_path") or "")).resolve(strict=True)
    for path, hash_field in (
        (labels, "labels_sha256"),
        (core, "core_mask_sha256"),
    ):
        if path.parent != root or _sha256_file(path) != manifest.get(hash_field):
            raise ExternalMemoryDBSCANError(
                f"terminal shortcut artifact closure mismatch: {hash_field}"
            )
    if (
        proof.get("labels_path") != str(labels)
        or proof.get("labels_sha256") != manifest.get("labels_sha256")
        or proof.get("core_mask_path") != str(core)
        or proof.get("core_mask_sha256") != manifest.get("core_mask_sha256")
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut proof/output mismatch")
    return labels, core, proof_path, artifacts[0], artifacts[2]


def _fit_all_core_one_component_shortcut(
    *,
    vectors: np.ndarray,
    root: Path,
    state_path: Path,
    final_manifest_path: Path,
    identity: Mapping[str, Any],
    contract: ExternalDBSCANContract,
    full_fit_method: str,
    sklearn_version: str,
    peak_rss_bytes: int,
) -> ExternalDBSCANResult | None:
    """Prove the only safe shortcut or return ``None`` for exact fallback.

    Let ``A`` be a set of distinct sample-index anchors.  For every sample
    ``x`` we count distinct anchor *indices* within sklearn's inclusive
    epsilon radius, excluding ``x`` itself when it is an anchor.  If that
    lower bound is at least ``min_samples - 1``, every sample is core because
    DBSCAN also counts its own sample index.  If the anchor epsilon graph is
    connected and every non-anchor has an anchor neighbor, every sample lies
    in that one core component.  sklearn must therefore emit label zero for
    every row, in original order.

    No exact full-data neighbor count is claimed or materialized.
    """

    n_samples, n_features = (int(vectors.shape[0]), int(vectors.shape[1]))
    fallback_limit = int(contract.exact_fallback_max_samples)
    state = _load_object(state_path)
    phase = str(state.get("phase"))
    failure_path = root / "shortcut_failure.json"
    if phase == "shortcut_blocked":
        if not failure_path.is_file() or _sha256_file(failure_path) != state.get(
            "shortcut_failure_sha256"
        ):
            raise ExternalMemoryDBSCANError(
                "blocked shortcut failure artifact mismatch"
            )
        failure = _load_object(failure_path)
        raise ExternalMemoryDBSCANError(
            "EXACT_DBSCAN_COMPLEXITY_BLOCKED:"
            f"samples={n_samples}:fallback_limit={fallback_limit}:"
            f"reason={failure.get('reason')}"
        )
    if phase not in {"shortcut_anchor_scan", "shortcut_finalize"}:
        return None

    def inconclusive(reason: str, details: Mapping[str, Any]) -> None:
        failure = _shortcut_failure(
            root=root,
            identity=identity,
            reason=reason,
            num_samples=n_samples,
            fallback_limit=fallback_limit,
            details=details,
        )
        next_phase = (
            "neighbor_counts" if n_samples <= fallback_limit else "shortcut_blocked"
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase=next_phase,
            next_offset=0,
            peak_rss_bytes=peak_rss_bytes,
            extra={
                "shortcut_failure_path": failure["path"],
                "shortcut_failure_sha256": failure["sha256"],
                "shortcut_approximation_used": False,
            },
        )
        if next_phase == "shortcut_blocked":
            raise ExternalMemoryDBSCANError(
                "EXACT_DBSCAN_COMPLEXITY_BLOCKED:"
                f"samples={n_samples}:fallback_limit={fallback_limit}:reason={reason}"
            )

    if int(contract.min_samples) < 2:
        inconclusive(
            "shortcut_requires_min_samples_at_least_two",
            {"min_samples": int(contract.min_samples)},
        )
        return None
    if n_samples < int(contract.min_samples):
        inconclusive(
            "sample_count_below_min_samples",
            {"sample_count": n_samples, "min_samples": int(contract.min_samples)},
        )
        return None
    anchor_indices = _deterministic_anchor_indices(
        n_samples, int(contract.shortcut_anchor_count)
    )
    if len(anchor_indices) < int(contract.min_samples):
        inconclusive(
            "insufficient_distinct_anchors",
            {
                "anchor_count": int(len(anchor_indices)),
                "min_samples": int(contract.min_samples),
            },
        )
        return None
    if full_fit_method != "brute":
        inconclusive(
            "full_sklearn_fit_method_is_not_brute",
            {"full_fit_method": str(full_fit_method)},
        )
        return None

    anchor_reservation = (
        2 * len(anchor_indices) * n_features * int(vectors.dtype.itemsize)
        + 2 * len(anchor_indices) ** 2 * np.dtype(np.intp).itemsize
        + 64 * 1024**2
    )
    _check_rss(
        int(contract.max_rss_bytes),
        phase="shortcut_anchor_graph",
        reserved_bytes=anchor_reservation,
    )
    anchor_vectors = np.asarray(vectors[anchor_indices])
    anchor_model, anchor_sklearn_version = _fit_anchor_neighbors(
        anchor_vectors, eps=float(contract.eps)
    )
    if anchor_sklearn_version != sklearn_version:
        raise ExternalMemoryDBSCANError("anchor/full sklearn versions differ")
    anchor_edges, anchor_connected, reached, anchor_rows = _anchor_graph(
        model=anchor_model, anchor_vectors=anchor_vectors
    )
    anchor_indices_path = root / "shortcut_anchor_indices.npy"
    anchor_edges_path = root / "shortcut_anchor_edges.npy"
    anchor_indices_sha = _ensure_exact_npy(
        anchor_indices_path, anchor_indices, label="shortcut anchor indices"
    )
    anchor_edges_sha = _ensure_exact_npy(
        anchor_edges_path, anchor_edges, label="shortcut anchor edges"
    )
    if not anchor_connected:
        inconclusive(
            "anchor_epsilon_graph_disconnected",
            {
                "anchor_count": int(len(anchor_indices)),
                "anchor_component_reached_count": int(reached),
                "anchor_edge_count": int(len(anchor_edges)),
                "anchor_neighborhoods_sha256": hashlib.sha256(
                    json.dumps(anchor_rows, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        )
        return None

    lower_partial = root / "shortcut_anchor_neighbor_lower_bounds.partial.npy"
    lower_final = root / "shortcut_anchor_neighbor_lower_bounds.npy"
    peak = max(int(peak_rss_bytes), _rss_bytes())
    if phase == "shortcut_anchor_scan":
        if lower_partial.exists():
            lower = _open_npy_memmap(lower_partial, mode="r+")
            _validate_partial(
                lower,
                shape=(n_samples,),
                dtype=np.uint32,
                label="anchor-neighbor lower bounds",
            )
        else:
            if int(state.get("next_offset", 0)) != 0:
                raise ExternalMemoryDBSCANError(
                    "missing anchor-neighbor lower-bound partial"
                )
            lower = _new_memmap(
                lower_partial, dtype=np.uint32, shape=(n_samples,)
            )
        start = int(state.get("next_offset", 0))
        block = _bounded_shortcut_block_size(
            requested=int(contract.shortcut_query_block_size),
            anchor_count=len(anchor_indices),
            max_rss_bytes=int(contract.max_rss_bytes),
            phase="shortcut_anchor_scan",
        )
        anchor_local_by_global = {
            int(global_index): local_index
            for local_index, global_index in enumerate(anchor_indices.tolist())
        }
        minimum_lower = int(
            state.get("minimum_distinct_anchor_neighbors_excluding_self", len(anchor_indices))
        )
        minimum_non_anchor = state.get("minimum_non_anchor_anchor_neighbors")
        minimum_non_anchor_value = (
            None if minimum_non_anchor is None else int(minimum_non_anchor)
        )
        blocks_since_checkpoint = 0
        for offset in range(start, n_samples, block):
            stop = min(n_samples, offset + block)
            reserved = _shortcut_query_reservation_bytes(
                stop - offset, len(anchor_indices)
            )
            _check_rss(
                int(contract.max_rss_bytes),
                phase="shortcut_anchor_scan.query",
                reserved_bytes=reserved,
            )
            neighborhoods = anchor_model.radius_neighbors(
                vectors[offset:stop], return_distance=False
            )
            block_lower = np.empty(stop - offset, dtype=np.uint32)
            first_failure: dict[str, Any] | None = None
            for local_row, raw in enumerate(neighborhoods):
                global_index = offset + local_row
                values = np.asarray(raw, dtype=np.intp)
                if len(values) != len(np.unique(values)):
                    raise ExternalMemoryDBSCANError(
                        "anchor scan returned duplicate sample indices"
                    )
                own_anchor = anchor_local_by_global.get(global_index)
                if own_anchor is not None and own_anchor not in values:
                    raise ExternalMemoryDBSCANError(
                        "anchor scan omitted explicit query self sample"
                    )
                count_excluding_self = len(values) - int(own_anchor is not None)
                block_lower[local_row] = count_excluding_self
                minimum_lower = min(minimum_lower, count_excluding_self)
                if own_anchor is None:
                    minimum_non_anchor_value = (
                        count_excluding_self
                        if minimum_non_anchor_value is None
                        else min(minimum_non_anchor_value, count_excluding_self)
                    )
                if (
                    count_excluding_self < int(contract.min_samples) - 1
                    or (own_anchor is None and count_excluding_self < 1)
                ) and first_failure is None:
                    first_failure = {
                        "sample_index": int(global_index),
                        "distinct_anchor_neighbors_excluding_self": int(
                            count_excluding_self
                        ),
                        "required_for_core": int(contract.min_samples) - 1,
                        "required_for_anchor_attachment_if_non_anchor": 1,
                        "sample_is_anchor": own_anchor is not None,
                    }
            lower[offset:stop] = block_lower
            del neighborhoods, block_lower
            peak = max(
                peak,
                _check_rss(
                    int(contract.max_rss_bytes), phase="shortcut_anchor_scan"
                ),
            )
            if first_failure is not None:
                _fsync_memmap(lower)
                inconclusive("per_sample_anchor_lower_bound_failed", first_failure)
                del lower
                return None
            blocks_since_checkpoint += 1
            if (
                blocks_since_checkpoint >= int(contract.checkpoint_interval_blocks)
                or stop == n_samples
            ):
                _fsync_memmap(lower)
                _checkpoint(
                    state_path,
                    identity=identity,
                    phase="shortcut_anchor_scan",
                    next_offset=stop,
                    peak_rss_bytes=peak,
                    extra={
                        "effective_shortcut_query_block_size": int(block),
                        "anchor_indices_sha256": anchor_indices_sha,
                        "anchor_edges_sha256": anchor_edges_sha,
                        "minimum_distinct_anchor_neighbors_excluding_self": int(
                            minimum_lower
                        ),
                        "minimum_non_anchor_anchor_neighbors": minimum_non_anchor_value,
                    },
                )
                blocks_since_checkpoint = 0
        _fsync_memmap(lower)
        lower_sha = _sha256_file(lower_partial)
        _checkpoint(
            state_path,
            identity=identity,
            phase="shortcut_finalize",
            next_offset=n_samples,
            peak_rss_bytes=peak,
            extra={
                "effective_shortcut_query_block_size": int(block),
                "anchor_indices_sha256": anchor_indices_sha,
                "anchor_edges_sha256": anchor_edges_sha,
                "anchor_neighbor_lower_bounds_sha256": lower_sha,
                "minimum_distinct_anchor_neighbors_excluding_self": int(
                    minimum_lower
                ),
                "minimum_non_anchor_anchor_neighbors": minimum_non_anchor_value,
                "anchor_edge_count": int(len(anchor_edges)),
                "anchor_count": int(len(anchor_indices)),
            },
        )
        del lower
        phase = "shortcut_finalize"
        state = _load_object(state_path)

    if phase != "shortcut_finalize":
        raise ExternalMemoryDBSCANError(f"unexpected shortcut phase: {phase}")
    lower_final = _reconcile_promoted_array(
        partial=lower_partial,
        final=lower_final,
        shape=(n_samples,),
        dtype=np.uint32,
        expected_sha256=str(state.get("anchor_neighbor_lower_bounds_sha256") or ""),
        label="anchor-neighbor lower bounds",
    )
    if anchor_indices_sha != state.get("anchor_indices_sha256"):
        raise ExternalMemoryDBSCANError("shortcut anchor-index checkpoint mismatch")
    if anchor_edges_sha != state.get("anchor_edges_sha256"):
        raise ExternalMemoryDBSCANError("shortcut anchor-edge checkpoint mismatch")

    labels_path = root / "labels.npy"
    core_path = root / "core_mask.npy"
    _constant_npy(
        labels_path, shape=(n_samples,), dtype=np.intp, fill_value=0
    )
    _constant_npy(
        core_path, shape=(n_samples,), dtype=np.bool_, fill_value=True
    )
    labels_sha = _sha256_file(labels_path)
    core_sha = _sha256_file(core_path)
    lower_sha = _sha256_file(lower_final)
    proof_path = root / "shortcut_proof.json"
    proof = {
        "schema_version": SHORTCUT_PROOF_SCHEMA_VERSION,
        "status": "PASS",
        "shortcut": ALL_CORE_ONE_COMPONENT_SHORTCUT,
        "scientific_identity_sha256": _stable_hash(identity),
        "vectors_path": str(Path(identity["vectors_path"])),
        "vectors_sha256": identity["vectors_sha256"],
        "vectors_dtype": identity["vectors_dtype"],
        "vectors_shape": identity["vectors_shape"],
        "sklearn_version": sklearn_version,
        "distance_kernel": "sklearn NearestNeighbors brute euclidean",
        "epsilon_comparison": "distance <= float(eps)",
        "eps": float(contract.eps),
        "min_samples": int(contract.min_samples),
        "self_count_semantics": "each sample index counts itself exactly once",
        "duplicate_semantics": "duplicate vectors remain distinct sample indices",
        "anchor_selection": "floor(i*(N-1)/(A-1)); endpoints included",
        "selected_anchor_indices_sha256": identity["shortcut_contract"][
            "selected_anchor_indices_sha256"
        ],
        "anchor_indices_path": str(anchor_indices_path),
        "anchor_indices_sha256": anchor_indices_sha,
        "anchor_count": int(len(anchor_indices)),
        "anchor_edges_path": str(anchor_edges_path),
        "anchor_edges_sha256": anchor_edges_sha,
        "anchor_edge_count": int(len(anchor_edges)),
        "anchor_epsilon_graph_connected": True,
        "anchor_neighbor_lower_bounds_path": str(lower_final),
        "anchor_neighbor_lower_bounds_sha256": lower_sha,
        "anchor_neighbor_lower_bound_definition": (
            "distinct anchor sample indices within eps, excluding the query's "
            "own sample index when it is an anchor"
        ),
        "minimum_distinct_anchor_neighbors_excluding_self": int(
            state["minimum_distinct_anchor_neighbors_excluding_self"]
        ),
        "minimum_non_anchor_anchor_neighbors": state.get(
            "minimum_non_anchor_anchor_neighbors"
        ),
        "all_points_core_proven": True,
        "single_epsilon_component_proven": True,
        "labels_are_exact_sklearn_order": True,
        "label_value": 0,
        "core_mask_value": True,
        "exact_neighbor_counts_materialized": False,
        "approximation_used": False,
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha,
        "core_mask_path": str(core_path),
        "core_mask_sha256": core_sha,
        "completed_at": _utc_now(),
    }
    _atomic_json(proof_path, proof)
    proof_sha = _sha256_file(proof_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_complete": True,
        "scientific_identity": dict(identity),
        "scientific_identity_sha256": _stable_hash(identity),
        "num_samples": n_samples,
        "num_features": n_features,
        "core_count": n_samples,
        "cluster_count": 1,
        "noise_count": 0,
        "neighbor_counts_available": False,
        "neighbor_counts_path": None,
        "neighbor_counts_sha256": None,
        "neighbor_counts_unavailable_reason": (
            "exact labels/core proven by anchor lower-bound witness; full exact "
            "neighbor counts were neither needed nor computed"
        ),
        "core_mask_path": str(core_path),
        "core_mask_sha256": core_sha,
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha,
        "shortcut_proof_path": str(proof_path),
        "shortcut_proof_sha256": proof_sha,
        "clustering_path": ALL_CORE_ONE_COMPONENT_SHORTCUT,
        "peak_rss_bytes_observed": max(peak, _rss_bytes()),
        "max_rss_bytes": int(contract.max_rss_bytes),
        "all_neighborhoods_materialized_simultaneously": False,
        "passes": ["anchor_graph", "anchor_lower_bound_scan", "constant_exact_labels"],
        "sklearn_dbscan_label_semantics_preserved": True,
        "approximation_used": False,
        "completed_at": _utc_now(),
    }
    if manifest["peak_rss_bytes_observed"] > int(contract.max_rss_bytes):
        raise ExternalMemoryDBSCANError("recorded peak RSS exceeded the frozen budget")
    _atomic_json(final_manifest_path, manifest)
    _checkpoint(
        state_path,
        identity=identity,
        phase="complete",
        next_offset=n_samples,
        peak_rss_bytes=int(manifest["peak_rss_bytes_observed"]),
        extra={
            "shortcut_proof_sha256": proof_sha,
            "labels_sha256": labels_sha,
            "core_mask_sha256": core_sha,
        },
    )
    return ExternalDBSCANResult(
        labels_path=labels_path,
        core_mask_path=core_path,
        neighbor_counts_path=None,
        shortcut_proof_path=proof_path,
        manifest_path=final_manifest_path,
        num_samples=n_samples,
        num_features=n_features,
        cluster_count=1,
        noise_count=0,
        core_count=n_samples,
        manifest_sha256=_sha256_file(final_manifest_path),
    )


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
        shortcut_path: Path | None = None
        if manifest.get("clustering_path") == ALL_CORE_ONE_COMPONENT_SHORTCUT:
            labels_path, core_path, shortcut_path, _anchors, _lower = (
                _validate_shortcut_proof_closure(manifest=manifest, root=root)
            )
            counts_path = None
        else:
            if manifest.get("neighbor_counts_available", True) is not True:
                raise ExternalMemoryDBSCANError(
                    "terminal DBSCAN exact-neighbor contract is invalid"
                )
            labels_path = Path(str(manifest["labels_path"])).resolve(strict=True)
            core_path = Path(str(manifest["core_mask_path"])).resolve(strict=True)
            counts_path = Path(str(manifest["neighbor_counts_path"])).resolve(
                strict=True
            )
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
            shortcut_proof_path=shortcut_path,
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
        "shortcut_contract": {
            "mode": contract.shortcut_mode,
            "anchor_count": int(contract.shortcut_anchor_count),
            "query_block_size": int(contract.shortcut_query_block_size),
            "exact_fallback_max_samples": int(
                contract.exact_fallback_max_samples
            ),
            "anchor_selection": "floor(i*(N-1)/(A-1)); endpoints included",
            "selected_anchor_indices_sha256": hashlib.sha256(
                json.dumps(
                    _deterministic_anchor_indices(
                        n_samples, int(contract.shortcut_anchor_count)
                    ).tolist(),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
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
            "phase": (
                "shortcut_anchor_scan"
                if contract.shortcut_mode == ALL_CORE_ONE_COMPONENT_SHORTCUT
                else "neighbor_counts"
            ),
            "next_offset": 0,
            "peak_rss_bytes": _rss_bytes(),
        }
        _checkpoint(
            state_path,
            identity=identity,
            phase=str(state["phase"]),
            next_offset=0,
            peak_rss_bytes=int(state["peak_rss_bytes"]),
        )

    peak = max(int(state.get("peak_rss_bytes", 0)), _rss_bytes())
    if contract.shortcut_mode == ALL_CORE_ONE_COMPONENT_SHORTCUT:
        shortcut_result = _fit_all_core_one_component_shortcut(
            vectors=vectors,
            root=root,
            state_path=state_path,
            final_manifest_path=final_manifest_path,
            identity=identity,
            contract=contract,
            full_fit_method=fit_method,
            sklearn_version=sklearn_version,
            peak_rss_bytes=peak,
        )
        if shortcut_result is not None:
            return shortcut_result
        state = _load_object(state_path)
        peak = max(peak, int(state.get("peak_rss_bytes", 0)), _rss_bytes())
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
        "neighbor_counts_available": True,
        "core_mask_path": str(core_path),
        "core_mask_sha256": _sha256_file(core_path),
        "labels_path": str(labels_final),
        "labels_sha256": _sha256_file(labels_final),
        "peak_rss_bytes_observed": max(peak, _rss_bytes()),
        "max_rss_bytes": int(contract.max_rss_bytes),
        "all_neighborhoods_materialized_simultaneously": False,
        "passes": ["neighbor_counts", "core_union", "border_assignment"],
        "clustering_path": "three_pass_exact_radius_graph_v1",
        "shortcut_proof_path": None,
        "shortcut_proof_sha256": None,
        "approximation_used": False,
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
        shortcut_proof_path=None,
        manifest_path=final_manifest_path,
        num_samples=n_samples,
        num_features=n_features,
        cluster_count=int(cluster_labels.size),
        noise_count=int(manifest["noise_count"]),
        core_count=int(core_indices.size),
        manifest_sha256=_sha256_file(final_manifest_path),
    )


__all__ = [
    "ALL_CORE_ONE_COMPONENT_SHORTCUT",
    "ExternalDBSCANContract",
    "ExternalDBSCANResult",
    "ExternalMemoryDBSCANError",
    "SCHEMA_VERSION",
    "SHORTCUT_DISABLED",
    "SHORTCUT_PROOF_SCHEMA_VERSION",
    "fit_external_memory_dbscan",
]
