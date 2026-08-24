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
The passes are checkpointed independently.  Shortcut checkpoints bind every
committed prefix through a forward hash-chain ledger and replay that prefix
from the source/model on resume.  Partial arrays live only in the caller-owned
fresh work directory and are fsynced before an atomic state update, so
replaying the last incomplete block is idempotent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "comrecgc_external_memory_dbscan_v3"
SHORTCUT_DISABLED = "disabled"
ALL_CORE_ONE_COMPONENT_SHORTCUT = "all_core_one_component_anchor_v1"
ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT = (
    "all_core_one_component_adaptive_anchor_v1"
)
ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY = (
    "all_core_adaptive_anchor_component_recovery_v1"
)
SHORTCUT_PROOF_SCHEMA_VERSION = "comrecgc_dbscan_anchor_proof_v2"
ADAPTIVE_SELECTION_SCHEMA_VERSION = "comrecgc_adaptive_anchor_selection_v2"
PROGRESS_LEDGER_SCHEMA_VERSION = "comrecgc_shortcut_progress_ledger_v1"
ALL_CORE_CERTIFICATE_SCHEMA_VERSION = "comrecgc_dbscan_all_core_certificate_v1"
CONNECTIVITY_CERTIFICATE_SCHEMA_VERSION = (
    "comrecgc_dbscan_connectivity_certificate_v1"
)
BOUNDARY_CERTIFICATE_SCHEMA_VERSION = "comrecgc_dbscan_boundary_certificate_v1"
CLUSTER_PARTITION_SCHEMA_VERSION = "comrecgc_dbscan_cluster_partition_v1"
COMPONENT_RECOVERY_SCHEMA_VERSION = (
    "comrecgc_dbscan_adaptive_component_recovery_v1"
)
COMPONENT_PRIMARY_LEDGER_PHASE = "shortcut_anchor_scan"
COMPONENT_EXPANSION_LEDGER_PHASE = "adaptive_component_expansion_scan"
COMPONENT_PRIMARY_PHASE = "adaptive_component_primary"
COMPONENT_EXPANSION_PHASE = "adaptive_component_expansion"
COMPONENT_FINALIZE_PHASE = "adaptive_component_finalize"


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
    shortcut_seed_count: int = 3
    shortcut_failure_cap: int = 4_096
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
            ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
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
        if int(self.shortcut_seed_count) <= 0:
            raise ExternalMemoryDBSCANError("shortcut_seed_count must be positive")
        if int(self.shortcut_seed_count) > 65_535:
            raise ExternalMemoryDBSCANError(
                "shortcut_seed_count exceeds the auditable finite-anchor limit"
            )
        if int(self.shortcut_failure_cap) <= 0:
            raise ExternalMemoryDBSCANError("shortcut_failure_cap must be positive")
        if int(self.shortcut_failure_cap) > 65_535:
            raise ExternalMemoryDBSCANError(
                "shortcut_failure_cap exceeds the auditable finite-anchor limit"
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


def _source_stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ExternalMemoryDBSCANError(
            f"DBSCAN vector source is not a regular file: {path}"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _hash_source_with_stable_stat(path: Path) -> tuple[str, dict[str, int]]:
    before = _source_stat_identity(path)
    digest = _sha256_file(path)
    after = _source_stat_identity(path)
    if before != after:
        raise ExternalMemoryDBSCANError(
            "DBSCAN vector source changed during identity hashing"
        )
    return digest, before


def _verify_source_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_stat: Mapping[str, Any],
    phase: str,
) -> None:
    before = _source_stat_identity(path)
    if before != dict(expected_stat):
        raise ExternalMemoryDBSCANError(
            f"DBSCAN vector source stat identity changed before {phase}"
        )
    digest = _sha256_file(path)
    after = _source_stat_identity(path)
    if (
        after != before
        or after != dict(expected_stat)
        or digest != str(expected_sha256)
    ):
        raise ExternalMemoryDBSCANError(
            f"DBSCAN vector source content identity changed before {phase}"
        )


def _assert_source_stat_identity(
    path: Path, *, expected_stat: Mapping[str, Any], phase: str
) -> None:
    if _source_stat_identity(path) != dict(expected_stat):
        raise ExternalMemoryDBSCANError(
            f"DBSCAN vector source stat identity changed before {phase}"
        )


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
    payload["checkpoint_payload_sha256"] = _stable_hash(payload)
    _atomic_json(state_path, payload)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load one checkpoint and reject any mutation of its atomic payload."""

    payload = _load_object(path)
    expected = payload.get("checkpoint_payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("checkpoint_payload_sha256", None)
    if not isinstance(expected, str) or expected != _stable_hash(unsigned):
        raise ExternalMemoryDBSCANError("checkpoint authentication mismatch")
    return payload


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


def _sample_indices_sha256(values: Sequence[int] | np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps(
            [int(value) for value in values], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


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


def _bounded_norm_block_size(
    *,
    requested: int,
    num_features: int,
    source_itemsize: int,
    max_rss_bytes: int,
) -> int:
    current = _check_rss(max_rss_bytes, phase="adaptive_seed_scan")
    available = int(max_rss_bytes) - current - 64 * 1024**2
    per_row = max(
        1,
        2 * int(num_features) * np.dtype(np.float64).itemsize
        + int(num_features) * int(source_itemsize)
        + 4096,
    )
    safe = available // per_row
    if safe < 1:
        raise ExternalMemoryDBSCANError(
            "RSS budget cannot hold one adaptive seed-selection row"
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


def _component_find(parent: np.ndarray, value: int) -> int:
    # Component recovery deliberately remains in this module: its source-stat
    # authority, authenticated progress-ledger schema, two-phase array
    # promotion, and terminal proof closure are the same transaction as the
    # existing exact DBSCAN engine.  Splitting only the numerical helpers would
    # either duplicate those cryptographic primitives or introduce a circular
    # checkpoint dependency.
    root = int(value)
    while int(parent[root]) != root:
        root = int(parent[root])
    cursor = int(value)
    while int(parent[cursor]) != cursor:
        following = int(parent[cursor])
        parent[cursor] = root
        cursor = following
    return root


def _component_union(parent: np.ndarray, left: int, right: int) -> bool:
    left_root = _component_find(parent, int(left))
    right_root = _component_find(parent, int(right))
    if left_root == right_root:
        return False
    lower, upper = sorted((left_root, right_root))
    parent[upper] = lower
    return True


def _canonical_anchor_components(
    *, anchor_indices: np.ndarray, anchor_rows: Sequence[Sequence[int]]
) -> tuple[np.ndarray, list[list[int]]]:
    """Return exact anchor components ordered by minimum global sample index."""

    anchor_count = int(len(anchor_indices))
    if len(anchor_rows) != anchor_count:
        raise ExternalMemoryDBSCANError("anchor component row count mismatch")
    visited = np.zeros(anchor_count, dtype=np.bool_)
    raw_components: list[list[int]] = []
    for start in range(anchor_count):
        if bool(visited[start]):
            continue
        frontier = [start]
        visited[start] = True
        members: list[int] = []
        while frontier:
            source = int(frontier.pop())
            members.append(source)
            for target in anchor_rows[source]:
                target_index = int(target)
                if target_index < 0 or target_index >= anchor_count:
                    raise ExternalMemoryDBSCANError(
                        "anchor component edge escaped anchor set"
                    )
                if not bool(visited[target_index]):
                    visited[target_index] = True
                    frontier.append(target_index)
        raw_components.append(sorted(members))
    components = sorted(
        raw_components,
        key=lambda members: min(int(anchor_indices[value]) for value in members),
    )
    component_by_anchor = np.empty(anchor_count, dtype=np.int32)
    for component, members in enumerate(components):
        component_by_anchor[np.asarray(members, dtype=np.intp)] = int(component)
    return component_by_anchor, components


def _float64_anchor_graph_recheck(
    *,
    anchor_vectors: np.ndarray,
    anchor_rows: Sequence[Sequence[int]],
    eps: float,
) -> tuple[np.ndarray, float, int]:
    """Fail closed unless float64 and frozen sklearn agree on every anchor edge."""

    anchors = np.asarray(anchor_vectors, dtype=np.float64)
    distances = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=2)
    within = distances <= float(eps)
    expected = np.zeros(within.shape, dtype=np.bool_)
    for source, raw in enumerate(anchor_rows):
        expected[source, np.asarray(raw, dtype=np.intp)] = True
    if not np.array_equal(within, expected):
        raise ExternalMemoryDBSCANError(
            "FLOAT_BOUNDARY_UNCERTAIN: sklearn/float64 anchor graph differs"
        )
    edge_distances = distances[within]
    return (
        distances,
        float(np.min(float(eps) - edge_distances)),
        int(np.count_nonzero(edge_distances == float(eps))),
    )


def _adaptive_primary_query_anchors(
    *,
    anchor_indices: np.ndarray,
    anchor_vectors64: np.ndarray,
    component_by_anchor: np.ndarray,
    seed_indices: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Choose every seed plus one exact nearest-to-seed anchor per other component."""

    local_by_global = {
        int(global_index): local
        for local, global_index in enumerate(anchor_indices.tolist())
    }
    try:
        seed_locals = [local_by_global[int(value)] for value in seed_indices]
    except KeyError as exc:
        raise ExternalMemoryDBSCANError(
            "adaptive seed is absent from selected anchors"
        ) from exc
    seed_components = {int(component_by_anchor[value]) for value in seed_locals}
    selected = set(seed_locals)
    boundary_rows: list[dict[str, Any]] = []
    component_count = int(np.max(component_by_anchor)) + 1
    for component in range(component_count):
        if component in seed_components:
            continue
        members = np.flatnonzero(component_by_anchor == component).tolist()
        choices: list[tuple[float, int, int, int]] = []
        for local in members:
            distances = np.linalg.norm(
                anchor_vectors64[np.asarray(seed_locals, dtype=np.intp)]
                - anchor_vectors64[int(local)],
                axis=1,
            )
            seed_position = int(np.argmin(distances))
            choices.append(
                (
                    float(distances[seed_position]),
                    int(anchor_indices[int(local)]),
                    int(local),
                    int(seed_locals[seed_position]),
                )
            )
        distance, _global, local, seed_local = min(choices)
        selected.add(local)
        boundary_rows.append(
            {
                "component": int(component),
                "anchor_local_index": int(local),
                "anchor_global_index": int(anchor_indices[local]),
                "nearest_seed_anchor_local_index": int(seed_local),
                "nearest_seed_anchor_global_index": int(anchor_indices[seed_local]),
                "anchor_to_seed_distance_float64_hex": float(distance).hex(),
            }
        )
    return np.asarray(sorted(selected), dtype=np.intp), boundary_rows


def _exact_query_membership(
    *,
    model: Any,
    vectors: np.ndarray,
    query_vectors64: np.ndarray,
    start: int,
    stop: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return sklearn membership after exact float64 elementwise agreement."""

    neighborhoods = model.radius_neighbors(
        vectors[start:stop], return_distance=False
    )
    width = int(query_vectors64.shape[0])
    sklearn_within = np.zeros((stop - start, width), dtype=np.bool_)
    for row, raw in enumerate(neighborhoods):
        values = np.asarray(raw, dtype=np.intp)
        if len(values) != len(np.unique(values)):
            raise ExternalMemoryDBSCANError(
                "component recovery radius query returned duplicate indices"
            )
        if bool(np.any(values < 0)) or bool(np.any(values >= width)):
            raise ExternalMemoryDBSCANError(
                "component recovery radius query escaped query-anchor set"
            )
        sklearn_within[row, values] = True
    del neighborhoods
    block = np.asarray(vectors[start:stop], dtype=np.float64)
    block_squared = np.einsum("ij,ij->i", block, block)
    query_squared = np.einsum("ij,ij->i", query_vectors64, query_vectors64)
    squared = (
        block_squared[:, None]
        + query_squared[None, :]
        - 2.0 * np.matmul(block, query_vectors64.T)
    )
    np.maximum(squared, 0.0, out=squared)
    distances = np.sqrt(squared, out=squared)
    comparison_guard = max(
        64.0 * np.finfo(np.float64).eps * max(1.0, abs(float(eps))), 1.0e-15
    )
    near = np.abs(distances - float(eps)) <= comparison_guard
    near_count = int(np.count_nonzero(near))
    if near_count:
        for row, column in np.argwhere(near).tolist():
            distances[int(row), int(column)] = float(
                np.linalg.norm(block[int(row)] - query_vectors64[int(column)])
            )
    float64_within = distances <= float(eps)
    if not np.array_equal(sklearn_within, float64_within):
        raise ExternalMemoryDBSCANError(
            "FLOAT_BOUNDARY_UNCERTAIN: sklearn/float64 recovery membership differs"
        )
    return sklearn_within, distances, near_count


def _failure_mask_for_block(
    failures: np.ndarray, *, start: int, stop: int
) -> np.ndarray:
    rows = np.arange(start, stop, dtype=np.intp)
    positions = np.searchsorted(failures, rows)
    valid = positions < len(failures)
    result = np.zeros(stop - start, dtype=np.bool_)
    result[valid] = failures[positions[valid]] == rows[valid]
    return result


def _recovery_scan_block(
    *,
    model: Any,
    vectors: np.ndarray,
    start: int,
    stop: int,
    eps: float,
    min_samples: int,
    query_anchor_locals: np.ndarray,
    query_anchor_globals: np.ndarray,
    query_vectors64: np.ndarray,
    component_by_anchor: np.ndarray,
    component_parent: np.ndarray,
    seed_indices: Sequence[int],
    failures: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_degrees: np.ndarray,
    lower_output: np.ndarray | None,
    attachment_output: np.ndarray | None,
    verify_core_contract: bool,
    write_core_outputs: bool,
) -> dict[str, Any]:
    within, distances, near_count = _exact_query_membership(
        model=model,
        vectors=vectors,
        query_vectors64=query_vectors64,
        start=start,
        stop=stop,
        eps=float(eps),
    )
    failure_mask = _failure_mask_for_block(failures, start=start, stop=stop)
    nonfailure_mask = ~failure_mask
    query_column_by_global = {
        int(value): column for column, value in enumerate(query_anchor_globals.tolist())
    }
    seed_columns = np.asarray(
        [query_column_by_global[int(value)] for value in seed_indices], dtype=np.intp
    )
    seed_within = np.asarray(within[:, seed_columns], dtype=np.bool_)
    seed_counts = np.sum(seed_within, axis=1, dtype=np.uint32)
    for seed_position, global_index in enumerate(seed_indices):
        if start <= int(global_index) < stop:
            local_row = int(global_index) - start
            if not bool(seed_within[local_row, seed_position]):
                raise ExternalMemoryDBSCANError(
                    "component recovery seed query omitted explicit self"
                )
            seed_counts[local_row] -= 1
    classified_failure = (seed_counts < int(min_samples) - 1) | (
        nonfailure_mask & (seed_counts < 1)
    )
    if not np.array_equal(classified_failure, failure_mask):
        first = int(np.flatnonzero(classified_failure != failure_mask)[0]) + start
        raise ExternalMemoryDBSCANError(
            "adaptive recovery/failure-ledger classification mismatch:"
            f"sample={first}"
        )

    block_lower = np.asarray(seed_counts, dtype=np.uint32)
    block_attachment = np.full(stop - start, -1, dtype=np.int32)
    seed_components = component_by_anchor[
        np.asarray(
            [
                int(np.searchsorted(anchor_indices, int(value)))
                for value in seed_indices
            ],
            dtype=np.intp,
        )
    ]
    for seed_position, component in enumerate(seed_components.tolist()):
        mask = nonfailure_mask & seed_within[:, seed_position]
        current = block_attachment[mask]
        block_attachment[mask] = np.where(
            current < 0, int(component), np.minimum(current, int(component))
        )
    failure_rows = np.flatnonzero(failure_mask)
    if failure_rows.size:
        failure_globals = failure_rows.astype(np.intp, copy=False) + int(start)
        anchor_locals = np.searchsorted(anchor_indices, failure_globals)
        if bool(np.any(anchor_locals >= len(anchor_indices))) or not np.array_equal(
            anchor_indices[anchor_locals], failure_globals
        ):
            raise ExternalMemoryDBSCANError(
                "adaptive recovery failure is absent from anchor closure"
            )
        failure_lower = anchor_degrees[anchor_locals] - 1
        block_lower[failure_rows] = np.asarray(failure_lower, dtype=np.uint32)
        block_attachment[failure_rows] = component_by_anchor[anchor_locals]
    if verify_core_contract:
        if bool(np.any(block_lower < int(min_samples) - 1)):
            raise ExternalMemoryDBSCANError(
                "adaptive disconnected recovery cannot prove every row core"
            )
        if bool(np.any(block_attachment < 0)):
            raise ExternalMemoryDBSCANError(
                "adaptive disconnected recovery cannot attach every row"
            )
        if lower_output is None or attachment_output is None:
            raise ExternalMemoryDBSCANError(
                "adaptive recovery core outputs are absent"
            )
        if write_core_outputs:
            lower_output[start:stop] = block_lower
            attachment_output[start:stop] = block_attachment
        elif (
            not np.array_equal(np.asarray(lower_output[start:stop]), block_lower)
            or not np.array_equal(
                np.asarray(attachment_output[start:stop]), block_attachment
            )
        ):
            raise ExternalMemoryDBSCANError(
                "adaptive recovery replay/core artifact mismatch"
            )

    # Each witness is x--a and x--b for one nonfailure/core x and anchors in
    # different exact anchor components.  Record only deterministic spanning
    # edges; at most component_count-1 witnesses survive the whole scan.
    columns_by_component: dict[int, list[int]] = {}
    for column, local_anchor in enumerate(query_anchor_locals.tolist()):
        component = int(component_by_anchor[int(local_anchor)])
        columns_by_component.setdefault(component, []).append(column)
    component_membership = {
        component: np.any(within[:, np.asarray(columns, dtype=np.intp)], axis=1)
        for component, columns in columns_by_component.items()
    }
    candidates: list[tuple[int, int, int]] = []
    component_values = sorted(columns_by_component)
    for left_index, left in enumerate(component_values):
        for right in component_values[left_index + 1 :]:
            rows = np.flatnonzero(
                nonfailure_mask
                & component_membership[left]
                & component_membership[right]
            )
            if rows.size:
                candidates.append((int(rows[0]) + start, int(left), int(right)))
    witnesses: list[dict[str, Any]] = []
    for global_row, left, right in sorted(candidates):
        if not _component_union(component_parent, left, right):
            continue
        local_row = global_row - start
        left_columns = [
            column
            for column in columns_by_component[left]
            if bool(within[local_row, column])
        ]
        right_columns = [
            column
            for column in columns_by_component[right]
            if bool(within[local_row, column])
        ]
        left_column = min(
            left_columns, key=lambda value: int(query_anchor_globals[value])
        )
        right_column = min(
            right_columns, key=lambda value: int(query_anchor_globals[value])
        )
        source_row64 = np.asarray(vectors[global_row], dtype=np.float64)
        left_distance = float(
            np.linalg.norm(source_row64 - query_vectors64[left_column])
        )
        right_distance = float(
            np.linalg.norm(source_row64 - query_vectors64[right_column])
        )
        witness_seed_columns = [
            int(column)
            for column in seed_columns.tolist()
            if bool(within[local_row, int(column)])
            and int(query_anchor_globals[int(column)]) != int(global_row)
        ]
        if len(witness_seed_columns) < int(min_samples) - 1:
            raise ExternalMemoryDBSCANError(
                "adaptive recovery bridge lacks its exact seed core witness"
            )
        witness_seed_rows = sorted([
            {
                "anchor_global_index": int(query_anchor_globals[column]),
                "distance_float64_hex": float(
                    np.linalg.norm(source_row64 - query_vectors64[column])
                ).hex(),
            }
            for column in witness_seed_columns
        ], key=lambda value: int(value["anchor_global_index"]))
        witnesses.append(
            {
                "sample_index": int(global_row),
                "left_initial_component": int(left),
                "right_initial_component": int(right),
                "left_anchor_local_index": int(query_anchor_locals[left_column]),
                "right_anchor_local_index": int(query_anchor_locals[right_column]),
                "left_anchor_global_index": int(query_anchor_globals[left_column]),
                "right_anchor_global_index": int(query_anchor_globals[right_column]),
                "left_distance_float64_hex": left_distance.hex(),
                "right_distance_float64_hex": right_distance.hex(),
                "seed_core_witnesses": witness_seed_rows,
            }
        )

    needed = int(min_samples) - 1
    certifying_margins: list[float] = []
    exact_boundary_count = 0
    if verify_core_contract and needed:
        candidate_distances = np.asarray(distances[:, seed_columns]).copy()
        for seed_position, global_index in enumerate(seed_indices):
            if start <= int(global_index) < stop:
                candidate_distances[int(global_index) - start, seed_position] = np.inf
        nonfailure_rows = np.flatnonzero(nonfailure_mask)
        if nonfailure_rows.size:
            kth = np.partition(
                candidate_distances[nonfailure_rows], needed - 1, axis=1
            )[:, needed - 1]
            if not bool(np.isfinite(kth).all()) or bool(np.any(kth > float(eps))):
                raise ExternalMemoryDBSCANError(
                    "adaptive recovery float64 seed core witness failed"
                )
            certifying_margins.append(float(np.min(float(eps) - kth)))
            exact_boundary_count += int(np.count_nonzero(kth == float(eps)))
    for witness in witnesses:
        for field in ("left_distance_float64_hex", "right_distance_float64_hex"):
            value = float.fromhex(str(witness[field]))
            if value > float(eps):
                raise ExternalMemoryDBSCANError(
                    "adaptive recovery bridge escaped inclusive epsilon"
                )
            certifying_margins.append(float(eps) - value)
            exact_boundary_count += int(value == float(eps))
    payload = {
        "row_count": int(stop - start),
        "query_anchor_indices_sha256": _sample_indices_sha256(
            query_anchor_globals
        ),
        "sklearn_float64_membership_equal": True,
        "membership_bool_sha256": hashlib.sha256(
            np.ascontiguousarray(within).tobytes(order="C")
        ).hexdigest(),
        "near_boundary_direct_norm_recompute_count": int(near_count),
        "new_bridge_witnesses": witnesses,
        "component_parent_after": [
            _component_find(component_parent, value)
            for value in range(len(component_parent))
        ],
        "minimum_certifying_margin_hex": (
            None if not certifying_margins else float(min(certifying_margins)).hex()
        ),
        "count_certifying_edges_exactly_at_eps": int(exact_boundary_count),
    }
    if verify_core_contract:
        payload.update(
            {
                "core_lower_bounds_uint32_le_sha256": _uint32_values_sha256(
                    block_lower
                ),
                "attachment_components_uint32_le_sha256": _uint32_values_sha256(
                    np.asarray(block_attachment, dtype=np.uint32)
                ),
                "minimum_core_lower_bound": int(np.min(block_lower)),
                "failure_classification_exact": True,
            }
        )
    return payload


def _write_shortcut_split_certificates(
    *,
    root: Path,
    identity: Mapping[str, Any],
    vectors: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_indices_path: Path,
    anchor_indices_sha256: str,
    anchor_edges: np.ndarray,
    anchor_edges_path: Path,
    anchor_edges_sha256: str,
    lower_path: Path,
    lower_sha256: str,
    labels_path: Path,
    labels_sha256: str,
    core_path: Path,
    core_sha256: str,
    contract: ExternalDBSCANContract,
) -> dict[str, Any]:
    """Float64-revalidate every shortcut witness and publish split proofs.

    The sklearn radius scan remains the paper-compatible edge authority.  This
    terminal pass recomputes every point-to-anchor comparison in float64 and
    requires the resulting distinct-anchor count to equal the committed
    sklearn lower-bound witness exactly.  Any precision disagreement therefore
    fails closed instead of being hidden behind a tolerance.
    """

    n_samples, n_features = map(int, vectors.shape)
    eps = float(contract.eps)
    needed_other_neighbors = max(0, int(contract.min_samples) - 1)
    anchors = np.asarray(vectors[anchor_indices], dtype=np.float64)
    anchor_count = int(len(anchor_indices))
    lower = _open_npy_memmap(lower_path, mode="r")
    if lower.shape != (n_samples,) or lower.dtype != np.dtype(np.uint32):
        raise ExternalMemoryDBSCANError("float64 boundary witness shape mismatch")
    anchor_column_by_sample = {
        int(sample): int(column)
        for column, sample in enumerate(anchor_indices.tolist())
    }
    block_size = _bounded_shortcut_block_size(
        requested=int(contract.shortcut_query_block_size),
        anchor_count=anchor_count,
        max_rss_bytes=int(contract.max_rss_bytes),
        phase="shortcut_float64_boundary_revalidation",
    )
    anchor_squared = np.einsum("ij,ij->i", anchors, anchors)
    exact_boundary_edge_count = 0
    near_boundary_recompute_count = 0
    non_anchor_attachment_count = 0
    minimum_certifying_margin = float("inf")
    maximum_certifying_distance = 0.0
    witness_digest = hashlib.sha256()
    comparison_guard = max(
        64.0 * np.finfo(np.float64).eps * max(1.0, abs(eps)), 1.0e-15
    )
    for offset in range(0, n_samples, block_size):
        stop = min(n_samples, offset + block_size)
        block = np.asarray(vectors[offset:stop], dtype=np.float64)
        block_squared = np.einsum("ij,ij->i", block, block)
        squared = (
            block_squared[:, None]
            + anchor_squared[None, :]
            - 2.0 * np.matmul(block, anchors.T)
        )
        np.maximum(squared, 0.0, out=squared)
        distances = np.sqrt(squared, out=squared)
        near = np.abs(distances - eps) <= comparison_guard
        if bool(np.any(near)):
            for row, column in np.argwhere(near).tolist():
                distances[int(row), int(column)] = float(
                    np.linalg.norm(block[int(row)] - anchors[int(column)])
                )
            near_boundary_recompute_count += int(np.count_nonzero(near))
        within = distances <= eps
        for sample, column in anchor_column_by_sample.items():
            if offset <= sample < stop:
                local = sample - offset
                if not bool(within[local, column]) or distances[local, column] != 0.0:
                    raise ExternalMemoryDBSCANError(
                        "float64 boundary pass lost an anchor self-neighbor"
                    )
                within[local, column] = False
        counts = np.sum(within, axis=1, dtype=np.uint32)
        if not np.array_equal(counts, np.asarray(lower[offset:stop])):
            raise ExternalMemoryDBSCANError(
                "FLOAT_BOUNDARY_UNCERTAIN: sklearn/float64 anchor counts differ"
            )
        if bool(np.any(counts < needed_other_neighbors)):
            raise ExternalMemoryDBSCANError(
                "float64 boundary pass cannot certify min_samples including self"
            )
        non_anchor = np.ones(stop - offset, dtype=np.bool_)
        for sample in anchor_column_by_sample:
            if offset <= sample < stop:
                non_anchor[sample - offset] = False
        if bool(np.any(counts[non_anchor] < 1)):
            raise ExternalMemoryDBSCANError(
                "float64 boundary pass found an unattached non-anchor"
            )
        non_anchor_attachment_count += int(np.count_nonzero(non_anchor))
        if needed_other_neighbors:
            candidates = np.where(within, distances, np.inf)
            kth = np.partition(
                candidates, needed_other_neighbors - 1, axis=1
            )[:, needed_other_neighbors - 1]
            if not bool(np.isfinite(kth).all()) or bool(np.any(kth > eps)):
                raise ExternalMemoryDBSCANError(
                    "float64 certifying edge escaped inclusive epsilon"
                )
            minimum_certifying_margin = min(
                minimum_certifying_margin, float(np.min(eps - kth))
            )
            maximum_certifying_distance = max(
                maximum_certifying_distance, float(np.max(kth))
            )
            exact_boundary_edge_count += int(np.count_nonzero(kth == eps))
            witness_digest.update(np.asarray(kth, dtype=np.float64).tobytes())
        witness_digest.update(np.asarray(counts, dtype=np.uint32).tobytes())
        _check_rss(
            int(contract.max_rss_bytes),
            phase="shortcut_float64_boundary_revalidation",
        )
    del lower

    expected_edges = {
        (int(source), int(target)) for source, target in anchor_edges.tolist()
    }
    observed_edge_count = 0
    anchor_exact_boundary_edge_count = 0
    for source in range(anchor_count):
        distances = np.linalg.norm(anchors - anchors[source], axis=1)
        if distances[source] != 0.0:
            raise ExternalMemoryDBSCANError("float64 anchor graph lost self")
        for target in np.flatnonzero(distances <= eps).tolist():
            if source >= int(target):
                continue
            edge = (source, int(target))
            if edge not in expected_edges:
                raise ExternalMemoryDBSCANError(
                    "FLOAT_BOUNDARY_UNCERTAIN: float64 added an anchor edge"
                )
            observed_edge_count += 1
            anchor_exact_boundary_edge_count += int(distances[int(target)] == eps)
    if observed_edge_count != len(expected_edges):
        raise ExternalMemoryDBSCANError(
            "FLOAT_BOUNDARY_UNCERTAIN: float64 removed an anchor edge"
        )

    scientific_identity_sha = _stable_hash(identity)
    boundary_path = root / "boundary_certificate.json"
    boundary = {
        "schema_version": BOUNDARY_CERTIFICATE_SCHEMA_VERSION,
        "status": "PASS",
        "scientific_identity_sha256": scientific_identity_sha,
        "input_rows": n_samples,
        "input_features": n_features,
        "source_dtype": str(vectors.dtype),
        "recheck_dtype": "float64",
        "metric": "euclidean",
        "comparison": "distance <= eps",
        "eps": eps,
        "float64_revalidation_complete": True,
        "float64_revalidated_row_count": n_samples,
        "sklearn_anchor_count_witness_exactly_matched": True,
        "near_boundary_recompute_guard": comparison_guard,
        "near_boundary_direct_norm_recompute_count": near_boundary_recompute_count,
        "minimum_margin_to_eps_among_certifying_edges": (
            None
            if needed_other_neighbors == 0
            else minimum_certifying_margin
        ),
        "maximum_certifying_edge_distance": (
            None if needed_other_neighbors == 0 else maximum_certifying_distance
        ),
        "count_certifying_edges_exactly_at_eps": exact_boundary_edge_count,
        "count_anchor_edges_exactly_at_eps": anchor_exact_boundary_edge_count,
        "certifying_witness_stream_sha256": witness_digest.hexdigest(),
        "uncertain_edges_accepted": 0,
        "approximation_used": False,
    }
    _atomic_json(boundary_path, boundary)
    boundary_sha = _sha256_file(boundary_path)

    all_core_path = root / "all_core_certificate.json"
    all_core = {
        "schema_version": ALL_CORE_CERTIFICATE_SCHEMA_VERSION,
        "status": "PASS",
        "scientific_identity_sha256": scientific_identity_sha,
        "input_rows": n_samples,
        "eps": eps,
        "min_samples": int(contract.min_samples),
        "metric": "euclidean",
        "comparison": "distance <= eps",
        "self_neighbor_counted_exactly_once": True,
        "distinct_other_anchor_neighbors_required": needed_other_neighbors,
        "all_points_core_proven": True,
        "core_point_count": n_samples,
        "anchor_neighbor_lower_bounds_path": str(lower_path),
        "anchor_neighbor_lower_bounds_sha256": lower_sha256,
        "minimum_distinct_anchor_neighbors_excluding_self": int(
            np.min(np.load(lower_path, mmap_mode="r", allow_pickle=False))
        ),
        "boundary_certificate_path": str(boundary_path),
        "boundary_certificate_sha256": boundary_sha,
        "exact": True,
        "approximation_used": False,
    }
    _atomic_json(all_core_path, all_core)
    all_core_sha = _sha256_file(all_core_path)

    connectivity_path = root / "connectivity_certificate.json"
    connectivity = {
        "schema_version": CONNECTIVITY_CERTIFICATE_SCHEMA_VERSION,
        "status": "PASS",
        "scientific_identity_sha256": scientific_identity_sha,
        "input_rows": n_samples,
        "eps": eps,
        "metric": "euclidean",
        "comparison": "distance <= eps",
        "anchor_indices_path": str(anchor_indices_path),
        "anchor_indices_sha256": anchor_indices_sha256,
        "anchor_edges_path": str(anchor_edges_path),
        "anchor_edges_sha256": anchor_edges_sha256,
        "anchor_count": anchor_count,
        "anchor_edge_count": len(expected_edges),
        "anchor_graph_exact_connected": True,
        "non_anchor_row_count": n_samples - anchor_count,
        "non_anchor_rows_with_exact_anchor_attachment": (
            non_anchor_attachment_count
        ),
        "every_close_row_attached_to_anchor_component": True,
        "attached_or_anchor_row_count": n_samples,
        "single_epsilon_component_proven": True,
        "all_core_certificate_path": str(all_core_path),
        "all_core_certificate_sha256": all_core_sha,
        "boundary_certificate_path": str(boundary_path),
        "boundary_certificate_sha256": boundary_sha,
        "exact": True,
        "approximation_used": False,
    }
    _atomic_json(connectivity_path, connectivity)
    connectivity_sha = _sha256_file(connectivity_path)

    partition_path = root / "cluster_partition.json"
    partition = {
        "schema_version": CLUSTER_PARTITION_SCHEMA_VERSION,
        "status": "PASS",
        "scientific_identity_sha256": scientific_identity_sha,
        "input_rows": n_samples,
        "cluster_count": 1,
        "noise_count": 0,
        "core_count": n_samples,
        "canonical_cluster_labels": [0],
        "partition": "one_cluster_zero_noise",
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha256,
        "core_mask_path": str(core_path),
        "core_mask_sha256": core_sha256,
        "all_core_certificate_sha256": all_core_sha,
        "connectivity_certificate_sha256": connectivity_sha,
        "boundary_certificate_sha256": boundary_sha,
        "sklearn_partition_semantics_preserved": True,
        "approximation_used": False,
    }
    _atomic_json(partition_path, partition)
    partition_sha = _sha256_file(partition_path)
    return {
        "all_core_certificate_path": str(all_core_path),
        "all_core_certificate_sha256": all_core_sha,
        "connectivity_certificate_path": str(connectivity_path),
        "connectivity_certificate_sha256": connectivity_sha,
        "boundary_certificate_path": str(boundary_path),
        "boundary_certificate_sha256": boundary_sha,
        "cluster_partition_path": str(partition_path),
        "cluster_partition_sha256": partition_sha,
    }


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


def _progress_genesis(*, phase: str, identity_sha256: str) -> str:
    return _stable_hash(
        {
            "schema_version": PROGRESS_LEDGER_SCHEMA_VERSION,
            "phase": str(phase),
            "identity_sha256": str(identity_sha256),
            "entry": "GENESIS",
        }
    )


def _new_progress_ledger(
    *, phase: str, identity: Mapping[str, Any]
) -> dict[str, Any]:
    identity_sha = _stable_hash(identity)
    return {
        "schema_version": PROGRESS_LEDGER_SCHEMA_VERSION,
        "phase": str(phase),
        "identity_sha256": identity_sha,
        "entries": [],
        "committed_offset": 0,
        "head_sha256": _progress_genesis(
            phase=str(phase), identity_sha256=identity_sha
        ),
        "complete": False,
        "result": None,
    }


def _progress_ledger_sha256(ledger: Mapping[str, Any]) -> str:
    return _stable_hash(ledger)


def _progress_ledgers_sha256(
    ledgers: Mapping[str, Mapping[str, Any]], *, identity: Mapping[str, Any]
) -> str:
    return _stable_hash(
        {
            "schema_version": PROGRESS_LEDGER_SCHEMA_VERSION,
            "identity_sha256": _stable_hash(identity),
            "progress_ledgers": dict(ledgers),
        }
    )


def _progress_checkpoint_extra(
    ledgers: Mapping[str, Mapping[str, Any]], *, identity: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "progress_ledgers": dict(ledgers),
        "progress_ledgers_sha256": _progress_ledgers_sha256(
            ledgers, identity=identity
        ),
    }


def _append_progress_entry(
    ledger: dict[str, Any],
    *,
    start: int,
    stop: int,
    payload: Mapping[str, Any],
) -> None:
    if ledger.get("complete") is True:
        raise ExternalMemoryDBSCANError("cannot append to a complete progress ledger")
    if int(start) != int(ledger.get("committed_offset", -1)) or int(stop) <= int(
        start
    ):
        raise ExternalMemoryDBSCANError("progress ledger block is noncontiguous")
    entry_core = {
        "schema_version": PROGRESS_LEDGER_SCHEMA_VERSION,
        "phase": str(ledger.get("phase")),
        "identity_sha256": str(ledger.get("identity_sha256")),
        "start": int(start),
        "stop": int(stop),
        "previous_chain_sha256": str(ledger.get("head_sha256")),
        "payload": dict(payload),
    }
    entry = dict(entry_core)
    entry["entry_sha256"] = _stable_hash(entry_core)
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ExternalMemoryDBSCANError("progress ledger entries are invalid")
    entries.append(entry)
    ledger["committed_offset"] = int(stop)
    ledger["head_sha256"] = entry["entry_sha256"]


def _complete_progress_ledger(
    ledger: dict[str, Any], *, num_samples: int, result: Mapping[str, Any]
) -> None:
    if int(ledger.get("committed_offset", -1)) != int(num_samples):
        raise ExternalMemoryDBSCANError("cannot complete a partial progress ledger")
    ledger["complete"] = True
    ledger["result"] = dict(result)


def _validate_progress_ledger_structure(
    ledger: Mapping[str, Any],
    *,
    phase: str,
    identity: Mapping[str, Any],
    num_samples: int,
) -> None:
    required_keys = {
        "schema_version",
        "phase",
        "identity_sha256",
        "entries",
        "committed_offset",
        "head_sha256",
        "complete",
        "result",
    }
    identity_sha = _stable_hash(identity)
    if (
        set(ledger) != required_keys
        or ledger.get("schema_version") != PROGRESS_LEDGER_SCHEMA_VERSION
        or ledger.get("phase") != phase
        or ledger.get("identity_sha256") != identity_sha
        or not isinstance(ledger.get("entries"), list)
        or ledger.get("complete") not in {True, False}
    ):
        raise ExternalMemoryDBSCANError(
            f"{phase} progress ledger schema/identity mismatch"
        )
    expected_start = 0
    previous = _progress_genesis(phase=phase, identity_sha256=identity_sha)
    for entry in ledger["entries"]:
        if not isinstance(entry, Mapping):
            raise ExternalMemoryDBSCANError(
                f"{phase} progress ledger entry is invalid"
            )
        entry_core = dict(entry)
        entry_sha = entry_core.pop("entry_sha256", None)
        if (
            set(entry_core)
            != {
                "schema_version",
                "phase",
                "identity_sha256",
                "start",
                "stop",
                "previous_chain_sha256",
                "payload",
            }
            or entry_core.get("schema_version") != PROGRESS_LEDGER_SCHEMA_VERSION
            or entry_core.get("phase") != phase
            or entry_core.get("identity_sha256") != identity_sha
            or not isinstance(entry_core.get("payload"), Mapping)
            or int(entry_core.get("start", -1)) != expected_start
            or int(entry_core.get("stop", -1)) <= expected_start
            or int(entry_core.get("stop", -1)) > int(num_samples)
            or entry_core.get("previous_chain_sha256") != previous
            or entry_sha != _stable_hash(entry_core)
        ):
            raise ExternalMemoryDBSCANError(
                f"{phase} progress ledger chain mismatch"
            )
        expected_start = int(entry_core["stop"])
        previous = str(entry_sha)
    if (
        int(ledger.get("committed_offset", -1)) != expected_start
        or ledger.get("head_sha256") != previous
        or (ledger.get("complete") is True and expected_start != int(num_samples))
        or (ledger.get("complete") is True and not isinstance(ledger.get("result"), Mapping))
        or (ledger.get("complete") is False and ledger.get("result") is not None)
    ):
        raise ExternalMemoryDBSCANError(
            f"{phase} progress ledger closure mismatch"
        )


def _load_progress_ledgers(
    state: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    num_samples: int,
) -> dict[str, dict[str, Any]]:
    raw = state.get("progress_ledgers")
    if not isinstance(raw, Mapping):
        raise ExternalMemoryDBSCANError("shortcut progress ledgers are missing")
    ledgers = json.loads(json.dumps(dict(raw)))
    if state.get("progress_ledgers_sha256") != _progress_ledgers_sha256(
        ledgers, identity=identity
    ):
        raise ExternalMemoryDBSCANError("shortcut progress-ledger hash mismatch")
    for phase, ledger in ledgers.items():
        if not isinstance(phase, str) or not isinstance(ledger, Mapping):
            raise ExternalMemoryDBSCANError("shortcut progress ledger is invalid")
        _validate_progress_ledger_structure(
            ledger,
            phase=phase,
            identity=identity,
            num_samples=num_samples,
        )
    return ledgers


def _require_progress_ledger(
    ledgers: dict[str, dict[str, Any]],
    *,
    phase: str,
    identity: Mapping[str, Any],
    create: bool = False,
) -> dict[str, Any]:
    ledger = ledgers.get(phase)
    if ledger is None and create:
        ledger = _new_progress_ledger(phase=phase, identity=identity)
        ledgers[phase] = ledger
    if not isinstance(ledger, dict):
        raise ExternalMemoryDBSCANError(f"{phase} progress ledger is missing")
    return ledger


def _seed_candidate_rows(
    candidates: Sequence[tuple[float, int]],
) -> list[dict[str, Any]]:
    return [
        {"index": int(index), "squared_norm_hex": float(norm).hex()}
        for norm, index in candidates
    ]


def _parse_seed_candidate_rows(
    rows: Any, *, label: str
) -> list[tuple[float, int]]:
    if not isinstance(rows, list):
        raise ExternalMemoryDBSCANError(f"{label} is invalid")
    parsed: list[tuple[float, int]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "index",
            "squared_norm_hex",
        }:
            raise ExternalMemoryDBSCANError(f"{label} is invalid")
        try:
            parsed.append(
                (float.fromhex(str(row["squared_norm_hex"])), int(row["index"]))
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExternalMemoryDBSCANError(f"{label} is invalid") from exc
    if any(not np.isfinite(norm) for norm, _index in parsed):
        raise ExternalMemoryDBSCANError(f"{label} contains NaN/Inf")
    if parsed != sorted(parsed, key=lambda row: (row[0], row[1])):
        raise ExternalMemoryDBSCANError(f"{label} is not canonically ordered")
    return parsed


def _adaptive_seed_block_candidates(
    vectors: np.ndarray, *, start: int, stop: int, seed_count: int
) -> list[tuple[float, int]]:
    rows = np.asarray(vectors[start:stop], dtype=np.float64)
    norms = np.einsum("ij,ij->i", rows, rows, dtype=np.float64)
    if not np.isfinite(norms).all():
        raise ExternalMemoryDBSCANError("adaptive seed norm scan found NaN/Inf")
    indices = np.arange(start, stop, dtype=np.intp)
    order = np.lexsort((indices, norms))[: int(seed_count)]
    result = [
        (float(norms[position]), int(indices[position]))
        for position in order.tolist()
    ]
    del rows, norms, indices, order
    return result


def _fold_seed_ledger(
    ledger: Mapping[str, Any], *, seed_count: int
) -> list[tuple[float, int]]:
    candidates: list[tuple[float, int]] = []
    for entry in ledger["entries"]:
        payload = entry["payload"]
        local = _parse_seed_candidate_rows(
            payload.get("local_seed_candidates"),
            label="adaptive seed progress payload",
        )
        if int(payload.get("row_count", -1)) != int(entry["stop"]) - int(
            entry["start"]
        ) or (
            len(local)
            != min(int(seed_count), int(entry["stop"]) - int(entry["start"]))
            or len({index for _norm, index in local}) != len(local)
            or any(
                index < int(entry["start"]) or index >= int(entry["stop"])
                for _norm, index in local
            )
        ):
            raise ExternalMemoryDBSCANError(
                "adaptive seed progress row count mismatch"
            )
        candidates = sorted(candidates + local, key=lambda row: (row[0], row[1]))[
            : int(seed_count)
        ]
    return candidates


def _adaptive_failure_block(
    *,
    model: Any,
    vectors: np.ndarray,
    start: int,
    stop: int,
    seed_local_by_global: Mapping[int, int],
    min_samples: int,
) -> list[int]:
    neighborhoods = model.radius_neighbors(
        vectors[start:stop], return_distance=False
    )
    failures: list[int] = []
    for local_row, raw in enumerate(neighborhoods):
        global_index = int(start) + local_row
        values = np.asarray(raw, dtype=np.intp)
        if len(values) != len(np.unique(values)):
            raise ExternalMemoryDBSCANError(
                "adaptive failure scan returned duplicate seed indices"
            )
        own_seed = seed_local_by_global.get(global_index)
        if own_seed is not None and own_seed not in values:
            raise ExternalMemoryDBSCANError(
                "adaptive failure scan omitted explicit seed self sample"
            )
        lower = len(values) - int(own_seed is not None)
        if lower < int(min_samples) - 1 or (own_seed is None and lower < 1):
            failures.append(global_index)
    del neighborhoods
    return failures


def _fold_failure_ledger(ledger: Mapping[str, Any]) -> list[int]:
    failures: list[int] = []
    for entry in ledger["entries"]:
        payload = entry["payload"]
        block_failures = payload.get("failure_indices")
        if not isinstance(block_failures, list):
            raise ExternalMemoryDBSCANError(
                "adaptive failure progress payload is invalid"
            )
        parsed = [int(value) for value in block_failures]
        if (
            parsed != sorted(set(parsed))
            or any(
                value < int(entry["start"]) or value >= int(entry["stop"])
                for value in parsed
            )
            or _sample_indices_sha256(parsed)
            != payload.get("failure_indices_sha256")
            or int(payload.get("row_count", -1))
            != int(entry["stop"]) - int(entry["start"])
        ):
            raise ExternalMemoryDBSCANError(
                "adaptive failure progress payload mismatch"
            )
        failures.extend(parsed)
    if failures != sorted(set(failures)):
        raise ExternalMemoryDBSCANError(
            "adaptive failure progress ledger is noncanonical"
        )
    return failures


def _uint32_values_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<u4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _anchor_lower_block(
    *,
    model: Any,
    vectors: np.ndarray,
    start: int,
    stop: int,
    anchor_local_by_global: Mapping[int, int],
    min_samples: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    neighborhoods = model.radius_neighbors(
        vectors[start:stop], return_distance=False
    )
    block_lower = np.empty(int(stop) - int(start), dtype=np.uint32)
    first_failure: dict[str, Any] | None = None
    for local_row, raw in enumerate(neighborhoods):
        global_index = int(start) + local_row
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
        if (
            count_excluding_self < int(min_samples) - 1
            or (own_anchor is None and count_excluding_self < 1)
        ) and first_failure is None:
            first_failure = {
                "sample_index": int(global_index),
                "distinct_anchor_neighbors_excluding_self": int(
                    count_excluding_self
                ),
                "required_for_core": int(min_samples) - 1,
                "required_for_anchor_attachment_if_non_anchor": 1,
                "sample_is_anchor": own_anchor is not None,
            }
    del neighborhoods
    return block_lower, first_failure


def _lower_block_payload(
    values: np.ndarray,
    *,
    start: int,
    anchor_local_by_global: Mapping[int, int],
) -> dict[str, Any]:
    minimum = int(np.min(values))
    anchor_positions = sorted(
        global_index - int(start)
        for global_index in anchor_local_by_global
        if int(start) <= global_index < int(start) + len(values)
    )
    non_anchor_minima: list[int] = []
    cursor = 0
    for position in anchor_positions:
        if cursor < position:
            non_anchor_minima.append(int(np.min(values[cursor:position])))
        cursor = position + 1
    if cursor < len(values):
        non_anchor_minima.append(int(np.min(values[cursor:])))
    return {
        "row_count": int(len(values)),
        "lower_bounds_uint32_le_sha256": _uint32_values_sha256(values),
        "minimum_distinct_anchor_neighbors_excluding_self": minimum,
        "minimum_non_anchor_anchor_neighbors": (
            min(non_anchor_minima) if non_anchor_minima else None
        ),
    }


def _assert_active_progress_offset(
    state: Mapping[str, Any], ledger: Mapping[str, Any], *, phase: str
) -> None:
    if (
        state.get("phase") != phase
        or int(state.get("next_offset", -1))
        != int(ledger.get("committed_offset", -2))
    ):
        raise ExternalMemoryDBSCANError(
            f"{phase} checkpoint offset/progress-ledger mismatch"
        )


def _replay_seed_progress(
    *,
    vectors: np.ndarray,
    ledger: Mapping[str, Any],
    seed_count: int,
    max_rss_bytes: int,
) -> tuple[list[tuple[float, int]], int]:
    peak = _rss_bytes()
    candidates: list[tuple[float, int]] = []
    for entry in ledger["entries"]:
        start, stop = int(entry["start"]), int(entry["stop"])
        reservation = (stop - start) * (
            2 * int(vectors.shape[1]) * np.dtype(np.float64).itemsize
            + int(vectors.shape[1]) * int(vectors.dtype.itemsize)
            + 4096
        )
        _check_rss(
            int(max_rss_bytes),
            phase="adaptive_seed_scan.resume_replay",
            reserved_bytes=reservation,
        )
        local = _adaptive_seed_block_candidates(
            vectors, start=start, stop=stop, seed_count=int(seed_count)
        )
        expected_payload = {
            "row_count": stop - start,
            "local_seed_candidates": _seed_candidate_rows(local),
        }
        if entry.get("payload") != expected_payload:
            raise ExternalMemoryDBSCANError(
                "adaptive seed resume replay mismatch"
            )
        candidates = sorted(candidates + local, key=lambda row: (row[0], row[1]))[
            : int(seed_count)
        ]
        peak = max(
            peak,
            _check_rss(
                int(max_rss_bytes), phase="adaptive_seed_scan.resume_replay"
            ),
        )
    if ledger.get("complete") is True:
        expected_result = {
            "seed_candidates": _seed_candidate_rows(candidates),
            "seed_indices_sha256": _sample_indices_sha256(
                [index for _norm, index in candidates]
            ),
        }
        if ledger.get("result") != expected_result:
            raise ExternalMemoryDBSCANError(
                "adaptive seed complete-ledger result mismatch"
            )
    return candidates, peak


def _replay_failure_progress(
    *,
    vectors: np.ndarray,
    model: Any,
    seed_indices: Sequence[int],
    ledger: Mapping[str, Any],
    min_samples: int,
    max_rss_bytes: int,
) -> tuple[list[int], int]:
    peak = _rss_bytes()
    failures: list[int] = []
    seed_hash = _sample_indices_sha256(seed_indices)
    seed_local_by_global = {
        int(global_index): local_index
        for local_index, global_index in enumerate(seed_indices)
    }
    for entry in ledger["entries"]:
        start, stop = int(entry["start"]), int(entry["stop"])
        _check_rss(
            int(max_rss_bytes),
            phase="adaptive_failure_scan.resume_replay",
            reserved_bytes=_shortcut_query_reservation_bytes(
                stop - start, len(seed_indices)
            ),
        )
        block_failures = _adaptive_failure_block(
            model=model,
            vectors=vectors,
            start=start,
            stop=stop,
            seed_local_by_global=seed_local_by_global,
            min_samples=int(min_samples),
        )
        expected_payload = {
            "row_count": stop - start,
            "seed_indices_sha256": seed_hash,
            "failure_indices": block_failures,
            "failure_indices_sha256": _sample_indices_sha256(block_failures),
        }
        if entry.get("payload") != expected_payload:
            raise ExternalMemoryDBSCANError(
                "adaptive failure resume replay mismatch"
            )
        failures.extend(block_failures)
        peak = max(
            peak,
            _check_rss(
                int(max_rss_bytes), phase="adaptive_failure_scan.resume_replay"
            ),
        )
    if failures != sorted(set(failures)):
        raise ExternalMemoryDBSCANError(
            "adaptive failure replay produced noncanonical indices"
        )
    if ledger.get("complete") is True:
        expected_result = {
            "first_pass_complete": True,
            "failure_indices": failures,
            "failure_indices_sha256": _sample_indices_sha256(failures),
        }
        if ledger.get("result") != expected_result:
            raise ExternalMemoryDBSCANError(
                "adaptive failure complete-ledger result mismatch"
            )
    return failures, peak


def _replay_lower_progress(
    *,
    vectors: np.ndarray,
    lower: np.ndarray,
    model: Any,
    anchor_indices: np.ndarray,
    ledger: Mapping[str, Any],
    min_samples: int,
    max_rss_bytes: int,
) -> tuple[int | None, int | None, int]:
    anchor_local_by_global = {
        int(global_index): local_index
        for local_index, global_index in enumerate(anchor_indices.tolist())
    }
    minimum: int | None = None
    minimum_non_anchor: int | None = None
    peak = _rss_bytes()
    for entry in ledger["entries"]:
        start, stop = int(entry["start"]), int(entry["stop"])
        _check_rss(
            int(max_rss_bytes),
            phase="shortcut_anchor_scan.resume_replay",
            reserved_bytes=_shortcut_query_reservation_bytes(
                stop - start, len(anchor_indices)
            ),
        )
        replayed, first_failure = _anchor_lower_block(
            model=model,
            vectors=vectors,
            start=start,
            stop=stop,
            anchor_local_by_global=anchor_local_by_global,
            min_samples=int(min_samples),
        )
        if first_failure is not None:
            raise ExternalMemoryDBSCANError(
                "shortcut lower-bound resume replay found a committed failure"
            )
        expected_payload = _lower_block_payload(
            replayed,
            start=start,
            anchor_local_by_global=anchor_local_by_global,
        )
        if entry.get("payload") != expected_payload or not np.array_equal(
            np.asarray(lower[start:stop]), replayed
        ):
            raise ExternalMemoryDBSCANError(
                "shortcut lower-bound resume replay mismatch"
            )
        block_minimum = int(expected_payload[
            "minimum_distinct_anchor_neighbors_excluding_self"
        ])
        minimum = block_minimum if minimum is None else min(minimum, block_minimum)
        block_non_anchor = expected_payload[
            "minimum_non_anchor_anchor_neighbors"
        ]
        if block_non_anchor is not None:
            minimum_non_anchor = (
                int(block_non_anchor)
                if minimum_non_anchor is None
                else min(minimum_non_anchor, int(block_non_anchor))
            )
        peak = max(
            peak,
            _check_rss(
                int(max_rss_bytes), phase="shortcut_anchor_scan.resume_replay"
            ),
        )
    return minimum, minimum_non_anchor, peak


def _validate_complete_lower_witness(
    *,
    lower: np.ndarray,
    anchor_indices: np.ndarray,
    ledger: Mapping[str, Any],
    num_samples: int,
    min_samples: int,
) -> tuple[int, int | None]:
    """Validate every lower slot and the complete proof domain before PASS."""

    if ledger.get("complete") is not True or int(
        ledger.get("committed_offset", -1)
    ) != int(num_samples):
        raise ExternalMemoryDBSCANError(
            "shortcut lower-bound progress ledger is incomplete"
        )
    anchor_local_by_global = {
        int(global_index): local_index
        for local_index, global_index in enumerate(anchor_indices.tolist())
    }
    minimum: int | None = None
    minimum_non_anchor: int | None = None
    for entry in ledger["entries"]:
        start, stop = int(entry["start"]), int(entry["stop"])
        values = np.asarray(lower[start:stop])
        payload = _lower_block_payload(
            values,
            start=start,
            anchor_local_by_global=anchor_local_by_global,
        )
        if entry.get("payload") != payload:
            raise ExternalMemoryDBSCANError(
                "shortcut final lower-bound ledger/array mismatch"
            )
        block_minimum = int(
            payload["minimum_distinct_anchor_neighbors_excluding_self"]
        )
        minimum = block_minimum if minimum is None else min(minimum, block_minimum)
        block_non_anchor = payload["minimum_non_anchor_anchor_neighbors"]
        if block_non_anchor is not None:
            minimum_non_anchor = (
                int(block_non_anchor)
                if minimum_non_anchor is None
                else min(minimum_non_anchor, int(block_non_anchor))
            )
    if minimum is None or minimum < int(min_samples) - 1:
        raise ExternalMemoryDBSCANError(
            "shortcut final lower-bound core coverage failed"
        )
    if int(num_samples) > len(anchor_indices) and (
        minimum_non_anchor is None or minimum_non_anchor < 1
    ):
        raise ExternalMemoryDBSCANError(
            "shortcut final non-anchor attachment coverage failed"
        )
    expected_result = {
        "full_scan_complete": True,
        "minimum_distinct_anchor_neighbors_excluding_self": int(minimum),
        "minimum_non_anchor_anchor_neighbors": minimum_non_anchor,
        "all_rows_core_lower_bound_pass": True,
        "all_non_anchors_attached": True,
    }
    if ledger.get("result") != expected_result:
        raise ExternalMemoryDBSCANError(
            "shortcut lower-bound complete-ledger result mismatch"
        )
    return int(minimum), minimum_non_anchor


def _validate_constant_output(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    expected: int | bool,
    label: str,
    block_size: int = 1_000_000,
) -> None:
    value = _open_npy_memmap(path, mode="r")
    _validate_partial(value, shape=shape, dtype=dtype, label=label)
    for start in range(0, int(shape[0]), int(block_size)):
        if not np.all(value[start : min(shape[0], start + block_size)] == expected):
            raise ExternalMemoryDBSCANError(f"{label} constant-value mismatch")
    del value


def _write_progress_ledger_artifact(
    *,
    path: Path,
    ledgers: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Any],
    required_phases: Sequence[str],
    num_samples: int,
) -> str:
    if set(ledgers) != set(required_phases):
        raise ExternalMemoryDBSCANError(
            "shortcut progress ledger phase set mismatch"
        )
    for phase in required_phases:
        ledger = ledgers.get(phase)
        if not isinstance(ledger, Mapping):
            raise ExternalMemoryDBSCANError(
                f"required progress ledger is missing: {phase}"
            )
        _validate_progress_ledger_structure(
            ledger,
            phase=phase,
            identity=identity,
            num_samples=num_samples,
        )
        if ledger.get("complete") is not True:
            raise ExternalMemoryDBSCANError(
                f"required progress ledger is incomplete: {phase}"
            )
    payload = {
        "schema_version": PROGRESS_LEDGER_SCHEMA_VERSION,
        "scientific_identity_sha256": _stable_hash(identity),
        "required_phases": list(required_phases),
        "progress_ledgers": dict(ledgers),
        "progress_ledgers_sha256": _progress_ledgers_sha256(
            ledgers, identity=identity
        ),
        "all_required_prefixes_complete": True,
        "created_at": _utc_now(),
    }
    _atomic_json(path, payload)
    return _sha256_file(path)


def _validate_progress_ledger_artifact(
    *,
    path: Path,
    expected_sha256: str,
    root: Path,
    identity: Mapping[str, Any],
    num_samples: int,
    required_phases: Sequence[str],
) -> dict[str, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if resolved.parent != root or _sha256_file(resolved) != str(expected_sha256):
        raise ExternalMemoryDBSCANError(
            "shortcut progress-ledger artifact closure mismatch"
        )
    payload = _load_object(resolved)
    ledgers_raw = payload.get("progress_ledgers")
    if (
        payload.get("schema_version") != PROGRESS_LEDGER_SCHEMA_VERSION
        or payload.get("scientific_identity_sha256") != _stable_hash(identity)
        or payload.get("required_phases") != list(required_phases)
        or payload.get("all_required_prefixes_complete") is not True
        or not isinstance(ledgers_raw, Mapping)
    ):
        raise ExternalMemoryDBSCANError(
            "shortcut progress-ledger artifact identity mismatch"
        )
    ledgers = json.loads(json.dumps(dict(ledgers_raw)))
    if set(ledgers) != set(required_phases):
        raise ExternalMemoryDBSCANError(
            "shortcut progress-ledger artifact phase set mismatch"
        )
    if payload.get("progress_ledgers_sha256") != _progress_ledgers_sha256(
        ledgers, identity=identity
    ):
        raise ExternalMemoryDBSCANError(
            "shortcut progress-ledger artifact hash mismatch"
        )
    for phase in required_phases:
        ledger = ledgers.get(phase)
        if not isinstance(ledger, Mapping):
            raise ExternalMemoryDBSCANError(
                f"shortcut progress-ledger phase is missing: {phase}"
            )
        _validate_progress_ledger_structure(
            ledger,
            phase=phase,
            identity=identity,
            num_samples=num_samples,
        )
        if ledger.get("complete") is not True:
            raise ExternalMemoryDBSCANError(
                f"shortcut progress-ledger phase is incomplete: {phase}"
            )
    return ledgers


def _validate_adaptive_selection_manifest(
    *,
    path: Path,
    expected_sha256: str,
    root: Path,
    identity: Mapping[str, Any],
    progress_ledgers: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if resolved.parent != root or _sha256_file(resolved) != str(expected_sha256):
        raise ExternalMemoryDBSCANError(
            "adaptive anchor-selection manifest closure mismatch"
        )
    manifest = _load_object(resolved)
    selection = manifest.get("selection_identity")
    if (
        manifest.get("schema_version") != ADAPTIVE_SELECTION_SCHEMA_VERSION
        or manifest.get("run_complete") is not True
        or not isinstance(selection, Mapping)
        or manifest.get("selection_identity_sha256") != _stable_hash(selection)
        or selection.get("vectors_path") != identity.get("vectors_path")
        or selection.get("vectors_sha256") != identity.get("vectors_sha256")
        or selection.get("vectors_stat_identity")
        != identity.get("vectors_stat_identity")
        or selection.get("vectors_dtype") != identity.get("vectors_dtype")
        or selection.get("vectors_shape") != identity.get("vectors_shape")
        or selection.get("selection_contract") != identity.get("shortcut_contract")
        or selection.get("seed_selection_algorithm")
        != "global_k_minimum_squared_l2_float64_tie_sample_index_v1"
        or selection.get("failure_selection_rule")
        != "all_first_pass_insufficient_seed_lower_bound_indices_v1"
        or selection.get("anchor_selection_rule")
        != "sorted_unique_union_of_seed_and_all_failure_indices_v1"
        or not isinstance(
            selection.get("adaptive_seed_progress_ledger_sha256"), str
        )
        or not isinstance(
            selection.get("adaptive_failure_progress_ledger_sha256"), str
        )
    ):
        raise ExternalMemoryDBSCANError(
            "adaptive anchor-selection scientific identity mismatch"
        )
    seed_indices = [int(value) for value in selection.get("seed_indices") or []]
    seed_norm_hex = [str(value) for value in selection.get("seed_squared_norm_hex") or []]
    seed_count = int(identity["shortcut_contract"]["seed_count"])
    if (
        len(seed_indices) != seed_count
        or len(seed_norm_hex) != seed_count
        or len(set(seed_indices)) != seed_count
        or _sample_indices_sha256(seed_indices)
        != selection.get("seed_indices_sha256")
    ):
        raise ExternalMemoryDBSCANError("adaptive seed identity mismatch")
    seed_order = sorted(
        zip((float.fromhex(value) for value in seed_norm_hex), seed_indices),
        key=lambda row: (row[0], row[1]),
    )
    if [index for _norm, index in seed_order] != seed_indices:
        raise ExternalMemoryDBSCANError("adaptive seeds are not canonically ordered")

    failure_path = Path(str(selection.get("failure_indices_path") or "")).resolve(
        strict=True
    )
    anchor_indices_path = Path(
        str(selection.get("anchor_indices_path") or "")
    ).resolve(strict=True)
    anchor_rows_path = Path(str(selection.get("anchor_rows_path") or "")).resolve(
        strict=True
    )
    for artifact, hash_field in (
        (failure_path, "failure_indices_sha256"),
        (anchor_indices_path, "anchor_indices_sha256"),
        (anchor_rows_path, "anchor_rows_sha256"),
    ):
        if artifact.parent != root or _sha256_file(artifact) != selection.get(
            hash_field
        ):
            raise ExternalMemoryDBSCANError(
                f"adaptive selection artifact mismatch: {hash_field}"
            )
    failures = np.load(failure_path, allow_pickle=False)
    anchors = np.load(anchor_indices_path, allow_pickle=False)
    anchor_rows = np.load(anchor_rows_path, allow_pickle=False)
    n_samples, n_features = (int(value) for value in identity["vectors_shape"])
    if (
        failures.dtype != np.dtype(np.intp)
        or failures.ndim != 1
        or len(failures) != int(selection.get("failure_count", -1))
        or len(failures) > int(identity["shortcut_contract"]["failure_cap"])
        or not np.array_equal(failures, np.unique(failures))
        or np.any(failures < 0)
        or np.any(failures >= n_samples)
        or _sample_indices_sha256(failures)
        != selection.get("failure_index_list_sha256")
        or anchors.dtype != np.dtype(np.intp)
        or anchors.ndim != 1
        or not np.array_equal(anchors, np.unique(anchors))
        or np.any(anchors < 0)
        or np.any(anchors >= n_samples)
        or _sample_indices_sha256(anchors)
        != selection.get("selected_anchor_indices_sha256")
        or not np.array_equal(
            anchors,
            np.asarray(sorted(set(seed_indices).union(failures.tolist())), dtype=np.intp),
        )
        or anchor_rows.shape != (len(anchors), n_features)
        or str(anchor_rows.dtype) != str(identity["vectors_dtype"])
    ):
        raise ExternalMemoryDBSCANError("adaptive selection array schema mismatch")
    source = _open_npy_memmap(
        Path(str(identity["vectors_path"])).resolve(strict=True), mode="r"
    )
    if not np.array_equal(anchor_rows, np.asarray(source[anchors])):
        raise ExternalMemoryDBSCANError(
            "adaptive anchor rows do not match the promoted source vectors"
        )
    actual_seed_rows = np.asarray(source[np.asarray(seed_indices, dtype=np.intp)])
    actual_seed_rows64 = np.asarray(actual_seed_rows, dtype=np.float64)
    actual_seed_norm_hex = [
        float(value).hex()
        for value in np.einsum(
            "ij,ij->i", actual_seed_rows64, actual_seed_rows64, dtype=np.float64
        ).tolist()
    ]
    if actual_seed_norm_hex != seed_norm_hex:
        raise ExternalMemoryDBSCANError(
            "adaptive seed norms do not match the promoted source vectors"
        )
    if progress_ledgers is not None:
        seed_ledger = progress_ledgers.get("adaptive_seed_scan")
        failure_ledger = progress_ledgers.get("adaptive_failure_scan")
        if (
            not isinstance(seed_ledger, Mapping)
            or not isinstance(failure_ledger, Mapping)
            or seed_ledger.get("complete") is not True
            or failure_ledger.get("complete") is not True
            or selection.get("adaptive_seed_progress_ledger_sha256")
            != _progress_ledger_sha256(seed_ledger)
            or selection.get("adaptive_failure_progress_ledger_sha256")
            != _progress_ledger_sha256(failure_ledger)
        ):
            raise ExternalMemoryDBSCANError(
                "adaptive selection/progress-ledger mismatch"
            )
        ledger_seed_candidates = _fold_seed_ledger(
            seed_ledger, seed_count=seed_count
        )
        ledger_failures = _fold_failure_ledger(failure_ledger)
        expected_seed_result = {
            "seed_candidates": _seed_candidate_rows(ledger_seed_candidates),
            "seed_indices_sha256": _sample_indices_sha256(
                [index for _norm, index in ledger_seed_candidates]
            ),
        }
        expected_failure_result = {
            "first_pass_complete": True,
            "failure_indices": ledger_failures,
            "failure_indices_sha256": _sample_indices_sha256(ledger_failures),
        }
        if (
            seed_ledger.get("result") != expected_seed_result
            or failure_ledger.get("result") != expected_failure_result
            or [index for _norm, index in ledger_seed_candidates] != seed_indices
            or [float(norm).hex() for norm, _index in ledger_seed_candidates]
            != seed_norm_hex
            or ledger_failures != failures.tolist()
        ):
            raise ExternalMemoryDBSCANError(
                "adaptive selection/progress-ledger result mismatch"
            )
    del source, actual_seed_rows, actual_seed_rows64
    return np.asarray(anchors, dtype=np.intp), dict(manifest)


def _resolve_adaptive_anchor_selection(
    *,
    vectors: np.ndarray,
    root: Path,
    state_path: Path,
    identity: Mapping[str, Any],
    contract: ExternalDBSCANContract,
    sklearn_version: str,
    peak_rss_bytes: int,
    resume_replay_required: bool,
) -> tuple[np.ndarray, dict[str, Any], int]:
    """Select deterministic seeds, collect every failure, then freeze anchors."""

    selection_path = root / "adaptive_anchor_selection.json"
    state = _load_checkpoint(state_path)
    phase = str(state.get("phase"))
    n_samples, n_features = (int(vectors.shape[0]), int(vectors.shape[1]))
    seed_count = int(contract.shortcut_seed_count)
    if seed_count > n_samples:
        raise ExternalMemoryDBSCANError(
            "adaptive seed count exceeds the number of samples"
        )
    ledgers = _load_progress_ledgers(
        state, identity=identity, num_samples=n_samples
    )

    seed_ledger = _require_progress_ledger(
        ledgers, phase="adaptive_seed_scan", identity=identity
    )
    if phase == "adaptive_seed_scan":
        _assert_active_progress_offset(
            state, seed_ledger, phase="adaptive_seed_scan"
        )
    candidates = _fold_seed_ledger(seed_ledger, seed_count=seed_count)
    peak = max(int(peak_rss_bytes), _rss_bytes())
    if resume_replay_required and seed_ledger["entries"]:
        replayed_candidates, replay_peak = _replay_seed_progress(
            vectors=vectors,
            ledger=seed_ledger,
            seed_count=seed_count,
            max_rss_bytes=int(contract.max_rss_bytes),
        )
        if replayed_candidates != candidates:
            raise ExternalMemoryDBSCANError(
                "adaptive seed ledger fold/replay mismatch"
            )
        peak = max(peak, replay_peak)
    expected_seed_result = {
        "seed_candidates": _seed_candidate_rows(candidates),
        "seed_indices_sha256": _sample_indices_sha256(
            [index for _norm, index in candidates]
        ),
    }
    if seed_ledger.get("complete") is True and seed_ledger.get(
        "result"
    ) != expected_seed_result:
        raise ExternalMemoryDBSCANError(
            "adaptive seed complete-ledger identity mismatch"
        )
    if phase == "adaptive_seed_scan":
        checkpoint_candidates = _parse_seed_candidate_rows(
            state.get("adaptive_seed_candidates") or [],
            label="adaptive seed checkpoint candidates",
        )
        if checkpoint_candidates != candidates:
            raise ExternalMemoryDBSCANError(
                "adaptive seed checkpoint/ledger mismatch"
            )

    failure_ledger = ledgers.get("adaptive_failure_scan")
    seed_model: Any | None = None
    failures: list[int] = []
    if failure_ledger is not None:
        if seed_ledger.get("complete") is not True:
            raise ExternalMemoryDBSCANError(
                "adaptive failure scan started before seed completion"
            )
        _validate_progress_ledger_structure(
            failure_ledger,
            phase="adaptive_failure_scan",
            identity=identity,
            num_samples=n_samples,
        )
        if phase == "adaptive_failure_scan":
            _assert_active_progress_offset(
                state, failure_ledger, phase="adaptive_failure_scan"
            )
        seed_indices_for_replay = [index for _norm, index in candidates]
        seed_vectors_for_replay = np.asarray(
            vectors[np.asarray(seed_indices_for_replay, dtype=np.intp)]
        )
        seed_model, seed_sklearn_version = _fit_anchor_neighbors(
            seed_vectors_for_replay, eps=float(contract.eps)
        )
        if seed_sklearn_version != sklearn_version:
            raise ExternalMemoryDBSCANError(
                "adaptive seed/full sklearn versions differ"
            )
        if resume_replay_required and failure_ledger["entries"]:
            failures, replay_peak = _replay_failure_progress(
                vectors=vectors,
                model=seed_model,
                seed_indices=seed_indices_for_replay,
                ledger=failure_ledger,
                min_samples=int(contract.min_samples),
                max_rss_bytes=int(contract.max_rss_bytes),
            )
            peak = max(peak, replay_peak)
        else:
            failures = _fold_failure_ledger(failure_ledger)
        if (
            len(failures) > int(contract.shortcut_failure_cap)
            and phase != "shortcut_blocked"
        ):
            raise ExternalMemoryDBSCANError(
                "adaptive failure progress ledger exceeds the frozen cap"
            )
        expected_failure_result = {
            "first_pass_complete": True,
            "failure_indices": failures,
            "failure_indices_sha256": _sample_indices_sha256(failures),
        }
        if failure_ledger.get("complete") is True and failure_ledger.get(
            "result"
        ) != expected_failure_result:
            raise ExternalMemoryDBSCANError(
                "adaptive failure complete-ledger identity mismatch"
            )
        if phase == "adaptive_failure_scan" or (
            phase == "shortcut_blocked"
            and "adaptive_failure_indices" in state
        ):
            checkpoint_failures = [
                int(value) for value in state.get("adaptive_failure_indices") or []
            ]
            if (
                checkpoint_failures != failures
                or _sample_indices_sha256(checkpoint_failures)
                != state.get("adaptive_failure_indices_sha256")
            ):
                raise ExternalMemoryDBSCANError(
                    "adaptive failure checkpoint/ledger mismatch"
                )

    if phase == "shortcut_blocked":
        failure_path = Path(str(state.get("shortcut_failure_path") or ""))
        if (
            failure_path.parent != root
            or not failure_path.is_file()
            or _sha256_file(failure_path)
            != state.get("shortcut_failure_sha256")
        ):
            raise ExternalMemoryDBSCANError(
                "blocked adaptive-selection failure artifact mismatch"
            )
        failure = _load_object(failure_path)
        if failure.get("reason") == "adaptive_failure_cap_exceeded" and (
            not isinstance(failure_ledger, Mapping)
            or int(state.get("next_offset", -1))
            != int(failure_ledger.get("committed_offset", -2))
        ):
            raise ExternalMemoryDBSCANError(
                "blocked adaptive failure checkpoint/ledger offset mismatch"
            )
        raise ExternalMemoryDBSCANError(
            "EXACT_DBSCAN_COMPLEXITY_BLOCKED:"
            f"reason={failure.get('reason')}"
        )
    if selection_path.exists():
        expected_sha = str(state.get("adaptive_selection_manifest_sha256") or "")
        if not expected_sha:
            if not (
                resume_replay_required
                and phase == "adaptive_failure_scan"
                and seed_ledger.get("complete") is True
                and isinstance(failure_ledger, Mapping)
                and failure_ledger.get("complete") is True
                and int(failure_ledger.get("committed_offset", -1)) == n_samples
            ):
                raise ExternalMemoryDBSCANError(
                    "adaptive selection manifest is not checkpoint-bound"
                )
            # Crash window: the selection rename completed after the full
            # first-pass ledger checkpoint but before the transition
            # checkpoint.  The source-derived seed/failure prefixes were just
            # replayed above, so validate the complete manifest semantically
            # and bind its observed bytes in the next atomic checkpoint.
            expected_sha = _sha256_file(selection_path)
        anchors, manifest = _validate_adaptive_selection_manifest(
            path=selection_path,
            expected_sha256=expected_sha,
            root=root,
            identity=identity,
            progress_ledgers=ledgers,
        )
        if phase in {"adaptive_seed_scan", "adaptive_failure_scan"}:
            if (
                seed_ledger.get("complete") is not True
                or not isinstance(failure_ledger, Mapping)
                or failure_ledger.get("complete") is not True
            ):
                raise ExternalMemoryDBSCANError(
                    "adaptive selection exists before complete scan ledgers"
                )
            lower_ledger = _require_progress_ledger(
                ledgers,
                phase="shortcut_anchor_scan",
                identity=identity,
                create=True,
            )
            if lower_ledger["entries"]:
                raise ExternalMemoryDBSCANError(
                    "new shortcut lower ledger is unexpectedly nonempty"
                )
            extra = _progress_checkpoint_extra(ledgers, identity=identity)
            extra.update(
                {
                    "adaptive_selection_manifest_path": str(selection_path),
                    "adaptive_selection_manifest_sha256": _sha256_file(
                        selection_path
                    ),
                    "selected_anchor_indices_sha256": manifest[
                        "selection_identity"
                    ]["selected_anchor_indices_sha256"],
                }
            )
            _checkpoint(
                state_path,
                identity=identity,
                phase="shortcut_anchor_scan",
                next_offset=0,
                peak_rss_bytes=max(int(peak_rss_bytes), _rss_bytes()),
                extra=extra,
            )
        return anchors, manifest, max(int(peak_rss_bytes), _rss_bytes())
    if phase not in {"adaptive_seed_scan", "adaptive_failure_scan"}:
        raise ExternalMemoryDBSCANError(
            "adaptive selection manifest is missing after selection phase"
        )

    if phase == "adaptive_seed_scan":
        start = int(seed_ledger["committed_offset"])
        block = _bounded_norm_block_size(
            requested=int(contract.shortcut_query_block_size),
            num_features=n_features,
            source_itemsize=int(vectors.dtype.itemsize),
            max_rss_bytes=int(contract.max_rss_bytes),
        )
        blocks_since_checkpoint = 0
        for offset in range(start, n_samples, block):
            stop = min(n_samples, offset + block)
            local_candidates = _adaptive_seed_block_candidates(
                vectors, start=offset, stop=stop, seed_count=seed_count
            )
            candidates = sorted(
                candidates + local_candidates, key=lambda row: (row[0], row[1])
            )[
                :seed_count
            ]
            _append_progress_entry(
                seed_ledger,
                start=offset,
                stop=stop,
                payload={
                    "row_count": stop - offset,
                    "local_seed_candidates": _seed_candidate_rows(
                        local_candidates
                    ),
                },
            )
            peak = max(
                peak,
                _check_rss(
                    int(contract.max_rss_bytes), phase="adaptive_seed_scan"
                ),
            )
            blocks_since_checkpoint += 1
            if (
                blocks_since_checkpoint >= int(contract.checkpoint_interval_blocks)
                or stop == n_samples
            ):
                extra = _progress_checkpoint_extra(ledgers, identity=identity)
                extra.update(
                    {
                        "effective_shortcut_query_block_size": int(block),
                        "adaptive_seed_candidates": _seed_candidate_rows(
                            candidates
                        ),
                    }
                )
                _checkpoint(
                    state_path,
                    identity=identity,
                    phase="adaptive_seed_scan",
                    next_offset=stop,
                    peak_rss_bytes=peak,
                    extra=extra,
                )
                blocks_since_checkpoint = 0
        if len(candidates) != seed_count:
            raise ExternalMemoryDBSCANError("adaptive seed scan is incomplete")
        seed_indices = [index for _norm, index in candidates]
        seed_norm_hex = [float(norm).hex() for norm, _index in candidates]
        _complete_progress_ledger(
            seed_ledger,
            num_samples=n_samples,
            result={
                "seed_candidates": _seed_candidate_rows(candidates),
                "seed_indices_sha256": _sample_indices_sha256(seed_indices),
            },
        )
        failure_ledger = _require_progress_ledger(
            ledgers,
            phase="adaptive_failure_scan",
            identity=identity,
            create=True,
        )
        extra = _progress_checkpoint_extra(ledgers, identity=identity)
        extra.update(
            {
                "seed_indices": seed_indices,
                "seed_indices_sha256": _sample_indices_sha256(seed_indices),
                "seed_squared_norm_hex": seed_norm_hex,
                "adaptive_failure_indices": [],
                "adaptive_failure_indices_sha256": _sample_indices_sha256([]),
            }
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase="adaptive_failure_scan",
            next_offset=0,
            peak_rss_bytes=peak,
            extra=extra,
        )
        state = _load_checkpoint(state_path)
        ledgers = _load_progress_ledgers(
            state, identity=identity, num_samples=n_samples
        )
        seed_ledger = _require_progress_ledger(
            ledgers, phase="adaptive_seed_scan", identity=identity
        )
        failure_ledger = _require_progress_ledger(
            ledgers, phase="adaptive_failure_scan", identity=identity
        )
        phase = "adaptive_failure_scan"

    candidates = _fold_seed_ledger(seed_ledger, seed_count=seed_count)
    seed_indices = [index for _norm, index in candidates]
    seed_norm_hex = [float(norm).hex() for norm, _index in candidates]
    state_seed_indices = [int(value) for value in state.get("seed_indices") or []]
    state_seed_norm_hex = [
        str(value) for value in state.get("seed_squared_norm_hex") or []
    ]
    if (
        len(seed_indices) != seed_count
        or len(seed_norm_hex) != seed_count
        or state_seed_indices != seed_indices
        or state_seed_norm_hex != seed_norm_hex
        or _sample_indices_sha256(seed_indices) != state.get("seed_indices_sha256")
    ):
        raise ExternalMemoryDBSCANError("adaptive seed checkpoint identity mismatch")
    if failure_ledger is None:
        raise ExternalMemoryDBSCANError("adaptive failure progress ledger is missing")
    failures = _fold_failure_ledger(failure_ledger)
    state_failures = [
        int(value) for value in state.get("adaptive_failure_indices") or []
    ]
    if (
        failures != sorted(set(failures))
        or state_failures != failures
        or _sample_indices_sha256(failures)
        != state.get("adaptive_failure_indices_sha256")
    ):
        raise ExternalMemoryDBSCANError("adaptive failure checkpoint identity mismatch")
    seed_array = np.asarray(seed_indices, dtype=np.intp)
    seed_vectors = np.asarray(vectors[seed_array])
    if seed_model is None:
        seed_model, seed_sklearn_version = _fit_anchor_neighbors(
            seed_vectors, eps=float(contract.eps)
        )
        if seed_sklearn_version != sklearn_version:
            raise ExternalMemoryDBSCANError("adaptive seed/full sklearn versions differ")
    seed_local_by_global = {
        int(global_index): local_index
        for local_index, global_index in enumerate(seed_indices)
    }
    start = int(failure_ledger["committed_offset"])
    block = _bounded_shortcut_block_size(
        requested=int(contract.shortcut_query_block_size),
        anchor_count=seed_count,
        max_rss_bytes=int(contract.max_rss_bytes),
        phase="adaptive_failure_scan",
    )
    blocks_since_checkpoint = 0
    for offset in range(start, n_samples, block):
        stop = min(n_samples, offset + block)
        _check_rss(
            int(contract.max_rss_bytes),
            phase="adaptive_failure_scan.query",
            reserved_bytes=_shortcut_query_reservation_bytes(
                stop - offset, seed_count
            ),
        )
        block_failures = _adaptive_failure_block(
            model=seed_model,
            vectors=vectors,
            start=offset,
            stop=stop,
            seed_local_by_global=seed_local_by_global,
            min_samples=int(contract.min_samples),
        )
        failures.extend(block_failures)
        _append_progress_entry(
            failure_ledger,
            start=offset,
            stop=stop,
            payload={
                "row_count": stop - offset,
                "seed_indices_sha256": _sample_indices_sha256(seed_indices),
                "failure_indices": block_failures,
                "failure_indices_sha256": _sample_indices_sha256(block_failures),
            },
        )
        if len(failures) > int(contract.shortcut_failure_cap):
            observed = np.asarray(failures, dtype=np.intp)
            observed_path = root / "adaptive_failure_cap_exceeded_indices.npy"
            observed_sha = _ensure_exact_npy(
                observed_path,
                observed,
                label="adaptive cap-exceeded failure indices",
            )
            failure = _shortcut_failure(
                root=root,
                identity=identity,
                reason="adaptive_failure_cap_exceeded",
                num_samples=n_samples,
                fallback_limit=int(contract.exact_fallback_max_samples),
                details={
                    "failure_cap": int(contract.shortcut_failure_cap),
                    "observed_failure_count": int(len(failures)),
                    "observed_failure_indices_path": str(observed_path),
                    "observed_failure_indices_sha256": observed_sha,
                    "first_pass_complete": False,
                },
            )
            extra = _progress_checkpoint_extra(ledgers, identity=identity)
            extra.update(
                {
                    "seed_indices": seed_indices,
                    "seed_indices_sha256": _sample_indices_sha256(seed_indices),
                    "seed_squared_norm_hex": seed_norm_hex,
                    "adaptive_failure_indices": failures,
                    "adaptive_failure_indices_sha256": _sample_indices_sha256(
                        failures
                    ),
                    "shortcut_failure_path": failure["path"],
                    "shortcut_failure_sha256": failure["sha256"],
                    "shortcut_approximation_used": False,
                }
            )
            _checkpoint(
                state_path,
                identity=identity,
                phase="shortcut_blocked",
                next_offset=stop,
                peak_rss_bytes=max(peak, _rss_bytes()),
                extra=extra,
            )
            raise ExternalMemoryDBSCANError(
                "EXACT_DBSCAN_COMPLEXITY_BLOCKED:"
                "reason=adaptive_failure_cap_exceeded:"
                f"count={len(failures)}:cap={contract.shortcut_failure_cap}"
            )
        peak = max(
            peak,
            _check_rss(
                int(contract.max_rss_bytes), phase="adaptive_failure_scan"
            ),
        )
        blocks_since_checkpoint += 1
        if (
            blocks_since_checkpoint >= int(contract.checkpoint_interval_blocks)
            or stop == n_samples
        ):
            extra = _progress_checkpoint_extra(ledgers, identity=identity)
            extra.update(
                {
                    "seed_indices": seed_indices,
                    "seed_indices_sha256": _sample_indices_sha256(seed_indices),
                    "seed_squared_norm_hex": seed_norm_hex,
                    "adaptive_failure_indices": failures,
                    "adaptive_failure_indices_sha256": _sample_indices_sha256(
                        failures
                    ),
                    "effective_shortcut_query_block_size": int(block),
                }
            )
            _checkpoint(
                state_path,
                identity=identity,
                phase="adaptive_failure_scan",
                next_offset=stop,
                peak_rss_bytes=peak,
                extra=extra,
            )
            blocks_since_checkpoint = 0

    failure_array = np.asarray(failures, dtype=np.intp)
    if len(failure_array) > int(contract.shortcut_failure_cap):
        raise ExternalMemoryDBSCANError(
            "adaptive failure set exceeds the frozen cap before selection"
        )
    if not np.array_equal(failure_array, np.unique(failure_array)):
        raise ExternalMemoryDBSCANError(
            "adaptive first-pass failure indices are not canonical"
        )
    _complete_progress_ledger(
        failure_ledger,
        num_samples=n_samples,
        result={
            "first_pass_complete": True,
            "failure_indices": failures,
            "failure_indices_sha256": _sample_indices_sha256(failures),
        },
    )
    # Persist the completed first-pass ledger before publishing any selection
    # artifact.  A crash after the selection JSON rename must never leave that
    # manifest referring to a completion state that existed only in memory.
    completed_failure_extra = _progress_checkpoint_extra(
        ledgers, identity=identity
    )
    completed_failure_extra.update(
        {
            "seed_indices": seed_indices,
            "seed_indices_sha256": _sample_indices_sha256(seed_indices),
            "seed_squared_norm_hex": seed_norm_hex,
            "adaptive_failure_indices": failures,
            "adaptive_failure_indices_sha256": _sample_indices_sha256(failures),
            "effective_shortcut_query_block_size": int(block),
        }
    )
    _checkpoint(
        state_path,
        identity=identity,
        phase="adaptive_failure_scan",
        next_offset=n_samples,
        peak_rss_bytes=peak,
        extra=completed_failure_extra,
    )
    anchor_indices = np.asarray(
        sorted(set(seed_indices).union(failures)), dtype=np.intp
    )
    if len(anchor_indices) > 65_535:
        raise ExternalMemoryDBSCANError(
            "EXACT_DBSCAN_COMPLEXITY_BLOCKED:adaptive anchor set exceeds limit"
        )
    failure_path = root / "adaptive_first_pass_failure_indices.npy"
    anchor_indices_path = root / "shortcut_anchor_indices.npy"
    anchor_rows_path = root / "adaptive_selected_anchor_rows.npy"
    failure_sha = _ensure_exact_npy(
        failure_path, failure_array, label="adaptive first-pass failure indices"
    )
    anchor_indices_sha = _ensure_exact_npy(
        anchor_indices_path, anchor_indices, label="adaptive anchor indices"
    )
    anchor_rows = np.asarray(vectors[anchor_indices])
    anchor_rows_sha = _ensure_exact_npy(
        anchor_rows_path, anchor_rows, label="adaptive anchor rows"
    )
    selection_identity = {
        "schema_version": ADAPTIVE_SELECTION_SCHEMA_VERSION,
        "vectors_path": identity["vectors_path"],
        "vectors_sha256": identity["vectors_sha256"],
        "vectors_stat_identity": identity["vectors_stat_identity"],
        "vectors_dtype": identity["vectors_dtype"],
        "vectors_shape": identity["vectors_shape"],
        "selection_contract": identity["shortcut_contract"],
        "sklearn_version": sklearn_version,
        "seed_selection_algorithm": (
            "global_k_minimum_squared_l2_float64_tie_sample_index_v1"
        ),
        "seed_indices": seed_indices,
        "seed_indices_sha256": _sample_indices_sha256(seed_indices),
        "seed_squared_norm_hex": seed_norm_hex,
        "failure_selection_rule": (
            "all_first_pass_insufficient_seed_lower_bound_indices_v1"
        ),
        "first_pass_complete": True,
        "failure_count": int(len(failure_array)),
        "failure_indices_path": str(failure_path),
        "failure_indices_sha256": failure_sha,
        "failure_index_list_sha256": _sample_indices_sha256(failure_array),
        "adaptive_seed_progress_ledger_sha256": _progress_ledger_sha256(
            seed_ledger
        ),
        "adaptive_failure_progress_ledger_sha256": _progress_ledger_sha256(
            failure_ledger
        ),
        "anchor_selection_rule": (
            "sorted_unique_union_of_seed_and_all_failure_indices_v1"
        ),
        "anchor_count": int(len(anchor_indices)),
        "anchor_indices_path": str(anchor_indices_path),
        "anchor_indices_sha256": anchor_indices_sha,
        "selected_anchor_indices_sha256": _sample_indices_sha256(anchor_indices),
        "anchor_rows_path": str(anchor_rows_path),
        "anchor_rows_sha256": anchor_rows_sha,
        "approximation_used": False,
    }
    selection_manifest = {
        "schema_version": ADAPTIVE_SELECTION_SCHEMA_VERSION,
        "run_complete": True,
        "selection_identity": selection_identity,
        "selection_identity_sha256": _stable_hash(selection_identity),
        "completed_at": _utc_now(),
    }
    _atomic_json(selection_path, selection_manifest)
    selection_sha = _sha256_file(selection_path)
    lower_ledger = _require_progress_ledger(
        ledgers,
        phase="shortcut_anchor_scan",
        identity=identity,
        create=True,
    )
    if lower_ledger["entries"]:
        raise ExternalMemoryDBSCANError(
            "new shortcut lower ledger is unexpectedly nonempty"
        )
    extra = _progress_checkpoint_extra(ledgers, identity=identity)
    extra.update(
        {
            "adaptive_selection_manifest_path": str(selection_path),
            "adaptive_selection_manifest_sha256": selection_sha,
            "selected_anchor_indices_sha256": selection_identity[
                "selected_anchor_indices_sha256"
            ],
        }
    )
    _checkpoint(
        state_path,
        identity=identity,
        phase="shortcut_anchor_scan",
        next_offset=0,
        peak_rss_bytes=max(peak, _rss_bytes()),
        extra=extra,
    )
    validated_anchors, validated_manifest = _validate_adaptive_selection_manifest(
        path=selection_path,
        expected_sha256=selection_sha,
        root=root,
        identity=identity,
        progress_ledgers=ledgers,
    )
    return validated_anchors, validated_manifest, max(peak, _rss_bytes())


def _validate_split_shortcut_certificates(
    *,
    manifest: Mapping[str, Any],
    proof: Mapping[str, Any],
    root: Path,
    scientific_identity_sha256: str,
    n_samples: int,
    eps: float,
    min_samples: int,
    labels: Path,
    core: Path,
) -> None:
    loaded: dict[str, dict[str, Any]] = {}
    schemas = {
        "all_core": ALL_CORE_CERTIFICATE_SCHEMA_VERSION,
        "connectivity": CONNECTIVITY_CERTIFICATE_SCHEMA_VERSION,
        "boundary": BOUNDARY_CERTIFICATE_SCHEMA_VERSION,
        "cluster_partition": CLUSTER_PARTITION_SCHEMA_VERSION,
    }
    for name, schema in schemas.items():
        path_field = f"{name}_certificate_path" if name != "cluster_partition" else "cluster_partition_path"
        hash_field = f"{name}_certificate_sha256" if name != "cluster_partition" else "cluster_partition_sha256"
        raw_path = proof.get(path_field)
        path = Path(str(raw_path or "")).resolve(strict=True)
        if (
            path.parent != root
            or manifest.get(path_field) != str(path)
            or manifest.get(hash_field) != proof.get(hash_field)
            or _sha256_file(path) != proof.get(hash_field)
        ):
            raise ExternalMemoryDBSCANError(
                f"terminal split certificate closure mismatch: {name}"
            )
        payload = _load_object(path)
        if (
            payload.get("schema_version") != schema
            or payload.get("status") != "PASS"
            or payload.get("scientific_identity_sha256")
            != scientific_identity_sha256
            or int(payload.get("input_rows", -1)) != int(n_samples)
            or payload.get("approximation_used") is not False
        ):
            raise ExternalMemoryDBSCANError(
                f"terminal split certificate is incomplete: {name}"
            )
        loaded[name] = payload
    boundary = loaded["boundary"]
    all_core = loaded["all_core"]
    connectivity = loaded["connectivity"]
    partition = loaded["cluster_partition"]
    if (
        boundary.get("metric") != "euclidean"
        or boundary.get("comparison") != "distance <= eps"
        or boundary.get("eps") != eps
        or boundary.get("recheck_dtype") != "float64"
        or boundary.get("float64_revalidation_complete") is not True
        or int(boundary.get("float64_revalidated_row_count", -1)) != n_samples
        or boundary.get("sklearn_anchor_count_witness_exactly_matched") is not True
        or int(boundary.get("uncertain_edges_accepted", -1)) != 0
    ):
        raise ExternalMemoryDBSCANError("terminal boundary certificate is incomplete")
    if (
        all_core.get("metric") != "euclidean"
        or all_core.get("comparison") != "distance <= eps"
        or all_core.get("eps") != eps
        or int(all_core.get("min_samples", -1)) != min_samples
        or all_core.get("self_neighbor_counted_exactly_once") is not True
        or all_core.get("all_points_core_proven") is not True
        or int(all_core.get("core_point_count", -1)) != n_samples
        or all_core.get("exact") is not True
        or all_core.get("boundary_certificate_sha256")
        != proof.get("boundary_certificate_sha256")
    ):
        raise ExternalMemoryDBSCANError("terminal all-core certificate is incomplete")
    if (
        connectivity.get("metric") != "euclidean"
        or connectivity.get("comparison") != "distance <= eps"
        or connectivity.get("eps") != eps
        or connectivity.get("anchor_graph_exact_connected") is not True
        or connectivity.get("every_close_row_attached_to_anchor_component") is not True
        or int(connectivity.get("attached_or_anchor_row_count", -1)) != n_samples
        or connectivity.get("single_epsilon_component_proven") is not True
        or connectivity.get("exact") is not True
        or connectivity.get("all_core_certificate_sha256")
        != proof.get("all_core_certificate_sha256")
        or connectivity.get("boundary_certificate_sha256")
        != proof.get("boundary_certificate_sha256")
    ):
        raise ExternalMemoryDBSCANError(
            "terminal connectivity certificate is incomplete"
        )
    if (
        partition.get("cluster_count") != 1
        or partition.get("noise_count") != 0
        or partition.get("core_count") != n_samples
        or partition.get("canonical_cluster_labels") != [0]
        or partition.get("partition") != "one_cluster_zero_noise"
        or partition.get("labels_path") != str(labels)
        or partition.get("labels_sha256") != manifest.get("labels_sha256")
        or partition.get("core_mask_path") != str(core)
        or partition.get("core_mask_sha256") != manifest.get("core_mask_sha256")
        or partition.get("all_core_certificate_sha256")
        != proof.get("all_core_certificate_sha256")
        or partition.get("connectivity_certificate_sha256")
        != proof.get("connectivity_certificate_sha256")
        or partition.get("boundary_certificate_sha256")
        != proof.get("boundary_certificate_sha256")
        or partition.get("sklearn_partition_semantics_preserved") is not True
    ):
        raise ExternalMemoryDBSCANError(
            "terminal cluster-partition certificate is incomplete"
        )


def _validate_shortcut_proof_closure(
    *, manifest: Mapping[str, Any], root: Path
) -> tuple[Path, Path, Path, Path, Path]:
    """Validate a terminal shortcut without inventing exact neighbor counts."""

    if (
        manifest.get("clustering_path")
        not in {
            ALL_CORE_ONE_COMPONENT_SHORTCUT,
            ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        }
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
    full_contract = scientific_identity.get("contract")
    vectors_shape = scientific_identity.get("vectors_shape")
    if (
        not isinstance(shortcut_contract, Mapping)
        or not isinstance(full_contract, Mapping)
        or not isinstance(vectors_shape, list)
        or len(vectors_shape) != 2
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut contract is absent")
    identity_n_samples, identity_n_features = (int(value) for value in vectors_shape)
    if (
        proof.get("schema_version") != SHORTCUT_PROOF_SCHEMA_VERSION
        or proof.get("status") != "PASS"
        or proof.get("shortcut") != manifest.get("clustering_path")
        or proof.get("scientific_identity_sha256")
        != manifest.get("scientific_identity_sha256")
        or proof.get("vectors_path") != scientific_identity.get("vectors_path")
        or proof.get("vectors_sha256") != scientific_identity.get("vectors_sha256")
        or proof.get("vectors_stat_identity")
        != scientific_identity.get("vectors_stat_identity")
        or proof.get("vectors_dtype") != scientific_identity.get("vectors_dtype")
        or proof.get("vectors_shape") != vectors_shape
        or proof.get("sklearn_version")
        != scientific_identity.get("sklearn_version")
        or proof.get("eps") != full_contract.get("eps")
        or proof.get("min_samples") != full_contract.get("min_samples")
        or shortcut_contract.get("mode") != manifest.get("clustering_path")
        or scientific_identity.get("nearest_neighbors_fit_method") != "brute"
        or scientific_identity.get("nearest_neighbors_metric") != "euclidean"
        or manifest.get("num_samples") != identity_n_samples
        or manifest.get("num_features") != identity_n_features
        or manifest.get("core_count") != identity_n_samples
        or manifest.get("cluster_count") != 1
        or manifest.get("noise_count") != 0
        or manifest.get("approximation_used") is not False
        or manifest.get("sklearn_dbscan_label_semantics_preserved") is not True
        or proof.get("all_points_core_proven") is not True
        or proof.get("single_epsilon_component_proven") is not True
        or proof.get("labels_are_exact_sklearn_order") is not True
        or proof.get("label_value") != 0
        or proof.get("core_mask_value") is not True
        or proof.get("exact_neighbor_counts_materialized") is not False
        or proof.get("approximation_used") is not False
        or proof.get("all_progress_prefixes_complete") is not True
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut proof is incomplete")
    required_progress_phases = (
        ["adaptive_seed_scan", "adaptive_failure_scan", "shortcut_anchor_scan"]
        if manifest.get("clustering_path")
        == ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
        else ["shortcut_anchor_scan"]
    )
    progress_path = Path(str(proof.get("progress_ledger_path") or ""))
    if (
        proof.get("progress_ledger_required_phases")
        != required_progress_phases
        or manifest.get("progress_ledger_path") != str(progress_path)
        or manifest.get("progress_ledger_sha256")
        != proof.get("progress_ledger_sha256")
    ):
        raise ExternalMemoryDBSCANError(
            "terminal shortcut progress-ledger binding mismatch"
        )
    progress_ledgers = _validate_progress_ledger_artifact(
        path=progress_path,
        expected_sha256=str(proof.get("progress_ledger_sha256") or ""),
        root=root,
        identity=scientific_identity,
        num_samples=int(manifest.get("num_samples", -1)),
        required_phases=required_progress_phases,
    )
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
    edge_rows = (
        [tuple(int(value) for value in row) for row in anchor_edges.tolist()]
        if anchor_edges.ndim == 2 and anchor_edges.shape[1:] == (2,)
        else []
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
        or len(anchor_edges) != int(proof.get("anchor_edge_count", -1))
        or edge_rows != sorted(set(edge_rows))
        or lower.dtype != np.dtype(np.uint32)
        or lower.shape != (n_samples,)
        or recorded_minimum < min_samples - 1
        or (
            n_samples > anchor_count
            and int(proof.get("minimum_non_anchor_anchor_neighbors", -1)) < 1
        )
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut witness schema mismatch")
    validated_minimum, validated_non_anchor = _validate_complete_lower_witness(
        lower=lower,
        anchor_indices=anchor_indices,
        ledger=progress_ledgers["shortcut_anchor_scan"],
        num_samples=n_samples,
        min_samples=min_samples,
    )
    if (
        validated_minimum != recorded_minimum
        or validated_non_anchor
        != proof.get("minimum_non_anchor_anchor_neighbors")
    ):
        raise ExternalMemoryDBSCANError(
            "terminal shortcut lower-bound minimum mismatch"
        )
    expected_indices_sha = hashlib.sha256(
        json.dumps(anchor_indices.tolist(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("clustering_path") == ALL_CORE_ONE_COMPONENT_SHORTCUT:
        if (
            proof.get("selected_anchor_indices_sha256")
            != shortcut_contract.get("selected_anchor_indices_sha256")
            or expected_indices_sha
            != shortcut_contract.get("selected_anchor_indices_sha256")
        ):
            raise ExternalMemoryDBSCANError(
                "terminal shortcut anchor identity mismatch"
            )
    else:
        selection_path = Path(
            str(proof.get("adaptive_selection_manifest_path") or "")
        )
        selected, selection_manifest = _validate_adaptive_selection_manifest(
            path=selection_path,
            expected_sha256=str(
                proof.get("adaptive_selection_manifest_sha256") or ""
            ),
            root=root,
            identity=scientific_identity,
            progress_ledgers=progress_ledgers,
        )
        selection_identity = selection_manifest["selection_identity"]
        if (
            not np.array_equal(selected, anchor_indices)
            or expected_indices_sha
            != selection_identity.get("selected_anchor_indices_sha256")
            or proof.get("selected_anchor_indices_sha256")
            != selection_identity.get("selected_anchor_indices_sha256")
            or proof.get("adaptive_selection_identity_sha256")
            != selection_manifest.get("selection_identity_sha256")
            or proof.get("first_pass_failure_indices_sha256")
            != selection_identity.get("failure_indices_sha256")
            or proof.get("first_pass_failure_index_list_sha256")
            != selection_identity.get("failure_index_list_sha256")
            or proof.get("selected_anchor_rows_sha256")
            != selection_identity.get("anchor_rows_sha256")
            or proof.get("second_pass_complete") is not True
        ):
            raise ExternalMemoryDBSCANError(
                "terminal adaptive selection/proof mismatch"
            )
    adjacency = [{index} for index in range(anchor_count)]
    for source, target in edge_rows:
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
    _validate_constant_output(
        labels,
        shape=(n_samples,),
        dtype=np.intp,
        expected=0,
        label="terminal shortcut labels",
    )
    _validate_constant_output(
        core,
        shape=(n_samples,),
        dtype=np.bool_,
        expected=True,
        label="terminal shortcut core mask",
    )
    if (
        proof.get("labels_path") != str(labels)
        or proof.get("labels_sha256") != manifest.get("labels_sha256")
        or proof.get("core_mask_path") != str(core)
        or proof.get("core_mask_sha256") != manifest.get("core_mask_sha256")
    ):
        raise ExternalMemoryDBSCANError("terminal shortcut proof/output mismatch")
    _validate_split_shortcut_certificates(
        manifest=manifest,
        proof=proof,
        root=root,
        scientific_identity_sha256=str(manifest["scientific_identity_sha256"]),
        n_samples=n_samples,
        eps=float(proof["eps"]),
        min_samples=min_samples,
        labels=labels,
        core=core,
    )
    return labels, core, proof_path, artifacts[0], artifacts[2]


def _component_scan_result(
    *,
    ledger: Mapping[str, Any],
    component_parent: np.ndarray,
    query_anchor_globals: np.ndarray,
    verify_core_contract: bool,
) -> dict[str, Any]:
    margins: list[float] = []
    boundary_edges = 0
    near_count = 0
    witness_count = 0
    minimum_lower: int | None = None
    for entry in ledger["entries"]:
        payload = entry["payload"]
        margin = payload.get("minimum_certifying_margin_hex")
        if margin is not None:
            margins.append(float.fromhex(str(margin)))
        boundary_edges += int(payload["count_certifying_edges_exactly_at_eps"])
        near_count += int(payload["near_boundary_direct_norm_recompute_count"])
        witness_count += len(payload["new_bridge_witnesses"])
        if verify_core_contract:
            block_minimum = int(payload["minimum_core_lower_bound"])
            minimum_lower = (
                block_minimum
                if minimum_lower is None
                else min(minimum_lower, block_minimum)
            )
    roots = [_component_find(component_parent, value) for value in range(len(component_parent))]
    return {
        "full_scan_complete": True,
        "query_anchor_indices_sha256": _sample_indices_sha256(
            query_anchor_globals
        ),
        "component_parent_after": roots,
        "component_count_after": len(set(roots)),
        "bridge_witness_count": int(witness_count),
        "minimum_certifying_margin_hex": (
            None if not margins else float(min(margins)).hex()
        ),
        "count_certifying_edges_exactly_at_eps": int(boundary_edges),
        "near_boundary_direct_norm_recompute_count": int(near_count),
        "all_sklearn_float64_memberships_equal": True,
        "all_rows_core_proven": bool(verify_core_contract),
        "minimum_core_lower_bound": minimum_lower,
    }


def _run_component_scan(
    *,
    vectors: np.ndarray,
    root: Path,
    state_path: Path,
    identity: Mapping[str, Any],
    contract: ExternalDBSCANContract,
    sklearn_version: str,
    ledgers: dict[str, dict[str, Any]],
    ledger_phase: str,
    state_phase: str,
    query_anchor_locals: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_vectors64: np.ndarray,
    component_by_anchor: np.ndarray,
    initial_component_parent: np.ndarray,
    seed_indices: Sequence[int],
    failures: np.ndarray,
    anchor_degrees: np.ndarray,
    lower_output: np.ndarray | None,
    attachment_output: np.ndarray | None,
    verify_core_contract: bool,
    peak_rss_bytes: int,
    allow_append: bool,
) -> tuple[np.ndarray, int]:
    """Replay and extend one hash-ledgered exact component scan."""

    n_samples = int(vectors.shape[0])
    ledger = _require_progress_ledger(
        ledgers, phase=ledger_phase, identity=identity, create=allow_append
    )
    _validate_progress_ledger_structure(
        ledger,
        phase=ledger_phase,
        identity=identity,
        num_samples=n_samples,
    )
    query_anchor_locals = np.asarray(query_anchor_locals, dtype=np.intp)
    if (
        query_anchor_locals.ndim != 1
        or not np.array_equal(query_anchor_locals, np.unique(query_anchor_locals))
        or bool(np.any(query_anchor_locals < 0))
        or bool(np.any(query_anchor_locals >= len(anchor_indices)))
    ):
        raise ExternalMemoryDBSCANError(
            "adaptive recovery query anchors are noncanonical"
        )
    query_globals = np.asarray(anchor_indices[query_anchor_locals], dtype=np.intp)
    query_vectors = np.asarray(anchor_vectors64[query_anchor_locals], dtype=np.float64)
    query_model, observed_version = _fit_anchor_neighbors(
        np.asarray(vectors[query_globals]), eps=float(contract.eps)
    )
    if observed_version != sklearn_version:
        raise ExternalMemoryDBSCANError(
            "adaptive recovery/full sklearn versions differ"
        )
    component_parent = np.asarray(initial_component_parent, dtype=np.int32).copy()
    peak = max(int(peak_rss_bytes), _rss_bytes())

    # Every committed prefix is recomputed from the immutable source.  The
    # witness union is reconstructed solely from authenticated entries.
    for entry in ledger["entries"]:
        start, stop = int(entry["start"]), int(entry["stop"])
        payload = _recovery_scan_block(
            model=query_model,
            vectors=vectors,
            start=start,
            stop=stop,
            eps=float(contract.eps),
            min_samples=int(contract.min_samples),
            query_anchor_locals=query_anchor_locals,
            query_anchor_globals=query_globals,
            query_vectors64=query_vectors,
            component_by_anchor=component_by_anchor,
            component_parent=component_parent,
            seed_indices=seed_indices,
            failures=failures,
            anchor_indices=anchor_indices,
            anchor_degrees=anchor_degrees,
            lower_output=lower_output,
            attachment_output=attachment_output,
            verify_core_contract=verify_core_contract,
            write_core_outputs=False,
        )
        if payload != entry.get("payload"):
            raise ExternalMemoryDBSCANError(
                f"{ledger_phase} committed-prefix replay mismatch"
            )
        peak = max(
            peak,
            _check_rss(
                int(contract.max_rss_bytes),
                phase=f"{ledger_phase}.resume_replay",
            ),
        )
    if ledger.get("complete") is True:
        expected_result = _component_scan_result(
            ledger=ledger,
            component_parent=component_parent,
            query_anchor_globals=query_globals,
            verify_core_contract=verify_core_contract,
        )
        if ledger.get("result") != expected_result:
            raise ExternalMemoryDBSCANError(
                f"{ledger_phase} complete-ledger result mismatch"
            )
        return component_parent, peak
    if not allow_append:
        raise ExternalMemoryDBSCANError(
            f"{ledger_phase} is incomplete outside its active phase"
        )
    state = _load_checkpoint(state_path)
    if (
        state.get("phase") != state_phase
        or int(state.get("next_offset", -1))
        != int(ledger.get("committed_offset", -2))
    ):
        raise ExternalMemoryDBSCANError(
            f"{ledger_phase} checkpoint/ledger offset mismatch"
        )
    block = _bounded_shortcut_block_size(
        requested=int(contract.shortcut_query_block_size),
        anchor_count=len(query_anchor_locals),
        max_rss_bytes=int(contract.max_rss_bytes),
        phase=ledger_phase,
    )
    blocks_since_checkpoint = 0
    for offset in range(int(ledger["committed_offset"]), n_samples, block):
        stop = min(n_samples, offset + block)
        _check_rss(
            int(contract.max_rss_bytes),
            phase=f"{ledger_phase}.query",
            reserved_bytes=_shortcut_query_reservation_bytes(
                stop - offset, len(query_anchor_locals)
            ),
        )
        payload = _recovery_scan_block(
            model=query_model,
            vectors=vectors,
            start=offset,
            stop=stop,
            eps=float(contract.eps),
            min_samples=int(contract.min_samples),
            query_anchor_locals=query_anchor_locals,
            query_anchor_globals=query_globals,
            query_vectors64=query_vectors,
            component_by_anchor=component_by_anchor,
            component_parent=component_parent,
            seed_indices=seed_indices,
            failures=failures,
            anchor_indices=anchor_indices,
            anchor_degrees=anchor_degrees,
            lower_output=lower_output,
            attachment_output=attachment_output,
            verify_core_contract=verify_core_contract,
            write_core_outputs=True,
        )
        _append_progress_entry(
            ledger, start=offset, stop=stop, payload=payload
        )
        peak = max(
            peak,
            _check_rss(int(contract.max_rss_bytes), phase=ledger_phase),
        )
        blocks_since_checkpoint += 1
        if (
            blocks_since_checkpoint >= int(contract.checkpoint_interval_blocks)
            or stop == n_samples
        ):
            if lower_output is not None:
                _fsync_memmap(lower_output)
            if attachment_output is not None:
                _fsync_memmap(attachment_output)
            extra = _progress_checkpoint_extra(ledgers, identity=identity)
            extra.update(
                {
                    **_adaptive_selection_checkpoint_fields(root),
                    "component_recovery_query_anchor_indices_sha256": (
                        _sample_indices_sha256(query_globals)
                    ),
                    "component_parent_after": [
                        _component_find(component_parent, value)
                        for value in range(len(component_parent))
                    ],
                    "effective_shortcut_query_block_size": int(block),
                }
            )
            _checkpoint(
                state_path,
                identity=identity,
                phase=state_phase,
                next_offset=stop,
                peak_rss_bytes=peak,
                extra=extra,
            )
            blocks_since_checkpoint = 0
    _complete_progress_ledger(
        ledger,
        num_samples=n_samples,
        result=_component_scan_result(
            ledger=ledger,
            component_parent=component_parent,
            query_anchor_globals=query_globals,
            verify_core_contract=verify_core_contract,
        ),
    )
    return component_parent, peak


def _component_bridge_witnesses(
    ledgers: Mapping[str, Mapping[str, Any]], *, phases: Sequence[str]
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for phase in phases:
        for entry in ledgers[phase]["entries"]:
            values.extend(dict(row) for row in entry["payload"]["new_bridge_witnesses"])
    return values


def _adaptive_selection_checkpoint_fields(root: Path) -> dict[str, Any]:
    path = root / "adaptive_anchor_selection.json"
    manifest = _load_object(path)
    selection = manifest.get("selection_identity")
    if not isinstance(selection, Mapping):
        raise ExternalMemoryDBSCANError(
            "adaptive selection checkpoint identity is absent"
        )
    return {
        "adaptive_selection_manifest_path": str(path),
        "adaptive_selection_manifest_sha256": _sha256_file(path),
        "selected_anchor_indices_sha256": selection[
            "selected_anchor_indices_sha256"
        ],
    }


def _write_component_recovery_outputs(
    *,
    vectors: np.ndarray,
    root: Path,
    state_path: Path,
    final_manifest_path: Path,
    identity: Mapping[str, Any],
    contract: ExternalDBSCANContract,
    sklearn_version: str,
    selection_manifest: Mapping[str, Any],
    anchor_indices: np.ndarray,
    anchor_edges: np.ndarray,
    anchor_rows: Sequence[Sequence[int]],
    component_by_anchor: np.ndarray,
    component_parent: np.ndarray,
    anchor_degrees: np.ndarray,
    anchor_margin: float,
    anchor_boundary_count: int,
    primary_query_locals: np.ndarray,
    boundary_rows: Sequence[Mapping[str, Any]],
    failures: np.ndarray,
    lower_path: Path,
    attachment_path: Path,
    ledgers: dict[str, dict[str, Any]],
    required_progress_phases: Sequence[str],
    exhaustive_scan_used: bool,
    peak_rss_bytes: int,
) -> ExternalDBSCANResult:
    n_samples, n_features = map(int, vectors.shape)
    component_count = int(np.max(component_by_anchor)) + 1
    final_roots = np.asarray(
        [_component_find(component_parent, value) for value in range(component_count)],
        dtype=np.int32,
    )
    attachment = _open_npy_memmap(attachment_path, mode="r")
    lower = _open_npy_memmap(lower_path, mode="r")
    _validate_partial(
        attachment,
        shape=(n_samples,),
        dtype=np.uint32,
        label="component recovery attachments",
    )
    _validate_partial(
        lower,
        shape=(n_samples,),
        dtype=np.uint32,
        label="component recovery core lower bounds",
    )
    if bool(np.any(lower < int(contract.min_samples) - 1)):
        raise ExternalMemoryDBSCANError(
            "component recovery final core lower bound failed"
        )
    if bool(np.any(attachment >= component_count)):
        raise ExternalMemoryDBSCANError(
            "component recovery final attachment escaped component set"
        )
    minimum_by_root: dict[int, int] = {}
    scan_block = max(1, min(1_000_000, n_samples))
    for offset in range(0, n_samples, scan_block):
        stop = min(n_samples, offset + scan_block)
        roots = final_roots[np.asarray(attachment[offset:stop], dtype=np.intp)]
        for component_root in np.unique(roots).tolist():
            local = int(np.flatnonzero(roots == int(component_root))[0]) + offset
            minimum_by_root[int(component_root)] = min(
                minimum_by_root.get(int(component_root), n_samples), local
            )
    ordered_roots = sorted(minimum_by_root, key=lambda value: minimum_by_root[value])
    label_by_root = {
        component_root: label
        for label, component_root in enumerate(ordered_roots)
    }
    labels_path = root / "labels.npy"
    labels_temporary = root / "labels.partial.npy"
    labels_temporary.unlink(missing_ok=True)
    labels = _new_memmap(labels_temporary, dtype=np.intp, shape=(n_samples,))
    for offset in range(0, n_samples, scan_block):
        stop = min(n_samples, offset + scan_block)
        roots = final_roots[np.asarray(attachment[offset:stop], dtype=np.intp)]
        labels[offset:stop] = np.fromiter(
            (label_by_root[int(value)] for value in roots.tolist()),
            dtype=np.intp,
            count=stop - offset,
        )
    _fsync_memmap(labels)
    del labels
    labels_sha = _sha256_file(labels_temporary)
    os.replace(labels_temporary, labels_path)
    _fsync_directory(root)
    core_path = root / "core_mask.npy"
    _constant_npy(core_path, shape=(n_samples,), dtype=np.bool_, fill_value=True)
    core_sha = _sha256_file(core_path)
    lower_sha = _sha256_file(lower_path)
    attachment_sha = _sha256_file(attachment_path)
    component_path = root / "anchor_initial_components.npy"
    component_sha = _ensure_exact_npy(
        component_path,
        np.asarray(component_by_anchor, dtype=np.int32),
        label="anchor initial components",
    )
    final_root_path = root / "final_component_roots.npy"
    final_root_sha = _ensure_exact_npy(
        final_root_path,
        final_roots,
        label="final component roots",
    )
    bridge_phases = [COMPONENT_PRIMARY_LEDGER_PHASE]
    if exhaustive_scan_used:
        bridge_phases.append(COMPONENT_EXPANSION_LEDGER_PHASE)
    witnesses = _component_bridge_witnesses(ledgers, phases=bridge_phases)
    witness_path = root / "component_bridge_witnesses.json"
    _atomic_json(
        witness_path,
        {
            "schema_version": COMPONENT_RECOVERY_SCHEMA_VERSION,
            "witnesses": witnesses,
            "witness_count": len(witnesses),
            "approximation_used": False,
        },
    )
    witness_sha = _sha256_file(witness_path)
    progress_path = root / "shortcut_progress_ledger.json"
    progress_sha = _write_progress_ledger_artifact(
        path=progress_path,
        ledgers=ledgers,
        identity=identity,
        required_phases=required_progress_phases,
        num_samples=n_samples,
    )
    selection_path = root / "adaptive_anchor_selection.json"
    selection_sha = _sha256_file(selection_path)
    anchor_indices_path = root / "shortcut_anchor_indices.npy"
    anchor_edges_path = root / "shortcut_anchor_edges.npy"
    anchor_indices_sha = _sha256_file(anchor_indices_path)
    anchor_edges_sha = _sha256_file(anchor_edges_path)
    primary_globals = np.asarray(anchor_indices[primary_query_locals], dtype=np.intp)
    primary_query_path = root / "component_primary_query_anchor_indices.npy"
    primary_query_sha = _ensure_exact_npy(
        primary_query_path,
        primary_globals,
        label="component primary query anchors",
    )
    margins = [float(anchor_margin)]
    exact_boundary_edges = int(anchor_boundary_count)
    near_boundary_count = 0
    for phase in bridge_phases:
        result = ledgers[phase]["result"]
        margin = result.get("minimum_certifying_margin_hex")
        if margin is not None:
            margins.append(float.fromhex(str(margin)))
        exact_boundary_edges += int(result["count_certifying_edges_exactly_at_eps"])
        near_boundary_count += int(result["near_boundary_direct_norm_recompute_count"])
    scientific_identity_sha = _stable_hash(identity)
    boundary_path = root / "boundary_certificate.json"
    boundary = {
        "schema_version": BOUNDARY_CERTIFICATE_SCHEMA_VERSION,
        "status": "PASS",
        "recovery_schema_version": COMPONENT_RECOVERY_SCHEMA_VERSION,
        "scientific_identity_sha256": scientific_identity_sha,
        "input_rows": n_samples,
        "input_features": n_features,
        "source_dtype": str(vectors.dtype),
        "recheck_dtype": "float64",
        "metric": "euclidean",
        "comparison": "distance <= eps",
        "eps": float(contract.eps),
        "float64_revalidation_complete": True,
        "float64_revalidated_row_count": n_samples,
        "sklearn_membership_exactly_matched": True,
        "anchor_graph_float64_exactly_matched": True,
        "near_boundary_direct_norm_recompute_count": near_boundary_count,
        "minimum_margin_to_eps_among_certifying_edges": float(min(margins)),
        "count_certifying_edges_exactly_at_eps": exact_boundary_edges,
        "uncertain_edges_accepted": 0,
        "approximation_used": False,
    }
    _atomic_json(boundary_path, boundary)
    boundary_sha = _sha256_file(boundary_path)
    all_core_path = root / "all_core_certificate.json"
    all_core = {
        "schema_version": ALL_CORE_CERTIFICATE_SCHEMA_VERSION,
        "status": "PASS",
        "recovery_schema_version": COMPONENT_RECOVERY_SCHEMA_VERSION,
        "scientific_identity_sha256": scientific_identity_sha,
        "input_rows": n_samples,
        "eps": float(contract.eps),
        "min_samples": int(contract.min_samples),
        "metric": "euclidean",
        "comparison": "distance <= eps",
        "self_neighbor_counted_exactly_once": True,
        "all_points_core_proven": True,
        "core_point_count": n_samples,
        "nonfailure_core_authority": "complete_seed_failure_ledger",
        "failure_core_authority": "exact_anchor_degree_including_self",
        "failure_count": int(len(failures)),
        "minimum_failure_anchor_degree_including_self": int(
            np.min(anchor_degrees[np.searchsorted(anchor_indices, failures)])
        ) if len(failures) else None,
        "core_lower_bounds_path": str(lower_path),
        "core_lower_bounds_sha256": lower_sha,
        "minimum_distinct_certifying_neighbors_excluding_self": int(np.min(lower)),
        "boundary_certificate_path": str(boundary_path),
        "boundary_certificate_sha256": boundary_sha,
        "exact": True,
        "approximation_used": False,
    }
    _atomic_json(all_core_path, all_core)
    all_core_sha = _sha256_file(all_core_path)
    connectivity_path = root / "connectivity_certificate.json"
    connectivity = {
        "schema_version": CONNECTIVITY_CERTIFICATE_SCHEMA_VERSION,
        "status": "PASS",
        "recovery_schema_version": COMPONENT_RECOVERY_SCHEMA_VERSION,
        "scientific_identity_sha256": scientific_identity_sha,
        "input_rows": n_samples,
        "eps": float(contract.eps),
        "metric": "euclidean",
        "comparison": "distance <= eps",
        "anchor_count": int(len(anchor_indices)),
        "initial_anchor_component_count": component_count,
        "final_exact_component_count": int(len(ordered_roots)),
        "anchor_graph_exact_connected": component_count == 1,
        "component_bridge_graph_exact": True,
        "single_epsilon_component_proven": len(ordered_roots) == 1,
        "exact_multicomponent_partition_proven": True,
        "primary_boundary_anchor_selection": [dict(value) for value in boundary_rows],
        "primary_query_anchor_indices_path": str(primary_query_path),
        "primary_query_anchor_indices_sha256": primary_query_sha,
        "exhaustive_all_anchor_scan_used": bool(exhaustive_scan_used),
        "bridge_witnesses_path": str(witness_path),
        "bridge_witnesses_sha256": witness_sha,
        "attachment_components_path": str(attachment_path),
        "attachment_components_sha256": attachment_sha,
        "anchor_initial_components_path": str(component_path),
        "anchor_initial_components_sha256": component_sha,
        "final_component_roots_path": str(final_root_path),
        "final_component_roots_sha256": final_root_sha,
        "all_core_certificate_path": str(all_core_path),
        "all_core_certificate_sha256": all_core_sha,
        "boundary_certificate_path": str(boundary_path),
        "boundary_certificate_sha256": boundary_sha,
        "exact": True,
        "approximation_used": False,
    }
    _atomic_json(connectivity_path, connectivity)
    connectivity_sha = _sha256_file(connectivity_path)
    partition_path = root / "cluster_partition.json"
    partition = {
        "schema_version": CLUSTER_PARTITION_SCHEMA_VERSION,
        "status": "PASS",
        "recovery_schema_version": COMPONENT_RECOVERY_SCHEMA_VERSION,
        "scientific_identity_sha256": scientific_identity_sha,
        "input_rows": n_samples,
        "cluster_count": int(len(ordered_roots)),
        "noise_count": 0,
        "core_count": n_samples,
        "canonical_cluster_labels": list(range(len(ordered_roots))),
        "partition": "all_core_exact_components",
        "cluster_order": "minimum_global_core_sample_index",
        "component_minimum_global_sample_indices": [
            int(minimum_by_root[value]) for value in ordered_roots
        ],
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha,
        "core_mask_path": str(core_path),
        "core_mask_sha256": core_sha,
        "all_core_certificate_sha256": all_core_sha,
        "connectivity_certificate_sha256": connectivity_sha,
        "boundary_certificate_sha256": boundary_sha,
        "sklearn_partition_semantics_preserved": True,
        "approximation_used": False,
    }
    _atomic_json(partition_path, partition)
    partition_sha = _sha256_file(partition_path)
    proof_path = root / "shortcut_proof.json"
    proof = {
        "schema_version": SHORTCUT_PROOF_SCHEMA_VERSION,
        "status": "PASS",
        "shortcut": ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
        "recovery_schema_version": COMPONENT_RECOVERY_SCHEMA_VERSION,
        "scientific_identity_sha256": scientific_identity_sha,
        "vectors_path": identity["vectors_path"],
        "vectors_sha256": identity["vectors_sha256"],
        "vectors_stat_identity": identity["vectors_stat_identity"],
        "vectors_dtype": identity["vectors_dtype"],
        "vectors_shape": identity["vectors_shape"],
        "sklearn_version": sklearn_version,
        "eps": float(contract.eps),
        "min_samples": int(contract.min_samples),
        "self_count_semantics": "each sample index counts itself exactly once",
        "duplicate_semantics": "duplicate vectors remain distinct sample indices",
        "adaptive_selection_manifest_path": str(selection_path),
        "adaptive_selection_manifest_sha256": selection_sha,
        "adaptive_selection_identity_sha256": selection_manifest[
            "selection_identity_sha256"
        ],
        "anchor_indices_path": str(anchor_indices_path),
        "anchor_indices_sha256": anchor_indices_sha,
        "anchor_edges_path": str(anchor_edges_path),
        "anchor_edges_sha256": anchor_edges_sha,
        "anchor_count": int(len(anchor_indices)),
        "anchor_edge_count": int(len(anchor_edges)),
        "initial_anchor_component_count": component_count,
        "all_points_core_proven": True,
        "single_epsilon_component_proven": len(ordered_roots) == 1,
        "exact_multicomponent_partition_proven": True,
        "labels_are_exact_sklearn_order": True,
        "exact_neighbor_counts_materialized": False,
        "approximation_used": False,
        "progress_ledger_path": str(progress_path),
        "progress_ledger_sha256": progress_sha,
        "progress_ledger_required_phases": list(required_progress_phases),
        "all_progress_prefixes_complete": True,
        "core_lower_bounds_path": str(lower_path),
        "core_lower_bounds_sha256": lower_sha,
        "attachment_components_path": str(attachment_path),
        "attachment_components_sha256": attachment_sha,
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha,
        "core_mask_path": str(core_path),
        "core_mask_sha256": core_sha,
        "all_core_certificate_path": str(all_core_path),
        "all_core_certificate_sha256": all_core_sha,
        "connectivity_certificate_path": str(connectivity_path),
        "connectivity_certificate_sha256": connectivity_sha,
        "boundary_certificate_path": str(boundary_path),
        "boundary_certificate_sha256": boundary_sha,
        "cluster_partition_path": str(partition_path),
        "cluster_partition_sha256": partition_sha,
        "completed_at": _utc_now(),
    }
    _atomic_json(proof_path, proof)
    proof_sha = _sha256_file(proof_path)
    _verify_source_identity(
        Path(str(identity["vectors_path"])),
        expected_sha256=str(identity["vectors_sha256"]),
        expected_stat=identity["vectors_stat_identity"],
        phase="adaptive component recovery PASS",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_complete": True,
        "scientific_identity": dict(identity),
        "scientific_identity_sha256": scientific_identity_sha,
        "num_samples": n_samples,
        "num_features": n_features,
        "core_count": n_samples,
        "cluster_count": int(len(ordered_roots)),
        "noise_count": 0,
        "neighbor_counts_available": False,
        "neighbor_counts_path": None,
        "neighbor_counts_sha256": None,
        "neighbor_counts_unavailable_reason": (
            "exact core/partition proven by complete seed-failure closure, exact "
            "anchor degrees, and exact component bridge scans"
        ),
        "core_mask_path": str(core_path),
        "core_mask_sha256": core_sha,
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha,
        "shortcut_proof_path": str(proof_path),
        "shortcut_proof_sha256": proof_sha,
        "all_core_certificate_path": str(all_core_path),
        "all_core_certificate_sha256": all_core_sha,
        "connectivity_certificate_path": str(connectivity_path),
        "connectivity_certificate_sha256": connectivity_sha,
        "boundary_certificate_path": str(boundary_path),
        "boundary_certificate_sha256": boundary_sha,
        "cluster_partition_path": str(partition_path),
        "cluster_partition_sha256": partition_sha,
        "progress_ledger_path": str(progress_path),
        "progress_ledger_sha256": progress_sha,
        "clustering_path": ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
        "peak_rss_bytes_observed": max(int(peak_rss_bytes), _rss_bytes()),
        "max_rss_bytes": int(contract.max_rss_bytes),
        "all_neighborhoods_materialized_simultaneously": False,
        "passes": [
            "adaptive_seed_failure_closure",
            "exact_anchor_component_graph",
            "targeted_component_bridge_scan",
            *( ["exhaustive_all_anchor_component_scan"] if exhaustive_scan_used else [] ),
            "canonical_all_core_partition",
        ],
        "sklearn_dbscan_label_semantics_preserved": True,
        "approximation_used": False,
        "completed_at": _utc_now(),
    }
    if int(manifest["peak_rss_bytes_observed"]) > int(contract.max_rss_bytes):
        raise ExternalMemoryDBSCANError(
            "recorded peak RSS exceeded the frozen budget"
        )
    _atomic_json(final_manifest_path, manifest)
    _checkpoint(
        state_path,
        identity=identity,
        phase="complete",
        next_offset=n_samples,
        peak_rss_bytes=int(manifest["peak_rss_bytes_observed"]),
        extra={
            **_progress_checkpoint_extra(ledgers, identity=identity),
            "shortcut_proof_sha256": proof_sha,
            "labels_sha256": labels_sha,
            "core_mask_sha256": core_sha,
        },
    )
    del attachment, lower
    return ExternalDBSCANResult(
        labels_path=labels_path,
        core_mask_path=core_path,
        neighbor_counts_path=None,
        shortcut_proof_path=proof_path,
        manifest_path=final_manifest_path,
        num_samples=n_samples,
        num_features=n_features,
        cluster_count=int(len(ordered_roots)),
        noise_count=0,
        core_count=n_samples,
        manifest_sha256=_sha256_file(final_manifest_path),
    )


def _fit_adaptive_disconnected_component_recovery(
    *,
    vectors: np.ndarray,
    root: Path,
    state_path: Path,
    final_manifest_path: Path,
    identity: Mapping[str, Any],
    contract: ExternalDBSCANContract,
    sklearn_version: str,
    peak_rss_bytes: int,
    anchor_indices: np.ndarray,
    anchor_edges: np.ndarray,
    anchor_rows: Sequence[Sequence[int]],
    selection_manifest: Mapping[str, Any],
) -> ExternalDBSCANResult | None:
    """Recover the exact all-core partition when adaptive anchors disconnect."""

    state = _load_checkpoint(state_path)
    phase = str(state.get("phase"))
    n_samples = int(vectors.shape[0])
    ledgers = _load_progress_ledgers(
        state, identity=identity, num_samples=n_samples
    )
    selection = selection_manifest["selection_identity"]
    failures = np.load(
        Path(str(selection["failure_indices_path"])), allow_pickle=False
    )
    seed_indices = [int(value) for value in selection["seed_indices"]]
    anchor_vectors64 = np.asarray(vectors[anchor_indices], dtype=np.float64)
    anchor_distances, anchor_margin, anchor_boundary_count = (
        _float64_anchor_graph_recheck(
            anchor_vectors=anchor_vectors64,
            anchor_rows=anchor_rows,
            eps=float(contract.eps),
        )
    )
    del anchor_distances
    component_by_anchor, components = _canonical_anchor_components(
        anchor_indices=anchor_indices, anchor_rows=anchor_rows
    )
    component_count = len(components)
    anchor_degrees = np.asarray([len(value) for value in anchor_rows], dtype=np.uint32)
    if len(failures):
        failure_locals = np.searchsorted(anchor_indices, failures)
        if (
            bool(np.any(failure_locals >= len(anchor_indices)))
            or not np.array_equal(anchor_indices[failure_locals], failures)
            or bool(np.any(anchor_degrees[failure_locals] < int(contract.min_samples)))
        ):
            # This proof family applies only when the complete failure closure
            # itself is all core.  Small fixtures can use the ordinary exact
            # route; a production-scale caller gets an explicit non-quadratic
            # recovery block rather than silently starting the old path.
            if n_samples <= int(contract.exact_fallback_max_samples):
                failure = _shortcut_failure(
                    root=root,
                    identity=identity,
                    reason="adaptive_failure_anchor_not_core",
                    num_samples=n_samples,
                    fallback_limit=int(contract.exact_fallback_max_samples),
                    details={
                        "minimum_failure_anchor_degree_including_self": int(
                            np.min(anchor_degrees[failure_locals])
                        ) if len(failure_locals) else None,
                        "general_exact_route": "three_pass_exact_radius_graph_v1",
                    },
                )
                _checkpoint(
                    state_path,
                    identity=identity,
                    phase="neighbor_counts",
                    next_offset=0,
                    peak_rss_bytes=max(int(peak_rss_bytes), _rss_bytes()),
                    extra={
                        "shortcut_failure_path": failure["path"],
                        "shortcut_failure_sha256": failure["sha256"],
                    },
                )
                return None
            failure = _shortcut_failure(
                root=root,
                identity=identity,
                reason="adaptive_failure_anchor_not_core_general_exact_required",
                num_samples=n_samples,
                fallback_limit=int(contract.exact_fallback_max_samples),
                details={
                    "old_quadratic_route_started": False,
                    "required_engine": "partitioned_external_memory_exact_dbscan",
                },
            )
            _checkpoint(
                state_path,
                identity=identity,
                phase="shortcut_blocked",
                next_offset=0,
                peak_rss_bytes=max(int(peak_rss_bytes), _rss_bytes()),
                extra={
                    **_progress_checkpoint_extra(ledgers, identity=identity),
                    "shortcut_failure_path": failure["path"],
                    "shortcut_failure_sha256": failure["sha256"],
                },
            )
            raise ExternalMemoryDBSCANError(
                "EXACT_DBSCAN_GENERAL_EXTERNAL_REQUIRED:"
                "reason=adaptive_failure_anchor_not_core"
            )
    primary_query_locals, boundary_rows = _adaptive_primary_query_anchors(
        anchor_indices=anchor_indices,
        anchor_vectors64=anchor_vectors64,
        component_by_anchor=component_by_anchor,
        seed_indices=seed_indices,
    )
    initial_parent = np.arange(component_count, dtype=np.int32)
    lower_partial = root / "component_core_lower_bounds.partial.npy"
    lower_final = root / "component_core_lower_bounds.npy"
    attachment_partial = root / "component_attachments.partial.npy"
    attachment_final = root / "component_attachments.npy"
    if phase == "shortcut_anchor_scan":
        primary_ledger = _require_progress_ledger(
            ledgers,
            phase=COMPONENT_PRIMARY_LEDGER_PHASE,
            identity=identity,
        )
        if primary_ledger["entries"]:
            raise ExternalMemoryDBSCANError(
                "disconnected recovery found preexisting ordinary anchor entries"
            )
        _checkpoint(
            state_path,
            identity=identity,
            phase=COMPONENT_PRIMARY_PHASE,
            next_offset=0,
            peak_rss_bytes=max(int(peak_rss_bytes), _rss_bytes()),
            extra={
                **_progress_checkpoint_extra(ledgers, identity=identity),
                **_adaptive_selection_checkpoint_fields(root),
                "initial_anchor_component_count": component_count,
                "component_primary_query_anchor_indices_sha256": (
                    _sample_indices_sha256(anchor_indices[primary_query_locals])
                ),
            },
        )
        phase = COMPONENT_PRIMARY_PHASE
        state = _load_checkpoint(state_path)
    if phase == COMPONENT_PRIMARY_PHASE:
        if lower_partial.exists():
            lower = _open_npy_memmap(lower_partial, mode="r+")
            attachment = _open_npy_memmap(attachment_partial, mode="r+")
            _validate_partial(
                lower,
                shape=(n_samples,),
                dtype=np.uint32,
                label="component core lower partial",
            )
            _validate_partial(
                attachment,
                shape=(n_samples,),
                dtype=np.uint32,
                label="component attachment partial",
            )
        else:
            if attachment_partial.exists() or int(state.get("next_offset", -1)) != 0:
                raise ExternalMemoryDBSCANError(
                    "component recovery partial arrays are inconsistent"
                )
            lower = _new_memmap(lower_partial, dtype=np.uint32, shape=(n_samples,))
            attachment = _new_memmap(
                attachment_partial, dtype=np.uint32, shape=(n_samples,)
            )
        primary_parent, peak_rss_bytes = _run_component_scan(
            vectors=vectors,
            root=root,
            state_path=state_path,
            identity=identity,
            contract=contract,
            sklearn_version=sklearn_version,
            ledgers=ledgers,
            ledger_phase=COMPONENT_PRIMARY_LEDGER_PHASE,
            state_phase=COMPONENT_PRIMARY_PHASE,
            query_anchor_locals=primary_query_locals,
            anchor_indices=anchor_indices,
            anchor_vectors64=anchor_vectors64,
            component_by_anchor=component_by_anchor,
            initial_component_parent=initial_parent,
            seed_indices=seed_indices,
            failures=failures,
            anchor_degrees=anchor_degrees,
            lower_output=lower,
            attachment_output=attachment,
            verify_core_contract=True,
            peak_rss_bytes=peak_rss_bytes,
            allow_append=True,
        )
        _fsync_memmap(lower)
        _fsync_memmap(attachment)
        lower_sha = _sha256_file(lower_partial)
        attachment_sha = _sha256_file(attachment_partial)
        del lower, attachment
        connected = len(
            {
                _component_find(primary_parent, value)
                for value in range(component_count)
            }
        ) == 1
        primary_is_exhaustive = len(primary_query_locals) == len(anchor_indices)
        next_phase = (
            COMPONENT_FINALIZE_PHASE
            if connected or primary_is_exhaustive
            else COMPONENT_EXPANSION_PHASE
        )
        if next_phase == COMPONENT_EXPANSION_PHASE:
            expansion_ledger = _require_progress_ledger(
                ledgers,
                phase=COMPONENT_EXPANSION_LEDGER_PHASE,
                identity=identity,
                create=True,
            )
            if expansion_ledger["entries"]:
                raise ExternalMemoryDBSCANError(
                    "new component expansion ledger is unexpectedly nonempty"
                )
        _checkpoint(
            state_path,
            identity=identity,
            phase=next_phase,
            next_offset=0 if next_phase == COMPONENT_EXPANSION_PHASE else n_samples,
            peak_rss_bytes=max(int(peak_rss_bytes), _rss_bytes()),
            extra={
                **_progress_checkpoint_extra(ledgers, identity=identity),
                **_adaptive_selection_checkpoint_fields(root),
                "component_core_lower_bounds_sha256": lower_sha,
                "component_attachments_sha256": attachment_sha,
                "component_primary_connected": bool(connected),
                "component_primary_exhaustive": bool(primary_is_exhaustive),
            },
        )
        os.replace(lower_partial, lower_final)
        os.replace(attachment_partial, attachment_final)
        _fsync_directory(root)
        phase = next_phase
        state = _load_checkpoint(state_path)
    if phase in {COMPONENT_EXPANSION_PHASE, COMPONENT_FINALIZE_PHASE}:
        lower_path = _reconcile_promoted_array(
            partial=lower_partial,
            final=lower_final,
            shape=(n_samples,),
            dtype=np.uint32,
            expected_sha256=str(state.get("component_core_lower_bounds_sha256") or ""),
            label="component core lower bounds",
        )
        attachment_path = _reconcile_promoted_array(
            partial=attachment_partial,
            final=attachment_final,
            shape=(n_samples,),
            dtype=np.uint32,
            expected_sha256=str(state.get("component_attachments_sha256") or ""),
            label="component attachments",
        )
        lower_read = _open_npy_memmap(lower_path, mode="r")
        attachment_read = _open_npy_memmap(attachment_path, mode="r")
        primary_parent, peak_rss_bytes = _run_component_scan(
            vectors=vectors,
            root=root,
            state_path=state_path,
            identity=identity,
            contract=contract,
            sklearn_version=sklearn_version,
            ledgers=ledgers,
            ledger_phase=COMPONENT_PRIMARY_LEDGER_PHASE,
            state_phase=COMPONENT_PRIMARY_PHASE,
            query_anchor_locals=primary_query_locals,
            anchor_indices=anchor_indices,
            anchor_vectors64=anchor_vectors64,
            component_by_anchor=component_by_anchor,
            initial_component_parent=initial_parent,
            seed_indices=seed_indices,
            failures=failures,
            anchor_degrees=anchor_degrees,
            lower_output=lower_read,
            attachment_output=attachment_read,
            verify_core_contract=True,
            peak_rss_bytes=peak_rss_bytes,
            allow_append=False,
        )
        del lower_read, attachment_read
    exhaustive_scan_used = COMPONENT_EXPANSION_LEDGER_PHASE in ledgers
    final_parent = primary_parent
    if phase == COMPONENT_EXPANSION_PHASE:
        final_parent, peak_rss_bytes = _run_component_scan(
            vectors=vectors,
            root=root,
            state_path=state_path,
            identity=identity,
            contract=contract,
            sklearn_version=sklearn_version,
            ledgers=ledgers,
            ledger_phase=COMPONENT_EXPANSION_LEDGER_PHASE,
            state_phase=COMPONENT_EXPANSION_PHASE,
            query_anchor_locals=np.arange(len(anchor_indices), dtype=np.intp),
            anchor_indices=anchor_indices,
            anchor_vectors64=anchor_vectors64,
            component_by_anchor=component_by_anchor,
            initial_component_parent=primary_parent,
            seed_indices=seed_indices,
            failures=failures,
            anchor_degrees=anchor_degrees,
            lower_output=None,
            attachment_output=None,
            verify_core_contract=False,
            peak_rss_bytes=peak_rss_bytes,
            allow_append=True,
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase=COMPONENT_FINALIZE_PHASE,
            next_offset=n_samples,
            peak_rss_bytes=max(int(peak_rss_bytes), _rss_bytes()),
            extra={
                **_progress_checkpoint_extra(ledgers, identity=identity),
                **_adaptive_selection_checkpoint_fields(root),
                "component_core_lower_bounds_sha256": _sha256_file(lower_final),
                "component_attachments_sha256": _sha256_file(attachment_final),
                "component_expansion_complete": True,
            },
        )
        phase = COMPONENT_FINALIZE_PHASE
    if phase != COMPONENT_FINALIZE_PHASE:
        raise ExternalMemoryDBSCANError(
            f"unexpected adaptive component recovery phase: {phase}"
        )
    required_phases = [
        "adaptive_seed_scan",
        "adaptive_failure_scan",
        COMPONENT_PRIMARY_LEDGER_PHASE,
    ]
    if exhaustive_scan_used:
        required_phases.append(COMPONENT_EXPANSION_LEDGER_PHASE)
    return _write_component_recovery_outputs(
        vectors=vectors,
        root=root,
        state_path=state_path,
        final_manifest_path=final_manifest_path,
        identity=identity,
        contract=contract,
        sklearn_version=sklearn_version,
        selection_manifest=selection_manifest,
        anchor_indices=anchor_indices,
        anchor_edges=anchor_edges,
        anchor_rows=anchor_rows,
        component_by_anchor=component_by_anchor,
        component_parent=final_parent,
        anchor_degrees=anchor_degrees,
        anchor_margin=anchor_margin,
        anchor_boundary_count=anchor_boundary_count,
        primary_query_locals=primary_query_locals,
        boundary_rows=boundary_rows,
        failures=failures,
        lower_path=lower_final,
        attachment_path=attachment_final,
        ledgers=ledgers,
        required_progress_phases=required_phases,
        exhaustive_scan_used=exhaustive_scan_used,
        peak_rss_bytes=peak_rss_bytes,
    )


def _validate_component_recovery_closure(
    *, manifest: Mapping[str, Any], root: Path
) -> tuple[Path, Path, Path]:
    """Reopen every exact recovery artifact and reconstruct its partition."""

    if (
        manifest.get("clustering_path") != ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
        or manifest.get("neighbor_counts_available") is not False
        or manifest.get("neighbor_counts_path") is not None
        or manifest.get("neighbor_counts_sha256") is not None
        or manifest.get("approximation_used") is not False
        or manifest.get("sklearn_dbscan_label_semantics_preserved") is not True
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery contract is invalid"
        )
    scientific_identity = manifest.get("scientific_identity")
    if not isinstance(scientific_identity, Mapping):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery identity is absent"
        )
    scientific_sha = _stable_hash(scientific_identity)
    n_samples, _n_features = [
        int(value) for value in scientific_identity["vectors_shape"]
    ]
    contract = scientific_identity["contract"]
    proof_path = Path(str(manifest.get("shortcut_proof_path") or "")).resolve(
        strict=True
    )
    if (
        proof_path.parent != root
        or _sha256_file(proof_path) != manifest.get("shortcut_proof_sha256")
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery proof closure mismatch"
        )
    proof = _load_object(proof_path)
    required_phases = proof.get("progress_ledger_required_phases")
    if (
        proof.get("schema_version") != SHORTCUT_PROOF_SCHEMA_VERSION
        or proof.get("status") != "PASS"
        or proof.get("shortcut") != ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
        or proof.get("recovery_schema_version")
        != COMPONENT_RECOVERY_SCHEMA_VERSION
        or proof.get("scientific_identity_sha256") != scientific_sha
        or proof.get("vectors_path") != scientific_identity.get("vectors_path")
        or proof.get("vectors_sha256") != scientific_identity.get("vectors_sha256")
        or proof.get("vectors_stat_identity")
        != scientific_identity.get("vectors_stat_identity")
        or proof.get("vectors_dtype") != scientific_identity.get("vectors_dtype")
        or proof.get("vectors_shape") != scientific_identity.get("vectors_shape")
        or proof.get("sklearn_version") != scientific_identity.get("sklearn_version")
        or proof.get("eps") != contract.get("eps")
        or proof.get("min_samples") != contract.get("min_samples")
        or proof.get("all_points_core_proven") is not True
        or proof.get("exact_multicomponent_partition_proven") is not True
        or proof.get("labels_are_exact_sklearn_order") is not True
        or proof.get("exact_neighbor_counts_materialized") is not False
        or proof.get("all_progress_prefixes_complete") is not True
        or proof.get("approximation_used") is not False
        or not isinstance(required_phases, list)
        or required_phases[:3]
        != [
            "adaptive_seed_scan",
            "adaptive_failure_scan",
            COMPONENT_PRIMARY_LEDGER_PHASE,
        ]
        or required_phases[3:] not in ([], [COMPONENT_EXPANSION_LEDGER_PHASE])
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery proof is incomplete"
        )
    progress_path = Path(str(proof.get("progress_ledger_path") or ""))
    if (
        manifest.get("progress_ledger_path") != str(progress_path)
        or manifest.get("progress_ledger_sha256")
        != proof.get("progress_ledger_sha256")
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery ledger binding mismatch"
        )
    ledgers = _validate_progress_ledger_artifact(
        path=progress_path,
        expected_sha256=str(proof["progress_ledger_sha256"]),
        root=root,
        identity=scientific_identity,
        num_samples=n_samples,
        required_phases=required_phases,
    )
    selection_path = Path(
        str(proof.get("adaptive_selection_manifest_path") or "")
    )
    anchors, selection_manifest = _validate_adaptive_selection_manifest(
        path=selection_path,
        expected_sha256=str(proof.get("adaptive_selection_manifest_sha256") or ""),
        root=root,
        identity=scientific_identity,
        progress_ledgers=ledgers,
    )
    if (
        proof.get("adaptive_selection_identity_sha256")
        != selection_manifest.get("selection_identity_sha256")
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery selection identity mismatch"
        )
    paths: dict[str, Path] = {}
    for field, hash_field in (
        ("labels_path", "labels_sha256"),
        ("core_mask_path", "core_mask_sha256"),
        ("core_lower_bounds_path", "core_lower_bounds_sha256"),
        ("attachment_components_path", "attachment_components_sha256"),
        ("anchor_indices_path", "anchor_indices_sha256"),
        ("anchor_edges_path", "anchor_edges_sha256"),
    ):
        path = Path(str(proof.get(field) or "")).resolve(strict=True)
        if (
            path.parent != root
            or _sha256_file(path) != proof.get(hash_field)
            or (
                field in {"labels_path", "core_mask_path"}
                and manifest.get(hash_field) != proof.get(hash_field)
            )
        ):
            raise ExternalMemoryDBSCANError(
                f"terminal component-recovery artifact mismatch: {field}"
            )
        paths[field] = path
    loaded_certificates: dict[str, dict[str, Any]] = {}
    for name, schema in (
        ("all_core", ALL_CORE_CERTIFICATE_SCHEMA_VERSION),
        ("connectivity", CONNECTIVITY_CERTIFICATE_SCHEMA_VERSION),
        ("boundary", BOUNDARY_CERTIFICATE_SCHEMA_VERSION),
        ("cluster_partition", CLUSTER_PARTITION_SCHEMA_VERSION),
    ):
        field = f"{name}_certificate_path" if name != "cluster_partition" else "cluster_partition_path"
        hash_field = f"{name}_certificate_sha256" if name != "cluster_partition" else "cluster_partition_sha256"
        path = Path(str(proof.get(field) or "")).resolve(strict=True)
        if (
            path.parent != root
            or manifest.get(field) != str(path)
            or manifest.get(hash_field) != proof.get(hash_field)
            or _sha256_file(path) != proof.get(hash_field)
        ):
            raise ExternalMemoryDBSCANError(
                f"terminal component-recovery certificate mismatch: {name}"
            )
        payload = _load_object(path)
        if (
            payload.get("schema_version") != schema
            or payload.get("status") != "PASS"
            or payload.get("recovery_schema_version")
            != COMPONENT_RECOVERY_SCHEMA_VERSION
            or payload.get("scientific_identity_sha256") != scientific_sha
            or int(payload.get("input_rows", -1)) != n_samples
            or payload.get("approximation_used") is not False
        ):
            raise ExternalMemoryDBSCANError(
                f"terminal component-recovery certificate incomplete: {name}"
            )
        loaded_certificates[name] = payload
    boundary = loaded_certificates["boundary"]
    all_core = loaded_certificates["all_core"]
    connectivity = loaded_certificates["connectivity"]
    partition = loaded_certificates["cluster_partition"]
    if (
        boundary.get("comparison") != "distance <= eps"
        or boundary.get("recheck_dtype") != "float64"
        or boundary.get("float64_revalidation_complete") is not True
        or int(boundary.get("float64_revalidated_row_count", -1)) != n_samples
        or boundary.get("sklearn_membership_exactly_matched") is not True
        or boundary.get("anchor_graph_float64_exactly_matched") is not True
        or int(boundary.get("uncertain_edges_accepted", -1)) != 0
        or all_core.get("all_points_core_proven") is not True
        or int(all_core.get("core_point_count", -1)) != n_samples
        or all_core.get("self_neighbor_counted_exactly_once") is not True
        or all_core.get("boundary_certificate_sha256")
        != proof.get("boundary_certificate_sha256")
        or connectivity.get("component_bridge_graph_exact") is not True
        or connectivity.get("exact_multicomponent_partition_proven") is not True
        or connectivity.get("all_core_certificate_sha256")
        != proof.get("all_core_certificate_sha256")
        or partition.get("sklearn_partition_semantics_preserved") is not True
        or int(partition.get("noise_count", -1)) != 0
        or int(partition.get("core_count", -1)) != n_samples
        or int(partition.get("cluster_count", -1))
        != int(manifest.get("cluster_count", -2))
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery certificate semantics are incomplete"
        )
    labels = _open_npy_memmap(paths["labels_path"], mode="r")
    core = _open_npy_memmap(paths["core_mask_path"], mode="r")
    lower = _open_npy_memmap(paths["core_lower_bounds_path"], mode="r")
    attachment = _open_npy_memmap(paths["attachment_components_path"], mode="r")
    _validate_partial(labels, shape=(n_samples,), dtype=np.intp, label="recovery labels")
    _validate_partial(core, shape=(n_samples,), dtype=np.bool_, label="recovery core")
    _validate_partial(lower, shape=(n_samples,), dtype=np.uint32, label="recovery lower")
    _validate_partial(
        attachment,
        shape=(n_samples,),
        dtype=np.uint32,
        label="recovery attachment",
    )
    if (
        not bool(np.all(core))
        or bool(np.any(lower < int(contract["min_samples"]) - 1))
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery all-core artifact failed"
        )
    component_path = Path(
        str(connectivity.get("anchor_initial_components_path") or "")
    ).resolve(strict=True)
    roots_path = Path(
        str(connectivity.get("final_component_roots_path") or "")
    ).resolve(strict=True)
    witness_path = Path(
        str(connectivity.get("bridge_witnesses_path") or "")
    ).resolve(strict=True)
    for path, expected in (
        (component_path, connectivity.get("anchor_initial_components_sha256")),
        (roots_path, connectivity.get("final_component_roots_sha256")),
        (witness_path, connectivity.get("bridge_witnesses_sha256")),
    ):
        if path.parent != root or _sha256_file(path) != expected:
            raise ExternalMemoryDBSCANError(
                "terminal component-recovery connectivity artifact mismatch"
            )
    component_by_anchor = np.load(component_path, allow_pickle=False)
    final_roots = np.load(roots_path, allow_pickle=False)
    anchor_edges = np.load(paths["anchor_edges_path"], allow_pickle=False)
    if (
        component_by_anchor.dtype != np.dtype(np.int32)
        or component_by_anchor.shape != (len(anchors),)
        or final_roots.dtype != np.dtype(np.int32)
        or final_roots.ndim != 1
        or bool(np.any(component_by_anchor < 0))
        or bool(np.any(component_by_anchor >= len(final_roots)))
        or bool(np.any(final_roots < 0))
        or bool(np.any(final_roots >= len(final_roots)))
        or not np.array_equal(final_roots[final_roots], final_roots)
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery component map is invalid"
        )
    for left, right in anchor_edges.tolist():
        if component_by_anchor[int(left)] != component_by_anchor[int(right)]:
            raise ExternalMemoryDBSCANError(
                "terminal anchor edge crosses initial components"
            )
    witnesses_payload = _load_object(witness_path)
    witnesses = witnesses_payload.get("witnesses")
    if (
        witnesses_payload.get("schema_version") != COMPONENT_RECOVERY_SCHEMA_VERSION
        or not isinstance(witnesses, list)
        or int(witnesses_payload.get("witness_count", -1)) != len(witnesses)
        or witnesses_payload.get("approximation_used") is not False
    ):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery witness artifact is invalid"
        )
    source = _open_npy_memmap(
        Path(str(scientific_identity["vectors_path"])), mode="r"
    )
    witness_parent = np.arange(len(final_roots), dtype=np.int32)
    selection = selection_manifest["selection_identity"]
    failures = np.load(Path(str(selection["failure_indices_path"])), allow_pickle=False)
    seed_set = {int(value) for value in selection["seed_indices"]}
    for row in witnesses:
        sample = int(row["sample_index"])
        position = int(np.searchsorted(failures, sample))
        if position < len(failures) and int(failures[position]) == sample:
            raise ExternalMemoryDBSCANError(
                "component bridge witness is a failure anchor"
            )
        left_local = int(row["left_anchor_local_index"])
        right_local = int(row["right_anchor_local_index"])
        left_distance = float(
            np.linalg.norm(
                np.asarray(source[sample], dtype=np.float64)
                - np.asarray(source[int(anchors[left_local])], dtype=np.float64)
            )
        )
        right_distance = float(
            np.linalg.norm(
                np.asarray(source[sample], dtype=np.float64)
                - np.asarray(source[int(anchors[right_local])], dtype=np.float64)
            )
        )
        seed_witnesses = row.get("seed_core_witnesses")
        if not isinstance(seed_witnesses, list):
            raise ExternalMemoryDBSCANError(
                "component bridge seed-core witness is absent"
            )
        observed_seed_globals: list[int] = []
        for seed_row in seed_witnesses:
            if not isinstance(seed_row, Mapping):
                raise ExternalMemoryDBSCANError(
                    "component bridge seed-core witness is invalid"
                )
            seed_global = int(seed_row["anchor_global_index"])
            seed_distance = float(
                np.linalg.norm(
                    np.asarray(source[sample], dtype=np.float64)
                    - np.asarray(source[seed_global], dtype=np.float64)
                )
            )
            if (
                seed_global not in seed_set
                or seed_global == sample
                or seed_distance.hex() != seed_row.get("distance_float64_hex")
                or seed_distance > float(contract["eps"])
            ):
                raise ExternalMemoryDBSCANError(
                    "component bridge seed-core edge failed exact recheck"
                )
            observed_seed_globals.append(seed_global)
        if (
            len(observed_seed_globals) < int(contract["min_samples"]) - 1
            or observed_seed_globals != sorted(set(observed_seed_globals))
        ):
            raise ExternalMemoryDBSCANError(
                "component bridge seed-core witness multiplicity is invalid"
            )
        if (
            left_distance.hex() != row["left_distance_float64_hex"]
            or right_distance.hex() != row["right_distance_float64_hex"]
            or left_distance > float(contract["eps"])
            or right_distance > float(contract["eps"])
            or int(component_by_anchor[left_local])
            != int(row["left_initial_component"])
            or int(component_by_anchor[right_local])
            != int(row["right_initial_component"])
        ):
            raise ExternalMemoryDBSCANError(
                "terminal component bridge witness failed exact recheck"
            )
        _component_union(
            witness_parent,
            int(row["left_initial_component"]),
            int(row["right_initial_component"]),
        )
    replayed_roots = np.asarray(
        [_component_find(witness_parent, value) for value in range(len(final_roots))],
        dtype=np.int32,
    )
    if not np.array_equal(replayed_roots, final_roots):
        raise ExternalMemoryDBSCANError(
            "terminal component bridge witness/root mismatch"
        )
    minimum_by_root: dict[int, int] = {}
    block = max(1, min(1_000_000, n_samples))
    expected_labels = np.empty(block, dtype=np.intp)
    for offset in range(0, n_samples, block):
        stop = min(n_samples, offset + block)
        roots = final_roots[np.asarray(attachment[offset:stop], dtype=np.intp)]
        for root_value in np.unique(roots).tolist():
            first = int(np.flatnonzero(roots == int(root_value))[0]) + offset
            minimum_by_root[int(root_value)] = min(
                minimum_by_root.get(int(root_value), n_samples), first
            )
    ordered = sorted(minimum_by_root, key=lambda value: minimum_by_root[value])
    label_by_root = {value: label for label, value in enumerate(ordered)}
    for offset in range(0, n_samples, block):
        stop = min(n_samples, offset + block)
        roots = final_roots[np.asarray(attachment[offset:stop], dtype=np.intp)]
        current = expected_labels[: stop - offset]
        current[:] = np.fromiter(
            (label_by_root[int(value)] for value in roots.tolist()),
            dtype=np.intp,
            count=stop - offset,
        )
        if not np.array_equal(np.asarray(labels[offset:stop]), current):
            raise ExternalMemoryDBSCANError(
                "terminal component-recovery canonical labels mismatch"
            )
    if partition.get("component_minimum_global_sample_indices") != [
        int(minimum_by_root[value]) for value in ordered
    ]:
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery cluster order mismatch"
        )
    if len(ordered) != int(manifest["cluster_count"]):
        raise ExternalMemoryDBSCANError(
            "terminal component-recovery cluster count mismatch"
        )
    del source, labels, core, lower, attachment
    return paths["labels_path"], paths["core_mask_path"], proof_path


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
    anchor_indices_override: np.ndarray | None = None,
    adaptive_selection_manifest: Mapping[str, Any] | None = None,
    resume_replay_required: bool = False,
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
    state = _load_checkpoint(state_path)
    phase = str(state.get("phase"))
    ledgers = _load_progress_ledgers(
        state, identity=identity, num_samples=n_samples
    )
    adaptive_checkpoint_fields: dict[str, Any] = {}
    if adaptive_selection_manifest is not None:
        adaptive_identity = adaptive_selection_manifest.get("selection_identity")
        if not isinstance(adaptive_identity, Mapping):
            raise ExternalMemoryDBSCANError(
                "adaptive anchor selection identity is absent"
            )
        adaptive_path = root / "adaptive_anchor_selection.json"
        adaptive_checkpoint_fields = {
            "adaptive_selection_manifest_path": str(adaptive_path),
            "adaptive_selection_manifest_sha256": _sha256_file(adaptive_path),
            "selected_anchor_indices_sha256": adaptive_identity[
                "selected_anchor_indices_sha256"
            ],
        }

    def shortcut_progress_extra() -> dict[str, Any]:
        extra = _progress_checkpoint_extra(ledgers, identity=identity)
        extra.update(adaptive_checkpoint_fields)
        return extra
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
    if phase not in {
        "shortcut_anchor_scan",
        "shortcut_finalize",
        COMPONENT_PRIMARY_PHASE,
        COMPONENT_EXPANSION_PHASE,
        COMPONENT_FINALIZE_PHASE,
    }:
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
        extra = shortcut_progress_extra()
        extra.update(
            {
                "shortcut_failure_path": failure["path"],
                "shortcut_failure_sha256": failure["sha256"],
                "shortcut_approximation_used": False,
            }
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase=next_phase,
            next_offset=0,
            peak_rss_bytes=peak_rss_bytes,
            extra=extra,
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
    if anchor_indices_override is None:
        anchor_indices = _deterministic_anchor_indices(
            n_samples, int(contract.shortcut_anchor_count)
        )
        anchor_selection_rule = "floor(i*(N-1)/(A-1)); endpoints included"
        selected_anchor_indices_sha = identity["shortcut_contract"][
            "selected_anchor_indices_sha256"
        ]
    else:
        anchor_indices = np.asarray(anchor_indices_override, dtype=np.intp)
        if adaptive_selection_manifest is None:
            raise ExternalMemoryDBSCANError(
                "adaptive anchors require their selection manifest"
            )
        selection_identity = adaptive_selection_manifest.get("selection_identity")
        if not isinstance(selection_identity, Mapping):
            raise ExternalMemoryDBSCANError(
                "adaptive anchor selection identity is absent"
            )
        anchor_selection_rule = str(selection_identity["anchor_selection_rule"])
        selected_anchor_indices_sha = str(
            selection_identity["selected_anchor_indices_sha256"]
        )
        if _sample_indices_sha256(anchor_indices) != selected_anchor_indices_sha:
            raise ExternalMemoryDBSCANError(
                "adaptive selected-anchor hash mismatch"
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
        if adaptive_selection_manifest is not None:
            return _fit_adaptive_disconnected_component_recovery(
                vectors=vectors,
                root=root,
                state_path=state_path,
                final_manifest_path=final_manifest_path,
                identity=identity,
                contract=contract,
                sklearn_version=sklearn_version,
                peak_rss_bytes=peak_rss_bytes,
                anchor_indices=anchor_indices,
                anchor_edges=anchor_edges,
                anchor_rows=anchor_rows,
                selection_manifest=adaptive_selection_manifest,
            )
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
    lower_ledger = _require_progress_ledger(
        ledgers, phase="shortcut_anchor_scan", identity=identity
    )
    if phase == "shortcut_anchor_scan":
        _assert_active_progress_offset(
            state, lower_ledger, phase="shortcut_anchor_scan"
        )
    if resume_replay_required and lower_ledger["entries"]:
        replay_path = lower_final if lower_final.exists() else lower_partial
        if not replay_path.exists():
            raise ExternalMemoryDBSCANError(
                "missing lower-bound array for resume replay"
            )
        replay_lower = _open_npy_memmap(replay_path, mode="r")
        _validate_partial(
            replay_lower,
            shape=(n_samples,),
            dtype=np.uint32,
            label="anchor-neighbor lower bounds",
        )
        replay_minimum, replay_non_anchor, replay_peak = _replay_lower_progress(
            vectors=vectors,
            lower=replay_lower,
            model=anchor_model,
            anchor_indices=anchor_indices,
            ledger=lower_ledger,
            min_samples=int(contract.min_samples),
            max_rss_bytes=int(contract.max_rss_bytes),
        )
        del replay_lower
        if (
            replay_minimum
            != state.get("minimum_distinct_anchor_neighbors_excluding_self")
            or replay_non_anchor
            != state.get("minimum_non_anchor_anchor_neighbors")
        ):
            raise ExternalMemoryDBSCANError(
                "shortcut lower-bound checkpoint minima/replay mismatch"
            )
        peak_rss_bytes = max(int(peak_rss_bytes), int(replay_peak))
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
        start = int(lower_ledger["committed_offset"])
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
        committed_payloads = [entry["payload"] for entry in lower_ledger["entries"]]
        minimum_lower = (
            min(
                int(payload["minimum_distinct_anchor_neighbors_excluding_self"])
                for payload in committed_payloads
            )
            if committed_payloads
            else int(len(anchor_indices))
        )
        committed_non_anchor = [
            int(payload["minimum_non_anchor_anchor_neighbors"])
            for payload in committed_payloads
            if payload.get("minimum_non_anchor_anchor_neighbors") is not None
        ]
        minimum_non_anchor_value = (
            min(committed_non_anchor) if committed_non_anchor else None
        )
        if lower_ledger["entries"] and (
            minimum_lower
            != state.get("minimum_distinct_anchor_neighbors_excluding_self")
            or minimum_non_anchor_value
            != state.get("minimum_non_anchor_anchor_neighbors")
        ):
            raise ExternalMemoryDBSCANError(
                "shortcut lower-bound checkpoint/ledger minima mismatch"
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
            block_lower, first_failure = _anchor_lower_block(
                model=anchor_model,
                vectors=vectors,
                start=offset,
                stop=stop,
                anchor_local_by_global=anchor_local_by_global,
                min_samples=int(contract.min_samples),
            )
            lower[offset:stop] = block_lower
            block_payload = _lower_block_payload(
                block_lower,
                start=offset,
                anchor_local_by_global=anchor_local_by_global,
            )
            block_minimum = int(
                block_payload["minimum_distinct_anchor_neighbors_excluding_self"]
            )
            minimum_lower = min(minimum_lower, block_minimum)
            block_non_anchor = block_payload["minimum_non_anchor_anchor_neighbors"]
            if block_non_anchor is not None:
                minimum_non_anchor_value = (
                    int(block_non_anchor)
                    if minimum_non_anchor_value is None
                    else min(minimum_non_anchor_value, int(block_non_anchor))
                )
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
            _append_progress_entry(
                lower_ledger,
                start=offset,
                stop=stop,
                payload=block_payload,
            )
            del block_lower
            blocks_since_checkpoint += 1
            if (
                blocks_since_checkpoint >= int(contract.checkpoint_interval_blocks)
                or stop == n_samples
            ):
                _fsync_memmap(lower)
                extra = shortcut_progress_extra()
                extra.update(
                    {
                        "effective_shortcut_query_block_size": int(block),
                        "anchor_indices_sha256": anchor_indices_sha,
                        "anchor_edges_sha256": anchor_edges_sha,
                        "minimum_distinct_anchor_neighbors_excluding_self": int(
                            minimum_lower
                        ),
                        "minimum_non_anchor_anchor_neighbors": minimum_non_anchor_value,
                    }
                )
                _checkpoint(
                    state_path,
                    identity=identity,
                    phase="shortcut_anchor_scan",
                    next_offset=stop,
                    peak_rss_bytes=peak,
                    extra=extra,
                )
                blocks_since_checkpoint = 0
        _fsync_memmap(lower)
        lower_sha = _sha256_file(lower_partial)
        _complete_progress_ledger(
            lower_ledger,
            num_samples=n_samples,
            result={
                "full_scan_complete": True,
                "minimum_distinct_anchor_neighbors_excluding_self": int(
                    minimum_lower
                ),
                "minimum_non_anchor_anchor_neighbors": minimum_non_anchor_value,
                "all_rows_core_lower_bound_pass": True,
                "all_non_anchors_attached": True,
            },
        )
        extra = shortcut_progress_extra()
        extra.update(
            {
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
            }
        )
        _checkpoint(
            state_path,
            identity=identity,
            phase="shortcut_finalize",
            next_offset=n_samples,
            peak_rss_bytes=peak,
            extra=extra,
        )
        del lower
        phase = "shortcut_finalize"
        state = _load_checkpoint(state_path)
        ledgers = _load_progress_ledgers(
            state, identity=identity, num_samples=n_samples
        )
        lower_ledger = _require_progress_ledger(
            ledgers, phase="shortcut_anchor_scan", identity=identity
        )

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

    lower_for_closure = _open_npy_memmap(lower_final, mode="r")
    validated_minimum, validated_non_anchor = _validate_complete_lower_witness(
        lower=lower_for_closure,
        anchor_indices=anchor_indices,
        ledger=lower_ledger,
        num_samples=n_samples,
        min_samples=int(contract.min_samples),
    )
    del lower_for_closure
    if (
        validated_minimum
        != state.get("minimum_distinct_anchor_neighbors_excluding_self")
        or validated_non_anchor
        != state.get("minimum_non_anchor_anchor_neighbors")
    ):
        raise ExternalMemoryDBSCANError(
            "shortcut final lower-bound checkpoint/full-scan mismatch"
        )
    vector_source = Path(str(identity["vectors_path"])).resolve(strict=True)
    _verify_source_identity(
        vector_source,
        expected_sha256=str(identity["vectors_sha256"]),
        expected_stat=identity["vectors_stat_identity"],
        phase="shortcut PASS",
    )
    required_progress_phases = (
        ["adaptive_seed_scan", "adaptive_failure_scan", "shortcut_anchor_scan"]
        if contract.shortcut_mode == ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
        else ["shortcut_anchor_scan"]
    )
    progress_path = root / "shortcut_progress_ledger.json"
    progress_sha = _write_progress_ledger_artifact(
        path=progress_path,
        ledgers=ledgers,
        identity=identity,
        required_phases=required_progress_phases,
        num_samples=n_samples,
    )

    labels_path = root / "labels.npy"
    core_path = root / "core_mask.npy"
    _constant_npy(
        labels_path, shape=(n_samples,), dtype=np.intp, fill_value=0
    )
    _constant_npy(
        core_path, shape=(n_samples,), dtype=np.bool_, fill_value=True
    )
    _validate_constant_output(
        labels_path,
        shape=(n_samples,),
        dtype=np.intp,
        expected=0,
        label="shortcut labels",
    )
    _validate_constant_output(
        core_path,
        shape=(n_samples,),
        dtype=np.bool_,
        expected=True,
        label="shortcut core mask",
    )
    labels_sha = _sha256_file(labels_path)
    core_sha = _sha256_file(core_path)
    lower_sha = _sha256_file(lower_final)
    split_certificates = _write_shortcut_split_certificates(
        root=root,
        identity=identity,
        vectors=vectors,
        anchor_indices=anchor_indices,
        anchor_indices_path=anchor_indices_path,
        anchor_indices_sha256=anchor_indices_sha,
        anchor_edges=anchor_edges,
        anchor_edges_path=anchor_edges_path,
        anchor_edges_sha256=anchor_edges_sha,
        lower_path=lower_final,
        lower_sha256=lower_sha,
        labels_path=labels_path,
        labels_sha256=labels_sha,
        core_path=core_path,
        core_sha256=core_sha,
        contract=contract,
    )
    proof_path = root / "shortcut_proof.json"
    proof = {
        "schema_version": SHORTCUT_PROOF_SCHEMA_VERSION,
        "status": "PASS",
        "shortcut": contract.shortcut_mode,
        "scientific_identity_sha256": _stable_hash(identity),
        "vectors_path": str(Path(identity["vectors_path"])),
        "vectors_sha256": identity["vectors_sha256"],
        "vectors_stat_identity": identity["vectors_stat_identity"],
        "vectors_dtype": identity["vectors_dtype"],
        "vectors_shape": identity["vectors_shape"],
        "sklearn_version": sklearn_version,
        "distance_kernel": "sklearn NearestNeighbors brute euclidean",
        "epsilon_comparison": "distance <= float(eps)",
        "eps": float(contract.eps),
        "min_samples": int(contract.min_samples),
        "self_count_semantics": "each sample index counts itself exactly once",
        "duplicate_semantics": "duplicate vectors remain distinct sample indices",
        "anchor_selection": anchor_selection_rule,
        "selected_anchor_indices_sha256": selected_anchor_indices_sha,
        "anchor_indices_path": str(anchor_indices_path),
        "anchor_indices_sha256": anchor_indices_sha,
        "anchor_count": int(len(anchor_indices)),
        "anchor_edges_path": str(anchor_edges_path),
        "anchor_edges_sha256": anchor_edges_sha,
        "anchor_edge_count": int(len(anchor_edges)),
        "anchor_epsilon_graph_connected": True,
        "anchor_neighbor_lower_bounds_path": str(lower_final),
        "anchor_neighbor_lower_bounds_sha256": lower_sha,
        "progress_ledger_path": str(progress_path),
        "progress_ledger_sha256": progress_sha,
        "progress_ledger_required_phases": required_progress_phases,
        "all_progress_prefixes_complete": True,
        "anchor_neighbor_lower_bound_definition": (
            "distinct anchor sample indices within eps, excluding the query's "
            "own sample index when it is an anchor"
        ),
        "minimum_distinct_anchor_neighbors_excluding_self": int(
            validated_minimum
        ),
        "minimum_non_anchor_anchor_neighbors": validated_non_anchor,
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
        **split_certificates,
        "completed_at": _utc_now(),
    }
    if adaptive_selection_manifest is not None:
        proof.update(
            {
                "adaptive_selection_manifest_path": str(
                    root / "adaptive_anchor_selection.json"
                ),
                "adaptive_selection_manifest_sha256": _sha256_file(
                    root / "adaptive_anchor_selection.json"
                ),
                "adaptive_selection_identity_sha256": adaptive_selection_manifest[
                    "selection_identity_sha256"
                ],
                "first_pass_failure_indices_sha256": adaptive_selection_manifest[
                    "selection_identity"
                ]["failure_indices_sha256"],
                "first_pass_failure_index_list_sha256": adaptive_selection_manifest[
                    "selection_identity"
                ]["failure_index_list_sha256"],
                "selected_anchor_rows_sha256": adaptive_selection_manifest[
                    "selection_identity"
                ]["anchor_rows_sha256"],
                "second_pass_complete": True,
            }
        )
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
        **split_certificates,
        "progress_ledger_path": str(progress_path),
        "progress_ledger_sha256": progress_sha,
        "clustering_path": contract.shortcut_mode,
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
    _assert_source_stat_identity(
        vector_source,
        expected_stat=identity["vectors_stat_identity"],
        phase="shortcut manifest publication",
    )
    _atomic_json(final_manifest_path, manifest)
    complete_extra = shortcut_progress_extra()
    complete_extra.update(
        {
            "shortcut_proof_sha256": proof_sha,
            "progress_ledger_sha256": progress_sha,
            "labels_sha256": labels_sha,
            "core_mask_sha256": core_sha,
        }
    )
    _checkpoint(
        state_path,
        identity=identity,
        phase="complete",
        next_offset=n_samples,
        peak_rss_bytes=int(manifest["peak_rss_bytes_observed"]),
        extra=complete_extra,
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
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ExternalMemoryDBSCANError("terminal DBSCAN schema mismatch")
        if manifest.get("run_complete") is not True:
            raise ExternalMemoryDBSCANError("terminal DBSCAN manifest is incomplete")
        scientific_identity = manifest.get("scientific_identity")
        if not isinstance(scientific_identity, Mapping):
            raise ExternalMemoryDBSCANError("terminal DBSCAN identity is absent")
        if scientific_identity.get("schema_version") != SCHEMA_VERSION:
            raise ExternalMemoryDBSCANError(
                "terminal DBSCAN scientific schema mismatch"
            )
        if scientific_identity.get("vectors_path") != str(source):
            raise ExternalMemoryDBSCANError("terminal DBSCAN vector path mismatch")
        if scientific_identity.get("contract") != asdict(contract):
            raise ExternalMemoryDBSCANError("terminal DBSCAN contract mismatch")
        expected_source_stat = scientific_identity.get("vectors_stat_identity")
        if not isinstance(expected_source_stat, Mapping):
            raise ExternalMemoryDBSCANError(
                "terminal DBSCAN vector stat identity is absent"
            )
        _verify_source_identity(
            source,
            expected_sha256=str(scientific_identity.get("vectors_sha256") or ""),
            expected_stat=expected_source_stat,
            phase="terminal reopen",
        )
        actual_source_sha = str(scientific_identity["vectors_sha256"])
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
        if manifest.get("clustering_path") == ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY:
            labels_path, core_path, shortcut_path = (
                _validate_component_recovery_closure(
                    manifest=manifest, root=root
                )
            )
            counts_path = None
        elif manifest.get("clustering_path") in {
            ALL_CORE_ONE_COMPONENT_SHORTCUT,
            ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        }:
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
        _assert_source_stat_identity(
            source,
            expected_stat=expected_source_stat,
            phase="terminal result return",
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
    checkpoint_existed_at_entry = state_path.exists()

    actual_vectors_sha, vectors_stat_identity = _hash_source_with_stable_stat(source)
    if expected_vectors_sha256 and actual_vectors_sha != expected_vectors_sha256:
        raise ExternalMemoryDBSCANError("recourse vector SHA256 mismatch")
    vectors = _open_npy_memmap(source, mode="r")
    _assert_source_stat_identity(
        source,
        expected_stat=vectors_stat_identity,
        phase="vector mmap validation",
    )
    if vectors.ndim != 2 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ExternalMemoryDBSCANError("vectors must be a nonempty 2-D ndarray")
    if vectors.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ExternalMemoryDBSCANError(
            f"vectors must preserve float32/float64 semantics, got {vectors.dtype}"
        )
    n_samples, n_features = (int(vectors.shape[0]), int(vectors.shape[1]))
    neighbors, sklearn_version, fit_method = _fit_neighbors(
        vectors, eps=float(contract.eps)
    )
    if sklearn_version != contract.expected_sklearn_version:
        raise ExternalMemoryDBSCANError(
            "SKLEARN_VERSION_MISMATCH:"
            f"actual={sklearn_version}:expected={contract.expected_sklearn_version}"
        )
    if contract.shortcut_mode == ALL_CORE_ONE_COMPONENT_SHORTCUT:
        shortcut_identity = {
            "mode": contract.shortcut_mode,
            "anchor_count": int(contract.shortcut_anchor_count),
            "query_block_size": int(contract.shortcut_query_block_size),
            "exact_fallback_max_samples": int(
                contract.exact_fallback_max_samples
            ),
            "anchor_selection": "floor(i*(N-1)/(A-1)); endpoints included",
            "selected_anchor_indices_sha256": _sample_indices_sha256(
                _deterministic_anchor_indices(
                    n_samples, int(contract.shortcut_anchor_count)
                )
            ),
        }
    elif contract.shortcut_mode == ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT:
        shortcut_identity = {
            "mode": contract.shortcut_mode,
            "seed_count": int(contract.shortcut_seed_count),
            "failure_cap": int(contract.shortcut_failure_cap),
            "query_block_size": int(contract.shortcut_query_block_size),
            "exact_fallback_max_samples": int(
                contract.exact_fallback_max_samples
            ),
            "seed_selection_algorithm": (
                "global_k_minimum_squared_l2_float64_tie_sample_index_v1"
            ),
            "failure_selection_rule": (
                "all_first_pass_insufficient_seed_lower_bound_indices_v1"
            ),
            "anchor_selection_rule": (
                "sorted_unique_union_of_seed_and_all_failure_indices_v1"
            ),
            "selected_anchor_indices_sha256": None,
        }
    else:
        shortcut_identity = {
            "mode": SHORTCUT_DISABLED,
            "query_block_size": int(contract.shortcut_query_block_size),
            "exact_fallback_max_samples": int(
                contract.exact_fallback_max_samples
            ),
        }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "vectors_path": str(source),
        "vectors_sha256": actual_vectors_sha,
        "vectors_stat_identity": vectors_stat_identity,
        "vectors_dtype": str(vectors.dtype),
        "vectors_shape": [n_samples, n_features],
        "contract": asdict(contract),
        "sklearn_version": sklearn_version,
        "nearest_neighbors_fit_method": fit_method,
        "nearest_neighbors_metric": "euclidean",
        "nearest_neighbors_algorithm": "auto",
        "border_assignment": "minimum_cluster_label_of_adjacent_core_component",
        "shortcut_contract": shortcut_identity,
    }
    state: dict[str, Any]
    if state_path.exists():
        state = _load_checkpoint(state_path)
        if not resume:
            raise ExternalMemoryDBSCANError("checkpoint exists but resume=false")
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("identity_sha256") != _stable_hash(identity)
            or state.get("identity") != identity
        ):
            raise ExternalMemoryDBSCANError("checkpoint scientific identity mismatch")
    else:
        initial_phase = {
            ALL_CORE_ONE_COMPONENT_SHORTCUT: "shortcut_anchor_scan",
            ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT: "adaptive_seed_scan",
        }.get(contract.shortcut_mode, "neighbor_counts")
        state = {
            "phase": initial_phase,
            "next_offset": 0,
            "peak_rss_bytes": _rss_bytes(),
        }
        initial_extra: dict[str, Any] = {}
        if contract.shortcut_mode in {
            ALL_CORE_ONE_COMPONENT_SHORTCUT,
            ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        }:
            initial_ledgers = {
                initial_phase: _new_progress_ledger(
                    phase=initial_phase, identity=identity
                )
            }
            initial_extra = _progress_checkpoint_extra(
                initial_ledgers, identity=identity
            )
            if initial_phase == "adaptive_seed_scan":
                initial_extra["adaptive_seed_candidates"] = []
        _checkpoint(
            state_path,
            identity=identity,
            phase=str(state["phase"]),
            next_offset=0,
            peak_rss_bytes=int(state["peak_rss_bytes"]),
            extra=initial_extra,
        )

    peak = max(int(state.get("peak_rss_bytes", 0)), _rss_bytes())
    shortcut_active_phases = {
        "adaptive_seed_scan",
        "adaptive_failure_scan",
        "shortcut_anchor_scan",
        "shortcut_finalize",
        "shortcut_blocked",
        COMPONENT_PRIMARY_PHASE,
        COMPONENT_EXPANSION_PHASE,
        COMPONENT_FINALIZE_PHASE,
    }
    if contract.shortcut_mode in {
        ALL_CORE_ONE_COMPONENT_SHORTCUT,
        ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    } and str(state.get("phase")) in shortcut_active_phases:
        adaptive_anchors: np.ndarray | None = None
        adaptive_manifest: Mapping[str, Any] | None = None
        if contract.shortcut_mode == ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT:
            adaptive_anchors, adaptive_manifest, peak = (
                _resolve_adaptive_anchor_selection(
                    vectors=vectors,
                    root=root,
                    state_path=state_path,
                    identity=identity,
                    contract=contract,
                    sklearn_version=sklearn_version,
                    peak_rss_bytes=peak,
                    resume_replay_required=checkpoint_existed_at_entry,
                )
            )
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
            anchor_indices_override=adaptive_anchors,
            adaptive_selection_manifest=adaptive_manifest,
            resume_replay_required=checkpoint_existed_at_entry,
        )
        if shortcut_result is not None:
            return shortcut_result
        state = _load_checkpoint(state_path)
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
        state = _load_checkpoint(state_path)
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
        state = _load_checkpoint(state_path)

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
        state = _load_checkpoint(state_path)

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
        state = _load_checkpoint(state_path)
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
    _verify_source_identity(
        source,
        expected_sha256=actual_vectors_sha,
        expected_stat=vectors_stat_identity,
        phase="exact DBSCAN PASS",
    )
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
    _assert_source_stat_identity(
        source,
        expected_stat=vectors_stat_identity,
        phase="exact DBSCAN manifest publication",
    )
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
    "ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY",
    "ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT",
    "ADAPTIVE_SELECTION_SCHEMA_VERSION",
    "ALL_CORE_ONE_COMPONENT_SHORTCUT",
    "ExternalDBSCANContract",
    "ExternalDBSCANResult",
    "ExternalMemoryDBSCANError",
    "PROGRESS_LEDGER_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SHORTCUT_DISABLED",
    "SHORTCUT_PROOF_SCHEMA_VERSION",
    "fit_external_memory_dbscan",
]
