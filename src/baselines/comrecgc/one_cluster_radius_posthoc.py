"""Post-hoc scalar-boundary audit for a terminal one-cluster summary.

This module never clusters.  It reopens the hash-closed terminal summary and
compares the historical widened NumPy trace mask with the corrected
distance-dtype mask and the official Torch mask in the terminal block order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import numpy as np

from .close_pair_view import validate_theta_close_pair_view
from .external_memory_dbscan import ExternalMemoryDBSCANError
from .external_memory_recourse import (
    _validate_exact_one_cluster_source,
    validate_proven_one_cluster_summary,
)


SCHEMA_VERSION = "comrecgc_one_cluster_radius_posthoc_v1"
MASK_DIGEST_CONTRACT = "sha256(bool-c-order-data-bytes)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalMemoryDBSCANError(f"invalid post-hoc JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExternalMemoryDBSCANError(f"post-hoc JSON is not an object: {path}")
    return value


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    if path.is_symlink() or not stat.S_ISREG(value.st_mode):
        raise ExternalMemoryDBSCANError(
            f"post-hoc source is not a physical regular file: {path}"
        )
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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_marker(root: Path, name: str, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / name)
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _update_sets(
    pairs: Any,
    mask: np.ndarray,
    *,
    parents: set[int],
    candidates: set[int],
    first_by_parent: dict[int, int] | None = None,
) -> None:
    if not bool(np.any(mask)):
        return
    selected = np.asarray(pairs[mask], dtype=np.int64)
    parents.update(map(int, np.unique(selected[:, 0]).tolist()))
    candidates.update(map(int, np.unique(selected[:, 1]).tolist()))
    if first_by_parent is not None:
        unique_parents, first = np.unique(selected[:, 0], return_index=True)
        for parent, local in zip(unique_parents.tolist(), first.tolist()):
            first_by_parent.setdefault(int(parent), int(selected[int(local), 1]))


@dataclass
class _MaskState:
    digest: Any
    count: int
    vector_sum: np.ndarray
    parents: set[int]
    candidates: set[int]
    first_by_parent: dict[int, int]
    exactly_at_delta: int = 0


def _new_state(num_features: int, dtype: np.dtype[Any]) -> _MaskState:
    return _MaskState(
        digest=hashlib.sha256(),
        count=0,
        vector_sum=np.zeros(num_features, dtype=dtype),
        parents=set(),
        candidates=set(),
        first_by_parent={},
    )


def _fold_mask(
    state: _MaskState,
    *,
    mask: np.ndarray,
    vectors: np.ndarray,
    pairs: Any,
    exactly: np.ndarray | None = None,
) -> None:
    mask = np.asarray(mask, dtype=np.bool_)
    state.digest.update(mask.tobytes(order="C"))
    selected = np.asarray(vectors[mask])
    if len(selected):
        state.vector_sum = np.add(
            state.vector_sum,
            np.sum(selected, axis=0, dtype=vectors.dtype),
            dtype=vectors.dtype,
        )
        state.count += int(len(selected))
    _update_sets(
        pairs,
        mask,
        parents=state.parents,
        candidates=state.candidates,
        first_by_parent=state.first_by_parent,
    )
    if exactly is not None:
        state.exactly_at_delta += int(np.count_nonzero(exactly))


def _center(state: _MaskState, dtype: np.dtype[Any]) -> np.ndarray | None:
    if state.count == 0:
        return None
    return np.asarray(
        state.vector_sum / np.asarray(state.count, dtype=dtype), dtype=dtype
    )


def _medoid(
    *,
    vectors: np.ndarray,
    pairs: Any,
    full_center: np.ndarray,
    radius: float,
    retained_center: np.ndarray | None,
    block_size: int,
    widened: bool,
) -> dict[str, Any] | None:
    if retained_center is None:
        return None
    winner = -1
    winner_distance = float("inf")
    for offset in range(0, len(vectors), block_size):
        stop = min(len(vectors), offset + block_size)
        block = np.asarray(vectors[offset:stop])
        distances = np.linalg.norm(block - full_center, axis=1)
        mask = (
            distances.astype(np.float64, copy=False) < float(radius)
            if widened
            else distances < np.asarray(float(radius), dtype=distances.dtype)
        )
        local = np.flatnonzero(mask)
        if not len(local):
            continue
        selected = block[local]
        retained_distances = np.linalg.norm(selected - retained_center, axis=1)
        local_winner = int(np.argmin(retained_distances))
        local_distance = float(retained_distances[local_winner])
        if local_distance < winner_distance:
            winner = int(offset + local[local_winner])
            winner_distance = local_distance
    if winner < 0:
        raise ExternalMemoryDBSCANError("post-hoc medoid replay lost retained rows")
    parent, candidate = pairs[winner]
    return {
        "position": winner,
        "parent_index": int(parent),
        "candidate_index": int(candidate),
        "distance": winner_distance,
        "distance_hex": winner_distance.hex(),
    }


def _selected_trace(
    *,
    state: _MaskState,
    medoid: Mapping[str, Any] | None,
    cluster_size: int,
    centroid_norm: float,
    radius: float,
    theta: float,
) -> list[dict[str, Any]]:
    if medoid is None or not state.parents or not centroid_norm < theta:
        return []
    return [
        {
            "rank": 1,
            "selected_rank": 1,
            "cluster_label": 0,
            "cluster_id": 0,
            "cluster_center_norm": centroid_norm,
            "centroid_norm": centroid_norm,
            "cluster_radius": radius,
            "cluster_size": cluster_size,
            "representative_source_index": int(medoid["parent_index"]),
            "representative_counterfactual_index": int(
                medoid["candidate_index"]
            ),
            "representative_distance_to_center": float(medoid["distance"]),
            "covered_parent_indices_native": sorted(state.parents),
            "native_cumulative_covered_count": len(state.parents),
            "cumulative_covered_count": len(state.parents),
            "native_cumulative_cost": centroid_norm,
            "member_counterfactual_indices": sorted(state.candidates),
            "representative_candidate_ids": [int(medoid["candidate_index"])],
        }
    ]


def run_one_cluster_radius_posthoc_audit(
    *,
    terminal_manifest_path: str | Path,
    expected_terminal_manifest_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Audit a c766 terminal summary without mutating or reclustering it."""

    terminal_path = Path(terminal_manifest_path).expanduser().resolve(strict=True)
    terminal_stat = _stat_identity(terminal_path)
    if _sha256_file(terminal_path) != expected_terminal_manifest_sha256:
        raise ExternalMemoryDBSCANError("terminal one-cluster manifest SHA mismatch")
    validated = validate_proven_one_cluster_summary(terminal_path)
    manifest = _load_object(terminal_path)
    identity = manifest["scientific_identity"]
    vectors_path = Path(identity["vectors_path"]).resolve(strict=True)
    vectors_stat_before = _stat_identity(vectors_path)
    vectors_sha = _sha256_file(vectors_path)
    if (
        vectors_sha != identity.get("vectors_sha256")
        or _stat_identity(vectors_path) != vectors_stat_before
    ):
        raise ExternalMemoryDBSCANError("terminal vector source SHA mismatch")
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != tuple(identity["vectors_shape"]):
        raise ExternalMemoryDBSCANError("terminal vector source shape mismatch")
    dbscan_path = Path(identity["dbscan_manifest_path"]).resolve(strict=True)
    if _sha256_file(dbscan_path) != identity["dbscan_manifest_sha256"]:
        raise ExternalMemoryDBSCANError("terminal DBSCAN manifest SHA mismatch")
    _validate_exact_one_cluster_source(
        dbscan_manifest_path=dbscan_path,
        dbscan_manifest_sha256=identity["dbscan_manifest_sha256"],
        recourse_vectors=vectors,
    )
    pairs_storage = identity.get("pairs_storage")
    if pairs_storage == "physical_npy":
        pairs_path = Path(identity["pairs_path"]).resolve(strict=True)
        pair_source_path = pairs_path
        pairs = np.load(pairs_path, mmap_mode="r", allow_pickle=False)
        pair_authority = {
            "storage": pairs_storage,
            "path": str(pairs_path),
            "sha256": identity["pairs_sha256"],
        }
    elif pairs_storage == "implicit_cartesian_v1":
        authority_path = Path(identity["pair_authority_manifest_path"])
        pair_source_path = authority_path.resolve(strict=True)
        view = validate_theta_close_pair_view(
            authority_path,
            require_dbscan_eligible=True,
            require_pair_semantics_authority=True,
        )
        if view.pairs_sha256 != identity["pairs_sha256"]:
            raise ExternalMemoryDBSCANError("terminal implicit pair SHA mismatch")
        pairs = view.open_pairs()
        pair_authority = {
            "storage": pairs_storage,
            "path": str(authority_path.resolve(strict=True)),
            "sha256": identity["pair_authority_manifest_sha256"],
        }
    else:
        raise ExternalMemoryDBSCANError("terminal pair storage is unsupported")
    if len(pairs) != len(vectors):
        raise ExternalMemoryDBSCANError("terminal vectors/pairs are not aligned")

    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"post-hoc output already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    numpy_center_path = Path(manifest["numpy_centroid_path"]).resolve(strict=True)
    torch_center_path = Path(manifest["torch_centroid_path"]).resolve(strict=True)
    live_mask_path = validated.retained_mask_path.resolve(strict=True)
    source_stats_before = {
        "terminal_manifest": terminal_stat,
        "vectors": vectors_stat_before,
        "dbscan_manifest": _stat_identity(dbscan_path),
        "pair_source": _stat_identity(pair_source_path),
        "numpy_centroid": _stat_identity(numpy_center_path),
        "torch_centroid": _stat_identity(torch_center_path),
        "live_retained_mask": _stat_identity(live_mask_path),
    }
    small_source_hashes = {
        "terminal_manifest": expected_terminal_manifest_sha256,
        "dbscan_manifest": identity["dbscan_manifest_sha256"],
        "pair_source": _sha256_file(pair_source_path),
        "numpy_centroid": _sha256_file(numpy_center_path),
        "torch_centroid": _sha256_file(torch_center_path),
        "live_retained_mask": _sha256_file(live_mask_path),
    }
    numpy_center = np.load(numpy_center_path, allow_pickle=False)
    torch_center = np.load(torch_center_path, allow_pickle=False)
    live_mask = np.load(
        live_mask_path, mmap_mode="r", allow_pickle=False
    )
    block_size = int(identity["block_size"])
    radius = float(identity["radius"])
    theta = float(identity["theta"])
    old = _new_state(vectors.shape[1], vectors.dtype)
    corrected = _new_state(vectors.shape[1], vectors.dtype)
    official = _new_state(vectors.shape[1], vectors.dtype)
    live_digest = hashlib.sha256()
    old_vs_corrected = 0
    corrected_vs_official = 0
    old_vs_official = 0
    import torch

    if str(torch.__version__) != identity.get("torch_version"):
        raise ExternalMemoryDBSCANError("terminal Torch version mismatch")
    torch_center_tensor = torch.from_numpy(torch_center)
    for offset in range(0, len(vectors), block_size):
        stop = min(len(vectors), offset + block_size)
        block = np.asarray(vectors[offset:stop])
        block_pairs = pairs[offset:stop]
        distances = np.linalg.norm(block - numpy_center, axis=1)
        radius_scalar = np.asarray(radius, dtype=distances.dtype)
        old_mask = distances.astype(np.float64, copy=False) < radius
        corrected_mask = distances < radius_scalar
        exact_corrected = distances == radius_scalar
        torch_distances = torch.norm(
            torch.from_numpy(block) - torch_center_tensor, dim=-1
        )
        official_mask = (torch_distances < radius).detach().cpu().numpy()
        exact_official = (torch_distances == radius).detach().cpu().numpy()
        _fold_mask(old, mask=old_mask, vectors=block, pairs=block_pairs)
        _fold_mask(
            corrected,
            mask=corrected_mask,
            vectors=block,
            pairs=block_pairs,
            exactly=exact_corrected,
        )
        _fold_mask(
            official,
            mask=official_mask,
            vectors=block,
            pairs=block_pairs,
            exactly=exact_official,
        )
        live_digest.update(
            np.asarray(live_mask[offset:stop], dtype=np.bool_).tobytes(order="C")
        )
        old_vs_corrected += int(np.count_nonzero(old_mask != corrected_mask))
        corrected_vs_official += int(
            np.count_nonzero(corrected_mask != official_mask)
        )
        old_vs_official += int(np.count_nonzero(old_mask != official_mask))

    if live_digest.hexdigest() != old.digest.hexdigest():
        raise ExternalMemoryDBSCANError(
            "terminal retained mask does not replay historical widened semantics"
        )
    if (
        official.count != int(manifest["within_centroid_radius_count"])
        or official.exactly_at_delta != int(manifest["count_exactly_at_delta"])
        or sorted(official.parents)
        != manifest["official_covered_parent_indices"]
        or sorted(official.candidates)
        != manifest["official_radius_counterfactual_indices"]
        or sorted(set(official.first_by_parent.values()))
        != manifest["official_first_counterfactual_indices"]
    ):
        raise ExternalMemoryDBSCANError("official Torch terminal replay mismatch")

    old_center = _center(old, vectors.dtype)
    corrected_center = _center(corrected, vectors.dtype)
    old_medoid = _medoid(
        vectors=vectors,
        pairs=pairs,
        full_center=numpy_center,
        radius=radius,
        retained_center=old_center,
        block_size=block_size,
        widened=True,
    )
    corrected_medoid = _medoid(
        vectors=vectors,
        pairs=pairs,
        full_center=numpy_center,
        radius=radius,
        retained_center=corrected_center,
        block_size=block_size,
        widened=False,
    )
    centroid_norm = float(np.linalg.norm(numpy_center))
    old_selected = _selected_trace(
        state=old,
        medoid=old_medoid,
        cluster_size=len(vectors),
        centroid_norm=centroid_norm,
        radius=radius,
        theta=theta,
    )
    corrected_selected = _selected_trace(
        state=corrected,
        medoid=corrected_medoid,
        cluster_size=len(vectors),
        centroid_norm=centroid_norm,
        radius=radius,
        theta=theta,
    )
    if old_selected != manifest["selected"]:
        raise ExternalMemoryDBSCANError("terminal selected trace does not replay")
    corrected_trace = {
        "schema_version": SCHEMA_VERSION,
        "source_terminal_manifest": str(terminal_path),
        "source_terminal_manifest_sha256": expected_terminal_manifest_sha256,
        "semantics": "numpy_distance < radius_cast_to_distance_dtype",
        "retained_count": corrected.count,
        "retained_mask_raw_sha256": corrected.digest.hexdigest(),
        "covered_parent_indices": sorted(corrected.parents),
        "counterfactual_indices": sorted(corrected.candidates),
        "retained_centroid": (
            None if corrected_center is None else corrected_center.tolist()
        ),
        "medoid": corrected_medoid,
        "selected": corrected_selected,
        "dbscan_recomputed": False,
        "close_filter_recomputed": False,
    }
    _atomic_json(root / "corrected_downstream_trace.json", corrected_trace)
    diff_free = old_vs_corrected == 0
    status = "PASS" if diff_free else "BLOCKED_BOUNDARY_DIFF"
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_complete": True,
        "terminal_manifest_path": str(terminal_path),
        "terminal_manifest_sha256": expected_terminal_manifest_sha256,
        "vectors_path": str(vectors_path),
        "vectors_sha256": vectors_sha,
        "source_stats_before": source_stats_before,
        "source_stats_after": {
            "terminal_manifest": _stat_identity(terminal_path),
            "vectors": _stat_identity(vectors_path),
            "dbscan_manifest": _stat_identity(dbscan_path),
            "pair_source": _stat_identity(pair_source_path),
            "numpy_centroid": _stat_identity(numpy_center_path),
            "torch_centroid": _stat_identity(torch_center_path),
            "live_retained_mask": _stat_identity(live_mask_path),
        },
        "small_source_hashes": small_source_hashes,
        "pair_authority": pair_authority,
        "block_size": block_size,
        "radius": radius,
        "theta": theta,
        "mask_digest_contract": MASK_DIGEST_CONTRACT,
        "live_retained_mask_file_sha256": _sha256_file(
            live_mask_path
        ),
        "live_retained_mask_raw_sha256": live_digest.hexdigest(),
        "old_widened_mask_raw_sha256": old.digest.hexdigest(),
        "dtype_cast_mask_raw_sha256": corrected.digest.hexdigest(),
        "official_torch_mask_raw_sha256": official.digest.hexdigest(),
        "old_vs_dtype_cast_diff_count": old_vs_corrected,
        "dtype_cast_vs_official_diff_count": corrected_vs_official,
        "old_vs_official_diff_count": old_vs_official,
        "old_vs_dtype_cast_parent_sets_equal": old.parents == corrected.parents,
        "old_vs_dtype_cast_candidate_sets_equal": (
            old.candidates == corrected.candidates
        ),
        "old_vs_dtype_cast_medoid_equal": old_medoid == corrected_medoid,
        "old_vs_dtype_cast_selected_trace_equal": (
            old_selected == corrected_selected
        ),
        "dtype_cast_vs_official_parent_sets_equal": (
            corrected.parents == official.parents
        ),
        "dtype_cast_vs_official_candidate_sets_equal": (
            corrected.candidates == official.candidates
        ),
        "old_widened": {
            "retained_count": old.count,
            "covered_parent_indices": sorted(old.parents),
            "counterfactual_indices": sorted(old.candidates),
            "medoid": old_medoid,
            "selected": old_selected,
        },
        "dtype_cast": {
            "retained_count": corrected.count,
            "count_exactly_at_delta": corrected.exactly_at_delta,
            "covered_parent_indices": sorted(corrected.parents),
            "counterfactual_indices": sorted(corrected.candidates),
            "medoid": corrected_medoid,
            "selected": corrected_selected,
        },
        "official_torch": {
            "retained_count": official.count,
            "count_exactly_at_delta": official.exactly_at_delta,
            "covered_parent_indices": sorted(official.parents),
            "counterfactual_indices": sorted(official.candidates),
        },
        "live_output_adoptable": diff_free,
        "final_standardization_blocked": not diff_free,
        "recommended_action": (
            "adopt_live_terminal_without_dbscan_rerun"
            if diff_free
            else "fresh_downstream_only_replay_from_existing_dbscan_manifest"
        ),
        "corrected_downstream_trace_path": str(
            root / "corrected_downstream_trace.json"
        ),
        "corrected_downstream_trace_sha256": _sha256_file(
            root / "corrected_downstream_trace.json"
        ),
        "dbscan_recomputed": False,
        "close_filter_recomputed": False,
        "completed_at": _utc_now(),
    }
    if audit["source_stats_before"] != audit["source_stats_after"]:
        raise ExternalMemoryDBSCANError("source changed during post-hoc audit")
    for name, expected_sha in small_source_hashes.items():
        source_path = {
            "terminal_manifest": terminal_path,
            "dbscan_manifest": dbscan_path,
            "pair_source": pair_source_path,
            "numpy_centroid": numpy_center_path,
            "torch_centroid": torch_center_path,
            "live_retained_mask": live_mask_path,
        }[name]
        if _sha256_file(source_path) != expected_sha:
            raise ExternalMemoryDBSCANError(
                f"small source changed during post-hoc audit: {name}"
            )
    _atomic_json(root / "radius_boundary_audit.json", audit)
    if audit["source_stats_before"] != {
        "terminal_manifest": _stat_identity(terminal_path),
        "vectors": _stat_identity(vectors_path),
        "dbscan_manifest": _stat_identity(dbscan_path),
        "pair_source": _stat_identity(pair_source_path),
        "numpy_centroid": _stat_identity(numpy_center_path),
        "torch_centroid": _stat_identity(torch_center_path),
        "live_retained_mask": _stat_identity(live_mask_path),
    }:
        raise ExternalMemoryDBSCANError(
            "source changed after post-hoc audit publication"
        )
    if diff_free:
        _write_marker(root, "PASS", "PASS")
    else:
        _write_marker(root, "BLOCKED", "BLOCKED_BOUNDARY_DIFF")
    return audit


__all__ = [
    "MASK_DIGEST_CONTRACT",
    "SCHEMA_VERSION",
    "run_one_cluster_radius_posthoc_audit",
]
