"""Exact, resumable materialization of a logical theta-close pair view.

This module deliberately knows nothing about GREED, NeuroSED, or molecular
graphs.  A caller supplies distance blocks produced by the frozen model
interface.  The scanner validates the physical pair-axis contract while it
writes a small uint8 bitmap and the scalar normalized distances needed for an
auditable boundary/statistics report.  Recourse-vector norms are never used to
decide whether a pair is close.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import atomic_write_bytes, sha256_file, stable_json_sha256


CLOSE_PAIR_SCAN_SCHEMA = "comrecgc_theta_close_distance_scan_v1"
CLOSE_BITMAP_DTYPE = np.dtype(np.uint8)
DISTANCE_DTYPE = np.dtype(np.float32)


class ClosePairScanError(RuntimeError):
    """Raised when the physical store or distance stream violates its contract."""


@dataclass(frozen=True)
class PairChunk:
    chunk_index: int
    candidate_start: int
    candidate_stop: int
    row_count: int
    first_pair: tuple[int, int]
    last_pair: tuple[int, int]

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "PairChunk":
        scientific = value.get("scientific_identity")
        if not isinstance(scientific, Mapping):
            raise ClosePairScanError("pair chunk lacks scientific_identity")
        first = value.get("first_pair")
        last = value.get("last_pair")
        if not isinstance(first, list) or len(first) != 2:
            raise ClosePairScanError("pair chunk first_pair is invalid")
        if not isinstance(last, list) or len(last) != 2:
            raise ClosePairScanError("pair chunk last_pair is invalid")
        return cls(
            chunk_index=int(value["chunk_index"]),
            candidate_start=int(scientific["candidate_start"]),
            candidate_stop=int(scientific["candidate_stop"]),
            row_count=int(value["row_count"]),
            first_pair=(int(first[0]), int(first[1])),
            last_pair=(int(last[0]), int(last[1])),
        )


@dataclass(frozen=True)
class ClosePairScanResult:
    manifest_path: Path
    manifest_sha256: str
    distance_path: Path
    distance_sha256: str
    close_bitmap_path: Path
    close_bitmap_sha256: str
    physical_pair_count: int
    logical_close_pair_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cartesian_row(
    *, parent_index: int, candidate_index: int, parent_count: int
) -> int:
    if parent_count <= 0 or parent_index < 0 or parent_index >= parent_count:
        raise ValueError("parent index is outside the Cartesian contract")
    if candidate_index < 0:
        raise ValueError("candidate index is negative")
    return int(candidate_index) * int(parent_count) + int(parent_index)


def pair_for_cartesian_row(row: int, *, parent_count: int) -> tuple[int, int]:
    if row < 0 or parent_count <= 0:
        raise ValueError("invalid Cartesian row contract")
    return int(row % parent_count), int(row // parent_count)


def normalize_distance_block(
    raw_distances: Any,
    *,
    candidate_element_counts: Any,
    parent_element_counts: Any,
) -> np.ndarray:
    """Apply the official ``distance / (candidate_count + parent_count)`` rule."""

    raw = np.asarray(raw_distances)
    candidate = np.asarray(candidate_element_counts).reshape(-1)
    parent = np.asarray(parent_element_counts).reshape(-1)
    expected = (len(candidate), len(parent))
    if raw.shape != expected:
        raise ClosePairScanError(
            f"distance block has shape {raw.shape}, expected {expected}"
        )
    if raw.dtype.kind != "f":
        raise ClosePairScanError(f"distance block dtype is not floating: {raw.dtype}")
    if np.any(candidate <= 0) or np.any(parent <= 0):
        raise ClosePairScanError("element counts must be strictly positive")
    scale = candidate[:, None] + parent[None, :]
    # PyTorch's production expression divides the float32 cdist tensor by an
    # integer scale and returns float32. NumPy instead promotes
    # ``float32 / int64`` to float64, so cast the scale explicitly.
    raw_float32 = np.asarray(raw, dtype=DISTANCE_DTYPE)
    normalized = raw_float32 / scale.astype(DISTANCE_DTYPE, copy=False)
    return np.asarray(normalized, dtype=DISTANCE_DTYPE)


def recourse_vector_formula(
    candidate_embedding: Any,
    parent_embedding: Any,
    *,
    candidate_element_count: int,
    parent_element_count: int,
) -> np.ndarray:
    candidate = np.asarray(candidate_embedding)
    parent = np.asarray(parent_embedding)
    if candidate.shape != parent.shape or candidate.ndim != 1:
        raise ClosePairScanError("candidate and parent embeddings must be aligned vectors")
    scale = int(candidate_element_count) + int(parent_element_count)
    if scale <= 0:
        raise ClosePairScanError("recourse-vector scale must be positive")
    return np.asarray((candidate - parent) / scale)


def close_mask(distances: Any, *, theta: float) -> np.ndarray:
    values = np.asarray(distances)
    if values.dtype.kind != "f" or values.ndim != 2:
        raise ClosePairScanError("normalized distance block must be a floating matrix")
    if not math.isfinite(theta) or theta < 0.0:
        raise ClosePairScanError("theta must be finite and nonnegative")
    finite = np.isfinite(values)
    if np.any(values[finite] < values.dtype.type(0.0)):
        raise ClosePairScanError("normalized distances must be nonnegative")
    # The official predicate is inclusive and compares in the tensor dtype.
    # Non-finite values fail closed.
    threshold = values.dtype.type(theta)
    return finite & (values <= threshold)


def _array_bytes_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, encoded)


def _file_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_chunks(
    chunks: Sequence[PairChunk], *, parent_count: int, candidate_count: int
) -> None:
    if not chunks:
        raise ClosePairScanError("physical pair manifest contains no chunks")
    expected_candidate = 0
    for expected_index, chunk in enumerate(chunks):
        expected_rows = (chunk.candidate_stop - chunk.candidate_start) * parent_count
        if (
            chunk.chunk_index != expected_index
            or chunk.candidate_start != expected_candidate
            or chunk.candidate_stop <= chunk.candidate_start
            or chunk.row_count != expected_rows
            or chunk.first_pair != (0, chunk.candidate_start)
            or chunk.last_pair != (parent_count - 1, chunk.candidate_stop - 1)
        ):
            raise ClosePairScanError(
                f"physical pair chunk {expected_index} violates candidate-major order"
            )
        expected_candidate = chunk.candidate_stop
    if expected_candidate != candidate_count:
        raise ClosePairScanError(
            f"pair chunks end at candidate {expected_candidate}, expected {candidate_count}"
        )


def _merge_boundary_rows(
    current_rows: np.ndarray,
    current_distances: np.ndarray,
    new_rows: np.ndarray,
    new_distances: np.ndarray,
    *,
    theta: float,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.concatenate((current_rows, np.asarray(new_rows, dtype=np.int64)))
    distances = np.concatenate(
        (current_distances, np.asarray(new_distances, dtype=np.float32))
    )
    if len(rows) > limit:
        margins = np.abs(distances.astype(np.float64) - float(theta))
        keep = np.argpartition(margins, limit - 1)[:limit]
        rows = rows[keep]
        distances = distances[keep]
    order = np.lexsort((rows, np.abs(distances.astype(np.float64) - float(theta))))
    return rows[order], distances[order]


def _block_boundary_rows(
    values: np.ndarray, *, row_start: int, theta: float, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1)
    finite_local = np.flatnonzero(np.isfinite(flat))
    count = min(int(limit), int(finite_local.size))
    if count <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    finite_distances = flat[finite_local]
    margins = np.abs(finite_distances.astype(np.float64) - float(theta))
    keep = np.argpartition(margins, count - 1)[:count]
    local = finite_local[keep]
    return local.astype(np.int64) + int(row_start), flat[local].astype(np.float32)


def _validate_resume_prefix(
    *,
    checkpoint: Mapping[str, Any],
    chunks: Sequence[PairChunk],
    distances: np.ndarray,
    bitmap: np.ndarray,
    parent_count: int,
) -> None:
    records = checkpoint.get("chunks")
    if not isinstance(records, list):
        raise ClosePairScanError("resume checkpoint lacks chunk records")
    next_index = int(checkpoint.get("next_chunk_index", -1))
    if len(records) != next_index or next_index < 0 or next_index > len(chunks):
        raise ClosePairScanError("resume checkpoint chunk cursor is inconsistent")
    for index, record in enumerate(records):
        chunk = chunks[index]
        row_start = chunk.candidate_start * parent_count
        row_stop = chunk.candidate_stop * parent_count
        distance_hash = _array_bytes_sha256(distances[row_start:row_stop])
        bitmap_hash = _array_bytes_sha256(bitmap[row_start:row_stop])
        if (
            int(record.get("chunk_index", -1)) != index
            or int(record.get("row_start", -1)) != row_start
            or int(record.get("row_stop", -1)) != row_stop
            or record.get("distance_block_sha256") != distance_hash
            or record.get("close_bitmap_block_sha256") != bitmap_hash
        ):
            raise ClosePairScanError(f"resume prefix differs at chunk {index}")


def scan_theta_close_pairs(
    *,
    output_dir: str | Path,
    pair_indices_path: str | Path,
    pair_chunks: Sequence[Mapping[str, Any]],
    parent_count: int,
    candidate_count: int,
    theta: float,
    scientific_identity: Mapping[str, Any],
    distance_provider: Callable[[int, int], np.ndarray],
    resume: bool = False,
    max_chunks: int | None = None,
    boundary_sample_size: int = 1_000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> ClosePairScanResult | None:
    """Scan exact distance blocks and materialize the logical close-pair view.

    ``distance_provider(start, stop)`` must return normalized distances for the
    candidate half-open interval in candidate-major, parent-minor order.
    Returning ``None`` means a bounded benchmark stopped before the full scan;
    partial arrays and an authenticated checkpoint remain resumable.
    """

    root = Path(output_dir).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    if max_chunks is not None and int(max_chunks) <= 0:
        raise ValueError("max_chunks must be positive when provided")
    if int(boundary_sample_size) <= 0:
        raise ValueError("boundary_sample_size must be positive")
    pair_path = Path(pair_indices_path).expanduser().resolve(strict=True)
    chunks = [PairChunk.from_manifest(value) for value in pair_chunks]
    _validate_chunks(
        chunks, parent_count=int(parent_count), candidate_count=int(candidate_count)
    )
    total_rows = int(parent_count) * int(candidate_count)
    identity = {
        "schema_version": CLOSE_PAIR_SCAN_SCHEMA,
        "scientific_identity": dict(scientific_identity),
        "parent_count": int(parent_count),
        "candidate_count": int(candidate_count),
        "physical_pair_count": total_rows,
        "theta": float(theta),
        "filter_operator": "<=",
        "pair_orientation": ["parent_index", "candidate_index"],
        "pair_order": "candidate_major_parent_minor",
        "distance_dtype": str(DISTANCE_DTYPE),
        "close_bitmap_dtype": str(CLOSE_BITMAP_DTYPE),
        "chunk_count": len(chunks),
    }
    identity_sha = stable_json_sha256(identity)
    checkpoint_path = root / "checkpoint.json"
    distances_partial = root / "normalized_distances.greed.float32.partial.npy"
    bitmap_partial = root / "close_pair_bitmap.greed.uint8.partial.npy"
    distances_final = root / "normalized_distances.greed.float32.npy"
    bitmap_final = root / "close_pair_bitmap.greed.uint8.npy"
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"close-pair scan is already terminal: {root}")
        terminal = json.loads(manifest_path.read_text(encoding="utf-8"))
        if terminal.get("identity") != identity:
            raise ClosePairScanError("terminal close-pair identity differs")
        resolved_distances = distances_final.resolve(strict=True)
        resolved_bitmap = bitmap_final.resolve(strict=True)
        distance_sha256 = sha256_file(resolved_distances)
        bitmap_sha256 = sha256_file(resolved_bitmap)
        if (
            distance_sha256 != terminal.get("normalized_distances_sha256")
            or bitmap_sha256 != terminal.get("close_bitmap_sha256")
        ):
            raise ClosePairScanError("terminal close-pair artifacts differ")
        return ClosePairScanResult(
            manifest_path=manifest_path,
            manifest_sha256=sha256_file(manifest_path),
            distance_path=resolved_distances,
            distance_sha256=distance_sha256,
            close_bitmap_path=resolved_bitmap,
            close_bitmap_sha256=bitmap_sha256,
            physical_pair_count=total_rows,
            logical_close_pair_count=int(terminal["logical_close_pair_count"]),
        )
    pairs = np.load(pair_path, mmap_mode="r", allow_pickle=False)
    if pairs.shape != (total_rows, 2) or pairs.dtype != np.dtype(np.int64):
        raise ClosePairScanError(
            f"physical pair array schema mismatch: shape={pairs.shape}, dtype={pairs.dtype}"
        )

    if resume:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity or checkpoint.get(
            "identity_sha256"
        ) != identity_sha:
            raise ClosePairScanError("resume checkpoint identity differs")
        next_index = int(checkpoint["next_chunk_index"])
        if next_index < len(chunks) and (
            not distances_partial.exists() or not bitmap_partial.exists()
        ):
            raise ClosePairScanError("incomplete scan is missing a partial array")
        distance_resume_path = (
            distances_partial if distances_partial.exists() else distances_final
        )
        bitmap_resume_path = bitmap_partial if bitmap_partial.exists() else bitmap_final
        distances = np.load(
            distance_resume_path, mmap_mode="r+", allow_pickle=False
        )
        bitmap = np.load(bitmap_resume_path, mmap_mode="r+", allow_pickle=False)
        if (
            distances.shape != (total_rows,)
            or distances.dtype != DISTANCE_DTYPE
            or bitmap.shape != (total_rows,)
            or bitmap.dtype != CLOSE_BITMAP_DTYPE
        ):
            raise ClosePairScanError("resume partial array schema differs")
        _validate_resume_prefix(
            checkpoint=checkpoint,
            chunks=chunks,
            distances=distances,
            bitmap=bitmap,
            parent_count=parent_count,
        )
    else:
        occupied = [
            path
            for path in (
                checkpoint_path,
                distances_partial,
                bitmap_partial,
                distances_final,
                bitmap_final,
            )
            if path.exists()
        ]
        if occupied:
            raise FileExistsError(f"fresh close-pair scan artifacts exist: {occupied}")
        distances = np.lib.format.open_memmap(
            distances_partial, mode="w+", dtype=DISTANCE_DTYPE, shape=(total_rows,)
        )
        bitmap = np.lib.format.open_memmap(
            bitmap_partial, mode="w+", dtype=CLOSE_BITMAP_DTYPE, shape=(total_rows,)
        )
        checkpoint = {
            "schema_version": CLOSE_PAIR_SCAN_SCHEMA,
            "identity": identity,
            "identity_sha256": identity_sha,
            "next_chunk_index": 0,
            "rows_processed": 0,
            "finite_count": 0,
            "nonfinite_count": 0,
            "logical_close_pair_count": 0,
            "count_distance_eq_theta": 0,
            "distance_min": None,
            "distance_max": None,
            "distance_sum": 0.0,
            "distance_square_sum": 0.0,
            "pair_axis_mismatch_count": 0,
            "boundary_rows": [],
            "boundary_distances": [],
            "chunks": [],
            "started_at": utc_now(),
            "updated_at": utc_now(),
        }
        _atomic_json(checkpoint_path, checkpoint)

    next_chunk = int(checkpoint["next_chunk_index"])
    resume_rows = int(checkpoint["rows_processed"])
    boundary_rows = np.asarray(checkpoint.get("boundary_rows") or [], dtype=np.int64)
    boundary_distances = np.asarray(
        checkpoint.get("boundary_distances") or [], dtype=np.float32
    )
    started = time.monotonic()
    limit = len(chunks) if max_chunks is None else min(len(chunks), next_chunk + max_chunks)
    for chunk in chunks[next_chunk:limit]:
        row_start = chunk.candidate_start * parent_count
        row_stop = chunk.candidate_stop * parent_count
        values = np.asarray(
            distance_provider(chunk.candidate_start, chunk.candidate_stop),
            dtype=DISTANCE_DTYPE,
        )
        expected_shape = (chunk.candidate_stop - chunk.candidate_start, parent_count)
        if values.shape != expected_shape:
            raise ClosePairScanError(
                f"distance provider shape {values.shape}, expected {expected_shape}"
            )
        flat = values.reshape(-1)
        pair_block = np.asarray(pairs[row_start:row_stop])
        rows = np.arange(row_start, row_stop, dtype=np.int64)
        expected_parent = rows % int(parent_count)
        expected_candidate = rows // int(parent_count)
        mismatches = int(
            np.count_nonzero(
                (pair_block[:, 0] != expected_parent)
                | (pair_block[:, 1] != expected_candidate)
            )
        )
        if mismatches:
            raise ClosePairScanError(
                f"physical pair axes differ in chunk {chunk.chunk_index}: "
                f"mismatches={mismatches}"
            )
        mask = close_mask(values, theta=theta).reshape(-1)
        distances[row_start:row_stop] = flat
        bitmap[row_start:row_stop] = mask.astype(CLOSE_BITMAP_DTYPE, copy=False)
        distances.flush()
        bitmap.flush()
        with distances_partial.open("rb") as handle:
            os.fsync(handle.fileno())
        with bitmap_partial.open("rb") as handle:
            os.fsync(handle.fileno())
        finite = np.isfinite(flat)
        finite_values = flat[finite]
        candidate_rows, candidate_distances = _block_boundary_rows(
            values,
            row_start=row_start,
            theta=theta,
            limit=boundary_sample_size,
        )
        boundary_rows, boundary_distances = _merge_boundary_rows(
            boundary_rows,
            boundary_distances,
            candidate_rows,
            candidate_distances,
            theta=theta,
            limit=boundary_sample_size,
        )
        elapsed = max(time.monotonic() - started, 1e-9)
        processed_this_run = row_stop - resume_rows
        throughput = processed_this_run / elapsed
        remaining = total_rows - row_stop
        record = {
            "chunk_index": chunk.chunk_index,
            "candidate_start": chunk.candidate_start,
            "candidate_stop": chunk.candidate_stop,
            "row_start": row_start,
            "row_stop": row_stop,
            "row_count": int(flat.size),
            "finite_count": int(np.count_nonzero(finite)),
            "close_pair_count": int(np.count_nonzero(mask)),
            "count_distance_eq_theta": int(
                np.count_nonzero(flat == DISTANCE_DTYPE.type(theta))
            ),
            "distance_min": (
                None if finite_values.size == 0 else float(np.min(finite_values))
            ),
            "distance_max": (
                None if finite_values.size == 0 else float(np.max(finite_values))
            ),
            "pair_axis_mismatch_count": mismatches,
            "distance_block_sha256": _array_bytes_sha256(flat),
            "close_bitmap_block_sha256": _array_bytes_sha256(
                mask.astype(CLOSE_BITMAP_DTYPE, copy=False)
            ),
        }
        previous_min = checkpoint.get("distance_min")
        previous_max = checkpoint.get("distance_max")
        checkpoint = {
            **checkpoint,
            "next_chunk_index": chunk.chunk_index + 1,
            "rows_processed": row_stop,
            "finite_count": int(checkpoint["finite_count"])
            + int(np.count_nonzero(finite)),
            "nonfinite_count": int(checkpoint["nonfinite_count"])
            + int(flat.size - np.count_nonzero(finite)),
            "logical_close_pair_count": int(checkpoint["logical_close_pair_count"])
            + int(np.count_nonzero(mask)),
            "count_distance_eq_theta": int(checkpoint["count_distance_eq_theta"])
            + int(np.count_nonzero(flat == DISTANCE_DTYPE.type(theta))),
            "distance_min": (
                previous_min
                if finite_values.size == 0
                else min(
                    float(np.min(finite_values)),
                    float(previous_min)
                    if previous_min is not None
                    else math.inf,
                )
            ),
            "distance_max": (
                previous_max
                if finite_values.size == 0
                else max(
                    float(np.max(finite_values)),
                    float(previous_max)
                    if previous_max is not None
                    else -math.inf,
                )
            ),
            "distance_sum": float(checkpoint["distance_sum"])
            + float(np.sum(finite_values, dtype=np.float64)),
            "distance_square_sum": float(checkpoint["distance_square_sum"])
            + float(
                np.sum(
                    finite_values.astype(np.float64)
                    * finite_values.astype(np.float64),
                    dtype=np.float64,
                )
            ),
            "pair_axis_mismatch_count": int(
                checkpoint["pair_axis_mismatch_count"]
            )
            + mismatches,
            "boundary_rows": [int(value) for value in boundary_rows],
            "boundary_distances": [float(value) for value in boundary_distances],
            "chunks": [*list(checkpoint["chunks"]), record],
            "rolling_throughput_rows_per_second": float(throughput),
            "eta_seconds": float(remaining / throughput) if throughput > 0 else None,
            "updated_at": utc_now(),
        }
        _atomic_json(checkpoint_path, checkpoint)
        if progress_callback is not None:
            progress_callback(checkpoint)

    if int(checkpoint["next_chunk_index"]) != len(chunks):
        return None
    if int(checkpoint["pair_axis_mismatch_count"]) != 0:
        raise ClosePairScanError("physical pair axes differ from the all-row contract")
    if int(checkpoint["rows_processed"]) != total_rows:
        raise ClosePairScanError("terminal distance scan row count differs")
    distances.flush()
    bitmap.flush()
    del distances, bitmap
    if distances_partial.exists():
        os.replace(distances_partial, distances_final)
        _fsync_directory(root)
    elif not distances_final.exists():
        raise ClosePairScanError("final distance array is missing")
    if bitmap_partial.exists():
        os.replace(bitmap_partial, bitmap_final)
        _fsync_directory(root)
    elif not bitmap_final.exists():
        raise ClosePairScanError("final close bitmap is missing")
    distances_read = np.load(distances_final, mmap_mode="r", allow_pickle=False)
    bitmap_read = np.load(bitmap_final, mmap_mode="r", allow_pickle=False)
    finite_values = distances_read[np.isfinite(distances_read)]
    if finite_values.size:
        quantiles = np.quantile(
            finite_values, [0.0, 0.01, 0.5, 0.99, 1.0], method="linear"
        )
        q_values: list[float | None] = [float(value) for value in quantiles]
    else:
        q_values = [None, None, None, None, None]
    logical_close = int(np.count_nonzero(bitmap_read))
    if logical_close != int(checkpoint["logical_close_pair_count"]):
        raise ClosePairScanError("terminal close bitmap count differs from checkpoint")
    mean = (
        float(checkpoint["distance_sum"]) / int(checkpoint["finite_count"])
        if int(checkpoint["finite_count"])
        else None
    )
    variance = (
        max(
            0.0,
            float(checkpoint["distance_square_sum"])
            / int(checkpoint["finite_count"])
            - float(mean) ** 2,
        )
        if mean is not None
        else None
    )
    manifest = {
        "schema_version": CLOSE_PAIR_SCAN_SCHEMA,
        "run_complete": True,
        "status": "PASS",
        "identity": identity,
        "identity_sha256": identity_sha,
        "physical_pair_count": total_rows,
        "logical_close_pair_count": logical_close,
        "close_pair_rate": logical_close / total_rows,
        "all_pairs_close": logical_close == total_rows,
        "dbscan_input_count_contract": logical_close,
        "theta": float(theta),
        "filter_operator": "<=",
        "distance_statistics": {
            "finite_count": int(checkpoint["finite_count"]),
            "nonfinite_count": int(checkpoint["nonfinite_count"]),
            "min": q_values[0],
            "q01": q_values[1],
            "q50": q_values[2],
            "q99": q_values[3],
            "max": q_values[4],
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
            "count_distance_eq_theta": int(
                checkpoint["count_distance_eq_theta"]
            ),
        },
        "pair_axis": {
            "columns": ["parent_index", "candidate_index"],
            "formula": "parent=row%parent_count;candidate=row//parent_count",
            "all_rows_checked": True,
            "mismatch_count": int(checkpoint["pair_axis_mismatch_count"]),
        },
        "boundary_rows": [int(value) for value in boundary_rows],
        "boundary_distances": [float(value) for value in boundary_distances],
        "chunks": list(checkpoint["chunks"]),
        "normalized_distances_path": str(distances_final),
        "normalized_distances_sha256": sha256_file(distances_final),
        "close_bitmap_path": str(bitmap_final),
        "close_bitmap_sha256": sha256_file(bitmap_final),
        "close_bitmap_encoding": "numpy uint8; one value per physical row; 1=close",
        "pair_indices_path": str(pair_path),
        "pair_indices_stat": _file_stat(pair_path),
        "completed_at": utc_now(),
    }
    _atomic_json(manifest_path, manifest)
    return ClosePairScanResult(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        distance_path=distances_final,
        distance_sha256=str(manifest["normalized_distances_sha256"]),
        close_bitmap_path=bitmap_final,
        close_bitmap_sha256=str(manifest["close_bitmap_sha256"]),
        physical_pair_count=total_rows,
        logical_close_pair_count=logical_close,
    )
