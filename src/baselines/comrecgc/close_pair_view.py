"""Hash-closed logical :math:`\theta`-close views over Cartesian pair stores.

The recovered AIDS pair snapshot is a *physical* candidate-by-parent Cartesian
store.  COMRECGC, however, clusters only rows whose normalized NeuroSED/GREED
distance is at most ``theta``.  A physical row therefore cannot become a
DBSCAN sample merely because its recourse vector was persisted.

This module turns an aligned, independently-computed normalized-distance array
into a resumable logical view.  The default partial-close representation stores
only a bitmap plus selected physical row indices and reads vector rows directly
from the physical mmap.  A compact vector copy is legal only under an explicit
byte budget.  The terminal manifest binds the logical mapping,
distance/checkpoint provenance, storage mode, and inclusive ``<=`` predicate.
Validation replays the predicate and every row mapping before a consumer can
use the view as DBSCAN input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import numpy as np

from .external_memory_dbscan import ExternalMemoryDBSCANError


CLOSE_PAIR_VIEW_SCHEMA = "comrecgc_theta_close_pair_view_v1"
CLOSE_PAIR_CHECKPOINT_SCHEMA = "comrecgc_theta_close_pair_checkpoint_v1"
ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA = "comrecgc_all_pairs_close_certificate_v1"
PAIR_ORIENTATION = "col0_parent_col1_candidate"
PAIR_ORDER = "candidate_major_parent_minor"
FILTER_OPERATOR = "<="
SCALE_CONTRACT = "element_count(parent)+element_count(candidate)"
NORMALIZED_DISTANCE_CONTRACT = (
    "GREED/NeuroSED.predict_outer(parent,candidate)/"
    "(element_count(parent)+element_count(candidate))"
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


def _sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ExternalMemoryDBSCANError(f"close-view source is not regular: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _hash_stable_file(path: Path) -> tuple[str, dict[str, int]]:
    before = _stat_identity(path)
    digest = _sha256_file(path)
    after = _stat_identity(path)
    if before != after:
        raise ExternalMemoryDBSCANError(
            f"close-view source changed while hashing: {path}"
        )
    return digest, before


def _verify_file(path: Path, *, sha256: str, identity: Mapping[str, Any]) -> None:
    before = _stat_identity(path)
    if before != dict(identity) or _sha256_file(path) != str(sha256):
        raise ExternalMemoryDBSCANError(f"close-view source identity drift: {path}")
    if _stat_identity(path) != before:
        raise ExternalMemoryDBSCANError(f"close-view source changed: {path}")


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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalMemoryDBSCANError(f"invalid close-view JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExternalMemoryDBSCANError(f"close-view JSON is not an object: {path}")
    return value


def _new_memmap(path: Path, *, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    if path.exists() or path.is_symlink():
        raise ExternalMemoryDBSCANError(f"close-view partial already exists: {path}")
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.dtype(dtype), shape=shape
    )


def _flush(value: np.memmap) -> None:
    value.flush()
    with Path(value.filename).open("rb") as handle:
        os.fsync(handle.fileno())


def _promote(partial: Path, final: Path, *, expected_sha256: str) -> None:
    candidates = [path for path in (partial, final) if path.exists()]
    if len(candidates) != 1 or _sha256_file(candidates[0]) != expected_sha256:
        raise ExternalMemoryDBSCANError(
            f"close-view promotion closure mismatch: {final.name}"
        )
    if candidates[0] == partial:
        os.replace(partial, final)
        directory = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    phase: str,
    next_offset: int,
    close_count: int,
    materialized_count: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CLOSE_PAIR_CHECKPOINT_SCHEMA,
        "scientific_identity": dict(identity),
        "scientific_identity_sha256": _stable_hash(identity),
        "phase": str(phase),
        "next_offset": int(next_offset),
        "close_count": int(close_count),
        "materialized_count": int(materialized_count),
    }
    if extra:
        payload.update(dict(extra))
    payload["checkpoint_payload_sha256"] = _stable_hash(payload)
    _atomic_json(path, payload)
    return payload


def _load_checkpoint(path: Path, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_object(path)
    expected = payload.get("checkpoint_payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("checkpoint_payload_sha256", None)
    if (
        payload.get("schema_version") != CLOSE_PAIR_CHECKPOINT_SCHEMA
        or payload.get("scientific_identity") != dict(identity)
        or payload.get("scientific_identity_sha256") != _stable_hash(identity)
        or not isinstance(expected, str)
        or expected != _stable_hash(unsigned)
    ):
        raise ExternalMemoryDBSCANError("close-view checkpoint identity mismatch")
    return payload


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class ThetaClosePairContract:
    """Scientific identity of the logical filtering operation."""

    theta: float
    parent_count: int
    candidate_count: int
    distance_checkpoint_sha256: str
    embedding_checkpoint_sha256: str
    scale_contract: str
    normalized_distance_contract: str
    chunk_order: str = PAIR_ORDER
    pair_orientation: str = PAIR_ORIENTATION
    filter_operator: str = FILTER_OPERATOR

    def validate(self) -> None:
        if not math.isfinite(float(self.theta)) or float(self.theta) < 0:
            raise ExternalMemoryDBSCANError("theta must be finite and nonnegative")
        if int(self.parent_count) <= 0 or int(self.candidate_count) <= 0:
            raise ExternalMemoryDBSCANError(
                "close-view parent/candidate counts must be positive"
            )
        if not _is_sha256(str(self.distance_checkpoint_sha256)):
            raise ExternalMemoryDBSCANError(
                "distance checkpoint SHA256 is not frozen"
            )
        if not _is_sha256(str(self.embedding_checkpoint_sha256)):
            raise ExternalMemoryDBSCANError(
                "embedding checkpoint SHA256 is not frozen"
            )
        if not str(self.scale_contract) or not str(self.normalized_distance_contract):
            raise ExternalMemoryDBSCANError(
                "distance and scale contracts must be explicit"
            )
        if self.chunk_order != PAIR_ORDER:
            raise ExternalMemoryDBSCANError("close-view chunk order changed")
        if self.pair_orientation != PAIR_ORIENTATION:
            raise ExternalMemoryDBSCANError("close-view pair orientation changed")
        if self.filter_operator != FILTER_OPERATOR:
            raise ExternalMemoryDBSCANError("close-view filter must be inclusive <=")


@dataclass(frozen=True)
class ThetaClosePairView:
    manifest_path: Path
    manifest_sha256: str
    bitmap_path: Path
    physical_row_indices_path: Path | None
    pairs_path: Path | None
    vectors_path: Path
    physical_store_rows: int
    logical_close_rows: int
    parent_count: int
    candidate_count: int
    theta: float
    pairs_sha256: str
    vectors_sha256: str
    all_pairs_close: bool
    view_storage: str
    eligible_for_dbscan: bool
    blocking_reason: str | None
    pair_semantics_contract_path: Path | None
    pair_semantics_contract_sha256: str | None

    def open_vectors(self) -> Any:
        """Open the logical vector view without copying selected rows."""

        physical = np.load(self.vectors_path, mmap_mode="r", allow_pickle=False)
        if self.all_pairs_close or self.view_storage == "materialized_selected_rows":
            return physical
        if self.physical_row_indices_path is None:
            raise ExternalMemoryDBSCANError("indexed close view has no row index")
        rows = np.load(
            self.physical_row_indices_path, mmap_mode="r", allow_pickle=False
        )
        return IndexedThetaCloseVectors(physical=physical, physical_rows=rows)

    def open_pairs(self) -> Any:
        """Open logical ``(parent, candidate)`` rows in close-view order."""

        if self.pairs_path is not None:
            return np.load(self.pairs_path, mmap_mode="r", allow_pickle=False)
        if self.all_pairs_close:
            return CartesianThetaClosePairs(
                physical_rows=None,
                logical_count=self.logical_close_rows,
                parent_count=self.parent_count,
            )
        if self.physical_row_indices_path is None:
            raise ExternalMemoryDBSCANError("indexed close view has no row index")
        return CartesianThetaClosePairs(
            physical_rows=np.load(
                self.physical_row_indices_path, mmap_mode="r", allow_pickle=False
            ),
            logical_count=self.logical_close_rows,
            parent_count=self.parent_count,
        )


class IndexedThetaCloseVectors:
    """Read-only logical array backed by physical vectors plus row indices."""

    def __init__(self, *, physical: np.ndarray, physical_rows: np.ndarray) -> None:
        self._physical = physical
        self._physical_rows = physical_rows
        self.shape = (len(physical_rows), int(physical.shape[1]))
        self.dtype = physical.dtype
        self.ndim = 2

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            row_key, column_key = key
            return np.asarray(self._physical[self._physical_rows[row_key], column_key])
        return np.asarray(self._physical[self._physical_rows[key]])


class CartesianThetaClosePairs:
    """Read-only computed pair columns for full or indexed Cartesian rows."""

    dtype = np.dtype(np.int64)
    ndim = 2

    def __init__(
        self,
        *,
        physical_rows: np.ndarray | None,
        logical_count: int,
        parent_count: int,
    ) -> None:
        self._physical_rows = physical_rows
        self._logical_count = int(logical_count)
        self._parent_count = int(parent_count)
        self.shape = (self._logical_count, 2)

    def __len__(self) -> int:
        return self._logical_count

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            row_key, column_key = key
            return self[row_key][..., column_key]
        logical = np.arange(self._logical_count, dtype=np.int64)[key]
        physical = (
            logical
            if self._physical_rows is None
            else np.asarray(self._physical_rows[key], dtype=np.int64)
        )
        return _expected_pairs(
            np.asarray(physical, dtype=np.int64), parent_count=self._parent_count
        )


def _mask(
    values: np.ndarray, *, theta: float, dtype: np.dtype[Any]
) -> np.ndarray:
    # PyTorch/official comparison casts the scalar to the tensor dtype.  Make
    # that conversion explicit so NumPy-version scalar-promotion changes
    # cannot alter the close set.
    threshold = np.asarray(float(theta), dtype=dtype)
    return np.less_equal(values, threshold)


def _expected_pairs(rows: np.ndarray, *, parent_count: int) -> np.ndarray:
    return np.column_stack(
        (rows % int(parent_count), rows // int(parent_count))
    ).astype(np.int64, copy=False)


def _implicit_pair_sha256(*, row_count: int, parent_count: int) -> str:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.int64)),
            "fortran_order": False,
            "shape": (int(row_count), 2),
        },
    )
    digest = hashlib.sha256(stream.getvalue())
    for start in range(0, int(row_count), 1_000_000):
        stop = min(int(row_count), start + 1_000_000)
        rows = np.arange(start, stop, dtype=np.int64)
        digest.update(
            _expected_pairs(rows, parent_count=int(parent_count)).tobytes(order="C")
        )
    return digest.hexdigest()


def _indexed_pair_sha256(
    physical_rows: np.ndarray, *, parent_count: int, block_size: int = 1_000_000
) -> str:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.int64)),
            "fortran_order": False,
            "shape": (int(len(physical_rows)), 2),
        },
    )
    digest = hashlib.sha256(stream.getvalue())
    for start in range(0, len(physical_rows), int(block_size)):
        stop = min(len(physical_rows), start + int(block_size))
        digest.update(
            _expected_pairs(
                np.asarray(physical_rows[start:stop], dtype=np.int64),
                parent_count=int(parent_count),
            ).tobytes(order="C")
        )
    return digest.hexdigest()


def _validate_all_pairs_close_certificate(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    contract: ThetaClosePairContract,
    physical_rows: int,
    count_equal_theta: int,
) -> tuple[Path, str]:
    certificate_path = Path(path).expanduser().resolve(strict=True)
    certificate = _load_object(certificate_path)
    if (
        certificate.get("schema_version") != ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA
        or certificate.get("status") != "PASS"
        or certificate.get("all_pairs_close_proven") is not True
        or certificate.get("full_distance_scan_complete") is not True
        or certificate.get("official_sample_comparison_pass") is not True
        or certificate.get("normalization_audit_pass") is not True
        or certificate.get("filter_operator") != FILTER_OPERATOR
        or certificate.get("pair_orientation") != PAIR_ORIENTATION
        or certificate.get("pair_order") != PAIR_ORDER
        or int(certificate.get("physical_store_rows", -1)) != int(physical_rows)
        or int(certificate.get("count_distance_le_theta", -1))
        != int(physical_rows)
        or int(certificate.get("count_distance_gt_theta", -1)) != 0
        or int(certificate.get("count_distance_eq_theta", -1))
        != int(count_equal_theta)
        or float(certificate.get("theta", float("nan"))) != float(contract.theta)
        or certificate.get("physical_vectors_sha256")
        != identity.get("physical_vectors_sha256")
        or certificate.get("normalized_distances_sha256")
        != identity.get("normalized_distances_sha256")
        or certificate.get("distance_checkpoint_sha256")
        != contract.distance_checkpoint_sha256
        or certificate.get("embedding_checkpoint_sha256")
        != contract.embedding_checkpoint_sha256
        or certificate.get("scale_contract") != contract.scale_contract
        or certificate.get("normalized_distance_contract")
        != contract.normalized_distance_contract
        or certificate.get("approximation_used") is not False
    ):
        raise ExternalMemoryDBSCANError(
            "ALL_PAIRS_CLOSE certificate is absent or incomplete"
        )
    return certificate_path, _sha256_file(certificate_path)


def _verify_scan_prefix(
    *,
    bitmap: np.ndarray,
    distances: np.ndarray,
    stop: int,
    theta: float,
    block_size: int,
) -> int:
    count = 0
    for offset in range(0, int(stop), int(block_size)):
        end = min(int(stop), offset + int(block_size))
        expected = _mask(
            distances[offset:end], theta=float(theta), dtype=distances.dtype
        )
        if not np.array_equal(bitmap[offset:end], expected):
            raise ExternalMemoryDBSCANError(
                "close-view committed bitmap prefix does not replay"
            )
        count += int(np.count_nonzero(expected))
    return count


def _verify_materialized_prefix(
    *,
    bitmap: np.ndarray,
    physical_vectors: np.ndarray,
    physical_rows: np.ndarray,
    pairs: np.ndarray,
    close_vectors: np.ndarray,
    physical_stop: int,
    logical_stop: int,
    parent_count: int,
    block_size: int,
) -> None:
    cursor = 0
    for offset in range(0, int(physical_stop), int(block_size)):
        end = min(int(physical_stop), offset + int(block_size))
        selected = np.flatnonzero(bitmap[offset:end]).astype(np.int64) + offset
        next_cursor = cursor + len(selected)
        if next_cursor > int(logical_stop):
            raise ExternalMemoryDBSCANError(
                "close-view materialized prefix cursor exceeds checkpoint"
            )
        if (
            not np.array_equal(physical_rows[cursor:next_cursor], selected)
            or not np.array_equal(
                pairs[cursor:next_cursor],
                _expected_pairs(selected, parent_count=int(parent_count)),
            )
            or not np.array_equal(
                close_vectors[cursor:next_cursor], physical_vectors[selected]
            )
        ):
            raise ExternalMemoryDBSCANError(
                "close-view materialized prefix does not map byte-exact rows"
            )
        cursor = next_cursor
    if cursor != int(logical_stop):
        raise ExternalMemoryDBSCANError(
            "close-view materialized checkpoint cursor mismatch"
        )


def _verify_index_prefix(
    *,
    bitmap: np.ndarray,
    physical_rows: np.ndarray,
    physical_stop: int,
    logical_stop: int,
    block_size: int,
) -> None:
    cursor = 0
    for offset in range(0, int(physical_stop), int(block_size)):
        end = min(int(physical_stop), offset + int(block_size))
        selected = np.flatnonzero(bitmap[offset:end]).astype(np.int64) + offset
        next_cursor = cursor + len(selected)
        if next_cursor > int(logical_stop) or not np.array_equal(
            physical_rows[cursor:next_cursor], selected
        ):
            raise ExternalMemoryDBSCANError(
                "close-view indexed prefix does not map exact physical rows"
            )
        cursor = next_cursor
    if cursor != int(logical_stop):
        raise ExternalMemoryDBSCANError(
            "close-view indexed checkpoint cursor mismatch"
        )


def materialize_theta_close_pair_view(
    *,
    physical_vectors_path: str | Path,
    normalized_distances_path: str | Path,
    output_dir: str | Path,
    contract: ThetaClosePairContract,
    expected_physical_vectors_sha256: str | None = None,
    expected_normalized_distances_sha256: str | None = None,
    pair_semantics_contract_path: str | Path | None = None,
    expected_pair_semantics_contract_sha256: str | None = None,
    all_pairs_close_certificate_path: str | Path | None = None,
    max_compact_bytes: int = 0,
    block_size: int = 1_000_000,
    resume: bool = False,
) -> ThetaClosePairView:
    """Build a resumable exact logical view from aligned scalar distances."""

    contract.validate()
    if int(block_size) <= 0 or int(max_compact_bytes) < 0:
        raise ExternalMemoryDBSCANError(
            "close-view block size/budget must be nonnegative"
        )
    physical_path = Path(physical_vectors_path).expanduser().resolve(strict=True)
    distance_path = Path(normalized_distances_path).expanduser().resolve(strict=True)
    pair_semantics_path = (
        None
        if pair_semantics_contract_path is None
        else Path(pair_semantics_contract_path).expanduser().resolve(strict=True)
    )
    root = Path(output_dir).expanduser().resolve(strict=False)
    for source in (physical_path, distance_path):
        try:
            source.relative_to(root)
        except ValueError:
            pass
        else:
            raise ExternalMemoryDBSCANError(
                "close-view source must be outside the derived output root"
            )
    manifest_path = root / "close_pair_contract.json"
    if manifest_path.exists():
        return validate_theta_close_pair_view(
            manifest_path,
            expected_contract=contract,
            expected_physical_vectors_path=physical_path,
            expected_physical_vectors_sha256=expected_physical_vectors_sha256,
            expected_normalized_distances_path=distance_path,
            expected_normalized_distances_sha256=(
                expected_normalized_distances_sha256
            ),
            expected_pair_semantics_contract_path=pair_semantics_path,
            expected_pair_semantics_contract_sha256=(
                expected_pair_semantics_contract_sha256
            ),
        )
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"close-view output is non-empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    physical_sha, physical_stat = _hash_stable_file(physical_path)
    distance_sha, distance_stat = _hash_stable_file(distance_path)
    pair_semantics_sha: str | None = None
    pair_semantics_stat: dict[str, int] | None = None
    if pair_semantics_path is not None:
        pair_semantics_sha, pair_semantics_stat = _hash_stable_file(
            pair_semantics_path
        )
    if (
        expected_physical_vectors_sha256 is not None
        and physical_sha != str(expected_physical_vectors_sha256)
    ):
        raise ExternalMemoryDBSCANError("physical recourse-vector SHA256 mismatch")
    if (
        expected_normalized_distances_sha256 is not None
        and distance_sha != str(expected_normalized_distances_sha256)
    ):
        raise ExternalMemoryDBSCANError("normalized-distance SHA256 mismatch")
    if (
        expected_pair_semantics_contract_sha256 is not None
        and pair_semantics_sha != str(expected_pair_semantics_contract_sha256)
    ):
        raise ExternalMemoryDBSCANError("pair-semantics contract SHA256 mismatch")

    physical = np.load(physical_path, mmap_mode="r", allow_pickle=False)
    distances = np.load(distance_path, mmap_mode="r", allow_pickle=False)
    expected_rows = int(contract.parent_count) * int(contract.candidate_count)
    if (
        not isinstance(physical, np.memmap)
        or physical.ndim != 2
        or physical.shape[0] != expected_rows
        or physical.shape[1] <= 0
        or physical.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
    ):
        raise ExternalMemoryDBSCANError(
            "physical vectors are not the frozen Cartesian float matrix"
        )
    if (
        not isinstance(distances, np.memmap)
        or distances.shape != (expected_rows,)
        or distances.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
    ):
        raise ExternalMemoryDBSCANError(
            "normalized distances must be one aligned float scalar per physical row"
        )
    identity = {
        "schema_version": CLOSE_PAIR_VIEW_SCHEMA,
        "contract": asdict(contract),
        "physical_vectors_path": str(physical_path),
        "physical_vectors_sha256": physical_sha,
        "physical_vectors_stat_identity": physical_stat,
        "physical_vectors_shape": [int(value) for value in physical.shape],
        "physical_vectors_dtype": str(physical.dtype),
        "normalized_distances_path": str(distance_path),
        "normalized_distances_sha256": distance_sha,
        "normalized_distances_stat_identity": distance_stat,
        "normalized_distances_shape": [expected_rows],
        "normalized_distances_dtype": str(distances.dtype),
        "pair_semantics_contract_path": (
            None if pair_semantics_path is None else str(pair_semantics_path)
        ),
        "pair_semantics_contract_sha256": pair_semantics_sha,
        "pair_semantics_contract_stat_identity": pair_semantics_stat,
        "physical_store_rows": expected_rows,
        "full_cartesian_count": expected_rows,
        "physical_store_is_full_cartesian": True,
        "filter_operator": FILTER_OPERATOR,
        "filter_scalar_cast": "theta_cast_to_normalized_distance_dtype",
        "pair_orientation": PAIR_ORIENTATION,
        "pair_order": PAIR_ORDER,
        "max_compact_bytes": int(max_compact_bytes),
    }
    state_path = root / "checkpoint.json"
    if state_path.exists():
        if not resume:
            raise ExternalMemoryDBSCANError("close-view checkpoint requires resume")
        state = _load_checkpoint(state_path, identity=identity)
    else:
        state = _checkpoint(
            state_path,
            identity=identity,
            phase="scan",
            next_offset=0,
            close_count=0,
            materialized_count=0,
        )

    bitmap_partial = root / "close_bitmap.partial.npy"
    bitmap_final = root / "close_bitmap.npy"
    phase = str(state.get("phase"))
    if phase == "scan":
        if bitmap_partial.exists():
            bitmap = np.load(bitmap_partial, mmap_mode="r+", allow_pickle=False)
            if bitmap.shape != (expected_rows,) or bitmap.dtype != np.dtype(np.bool_):
                raise ExternalMemoryDBSCANError("close bitmap partial schema mismatch")
        else:
            if int(state.get("next_offset", -1)) != 0:
                raise ExternalMemoryDBSCANError("close bitmap partial is missing")
            bitmap = _new_memmap(
                bitmap_partial, dtype=np.bool_, shape=(expected_rows,)
            )
        start = int(state.get("next_offset", 0))
        if start < 0 or start > expected_rows or (
            start != expected_rows and start % int(block_size) != 0
        ):
            raise ExternalMemoryDBSCANError("close-view scan checkpoint is invalid")
        replayed_count = _verify_scan_prefix(
            bitmap=bitmap,
            distances=distances,
            stop=start,
            theta=float(contract.theta),
            block_size=int(block_size),
        )
        if replayed_count != int(state.get("close_count", -1)):
            raise ExternalMemoryDBSCANError("close-view scan count does not replay")
        close_count = replayed_count
        for offset in range(start, expected_rows, int(block_size)):
            stop = min(expected_rows, offset + int(block_size))
            block = np.asarray(distances[offset:stop])
            if not bool(np.isfinite(block).all()) or bool(np.any(block < 0)):
                raise ExternalMemoryDBSCANError(
                    "normalized distances contain negative or non-finite values"
                )
            selected = _mask(
                block, theta=float(contract.theta), dtype=distances.dtype
            )
            bitmap[offset:stop] = selected
            close_count += int(np.count_nonzero(selected))
            _flush(bitmap)
            state = _checkpoint(
                state_path,
                identity=identity,
                phase="scan",
                next_offset=stop,
                close_count=close_count,
                materialized_count=0,
            )
        _flush(bitmap)
        bitmap_sha = _sha256_file(bitmap_partial)
        count_equal_theta = 0
        distance_min = float("inf")
        distance_max = float("-inf")
        threshold = np.asarray(float(contract.theta), dtype=distances.dtype)
        for offset in range(0, expected_rows, int(block_size)):
            stop = min(expected_rows, offset + int(block_size))
            block = np.asarray(distances[offset:stop])
            distance_min = min(distance_min, float(np.min(block)))
            distance_max = max(distance_max, float(np.max(block)))
            count_equal_theta += int(np.count_nonzero(np.equal(block, threshold)))
        compact_estimated_bytes = close_count * (
            8 + 16 + int(physical.shape[1]) * int(physical.dtype.itemsize)
        )
        next_phase = (
            "index_only"
            if close_count < expected_rows
            and compact_estimated_bytes > int(max_compact_bytes)
            else "materialize"
        )
        state = _checkpoint(
            state_path,
            identity=identity,
            phase=next_phase,
            next_offset=0,
            close_count=close_count,
            materialized_count=0,
            extra={
                "close_bitmap_sha256": bitmap_sha,
                "distance_min": distance_min,
                "distance_max": distance_max,
                "count_distance_eq_theta": count_equal_theta,
                "compact_estimated_bytes": compact_estimated_bytes,
            },
        )
        del bitmap
        phase = next_phase

    close_count = int(state.get("close_count", -1))
    if close_count < 0 or close_count > expected_rows:
        raise ExternalMemoryDBSCANError("close-view logical count is invalid")
    all_pairs_certificate: tuple[Path, str] | None = None
    if close_count == expected_rows and all_pairs_close_certificate_path is None:
        diagnostic = {
            "schema_version": CLOSE_PAIR_VIEW_SCHEMA,
            "status": "ALL_PAIRS_CLOSE_REVIEW_REQUIRED",
            "eligible_for_dbscan": False,
            "scientific_identity": identity,
            "scientific_identity_sha256": _stable_hash(identity),
            "physical_store_rows": expected_rows,
            "logical_close_rows": close_count,
            "theta": float(contract.theta),
        }
        _atomic_json(root / "all_pairs_close_review.json", diagnostic)
        raise ExternalMemoryDBSCANError(
            "ALL_PAIRS_CLOSE_REVIEW_REQUIRED: refusing automatic DBSCAN adoption"
        )
    if close_count == expected_rows:
        all_pairs_certificate = _validate_all_pairs_close_certificate(
            all_pairs_close_certificate_path,  # type: ignore[arg-type]
            identity=identity,
            contract=contract,
            physical_rows=expected_rows,
            count_equal_theta=int(state["count_distance_eq_theta"]),
        )

    if close_count == expected_rows and phase == "materialize":
        bitmap_path = bitmap_final if bitmap_final.exists() else bitmap_partial
        if (
            not bitmap_path.exists()
            or _sha256_file(bitmap_path) != state.get("close_bitmap_sha256")
        ):
            raise ExternalMemoryDBSCANError("ALL_PAIRS_CLOSE bitmap mismatch")
        bitmap = np.load(bitmap_path, mmap_mode="r", allow_pickle=False)
        if bitmap.shape != (expected_rows,) or not bool(np.all(bitmap)):
            raise ExternalMemoryDBSCANError(
                "ALL_PAIRS_CLOSE bitmap does not cover every physical row"
            )
        del bitmap
        _promote(
            bitmap_partial,
            bitmap_final,
            expected_sha256=str(state["close_bitmap_sha256"]),
        )
        state = _checkpoint(
            state_path,
            identity=identity,
            phase="complete",
            next_offset=expected_rows,
            close_count=close_count,
            materialized_count=close_count,
            extra={
                "close_bitmap_sha256": state["close_bitmap_sha256"],
                "distance_min": state["distance_min"],
                "distance_max": state["distance_max"],
                "count_distance_eq_theta": state["count_distance_eq_theta"],
                "all_pairs_close_certificate_path": str(all_pairs_certificate[0]),
                "all_pairs_close_certificate_sha256": all_pairs_certificate[1],
                "implicit_pair_indices_sha256": _implicit_pair_sha256(
                    row_count=expected_rows,
                    parent_count=int(contract.parent_count),
                ),
            },
        )
        phase = "complete"

    if phase == "index_only":
        bitmap_path = bitmap_final if bitmap_final.exists() else bitmap_partial
        if (
            not bitmap_path.exists()
            or _sha256_file(bitmap_path) != state.get("close_bitmap_sha256")
        ):
            raise ExternalMemoryDBSCANError("close bitmap checksum mismatch")
        bitmap = np.load(bitmap_path, mmap_mode="r", allow_pickle=False)
        rows_partial = root / "physical_row_indices.partial.npy"
        rows_final = root / "physical_row_indices.npy"
        if rows_partial.exists():
            rows = np.load(rows_partial, mmap_mode="r+", allow_pickle=False)
        else:
            if int(state.get("materialized_count", -1)) != 0:
                raise ExternalMemoryDBSCANError(
                    "close-view indexed partial is missing"
                )
            rows = _new_memmap(rows_partial, dtype=np.int64, shape=(close_count,))
        if rows.shape != (close_count,) or rows.dtype != np.dtype(np.int64):
            raise ExternalMemoryDBSCANError("close-view index schema mismatch")
        start = int(state.get("next_offset", 0))
        cursor = int(state.get("materialized_count", 0))
        _verify_index_prefix(
            bitmap=bitmap,
            physical_rows=rows,
            physical_stop=start,
            logical_stop=cursor,
            block_size=int(block_size),
        )
        for offset in range(start, expected_rows, int(block_size)):
            stop = min(expected_rows, offset + int(block_size))
            selected = np.flatnonzero(bitmap[offset:stop]).astype(np.int64) + offset
            next_cursor = cursor + len(selected)
            rows[cursor:next_cursor] = selected
            _flush(rows)
            cursor = next_cursor
            state = _checkpoint(
                state_path,
                identity=identity,
                phase="index_only",
                next_offset=stop,
                close_count=close_count,
                materialized_count=cursor,
                extra={
                    key: state[key]
                    for key in (
                        "close_bitmap_sha256",
                        "distance_min",
                        "distance_max",
                        "count_distance_eq_theta",
                        "compact_estimated_bytes",
                    )
                },
            )
        if cursor != close_count:
            raise ExternalMemoryDBSCANError("close-view indexed count drift")
        _flush(rows)
        rows_sha = _sha256_file(rows_partial)
        del rows, bitmap
        _promote(
            bitmap_partial,
            bitmap_final,
            expected_sha256=str(state["close_bitmap_sha256"]),
        )
        _promote(
            rows_partial,
            rows_final,
            expected_sha256=rows_sha,
        )
        state = _checkpoint(
            state_path,
            identity=identity,
            phase="complete",
            next_offset=expected_rows,
            close_count=close_count,
            materialized_count=close_count,
            extra={
                "close_bitmap_sha256": state["close_bitmap_sha256"],
                "distance_min": state["distance_min"],
                "distance_max": state["distance_max"],
                "count_distance_eq_theta": state["count_distance_eq_theta"],
                "compact_estimated_bytes": state["compact_estimated_bytes"],
                "physical_row_indices_sha256": rows_sha,
            },
        )
        phase = "complete"

    if phase == "materialize":
        bitmap_path = bitmap_final if bitmap_final.exists() else bitmap_partial
        if (
            not bitmap_path.exists()
            or _sha256_file(bitmap_path) != state.get("close_bitmap_sha256")
        ):
            raise ExternalMemoryDBSCANError("close bitmap checksum mismatch")
        bitmap = np.load(bitmap_path, mmap_mode="r", allow_pickle=False)
        rows_partial = root / "physical_row_indices.partial.npy"
        pairs_partial = root / "pair_indices.partial.npy"
        vectors_partial = root / "recourse_vectors.partial.npy"
        rows_final = root / "physical_row_indices.npy"
        pairs_final = root / "pair_indices.npy"
        vectors_final = root / "recourse_vectors.npy"
        if rows_partial.exists():
            rows = np.load(rows_partial, mmap_mode="r+", allow_pickle=False)
            pairs = np.load(pairs_partial, mmap_mode="r+", allow_pickle=False)
            close_vectors = np.load(
                vectors_partial, mmap_mode="r+", allow_pickle=False
            )
        else:
            if any(path.exists() for path in (pairs_partial, vectors_partial)):
                raise ExternalMemoryDBSCANError(
                    "close-view partial materialization is incomplete"
                )
            rows = _new_memmap(rows_partial, dtype=np.int64, shape=(close_count,))
            pairs = _new_memmap(
                pairs_partial, dtype=np.int64, shape=(close_count, 2)
            )
            close_vectors = _new_memmap(
                vectors_partial,
                dtype=physical.dtype,
                shape=(close_count, int(physical.shape[1])),
            )
        if (
            rows.shape != (close_count,)
            or rows.dtype != np.dtype(np.int64)
            or pairs.shape != (close_count, 2)
            or pairs.dtype != np.dtype(np.int64)
            or close_vectors.shape != (close_count, int(physical.shape[1]))
            or close_vectors.dtype != physical.dtype
        ):
            raise ExternalMemoryDBSCANError(
                "close-view materialized partial schema mismatch"
            )
        start = int(state.get("next_offset", 0))
        cursor = int(state.get("materialized_count", 0))
        if start < 0 or start > expected_rows or cursor < 0 or cursor > close_count:
            raise ExternalMemoryDBSCANError(
                "close-view materialization checkpoint is invalid"
            )
        _verify_materialized_prefix(
            bitmap=bitmap,
            physical_vectors=physical,
            physical_rows=rows,
            pairs=pairs,
            close_vectors=close_vectors,
            physical_stop=start,
            logical_stop=cursor,
            parent_count=int(contract.parent_count),
            block_size=int(block_size),
        )
        for offset in range(start, expected_rows, int(block_size)):
            stop = min(expected_rows, offset + int(block_size))
            selected = np.flatnonzero(bitmap[offset:stop]).astype(np.int64) + offset
            next_cursor = cursor + len(selected)
            rows[cursor:next_cursor] = selected
            pairs[cursor:next_cursor] = _expected_pairs(
                selected, parent_count=int(contract.parent_count)
            )
            close_vectors[cursor:next_cursor] = physical[selected]
            _flush(rows)
            _flush(pairs)
            _flush(close_vectors)
            cursor = next_cursor
            state = _checkpoint(
                state_path,
                identity=identity,
                phase="materialize",
                next_offset=stop,
                close_count=close_count,
                materialized_count=cursor,
                extra={
                    "close_bitmap_sha256": state["close_bitmap_sha256"],
                    "distance_min": state["distance_min"],
                    "distance_max": state["distance_max"],
                    "count_distance_eq_theta": state[
                        "count_distance_eq_theta"
                    ],
                },
            )
        if cursor != close_count:
            raise ExternalMemoryDBSCANError("close-view final count drift")
        for value in (rows, pairs, close_vectors):
            _flush(value)
        hashes = {
            "physical_row_indices_sha256": _sha256_file(rows_partial),
            "pair_indices_sha256": _sha256_file(pairs_partial),
            "recourse_vectors_sha256": _sha256_file(vectors_partial),
        }
        del rows, pairs, close_vectors, bitmap
        _promote(
            bitmap_partial,
            bitmap_final,
            expected_sha256=str(state["close_bitmap_sha256"]),
        )
        _promote(
            rows_partial,
            rows_final,
            expected_sha256=hashes["physical_row_indices_sha256"],
        )
        _promote(
            pairs_partial,
            pairs_final,
            expected_sha256=hashes["pair_indices_sha256"],
        )
        _promote(
            vectors_partial,
            vectors_final,
            expected_sha256=hashes["recourse_vectors_sha256"],
        )
        state = _checkpoint(
            state_path,
            identity=identity,
            phase="complete",
            next_offset=expected_rows,
            close_count=close_count,
            materialized_count=close_count,
            extra={
                "close_bitmap_sha256": state["close_bitmap_sha256"],
                "distance_min": state["distance_min"],
                "distance_max": state["distance_max"],
                "count_distance_eq_theta": state["count_distance_eq_theta"],
                **hashes,
            },
        )
        phase = "complete"

    if phase != "complete":
        raise ExternalMemoryDBSCANError(f"unknown close-view phase: {phase}")
    _verify_file(physical_path, sha256=physical_sha, identity=physical_stat)
    _verify_file(distance_path, sha256=distance_sha, identity=distance_stat)
    if pair_semantics_path is not None:
        assert pair_semantics_sha is not None and pair_semantics_stat is not None
        _verify_file(
            pair_semantics_path,
            sha256=pair_semantics_sha,
            identity=pair_semantics_stat,
        )
    all_pairs_close = close_count == expected_rows
    compact_materialized = (
        not all_pairs_close and "recourse_vectors_sha256" in state
    )
    indexed_only = not all_pairs_close and not compact_materialized
    indexed_pair_sha: str | None = None
    logical_vector_view_sha: str | None = None
    if indexed_only:
        indexed_rows = np.load(
            root / "physical_row_indices.npy", mmap_mode="r", allow_pickle=False
        )
        indexed_pair_sha = _indexed_pair_sha256(
            indexed_rows,
            parent_count=int(contract.parent_count),
            block_size=int(block_size),
        )
        logical_vector_view_sha = _stable_hash(
            {
                "physical_vectors_sha256": physical_sha,
                "physical_row_indices_sha256": state[
                    "physical_row_indices_sha256"
                ],
                "logical_shape": [close_count, int(physical.shape[1])],
                "dtype": str(physical.dtype),
                "order": "physical_row_indices_ascending",
            }
        )
        del indexed_rows
    manifest = {
        "schema_version": CLOSE_PAIR_VIEW_SCHEMA,
        "status": "PASS",
        "run_complete": True,
        "eligible_for_dbscan": not indexed_only,
        "blocking_reason": (
            "BLOCKED_STORAGE_INDEXED_DBSCAN_ENGINE_REQUIRED"
            if indexed_only
            else None
        ),
        "scientific_identity": identity,
        "scientific_identity_sha256": _stable_hash(identity),
        "physical_pair_store_adopted": True,
        "pair_store_regenerated": False,
        "physical_store_rows": expected_rows,
        "physical_pair_count": expected_rows,
        "full_cartesian_count": expected_rows,
        "physical_store_is_full_cartesian": True,
        "logical_close_rows": close_count,
        "logical_close_pair_count": close_count,
        "close_pair_rate": close_count / expected_rows,
        "theta": float(contract.theta),
        "filter_operator": FILTER_OPERATOR,
        "dbscan_input": "theta_close_recourse_only",
        "dbscan_input_count": close_count,
        "pair_orientation": PAIR_ORIENTATION,
        "pair_axis": "col0=parent;col1=candidate",
        "chunk_order": PAIR_ORDER,
        "scale_contract": contract.scale_contract,
        "distance_checkpoint_hash": contract.distance_checkpoint_sha256,
        "embedding_checkpoint_hash": contract.embedding_checkpoint_sha256,
        "normalized_distance_dtype": str(distances.dtype),
        "recourse_vector_dtype": str(physical.dtype),
        "distance_min": float(state["distance_min"]),
        "distance_max": float(state["distance_max"]),
        "count_distance_eq_theta": int(state["count_distance_eq_theta"]),
        "view_storage": (
            "zero_copy_full_cartesian"
            if all_pairs_close
            else (
                "materialized_selected_rows"
                if compact_materialized
                else "bitmap_index_zero_copy"
            )
        ),
        "compact_estimated_bytes": int(state.get("compact_estimated_bytes", 0)),
        "compact_storage_budget_bytes": int(max_compact_bytes),
        "large_vector_copy_materialized": compact_materialized,
        "close_bitmap_path": str(root / "close_bitmap.npy"),
        "close_bitmap_hash": str(state["close_bitmap_sha256"]),
        "physical_row_indices_path": (
            None if all_pairs_close else str(root / "physical_row_indices.npy")
        ),
        "physical_row_indices_sha256": (
            None if all_pairs_close else str(state["physical_row_indices_sha256"])
        ),
        "pairs_storage": (
            "implicit_cartesian_v1"
            if all_pairs_close
            else (
                "physical_npy"
                if compact_materialized
                else "indexed_cartesian_v1"
            )
        ),
        "pair_indices_path": (
            str(root / "pair_indices.npy") if compact_materialized else None
        ),
        "pair_indices_sha256": (
            str(state["implicit_pair_indices_sha256"])
            if all_pairs_close
            else (
                str(state["pair_indices_sha256"])
                if compact_materialized
                else indexed_pair_sha
            )
        ),
        "recourse_vectors_path": (
            str(physical_path)
            if all_pairs_close or indexed_only
            else str(root / "recourse_vectors.npy")
        ),
        "recourse_vectors_sha256": (
            physical_sha
            if all_pairs_close or indexed_only
            else str(state["recourse_vectors_sha256"])
        ),
        "logical_vector_view_sha256": logical_vector_view_sha,
        "recourse_vectors_copied_byte_exact_from_physical_rows": (
            all_pairs_close or compact_materialized
        ),
        "recourse_vectors_zero_copy_indexed_from_physical_rows": indexed_only,
        "recourse_vectors_recomputed": False,
        "all_pairs_close": all_pairs_close,
        "all_pairs_close_certificate_path": (
            str(state["all_pairs_close_certificate_path"])
            if all_pairs_close
            else None
        ),
        "all_pairs_close_certificate_sha256": (
            str(state["all_pairs_close_certificate_sha256"])
            if all_pairs_close
            else None
        ),
        "approximation_used": False,
    }
    _atomic_json(manifest_path, manifest)
    return validate_theta_close_pair_view(
        manifest_path,
        expected_contract=contract,
        expected_physical_vectors_path=physical_path,
        expected_physical_vectors_sha256=physical_sha,
        expected_normalized_distances_path=distance_path,
        expected_normalized_distances_sha256=distance_sha,
        expected_pair_semantics_contract_path=pair_semantics_path,
        expected_pair_semantics_contract_sha256=pair_semantics_sha,
        replay_block_size=int(block_size),
    )


def validate_theta_close_pair_view(
    manifest_path: str | Path,
    *,
    expected_contract: ThetaClosePairContract | None = None,
    expected_physical_vectors_path: str | Path | None = None,
    expected_physical_vectors_sha256: str | None = None,
    expected_normalized_distances_path: str | Path | None = None,
    expected_normalized_distances_sha256: str | None = None,
    expected_pair_semantics_contract_path: str | Path | None = None,
    expected_pair_semantics_contract_sha256: str | None = None,
    replay_block_size: int = 1_000_000,
    require_dbscan_eligible: bool = False,
    require_pair_semantics_authority: bool = False,
) -> ThetaClosePairView:
    """Reopen a terminal view and replay its exact predicate/mapping closure."""

    if int(replay_block_size) <= 0:
        raise ExternalMemoryDBSCANError("close-view replay block size must be positive")
    path = Path(manifest_path).expanduser().resolve(strict=True)
    root = path.parent
    manifest = _load_object(path)
    identity = manifest.get("scientific_identity")
    if (
        manifest.get("schema_version") != CLOSE_PAIR_VIEW_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("run_complete") is not True
        or manifest.get("eligible_for_dbscan") not in {True, False}
        or manifest.get("approximation_used") is not False
        or not isinstance(identity, Mapping)
        or manifest.get("scientific_identity_sha256") != _stable_hash(identity)
        or identity.get("schema_version") != CLOSE_PAIR_VIEW_SCHEMA
        or manifest.get("filter_operator") != FILTER_OPERATOR
        or manifest.get("pair_orientation") != PAIR_ORIENTATION
        or manifest.get("pair_axis") != "col0=parent;col1=candidate"
        or manifest.get("chunk_order") != PAIR_ORDER
        or manifest.get("dbscan_input") != "theta_close_recourse_only"
        or manifest.get("physical_store_is_full_cartesian") is not True
        or manifest.get("recourse_vectors_recomputed") is not False
    ):
        raise ExternalMemoryDBSCANError("logical theta-close manifest is not PASS")
    eligible_for_dbscan = manifest.get("eligible_for_dbscan") is True
    if require_dbscan_eligible and not eligible_for_dbscan:
        raise ExternalMemoryDBSCANError(
            str(manifest.get("blocking_reason") or "CLOSE_VIEW_NOT_DBSCAN_ELIGIBLE")
        )
    raw_contract = identity.get("contract")
    if not isinstance(raw_contract, Mapping):
        raise ExternalMemoryDBSCANError("logical theta-close contract is absent")
    try:
        contract = ThetaClosePairContract(**dict(raw_contract))
    except TypeError as exc:
        raise ExternalMemoryDBSCANError("logical theta-close contract is invalid") from exc
    contract.validate()
    if expected_contract is not None and contract != expected_contract:
        raise ExternalMemoryDBSCANError("logical theta-close contract mismatch")

    physical_path = Path(str(identity.get("physical_vectors_path") or "")).resolve(
        strict=True
    )
    distance_path = Path(
        str(identity.get("normalized_distances_path") or "")
    ).resolve(strict=True)
    if (
        expected_physical_vectors_path is not None
        and physical_path
        != Path(expected_physical_vectors_path).expanduser().resolve(strict=True)
    ):
        raise ExternalMemoryDBSCANError("logical view physical source path mismatch")
    if (
        expected_physical_vectors_sha256 is not None
        and identity.get("physical_vectors_sha256")
        != str(expected_physical_vectors_sha256)
    ):
        raise ExternalMemoryDBSCANError("logical view physical source hash mismatch")
    if (
        expected_normalized_distances_path is not None
        and distance_path
        != Path(expected_normalized_distances_path).expanduser().resolve(strict=True)
    ):
        raise ExternalMemoryDBSCANError("logical view distance source path mismatch")
    if (
        expected_normalized_distances_sha256 is not None
        and identity.get("normalized_distances_sha256")
        != str(expected_normalized_distances_sha256)
    ):
        raise ExternalMemoryDBSCANError("logical view distance source hash mismatch")
    pair_semantics_raw = identity.get("pair_semantics_contract_path")
    pair_semantics_sha = identity.get("pair_semantics_contract_sha256")
    pair_semantics_stat = identity.get("pair_semantics_contract_stat_identity")
    authority_absent = (
        pair_semantics_raw is None
        and pair_semantics_sha is None
        and pair_semantics_stat is None
    )
    if require_pair_semantics_authority and authority_absent:
        raise ExternalMemoryDBSCANError("PAIR_SEMANTICS_AUTHORITY_NOT_BOUND")
    pair_semantics_path: Path | None = None
    if not authority_absent:
        if (
            pair_semantics_raw is None
            or not isinstance(pair_semantics_sha, str)
            or not isinstance(pair_semantics_stat, Mapping)
        ):
            raise ExternalMemoryDBSCANError(
                "logical view pair-semantics authority closure mismatch"
            )
        pair_semantics_path = Path(str(pair_semantics_raw)).resolve(strict=True)
        _verify_file(
            pair_semantics_path,
            sha256=pair_semantics_sha,
            identity=pair_semantics_stat,
        )
    if expected_pair_semantics_contract_path is not None:
        expected_authority_path = Path(
            expected_pair_semantics_contract_path
        ).expanduser().resolve(strict=True)
        if pair_semantics_path != expected_authority_path:
            raise ExternalMemoryDBSCANError(
                "logical view pair-semantics authority path mismatch"
            )
    if (
        expected_pair_semantics_contract_sha256 is not None
        and pair_semantics_sha != str(expected_pair_semantics_contract_sha256)
    ):
        raise ExternalMemoryDBSCANError(
            "logical view pair-semantics authority hash mismatch"
        )
    physical_stat = identity.get("physical_vectors_stat_identity")
    distance_stat = identity.get("normalized_distances_stat_identity")
    if not isinstance(physical_stat, Mapping) or not isinstance(distance_stat, Mapping):
        raise ExternalMemoryDBSCANError("logical view source stat identity is absent")
    _verify_file(
        physical_path,
        sha256=str(identity.get("physical_vectors_sha256") or ""),
        identity=physical_stat,
    )
    _verify_file(
        distance_path,
        sha256=str(identity.get("normalized_distances_sha256") or ""),
        identity=distance_stat,
    )
    all_pairs_close = manifest.get("all_pairs_close") is True
    indexed_only = manifest.get("view_storage") == "bitmap_index_zero_copy"
    if manifest.get("all_pairs_close") not in {True, False}:
        raise ExternalMemoryDBSCANError("logical theta-close all-pairs state is absent")
    artifacts: dict[str, tuple[Path, str]] = {}
    artifact_fields = [("close_bitmap_path", "close_bitmap_hash")]
    if not all_pairs_close:
        artifact_fields.extend(
            [("physical_row_indices_path", "physical_row_indices_sha256")]
        )
        if not indexed_only:
            artifact_fields.extend(
                [
                    ("pair_indices_path", "pair_indices_sha256"),
                    ("recourse_vectors_path", "recourse_vectors_sha256"),
                ]
            )
    for path_field, hash_field in artifact_fields:
        artifact = Path(str(manifest.get(path_field) or "")).resolve(strict=True)
        if artifact.parent != root or _sha256_file(artifact) != manifest.get(hash_field):
            raise ExternalMemoryDBSCANError(
                f"logical theta-close artifact closure mismatch: {path_field}"
            )
        artifacts[path_field] = (artifact, str(manifest[hash_field]))

    if all_pairs_close:
        if (
            manifest.get("view_storage") != "zero_copy_full_cartesian"
            or manifest.get("pairs_storage") != "implicit_cartesian_v1"
            or manifest.get("physical_row_indices_path") is not None
            or manifest.get("physical_row_indices_sha256") is not None
            or manifest.get("pair_indices_path") is not None
            or manifest.get("recourse_vectors_path") != str(physical_path)
            or manifest.get("recourse_vectors_sha256")
            != identity.get("physical_vectors_sha256")
        ):
            raise ExternalMemoryDBSCANError(
                "ALL_PAIRS_CLOSE zero-copy storage contract mismatch"
            )
        _certificate_path, certificate_sha = _validate_all_pairs_close_certificate(
            str(manifest.get("all_pairs_close_certificate_path") or ""),
            identity=identity,
            contract=contract,
            physical_rows=int(contract.parent_count) * int(contract.candidate_count),
            count_equal_theta=int(manifest.get("count_distance_eq_theta", -1)),
        )
        if certificate_sha != manifest.get("all_pairs_close_certificate_sha256"):
            raise ExternalMemoryDBSCANError(
                "ALL_PAIRS_CLOSE certificate hash mismatch"
            )
        if manifest.get("pair_indices_sha256") != _implicit_pair_sha256(
            row_count=int(contract.parent_count) * int(contract.candidate_count),
            parent_count=int(contract.parent_count),
        ):
            raise ExternalMemoryDBSCANError(
                "ALL_PAIRS_CLOSE implicit pair hash mismatch"
            )
        artifacts["recourse_vectors_path"] = (
            physical_path,
            str(identity["physical_vectors_sha256"]),
        )
    elif indexed_only:
        if (
            eligible_for_dbscan
            or manifest.get("blocking_reason")
            != "BLOCKED_STORAGE_INDEXED_DBSCAN_ENGINE_REQUIRED"
            or manifest.get("pairs_storage") != "indexed_cartesian_v1"
            or manifest.get("pair_indices_path") is not None
            or manifest.get("recourse_vectors_path") != str(physical_path)
            or manifest.get("recourse_vectors_sha256")
            != identity.get("physical_vectors_sha256")
            or manifest.get("large_vector_copy_materialized") is not False
            or manifest.get(
                "recourse_vectors_zero_copy_indexed_from_physical_rows"
            )
            is not True
        ):
            raise ExternalMemoryDBSCANError(
                "indexed close-view storage contract mismatch"
            )
        artifacts["recourse_vectors_path"] = (
            physical_path,
            str(identity["physical_vectors_sha256"]),
        )
    elif (
        manifest.get("view_storage") != "materialized_selected_rows"
        or not eligible_for_dbscan
        or manifest.get("large_vector_copy_materialized") is not True
        or manifest.get("recourse_vectors_copied_byte_exact_from_physical_rows")
        is not True
    ):
        raise ExternalMemoryDBSCANError(
            "materialized close-view storage contract mismatch"
        )

    physical = np.load(physical_path, mmap_mode="r", allow_pickle=False)
    distances = np.load(distance_path, mmap_mode="r", allow_pickle=False)
    bitmap = np.load(artifacts["close_bitmap_path"][0], mmap_mode="r", allow_pickle=False)
    rows = (
        None
        if all_pairs_close
        else np.load(
            artifacts["physical_row_indices_path"][0],
            mmap_mode="r",
            allow_pickle=False,
        )
    )
    pairs = (
        None
        if all_pairs_close or indexed_only
        else np.load(
            artifacts["pair_indices_path"][0], mmap_mode="r", allow_pickle=False
        )
    )
    vectors = np.load(
        artifacts["recourse_vectors_path"][0], mmap_mode="r", allow_pickle=False
    )
    physical_count = int(contract.parent_count) * int(contract.candidate_count)
    close_count = int(manifest.get("logical_close_rows", -1))
    if (
        physical.ndim != 2
        or physical.shape[0] != physical_count
        or physical.shape[1] <= 0
        or distances.shape != (physical_count,)
        or bitmap.shape != (physical_count,)
        or bitmap.dtype != np.dtype(np.bool_)
        or (
            not all_pairs_close
            and (
                rows is None
                or rows.shape != (close_count,)
                or rows.dtype != np.dtype(np.int64)
                or (
                    not indexed_only
                    and (
                        pairs is None
                        or pairs.shape != (close_count, 2)
                        or pairs.dtype != np.dtype(np.int64)
                    )
                )
            )
        )
        or (
            indexed_only
            and vectors.shape != physical.shape
        )
        or (
            not indexed_only
            and vectors.shape != (close_count, int(physical.shape[1]))
        )
        or vectors.dtype != physical.dtype
        or close_count < 0
        or close_count > physical_count
        or (all_pairs_close and close_count != physical_count)
        or (not all_pairs_close and close_count >= physical_count)
        or int(manifest.get("physical_store_rows", -1)) != physical_count
        or int(manifest.get("physical_pair_count", -1)) != physical_count
        or int(manifest.get("full_cartesian_count", -1)) != physical_count
        or int(manifest.get("logical_close_pair_count", -1)) != close_count
        or int(manifest.get("dbscan_input_count", -1)) != close_count
    ):
        raise ExternalMemoryDBSCANError("logical theta-close array/count contract mismatch")
    replayed_count = _verify_scan_prefix(
        bitmap=bitmap,
        distances=distances,
        stop=physical_count,
        theta=float(contract.theta),
        block_size=int(replay_block_size),
    )
    if replayed_count != close_count or int(np.count_nonzero(bitmap)) != close_count:
        raise ExternalMemoryDBSCANError("logical theta-close count does not replay")
    if all_pairs_close:
        # Path/hash identity above is the zero-copy vector proof; a separate
        # mmap object does not necessarily share a Python base object.
        if not bool(np.all(bitmap)):
            raise ExternalMemoryDBSCANError(
                "ALL_PAIRS_CLOSE bitmap lost a physical row"
            )
    elif indexed_only:
        assert rows is not None
        _verify_index_prefix(
            bitmap=bitmap,
            physical_rows=rows,
            physical_stop=physical_count,
            logical_stop=close_count,
            block_size=int(replay_block_size),
        )
        if manifest.get("pair_indices_sha256") != _indexed_pair_sha256(
            rows,
            parent_count=int(contract.parent_count),
            block_size=int(replay_block_size),
        ):
            raise ExternalMemoryDBSCANError("indexed pair identity mismatch")
        expected_view_sha = _stable_hash(
            {
                "physical_vectors_sha256": identity["physical_vectors_sha256"],
                "physical_row_indices_sha256": manifest[
                    "physical_row_indices_sha256"
                ],
                "logical_shape": [close_count, int(physical.shape[1])],
                "dtype": str(physical.dtype),
                "order": "physical_row_indices_ascending",
            }
        )
        if manifest.get("logical_vector_view_sha256") != expected_view_sha:
            raise ExternalMemoryDBSCANError("indexed vector-view identity mismatch")
    else:
        assert rows is not None and pairs is not None
        _verify_materialized_prefix(
            bitmap=bitmap,
            physical_vectors=physical,
            physical_rows=rows,
            pairs=pairs,
            close_vectors=vectors,
            physical_stop=physical_count,
            logical_stop=close_count,
            parent_count=int(contract.parent_count),
            block_size=int(replay_block_size),
        )
    return ThetaClosePairView(
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        bitmap_path=artifacts["close_bitmap_path"][0],
        physical_row_indices_path=(
            None
            if all_pairs_close
            else artifacts["physical_row_indices_path"][0]
        ),
        pairs_path=(
            None
            if all_pairs_close or indexed_only
            else artifacts["pair_indices_path"][0]
        ),
        vectors_path=artifacts["recourse_vectors_path"][0],
        physical_store_rows=physical_count,
        logical_close_rows=close_count,
        parent_count=int(contract.parent_count),
        candidate_count=int(contract.candidate_count),
        theta=float(contract.theta),
        pairs_sha256=str(manifest["pair_indices_sha256"]),
        vectors_sha256=artifacts["recourse_vectors_path"][1],
        all_pairs_close=all_pairs_close,
        view_storage=str(manifest["view_storage"]),
        eligible_for_dbscan=eligible_for_dbscan,
        blocking_reason=(
            None
            if manifest.get("blocking_reason") is None
            else str(manifest["blocking_reason"])
        ),
        pair_semantics_contract_path=pair_semantics_path,
        pair_semantics_contract_sha256=(
            None if pair_semantics_sha is None else str(pair_semantics_sha)
        ),
    )


__all__ = [
    "ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA",
    "CartesianThetaClosePairs",
    "CLOSE_PAIR_VIEW_SCHEMA",
    "FILTER_OPERATOR",
    "PAIR_ORDER",
    "PAIR_ORIENTATION",
    "SCALE_CONTRACT",
    "NORMALIZED_DISTANCE_CONTRACT",
    "ThetaClosePairContract",
    "ThetaClosePairView",
    "IndexedThetaCloseVectors",
    "materialize_theta_close_pair_view",
    "validate_theta_close_pair_view",
]
