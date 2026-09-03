"""Bounded, dataset-specific state for the TasteMolNet T12 VRRW route.

The bounded replay canary intentionally keeps every complete bridge row in
memory.  That is useful for exact replay evidence, but it is not a production
layout: the pinned 20,000-step walk may score as many as 200,000,001 rows at
the official ``sample_size=10000``.  This module is the production-only
substrate.  It retains no historical graph payload, coverage vector, or lineage
payload.  It appends one fixed-size observation to an authenticated chain and
keeps the first compact semantic row in a derived SQLite index with a bounded
page cache.  First-seen GINE embedding bytes are the one deliberate exception:
they live in a separate, bounded append-only binary authority so eviction and
process restart never make a new GPU reduction the canonical value.

The journal is the authority; SQLite is only a disposable lookup index and is
rebuilt from the committed journal prefix after a process restart.  A
checkpoint binds a byte prefix, so an uncommitted tail left by a failed worker
does not need to be truncated or trusted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import struct
import time
from typing import Any, Mapping, Sequence
import uuid


HISTORY_SCHEMA = "tastemolnet_t12_compact_history_v2"
HISTORY_SEGMENT_SCHEMA = "tastemolnet_t12_compact_history_segment_v2"
HISTORY_SNAPSHOT_SCHEMA = "tastemolnet_t12_compact_history_snapshot_v2"
BOUNDS_SCHEMA = "tastemolnet_t12_bounded_production_limits_v2"
HISTORY_MAGIC = b"T12HST2\n"
HISTORY_HEADER_LIMIT_BYTES = 4096
HISTORY_MAX_SEGMENTS = 2
HISTORY_INDEX_CACHE_KIB = 64 * 1024
HISTORY_INDEX_BUSY_TIMEOUT_MILLISECONDS = 120_000
HISTORY_INDEX_RETRY_INITIAL_SECONDS = 0.05
HISTORY_INDEX_RETRY_MAX_SECONDS = 0.5

FIRST_EMBEDDING_STORE_SCHEMA = "tastemolnet_t12_first_embedding_store_v1"
FIRST_EMBEDDING_SEGMENT_SCHEMA = "tastemolnet_t12_first_embedding_segment_v1"
FIRST_EMBEDDING_SNAPSHOT_SCHEMA = "tastemolnet_t12_first_embedding_snapshot_v1"
FIRST_EMBEDDING_MAGIC = b"T12EMB1\n"
FIRST_EMBEDDING_HEADER_LIMIT_BYTES = 4096
FIRST_EMBEDDING_DTYPE_LIMIT_BYTES = 64
FIRST_EMBEDDING_MAX_NDIM = 8
# sequence; graph/raw/semantic-embedding/model/feature SHA-256; dtype length;
# ndim; raw length.
_FIRST_EMBEDDING_PREFIX = struct.Struct(">Q32s32s32s32s32sHHQ")

PINNED_TOTAL_STEPS = 20_000
PINNED_CHECKPOINT_CURSORS = (10_000, 20_000)
PINNED_SAMPLE_SIZE = 10_000
PINNED_CANDIDATE_CAPACITY = 100_000

# Sequence; graph/semantic/embedding/coverage/failure/raw-query digests; three
# probabilities; prediction; flags; padding; covered-parent count; and lineage
# digest.  The resulting chain head follows as a final 32-byte field.
_OBSERVATION_BODY = struct.Struct(">Q32s32s32s32s32s32sdddBB6xQ32s")
HISTORY_RECORD_BYTES = _OBSERVATION_BODY.size + 32

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TasteT12ProductionStateError(RuntimeError):
    """A compact-history or production resource contract is invalid."""


def _is_sqlite_lock_error(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if type(code) is int and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return True
    message = str(error).lower()
    return "database is locked" in message or "database is busy" in message


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TasteT12ProductionStateError(f"{field} must be lowercase SHA-256")
    return value


def _require_native_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TasteT12ProductionStateError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _normalized_absolute(path: str | Path, *, field: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or Path(os.path.abspath(value)) != value:
        raise TasteT12ProductionStateError(
            f"{field} must be normalized and absolute"
        )
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class T12ProductionBounds:
    """Finite upper bounds for the pinned 20k production bridge.

    These values are deliberately conservative.  They are gates, not resource
    estimates: a row larger than either per-row cap, a journal beyond its disk
    cap, or a published checkpoint beyond its cap fails before publication.
    """

    schema_version: str
    total_steps: int
    checkpoint_cursors: tuple[int, int]
    sample_size: int
    candidate_capacity: int
    parent_count: int
    max_scored_observations: int
    max_full_live_records: int
    max_transient_full_records: int
    history_record_bytes: int
    max_history_bytes: int
    max_live_record_deep_bytes: int
    max_live_record_serialized_bytes: int
    max_bridge_ram_bytes: int
    max_bridge_checkpoint_bytes: int
    max_full_checkpoint_bytes: int
    sqlite_page_cache_bytes: int

    @classmethod
    def pinned(
        cls,
        *,
        parent_count: int,
        max_live_record_deep_bytes: int = 512 * 1024,
        max_live_record_serialized_bytes: int = 256 * 1024,
        max_bridge_ram_bytes: int = 16 * 1024**3,
        max_bridge_checkpoint_bytes: int = 8 * 1024**3,
        max_full_checkpoint_bytes: int = 32 * 1024**3,
        max_history_bytes: int = 64 * 1024**3,
    ) -> "T12ProductionBounds":
        parents = _require_native_int(
            parent_count, field="T12 production parent_count", minimum=1
        )
        # At most one selected graph enters graph_map per completed step, plus
        # the initial restart graph.  Candidate capacity is larger (100k), but
        # cannot be reached in a 20k walk.
        live_records = min(
            PINNED_CANDIDATE_CAPACITY, PINNED_TOTAL_STEPS + 1
        )
        transient_records = live_records + PINNED_SAMPLE_SIZE
        max_observations = 1 + PINNED_TOTAL_STEPS * PINNED_SAMPLE_SIZE
        minimum_history_bytes = (
            max_observations * HISTORY_RECORD_BYTES
            + HISTORY_MAX_SEGMENTS * (len(HISTORY_MAGIC) + 4 + HISTORY_HEADER_LIMIT_BYTES)
        )
        bridge_ram_formula = (
            transient_records * int(max_live_record_deep_bytes)
            + HISTORY_INDEX_CACHE_KIB * 1024
            + 64 * 1024**2
        )
        bridge_checkpoint_formula = (
            live_records * int(max_live_record_serialized_bytes)
            + 64 * 1024**2
        )
        if minimum_history_bytes > int(max_history_bytes):
            raise TasteT12ProductionStateError(
                "T12 compact history disk cap cannot hold the pinned 20k/10k route"
            )
        if bridge_ram_formula > int(max_bridge_ram_bytes):
            raise TasteT12ProductionStateError(
                "T12 bridge RAM cap cannot hold the pinned live-record bound"
            )
        if bridge_checkpoint_formula > int(max_bridge_checkpoint_bytes):
            raise TasteT12ProductionStateError(
                "T12 bridge checkpoint cap cannot hold the pinned live-record bound"
            )
        if int(max_bridge_checkpoint_bytes) >= int(max_full_checkpoint_bytes):
            raise TasteT12ProductionStateError(
                "T12 full checkpoint cap must exceed the bridge component cap"
            )
        return cls(
            schema_version=BOUNDS_SCHEMA,
            total_steps=PINNED_TOTAL_STEPS,
            checkpoint_cursors=PINNED_CHECKPOINT_CURSORS,
            sample_size=PINNED_SAMPLE_SIZE,
            candidate_capacity=PINNED_CANDIDATE_CAPACITY,
            parent_count=parents,
            max_scored_observations=max_observations,
            max_full_live_records=live_records,
            max_transient_full_records=transient_records,
            history_record_bytes=HISTORY_RECORD_BYTES,
            max_history_bytes=int(max_history_bytes),
            max_live_record_deep_bytes=int(max_live_record_deep_bytes),
            max_live_record_serialized_bytes=int(
                max_live_record_serialized_bytes
            ),
            max_bridge_ram_bytes=int(max_bridge_ram_bytes),
            max_bridge_checkpoint_bytes=int(max_bridge_checkpoint_bytes),
            max_full_checkpoint_bytes=int(max_full_checkpoint_bytes),
            sqlite_page_cache_bytes=HISTORY_INDEX_CACHE_KIB * 1024,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "T12ProductionBounds":
        if type(value) is not dict or set(value) != {
            field.name for field in cls.__dataclass_fields__.values()
        }:
            raise TasteT12ProductionStateError("T12 production bounds keys changed")
        try:
            result = cls(
                **{
                    **value,
                    "checkpoint_cursors": tuple(value["checkpoint_cursors"]),
                }
            )
        except (TypeError, ValueError) as exc:
            raise TasteT12ProductionStateError(
                "T12 production bounds are malformed"
            ) from exc
        expected = cls.pinned(
            parent_count=result.parent_count,
            max_live_record_deep_bytes=result.max_live_record_deep_bytes,
            max_live_record_serialized_bytes=(
                result.max_live_record_serialized_bytes
            ),
            max_bridge_ram_bytes=result.max_bridge_ram_bytes,
            max_bridge_checkpoint_bytes=result.max_bridge_checkpoint_bytes,
            max_full_checkpoint_bytes=result.max_full_checkpoint_bytes,
            max_history_bytes=result.max_history_bytes,
        )
        if result != expected:
            raise TasteT12ProductionStateError(
                "T12 production bounds differ from the pinned formula"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checkpoint_cursors"] = list(self.checkpoint_cursors)
        return value

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.to_dict()))

    def proof(self) -> dict[str, Any]:
        formula_ram = (
            self.max_transient_full_records * self.max_live_record_deep_bytes
            + self.sqlite_page_cache_bytes
            + 64 * 1024**2
        )
        formula_checkpoint = (
            self.max_full_live_records
            * self.max_live_record_serialized_bytes
            + 64 * 1024**2
        )
        formula_history = (
            self.max_scored_observations * self.history_record_bytes
            + HISTORY_MAX_SEGMENTS
            * (len(HISTORY_MAGIC) + 4 + HISTORY_HEADER_LIMIT_BYTES)
        )
        return {
            "schema_version": "tastemolnet_t12_resource_bound_proof_v1",
            "bounds_sha256": self.sha256,
            "scientific_parameters": {
                "M": self.total_steps,
                "sample_size": self.sample_size,
                "candidate_capacity": self.candidate_capacity,
                "checkpoint_cursors": list(self.checkpoint_cursors),
            },
            "max_scored_observations_formula": "1 + M * sample_size",
            "max_scored_observations": self.max_scored_observations,
            "max_full_live_records_formula": "min(candidate_capacity, M + 1)",
            "max_full_live_records": self.max_full_live_records,
            "max_transient_full_records_formula": (
                "max_full_live_records + sample_size"
            ),
            "max_transient_full_records": self.max_transient_full_records,
            "history_record_bytes": self.history_record_bytes,
            "history_formula_bytes": formula_history,
            "history_cap_bytes": self.max_history_bytes,
            "bridge_ram_formula_bytes": formula_ram,
            "bridge_ram_cap_bytes": self.max_bridge_ram_bytes,
            "bridge_checkpoint_formula_bytes": formula_checkpoint,
            "bridge_checkpoint_cap_bytes": self.max_bridge_checkpoint_bytes,
            "full_checkpoint_hard_cap_bytes": self.max_full_checkpoint_bytes,
            "history_payload_retained": False,
            "history_embedding_values_retained": False,
            "first_seen_embedding_raw_bytes_retained_append_only": True,
            "history_lineage_payload_retained": False,
            "history_neurosed_query_sha256_retained": True,
            "bound_pass": (
                formula_history <= self.max_history_bytes
                and formula_ram <= self.max_bridge_ram_bytes
                and formula_checkpoint <= self.max_bridge_checkpoint_bytes
                and self.max_bridge_checkpoint_bytes
                < self.max_full_checkpoint_bytes
            ),
        }


@dataclass(frozen=True, slots=True)
class CompactFirstObservation:
    probabilities: tuple[float, float, float]
    prediction: int
    candidate: bool
    valid_fullgraph: bool
    covered_parent_count: int
    embedding_sha256: str
    coverage_sha256: str
    failure_sha256: str
    neurosed_query_sha256: str


def compact_semantic_sha256(
    *,
    graph_identity_sha256: str,
    probabilities: Sequence[float],
    prediction: int,
    candidate: bool,
    valid_fullgraph: bool,
    covered_parent_count: int,
    embedding_sha256: str,
    coverage_sha256: str,
    failure_sha256: str,
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": "tastemolnet_t12_compact_observation_v1",
                "graph_identity_sha256": graph_identity_sha256,
                "probability_hex": [float(value).hex() for value in probabilities],
                "prediction": prediction,
                "candidate": candidate,
                "valid_fullgraph": valid_fullgraph,
                "covered_parent_count": covered_parent_count,
                "embedding_sha256": embedding_sha256,
                "coverage_sha256": coverage_sha256,
                "failure_sha256": failure_sha256,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class FirstSeenEmbedding:
    """One exact first-seen embedding reconstructed from the binary authority."""

    graph_identity_sha256: str
    dtype: str
    shape: tuple[int, ...]
    raw_bytes: bytes
    raw_sha256: str
    embedding_sha256: str
    model_sha256: str
    feature_schema_sha256: str


@dataclass(frozen=True, slots=True)
class _FirstEmbeddingLocator:
    path: Path
    raw_offset: int
    raw_length: int
    dtype: str
    shape: tuple[int, ...]
    raw_sha256: str
    embedding_sha256: str


def _validate_first_embedding_payload(
    *, dtype: Any, shape: Sequence[int], raw_bytes: Any
) -> tuple[str, tuple[int, ...], bytes, str, str]:
    """Normalize one NumPy-compatible embedding without changing its bytes."""

    import numpy as np

    if (
        type(dtype) is not str
        or not dtype
        or not dtype.isascii()
        or len(dtype) > FIRST_EMBEDDING_DTYPE_LIMIT_BYTES
    ):
        raise TasteT12ProductionStateError("T12 first embedding dtype is invalid")
    try:
        normalized_dtype = np.dtype(dtype)
    except TypeError as exc:
        raise TasteT12ProductionStateError(
            "T12 first embedding dtype is invalid"
        ) from exc
    if normalized_dtype.str != dtype or normalized_dtype.kind != "f":
        raise TasteT12ProductionStateError(
            "T12 first embedding dtype must be one canonical float dtype"
        )
    if not isinstance(shape, (tuple, list)):
        raise TasteT12ProductionStateError("T12 first embedding shape is invalid")
    normalized_shape = tuple(shape)
    if (
        not 1 <= len(normalized_shape) <= FIRST_EMBEDDING_MAX_NDIM
        or any(type(value) is not int or value <= 0 for value in normalized_shape)
    ):
        raise TasteT12ProductionStateError("T12 first embedding shape is invalid")
    if type(raw_bytes) is not bytes:
        raise TasteT12ProductionStateError("T12 first embedding raw value is not bytes")
    expected_bytes = math.prod(normalized_shape) * int(normalized_dtype.itemsize)
    if len(raw_bytes) != expected_bytes:
        raise TasteT12ProductionStateError(
            "T12 first embedding byte length differs from dtype/shape"
        )
    raw_sha256 = _sha256_bytes(raw_bytes)
    semantic = hashlib.sha256()
    semantic.update(str(normalized_dtype).encode("ascii"))
    semantic.update(_canonical_bytes(list(normalized_shape)))
    semantic.update(raw_bytes)
    return (
        dtype,
        normalized_shape,
        raw_bytes,
        raw_sha256,
        semantic.hexdigest(),
    )


class T12FirstSeenEmbeddingStore:
    """Bounded append-only authority for exact first-seen GINE bytes.

    The in-memory index contains only file offsets and compact metadata.  Raw
    values are read from their authenticated segment on demand, so keeping the
    first observation stable does not turn the 20k production bridge into an
    unbounded RAM cache.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        bounds: T12ProductionBounds,
        contract_sha256: str,
        attempt_id: str,
        generation_token: str,
        model_sha256: str,
        feature_schema_sha256: str,
        resume_snapshot: Mapping[str, Any] | None = None,
        open_writer: bool = True,
    ) -> None:
        self.root = _normalized_absolute(root, field="T12 first embedding root")
        self.bounds = bounds
        self.contract_sha256 = _require_sha256(
            contract_sha256, field="T12 first embedding contract"
        )
        self.attempt_id = attempt_id
        self.generation_token = _require_sha256(
            generation_token, field="T12 first embedding generation token"
        )
        self.model_sha256 = _require_sha256(
            model_sha256, field="T12 first embedding model"
        )
        self.feature_schema_sha256 = _require_sha256(
            feature_schema_sha256, field="T12 first embedding feature schema"
        )
        if resume_snapshot is None:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        elif (
            not self.root.is_dir()
            or self.root.is_symlink()
            or self.root.resolve(strict=True) != self.root
        ):
            raise TasteT12ProductionStateError(
                "T12 resume first embedding root is not one physical directory"
            )
        self._segments: list[dict[str, Any]] = []
        self._locators: dict[str, _FirstEmbeddingLocator] = {}
        self.sequence = 0
        self.chain_head = "0" * 64
        self._active_stream: Any | None = None
        self._active_path: Path | None = None
        self._active_header: dict[str, Any] | None = None
        self._active_header_bytes = b""
        self._active_record_count = 0
        self._active_prefix_digest: Any | None = None
        if resume_snapshot is not None:
            self._restore_snapshot(resume_snapshot)
        if open_writer:
            self.start_writer()

    @property
    def record_count(self) -> int:
        return self.sequence

    @property
    def graph_hashes(self) -> frozenset[str]:
        return frozenset(self._locators)

    def start_writer(self) -> None:
        if self._active_stream is not None:
            raise TasteT12ProductionStateError(
                "T12 first embedding segment is already open"
            )
        if len(self._segments) >= HISTORY_MAX_SEGMENTS:
            raise TasteT12ProductionStateError(
                "T12 first embedding store exceeded the two process segments"
            )
        segment_id = str(uuid.uuid4())
        header = {
            "schema_version": FIRST_EMBEDDING_SEGMENT_SCHEMA,
            "segment_id": segment_id,
            "contract_sha256": self.contract_sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "model_sha256": self.model_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "bounds_sha256": self.bounds.sha256,
            "anchor_sequence": self.sequence,
            "anchor_chain_head": self.chain_head,
        }
        header_bytes = _canonical_bytes(header)
        if len(header_bytes) > FIRST_EMBEDDING_HEADER_LIMIT_BYTES:
            raise TasteT12ProductionStateError(
                "T12 first embedding header exceeded its cap"
            )
        path = self.root / f"embeddings-{segment_id}.bin"
        stream = path.open("xb", buffering=0)
        prefix = (
            FIRST_EMBEDDING_MAGIC
            + struct.pack(">I", len(header_bytes))
            + header_bytes
        )
        stream.write(prefix)
        os.fsync(stream.fileno())
        _fsync_directory(self.root)
        self._active_stream = stream
        self._active_path = path
        self._active_header = header
        self._active_header_bytes = header_bytes
        self._active_record_count = 0
        self._active_prefix_digest = hashlib.sha256(prefix)

    def raw_sha256_for(self, graph_identity_sha256: str) -> str | None:
        graph_hash = _require_sha256(
            graph_identity_sha256, field="T12 first embedding graph identity"
        )
        locator = self._locators.get(graph_hash)
        return None if locator is None else locator.raw_sha256

    def embedding_sha256_for(self, graph_identity_sha256: str) -> str | None:
        graph_hash = _require_sha256(
            graph_identity_sha256, field="T12 first embedding graph identity"
        )
        locator = self._locators.get(graph_hash)
        return None if locator is None else locator.embedding_sha256

    def lookup(self, graph_identity_sha256: str) -> FirstSeenEmbedding | None:
        graph_hash = _require_sha256(
            graph_identity_sha256, field="T12 first embedding graph identity"
        )
        locator = self._locators.get(graph_hash)
        if locator is None:
            return None
        with locator.path.open("rb", buffering=0) as stream:
            stream.seek(locator.raw_offset)
            raw = stream.read(locator.raw_length)
        if len(raw) != locator.raw_length or _sha256_bytes(raw) != locator.raw_sha256:
            raise TasteT12ProductionStateError(
                "T12 first embedding raw bytes changed after indexing"
            )
        return FirstSeenEmbedding(
            graph_identity_sha256=graph_hash,
            dtype=locator.dtype,
            shape=locator.shape,
            raw_bytes=raw,
            raw_sha256=locator.raw_sha256,
            embedding_sha256=locator.embedding_sha256,
            model_sha256=self.model_sha256,
            feature_schema_sha256=self.feature_schema_sha256,
        )

    def append(
        self,
        *,
        graph_identity_sha256: str,
        dtype: str,
        shape: Sequence[int],
        raw_bytes: bytes,
    ) -> FirstSeenEmbedding:
        if (
            self._active_stream is None
            or self._active_path is None
            or self._active_prefix_digest is None
        ):
            raise TasteT12ProductionStateError(
                "T12 first embedding writer is closed"
            )
        graph_hash = _require_sha256(
            graph_identity_sha256, field="T12 first embedding graph identity"
        )
        if graph_hash in self._locators:
            raise TasteT12ProductionStateError(
                "T12 first embedding was appended more than once"
            )
        (
            dtype,
            normalized_shape,
            raw,
            raw_sha,
            embedding_sha,
        ) = _validate_first_embedding_payload(
            dtype=dtype, shape=shape, raw_bytes=raw_bytes
        )
        dtype_bytes = dtype.encode("ascii")
        shape_bytes = b"".join(
            struct.pack(">Q", dimension) for dimension in normalized_shape
        )
        sequence = self.sequence + 1
        prefix = _FIRST_EMBEDDING_PREFIX.pack(
            sequence,
            bytes.fromhex(graph_hash),
            bytes.fromhex(raw_sha),
            bytes.fromhex(embedding_sha),
            bytes.fromhex(self.model_sha256),
            bytes.fromhex(self.feature_schema_sha256),
            len(dtype_bytes),
            len(normalized_shape),
            len(raw),
        )
        body = prefix + shape_bytes + dtype_bytes + raw
        if len(body) + 32 > self.bounds.max_live_record_serialized_bytes:
            raise TasteT12ProductionStateError(
                "T12 first embedding record exceeds the per-row disk bound"
            )
        projected = (
            sum(int(row["committed_bytes"]) for row in self._segments)
            + int(self._active_stream.tell())
            + len(body)
            + 32
        )
        if sequence > self.bounds.max_full_live_records:
            raise TasteT12ProductionStateError(
                "T12 first embedding count exceeded the 20k graph bound"
            )
        if projected > self.bounds.max_history_bytes:
            raise TasteT12ProductionStateError(
                "T12 first embedding store exceeded its disk cap"
            )
        chain = hashlib.sha256(bytes.fromhex(self.chain_head) + body).digest()
        record_start = int(self._active_stream.tell())
        self._active_stream.write(body + chain)
        self._active_prefix_digest.update(body + chain)
        raw_offset = (
            record_start
            + _FIRST_EMBEDDING_PREFIX.size
            + len(shape_bytes)
            + len(dtype_bytes)
        )
        self._locators[graph_hash] = _FirstEmbeddingLocator(
            path=self._active_path,
            raw_offset=raw_offset,
            raw_length=len(raw),
            dtype=dtype,
            shape=normalized_shape,
            raw_sha256=raw_sha,
            embedding_sha256=embedding_sha,
        )
        self.sequence = sequence
        self.chain_head = chain.hex()
        self._active_record_count += 1
        return FirstSeenEmbedding(
            graph_identity_sha256=graph_hash,
            dtype=dtype,
            shape=normalized_shape,
            raw_bytes=raw,
            raw_sha256=raw_sha,
            embedding_sha256=embedding_sha,
            model_sha256=self.model_sha256,
            feature_schema_sha256=self.feature_schema_sha256,
        )

    def _active_manifest(self) -> dict[str, Any] | None:
        if self._active_stream is None or self._active_path is None:
            return None
        if self._active_record_count == 0:
            return None
        self._active_stream.flush()
        os.fsync(self._active_stream.fileno())
        _fsync_directory(self.root)
        return {
            "schema_version": FIRST_EMBEDDING_SEGMENT_SCHEMA,
            "segment_file": self._active_path.name,
            "header_sha256": _sha256_bytes(self._active_header_bytes),
            "anchor_sequence": int(self._active_header["anchor_sequence"]),
            "anchor_chain_head": str(self._active_header["anchor_chain_head"]),
            "record_count": self._active_record_count,
            "terminal_sequence": self.sequence,
            "terminal_chain_head": self.chain_head,
            "committed_bytes": int(self._active_stream.tell()),
            "committed_prefix_sha256": (
                self._active_prefix_digest.copy().hexdigest()
            ),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        segments = [dict(row) for row in self._segments]
        active = self._active_manifest()
        if active is not None:
            segments.append(active)
        return {
            "schema_version": FIRST_EMBEDDING_SNAPSHOT_SCHEMA,
            "store_root": str(self.root),
            "contract_sha256": self.contract_sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "model_sha256": self.model_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "bounds_sha256": self.bounds.sha256,
            "segments": segments,
            "record_count": self.sequence,
            "chain_head": self.chain_head,
            "raw_bytes_authoritative": True,
            "append_only_hash_chain": True,
            "sqlite_index_authoritative": False,
        }

    def _read_committed_segment(
        self,
        manifest: Mapping[str, Any],
        *,
        expected_sequence: int,
        expected_head: str,
    ) -> tuple[int, str]:
        expected_keys = {
            "schema_version", "segment_file", "header_sha256",
            "anchor_sequence", "anchor_chain_head", "record_count",
            "terminal_sequence", "terminal_chain_head", "committed_bytes",
            "committed_prefix_sha256",
        }
        if type(manifest) is not dict or set(manifest) != expected_keys:
            raise TasteT12ProductionStateError(
                "T12 first embedding segment manifest keys changed"
            )
        name = manifest.get("segment_file")
        if (
            type(name) is not str
            or Path(name).name != name
            or not name.startswith("embeddings-")
        ):
            raise TasteT12ProductionStateError(
                "T12 first embedding segment name is invalid"
            )
        path = self.root / name
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise TasteT12ProductionStateError(
                "T12 first embedding segment is an alias"
            )
        committed_bytes = _require_native_int(
            manifest.get("committed_bytes"),
            field="T12 committed first embedding bytes",
            minimum=len(FIRST_EMBEDDING_MAGIC) + 4,
        )
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size < committed_bytes:
            raise TasteT12ProductionStateError(
                "T12 first embedding committed prefix is incomplete"
            )
        with path.open("rb", buffering=0) as stream:
            magic = stream.read(len(FIRST_EMBEDDING_MAGIC))
            raw_header_length = stream.read(4)
            if magic != FIRST_EMBEDDING_MAGIC or len(raw_header_length) != 4:
                raise TasteT12ProductionStateError(
                    "T12 first embedding header is invalid"
                )
            header_length = struct.unpack(">I", raw_header_length)[0]
            if not 0 < header_length <= FIRST_EMBEDDING_HEADER_LIMIT_BYTES:
                raise TasteT12ProductionStateError(
                    "T12 first embedding header length is invalid"
                )
            header_bytes = stream.read(header_length)
            try:
                header = json.loads(header_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TasteT12ProductionStateError(
                    "T12 first embedding header is unreadable"
                ) from exc
            if (
                type(header) is not dict
                or header.get("schema_version") != FIRST_EMBEDDING_SEGMENT_SCHEMA
                or header.get("contract_sha256") != self.contract_sha256
                or header.get("attempt_id") != self.attempt_id
                or header.get("generation_token") != self.generation_token
                or header.get("model_sha256") != self.model_sha256
                or header.get("feature_schema_sha256")
                != self.feature_schema_sha256
                or header.get("bounds_sha256") != self.bounds.sha256
                or header.get("anchor_sequence") != expected_sequence
                or header.get("anchor_chain_head") != expected_head
                or manifest.get("anchor_sequence") != expected_sequence
                or manifest.get("anchor_chain_head") != expected_head
                or _sha256_bytes(header_bytes) != manifest.get("header_sha256")
            ):
                raise TasteT12ProductionStateError(
                    "T12 first embedding header semantics changed"
                )
            digest = hashlib.sha256(magic + raw_header_length + header_bytes)
            sequence = expected_sequence
            chain_head = expected_head
            record_count = _require_native_int(
                manifest.get("record_count"),
                field="T12 first embedding record count",
                minimum=1,
            )
            for _ in range(record_count):
                record_start = int(stream.tell())
                prefix = stream.read(_FIRST_EMBEDDING_PREFIX.size)
                if len(prefix) != _FIRST_EMBEDDING_PREFIX.size:
                    raise TasteT12ProductionStateError(
                        "T12 first embedding record prefix is incomplete"
                    )
                (
                    observed_sequence,
                    graph_raw,
                    raw_sha,
                    embedding_sha,
                    model_raw,
                    feature_raw,
                    dtype_length,
                    ndim,
                    raw_length,
                ) = _FIRST_EMBEDDING_PREFIX.unpack(prefix)
                if (
                    observed_sequence != sequence + 1
                    or not 0 < dtype_length <= FIRST_EMBEDDING_DTYPE_LIMIT_BYTES
                    or not 1 <= ndim <= FIRST_EMBEDDING_MAX_NDIM
                    or raw_length <= 0
                    or model_raw.hex() != self.model_sha256
                    or feature_raw.hex() != self.feature_schema_sha256
                ):
                    raise TasteT12ProductionStateError(
                        "T12 first embedding record metadata changed"
                    )
                shape_bytes = stream.read(ndim * 8)
                dtype_bytes = stream.read(dtype_length)
                if len(shape_bytes) != ndim * 8 or len(dtype_bytes) != dtype_length:
                    raise TasteT12ProductionStateError(
                        "T12 first embedding dtype/shape is incomplete"
                    )
                shape = tuple(
                    struct.unpack(">Q", shape_bytes[index : index + 8])[0]
                    for index in range(0, len(shape_bytes), 8)
                )
                try:
                    dtype = dtype_bytes.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise TasteT12ProductionStateError(
                        "T12 first embedding dtype is unreadable"
                    ) from exc
                raw_offset = int(stream.tell())
                raw = stream.read(raw_length)
                observed_chain = stream.read(32)
                if len(raw) != raw_length or len(observed_chain) != 32:
                    raise TasteT12ProductionStateError(
                        "T12 first embedding record is incomplete"
                    )
                validated = _validate_first_embedding_payload(
                    dtype=dtype, shape=shape, raw_bytes=raw
                )
                if (
                    validated[3] != raw_sha.hex()
                    or validated[4] != embedding_sha.hex()
                ):
                    raise TasteT12ProductionStateError(
                        "T12 first embedding raw/semantic SHA changed"
                    )
                body = prefix + shape_bytes + dtype_bytes + raw
                expected_chain = hashlib.sha256(
                    bytes.fromhex(chain_head) + body
                ).digest()
                if observed_chain != expected_chain:
                    raise TasteT12ProductionStateError(
                        "T12 first embedding hash chain changed"
                    )
                record = body + observed_chain
                digest.update(record)
                graph_hash = graph_raw.hex()
                if graph_hash in self._locators:
                    raise TasteT12ProductionStateError(
                        "T12 first embedding graph was serialized twice"
                    )
                self._locators[graph_hash] = _FirstEmbeddingLocator(
                    path=path,
                    raw_offset=raw_offset,
                    raw_length=raw_length,
                    dtype=dtype,
                    shape=shape,
                    raw_sha256=raw_sha.hex(),
                    embedding_sha256=embedding_sha.hex(),
                )
                sequence = int(observed_sequence)
                chain_head = observed_chain.hex()
                if int(stream.tell()) <= record_start:
                    raise AssertionError("T12 first embedding reader did not advance")
            if int(stream.tell()) != committed_bytes:
                raise TasteT12ProductionStateError(
                    "T12 first embedding committed byte boundary changed"
                )
            if (
                digest.hexdigest()
                != _require_sha256(
                    manifest.get("committed_prefix_sha256"),
                    field="T12 first embedding prefix SHA",
                )
                or manifest.get("terminal_sequence") != sequence
                or manifest.get("terminal_chain_head") != chain_head
            ):
                raise TasteT12ProductionStateError(
                    "T12 first embedding committed prefix digest changed"
                )
        return sequence, chain_head

    def _restore_snapshot(self, value: Mapping[str, Any]) -> None:
        expected = {
            "schema_version", "store_root", "contract_sha256", "attempt_id",
            "generation_token", "model_sha256", "feature_schema_sha256",
            "bounds_sha256", "segments", "record_count", "chain_head",
            "raw_bytes_authoritative", "append_only_hash_chain",
            "sqlite_index_authoritative",
        }
        if type(value) is not dict or set(value) != expected:
            raise TasteT12ProductionStateError(
                "T12 first embedding snapshot keys changed"
            )
        if (
            value.get("schema_version") != FIRST_EMBEDDING_SNAPSHOT_SCHEMA
            or value.get("store_root") != str(self.root)
            or value.get("contract_sha256") != self.contract_sha256
            or value.get("attempt_id") != self.attempt_id
            or value.get("generation_token") != self.generation_token
            or value.get("model_sha256") != self.model_sha256
            or value.get("feature_schema_sha256") != self.feature_schema_sha256
            or value.get("bounds_sha256") != self.bounds.sha256
            or value.get("raw_bytes_authoritative") is not True
            or value.get("append_only_hash_chain") is not True
            or value.get("sqlite_index_authoritative") is not False
        ):
            raise TasteT12ProductionStateError(
                "T12 first embedding snapshot semantics changed"
            )
        segments = value.get("segments")
        if (
            type(segments) is not list
            or len(segments) > HISTORY_MAX_SEGMENTS
            or (not segments and value.get("record_count") != 0)
        ):
            raise TasteT12ProductionStateError(
                "T12 first embedding segment count is invalid"
            )
        sequence = 0
        chain_head = "0" * 64
        restored: list[dict[str, Any]] = []
        for manifest in segments:
            sequence, chain_head = self._read_committed_segment(
                manifest, expected_sequence=sequence, expected_head=chain_head
            )
            restored.append(dict(manifest))
        if (
            value.get("record_count") != sequence
            or value.get("chain_head") != chain_head
            or len(self._locators) != sequence
        ):
            raise TasteT12ProductionStateError(
                "T12 first embedding snapshot counters changed"
            )
        self.sequence = sequence
        self.chain_head = chain_head
        self._segments = restored

    def close(self) -> None:
        if self._active_stream is not None:
            self._active_stream.close()
            self._active_stream = None


class T12CompactHistoryJournal:
    """Fixed-record append-only history with a disposable on-disk index."""

    def __init__(
        self,
        *,
        root: str | Path,
        index_root: str | Path,
        bounds: T12ProductionBounds,
        contract_sha256: str,
        attempt_id: str,
        generation_token: str,
        resume_snapshot: Mapping[str, Any] | None = None,
        open_writer: bool = True,
    ) -> None:
        self.root = _normalized_absolute(root, field="T12 history root")
        self.index_root = _normalized_absolute(
            index_root, field="T12 history index root"
        )
        self.bounds = bounds
        self.contract_sha256 = _require_sha256(
            contract_sha256, field="T12 history contract"
        )
        try:
            parsed_attempt = uuid.UUID(attempt_id)
        except (ValueError, AttributeError) as exc:
            raise TasteT12ProductionStateError(
                "T12 history attempt ID is invalid"
            ) from exc
        if parsed_attempt.version != 4 or str(parsed_attempt) != attempt_id:
            raise TasteT12ProductionStateError(
                "T12 history attempt ID must be canonical UUIDv4"
            )
        self.attempt_id = attempt_id
        self.generation_token = _require_sha256(
            generation_token, field="T12 history generation token"
        )
        if resume_snapshot is None:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        else:
            if (
                not self.root.is_dir()
                or self.root.is_symlink()
                or self.root.resolve(strict=True) != self.root
            ):
                raise TasteT12ProductionStateError(
                    "T12 resume history root is not one physical directory"
                )
        self.index_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.index_root.is_symlink() or self.index_root.resolve(strict=True) != self.index_root:
            raise TasteT12ProductionStateError("T12 history index root is an alias")
        self._index_path = self.index_root / f"t12-history-{uuid.uuid4()}.sqlite3"
        # The index is explicitly non-authoritative and is rebuilt from the
        # authenticated history segments on resume.  A read-only observer can
        # nevertheless take a SQLite SHARED lock, so use a bounded busy wait
        # instead of aborting an otherwise valid production walk immediately.
        # This does not change any scientific state or history semantics.
        self._connection = sqlite3.connect(
            self._index_path,
            timeout=HISTORY_INDEX_BUSY_TIMEOUT_MILLISECONDS / 1_000,
        )
        self._connection.execute(
            f"PRAGMA busy_timeout={HISTORY_INDEX_BUSY_TIMEOUT_MILLISECONDS}"
        )
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute(f"PRAGMA cache_size=-{HISTORY_INDEX_CACHE_KIB}")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.executescript(
            """
            CREATE TABLE first_observation (
                graph_hash TEXT PRIMARY KEY,
                p0 REAL NOT NULL,
                p1 REAL NOT NULL,
                p2 REAL NOT NULL,
                prediction INTEGER NOT NULL,
                candidate INTEGER NOT NULL,
                valid_fullgraph INTEGER NOT NULL,
                covered_parent_count INTEGER NOT NULL,
                embedding_sha256 TEXT NOT NULL,
                coverage_sha256 TEXT NOT NULL,
                failure_sha256 TEXT NOT NULL,
                neurosed_query_sha256 TEXT NOT NULL
            );
            CREATE TABLE lineage_identity (
                graph_hash TEXT NOT NULL,
                lineage_sha256 TEXT NOT NULL,
                PRIMARY KEY (graph_hash, lineage_sha256)
            );
            """
        )
        self._connection.execute("BEGIN")
        self._pending_index_writes = 0
        self._segments: list[dict[str, Any]] = []
        self.sequence = 0
        self.chain_head = "0" * 64
        self.first_seen_graph_count = 0
        self.first_seen_lineage_count = 0
        self.candidate_first_seen_count = 0
        self.destination_first_seen_counts = {0: 0, 2: 0}
        self._active_stream: Any | None = None
        self._active_path: Path | None = None
        self._active_header: dict[str, Any] | None = None
        self._active_header_bytes = b""
        self._active_record_count = 0
        self._active_prefix_digest: Any | None = None
        self._first_embedding_store: T12FirstSeenEmbeddingStore | None = None
        self._first_embedding_store_required = False
        self._open_writer_requested = bool(open_writer)
        if resume_snapshot is not None:
            self._restore_snapshot(resume_snapshot)
        if open_writer:
            if self._first_embedding_store is not None:
                self._first_embedding_store.start_writer()
            self._start_segment()

    @property
    def observation_count(self) -> int:
        return self.sequence

    @property
    def first_seen_strict_counterfactual_count(self) -> int:
        return self.candidate_first_seen_count

    @property
    def first_seen_embedding_store(self) -> T12FirstSeenEmbeddingStore | None:
        return self._first_embedding_store

    def bind_first_seen_embedding_authority(
        self, *, model_sha256: str, feature_schema_sha256: str
    ) -> None:
        """Bind this production journal to one frozen GINE byte authority."""

        model_hash = _require_sha256(
            model_sha256, field="T12 first embedding model"
        )
        feature_hash = _require_sha256(
            feature_schema_sha256, field="T12 first embedding feature schema"
        )
        if self._first_embedding_store is None:
            self._first_embedding_store = T12FirstSeenEmbeddingStore(
                root=(self.root / "first-seen-embeddings").resolve(),
                bounds=self.bounds,
                contract_sha256=self.contract_sha256,
                attempt_id=self.attempt_id,
                generation_token=self.generation_token,
                model_sha256=model_hash,
                feature_schema_sha256=feature_hash,
                open_writer=self._open_writer_requested,
            )
        elif (
            self._first_embedding_store.model_sha256 != model_hash
            or self._first_embedding_store.feature_schema_sha256 != feature_hash
        ):
            raise TasteT12ProductionStateError(
                "T12 first embedding model/feature authority changed"
            )
        self._first_embedding_store_required = True
        self._validate_first_embedding_alignment()

    def lookup_first_embedding(
        self, graph_identity_sha256: str
    ) -> FirstSeenEmbedding | None:
        if not self._first_embedding_store_required or self._first_embedding_store is None:
            raise TasteT12ProductionStateError(
                "T12 first embedding authority is not bound"
            )
        return self._first_embedding_store.lookup(graph_identity_sha256)

    def append_first_embedding(
        self,
        *,
        graph_identity_sha256: str,
        dtype: str,
        shape: Sequence[int],
        raw_bytes: bytes,
    ) -> FirstSeenEmbedding:
        if not self._first_embedding_store_required or self._first_embedding_store is None:
            raise TasteT12ProductionStateError(
                "T12 first embedding authority is not bound"
            )
        graph_hash = _require_sha256(
            graph_identity_sha256, field="T12 first embedding graph identity"
        )
        if self._first_observation(graph_hash) is not None:
            raise TasteT12ProductionStateError(
                "T12 compact first observation exists without first embedding bytes"
            )
        return self._first_embedding_store.append(
            graph_identity_sha256=graph_hash,
            dtype=dtype,
            shape=shape,
            raw_bytes=raw_bytes,
        )

    def _validate_first_embedding_alignment(self) -> None:
        store = self._first_embedding_store
        if store is None:
            if self._first_embedding_store_required:
                raise TasteT12ProductionStateError(
                    "T12 required first embedding store is absent"
                )
            return
        rows = self._connection.execute(
            "SELECT graph_hash,embedding_sha256 FROM first_observation"
        ).fetchall()
        indexed = {str(graph_hash): str(raw_sha) for graph_hash, raw_sha in rows}
        if set(indexed) != set(store.graph_hashes) or any(
            store.embedding_sha256_for(graph_hash) != raw_sha
            for graph_hash, raw_sha in indexed.items()
        ):
            raise TasteT12ProductionStateError(
                "T12 compact history and first embedding authority differ"
            )

    def _commit_index(self, *, begin_next: bool = True) -> None:
        deadline = (
            time.monotonic() + HISTORY_INDEX_BUSY_TIMEOUT_MILLISECONDS / 1_000
        )
        delay = HISTORY_INDEX_RETRY_INITIAL_SECONDS
        while True:
            try:
                self._connection.commit()
                break
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_lock_error(exc):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TasteT12ProductionStateError(
                        "T12 disposable history index remained locked past the "
                        "bounded busy timeout"
                    ) from exc
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, HISTORY_INDEX_RETRY_MAX_SECONDS)
        if begin_next:
            self._connection.execute("BEGIN")
        self._pending_index_writes = 0

    def _start_segment(self) -> None:
        if self._active_stream is not None:
            raise TasteT12ProductionStateError("T12 history segment is already open")
        if len(self._segments) >= HISTORY_MAX_SEGMENTS:
            raise TasteT12ProductionStateError(
                "T12 history exceeded the two 10k/20k process segments"
            )
        segment_id = str(uuid.uuid4())
        header = {
            "schema_version": HISTORY_SEGMENT_SCHEMA,
            "segment_id": segment_id,
            "contract_sha256": self.contract_sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "anchor_sequence": self.sequence,
            "anchor_chain_head": self.chain_head,
            "record_bytes": HISTORY_RECORD_BYTES,
            "bounds_sha256": self.bounds.sha256,
        }
        header_bytes = _canonical_bytes(header)
        if len(header_bytes) > HISTORY_HEADER_LIMIT_BYTES:
            raise TasteT12ProductionStateError("T12 history header exceeded its cap")
        path = self.root / f"history-{segment_id}.bin"
        stream = path.open("xb", buffering=0)
        prefix = HISTORY_MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes
        stream.write(prefix)
        os.fsync(stream.fileno())
        _fsync_directory(self.root)
        self._active_stream = stream
        self._active_path = path
        self._active_header = header
        self._active_header_bytes = header_bytes
        self._active_record_count = 0
        self._active_prefix_digest = hashlib.sha256(prefix)

    def _first_observation(self, graph_hash: str) -> CompactFirstObservation | None:
        row = self._connection.execute(
            """
            SELECT p0,p1,p2,prediction,candidate,valid_fullgraph,
                   covered_parent_count,embedding_sha256,coverage_sha256,
                   failure_sha256,neurosed_query_sha256
            FROM first_observation WHERE graph_hash=?
            """,
            (graph_hash,),
        ).fetchone()
        if row is None:
            return None
        return CompactFirstObservation(
            probabilities=(float(row[0]), float(row[1]), float(row[2])),
            prediction=int(row[3]),
            candidate=bool(row[4]),
            valid_fullgraph=bool(row[5]),
            covered_parent_count=int(row[6]),
            embedding_sha256=str(row[7]),
            coverage_sha256=str(row[8]),
            failure_sha256=str(row[9]),
            neurosed_query_sha256=str(row[10]),
        )

    def lookup_first(self, graph_identity_sha256: str) -> CompactFirstObservation | None:
        graph_hash = _require_sha256(
            graph_identity_sha256, field="T12 history graph identity"
        )
        return self._first_observation(graph_hash)

    def append_observation(
        self,
        *,
        graph_identity_sha256: str,
        probabilities: Sequence[float],
        prediction: int,
        candidate: bool,
        valid_fullgraph: bool,
        coverage_vector: Sequence[int],
        embedding_sha256: str,
        failure_reason: str,
        lineage_sha256: str,
        neurosed_query_sha256: str,
    ) -> CompactFirstObservation:
        if self._active_stream is None or self._active_prefix_digest is None:
            raise TasteT12ProductionStateError("T12 history writer is closed")
        graph_hash = _require_sha256(
            graph_identity_sha256, field="T12 history graph identity"
        )
        embedding_hash = _require_sha256(
            embedding_sha256, field="T12 history embedding digest"
        )
        lineage_hash = _require_sha256(
            lineage_sha256, field="T12 history lineage digest"
        )
        query_hash = _require_sha256(
            neurosed_query_sha256, field="T12 history NeuroSED query digest"
        )
        if self._first_embedding_store_required:
            if self._first_embedding_store is None or (
                self._first_embedding_store.embedding_sha256_for(graph_hash)
                != embedding_hash
            ):
                raise TasteT12ProductionStateError(
                    "T12 observation is not bound to its first embedding bytes"
                )
        if (
            type(probabilities) not in (list, tuple)
            or len(probabilities) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in probabilities
            )
        ):
            raise TasteT12ProductionStateError(
                "T12 history probabilities must be three finite values"
            )
        probability_row = tuple(float(value) for value in probabilities)
        if type(prediction) is not int or prediction not in (0, 1, 2):
            raise TasteT12ProductionStateError("T12 history prediction is invalid")
        if type(candidate) is not bool or type(valid_fullgraph) is not bool:
            raise TasteT12ProductionStateError("T12 history flags are invalid")
        if type(failure_reason) is not str:
            raise TasteT12ProductionStateError("T12 history failure reason is invalid")
        values = tuple(coverage_vector)
        if (
            len(values) != self.bounds.parent_count
            or any(type(value) is not int or value not in (0, 1) for value in values)
        ):
            raise TasteT12ProductionStateError("T12 history coverage is invalid")
        covered = sum(values)
        coverage_hash = _sha256_bytes(bytes(values))
        failure_hash = _sha256_bytes(failure_reason.encode("utf-8"))
        semantic_hash = compact_semantic_sha256(
            graph_identity_sha256=graph_hash,
            probabilities=probability_row,
            prediction=prediction,
            candidate=candidate,
            valid_fullgraph=valid_fullgraph,
            covered_parent_count=covered,
            embedding_sha256=embedding_hash,
            coverage_sha256=coverage_hash,
            failure_sha256=failure_hash,
        )
        sequence = self.sequence + 1
        flags = int(candidate) | (int(valid_fullgraph) << 1)
        body = _OBSERVATION_BODY.pack(
            sequence,
            bytes.fromhex(graph_hash),
            bytes.fromhex(semantic_hash),
            bytes.fromhex(embedding_hash),
            bytes.fromhex(coverage_hash),
            bytes.fromhex(failure_hash),
            bytes.fromhex(query_hash),
            *probability_row,
            prediction,
            flags,
            covered,
            bytes.fromhex(lineage_hash),
        )
        chain = hashlib.sha256(bytes.fromhex(self.chain_head) + body).digest()
        record = body + chain
        if len(record) != HISTORY_RECORD_BYTES:
            raise AssertionError("T12 compact history record size drifted")
        projected = (
            sum(int(row["committed_bytes"]) for row in self._segments)
            + int(self._active_stream.tell())
            + len(record)
        )
        if sequence > self.bounds.max_scored_observations:
            raise TasteT12ProductionStateError(
                "T12 compact history exceeded the 20k observation bound"
            )
        if projected > self.bounds.max_history_bytes:
            raise TasteT12ProductionStateError(
                "T12 compact history exceeded its disk cap"
            )
        self._active_stream.write(record)
        self._active_prefix_digest.update(record)
        self._active_record_count += 1
        self.sequence = sequence
        self.chain_head = chain.hex()

        inserted = self._connection.execute(
            """
            INSERT OR IGNORE INTO first_observation
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                graph_hash,
                *probability_row,
                prediction,
                int(candidate),
                int(valid_fullgraph),
                covered,
                embedding_hash,
                coverage_hash,
                failure_hash,
                query_hash,
            ),
        ).rowcount
        if inserted:
            self.first_seen_graph_count += 1
            if candidate:
                self.candidate_first_seen_count += 1
                if prediction in self.destination_first_seen_counts:
                    self.destination_first_seen_counts[prediction] += 1
        lineage_inserted = self._connection.execute(
            "INSERT OR IGNORE INTO lineage_identity VALUES (?,?)",
            (graph_hash, lineage_hash),
        ).rowcount
        if lineage_inserted:
            self.first_seen_lineage_count += 1
        self._pending_index_writes += 2
        if self._pending_index_writes >= 2048:
            self._commit_index()
        result = self._first_observation(graph_hash)
        if result is None:
            raise TasteT12ProductionStateError(
                "T12 history index lost its first observation"
            )
        return result

    def _active_manifest(self) -> dict[str, Any] | None:
        if self._active_stream is None or self._active_path is None:
            return None
        if self._active_record_count == 0:
            return None
        self._active_stream.flush()
        os.fsync(self._active_stream.fileno())
        _fsync_directory(self.root)
        committed_bytes = int(self._active_stream.tell())
        return {
            "schema_version": HISTORY_SEGMENT_SCHEMA,
            "segment_file": self._active_path.name,
            "header_sha256": _sha256_bytes(self._active_header_bytes),
            "anchor_sequence": int(self._active_header["anchor_sequence"]),
            "anchor_chain_head": str(self._active_header["anchor_chain_head"]),
            "record_count": self._active_record_count,
            "terminal_sequence": self.sequence,
            "terminal_chain_head": self.chain_head,
            "committed_bytes": committed_bytes,
            "committed_prefix_sha256": self._active_prefix_digest.copy().hexdigest(),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        self._commit_index()
        if self._first_embedding_store_required:
            self._validate_first_embedding_alignment()
        segments = [dict(row) for row in self._segments]
        active = self._active_manifest()
        if active is not None:
            segments.append(active)
        embedding_snapshot = (
            None
            if self._first_embedding_store is None
            else self._first_embedding_store.checkpoint_state()
        )
        return {
            "schema_version": HISTORY_SNAPSHOT_SCHEMA,
            "history_root": str(self.root),
            "contract_sha256": self.contract_sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "bounds": self.bounds.to_dict(),
            "bounds_sha256": self.bounds.sha256,
            "segments": segments,
            "observation_count": self.sequence,
            "chain_head": self.chain_head,
            "first_seen_graph_count": self.first_seen_graph_count,
            "first_seen_lineage_count": self.first_seen_lineage_count,
            "candidate_first_seen_count": self.candidate_first_seen_count,
            "destination_first_seen_counts": {
                str(key): value
                for key, value in sorted(self.destination_first_seen_counts.items())
            },
            "historical_payload_retained": False,
            "historical_embedding_values_retained": (
                embedding_snapshot is not None
            ),
            "first_seen_embedding_store": embedding_snapshot,
            "historical_lineage_payload_retained": False,
            "historical_neurosed_query_sha256_retained": True,
            "sqlite_index_authoritative": False,
            "append_only_hash_chain": True,
        }

    def _read_committed_segment(
        self, manifest: Mapping[str, Any], *, expected_sequence: int, expected_head: str
    ) -> tuple[int, str]:
        expected_keys = {
            "schema_version",
            "segment_file",
            "header_sha256",
            "anchor_sequence",
            "anchor_chain_head",
            "record_count",
            "terminal_sequence",
            "terminal_chain_head",
            "committed_bytes",
            "committed_prefix_sha256",
        }
        if type(manifest) is not dict or set(manifest) != expected_keys:
            raise TasteT12ProductionStateError(
                "T12 history segment manifest keys changed"
            )
        name = manifest.get("segment_file")
        if type(name) is not str or Path(name).name != name or not name.startswith("history-"):
            raise TasteT12ProductionStateError("T12 history segment name is invalid")
        path = self.root / name
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise TasteT12ProductionStateError("T12 history segment is an alias")
        info = path.stat()
        committed_bytes = _require_native_int(
            manifest.get("committed_bytes"),
            field="T12 committed history bytes",
            minimum=len(HISTORY_MAGIC) + 4,
        )
        if not stat.S_ISREG(info.st_mode) or info.st_size < committed_bytes:
            raise TasteT12ProductionStateError("T12 history prefix is incomplete")
        with path.open("rb", buffering=0) as stream:
            magic = stream.read(len(HISTORY_MAGIC))
            raw_length = stream.read(4)
            if magic != HISTORY_MAGIC or len(raw_length) != 4:
                raise TasteT12ProductionStateError("T12 history header is invalid")
            header_length = struct.unpack(">I", raw_length)[0]
            if not 0 < header_length <= HISTORY_HEADER_LIMIT_BYTES:
                raise TasteT12ProductionStateError("T12 history header length is invalid")
            header_bytes = stream.read(header_length)
            try:
                header = json.loads(header_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TasteT12ProductionStateError(
                    "T12 history header is unreadable"
                ) from exc
            if (
                type(header) is not dict
                or header.get("schema_version") != HISTORY_SEGMENT_SCHEMA
                or header.get("contract_sha256") != self.contract_sha256
                or header.get("attempt_id") != self.attempt_id
                or header.get("generation_token") != self.generation_token
                or header.get("bounds_sha256") != self.bounds.sha256
                or header.get("record_bytes") != HISTORY_RECORD_BYTES
                or header.get("anchor_sequence") != expected_sequence
                or header.get("anchor_chain_head") != expected_head
                or manifest.get("anchor_sequence") != expected_sequence
                or manifest.get("anchor_chain_head") != expected_head
                or _sha256_bytes(header_bytes) != manifest.get("header_sha256")
            ):
                raise TasteT12ProductionStateError("T12 history header semantics changed")
            prefix_digest = hashlib.sha256(magic + raw_length + header_bytes)
            sequence = expected_sequence
            chain_head = expected_head
            record_count = _require_native_int(
                manifest.get("record_count"),
                field="T12 history record count",
                minimum=1,
            )
            for _ in range(record_count):
                record = stream.read(HISTORY_RECORD_BYTES)
                if len(record) != HISTORY_RECORD_BYTES:
                    raise TasteT12ProductionStateError(
                        "T12 history committed record is incomplete"
                    )
                prefix_digest.update(record)
                body, observed_chain = record[:-32], record[-32:]
                unpacked = _OBSERVATION_BODY.unpack(body)
                observed_sequence = int(unpacked[0])
                if observed_sequence != sequence + 1:
                    raise TasteT12ProductionStateError("T12 history sequence has a gap")
                expected_chain = hashlib.sha256(bytes.fromhex(chain_head) + body).digest()
                if observed_chain != expected_chain:
                    raise TasteT12ProductionStateError("T12 history hash chain changed")
                (
                    _, graph_raw, semantic_raw, embedding_raw, coverage_raw,
                    failure_raw, query_raw, p0, p1, p2, prediction, flags, covered,
                    lineage_raw,
                ) = unpacked
                graph_hash = graph_raw.hex()
                candidate = bool(flags & 1)
                valid_fullgraph = bool(flags & 2)
                if flags & ~3 or prediction not in (0, 1, 2) or covered > self.bounds.parent_count:
                    raise TasteT12ProductionStateError(
                        "T12 history observation flags changed"
                    )
                semantic = compact_semantic_sha256(
                    graph_identity_sha256=graph_hash,
                    probabilities=(p0, p1, p2),
                    prediction=prediction,
                    candidate=candidate,
                    valid_fullgraph=valid_fullgraph,
                    covered_parent_count=covered,
                    embedding_sha256=embedding_raw.hex(),
                    coverage_sha256=coverage_raw.hex(),
                    failure_sha256=failure_raw.hex(),
                )
                if semantic_raw.hex() != semantic:
                    raise TasteT12ProductionStateError(
                        "T12 history compact semantics changed"
                    )
                inserted = self._connection.execute(
                    "INSERT OR IGNORE INTO first_observation VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        graph_hash, p0, p1, p2, prediction, int(candidate),
                        int(valid_fullgraph), covered, embedding_raw.hex(),
                        coverage_raw.hex(), failure_raw.hex(),
                        query_raw.hex(),
                    ),
                ).rowcount
                if inserted:
                    self.first_seen_graph_count += 1
                    if candidate:
                        self.candidate_first_seen_count += 1
                        if prediction in self.destination_first_seen_counts:
                            self.destination_first_seen_counts[prediction] += 1
                lineage_inserted = self._connection.execute(
                    "INSERT OR IGNORE INTO lineage_identity VALUES (?,?)",
                    (graph_hash, lineage_raw.hex()),
                ).rowcount
                if lineage_inserted:
                    self.first_seen_lineage_count += 1
                sequence = observed_sequence
                chain_head = observed_chain.hex()
            if stream.tell() != committed_bytes:
                raise TasteT12ProductionStateError(
                    "T12 history committed byte boundary changed"
                )
            if (
                prefix_digest.hexdigest()
                != _require_sha256(
                    manifest.get("committed_prefix_sha256"),
                    field="T12 history prefix SHA",
                )
                or manifest.get("terminal_sequence") != sequence
                or manifest.get("terminal_chain_head") != chain_head
            ):
                raise TasteT12ProductionStateError("T12 history prefix digest changed")
        return sequence, chain_head

    def _restore_snapshot(self, value: Mapping[str, Any]) -> None:
        expected = {
            "schema_version", "history_root", "contract_sha256", "attempt_id",
            "generation_token", "bounds", "bounds_sha256", "segments",
            "observation_count", "chain_head", "first_seen_graph_count",
            "first_seen_lineage_count", "candidate_first_seen_count",
            "destination_first_seen_counts", "historical_payload_retained",
            "historical_embedding_values_retained",
            "first_seen_embedding_store",
            "historical_lineage_payload_retained",
            "historical_neurosed_query_sha256_retained",
            "sqlite_index_authoritative",
            "append_only_hash_chain",
        }
        if type(value) is not dict or set(value) != expected:
            raise TasteT12ProductionStateError("T12 history snapshot keys changed")
        embedding_snapshot = value.get("first_seen_embedding_store")
        if embedding_snapshot is not None and type(embedding_snapshot) is not dict:
            raise TasteT12ProductionStateError(
                "T12 first embedding snapshot is invalid"
            )
        if (
            value.get("schema_version") != HISTORY_SNAPSHOT_SCHEMA
            or value.get("history_root") != str(self.root)
            or value.get("contract_sha256") != self.contract_sha256
            or value.get("attempt_id") != self.attempt_id
            or value.get("generation_token") != self.generation_token
            or value.get("bounds_sha256") != self.bounds.sha256
            or T12ProductionBounds.from_dict(value.get("bounds")) != self.bounds
            or value.get("historical_payload_retained") is not False
            or value.get("historical_embedding_values_retained")
            is not (embedding_snapshot is not None)
            or value.get("historical_lineage_payload_retained") is not False
            or value.get("historical_neurosed_query_sha256_retained") is not True
            or value.get("sqlite_index_authoritative") is not False
            or value.get("append_only_hash_chain") is not True
        ):
            raise TasteT12ProductionStateError("T12 history snapshot semantics changed")
        segments = value.get("segments")
        if type(segments) is not list or not 1 <= len(segments) <= HISTORY_MAX_SEGMENTS:
            raise TasteT12ProductionStateError("T12 history segment count is invalid")
        sequence = 0
        chain_head = "0" * 64
        restored: list[dict[str, Any]] = []
        for manifest in segments:
            sequence, chain_head = self._read_committed_segment(
                manifest, expected_sequence=sequence, expected_head=chain_head
            )
            restored.append(dict(manifest))
        self._commit_index()
        expected_destinations = {
            str(key): count
            for key, count in sorted(self.destination_first_seen_counts.items())
        }
        if (
            value.get("observation_count") != sequence
            or value.get("chain_head") != chain_head
            or value.get("first_seen_graph_count") != self.first_seen_graph_count
            or value.get("first_seen_lineage_count") != self.first_seen_lineage_count
            or value.get("candidate_first_seen_count")
            != self.candidate_first_seen_count
            or value.get("destination_first_seen_counts") != expected_destinations
        ):
            raise TasteT12ProductionStateError("T12 history counters changed")
        self.sequence = sequence
        self.chain_head = chain_head
        self._segments = restored
        if embedding_snapshot is not None:
            self._first_embedding_store = T12FirstSeenEmbeddingStore(
                root=(self.root / "first-seen-embeddings").resolve(),
                bounds=self.bounds,
                contract_sha256=self.contract_sha256,
                attempt_id=self.attempt_id,
                generation_token=self.generation_token,
                model_sha256=embedding_snapshot.get("model_sha256"),
                feature_schema_sha256=embedding_snapshot.get(
                    "feature_schema_sha256"
                ),
                resume_snapshot=embedding_snapshot,
                open_writer=False,
            )
            self._first_embedding_store_required = True
            self._validate_first_embedding_alignment()

    def close(self, *, commit_index: bool = True) -> None:
        if self._active_stream is not None:
            self._active_stream.close()
            self._active_stream = None
        if self._first_embedding_store is not None:
            self._first_embedding_store.close()
        try:
            if commit_index:
                self._commit_index(begin_next=False)
            else:
                # SQLite is a disposable index.  On a failed science body the
                # authenticated append-only journal remains the evidence, and
                # rolling back here prevents close() from masking that primary
                # exception with a second lock error.
                self._connection.rollback()
        finally:
            self._connection.close()


__all__ = [
    "BOUNDS_SCHEMA",
    "CompactFirstObservation",
    "FIRST_EMBEDDING_SNAPSHOT_SCHEMA",
    "FirstSeenEmbedding",
    "HISTORY_RECORD_BYTES",
    "HISTORY_SCHEMA",
    "HISTORY_SNAPSHOT_SCHEMA",
    "PINNED_CANDIDATE_CAPACITY",
    "PINNED_CHECKPOINT_CURSORS",
    "PINNED_SAMPLE_SIZE",
    "PINNED_TOTAL_STEPS",
    "T12CompactHistoryJournal",
    "T12FirstSeenEmbeddingStore",
    "T12ProductionBounds",
    "TasteT12ProductionStateError",
    "compact_semantic_sha256",
]
