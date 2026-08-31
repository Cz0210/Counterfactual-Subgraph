"""External, exact transition storage for the TasteMolNet T12 VRRW walk.

The pinned official GCFExplainer implementation stores one four-part tuple per
visited graph.  Its final element is a ``(neighbours, parents)`` binary
coverage tensor.  Keeping every such tensor in the Python transition mapping
can require terabytes at the authorized 20k/10k route.  This module changes
only the lifetime and representation of that already-computed state:

* hashes, actions, importance rows, and binary coverage are appended to an
  authenticated dataset-specific journal;
* coverage is bit-packed on disk and reconstructed exactly as the same Torch
  dtype when official code reads a transition;
* one (by default) expanded official tuple is retained in a deterministic
  LRU; and
* checkpoints export a small immutable prefix closure.  Reopen scans and
  rehashes that prefix before rebuilding the derived in-memory source index.

No graph/model call, edit enumeration, random draw, ordering decision, or
scientific parameter is added or repeated.  A non-binary coverage tensor is a
hard error rather than a lossy conversion.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any

import numpy as np


TRANSITION_STORE_SCHEMA = "tastemolnet_t12_external_transition_store_v1"
TRANSITION_SNAPSHOT_SCHEMA = (
    "tastemolnet_t12_external_transition_checkpoint_v1"
)
TRANSITION_SEGMENT_SCHEMA = "tastemolnet_t12_external_transition_segment_v1"
TRANSITION_STORE_POLICY = "exact_bitpacked_coverage_bounded_expanded_lru_v1"
TRANSITION_MAGIC = b"T12TRN1\n"
TRANSITION_MAX_SEGMENTS = 2
TRANSITION_DEFAULT_EXPANDED_CAPACITY = 1
TRANSITION_DEFAULT_MAX_STORE_BYTES = 128 * 1024**3

_HEADER_LENGTH = struct.Struct(">I")
_EVENT_PREFIX = struct.Struct(">IQ")
_CHAIN_BYTES = 32
_MAX_EVENT_HEADER_BYTES = 4 * 1024**2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TasteT12TransitionStoreError(RuntimeError):
    """The external transition journal or its exact replay contract failed."""


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
        raise TasteT12TransitionStoreError(f"{field} must be lowercase SHA-256")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TasteT12TransitionStoreError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _normalized_absolute(path: str | Path, *, field: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or Path(os.path.abspath(value)) != value:
        raise TasteT12TransitionStoreError(
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


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in {str, int, float, bool}:
        if isinstance(value, float) and not math.isfinite(value):
            raise TasteT12TransitionStoreError(
                "T12 transition action contains a non-finite scalar"
            )
        return value
    if isinstance(value, (list, tuple)):
        return [_freeze_json_value(item) for item in value]
    raise TasteT12TransitionStoreError(
        f"T12 transition action contains unsupported {type(value).__name__}"
    )


def _freeze_action(action: Any) -> list[Any]:
    if not isinstance(action, (list, tuple)) or not action:
        raise TasteT12TransitionStoreError("T12 transition action is malformed")
    frozen = _freeze_json_value(action)
    if type(frozen[0]) is not str or not frozen[0]:
        raise TasteT12TransitionStoreError("T12 transition action name is invalid")
    return frozen


def _dtype_string(value: Any) -> str:
    try:
        dtype = np.dtype(str(value).removeprefix("torch."))
    except TypeError as exc:
        raise TasteT12TransitionStoreError(
            f"unsupported T12 transition coverage dtype: {value}"
        ) from exc
    if dtype.kind not in "bifu":
        raise TasteT12TransitionStoreError(
            "T12 transition coverage dtype must be numeric"
        )
    return dtype.str


@dataclass(frozen=True, slots=True)
class _RecordLocation:
    segment_file: str
    offset: int
    event_bytes: int
    event_sequence: int


class T12ExternalTransitionStore(MutableMapping[str, Any]):
    """Drop-in official transition mapping backed by an authenticated journal."""

    T12_BOUNDED_TRANSITION_STATE = True

    def __init__(
        self,
        *,
        root: str | Path,
        parent_count: int,
        sample_size: int,
        candidate_capacity: int,
        contract_sha256: str,
        attempt_id: str,
        generation_token: str,
        expanded_capacity: int = TRANSITION_DEFAULT_EXPANDED_CAPACITY,
        max_store_bytes: int = TRANSITION_DEFAULT_MAX_STORE_BYTES,
        resume_snapshot: Mapping[str, Any] | None = None,
        open_writer: bool = True,
    ) -> None:
        self.root = _normalized_absolute(root, field="T12 transition store root")
        self.parent_count = _require_int(
            parent_count, field="T12 transition parent count", minimum=1
        )
        self.sample_size = _require_int(
            sample_size, field="T12 transition sample size", minimum=1
        )
        self.candidate_capacity = _require_int(
            candidate_capacity,
            field="T12 transition candidate capacity",
            minimum=1,
        )
        self.contract_sha256 = _require_sha256(
            contract_sha256, field="T12 transition contract"
        )
        self.attempt_id = str(attempt_id)
        if not self.attempt_id:
            raise TasteT12TransitionStoreError(
                "T12 transition attempt identity is empty"
            )
        self.generation_token = _require_sha256(
            generation_token, field="T12 transition generation token"
        )
        self.expanded_capacity = _require_int(
            expanded_capacity,
            field="T12 expanded transition capacity",
            minimum=1,
        )
        if self.expanded_capacity > 2:
            raise TasteT12TransitionStoreError(
                "T12 expanded transition capacity may not exceed two"
            )
        self.max_store_bytes = _require_int(
            max_store_bytes, field="T12 transition store cap", minimum=1
        )
        self._open_writer = bool(open_writer)
        self._index: dict[str, _RecordLocation] = {}
        self._expanded: OrderedDict[str, Any] = OrderedDict()
        self._segments: list[dict[str, Any]] = []
        self._chain = bytes(32)
        self._event_count = 0
        self._writer = None
        self._writer_segment_index: int | None = None
        self._writer_path: Path | None = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.expanded_reconstruction_count = 0
        self.put_count = 0
        self.delete_count = 0
        self.max_active_entry_count = 0
        self.max_expanded_entry_count = 0
        self.max_event_bytes = 0

        if resume_snapshot is None:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
            if self.root.resolve(strict=True) != self.root or self.root.is_symlink():
                raise TasteT12TransitionStoreError(
                    "T12 transition root is an alias"
                )
        else:
            self._restore_snapshot(resume_snapshot)

    @property
    def store_bytes(self) -> int:
        return sum(int(row["committed_bytes"]) for row in self._segments)

    def _segment_header(self, *, segment_index: int) -> dict[str, Any]:
        return {
            "schema_version": TRANSITION_SEGMENT_SCHEMA,
            "store_schema_version": TRANSITION_STORE_SCHEMA,
            "segment_index": segment_index,
            "event_sequence_start": self._event_count + 1,
            "previous_chain_sha256": self._chain.hex(),
            "contract_sha256": self.contract_sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "parent_count": self.parent_count,
            "sample_size": self.sample_size,
            "candidate_capacity": self.candidate_capacity,
            "expanded_capacity": self.expanded_capacity,
            "max_store_bytes": self.max_store_bytes,
            "coverage_contract": "binary_row_major_packbits_little_v1",
        }

    def _ensure_writer(self) -> None:
        if not self._open_writer:
            raise TasteT12TransitionStoreError(
                "T12 transition store is opened read-only"
            )
        if self._writer is not None:
            return
        segment_index = len(self._segments)
        if segment_index >= TRANSITION_MAX_SEGMENTS:
            raise TasteT12TransitionStoreError(
                "T12 transition store exceeded the 10k/20k segment plan"
            )
        filename = f"transitions-{segment_index:02d}.bin"
        path = self.root / filename
        header = _canonical_bytes(self._segment_header(segment_index=segment_index))
        if len(header) > _MAX_EVENT_HEADER_BYTES:
            raise TasteT12TransitionStoreError(
                "T12 transition segment header exceeds its fixed bound"
            )
        prefix = TRANSITION_MAGIC + _HEADER_LENGTH.pack(len(header)) + header
        stream = path.open("xb")
        try:
            stream.write(prefix)
            stream.flush()
        except BaseException:
            stream.close()
            raise
        self._chain = hashlib.sha256(self._chain + prefix).digest()
        self._segments.append(
            {
                "segment_file": filename,
                "segment_index": segment_index,
                "committed_bytes": len(prefix),
                "event_count": 0,
                "final_chain_sha256": self._chain.hex(),
            }
        )
        self._writer = stream
        self._writer_segment_index = segment_index
        self._writer_path = path

    def _append_event(
        self, *, header: Mapping[str, Any], payload: bytes
    ) -> _RecordLocation:
        self._ensure_writer()
        assert self._writer is not None
        assert self._writer_segment_index is not None
        assert self._writer_path is not None
        header_bytes = _canonical_bytes(dict(header))
        if not 0 < len(header_bytes) <= _MAX_EVENT_HEADER_BYTES:
            raise TasteT12TransitionStoreError(
                "T12 transition event header exceeds its fixed bound"
            )
        frame = _EVENT_PREFIX.pack(len(header_bytes), len(payload))
        event_chain = hashlib.sha256(
            self._chain + frame + header_bytes + payload
        ).digest()
        event = frame + header_bytes + payload + event_chain
        projected = self.store_bytes + len(event)
        if projected > self.max_store_bytes:
            raise TasteT12TransitionStoreError(
                "T12 transition journal exceeded its external disk cap"
            )
        offset = self._writer.tell()
        self._writer.write(event)
        self._writer.flush()
        self._chain = event_chain
        self._event_count += 1
        row = self._segments[self._writer_segment_index]
        row["committed_bytes"] = int(row["committed_bytes"]) + len(event)
        row["event_count"] = int(row["event_count"]) + 1
        row["final_chain_sha256"] = self._chain.hex()
        self.max_event_bytes = max(self.max_event_bytes, len(event))
        return _RecordLocation(
            segment_file=self._writer_path.name,
            offset=offset,
            event_bytes=len(event),
            event_sequence=self._event_count,
        )

    def _encode_transition(self, source: str, value: Any) -> tuple[dict[str, Any], bytes]:
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            raise TasteT12TransitionStoreError(
                "T12 transition must use the official four-part tuple"
            )
        target_hashes_raw, actions_raw, importance_raw, coverage = value
        if not isinstance(target_hashes_raw, (list, tuple)) or not isinstance(
            actions_raw, (list, tuple)
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition hashes/actions are malformed"
            )
        target_hashes = [
            _require_sha256(item, field="T12 transition target")
            for item in target_hashes_raw
        ]
        actions = [_freeze_action(item) for item in actions_raw]
        if (
            not target_hashes
            or len(target_hashes) > self.sample_size + 1
            or len(actions) != len(target_hashes)
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition target/action rows exceed the pinned sample size"
            )
        importance = np.asarray(importance_raw)
        if (
            importance.ndim != 2
            or importance.shape[0] != len(target_hashes)
            or importance.dtype.kind not in "fiu"
            or not np.isfinite(importance).all()
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition importance rows are not finite aligned numerics"
            )
        importance = np.ascontiguousarray(importance)
        try:
            dense = coverage.to_dense() if bool(coverage.is_sparse) else coverage
            dense = dense.detach().cpu().contiguous()
            shape = tuple(int(item) for item in dense.shape)
            dense_numpy = np.ascontiguousarray(dense.numpy())
        except (AttributeError, TypeError, ValueError) as exc:
            raise TasteT12TransitionStoreError(
                "T12 transition coverage is not a CPU-convertible Torch tensor"
            ) from exc
        if shape != (len(target_hashes), self.parent_count):
            raise TasteT12TransitionStoreError(
                "T12 transition coverage shape differs from targets/parents"
            )
        if not np.isfinite(dense_numpy).all() or not np.logical_or(
            dense_numpy == 0, dense_numpy == 1
        ).all():
            raise TasteT12TransitionStoreError(
                "T12 transition coverage must be exactly binary"
            )
        coverage_dtype = _dtype_string(dense_numpy.dtype)
        coverage_bits = np.packbits(
            dense_numpy.reshape(-1).astype(np.uint8, copy=False),
            bitorder="little",
        ).tobytes(order="C")
        importance_bytes = importance.tobytes(order="C")
        payload = importance_bytes + coverage_bits
        header = {
            "schema_version": TRANSITION_STORE_SCHEMA,
            "op": "PUT",
            "event_sequence": self._event_count + 1,
            "source_hash": source,
            "target_hashes": target_hashes,
            "actions": actions,
            "importance_dtype": importance.dtype.str,
            "importance_shape": list(importance.shape),
            "importance_bytes": len(importance_bytes),
            "importance_sha256": _sha256_bytes(importance_bytes),
            "coverage_dtype": coverage_dtype,
            "coverage_shape": list(shape),
            "coverage_numel": int(dense_numpy.size),
            "coverage_packed_bytes": len(coverage_bits),
            "coverage_sha256": _sha256_bytes(coverage_bits),
            "payload_sha256": _sha256_bytes(payload),
        }
        return header, payload

    def __setitem__(self, key: str, value: Any) -> None:
        source = _require_sha256(key, field="T12 transition source")
        header, payload = self._encode_transition(source, value)
        location = self._append_event(header=header, payload=payload)
        # Native dict replacement retains insertion position.  Reassignment is
        # not expected on the official path but preserving it is inexpensive.
        self._index[source] = location
        self._expanded.pop(source, None)
        self._remember_expanded(source, value)
        self.put_count += 1
        self.max_active_entry_count = max(
            self.max_active_entry_count, len(self._index)
        )

    def __delitem__(self, key: str) -> None:
        source = _require_sha256(key, field="T12 transition source")
        if source not in self._index:
            raise KeyError(source)
        header = {
            "schema_version": TRANSITION_STORE_SCHEMA,
            "op": "DELETE",
            "event_sequence": self._event_count + 1,
            "source_hash": source,
            "payload_sha256": _sha256_bytes(b""),
        }
        self._append_event(header=header, payload=b"")
        del self._index[source]
        self._expanded.pop(source, None)
        self.delete_count += 1

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def _read_event(
        self, location: _RecordLocation
    ) -> tuple[dict[str, Any], bytes]:
        path = self.root / location.segment_file
        with path.open("rb") as stream:
            stream.seek(location.offset)
            prefix = stream.read(_EVENT_PREFIX.size)
            if len(prefix) != _EVENT_PREFIX.size:
                raise TasteT12TransitionStoreError(
                    "T12 transition event prefix is truncated"
                )
            header_size, payload_size = _EVENT_PREFIX.unpack(prefix)
            header_bytes = stream.read(header_size)
            payload = stream.read(payload_size)
            chain = stream.read(_CHAIN_BYTES)
        if (
            len(header_bytes) != header_size
            or len(payload) != payload_size
            or len(chain) != _CHAIN_BYTES
            or _EVENT_PREFIX.size
            + header_size
            + payload_size
            + _CHAIN_BYTES
            != location.event_bytes
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition event bytes are truncated"
            )
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TasteT12TransitionStoreError(
                "T12 transition event header is unreadable"
            ) from exc
        if (
            type(header) is not dict
            or header.get("op") != "PUT"
            or header.get("event_sequence") != location.event_sequence
            or header.get("payload_sha256") != _sha256_bytes(payload)
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition event content changed"
            )
        # Full chain membership is checked on checkpoint reopen.  Per-read
        # payload hashes catch mutation without rescanning multi-gigabyte
        # prefixes inside each official move.
        return header, payload

    def _decode(self, source: str, location: _RecordLocation) -> Any:
        header, payload = self._read_event(location)
        if header.get("source_hash") != source:
            raise TasteT12TransitionStoreError(
                "T12 transition source index changed"
            )
        target_hashes = list(header.get("target_hashes") or ())
        actions = [tuple(item) for item in header.get("actions") or ()]
        importance_bytes = _require_int(
            header.get("importance_bytes"),
            field="T12 transition importance bytes",
        )
        importance_payload = payload[:importance_bytes]
        coverage_payload = payload[importance_bytes:]
        if (
            header.get("importance_sha256")
            != _sha256_bytes(importance_payload)
            or header.get("coverage_sha256") != _sha256_bytes(coverage_payload)
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition numeric payload changed"
            )
        try:
            importance_dtype = np.dtype(header["importance_dtype"])
            importance_shape = tuple(int(item) for item in header["importance_shape"])
            importance = np.frombuffer(
                importance_payload, dtype=importance_dtype
            ).reshape(importance_shape).copy()
            coverage_dtype = np.dtype(header["coverage_dtype"])
            coverage_shape = tuple(int(item) for item in header["coverage_shape"])
            coverage_numel = int(header["coverage_numel"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TasteT12TransitionStoreError(
                "T12 transition numeric metadata is malformed"
            ) from exc
        if (
            importance.shape[0] != len(target_hashes)
            or len(actions) != len(target_hashes)
            or coverage_shape != (len(target_hashes), self.parent_count)
            or coverage_numel != math.prod(coverage_shape)
            or len(coverage_payload) != (coverage_numel + 7) // 8
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition numeric shapes changed"
            )
        unpacked = np.unpackbits(
            np.frombuffer(coverage_payload, dtype=np.uint8), bitorder="little"
        )[:coverage_numel]
        dense_numpy = unpacked.reshape(coverage_shape).astype(
            coverage_dtype, copy=False
        )
        try:
            import torch

            coverage = torch.from_numpy(np.array(dense_numpy, copy=True)).to_sparse()
        except (ImportError, RuntimeError, TypeError) as exc:
            raise TasteT12TransitionStoreError(
                "T12 transition coverage could not be reconstructed"
            ) from exc
        return (
            target_hashes,
            actions,
            [np.array(row, copy=True) for row in importance],
            coverage,
        )

    def _remember_expanded(self, source: str, value: Any) -> None:
        self._expanded.pop(source, None)
        self._expanded[source] = value
        while len(self._expanded) > self.expanded_capacity:
            self._expanded.popitem(last=False)
        self.max_expanded_entry_count = max(
            self.max_expanded_entry_count, len(self._expanded)
        )

    def __getitem__(self, key: str) -> Any:
        source = _require_sha256(key, field="T12 transition source")
        location = self._index.get(source)
        if location is None:
            raise KeyError(source)
        if source in self._expanded:
            self.cache_hit_count += 1
            value = self._expanded.pop(source)
            self._expanded[source] = value
            return value
        self.cache_miss_count += 1
        value = self._decode(source, location)
        self.expanded_reconstruction_count += 1
        self._remember_expanded(source, value)
        return value

    def clear_expanded(self) -> None:
        self._expanded.clear()

    def _scan_segment(
        self,
        *,
        row: Mapping[str, Any],
        expected_chain: bytes,
        index: dict[str, _RecordLocation],
    ) -> tuple[bytes, int]:
        filename = row.get("segment_file")
        if (
            type(filename) is not str
            or Path(filename).name != filename
            or not filename.startswith("transitions-")
            or not filename.endswith(".bin")
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition segment filename is invalid"
            )
        committed_bytes = _require_int(
            row.get("committed_bytes"),
            field="T12 transition committed bytes",
            minimum=len(TRANSITION_MAGIC) + _HEADER_LENGTH.size + 2,
        )
        path = self.root / filename
        resolved = path.resolve(strict=True)
        info = path.stat()
        if (
            resolved != path
            or path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_size < committed_bytes
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition segment physical identity is invalid"
            )
        with path.open("rb") as stream:
            magic = stream.read(len(TRANSITION_MAGIC))
            length_bytes = stream.read(_HEADER_LENGTH.size)
            if magic != TRANSITION_MAGIC or len(length_bytes) != _HEADER_LENGTH.size:
                raise TasteT12TransitionStoreError(
                    "T12 transition segment header is truncated"
                )
            (header_size,) = _HEADER_LENGTH.unpack(length_bytes)
            header_bytes = stream.read(header_size)
            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TasteT12TransitionStoreError(
                    "T12 transition segment header is unreadable"
                ) from exc
            segment_index = _require_int(
                row.get("segment_index"), field="T12 transition segment index"
            )
            expected_header = self._segment_header(segment_index=segment_index)
            # During scan ``event_count`` and ``chain`` reflect the preceding
            # committed segment and therefore reproduce the original header.
            if header != expected_header:
                raise TasteT12TransitionStoreError(
                    "T12 transition segment contract changed"
                )
            prefix = magic + length_bytes + header_bytes
            chain = hashlib.sha256(expected_chain + prefix).digest()
            position = len(prefix)
            event_count = 0
            while position < committed_bytes:
                offset = position
                event_prefix = stream.read(_EVENT_PREFIX.size)
                if len(event_prefix) != _EVENT_PREFIX.size:
                    raise TasteT12TransitionStoreError(
                        "T12 transition committed event prefix is truncated"
                    )
                event_header_size, payload_size = _EVENT_PREFIX.unpack(event_prefix)
                if not 0 < event_header_size <= _MAX_EVENT_HEADER_BYTES:
                    raise TasteT12TransitionStoreError(
                        "T12 transition event header size is invalid"
                    )
                event_header_bytes = stream.read(event_header_size)
                payload = stream.read(payload_size)
                observed_chain = stream.read(_CHAIN_BYTES)
                event_bytes = (
                    _EVENT_PREFIX.size
                    + event_header_size
                    + payload_size
                    + _CHAIN_BYTES
                )
                position += event_bytes
                if (
                    position > committed_bytes
                    or len(event_header_bytes) != event_header_size
                    or len(payload) != payload_size
                    or len(observed_chain) != _CHAIN_BYTES
                ):
                    raise TasteT12TransitionStoreError(
                        "T12 transition committed event is truncated"
                    )
                computed = hashlib.sha256(
                    chain + event_prefix + event_header_bytes + payload
                ).digest()
                if computed != observed_chain:
                    raise TasteT12TransitionStoreError(
                        "T12 transition checkpoint hash chain changed"
                    )
                chain = computed
                try:
                    event_header = json.loads(event_header_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TasteT12TransitionStoreError(
                        "T12 transition event header is unreadable"
                    ) from exc
                sequence = self._event_count + event_count + 1
                source = _require_sha256(
                    event_header.get("source_hash"),
                    field="T12 transition event source",
                )
                if (
                    event_header.get("schema_version") != TRANSITION_STORE_SCHEMA
                    or event_header.get("event_sequence") != sequence
                    or event_header.get("payload_sha256") != _sha256_bytes(payload)
                ):
                    raise TasteT12TransitionStoreError(
                        "T12 transition event identity changed"
                    )
                op = event_header.get("op")
                if op == "PUT":
                    index[source] = _RecordLocation(
                        segment_file=filename,
                        offset=offset,
                        event_bytes=event_bytes,
                        event_sequence=sequence,
                    )
                elif op == "DELETE" and source in index and not payload:
                    del index[source]
                else:
                    raise TasteT12TransitionStoreError(
                        "T12 transition event operation is invalid"
                    )
                event_count += 1
            if position != committed_bytes:
                raise TasteT12TransitionStoreError(
                    "T12 transition segment prefix boundary changed"
                )
        if (
            event_count != row.get("event_count")
            or chain.hex() != row.get("final_chain_sha256")
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition segment checkpoint summary changed"
            )
        return chain, event_count

    def _snapshot_base(self) -> dict[str, Any]:
        return {
            "schema_version": TRANSITION_SNAPSHOT_SCHEMA,
            "store_schema_version": TRANSITION_STORE_SCHEMA,
            "policy": TRANSITION_STORE_POLICY,
            "root": str(self.root),
            "parent_count": self.parent_count,
            "sample_size": self.sample_size,
            "candidate_capacity": self.candidate_capacity,
            "contract_sha256": self.contract_sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "expanded_capacity": self.expanded_capacity,
            "max_store_bytes": self.max_store_bytes,
        }

    def export_checkpoint_state(self) -> dict[str, Any]:
        """Fsync and export an exact graph-object-free committed prefix."""

        if self._writer is not None:
            self._writer.flush()
            os.fsync(self._writer.fileno())
            _fsync_directory(self.root)
        counters = {
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "expanded_reconstruction_count": self.expanded_reconstruction_count,
            "put_count": self.put_count,
            "delete_count": self.delete_count,
            "max_active_entry_count": self.max_active_entry_count,
            "max_expanded_entry_count": self.max_expanded_entry_count,
            "max_event_bytes": self.max_event_bytes,
        }
        return {
            **self._snapshot_base(),
            "segments": [dict(row) for row in self._segments],
            "event_count": self._event_count,
            "active_sources": list(self._index),
            "active_entry_count": len(self._index),
            "expanded_keys_lru_order": list(self._expanded),
            "chain_sha256": self._chain.hex(),
            "committed_store_bytes": self.store_bytes,
            "counters": counters,
            "checkpoint_export_contains_coverage_payload": False,
            "external_journal_is_authority": True,
            "model_recomputation_count": 0,
            "rng_calls_added": 0,
            "neighbor_order_changed": False,
            "candidate_order_changed": False,
            "scientific_parameters_changed": False,
        }

    def _restore_snapshot(self, raw: Mapping[str, Any]) -> None:
        expected_keys = {
            *self._snapshot_base(),
            "segments",
            "event_count",
            "active_sources",
            "active_entry_count",
            "expanded_keys_lru_order",
            "chain_sha256",
            "committed_store_bytes",
            "counters",
            "checkpoint_export_contains_coverage_payload",
            "external_journal_is_authority",
            "model_recomputation_count",
            "rng_calls_added",
            "neighbor_order_changed",
            "candidate_order_changed",
            "scientific_parameters_changed",
        }
        if type(raw) is not dict or set(raw) != expected_keys:
            raise TasteT12TransitionStoreError(
                "T12 transition checkpoint keys changed"
            )
        for field, value in self._snapshot_base().items():
            if raw.get(field) != value:
                raise TasteT12TransitionStoreError(
                    f"T12 transition checkpoint {field} changed"
                )
        if any(
            raw.get(field) is not expected
            for field, expected in {
                "checkpoint_export_contains_coverage_payload": False,
                "external_journal_is_authority": True,
                "neighbor_order_changed": False,
                "candidate_order_changed": False,
                "scientific_parameters_changed": False,
            }.items()
        ) or raw.get("model_recomputation_count") != 0 or raw.get(
            "rng_calls_added"
        ) != 0:
            raise TasteT12TransitionStoreError(
                "T12 transition checkpoint scientific semantics changed"
            )
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.resolve(strict=True) != self.root or self.root.is_symlink():
            raise TasteT12TransitionStoreError("T12 transition root is an alias")
        segments = raw.get("segments")
        if (
            type(segments) is not list
            or not 0 <= len(segments) <= TRANSITION_MAX_SEGMENTS
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition checkpoint segment plan changed"
            )
        self._index = {}
        self._chain = bytes(32)
        self._event_count = 0
        self._segments = []
        for expected_index, row in enumerate(segments):
            if (
                type(row) is not dict
                or set(row)
                != {
                    "segment_file",
                    "segment_index",
                    "committed_bytes",
                    "event_count",
                    "final_chain_sha256",
                }
                or row.get("segment_index") != expected_index
            ):
                raise TasteT12TransitionStoreError(
                    "T12 transition segment snapshot changed"
                )
            chain, count = self._scan_segment(
                row=row, expected_chain=self._chain, index=self._index
            )
            self._chain = chain
            self._event_count += count
            self._segments.append(dict(row))
        active = raw.get("active_sources")
        expanded = raw.get("expanded_keys_lru_order")
        if (
            active != list(self._index)
            or raw.get("active_entry_count") != len(self._index)
            or raw.get("event_count") != self._event_count
            or raw.get("chain_sha256") != self._chain.hex()
            or raw.get("committed_store_bytes") != self.store_bytes
            or type(expanded) is not list
            or len(expanded) > self.expanded_capacity
            or len(set(expanded)) != len(expanded)
            or any(key not in self._index for key in expanded)
        ):
            raise TasteT12TransitionStoreError(
                "T12 transition checkpoint derived state changed"
            )
        counters = raw.get("counters")
        counter_names = {
            "cache_hit_count",
            "cache_miss_count",
            "expanded_reconstruction_count",
            "put_count",
            "delete_count",
            "max_active_entry_count",
            "max_expanded_entry_count",
            "max_event_bytes",
        }
        if type(counters) is not dict or set(counters) != counter_names:
            raise TasteT12TransitionStoreError(
                "T12 transition checkpoint counters changed"
            )
        for name in counter_names:
            setattr(self, name, _require_int(counters[name], field=f"T12 {name}"))
        if self.max_active_entry_count < len(self._index):
            raise TasteT12TransitionStoreError(
                "T12 transition maximum active count changed"
            )
        # Reconstruct the exact bounded LRU without incrementing scientific or
        # diagnostic counters.  This invokes no model, action enumeration, or RNG.
        for source in expanded:
            self._remember_expanded(source, self._decode(source, self._index[source]))
        if self.export_checkpoint_state() != dict(raw):
            raise TasteT12TransitionStoreError(
                "T12 transition checkpoint did not reopen exactly"
            )

    def restore_checkpoint_state(self, value: Mapping[str, Any]) -> None:
        """Restore into an empty store or prove an already reopened store exact."""

        if self._event_count or self._segments or self._index or self._writer is not None:
            if self.export_checkpoint_state() != dict(value):
                raise TasteT12TransitionStoreError(
                    "T12 transition restore target differs from checkpoint"
                )
            return
        self._restore_snapshot(value)

    @classmethod
    def verify_checkpoint_state(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        """Independently rescan one external prefix without opening a writer."""

        if type(value) is not dict:
            raise TasteT12TransitionStoreError(
                "T12 transition checkpoint is not one mapping"
            )
        store = cls(
            root=value.get("root"),
            parent_count=value.get("parent_count"),
            sample_size=value.get("sample_size"),
            candidate_capacity=value.get("candidate_capacity"),
            contract_sha256=value.get("contract_sha256"),
            attempt_id=value.get("attempt_id"),
            generation_token=value.get("generation_token"),
            expanded_capacity=value.get("expanded_capacity"),
            max_store_bytes=value.get("max_store_bytes"),
            resume_snapshot=value,
            open_writer=False,
        )
        try:
            observed = store.export_checkpoint_state()
        finally:
            store.close()
        if observed != dict(value):
            raise TasteT12TransitionStoreError(
                "T12 transition independent checkpoint verification changed"
            )
        return observed

    def validate_live_domain(
        self,
        *,
        live_sources: set[str],
        known_target: Any,
    ) -> dict[str, int]:
        """Validate transition metadata without expanding coverage matrices."""

        target_count = 0
        for source, location in self._index.items():
            if source not in live_sources:
                raise TasteT12TransitionStoreError(
                    "T12 external transition source left the official live domain"
                )
            header, _payload = self._read_event(location)
            targets = header.get("target_hashes")
            if (
                type(targets) is not list
                or not 1 <= len(targets) <= self.sample_size + 1
            ):
                raise TasteT12TransitionStoreError(
                    "T12 external transition target metadata changed"
                )
            for target in targets:
                target = _require_sha256(target, field="T12 transition target")
                if not bool(known_target(target)):
                    raise TasteT12TransitionStoreError(
                        "T12 transition target is absent from compact history"
                    )
            target_count += len(targets)
        return {
            "transition_entry_count": len(self._index),
            "transition_target_reference_count": target_count,
        }

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": TRANSITION_STORE_SCHEMA,
            "policy": TRANSITION_STORE_POLICY,
            "active_entry_count": len(self._index),
            "parent_count": self.parent_count,
            "sample_size": self.sample_size,
            "candidate_capacity": self.candidate_capacity,
            "contract_sha256": self.contract_sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "event_count": self._event_count,
            "expanded_capacity": self.expanded_capacity,
            "expanded_entry_count": len(self._expanded),
            "max_active_entry_count": self.max_active_entry_count,
            "max_expanded_entry_count": self.max_expanded_entry_count,
            "committed_store_bytes": self.store_bytes,
            "max_store_bytes": self.max_store_bytes,
            "chain_sha256": self._chain.hex(),
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "expanded_reconstruction_count": self.expanded_reconstruction_count,
            "put_count": self.put_count,
            "delete_count": self.delete_count,
            "coverage_storage": "binary_row_major_packbits_little",
            "coverage_payload_in_ram_is_lru_bounded": True,
            "external_journal_is_authority": True,
            "model_recomputation_count": 0,
            "rng_calls_added": 0,
            "neighbor_order_changed": False,
            "candidate_order_changed": False,
            "scientific_parameters_changed": False,
        }

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            os.fsync(self._writer.fileno())
            self._writer.close()
            self._writer = None
            _fsync_directory(self.root)

    def __enter__(self) -> "T12ExternalTransitionStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = [
    "TRANSITION_DEFAULT_EXPANDED_CAPACITY",
    "TRANSITION_DEFAULT_MAX_STORE_BYTES",
    "TRANSITION_SNAPSHOT_SCHEMA",
    "TRANSITION_STORE_POLICY",
    "TRANSITION_STORE_SCHEMA",
    "T12ExternalTransitionStore",
    "TasteT12TransitionStoreError",
]
