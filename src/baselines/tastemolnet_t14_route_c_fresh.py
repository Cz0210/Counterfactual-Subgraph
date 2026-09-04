"""Dataset-specific ownership contract for a fresh low-memory Taste T14 run.

Route C deliberately starts from step zero.  A fresh production attempt seals
early recovery checkpoints at steps 50, 100, and 250; the parity-qualified
canary becomes promotable at step 500, and it never treats a legacy checkpoint
as an input.  Promotion performs payload reload after the science process has
released its live state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from collections import OrderedDict, deque
from collections.abc import Iterator, MutableMapping, MutableSequence, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
import pickle
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
from typing import Any, Callable, Mapping
from uuid import UUID

from src.baselines.comrecgc.live_graph_state import LiveGraphState


SPEC_SCHEMA = "tastemolnet_t14_route_c_fresh_spec_v1"
BOUNDARY_SCHEMA = "tastemolnet_t14_route_c_first_checkpoint_v1"
PROMOTION_SCHEMA = "tastemolnet_t14_route_c_promotion_v1"
FIRST_CHECKPOINT_STEP = 50
PARITY_RELOAD_CHECKPOINT_STEP = 250
PROMOTABLE_CHECKPOINT_STEP = 500
RELOAD_REPLAY_END_STEP = 510
EARLY_CHECKPOINT_STEPS = (50, 100, 250)
PRODUCTION_CHECKPOINT_STEPS = (
    500,
    2_500,
    5_000,
    7_500,
    10_000,
    12_500,
    15_000,
    17_500,
    20_000,
    25_000,
)
GPU_INDEX = 2
M_MAX = 20_000
M_FALLBACK_MAX = 25_000
GRAPH_STORE_SCHEMA = "tastemolnet_t14_route_c_append_only_graph_store_v1"
CANDIDATE_STATE_SCHEMA = "tastemolnet_t14_route_c_mmap_candidate_state_v1"
STEP_LEDGER_SCHEMA = "tastemolnet_t14_route_c_step_state_v1"
PARITY_SCHEMA = "tastemolnet_t14_route_c_parity_v1"
RECOVERY_SCHEMA = "tastemolnet_t14_route_c_external_state_recovery_v1"
CONVERGENCE_RECEIPT_SCHEMA = "tastemolnet_t14_route_c_convergence_receipt_v1"
FRESH_RETRY_RECEIPT_SCHEMA = "tastemolnet_t14_route_c_failed_attempt_retirement_v1"
FRESH_RETRY_CONTRACT_SCHEMA = "tastemolnet_t14_route_c_fresh_retry_task_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class T14RouteCFreshError(RuntimeError):
    """The fresh Route C ownership or checkpoint contract was violated."""


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_numeric_graph_id(graph_hash: Any) -> int:
    """Return a process-independent positive signed-63-bit storage identity."""

    encoded = canonical_bytes({"official_graph_hash": graph_hash})
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return (value & ((1 << 63) - 1)) or 1


def _pickle_sha256(value: Any) -> tuple[bytes, str]:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return payload, hashlib.sha256(payload).hexdigest()


class RouteCAppendOnlyGraphStore:
    """One T14-only append-only blob store with a compact SQLite index.

    The SQLite database contains only stable numeric IDs and blob locators;
    PyG objects live once in ``graphs.bin``.  Runtime eviction removes only a
    logical active-map entry.  It never deletes or rewrites a graph blob.
    """

    def __init__(self, root: Path, *, lru_capacity: int = 128) -> None:
        if lru_capacity <= 0:
            raise T14RouteCFreshError("Route C graph LRU capacity must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise T14RouteCFreshError("Route C graph-store root cannot be a symlink")
        self.data_path = self.root / "graphs.bin"
        self.index_path = self.root / "graph_index.sqlite3"
        self._data = self.data_path.open("a+b", buffering=0)
        self._connection = sqlite3.connect(str(self.index_path), timeout=60.0)
        self._connection.execute("PRAGMA busy_timeout=60000")
        mode = str(self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0])
        if mode.lower() != "wal":
            raise T14RouteCFreshError("Route C graph index requires SQLite WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS graphs (
                graph_id INTEGER PRIMARY KEY,
                key_blob BLOB NOT NULL,
                key_sha256 TEXT NOT NULL UNIQUE,
                offset INTEGER NOT NULL,
                length INTEGER NOT NULL,
                payload_sha256 TEXT NOT NULL,
                graph_sha256 TEXT NOT NULL,
                sequence_id INTEGER NOT NULL,
                chain_sha256 TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self._lru_capacity = int(lru_capacity)
        self._lru: OrderedDict[int, Any] = OrderedDict()
        self._writer_pid = os.getpid()
        self._writer_thread = threading.get_ident()
        self.read_count = 0
        self.write_count = 0

    @property
    def checkpoint_connection(self) -> sqlite3.Connection:
        return self._connection

    def _assert_writer(self) -> None:
        if os.getpid() != self._writer_pid or threading.get_ident() != self._writer_thread:
            raise T14RouteCFreshError(
                "Route C scientific state accepts exactly one process/thread updater"
            )

    @staticmethod
    def _key(key: Any) -> tuple[bytes, str, int]:
        key_blob, key_sha = _pickle_sha256(key)
        return key_blob, key_sha, _stable_numeric_graph_id(key)

    def graph_id(self, key: Any) -> int | None:
        _blob, key_sha, graph_id = self._key(key)
        row = self._connection.execute(
            "SELECT graph_id FROM graphs WHERE key_sha256 = ?", (key_sha,)
        ).fetchone()
        if row is None:
            return None
        if int(row[0]) != graph_id:
            raise T14RouteCFreshError("Route C stable numeric graph ID changed")
        return graph_id

    def contains(self, key: Any) -> bool:
        return self.graph_id(key) is not None

    def put(self, key: Any, value: Any, *, sequence_id: int) -> int:
        self._assert_writer()
        key_blob, key_sha, graph_id = self._key(key)
        existing = self._connection.execute(
            "SELECT graph_id,payload_sha256,graph_sha256 FROM graphs WHERE key_sha256=?",
            (key_sha,),
        ).fetchone()
        try:
            from src.baselines.comrecgc.live_graph_state import _entry_graph_sha256

            graph_sha = _entry_graph_sha256(value)
        except Exception as exc:
            raise T14RouteCFreshError("Route C graph payload is invalid") from exc
        payload, payload_sha = _pickle_sha256(value)
        if existing is not None:
            if (
                int(existing[0]) != graph_id
                or str(existing[1]) != payload_sha
                or str(existing[2]) != graph_sha
            ):
                raise T14RouteCFreshError(
                    "Route C graph identity was observed with different bytes"
                )
            self._remember(graph_id, value)
            return graph_id
        collision = self._connection.execute(
            "SELECT key_sha256 FROM graphs WHERE graph_id=?", (graph_id,)
        ).fetchone()
        if collision is not None:
            raise T14RouteCFreshError("Route C stable numeric graph-ID collision")
        self._data.seek(0, os.SEEK_END)
        offset = self._data.tell()
        self._data.write(payload)
        previous = self._connection.execute(
            "SELECT chain_sha256 FROM graphs ORDER BY sequence_id DESC,graph_id DESC LIMIT 1"
        ).fetchone()
        previous_sha = str(previous[0]) if previous else "0" * 64
        chain_sha = hashlib.sha256(
            bytes.fromhex(previous_sha)
            + bytes.fromhex(key_sha)
            + bytes.fromhex(payload_sha)
            + bytes.fromhex(graph_sha)
        ).hexdigest()
        with self._connection:
            self._connection.execute(
                "INSERT INTO graphs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    graph_id,
                    key_blob,
                    key_sha,
                    offset,
                    len(payload),
                    payload_sha,
                    graph_sha,
                    int(sequence_id),
                    chain_sha,
                ),
            )
        self.write_count += 1
        self._remember(graph_id, value)
        return graph_id

    def _remember(self, graph_id: int, value: Any) -> None:
        self._lru.pop(graph_id, None)
        self._lru[graph_id] = value
        while len(self._lru) > self._lru_capacity:
            self._lru.popitem(last=False)

    def get(self, key: Any) -> Any:
        key_blob, key_sha, graph_id = self._key(key)
        row = self._connection.execute(
            "SELECT key_blob,offset,length,payload_sha256 FROM graphs WHERE key_sha256=?",
            (key_sha,),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        if int(graph_id) in self._lru:
            value = self._lru.pop(graph_id)
            self._lru[graph_id] = value
            self.read_count += 1
            return value
        if bytes(row[0]) != key_blob:
            raise T14RouteCFreshError("Route C graph-store key digest collision")
        self._data.seek(int(row[1]))
        payload = self._data.read(int(row[2]))
        if len(payload) != int(row[2]) or hashlib.sha256(payload).hexdigest() != row[3]:
            raise T14RouteCFreshError("Route C append-only graph payload changed")
        value = pickle.loads(payload)
        self._remember(graph_id, value)
        self.read_count += 1
        return value

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM graphs").fetchone()[0])

    def flush(self) -> None:
        self._assert_writer()
        self._data.flush()
        os.fsync(self._data.fileno())
        self._connection.commit()
        self._connection.execute("PRAGMA wal_checkpoint(FULL)")
        _fsync_dir(self.root)

    def checkpoint_state(self) -> dict[str, Any]:
        self.flush()
        last = self._connection.execute(
            "SELECT chain_sha256 FROM graphs ORDER BY sequence_id DESC,graph_id DESC LIMIT 1"
        ).fetchone()
        return {
            "schema_version": GRAPH_STORE_SCHEMA,
            "data_path": self.data_path.name,
            "index_path": self.index_path.name,
            "entry_count": self.count(),
            "data_bytes": self.data_path.stat().st_size,
            "chain_sha256": str(last[0]) if last else "0" * 64,
            "lru_capacity": self._lru_capacity,
            "graph_objects_in_checkpoint": 0,
            "append_only": True,
        }

    def validate_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema_version") != GRAPH_STORE_SCHEMA
            or state.get("data_path") != self.data_path.name
            or state.get("index_path") != self.index_path.name
            or state.get("append_only") is not True
            or state.get("graph_objects_in_checkpoint") != 0
            or int(state.get("lru_capacity", -1)) != self._lru_capacity
        ):
            raise T14RouteCFreshError("Route C graph-store checkpoint contract changed")
        count = int(state.get("entry_count", -1))
        data_bytes = int(state.get("data_bytes", -1))
        if self.count() != count or self.data_path.stat().st_size != data_bytes:
            raise T14RouteCFreshError(
                "Route C graph-store differs from the clean checkpoint boundary"
            )
        row = self._connection.execute(
            "SELECT chain_sha256 FROM graphs ORDER BY sequence_id,graph_id LIMIT 1 OFFSET ?",
            (max(count - 1, 0),),
        ).fetchone()
        observed = str(row[0]) if row is not None and count else "0" * 64
        if observed != state.get("chain_sha256"):
            raise T14RouteCFreshError("Route C graph-store committed prefix changed")

    def close(self) -> None:
        self.flush()
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._connection.close()
        self._data.close()


class RouteCGraphMap(MutableMapping[Any, Any]):
    """Logical official graph map backed by one bounded lazy object cache."""

    def __init__(
        self,
        store: RouteCAppendOnlyGraphStore,
        values: Mapping[Any, Any],
        *,
        next_sequence: Callable[[], int],
        module: Any | None = None,
    ) -> None:
        self.store = store
        self.module = module
        self._active: OrderedDict[Any, int] = OrderedDict()
        self._next_sequence = next_sequence
        self.current_step = 0
        self.current_graph_hashes: tuple[Any, ...] = ()
        self.pin_counts: dict[Any, int] = {}
        self.deferred_deletions: set[Any] = set()
        self.eviction_attempts = 0
        self.eviction_committed = 0
        self.eviction_deferred = 0
        self.active_eviction_prevented = 0
        self.deferred_flushed = 0
        self.rehydrations = 0
        self.unresolved_lookups = 0
        self.max_hot_cache_size = 0
        self.missing_unmaterialized_eviction_count = 0
        self.recent_evictions: deque[dict[str, Any]] = deque(maxlen=32)
        for key, value in values.items():
            self[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._active

    def __len__(self) -> int:
        return len(self._active)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._active)

    def __getitem__(self, key: Any) -> Any:
        try:
            value = self.store.get(key)
        except KeyError as exc:
            self.unresolved_lookups += 1
            raise KeyError(key) from exc
        if key not in self._active:
            self.rehydrations += 1
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        graph_id = self.store.put(key, value, sequence_id=self._next_sequence())
        self._active[key] = graph_id
        self.max_hot_cache_size = max(self.max_hot_cache_size, len(self._active))

    def __delitem__(self, key: Any) -> None:
        self.eviction_attempts += 1
        if key not in self._active:
            candidates = getattr(self.module, "counterfactual_candidates", ())
            tail_hash = candidates[-1].get("graph_hash") if candidates else None
            index_map = getattr(self.module, "graph_index_map", {})
            if key != tail_hash or key in index_map:
                raise KeyError(key)
            self.missing_unmaterialized_eviction_count += 1
            return
        value = self.store.get(key)
        try:
            from src.baselines.comrecgc.live_graph_state import _entry_graph_sha256

            graph_sha256 = _entry_graph_sha256(value)
        except Exception as exc:
            raise T14RouteCFreshError("Route C evicted graph payload is invalid") from exc
        was_pinned = self.pin_counts.get(key, 0) > 0
        if was_pinned:
            self.active_eviction_prevented += 1
            self.eviction_deferred += 1
            self.deferred_deletions.add(key)
        self._active.pop(key)
        self.eviction_committed += 1
        self.recent_evictions.append(
            {
                "graph_hash": str(key),
                "current_step": self.current_step,
                "was_pinned": was_pinned,
                "graph_sha256": graph_sha256,
            }
        )

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self._active and not self.store.contains(key):
            return default
        return self[key]

    def contains_resolvable(self, key: Any) -> bool:
        return key in self._active or self.store.contains(key)

    def begin_move(self, graph_hashes: Sequence[Any], *, current_step: int) -> None:
        self.current_step = int(current_step)
        self.current_graph_hashes = tuple(graph_hashes)

    def end_move(self) -> None:
        self.current_graph_hashes = ()
        self.flush_deferred()

    @contextmanager
    def pin_many(self, keys: Sequence[Any]) -> Iterator[None]:
        for key in keys:
            self.pin_counts[key] = self.pin_counts.get(key, 0) + 1
        try:
            yield
        finally:
            for key in keys:
                count = self.pin_counts[key] - 1
                if count:
                    self.pin_counts[key] = count
                else:
                    self.pin_counts.pop(key, None)
            self.flush_deferred()

    def flush_deferred(self) -> None:
        for key in tuple(self.deferred_deletions):
            if self.pin_counts.get(key, 0) == 0:
                self.deferred_deletions.remove(key)
                self.deferred_flushed += 1

    def export_checkpoint_state(self) -> dict[str, Any]:
        if self.pin_counts or self.deferred_deletions or self.current_graph_hashes:
            raise T14RouteCFreshError("Route C graph map checkpointed inside a move")
        return {
            "schema_version": "tastemolnet_t14_route_c_graph_map_v1",
            "active_keys": list(self._active),
            "active_graph_ids": list(self._active.values()),
            "current_step": self.current_step,
            "eviction_attempts": self.eviction_attempts,
            "eviction_committed": self.eviction_committed,
            "eviction_deferred": self.eviction_deferred,
            "active_eviction_prevented": self.active_eviction_prevented,
            "deferred_flushed": self.deferred_flushed,
            "rehydrations": self.rehydrations,
            "unresolved_lookups": self.unresolved_lookups,
            "missing_unmaterialized_eviction_count": (
                self.missing_unmaterialized_eviction_count
            ),
            "max_hot_cache_size": self.max_hot_cache_size,
            "recent_evictions": list(self.recent_evictions),
            "store": self.store.checkpoint_state(),
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "tastemolnet_t14_route_c_graph_map_v1":
            raise T14RouteCFreshError("Route C graph-map checkpoint schema changed")
        self.store.validate_checkpoint_state(state["store"])
        keys = list(state.get("active_keys") or ())
        graph_ids = [int(value) for value in state.get("active_graph_ids") or ()]
        if len(keys) != len(graph_ids) or len(set(keys)) != len(keys):
            raise T14RouteCFreshError("Route C active graph index is malformed")
        for key, graph_id in zip(keys, graph_ids, strict=True):
            if self.store.graph_id(key) != graph_id:
                raise T14RouteCFreshError("Route C active graph ID changed")
        self._active = OrderedDict(zip(keys, graph_ids, strict=True))
        for name in (
            "current_step",
            "eviction_attempts",
            "eviction_committed",
            "eviction_deferred",
            "active_eviction_prevented",
            "deferred_flushed",
            "rehydrations",
            "unresolved_lookups",
            "missing_unmaterialized_eviction_count",
            "max_hot_cache_size",
        ):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise T14RouteCFreshError(
                    f"Route C graph-map checkpoint counter is invalid: {name}"
                )
            setattr(self, name, value)
        if self.max_hot_cache_size < len(self._active):
            raise T14RouteCFreshError("Route C maximum active graph size is inconsistent")
        recent = state.get("recent_evictions")
        if not isinstance(recent, list) or any(not isinstance(row, dict) for row in recent):
            raise T14RouteCFreshError("Route C recent eviction state is malformed")
        self.recent_evictions = deque(recent, maxlen=32)

    def runtime_diagnostics(self) -> dict[str, Any]:
        return {
            "policy": "t14_route_c_append_only_graph_store_bounded_lru_v1",
            "active_graph_count": len(self),
            "persistent_graph_count": self.store.count(),
            "hot_graph_count": len(self.store._lru),  # noqa: SLF001
            "max_hot_graph_count": self.max_hot_cache_size,
            "pins": sum(self.pin_counts.values()),
            "deferred_deletions": len(self.deferred_deletions),
            "eviction_attempts": self.eviction_attempts,
            "eviction_committed": self.eviction_committed,
            "eviction_deferred": self.eviction_deferred,
            "active_eviction_prevented": self.active_eviction_prevented,
            "deferred_flushed": self.deferred_flushed,
            "rehydrations": self.rehydrations,
            "unresolved_lookups": self.unresolved_lookups,
            "missing_unmaterialized_eviction_count": (
                self.missing_unmaterialized_eviction_count
            ),
            "stable_numeric_graph_ids": True,
            "append_only_graph_store": True,
            "recent_evictions": list(self.recent_evictions),
        }


class RouteCLiveGraphState(LiveGraphState):
    """T14-only adapter consumed by the existing exact transition runtime."""

    def __init__(self, updater: "RouteCStateUpdater", module: Any, values: Mapping[Any, Any]):
        self.module = module
        self.store = updater.graph_store
        self.graph_map = RouteCGraphMap(
            self.store,
            values,
            next_sequence=updater.next_sequence,
            module=module,
        )
        self.move_count = 0

    def resolve_graph(self, graph_hash: Any) -> Any:
        return self.graph_map[graph_hash][0]

    def contains(self, graph_hash: Any) -> bool:
        return self.graph_map.contains_resolvable(graph_hash)

    @contextmanager
    def pin_many(self, graph_hashes: Sequence[Any]) -> Iterator[None]:
        with self.graph_map.pin_many(graph_hashes):
            yield

    def wrap_move(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            hashes = list(kwargs.get("graphs_hash", args[0] if args else []))
            self.move_count += 1
            self.graph_map.begin_move(hashes, current_step=self.move_count)
            try:
                with self.pin_many(hashes):
                    return original(*args, **kwargs)
            finally:
                self.graph_map.end_move()

        return wrapped

    def export_checkpoint_state(self) -> dict[str, Any]:
        state = self.graph_map.export_checkpoint_state()
        if self.move_count != int(state["current_step"]):
            raise T14RouteCFreshError("Route C graph move counters differ")
        return {**state, "move_count": self.move_count}

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.graph_map.restore_checkpoint_state(state)
        self.move_count = int(state.get("move_count", -1))
        if self.move_count != self.graph_map.current_step:
            raise T14RouteCFreshError("Route C restored graph move counter changed")

    def runtime_diagnostics(self) -> dict[str, Any]:
        return {"move_count": self.move_count, **self.graph_map.runtime_diagnostics()}

    def close(self) -> None:
        # The state updater owns the shared graph/index files.
        pass


class _CandidateProxy(MutableMapping[str, Any]):
    def __init__(self, owner: "RouteCMMapCandidateState", record_id: int) -> None:
        self._owner = owner
        self.record_id = int(record_id)

    def __getitem__(self, key: str) -> Any:
        if key == "frequency":
            return int(self._owner.frequency[self.record_id])
        return self._owner._payload(self.record_id)[key]  # noqa: SLF001

    def __setitem__(self, key: str, value: Any) -> None:
        if key != "frequency" or type(value) is not int or value < 0:
            raise T14RouteCFreshError(
                "Route C candidate records permit only integer frequency mutation"
            )
        self._owner._assert_writer()  # noqa: SLF001
        self._owner.frequency[self.record_id] = value

    def __delitem__(self, key: str) -> None:
        raise T14RouteCFreshError("Route C candidate fields are append-only")

    def __iter__(self) -> Iterator[str]:
        return iter(("frequency", *self._owner._payload(self.record_id).keys()))  # noqa: SLF001

    def __len__(self) -> int:
        return 1 + len(self._owner._payload(self.record_id))  # noqa: SLF001


class RouteCMMapCandidateState(MutableSequence[MutableMapping[str, Any]]):
    """Official list interface over mmap frequency/order/metadata arrays."""

    _META_DTYPE = [
        ("graph_id", "<i8"),
        ("predecessor_id", "<i8"),
        ("downstream_id", "<i8"),
        ("sequence_id", "<i8"),
    ]

    def __init__(
        self,
        root: Path,
        *,
        capacity: int,
        record_capacity: int,
        graph_store: RouteCAppendOnlyGraphStore,
        next_sequence: Callable[[], int],
        resume: bool = False,
    ) -> None:
        import numpy as np

        if capacity <= 0 or record_capacity < capacity:
            raise T14RouteCFreshError(
                "Route C candidate active/record capacities are invalid"
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.record_capacity = int(record_capacity)
        self.graph_store = graph_store
        self._next_sequence = next_sequence
        self._writer_pid = os.getpid()
        self._writer_thread = threading.get_ident()
        self._size = 0
        self._next_record_id = 0
        self._payload_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._payload_cache_capacity = 256
        self.payload_path = self.root / "candidate_payloads.bin"
        self.payload_index_path = self.root / "candidate_payload_index.jsonl"
        self._payload_file = self.payload_path.open("a+b", buffering=0)
        self._payload_index = self.payload_index_path.open("a+b", buffering=0)
        mode = "r+" if resume else "w+"
        required = (
            self.root / "frequency.i64",
            self.root / "order.i64",
            self.root / "metadata.bin",
        )
        if resume and any(not path.is_file() for path in required):
            raise T14RouteCFreshError("Route C candidate mmap files are incomplete")
        self.frequency = np.memmap(
            required[0], mode=mode, dtype="<i8", shape=(record_capacity,)
        )
        self.order = np.memmap(
            required[1], mode=mode, dtype="<i8", shape=(capacity,)
        )
        self.metadata = np.memmap(
            required[2], mode=mode, dtype=self._META_DTYPE, shape=(record_capacity,)
        )
        if not resume:
            self.frequency[:] = 0
            self.order[:] = -1
            self.metadata["graph_id"][:] = -1
            self.metadata["predecessor_id"][:] = -1
            self.metadata["downstream_id"][:] = -1
            self.metadata["sequence_id"][:] = -1
        self._offsets: dict[int, tuple[int, int, str]] = {}
        self._active_record_by_graph_id: dict[int, int] = {}
        self._active_record_refcount: dict[int, int] = {}

    def _assert_writer(self) -> None:
        if os.getpid() != self._writer_pid or threading.get_ident() != self._writer_thread:
            raise T14RouteCFreshError("Route C candidate state has multiple updaters")

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _normalize_index(index: int, size: int) -> int:
        value = int(index)
        if value < 0:
            value += size
        if not 0 <= value < size:
            raise IndexError(index)
        return value

    def _record_at(self, index: int) -> int:
        return int(self.order[self._normalize_index(index, self._size)])

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._size))]
        return _CandidateProxy(self, self._record_at(index))

    def __setitem__(self, index: int | slice, value: Any) -> None:
        self._assert_writer()
        if isinstance(index, slice):
            raise T14RouteCFreshError("Route C candidate slice mutation is forbidden")
        position = self._normalize_index(index, self._size)
        if isinstance(value, _CandidateProxy) and value._owner is self:  # noqa: SLF001
            record_id = value.record_id
        elif isinstance(value, Mapping):
            record_id = self._append_record(value)
        else:
            raise T14RouteCFreshError("Route C candidate assignment is malformed")
        previous_record_id = int(self.order[position])
        self.order[position] = record_id
        previous_count = self._active_record_refcount.get(previous_record_id, 0) - 1
        if previous_count < 0:
            raise T14RouteCFreshError("Route C candidate active index is inconsistent")
        if previous_count:
            self._active_record_refcount[previous_record_id] = previous_count
        else:
            self._active_record_refcount.pop(previous_record_id, None)
            previous_graph_id = int(self.metadata[previous_record_id]["graph_id"])
            if self._active_record_by_graph_id.get(previous_graph_id) == previous_record_id:
                self._active_record_by_graph_id.pop(previous_graph_id, None)
        self._active_record_refcount[record_id] = (
            self._active_record_refcount.get(record_id, 0) + 1
        )
        graph_id = int(self.metadata[record_id]["graph_id"])
        self._active_record_by_graph_id[graph_id] = record_id

    def __delitem__(self, index: int | slice) -> None:
        raise T14RouteCFreshError("Route C candidate deletion is forbidden")

    def insert(self, index: int, value: MutableMapping[str, Any]) -> None:
        self._assert_writer()
        if index != self._size or self._size >= self.capacity:
            raise T14RouteCFreshError("Route C candidates support append only")
        record_id = self._append_record(value)
        self.order[self._size] = record_id
        graph_id = int(self.metadata[record_id]["graph_id"])
        self._active_record_by_graph_id[graph_id] = record_id
        self._active_record_refcount[record_id] = 1
        self._size += 1

    def _append_record(self, value: Mapping[str, Any]) -> int:
        self._assert_writer()
        required = {
            "frequency",
            "graph_hash",
            "importance_parts",
            "input_graphs_covering_list",
        }
        if set(value) != required or type(value.get("frequency")) is not int:
            raise T14RouteCFreshError("Route C candidate schema changed")
        record_id = self._next_record_id
        if record_id >= self.record_capacity:
            raise T14RouteCFreshError("Route C candidate record capacity exhausted")
        graph_hash = value["graph_hash"]
        graph_id = self.graph_store.graph_id(graph_hash)
        if graph_id is None:
            raise T14RouteCFreshError("Route C candidate graph is not in graph store")
        immutable = {key: item for key, item in value.items() if key != "frequency"}
        payload, payload_sha = _pickle_sha256(immutable)
        self._payload_file.seek(0, os.SEEK_END)
        offset = self._payload_file.tell()
        self._payload_file.write(payload)
        sequence = self._next_sequence()
        row = {
            "record_id": record_id,
            "offset": offset,
            "length": len(payload),
            "payload_sha256": payload_sha,
            "graph_id": graph_id,
            "sequence_id": sequence,
        }
        self._payload_index.write(canonical_bytes(row) + b"\n")
        self._offsets[record_id] = (offset, len(payload), payload_sha)
        self.frequency[record_id] = int(value["frequency"])
        self.metadata[record_id] = (graph_id, -1, graph_id, sequence)
        self._payload_cache[record_id] = immutable
        self._next_record_id += 1
        return record_id

    def _payload(self, record_id: int) -> dict[str, Any]:
        if record_id in self._payload_cache:
            value = self._payload_cache.pop(record_id)
            self._payload_cache[record_id] = value
            return value
        try:
            offset, length, expected = self._offsets[record_id]
        except KeyError as exc:
            raise T14RouteCFreshError("Route C candidate locator is absent") from exc
        self._payload_file.seek(offset)
        payload = self._payload_file.read(length)
        if len(payload) != length or hashlib.sha256(payload).hexdigest() != expected:
            raise T14RouteCFreshError("Route C candidate payload changed")
        value = pickle.loads(payload)
        if not isinstance(value, dict):
            raise T14RouteCFreshError("Route C candidate payload is malformed")
        self._payload_cache[record_id] = value
        while len(self._payload_cache) > self._payload_cache_capacity:
            self._payload_cache.popitem(last=False)
        return value

    def graph_hash_at(self, index: int) -> Any:
        return self._payload(self._record_at(index))["graph_hash"]

    def record_transition(self, *, source_hash: Any, target_hash: Any) -> None:
        target_id = self.graph_store.graph_id(target_hash)
        if target_id is None:
            return
        source_id = self.graph_store.graph_id(source_hash)
        record_id = self._active_record_by_graph_id.get(target_id)
        if record_id is not None:
            self.metadata[record_id]["predecessor_id"] = (
                -1 if source_id is None else source_id
            )
            self.metadata[record_id]["downstream_id"] = target_id

    def flush(self) -> None:
        self._payload_file.flush()
        self._payload_index.flush()
        os.fsync(self._payload_file.fileno())
        os.fsync(self._payload_index.fileno())
        self.frequency.flush()
        self.order.flush()
        self.metadata.flush()
        _fsync_dir(self.root)

    def checkpoint_state(self) -> dict[str, Any]:
        import numpy as np

        self.flush()
        record_count = self._next_record_id
        return {
            "schema_version": CANDIDATE_STATE_SCHEMA,
            "capacity": self.capacity,
            "record_capacity": self.record_capacity,
            "size": self._size,
            "record_count": record_count,
            "frequency": np.asarray(self.frequency[:record_count]).copy(),
            "order": np.asarray(self.order[: self._size]).copy(),
            "metadata": np.asarray(self.metadata[:record_count]).copy(),
            "payload_bytes": self.payload_path.stat().st_size,
            "payload_index_bytes": self.payload_index_path.stat().st_size,
            "full_python_candidate_list_saved": False,
            "mmap_frequency": True,
            "mmap_metadata": True,
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        import numpy as np

        if (
            state.get("schema_version") != CANDIDATE_STATE_SCHEMA
            or int(state.get("capacity", -1)) != self.capacity
            or int(state.get("record_capacity", -1)) != self.record_capacity
            or state.get("full_python_candidate_list_saved") is not False
            or state.get("mmap_frequency") is not True
            or state.get("mmap_metadata") is not True
        ):
            raise T14RouteCFreshError("Route C candidate checkpoint contract changed")
        record_count = int(state.get("record_count", -1))
        size = int(state.get("size", -1))
        if not 0 <= size <= self.capacity or not size <= record_count <= self.record_capacity:
            raise T14RouteCFreshError("Route C candidate checkpoint counts are invalid")
        frequency = np.asarray(state["frequency"], dtype="<i8")
        order = np.asarray(state["order"], dtype="<i8")
        metadata = np.asarray(state["metadata"], dtype=self._META_DTYPE)
        if (
            frequency.shape != (record_count,)
            or order.shape != (size,)
            or metadata.shape != (record_count,)
        ):
            raise T14RouteCFreshError("Route C candidate checkpoint arrays changed")
        if (
            self.payload_path.stat().st_size != int(state["payload_bytes"])
            or self.payload_index_path.stat().st_size
            != int(state["payload_index_bytes"])
        ):
            raise T14RouteCFreshError(
                "Route C candidate files differ from the clean checkpoint boundary"
            )
        self.frequency[:record_count] = frequency
        self.order[:size] = order
        self.metadata[:record_count] = metadata
        self._size = size
        self._next_record_id = record_count
        self._rebuild_offsets(limit_bytes=int(state["payload_index_bytes"]))
        self._active_record_by_graph_id = {}
        self._active_record_refcount = {}
        for position in range(size):
            record_id = int(self.order[position])
            if not 0 <= record_id < record_count:
                raise T14RouteCFreshError("Route C candidate active record is invalid")
            graph_id = int(self.metadata[record_id]["graph_id"])
            existing = self._active_record_by_graph_id.get(graph_id)
            if existing is not None and existing != record_id:
                raise T14RouteCFreshError("Route C candidate graph is active twice")
            self._active_record_by_graph_id[graph_id] = record_id
            self._active_record_refcount[record_id] = (
                self._active_record_refcount.get(record_id, 0) + 1
            )
        self.flush()

    def _rebuild_offsets(self, *, limit_bytes: int) -> None:
        self._offsets.clear()
        self._payload_index.seek(0)
        consumed = 0
        while consumed < limit_bytes:
            line = self._payload_index.readline()
            if not line:
                break
            consumed += len(line)
            row = json.loads(line)
            self._offsets[int(row["record_id"])] = (
                int(row["offset"]),
                int(row["length"]),
                str(row["payload_sha256"]),
            )
        if consumed != limit_bytes or len(self._offsets) < self._next_record_id:
            raise T14RouteCFreshError("Route C candidate index prefix changed")

    def close(self) -> None:
        self.flush()
        self._payload_file.close()
        self._payload_index.close()


class RouteCStateUpdater:
    """Single owner of Route C sequence IDs and disk-backed state mutation."""

    def __init__(
        self,
        root: Path,
        *,
        candidate_capacity: int,
        record_capacity: int = 200_000,
        lru_capacity: int = 128,
        resume: bool = False,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._writer_pid = os.getpid()
        self._writer_thread = threading.get_ident()
        self.graph_store = RouteCAppendOnlyGraphStore(
            self.root / "graph_store", lru_capacity=lru_capacity
        )
        self.candidates = RouteCMMapCandidateState(
            self.root / "candidate_state",
            capacity=candidate_capacity,
            record_capacity=record_capacity,
            graph_store=self.graph_store,
            next_sequence=self.next_sequence,
            resume=resume,
        )

    def next_sequence(self) -> int:
        if os.getpid() != self._writer_pid or threading.get_ident() != self._writer_thread:
            raise T14RouteCFreshError("Route C has more than one state updater")
        self._sequence += 1
        return self._sequence

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": "tastemolnet_t14_route_c_state_updater_v1",
            "sequence_id": self._sequence,
            "graph_store": self.graph_store.checkpoint_state(),
            "candidates": self.candidates.checkpoint_state(),
            "single_scientific_state_updater": True,
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema_version")
            != "tastemolnet_t14_route_c_state_updater_v1"
            or state.get("single_scientific_state_updater") is not True
        ):
            raise T14RouteCFreshError("Route C state-updater checkpoint changed")
        self.graph_store.validate_checkpoint_state(state["graph_store"])
        self.candidates.restore_checkpoint_state(state["candidates"])
        self._sequence = int(state.get("sequence_id", -1))
        if self._sequence < 0:
            raise T14RouteCFreshError("Route C sequence checkpoint is invalid")

    def close(self) -> None:
        self.candidates.close()
        self.graph_store.close()


def checkpoint_targets(
    *, completed_step: int, stop_step: int, route_c: bool
) -> tuple[int, ...]:
    """Return the exact T14 checkpoint cadence for one invocation."""

    if not route_c:
        return tuple(
            range(((completed_step // 2_500) + 1) * 2_500, stop_step + 1, 2_500)
        )
    allowed = (*EARLY_CHECKPOINT_STEPS, *PRODUCTION_CHECKPOINT_STEPS)
    return tuple(step for step in allowed if completed_step < step <= stop_step)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise T14RouteCFreshError(
                    f"Route C step ledger line {line_number} is invalid"
                ) from exc
            if not isinstance(value, dict):
                raise T14RouteCFreshError("Route C step ledger row is not an object")
            rows.append(value)
    return rows


def _semantic_value(value: Any, *, field: str) -> Any:
    """Return a deterministic typed representation for parity-only science state."""

    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise T14RouteCFreshError(f"Route C parity field is non-finite: {field}")
        return {"type": "float", "value": value.hex()}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {"type": "bytes", "size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        rows = [
            {
                "key": _semantic_value(key, field=f"{field}.key"),
                "value": _semantic_value(item, field=f"{field}[{key!r}]"),
            }
            for key, item in value.items()
        ]
        rows.sort(key=lambda row: canonical_bytes(row["key"]))
        return {"type": "mapping", "items": rows}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _semantic_value(item, field=f"{field}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _semantic_value(item, field=f"{field}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (set, frozenset)):
        items = [_semantic_value(item, field=f"{field}.item") for item in value]
        items.sort(key=canonical_bytes)
        return {"type": "set", "items": items}
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if getattr(value, "is_sparse", False):
            value = value.coalesce()
            return {
                "type": "torch_sparse_coo",
                "shape": list(value.shape),
                "indices": _semantic_value(
                    value.indices(), field=f"{field}.indices"
                ),
                "values": _semantic_value(value.values(), field=f"{field}.values"),
            }
    if hasattr(value, "numpy") and hasattr(value, "shape"):
        value = value.numpy()
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        import numpy as np

        array = np.ascontiguousarray(value)
        if array.dtype.hasobject or (
            array.dtype.kind in {"f", "c"} and not np.isfinite(array).all()
        ):
            raise T14RouteCFreshError(f"Route C parity array is invalid: {field}")
        raw = array.tobytes(order="C")
        return {
            "type": "array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return {
            "type": "object_to_dict",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _semantic_value(to_dict(), field=f"{field}.to_dict"),
        }
    raise T14RouteCFreshError(
        f"Route C parity field has unsupported type: {field}={type(value)!r}"
    )


def _semantic_sha(value: Any, *, field: str) -> str:
    return stable_sha256(_semantic_value(value, field=field))


def scientific_state_digest(
    *,
    module: Any,
    bridge: Any,
    loop_state: Any,
    selected: Mapping[str, Any],
    transition_map: Any | None = None,
    live_graph_state: Any | None = None,
) -> dict[str, Any]:
    """Create one complete storage-independent semantic digest after a move."""

    candidates = [dict(row) for row in module.counterfactual_candidates]
    candidate_payload = _semantic_value(candidates, field="candidates")
    candidate_digest = stable_sha256(candidate_payload)
    lineages = {
        str(key): dict(sorted(value.items()))
        for key, value in sorted(bridge.lineage_occurrences.items())
    }
    records = {
        str(key): {
            "graph_identity_sha256": value.graph_identity_sha256,
            "canonical_graph": value.canonical_graph,
            "probabilities": [float(item).hex() for item in value.probabilities],
            "prediction": int(value.prediction),
            "score": float(value.score).hex(),
            "candidate": bool(value.candidate),
            "valid_fullgraph": bool(value.valid_fullgraph),
            "model_graph_sha256": value.model_graph_sha256,
            "model_graph_payload_sha256": _semantic_sha(
                value.model_graph_payload, field=f"bridge.records[{key}].model_graph"
            ),
            "embedding_sha256": value.embedding_sha256,
            "embedding_dtype": value.embedding_dtype,
        }
        for key, value in sorted(bridge.records.items())
    }
    collision_payloads = {
        str(key): _semantic_sha(value, field=f"bridge.collisions[{key}]")
        for key, value in sorted(bridge.graph_collision_payloads.items())
    }
    module_state = {
        "graph_index_map": dict(module.graph_index_map),
        "input_graphs_covered": module.input_graphs_covered,
        "covering_graphs": set(module.covering_graphs),
        "start": dict(module.start),
        "is_sample": bool(module.is_sample),
        "starting_step": int(module.starting_step),
        "traversed_hashes": list(module.traversed_hashes),
        "sample_size": int(module.sample_size),
        "max_counterfactual_size": int(module.MAX_COUNTERFACTUAL_SIZE),
    }
    active_graph_rows = []
    graph_map = getattr(module, "graph_map", {})
    for key in graph_map:
        value = graph_map[key]
        try:
            from src.baselines.comrecgc.live_graph_state import _entry_graph_sha256

            payload_sha = _entry_graph_sha256(value)
        except Exception as exc:
            raise T14RouteCFreshError(
                f"Route C active graph payload is invalid: {key!r}"
            ) from exc
        record = bridge.records.get(str(key))
        active_graph_rows.append(
            {
                "identity": _semantic_value(key, field="active_graph.identity"),
                "payload_sha256": payload_sha,
                "bridge_record_sha256": (
                    None
                    if record is None
                    else _semantic_sha(
                        {
                            "collision": bridge.graph_collision_payloads.get(str(key)),
                            "model_graph_sha256": record.model_graph_sha256,
                            "embedding_sha256": record.embedding_sha256,
                            "probabilities": tuple(record.probabilities),
                        },
                        field=f"active_graph[{key!r}].bridge_record",
                    )
                ),
            }
        )
    active_graph_rows.sort(key=lambda row: canonical_bytes(row["identity"]))
    transitions = transition_map if transition_map is not None else module.transitions
    transition_rows = []
    raw_entries = getattr(transitions, "_entries", {})
    for source_hash, entry in raw_entries.items():
        transition_rows.append(
            {
                "source_hash": _semantic_value(source_hash, field="transition.source"),
                "target_hashes": _semantic_value(
                    tuple(entry.target_hashes), field="transition.targets"
                ),
                "actions": _semantic_value(tuple(entry.actions), field="transition.actions"),
                "importance_parts": _semantic_value(
                    entry.importance_parts, field="transition.importance_parts"
                ),
                "embeddings": _semantic_value(
                    entry.embeddings, field="transition.embeddings"
                ),
            }
        )
    transition_rows.sort(key=lambda row: canonical_bytes(row["source_hash"]))
    transition_counters = {
        name: int(getattr(transitions, name, 0))
        for name in (
            "move_count",
            "deferred_deletion_count",
            "applied_deferred_deletion_count",
            "cancelled_deferred_deletion_count",
            "missing_lookup_count",
            "captured_action_count",
        )
    }
    graph_diagnostics = (
        live_graph_state.runtime_diagnostics()
        if live_graph_state is not None
        else {}
    )
    # Cache/rehydration counters intentionally differ between reference and
    # low-memory storage.  Only scientific collision/eviction state is bound.
    eviction_state = {
        "candidate_record_count": int(
            getattr(getattr(module, "counterfactual_candidates", None), "_next_record_id", len(candidates))
        ),
        "candidate_active_count": len(candidates),
        "missing_unmaterialized_eviction_count": int(
            graph_diagnostics.get("missing_unmaterialized_eviction_count", 0)
        ),
        "eviction_attempts": int(graph_diagnostics.get("eviction_attempts", 0)),
        "eviction_committed": int(graph_diagnostics.get("eviction_committed", 0)),
        "eviction_deferred": int(graph_diagnostics.get("eviction_deferred", 0)),
        "active_eviction_prevented": int(
            graph_diagnostics.get("active_eviction_prevented", 0)
        ),
        "deferred_flushed": int(graph_diagnostics.get("deferred_flushed", 0)),
        "deferred_deletions": int(graph_diagnostics.get("deferred_deletions", 0)),
        "recent_evictions": graph_diagnostics.get("recent_evictions", []),
        "transition_applied_deferred_deletion_count": transition_counters[
            "applied_deferred_deletion_count"
        ],
        "transition_cancelled_deferred_deletion_count": transition_counters[
            "cancelled_deferred_deletion_count"
        ],
    }
    from src.baselines.comrecgc.generation_checkpoint import capture_rng_state

    rng_payload, rng_sha = _pickle_sha256(capture_rng_state())
    del rng_payload
    return {
        "schema_version": STEP_LEDGER_SCHEMA,
        "sequence_id": int(loop_state.completed_step),
        "completed_step": int(loop_state.completed_step),
        "rng_state_sha256": rng_sha,
        "start_graph_hashes": [str(value) for value in loop_state.start_graph_hashes],
        "current_graph_hashes": [str(value) for value in loop_state.current_graph_hashes],
        "restart_indices": list(loop_state.restart_indices),
        "selected": _semantic_value(dict(selected), field="selected_transition"),
        "candidate_order_frequency_sha256": candidate_digest,
        "candidate_records_sha256": candidate_digest,
        "candidate_universe_sha256": stable_sha256(
            sorted(str(row["graph_hash"]) for row in candidates)
        ),
        "candidate_covering_lists_sha256": _semantic_sha(
            [row["input_graphs_covering_list"] for row in candidates],
            field="candidate_covering_lists",
        ),
        "module_scientific_state_sha256": _semantic_sha(
            module_state, field="module_scientific_state"
        ),
        "lineage_sha256": stable_sha256(lineages),
        "record_semantics_sha256": stable_sha256(records),
        "collision_state_sha256": stable_sha256(collision_payloads),
        "bridge_counters": {
            "call_count": int(bridge.call_count),
            "evaluated_graph_count": int(bridge.evaluated_graph_count),
            "calculate_hash_count": int(bridge.calculate_hash_count),
            "pending_hash_count": int(bridge.pending_hash_count),
        },
        "active_graph_state_sha256": stable_sha256(active_graph_rows),
        "transition_state_sha256": stable_sha256(transition_rows),
        "transition_counters": transition_counters,
        "lineage_collision_eviction_sha256": _semantic_sha(
            {
                "lineages": lineages,
                "collisions": collision_payloads,
                "evictions": eviction_state,
            },
            field="lineage_collision_eviction",
        ),
        "graph_index_sha256": _semantic_sha(
            dict(module.graph_index_map), field="graph_index_map"
        ),
        "test_loaded": False,
        "calibration_loaded": False,
    }


def append_step_state(path: Path, row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != STEP_LEDGER_SCHEMA:
        raise T14RouteCFreshError("Route C step-state schema changed")
    existing = _jsonl_rows(path)
    step = int(row["completed_step"])
    if step <= len(existing):
        if existing[step - 1] != dict(row):
            raise T14RouteCFreshError(
                f"Route C deterministic replay diverged at step {step}"
            )
        return
    if step != len(existing) + 1:
        raise T14RouteCFreshError("Route C step ledger is not contiguous")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        handle.write(canonical_bytes(dict(row)) + b"\n")
        os.fsync(handle.fileno())


def compare_step_ledgers(
    reference: Path,
    candidate: Path,
    *,
    start_step: int = 1,
    end_step: int = PROMOTABLE_CHECKPOINT_STEP,
) -> dict[str, Any]:
    left = _jsonl_rows(reference)
    right = _jsonl_rows(candidate)
    if start_step <= 0 or end_step < start_step:
        raise T14RouteCFreshError("Route C parity interval is invalid")
    first = None
    differing_fields: list[str] = []
    for step in range(start_step, end_step + 1):
        if step > len(left) or step > len(right):
            first = step
            differing_fields = ["missing_step"]
            break
        if left[step - 1] != right[step - 1]:
            first = step
            differing_fields = sorted(
                key
                for key in set(left[step - 1]) | set(right[step - 1])
                if left[step - 1].get(key) != right[step - 1].get(key)
            )
            break
    receipt = {
        "schema_version": PARITY_SCHEMA,
        "status": "PASS" if first is None else "FAILED",
        "reference": str(reference),
        "candidate": str(candidate),
        "start_step": start_step,
        "end_step": end_step,
        "first_semantic_divergence_step": first,
        "differing_fields": differing_fields,
        "discrete_state_exact": first is None,
        "reference_sha256": file_sha256(reference),
        "candidate_sha256": file_sha256(candidate),
    }
    receipt["receipt_sha256"] = stable_sha256(receipt)
    return receipt


def _route_c_checkpoint_summary(
    *, spec: Mapping[str, Any], step: int
) -> dict[str, Any]:
    """Project one sealed Route C checkpoint into the frozen convergence schema."""

    import math
    import numpy as np

    value = validate_spec(spec, check_files=False)
    identity = _checkpoint_identity(Path(value["output_root"]))
    checkpoint_dir = Path(value["output_root"]) / "checkpoints" / f"step-{step:012d}"
    from src.baselines.comrecgc.generation_checkpoint import load_generation_checkpoint

    loaded = load_generation_checkpoint(
        checkpoint_dir,
        expected_provenance=identity["provenance"],
        expected_scientific_argv=identity["scientific_argv"],
        expected_command_sha256=identity["command_sha256"],
        expected_total_steps=M_FALLBACK_MAX,
        expected_completed_step=step,
        single_pass=True,
    )
    algorithm = loaded.algorithm_state
    if algorithm.get("schema_version") != "tastemolnet_t14_route_c_runtime_v1":
        raise T14RouteCFreshError("Route C convergence runtime schema changed")
    route_state = algorithm.get("route_c_state")
    official = algorithm.get("official_state")
    bridge = algorithm.get("bridge_state")
    if not all(isinstance(item, Mapping) for item in (route_state, official, bridge)):
        raise T14RouteCFreshError("Route C convergence state is incomplete")
    candidates = route_state.get("candidates")
    if (
        not isinstance(candidates, Mapping)
        or candidates.get("schema_version") != CANDIDATE_STATE_SCHEMA
        or bridge.get("schema_version")
        not in {
            "tastemolnet_comrecgc_bridge_checkpoint_v3",
            "tastemolnet_comrecgc_bridge_checkpoint_v4",
        }
    ):
        raise T14RouteCFreshError("Route C convergence checkpoint contract changed")
    size = int(candidates.get("size", -1))
    record_count = int(candidates.get("record_count", -1))
    order = np.asarray(candidates.get("order"), dtype="<i8")
    frequency = np.asarray(candidates.get("frequency"), dtype="<i8")
    metadata = np.asarray(
        candidates.get("metadata"), dtype=RouteCMMapCandidateState._META_DTYPE
    )
    if (
        not 0 <= size <= record_count
        or order.shape != (size,)
        or frequency.shape != (record_count,)
        or metadata.shape != (record_count,)
    ):
        raise T14RouteCFreshError("Route C convergence candidate arrays changed")
    connection = sqlite3.connect(
        f"{loaded.sqlite_snapshot_path.resolve().as_uri()}?mode=ro&immutable=1",
        uri=True,
        timeout=60.0,
    )
    try:
        if str(connection.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise T14RouteCFreshError("Route C convergence SQLite integrity failed")
        graph_keys = {
            int(graph_id): pickle.loads(bytes(key_blob))
            for graph_id, key_blob in connection.execute(
                "SELECT graph_id,key_blob FROM graphs"
            ).fetchall()
        }
    finally:
        connection.close()
    frequencies: dict[str, int] = {}
    for position in range(size):
        record_id = int(order[position])
        if not 0 <= record_id < record_count:
            raise T14RouteCFreshError("Route C convergence candidate order is invalid")
        graph_id = int(metadata[record_id]["graph_id"])
        graph_hash = graph_keys.get(graph_id)
        if type(graph_hash) is not str or _SHA256.fullmatch(graph_hash) is None:
            raise T14RouteCFreshError("Route C convergence graph identity is invalid")
        count = int(frequency[record_id])
        if count <= 0 or graph_hash in frequencies:
            raise T14RouteCFreshError("Route C convergence candidate frequency is invalid")
        frequencies[graph_hash] = count

    records = bridge.get("records")
    collisions = bridge.get("graph_collision_payloads")
    lineages = bridge.get("lineage_occurrences")
    if not all(isinstance(item, Mapping) for item in (records, collisions, lineages)):
        raise T14RouteCFreshError("Route C convergence lineage state is incomplete")
    from src.baselines.tastemolnet_comrecgc_smoke import _identity_graph_sha256

    valid: set[str] = set()
    lineage_errors: list[dict[str, Any]] = []
    for raw_key, record in records.items():
        key = str(raw_key)
        failures: list[str] = []
        collision = collisions.get(key)
        lineage = lineages.get(key)
        try:
            collision_matches = (
                isinstance(collision, dict) and _identity_graph_sha256(collision) == key
            )
        except Exception:
            collision_matches = False
        if _SHA256.fullmatch(key) is None or not isinstance(record, Mapping):
            failures.append("record_identity")
        else:
            if record.get("graph_identity_sha256") != key:
                failures.append("record_graph_identity")
            if not collision_matches:
                failures.append("collision_payload")
            if (
                not isinstance(lineage, Mapping)
                or not lineage
                or any(
                    _SHA256.fullmatch(str(parent)) is None
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                    for parent, count in lineage.items()
                )
            ):
                failures.append("lineage")
            if (
                record.get("valid_fullgraph") is True
                and record.get("candidate") is True
                and record.get("prediction") in (0, 2)
                and isinstance(record.get("canonical_graph"), str)
                and record.get("canonical_graph")
                and not failures
            ):
                valid.add(key)
        if failures:
            lineage_errors.append({"graph_identity_sha256": key, "failures": failures})
    coverage = official.get("input_graphs_covered")
    if hasattr(coverage, "detach"):
        coverage = coverage.detach().cpu()
    if hasattr(coverage, "tolist"):
        coverage = coverage.tolist()
    if not isinstance(coverage, list) or not coverage:
        raise T14RouteCFreshError("Route C convergence train coverage is absent")
    coverage_values = [float(item) for item in coverage]
    if any(not math.isfinite(item) or item < 0 for item in coverage_values):
        raise T14RouteCFreshError("Route C convergence train coverage is invalid")
    ranking = [
        key for key, _count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    ]
    valid_ranking = [key for key in ranking if key in valid]
    return {
        "schema_version": "tastemolnet_t14_checkpoint_summary_v1",
        "step": step,
        "checkpoint_root": str(checkpoint_dir),
        "checkpoint_digest": loaded.validation.checkpoint_digest,
        "candidate_frequency": dict(sorted(frequencies.items())),
        "top100_candidate_hashes": ranking[:100],
        "top20_provisional_rule_hashes": valid_ranking[:20],
        "valid_unique_rule_count": len(valid),
        "valid_unique_rule_hashes": sorted(valid),
        "lineage_error_count": len(lineage_errors),
        "lineage_errors": lineage_errors,
        "train_coverage": sum(item > 0 for item in coverage_values) / len(coverage_values),
        "train_parent_count": len(coverage_values),
        "split": "train",
        "calibration_loaded": False,
        "test_loaded": False,
        "sqlite_accessed": True,
        "sqlite_access_reason": "ROUTE_C_CANDIDATE_GRAPH_ID_BINDING",
    }


def audit_route_c_convergence(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the existing frozen T14 train-side gate to Route C checkpoints."""

    from dataclasses import asdict
    from src.eval.tastemolnet_t14_external_convergence import POLICY, compare_summaries

    value = validate_spec(spec, check_files=False)
    checkpoint_root = Path(value["output_root"]) / "checkpoints"
    required = (5_000, 10_000, 12_500)
    available = [
        step
        for step in PRODUCTION_CHECKPOINT_STEPS
        if step >= 5_000 and (checkpoint_root / f"step-{step:012d}").is_dir()
    ]
    if not set(required).issubset(available):
        return {
            "schema_version": "tastemolnet_t14_route_c_convergence_audit_v1",
            "status": "WAITING_FOR_12500",
            "policy": asdict(POLICY),
            "available_steps": available,
            "required_initial_steps": list(required),
            "checkpoint_state_loaded": False,
            "converged": False,
            "safe_stop_authorized": False,
            "calibration_loaded": False,
            "test_loaded": False,
        }
    summaries = [
        _route_c_checkpoint_summary(spec=value, step=step) for step in available
    ]
    windows = [
        compare_summaries(before, after)
        for before, after in zip(summaries, summaries[1:])
    ]
    consecutive = 0
    for window in windows:
        consecutive = consecutive + 1 if window["pass"] else 0
    converged = consecutive >= POLICY.required_consecutive_windows
    return {
        "schema_version": "tastemolnet_t14_route_c_convergence_audit_v1",
        "status": "CONVERGED_EARLY_STOP" if converged else "CONTINUE_T14",
        "policy": asdict(POLICY),
        "available_steps": available,
        "checkpoint_summaries": summaries,
        "windows": windows,
        "consecutive_passing_windows": consecutive,
        "checkpoint_state_loaded": True,
        "sqlite_accessed": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "converged": converged,
        "safe_stop_authorized": converged,
    }


def write_route_c_convergence_receipt(
    path: Path, *, spec: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    value = validate_spec(spec, check_files=False)
    summaries = audit.get("checkpoint_summaries")
    if (
        audit.get("schema_version")
        != "tastemolnet_t14_route_c_convergence_audit_v1"
        or audit.get("converged") is not True
        or audit.get("safe_stop_authorized") is not True
        or int(audit.get("consecutive_passing_windows", 0)) < 2
        or not isinstance(summaries, list)
        or not summaries
    ):
        raise T14RouteCFreshError("Route C convergence receipt requires a PASS audit")
    latest = summaries[-1]
    step = int(latest.get("step", -1))
    if step not in PRODUCTION_CHECKPOINT_STEPS or not 10_000 <= step < M_MAX:
        raise T14RouteCFreshError("Route C convergence stop step is invalid")
    receipt = {
        "schema_version": CONVERGENCE_RECEIPT_SCHEMA,
        "status": "PASS",
        "spec_sha256": value["spec_sha256"],
        "output_root": value["output_root"],
        "m_effective": step,
        "checkpoint_digest": latest["checkpoint_digest"],
        "audit_sha256": stable_sha256(dict(audit)),
        "policy": audit["policy"],
        "consecutive_passing_windows": int(audit["consecutive_passing_windows"]),
        "stop_reason": "TRAIN_SIDE_CONVERGENCE_TWO_CONSECUTIVE_WINDOWS",
        "calibration_loaded": False,
        "test_loaded": False,
        "written_at": _utc_now(),
    }
    receipt["receipt_sha256"] = stable_sha256(receipt)
    atomic_json(path, receipt)
    return validate_route_c_convergence_receipt(
        path,
        spec=value,
        expected_step=step,
        expected_checkpoint_digest=str(latest["checkpoint_digest"]),
    )


def validate_route_c_convergence_receipt(
    path: Path,
    *,
    spec: Mapping[str, Any],
    expected_step: int,
    expected_checkpoint_digest: str,
) -> dict[str, Any]:
    value = validate_spec(spec, check_files=False)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise T14RouteCFreshError("Route C convergence receipt is absent or indirect")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    from dataclasses import asdict
    from src.eval.tastemolnet_t14_external_convergence import POLICY

    if (
        receipt.get("schema_version") != CONVERGENCE_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("spec_sha256") != value["spec_sha256"]
        or receipt.get("output_root") != value["output_root"]
        or receipt.get("m_effective") != expected_step
        or receipt.get("checkpoint_digest") != expected_checkpoint_digest
        or receipt.get("policy") != asdict(POLICY)
        or int(receipt.get("consecutive_passing_windows", 0)) < 2
        or receipt.get("stop_reason")
        != "TRAIN_SIDE_CONVERGENCE_TWO_CONSECUTIVE_WINDOWS"
        or receipt.get("calibration_loaded") is not False
        or receipt.get("test_loaded") is not False
        or receipt.get("receipt_sha256") != stable_sha256(unsigned)
    ):
        raise T14RouteCFreshError("Route C convergence receipt changed")
    return receipt


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_physical_file(path: Path, *, field: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise T14RouteCFreshError(f"Route C {field} is absent or indirect")
    return path


def _bytes_sha256(value: Any) -> str:
    try:
        return hashlib.sha256(memoryview(value).cast("B")).hexdigest()
    except (TypeError, ValueError):
        return hashlib.sha256(bytes(value)).hexdigest()


def recover_route_c_external_state(
    *,
    output_root: Path,
    loaded: Any,
    promotion_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore append-only Route C files to one promoted checkpoint boundary.

    A process may die after appending a suffix but before publishing its next
    checkpoint.  The checkpoint payload cannot be applied until that suffix is
    removed.  This routine first validates the promoted receipt, checkpoint
    digest, sealed SQLite index, and every immutable blob in the committed
    prefix.  Only then does it atomically record an intent and restore/truncate
    the external files.  Repeating recovery for the same checkpoint is
    idempotent.
    """

    import numpy as np

    root = Path(output_root)
    if not root.is_absolute() or root.is_symlink():
        raise T14RouteCFreshError("Route C recovery root must be physical and absolute")
    validation = getattr(loaded, "validation", None)
    algorithm_state = getattr(loaded, "algorithm_state", None)
    if validation is None or not isinstance(algorithm_state, Mapping):
        raise T14RouteCFreshError("Route C recovery lacks a validated checkpoint")
    completed = int(getattr(validation, "completed_step", -1))
    checkpoint_digest = str(getattr(validation, "checkpoint_digest", ""))
    unsigned_promotion = {
        key: value for key, value in promotion_receipt.items() if key != "receipt_sha256"
    }
    if (
        promotion_receipt.get("schema_version") != PROMOTION_SCHEMA
        or promotion_receipt.get("status") != "PASS"
        or promotion_receipt.get("output_root") != str(root)
        or promotion_receipt.get("completed_step") != completed
        or promotion_receipt.get("checkpoint_digest") != checkpoint_digest
        or promotion_receipt.get("payload_reload_pass") is not True
        or promotion_receipt.get("latest_promoted") is not True
        or promotion_receipt.get("receipt_sha256") != stable_sha256(unsigned_promotion)
    ):
        raise T14RouteCFreshError(
            "Route C recovery refused an unbound promotion receipt/checkpoint"
        )
    if algorithm_state.get("schema_version") != "tastemolnet_t14_route_c_runtime_v1":
        raise T14RouteCFreshError("Route C recovery checkpoint schema changed")
    updater_state = algorithm_state.get("route_c_state")
    live_state = algorithm_state.get("live_graph_state")
    if (
        not isinstance(updater_state, Mapping)
        or updater_state.get("schema_version")
        != "tastemolnet_t14_route_c_state_updater_v1"
        or not isinstance(live_state, Mapping)
    ):
        raise T14RouteCFreshError("Route C external checkpoint state is incomplete")
    graph_state = updater_state.get("graph_store")
    candidate_state = updater_state.get("candidates")
    if not isinstance(graph_state, Mapping) or not isinstance(candidate_state, Mapping):
        raise T14RouteCFreshError("Route C external checkpoint payload is malformed")
    if live_state.get("store") != graph_state:
        raise T14RouteCFreshError("Route C graph-store checkpoint copies diverged")

    route_root = root / "route_c_state"
    graph_root = route_root / "graph_store"
    candidate_root = route_root / "candidate_state"
    graph_blob = _regular_physical_file(graph_root / "graphs.bin", field="graph blob")
    live_index = graph_root / "graph_index.sqlite3"
    if live_index.is_symlink():
        raise T14RouteCFreshError("Route C live graph index is indirect")
    candidate_blob = _regular_physical_file(
        candidate_root / "candidate_payloads.bin", field="candidate blob"
    )
    candidate_index = _regular_physical_file(
        candidate_root / "candidate_payload_index.jsonl", field="candidate index"
    )
    frequency_path = _regular_physical_file(
        candidate_root / "frequency.i64", field="candidate frequency mmap"
    )
    order_path = _regular_physical_file(
        candidate_root / "order.i64", field="candidate order mmap"
    )
    metadata_path = _regular_physical_file(
        candidate_root / "metadata.bin", field="candidate metadata mmap"
    )
    snapshot = _regular_physical_file(
        Path(loaded.sqlite_snapshot_path), field="sealed checkpoint SQLite snapshot"
    )

    graph_bytes = int(graph_state.get("data_bytes", -1))
    graph_count = int(graph_state.get("entry_count", -1))
    candidate_bytes = int(candidate_state.get("payload_bytes", -1))
    candidate_index_bytes = int(candidate_state.get("payload_index_bytes", -1))
    candidate_count = int(candidate_state.get("record_count", -1))
    candidate_size = int(candidate_state.get("size", -1))
    if min(
        graph_bytes,
        graph_count,
        candidate_bytes,
        candidate_index_bytes,
        candidate_count,
        candidate_size,
    ) < 0:
        raise T14RouteCFreshError("Route C recovery boundary sizes are invalid")
    if (
        graph_blob.stat().st_size < graph_bytes
        or candidate_blob.stat().st_size < candidate_bytes
        or candidate_index.stat().st_size < candidate_index_bytes
    ):
        raise T14RouteCFreshError("Route C committed external prefix was truncated")

    # Validate the graph blob against the independently sealed SQLite snapshot,
    # not against the possibly suffix-bearing live index.
    connection = sqlite3.connect(
        f"{snapshot.resolve().as_uri()}?mode=ro&immutable=1", uri=True, timeout=60.0
    )
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        rows = connection.execute(
            "SELECT graph_id,key_blob,key_sha256,offset,length,payload_sha256,"
            "graph_sha256,sequence_id,chain_sha256 "
            "FROM graphs ORDER BY sequence_id,graph_id"
        ).fetchall()
    finally:
        connection.close()
    if integrity != "ok" or len(rows) != graph_count:
        raise T14RouteCFreshError("Route C sealed graph index failed recovery validation")
    previous_chain = "0" * 64
    expected_offset = 0
    with graph_blob.open("rb") as stream:
        for row in rows:
            graph_id, key_blob, key_sha, offset, length, payload_sha, graph_sha, _, chain = row
            if int(offset) != expected_offset or int(length) < 0:
                raise T14RouteCFreshError("Route C graph prefix is not contiguous")
            key_blob = bytes(key_blob)
            if (
                hashlib.sha256(key_blob).hexdigest() != str(key_sha)
                or _stable_numeric_graph_id(pickle.loads(key_blob)) != int(graph_id)
            ):
                raise T14RouteCFreshError("Route C graph key identity changed")
            stream.seek(int(offset))
            payload = stream.read(int(length))
            if len(payload) != int(length) or hashlib.sha256(payload).hexdigest() != str(
                payload_sha
            ):
                raise T14RouteCFreshError("Route C committed graph payload changed")
            expected_chain = hashlib.sha256(
                bytes.fromhex(previous_chain)
                + bytes.fromhex(str(key_sha))
                + bytes.fromhex(str(payload_sha))
                + bytes.fromhex(str(graph_sha))
            ).hexdigest()
            if str(chain) != expected_chain:
                raise T14RouteCFreshError("Route C graph prefix chain changed")
            previous_chain = expected_chain
            expected_offset += int(length)
    if (
        expected_offset != graph_bytes
        or previous_chain != graph_state.get("chain_sha256")
    ):
        raise T14RouteCFreshError("Route C graph prefix boundary changed")

    # Validate every candidate locator and immutable payload in the committed
    # index prefix before permitting truncation of an uncommitted suffix.
    with candidate_index.open("rb") as stream:
        index_prefix = stream.read(candidate_index_bytes)
    if candidate_index_bytes and not index_prefix.endswith(b"\n"):
        raise T14RouteCFreshError("Route C candidate index boundary splits one row")
    index_rows = []
    for line in index_prefix.splitlines():
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise T14RouteCFreshError("Route C candidate index prefix is invalid") from exc
        if not isinstance(row, dict):
            raise T14RouteCFreshError("Route C candidate index row is malformed")
        index_rows.append(row)
    if len(index_rows) != candidate_count:
        raise T14RouteCFreshError("Route C candidate index count changed")
    expected_offset = 0
    with candidate_blob.open("rb") as stream:
        for record_id, row in enumerate(index_rows):
            offset = int(row.get("offset", -1))
            length = int(row.get("length", -1))
            if int(row.get("record_id", -1)) != record_id or offset != expected_offset:
                raise T14RouteCFreshError("Route C candidate locator sequence changed")
            stream.seek(offset)
            payload = stream.read(length)
            if len(payload) != length or hashlib.sha256(payload).hexdigest() != row.get(
                "payload_sha256"
            ):
                raise T14RouteCFreshError("Route C committed candidate payload changed")
            expected_offset += length
    if expected_offset != candidate_bytes:
        raise T14RouteCFreshError("Route C candidate payload boundary changed")

    frequency = np.asarray(candidate_state["frequency"], dtype="<i8")
    order = np.asarray(candidate_state["order"], dtype="<i8")
    metadata = np.asarray(candidate_state["metadata"], dtype=RouteCMMapCandidateState._META_DTYPE)
    if (
        frequency.shape != (candidate_count,)
        or order.shape != (candidate_size,)
        or metadata.shape != (candidate_count,)
    ):
        raise T14RouteCFreshError("Route C candidate mmap checkpoint shapes changed")
    record_capacity = int(candidate_state.get("record_capacity", -1))
    capacity = int(candidate_state.get("capacity", -1))
    expected_mmap_sizes = {
        frequency_path: record_capacity * np.dtype("<i8").itemsize,
        order_path: capacity * np.dtype("<i8").itemsize,
        metadata_path: record_capacity
        * np.dtype(RouteCMMapCandidateState._META_DTYPE).itemsize,
    }
    if any(path.stat().st_size != size for path, size in expected_mmap_sizes.items()):
        raise T14RouteCFreshError("Route C candidate mmap file size changed")

    target_state_sha = stable_sha256(
        {
            "sequence_id": int(updater_state.get("sequence_id", -1)),
            "graph_state": dict(graph_state),
            "candidate_counts": [candidate_count, candidate_size],
            "frequency_sha256": _bytes_sha256(frequency),
            "order_sha256": _bytes_sha256(order),
            "metadata_sha256": _bytes_sha256(metadata),
        }
    )
    base_receipt_path = root / f"route_c_recovery_receipt_{completed:06d}.json"
    base_intent_path = root / f"route_c_recovery_intent_{completed:06d}.json"
    base_quarantine = root / "route_c_recovery_quarantine" / f"step-{completed:012d}"
    attempt = 0
    existing: dict[str, Any] | None = None
    if base_receipt_path.is_file() and not base_receipt_path.is_symlink():
        existing = json.loads(base_receipt_path.read_text(encoding="utf-8"))
        unsigned = {
            key: value for key, value in existing.items() if key != "receipt_sha256"
        }
        if (
            existing.get("schema_version") != RECOVERY_SCHEMA
            or existing.get("status") != "PASS"
            or existing.get("completed_step") != completed
            or existing.get("checkpoint_digest") != checkpoint_digest
            or existing.get("promotion_receipt_sha256")
            != promotion_receipt.get("receipt_sha256")
            or existing.get("target_state_sha256") != target_state_sha
            or existing.get("receipt_sha256") != stable_sha256(unsigned)
        ):
            raise T14RouteCFreshError("Route C recovery receipt already differs")
        expected_after = existing.get("after")
        clean = isinstance(expected_after, Mapping) and all(
            (
                graph_blob.stat().st_size
                == int(expected_after.get("graph_blob_bytes", -1)),
                candidate_blob.stat().st_size
                == int(expected_after.get("candidate_blob_bytes", -1)),
                candidate_index.stat().st_size
                == int(expected_after.get("candidate_index_bytes", -1)),
                file_sha256(frequency_path)
                == expected_after.get("frequency_sha256"),
                file_sha256(order_path) == expected_after.get("order_sha256"),
                file_sha256(metadata_path) == expected_after.get("metadata_sha256"),
                live_index.is_file(),
                not live_index.is_symlink(),
                file_sha256(live_index) == expected_after.get("sqlite_sha256")
                if live_index.is_file() and not live_index.is_symlink()
                else False,
                not Path(f"{live_index}-wal").exists(),
                not Path(f"{live_index}-shm").exists(),
            )
        )
        if clean:
            return existing
        attempt = 1
    elif base_receipt_path.exists() or base_receipt_path.is_symlink():
        raise T14RouteCFreshError("Route C recovery receipt is not a physical file")

    # A prior recovery may itself have crashed after publishing its intent.
    # Never overwrite that evidence or reuse its quarantine directory: select
    # a fresh deterministic retry suffix and re-run the verified rollback.
    while True:
        suffix = "" if attempt == 0 else f".retry-{attempt:03d}"
        receipt_path = root / f"route_c_recovery_receipt_{completed:06d}{suffix}.json"
        intent_path = root / f"route_c_recovery_intent_{completed:06d}{suffix}.json"
        quarantine = Path(f"{base_quarantine}{suffix}")
        if not any(
            path.exists() or path.is_symlink()
            for path in (receipt_path, intent_path, quarantine)
        ):
            break
        attempt += 1

    before = {
        "graph_blob_bytes": graph_blob.stat().st_size,
        "candidate_blob_bytes": candidate_blob.stat().st_size,
        "candidate_index_bytes": candidate_index.stat().st_size,
        "frequency_sha256": file_sha256(frequency_path),
        "order_sha256": file_sha256(order_path),
        "metadata_sha256": file_sha256(metadata_path),
    }
    intent = {
        "schema_version": RECOVERY_SCHEMA,
        "status": "VERIFIED_READY_TO_RECOVER",
        "recovery_attempt": attempt,
        "checkpoint_digest": checkpoint_digest,
        "promotion_receipt_sha256": promotion_receipt["receipt_sha256"],
        "target_state_sha256": target_state_sha,
        "before": before,
        "written_at": _utc_now(),
    }
    intent["receipt_sha256"] = stable_sha256(intent)
    atomic_json(intent_path, intent)

    for path, target in (
        (graph_blob, graph_bytes),
        (candidate_blob, candidate_bytes),
        (candidate_index, candidate_index_bytes),
    ):
        with path.open("r+b", buffering=0) as stream:
            stream.truncate(target)
            os.fsync(stream.fileno())

    quarantine.mkdir(parents=True, exist_ok=True)
    sidecars: list[dict[str, Any]] = []
    for suffix in ("-wal", "-shm"):
        path = Path(f"{live_index}{suffix}")
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise T14RouteCFreshError("Route C SQLite sidecar is indirect")
        destination = quarantine / path.name
        if destination.exists():
            raise T14RouteCFreshError("Route C recovery quarantine collision")
        evidence = {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        os.replace(path, destination)
        sidecars.append(evidence)
    from src.baselines.comrecgc.generation_checkpoint import restore_sqlite_snapshot

    restore_sqlite_snapshot(snapshot, live_index)

    frequency_mmap = np.memmap(
        frequency_path, mode="r+", dtype="<i8", shape=(record_capacity,)
    )
    order_mmap = np.memmap(order_path, mode="r+", dtype="<i8", shape=(capacity,))
    metadata_mmap = np.memmap(
        metadata_path,
        mode="r+",
        dtype=RouteCMMapCandidateState._META_DTYPE,
        shape=(record_capacity,),
    )
    frequency_mmap[:] = 0
    frequency_mmap[:candidate_count] = frequency
    order_mmap[:] = -1
    order_mmap[:candidate_size] = order
    for field in metadata_mmap.dtype.names or ():
        metadata_mmap[field][:] = -1
    metadata_mmap[:candidate_count] = metadata
    frequency_mmap.flush()
    order_mmap.flush()
    metadata_mmap.flush()
    del frequency_mmap, order_mmap, metadata_mmap
    for path in (frequency_path, order_path, metadata_path, live_index):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    _fsync_dir(graph_root)
    _fsync_dir(candidate_root)
    _fsync_dir(quarantine)

    after = {
        "graph_blob_bytes": graph_blob.stat().st_size,
        "candidate_blob_bytes": candidate_blob.stat().st_size,
        "candidate_index_bytes": candidate_index.stat().st_size,
        "frequency_sha256": file_sha256(frequency_path),
        "order_sha256": file_sha256(order_path),
        "metadata_sha256": file_sha256(metadata_path),
        "sqlite_sha256": file_sha256(live_index),
    }
    receipt = {
        "schema_version": RECOVERY_SCHEMA,
        "status": "PASS",
        "recovery_attempt": attempt,
        "completed_step": completed,
        "checkpoint_digest": checkpoint_digest,
        "promotion_receipt_sha256": promotion_receipt["receipt_sha256"],
        "checkpoint_sqlite_sha256": file_sha256(snapshot),
        "target_state_sha256": target_state_sha,
        "before": before,
        "after": after,
        "removed_suffix_bytes": {
            "graph_blob": int(before["graph_blob_bytes"]) - graph_bytes,
            "candidate_blob": int(before["candidate_blob_bytes"]) - candidate_bytes,
            "candidate_index": int(before["candidate_index_bytes"])
            - candidate_index_bytes,
        },
        "quarantined_sqlite_sidecars": sidecars,
        "same_root_resume": True,
        "recovered_at": _utc_now(),
    }
    receipt["receipt_sha256"] = stable_sha256(receipt)
    atomic_json(receipt_path, receipt)
    return receipt


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(dict(value)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def audit_no_live_t14_science_owner(
    *,
    control_root: Path,
    receipt_path: Path,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Fail closed if an exact legacy/current T14 owner or science PID is live.

    This is a launch-time observation only.  It never signals a process and it
    deliberately matches exact entrypoint basenames rather than fuzzy task
    names.  Existing T14-named control files are inventoried into the receipt
    so the process observation remains reviewable against prior ownership
    evidence.
    """

    control_root = Path(control_root)
    receipt_path = Path(receipt_path)
    proc_root = Path(proc_root)
    if not control_root.is_absolute() or control_root.is_symlink():
        raise T14RouteCFreshError("Route C control root must be physical and absolute")
    if not receipt_path.is_absolute() or receipt_path.is_symlink():
        raise T14RouteCFreshError("Route C owner-audit receipt must be physical and absolute")
    exact_entrypoints = {
        "run_tastemolnet_comrecgc_full.py",
        "run_tastemolnet_t14_comrecgc_full.sh",
        "run_t14_low_memory_resume_owner.py",
        "run_t14_route_c_owner.py",
    }
    live: list[dict[str, Any]] = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            tokens = [
                value.decode("utf-8", errors="replace")
                for value in (directory / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except OSError:
            continue
        matched = sorted(
            {
                Path(token).name
                for token in tokens
                if Path(token).name in exact_entrypoints
            }
        )
        if not matched:
            continue
        try:
            stat_payload = (directory / "stat").read_text(encoding="utf-8")
            close = stat_payload.rfind(")")
            stat_fields = stat_payload[close + 2 :].split()
            if close < 0 or len(stat_fields) <= 19:
                raise ValueError("malformed proc stat")
            start_ticks = int(stat_fields[19])
            cwd = str((directory / "cwd").resolve(strict=True))
        except (OSError, ValueError, IndexError):
            start_ticks = None
            cwd = None
        live.append(
            {
                "pid": int(directory.name),
                "start_ticks": start_ticks,
                "cwd": cwd,
                "matched_entrypoints": matched,
                "command_sha256": hashlib.sha256(b"\0".join(
                    token.encode("utf-8") for token in tokens
                )).hexdigest(),
            }
        )
    evidence_files: list[str] = []
    if control_root.is_dir():
        for candidate in control_root.rglob("*"):
            try:
                if (
                    "t14" in str(candidate.relative_to(control_root)).lower()
                    and candidate.is_file()
                    and not candidate.is_symlink()
                    and candidate.stat().st_size <= 2 * 1024 * 1024
                ):
                    evidence_files.append(str(candidate))
            except OSError:
                continue
    payload: dict[str, Any] = {
        "schema_version": "tastemolnet_t14_no_live_science_owner_audit_v1",
        "status": "PASS" if not live else "BLOCKED_LIVE_T14_OWNER",
        "control_root": str(control_root),
        "control_evidence_files": sorted(evidence_files),
        "exact_entrypoints": sorted(exact_entrypoints),
        "live_exact_processes": live,
        "process_signal_sent": False,
        "checked_at": _utc_now(),
    }
    unsigned = dict(payload)
    payload["receipt_sha256"] = stable_sha256(unsigned)
    atomic_json(receipt_path, payload)
    if live:
        raise T14RouteCFreshError(
            "Route C launch blocked by an exact live T14 owner/science process"
        )
    return payload


def audit_route_c_matrix_cell_absent(
    *, state_path: Path, lock_path: Path, receipt_path: Path
) -> dict[str, Any]:
    """Bind a Route C launch to the unique matrix while its cell is absent."""

    from src.eval.fast16_matrix_authority_pointer import read_authority_pointer

    state = Path(state_path)
    lock = Path(lock_path)
    receipt = Path(receipt_path)
    if (
        not state.is_absolute()
        or not lock.is_absolute()
        or not receipt.is_absolute()
        or state.is_symlink()
        or lock.is_symlink()
        or receipt.is_symlink()
    ):
        raise T14RouteCFreshError("Route C matrix launch-gate paths are invalid")
    pointer = read_authority_pointer(
        state_path=state,
        lock_path=lock,
        initial_authority_root=None,
    )
    cell = "TasteMolNet/ComRecGC"
    applied = pointer.get("applied_cells")
    if not isinstance(applied, list) or any(type(value) is not str for value in applied):
        raise T14RouteCFreshError("Route C matrix pointer cell inventory is invalid")
    already_published = cell in applied
    payload: dict[str, Any] = {
        "schema_version": "tastemolnet_t14_route_c_matrix_launch_gate_v1",
        "status": "ALREADY_PUBLISHED" if already_published else "PASS_CELL_ABSENT",
        "matrix_state_path": str(state),
        "matrix_state_sha256": file_sha256(state),
        "latest_authority_root": pointer["latest_authority_root"],
        "latest_count": int(pointer["latest_count"]),
        "applied_cells": list(applied),
        "required_absent_cell": cell,
        "science_launch_allowed": not already_published,
        "checked_at": _utc_now(),
    }
    payload["receipt_sha256"] = stable_sha256(payload)
    atomic_json(receipt, payload)
    if already_published:
        raise T14RouteCFreshError(
            "Route C science is already present in the unique matrix authority"
        )
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(value: Any, *, field: str) -> Path:
    if not isinstance(value, str):
        raise T14RouteCFreshError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise T14RouteCFreshError(f"{field} must be normalized and absolute")
    return path


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_fresh_retry_contract(
    raw: Mapping[str, Any], *, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the one authorized post-watchdog retry without opening old state."""

    value = dict(raw)
    required = {
        "schema_version",
        "retry_index",
        "max_retries",
        "reuse_partial_step161",
        "preserve_failed_attempt",
        "fresh_uuid",
        "fresh_output_root",
        "previous_attempt_uuid",
        "previous_output_root",
        "retirement_receipt",
        "retirement_receipt_sha256",
        "dataset_sha256",
        "train_split_sha256",
        "cohort_sha256",
        "t3_gine_sha256",
        "seed",
        "config_sha256",
        "candidate_capacity",
        "m_configured_max",
        "m_fallback_max",
        "min_valid_unique",
        "gpu_index",
        "memory_policy",
        "checkpoint_policy",
        "matrix_authority_root",
        "matrix_authority_state",
        "matrix_authority_lock",
    }
    if set(value) != required:
        raise T14RouteCFreshError("Route C fresh-retry contract fields changed")
    if (
        value.get("schema_version") != FRESH_RETRY_CONTRACT_SCHEMA
        or value.get("retry_index") != 1
        or value.get("max_retries") != 1
        or value.get("reuse_partial_step161") is not False
        or value.get("preserve_failed_attempt") is not True
        or value.get("fresh_uuid") != spec.get("attempt_uuid")
        or value.get("fresh_output_root") != spec.get("output_root")
        or value.get("seed") != 7
        or value.get("candidate_capacity") != 50_000
        or value.get("m_configured_max") != M_MAX
        or value.get("m_fallback_max") != M_FALLBACK_MAX
        or value.get("min_valid_unique") != 10
        or value.get("gpu_index") != GPU_INDEX
    ):
        raise T14RouteCFreshError("Route C fresh-retry fixed contract changed")
    try:
        previous = UUID(str(value.get("previous_attempt_uuid")))
    except (TypeError, ValueError, AttributeError) as exc:
        raise T14RouteCFreshError("Route C previous attempt UUID is invalid") from exc
    if previous.version != 4 or str(previous) != value["previous_attempt_uuid"]:
        raise T14RouteCFreshError("Route C previous attempt UUID is not canonical")
    for field in (
        "retirement_receipt_sha256",
        "dataset_sha256",
        "train_split_sha256",
        "cohort_sha256",
        "t3_gine_sha256",
        "config_sha256",
    ):
        if _SHA256.fullmatch(str(value.get(field) or "")) is None:
            raise T14RouteCFreshError(f"Route C fresh-retry {field} is invalid")
    for field in (
        "previous_output_root",
        "retirement_receipt",
        "matrix_authority_root",
        "matrix_authority_state",
        "matrix_authority_lock",
    ):
        _absolute(value.get(field), field=f"fresh_retry.{field}")
    retirement_path = Path(value["retirement_receipt"])
    retirement = validate_fresh_retry_retirement_receipt(retirement_path)
    if (
        file_sha256(retirement_path) != value["retirement_receipt_sha256"]
        or retirement.get("old_attempt_uuid") != value["previous_attempt_uuid"]
        or retirement.get("old_output_root") != value["previous_output_root"]
    ):
        raise T14RouteCFreshError("Route C retry retirement binding changed")
    matrix_root = Path(value["matrix_authority_root"])
    if (
        Path(value["matrix_authority_state"]).parent != matrix_root
        or Path(value["matrix_authority_lock"]).parent != matrix_root
    ):
        raise T14RouteCFreshError("Route C retry matrix authority binding changed")
    previous_root = Path(retirement["preserved_science_root"])
    fresh_root = Path(str(spec["output_root"]))
    if not previous_root.is_dir() or previous_root.is_symlink():
        raise T14RouteCFreshError("Route C failed attempt was not preserved")
    if previous_root == fresh_root or _is_within(previous_root, fresh_root) or _is_within(
        fresh_root, previous_root
    ):
        raise T14RouteCFreshError("Route C retry overlaps the failed attempt")
    environment_bytes = canonical_bytes(spec["science_environment"]).decode("utf-8")
    if (
        str(previous_root) in environment_bytes
        or str(value["previous_output_root"]) in environment_bytes
    ):
        raise T14RouteCFreshError("Route C retry environment references failed state")
    memory = value.get("memory_policy")
    if memory != {
        "start_headroom_bytes": 384 * 1024**3,
        "runtime_reserve_bytes": 96 * 1024**3,
        "launch_samples_required": 3,
        "runtime_low_headroom_samples": 3,
        "sample_seconds": 30.0,
    }:
        raise T14RouteCFreshError("Route C retry memory policy changed")
    spec_memory = spec.get("memory")
    if (
        not isinstance(spec_memory, Mapping)
        or spec_memory.get("launch_headroom_bytes") != memory["start_headroom_bytes"]
        or spec_memory.get("runtime_headroom_bytes") != memory["runtime_reserve_bytes"]
        or spec_memory.get("launch_samples_required", 1)
        != memory["launch_samples_required"]
        or spec_memory.get("runtime_low_headroom_samples", 1)
        != memory["runtime_low_headroom_samples"]
        or float(spec_memory.get("sample_seconds", -1)) != memory["sample_seconds"]
    ):
        raise T14RouteCFreshError("Route C retry memory/spec binding changed")
    checkpoints = value.get("checkpoint_policy")
    if checkpoints != {
        "early_steps": [50, 100, 250, 500],
        "production_steps": [
            2_500,
            5_000,
            7_500,
            10_000,
            12_500,
            15_000,
            17_500,
            20_000,
        ],
        "fresh_process_reload_each_checkpoint": True,
        "route_c_500_promoted_to_full_without_replay": True,
    }:
        raise T14RouteCFreshError("Route C retry checkpoint policy changed")
    return value


def validate_fresh_retry_retirement_receipt(path: Path) -> dict[str, Any]:
    """Read one immutable receipt proving the failed attempt was only superseded."""

    receipt_path = Path(path)
    if not receipt_path.is_absolute() or receipt_path.is_symlink() or not receipt_path.is_file():
        raise T14RouteCFreshError("Route C retirement receipt is absent or indirect")
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14RouteCFreshError("Route C retirement receipt is unreadable") from exc
    if not isinstance(value, dict):
        raise T14RouteCFreshError("Route C retirement receipt is not an object")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("schema_version") != FRESH_RETRY_RECEIPT_SCHEMA
        or value.get("terminal_state") != "TERMINAL_FAILED_RESOURCE_WATCHDOG"
        or value.get("retirement_state") != "SUPERSEDED_BY_FRESH_RETRY"
        or value.get("retry_index") != 1
        or value.get("max_retries") != 1
        or value.get("reuse_partial_step161") is not False
        or value.get("preserve_failed_attempt") is not True
        or value.get("old_root_deleted") is not False
        or value.get("process_signal_sent") is not False
        or _SHA256.fullmatch(str(value.get("old_terminal_sha256") or "")) is None
        or value.get("receipt_sha256") != stable_sha256(unsigned)
    ):
        raise T14RouteCFreshError("Route C retirement receipt changed")
    for field in ("old_output_root", "old_owner_root", "retired_pointer_root"):
        _absolute(value.get(field), field=f"retirement.{field}")
    old_output = Path(value["old_output_root"])
    old_owner = Path(value["old_owner_root"])
    old_terminal = old_owner / "terminal.json"
    if (
        not old_owner.is_dir()
        or old_owner.is_symlink()
        or not old_terminal.is_file()
        or old_terminal.is_symlink()
        or file_sha256(old_terminal) != value["old_terminal_sha256"]
    ):
        raise T14RouteCFreshError("Route C failed attempt preservation changed")
    # The original owner wrote REFERENCE_500 below its sealed canary plan.  Its
    # master output_root is only a prospective production destination and was
    # never created when the watchdog fired during that child.  Bind the
    # preserved attempt to the child spec/plan rather than claiming the unused
    # master destination contained science.
    preserved_science_root = old_output
    preservation_source = "MASTER_OUTPUT_ROOT"
    if not old_output.is_dir() or old_output.is_symlink():
        plan_path = _regular_physical_file(
            old_owner / "owner_plan.json", field="failed owner plan"
        )
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise T14RouteCFreshError("Route C failed owner plan is unreadable") from exc
        reference = (
            plan.get("children", {}).get("REFERENCE_500")
            if isinstance(plan, Mapping)
            else None
        )
        if not isinstance(reference, Mapping):
            raise T14RouteCFreshError("Route C failed reference child is absent")
        child_spec_path = _regular_physical_file(
            Path(str(reference.get("spec_path", ""))),
            field="failed reference child spec",
        )
        expected_child_sha = str(reference.get("spec_sha256") or "")
        try:
            child_spec = json.loads(child_spec_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise T14RouteCFreshError(
                "Route C failed reference child spec is unreadable"
            ) from exc
        # owner_plan records the task spec's canonical self hash, not the
        # byte-level hash of the newline-terminated JSON file.  Recompute that
        # documented self hash and bind both copies; comparing the plan value
        # to file_sha256() incorrectly rejects every real sealed child spec.
        child_spec_unsigned = {
            key: item for key, item in child_spec.items() if key != "spec_sha256"
        }
        if (
            _SHA256.fullmatch(expected_child_sha) is None
            or child_spec.get("spec_sha256") != expected_child_sha
            or stable_sha256(child_spec_unsigned) != expected_child_sha
        ):
            raise T14RouteCFreshError("Route C failed reference child spec changed")
        preserved_science_root = Path(str(reference.get("output_root", "")))
        if (
            child_spec.get("output_root") != str(preserved_science_root)
            or not preserved_science_root.is_dir()
            or preserved_science_root.is_symlink()
            or not _is_within(preserved_science_root, old_owner)
            or not (preserved_science_root / "cohort_manifest.json").is_file()
            or not (preserved_science_root / "route_c_step_states.jsonl").is_file()
        ):
            raise T14RouteCFreshError(
                "Route C failed reference science root was not preserved"
            )
        preservation_source = "OWNER_PLAN_REFERENCE_500"
    result = dict(value)
    result["preserved_science_root"] = str(preserved_science_root)
    result["preservation_source"] = preservation_source
    result["declared_master_output_materialized"] = old_output.is_dir()
    return result


def retire_failed_route_c_current(
    *,
    current_root: Path,
    retired_root: Path,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Atomically retire only a dead resource-watchdog pointer for one fresh retry.

    The failed science and owner roots are never modified or removed.  The
    caller must hold the Route C launch lock, making the absence check and
    pointer rename a one-shot operation.
    """

    current = Path(current_root)
    retired = Path(retired_root)
    if (
        not current.is_absolute()
        or not retired.is_absolute()
        or current.is_symlink()
        or retired.is_symlink()
        or not current.is_dir()
    ):
        raise T14RouteCFreshError("Route C retry pointer roots are invalid")
    required = {
        "owner.pid": current / "owner.pid",
        "owner.start_ticks": current / "owner.start_ticks",
        "task_spec.path": current / "task_spec.path",
    }
    for label, path in required.items():
        _regular_physical_file(path, field=f"current {label}")
    try:
        old_pid = int(required["owner.pid"].read_text(encoding="utf-8").strip())
        old_ticks = int(
            required["owner.start_ticks"].read_text(encoding="utf-8").strip()
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise T14RouteCFreshError("Route C failed owner identity is unreadable") from exc
    spec_path = Path(required["task_spec.path"].read_text(encoding="utf-8").strip())
    _regular_physical_file(spec_path, field="failed task spec")
    try:
        old_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14RouteCFreshError("Route C failed task spec is unreadable") from exc
    if not isinstance(old_spec, dict):
        raise T14RouteCFreshError("Route C failed task spec is not an object")
    if old_spec.get("fresh_retry") is not None:
        raise T14RouteCFreshError("Route C one fresh retry was already consumed")
    old_owner_root = _absolute(old_spec.get("owner_root"), field="failed owner_root")
    old_output_root = _absolute(old_spec.get("output_root"), field="failed output_root")
    if spec_path.parent != old_owner_root:
        raise T14RouteCFreshError("Route C failed task spec escaped its owner root")
    owner_path = _regular_physical_file(
        old_owner_root / "owner.json", field="failed owner evidence"
    )
    terminal_path = _regular_physical_file(
        old_owner_root / "terminal.json", field="failed terminal evidence"
    )
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        not isinstance(owner, dict)
        or owner.get("owner_pid") != old_pid
        or owner.get("owner_start_ticks") != old_ticks
        or owner.get("task_spec") != str(spec_path)
        or not isinstance(terminal, dict)
        or terminal.get("status") != "FAILED"
        or "resource watchdog" not in str(terminal.get("error", "")).lower()
    ):
        raise T14RouteCFreshError("Route C prior attempt was not a resource-watchdog failure")
    live_stat = proc_root / str(old_pid) / "stat"
    if live_stat.is_file():
        try:
            raw = live_stat.read_text(encoding="utf-8")
            close = raw.rfind(")")
            fields = raw[close + 2 :].split()
            live_ticks = int(fields[19])
        except (OSError, ValueError, IndexError) as exc:
            raise T14RouteCFreshError("Route C prior PID identity is ambiguous") from exc
        if live_ticks == old_ticks:
            raise T14RouteCFreshError("Route C failed owner PID is still live")
    exact_entrypoints = {
        "run_tastemolnet_comrecgc_full.py",
        "run_tastemolnet_t14_comrecgc_full.sh",
        "run_t14_route_c_owner.py",
    }
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            tokens = [
                item.decode("utf-8", errors="replace")
                for item in (directory / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except OSError:
            continue
        if not any(Path(token).name in exact_entrypoints for token in tokens):
            continue
        command = "\0".join(tokens)
        if str(old_output_root) in command or str(spec_path) in command:
            raise T14RouteCFreshError("Route C failed root still has an exact writer")
    attempt = str(old_spec.get("attempt_uuid") or "")
    try:
        attempt_uuid = UUID(attempt)
    except (TypeError, ValueError, AttributeError) as exc:
        raise T14RouteCFreshError("Route C failed attempt UUID is invalid") from exc
    if attempt_uuid.version != 4 or str(attempt_uuid) != attempt:
        raise T14RouteCFreshError("Route C failed attempt UUID is not canonical")
    retired.mkdir(parents=True, exist_ok=True)
    for prior_receipt in retired.glob(
        "*-superseded-by-fresh-retry-1/retirement_receipt.json"
    ):
        if prior_receipt.is_file() and not prior_receipt.is_symlink():
            validate_fresh_retry_retirement_receipt(prior_receipt)
            raise T14RouteCFreshError("Route C one fresh retry was already consumed")
    destination = retired / f"{attempt}-superseded-by-fresh-retry-1"
    if destination.exists() or destination.is_symlink():
        raise T14RouteCFreshError("Route C fresh retry was already consumed")
    completed_step = 0
    ledger = old_output_root / "route_c_step_states.jsonl"
    if ledger.is_file() and not ledger.is_symlink():
        with ledger.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    completed_step = max(completed_step, int(row["completed_step"]))
    receipt: dict[str, Any] = {
        "schema_version": FRESH_RETRY_RECEIPT_SCHEMA,
        "terminal_state": "TERMINAL_FAILED_RESOURCE_WATCHDOG",
        "retirement_state": "SUPERSEDED_BY_FRESH_RETRY",
        "retry_index": 1,
        "max_retries": 1,
        "reuse_partial_step161": False,
        "preserve_failed_attempt": True,
        "old_attempt_uuid": attempt,
        "old_owner_pid": old_pid,
        "old_owner_start_ticks": old_ticks,
        "old_owner_root": str(old_owner_root),
        "old_output_root": str(old_output_root),
        "old_terminal_sha256": file_sha256(terminal_path),
        "observed_uncommitted_step": completed_step,
        "old_root_deleted": False,
        "process_signal_sent": False,
        "retired_pointer_root": str(destination),
        "retired_at": _utc_now(),
    }
    receipt["receipt_sha256"] = stable_sha256(receipt)
    atomic_json(current / "retirement_receipt.json", receipt)
    os.replace(current, destination)
    _fsync_dir(retired)
    _fsync_dir(retired.parent)
    return validate_fresh_retry_retirement_receipt(
        destination / "retirement_receipt.json"
    )


def build_spec(
    *,
    attempt_uuid: str,
    execution_commit: str,
    python: Path,
    science_wrapper: Path,
    owner_entrypoint: Path,
    output_root: Path,
    owner_root: Path,
    cgroup_limit_path: Path,
    cgroup_current_path: Path,
    cgroup_failcnt_path: Path,
    forbidden_legacy_root: Path,
    science_environment: Mapping[str, str],
    storage_mode: str = "lowmemory",
    canary_role: str = "PROMOTABLE_LOW_MEMORY",
    max_process_rss_bytes: int,
    launch_headroom_bytes: int,
    runtime_headroom_bytes: int,
    sample_seconds: float = 30.0,
    launch_samples_required: int = 3,
    runtime_low_headroom_samples: int = 3,
    fresh_retry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one immutable fresh-route spec without touching the legacy root."""

    value: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA,
        "attempt_uuid": attempt_uuid,
        "run_id": f"tastemolnet-t14-route-c-{attempt_uuid}",
        "execution_commit": execution_commit,
        "python": str(python),
        "science_wrapper": str(science_wrapper),
        "science_wrapper_sha256": file_sha256(science_wrapper),
        "owner_entrypoint": str(owner_entrypoint),
        "owner_entrypoint_sha256": file_sha256(owner_entrypoint),
        "output_root": str(output_root),
        "owner_root": str(owner_root),
        "promotion_root": str(owner_root / "promotions"),
        "gpu_index": GPU_INDEX,
        "storage_mode": storage_mode,
        "canary_role": canary_role,
        "fresh_resume_flag": 0,
        "continuation_resume_flag": 1,
        "first_checkpoint_step": FIRST_CHECKPOINT_STEP,
        "promotable_checkpoint_step": PROMOTABLE_CHECKPOINT_STEP,
        "reload_replay_end_step": RELOAD_REPLAY_END_STEP,
        "production_checkpoint_steps": list(PRODUCTION_CHECKPOINT_STEPS),
        "m_configured_max": M_MAX,
        "m_fallback_max": M_FALLBACK_MAX,
        "checkpoint_reload_process": "INDEPENDENT_AFTER_FRESH_SCIENCE_EXIT",
        "forbidden_legacy_root": str(forbidden_legacy_root),
        "forbidden_legacy_checkpoint_step": 12_500,
        "legacy_checkpoint_loaded": False,
        "route_c_state": {
            "append_only_graph_store": True,
            "stable_numeric_graph_ids": True,
            "mmap_candidate_frequency": True,
            "mmap_candidate_metadata": True,
            "compact_transition_records": True,
            "stable_predecessor_downstream_ids": True,
            "bounded_graph_lru": 128,
            "single_scientific_state_updater": True,
            "deterministic_sequence_ids": True,
            "atomic_index_checkpoint": True,
            "lazy_pyg_reconstruction": True,
            "candidate_record_capacity": 200_000,
        },
        "science_environment": dict(science_environment),
        "memory": {
            "cgroup_limit_path": str(cgroup_limit_path),
            "cgroup_current_path": str(cgroup_current_path),
            "cgroup_failcnt_path": str(cgroup_failcnt_path),
            "max_process_rss_bytes": int(max_process_rss_bytes),
            "launch_headroom_bytes": int(launch_headroom_bytes),
            "runtime_headroom_bytes": int(runtime_headroom_bytes),
            "sample_seconds": float(sample_seconds),
            "launch_samples_required": int(launch_samples_required),
            "runtime_low_headroom_samples": int(runtime_low_headroom_samples),
        },
        "created_at": _utc_now(),
    }
    if fresh_retry is not None:
        value["fresh_retry"] = dict(fresh_retry)
    value["spec_sha256"] = stable_sha256(value)
    return validate_spec(value, check_files=True)


def validate_spec(
    raw: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise T14RouteCFreshError("Route C spec must be a JSON object")
    value = dict(raw)
    required = {
        "schema_version",
        "attempt_uuid",
        "run_id",
        "execution_commit",
        "python",
        "science_wrapper",
        "science_wrapper_sha256",
        "owner_entrypoint",
        "owner_entrypoint_sha256",
        "output_root",
        "owner_root",
        "promotion_root",
        "gpu_index",
        "storage_mode",
        "canary_role",
        "fresh_resume_flag",
        "continuation_resume_flag",
        "first_checkpoint_step",
        "promotable_checkpoint_step",
        "reload_replay_end_step",
        "production_checkpoint_steps",
        "m_configured_max",
        "m_fallback_max",
        "checkpoint_reload_process",
        "forbidden_legacy_root",
        "forbidden_legacy_checkpoint_step",
        "legacy_checkpoint_loaded",
        "route_c_state",
        "science_environment",
        "memory",
        "created_at",
        "spec_sha256",
    }
    optional = {"fresh_retry"}
    if not required.issubset(value) or set(value) - required - optional:
        raise T14RouteCFreshError("Route C spec fields changed")
    if value.get("schema_version") != SPEC_SCHEMA:
        raise T14RouteCFreshError("Route C spec schema changed")
    try:
        attempt = UUID(str(value.get("attempt_uuid")))
    except (ValueError, TypeError, AttributeError) as exc:
        raise T14RouteCFreshError("Route C attempt must be one UUIDv4") from exc
    if attempt.version != 4 or str(attempt) != value["attempt_uuid"]:
        raise T14RouteCFreshError("Route C attempt must be one canonical UUIDv4")
    if value.get("run_id") != f"tastemolnet-t14-route-c-{attempt}":
        raise T14RouteCFreshError("Route C run ID differs from its attempt UUID")
    if _GIT_SHA.fullmatch(str(value.get("execution_commit") or "")) is None:
        raise T14RouteCFreshError("Route C execution commit must be one full Git SHA")
    if value.get("gpu_index") != GPU_INDEX:
        raise T14RouteCFreshError("Route C is bound to physical GPU2")
    if (
        value.get("fresh_resume_flag") != 0
        or value.get("continuation_resume_flag") != 1
        or value.get("first_checkpoint_step") != FIRST_CHECKPOINT_STEP
        or value.get("promotable_checkpoint_step") != PROMOTABLE_CHECKPOINT_STEP
        or value.get("reload_replay_end_step") != RELOAD_REPLAY_END_STEP
        or value.get("production_checkpoint_steps")
        != list(PRODUCTION_CHECKPOINT_STEPS)
        or value.get("m_configured_max") != M_MAX
        or value.get("m_fallback_max") != M_FALLBACK_MAX
        or value.get("checkpoint_reload_process")
        != "INDEPENDENT_AFTER_FRESH_SCIENCE_EXIT"
        or value.get("forbidden_legacy_checkpoint_step") != 12_500
        or value.get("legacy_checkpoint_loaded") is not False
    ):
        raise T14RouteCFreshError("Route C fixed scientific/operational contract changed")
    paths = {
        field: _absolute(value.get(field), field=field)
        for field in (
            "python",
            "science_wrapper",
            "owner_entrypoint",
            "output_root",
            "owner_root",
            "promotion_root",
            "forbidden_legacy_root",
        )
    }
    if paths["promotion_root"].parent != paths["owner_root"]:
        raise T14RouteCFreshError("Route C promotion root escaped its owner root")
    if value.get("storage_mode") not in {"reference", "lowmemory"}:
        raise T14RouteCFreshError("Route C storage mode changed")
    if value.get("canary_role") not in {
        "REFERENCE_500",
        "LOW_MEMORY_CONTINUOUS_510",
        "LOW_MEMORY_RELOAD_510",
        "PROMOTABLE_LOW_MEMORY",
    }:
        raise T14RouteCFreshError("Route C canary role changed")
    state_contract = value.get("route_c_state")
    if state_contract != {
        "append_only_graph_store": True,
        "stable_numeric_graph_ids": True,
        "mmap_candidate_frequency": True,
        "mmap_candidate_metadata": True,
        "compact_transition_records": True,
        "stable_predecessor_downstream_ids": True,
        "bounded_graph_lru": 128,
        "single_scientific_state_updater": True,
        "deterministic_sequence_ids": True,
        "atomic_index_checkpoint": True,
        "lazy_pyg_reconstruction": True,
        "candidate_record_capacity": 200_000,
    }:
        raise T14RouteCFreshError("Route C state design contract changed")
    if not paths["output_root"].name.endswith(str(attempt)):
        raise T14RouteCFreshError("Route C output root does not bind its UUID")
    if not paths["owner_root"].name.endswith(str(attempt)):
        raise T14RouteCFreshError("Route C owner root does not bind its UUID")
    forbidden = paths["forbidden_legacy_root"]
    if (
        _is_within(paths["output_root"], forbidden)
        or _is_within(forbidden, paths["output_root"])
        or _is_within(paths["owner_root"], forbidden)
        or _is_within(forbidden, paths["owner_root"])
    ):
        raise T14RouteCFreshError("Route C fresh roots overlap the forbidden legacy root")
    for field in ("science_wrapper_sha256", "owner_entrypoint_sha256", "spec_sha256"):
        if _SHA256.fullmatch(str(value.get(field) or "")) is None:
            raise T14RouteCFreshError(f"Route C {field} is not SHA-256")
    unsigned = {key: item for key, item in value.items() if key != "spec_sha256"}
    if value["spec_sha256"] != stable_sha256(unsigned):
        raise T14RouteCFreshError("Route C spec self hash changed")
    environment = value.get("science_environment")
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        for key, item in environment.items()
    ):
        raise T14RouteCFreshError("Route C science environment is invalid")
    required_environment = {
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "RUN_GNN_ABLATION": "0",
        "RUN_LLM_ABLATION": "0",
    }
    for key, expected in required_environment.items():
        if environment.get(key) != expected:
            raise T14RouteCFreshError(f"Route C environment changed: {key}")
    if str(forbidden) in canonical_bytes(environment).decode("utf-8"):
        raise T14RouteCFreshError("Route C science environment references the legacy root")
    memory = value.get("memory")
    base_memory_fields = {
        "cgroup_limit_path",
        "cgroup_current_path",
        "cgroup_failcnt_path",
        "max_process_rss_bytes",
        "launch_headroom_bytes",
        "runtime_headroom_bytes",
        "sample_seconds",
    }
    extended_memory_fields = {
        *base_memory_fields,
        "launch_samples_required",
        "runtime_low_headroom_samples",
    }
    if not isinstance(memory, Mapping) or frozenset(memory) not in {
        frozenset(base_memory_fields),
        frozenset(extended_memory_fields),
    }:
        raise T14RouteCFreshError("Route C memory contract changed")
    for field in ("cgroup_limit_path", "cgroup_current_path", "cgroup_failcnt_path"):
        _absolute(memory.get(field), field=f"memory.{field}")
    for field in (
        "max_process_rss_bytes",
        "launch_headroom_bytes",
        "runtime_headroom_bytes",
    ):
        if type(memory.get(field)) is not int or int(memory[field]) <= 0:
            raise T14RouteCFreshError(f"Route C memory.{field} must be positive")
    if int(memory["runtime_headroom_bytes"]) >= int(memory["launch_headroom_bytes"]):
        raise T14RouteCFreshError("Route C runtime headroom must be below launch headroom")
    if (
        not isinstance(memory.get("sample_seconds"), (int, float))
        or isinstance(memory.get("sample_seconds"), bool)
        or not 1 <= float(memory["sample_seconds"]) <= 60
    ):
        raise T14RouteCFreshError("Route C sample interval is invalid")
    for field in ("launch_samples_required", "runtime_low_headroom_samples"):
        observed = memory.get(field, 1)
        if type(observed) is not int or not 1 <= int(observed) <= 10:
            raise T14RouteCFreshError(f"Route C memory.{field} is invalid")
    retry = value.get("fresh_retry")
    if retry is not None:
        _validate_fresh_retry_contract(retry, spec=value)
    if check_files:
        for field in ("science_wrapper", "owner_entrypoint"):
            path = paths[field]
            if not path.is_file() or path.is_symlink():
                raise T14RouteCFreshError(f"Route C {field} is absent or indirect")
            if file_sha256(path) != value[f"{field}_sha256"]:
                raise T14RouteCFreshError(f"Route C {field} bytes changed")
        python = paths["python"]
        try:
            resolved_python = python.resolve(strict=True)
        except OSError as exc:
            raise T14RouteCFreshError("Route C Python is absent") from exc
        if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
            raise T14RouteCFreshError("Route C Python is not executable")
        for field in ("cgroup_limit_path", "cgroup_current_path", "cgroup_failcnt_path"):
            path = Path(str(memory[field]))
            if not path.is_file() or path.is_symlink():
                raise T14RouteCFreshError(f"Route C memory counter is absent: {field}")
    return value


def write_spec(path: Path, spec: Mapping[str, Any]) -> None:
    validated = validate_spec(spec, check_files=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Route C spec path must be fresh")
    atomic_json(path, validated)


def load_spec(path: Path, *, check_files: bool = True) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise T14RouteCFreshError("Route C spec must be one absolute physical file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14RouteCFreshError("Route C spec is unreadable") from exc
    return validate_spec(raw, check_files=check_files)


def _checkpoint_identity(output_root: Path) -> dict[str, Any]:
    path = output_root / "checkpoint_identity.json"
    if not path.is_file() or path.is_symlink():
        raise T14RouteCFreshError("Route C checkpoint identity is absent")
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14RouteCFreshError("Route C checkpoint identity is unreadable") from exc
    if (
        not isinstance(identity, dict)
        or identity.get("schema_version")
        != "tastemolnet_t14_checkpoint_provenance_v1"
        or identity.get("total_steps") != M_FALLBACK_MAX
        or identity.get("checkpoint_interval") != FIRST_CHECKPOINT_STEP
        or not isinstance(identity.get("provenance"), dict)
        or not isinstance(identity.get("scientific_argv"), list)
        or _SHA256.fullmatch(str(identity.get("command_sha256") or "")) is None
    ):
        raise T14RouteCFreshError("Route C checkpoint identity changed")
    return identity


def _registered_checkpoint_step(step: int) -> int:
    value = int(step)
    if value not in {*EARLY_CHECKPOINT_STEPS, *PRODUCTION_CHECKPOINT_STEPS}:
        raise T14RouteCFreshError(f"Route C checkpoint step is not registered: {value}")
    return value


def promotion_receipt_path(spec: Mapping[str, Any], step: int) -> Path:
    value = validate_spec(spec, check_files=False)
    completed = _registered_checkpoint_step(step)
    return Path(value["promotion_root"]) / f"checkpoint-{completed:06d}.json"


def validate_checkpoint_boundary(
    spec: Mapping[str, Any],
    *,
    step: int,
    validate_envelope: bool = True,
) -> dict[str, Any]:
    """Validate one clean Route C checkpoint without opening the legacy 12.5k root."""

    value = validate_spec(spec, check_files=False)
    completed = _registered_checkpoint_step(step)
    root = Path(value["output_root"])
    boundary_path = root / f"route_c_boundary_{completed:06d}.json"
    pending_path = root / "checkpoints" / "PENDING_LATEST.json"
    if not boundary_path.is_file() or boundary_path.is_symlink():
        raise T14RouteCFreshError("Route C checkpoint-boundary receipt is absent")
    if not pending_path.is_file() or pending_path.is_symlink():
        raise T14RouteCFreshError("Route C checkpoint is not pending reload")
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    checkpoint_name = f"step-{completed:012d}"
    checkpoint_dir = root / "checkpoints" / checkpoint_name
    if (
        not isinstance(boundary, dict)
        or boundary.get("schema_version") != BOUNDARY_SCHEMA
        or boundary.get("status") != "CHECKPOINT_BOUNDARY_REACHED"
        or boundary.get("completed_step") != completed
        or boundary.get("next_step") != completed + 1
        or boundary.get("attempt_uuid") != value["attempt_uuid"]
        or boundary.get("spec_sha256") != value["spec_sha256"]
        or boundary.get("output_root") != str(root)
        or boundary.get("legacy_checkpoint_loaded") is not False
        or boundary.get("payload_reload_state") != "PENDING_INDEPENDENT_RELOAD"
        or boundary.get("latest_promoted") is not False
        or _SHA256.fullmatch(str(boundary.get("checkpoint_digest") or "")) is None
    ):
        raise T14RouteCFreshError("Route C checkpoint-boundary receipt changed")
    expected_pending = {
        "schema_version": "comrecgc_generation_checkpoint_pending_v1",
        "checkpoint_dir": checkpoint_name,
        "completed_step": completed,
        "checkpoint_digest": boundary["checkpoint_digest"],
        "payload_reload_state": "PENDING_INDEPENDENT_RELOAD",
    }
    if pending != expected_pending:
        raise T14RouteCFreshError("Route C pending pointer differs from boundary")
    if (root / "GENERATION_PASS").exists():
        raise T14RouteCFreshError("Route C canary unexpectedly claims generation PASS")
    identity = _checkpoint_identity(root)
    if identity["provenance"].get("execution_commit") != value["execution_commit"]:
        raise T14RouteCFreshError("Route C checkpoint execution commit changed")
    if identity["provenance"].get("physical_gpu_index") != str(GPU_INDEX):
        raise T14RouteCFreshError("Route C checkpoint was not produced on GPU2")
    if boundary.get("checkpoint_identity_sha256") != file_sha256(
        root / "checkpoint_identity.json"
    ):
        raise T14RouteCFreshError("Route C boundary identity binding changed")
    if validate_envelope:
        from src.baselines.comrecgc.generation_checkpoint import (
            validate_generation_checkpoint_envelope,
        )

        validation = validate_generation_checkpoint_envelope(
            checkpoint_dir,
            expected_provenance=identity["provenance"],
            expected_scientific_argv=identity["scientific_argv"],
            expected_command_sha256=identity["command_sha256"],
            expected_total_steps=M_FALLBACK_MAX,
            expected_completed_step=completed,
        )
        if validation.checkpoint_digest != boundary["checkpoint_digest"]:
            raise T14RouteCFreshError("Route C checkpoint digest changed")
    return {
        "boundary": boundary,
        "identity": identity,
        "checkpoint_dir": str(checkpoint_dir),
    }


def validate_first_checkpoint(
    spec: Mapping[str, Any], *, validate_envelope: bool = True
) -> dict[str, Any]:
    return validate_checkpoint_boundary(
        spec, step=FIRST_CHECKPOINT_STEP, validate_envelope=validate_envelope
    )


def promote_checkpoint(spec: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    """Independently reload and promote one clean Route C boundary."""

    value = validate_spec(spec, check_files=True)
    completed = _registered_checkpoint_step(step)
    evidence = validate_checkpoint_boundary(
        value, step=completed, validate_envelope=True
    )
    identity = evidence["identity"]
    checkpoint_dir = Path(evidence["checkpoint_dir"])
    from src.baselines.comrecgc.generation_checkpoint import (
        promote_generation_checkpoint,
    )

    validation = promote_generation_checkpoint(
        checkpoint_dir,
        expected_provenance=identity["provenance"],
        expected_scientific_argv=identity["scientific_argv"],
        expected_command_sha256=identity["command_sha256"],
        expected_total_steps=M_FALLBACK_MAX,
        expected_completed_step=completed,
    )
    receipt: dict[str, Any] = {
        "schema_version": PROMOTION_SCHEMA,
        "status": "PASS",
        "spec_sha256": value["spec_sha256"],
        "attempt_uuid": value["attempt_uuid"],
        "output_root": value["output_root"],
        "checkpoint_dir": str(validation.checkpoint_dir),
        "completed_step": validation.completed_step,
        "checkpoint_digest": validation.checkpoint_digest,
        "checkpoint_identity_sha256": file_sha256(
            Path(value["output_root"]) / "checkpoint_identity.json"
        ),
        "payload_reload_pass": True,
        "latest_promoted": True,
        "independent_process_required": True,
        "legacy_checkpoint_loaded": False,
        "forbidden_legacy_checkpoint_step": 12_500,
        "promoted_at": _utc_now(),
    }
    receipt["receipt_sha256"] = stable_sha256(receipt)
    receipt_path = promotion_receipt_path(value, completed)
    if receipt_path.exists() or receipt_path.is_symlink():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        stable_fields = {key for key in receipt if key != "promoted_at"}
        if any(existing.get(key) != receipt.get(key) for key in stable_fields):
            raise T14RouteCFreshError("Route C promotion receipt already differs")
        return validate_promotion_receipt(
            receipt_path, spec=value, expected_step=completed
        )
    atomic_json(receipt_path, receipt)
    return validate_promotion_receipt(
        receipt_path, spec=value, expected_step=completed
    )


def promote_first_checkpoint(spec: Mapping[str, Any]) -> dict[str, Any]:
    return promote_checkpoint(spec, step=FIRST_CHECKPOINT_STEP)


def validate_promotion_receipt(
    path: Path,
    *,
    spec: Mapping[str, Any],
    expected_step: int | None,
) -> dict[str, Any]:
    value = validate_spec(spec, check_files=False)
    root = Path(value["output_root"])
    latest_path = root / "checkpoints" / "LATEST"
    if not latest_path.is_file() or latest_path.is_symlink():
        raise T14RouteCFreshError("Route C promoted LATEST pointer is absent")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    completed = _registered_checkpoint_step(
        int(latest.get("completed_step", -1)) if expected_step is None else expected_step
    )
    if path != promotion_receipt_path(value, completed):
        raise T14RouteCFreshError("Route C promotion receipt path changed")
    if not path.is_file() or path.is_symlink():
        raise T14RouteCFreshError("Route C promotion receipt is absent or indirect")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14RouteCFreshError("Route C promotion receipt is unreadable") from exc
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema_version") != PROMOTION_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("spec_sha256") != value["spec_sha256"]
        or receipt.get("attempt_uuid") != value["attempt_uuid"]
        or receipt.get("output_root") != str(root)
        or receipt.get("completed_step") != completed
        or receipt.get("payload_reload_pass") is not True
        or receipt.get("latest_promoted") is not True
        or receipt.get("independent_process_required") is not True
        or receipt.get("legacy_checkpoint_loaded") is not False
        or receipt.get("forbidden_legacy_checkpoint_step") != 12_500
        or receipt.get("checkpoint_identity_sha256")
        != file_sha256(root / "checkpoint_identity.json")
        or receipt.get("receipt_sha256") != stable_sha256(unsigned)
        or latest.get("completed_step") != completed
        or latest.get("checkpoint_digest") != receipt.get("checkpoint_digest")
        or latest.get("checkpoint_dir") != f"step-{completed:012d}"
    ):
        raise T14RouteCFreshError("Route C promotion receipt changed")
    return receipt
