"""Fail-closed live graph resolution for long COMRECGC random walks.

Pinned upstream COMRECGC intentionally evicts entries from its active
``graph_map`` when the candidate array reaches capacity.  A hash can still be
held by a random-walk head or transition while that logical candidate eviction
is taking place.  This module preserves the evicted value in an authoritative
SQLite store and resolves those live references without changing active-map
membership, candidate ordering, or RNG behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
from collections import Counter, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .graph_trace import stable_untyped_graph_sha256


GRAPH_STATE_POLICY = "authoritative_backing_live_graph_resolution_v2"
LIVE_GRAPH_CHECKPOINT_SCHEMA = "comrecgc_live_graph_state_v1"


class ComRecGCLiveGraphResolutionError(RuntimeError):
    """Raised when a live COMRECGC graph hash cannot be resolved exactly."""

    def __init__(self, graph_hash: Any, diagnostics: Mapping[str, Any]) -> None:
        self.graph_hash = graph_hash
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "[COMRECGC_LIVE_GRAPH_RESOLUTION_ERROR] "
            + json.dumps(self.diagnostics, sort_keys=True, default=str)
        )


class ComRecGCGraphHashCollisionError(RuntimeError):
    """Raised when one official hash is observed with two different graphs."""


def _key_blob(key: Any) -> bytes:
    return pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)


def _key_id(key: Any) -> str:
    return hashlib.sha256(_key_blob(key)).hexdigest()


def _entry_blob(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _entry_graph_sha256(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("COMRECGC graph_map values must contain a graph at index 0.")
    return stable_untyped_graph_sha256(value[0])


class AuthoritativeGraphStore:
    """SQLite graph store with explicit writer and immutable-reader modes."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"COMRECGC authoritative graph store missing: {self.path}"
                )
            wal_path = Path(f"{self.path}-wal")
            if wal_path.exists() and wal_path.stat().st_size > 0:
                raise RuntimeError(
                    "Immutable COMRECGC graph-store audit rejects a non-empty WAL: "
                    f"{wal_path}"
                )
            self._connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro&immutable=1",
                uri=True,
                timeout=60.0,
            )
            self._connection.execute("PRAGMA query_only=ON")
            self._connection.execute("PRAGMA busy_timeout=60000")
            self.write_count = 0
            self.read_count = 0
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=60.0)
        self._connection.execute("PRAGMA busy_timeout=60000")
        journal_mode = str(
            self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        ).lower()
        if journal_mode != "wal":
            self._connection.close()
            raise RuntimeError(
                "COMRECGC authoritative graph store requires SQLite WAL support: "
                f"path={self.path}, journal_mode={journal_mode!r}"
            )
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA wal_autocheckpoint=1000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS graphs (
                key_id TEXT PRIMARY KEY,
                key_blob BLOB NOT NULL,
                value_blob BLOB NOT NULL,
                value_sha256 TEXT NOT NULL,
                graph_sha256 TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self.write_count = 0
        self.read_count = 0

    def close(self) -> None:
        if self.read_only:
            self._connection.close()
            return
        self._connection.commit()
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._connection.close()

    @property
    def checkpoint_connection(self) -> sqlite3.Connection:
        """Expose the single writer solely for SQLite's consistent backup API."""

        if self.read_only:
            raise RuntimeError("Read-only COMRECGC graph stores cannot be checkpointed.")
        return self._connection

    def checkpoint_wal(self, *, truncate: bool = False) -> dict[str, int]:
        """Flush committed WAL pages without weakening durability."""

        if self.read_only:
            raise RuntimeError("Read-only COMRECGC graph stores cannot checkpoint WAL.")
        mode = "TRUNCATE" if truncate else "PASSIVE"
        busy, log_pages, checkpointed_pages = self._connection.execute(
            f"PRAGMA wal_checkpoint({mode})"
        ).fetchone()
        return {
            "busy": int(busy),
            "log_pages": int(log_pages),
            "checkpointed_pages": int(checkpointed_pages),
        }

    def contains(self, key: Any) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM graphs WHERE key_id = ?", (_key_id(key),)
        ).fetchone()
        return row is not None

    def metadata(self, key: Any) -> dict[str, str] | None:
        row = self._connection.execute(
            "SELECT value_sha256, graph_sha256 FROM graphs WHERE key_id = ?",
            (_key_id(key),),
        ).fetchone()
        if row is None:
            return None
        return {"value_sha256": str(row[0]), "graph_sha256": str(row[1])}

    def put(self, key: Any, value: Any) -> dict[str, str]:
        if self.read_only:
            raise RuntimeError("Read-only COMRECGC graph stores reject writes.")
        key_blob = _key_blob(key)
        value_blob = _entry_blob(value)
        value_sha256 = hashlib.sha256(value_blob).hexdigest()
        graph_sha256 = _entry_graph_sha256(value)
        existing = self.metadata(key)
        if existing is not None:
            if existing["graph_sha256"] != graph_sha256:
                raise ComRecGCGraphHashCollisionError(
                    "[COMRECGC_GRAPH_HASH_COLLISION] "
                    f"hash={key!r} existing_graph_sha256={existing['graph_sha256']} "
                    f"new_graph_sha256={graph_sha256}"
                )
            return existing
        with self._connection:
            self._connection.execute(
                "INSERT INTO graphs VALUES (?, ?, ?, ?, ?)",
                (_key_id(key), key_blob, value_blob, value_sha256, graph_sha256),
            )
        self.write_count += 1
        return {"value_sha256": value_sha256, "graph_sha256": graph_sha256}

    def get(self, key: Any) -> Any:
        row = self._connection.execute(
            "SELECT key_blob, value_blob, value_sha256, graph_sha256 "
            "FROM graphs WHERE key_id = ?",
            (_key_id(key),),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        key_blob, value_blob, expected_value_sha, expected_graph_sha = row
        if pickle.loads(key_blob) != key:
            raise RuntimeError(f"COMRECGC backing-store key digest mismatch: {key!r}")
        actual_value_sha = hashlib.sha256(value_blob).hexdigest()
        if actual_value_sha != expected_value_sha:
            raise RuntimeError(f"COMRECGC backing-store value checksum mismatch: {key!r}")
        value = pickle.loads(value_blob)
        actual_graph_sha = _entry_graph_sha256(value)
        if actual_graph_sha != expected_graph_sha:
            raise RuntimeError(f"COMRECGC backing-store graph checksum mismatch: {key!r}")
        self.read_count += 1
        return value

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM graphs").fetchone()[0])

    def stored_keys(self) -> set[Any]:
        """Return one in-memory index for bounded integrity scans.

        Transition tables can contain millions of destinations.  Querying
        SQLite once per destination would turn the final integrity gate into a
        second long-running workload, while the authoritative store normally
        contains only the comparatively small evicted subset.
        """

        return {
            pickle.loads(row[0])
            for row in self._connection.execute("SELECT key_blob FROM graphs")
        }

    def find_keys_by_graph_sha256(self, graph_sha256: str) -> list[Any]:
        """Return exact stored keys for one canonical graph fingerprint."""

        return [
            pickle.loads(row[0])
            for row in self._connection.execute(
                "SELECT key_blob FROM graphs WHERE graph_sha256 = ? ORDER BY key_id",
                (str(graph_sha256),),
            )
        ]

    def integrity_audit(self) -> dict[str, Any]:
        integrity = str(self._connection.execute("PRAGMA integrity_check").fetchone()[0])
        digest = hashlib.sha256()
        row_count = 0
        for key_id, value_blob, value_sha256, graph_sha256 in self._connection.execute(
            "SELECT key_id, value_blob, value_sha256, graph_sha256 FROM graphs ORDER BY key_id"
        ):
            actual_value_sha = hashlib.sha256(value_blob).hexdigest()
            if actual_value_sha != value_sha256:
                raise RuntimeError(f"COMRECGC backing-store checksum mismatch: {key_id}")
            value = pickle.loads(value_blob)
            if _entry_graph_sha256(value) != graph_sha256:
                raise RuntimeError(f"COMRECGC backing-store graph mismatch: {key_id}")
            digest.update(str(key_id).encode("ascii"))
            digest.update(str(value_sha256).encode("ascii"))
            digest.update(str(graph_sha256).encode("ascii"))
            row_count += 1
        wal_path = Path(f"{self.path}-wal")
        shm_path = Path(f"{self.path}-shm")
        return {
            "integrity_check": integrity,
            "integrity_passed": integrity == "ok",
            "entry_count": row_count,
            "content_sha256": digest.hexdigest(),
            "path": str(self.path),
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
            "wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
            "shm_bytes": shm_path.stat().st_size if shm_path.exists() else 0,
            "journal_mode": str(
                self._connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower(),
        }


class LiveGraphMap(dict[Any, Any]):
    """Active upstream graph map with lossless fallback resolution.

    ``in``/``keys``/``len`` deliberately retain upstream *active candidate*
    semantics.  ``resolve`` and ``get`` additionally consult the authoritative
    store.  This distinction lets upstream's existing final-loop restoration
    run at the same point while graph reads remain lossless.
    """

    def __init__(
        self,
        module: Any,
        values: Mapping[Any, Any],
        *,
        store: AuthoritativeGraphStore,
        seed: int,
    ) -> None:
        super().__init__()
        self._module = module
        self.store = store
        self.seed = int(seed)
        self.pin_counts: Counter[Any] = Counter()
        self.deferred_deletions: set[Any] = set()
        self.current_step = 0
        self.current_graph_hashes: tuple[Any, ...] = ()
        self.eviction_attempts = 0
        self.eviction_committed = 0
        self.eviction_deferred = 0
        self.active_eviction_prevented = 0
        self.deferred_flushed = 0
        self.rehydrations = 0
        self.unresolved_lookups = 0
        self.missing_unmaterialized_eviction_count = 0
        self.max_hot_cache_size = 0
        self.recent_evictions: deque[dict[str, Any]] = deque(maxlen=32)
        for key, value in values.items():
            self.__setitem__(key, value)

    def __setitem__(self, key: Any, value: Any) -> None:
        new_sha = _entry_graph_sha256(value)
        if dict.__contains__(self, key):
            existing_sha = _entry_graph_sha256(dict.__getitem__(self, key))
            if existing_sha != new_sha:
                raise ComRecGCGraphHashCollisionError(
                    "[COMRECGC_GRAPH_HASH_COLLISION] "
                    f"hash={key!r} existing_graph_sha256={existing_sha} new_graph_sha256={new_sha}"
                )
        stored = self.store.metadata(key)
        if stored is not None and stored["graph_sha256"] != new_sha:
            raise ComRecGCGraphHashCollisionError(
                "[COMRECGC_GRAPH_HASH_COLLISION] "
                f"hash={key!r} stored_graph_sha256={stored['graph_sha256']} new_graph_sha256={new_sha}"
            )
        dict.__setitem__(self, key, value)
        self.max_hot_cache_size = max(self.max_hot_cache_size, dict.__len__(self))

    def __getitem__(self, key: Any) -> Any:
        return self.resolve(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if not dict.__contains__(self, key) and not self.store.contains(key):
            return default
        return self.resolve(key)

    def __delitem__(self, key: Any) -> None:
        self.eviction_attempts += 1
        if not dict.__contains__(self, key):
            candidates = getattr(self._module, "counterfactual_candidates", ())
            tail_hash = candidates[-1].get("graph_hash") if candidates else None
            index_map = getattr(self._module, "graph_index_map", {})
            if key != tail_hash or key in index_map:
                raise KeyError(key)
            self.missing_unmaterialized_eviction_count += 1
            return
        value = dict.__getitem__(self, key)
        metadata = self.store.put(key, value)
        if self.pin_counts[key] > 0:
            self.active_eviction_prevented += 1
            self.eviction_deferred += 1
            self.deferred_deletions.add(key)
        dict.__delitem__(self, key)
        self.eviction_committed += 1
        self.recent_evictions.append(
            {
                "graph_hash": str(key),
                "current_step": self.current_step,
                "was_pinned": self.pin_counts[key] > 0,
                "graph_sha256": metadata["graph_sha256"],
            }
        )

    def resolve(self, key: Any) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        try:
            value = self.store.get(key)
        except KeyError as exc:
            self.unresolved_lookups += 1
            raise ComRecGCLiveGraphResolutionError(key, self.diagnostics(key)) from exc
        self.rehydrations += 1
        return value

    def contains_resolvable(self, key: Any) -> bool:
        return dict.__contains__(self, key) or self.store.contains(key)

    def resolve_many(self, keys: Sequence[Any]) -> list[Any]:
        return [self.resolve(key) for key in keys]

    @contextmanager
    def pin_many(self, keys: Sequence[Any]) -> Iterator[None]:
        values = tuple(keys)
        for key in values:
            self.pin_counts[key] += 1
        try:
            yield
        finally:
            for key in values:
                self.pin_counts[key] -= 1
                if self.pin_counts[key] <= 0:
                    self.pin_counts.pop(key, None)
            self.flush_deferred()

    def flush_deferred(self) -> None:
        for key in tuple(self.deferred_deletions):
            if self.pin_counts[key] == 0:
                self.deferred_deletions.remove(key)
                self.deferred_flushed += 1

    def begin_move(self, graph_hashes: Sequence[Any], *, current_step: int) -> None:
        self.current_step = int(current_step)
        self.current_graph_hashes = tuple(graph_hashes)

    def end_move(self) -> None:
        self.current_graph_hashes = ()
        self.flush_deferred()

    def live_reference_hashes(self) -> set[Any]:
        live = set(self.current_graph_hashes) | set(self.pin_counts)
        transitions = getattr(self._module, "transitions", {})
        live.update(transitions.keys())
        for transition in transitions.values():
            if isinstance(transition, tuple) and transition:
                live.update(transition[0])
        return live

    def transition_integrity(self) -> dict[str, int]:
        transitions = getattr(self._module, "transitions", {})
        stored_keys = self.store.stored_keys()
        stored_keys_by_string = {str(value): value for value in stored_keys}
        if len(stored_keys_by_string) != len(stored_keys):
            raise ComRecGCGraphHashCollisionError(
                "[COMRECGC_GRAPH_HASH_COLLISION] backing store contains "
                "ambiguous string-normalized official hashes"
            )
        unresolved_sources = 0
        invalid_destinations = 0
        destination_count = 0
        for source_hash, transition in transitions.items():
            if not self.contains_resolvable(source_hash):
                unresolved_sources += 1
            if not isinstance(transition, tuple) or len(transition) < 2:
                invalid_destinations += 1
                continue
            hashes, graphs = transition[0], transition[1]
            if len(hashes) != len(graphs):
                invalid_destinations += abs(len(hashes) - len(graphs)) or 1
                continue
            destination_count += len(hashes)
            for graph_hash, graph in zip(hashes, graphs, strict=True):
                stored_key = stored_keys_by_string.get(str(graph_hash))
                if stored_key is None:
                    continue
                stored = self.store.metadata(stored_key)
                if stored is None or stored["graph_sha256"] != stable_untyped_graph_sha256(graph):
                    invalid_destinations += 1
        return {
            "transition_source_count": len(transitions),
            "transition_destination_count": destination_count,
            "unresolved_transition_source_count": unresolved_sources,
            "invalid_transition_destination_count": invalid_destinations,
        }

    def diagnostics(self, key: Any | None = None) -> dict[str, Any]:
        transitions = getattr(self._module, "transitions", {})
        referenced_by = []
        for source_hash, transition in transitions.items():
            if source_hash == key:
                referenced_by.append({"kind": "transition_source", "source": str(source_hash)})
            if isinstance(transition, tuple) and transition and key in transition[0]:
                referenced_by.append({"kind": "transition_destination", "source": str(source_hash)})
                if len(referenced_by) >= 32:
                    break
        return {
            "graph_hash": str(key),
            "current_step": self.current_step,
            "seed": self.seed,
            "current_graph_hashes": [str(value) for value in self.current_graph_hashes],
            "transition_references": referenced_by,
            "hot_cache_size": dict.__len__(self),
            "backing_store_size": self.store.count(),
            "pin_count": sum(self.pin_counts.values()),
            "deferred_deletion_count": len(self.deferred_deletions),
            "recent_eviction_record": self.recent_evictions[-1] if self.recent_evictions else None,
            "unresolved_lookups": self.unresolved_lookups,
        }

    def audit(self) -> dict[str, Any]:
        transition = self.transition_integrity()
        backing = self.store.integrity_audit()
        return {
            "policy": GRAPH_STATE_POLICY,
            "hot_cache_size": dict.__len__(self),
            "max_hot_cache_size": self.max_hot_cache_size,
            "backing_store_size": self.store.count(),
            "pins": sum(self.pin_counts.values()),
            "deferred_deletions": len(self.deferred_deletions),
            "eviction_attempts": self.eviction_attempts,
            "eviction_committed": self.eviction_committed,
            "eviction_deferred": self.eviction_deferred,
            "active_eviction_prevented": self.active_eviction_prevented,
            "deferred_flushed": self.deferred_flushed,
            "rehydrations": self.rehydrations,
            "unresolved_lookups": self.unresolved_lookups,
            "missing_unmaterialized_eviction_count": self.missing_unmaterialized_eviction_count,
            "recent_evictions": list(self.recent_evictions),
            "backing_store": backing,
            **transition,
            "rng_calls_added": 0,
            "candidate_order_changed": False,
            "scientific_parameters_changed": False,
        }

    def runtime_diagnostics(self) -> dict[str, Any]:
        """Return counters suitable for frequent progress heartbeats.

        The full ``audit`` deliberately verifies every backing-store row.  A
        long random walk must not repeat that O(database size) scan every few
        hundred moves, so storage monitoring uses this constant-time view.
        """

        return {
            "policy": GRAPH_STATE_POLICY,
            "hot_cache_size": dict.__len__(self),
            "max_hot_cache_size": self.max_hot_cache_size,
            "backing_store_size": self.store.count(),
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
        }

    def export_checkpoint_state(self) -> dict[str, Any]:
        if self.pin_counts or self.deferred_deletions or self.current_graph_hashes:
            raise RuntimeError("COMRECGC live graph checkpoint requested inside a move.")
        return {
            "schema_version": LIVE_GRAPH_CHECKPOINT_SCHEMA,
            "seed": self.seed,
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
            "store_write_count": self.store.write_count,
            "store_read_count": self.store.read_count,
        }

    def restore_checkpoint_state(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != LIVE_GRAPH_CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported COMRECGC live graph checkpoint schema.")
        if int(value.get("seed", -1)) != self.seed:
            raise ValueError("COMRECGC live graph checkpoint seed differs.")
        if self.pin_counts or self.deferred_deletions or self.current_graph_hashes:
            raise RuntimeError("COMRECGC live graph restore target is active.")
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
            setattr(self, name, int(value[name]))
        if self.max_hot_cache_size < dict.__len__(self):
            raise ValueError("COMRECGC live graph maximum hot size is inconsistent.")
        self.recent_evictions = deque(value.get("recent_evictions") or (), maxlen=32)
        self.store.write_count = int(value.get("store_write_count", 0))
        self.store.read_count = int(value.get("store_read_count", 0))

    def __reduce__(self) -> tuple[Any, tuple[dict[Any, Any]]]:
        return dict, (dict(self),)


class LiveGraphState:
    """Own the active map, authoritative store, and scoped move pins."""

    def __init__(
        self,
        module: Any,
        values: Mapping[Any, Any],
        *,
        store_path: str | Path,
        seed: int,
        on_step: Callable[[int, "LiveGraphState"], None] | None = None,
    ) -> None:
        self.module = module
        self.store = AuthoritativeGraphStore(store_path)
        self.graph_map = LiveGraphMap(module, values, store=self.store, seed=seed)
        self.move_count = 0
        self._on_step = on_step

    def resolve_graph(self, graph_hash: Any) -> Any:
        return self.graph_map.resolve(graph_hash)[0]

    def resolve_graphs(self, graph_hashes: Sequence[Any]) -> list[Any]:
        return [self.resolve_graph(value) for value in graph_hashes]

    def contains(self, graph_hash: Any) -> bool:
        return self.graph_map.contains_resolvable(graph_hash)

    @contextmanager
    def pin_many(self, graph_hashes: Sequence[Any]) -> Iterator[None]:
        with self.graph_map.pin_many(graph_hashes):
            yield

    def wrap_move(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            graph_hashes = list(kwargs.get("graphs_hash", args[0] if args else []))
            self.move_count += 1
            self.graph_map.begin_move(graph_hashes, current_step=self.move_count)
            try:
                with self.pin_many(graph_hashes):
                    result = original(*args, **kwargs)
            finally:
                self.graph_map.end_move()
            if self._on_step is not None:
                self._on_step(self.move_count, self)
            return result

        return wrapped

    def audit(self) -> dict[str, Any]:
        return {"move_count": self.move_count, **self.graph_map.audit()}

    def runtime_diagnostics(self) -> dict[str, Any]:
        return {
            "move_count": self.move_count,
            **self.graph_map.runtime_diagnostics(),
        }

    def export_checkpoint_state(self) -> dict[str, Any]:
        state = self.graph_map.export_checkpoint_state()
        if self.move_count != int(state["current_step"]):
            raise RuntimeError("COMRECGC live graph move counters are inconsistent.")
        return {**state, "move_count": self.move_count}

    def restore_checkpoint_state(self, value: Mapping[str, Any]) -> None:
        self.graph_map.restore_checkpoint_state(value)
        self.move_count = int(value.get("move_count", -1))
        if self.move_count < 0 or self.move_count != self.graph_map.current_step:
            raise ValueError("COMRECGC live graph restored move count is inconsistent.")

    def close(self) -> None:
        self.store.close()


def graph_state(module: Any) -> LiveGraphState | None:
    value = getattr(module, "comrecgc_live_graph_state", None)
    return value if isinstance(value, LiveGraphState) else None


def resolve_graph(module: Any, graph_hash: Any) -> Any:
    state = graph_state(module)
    if state is not None:
        return state.resolve_graph(graph_hash)
    try:
        return module.graph_map[graph_hash][0]
    except KeyError as exc:
        raise ComRecGCLiveGraphResolutionError(
            graph_hash,
            {
                "graph_hash": str(graph_hash),
                "current_step": None,
                "seed": None,
                "current_graph_hashes": [],
                "transition_references": [],
                "hot_cache_size": len(getattr(module, "graph_map", {})),
                "backing_store_size": 0,
                "pin_count": 0,
                "deferred_deletion_count": 0,
                "recent_eviction_record": None,
                "unresolved_lookups": 1,
            },
        ) from exc


def resolve_graphs(module: Any, graph_hashes: Sequence[Any]) -> list[Any]:
    return [resolve_graph(module, value) for value in graph_hashes]


@contextmanager
def pin_graphs(module: Any, graph_hashes: Sequence[Any]) -> Iterator[None]:
    state = graph_state(module)
    if state is None:
        yield
        return
    with state.pin_many(graph_hashes):
        yield


def current_rss_mib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None
