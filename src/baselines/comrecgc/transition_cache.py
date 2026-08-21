"""Memory-bounded transition storage for the pinned COMRECGC random walk."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import resource
import sys
from typing import Any, Callable

import numpy as np


COMPACT_TRANSITION_CACHE_PATCH = "compact_transition_action_replay_lru_v1"
COMPACT_TRANSITION_CHECKPOINT_SCHEMA = "comrecgc_compact_transition_state_v1"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _freeze_action(action: Sequence[Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for value in action:
        if isinstance(value, np.generic):
            value = value.item()
        values.append(value)
    return tuple(values)


def _stack_numeric_rows(values: Sequence[Any], *, field: str) -> np.ndarray:
    rows = [np.asarray(value) for value in values]
    if not rows:
        return np.empty((0,), dtype=np.float64)
    try:
        result = np.stack(rows, axis=0)
    except ValueError as exc:
        raise RuntimeError(
            f"COMRECGC compact transition {field} rows have inconsistent shapes."
        ) from exc
    if result.dtype == object or not np.isfinite(result).all():
        raise RuntimeError(
            f"COMRECGC compact transition {field} must be finite numeric data."
        )
    return np.array(result, copy=True, order="C")


@dataclass(frozen=True)
class _CompactTransition:
    target_hashes: tuple[Any, ...]
    actions: tuple[tuple[Any, ...], ...]
    importance_parts: np.ndarray
    embeddings: np.ndarray

    @property
    def numeric_bytes(self) -> int:
        return int(self.importance_parts.nbytes + self.embeddings.nbytes)


class CompactMoveScopedTransitionMap(MutableMapping[Any, Any]):
    """Retain transition numerics while bounding complete neighbor graphs.

    Pinned upstream stores every neighbor ``Data`` object in ``transitions``.
    This mapping captures the exact enumerated edit action, keeps the already
    computed hashes/importances/embeddings, and deterministically reconstructs
    neighbor graphs after their small expanded LRU entry is evicted.  No model
    call, random draw, neighbor re-enumeration, or ordering decision is repeated.
    """

    def __init__(
        self,
        module: Any,
        values: Mapping[Any, Any],
        *,
        seed: int,
        expanded_capacity: int,
        rebuild_target: Callable[[Any, tuple[Any, ...]], Any],
    ) -> None:
        if values:
            raise ValueError(
                "Compact COMRECGC transition storage must be installed before "
                "the full random walk creates transitions."
            )
        if int(expanded_capacity) <= 0:
            raise ValueError("COMRECGC expanded transition capacity must be positive.")
        self._module = module
        self._seed = int(seed)
        self._expanded_capacity = int(expanded_capacity)
        self._rebuild_target = rebuild_target
        self._entries: dict[Any, _CompactTransition] = {}
        self._expanded: OrderedDict[Any, tuple[Any, ...]] = OrderedDict()
        self._actions_by_object_id: dict[int, tuple[Any, ...]] = {}
        self._active_keys: tuple[Any, ...] = ()
        self._active_graphs: dict[Any, Any] = {}
        self._deferred_deletions: set[Any] = set()
        self._current_step = 0
        self.move_count = 0
        self.deferred_deletion_count = 0
        self.applied_deferred_deletion_count = 0
        self.cancelled_deferred_deletion_count = 0
        self.missing_lookup_count = 0
        self.max_transition_size = 0
        self.max_expanded_entry_count = 0
        self.max_expanded_graph_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.reconstructed_graph_count = 0
        self.captured_action_count = 0

    def record_enumerated(self, target_graph: Any, action: Sequence[Any]) -> None:
        object_id = id(target_graph)
        if object_id in self._actions_by_object_id:
            raise RuntimeError(
                "COMRECGC neighbor object was recorded with more than one action."
            )
        self._actions_by_object_id[object_id] = _freeze_action(action)
        self.captured_action_count += 1

    def _remember_expanded(self, key: Any, value: tuple[Any, ...]) -> None:
        self._expanded.pop(key, None)
        self._expanded[key] = value
        while len(self._expanded) > self._expanded_capacity:
            self._expanded.popitem(last=False)
        self.max_expanded_entry_count = max(
            self.max_expanded_entry_count, len(self._expanded)
        )
        graph_count = sum(len(item[1]) for item in self._expanded.values())
        self.max_expanded_graph_count = max(self.max_expanded_graph_count, graph_count)

    def _drop(self, key: Any) -> None:
        del self._entries[key]
        self._expanded.pop(key, None)

    def begin_move(self, graph_hashes: Sequence[Any]) -> int:
        if self._active_keys or self._deferred_deletions:
            raise RuntimeError("COMRECGC transition move scopes cannot be nested.")
        self.move_count += 1
        self._current_step = self.move_count
        self._active_keys = tuple(graph_hashes)
        # Resolve through the same fail-closed authority used by graph tracing.
        # ``LiveGraphMap.__contains__`` intentionally reflects only the bounded
        # hot set, so membership checks followed by raw indexing would bypass
        # the authoritative backing store after an eviction.
        from .live_graph_state import resolve_graph

        self._active_graphs = {
            key: resolve_graph(self._module, key) for key in self._active_keys
        }
        return self._current_step

    def end_move(self) -> None:
        index_map = getattr(self._module, "graph_index_map", {})
        for key in tuple(self._deferred_deletions):
            if key in index_map:
                self.cancelled_deferred_deletion_count += 1
            elif key in self._entries:
                self._drop(key)
                self.applied_deferred_deletion_count += 1
        self._deferred_deletions.clear()
        self._active_graphs.clear()
        self._active_keys = ()

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._entries)

    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(value, tuple) or len(value) != 4:
            raise RuntimeError(
                "COMRECGC compact transition values must use the official four-part tuple."
            )
        target_hashes, target_graphs, importance_parts, embeddings = value
        if not (
            len(target_hashes)
            == len(target_graphs)
            == len(importance_parts)
            == len(embeddings)
        ):
            raise RuntimeError("COMRECGC transition tuple rows are not aligned.")
        try:
            actions = tuple(
                self._actions_by_object_id[id(graph)] for graph in target_graphs
            )
        except KeyError as exc:
            raise RuntimeError(
                "COMRECGC compact transition cache could not resolve an exact "
                "enumerated action for a target graph."
            ) from exc
        finally:
            self._actions_by_object_id.clear()
        entry = _CompactTransition(
            target_hashes=tuple(target_hashes),
            actions=actions,
            importance_parts=_stack_numeric_rows(
                importance_parts, field="importance_parts"
            ),
            embeddings=_stack_numeric_rows(embeddings, field="embeddings"),
        )
        self._entries[key] = entry
        self._remember_expanded(
            key,
            (
                list(entry.target_hashes),
                list(target_graphs),
                [row for row in entry.importance_parts],
                [row for row in entry.embeddings],
            ),
        )
        self.max_transition_size = max(self.max_transition_size, len(self._entries))

    def _source_graph(self, key: Any) -> Any:
        if key in self._active_graphs:
            return self._active_graphs[key]
        from .live_graph_state import resolve_graph

        return resolve_graph(self._module, key)

    def __getitem__(self, key: Any) -> Any:
        if key not in self._entries:
            self.missing_lookup_count += 1
            head = self._active_keys.index(key) if key in self._active_keys else None
            graph_map = getattr(self._module, "graph_map", {})
            raise RuntimeError(
                "[COMRECGC_TRANSITION_STATE_ERROR] "
                f"current_step={self._current_step} head={head} seed={self._seed} "
                f"graph_hash={key} transition_size={len(self)} "
                f"cache_size={len(graph_map)} active_head_count={len(self._active_keys)}"
            )
        if key in self._expanded:
            self.cache_hit_count += 1
            value = self._expanded.pop(key)
            self._expanded[key] = value
            return value
        self.cache_miss_count += 1
        entry = self._entries[key]
        source_graph = self._source_graph(key)
        target_graphs = [
            self._rebuild_target(source_graph, action) for action in entry.actions
        ]
        self.reconstructed_graph_count += len(target_graphs)
        value = (
            list(entry.target_hashes),
            target_graphs,
            [row for row in entry.importance_parts],
            [row for row in entry.embeddings],
        )
        self._remember_expanded(key, value)
        return value

    def action_records(self, source_hash: Any, target_hash: Any) -> list[dict[str, Any]]:
        """Expose exact cached actions to the project trace without graph retention."""

        entry = self._entries.get(source_hash)
        if entry is None:
            return []
        return [
            {"action": list(action)}
            for resolved_hash, action in zip(
                entry.target_hashes, entry.actions, strict=True
            )
            if resolved_hash == target_hash
        ]

    def __delitem__(self, key: Any) -> None:
        if key not in self._entries:
            raise KeyError(key)
        if key in self._active_keys:
            if key not in self._deferred_deletions:
                self.deferred_deletion_count += 1
            self._deferred_deletions.add(key)
            return
        self._drop(key)

    def wrap_move(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            graph_hashes = list(kwargs.get("graphs_hash", args[0] if args else []))
            current_step = self.begin_move(graph_hashes)
            try:
                return original(*args, **kwargs)
            finally:
                self.end_move()
                if current_step == 1_000 or current_step % 10_000 == 0:
                    audit = self.audit()
                    print(
                        "[COMRECGC_COMPACT_TRANSITION_CACHE] "
                        f"current_step={current_step} seed={self._seed} "
                        f"transition_size={len(self)} "
                        f"expanded_entries={len(self._expanded)} "
                        f"expanded_graphs_max={audit['max_expanded_graph_count']} "
                        f"compact_numeric_bytes={audit['compact_numeric_bytes']} "
                        f"process_peak_rss_bytes={audit['process_peak_rss_bytes']} "
                        f"cache_hits={self.cache_hit_count} "
                        f"cache_misses={self.cache_miss_count}",
                        flush=True,
                    )

        return wrapped

    def audit(self) -> dict[str, Any]:
        return {
            "patch": COMPACT_TRANSITION_CACHE_PATCH,
            "policy": "exact_action_replay_with_bounded_expanded_lru",
            "move_count": self.move_count,
            "transition_entry_count": len(self._entries),
            "max_transition_size": self.max_transition_size,
            "expanded_capacity": self._expanded_capacity,
            "expanded_entry_count": len(self._expanded),
            "max_expanded_entry_count": self.max_expanded_entry_count,
            "max_expanded_graph_count": self.max_expanded_graph_count,
            "compact_numeric_bytes": sum(
                entry.numeric_bytes for entry in self._entries.values()
            ),
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "captured_action_count": self.captured_action_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "reconstructed_graph_count": self.reconstructed_graph_count,
            "deferred_deletion_count": self.deferred_deletion_count,
            "applied_deferred_deletion_count": self.applied_deferred_deletion_count,
            "cancelled_deferred_deletion_count": self.cancelled_deferred_deletion_count,
            "missing_lookup_count": self.missing_lookup_count,
            "graph_source_resolution": "unified_live_graph_resolver_v3",
            "model_recomputation_count": 0,
            "rng_calls_added": 0,
            "neighbor_order_changed": False,
            "candidate_order_changed": False,
            "scientific_parameters_changed": False,
        }

    def clear_expanded(self) -> None:
        self._expanded.clear()

    def export_checkpoint_state(self) -> dict[str, Any]:
        """Return an exact, graph-object-free state at a completed move boundary."""

        if (
            self._active_keys
            or self._active_graphs
            or self._deferred_deletions
            or self._actions_by_object_id
        ):
            raise RuntimeError(
                "COMRECGC transition checkpoint requested inside an active move."
            )
        entries = [
            {
                "source_hash": source_hash,
                "target_hashes": list(entry.target_hashes),
                "actions": [list(action) for action in entry.actions],
                "importance_parts": np.array(entry.importance_parts, copy=True),
                "embeddings": np.array(entry.embeddings, copy=True),
            }
            for source_hash, entry in self._entries.items()
        ]
        counters = {
            name: int(getattr(self, name))
            for name in (
                "move_count",
                "deferred_deletion_count",
                "applied_deferred_deletion_count",
                "cancelled_deferred_deletion_count",
                "missing_lookup_count",
                "max_transition_size",
                "max_expanded_entry_count",
                "max_expanded_graph_count",
                "cache_hit_count",
                "cache_miss_count",
                "reconstructed_graph_count",
                "captured_action_count",
            )
        }
        return {
            "schema_version": COMPACT_TRANSITION_CHECKPOINT_SCHEMA,
            "seed": self._seed,
            "expanded_capacity": self._expanded_capacity,
            "current_step": self._current_step,
            "entries": entries,
            "expanded_keys": list(self._expanded.keys()),
            "counters": counters,
        }

    def restore_checkpoint_state(self, value: Mapping[str, Any]) -> None:
        """Restore compact entries without replaying model calls or actions."""

        if value.get("schema_version") != COMPACT_TRANSITION_CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported COMRECGC compact transition checkpoint schema.")
        if int(value.get("seed", -1)) != self._seed or int(
            value.get("expanded_capacity", -1)
        ) != self._expanded_capacity:
            raise ValueError(
                "COMRECGC compact transition checkpoint provenance differs from runtime."
            )
        if (
            self._entries
            or self._expanded
            or self._active_keys
            or self._active_graphs
            or self._deferred_deletions
            or self._actions_by_object_id
        ):
            raise RuntimeError("COMRECGC compact transition restore target is not empty.")
        for row in value.get("entries") or ():
            source_hash = row["source_hash"]
            if source_hash in self._entries:
                raise ValueError("Duplicate COMRECGC transition source in checkpoint.")
            target_hashes = tuple(row["target_hashes"])
            actions = tuple(_freeze_action(action) for action in row["actions"])
            importance_parts = _stack_numeric_rows(
                row["importance_parts"], field="importance_parts"
            )
            embeddings = _stack_numeric_rows(row["embeddings"], field="embeddings")
            if not (
                len(target_hashes)
                == len(actions)
                == len(importance_parts)
                == len(embeddings)
            ):
                raise ValueError("COMRECGC compact transition checkpoint rows are misaligned.")
            self._entries[source_hash] = _CompactTransition(
                target_hashes=target_hashes,
                actions=actions,
                importance_parts=importance_parts,
                embeddings=embeddings,
            )
        expanded_keys = list(value.get("expanded_keys") or ())
        if len(set(expanded_keys)) != len(expanded_keys) or len(
            expanded_keys
        ) > self._expanded_capacity:
            raise ValueError("COMRECGC expanded transition LRU state is invalid.")
        for key in expanded_keys:
            entry = self._entries.get(key)
            if entry is None:
                raise ValueError("COMRECGC expanded transition key has no compact entry.")
            source_graph = self._source_graph(key)
            target_graphs = [
                self._rebuild_target(source_graph, action) for action in entry.actions
            ]
            self._remember_expanded(
                key,
                (
                    list(entry.target_hashes),
                    target_graphs,
                    [row for row in entry.importance_parts],
                    [row for row in entry.embeddings],
                ),
            )
        counters = value.get("counters")
        if not isinstance(counters, Mapping):
            raise ValueError("COMRECGC compact transition counters are missing.")
        for name in (
            "move_count",
            "deferred_deletion_count",
            "applied_deferred_deletion_count",
            "cancelled_deferred_deletion_count",
            "missing_lookup_count",
            "max_transition_size",
            "max_expanded_entry_count",
            "max_expanded_graph_count",
            "cache_hit_count",
            "cache_miss_count",
            "reconstructed_graph_count",
            "captured_action_count",
        ):
            setattr(self, name, int(counters[name]))
        self._current_step = int(value.get("current_step", self.move_count))
        if self.move_count != self._current_step:
            raise ValueError("COMRECGC compact transition move counter is inconsistent.")
        if self.max_transition_size < len(self._entries):
            raise ValueError("COMRECGC compact transition maximum size is inconsistent.")
