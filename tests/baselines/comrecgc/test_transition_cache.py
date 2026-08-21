from __future__ import annotations

import random
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.transition_cache import (
    COMPACT_TRANSITION_CACHE_PATCH,
    CompactMoveScopedTransitionMap,
)


@dataclass
class TinyGraph:
    value: int


def _rebuild(source: TinyGraph, action: tuple[object, ...]) -> TinyGraph:
    assert action[0] == "ADD"
    return TinyGraph(source.value + int(action[1]))


def _transition_rows(source: TinyGraph, count: int) -> tuple[object, ...]:
    targets = [TinyGraph(source.value + index + 1) for index in range(count)]
    return (
        [f"target-{source.value}-{index}" for index in range(count)],
        targets,
        [np.asarray([0.1 + index / 100.0, 1.0]) for index in range(count)],
        [np.asarray([source.value, index], dtype=np.float32) for index in range(count)],
    )


def _record_actions(
    cache: CompactMoveScopedTransitionMap, transition: tuple[object, ...]
) -> None:
    for index, graph in enumerate(transition[1]):
        cache.record_enumerated(graph, ("ADD", index + 1))


def test_compact_transition_cache_replays_exact_rows_without_model_calls() -> None:
    source = TinyGraph(10)
    module = SimpleNamespace(
        graph_map={"source": [source, None, None]},
        graph_index_map={"source": 0},
    )
    cache = CompactMoveScopedTransitionMap(
        module,
        {},
        seed=0,
        expanded_capacity=1,
        rebuild_target=_rebuild,
    )
    transition = _transition_rows(source, 4)
    _record_actions(cache, transition)
    cache["source"] = transition
    expected = cache["source"]

    cache.clear_expanded()
    replayed = cache["source"]

    assert replayed[0] == expected[0]
    assert [graph.value for graph in replayed[1]] == [graph.value for graph in expected[1]]
    np.testing.assert_array_equal(np.asarray(replayed[2]), np.asarray(expected[2]))
    np.testing.assert_array_equal(np.asarray(replayed[3]), np.asarray(expected[3]))
    assert cache.action_records("source", "target-10-2") == [
        {"action": ["ADD", 3]}
    ]
    audit = cache.audit()
    assert audit["patch"] == COMPACT_TRANSITION_CACHE_PATCH
    assert audit["model_recomputation_count"] == 0
    assert audit["rng_calls_added"] == 0
    assert audit["neighbor_order_changed"] is False
    assert audit["candidate_order_changed"] is False


def test_compact_transition_cache_preserves_1000_random_choices() -> None:
    source = TinyGraph(20)
    module = SimpleNamespace(
        graph_map={"source": [source, None, None]},
        graph_index_map={"source": 0},
    )
    cache = CompactMoveScopedTransitionMap(
        module,
        {},
        seed=13,
        expanded_capacity=1,
        rebuild_target=_rebuild,
    )
    baseline = _transition_rows(source, 8)
    _record_actions(cache, baseline)
    cache["source"] = baseline
    cache.clear_expanded()
    cached = cache["source"]
    baseline_rng = random.Random(13)
    cached_rng = random.Random(13)
    weights = [float(row[0]) for row in baseline[2]]

    for _ in range(1_000):
        baseline_index = baseline_rng.choices(range(len(weights)), weights=weights)[0]
        cached_index = cached_rng.choices(
            range(len(cached[2])),
            weights=[float(row[0]) for row in cached[2]],
        )[0]
        assert cached_index == baseline_index
        assert cached[0][cached_index] == baseline[0][baseline_index]
        assert cached[1][cached_index].value == baseline[1][baseline_index].value


def test_compact_transition_cache_bounds_complete_neighbor_graphs() -> None:
    graph_map = {
        f"source-{index}": [TinyGraph(index * 100), None, None]
        for index in range(100)
    }
    module = SimpleNamespace(
        graph_map=graph_map,
        graph_index_map={key: index for index, key in enumerate(graph_map)},
    )
    cache = CompactMoveScopedTransitionMap(
        module,
        {},
        seed=0,
        expanded_capacity=5,
        rebuild_target=_rebuild,
    )
    for key, graph_entry in graph_map.items():
        transition = _transition_rows(graph_entry[0], 20)
        _record_actions(cache, transition)
        cache[key] = transition

    audit = cache.audit()
    assert audit["transition_entry_count"] == 100
    assert audit["max_expanded_entry_count"] == 5
    assert audit["max_expanded_graph_count"] == 100
    assert audit["compact_numeric_bytes"] > 0


def test_compact_transition_cache_defers_active_deletion() -> None:
    source = TinyGraph(5)
    module = SimpleNamespace(
        graph_map={"source": [source, None, None]},
        graph_index_map={"source": 0},
    )
    cache = CompactMoveScopedTransitionMap(
        module,
        {},
        seed=0,
        expanded_capacity=1,
        rebuild_target=_rebuild,
    )
    transition = _transition_rows(source, 2)
    _record_actions(cache, transition)
    cache["source"] = transition
    cache.begin_move(["source"])
    module.graph_index_map.clear()
    module.graph_map.clear()
    del cache["source"]

    assert cache["source"][0] == transition[0]
    cache.end_move()
    assert "source" not in cache
    assert cache.audit()["applied_deferred_deletion_count"] == 1


def test_compact_transition_cache_rejects_untracked_target_action() -> None:
    source = TinyGraph(0)
    module = SimpleNamespace(
        graph_map={"source": [source, None, None]},
        graph_index_map={"source": 0},
    )
    cache = CompactMoveScopedTransitionMap(
        module,
        {},
        seed=0,
        expanded_capacity=1,
        rebuild_target=_rebuild,
    )

    with pytest.raises(RuntimeError, match="exact enumerated action"):
        cache["source"] = _transition_rows(source, 1)


def test_compact_transition_checkpoint_restores_entries_lru_and_counters() -> None:
    sources = {
        "one": [TinyGraph(10), None, None],
        "two": [TinyGraph(20), None, None],
    }
    module = SimpleNamespace(graph_map=sources, graph_index_map={"one": 0, "two": 1})
    cache = CompactMoveScopedTransitionMap(
        module, {}, seed=7, expanded_capacity=2, rebuild_target=_rebuild
    )
    for key in ("one", "two"):
        transition = _transition_rows(sources[key][0], 3)
        _record_actions(cache, transition)
        cache[key] = transition
    cache.begin_move(["one"])
    cache["one"]
    cache.end_move()
    expected = cache.export_checkpoint_state()

    restored = CompactMoveScopedTransitionMap(
        module, {}, seed=7, expanded_capacity=2, rebuild_target=_rebuild
    )
    restored.restore_checkpoint_state(expected)

    assert restored.export_checkpoint_state()["expanded_keys"] == ["two", "one"]
    assert restored.export_checkpoint_state()["counters"] == expected["counters"]
    assert restored["one"][0] == cache["one"][0]


def test_compact_transition_checkpoint_rejects_active_move() -> None:
    source = TinyGraph(1)
    module = SimpleNamespace(
        graph_map={"source": [source, None, None]}, graph_index_map={"source": 0}
    )
    cache = CompactMoveScopedTransitionMap(
        module, {}, seed=0, expanded_capacity=1, rebuild_target=_rebuild
    )
    cache.begin_move(["source"])
    with pytest.raises(RuntimeError, match="inside an active move"):
        cache.export_checkpoint_state()
    cache.end_move()
