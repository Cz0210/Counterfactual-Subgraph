from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc.live_graph_state import LiveGraphState


def entry(index: int) -> list[object]:
    graph = SimpleNamespace(
        x=np.asarray([[index], [index + 1]], dtype=np.int64),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
    )
    return [graph, np.asarray([index], dtype=float), np.asarray([2.0])]


def test_pinned_hot_eviction_preserves_authoritative_value_and_flushes(tmp_path) -> None:
    module = SimpleNamespace(
        graph_map={}, graph_index_map={}, counterfactual_candidates=[], transitions={}
    )
    state = LiveGraphState(module, {}, store_path=tmp_path / "graphs.sqlite3", seed=0)
    module.graph_map = state.graph_map
    try:
        module.graph_map[1] = entry(1)
        with state.pin_many([1]):
            del module.graph_map[1]
            assert state.contains(1)
            assert state.graph_map.active_eviction_prevented == 1
            assert state.graph_map.deferred_deletions == {1}
        assert not state.graph_map.deferred_deletions
        assert state.graph_map.deferred_flushed == 1
        assert state.graph_map.eviction_committed == 1
    finally:
        state.close()


def test_2164128_revisit_after_eviction_remains_resolvable(tmp_path) -> None:
    module = SimpleNamespace(
        graph_map={}, graph_index_map={}, counterfactual_candidates=[], transitions={}
    )
    state = LiveGraphState(module, {}, store_path=tmp_path / "graphs.sqlite3", seed=0)
    module.graph_map = state.graph_map
    try:
        missing_hash = -5763365003180206704
        module.graph_map[missing_hash] = entry(2)
        with state.pin_many([missing_hash]):
            del module.graph_map[missing_hash]
        state.graph_map.begin_move([missing_hash], current_step=46690)
        assert state.resolve_graph(missing_hash).num_nodes == 2
        assert state.graph_map.unresolved_lookups == 0
    finally:
        state.close()
