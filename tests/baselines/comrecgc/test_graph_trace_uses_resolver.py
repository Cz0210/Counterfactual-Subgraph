from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc import graph_trace
from src.baselines.comrecgc.live_graph_state import LiveGraphState
from src.baselines.comrecgc.transition_cache import CompactMoveScopedTransitionMap


def test_graph_trace_has_no_raw_module_graph_map_index() -> None:
    source = inspect.getsource(graph_trace)
    assert "module.graph_map[" not in source
    assert "resolve_graph(module" in source
    assert "resolve_graphs(module" in source


def test_compact_transition_active_source_uses_backing_resolver(tmp_path) -> None:
    graph = SimpleNamespace(
        x=np.asarray([[1], [2]], dtype=np.int64),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
    )
    module = SimpleNamespace(
        graph_map={}, graph_index_map={}, counterfactual_candidates=[], transitions={}
    )
    state = LiveGraphState(module, {}, store_path=tmp_path / "graphs.sqlite3", seed=0)
    module.graph_map = state.graph_map
    module.comrecgc_live_graph_state = state
    module.graph_map[42] = [graph, np.asarray([1.0]), np.asarray([2.0])]
    del module.graph_map[42]
    cache = CompactMoveScopedTransitionMap(
        module,
        {},
        seed=0,
        expanded_capacity=1,
        rebuild_target=lambda source, _action: source,
    )
    try:
        cache.begin_move([42])
        assert cache._source_graph(42) is not None
        assert state.graph_map.rehydrations >= 1
        assert cache.audit()["graph_source_resolution"] == (
            "unified_live_graph_resolver_v3"
        )
        cache.end_move()
    finally:
        state.close()
