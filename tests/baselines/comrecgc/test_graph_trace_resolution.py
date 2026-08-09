from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc.graph_trace import ActionTraceRecorder
from src.baselines.comrecgc.live_graph_state import LiveGraphState


def graph(label: int) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[label], [label + 1]], dtype=np.int64),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
        comrecgc_parent_id="parent",
    )


def test_graph_trace_uses_resolver_for_evicted_source_and_target(tmp_path) -> None:
    source = graph(1)
    target = graph(2)
    module = SimpleNamespace(
        graph_map={}, graph_index_map={}, counterfactual_candidates=[], transitions={}
    )
    state = LiveGraphState(module, {}, store_path=tmp_path / "graphs.sqlite3", seed=0)
    module.graph_map = state.graph_map
    module.comrecgc_live_graph_state = state
    module.graph_map["source"] = [source, np.asarray([1.0]), np.asarray([2.0])]
    module.graph_map["target"] = [target, np.asarray([2.0]), np.asarray([2.0])]
    del module.graph_map["source"]
    del module.graph_map["target"]
    recorder = ActionTraceRecorder(output_dir=tmp_path / "trace")

    def move(*_args, **_kwargs):
        return ["target"], False, None, None, None

    try:
        result = recorder.wrap_move(move, module)(graphs_hash=["source"])
        assert result[0] == ["target"]
        assert recorder.selected_transition_count == 1
        assert state.graph_map.rehydrations >= 2
        assert state.graph_map.unresolved_lookups == 0
    finally:
        state.close()
