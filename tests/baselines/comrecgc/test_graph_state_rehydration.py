from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256
from src.baselines.comrecgc.live_graph_state import LiveGraphState


def test_authoritative_store_reopens_with_exact_graph_and_checksum(tmp_path) -> None:
    graph = SimpleNamespace(
        x=np.asarray([[1], [4], [7]], dtype=np.int64),
        edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        num_nodes=3,
    )
    value = [graph, np.asarray([0.25, 0.5]), np.asarray([3.0, 4.0])]
    path = tmp_path / "graphs.sqlite3"
    first_module = SimpleNamespace(
        graph_map={}, graph_index_map={}, counterfactual_candidates=[], transitions={}
    )
    first = LiveGraphState(first_module, {}, store_path=path, seed=0)
    first_module.graph_map = first.graph_map
    first_module.graph_map[42] = value
    del first_module.graph_map[42]
    first_sha = first.store.integrity_audit()["content_sha256"]
    first.close()

    second_module = SimpleNamespace(
        graph_map={}, graph_index_map={}, counterfactual_candidates=[], transitions={}
    )
    second = LiveGraphState(second_module, {}, store_path=path, seed=0)
    try:
        restored = second.resolve_graph(42)
        assert stable_untyped_graph_sha256(restored) == stable_untyped_graph_sha256(graph)
        audit = second.store.integrity_audit()
        assert audit["integrity_passed"] is True
        assert audit["content_sha256"] == first_sha
    finally:
        second.close()
