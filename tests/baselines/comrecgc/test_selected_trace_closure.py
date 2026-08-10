from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc.frozen_payload import materialize_frozen_payload_closure
from src.baselines.comrecgc.graph_trace import (
    iter_candidate_lineage_from_selected_trace,
    stable_untyped_graph_sha256,
)
from src.baselines.comrecgc.live_graph_state import AuthoritativeGraphStore


def test_rehydrated_selected_trace_replays_exactly(tmp_path) -> None:
    source = SimpleNamespace(
        x=np.asarray([[1], [2]], dtype=np.int64),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
        comrecgc_parent_id="parent",
        comrecgc_node_origin=np.asarray([0, 1], dtype=np.int64),
    )
    target = SimpleNamespace(
        x=np.asarray([[2]], dtype=np.int64),
        edge_index=np.empty((2, 0), dtype=np.int64),
        num_nodes=1,
        comrecgc_parent_id="parent",
        comrecgc_node_origin=np.asarray([1], dtype=np.int64),
    )
    store_path = tmp_path / "graphs.sqlite3"
    store = AuthoritativeGraphStore(store_path)
    store.put("source", [source, np.asarray([1.0]), np.asarray([2.0])])
    store.close()
    payload = {
        "graph_map": {
            "target": [target, np.asarray([1.0]), np.asarray([2.0])]
        },
        "counterfactual_candidates": [{"graph_hash": "target", "frequency": 1}],
    }
    event = {
        "move_index": 1,
        "head_index": 0,
        "event": "selected_transition",
        "parent_id": "parent",
        "source_official_hash": "source",
        "target_official_hash": "target",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
        "action": ["NR", 0, 0],
    }
    frozen, _audit = materialize_frozen_payload_closure(
        payload, [event], backing_store_path=store_path
    )

    rows = list(
        iter_candidate_lineage_from_selected_trace(
            frozen,
            [event],
            source_graphs_by_parent_id={"parent": source},
        )
    )

    assert len(rows) == 1
    assert rows[0]["action_lineage_resolved"] is True
    assert rows[0]["actions"][0]["action_replay_exact"] is True
