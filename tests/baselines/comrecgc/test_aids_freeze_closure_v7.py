from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc.frozen_payload import build_frozen_payload_closure
from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256


def _graph(value: int) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[value]], dtype=np.int64),
        edge_index=np.empty((2, 0), dtype=np.int64),
        num_nodes=1,
        comrecgc_parent_id="aids-parent",
    )


def test_aids_payload_includes_trace_transition_frontier_and_recourse_hashes() -> None:
    source, target, frontier, recourse = (_graph(value) for value in range(4))
    payload = {
        "graph_map": {
            "source": [source],
            "target": [target],
            "frontier": [frontier],
            "recourse": [recourse],
        },
        "transitions": {"source": (["target"], [target])},
        "frontier_hashes": ["frontier"],
        "counterfactual_candidates": [{"graph_hash": "recourse"}],
    }
    trace = [{
        "event": "selected_transition",
        "parent_id": "aids-parent",
        "source_official_hash": "source",
        "target_official_hash": "target",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
    }]
    frozen, audit = build_frozen_payload_closure(
        payload, trace, backing_store_path=None
    )

    assert audit["closure_complete"] is True
    assert set(frozen["required_hashes"]) == {
        "source", "target", "frontier", "recourse"
    }
    assert audit["transition_hash_count"] == 2
    assert audit["selected_trace_hash_count"] == 2
