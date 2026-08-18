from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc.frozen_payload import (
    build_frozen_payload_closure,
    payload_graphs_by_official_hash,
)
from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256


def _graph(value: int) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[value]], dtype=np.int64),
        edge_index=np.empty((2, 0), dtype=np.int64),
        num_nodes=1,
        comrecgc_parent_id="p",
    )


def test_original_trace_hashes_survive_serialization_roundtrip() -> None:
    source, target = _graph(1), _graph(2)
    trace = [{
        "event": "selected_transition",
        "parent_id": "p",
        "source_official_hash": "trace-source",
        "target_official_hash": "trace-target",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
    }]
    frozen, _ = build_frozen_payload_closure(
        {
            "graph_map": {"canonical-source": [source], "canonical-target": [target]},
            "counterfactual_candidates": [{"graph_hash": "canonical-target"}],
        },
        trace,
        backing_store_path=None,
    )
    reloaded = pickle.loads(pickle.dumps(frozen))
    verified, audit = build_frozen_payload_closure(
        reloaded, trace, backing_store_path=None
    )
    resolved = payload_graphs_by_official_hash(verified)

    assert audit["closure_complete"] is True
    assert set(verified["original_trace_hashes"]) == {"trace-source", "trace-target"}
    assert {"trace-source", "trace-target"}.issubset(resolved)
