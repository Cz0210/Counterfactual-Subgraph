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


def test_validator_and_recover_build_identical_required_closure() -> None:
    source, target = _graph(1), _graph(2)
    payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    trace = [{
        "event": "selected_transition",
        "parent_id": "aids-parent",
        "source_official_hash": "source",
        "target_official_hash": "target",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
    }]

    validated, first = build_frozen_payload_closure(
        payload, trace, backing_store_path=None
    )
    recovered, second = build_frozen_payload_closure(
        validated, trace, backing_store_path=None
    )

    assert first["requirements_sha256"] == second["requirements_sha256"]
    assert validated["required_hashes"] == recovered["required_hashes"]
    assert recovered["frozen_payload_closure_requirements"] == (
        validated["frozen_payload_closure_requirements"]
    )
