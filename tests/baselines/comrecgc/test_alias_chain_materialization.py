from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.frozen_payload import (
    FrozenPayloadClosureError,
    build_frozen_payload_closure,
    payload_graphs_by_official_hash,
)


def _graph() -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[1]], dtype=np.int64),
        edge_index=np.empty((2, 0), dtype=np.int64),
        num_nodes=1,
    )


def test_alias_chain_is_flattened_and_materialized() -> None:
    graph = _graph()
    frozen, audit = build_frozen_payload_closure(
        {
            "graph_map": {"canonical": [graph]},
            "counterfactual_candidates": [{"graph_hash": "old"}],
            "alias_to_canonical": {"old": "middle", "middle": "canonical"},
        },
        (),
        backing_store_path=None,
    )
    assert audit["closure_complete"] is True
    assert frozen["alias_to_canonical"]["old"] == "canonical"
    assert payload_graphs_by_official_hash(frozen)["old"] is graph


def test_alias_cycle_fails_closed() -> None:
    with pytest.raises(FrozenPayloadClosureError, match="graph_alias_cycle"):
        build_frozen_payload_closure(
            {
                "graph_map": {"canonical": [_graph()]},
                "counterfactual_candidates": [{"graph_hash": "a"}],
                "alias_to_canonical": {"a": "b", "b": "a"},
            },
            (),
            backing_store_path=None,
        )
