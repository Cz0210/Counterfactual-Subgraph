from __future__ import annotations

import pytest

from src.baselines.comrecgc.runtime import model_counterfactual_graphs


def test_model_counterfactual_graphs_preserves_official_candidate_order() -> None:
    first = object()
    second = object()
    payload = {
        "graph_map": {"a": (first,), "b": (second,)},
        "counterfactual_candidates": [
            {"graph_hash": "missing", "importance_parts": [1.0]},
            {"graph_hash": "b", "importance_parts": [1.0]},
            {"graph_hash": "a", "importance_parts": [1.0]},
        ],
    }
    assert model_counterfactual_graphs(payload, limit=2) == [second, first]


def test_model_counterfactual_graphs_requires_actual_counterfactuals() -> None:
    payload = {
        "graph_map": {"a": (object(),)},
        "counterfactual_candidates": [
            {"graph_hash": "a", "importance_parts": [0.49]},
        ],
    }
    with pytest.raises(RuntimeError, match="no model-counterfactual"):
        model_counterfactual_graphs(payload, limit=5)
