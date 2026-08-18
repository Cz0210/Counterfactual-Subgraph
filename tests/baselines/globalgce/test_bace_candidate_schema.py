from __future__ import annotations

from src.baselines.globalgce_bace_action_adapter import (
    FULL_COUNTERFACTUAL_GRAPH,
    infer_globalgce_native_output_type,
)


def test_bace_frequency_candidate_is_a_full_counterfactual_graph() -> None:
    rows = [{
        "rank": "1",
        "candidate_id": "g1",
        "canonical_smiles": "CCO",
        "selection_mode": "globalgce_frequency_top20_train_support_v1",
    }]
    assert infer_globalgce_native_output_type(rows) == FULL_COUNTERFACTUAL_GRAPH
