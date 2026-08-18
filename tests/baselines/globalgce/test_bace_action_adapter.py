from __future__ import annotations

from src.baselines.globalgce_bace_action_adapter import (
    FULLGRAPH_ACTION_ADAPTER,
    adapt_globalgce_fullgraph_rows,
)


def test_fullgraph_adapter_does_not_create_a_deletion_fragment() -> None:
    adapted = adapt_globalgce_fullgraph_rows(
        [{"rank": 1, "candidate_id": "g1", "canonical_smiles": "CCO"}],
        expected_count=1,
    )
    assert adapted[0].candidate_smiles == "CCO"
    assert adapted[0].action_adapter == FULLGRAPH_ACTION_ADAPTER
    assert "final_fragment" not in adapted[0].source_row
