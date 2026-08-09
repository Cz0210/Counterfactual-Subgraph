from __future__ import annotations

from pathlib import Path

import pytest

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.gcf_style_recourse_report import CandidateRank, aggregate_detail_rows


def test_connected_report_rejects_finite_disconnected_strict_flip() -> None:
    candidates = [CandidateRank(1, "c", "C", "C", 0)]
    rows = [
        {
            "method": "ours_selected_subgraphs",
            "parent_id": "p",
            "candidate_id": "c",
            "label": 1,
            "pred_before": 1,
            "pred_after": 0,
            "distance": 0.001,
            "delete_valid": True,
            "residual_connected": False,
            "sanitize_ok": True,
            "contains_dot": True,
            "num_components": 2,
        }
    ]
    with pytest.raises(ValueError, match="invalid residual"):
        aggregate_detail_rows(
            rows,
            candidates=candidates,
            source=Path("details.csv"),
            action_semantics_version=CONNECTED_ACTION_SEMANTICS,
            match_selection_policy=CONNECTED_MATCH_SELECTION_POLICY,
        )
