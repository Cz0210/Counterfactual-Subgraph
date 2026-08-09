from __future__ import annotations

from typing import Any

from src.eval.mutagenicity_wnode_matrix import (
    CalibrationParent,
    evaluate_parent_candidate_pair,
)


class Teacher:
    def score_smiles(
        self, smiles: str, label: int | None = None, **_: Any
    ) -> dict[str, Any]:
        values = {
            "PARENT": (1, 0.9),
            "NONFLIP": (1, 0.8),
            "CONNECTED_FLIP": (0, 0.2),
        }
        prediction, probability = values[smiles]
        return {
            "teacher_result_ok": True,
            "teacher_label": prediction,
            "teacher_prob": probability,
        }


class Distance:
    def distance(self, _left: str, right: str) -> dict[str, Any]:
        return {
            "ok": True,
            "distance": {"NONFLIP": 0.001, "CONNECTED_FLIP": 0.2}[right],
        }


def test_all_winner_metrics_come_from_same_connected_flip_match() -> None:
    deletions = [
        {
            "match_index": 0,
            "match_atoms": [0],
            "delete_valid": True,
            "residual_smiles": "NONFLIP",
            "residual_connected": True,
            "residual_num_components": 1,
            "sanitize_ok": True,
            "contains_dot": False,
            "action_semantics_version": "connected_sanitized_residual_v1",
        },
        {
            "match_index": 1,
            "match_atoms": [3],
            "delete_valid": True,
            "residual_smiles": "CONNECTED_FLIP",
            "residual_connected": True,
            "residual_num_components": 1,
            "sanitize_ok": True,
            "contains_dot": False,
            "action_semantics_version": "connected_sanitized_residual_v1",
        },
    ]
    pair, _matches = evaluate_parent_candidate_pair(
        CalibrationParent("p", "PARENT", 1, "calibration"),
        {"candidate_id": "c", "canonical_fragment": "C"},
        teacher=Teacher(),
        distance_provider=Distance(),
        deletion_fn=lambda _parent, _candidate: deletions,
        match_selection_policy=(
            "existential_min_wnode_among_valid_connected_strict_flips_v1"
        ),
    )
    assert pair["best_match_index"] == 1
    assert pair["residual_smiles"] == "CONNECTED_FLIP"
    assert pair["pred_after"] == 0
    assert pair["cf_drop"] == 0.7
    assert pair["wnode_distance"] == 0.2
