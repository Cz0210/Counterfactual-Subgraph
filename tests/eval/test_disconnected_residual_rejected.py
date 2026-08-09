from __future__ import annotations

from typing import Any

from src.eval.close_counterfactual_coverage import (
    hard_delete_substructure_connected_matches,
)
from src.eval.mutagenicity_wnode_matrix import (
    CalibrationParent,
    evaluate_parent_candidate_pair,
)


class Teacher:
    def score_smiles(
        self, smiles: str, label: int | None = None, **_: Any
    ) -> dict[str, Any]:
        prediction = 1 if smiles == "CCOCC" else 0
        return {
            "teacher_result_ok": True,
            "teacher_label": prediction,
            "teacher_prob": 0.9 if prediction else 0.1,
        }


class Distance:
    calls = 0

    def distance(self, _left: str, _right: str) -> dict[str, Any]:
        self.calls += 1
        return {"ok": True, "distance": 0.001}


def test_disconnected_match_never_reaches_teacher_distance_or_coverage() -> None:
    distance = Distance()
    pair, matches = evaluate_parent_candidate_pair(
        CalibrationParent("p", "CCOCC", 1, "calibration"),
        {"candidate_id": "c", "canonical_fragment": "O"},
        teacher=Teacher(),
        distance_provider=distance,
        deletion_fn=hard_delete_substructure_connected_matches,
        match_selection_policy=(
            "existential_min_wnode_among_valid_connected_strict_flips_v1"
        ),
    )
    assert len(matches) == 1
    assert matches[0]["delete_valid"] is False
    assert matches[0]["residual_connected"] is False
    assert distance.calls == 0
    assert pair["pair_strict_flip"] is False
    assert pair["wnode_distance"] is None
