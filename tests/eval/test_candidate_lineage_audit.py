from __future__ import annotations

import pytest

from src.eval.candidate_lineage_audit import audit_candidate_lineage


def _row(index: int, split: str = "train") -> dict[str, object]:
    return {
        "dataset": "BBBP",
        "candidate_id": f"c{index}",
        "candidate_source": "chemllm_ppo",
        "parent_id": f"p{index}",
        "parent_split": split,
        "generation_seed": 13,
        "generation_rank": 1,
    }


def test_candidate_lineage_records_order_and_no_test_usage() -> None:
    audit = audit_candidate_lineage([_row(0), _row(1, "val")], expected_dataset="BBBP")
    assert audit["passed"] is True
    assert audit["candidate_count"] == 2
    assert audit["test_used_for_candidate_generation"] is False
    assert len(audit["candidate_order_sha256"]) == 64


def test_test_parent_candidate_fails_closed() -> None:
    with pytest.raises(ValueError, match="disallowed source splits"):
        audit_candidate_lineage([_row(0, "test")], expected_dataset="BBBP")


def test_selector_or_threshold_test_leakage_fails() -> None:
    with pytest.raises(ValueError, match="test_used_for_selector"):
        audit_candidate_lineage([_row(0)], selector_source_splits=("test",))
    with pytest.raises(ValueError, match="threshold_source_not_calibration"):
        audit_candidate_lineage([_row(0)], threshold_source="test")


def test_duplicate_candidate_id_fails() -> None:
    with pytest.raises(ValueError, match="unique"):
        audit_candidate_lineage([_row(0), {**_row(1), "candidate_id": "c0"}])
