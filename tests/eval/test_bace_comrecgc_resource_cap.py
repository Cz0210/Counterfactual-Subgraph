from __future__ import annotations

import pytest

from src.eval.bace_comrecgc_resource_cap import (
    BaceComRecGCResourceCapError,
    decide_bace_comrecgc_resource_cap,
)


def _audit(
    step: int,
    *,
    unique: int = 10,
    lineage: int = 0,
    status: str = "CONTINUE",
) -> dict[str, object]:
    return {
        "status": status,
        "evaluation_step": step,
        "checkpoint_summaries": [
            {
                "step": step,
                "valid_unique_count": unique,
                "lineage_error_count": lineage,
            }
        ],
        "checkpoint_evidence": [
            {"step": step, "checkpoint_digest": "a" * 64}
        ],
    }


def test_cap_never_stops_before_20000_without_convergence() -> None:
    result = decide_bace_comrecgc_resource_cap(_audit(17_500, unique=50_000))
    assert result["status"] == "CONTINUE"
    assert result["signals_sent"] is False


def test_cap_accepts_first_committed_20000_with_ten_rules_and_clean_lineage() -> None:
    result = decide_bace_comrecgc_resource_cap(_audit(20_000))
    assert result["status"] == "HANDOVER_ELIGIBLE"
    assert result["reason"] == "RESOURCE_CAP_20000"
    assert result["m_effective"] == 20_000
    assert result["postprocess_started"] is False


@pytest.mark.parametrize("unique,lineage", [(9, 0), (10, 1)])
def test_first_cap_extends_to_25000_when_gate_fails(unique: int, lineage: int) -> None:
    result = decide_bace_comrecgc_resource_cap(
        _audit(20_000, unique=unique, lineage=lineage)
    )
    assert result["status"] == "CONTINUE"
    assert result["reason"] == "EXTEND_TO_ABSOLUTE_CAP_25000"


def test_absolute_cap_science_fails_when_still_short() -> None:
    result = decide_bace_comrecgc_resource_cap(_audit(25_000, unique=9))
    assert result["status"] == "SCI_FAILED_ELIGIBLE_FOR_EXACT_GRACEFUL_STOP"
    assert result["m_effective"] == 25_000


def test_preregistered_convergence_can_handover_before_resource_cap() -> None:
    result = decide_bace_comrecgc_resource_cap(
        _audit(17_500, status="CONVERGED_EARLY_STOP")
    )
    assert result["status"] == "HANDOVER_ELIGIBLE"
    assert result["reason"] == "PREREGISTERED_CONVERGENCE_PASS"


def test_mismatched_committed_boundary_fails_closed() -> None:
    audit = _audit(20_000)
    audit["evaluation_step"] = 17_500
    with pytest.raises(BaceComRecGCResourceCapError, match="boundary"):
        decide_bace_comrecgc_resource_cap(audit)
