from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines.comrecgc.freeze_recovery import recovery_population_counts


FIXTURE = (
    Path(__file__).parent
    / "fixtures/comrecgc_lineage/mutagenicity_recovery_counts.json"
)


def test_mutagenicity_historical_failure_payload() -> None:
    """Keep the real recovered candidate population distinct from trace rows."""

    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    counts = recovery_population_counts(
        candidate_count=document["candidate_count"],
        candidate_lineage_resolved_count=document[
            "candidate_lineage_resolved_count"
        ],
        lineage_recovery_audit={
            "selected_transition_count": document["selected_transition_count"],
            "recorded_action_present_count": document[
                "recorded_action_present_count"
            ],
            "legacy_missing_action_count": document["legacy_missing_action_count"],
        },
    )

    assert counts == {
        "candidate_count": 100235,
        "candidate_lineage_resolved_count": 100235,
        "selected_transition_count": 224690,
    }
    assert counts["selected_transition_count"] != counts["candidate_count"]


def test_recovery_population_counts_fail_closed_on_inconsistent_trace_count() -> None:
    with pytest.raises(ValueError, match="selected-transition counters are inconsistent"):
        recovery_population_counts(
            candidate_count=2,
            candidate_lineage_resolved_count=2,
            lineage_recovery_audit={
                "selected_transition_count": 4,
                "recorded_action_present_count": 3,
                "legacy_missing_action_count": 0,
            },
        )
