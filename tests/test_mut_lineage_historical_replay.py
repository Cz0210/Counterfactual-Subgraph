from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl.run_three_lines_stage import MUT_SOURCE_COMMIT
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


def test_mutagenicity_failed_v2_was_bound_to_one_extra_commit_character() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    source = document["source"]
    regression = document["failed_v2_project_commit_gate"]
    actual = regression["actual_project_commit"]
    incorrect = regression["incorrect_expected_project_commit"]

    assert source["resolved_config_sha256"] == (
        "7d89926de8d17907982189af1cb80862a813fd35bd9626557616738d8eefa0dd"
    )
    assert source["resolved_config_project_commit"] == actual
    assert MUT_SOURCE_COMMIT == actual
    assert len(actual) == regression["actual_length"] == 40
    assert len(incorrect) == regression["incorrect_expected_length"] == 41
    assert incorrect == actual + "b"
    assert regression["only_failed_check"] == "project_commit_matches"
    assert regression["closure_error"] is None
    assert regression["frozen_payload_closure_complete"] is True
    assert regression["sha_mismatch_count"] == 0


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
