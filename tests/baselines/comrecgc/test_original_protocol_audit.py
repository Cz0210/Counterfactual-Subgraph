from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.baselines.comrecgc.contracts import (
    GenerationParameters,
    RecourseParameters,
    UPSTREAM_COMMIT,
)


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "comrecgc_original_protocol_v1.json"
)


def _contract() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_comrecgc_official_close_pair_filter() -> None:
    contract = _contract()
    deployment = contract["deployment_contract"]

    assert deployment["close_pair_filter_operator"] == "<="
    assert deployment["dbscan_input"] == "theta_close_recourse_vectors_only"
    assert deployment["physical_cartesian_store_is_not_logical_dbscan_input"] is True
    assert deployment["parent_axis"] == 0
    assert deployment["candidate_axis"] == 1
    assert deployment["recourse_vector"] == (
        "(embedding(candidate) - embedding(parent)) / "
        "(element_count(candidate) + element_count(parent))"
    )


def test_comrecgc_candidate_cap_effective() -> None:
    cap = _contract()["official_code_defaults"]["candidate_cap"]

    assert cap["effective_source"] == "comrecgc.py --k -> MAX_COUNTERFACTUAL_SIZE"
    assert cap["nominal_maximum"] == 100_000
    assert cap["cf_size_declared"] == 100_000
    assert cap["cf_size_applied"] is False
    assert cap["downstream_slice_applied"] is False
    assert cap["candidate_count_must_come_from_run_payload"] is True
    assert cap["candidate_count_before"] is None
    assert cap["candidate_count_after"] is None


def test_comrecgc_paper_code_differences_are_explicit() -> None:
    contract = _contract()

    assert contract["paper_config"]["random_walk"]["teleport_tau"] == 0.05
    assert contract["official_code_defaults"]["generation"]["teleport_tau"] == 0.1
    assert contract["official_code_defaults"]["run_experiments_overrides"][
        "teleport_overridden"
    ] is False
    assert contract["paper_config"]["candidate_predicate"]["operator"] == ">"
    assert contract["official_code_defaults"]["candidate_predicate"]["operator"] == ">="
    assert contract["official_code_defaults"]["generation"][
        "theta_consumed_by_generation"
    ] is False


def test_comrecgc_frozen_project_defaults_match_audit_fixture() -> None:
    contract = _contract()
    project = contract["project_frozen_source_contract"]

    assert project["upstream_commit"] == UPSTREAM_COMMIT
    assert project["project_full_generation"] == asdict(
        GenerationParameters.for_mode("full")
    )
    assert project["project_full_recourse"] == asdict(
        RecourseParameters.for_mode("full")
    )
    assert project["actual_aids_run_manifest_inspected"] is False
    assert project["actual_run_values_must_not_be_inferred_from_defaults"] is True


def test_comrecgc_official_source_identity_is_closed() -> None:
    source = _contract()["sources"]["official_implementation"]

    assert source["pinned_commit"] == UPSTREAM_COMMIT
    assert source["remote_head_observed"] == UPSTREAM_COMMIT
    assert source["remote_head_matches_project_pin"] is True
    assert set(source["key_file_sha256"]) == {
        "README.md",
        "common_recourse.py",
        "comrecgc.py",
        "data.py",
        "distance.py",
        "gnn.py",
        "run_experiments.sh",
        "util.py",
    }
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in source["key_file_sha256"].values()
    )
