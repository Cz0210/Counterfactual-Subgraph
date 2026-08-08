from __future__ import annotations

import pytest

from src.eval.heldout_molecule_protocol import (
    EXPECTED_ROLES,
    build_heldout_protocol_manifest,
    transductive_vs_heldout_schema,
    validate_heldout_roles,
)


def test_heldout_roles_are_inductive_and_calibration_only() -> None:
    audit = validate_heldout_roles(EXPECTED_ROLES)
    assert audit["passed"] is True
    assert audit["test_used_for_candidate_generation"] is False
    assert audit["threshold_fitted_on_test"] is False


def test_heldout_test_leakage_fails_closed() -> None:
    roles = {key: tuple(value) for key, value in EXPECTED_ROLES.items()}
    roles["candidate_discovery"] = ("train", "test")
    with pytest.raises(ValueError, match="test_used_for_candidate_discovery"):
        validate_heldout_roles(roles)


def test_heldout_parent_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="parent overlap"):
        build_heldout_protocol_manifest(
            dataset="BBBP",
            method="Ours",
            split_manifest_sha256="a" * 64,
            parent_ids_by_split={
                "train": ["p1"],
                "val": ["p2"],
                "calibration": ["p3"],
                "test": ["p1"],
            },
        )


def test_combined_protocol_schema_is_stable() -> None:
    assert transductive_vs_heldout_schema()[:3] == ("dataset", "method", "protocol")
    assert transductive_vs_heldout_schema()[-1] == "status"
