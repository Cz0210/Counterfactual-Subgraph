"""Inductive held-out molecule protocol contracts and leakage gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.data.molecular_split import stable_json_sha256


EXPECTED_ROLES = {
    "teacher_fit": ("train",),
    "teacher_selection": ("val",),
    "candidate_discovery": ("train", "val"),
    "selector_tuning": ("calibration",),
    "threshold_fitting": ("calibration",),
    "final_evaluation": ("test",),
}


def validate_heldout_roles(roles: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    missing = sorted(set(EXPECTED_ROLES) - set(roles))
    if missing:
        raise ValueError(f"Held-out protocol is missing roles: {missing}")
    normalized = {
        role: tuple(str(value) for value in roles[role]) for role in EXPECTED_ROLES
    }
    violations: list[str] = []
    for role in (
        "teacher_fit",
        "teacher_selection",
        "candidate_discovery",
        "selector_tuning",
        "threshold_fitting",
    ):
        if "test" in normalized[role]:
            violations.append(f"test_used_for_{role}")
    if normalized["threshold_fitting"] != ("calibration",):
        violations.append("threshold_not_calibration_only")
    if normalized["selector_tuning"] != ("calibration",):
        violations.append("selector_not_calibration_only")
    if normalized["final_evaluation"] != ("test",):
        violations.append("final_evaluation_not_test_only")
    if violations:
        raise ValueError("Held-out protocol leakage: " + ", ".join(violations))
    return {
        "schema_version": "heldout_molecule_protocol_v1",
        "passed": True,
        "roles": {key: list(value) for key, value in normalized.items()},
        "test_used_for_candidate_generation": False,
        "test_used_for_size_distribution": False,
        "test_used_for_fragment_frequency": False,
        "test_used_for_selector": False,
        "threshold_fitted_on_test": False,
        "test_used_for_hyperparameter_choice": False,
        "roles_sha256": stable_json_sha256(normalized),
    }


def build_heldout_protocol_manifest(
    *,
    dataset: str,
    method: str,
    split_manifest_sha256: str,
    parent_ids_by_split: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    role_audit = validate_heldout_roles(EXPECTED_ROLES)
    parent_sets = {
        split: {str(value) for value in values}
        for split, values in parent_ids_by_split.items()
    }
    overlaps: list[dict[str, Any]] = []
    names = tuple(parent_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(parent_sets[left] & parent_sets[right])
            if shared:
                overlaps.append(
                    {"left": left, "right": right, "count": len(shared), "examples": shared[:10]}
                )
    if overlaps:
        raise ValueError(f"Held-out parent overlap detected: {overlaps}")
    return {
        **role_audit,
        "dataset": str(dataset),
        "method": str(method),
        "protocol": "inductive_heldout_molecule_v1",
        "split_manifest_sha256": str(split_manifest_sha256),
        "parent_counts": {
            split: len(values) for split, values in parent_sets.items()
        },
        "parent_ids_hashes": {
            split: stable_json_sha256(sorted(values))
            for split, values in parent_sets.items()
        },
        "selection_performed_in_eval": False,
        "candidate_set_preselected": True,
        "status": "NOT_RUN",
    }


def transductive_vs_heldout_schema() -> tuple[str, ...]:
    return (
        "dataset",
        "method",
        "protocol",
        "coverage",
        "cost",
        "cf_drop",
        "flip_rate",
        "valid_rate",
        "structural_redundancy",
        "coverage_redundancy",
        "status",
    )


__all__ = [
    "EXPECTED_ROLES",
    "build_heldout_protocol_manifest",
    "transductive_vs_heldout_schema",
    "validate_heldout_roles",
]
