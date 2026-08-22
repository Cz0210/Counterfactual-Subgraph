"""Controller-facing blocked fragment for TasteMolNet multiclass baselines.

The current campaign has no approved data-reuse basis.  Consequently this
module can publish only immutable terminal ``BLOCKED_LICENSE_REVIEW`` tasks.
It also records the exact prerequisites for a *new* runnable fragment after
approval; it never mutates or silently releases an existing blocked fragment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.baselines.tastemolnet_multiclass_adapters import (
    CF_MODE,
    CLASSIFIER_FAMILY,
    DATASET,
    DESTINATION_LABELS,
    NUM_CLASSES,
    ORACLE_BACKEND,
    SOURCE_LABEL,
    TasteMulticlassContractError,
    canonical_manifest_hash,
    multiclass_extension_manifest,
)


FRAGMENT_SCHEMA = "tastemolnet_multiclass_baseline_blocked_fragment_v1"
LICENSE_GATE_SCHEMA = "tastemolnet_license_audit_v1"
BLOCKER = "BLOCKED_LICENSE_REVIEW"
METHODS = ("GCFExplainer", "GlobalGCE", "ComRecGC")


def _method_id(method: str) -> str:
    return method.lower()


def validate_current_license_block(
    gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return bounded gate evidence or fail if a PASS needs fresh release work."""

    if gate is None:
        return {
            "status": BLOCKER,
            "gate_supplied": False,
            "gate_hash": None,
            "reason": "no_explicit_tastemolnet_license_gate",
        }
    if gate.get("schema_version") != LICENSE_GATE_SCHEMA:
        raise TasteMulticlassContractError(
            "TasteMolNet license gate schema is not the audited exact-data contract"
        )
    if str(gate.get("dataset") or "").lower() != DATASET:
        raise TasteMulticlassContractError("License gate belongs to another dataset")
    status = str(gate.get("status") or "")
    if status == "PASS":
        raise TasteMulticlassContractError(
            "A PASS license gate requires a new runnable fragment with a frozen "
            "three-class GINE and native input manifests; this immutable blocked "
            "fragment cannot be relabeled or released in place"
        )
    if status != BLOCKER:
        raise TasteMulticlassContractError(
            f"Unsupported TasteMolNet license gate status: {status!r}"
        )
    if gate.get("heavy_route_authorized") is not False:
        raise TasteMulticlassContractError(
            "Blocked TasteMolNet gate must set heavy_route_authorized=false"
        )
    if gate.get("run_tastemolnet") is not False:
        raise TasteMulticlassContractError(
            "Blocked TasteMolNet gate must set run_tastemolnet=false"
        )
    return {
        "status": BLOCKER,
        "gate_supplied": True,
        "gate_hash": canonical_manifest_hash(gate),
        "reason": str(gate.get("blocked_reason") or BLOCKER),
    }


def release_contract() -> dict[str, Any]:
    """Describe the all-of gate for a future fresh runnable task fragment."""

    return {
        "schema_version": "tastemolnet_multiclass_baseline_release_contract_v1",
        "release_mode": "new_fresh_fragment_only",
        "blocked_fragment_mutation_forbidden": True,
        "all_of": [
            {
                "gate": "exact_data_license",
                "schema_version": LICENSE_GATE_SCHEMA,
                "status": "PASS",
                "heavy_route_authorized": True,
                "run_tastemolnet": True,
                "explicit_reuse_basis_required": True,
            },
            {
                "gate": "frozen_classifier",
                "dataset": DATASET,
                "oracle_backend": ORACLE_BACKEND,
                "classifier_family": CLASSIFIER_FAMILY,
                "num_classes": NUM_CLASSES,
                "source_label": SOURCE_LABEL,
                "rf_oracle_used": False,
                "test_loaded": False,
                "required_hashes": [
                    "oracle_checkpoint_hash",
                    "temperature_calibration_hash",
                    "feature_schema_hash",
                ],
            },
            {
                "gate": "shared_evaluation_inputs",
                "same_scaffold_split_hash_for_all_methods": True,
                "same_molclr_checkpoint_hash_for_all_methods": True,
                "cf_mode": CF_MODE,
                "destination_labels": list(DESTINATION_LABELS),
            },
            {
                "gate": "split_order",
                "generation": "train_only",
                "selection": "calibration_only",
                "test": "after_frozen_selector_only",
            },
        ],
        "method_extensions": {
            method: multiclass_extension_manifest(method) for method in METHODS
        },
        "future_task_order": {
            "GCFExplainer": [
                "native_fullgraph_generation",
                "calibration_ordering",
                "selector_freeze",
                "heldout_test_evaluation",
                "final_freeze",
            ],
            "GlobalGCE": [
                "target_0_native_rule_generation",
                "target_2_native_rule_generation",
                "deduplicate_merge_before_calibration",
                "calibration_selector",
                "selector_freeze",
                "heldout_test_evaluation",
                "final_freeze",
            ],
            "ComRecGC": [
                "native_generation",
                "global_hash_lineage_reconstruction",
                "unique_single_edit_freeze",
                "calibration_selector",
                "selector_freeze",
                "heldout_test_evaluation",
                "final_freeze",
            ],
        },
    }


def build_blocked_baseline_tasks(
    *, license_task_id: str = "tastemolnet_license_audit"
) -> list[dict[str, Any]]:
    """Return three terminal tasks that consume no CPU/GPU worker."""

    if not license_task_id or any(character.isspace() for character in license_task_id):
        raise TasteMulticlassContractError("license_task_id must be one stable task ID")
    contract = release_contract()
    contract_hash = canonical_manifest_hash(contract)
    tasks: list[dict[str, Any]] = []
    for priority, method in enumerate(METHODS, start=1001):
        method_id = _method_id(method)
        adapter = multiclass_extension_manifest(method)
        tasks.append(
            {
                "id": f"tastemolnet_{method_id}",
                "dataset": DATASET,
                "stage": f"TASTEMOLNET_{method_id.upper()}_MULTICLASS_ROUTE",
                "runner_dataset": f"tastemolnet-baseline-{method_id}",
                "runner_stage": f"TASTEMOLNET_{method_id.upper()}_MULTICLASS_ROUTE",
                "depends_on": [license_task_id],
                "resource": "cpu",
                "priority": priority,
                "enabled": True,
                "data_splits": [],
                "manifest_only": True,
                "command": None,
                "blocked_reason": BLOCKER,
                "blocker_code": BLOCKER,
                "heavy_route_authorized": False,
                "run_tastemolnet": False,
                "native_action_kind": adapter["action_kind"],
                "multiclass_adapter_contract": adapter,
                "release_contract_hash": contract_hash,
                "fresh_release_fragment_required": True,
            }
        )
    return tasks


def build_blocked_fragment(
    *,
    license_gate: Mapping[str, Any] | None = None,
    license_gate_path: str | Path | None = None,
    license_task_id: str = "tastemolnet_license_audit",
) -> dict[str, Any]:
    """Build the current non-runnable baseline fragment and its release gate."""

    evidence = validate_current_license_block(license_gate)
    if license_gate_path is not None:
        path = Path(license_gate_path).expanduser()
        if not path.is_absolute():
            raise TasteMulticlassContractError("license_gate_path must be absolute")
        evidence["gate_path"] = str(path.resolve(strict=False))
    contract = release_contract()
    return {
        "schema_version": FRAGMENT_SCHEMA,
        "dataset": DATASET,
        "status": BLOCKER,
        "heavy_route_authorized": False,
        "current_tasks_are_terminal_blocked": True,
        "license_evidence": evidence,
        "release_contract": contract,
        "release_contract_hash": canonical_manifest_hash(contract),
        "tasks": build_blocked_baseline_tasks(license_task_id=license_task_id),
    }


__all__ = [
    "BLOCKER",
    "FRAGMENT_SCHEMA",
    "LICENSE_GATE_SCHEMA",
    "METHODS",
    "build_blocked_baseline_tasks",
    "build_blocked_fragment",
    "release_contract",
    "validate_current_license_block",
]
