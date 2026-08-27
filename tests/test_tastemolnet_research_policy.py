from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.tastemolnet_research_policy import (
    ACTIVE_STATE,
    PENDING_STATE,
    SOURCE_CSV_SHA256,
    UPSTREAM_TERMS_STATUS,
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = PROJECT_ROOT / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"


def _active_policy(tmp_path: Path) -> Path:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    payload["authorization_basis"] = "explicit_user_instruction"
    payload["authorization_state"] = ACTIVE_STATE
    payload["authorization_source"] = "user_project_owner_instruction"
    payload["research_compute_allowed"] = True
    payload["paper_result_reporting_allowed"] = True
    payload["aggregated_metrics_release_allowed"] = True
    payload["figure_release_allowed"] = True
    payload["permissions"]["research_execution"] = "ALLOWED"
    payload["permissions"]["paper_reporting"] = "ALLOWED"
    payload["permissions"]["aggregate_publication"] = (
        "ALLOWED_AFTER_PUBLIC_ARTIFACT_AUDIT"
    )
    payload["execution"]["run_tastemolnet"] = 1
    path = tmp_path / "active-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _pending_policy(tmp_path: Path) -> Path:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    payload["authorization_basis"] = "forwarded_user_instruction_pending_root_activation"
    payload["authorization_state"] = PENDING_STATE
    payload["authorization_source"] = "PENDING_ROOT_ACTIVATION"
    payload["research_compute_allowed"] = False
    payload["paper_result_reporting_allowed"] = False
    payload["aggregated_metrics_release_allowed"] = False
    payload["figure_release_allowed"] = False
    payload["trained_model_release_allowed"] = False
    payload["permissions"]["research_execution"] = "PENDING_ROOT_ACTIVATION"
    payload["permissions"]["paper_reporting"] = "PENDING_ROOT_ACTIVATION"
    payload["permissions"]["aggregate_publication"] = (
        "ALLOWED_ONLY_AFTER_ACTIVATION_AND_PUBLIC_ARTIFACT_AUDIT"
    )
    payload["execution"]["run_tastemolnet"] = 0
    path = tmp_path / "pending-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_checked_policy_is_active_scoped_without_license_conclusion() -> None:
    policy = load_tastemolnet_research_policy(POLICY)
    assert policy.version == 2
    assert policy.payload["schema_version"] == (
        "tastemolnet_research_reporting_policy_v2"
    )
    assert policy.authorization_state == ACTIVE_STATE
    assert policy.active is True
    assert policy.payload["execution"]["run_tastemolnet"] == 1
    assert policy.payload["execution"]["gpu_index"] == 1
    assert policy.payload["execution"]["gpu_lock_mode"] == "exclusive"
    assert policy.payload["execution"]["hpc_execution_allowed"] is False
    assert policy.payload["execution"]["classifier"] == {
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "num_classes": 3,
        "source_label": 1,
    }
    assert policy.payload["execution"]["split_access"]["test_loaded"] is False
    assert policy.payload["dataset_identity"]["source_csv_sha256"] == SOURCE_CSV_SHA256
    assert policy.payload["dataset_identity"]["prepared_output_manifest_sha256"] == (
        "36aaf17bf45e0a092a96a0379fab31d9e6bfcd719b87cb4ffa4e57a6642bb645"
    )
    assert policy.payload["dataset_identity"]["split_manifest_sha256"] == (
        "841f3b911e5d353c1e00f010bafcc8a6f7b3433082dba8a8979fab1b558251af"
    )
    assert policy.payload["dataset_identity"]["upstream_terms_status"] == UPSTREAM_TERMS_STATUS
    assert policy.payload["permissions"]["dataset_redistribution"] == "FORBIDDEN"
    assert policy.payload["research_compute_allowed"] is True
    assert policy.payload["paper_result_reporting_allowed"] is True
    assert policy.payload["data_redistribution_allowed"] is False
    assert policy.payload["upstream_license_claimed_resolved"] is False
    assert policy.payload["raw_data_redistribution_allowed"] is False
    assert policy.payload["cleaned_dataset_redistribution_allowed"] is False
    assert policy.payload["full_smiles_label_table_release_allowed"] is False
    assert policy.payload["reconstructable_dataset_artifact_allowed"] is False
    assert policy.payload["preprocessing_code_release_allowed"] is True
    assert policy.payload["configuration_release_allowed"] is True
    assert policy.payload["trained_model_release_allowed"] == "review_required"
    policy.require_active()


def test_synthetic_pending_policy_remains_non_runnable(tmp_path: Path) -> None:
    policy = load_tastemolnet_research_policy(_pending_policy(tmp_path))
    assert policy.authorization_state == PENDING_STATE
    assert policy.active is False
    assert policy.payload["execution"]["run_tastemolnet"] == 0
    with pytest.raises(TasteResearchPolicyError, match="NOT_ACTIVATED"):
        policy.require_active()


def test_exact_active_shape_is_scoped_and_never_a_license_conclusion(tmp_path: Path) -> None:
    policy = load_tastemolnet_research_policy(_active_policy(tmp_path))
    assert policy.active is True
    policy.require_active()
    evidence = policy.evidence()
    assert evidence["research_execution_allowed"] is True
    assert evidence["paper_reporting_allowed"] is True
    assert evidence["dataset_redistribution_allowed"] is False
    assert evidence["upstream_terms_status"] == "NOT_EXPLICITLY_STATED"
    assert evidence["license_conclusion"] == "NOT_GRANTED_OR_INFERRED"
    assert "license_pass" not in evidence
    assert "passed" not in evidence


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_run_tastemolnet_requires_a_native_json_integer(
    tmp_path: Path, value: object
) -> None:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    payload["execution"]["run_tastemolnet"] = value
    path = tmp_path / "typed-run-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(TasteResearchPolicyError, match="execution boundary"):
        load_tastemolnet_research_policy(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (
            "permissions",
            "dataset_redistribution",
            "ALLOWED",
            "permissions.dataset_redistribution changed value",
        ),
        (
            "execution.classifier",
            "rf_oracle_used",
            True,
            "execution.classifier.rf_oracle_used changed value",
        ),
        (
            "execution.classifier",
            "num_classes",
            2,
            "execution.classifier.num_classes changed value",
        ),
        ("execution", "gpu_index", 2, "execution boundary"),
        (
            "execution.split_access",
            "test_loaded",
            True,
            "execution.split_access.test_loaded changed value",
        ),
        (
            "dataset_identity",
            "upstream_terms_status",
            "MIT",
            "dataset_identity.upstream_terms_status changed value",
        ),
    ],
)
def test_policy_weakening_fails_closed(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    target = payload
    for part in section.split("."):
        target = target[part]
    target[key] = value
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(TasteResearchPolicyError, match=message):
        load_tastemolnet_research_policy(path)


def test_raw_file_hash_binding_detects_policy_drift(tmp_path: Path) -> None:
    expected = load_tastemolnet_research_policy(POLICY).file_sha256
    copied = tmp_path / "policy.yaml"
    copied.write_bytes(POLICY.read_bytes() + b"\n")
    with pytest.raises(TasteResearchPolicyError, match="file SHA-256 changed"):
        load_tastemolnet_research_policy(copied, expected_file_sha256=expected)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("dataset_identity", "prepared_rows", 13421.0),
        ("data_handling", "reuse_existing_prepared_data_only", 1),
        ("execution.classifier", "num_classes", 3.0),
        ("execution.classifier", "rf_oracle_used", 0),
        ("execution.split_access", "test_metadata_hash_only", 1),
    ],
)
def test_nested_policy_authority_rejects_python_bool_int_numeric_coercions(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    target = payload
    for part in section.split("."):
        target = target[part]
    target[key] = value
    path = tmp_path / "nested-type-tamper.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TasteResearchPolicyError, match="native JSON type"):
        load_tastemolnet_research_policy(path)
