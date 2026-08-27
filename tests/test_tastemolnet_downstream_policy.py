from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.utils import tastemolnet_downstream_policy as policy_module
from src.utils.tastemolnet_downstream_policy import (
    DOWNSTREAM_POLICY_FILE_SHA256,
    TasteDownstreamPolicyError,
    hold_tastemolnet_downstream_policy,
    load_tastemolnet_downstream_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = (
    PROJECT_ROOT
    / "configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json"
)
BASE = (
    PROJECT_ROOT
    / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mutated(tmp_path: Path, path: tuple[str, ...], value: object) -> Path:
    payload = json.loads(DOWNSTREAM.read_text(encoding="utf-8"))
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    output = tmp_path / "policy.json"
    output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return output


def test_tracked_downstream_policy_is_exact_and_no_redistribution() -> None:
    policy = load_tastemolnet_downstream_policy(
        DOWNSTREAM, base_policy_path=BASE
    )
    assert policy.file_sha256 == DOWNSTREAM_POLICY_FILE_SHA256 == _sha(DOWNSTREAM)
    assert policy.stage("T3_GINE_CALIBRATED")["mode"] == (
        "adopt_existing_validation_fit"
    )
    assert policy.stage("T3_GINE_CALIBRATED")["split_payload_access"] == {
        "train": False,
        "validation": False,
        "calibration": False,
        "test": False,
    }
    assert policy.stage("T3_GINE_CALIBRATED")["device"] == "cpu"
    assert policy.stage("T3_GINE_CALIBRATED")["physical_gpu_index"] is None
    assert policy.stage("T4_ORACLE_SMOKE")["split_payload_access"] == {
        "train": False,
        "validation": False,
        "calibration": True,
        "test": False,
    }
    assert policy.payload["data_handling"]["temperature_refit_allowed"] is False
    assert policy.payload["permissions"]["data_redistribution_allowed"] is False
    assert policy.payload["permissions"]["per_example_public_reporting_allowed"] is False
    t6 = policy.stage("T6_OURS_SMOKE")
    assert t6["mode"] == "train_only_frozen_gine_reward_ppo_smoke"
    assert t6["split_payload_access"] == {
        "train": True,
        "validation": False,
        "calibration": False,
        "test": False,
    }
    assert t6["frozen_gine_reward_required"] is True
    assert t6["rf_oracle_used"] is False
    assert t6["minimum_optimizer_steps"] == 5
    assert t6["allowed_input_files"] == [
        "immutable_t2_bundle",
        "immutable_t3_stage_output",
        "immutable_t4_stage_output",
        "immutable_t5_clean_policy",
        "frozen_train_csv",
    ]
    assert policy.revalidate(stage="T6_OURS_SMOKE")["stage"] == "T6_OURS_SMOKE"
    policy.close()


def test_public_held_policy_api_exposes_t6_hash_binding() -> None:
    with hold_tastemolnet_downstream_policy(
        DOWNSTREAM, base_policy_path=BASE
    ) as policy:
        evidence = policy.revalidate(stage="T6_OURS_SMOKE")
        assert evidence["downstream_policy"]["file_sha256"] == _sha(DOWNSTREAM)
        assert evidence["base_policy"]["file_sha256"] == _sha(BASE)
        assert evidence["stage_contract"]["split_payload_access"]["train"] is True
        assert evidence["stage_contract"]["allowed_input_files"][-1] == (
            "frozen_train_csv"
        )


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
@pytest.mark.parametrize(
    "path",
    [
        ("execution", "stages", "T3_GINE_CALIBRATED", "run"),
        ("execution", "stages", "T3_GINE_CALIBRATED", "num_classes"),
        ("execution", "stages", "T3_GINE_CALIBRATED", "source_label"),
        ("execution", "stages", "T4_ORACLE_SMOKE", "physical_gpu_index"),
        ("execution", "stages", "T4_ORACLE_SMOKE", "run"),
        ("execution", "stages", "T4_ORACLE_SMOKE", "num_classes"),
        ("execution", "stages", "T4_ORACLE_SMOKE", "source_label"),
        ("execution", "stages", "T4_ORACLE_SMOKE", "source_count"),
        ("execution", "stages", "T4_ORACLE_SMOKE", "max_deletions_per_parent"),
        ("execution", "stages", "T4_ORACLE_SMOKE", "minimum_deletions_per_parent"),
        ("execution", "stages", "T6_OURS_SMOKE", "physical_gpu_index"),
        ("execution", "stages", "T6_OURS_SMOKE", "run"),
        ("execution", "stages", "T6_OURS_SMOKE", "num_classes"),
        ("execution", "stages", "T6_OURS_SMOKE", "source_label"),
        ("execution", "stages", "T6_OURS_SMOKE", "minimum_optimizer_steps"),
    ],
)
def test_downstream_integer_authority_rejects_bool_float_string_and_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
) -> None:
    candidate = _mutated(tmp_path, path, value)
    monkeypatch.setattr(policy_module, "TRACKED_DOWNSTREAM_POLICY_PATH", candidate)
    with pytest.raises(TasteDownstreamPolicyError, match="native JSON type"):
        load_tastemolnet_downstream_policy(
            candidate,
            base_policy_path=BASE,
            expected_file_sha256=_sha(candidate),
        )


def test_downstream_policy_rejects_permission_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _mutated(
        tmp_path,
        ("permissions", "data_redistribution_allowed"),
        True,
    )
    monkeypatch.setattr(policy_module, "TRACKED_DOWNSTREAM_POLICY_PATH", candidate)
    with pytest.raises(TasteDownstreamPolicyError, match="changed value"):
        load_tastemolnet_downstream_policy(
            candidate,
            base_policy_path=BASE,
            expected_file_sha256=_sha(candidate),
        )


def test_downstream_policy_rejects_ancestor_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    candidate = real / "policy.json"
    candidate.write_bytes(DOWNSTREAM.read_bytes())
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(TasteDownstreamPolicyError, match="exact tracked"):
        load_tastemolnet_downstream_policy(
            alias / "policy.json",
            base_policy_path=BASE,
            expected_file_sha256=_sha(candidate),
        )


def test_downstream_policy_rejects_equal_byte_base_policy_ancestor_alias(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    copied = real / BASE.name
    copied.write_bytes(BASE.read_bytes())
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(TasteDownstreamPolicyError, match="exact tracked"):
        load_tastemolnet_downstream_policy(
            DOWNSTREAM,
            base_policy_path=alias / BASE.name,
        )


@pytest.mark.parametrize("value", [1, True, 1.0, "1"])
def test_t3_cpu_authority_rejects_gpu_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    candidate = _mutated(
        tmp_path,
        ("execution", "stages", "T3_GINE_CALIBRATED", "physical_gpu_index"),
        value,
    )
    monkeypatch.setattr(policy_module, "TRACKED_DOWNSTREAM_POLICY_PATH", candidate)
    with pytest.raises(TasteDownstreamPolicyError, match="native JSON type"):
        load_tastemolnet_downstream_policy(
            candidate,
            base_policy_path=BASE,
            expected_file_sha256=_sha(candidate),
        )
