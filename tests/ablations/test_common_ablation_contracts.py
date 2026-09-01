from __future__ import annotations

import json

import pytest

from src.ablations.contracts import AblationRunContract, ContractError
from src.ablations.output_schema import output_inventory, run_manifest_template


SHA = "a" * 64


def _contract(**overrides):
    payload = {
        "dataset": "bace",
        "method": "ours",
        "variant": "BRICS_FIXED",
        "seed": 7,
        "train_split_sha": SHA,
        "validation_split_sha": SHA,
        "calibration_split_sha": SHA,
        "test_split_sha": SHA,
        "oracle_sha": SHA,
        "temperature_sha": SHA,
        "feature_schema_sha": SHA,
        "molclr_sha": SHA,
        "wnode_config_sha": SHA,
        "candidate_budget_contract": {
            "primary": "proposal_attempt_matched",
            "attempts_per_parent": 4,
        },
        "selector_config_sha": SHA,
        "threshold_config_sha": SHA,
        "evaluation_config_sha": SHA,
    }
    payload.update(overrides)
    return AblationRunContract(**payload)


def test_common_contract_is_hash_closed_and_deterministic() -> None:
    first = _contract().to_dict()
    second = _contract().to_dict()
    assert first == second
    assert len(first["contract_sha256"]) == 64


def test_common_contract_rejects_missing_or_fake_hash() -> None:
    with pytest.raises(ContractError):
        _contract(test_split_sha="unknown")
    serialized = _contract().to_dict()
    serialized["seed"] = 99
    with pytest.raises(ContractError, match="self-hash changed"):
        AblationRunContract.from_mapping(serialized)


def test_output_templates_contain_no_fake_metrics() -> None:
    for family in ("llm", "gnn"):
        template = run_manifest_template(family)
        assert template["science_started"] is False
        assert template["metrics"] is None
        assert set(template["artifacts"]) == set(output_inventory(family))
        assert "0.0" not in json.dumps(template)
