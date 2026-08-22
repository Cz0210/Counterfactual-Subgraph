from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.train.bace_stage_boundaries import validate_stage_data_access


def _selector(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "selection_split": "calibration",
                "selector_fitted_on_calibration": True,
                "selection_frozen": True,
                "test_used": False,
                "ordered_rule_ids": [f"rule-{index:02d}" for index in range(20)],
                "rule_hashes": {f"rule-{index:02d}": f"sha-{index}" for index in range(20)},
                "thresholds": {"wnode": 0.5},
                "selector_config": {"k": list(range(1, 21))},
                "calibration_input_hash": "calibration-sha",
                "candidate_pool_hash": "pool-sha",
                "gnn_checkpoint_hash": "gine-sha",
                "molclr_checkpoint_hash": "molclr-sha",
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "stage",
    ["B6_PPO_SMOKE_V2", "B7_PPO_FULL", "B8_BASE_POOL", "B9_HIGH_TEMP_POOL"],
)
def test_policy_and_generation_stages_reject_test(stage: str) -> None:
    with pytest.raises(ValueError, match="may load train only"):
        validate_stage_data_access(stage=stage, requested_split="test")


def test_test_access_requires_complete_frozen_calibration_selector(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires the frozen B12"):
        validate_stage_data_access(
            stage="B13_TEST_EVALUATION", requested_split="test"
        )
    selector = _selector(tmp_path / "selector.json")
    contract = validate_stage_data_access(
        stage="B13_TEST_EVALUATION",
        requested_split="test",
        selector_manifest=selector,
    )
    assert contract["allowed"] is True
    assert contract["test_used_only_after_freeze"] is True
    assert contract["selector_freeze"]["manifest"]["test_used"] is False


def test_incomplete_or_test_tuned_selector_fails_closed(tmp_path: Path) -> None:
    selector = _selector(tmp_path / "selector.json")
    payload = json.loads(selector.read_text(encoding="utf-8"))
    payload["test_used"] = True
    selector.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selector freeze gate failed"):
        validate_stage_data_access(
            stage="B13_TEST_EVALUATION",
            requested_split="test",
            selector_manifest=selector,
        )


def test_b14_is_manifest_only_and_never_reloads_raw_test(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="may load manifest_only only"):
        validate_stage_data_access(stage="B14_FINAL_AUDIT", requested_split="test")
    selector = _selector(tmp_path / "selector.json")
    contract = validate_stage_data_access(
        stage="B14_FINAL_AUDIT",
        requested_split="manifest_only",
        selector_manifest=selector,
    )
    assert contract["allowed"] is True
    assert contract["raw_test_loaded"] is False


def test_official_b6_b7_runner_disables_validation_and_never_opens_test() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "train_bace_gnn_ppo.py").read_text(
        encoding="utf-8"
    )
    assert "args.val_dataset_path = None" in source
    assert "args.eval_every_steps = 0" in source
    assert "test.csv" not in source
    assert "calibration.csv" not in source
