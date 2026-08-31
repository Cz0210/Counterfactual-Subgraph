from __future__ import annotations

import json

import pytest

from src.eval.frozen_threshold_manifest import load_shared_frozen_thresholds
from src.baselines.tastemolnet_globalgce_full import (
    load_threshold_contract as load_globalgce_threshold_contract,
)
from src.eval.tastemolnet_ours_full import load_threshold_contract
from src.eval.tastemolnet_threshold_authorities import (
    TasteThresholdAuthorityError,
    derive_shared_wnode_contract,
    derive_t7_neurosed_threshold,
)


def test_neurosed_threshold_is_linear_calibration_q30() -> None:
    payload = derive_t7_neurosed_threshold([float(value) for value in range(10)])
    assert payload["neurosed_distance_threshold"] == pytest.approx(2.7)
    assert payload["selection_split"] == "calibration"
    assert payload["inference_direction"] == "generated_query_to_original_target"
    assert payload["train_payload_loaded"] is False
    assert payload["validation_payload_loaded"] is False
    assert payload["test_payload_loaded"] is False
    assert payload["test_used_for_selection"] is False


@pytest.mark.parametrize(
    ("split", "test_loaded"),
    [("test", False), ("validation", False), ("calibration", True)],
)
def test_selectors_reject_noncalibration_or_test_access(
    split: str, test_loaded: bool
) -> None:
    with pytest.raises(TasteThresholdAuthorityError):
        derive_t7_neurosed_threshold(
            [0.1, 0.2], selection_split=split, test_loaded=test_loaded
        )
    with pytest.raises(TasteThresholdAuthorityError):
        derive_shared_wnode_contract(
            [0.1, 0.2], selection_split=split, test_loaded=test_loaded
        )


def test_shared_wnode_contract_loads_in_all_downstream_contracts(tmp_path) -> None:
    payload = derive_shared_wnode_contract(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    )
    path = tmp_path / "tastemolnet.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    generic = load_shared_frozen_thresholds(path)
    ours = load_threshold_contract(path)
    globalgce = load_globalgce_threshold_contract(path)
    assert generic["thresholds"] == payload["thresholds"]
    assert generic["theta_star"] == payload["theta_star"]
    assert ours.values == tuple(payload["thresholds"])
    assert ours.theta_star == payload["theta_star"]
    assert ours.cost_cap == payload["cost_cap"]
    assert ours.source_split == "calibration"
    assert globalgce.values == tuple(payload["thresholds"])
    assert globalgce.theta_star == payload["theta_star"]


def test_duplicate_wnode_quantiles_merge_deterministically() -> None:
    payload = derive_shared_wnode_contract([0.25] * 38)
    assert payload["thresholds"] == [0.25]
    assert payload["theta_star"] == 0.25
    assert payload["cost_cap"] == 0.25
    assert payload["duplicate_thresholds_merged"] is True
    assert payload["threshold_config_hash"]


@pytest.mark.parametrize(
    "values",
    [[], [float("nan")], [float("inf")], [-0.1, 0.2]],
)
def test_nonfinite_or_negative_distances_fail_closed(values) -> None:
    with pytest.raises(TasteThresholdAuthorityError):
        derive_t7_neurosed_threshold(values)
    with pytest.raises(TasteThresholdAuthorityError):
        derive_shared_wnode_contract(values)
