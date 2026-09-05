"""No model load or fit: configuration-only prevention of missing fit flags."""
import copy

import pytest

from src.ablations.gnn.temperature_preflight import (
    prepare_new_ablation_temperature_config,
    validate_explicit_validation_temperature_config,
)


def test_missing_flag_is_rejected_by_strict_preflight():
    with pytest.raises(ValueError, match="explicit fit_on_validation"):
        validate_explicit_validation_temperature_config({
            "calibration": {"method": "temperature_scaling", "split": "validation", "max_iter": 100}})


@pytest.mark.parametrize("backbone", ["gin", "gcn", "gatv2", "gatedgcn_plus"])
def test_new_config_records_fit_without_mutating_source(backbone):
    source = {"calibration": {"method": "temperature_scaling", "split": "validation"},
              "training": {"max_epochs": 100}, "gnn": {"backbone": backbone}}
    original = copy.deepcopy(source)
    result = prepare_new_ablation_temperature_config(source, backbone=backbone)
    assert source == original
    assert result["calibration"]["fit_on_validation"] is True
    assert result["calibration"]["max_iter"] == 100
    assert result["training"] == source["training"]
    gate = validate_explicit_validation_temperature_config(result)
    assert gate["state"] == "CONFIG_PREFLIGHT_PASS"
    assert gate["temperature_fit_performed"] is False
    assert gate["temperature_result_claimed"] is False
    assert gate["test_loaded"] is False


@pytest.mark.parametrize("field,value", [
    ("fit_on_validation", False), ("fit_on_validation", "true"),
    ("fit_on_validation", 1), ("split", "calibration"), ("split", "test"),
    ("method", "isotonic"), ("max_iter", True), ("max_iter", 0), ("max_iter", 100.0),
])
def test_explicit_contradictions_fail_not_silently_overwritten(field, value):
    source = {"calibration": {field: value}}
    original = copy.deepcopy(source)
    with pytest.raises(ValueError):
        prepare_new_ablation_temperature_config(source, backbone="gin")
    assert source == original


def test_gine_reference_cannot_enter_new_fit_path():
    with pytest.raises(ValueError, match="four new BACE ablations"):
        prepare_new_ablation_temperature_config({}, backbone="gine")


def test_explicit_fitter_iteration_contract_is_preserved():
    result = prepare_new_ablation_temperature_config({"calibration": {"max_iter": 3}}, backbone="gin")
    assert result["calibration"]["max_iter"] == 3


def test_new_cpu_effective_config_fills_historical_omission_without_source_write(tmp_path):
    from tests.test_bace_gnn_cpu_training import _bundle
    from src.ablations.gnn.cpu_training import effective_training_config, load_bundle, file_sha256
    import json
    root = _bundle(tmp_path)
    _, manifest = load_bundle(root)
    source = root / manifest["training_config_path"]
    config = json.loads(source.read_text())
    config["calibration"].pop("fit_on_validation")
    source.write_text(json.dumps(config))
    manifest["files"][manifest["training_config_path"]] = {
        "sha256": file_sha256(source), "size": source.stat().st_size}
    before = source.read_bytes()
    effective = effective_training_config(root, manifest, "gin")
    assert effective["calibration"]["fit_on_validation"] is True
    assert source.read_bytes() == before
