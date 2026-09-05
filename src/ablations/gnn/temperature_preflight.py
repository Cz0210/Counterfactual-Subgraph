"""Explicit validation-temperature intent for newly created BACE ablations.

This module never reads logits, fits a temperature or edits a source config.
Historical sealed attempts must use a separately authorized first-fit repair;
they must not resume under the new effective configuration.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping


TRAINED_BACE_BACKBONES = frozenset({"gin", "gcn", "gatv2", "gatedgcn_plus"})


def validate_explicit_validation_temperature_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject the exact missing-flag failure before any new training starts."""
    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("Temperature preflight requires an explicit calibration mapping")
    if calibration.get("fit_on_validation") is not True:
        raise ValueError("Temperature preflight requires explicit fit_on_validation=true")
    if calibration.get("method") != "temperature_scaling":
        raise ValueError("Temperature preflight requires method=temperature_scaling")
    if calibration.get("split") != "validation":
        raise ValueError("Temperature fitting is validation-only; calibration/test are forbidden")
    max_iter = calibration.get("max_iter")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("Temperature preflight requires a positive integer max_iter")
    return {
        "state": "CONFIG_PREFLIGHT_PASS",
        "fit_on_validation": True,
        "method": "temperature_scaling",
        "split": "validation",
        "max_iter": max_iter,
        "temperature_fit_performed": False,
        "temperature_result_claimed": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def prepare_new_ablation_temperature_config(config: Mapping[str, Any], *, backbone: str) -> dict[str, Any]:
    """Copy the frozen training config and express the authorized new-run fit.

    Only omitted settings receive the historical fitter's defaults. Explicit
    contradictory settings fail rather than being silently rewritten. The main
    GINE is always adopted and may not enter this new-training helper.
    """
    if backbone not in TRAINED_BACE_BACKBONES:
        raise ValueError("Only the four new BACE ablations may configure a first temperature fit")
    result = copy.deepcopy(dict(config))
    calibration = result.setdefault("calibration", {})
    if not isinstance(calibration, dict):
        raise ValueError("Temperature preflight requires a calibration mapping")
    for name, value in {
        "fit_on_validation": True,
        "method": "temperature_scaling",
        "split": "validation",
        "max_iter": 100,
    }.items():
        calibration.setdefault(name, value)
    validate_explicit_validation_temperature_config(result)
    return result
