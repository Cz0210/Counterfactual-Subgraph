from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines.globalgce_min_freq import (
    GlobalGCEMinFreqConfigurationError,
    bace_min_freq_grid,
    resolve_globalgce_min_freq,
    select_bace_min_freq,
)


def test_bace_grid_is_fixed_from_train_count() -> None:
    assert bace_min_freq_grid(360) == (2, 4, 7, 18)


def test_bace_requires_explicit_or_calibration_manifest(tmp_path: Path) -> None:
    with pytest.raises(GlobalGCEMinFreqConfigurationError, match="No GlobalGCE"):
        resolve_globalgce_min_freq("BACE")
    assert resolve_globalgce_min_freq("BACE", explicit_min_freq=7).value == 7


def test_bace_resolves_frozen_calibration_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "min_freq.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "BACE",
                "selected_min_freq": 4,
                "selection_split": "calibration",
                "test_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    result = resolve_globalgce_min_freq("BACE", calibration_manifest=manifest)
    assert result.value == 4
    assert result.source == "calibration_manifest"


def test_calibration_selection_uses_registered_tie_breaks() -> None:
    base = {
        "selection_split": "calibration",
        "test_loaded": False,
        "prefix_auc_k1_k10": 0.4,
        "multi_threshold_prefix_auc": 0.3,
        "cost": 0.02,
        "coverage_redundancy": 0.1,
        "rule_count": 20,
    }
    selected = select_bace_min_freq(
        [{**base, "min_freq": 7}, {**base, "min_freq": 4}]
    )
    assert int(selected["min_freq"]) == 4
