"""Downstream B8--B14 split-boundary tests (complements PPO boundary tests)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.bace_frozen_gnn_contracts import assert_stage_data_boundary


def _selection(path: Path, **overrides: object) -> Path:
    payload = {
        "dataset": "bace",
        "stage": "B12_SELECTOR",
        "status": "FROZEN",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "source_label": 1,
        "num_classes": 2,
        "selector_fitted_on_calibration": True,
        "selection_frozen": True,
        "K": 20,
        "test_loaded": False,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_b13_rejects_test_before_frozen_selector_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a frozen B12"):
        assert_stage_data_boundary(
            stage="B13_FINAL_EVAL",
            split_path=tmp_path / "test.csv",
            frozen_selection_manifest=None,
        )


def test_b13_rejects_nonfrozen_or_test_touched_selector(tmp_path: Path) -> None:
    manifest = _selection(tmp_path / "selection.json", selection_frozen=False)
    with pytest.raises(ValueError, match="incomplete B12"):
        assert_stage_data_boundary(
            stage="B13_FINAL_EVAL",
            split_path=tmp_path / "test.csv",
            frozen_selection_manifest=manifest,
        )
    manifest = _selection(tmp_path / "selection.json", test_loaded=True)
    with pytest.raises(ValueError, match="no-test boundary"):
        assert_stage_data_boundary(
            stage="B13_FINAL_EVAL",
            split_path=tmp_path / "test.csv",
            frozen_selection_manifest=manifest,
        )


def test_b13_authorizes_test_only_after_valid_b12_freeze(tmp_path: Path) -> None:
    manifest = _selection(tmp_path / "selection.json")
    payload = assert_stage_data_boundary(
        stage="B13_FINAL_EVAL",
        split_path=tmp_path / "test.csv",
        frozen_selection_manifest=manifest,
    )
    assert payload is not None
    assert payload["selection_frozen"] is True


def test_b8_and_b11_reject_test_named_split(tmp_path: Path) -> None:
    for stage in ("B8_POOL_BASE", "B9_POOL_HIGHTEMP", "B11_CROSS_PARENT_VERIFIED"):
        with pytest.raises(ValueError):
            assert_stage_data_boundary(stage=stage, split_path=tmp_path / "test.csv")
