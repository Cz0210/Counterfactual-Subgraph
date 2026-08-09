from __future__ import annotations

import json
from pathlib import Path

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.bace_paper_artifacts import (
    freeze_bace_connected_thresholds_from_matrix,
    load_bace_thresholds,
)


def test_connected_thresholds_are_frozen_from_calibration_matrix_only(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    inputs = {
        "calibration_cohort_hash": "cohort",
        "calibration_csv": {"path": "/calibration.csv", "sha256": "a" * 64},
        "teacher_path": {"path": "/teacher.pkl", "sha256": "b" * 64},
        "molclr_checkpoint": {"path": "/model.pth", "sha256": "c" * 64},
        "distance_implementation_version": "wnode-v1",
        "wnode_size_penalty_beta": 0.0,
    }
    (matrix / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "run_complete": True,
                "test_loaded": False,
                "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
                "inputs": inputs,
            }
        ),
        encoding="utf-8",
    )
    (matrix / "matrix_audit.json").write_text(
        json.dumps(
            {
                "run_complete": True,
                "disconnected_residual_used_count": 0,
                "all_winning_residuals_connected": True,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "pair_strict_flip": True,
            "residual_connected": True,
            "sanitize_ok": True,
            "residual_num_components": 1,
            "contains_dot": False,
            "wnode_distance": value,
        }
        for value in (0.01, 0.02, 0.03, 0.04, 0.05)
    ]
    (matrix / "pair_matrix.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "thresholds.json"
    frozen = freeze_bace_connected_thresholds_from_matrix(
        calibration_matrix_dir=matrix,
        output_path=output,
    )
    loaded = load_bace_thresholds(output)
    assert loaded == frozen
    assert loaded["schema_version"] == "bace_wnode_thresholds_v2"
    assert loaded["threshold_fitted_on_test"] is False
    assert loaded["action_semantics_version"] == CONNECTED_ACTION_SEMANTICS


def test_connected_threshold_freeze_rejects_disconnected_winner(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "run_complete": True,
                "test_loaded": False,
                "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
                "inputs": {},
            }
        ),
        encoding="utf-8",
    )
    (matrix / "matrix_audit.json").write_text(
        json.dumps(
            {
                "run_complete": True,
                "disconnected_residual_used_count": 1,
                "all_winning_residuals_connected": False,
            }
        ),
        encoding="utf-8",
    )
    (matrix / "pair_matrix.jsonl").write_text("", encoding="utf-8")
    try:
        freeze_bace_connected_thresholds_from_matrix(
            calibration_matrix_dir=matrix,
            output_path=tmp_path / "thresholds.json",
        )
    except ValueError as exc:
        assert "disconnected" in str(exc).lower()
    else:
        raise AssertionError("Disconnected threshold source was accepted")
