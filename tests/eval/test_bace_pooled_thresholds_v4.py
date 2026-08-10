from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.freeze_bace_pooled_connected_thresholds import freeze_thresholds


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def test_pooled_thresholds_are_method_balanced_and_test_independent(tmp_path: Path) -> None:
    ours = tmp_path / "ours"
    ours.mkdir()
    (ours / "matrix_manifest.json").write_text(json.dumps({
        "test_loaded": False,
        "action_semantics_version": "connected_sanitized_residual_v1",
    }))
    (ours / "matrix_audit.json").write_text(json.dumps({"disconnected_residual_used_count": 0}))
    ours_rows = []
    for index in range(60):
        ours_rows.append({
            "parent_id": f"p{index}", "pair_strict_flip": True,
            "residual_connected": True, "sanitize_ok": True,
            "residual_num_components": 1, "contains_dot": False,
            "wnode_distance": 0.01 + index / 10000,
        })
    (ours / "pair_matrix.jsonl").write_text("".join(json.dumps(row)+"\n" for row in ours_rows))
    gcf = tmp_path / "gcf"
    (gcf / "details").mkdir(parents=True)
    (gcf / "run_config.json").write_text(json.dumps({"cf_mode": "strict_flip"}))
    _csv(
        gcf / "details" / "pair_details.csv",
        ("parent_id", "teacher_strict_flip", "delete_valid", "candidate_smiles", "distance"),
        [{"parent_id": f"p{i}", "teacher_strict_flip": True, "delete_valid": True, "candidate_smiles": "CC", "distance": 0.02+i/10000} for i in range(60)],
    )
    calibration = tmp_path / "calibration.csv"
    _csv(calibration, ("molecule_id",), [{"molecule_id": f"p{i}"} for i in range(60)])

    result = freeze_thresholds(
        ours_matrix_root=ours,
        gcf_calibration_root=gcf,
        calibration_csv=calibration,
        output_dir=tmp_path / "out",
    )

    contract = result["thresholds"]
    assert contract["method_weights"] == {"ours": 0.5, "gcfexplainer": 0.5}
    assert contract["threshold_fitted_on_test"] is False
    assert contract["theta_star"] == contract["thresholds"][3]
    assert contract["standard_sensitivity_threshold"] == contract["thresholds"][4]
