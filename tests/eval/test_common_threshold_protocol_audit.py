from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_common_threshold_protocol import audit_threshold_protocol


def _threshold(path: Path, *, source: str, shared: bool | None) -> Path:
    path.write_text(
        json.dumps(
            {
                "threshold_source": source,
                "quantiles": [0.3, 0.5],
                "thresholds": [0.01, 0.02],
                "theta_star_quantile": 0.3,
                "theta_star": 0.01,
                "shared_across_methods": shared,
                "threshold_fitted_on_test": False,
                "selection_used_test": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_missing_aids_and_method_specific_bace_requires_pooled_protocol(
    tmp_path: Path,
) -> None:
    mut = _threshold(tmp_path / "mut.json", source="calibration_pairs", shared=None)
    old = _threshold(tmp_path / "bace_old.json", source="frozen_ours_calibration", shared=None)
    connected = _threshold(
        tmp_path / "bace_connected.json",
        source="connected_ours_calibration_matrix",
        shared=True,
    )
    result = audit_threshold_protocol(
        aids_thresholds=[],
        mut_thresholds=[mut],
        bace_old_threshold=old,
        bace_connected_threshold=connected,
        output_dir=tmp_path / "audit",
    )
    assert result["Q30_PRE_REGISTERED_ACROSS_DATASETS"] is False
    assert result["Q30_USED_FOR_AIDS"] is False
    assert result["Q30_USED_FOR_MUT"] is True
    assert result["THRESHOLD_METHOD_INDEPENDENT"] is False
    assert result["final_test_allowed"] is False
    assert result["required_protocol"]["strict_primary"].endswith("q30")
