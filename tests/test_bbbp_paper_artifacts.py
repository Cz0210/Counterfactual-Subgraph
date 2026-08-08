from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.bbbp_paper_artifacts import (
    CF_MODE,
    DISTANCE_LINE,
    FIGURE3_FIELDS,
    FIGURE4_FIELDS,
    METHODS,
    TABLE2_FIELDS,
    load_bbbp_thresholds,
)


def _threshold_payload() -> dict[str, object]:
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    return {
        "schema_version": "bbbp_wnode_thresholds_v1",
        "dataset": "BBBP",
        "distance_line": DISTANCE_LINE,
        "distance_type": "node_wasserstein",
        "cf_mode": CF_MODE,
        "threshold_source": "frozen_ours_calibration",
        "quantiles": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
        "thresholds": thresholds,
        "theta_star_quantile": 0.30,
        "theta_star": thresholds[3],
        "cost_cap_quantile": 0.90,
        "cost_cap": thresholds[-1],
        "selection_used_test": False,
        "threshold_fitted_on_test": False,
    }


def test_bbbp_plotting_schemas_and_common4_methods_are_frozen() -> None:
    assert FIGURE3_FIELDS == ("method", "k", "coverage", "cost")
    assert FIGURE4_FIELDS == ("method", "threshold", "coverage")
    assert TABLE2_FIELDS == (
        "method",
        "k",
        "coverage",
        "cost",
        "flip_rate",
        "cf_drop",
    )
    assert [METHODS[slug]["display"] for slug in METHODS] == [
        "Ours",
        "GlobalGCE",
        "GCFExplainer",
        "COMRECGC",
    ]
    assert DISTANCE_LINE == "MolCLR-Node-Wasserstein"
    assert CF_MODE == "strict_flip"


def test_bbbp_threshold_contract_loads_frozen_calibration_grid(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(_threshold_payload()), encoding="utf-8")
    loaded = load_bbbp_thresholds(path)
    assert loaded["threshold_source"] == "frozen_ours_calibration"
    assert loaded["selection_used_test"] is False
    assert loaded["threshold_fitted_on_test"] is False


@pytest.mark.parametrize("field", ["selection_used_test", "threshold_fitted_on_test"])
def test_bbbp_threshold_contract_rejects_test_leakage(
    tmp_path: Path, field: str
) -> None:
    payload = _threshold_payload()
    payload[field] = True
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test"):
        load_bbbp_thresholds(path)
