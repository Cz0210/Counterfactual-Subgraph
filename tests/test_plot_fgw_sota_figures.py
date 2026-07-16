from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_fgw_sota_figures.py"
SPEC = importlib.util.spec_from_file_location("plot_fgw_sota_figures", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_figure3_prefers_actual_conditional_median_cost_and_normalizes_methods(tmp_path: Path) -> None:
    path = tmp_path / "fgw_q30_k10_main_figure3_fgw_coverage_cost_vs_k.csv"
    raw_method_names = {
        "Ours": "ours_selected_subgraphs",
        "GlobalGCE": "GlobalGCE-Frequency-Top20",
        "CLEAR": "CLEAR-ParentFrequency-Top20",
        "GCFExplainer": "GCFExplainer-Top20",
    }
    rows: list[dict[str, object]] = []
    for normalized_name, raw_name in raw_method_names.items():
        for k in range(1, 21):
            rows.append(
                {
                    "method": raw_name,
                    "k": k,
                    "theta": 0.0328,
                    "coverage": k / 20,
                    "conditional_median_cost": 0.01 + k / 10000,
                    "theta_covered_conditional_median_cost": 0.99,
                }
            )
    _write_csv(path, list(rows[0]), rows)

    parsed, audit = PLOT.load_figure3_rows(path, q30=0.0328363645853374)

    assert audit["conditional_cost_field"] == "conditional_median_cost"
    assert {row.method for row in parsed} == set(raw_method_names)
    ours_k10 = next(row for row in parsed if row.method == "Ours" and row.k == 10)
    assert ours_k10.conditional_median_cost == pytest.approx(0.011)


def test_figure4_rejects_non_k20_input_to_prevent_wrong_curve(tmp_path: Path) -> None:
    path = tmp_path / "figure4.csv"
    rows = [
        {
            "method": method,
            "k": 10,
            "threshold": 0.0,
            "coverage": 0.0,
        }
        for method in ("Ours", "GlobalGCE", "CLEAR", "GCFExplainer")
    ]
    _write_csv(path, list(rows[0]), rows)

    with pytest.raises(ValueError, match="K=20"):
        PLOT.load_figure4_rows(path)


def test_write_audit_uses_lowercase_auc_schema_and_reports_actual_values(tmp_path: Path) -> None:
    auc_rows = [
        {
            "method": "Ours",
            "auc_min": 0.0,
            "auc_max": 0.0328,
            "low_cost_normalized_auc": 0.20,
            "coverage_at_q30": 0.38,
        },
        {
            "method": "GCFExplainer",
            "auc_min": 0.0,
            "auc_max": 0.0328,
            "low_cost_normalized_auc": 0.15,
            "coverage_at_q30": 0.37,
        },
    ]
    table_rows = [
        {"method": "Ours", "coverage": 0.38, "conditional_median_cost": 0.02},
        {
            "method": "GCFExplainer",
            "coverage": 0.37,
            "conditional_median_cost": 0.03,
        },
    ]
    PLOT._write_audit(
        tmp_path,
        figure3_audit={
            "figure3_csv": "figure3.csv",
            "conditional_cost_field": "conditional_median_cost",
            "selected_theta": 0.0328,
            "theta_delta_from_q30": 0.0,
        },
        figure4_audit={"figure4_csv": "figure4.csv", "selected_k": 20},
        table_rows=table_rows,
        auc_rows=auc_rows,
        q20=0.02,
        q30=0.0328,
        figure4_display_min=0.015,
    )

    audit = (tmp_path / "sota_presentation_audit.txt").read_text(encoding="utf-8")
    assert "K=10 q30 coverage SOTA: True" in audit
    assert "K=10 q30 conditional cost SOTA: True" in audit
    assert "[0,q30] normalized AUC SOTA: True" in audit
    assert "Ours low-cost normalized AUC: 0.2" in audit
    assert "Best baseline low-cost normalized AUC: 0.15" in audit


def test_auc_schema_validation_reports_actual_keys() -> None:
    with pytest.raises(ValueError, match="actual keys=.*Method"):
        PLOT._validate_auc_rows(
            [
                {
                    "Method": "Ours",
                    "auc_min": 0.0,
                    "auc_max": 0.0328,
                    "low_cost_normalized_auc": 0.2,
                    "coverage_at_q30": 0.38,
                }
            ]
        )


def test_normalized_auc_uses_canonical_snake_case_schema() -> None:
    rows = [
        PLOT.Figure4Row(
            method="Ours",
            k=20,
            threshold=0.0,
            coverage=0.0,
            mean=None,
            lower=None,
            upper=None,
        ),
        PLOT.Figure4Row(
            method="Ours",
            k=20,
            threshold=0.0328,
            coverage=1.0,
            mean=None,
            lower=None,
            upper=None,
        ),
    ]

    result = PLOT._normalized_auc(rows, q30=0.0328)

    assert set(result) == {"auc_min", "auc_max", "low_cost_normalized_auc"}
    assert result["auc_min"] == 0.0
    assert result["auc_max"] == pytest.approx(0.0328)
    assert result["low_cost_normalized_auc"] == pytest.approx(0.5)
