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
