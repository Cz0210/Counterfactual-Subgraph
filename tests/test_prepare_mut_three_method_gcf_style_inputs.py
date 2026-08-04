from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_mut_three_method_gcf_style_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_mut_three", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_roots(tmp_path: Path, *, duplicate_ours: bool = False) -> dict[str, Path]:
    roots = {method: tmp_path / method.lower() for method in PREPARE.METHOD_ORDER}
    thresholds = list(PREPARE.FROZEN_THRESHOLDS)
    for method, root in roots.items():
        figure3: list[dict[str, object]] = []
        for k in range(1, 21):
            common = {
                "k": k,
                "conditional_median_cost": 0.2 - k / 1000,
                "num_parents": 217,
            }
            if method == "Ours":
                row = {
                    **common,
                    "ccrcov_theta_star": k / 40,
                    "applicable_rate": 0.8,
                }
            elif method == "CLEAR":
                row = {
                    **common,
                    "ccrcov_theta_star": k / 50,
                    "applicable_rate": 0.7,
                    "theta_star": PREPARE.THETA_STAR,
                }
            else:
                row = {
                    **common,
                    "close_cf_coverage": k / 60,
                    "num_applicable_parents": 130,
                    "threshold": PREPARE.THETA_STAR,
                }
            figure3.append(row)
        _write(root / "figure3_coverage_vs_k.csv", figure3)

        figure4: list[dict[str, object]] = []
        for k in (10, 20):
            for index, (label, threshold) in enumerate(thresholds, start=1):
                coverage = index / 10
                common4 = {"k": k, "threshold": threshold, "num_parents": 217}
                if method == "Ours":
                    row4 = {
                        **common4,
                        "quantile_label": label,
                        "coverage": coverage,
                        "num_covered": index * 10,
                    }
                elif method == "CLEAR":
                    row4 = {
                        **common4,
                        "threshold_name": label,
                        "ccrcov": coverage,
                        "coverage": coverage,
                        "num_covered": index * 10,
                    }
                else:
                    row4 = {
                        **common4,
                        "close_cf_coverage": coverage,
                        "num_close_cf_covered": index * 10,
                    }
                figure4.append(row4)
        if method == "Ours" and duplicate_ours:
            figure4.append(dict(figure4[0]))
        _write(root / "figure4_coverage_vs_threshold.csv", figure4)

        k10 = figure3[9]
        if method == "Ours":
            table = {
                "k": 10,
                "theta": PREPARE.THETA_STAR,
                "coverage": k10["ccrcov_theta_star"],
                "conditional_median_cost": k10["conditional_median_cost"],
                "applicable_rate": 0.8,
                "mean_cf_drop": 0.3,
                "num_test_parents": 217,
            }
            filename = "table2_ours_k10.csv"
        elif method == "CLEAR":
            table = {
                "k": 10,
                "theta": PREPARE.THETA_STAR,
                "ccrcov_theta_star": k10["ccrcov_theta_star"],
                "coverage": k10["ccrcov_theta_star"],
                "conditional_median_cost": k10["conditional_median_cost"],
                "applicable_rate": 0.7,
                "mean_cf_drop": 0.2,
                "num_parents": 217,
            }
            filename = "table2_clear_k10.csv"
        else:
            table = {
                "k": 10,
                "theta": PREPARE.THETA_STAR,
                "ccrcov": k10["close_cf_coverage"],
                "coverage": k10["close_cf_coverage"],
                "conditional_median_cost": k10["conditional_median_cost"],
                "applicable_coverage": 130 / 217,
                "avg_cf_drop_among_covered": 0.1,
                "num_test_parents": 217,
            }
            filename = "table2_globalgce_k10.csv"
        _write(root / filename, [table])
    return roots


def _args(roots: dict[str, Path], output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        ours_dir=str(roots["Ours"]),
        clear_dir=str(roots["CLEAR"]),
        globalgce_dir=str(roots["GlobalGCE"]),
        output_dir=str(output),
        theta_star=PREPARE.THETA_STAR,
        figure4_k=10,
        expected_num_parents=217,
    )


def test_normalizes_three_method_schemas_and_filters_k10(tmp_path: Path) -> None:
    roots = _build_roots(tmp_path)
    output = tmp_path / "output"
    result = PREPARE.run(_args(roots, output))

    assert result["manifest"]["methods"] == ["Ours", "CLEAR", "GlobalGCE"]
    with (output / "mut_three_method_figure3_coverage_cost_vs_k.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        figure3 = list(csv.DictReader(handle))
    with (output / "mut_three_method_figure4_coverage_vs_threshold.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        figure4 = list(csv.DictReader(handle))
    assert len(figure3) == 60
    assert len(figure4) == 21
    assert {int(row["k"]) for row in figure4} == {10}
    assert {row["threshold_name"] for row in figure4} == {
        "q05", "q10", "q20", "q30", "q50", "q70", "q90"
    }
    audit = (output / "mut_three_method_normalization_audit.txt").read_text(
        encoding="utf-8"
    )
    assert "[MUT_THREE_METHOD_NORMALIZATION_PASS]" in audit
    assert "best among completed three methods" in audit
    assert "all-baseline SOTA" not in audit


def test_identical_figure4_duplicate_is_safely_removed(tmp_path: Path) -> None:
    roots = _build_roots(tmp_path, duplicate_ours=True)
    output = tmp_path / "output"
    result = PREPARE.run(_args(roots, output))
    assert result["manifest"]["figure4_duplicate_rows_removed"]["Ours"] == 1


def test_conflicting_figure4_duplicate_fails(tmp_path: Path) -> None:
    roots = _build_roots(tmp_path, duplicate_ours=True)
    path = roots["Ours"] / "figure4_coverage_vs_threshold.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["coverage"] = "0.999"
    _write(path, rows)

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        PREPARE.run(_args(roots, tmp_path / "output"))


def test_truncated_theta_cost_is_not_accepted_as_primary_cost(tmp_path: Path) -> None:
    roots = _build_roots(tmp_path)
    path = roots["Ours"] / "figure3_coverage_vs_k.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("conditional_median_cost")
        row["theta_star_conditional_median_cost"] = "0.01"
    _write(path, rows)

    with pytest.raises(ValueError, match="untruncated conditional median cost"):
        PREPARE.run(_args(roots, tmp_path / "output"))
