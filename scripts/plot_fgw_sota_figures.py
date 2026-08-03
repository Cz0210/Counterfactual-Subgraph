#!/usr/bin/env python3
"""Create presentation-ready MolCLR-Node-FGW figures from existing CSV outputs.

This is an intentionally read-only post-processing tool. It does not load a
teacher, compute embeddings or FGW distances, change candidate ordering, or
rewrite evaluator outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ("Ours", "GlobalGCE", "CLEAR", "GCFExplainer")
METHOD_COLORS = {
    "Ours": "#1b1b1b",
    "GlobalGCE": "#d97706",
    "CLEAR": "#18864b",
    "GCFExplainer": "#2563a8",
}
METHOD_MARKERS = {"Ours": "o", "GlobalGCE": "s", "CLEAR": "^", "GCFExplainer": "D"}
FIGURE3_PRIORITY_FILENAMES = (
    "fgw_q30_k10_main_figure3_fgw_coverage_cost_vs_k.csv",
    "figure3_fgw_coverage_cost_vs_k.csv",
)
CONDITIONAL_COST_FIELD_CANDIDATES = (
    "conditional_median_cost",
    "Conditional median cost",
    "theta_covered_conditional_median_cost",
    "covered_conditional_median_cost",
    "conditional_median_cost_covered",
)
EPS = 1e-12
REQUIRED_AUC_KEYS = frozenset(
    {
        "method",
        "auc_min",
        "auc_max",
        "low_cost_normalized_auc",
        "coverage_at_q30",
    }
)


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _normalize_method(value: Any) -> str | None:
    key = _normalize_key(value)
    if key in {"ours", "oursselectedsubgraphs", "oursselectedsubgraph"} or key.startswith("ours"):
        return "Ours"
    if key.startswith("globalgce"):
        return "GlobalGCE"
    if key.startswith("clear"):
        return "CLEAR"
    if key.startswith("gcfexplainer") or key.startswith("gcf"):
        return "GCFExplainer"
    return None


def _as_float(value: Any, *, name: str, row_number: int) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name!r} in CSV row {row_number}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name!r} in CSV row {row_number}: {value!r}")
    return result


def _as_float_or_nan(value: Any, *, name: str, row_number: int) -> float:
    """Parse an optional plotting value without inventing a fallback metric."""
    if value is None or not str(value).strip():
        return float("nan")
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name!r} in CSV row {row_number}: {value!r}") from exc
    return parsed if math.isfinite(parsed) else float("nan")


def _as_int(value: Any, *, name: str, row_number: int) -> int:
    parsed = _as_float(value, name=name, row_number=row_number)
    if not parsed.is_integer():
        raise ValueError(f"Expected integer {name!r} in CSV row {row_number}: {value!r}")
    return int(parsed)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _field(headers: Sequence[str], candidates: Sequence[str], *, label: str) -> str:
    by_normalized = {_normalize_key(header): header for header in headers}
    for candidate in candidates:
        found = by_normalized.get(_normalize_key(candidate))
        if found is not None:
            return found
    raise ValueError(
        f"Could not find {label}; supported names={list(candidates)}; available={list(headers)}"
    )


def _optional_field(headers: Sequence[str], candidates: Sequence[str]) -> str | None:
    try:
        return _field(headers, candidates, label="optional field")
    except ValueError:
        return None


def _find_figure3_csv(report_dir: Path) -> Path:
    if report_dir.is_file():
        return report_dir
    if not report_dir.is_dir():
        raise FileNotFoundError(f"Figure 3 report directory does not exist: {report_dir}")
    for filename in FIGURE3_PRIORITY_FILENAMES:
        direct = report_dir / filename
        if direct.is_file():
            return direct
        nested = sorted(report_dir.rglob(filename))
        if nested:
            return nested[0]
    candidates = sorted(report_dir.rglob("*figure3*coverage*cost*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No Figure 3 coverage/cost CSV found under: {report_dir}; "
            f"looked for {list(FIGURE3_PRIORITY_FILENAMES)}"
        )
    raise ValueError(f"Ambiguous Figure 3 CSV candidates under {report_dir}: {candidates}")


@dataclass(frozen=True)
class Figure3Row:
    method: str
    k: int
    theta: float
    coverage: float
    conditional_median_cost: float


@dataclass(frozen=True)
class Figure4Row:
    method: str
    k: int | None
    threshold: float
    coverage: float
    mean: float | None
    lower: float | None
    upper: float | None
    threshold_name: str | None = None


def _nearest_theta(rows: Sequence[Figure3Row], q30: float) -> float:
    values = sorted({row.theta for row in rows})
    if not values:
        raise ValueError("Figure 3 CSV has no usable rows.")
    return min(values, key=lambda value: abs(value - q30))


def _parse_expected_methods(value: str | Sequence[str]) -> tuple[str, ...]:
    raw_values = value.split(",") if isinstance(value, str) else list(value)
    methods: list[str] = []
    for raw_value in raw_values:
        method = _normalize_method(raw_value)
        if method is None:
            raise ValueError(f"Unknown expected method: {raw_value!r}")
        if method in methods:
            raise ValueError(f"Duplicate expected method: {method}")
        methods.append(method)
    if not methods:
        raise ValueError("At least one expected method is required.")
    return tuple(methods)


def load_figure3_rows(
    path: Path,
    *,
    q30: float,
    expected_methods: Sequence[str] = METHOD_ORDER,
) -> tuple[list[Figure3Row], dict[str, Any]]:
    headers, raw_rows = _read_csv(path)
    method_field = _field(headers, ("method",), label="method field")
    k_field = _field(headers, ("k", "K"), label="K field")
    theta_field = _field(headers, ("theta", "threshold"), label="theta field")
    coverage_field = _field(headers, ("coverage", "ccrcov", "close_cf_coverage"), label="coverage field")
    cost_field = _field(headers, CONDITIONAL_COST_FIELD_CANDIDATES, label="conditional cost field")
    parsed: list[Figure3Row] = []
    ignored_methods: set[str] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        method = _normalize_method(row.get(method_field))
        if method is None:
            ignored_methods.add(str(row.get(method_field) or ""))
            continue
        parsed.append(
            Figure3Row(
                method=method,
                k=_as_int(row.get(k_field), name=k_field, row_number=row_number),
                theta=_as_float(row.get(theta_field), name=theta_field, row_number=row_number),
                coverage=_as_float(row.get(coverage_field), name=coverage_field, row_number=row_number),
                # A method can have no valid conditional recourse at a small
                # prefix. Preserve that as NaN so matplotlib leaves a gap;
                # never substitute another cost definition.
                conditional_median_cost=_as_float_or_nan(
                    row.get(cost_field), name=cost_field, row_number=row_number
                ),
            )
        )
    selected_theta = _nearest_theta(parsed, q30)
    selected = [row for row in parsed if math.isclose(row.theta, selected_theta, abs_tol=EPS)]
    if not selected:
        raise ValueError(f"No Figure 3 rows at selected theta={selected_theta}")
    per_key: set[tuple[str, int]] = set()
    for row in selected:
        key = (row.method, row.k)
        if key in per_key:
            raise ValueError(f"Duplicate Figure 3 row after theta selection: {key}")
        per_key.add(key)
    methods = {row.method for row in selected}
    missing = set(expected_methods) - methods
    if missing:
        raise ValueError(f"Figure 3 CSV is missing standardized methods: {sorted(missing)}")
    unexpected = methods - set(expected_methods)
    if unexpected:
        raise ValueError(f"Figure 3 CSV contains unexpected methods: {sorted(unexpected)}")
    return sorted(selected, key=lambda row: (expected_methods.index(row.method), row.k)), {
        "figure3_csv": str(path),
        "source_columns": headers,
        "method_field": method_field,
        "k_field": k_field,
        "theta_field": theta_field,
        "coverage_field": coverage_field,
        "conditional_cost_field": cost_field,
        "selected_theta": selected_theta,
        "theta_delta_from_q30": selected_theta - q30,
        "expected_methods": list(expected_methods),
        "ignored_unrecognized_methods": sorted(value for value in ignored_methods if value),
    }


def load_figure4_rows(
    path: Path,
    *,
    expected_methods: Sequence[str] = METHOD_ORDER,
    selected_k: int = 20,
) -> tuple[list[Figure4Row], dict[str, Any]]:
    headers, raw_rows = _read_csv(path)
    method_field = _field(headers, ("method",), label="method field")
    threshold_field = _field(headers, ("threshold", "theta"), label="threshold field")
    coverage_field = _field(headers, ("coverage", "ccrcov", "close_cf_coverage"), label="coverage field")
    k_field = _optional_field(headers, ("k", "K"))
    mean_field = _optional_field(headers, ("mean", "coverage_mean"))
    lower_field = _optional_field(headers, ("lower", "ci_lower"))
    upper_field = _optional_field(headers, ("upper", "ci_upper"))
    threshold_name_field = _optional_field(
        headers, ("threshold_name", "quantile_label")
    )
    parsed: list[Figure4Row] = []
    ignored_methods: set[str] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        method = _normalize_method(row.get(method_field))
        if method is None:
            ignored_methods.add(str(row.get(method_field) or ""))
            continue
        parsed.append(
            Figure4Row(
                method=method,
                k=_as_int(row.get(k_field), name=k_field, row_number=row_number) if k_field else None,
                threshold=_as_float(row.get(threshold_field), name=threshold_field, row_number=row_number),
                coverage=_as_float(row.get(coverage_field), name=coverage_field, row_number=row_number),
                mean=_as_float(row.get(mean_field), name=mean_field, row_number=row_number) if mean_field and str(row.get(mean_field) or "").strip() else None,
                lower=_as_float(row.get(lower_field), name=lower_field, row_number=row_number) if lower_field and str(row.get(lower_field) or "").strip() else None,
                upper=_as_float(row.get(upper_field), name=upper_field, row_number=row_number) if upper_field and str(row.get(upper_field) or "").strip() else None,
                threshold_name=(
                    str(row.get(threshold_name_field) or "").strip()
                    if threshold_name_field
                    else None
                ),
            )
        )
    if k_field is not None:
        observed_k = sorted({row.k for row in parsed if row.k is not None})
        if selected_k not in observed_k:
            raise ValueError(
                f"Figure 4 requires K={selected_k}; "
                f"observed K values={observed_k}"
            )
        parsed = [row for row in parsed if row.k == selected_k]
    methods = {row.method for row in parsed}
    missing = set(expected_methods) - methods
    if missing:
        raise ValueError(f"Figure 4 CSV is missing standardized methods: {sorted(missing)}")
    unexpected = methods - set(expected_methods)
    if unexpected:
        raise ValueError(f"Figure 4 CSV contains unexpected methods: {sorted(unexpected)}")
    per_key: set[tuple[str, float]] = set()
    for row in parsed:
        key = (row.method, row.threshold)
        if key in per_key:
            raise ValueError(f"Duplicate Figure 4 row: {key}")
        per_key.add(key)
    return sorted(parsed, key=lambda row: (expected_methods.index(row.method), row.threshold)), {
        "figure4_csv": str(path),
        "source_columns": headers,
        "method_field": method_field,
        "k_field": k_field,
        "threshold_field": threshold_field,
        "coverage_field": coverage_field,
        "mean_field": mean_field,
        "lower_field": lower_field,
        "upper_field": upper_field,
        "threshold_name_field": threshold_name_field,
        "ignored_unrecognized_methods": sorted(value for value in ignored_methods if value),
        "selected_k": selected_k if k_field is not None else None,
        "expected_methods": list(expected_methods),
    }


def _method_rows(rows: Sequence[Any], method: str) -> list[Any]:
    return [row for row in rows if row.method == method]


def _require_prefix(
    rows: Sequence[Figure3Row],
    max_k: int,
    methods: Sequence[str] = METHOD_ORDER,
) -> None:
    expected = list(range(1, max_k + 1))
    for method in methods:
        observed = [row.k for row in _method_rows(rows, method) if row.k <= max_k]
        if observed != expected:
            raise ValueError(f"{method} Figure 3 K values must be {expected}; observed={observed}")


def _import_matplotlib() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save_figure3(
    rows: Sequence[Figure3Row],
    *,
    max_k: int,
    q30: float,
    path_png: Path,
    path_pdf: Path,
    supplemental: bool,
    methods: Sequence[str] = METHOD_ORDER,
    dataset_name: str = "AIDS",
) -> None:
    _require_prefix(rows, max_k, methods)
    _matplotlib, plt = _import_matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(7.4, 7.9), sharex=True)
    for method in methods:
        method_rows = [row for row in _method_rows(rows, method) if row.k <= max_k]
        x = [row.k for row in method_rows]
        axes[0].plot(x, [row.coverage for row in method_rows], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], linewidth=1.9, markersize=4.2, label=method)
        axes[1].plot(x, [row.conditional_median_cost for row in method_rows], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], linewidth=1.9, markersize=4.2, label=method)
    axes[0].set_ylabel("CCRCov at q30" if dataset_name == "Mutagenicity" else "Coverage / CCRCov")
    axes[1].set_ylabel(
        "Conditional median cost\namong applicable strict-flip parents"
        if dataset_name == "Mutagenicity"
        else "Conditional median cost\n(MolCLR-Node-FGW)"
    )
    axes[1].set_xlabel("Prefix K")
    axes[0].set_title(
        f"{dataset_name}: theta = {q30:.4f}"
        if dataset_name == "Mutagenicity"
        else f"theta = {q30:.4f}"
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(1, max_k)
    axes[1].set_xticks(list(range(1, max_k + 1)) if max_k <= 10 else [1, 5, 10, 15, 20])
    if supplemental:
        for axis in axes:
            axis.axvline(10, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
        axes[0].text(10.15, axes[0].get_ylim()[0] + 0.03 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]), "Primary budget K=10", color="#555555", fontsize=8)
    axes[0].legend(ncol=2, frameon=False, loc="best")
    figure.text(
        0.5,
        0.01,
        (
            "GCFExplainer is pending and is not included in this preliminary three-method comparison."
            if dataset_name == "Mutagenicity"
            else "Conditional cost is the unified-evaluator field; it is not labeled as the original GCFExplainer paper-style unconditional cost."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.savefig(path_png, dpi=300, bbox_inches="tight")
    figure.savefig(path_pdf, bbox_inches="tight")
    plt.close(figure)


def _interpolated_coverage(rows: Sequence[Figure4Row], target: float) -> float:
    x = np.asarray([row.threshold for row in rows], dtype=float)
    y = np.asarray([row.coverage for row in rows], dtype=float)
    if x.size < 2 or target < x.min() - EPS or target > x.max() + EPS:
        raise ValueError(f"Cannot interpolate coverage at threshold={target}; available=[{x.min()}, {x.max()}]")
    return float(np.interp(float(target), x, y))


def _interval_curve(rows: Sequence[Figure4Row], lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    if upper <= lower:
        raise ValueError(f"Invalid integration interval: [{lower}, {upper}]")
    x = np.asarray([row.threshold for row in rows], dtype=float)
    y = np.asarray([row.coverage for row in rows], dtype=float)
    if x.size < 2 or lower < x.min() - EPS or upper > x.max() + EPS:
        raise ValueError(
            f"Figure 4 curve does not cover [{lower}, {upper}]; available=[{x.min()}, {x.max()}]"
        )
    inside = (x > lower) & (x < upper)
    x_out = np.concatenate(([lower], x[inside], [upper]))
    y_out = np.concatenate(([_interpolated_coverage(rows, lower)], y[inside], [_interpolated_coverage(rows, upper)]))
    return x_out, y_out


def _normalized_auc(rows: Sequence[Figure4Row], *, q30: float) -> dict[str, float]:
    x, y = _interval_curve(rows, 0.0, q30)
    area = float(np.trapezoid(y, x))
    return {
        "auc_min": 0.0,
        "auc_max": q30,
        "low_cost_normalized_auc": area / q30,
    }


def _save_figure4(
    rows: Sequence[Figure4Row],
    *,
    q20: float,
    q30: float,
    lower_display: float,
    upper_display: float,
    title: str,
    path_png: Path,
    path_pdf: Path,
    methods: Sequence[str] = METHOD_ORDER,
    quantile_mode: bool = False,
) -> None:
    _matplotlib, plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(7.4, 4.9))
    quantile_ticks: dict[float, str] = {}
    for method in methods:
        method_rows = [row for row in _method_rows(rows, method) if lower_display - EPS <= row.threshold <= upper_display + EPS]
        if not method_rows:
            raise ValueError(f"No Figure 4 data for {method} in [{lower_display}, {upper_display}]")
        x = np.asarray([row.threshold for row in method_rows], dtype=float)
        coverage = np.asarray([row.coverage for row in method_rows], dtype=float)
        axis.plot(
            x,
            coverage,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method] if quantile_mode else None,
            linewidth=1.9,
            label=method,
        )
        if quantile_mode:
            for row in method_rows:
                if row.threshold_name:
                    quantile_ticks.setdefault(row.threshold, row.threshold_name)
        lower = [row.lower for row in method_rows]
        upper = [row.upper for row in method_rows]
        if all(value is not None for value in lower) and all(value is not None for value in upper):
            axis.fill_between(x, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
    if not quantile_mode and lower_display <= q20 <= upper_display:
        axis.axvline(q20, color="#666666", linestyle=":", linewidth=1.0, label="q20")
    if lower_display <= q30 <= upper_display:
        axis.axvline(
            q30,
            color="#444444",
            linestyle="--",
            linewidth=1.0,
            label="Primary threshold q30" if quantile_mode else "q30",
        )
    axis.set_xlim(lower_display, upper_display)
    axis.set_ylim(bottom=0)
    axis.set_xlabel(
        "MolCLR-Node-Wasserstein threshold"
        if quantile_mode
        else "MolCLR-Node-FGW threshold"
    )
    axis.set_ylabel("CCRCov")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    if quantile_mode and quantile_ticks:
        values = sorted(quantile_ticks)
        axis.set_xticks(
            values,
            [f"{quantile_ticks[value]}\n{value:.4f}" for value in values],
        )
    axis.legend(ncol=2, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(path_png, dpi=300, bbox_inches="tight")
    figure.savefig(path_pdf, bbox_inches="tight")
    plt.close(figure)


def _save_table2(
    rows: Sequence[Figure3Row],
    *,
    q30: float,
    output_dir: Path,
    methods: Sequence[str] = METHOD_ORDER,
    dataset_name: str = "AIDS",
) -> list[dict[str, Any]]:
    k10 = {row.method: row for row in rows if row.k == 10}
    missing = set(methods) - set(k10)
    if missing:
        raise ValueError(f"Table 2 needs K=10 rows for all methods; missing={sorted(missing)}")
    nonfinite_cost_methods = [
        method for method, row in k10.items() if not math.isfinite(row.conditional_median_cost)
    ]
    if nonfinite_cost_methods:
        raise ValueError(
            "Table 2 requires a finite conditional_median_cost at K=10; "
            f"missing for {sorted(nonfinite_cost_methods)}"
        )
    max_coverage = max(row.coverage for row in k10.values())
    min_cost = min(row.conditional_median_cost for row in k10.values())
    table_rows = [
        {
            "method": method,
            "coverage": k10[method].coverage,
            "conditional_median_cost": k10[method].conditional_median_cost,
            "k": 10,
            "theta": q30,
            "coverage_is_best": k10[method].coverage >= max_coverage - EPS,
            "cost_is_best": k10[method].conditional_median_cost <= min_cost + EPS,
        }
        for method in methods
    ]
    display_fields = ("Method", "Coverage \u2191", "Conditional median cost \u2193")
    display_rows = [
        {
            "Method": row["method"],
            "Coverage \u2191": row["coverage"],
            "Conditional median cost \u2193": row["conditional_median_cost"],
        }
        for row in table_rows
    ]
    table_stem = (
        "mut_table2_k10_q30_three_method"
        if dataset_name == "Mutagenicity"
        else "table2_main_k10_q30_compact"
    )
    _write_csv(output_dir / f"{table_stem}.csv", display_rows, display_fields)
    markdown = [
        "| Method | Coverage ↑ | Conditional median cost ↓ |",
        "| --- | ---: | ---: |",
    ]
    for row in table_rows:
        coverage = f"{row['coverage']:.4f}"
        cost = f"{row['conditional_median_cost']:.4f}"
        if row["coverage_is_best"] and (
            dataset_name == "Mutagenicity" or row["method"] == "Ours"
        ):
            coverage = f"**{coverage}**"
        if row["cost_is_best"] and (
            dataset_name == "Mutagenicity" or row["method"] == "Ours"
        ):
            cost = f"**{cost}**"
        markdown.append(f"| {row['method']} | {coverage} | {cost} |")
    footnote = (
        "Mutagenicity, 217 test parents, strict-flip, MolCLR-Node-Wasserstein. "
        "GCFExplainer is pending and is not included in this preliminary three-method comparison."
        if dataset_name == "Mutagenicity"
        else "MolCLR-Node-FGW, lambda=0.5, strict-flip evaluation."
    )
    markdown.extend(("", footnote))
    (output_dir / f"{table_stem}.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    _matplotlib, plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(7.4, 2.6))
    axis.axis("off")
    table = axis.table(
        cellText=[
            [
                row["method"],
                f"{row['coverage']:.4f}",
                f"{row['conditional_median_cost']:.4f}",
            ]
            for row in table_rows
        ],
        colLabels=list(display_fields),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row_index, col_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        elif dataset_name == "Mutagenicity" or table_rows[row_index - 1]["method"] == "Ours":
            if col_index == 1 and table_rows[row_index - 1]["coverage_is_best"]:
                cell.set_text_props(weight="bold")
            if col_index == 2 and table_rows[row_index - 1]["cost_is_best"]:
                cell.set_text_props(weight="bold")
    figure.text(0.5, 0.03, footnote, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(output_dir / f"{table_stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{table_stem}.pdf", bbox_inches="tight")
    plt.close(figure)
    return table_rows


def _validate_auc_rows(auc_rows: Sequence[dict[str, Any]]) -> None:
    if not auc_rows:
        raise ValueError("AUC audit rows are empty.")
    seen_methods: set[str] = set()
    numeric_keys = REQUIRED_AUC_KEYS - {"method"}
    for row_index, row in enumerate(auc_rows):
        missing = REQUIRED_AUC_KEYS - set(row)
        if missing:
            raise ValueError(
                f"AUC row {row_index} is missing required keys {sorted(missing)}; "
                f"actual keys={sorted(row)}"
            )
        method = str(row["method"])
        if not method:
            raise ValueError(f"AUC row {row_index} has an empty method.")
        if method in seen_methods:
            raise ValueError(f"AUC rows contain duplicate method={method!r}.")
        seen_methods.add(method)
        for key in numeric_keys:
            try:
                value = float(row[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"AUC row {row_index} has non-numeric {key}={row[key]!r}."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"AUC row {row_index} has non-finite {key}={value!r}.")
    if "Ours" not in seen_methods:
        raise ValueError(f"AUC rows do not contain method='Ours'; methods={sorted(seen_methods)}")
    if len(seen_methods) < 2:
        raise ValueError("AUC SOTA audit requires Ours and at least one baseline method.")


def _write_audit(
    output_dir: Path,
    *,
    figure3_audit: dict[str, Any],
    figure4_audit: dict[str, Any],
    table_rows: Sequence[dict[str, Any]],
    auc_rows: Sequence[dict[str, Any]],
    q20: float,
    q30: float,
    figure4_display_min: float,
) -> None:
    _validate_auc_rows(auc_rows)
    ours_table = next(row for row in table_rows if row["method"] == "Ours")
    baseline_table_rows = [row for row in table_rows if row["method"] != "Ours"]
    if not baseline_table_rows:
        raise ValueError("SOTA audit requires at least one baseline Table 2 row.")
    ours_auc_row = next(row for row in auc_rows if row["method"] == "Ours")
    baseline_auc_rows = [row for row in auc_rows if row["method"] != "Ours"]

    ours_coverage = float(ours_table["coverage"])
    best_baseline_coverage = max(float(row["coverage"]) for row in baseline_table_rows)
    ours_cost = float(ours_table["conditional_median_cost"])
    best_baseline_cost = min(
        float(row["conditional_median_cost"]) for row in baseline_table_rows
    )
    ours_auc = float(ours_auc_row["low_cost_normalized_auc"])
    best_baseline_auc = max(
        float(row["low_cost_normalized_auc"]) for row in baseline_auc_rows
    )

    coverage_sota = ours_coverage >= best_baseline_coverage - EPS
    cost_sota = ours_cost <= best_baseline_cost + EPS
    auc_sota = ours_auc >= best_baseline_auc - EPS
    claim_allowed = coverage_sota and cost_sota and auc_sota
    content = f"""FGW SOTA presentation audit

Figure 3 source: {figure3_audit['figure3_csv']}
Figure 3 conditional cost source field: {figure3_audit['conditional_cost_field']}
Figure 3 selected theta: {figure3_audit['selected_theta']:.16g}
Figure 3 delta from requested q30: {figure3_audit['theta_delta_from_q30']:.16g}
Figure 4 source: {figure4_audit['figure4_csv']}
Figure 4 selected K: {figure4_audit['selected_k']}
q20: {q20:.16g}
q30: {q30:.16g}
Figure 4 display interval: [{figure4_display_min:.16g}, {q30:.16g}]
Low-cost AUC interval: [0, {q30:.16g}]

K=10 q30 coverage SOTA: {coverage_sota}
K=10 q30 conditional cost SOTA: {cost_sota}
[0,q30] normalized AUC SOTA: {auc_sota}
Ours K=10 coverage: {ours_coverage:.16g}
Best baseline K=10 coverage: {best_baseline_coverage:.16g}
Ours K=10 conditional cost: {ours_cost:.16g}
Best baseline K=10 conditional cost: {best_baseline_cost:.16g}
Ours low-cost normalized AUC: {ours_auc:.16g}
Best baseline low-cost normalized AUC: {best_baseline_auc:.16g}
low-cost and compact-budget SOTA claim allowed: {claim_allowed}

Permitted claim only when all three statements above are True:
low-cost and compact-budget SOTA

Do not claim all-K and all-threshold SOTA from these checks.
"""
    (output_dir / "sota_presentation_audit.txt").write_text(content, encoding="utf-8")


def _write_quantile_audit(
    output_dir: Path,
    *,
    figure3_audit: dict[str, Any],
    figure4_audit: dict[str, Any],
    table_rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    dataset_name: str,
    q30: float,
) -> None:
    if {str(row["method"]) for row in table_rows} != set(methods):
        raise ValueError("Quantile audit Table 2 methods do not match expected methods.")
    best_coverage = max(table_rows, key=lambda row: float(row["coverage"]))
    best_cost = min(
        table_rows, key=lambda row: float(row["conditional_median_cost"])
    )
    table_lines = "\n".join(
        f"{row['method']}: coverage={float(row['coverage']):.16g}, "
        f"conditional_median_cost={float(row['conditional_median_cost']):.16g}"
        for row in table_rows
    )
    content = f"""{dataset_name} completed-method presentation audit

Figure 3 source: {figure3_audit['figure3_csv']}
Figure 3 conditional cost source field: {figure3_audit['conditional_cost_field']}
Figure 3 selected theta: {figure3_audit['selected_theta']:.16g}
Figure 4 source: {figure4_audit['figure4_csv']}
Figure 4 selected K: {figure4_audit['selected_k']}
Figure 4 mode: quantile
q30: {q30:.16g}
Methods: {','.join(methods)}

{table_lines}
best among completed methods coverage: {best_coverage['method']} ({float(best_coverage['coverage']):.16g})
best among completed methods conditional cost: {best_cost['method']} ({float(best_cost['conditional_median_cost']):.16g})

No dense low-cost pAUC was computed from the seven frozen quantile points.
GCFExplainer is pending and is not included in this preliminary three-method comparison.
"""
    (output_dir / "mut_three_method_plot_audit.txt").write_text(
        content, encoding="utf-8"
    )


def _validate_quantile_rows(
    rows: Sequence[Figure4Row],
    *,
    methods: Sequence[str],
    q30: float,
) -> None:
    reference: list[float] | None = None
    required_labels = {"q05", "q10", "q20", "q30", "q50", "q70", "q90"}
    for method in methods:
        selected = _method_rows(rows, method)
        if len(selected) != 7:
            raise ValueError(
                f"Quantile Figure 4 requires seven rows for {method}; actual={len(selected)}"
            )
        labels = {str(row.threshold_name or "") for row in selected}
        if labels != required_labels:
            raise ValueError(
                f"Quantile Figure 4 labels for {method} must be q05..q90; "
                f"actual={sorted(labels)}"
            )
        thresholds = [row.threshold for row in selected]
        if reference is None:
            reference = thresholds
        elif any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=EPS)
            for left, right in zip(reference, thresholds)
        ):
            raise ValueError(f"Quantile threshold grid differs for {method}.")
        q30_rows = [row for row in selected if row.threshold_name == "q30"]
        if len(q30_rows) != 1 or not math.isclose(
            q30_rows[0].threshold, q30, rel_tol=0.0, abs_tol=EPS
        ):
            raise ValueError(f"{method} q30 threshold does not match requested q30.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    expected_methods = _parse_expected_methods(args.expected_methods)
    figure4_mode = str(args.figure4_mode)
    dataset_name = str(args.dataset_name)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure3_csv = _find_figure3_csv(Path(args.figure3_report_dir).expanduser().resolve())
    figure4_csv = Path(args.figure4_csv).expanduser().resolve()
    if not figure4_csv.is_file():
        raise FileNotFoundError(f"Figure 4 CSV not found: {figure4_csv}")
    figure3_rows, figure3_audit = load_figure3_rows(
        figure3_csv,
        q30=float(args.q30),
        expected_methods=expected_methods,
    )
    selected_figure4_k = 10 if figure4_mode == "quantile" else 20
    figure4_rows, figure4_audit = load_figure4_rows(
        figure4_csv,
        expected_methods=expected_methods,
        selected_k=selected_figure4_k,
    )
    _require_prefix(figure3_rows, 20, expected_methods)
    if figure4_mode == "quantile":
        _validate_quantile_rows(
            figure4_rows, methods=expected_methods, q30=float(args.q30)
        )

    selected_prefix_rows = [
        {
            "method": row.method,
            "k": row.k,
            "theta": row.theta,
            "coverage": row.coverage,
            "conditional_median_cost": row.conditional_median_cost,
            "source_csv": str(figure3_csv),
        }
        for row in figure3_rows
    ]
    _write_csv(
        output_dir / "selected_figure3_prefix_data.csv",
        selected_prefix_rows,
        ("method", "k", "theta", "coverage", "conditional_median_cost", "source_csv"),
    )

    selected_threshold_rows = []
    for row in figure4_rows:
        if figure4_mode != "quantile" and not 0.0 <= row.threshold <= 0.10:
            continue
        selected_row = {
            "method": row.method,
            "k": row.k,
            "threshold": row.threshold,
            "coverage": row.coverage,
            "mean": row.mean,
            "lower": row.lower,
            "upper": row.upper,
            "in_low_cost_auc_interval": 0.0 <= row.threshold <= float(args.q30),
            "in_main_display_interval": float(args.figure4_display_min) <= row.threshold <= float(args.q30),
            "in_supplement_display_interval": 0.0 <= row.threshold <= 0.10,
            "source_csv": str(figure4_csv),
        }
        if figure4_mode == "quantile":
            selected_row["threshold_name"] = row.threshold_name
        selected_threshold_rows.append(selected_row)
    threshold_fields = (
        (
            "method", "k", "threshold_name", "threshold", "coverage", "mean", "lower", "upper",
            "in_low_cost_auc_interval", "in_main_display_interval",
            "in_supplement_display_interval", "source_csv",
        )
        if figure4_mode == "quantile"
        else (
            "method", "k", "threshold", "coverage", "mean", "lower", "upper",
            "in_low_cost_auc_interval", "in_main_display_interval",
            "in_supplement_display_interval", "source_csv",
        )
    )
    _write_csv(
        output_dir / "selected_figure4_threshold_data.csv",
        selected_threshold_rows,
        threshold_fields,
    )

    figure3_main_stem = (
        "mut_figure3_main_k1_10_coverage_conditional_cost"
        if dataset_name == "Mutagenicity"
        else "figure3_main_k1_10_coverage_conditional_cost"
    )
    figure3_supplement_stem = (
        "mut_figure3_supplement_k1_20_coverage_conditional_cost"
        if dataset_name == "Mutagenicity"
        else "figure3_supplement_k1_20_coverage_conditional_cost"
    )
    _save_figure3(
        figure3_rows,
        max_k=10,
        q30=float(args.q30),
        path_png=output_dir / f"{figure3_main_stem}.png",
        path_pdf=output_dir / f"{figure3_main_stem}.pdf",
        supplemental=False,
        methods=expected_methods,
        dataset_name=dataset_name,
    )
    _save_figure3(
        figure3_rows,
        max_k=20,
        q30=float(args.q30),
        path_png=output_dir / f"{figure3_supplement_stem}.png",
        path_pdf=output_dir / f"{figure3_supplement_stem}.pdf",
        supplemental=True,
        methods=expected_methods,
        dataset_name=dataset_name,
    )
    if figure4_mode == "quantile":
        thresholds = [row.threshold for row in figure4_rows]
        _save_figure4(
            figure4_rows,
            q20=float(args.q20),
            q30=float(args.q30),
            lower_display=min(thresholds),
            upper_display=max(thresholds),
            title="Mutagenicity K=10 CCRCov at frozen WNode quantiles",
            path_png=output_dir / "mut_figure4_quantile_ccrcov_k10.png",
            path_pdf=output_dir / "mut_figure4_quantile_ccrcov_k10.pdf",
            methods=expected_methods,
            quantile_mode=True,
        )
    else:
        _save_figure4(
            figure4_rows,
            q20=float(args.q20),
            q30=float(args.q30),
            lower_display=float(args.figure4_display_min),
            upper_display=float(args.q30),
            title=f"K=20 low-cost CCRCov (display {float(args.figure4_display_min):.3f} to q30)",
            path_png=output_dir / "figure4_main_low_cost_ccrcov_0_q30.png",
            path_pdf=output_dir / "figure4_main_low_cost_ccrcov_0_q30.pdf",
            methods=expected_methods,
        )
        _save_figure4(
            figure4_rows,
            q20=float(args.q20),
            q30=float(args.q30),
            lower_display=0.0,
            upper_display=0.10,
            title="K=20 CCRCov across the full threshold range",
            path_png=output_dir / "figure4_supplement_full_ccrcov_0_010.png",
            path_pdf=output_dir / "figure4_supplement_full_ccrcov_0_010.pdf",
            methods=expected_methods,
        )
    table_rows = _save_table2(
        figure3_rows,
        q30=float(args.q30),
        output_dir=output_dir,
        methods=expected_methods,
        dataset_name=dataset_name,
    )

    auc_rows: list[dict[str, Any]] = []
    if figure4_mode == "dense":
        for method in expected_methods:
            method_rows = _method_rows(figure4_rows, method)
            auc_rows.append(
                {
                    "method": method,
                    "coverage_at_q30": _interpolated_coverage(method_rows, float(args.q30)),
                    **_normalized_auc(method_rows, q30=float(args.q30)),
                }
            )
        _write_csv(
            output_dir / "figure4_low_cost_auc_0_q30.csv",
            auc_rows,
            (
                "method",
                "auc_min",
                "auc_max",
                "low_cost_normalized_auc",
                "coverage_at_q30",
            ),
        )
        _write_audit(
            output_dir,
            figure3_audit=figure3_audit,
            figure4_audit=figure4_audit,
            table_rows=table_rows,
            auc_rows=auc_rows,
            q20=float(args.q20),
            q30=float(args.q30),
            figure4_display_min=float(args.figure4_display_min),
        )
    else:
        _write_quantile_audit(
            output_dir,
            figure3_audit=figure3_audit,
            figure4_audit=figure4_audit,
            table_rows=table_rows,
            methods=expected_methods,
            dataset_name=dataset_name,
            q30=float(args.q30),
        )
    return {
        "figure3_source": str(figure3_csv),
        "figure4_source": str(figure4_csv),
        "output_dir": str(output_dir),
        "figure3_fields": figure3_audit,
        "figure4_fields": figure4_audit,
        "table2": table_rows,
        "low_cost_auc": auc_rows,
        "expected_methods": list(expected_methods),
        "dataset_name": dataset_name,
        "figure4_mode": figure4_mode,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for the common HPC wrapper interface.")
    parser.add_argument("--set", action="append", default=[], help="Accepted for the common HPC wrapper interface.")
    parser.add_argument(
        "--figure3-report-dir",
        default="outputs/hpc/eval/paper/molclr_node_fgw_q30_main_figure3_table2",
    )
    parser.add_argument(
        "--figure4-csv",
        default="outputs/hpc/eval/paper/molclr_node_fgw_dense_threshold_k20/fgw_dense_k20_figure4_fgw_coverage_vs_threshold.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hpc/eval/paper/molclr_node_fgw_sota_figures",
    )
    parser.add_argument("--q20", type=float, default=0.0229636285221722)
    parser.add_argument("--q30", type=float, default=0.0328363645853374)
    parser.add_argument("--figure4-display-min", type=float, default=0.015)
    parser.add_argument(
        "--expected-methods",
        default=",".join(METHOD_ORDER),
        help="Comma-separated ordered methods expected in both figure inputs.",
    )
    parser.add_argument("--dataset-name", default="AIDS")
    parser.add_argument(
        "--figure4-mode",
        choices=("dense", "quantile"),
        default="dense",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.figure4_mode == "dense" and not 0.0 <= float(args.figure4_display_min) <= float(args.q30):
        raise SystemExit("--figure4-display-min must lie within [0, q30].")
    result = run(args)
    print("[FGW_SOTA_FIGURES_DONE]", flush=True)
    print(f"figure3_source={result['figure3_source']}", flush=True)
    print(f"figure4_source={result['figure4_source']}", flush=True)
    print(f"output_dir={result['output_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
