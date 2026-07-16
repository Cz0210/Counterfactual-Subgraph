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
from typing import Any, Iterable, Sequence

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


def _nearest_theta(rows: Sequence[Figure3Row], q30: float) -> float:
    values = sorted({row.theta for row in rows})
    if not values:
        raise ValueError("Figure 3 CSV has no usable rows.")
    return min(values, key=lambda value: abs(value - q30))


def load_figure3_rows(path: Path, *, q30: float) -> tuple[list[Figure3Row], dict[str, Any]]:
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
    missing = set(METHOD_ORDER) - methods
    if missing:
        raise ValueError(f"Figure 3 CSV is missing standardized methods: {sorted(missing)}")
    return sorted(selected, key=lambda row: (METHOD_ORDER.index(row.method), row.k)), {
        "figure3_csv": str(path),
        "source_columns": headers,
        "method_field": method_field,
        "k_field": k_field,
        "theta_field": theta_field,
        "coverage_field": coverage_field,
        "conditional_cost_field": cost_field,
        "selected_theta": selected_theta,
        "theta_delta_from_q30": selected_theta - q30,
        "ignored_unrecognized_methods": sorted(value for value in ignored_methods if value),
    }


def load_figure4_rows(path: Path) -> tuple[list[Figure4Row], dict[str, Any]]:
    headers, raw_rows = _read_csv(path)
    method_field = _field(headers, ("method",), label="method field")
    threshold_field = _field(headers, ("threshold", "theta"), label="threshold field")
    coverage_field = _field(headers, ("coverage", "ccrcov", "close_cf_coverage"), label="coverage field")
    k_field = _optional_field(headers, ("k", "K"))
    mean_field = _optional_field(headers, ("mean", "coverage_mean"))
    lower_field = _optional_field(headers, ("lower", "ci_lower"))
    upper_field = _optional_field(headers, ("upper", "ci_upper"))
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
            )
        )
    if k_field is not None:
        observed_k = sorted({row.k for row in parsed if row.k is not None})
        if 20 not in observed_k:
            raise ValueError(
                "Figure 4 requires the dense K=20 input curve; "
                f"observed K values={observed_k}"
            )
        parsed = [row for row in parsed if row.k == 20]
    methods = {row.method for row in parsed}
    missing = set(METHOD_ORDER) - methods
    if missing:
        raise ValueError(f"Figure 4 CSV is missing standardized methods: {sorted(missing)}")
    per_key: set[tuple[str, float]] = set()
    for row in parsed:
        key = (row.method, row.threshold)
        if key in per_key:
            raise ValueError(f"Duplicate Figure 4 row: {key}")
        per_key.add(key)
    return sorted(parsed, key=lambda row: (METHOD_ORDER.index(row.method), row.threshold)), {
        "figure4_csv": str(path),
        "source_columns": headers,
        "method_field": method_field,
        "k_field": k_field,
        "threshold_field": threshold_field,
        "coverage_field": coverage_field,
        "mean_field": mean_field,
        "lower_field": lower_field,
        "upper_field": upper_field,
        "ignored_unrecognized_methods": sorted(value for value in ignored_methods if value),
        "selected_k": 20 if k_field is not None else None,
    }


def _method_rows(rows: Sequence[Any], method: str) -> list[Any]:
    return [row for row in rows if row.method == method]


def _require_prefix(rows: Sequence[Figure3Row], max_k: int) -> None:
    expected = list(range(1, max_k + 1))
    for method in METHOD_ORDER:
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
) -> None:
    _require_prefix(rows, max_k)
    _matplotlib, plt = _import_matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(7.4, 7.9), sharex=True)
    for method in METHOD_ORDER:
        method_rows = [row for row in _method_rows(rows, method) if row.k <= max_k]
        x = [row.k for row in method_rows]
        axes[0].plot(x, [row.coverage for row in method_rows], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], linewidth=1.9, markersize=4.2, label=method)
        axes[1].plot(x, [row.conditional_median_cost for row in method_rows], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], linewidth=1.9, markersize=4.2, label=method)
    axes[0].set_ylabel("Coverage / CCRCov")
    axes[1].set_ylabel("Conditional median cost\n(MolCLR-Node-FGW)")
    axes[1].set_xlabel("Prefix K")
    axes[0].set_title(f"theta = {q30:.4f}")
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
        "Conditional cost is the unified-evaluator field; it is not labeled as the original GCFExplainer paper-style unconditional cost.",
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
    area = float(np.trapz(y, x))
    return {
        "auc_0_q30": area,
        "normalized_auc_0_q30": area / q30,
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
) -> None:
    _matplotlib, plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(7.4, 4.9))
    for method in METHOD_ORDER:
        method_rows = [row for row in _method_rows(rows, method) if lower_display - EPS <= row.threshold <= upper_display + EPS]
        if not method_rows:
            raise ValueError(f"No Figure 4 data for {method} in [{lower_display}, {upper_display}]")
        x = np.asarray([row.threshold for row in method_rows], dtype=float)
        coverage = np.asarray([row.coverage for row in method_rows], dtype=float)
        axis.plot(x, coverage, color=METHOD_COLORS[method], linewidth=1.9, label=method)
        lower = [row.lower for row in method_rows]
        upper = [row.upper for row in method_rows]
        if all(value is not None for value in lower) and all(value is not None for value in upper):
            axis.fill_between(x, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
    if lower_display <= q20 <= upper_display:
        axis.axvline(q20, color="#666666", linestyle=":", linewidth=1.0, label="q20")
    if lower_display <= q30 <= upper_display:
        axis.axvline(q30, color="#444444", linestyle="--", linewidth=1.0, label="q30")
    axis.set_xlim(lower_display, upper_display)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("MolCLR-Node-FGW threshold")
    axis.set_ylabel("CCRCov")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(path_png, dpi=300, bbox_inches="tight")
    figure.savefig(path_pdf, bbox_inches="tight")
    plt.close(figure)


def _save_table2(rows: Sequence[Figure3Row], *, q30: float, output_dir: Path) -> list[dict[str, Any]]:
    k10 = {row.method: row for row in rows if row.k == 10}
    missing = set(METHOD_ORDER) - set(k10)
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
            "Method": method,
            "Coverage": k10[method].coverage,
            "Conditional median cost": k10[method].conditional_median_cost,
            "K": 10,
            "Theta": q30,
            "coverage_is_best": k10[method].coverage >= max_coverage - EPS,
            "cost_is_best": k10[method].conditional_median_cost <= min_cost + EPS,
        }
        for method in METHOD_ORDER
    ]
    display_fields = ("Method", "Coverage \u2191", "Conditional median cost \u2193")
    display_rows = [
        {
            "Method": row["Method"],
            "Coverage \u2191": row["Coverage"],
            "Conditional median cost \u2193": row["Conditional median cost"],
        }
        for row in table_rows
    ]
    _write_csv(output_dir / "table2_main_k10_q30_compact.csv", display_rows, display_fields)
    markdown = [
        "| Method | Coverage ↑ | Conditional median cost ↓ |",
        "| --- | ---: | ---: |",
    ]
    for row in table_rows:
        coverage = f"{row['Coverage']:.4f}"
        cost = f"{row['Conditional median cost']:.4f}"
        if row["Method"] == "Ours" and row["coverage_is_best"]:
            coverage = f"**{coverage}**"
        if row["Method"] == "Ours" and row["cost_is_best"]:
            cost = f"**{cost}**"
        markdown.append(f"| {row['Method']} | {coverage} | {cost} |")
    markdown.extend(("", "MolCLR-Node-FGW, lambda=0.5, strict-flip evaluation."))
    (output_dir / "table2_main_k10_q30_compact.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    _matplotlib, plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(7.4, 2.6))
    axis.axis("off")
    table = axis.table(
        cellText=[[row["Method"], f"{row['Coverage']:.4f}", f"{row['Conditional median cost']:.4f}"] for row in table_rows],
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
        elif table_rows[row_index - 1]["Method"] == "Ours":
            if col_index == 1 and table_rows[row_index - 1]["coverage_is_best"]:
                cell.set_text_props(weight="bold")
            if col_index == 2 and table_rows[row_index - 1]["cost_is_best"]:
                cell.set_text_props(weight="bold")
    figure.text(0.5, 0.03, "MolCLR-Node-FGW, lambda=0.5, strict-flip evaluation.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(output_dir / "table2_main_k10_q30_compact.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "table2_main_k10_q30_compact.pdf", bbox_inches="tight")
    plt.close(figure)
    return table_rows


def _sota_bool(value: float, values: Iterable[float], *, higher: bool) -> bool:
    values = list(values)
    best = max(values) if higher else min(values)
    return value >= best - EPS if higher else value <= best + EPS


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
    ours_table = next(row for row in table_rows if row["Method"] == "Ours")
    ours_auc = next(row for row in auc_rows if row["Method"] == "Ours")
    coverage_sota = _sota_bool(ours_table["Coverage"], (row["Coverage"] for row in table_rows), higher=True)
    cost_sota = _sota_bool(ours_table["Conditional median cost"], (row["Conditional median cost"] for row in table_rows), higher=False)
    auc_sota = _sota_bool(ours_auc["normalized_auc_0_q30"], (row["normalized_auc_0_q30"] for row in auc_rows), higher=True)
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
low-cost and compact-budget SOTA claim allowed: {claim_allowed}

Permitted claim only when all three statements above are True:
low-cost and compact-budget SOTA

Do not claim all-K and all-threshold SOTA from these checks.
"""
    (output_dir / "sota_presentation_audit.txt").write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure3_csv = _find_figure3_csv(Path(args.figure3_report_dir).expanduser().resolve())
    figure4_csv = Path(args.figure4_csv).expanduser().resolve()
    if not figure4_csv.is_file():
        raise FileNotFoundError(f"Figure 4 CSV not found: {figure4_csv}")
    figure3_rows, figure3_audit = load_figure3_rows(figure3_csv, q30=float(args.q30))
    figure4_rows, figure4_audit = load_figure4_rows(figure4_csv)
    _require_prefix(figure3_rows, 20)

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

    selected_threshold_rows = [
        {
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
        for row in figure4_rows
        if 0.0 <= row.threshold <= 0.10
    ]
    _write_csv(
        output_dir / "selected_figure4_threshold_data.csv",
        selected_threshold_rows,
        (
            "method", "k", "threshold", "coverage", "mean", "lower", "upper",
            "in_low_cost_auc_interval", "in_main_display_interval",
            "in_supplement_display_interval", "source_csv",
        ),
    )

    _save_figure3(
        figure3_rows,
        max_k=10,
        q30=float(args.q30),
        path_png=output_dir / "figure3_main_k1_10_coverage_conditional_cost.png",
        path_pdf=output_dir / "figure3_main_k1_10_coverage_conditional_cost.pdf",
        supplemental=False,
    )
    _save_figure3(
        figure3_rows,
        max_k=20,
        q30=float(args.q30),
        path_png=output_dir / "figure3_supplement_k1_20_coverage_conditional_cost.png",
        path_pdf=output_dir / "figure3_supplement_k1_20_coverage_conditional_cost.pdf",
        supplemental=True,
    )
    _save_figure4(
        figure4_rows,
        q20=float(args.q20),
        q30=float(args.q30),
        lower_display=float(args.figure4_display_min),
        upper_display=float(args.q30),
        title=f"K=20 low-cost CCRCov (display {float(args.figure4_display_min):.3f} to q30)",
        path_png=output_dir / "figure4_main_low_cost_ccrcov_0_q30.png",
        path_pdf=output_dir / "figure4_main_low_cost_ccrcov_0_q30.pdf",
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
    )
    table_rows = _save_table2(figure3_rows, q30=float(args.q30), output_dir=output_dir)

    auc_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = _method_rows(figure4_rows, method)
        auc_rows.append(
            {
                "method": method,
                "q20": float(args.q20),
                "q30": float(args.q30),
                "coverage_at_q20": _interpolated_coverage(method_rows, float(args.q20)),
                "coverage_at_q30": _interpolated_coverage(method_rows, float(args.q30)),
                **_normalized_auc(method_rows, q30=float(args.q30)),
            }
        )
    _write_csv(
        output_dir / "figure4_low_cost_auc_0_q30.csv",
        auc_rows,
        ("method", "q20", "q30", "coverage_at_q20", "coverage_at_q30", "auc_0_q30", "normalized_auc_0_q30"),
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
    return {
        "figure3_source": str(figure3_csv),
        "figure4_source": str(figure4_csv),
        "output_dir": str(output_dir),
        "figure3_fields": figure3_audit,
        "figure4_fields": figure4_audit,
        "table2": table_rows,
        "low_cost_auc": auc_rows,
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= float(args.figure4_display_min) <= float(args.q30):
        raise SystemExit("--figure4-display-min must lie within [0, q30].")
    result = run(args)
    print("[FGW_SOTA_FIGURES_DONE]", flush=True)
    print(f"figure3_source={result['figure3_source']}", flush=True)
    print(f"figure4_source={result['figure4_source']}", flush=True)
    print(f"output_dir={result['output_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
