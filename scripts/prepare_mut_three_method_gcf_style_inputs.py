#!/usr/bin/env python3
"""Normalize completed Mutagenicity WNode artifacts for GCF-style plots.

The script only reads existing aggregate CSV files. It does not load pair
details, models, embeddings, distances, teachers, or candidate selectors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET = "Mutagenicity"
DISTANCE_LABEL = "MolCLR-Node-Wasserstein"
CF_MODE = "strict_flip"
METHOD_ORDER = ("Ours", "CLEAR", "GlobalGCE")
THETA_STAR = 0.038576244576299636
EXPECTED_NUM_PARENTS = 217
EPS = 1e-12
FROZEN_THRESHOLDS = (
    ("q05", 0.0140881224447634),
    ("q10", 0.0228907585727511),
    ("q20", 0.0323756993249126),
    ("q30", 0.0385762445762996),
    ("q50", 0.0496184268839172),
    ("q70", 0.0640645252675410),
    ("q90", 0.0983224211544865),
)


@dataclass(frozen=True)
class InputSpec:
    method: str
    root: Path
    table_filename: str


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required non-empty CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not headers or not rows:
        raise ValueError(f"CSV must contain a header and rows: {path}")
    return headers, rows


def _write_csv(
    path: Path,
    headers: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column(
    headers: Sequence[str],
    candidates: Sequence[str],
    *,
    label: str,
    required: bool = True,
) -> str | None:
    lookup = {_normalize(header): header for header in headers}
    for candidate in candidates:
        found = lookup.get(_normalize(candidate))
        if found is not None:
            return found
    if required:
        raise ValueError(
            f"Missing {label}; supported={list(candidates)}, available={list(headers)}"
        )
    return None


def _float(value: Any, *, field: str, path: Path) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in {path}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} in {path}: {value!r}")
    return parsed


def _int(value: Any, *, field: str, path: Path) -> int:
    parsed = _float(value, field=field, path=path)
    if not parsed.is_integer():
        raise ValueError(f"Expected integer {field} in {path}: {value!r}")
    return int(parsed)


def _assert_close(left: float, right: float, *, label: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=EPS):
        raise ValueError(f"{label} mismatch: left={left}, right={right}")


def _assert_optional_equal(
    headers: Sequence[str],
    rows: Sequence[dict[str, str]],
    primary: str,
    secondary_candidates: Sequence[str],
    *,
    path: Path,
) -> None:
    secondary = _column(
        headers,
        secondary_candidates,
        label="equivalent metric",
        required=False,
    )
    if secondary is None or secondary == primary:
        return
    for row in rows:
        _assert_close(
            _float(row[primary], field=primary, path=path),
            _float(row[secondary], field=secondary, path=path),
            label=f"{path.name} {primary}/{secondary}",
        )


def _num_parents(
    headers: Sequence[str],
    row: Mapping[str, str],
    *,
    expected: int,
    path: Path,
) -> int:
    field = _column(
        headers,
        ("num_parents", "num_test_parents", "test_parent_count"),
        label="number of parents",
        required=False,
    )
    if field is not None:
        actual = _int(row[field], field=field, path=path)
        if actual != expected:
            raise ValueError(
                f"Parent count mismatch in {path}: expected={expected}, actual={actual}"
            )
    return expected


def _threshold_label(value: float) -> str:
    matches = [
        label
        for label, threshold in FROZEN_THRESHOLDS
        if math.isclose(value, threshold, rel_tol=0.0, abs_tol=EPS)
    ]
    if len(matches) != 1:
        raise ValueError(f"Threshold is not in the frozen q05..q90 grid: {value}")
    return matches[0]


def _normalize_figure3(
    spec: InputSpec,
    *,
    theta_star: float,
    expected_num_parents: int,
) -> list[dict[str, Any]]:
    path = spec.root / "figure3_coverage_vs_k.csv"
    headers, rows = _read_csv(path)
    k_field = _column(headers, ("k",), label="Figure 3 K")
    cost_field = _column(
        headers,
        ("conditional_median_cost",),
        label="untruncated conditional median cost",
    )
    if spec.method in {"Ours", "CLEAR"}:
        coverage_field = _column(
            headers, ("ccrcov_theta_star",), label="theta-star coverage"
        )
        applicable_field = _column(
            headers, ("applicable_rate",), label="applicable rate"
        )
    else:
        coverage_field = _column(
            headers, ("close_cf_coverage",), label="close-CF coverage"
        )
        applicable_field = _column(
            headers,
            ("applicable_coverage",),
            label="applicable coverage",
            required=False,
        )
    theta_field = None
    if spec.method == "CLEAR":
        theta_field = _column(
            headers, ("theta_star",), label="theta star", required=False
        )
    elif spec.method == "GlobalGCE":
        theta_field = _column(headers, ("threshold",), label="threshold")
    num_applicable_field = _column(
        headers,
        ("num_applicable_parents",),
        label="number of applicable parents",
        required=False,
    )

    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        k = _int(row[k_field], field=k_field, path=path)
        if k in seen:
            raise ValueError(f"Duplicate {spec.method} Figure 3 K={k}")
        seen.add(k)
        theta = (
            _float(row[theta_field], field=theta_field, path=path)
            if theta_field is not None
            else theta_star
        )
        _assert_close(theta, theta_star, label=f"{spec.method} Figure 3 theta")
        coverage = _float(row[coverage_field], field=coverage_field, path=path)
        cost = _float(row[cost_field], field=cost_field, path=path)
        if not 0.0 <= coverage <= 1.0 or cost < 0.0:
            raise ValueError(f"Invalid Figure 3 metric range for {spec.method} K={k}")
        num_parents = _num_parents(
            headers, row, expected=expected_num_parents, path=path
        )
        if applicable_field is not None:
            applicable_rate = _float(
                row[applicable_field], field=applicable_field, path=path
            )
        elif num_applicable_field is not None:
            applicable_rate = (
                _int(row[num_applicable_field], field=num_applicable_field, path=path)
                / num_parents
            )
        else:
            raise ValueError(f"No applicable-rate source found in {path}")
        if not 0.0 <= applicable_rate <= 1.0:
            raise ValueError(f"Invalid applicable rate for {spec.method} K={k}")
        output.append(
            {
                "method": spec.method,
                "dataset": DATASET,
                "k": k,
                "theta": theta,
                "coverage": coverage,
                "conditional_median_cost": cost,
                "applicable_rate": applicable_rate,
                "num_parents": num_parents,
                "distance_label": DISTANCE_LABEL,
                "cf_mode": CF_MODE,
                "source_file": str(path),
            }
        )
    output.sort(key=lambda row: int(row["k"]))
    expected_k = list(range(1, 21))
    observed_k = [int(row["k"]) for row in output]
    if observed_k != expected_k:
        raise ValueError(
            f"{spec.method} Figure 3 K must be exactly 1..20; observed={observed_k}"
        )
    return output


def _normalize_figure4(
    spec: InputSpec,
    *,
    figure4_k: int,
    expected_num_parents: int,
) -> tuple[list[dict[str, Any]], int]:
    path = spec.root / "figure4_coverage_vs_threshold.csv"
    headers, all_rows = _read_csv(path)
    k_field = _column(headers, ("k",), label="Figure 4 K")
    selected = [
        row
        for row in all_rows
        if _int(row[k_field], field=k_field, path=path) == figure4_k
    ]
    if not selected:
        raise ValueError(f"{spec.method} Figure 4 has no K={figure4_k} rows")
    threshold_field = _column(headers, ("threshold",), label="threshold")
    if spec.method == "Ours":
        coverage_field = _column(headers, ("coverage",), label="coverage")
        name_field = _column(
            headers, ("quantile_label",), label="quantile label", required=False
        )
        num_covered_field = _column(
            headers, ("num_covered",), label="number covered"
        )
    elif spec.method == "CLEAR":
        coverage_field = _column(headers, ("ccrcov",), label="CCRCov")
        _assert_optional_equal(
            headers, selected, coverage_field, ("coverage",), path=path
        )
        name_field = _column(
            headers, ("threshold_name",), label="threshold name", required=False
        )
        num_covered_field = _column(
            headers,
            ("num_covered", "num_close_cf_covered"),
            label="number covered",
        )
    else:
        coverage_field = _column(
            headers, ("close_cf_coverage",), label="close-CF coverage"
        )
        name_field = _column(
            headers, ("threshold_name",), label="threshold name", required=False
        )
        num_covered_field = _column(
            headers, ("num_close_cf_covered",), label="number close-CF covered"
        )

    normalized: list[dict[str, Any]] = []
    for row in selected:
        threshold = _float(row[threshold_field], field=threshold_field, path=path)
        expected_name = _threshold_label(threshold)
        threshold_name = str(row.get(name_field) or "").strip() if name_field else ""
        if threshold_name:
            if _normalize(threshold_name) != _normalize(expected_name):
                raise ValueError(
                    f"{spec.method} threshold label mismatch: "
                    f"expected={expected_name}, actual={threshold_name}"
                )
        else:
            threshold_name = expected_name
        coverage = _float(row[coverage_field], field=coverage_field, path=path)
        num_covered = _int(
            row[num_covered_field], field=num_covered_field, path=path
        )
        num_parents = _num_parents(
            headers, row, expected=expected_num_parents, path=path
        )
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"Invalid Figure 4 coverage for {spec.method}/{threshold_name}")
        normalized.append(
            {
                "method": spec.method,
                "dataset": DATASET,
                "k": figure4_k,
                "threshold_name": threshold_name,
                "threshold": threshold,
                "coverage": coverage,
                "num_covered": num_covered,
                "num_parents": num_parents,
                "distance_label": DISTANCE_LABEL,
                "cf_mode": CF_MODE,
                "source_file": str(path),
            }
        )

    deduplicated: list[dict[str, Any]] = []
    by_threshold: dict[float, dict[str, Any]] = {}
    duplicates_removed = 0
    for row in normalized:
        threshold = float(row["threshold"])
        existing_key = next(
            (
                value
                for value in by_threshold
                if math.isclose(value, threshold, rel_tol=0.0, abs_tol=EPS)
            ),
            None,
        )
        if existing_key is None:
            by_threshold[threshold] = row
            deduplicated.append(row)
            continue
        existing = by_threshold[existing_key]
        numeric_fields = ("coverage", "num_covered", "num_parents", "k")
        if any(
            not math.isclose(
                float(existing[field]), float(row[field]), rel_tol=0.0, abs_tol=EPS
            )
            for field in numeric_fields
        ) or existing["threshold_name"] != row["threshold_name"]:
            raise ValueError(
                f"Conflicting duplicate Figure 4 row for {spec.method}, "
                f"threshold={threshold}"
            )
        duplicates_removed += 1
    deduplicated.sort(key=lambda row: float(row["threshold"]))
    if len(deduplicated) != len(FROZEN_THRESHOLDS):
        raise ValueError(
            f"{spec.method} Figure 4 must have 7 thresholds after K filtering; "
            f"observed={len(deduplicated)}"
        )
    observed_names = [str(row["threshold_name"]) for row in deduplicated]
    expected_names = [label for label, _value in FROZEN_THRESHOLDS]
    if observed_names != expected_names:
        raise ValueError(
            f"{spec.method} Figure 4 labels mismatch: observed={observed_names}"
        )
    return deduplicated, duplicates_removed


def _normalize_table2(
    spec: InputSpec,
    *,
    theta_star: float,
    figure4_k: int,
    expected_num_parents: int,
) -> dict[str, Any]:
    path = spec.root / spec.table_filename
    headers, rows = _read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"{spec.method} Table 2 must contain exactly one row")
    row = rows[0]
    k_field = _column(headers, ("k",), label="Table 2 K")
    theta_field = _column(headers, ("theta",), label="Table 2 theta")
    if spec.method == "Ours":
        coverage_field = _column(headers, ("coverage",), label="coverage")
    elif spec.method == "CLEAR":
        coverage_field = _column(
            headers, ("ccrcov_theta_star",), label="theta-star CCRCov"
        )
        _assert_optional_equal(
            headers, rows, coverage_field, ("coverage",), path=path
        )
    else:
        coverage_field = _column(
            headers, ("ccrcov", "coverage"), label="CCRCov"
        )
        other = "coverage" if _normalize(coverage_field) == "ccrcov" else "ccrcov"
        _assert_optional_equal(headers, rows, coverage_field, (other,), path=path)
    cost_field = _column(
        headers,
        ("conditional_median_cost",),
        label="untruncated conditional median cost",
    )
    applicable_field = _column(
        headers,
        ("applicable_rate", "applicable_coverage"),
        label="applicable rate",
    )
    drop_field = _column(
        headers,
        ("mean_cf_drop", "avg_cf_drop_among_covered"),
        label="mean CFDrop",
    )
    k = _int(row[k_field], field=k_field, path=path)
    theta = _float(row[theta_field], field=theta_field, path=path)
    if k != figure4_k:
        raise ValueError(f"{spec.method} Table 2 K must be {figure4_k}; actual={k}")
    _assert_close(theta, theta_star, label=f"{spec.method} Table 2 theta")
    coverage = _float(row[coverage_field], field=coverage_field, path=path)
    cost = _float(row[cost_field], field=cost_field, path=path)
    applicable_rate = _float(row[applicable_field], field=applicable_field, path=path)
    mean_cf_drop = _float(row[drop_field], field=drop_field, path=path)
    num_parents = _num_parents(
        headers, row, expected=expected_num_parents, path=path
    )
    if not 0.0 <= coverage <= 1.0 or cost < 0.0:
        raise ValueError(f"Invalid Table 2 values for {spec.method}")
    return {
        "method": spec.method,
        "dataset": DATASET,
        "k": k,
        "theta": theta,
        "coverage": coverage,
        "conditional_median_cost": cost,
        "applicable_rate": applicable_rate,
        "mean_cf_drop": mean_cf_drop,
        "num_parents": num_parents,
        "distance_label": DISTANCE_LABEL,
        "cf_mode": CF_MODE,
        "source_file": str(path),
    }


def normalize_inputs(
    specs: Sequence[InputSpec],
    *,
    theta_star: float,
    figure4_k: int,
    expected_num_parents: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    figure3: list[dict[str, Any]] = []
    figure4: list[dict[str, Any]] = []
    table2: list[dict[str, Any]] = []
    duplicates: dict[str, int] = {}
    for spec in specs:
        figure3.extend(
            _normalize_figure3(
                spec,
                theta_star=theta_star,
                expected_num_parents=expected_num_parents,
            )
        )
        normalized4, removed = _normalize_figure4(
            spec,
            figure4_k=figure4_k,
            expected_num_parents=expected_num_parents,
        )
        figure4.extend(normalized4)
        duplicates[spec.method] = removed
        table2.append(
            _normalize_table2(
                spec,
                theta_star=theta_star,
                figure4_k=figure4_k,
                expected_num_parents=expected_num_parents,
            )
        )
    expected_k = set(range(1, 21))
    k_sets = {
        method: {int(row["k"]) for row in figure3 if row["method"] == method}
        for method in METHOD_ORDER
    }
    if any(values != expected_k for values in k_sets.values()):
        raise ValueError(f"Figure 3 K sets differ across methods: {k_sets}")
    grids = {
        method: [
            float(row["threshold"])
            for row in figure4
            if row["method"] == method
        ]
        for method in METHOD_ORDER
    }
    reference = grids[METHOD_ORDER[0]]
    for method, grid in grids.items():
        if len(grid) != len(reference) or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=EPS)
            for left, right in zip(reference, grid)
        ):
            raise ValueError(f"Figure 4 threshold grid differs for {method}: {grid}")
    table_by_method = {str(row["method"]): row for row in table2}
    for method in METHOD_ORDER:
        figure3_k10 = next(
            row for row in figure3 if row["method"] == method and int(row["k"]) == 10
        )
        _assert_close(
            float(figure3_k10["coverage"]),
            float(table_by_method[method]["coverage"]),
            label=f"{method} Figure 3/Table 2 coverage",
        )
        _assert_close(
            float(figure3_k10["conditional_median_cost"]),
            float(table_by_method[method]["conditional_median_cost"]),
            label=f"{method} Figure 3/Table 2 conditional cost",
        )
    return figure3, figure4, table2, duplicates


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    specs = (
        InputSpec("Ours", Path(args.ours_dir).expanduser().resolve(), "table2_ours_k10.csv"),
        InputSpec("CLEAR", Path(args.clear_dir).expanduser().resolve(), "table2_clear_k10.csv"),
        InputSpec(
            "GlobalGCE",
            Path(args.globalgce_dir).expanduser().resolve(),
            "table2_globalgce_k10.csv",
        ),
    )
    figure3, figure4, table2, duplicates = normalize_inputs(
        specs,
        theta_star=float(args.theta_star),
        figure4_k=int(args.figure4_k),
        expected_num_parents=int(args.expected_num_parents),
    )
    figure3_path = output / "mut_three_method_figure3_coverage_cost_vs_k.csv"
    figure4_path = output / "mut_three_method_figure4_coverage_vs_threshold.csv"
    table2_path = output / "mut_three_method_table2_k10_q30.csv"
    _write_csv(figure3_path, tuple(figure3[0]), figure3)
    _write_csv(figure4_path, tuple(figure4[0]), figure4)
    _write_csv(table2_path, tuple(table2[0]), table2)

    source_paths = {
        spec.method: {
            "figure3": spec.root / "figure3_coverage_vs_k.csv",
            "figure4": spec.root / "figure4_coverage_vs_threshold.csv",
            "table2": spec.root / spec.table_filename,
        }
        for spec in specs
    }
    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "methods": list(METHOD_ORDER),
        "num_parents": int(args.expected_num_parents),
        "theta_star": float(args.theta_star),
        "figure4_k": int(args.figure4_k),
        "frozen_thresholds": {label: value for label, value in FROZEN_THRESHOLDS},
        "distance_label": DISTANCE_LABEL,
        "cf_mode": CF_MODE,
        "source_files": {
            method: {
                key: {"path": str(path), "sha256": _sha256(path)}
                for key, path in paths.items()
            }
            for method, paths in source_paths.items()
        },
        "output_files": {
            path.name: {"sha256": _sha256(path), "rows": len(rows)}
            for path, rows in (
                (figure3_path, figure3),
                (figure4_path, figure4),
                (table2_path, table2),
            )
        },
        "figure4_duplicate_rows_removed": duplicates,
        "selection_performed": False,
        "distance_recomputed": False,
        "teacher_prediction_recomputed": False,
    }
    _write_json(output / "mut_three_method_source_manifest.json", manifest)

    coverage_best = max(table2, key=lambda row: float(row["coverage"]))
    cost_best = min(table2, key=lambda row: float(row["conditional_median_cost"]))
    lines = [
        "Mutagenicity three-method GCF-style normalization audit",
        f"dataset={DATASET}",
        f"methods={','.join(METHOD_ORDER)}",
        f"num_parents={int(args.expected_num_parents)}",
        f"theta_star={float(args.theta_star):.16g}",
        f"figure3_row_count={len(figure3)}",
    ]
    for method in METHOD_ORDER:
        method_k = [int(row["k"]) for row in figure3 if row["method"] == method]
        lines.append(
            f"figure3_{method}_k_min={min(method_k)} k_max={max(method_k)} count={len(method_k)}"
        )
    lines.extend(
        (
            f"figure4_selected_k={int(args.figure4_k)}",
            "figure4_threshold_grid="
            + ",".join(f"{label}:{value:.16g}" for label, value in FROZEN_THRESHOLDS),
            "figure4_duplicate_rows_removed="
            + json.dumps(duplicates, sort_keys=True),
        )
    )
    for row in table2:
        lines.append(
            f"table2_{row['method']}: coverage={row['coverage']:.16g} "
            f"conditional_median_cost={row['conditional_median_cost']:.16g} "
            f"applicable_rate={row['applicable_rate']:.16g}"
        )
    for method, paths in source_paths.items():
        for key, path in paths.items():
            lines.append(f"source_{method}_{key}={path}")
    lines.extend(
        (
            f"best among completed three methods coverage={coverage_best['method']}:{coverage_best['coverage']:.16g}",
            f"best among completed three methods conditional cost={cost_best['method']}:{cost_best['conditional_median_cost']:.16g}",
            "GCFExplainer=pending_not_included",
            "[MUT_THREE_METHOD_NORMALIZATION_PASS]",
        )
    )
    audit_path = output / "mut_three_method_normalization_audit.txt"
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[MUT_THREE_METHOD_NORMALIZATION_PASS]", flush=True)
    return {"output_dir": str(output), "manifest": manifest, "audit": str(audit_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ours-dir",
        default="outputs/hpc/mutagenicity/final_eval/wnode_frozen_a2_test_p217_k20_v3",
    )
    parser.add_argument(
        "--clear-dir",
        default="outputs/hpc/mutagenicity/final_eval/clear_parent_frequency_top20_plot_artifacts_p217_v1",
    )
    parser.add_argument(
        "--globalgce-dir",
        default="outputs/hpc/mutagenicity/final/globalgce_wnode_frequency_top20_test_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hpc/eval/paper/mut_three_method_gcf_style_inputs_v1",
    )
    parser.add_argument("--theta-star", type=float, default=THETA_STAR)
    parser.add_argument("--figure4-k", type=int, default=10)
    parser.add_argument("--expected-num-parents", type=int, default=EXPECTED_NUM_PARENTS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if int(args.figure4_k) != 10:
        raise SystemExit("--figure4-k must be 10 for the frozen Table 2 budget.")
    if int(args.expected_num_parents) != EXPECTED_NUM_PARENTS:
        raise SystemExit("--expected-num-parents must be 217.")
    _assert_close(float(args.theta_star), THETA_STAR, label="CLI theta_star")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
