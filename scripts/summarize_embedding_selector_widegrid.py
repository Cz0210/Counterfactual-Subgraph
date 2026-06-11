#!/usr/bin/env python3
"""Summarize beta/gamma embedding-MMR selector grids for ours vs GT-fullgraph."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_OURS_ROOT = Path("outputs/hpc/selectors/widegrid_ours_embedding_label1")
DEFAULT_GT_ROOT = Path("outputs/hpc/selectors/widegrid_gt_fullgraph_embedding_label1_relaxed")
DEFAULT_OUT_DIR = Path("outputs/hpc/selectors/embedding_selector_widegrid_comparison_label1_relaxed")
GRID_PATTERN = re.compile(
    r"beta_(?P<beta>\d+(?:p\d+)?)_gamma_(?P<gamma>\d+(?:p\d+)?)$"
)

METRIC_KEYS = (
    "coverage",
    "cf_drop",
    "flip",
    "emb_mean",
    "emb_max",
    "tanimoto_mean",
    "tanimoto_max",
    "atom_ratio",
)
ROW_COLUMNS = [
    "method",
    "seed",
    "beta_coverage",
    "gamma_redundancy",
    *METRIC_KEYS,
    "path",
]
AGG_COLUMNS = [
    "method",
    "beta_coverage",
    "gamma_redundancy",
    "n",
    "coverage_mean",
    "coverage_std",
    "cf_drop_mean",
    "cf_drop_std",
    "flip_mean",
    "flip_std",
    "emb_mean_mean",
    "emb_mean_std",
    "emb_max_mean",
    "emb_max_std",
    "tanimoto_mean_mean",
    "tanimoto_mean_std",
    "tanimoto_max_mean",
    "tanimoto_max_std",
    "atom_ratio_mean",
    "atom_ratio_std",
]
COMPARE_COLUMNS = [
    "beta_coverage",
    "gamma_redundancy",
    "ours_coverage",
    "gt_coverage_mean",
    "coverage_delta",
    "ours_cf_drop",
    "gt_cf_drop_mean",
    "cf_drop_delta",
    "ours_flip",
    "gt_flip_mean",
    "flip_delta",
    "ours_emb_mean",
    "gt_emb_mean_mean",
    "emb_mean_delta",
    "ours_tanimoto_mean",
    "gt_tanimoto_mean_mean",
    "tanimoto_mean_delta",
]
PARETO_COLUMNS = [
    "method",
    "beta_coverage",
    "gamma_redundancy",
    "coverage_mean",
    "cf_drop_mean",
    "flip_mean",
    "emb_mean_mean",
    "emb_max_mean",
    "tanimoto_mean_mean",
    "tanimoto_max_mean",
    "atom_ratio_mean",
    "recommendation_tags",
    "path",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for Slurm wrapper parity.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept for Slurm wrapper parity.",
    )
    parser.add_argument("--ours-root", default=str(DEFAULT_OURS_ROOT))
    parser.add_argument("--gt-root", default=str(DEFAULT_GT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _grid_from_path(summary_path: Path) -> tuple[float | None, float | None]:
    for part in reversed(summary_path.parts):
        match = GRID_PATTERN.match(part)
        if match:
            return (
                float(match.group("beta").replace("p", ".")),
                float(match.group("gamma").replace("p", ".")),
            )
    return None, None


def _summary_to_row(summary_path: Path, *, method: str, seed: str) -> dict[str, Any]:
    payload = _load_json(summary_path)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    beta = _as_float(metadata.get("beta_coverage"))
    gamma = _as_float(metadata.get("gamma_redundancy"))
    path_beta, path_gamma = _grid_from_path(summary_path)
    if beta is None:
        beta = path_beta
    if gamma is None:
        gamma = path_gamma
    return {
        "method": method,
        "seed": seed,
        "beta_coverage": beta,
        "gamma_redundancy": gamma,
        "coverage": _as_float(payload.get("final_cumulative_coverage")),
        "cf_drop": _as_float(payload.get("selected_mean_cf_drop")),
        "flip": _as_float(payload.get("selected_cf_flip_rate")),
        "emb_mean": _as_float(payload.get("selected_pairwise_embedding_cosine_mean")),
        "emb_max": _as_float(payload.get("selected_pairwise_embedding_cosine_max")),
        "tanimoto_mean": _as_float(payload.get("selected_pairwise_tanimoto_mean")),
        "tanimoto_max": _as_float(payload.get("selected_pairwise_tanimoto_max")),
        "atom_ratio": _as_float(payload.get("selected_mean_atom_ratio")),
        "path": str(summary_path),
    }


def collect_rows(ours_root: str | Path, gt_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ours_path = Path(ours_root).expanduser()
    gt_path = Path(gt_root).expanduser()

    for summary_path in sorted(ours_path.glob("beta_*_gamma_*/selector_summary.json")):
        rows.append(_summary_to_row(summary_path, method="ours_merged", seed="merged"))

    for seed_dir in sorted(gt_path.glob("label1_*/")):
        if not seed_dir.is_dir():
            continue
        for summary_path in sorted(seed_dir.glob("beta_*_gamma_*/selector_summary.json")):
            rows.append(
                _summary_to_row(
                    summary_path,
                    method="gt_fullgraph_greedy_proxy",
                    seed=seed_dir.name,
                )
            )
    return rows


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None, None
    if len(numeric) == 1:
        return mean(numeric), 0.0
    return mean(numeric), stdev(numeric)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        beta = _as_float(row.get("beta_coverage"))
        gamma = _as_float(row.get("gamma_redundancy"))
        if beta is None or gamma is None:
            continue
        grouped[(str(row["method"]), beta, gamma)].append(row)

    aggregates: list[dict[str, Any]] = []
    for (method, beta, gamma), group_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        payload: dict[str, Any] = {
            "method": method,
            "beta_coverage": beta,
            "gamma_redundancy": gamma,
            "n": len(group_rows),
        }
        for metric in METRIC_KEYS:
            metric_mean, metric_std = _mean_std([row.get(metric) for row in group_rows])
            payload[f"{metric}_mean"] = metric_mean
            payload[f"{metric}_std"] = metric_std
        payload["path"] = group_rows[0].get("path")
        aggregates.append(payload)
    return aggregates


def _lookup_by_method_grid(
    aggregates: list[dict[str, Any]],
) -> dict[tuple[str, float, float], dict[str, Any]]:
    lookup: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in aggregates:
        beta = _as_float(row.get("beta_coverage"))
        gamma = _as_float(row.get("gamma_redundancy"))
        if beta is not None and gamma is not None:
            lookup[(str(row["method"]), beta, gamma)] = row
    return lookup


def _delta(left: Any, right: Any) -> float | None:
    left_float = _as_float(left)
    right_float = _as_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def build_same_parameter_comparisons(
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = _lookup_by_method_grid(aggregates)
    grids = sorted(
        {
            (beta, gamma)
            for method, beta, gamma in lookup
            if method in {"ours_merged", "gt_fullgraph_greedy_proxy"}
        }
    )
    rows: list[dict[str, Any]] = []
    for beta, gamma in grids:
        ours = lookup.get(("ours_merged", beta, gamma))
        gt = lookup.get(("gt_fullgraph_greedy_proxy", beta, gamma))
        if ours is None or gt is None:
            continue
        rows.append(
            {
                "beta_coverage": beta,
                "gamma_redundancy": gamma,
                "ours_coverage": _as_float(ours.get("coverage_mean")),
                "gt_coverage_mean": _as_float(gt.get("coverage_mean")),
                "coverage_delta": _delta(ours.get("coverage_mean"), gt.get("coverage_mean")),
                "ours_cf_drop": _as_float(ours.get("cf_drop_mean")),
                "gt_cf_drop_mean": _as_float(gt.get("cf_drop_mean")),
                "cf_drop_delta": _delta(ours.get("cf_drop_mean"), gt.get("cf_drop_mean")),
                "ours_flip": _as_float(ours.get("flip_mean")),
                "gt_flip_mean": _as_float(gt.get("flip_mean")),
                "flip_delta": _delta(ours.get("flip_mean"), gt.get("flip_mean")),
                "ours_emb_mean": _as_float(ours.get("emb_mean_mean")),
                "gt_emb_mean_mean": _as_float(gt.get("emb_mean_mean")),
                "emb_mean_delta": _delta(ours.get("emb_mean_mean"), gt.get("emb_mean_mean")),
                "ours_tanimoto_mean": _as_float(ours.get("tanimoto_mean_mean")),
                "gt_tanimoto_mean_mean": _as_float(gt.get("tanimoto_mean_mean")),
                "tanimoto_mean_delta": _delta(
                    ours.get("tanimoto_mean_mean"),
                    gt.get("tanimoto_mean_mean"),
                ),
            }
        )
    return rows


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(value) for row in rows if (value := _as_float(row.get(key))) is not None]


def _max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def _min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def _gt_cf_drop_is_proxy(aggregates: list[dict[str, Any]]) -> bool:
    gt_values = [
        value
        for row in aggregates
        if row.get("method") == "gt_fullgraph_greedy_proxy"
        if (value := _as_float(row.get("cf_drop_mean"))) is not None
    ]
    return bool(gt_values) and max(abs(value) for value in gt_values) <= 1e-12


def build_feasible_ours(
    aggregates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ours_rows = [row for row in aggregates if row.get("method") == "ours_merged"]
    gt_rows = [row for row in aggregates if row.get("method") == "gt_fullgraph_greedy_proxy"]
    thresholds = {
        "best_gt_coverage_mean": _max_or_none(_numeric_values(gt_rows, "coverage_mean")),
        "best_gt_flip_mean": _max_or_none(_numeric_values(gt_rows, "flip_mean")),
        "lowest_gt_emb_mean_mean": _min_or_none(_numeric_values(gt_rows, "emb_mean_mean")),
        "best_gt_cf_drop_mean": _max_or_none(_numeric_values(gt_rows, "cf_drop_mean")),
        "gt_cf_drop_is_proxy": _gt_cf_drop_is_proxy(aggregates),
    }
    feasible: list[dict[str, Any]] = []
    for row in ours_rows:
        coverage = _as_float(row.get("coverage_mean"))
        flip = _as_float(row.get("flip_mean"))
        emb_mean = _as_float(row.get("emb_mean_mean"))
        cf_drop = _as_float(row.get("cf_drop_mean"))
        if coverage is None or flip is None or emb_mean is None:
            continue
        if thresholds["best_gt_coverage_mean"] is not None:
            if coverage < float(thresholds["best_gt_coverage_mean"]) - 0.01:
                continue
        if thresholds["best_gt_flip_mean"] is not None:
            if flip < float(thresholds["best_gt_flip_mean"]):
                continue
        if thresholds["lowest_gt_emb_mean_mean"] is not None:
            if emb_mean >= float(thresholds["lowest_gt_emb_mean_mean"]):
                continue
        if (
            not thresholds["gt_cf_drop_is_proxy"]
            and thresholds["best_gt_cf_drop_mean"] is not None
            and cf_drop is not None
            and cf_drop < float(thresholds["best_gt_cf_drop_mean"]) - 0.02
        ):
            continue
        feasible.append(row)
    return feasible, thresholds


def _domination_value(row: dict[str, Any], key: str, *, maximize: bool) -> float:
    value = _as_float(row.get(key))
    if value is None:
        return -math.inf if maximize else math.inf
    return value


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    checks = [
        ("coverage_mean", True),
        ("flip_mean", True),
        ("cf_drop_mean", True),
        ("emb_mean_mean", False),
    ]
    at_least_as_good = True
    strictly_better = False
    for key, maximize in checks:
        left_value = _domination_value(left, key, maximize=maximize)
        right_value = _domination_value(right, key, maximize=maximize)
        if maximize:
            if left_value < right_value:
                at_least_as_good = False
                break
            if left_value > right_value:
                strictly_better = True
        else:
            if left_value > right_value:
                at_least_as_good = False
                break
            if left_value < right_value:
                strictly_better = True
    return at_least_as_good and strictly_better


def build_pareto_frontier(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ours_rows = [row for row in aggregates if row.get("method") == "ours_merged"]
    flip_one_rows = [
        row for row in ours_rows if (_as_float(row.get("flip_mean")) or 0.0) >= 0.999
    ]
    candidate_rows = flip_one_rows if flip_one_rows else ours_rows
    frontier: list[dict[str, Any]] = []
    for row in candidate_rows:
        if any(other is not row and _dominates(other, row) for other in candidate_rows):
            continue
        frontier.append(row)
    return sorted(
        frontier,
        key=lambda row: (
            -(_as_float(row.get("coverage_mean")) or -math.inf),
            _as_float(row.get("emb_mean_mean")) if _as_float(row.get("emb_mean_mean")) is not None else math.inf,
            -(_as_float(row.get("cf_drop_mean")) or -math.inf),
        ),
    )


def _normalize(value: float | None, low: float | None, high: float | None, *, invert: bool = False) -> float:
    if value is None or low is None or high is None or math.isclose(low, high):
        return 0.0
    score = (value - low) / (high - low)
    score = max(0.0, min(1.0, score))
    return 1.0 - score if invert else score


def build_recommendations(
    feasible: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    candidates = feasible if feasible else frontier
    if not candidates:
        return {
            "conservative_recommended_ours": None,
            "balanced_recommended_ours": None,
            "low_redundancy_recommended_ours": None,
        }

    conservative = sorted(
        candidates,
        key=lambda row: (
            -(_as_float(row.get("coverage_mean")) or -math.inf),
            -(_as_float(row.get("flip_mean")) or -math.inf),
            -(_as_float(row.get("cf_drop_mean")) or -math.inf),
            _as_float(row.get("emb_mean_mean")) if _as_float(row.get("emb_mean_mean")) is not None else math.inf,
        ),
    )[0]
    low_redundancy = sorted(
        candidates,
        key=lambda row: (
            _as_float(row.get("emb_mean_mean")) if _as_float(row.get("emb_mean_mean")) is not None else math.inf,
            -(_as_float(row.get("coverage_mean")) or -math.inf),
            -(_as_float(row.get("cf_drop_mean")) or -math.inf),
        ),
    )[0]

    coverages = _numeric_values(candidates, "coverage_mean")
    cf_drops = _numeric_values(candidates, "cf_drop_mean")
    emb_means = _numeric_values(candidates, "emb_mean_mean")
    flips = _numeric_values(candidates, "flip_mean")
    low_cov, high_cov = _min_or_none(coverages), _max_or_none(coverages)
    low_cf, high_cf = _min_or_none(cf_drops), _max_or_none(cf_drops)
    low_emb, high_emb = _min_or_none(emb_means), _max_or_none(emb_means)
    low_flip, high_flip = _min_or_none(flips), _max_or_none(flips)

    def balanced_score(row: dict[str, Any]) -> float:
        return (
            0.45 * _normalize(_as_float(row.get("coverage_mean")), low_cov, high_cov)
            + 0.25 * _normalize(_as_float(row.get("emb_mean_mean")), low_emb, high_emb, invert=True)
            + 0.20 * _normalize(_as_float(row.get("cf_drop_mean")), low_cf, high_cf)
            + 0.10 * _normalize(_as_float(row.get("flip_mean")), low_flip, high_flip)
        )

    balanced = sorted(
        candidates,
        key=lambda row: (
            -balanced_score(row),
            -(_as_float(row.get("coverage_mean")) or -math.inf),
            _as_float(row.get("emb_mean_mean")) if _as_float(row.get("emb_mean_mean")) is not None else math.inf,
        ),
    )[0]
    return {
        "conservative_recommended_ours": conservative,
        "balanced_recommended_ours": balanced,
        "low_redundancy_recommended_ours": low_redundancy,
    }


def _tag_recommendations(
    frontier: list[dict[str, Any]],
    recommendations: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for row in frontier:
        tags = [
            key
            for key, value in recommendations.items()
            if value is row
            or (
                value is not None
                and value.get("beta_coverage") == row.get("beta_coverage")
                and value.get("gamma_redundancy") == row.get("gamma_redundancy")
            )
        ]
        payload = dict(row)
        payload["recommendation_tags"] = ",".join(tags)
        tagged.append(payload)
    return tagged


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fieldnames})


def render_report(
    *,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    same_parameter_rows: list[dict[str, Any]],
    feasible_ours: list[dict[str, Any]],
    thresholds: dict[str, Any],
    pareto_frontier: list[dict[str, Any]],
    recommendations: dict[str, dict[str, Any] | None],
) -> str:
    lines = [
        "Embedding Selector Widegrid Comparison",
        "",
        f"raw_result_count: {len(rows)}",
        f"aggregate_row_count: {len(aggregates)}",
        f"same_parameter_comparison_count: {len(same_parameter_rows)}",
        f"pareto_candidate_count: {len(pareto_frontier)}",
        f"feasible_ours_count: {len(feasible_ours)}",
        "",
        "GT reference thresholds:",
        f"- best_gt_coverage_mean: {_fmt(thresholds.get('best_gt_coverage_mean'))}",
        f"- best_gt_flip_mean: {_fmt(thresholds.get('best_gt_flip_mean'))}",
        f"- lowest_gt_emb_mean_mean: {_fmt(thresholds.get('lowest_gt_emb_mean_mean'))}",
        f"- best_gt_cf_drop_mean: {_fmt(thresholds.get('best_gt_cf_drop_mean'))}",
        f"- gt_cf_drop_is_proxy: {thresholds.get('gt_cf_drop_is_proxy')}",
    ]
    if thresholds.get("gt_cf_drop_is_proxy"):
        lines.append(
            "- note: GT cf_drop is proxy-filled 0.0 and is not used as a hard feasibility constraint."
        )

    lines.extend(["", "Recommendation:"])
    if feasible_ours:
        lines.append("PASS: at least one ours grid point beats the GT embedding-redundancy target.")
    else:
        lines.append(
            "NO_PASS: no ours grid point strictly beats the lowest GT embedding mean under the coverage/flip constraints."
        )

    for label, row in recommendations.items():
        if row is None:
            lines.append(f"- {label}: none")
            continue
        lines.append(
            "- {label}: beta={beta} gamma={gamma} coverage={coverage} flip={flip} "
            "cf_drop={cf_drop} emb_mean={emb_mean}".format(
                label=label,
                beta=_fmt(row.get("beta_coverage")),
                gamma=_fmt(row.get("gamma_redundancy")),
                coverage=_fmt(row.get("coverage_mean")),
                flip=_fmt(row.get("flip_mean")),
                cf_drop=_fmt(row.get("cf_drop_mean")),
                emb_mean=_fmt(row.get("emb_mean_mean")),
            )
        )

    lines.extend(["", "Pareto frontier:"])
    if not pareto_frontier:
        lines.append("- none")
    for row in pareto_frontier:
        lines.append(
            "- beta={beta} gamma={gamma} coverage={coverage} flip={flip} "
            "cf_drop={cf_drop} emb_mean={emb_mean}".format(
                beta=_fmt(row.get("beta_coverage")),
                gamma=_fmt(row.get("gamma_redundancy")),
                coverage=_fmt(row.get("coverage_mean")),
                flip=_fmt(row.get("flip_mean")),
                cf_drop=_fmt(row.get("cf_drop_mean")),
                emb_mean=_fmt(row.get("emb_mean_mean")),
            )
        )

    lines.extend(["", "Same-parameter deltas:"])
    if not same_parameter_rows:
        lines.append("- none")
    for row in same_parameter_rows:
        lines.append(
            "- beta={beta} gamma={gamma} coverage_delta={coverage_delta} "
            "emb_mean_delta={emb_mean_delta} tanimoto_mean_delta={tanimoto_mean_delta}".format(
                beta=_fmt(row.get("beta_coverage")),
                gamma=_fmt(row.get("gamma_redundancy")),
                coverage_delta=_fmt(row.get("coverage_delta")),
                emb_mean_delta=_fmt(row.get("emb_mean_delta")),
                tanimoto_mean_delta=_fmt(row.get("tanimoto_mean_delta")),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args.ours_root, args.gt_root)
    aggregates = aggregate_rows(rows)
    same_parameter_rows = build_same_parameter_comparisons(aggregates)
    feasible_ours, thresholds = build_feasible_ours(aggregates)
    pareto_frontier = build_pareto_frontier(aggregates)
    recommendations = build_recommendations(feasible_ours, pareto_frontier)
    tagged_frontier = _tag_recommendations(pareto_frontier, recommendations)

    output_paths = {
        "summary_table_tsv": str(out_dir / "summary_table.tsv"),
        "summary_by_method_beta_gamma_tsv": str(out_dir / "summary_by_method_beta_gamma.tsv"),
        "pareto_candidates_tsv": str(out_dir / "pareto_candidates.tsv"),
        "same_parameter_comparison_tsv": str(out_dir / "same_parameter_comparison.tsv"),
        "comparison_summary_json": str(out_dir / "comparison_summary.json"),
        "comparison_report_txt": str(out_dir / "comparison_report.txt"),
    }

    write_tsv(out_dir / "summary_table.tsv", rows, ROW_COLUMNS)
    write_tsv(out_dir / "summary_by_method_beta_gamma.tsv", aggregates, AGG_COLUMNS)
    write_tsv(out_dir / "same_parameter_comparison.tsv", same_parameter_rows, COMPARE_COLUMNS)
    write_tsv(out_dir / "pareto_candidates.tsv", tagged_frontier, PARETO_COLUMNS)

    report_text = render_report(
        rows=rows,
        aggregates=aggregates,
        same_parameter_rows=same_parameter_rows,
        feasible_ours=feasible_ours,
        thresholds=thresholds,
        pareto_frontier=pareto_frontier,
        recommendations=recommendations,
    )
    summary = {
        "inputs": {
            "ours_root": str(Path(args.ours_root).expanduser()),
            "gt_root": str(Path(args.gt_root).expanduser()),
        },
        "raw_rows": rows,
        "method_beta_gamma_summaries": aggregates,
        "same_parameter_comparisons": same_parameter_rows,
        "gt_reference_thresholds": thresholds,
        "feasible_ours": feasible_ours,
        "pareto_frontier": tagged_frontier,
        "recommendations": recommendations,
        "output_paths": output_paths,
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "comparison_report.txt").write_text(report_text, encoding="utf-8")

    print(report_text)
    print(json.dumps(output_paths, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
