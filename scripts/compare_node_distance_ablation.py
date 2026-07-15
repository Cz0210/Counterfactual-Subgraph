#!/usr/bin/env python3
"""Compare WNode and Node-FGW runs on Ours-reference quantile positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)


def _parse_specs(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Run must use Method=PATH: {value!r}")
        method, raw_path = value.split("=", 1)
        if not method.strip() or not raw_path.strip() or method.strip() in output:
            raise ValueError(f"Invalid/duplicate run specification: {value!r}")
        output[method.strip()] = Path(raw_path.strip()).expanduser().resolve()
    return output


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _calibration_map(path: str | None) -> dict[float, float]:
    if not path:
        return {}
    payload = _read_json(Path(path).expanduser().resolve())
    rows = payload.get("thresholds") or payload.get("distance_quantiles") or payload.get("rows") or []
    output: dict[float, float] = {}
    if isinstance(rows, dict):
        rows = [{"quantile": key, "threshold": value} for key, value in rows.items()]
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        quantile, threshold = _as_float(row.get("quantile")), _as_float(row.get("threshold"))
        if quantile is not None and threshold is not None:
            output[quantile] = threshold
    return output


def load_run_curve(run_dir: Path, display_method: str, calibration: dict[float, float]) -> dict[str, Any]:
    summary_path = run_dir / "combined" / "combined_threshold_summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    methods = sorted({str(row.get("method") or "") for row in rows if str(row.get("method") or "")})
    if len(methods) > 1:
        matching = [row for row in rows if str(row.get("method")) == display_method]
        if matching:
            rows = matching
        else:
            raise ValueError(f"Run contains multiple methods and none matches {display_method!r}: {run_dir}")
    curve: dict[float, dict[str, float]] = {}
    for row in rows:
        threshold = _as_float(row.get("threshold"))
        quantile = _as_float(row.get("quantile"))
        if quantile is None and threshold is not None and calibration:
            quantile = next((q for q, value in calibration.items() if math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-12)), None)
        coverage = _as_float(row.get("close_cf_coverage"))
        if quantile is None or threshold is None or coverage is None:
            continue
        curve[quantile] = {
            "threshold": threshold,
            "coverage": coverage,
            "avg_best_distance": _as_float(row.get("avg_best_distance")) or float("nan"),
        }
    config = _read_json(run_dir / "run_config.json")
    cache = _read_json(run_dir / "cache_stats.json")
    return {"method": display_method, "run_dir": str(run_dir), "curve": curve, "config": config, "cache": cache}


def average_ranks(values: dict[str, float]) -> dict[str, float]:
    """Descending ranks with average rank assigned to exact ties."""

    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    output: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for method, _value in ordered[cursor:end]:
            output[method] = average
        cursor = end
    return output


def normalized_pauc(curve: dict[float, dict[str, float]], lower: float, upper: float) -> float:
    points = sorted((q, row["coverage"]) for q, row in curve.items() if lower <= q <= upper)
    if not points or points[0][0] > lower or points[-1][0] < upper:
        return float("nan")
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    integral = getattr(np, "trapezoid", np.trapz)(y, x)
    return float(integral / (upper - lower))


def minimum_quantile_for_coverage(curve: dict[float, dict[str, float]], target: float) -> float:
    return next((q for q, row in sorted(curve.items()) if row["coverage"] >= target), float("nan"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_lines(path: Path, rows: list[dict[str, Any]], *, methods: Sequence[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for distance in ("MolCLR-Node-Wasserstein", "MolCLR-Node-FGW"):
        for method in methods:
            subset = sorted(
                (row for row in rows if row["distance"] == distance and row["method"] == method),
                key=lambda row: row["reference_quantile"],
            )
            if subset:
                ax.plot(
                    [row["reference_quantile"] for row in subset],
                    [row["coverage"] for row in subset],
                    marker="o", linewidth=1.5, label=f"{method} | {distance}",
                )
    ax.set_xlabel("Ours-reference quantile")
    ax.set_ylabel("CCRCov")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare_runs(args: argparse.Namespace) -> dict[str, Any]:
    wnode_specs, fgw_specs = _parse_specs(args.wnode_run), _parse_specs(args.fgw_run)
    if set(wnode_specs) != set(fgw_specs) or not wnode_specs:
        raise ValueError("WNode and FGW must provide the same non-empty method set.")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibrations = {
        "MolCLR-Node-Wasserstein": _calibration_map(args.wnode_calibration_json),
        "MolCLR-Node-FGW": _calibration_map(args.fgw_calibration_json),
    }
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for label, specs in (("MolCLR-Node-Wasserstein", wnode_specs), ("MolCLR-Node-FGW", fgw_specs)):
        runs[label] = {method: load_run_curve(path, method, calibrations[label]) for method, path in specs.items()}
    quantiles = tuple(float(value) for value in args.reference_quantiles.split(",") if value.strip())
    long_rows: list[dict[str, Any]] = []
    for distance, method_runs in runs.items():
        for method, run in method_runs.items():
            missing = [q for q in quantiles if q not in run["curve"]]
            if missing:
                raise ValueError(f"{distance}/{method} is missing reference quantiles: {missing}")
            for quantile in quantiles:
                long_rows.append({
                    "distance": distance, "method": method, "reference_quantile": quantile,
                    **run["curve"][quantile], "run_dir": run["run_dir"],
                })
    _write_csv(
        output / "coverage_by_reference_quantile.csv", long_rows,
        ["distance", "method", "reference_quantile", "threshold", "coverage", "avg_best_distance", "run_dir"],
    )

    rank_rows: list[dict[str, Any]] = []
    rank_lookup: dict[tuple[str, float, str], float] = {}
    for distance in runs:
        for quantile in quantiles:
            values = {method: runs[distance][method]["curve"][quantile]["coverage"] for method in runs[distance]}
            for method, rank in average_ranks(values).items():
                rank_lookup[(distance, quantile, method)] = rank
                rank_rows.append({"distance": distance, "reference_quantile": quantile, "method": method, "coverage": values[method], "rank": rank})
    _write_csv(output / "method_rank_by_quantile.csv", rank_rows, ["distance", "reference_quantile", "method", "coverage", "rank"])
    rank_changes = [
        {
            "method": method,
            "reference_quantile": quantile,
            "wnode_rank": rank_lookup[("MolCLR-Node-Wasserstein", quantile, method)],
            "fgw_rank": rank_lookup[("MolCLR-Node-FGW", quantile, method)],
            "rank_delta_wnode_minus_fgw": rank_lookup[("MolCLR-Node-Wasserstein", quantile, method)] - rank_lookup[("MolCLR-Node-FGW", quantile, method)],
        }
        for method in wnode_specs for quantile in quantiles
    ]
    _write_csv(output / "rank_change_summary.csv", rank_changes, list(rank_changes[0]))

    targets = [float(value) for value in args.coverage_targets.split(",") if value.strip()]
    target_rows = [
        {"distance": distance, "method": method, "coverage_target": target, "minimum_reference_quantile": minimum_quantile_for_coverage(run["curve"], target)}
        for distance, method_runs in runs.items() for method, run in method_runs.items() for target in targets
    ]
    _write_csv(output / "coverage_target_reference_position.csv", target_rows, list(target_rows[0]))

    summary_rows: list[dict[str, Any]] = []
    for distance, method_runs in runs.items():
        ours_name = next((name for name in method_runs if name.lower() == "ours"), next(iter(method_runs)))
        ours = method_runs[ours_name]
        cache = ours["cache"]
        summary_rows.append({
            "Distance": distance,
            "Ours CCRCov@q20": ours["curve"][0.20]["coverage"],
            "Ours CCRCov@q50": ours["curve"][0.50]["coverage"],
            "Avg best distance at q50": ours["curve"][0.50]["avg_best_distance"],
            "Runtime seconds": _as_float(cache.get("runtime_seconds")),
            "Pair-cache hits": cache.get("pair_distance_cache_hits"),
            "Pair-cache hit rate": cache.get("pair_distance_cache_hit_rate"),
            "Ours normalized low-q pAUC [0.05,0.30]": normalized_pauc(ours["curve"], 0.05, float(args.low_quantile)),
            "distance_scale_note": "Avg best distance is interpretable only within the same distance line.",
        })
    summary_fields = list(summary_rows[0])
    _write_csv(output / "distance_ablation_summary.csv", summary_rows, summary_fields)
    markdown = ["| " + " | ".join(summary_fields) + " |", "| " + " | ".join("---" for _ in summary_fields) + " |"]
    markdown.extend("| " + " | ".join(str(row.get(field, "")) for field in summary_fields) + " |" for row in summary_rows)
    (output / "distance_ablation_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    methods = list(wnode_specs)
    _plot_lines(output / "method_coverage_vs_reference_quantile.png", long_rows, methods=methods)
    _plot_lines(output / "ours_coverage_vs_reference_quantile.png", long_rows, methods=[next((name for name in methods if name.lower() == "ours"), methods[0])])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    labels = [row["Distance"] for row in summary_rows]
    axes[0].bar(labels, [row["Runtime seconds"] or 0.0 for row in summary_rows])
    axes[0].set_ylabel("Runtime seconds")
    axes[1].bar(labels, [row["Pair-cache hit rate"] or 0.0 for row in summary_rows])
    axes[1].set_ylabel("Pair-cache hit rate")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "runtime_and_cache_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"output_dir": str(output), "num_methods": len(methods), "num_quantiles": len(quantiles)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--wnode-run", action="append", default=[], metavar="Method=PATH")
    parser.add_argument("--fgw-run", action="append", default=[], metavar="Method=PATH")
    parser.add_argument("--wnode-calibration-json", default=None)
    parser.add_argument("--fgw-calibration-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-quantiles", default=",".join(str(value) for value in DEFAULT_QUANTILES))
    parser.add_argument("--low-quantile", type=float, default=0.30)
    parser.add_argument("--coverage-targets", default="0.30,0.50,0.70")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_runs(args)
    print("[NODE_DISTANCE_ABLATION_DONE]")
    print(f"output_dir={result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
