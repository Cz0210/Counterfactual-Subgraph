#!/usr/bin/env python3
"""Summarize embedding-MMR selector sweeps for ours vs GT-fullgraph motif pools."""

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


DEFAULT_OURS_ROOT = Path("outputs/hpc/selectors/param_sweep_ours_embedding_cov20")
DEFAULT_GT_ROOT = Path("outputs/hpc/selectors/param_sweep_gt_fullgraph_embedding_cov20")
DEFAULT_OUT_DIR = Path("outputs/hpc/selectors/embedding_selector_sweep_comparison_label1")
GAMMA_PATTERN = re.compile(r"gamma_(?P<gamma>\d+(?:p\d+)?)$")


ROW_COLUMNS = [
    "method",
    "seed",
    "gamma",
    "coverage",
    "cf_drop",
    "flip",
    "emb_mean",
    "emb_max",
    "tanimoto_mean",
    "tanimoto_max",
    "atom_ratio",
    "path",
]
AGG_COLUMNS = [
    "method",
    "gamma",
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
COMPARISON_COLUMNS = [
    "gamma",
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
    "recommendation_pass",
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
        return float(value) if math.isfinite(float(value)) else None
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
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _parse_gamma_from_path(summary_path: Path) -> float:
    for part in reversed(summary_path.parts):
        match = GAMMA_PATTERN.match(part)
        if match:
            return float(match.group("gamma").replace("p", "."))
    metadata_gamma = _as_float(_load_json(summary_path).get("metadata", {}).get("gamma_redundancy"))
    if metadata_gamma is not None:
        return metadata_gamma
    raise ValueError(f"Could not infer gamma from path: {summary_path}")


def _summary_to_row(
    summary_path: Path,
    *,
    method: str,
    seed: str,
) -> dict[str, Any]:
    payload = _load_json(summary_path)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    gamma = _as_float(metadata.get("gamma_redundancy"))
    if gamma is None:
        gamma = _parse_gamma_from_path(summary_path)
    return {
        "method": method,
        "seed": seed,
        "gamma": gamma,
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
    ours_path = Path(ours_root).expanduser()
    gt_path = Path(gt_root).expanduser()
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(ours_path.glob("gamma_*/selector_summary.json")):
        rows.append(
            _summary_to_row(
                summary_path,
                method="ours_merged",
                seed="merged",
            )
        )

    for seed_dir in sorted(gt_path.glob("label1_*/")):
        if not seed_dir.is_dir():
            continue
        seed = seed_dir.name
        for summary_path in sorted(seed_dir.glob("gamma_*/selector_summary.json")):
            rows.append(
                _summary_to_row(
                    summary_path,
                    method="gt_fullgraph_greedy_proxy",
                    seed=seed,
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
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gamma = _as_float(row.get("gamma"))
        if gamma is None:
            continue
        grouped[(str(row["method"]), float(gamma))].append(row)

    aggregate: list[dict[str, Any]] = []
    metrics = [
        ("coverage", "coverage"),
        ("cf_drop", "cf_drop"),
        ("flip", "flip"),
        ("emb_mean", "emb_mean"),
        ("emb_max", "emb_max"),
        ("tanimoto_mean", "tanimoto_mean"),
        ("tanimoto_max", "tanimoto_max"),
        ("atom_ratio", "atom_ratio"),
    ]
    for (method, gamma), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        payload: dict[str, Any] = {"method": method, "gamma": gamma, "n": len(group_rows)}
        for source_key, output_prefix in metrics:
            metric_mean, metric_std = _mean_std([row.get(source_key) for row in group_rows])
            payload[f"{output_prefix}_mean"] = metric_mean
            payload[f"{output_prefix}_std"] = metric_std
        aggregate.append(payload)
    return aggregate


def _by_method_gamma(rows: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    result: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        gamma = _as_float(row.get("gamma"))
        if gamma is None:
            continue
        result[(str(row["method"]), float(gamma))] = row
    return result


def _delta(left: Any, right: Any) -> float | None:
    left_float = _as_float(left)
    right_float = _as_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def build_comparison_rows(aggregate_rows_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _by_method_gamma(aggregate_rows_payload)
    gammas = sorted(
        {
            gamma
            for method, gamma in lookup
            if method in {"ours_merged", "gt_fullgraph_greedy_proxy"}
        }
    )
    rows: list[dict[str, Any]] = []
    for gamma in gammas:
        ours = lookup.get(("ours_merged", gamma))
        gt = lookup.get(("gt_fullgraph_greedy_proxy", gamma))
        if ours is None or gt is None:
            continue
        coverage_delta = _delta(ours.get("coverage_mean"), gt.get("coverage_mean"))
        cf_drop_delta = _delta(ours.get("cf_drop_mean"), gt.get("cf_drop_mean"))
        flip_delta = _delta(ours.get("flip_mean"), gt.get("flip_mean"))
        emb_mean_delta = _delta(ours.get("emb_mean_mean"), gt.get("emb_mean_mean"))
        tanimoto_mean_delta = _delta(ours.get("tanimoto_mean_mean"), gt.get("tanimoto_mean_mean"))
        ours_coverage = _as_float(ours.get("coverage_mean"))
        gt_coverage = _as_float(gt.get("coverage_mean"))
        ours_flip = _as_float(ours.get("flip_mean"))
        gt_flip = _as_float(gt.get("flip_mean"))
        ours_cf_drop = _as_float(ours.get("cf_drop_mean"))
        gt_cf_drop = _as_float(gt.get("cf_drop_mean"))
        ours_emb_mean = _as_float(ours.get("emb_mean_mean"))
        gt_emb_mean = _as_float(gt.get("emb_mean_mean"))
        cf_drop_ok = True
        if ours_cf_drop is not None and gt_cf_drop is not None:
            cf_drop_ok = ours_cf_drop >= gt_cf_drop - 0.02
        recommendation_pass = bool(
            ours_coverage is not None
            and gt_coverage is not None
            and ours_flip is not None
            and gt_flip is not None
            and ours_emb_mean is not None
            and gt_emb_mean is not None
            and ours_coverage >= gt_coverage - 0.01
            and ours_flip >= gt_flip
            and cf_drop_ok
            and ours_emb_mean < gt_emb_mean
        )
        rows.append(
            {
                "gamma": gamma,
                "ours_coverage": ours_coverage,
                "gt_coverage_mean": gt_coverage,
                "coverage_delta": coverage_delta,
                "ours_cf_drop": ours_cf_drop,
                "gt_cf_drop_mean": gt_cf_drop,
                "cf_drop_delta": cf_drop_delta,
                "ours_flip": ours_flip,
                "gt_flip_mean": gt_flip,
                "flip_delta": flip_delta,
                "ours_emb_mean": ours_emb_mean,
                "gt_emb_mean_mean": gt_emb_mean,
                "emb_mean_delta": emb_mean_delta,
                "ours_tanimoto_mean": _as_float(ours.get("tanimoto_mean_mean")),
                "gt_tanimoto_mean_mean": _as_float(gt.get("tanimoto_mean_mean")),
                "tanimoto_mean_delta": tanimoto_mean_delta,
                "recommendation_pass": recommendation_pass,
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fieldnames})


def render_report(
    *,
    rows: list[dict[str, Any]],
    aggregate_rows_payload: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    pass_rows = [row for row in comparison_rows if row.get("recommendation_pass")]
    lines = [
        "Embedding Selector Sweep Comparison",
        "",
        f"raw_result_count: {len(rows)}",
        f"aggregate_row_count: {len(aggregate_rows_payload)}",
        f"comparison_gamma_count: {len(comparison_rows)}",
        "",
        "Recommendation:",
    ]
    if pass_rows:
        gamma_list = ", ".join(_fmt(row["gamma"]) for row in pass_rows)
        lines.append("PASS: ours improves embedding redundancy without reducing coverage/flip.")
        lines.append(f"passing_ours_gamma: {gamma_list}")
    else:
        lines.append("NO_PASS: no gamma satisfied the coverage/flip/cf_drop/embedding-redundancy rule.")
    lines.extend(["", "Per-gamma deltas:"])
    if not comparison_rows:
        lines.append("- none")
    for row in comparison_rows:
        lines.append(
            "- gamma={gamma} coverage_delta={coverage_delta} cf_drop_delta={cf_drop_delta} "
            "flip_delta={flip_delta} emb_mean_delta={emb_mean_delta} "
            "tanimoto_mean_delta={tanimoto_mean_delta} pass={recommendation_pass}".format(
                gamma=_fmt(row.get("gamma")),
                coverage_delta=_fmt(row.get("coverage_delta")),
                cf_drop_delta=_fmt(row.get("cf_drop_delta")),
                flip_delta=_fmt(row.get("flip_delta")),
                emb_mean_delta=_fmt(row.get("emb_mean_delta")),
                tanimoto_mean_delta=_fmt(row.get("tanimoto_mean_delta")),
                recommendation_pass=row.get("recommendation_pass"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args.ours_root, args.gt_root)
    aggregate_rows_payload = aggregate_rows(rows)
    comparison_rows = build_comparison_rows(aggregate_rows_payload)
    output_paths = {
        "summary_table_tsv": str(out_dir / "summary_table.tsv"),
        "summary_by_method_gamma_tsv": str(out_dir / "summary_by_method_gamma.tsv"),
        "comparison_summary_json": str(out_dir / "comparison_summary.json"),
        "comparison_report_txt": str(out_dir / "comparison_report.txt"),
    }

    write_tsv(out_dir / "summary_table.tsv", rows, ROW_COLUMNS)
    aggregate_fieldnames = AGG_COLUMNS + COMPARISON_COLUMNS
    combined_rows = list(aggregate_rows_payload) + comparison_rows
    write_tsv(out_dir / "summary_by_method_gamma.tsv", combined_rows, aggregate_fieldnames)
    report_text = render_report(
        rows=rows,
        aggregate_rows_payload=aggregate_rows_payload,
        comparison_rows=comparison_rows,
    )
    summary = {
        "inputs": {
            "ours_root": str(Path(args.ours_root).expanduser()),
            "gt_root": str(Path(args.gt_root).expanduser()),
        },
        "raw_rows": rows,
        "method_gamma_summaries": aggregate_rows_payload,
        "comparison_rows": comparison_rows,
        "recommended_ours_gammas": [
            row["gamma"] for row in comparison_rows if row.get("recommendation_pass")
        ],
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
