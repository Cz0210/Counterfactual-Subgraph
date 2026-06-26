#!/usr/bin/env python3
"""Run ours/GCF close counterfactual coverage for GED and embedding distances."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_close_counterfactual_coverage import plot_summaries  # noqa: E402
from src.eval.close_counterfactual_coverage import (  # noqa: E402
    SUMMARY_FIELDS,
    evaluate_gcf_counterfactual_graphs,
    evaluate_ours_selected_subgraphs,
)
from src.utils.io import ensure_directory  # noqa: E402

DEFAULT_GED_THRESHOLDS = "0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.20"
DEFAULT_EMBEDDING_THRESHOLDS = "0.02,0.05,0.10,0.15,0.20,0.25,0.30"


def _parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in str(value or "").split(",") if part.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _render_combined_report(rows: list[dict[str, Any]]) -> str:
    columns = [
        "method",
        "distance_type",
        "threshold",
        "close_only_coverage",
        "close_cf_coverage",
        "avg_best_distance",
        "median_best_distance",
        "avg_cf_drop_among_covered",
        "flip_rate_among_covered",
        "match_rate",
        "delete_valid_rate",
    ]
    lines = [
        "# Combined Close Counterfactual Coverage Report",
        "",
        "低成本翻转 coverage@0.20 可以视为 close counterfactual coverage 的一个特例。当候选反事实图定义为 hard deletion 后的 residual graph G\\s，距离函数使用 normalized GED 或 embedding distance，并固定 threshold=0.20，同时要求 teacher prediction flip，则该指标等价于 CloseCFCoverage@0.20。区别是 GCFExplainer 原始定义的候选是完整 counterfactual graph C，而 ours 的候选首先是 selected subgraph s，需要通过 hard deletion 映射为 G\\s。",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row.get(column)) for column in columns) + "|")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional config path kept for HPC wrapper parity.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept for Slurm wrapper parity.",
    )
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--ours-selected-path", required=True)
    parser.add_argument("--gcf-candidates-path", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--ged-thresholds", default=DEFAULT_GED_THRESHOLDS)
    parser.add_argument("--embedding-thresholds", default=DEFAULT_EMBEDDING_THRESHOLDS)
    parser.add_argument("--require-flip-only", action="store_true")
    parser.add_argument("--min-cf-drop", type=float, default=0.0)
    parser.add_argument("--desired-label", type=int, default=None)
    parser.add_argument("--max-parents", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = ensure_directory(Path(args.output_root).expanduser().resolve())
    ged_thresholds = _parse_thresholds(args.ged_thresholds)
    embedding_thresholds = _parse_thresholds(args.embedding_thresholds)
    print(
        f"[CLOSE_CF_CONFIG] output_root={output_root} label={args.label} "
        f"ged_thresholds={ged_thresholds} embedding_thresholds={embedding_thresholds}"
    )

    runs = [
        (
            "Ours-GED",
            output_root / "ours_ged",
            evaluate_ours_selected_subgraphs(
                dataset_csv=args.dataset_csv,
                selected_subgraphs_path=args.ours_selected_path,
                teacher_path=args.teacher_path,
                label=int(args.label),
                distance_type="ged",
                thresholds=ged_thresholds,
                output_dir=output_root / "ours_ged",
                smiles_col=args.smiles_col,
                label_col=args.label_col,
                ged_mode="delete",
                require_flip_only=bool(args.require_flip_only),
                min_cf_drop=float(args.min_cf_drop),
                max_parents=args.max_parents,
            ),
        ),
        (
            "Ours-Embedding",
            output_root / "ours_embedding",
            evaluate_ours_selected_subgraphs(
                dataset_csv=args.dataset_csv,
                selected_subgraphs_path=args.ours_selected_path,
                teacher_path=args.teacher_path,
                label=int(args.label),
                distance_type="embedding",
                thresholds=embedding_thresholds,
                output_dir=output_root / "ours_embedding",
                smiles_col=args.smiles_col,
                label_col=args.label_col,
                ged_mode="delete",
                require_flip_only=bool(args.require_flip_only),
                min_cf_drop=float(args.min_cf_drop),
                max_parents=args.max_parents,
            ),
        ),
        (
            "GCF-GED",
            output_root / "gcf_ged",
            evaluate_gcf_counterfactual_graphs(
                dataset_csv=args.dataset_csv,
                gcf_candidates_path=args.gcf_candidates_path,
                teacher_path=args.teacher_path,
                label=int(args.label),
                distance_type="ged",
                thresholds=ged_thresholds,
                output_dir=output_root / "gcf_ged",
                smiles_col=args.smiles_col,
                label_col=args.label_col,
                desired_label=args.desired_label,
                require_flip_only=bool(args.require_flip_only),
                min_cf_drop=float(args.min_cf_drop),
                max_parents=args.max_parents,
                ged_mode="networkx",
            ),
        ),
        (
            "GCF-Embedding",
            output_root / "gcf_embedding",
            evaluate_gcf_counterfactual_graphs(
                dataset_csv=args.dataset_csv,
                gcf_candidates_path=args.gcf_candidates_path,
                teacher_path=args.teacher_path,
                label=int(args.label),
                distance_type="embedding",
                thresholds=embedding_thresholds,
                output_dir=output_root / "gcf_embedding",
                smiles_col=args.smiles_col,
                label_col=args.label_col,
                desired_label=args.desired_label,
                require_flip_only=bool(args.require_flip_only),
                min_cf_drop=float(args.min_cf_drop),
                max_parents=args.max_parents,
                ged_mode="networkx",
            ),
        ),
    ]

    combined_dir = ensure_directory(output_root / "combined")
    summary_paths: list[str] = []
    plot_labels: list[str] = []
    combined_rows: list[dict[str, Any]] = []
    for label, _, result in runs:
        summary_path = Path(result["outputs"]["threshold_summary_csv"])
        summary_paths.append(str(summary_path))
        plot_labels.append(label)
        for row in _read_csv(summary_path):
            payload = dict(row)
            payload["method"] = label
            payload["source_summary_csv"] = str(summary_path)
            combined_rows.append(payload)

    combined_csv = combined_dir / "combined_threshold_summary.csv"
    fieldnames = list(SUMMARY_FIELDS)
    if "source_summary_csv" not in fieldnames:
        fieldnames.append("source_summary_csv")
    _write_csv(combined_csv, combined_rows, fieldnames)
    report_path = combined_dir / "combined_report.md"
    report_path.write_text(_render_combined_report(combined_rows), encoding="utf-8")

    plot_summaries(
        summary_csvs=summary_paths,
        labels=plot_labels,
        output_dir=combined_dir,
        title_prefix=f"Label {args.label}",
        key_thresholds_ged=[0.10, 0.20],
        key_thresholds_embedding=[0.10, 0.20, 0.30],
    )
    print(f"combined_threshold_summary_csv: {combined_csv}")
    print(f"combined_report_md: {report_path}")
    print(f"combined_figures_dir: {combined_dir / 'figures'}")


if __name__ == "__main__":
    main()
