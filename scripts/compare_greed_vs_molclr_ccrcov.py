#!/usr/bin/env python3
"""Compare GREED-GED and MolCLR CCRCov summary CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_directory  # noqa: E402


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read(path: Path, tag: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["distance_line"] = tag
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(rows: list[dict[str, Any]], metric: str, output: Path, ylabel: str, percent: bool = False) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = f"{row.get('distance_line')}:{row.get('method')}"
        grouped.setdefault(label, []).append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, group in grouped.items():
        group = sorted(group, key=lambda item: _as_float(item.get("threshold")) or 0.0)
        xs = [_as_float(row.get("threshold")) or 0.0 for row in group]
        ys = [(_as_float(row.get(metric)) or 0.0) * (100.0 if percent else 1.0) for row in group]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label)
    ax.set_xlabel("threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(metric)
    ax.grid(True, alpha=0.25)
    ax.legend()
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
        print(f"[PLOT_SAVED] {output.with_suffix(suffix)}")
    plt.close(fig)


def _save_table(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    table_rows = []
    for row in rows:
        if str(row.get("threshold")) not in {"0.1", "0.10", "0.2", "0.20"}:
            continue
        table_rows.append(
            [
                row.get("distance_line"),
                row.get("method"),
                row.get("threshold"),
                f"{(_as_float(row.get('close_cf_coverage')) or 0.0) * 100:.1f}%",
                f"{_as_float(row.get('avg_best_distance')) or 0.0:.4f}",
            ]
        )
    if not table_rows:
        return
    fig, ax = plt.subplots(figsize=(10, max(2.5, len(table_rows) * 0.35 + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=["distance", "method", "theta", "CCRCov", "Cost"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
        print(f"[PLOT_SAVED] {output.with_suffix(suffix)}")
    plt.close(fig)


def _save_key_bar(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        threshold = _as_float(row.get("threshold"))
        if threshold is None or threshold not in {0.05, 0.1, 0.2, 0.3}:
            continue
        labels.append(f"{row.get('distance_line')}\n{row.get('method')}\n@{threshold:g}")
        values.append((_as_float(row.get("close_cf_coverage")) or 0.0) * 100.0)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(9.0, len(labels) * 0.7), 5.2))
    ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.set_title("Key-threshold CCRCov Comparison")
    ax.set_ylabel("close CF coverage (%)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
        print(f"[PLOT_SAVED] {output.with_suffix(suffix)}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--greed-summary", required=True)
    parser.add_argument("--molclr-summary", required=True)
    parser.add_argument("--output-dir", default="outputs/hpc/eval/ccrcov_distance_comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = ensure_directory(Path(args.output_dir).expanduser().resolve())
    rows = _read(Path(args.greed_summary), "GREED-GED") + _read(Path(args.molclr_summary), "MolCLR-Embedding")
    _write(out / "combined_distance_comparison.csv", rows)
    md = ["# GREED-GED vs MolCLR-Embedding CCRCov", "", "| distance | method | theta | close_cf_coverage | cost |", "|---|---|---:|---:|---:|"]
    for row in rows:
        md.append(f"| {row.get('distance_line')} | {row.get('method')} | {row.get('threshold')} | {row.get('close_cf_coverage')} | {row.get('avg_best_distance')} |")
    (out / "combined_distance_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    fig_dir = ensure_directory(out / "figures")
    _save_plot(rows, "close_cf_coverage", fig_dir / "coverage_vs_threshold_greed_vs_molclr", "coverage (%)", percent=True)
    _save_plot(rows, "avg_best_distance", fig_dir / "cost_vs_threshold_greed_vs_molclr", "avg best distance")
    _save_plot(rows, "avg_cf_drop_among_covered", fig_dir / "cfdrop_vs_threshold_greed_vs_molclr", "mean CFDrop")
    _save_plot(rows, "flip_rate_among_covered", fig_dir / "fliprate_vs_threshold_greed_vs_molclr", "flip rate", percent=True)
    _save_key_bar(rows, fig_dir / "key_threshold_bar")
    _save_table(rows, fig_dir / "method_comparison_table")


if __name__ == "__main__":
    main()
