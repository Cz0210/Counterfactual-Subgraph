#!/usr/bin/env python3
"""Plot close counterfactual coverage threshold sweeps."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_directory  # noqa: E402


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in _split_csv(value)]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read_summary_csv(path: Path, label: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["plot_label"] = label
        row["source_csv"] = str(path)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_by_label(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("plot_label") or row.get("method") or "method"), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda item: _as_float(item.get("threshold")) or 0.0)
    return grouped


def _save_figure(fig: Any, path_without_suffix: Path) -> None:
    for suffix in (".png", ".pdf"):
        out_path = path_without_suffix.with_suffix(suffix)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT_SAVED] {out_path}")


def _plot_line(rows: list[dict[str, Any]], *, metric: str, title: str, ylabel: str, out_path: Path, percent: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, group in _group_by_label(rows).items():
        xs: list[float] = []
        ys: list[float] = []
        for row in group:
            x = _as_float(row.get("threshold"))
            y = _as_float(row.get(metric))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y * 100.0 if percent else y)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("threshold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_path)
    plt.close(fig)


def _nearest_row(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    candidates = [
        (abs((_as_float(row.get("threshold")) or 0.0) - threshold), row)
        for row in rows
        if _as_float(row.get("threshold")) is not None
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if candidates[0][0] > 1e-6:
        return None
    return candidates[0][1]


def _plot_key_bar(
    rows: list[dict[str, Any]],
    *,
    ged_thresholds: list[float],
    embedding_thresholds: list[float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels: list[str] = []
    values: list[float] = []
    for label, group in _group_by_label(rows).items():
        distance_type = str(group[0].get("distance_type") or "")
        key_thresholds = embedding_thresholds if distance_type == "embedding" else ged_thresholds
        for threshold in key_thresholds:
            row = _nearest_row(group, threshold)
            if row is None:
                continue
            labels.append(f"{label}\n@{threshold:g}")
            values.append(float(_as_float(row.get("close_cf_coverage")) or 0.0) * 100.0)
    fig, ax = plt.subplots(figsize=(max(8.5, len(labels) * 0.75), 5.0))
    ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.set_title("Close Counterfactual Coverage at Key Thresholds")
    ax.set_ylabel("close CF coverage (%)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_delete_diagnostics(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ours_rows = [
        row
        for row in rows
        if "ours" in str(row.get("plot_label") or row.get("method") or "").lower()
    ]
    if not ours_rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for metric, linestyle in (("match_rate", "-"), ("delete_valid_rate", "--")):
        xs: list[float] = []
        ys: list[float] = []
        for row in sorted(ours_rows, key=lambda item: _as_float(item.get("threshold")) or 0.0):
            x = _as_float(row.get("threshold"))
            y = _as_float(row.get(metric))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y * 100.0)
        if xs:
            ax.plot(xs, ys, marker="o", linestyle=linestyle, linewidth=2.0, label=metric)
    ax.set_title("Ours Hard-Deletion Diagnostics")
    ax.set_xlabel("threshold")
    ax.set_ylabel("rate (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_table(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    table_rows: list[list[str]] = []
    for label, group in _group_by_label(rows).items():
        last = group[-1]
        table_rows.append(
            [
                label,
                str(last.get("distance_type") or ""),
                str(last.get("threshold") or ""),
                f"{float(_as_float(last.get('close_only_coverage')) or 0.0) * 100.0:.1f}%",
                f"{float(_as_float(last.get('close_cf_coverage')) or 0.0) * 100.0:.1f}%",
                f"{_as_float(last.get('avg_best_distance')) or 0.0:.4f}",
                f"{_as_float(last.get('avg_cf_drop_among_covered')) or 0.0:.4f}",
            ]
        )
    if not table_rows:
        return
    columns = ["method", "distance", "threshold", "close only", "close CF", "avg dist", "avg drop"]
    fig, ax = plt.subplots(figsize=(11.5, max(2.5, 0.45 * len(table_rows) + 1.2)))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("Close Counterfactual Coverage Method Comparison", pad=16)
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_summaries(
    *,
    summary_csvs: list[str | Path],
    labels: list[str],
    output_dir: str | Path,
    title_prefix: str = "",
    key_thresholds_ged: list[float] | None = None,
    key_thresholds_embedding: list[float] | None = None,
) -> dict[str, str]:
    if len(labels) != len(summary_csvs):
        raise ValueError("--labels must contain the same number of entries as --summary-csvs")
    out_root = ensure_directory(Path(output_dir).expanduser().resolve())
    figures_dir = ensure_directory(out_root / "figures")
    all_rows: list[dict[str, Any]] = []
    for path_like, label in zip(summary_csvs, labels):
        all_rows.extend(_read_summary_csv(Path(path_like).expanduser().resolve(), label))
    merged_path = out_root / "merged_summary_for_plot.csv"
    _write_csv(merged_path, all_rows)

    prefix = (str(title_prefix).strip() + " ") if str(title_prefix).strip() else ""
    _plot_line(
        all_rows,
        metric="close_cf_coverage",
        title=f"{prefix}Close Counterfactual Coverage vs Threshold",
        ylabel="close CF coverage (%)",
        out_path=figures_dir / "coverage_vs_threshold_close_cf",
        percent=True,
    )
    _plot_line(
        all_rows,
        metric="close_only_coverage",
        title=f"{prefix}Close-Only Coverage vs Threshold",
        ylabel="close-only coverage (%)",
        out_path=figures_dir / "coverage_vs_threshold_close_only",
        percent=True,
    )
    _plot_line(
        all_rows,
        metric="avg_best_distance",
        title=f"{prefix}Average Best Distance vs Threshold",
        ylabel="avg best distance",
        out_path=figures_dir / "avg_cost_vs_threshold",
        percent=False,
    )
    _plot_line(
        all_rows,
        metric="flip_rate_among_covered",
        title=f"{prefix}Flip Rate Among Covered Parents",
        ylabel="flip rate (%)",
        out_path=figures_dir / "flip_rate_vs_threshold",
        percent=True,
    )
    _plot_line(
        all_rows,
        metric="avg_cf_drop_among_covered",
        title=f"{prefix}Average CF Drop Among Covered Parents",
        ylabel="avg CF drop",
        out_path=figures_dir / "cf_drop_vs_threshold",
        percent=False,
    )
    _plot_key_bar(
        all_rows,
        ged_thresholds=key_thresholds_ged or [0.10, 0.20],
        embedding_thresholds=key_thresholds_embedding or [0.10, 0.20, 0.30],
        out_path=figures_dir / "key_threshold_coverage_bar",
    )
    _plot_delete_diagnostics(all_rows, figures_dir / "delete_diagnostics_ours")
    _plot_table(all_rows, figures_dir / "method_comparison_table")
    return {"merged_summary_csv": str(merged_path), "figures_dir": str(figures_dir)}


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
    parser.add_argument("--summary-csvs", required=True, help="Comma-separated threshold_summary.csv paths.")
    parser.add_argument("--labels", required=True, help='Comma-separated labels, e.g. "Ours-GED,GCF-GED".')
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title-prefix", default="")
    parser.add_argument("--key-thresholds-ged", default="0.10,0.20")
    parser.add_argument("--key-thresholds-embedding", default="0.10,0.20,0.30")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = plot_summaries(
        summary_csvs=_split_csv(args.summary_csvs),
        labels=_split_csv(args.labels),
        output_dir=args.output_dir,
        title_prefix=args.title_prefix,
        key_thresholds_ged=_parse_float_list(args.key_thresholds_ged),
        key_thresholds_embedding=_parse_float_list(args.key_thresholds_embedding),
    )
    print(f"merged_summary_for_plot_csv: {outputs['merged_summary_csv']}")
    print(f"figures_dir: {outputs['figures_dir']}")


if __name__ == "__main__":
    main()
