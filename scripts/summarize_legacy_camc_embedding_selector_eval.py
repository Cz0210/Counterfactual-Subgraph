#!/usr/bin/env python3
"""Summarize legacy HIV quick CAMC re-evaluation for embedding selector sets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1")
RUN_DIRS = {
    "old_morgan": "old_morgan_seed13",
    "embedding_conservative": "embedding_conservative_beta20_gamma5_seed13",
    "embedding_lowred": "embedding_lowred_beta10_gamma8_seed13",
}
REFERENCE_OLD_PPT = {
    "support_coverage": 0.9757,
    "camc_flip_coverage": 0.9127,
    "camc_delta_0.5": 0.6812,
    "mean_cf_drop_covered": 0.5816,
    "pairwise_tanimoto_mean": 0.0657,
}
SUMMARY_COLUMNS = [
    "run_name",
    "run_dir",
    "support_coverage",
    "camc_flip_coverage",
    "camc_delta_0.5",
    "mean_cf_drop_covered",
    "pairwise_tanimoto_mean",
    "pairwise_tanimoto_max",
    "motif_atom_count_mean",
    "motif_atom_ratio_mean",
    "support_coverage_delta_vs_old",
    "camc_flip_coverage_delta_vs_old",
    "camc_delta_0.5_delta_vs_old",
    "mean_cf_drop_covered_delta_vs_old",
    "tanimoto_mean_delta_vs_old",
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
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_ours_k20(path: Path) -> dict[str, str]:
    rows = _read_csv_rows(path)
    matches = [
        row
        for row in rows
        if str(row.get("method", "")).strip() == "ours_selected_subgraph"
        and int(float(str(row.get("k", "nan")).strip())) == 20
    ]
    if not matches:
        raise ValueError(f"No method=ours_selected_subgraph, k=20 row found in {path}")
    if len(matches) > 1:
        matches.sort(key=lambda row: str(row.get("action_source", "")))
    return matches[0]


def _delta(value: Any, old_value: Any) -> float | None:
    numeric = _as_float(value)
    old_numeric = _as_float(old_value)
    if numeric is None or old_numeric is None:
        return None
    return numeric - old_numeric


def collect_summary_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_by_run: dict[str, dict[str, str]] = {}
    for run_name, run_dir in RUN_DIRS.items():
        table_path = root / run_dir / "camc_comparison_table.csv"
        if not table_path.exists():
            raise FileNotFoundError(f"Missing CAMC table for {run_name}: {table_path}")
        raw_by_run[run_name] = _extract_ours_k20(table_path)

    old_row = raw_by_run["old_morgan"]
    for run_name, raw in raw_by_run.items():
        row = {
            "run_name": run_name,
            "run_dir": str(root / RUN_DIRS[run_name]),
            "support_coverage": _as_float(raw.get("support_coverage")),
            "camc_flip_coverage": _as_float(raw.get("camc_flip_coverage")),
            "camc_delta_0.5": _as_float(raw.get("camc_delta_0.5")),
            "mean_cf_drop_covered": _as_float(raw.get("mean_cf_drop_covered")),
            "pairwise_tanimoto_mean": _as_float(raw.get("pairwise_tanimoto_mean")),
            "pairwise_tanimoto_max": _as_float(raw.get("pairwise_tanimoto_max")),
            "motif_atom_count_mean": _as_float(raw.get("motif_atom_count_mean")),
            "motif_atom_ratio_mean": _as_float(raw.get("motif_atom_ratio_mean")),
        }
        row.update(
            {
                "support_coverage_delta_vs_old": _delta(
                    row["support_coverage"],
                    old_row.get("support_coverage"),
                ),
                "camc_flip_coverage_delta_vs_old": _delta(
                    row["camc_flip_coverage"],
                    old_row.get("camc_flip_coverage"),
                ),
                "camc_delta_0.5_delta_vs_old": _delta(
                    row["camc_delta_0.5"],
                    old_row.get("camc_delta_0.5"),
                ),
                "mean_cf_drop_covered_delta_vs_old": _delta(
                    row["mean_cf_drop_covered"],
                    old_row.get("mean_cf_drop_covered"),
                ),
                "tanimoto_mean_delta_vs_old": _delta(
                    row["pairwise_tanimoto_mean"],
                    old_row.get("pairwise_tanimoto_mean"),
                ),
            }
        )
        rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in SUMMARY_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    display_columns = [
        "run_name",
        "support_coverage",
        "camc_flip_coverage",
        "camc_delta_0.5",
        "mean_cf_drop_covered",
        "pairwise_tanimoto_mean",
        "support_coverage_delta_vs_old",
        "camc_flip_coverage_delta_vs_old",
        "camc_delta_0.5_delta_vs_old",
    ]
    lines = [
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join(["---"] * len(display_columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in display_columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _close_to_reference(old_row: dict[str, Any], tolerance: float = 0.015) -> tuple[bool, list[str]]:
    checks: list[str] = []
    ok = True
    for key, reference in REFERENCE_OLD_PPT.items():
        value = _as_float(old_row.get(key))
        if value is None:
            ok = False
            checks.append(f"{key}: missing, expected {reference}")
            continue
        diff = value - reference
        passed = abs(diff) <= tolerance
        ok = ok and passed
        checks.append(
            f"{key}: value={value:.6f}, reference={reference:.6f}, diff={diff:.6f}, pass={passed}"
        )
    return ok, checks


def _row_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["run_name"]): row for row in rows}


def render_report(rows: list[dict[str, Any]], root: Path) -> str:
    lookup = _row_by_name(rows)
    old_row = lookup.get("old_morgan", {})
    legacy_ok, legacy_checks = _close_to_reference(old_row)
    lines = [
        "Legacy CAMC Embedding Selector Re-evaluation",
        "",
        f"root: {root}",
        "source_table: camc_comparison_table.csv",
        "filter: method=ours_selected_subgraph, k=20",
        "",
        "Old Morgan reproduction check:",
    ]
    lines.extend(f"- {line}" for line in legacy_checks)
    if legacy_ok:
        lines.append("PASS: old_morgan is close to the old PPT CAMC values; legacy evaluator path is aligned.")
    else:
        lines.append("WARNING: old_morgan is not close to the old PPT CAMC values; inspect paths/data/teacher before interpreting deltas.")

    lines.extend(["", "Embedding selector deltas vs old_morgan:"])
    for run_name in ("embedding_conservative", "embedding_lowred"):
        row = lookup.get(run_name)
        if row is None:
            lines.append(f"- {run_name}: missing")
            continue
        lines.append(
            "- {run}: support_delta={support_delta}, flip_delta={flip_delta}, "
            "camc@0.5_delta={delta_delta}, cf_drop_delta={cf_delta}, "
            "tanimoto_mean_delta={tan_delta}".format(
                run=run_name,
                support_delta=_fmt(row.get("support_coverage_delta_vs_old")),
                flip_delta=_fmt(row.get("camc_flip_coverage_delta_vs_old")),
                delta_delta=_fmt(row.get("camc_delta_0.5_delta_vs_old")),
                cf_delta=_fmt(row.get("mean_cf_drop_covered_delta_vs_old")),
                tan_delta=_fmt(row.get("tanimoto_mean_delta_vs_old")),
            )
        )

    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- If old_morgan reproduces the old PPT and embedding rows remain much lower, embedding selector truly sacrificed CAMC coverage.",
            "- If embedding rows are close to old_morgan under this legacy evaluator, the previous final-table drop mainly came from evaluator protocol/evidence-source differences.",
            "- If old_morgan does not reproduce, treat all deltas as inconclusive until HIV CSV, RF teacher, selected dir, and seed are checked.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = collect_summary_rows(root)

    output_paths = {
        "summary_tsv": str(root / "legacy_camc_embedding_selector_summary.tsv"),
        "summary_md": str(root / "legacy_camc_embedding_selector_summary.md"),
        "report_txt": str(root / "legacy_camc_embedding_selector_report.txt"),
    }
    write_tsv(root / "legacy_camc_embedding_selector_summary.tsv", rows)
    write_markdown(root / "legacy_camc_embedding_selector_summary.md", rows)
    report = render_report(rows, root)
    (root / "legacy_camc_embedding_selector_report.txt").write_text(report, encoding="utf-8")
    (root / "legacy_camc_embedding_selector_summary.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "rows": rows,
                "reference_old_ppt": REFERENCE_OLD_PPT,
                "output_paths": output_paths,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(report)
    print(json.dumps(output_paths, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
