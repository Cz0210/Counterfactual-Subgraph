#!/usr/bin/env python3
"""Summarize label=1 Base vs SFT vs SFT+PPO ablation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/hpc/ablations/label1_sft_ppo")
METHODS = (
    ("A_base_chemmllm", "base_chemmllm_n4", "base_chemmllm_n4_cov20"),
    ("B_sft_only", "sft_only_n4", "sft_only_n4_cov20"),
    ("C_sft_ppo_stable300", "ppo_stable300_n4", "ppo_stable300_n4_cov20"),
)
SUMMARY_COLUMNS = [
    "method",
    "pool_path",
    "selector_path",
    "num_total",
    "num_unique_parent",
    "valid_rate",
    "parse_ok_rate",
    "connected_rate",
    "direct_substructure_rate",
    "final_substructure_rate",
    "projection_used_rate",
    "oracle_ok_rate",
    "cf_flip_rate",
    "cf_drop_mean",
    "atom_ratio_mean",
    "unique_final_fragment_rate",
    "top5_final_fragment_ratio",
    "mean_pairwise_tanimoto",
    "final_cumulative_coverage",
    "selected_mean_cf_drop",
    "selected_cf_flip_rate",
    "selected_mean_atom_ratio",
    "selected_pairwise_tanimoto_mean",
    "selected_pairwise_tanimoto_max",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _get(payload: dict[str, Any], key: str) -> Any:
    return payload.get(key)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def build_summary_rows(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for method, pool_name, selector_name in METHODS:
        pool_dir = root_path / pool_name
        audit_dir = root_path / "audits" / pool_name
        selector_dir = root_path / "selectors" / selector_name
        audit = _load_json(audit_dir / "audit_summary.json")
        diversity = _load_json(audit_dir / "diversity_summary.json")
        parent_coverage = _load_json(audit_dir / "parent_coverage_summary.json")
        selector = _load_json(selector_dir / "selector_summary.json")
        overall = dict(audit.get("overall") or {})
        row = {
            "method": method,
            "pool_path": str(pool_dir / "candidate_pool.jsonl"),
            "selector_path": str(selector_dir),
            "num_total": _first_present(_get(overall, "num_total"), _get(parent_coverage, "num_total")),
            "num_unique_parent": _first_present(
                _get(overall, "num_unique_parent"),
                _get(parent_coverage, "num_unique_parent"),
            ),
            "valid_rate": _first_present(_get(overall, "valid_rate"), _get(parent_coverage, "valid_rate")),
            "parse_ok_rate": _first_present(
                _get(overall, "parse_ok_rate"),
                _get(parent_coverage, "parse_ok_rate"),
            ),
            "connected_rate": _first_present(
                _get(overall, "connected_rate"),
                _get(parent_coverage, "connected_rate"),
            ),
            "direct_substructure_rate": _first_present(
                _get(overall, "direct_substructure_rate"),
                _get(parent_coverage, "direct_substructure_rate"),
            ),
            "final_substructure_rate": _first_present(
                _get(overall, "final_substructure_rate"),
                _get(parent_coverage, "final_substructure_rate"),
            ),
            "projection_used_rate": _first_present(
                _get(overall, "projection_used_rate"),
                _get(parent_coverage, "projection_used_rate"),
            ),
            "oracle_ok_rate": _first_present(
                _get(overall, "oracle_ok_rate"),
                _get(parent_coverage, "oracle_ok_rate"),
            ),
            "cf_flip_rate": _first_present(
                _get(overall, "cf_flip_rate"),
                _get(parent_coverage, "cf_flip_rate"),
            ),
            "cf_drop_mean": _first_present(
                _get(overall, "cf_drop_mean"),
                _get(parent_coverage, "cf_drop_mean"),
            ),
            "atom_ratio_mean": _get(overall, "atom_ratio_mean"),
            "unique_final_fragment_rate": _first_present(
                _get(overall, "unique_final_fragment_rate"),
                _get(diversity, "unique_final_fragment_rate"),
            ),
            "top5_final_fragment_ratio": _first_present(
                _get(overall, "top5_final_fragment_ratio"),
                _get(diversity, "top5_final_fragment_ratio"),
            ),
            "mean_pairwise_tanimoto": _first_present(
                _get(overall, "mean_pairwise_tanimoto"),
                _get(diversity, "mean_pairwise_tanimoto"),
            ),
            "final_cumulative_coverage": _get(selector, "final_cumulative_coverage"),
            "selected_mean_cf_drop": _get(selector, "selected_mean_cf_drop"),
            "selected_cf_flip_rate": _get(selector, "selected_cf_flip_rate"),
            "selected_mean_atom_ratio": _get(selector, "selected_mean_atom_ratio"),
            "selected_pairwise_tanimoto_mean": _get(selector, "selected_pairwise_tanimoto_mean"),
            "selected_pairwise_tanimoto_max": _get(selector, "selected_pairwise_tanimoto_max"),
        }
        rows.append(row)
    return rows


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    display_columns = [
        "method",
        "num_total",
        "valid_rate",
        "parse_ok_rate",
        "connected_rate",
        "direct_substructure_rate",
        "final_substructure_rate",
        "cf_flip_rate",
        "cf_drop_mean",
        "unique_final_fragment_rate",
        "top5_final_fragment_ratio",
        "mean_pairwise_tanimoto",
        "final_cumulative_coverage",
        "selected_mean_cf_drop",
        "selected_cf_flip_rate",
    ]
    lines = [
        "# Label=1 SFT/PPO Ablation Summary",
        "",
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join("---" for _ in display_columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(col)) for col in display_columns) + " |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- B_sft_only vs A_base_chemmllm isolates the effect of SFT on valid, parseable, connected parent-substructure generation.",
            "- C_sft_ppo_stable300 vs B_sft_only isolates the effect of PPO on counterfactuality and selector-level class coverage.",
            "- High-temp merged pools are intentionally excluded from this main ablation to avoid mixing sampling-diversity effects.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary(rows: list[dict[str, Any]], *, out_csv: str | Path, out_md: str | Path) -> None:
    csv_path = Path(out_csv).expanduser().resolve()
    md_path = Path(out_md).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})
    md_path.write_text(render_markdown(rows), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper parity.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT), help="Ablation root directory.")
    parser.add_argument("--out-csv", default="", help="Output CSV path.")
    parser.add_argument("--out-md", default="", help="Output Markdown path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    out_csv = args.out_csv or str(root_dir / "ablation_summary.csv")
    out_md = args.out_md or str(root_dir / "ablation_summary.md")
    rows = build_summary_rows(root_dir)
    write_summary(rows, out_csv=out_csv, out_md=out_md)
    print(render_markdown(rows))
    print(f"out_csv: {Path(out_csv).expanduser().resolve()}")
    print(f"out_md: {Path(out_md).expanduser().resolve()}")


if __name__ == "__main__":
    main()
