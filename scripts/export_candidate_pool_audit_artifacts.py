#!/usr/bin/env python3
"""Export compact audit sidecar artifacts from audit_summary.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper parity.")
    parser.add_argument("--audit-json", required=True, help="Input audit_summary.json.")
    parser.add_argument("--out-dir", required=True, help="Directory for sidecar artifacts.")
    parser.add_argument("--topk", type=int, default=30, help="Number of fragment rows to export.")
    return parser


def export_audit_artifacts(audit_json: str | Path, out_dir: str | Path, *, topk: int = 30) -> dict[str, str]:
    audit_path = Path(audit_json).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(audit_path.read_text(encoding="utf-8"))
    overall = dict(summary.get("overall") or {})

    diversity = {
        "audit_json": str(audit_path),
        "unique_raw_fragment_count": overall.get("unique_raw_fragment_count"),
        "unique_core_fragment_count": overall.get("unique_core_fragment_count"),
        "unique_projected_fragment_count": overall.get("unique_projected_fragment_count"),
        "unique_final_fragment_count": overall.get("unique_final_fragment_count"),
        "unique_final_fragment_rate": overall.get("unique_final_fragment_rate"),
        "top1_final_fragment_ratio": overall.get("top1_final_fragment_ratio"),
        "top5_final_fragment_ratio": overall.get("top5_final_fragment_ratio"),
        "top10_final_fragment_ratio": overall.get("top10_final_fragment_ratio"),
        "mean_pairwise_tanimoto": overall.get("mean_pairwise_tanimoto"),
        "median_pairwise_tanimoto": overall.get("median_pairwise_tanimoto"),
        "similarity_fragment_count": overall.get("similarity_fragment_count"),
        "similarity_pairs_evaluated": overall.get("similarity_pairs_evaluated"),
    }
    parent_coverage = {
        "audit_json": str(audit_path),
        "num_total": overall.get("num_total"),
        "num_unique_parent": overall.get("num_unique_parent"),
        "avg_candidates_per_parent": overall.get("avg_candidates_per_parent"),
        "num_by_label": overall.get("num_by_label"),
        "valid_rate": overall.get("valid_rate"),
        "parse_ok_rate": overall.get("parse_ok_rate"),
        "connected_rate": overall.get("connected_rate"),
        "direct_substructure_rate": overall.get("direct_substructure_rate"),
        "final_substructure_rate": overall.get("final_substructure_rate"),
        "projection_used_rate": overall.get("projection_used_rate"),
        "oracle_ok_rate": overall.get("oracle_ok_rate"),
        "cf_flip_rate": overall.get("cf_flip_rate"),
        "cf_drop_mean": overall.get("cf_drop_mean"),
    }

    diversity_path = output_dir / "diversity_summary.json"
    coverage_path = output_dir / "parent_coverage_summary.json"
    topk_path = output_dir / "fragment_frequency_topk.csv"
    _write_json(diversity_path, diversity)
    _write_json(coverage_path, parent_coverage)

    top_fragments = list(overall.get("top_final_fragments") or [])[: max(0, int(topk))]
    with topk_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "fragment", "count", "ratio"])
        writer.writeheader()
        for rank, row in enumerate(top_fragments, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "fragment": row.get("fragment"),
                    "count": row.get("count"),
                    "ratio": row.get("ratio"),
                }
            )

    return {
        "diversity_summary": str(diversity_path),
        "parent_coverage_summary": str(coverage_path),
        "fragment_frequency_topk": str(topk_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    outputs = export_audit_artifacts(args.audit_json, args.out_dir, topk=int(args.topk))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
