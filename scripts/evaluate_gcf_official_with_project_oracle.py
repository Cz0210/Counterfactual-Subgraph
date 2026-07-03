#!/usr/bin/env python3
"""Diagnostic RF-oracle evaluation for SMILES-converted official GCF graphs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--teacher-path", default="outputs/hpc/oracle/aids_rf_model.pkl")
    parser.add_argument("--out-dir", default="outputs/hpc/gcfexplainer_official/rf_oracle_eval")
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--low-conversion-threshold", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidates_csv = Path(args.candidates_csv).expanduser().resolve()
    rows = _read_csv(candidates_csv)
    scorer = TeacherSemanticScorer(args.teacher_path)
    details: list[dict[str, Any]] = []
    valid_rows = [row for row in rows if _as_bool(row.get("convert_ok")) and _as_bool(row.get("sanitize_ok")) and row.get("smiles")]
    for row in valid_rows:
        smiles = str(row.get("smiles") or "")
        result = scorer.score_smiles(smiles, label=int(args.target_label))
        pred = result.get("teacher_label")
        details.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "graph_hash": row.get("graph_hash", ""),
                "smiles": smiles,
                "teacher_ok": bool(result.get("teacher_result_ok")),
                "teacher_pred": pred,
                "teacher_prob_target": result.get("teacher_prob"),
                "strict_flip": bool(pred is not None and int(pred) != int(args.target_label)),
                "teacher_reason": result.get("teacher_reason"),
                "CF_MODE": "strict_flip",
                "GCF_MODE": "official_native",
                "TEACHER_TYPE": "project_rf_oracle_diagnostic",
            }
        )

    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    details_csv = output / "rf_oracle_eval_details.csv"
    _write_csv(
        details_csv,
        details,
        [
            "candidate_id",
            "graph_hash",
            "smiles",
            "teacher_ok",
            "teacher_pred",
            "teacher_prob_target",
            "strict_flip",
            "teacher_reason",
            "CF_MODE",
            "GCF_MODE",
            "TEACHER_TYPE",
        ],
    )
    total = len(rows)
    conversion_rate = len(valid_rows) / total if total else 0.0
    teacher_ok = sum(1 for row in details if row["teacher_ok"])
    strict_flips = sum(1 for row in details if row["strict_flip"])
    reliable = conversion_rate >= float(args.low_conversion_threshold)
    summary = {
        "method": "GCFExplainerOfficial",
        "diagnostic_only": True,
        "oracle_eval_reliable": bool(reliable),
        "reason": "" if reliable else "low_smiles_conversion_rate",
        "CF_MODE": "strict_flip",
        "GCF_MODE": "official_native",
        "TEACHER_TYPE": "project_rf_oracle_diagnostic",
        "teacher_path": str(Path(args.teacher_path).expanduser()),
        "candidates_csv": str(candidates_csv),
        "num_candidates_total": total,
        "num_smiles_valid": len(valid_rows),
        "smiles_conversion_rate": conversion_rate,
        "teacher_ok_count": teacher_ok,
        "teacher_ok_rate": teacher_ok / len(details) if details else 0.0,
        "strict_flip_count": strict_flips,
        "strict_flip_rate": strict_flips / teacher_ok if teacher_ok else 0.0,
    }
    summary_csv = output / "rf_oracle_eval_summary.csv"
    _write_csv(summary_csv, [summary], list(summary.keys()))
    (output / "rf_oracle_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[GCF_RF_ORACLE_DIAGNOSTIC_DONE]", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

