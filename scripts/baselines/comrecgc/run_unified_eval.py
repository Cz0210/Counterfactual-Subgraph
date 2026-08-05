#!/usr/bin/env python3
"""Invoke the existing WNode strict-flip evaluator for frozen COMRECGC candidates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import (  # noqa: E402
    CF_MODE,
    DISTANCE_LINE,
    ContractError,
    sha256_file,
    write_json,
)


def read_frozen_thresholds(path: str | Path) -> list[float]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    values: Any = payload.get("thresholds") if isinstance(payload, dict) else None
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    if not isinstance(values, list) and isinstance(payload, dict):
        merged = payload.get("merged_thresholds")
        if isinstance(merged, list):
            values = [row.get("threshold") for row in merged if isinstance(row, dict)]
    if not isinstance(values, list) or not values:
        raise ContractError(f"Frozen threshold list not found: {source}")
    resolved = [float(value) for value in values]
    if resolved != sorted(set(resolved)):
        raise ContractError("Frozen thresholds must be unique and sorted.")
    return resolved


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _find_pair_details(root: Path) -> Path:
    for candidate in (
        root / "details/pair_details.csv",
        root / "pair_details.csv",
        root / "test_pair_details.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unified evaluator pair details missing under {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aids", "mutagenicity"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--molclr-root", required=True)
    parser.add_argument("--molclr-checkpoint", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidate_manifest_path = Path(args.candidate_manifest).expanduser().resolve()
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidates = _csv_rows(Path(args.candidates_csv).expanduser().resolve())
    expected_candidates = int(candidate_manifest.get("candidate_count", -1))
    if len(candidates) != expected_candidates or not candidates:
        raise ContractError(
            f"Frozen candidate count mismatch: csv={len(candidates)}, manifest={expected_candidates}"
        )
    if args.mode == "full" and len(candidates) != 20:
        raise ContractError("Full unified evaluation requires exactly 20 frozen candidates.")
    if [int(row["rank"]) for row in candidates] != list(range(1, len(candidates) + 1)):
        raise ContractError("Frozen candidate prefix order is not rank 1..K.")
    thresholds = read_frozen_thresholds(args.thresholds_json)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Unified evaluation output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    parent_count = min(16, int(args.expected_parent_count)) if args.mode == "smoke" else int(
        args.expected_parent_count
    )
    argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py"),
        "--dataset-csv",
        str(Path(args.dataset_csv).expanduser().resolve()),
        "--teacher-path",
        str(Path(args.teacher_path).expanduser().resolve()),
        "--molclr-root",
        str(Path(args.molclr_root).expanduser().resolve()),
        "--molclr-checkpoint",
        str(Path(args.molclr_checkpoint).expanduser().resolve()),
        "--label",
        "1",
        "--smiles-col",
        "smiles",
        "--label-col",
        "label",
        "--cf-mode",
        CF_MODE,
        "--output-dir",
        str(output),
        "--max-parents",
        str(parent_count),
        "--max-candidates",
        str(len(candidates)),
        "--wnode-thresholds",
        ",".join(format(value, ".17g") for value in thresholds),
        "--feature-cost",
        "cosine",
        "--node-mass",
        "uniform",
        "--size-penalty-beta",
        "0.0",
        "--device",
        "cuda",
        "--run-ours",
        "0",
        "--run-fullgraph",
        "1",
        "--fullgraph-candidates-path",
        str(Path(args.candidates_csv).expanduser().resolve()),
        "--fullgraph-method-name",
        "COMRECGC",
        "--selection-method",
        "official_comrecgc_greedy_cluster_order_filtered_by_validity",
        "--preselected-topk",
        str(len(candidates)),
        "--require-preselected-topk",
        "1",
        "--skip-redundancy",
        "1",
        "--resume",
        "1" if args.resume else "0",
    ]
    write_json(
        output / "comrecgc_eval_command.json",
        {
            "argv": argv,
            "dataset": args.dataset,
            "mode": args.mode,
            "distance_line": DISTANCE_LINE,
            "cf_mode": CF_MODE,
            "thresholds_source": str(Path(args.thresholds_json).expanduser().resolve()),
            "thresholds_source_sha256": sha256_file(args.thresholds_json),
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
        },
    )
    subprocess.run(argv, cwd=PROJECT_ROOT, check=True, timeout=172800)
    pair_path = _find_pair_details(output)
    details = _csv_rows(pair_path)
    expected_pairs = parent_count * len(candidates)
    if len(details) != expected_pairs:
        raise ContractError(
            f"Unified evaluator pair count mismatch: actual={len(details)}, expected={expected_pairs}"
        )
    summary = {
        "method": "COMRECGC",
        "dataset": args.dataset,
        "mode": args.mode,
        "distance_line": DISTANCE_LINE,
        "cf_mode": CF_MODE,
        "parent_count": parent_count,
        "candidate_count": len(candidates),
        "pair_count": len(details),
        "complete_cartesian": True,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "pair_details_path": str(pair_path),
        "pair_details_sha256": sha256_file(pair_path),
        "run_complete": True,
    }
    write_json(output / "comrecgc_eval_manifest.json", summary)
    write_json(output / "_COMRECGC_EVAL_COMPLETE.json", {"run_complete": True})
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
