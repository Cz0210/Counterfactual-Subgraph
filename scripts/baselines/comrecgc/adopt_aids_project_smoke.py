#!/usr/bin/env python3
"""Adopt the completed project AIDS/HIV smoke without regenerating candidates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import (  # noqa: E402
    UPSTREAM_COMMIT,
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def adopt(
    *,
    project_root: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    teacher_path: str | Path,
    molclr_checkpoint: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    output = require_empty_output(output_dir)
    files = {
        "generation": source / "generation/run_manifest.json",
        "counterfactuals": source / "generation/counterfactuals.pt",
        "common": source / "common_recourse/run_manifest.json",
        "representatives": source / "common_recourse/representative_counterfactuals.pt",
        "export": source / "export/run_manifest.json",
        "filter_audit": source / "export/candidate_filter_audit.jsonl",
        "eval": source / "eval/comrecgc_eval_manifest.json",
        "pair_details": source / "eval/details/pair_details.csv",
        "gate": source / "gate.json",
        "gate_marker": source / "_GATE_PASS.json",
    }
    for path in files.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    inventory_before = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in files.items()
    }
    generation = _load(files["generation"])
    common = _load(files["common"])
    export = _load(files["export"])
    evaluation = _load(files["eval"])
    gate = _load(files["gate"])
    expected_generation = {
        "dataset": "aids",
        "route": "project_adapted",
        "mode": "smoke",
        "parent_limit": 64,
        "run_complete": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "calibration_loaded": False,
        "test_loaded": False,
        "official_source_modified": False,
    }
    failures = [
        f"generation:{field}"
        for field, expected in expected_generation.items()
        if generation.get(field) != expected
    ]
    parent_ids = list(generation.get("generation_parent_ids") or [])
    if len(parent_ids) != 64 or len(set(parent_ids)) != 64:
        failures.append("generation_parent_ids")
    if int(generation.get("counterfactual_candidate_count") or 0) <= 0:
        failures.append("generation_candidate_count")
    gnn = dict(generation.get("gnn") or {})
    gnn_path = Path(str(gnn.get("checkpoint_path") or ""))
    if not gnn_path.is_file() or sha256_file(gnn_path) != gnn.get("checkpoint_sha256"):
        failures.append("gnn_checkpoint")
    if common.get("run_complete") is not True or common.get("dataset") != "aids":
        failures.append("common_recourse")
    if common.get("official_greedy_order_preserved") is not True:
        failures.append("official_greedy_order")
    if export.get("decode_ok_count", 0) < 1 or export.get("rf_scored_count", 0) < 1:
        failures.append("chemistry_or_rf_bridge")
    if export.get("selection_performed_in_eval") is not False:
        failures.append("export_selection")
    expected_eval = {
        "dataset": "aids",
        "mode": "smoke",
        "run_complete": True,
        "parent_count": 16,
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "complete_cartesian": True,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
    }
    failures.extend(
        f"eval:{field}"
        for field, expected in expected_eval.items()
        if evaluation.get(field) != expected
    )
    if gate.get("audit_passed") is not True or gate.get("run_complete") is not True:
        failures.append("source_gate")
    checks = dict(gate.get("checks") or {})
    if checks.get("rf_bridge_called") is not True or checks.get("wnode_called") is not True:
        failures.append("rf_wnode_bridge")
    teacher = Path(teacher_path).expanduser().resolve()
    molclr = Path(molclr_checkpoint).expanduser().resolve()
    if not teacher.is_file() or sha256_file(teacher) != export.get("teacher_sha256"):
        failures.append("teacher_checkpoint")
    if not molclr.is_file():
        failures.append("molclr_checkpoint")
    pair_rows = _rows(files["pair_details"])
    if len(pair_rows) != int(evaluation.get("pair_count") or -1):
        failures.append("pair_count")
    if failures:
        failure = {
            "stage": "aids_project_smoke_adoption",
            "failed_hard_checks": failures,
            "run_complete": False,
        }
        write_json(output / "failure_summary.json", failure)
        write_json(output / "_RUN_FAILED.json", failure)
        raise ValueError("AIDS project smoke adoption failed: " + ", ".join(failures))
    inventory_after = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in files.items()
    }
    if inventory_after != inventory_before:
        raise ValueError("AIDS project smoke source changed during read-only adoption.")
    result = {
        "schema_version": 1,
        "stage": "aids_project_smoke_gate",
        "status": "ENGINEERING_SMOKE_PASS",
        "audit_passed": True,
        "run_complete": True,
        "adoption_mode": "ADOPT_EXISTING",
        "algorithm_rerun": False,
        "candidate_regeneration": False,
        "source_root": str(source),
        "source_inventory": inventory_before,
        "source_inventory_sha256": stable_json_sha256(inventory_before),
        "source_project_commit": generation.get("project_commit"),
        "project_commit": _git_commit(project),
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": "AIDS/HIV",
        "dataset_fingerprint": generation["dataset_audit"]["dataset_fingerprint"],
        "generation_parent_count": 64,
        "counterfactual_candidate_count": generation["counterfactual_candidate_count"],
        "common_recourse_count": common.get("common_recourse_count"),
        "chemistry_mapping_passed": export.get("decode_ok_count", 0) >= 1,
        "rf_bridge_passed": checks["rf_bridge_called"],
        "wnode_bridge_passed": checks["wnode_called"],
        "scientific_output_empty": int(export.get("selected_count") or 0) == 0,
        "candidate_yield_gate_passed": bool(gate.get("candidate_yield_gate_passed")),
        "selection_performed_in_eval": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "teacher_path": str(teacher),
        "teacher_sha256": sha256_file(teacher),
        "molclr_checkpoint": str(molclr),
        "molclr_checkpoint_sha256": sha256_file(molclr),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "audit.json", result)
    write_json(output / "run_manifest.json", result)
    write_json(output / "_RUN_COMPLETE.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--molclr-checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(adopt(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
