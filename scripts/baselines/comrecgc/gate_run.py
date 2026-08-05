#!/usr/bin/env python3
"""Gate a COMRECGC smoke/full project chain without performance thresholds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import write_json  # noqa: E402


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aids", "mutagenicity"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--base-root", required=True)
    args = parser.parse_args()
    root = Path(args.base_root).expanduser().resolve()
    checks = {}
    generation = load(root / "generation/run_manifest.json")
    recourse = load(root / "common_recourse/run_manifest.json")
    export = load(root / "export/run_manifest.json")
    evaluation = load(root / "eval/comrecgc_eval_manifest.json")
    checks["generation_complete"] = generation.get("run_complete") is True
    checks["candidate_count_positive"] = int(generation.get("counterfactual_candidate_count", 0)) > 0
    checks["recourse_complete"] = recourse.get("run_complete") is True
    checks["cluster_count_positive"] = int(recourse.get("common_recourse_count", 0)) > 0
    checks["serialization_reloadable"] = (root / "common_recourse/representative_counterfactuals.pt").stat().st_size > 0
    checks["rf_bridge_called"] = int(export.get("rf_scored_count", 0)) > 0
    checks["wnode_called"] = int(evaluation.get("pair_count", 0)) > 0
    checks["complete_cartesian"] = evaluation.get("complete_cartesian") is True
    checks["no_selection_in_eval"] = evaluation.get("selection_performed_in_eval") is False
    checks["no_calibration_test_generation"] = generation.get("calibration_loaded") is False and generation.get("test_loaded") is False
    checks["finite_counts"] = all(
        math.isfinite(float(value))
        for value in (
            generation.get("counterfactual_candidate_count", 0),
            recourse.get("common_recourse_count", 0),
            evaluation.get("pair_count", 0),
        )
    )
    if args.mode == "full":
        checks["top20_available"] = int(export.get("selected_count", 0)) == 20
    failed = sorted(name for name, passed in checks.items() if not passed)
    gate = {
        "schema_version": 1,
        "method": "COMRECGC",
        "dataset": args.dataset,
        "mode": args.mode,
        "checks": checks,
        "failed_hard_checks": failed,
        "audit_passed": not failed,
        "run_complete": not failed,
        "next_stage": "full_generation" if args.mode == "smoke" and not failed else None,
    }
    write_json(root / "gate.json", gate)
    marker = "_GATE_PASS.json" if not failed else "_GATE_FAILED.json"
    write_json(root / marker, gate)
    print(json.dumps(gate, sort_keys=True))
    if failed:
        print("[COMRECGC_GATE_FAIL]", flush=True)
        return 3
    print("[COMRECGC_GATE_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
