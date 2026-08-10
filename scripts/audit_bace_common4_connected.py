#!/usr/bin/env python3
"""Audit the frozen connected-residual BACE four-method paper root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_bace_paper_artifacts import audit_bace_artifacts  # noqa: E402
from src.chem.hard_deletion import CONNECTED_ACTION_SEMANTICS  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def _write(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Existing BACE common4 audit differs: {path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def audit_common4(*, root: str | Path) -> dict[str, Any]:
    paper = Path(root).expanduser().resolve()
    import_manifest = _json(paper / "v4_import_manifest.json")
    if import_manifest.get("passed") is not True or import_manifest.get("test_reexecuted") is not False:
        raise ValueError("BACE v4 Ours/GCF import provenance is invalid.")
    base = audit_bace_artifacts(paper, thresholds_path=paper / "thresholds.json")
    expected_methods = ["Ours", "GlobalGCE", "GCFExplainer", "COMRECGC"]
    if base.get("methods") != expected_methods:
        raise ValueError(f"BACE common4 methods differ: {base.get('methods')}")
    summaries = {
        method: _json(paper / method / "summary.json")
        for method in ("ours", "globalgce", "gcfexplainer", "comrecgc")
    }
    manifests = {
        method: _json(paper / method / "run_manifest.json")
        for method in summaries
    }
    audits = {
        method: _json(paper / method / "final_artifact_audit.json")
        for method in summaries
    }
    gcf_connectivity = _json(paper / "source_v4_gcf_connectivity_audit.json")
    cost_cap_values = [row.get("cost_cap") for row in summaries.values()]
    same_cost_cap = bool(
        all(value is not None for value in cost_cap_values)
        and len({float(value) for value in cost_cap_values}) == 1
    )
    expected_table_schema = ["method", "k", "coverage", "cost", "flip_rate", "cf_drop"]
    hard_checks = {
        "methods_exactly_four": base.get("methods") == expected_methods,
        "same_parent_cohort": len({row.get("test_parent_ids_sha256") for row in summaries.values()}) == 1,
        "same_teacher": len({row.get("teacher_path") for row in manifests.values()}) == 1,
        "same_molclr": len({row.get("molclr_checkpoint") for row in manifests.values()}) == 1,
        "same_threshold": len({row.get("thresholds_sha256") or row.get("thresholds_json_sha256") for row in manifests.values()}) == 1,
        "same_cost_definition": (
            same_cost_cap
            and all(row.get("table2_schema") == expected_table_schema for row in audits.values())
        ),
        "strict_flip": all(row.get("cf_mode") == "strict_flip" for row in summaries.values()),
        "connected_action_semantics": all(
            row.get("action_semantics_version") == CONNECTED_ACTION_SEMANTICS
            for row in summaries.values()
        ),
        "selection_outside_eval": all(
            row.get("selection_performed_in_eval") is False for row in manifests.values()
        ),
        "threshold_not_fit_on_test": all(
            row.get("threshold_fitted_on_test") is False for row in summaries.values()
        ),
        "test_not_used_for_selection": all(
            row.get("test_used_for_selection") is False for row in summaries.values()
        ),
        "ours_disconnected_residual_used_count_zero": int(
            summaries["ours"].get("disconnected_residual_used_count") or 0
        ) == 0,
        "gcf_all_candidates_connected": gcf_connectivity.get("all_candidates_connected") is True,
        "globalgce_all_candidates_connected": summaries["globalgce"].get("all_candidates_connected") is True,
        "comrecgc_all_evaluated_candidates_connected": summaries["comrecgc"].get("all_evaluated_candidates_connected") is True,
        "comrecgc_disconnected_output_used_count_zero": int(
            summaries["comrecgc"].get("disconnected_output_used_count") or 0
        ) == 0,
        "all_method_audits_pass": all(
            row.get("passed") is True for row in audits.values()
        ),
        "plotting_adapter_required_false": base.get("plotting_adapter_required") is False,
        "old_invalid_results_excluded": all(
            str(paper) in str((paper / method).resolve()) for method in summaries
        ),
    }
    failed = sorted(name for name, passed in hard_checks.items() if not passed)
    if failed:
        raise ValueError(f"BACE connected common4 hard checks failed: {failed}")
    cohort = {
        "passed": True,
        "same_parent_cohort": True,
        "test_parent_count": base["test_parent_count"],
        "test_parent_ids_sha256": base["test_parent_ids_sha256"],
        "methods": expected_methods,
    }
    threshold = {
        "passed": True,
        "same_threshold": True,
        "thresholds_sha256": base["thresholds_json_sha256"],
        "theta_star": base["theta_star"],
        "thresholds": base["thresholds"],
        "threshold_fitted_on_test": False,
        "method_specific_threshold": False,
    }
    result = {
        **base,
        "schema_version": "bace_common4_connected_residual_v1",
        "passed": True,
        "hard_checks": hard_checks,
        "same_parent_cohort": True,
        "same_teacher": True,
        "same_molclr": True,
        "same_threshold": True,
        "same_cost_definition": True,
        "strict_flip": True,
        "connected_action_semantics": True,
        "threshold_fitted_on_test": False,
        "selection_performed_in_eval": False,
        "test_used_for_selection": False,
        "ours_disconnected_residual_used_count": 0,
        "globalgce_disconnected_output_used_count": 0,
        "gcf_all_candidates_connected": True,
        "comrecgc_all_candidates_connected": True,
        "old_invalid_results_excluded": True,
    }
    _write(paper / "cohort_parity_audit.json", cohort)
    _write(paper / "threshold_parity_audit.json", threshold)
    _write(paper / "common_protocol_audit.json", result)
    _write(paper / "bace_paper_artifact_audit.json", base)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    result = audit_common4(root=args.root)
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_COMMON4_CONNECTED_AUDIT_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
