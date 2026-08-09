#!/usr/bin/env python3
"""Fail-closed audit for the BACE Ours v2 selector or frozen test artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"Expected object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_monotone(values: list[float], label: str) -> None:
    if any(right + 1e-12 < left for left, right in zip(values, values[1:])):
        raise AssertionError(f"{label} is not monotone non-decreasing")


def audit_selector(
    root: Path,
    *,
    require_frozen: bool = True,
    require_connected: bool = False,
) -> dict[str, Any]:
    required = (
        "selected_top20.csv",
        "selected_top20.json",
        "selected_top20.sha256",
        "selected_subgraphs.csv",
        "selector_cv_results.csv",
        "selector_variant_calibration.csv",
        "selected_variant_manifest.json",
        "rank_preservation_audit.json",
        "candidate_pool_limitation_audit.json",
        "summary.json",
        "run_manifest.json",
        "_RUN_COMPLETE.json",
    )
    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"Missing selector artifact: {path}")
    summary = _json(root / "summary.json")
    selection_frozen = summary.get("selection_frozen") is True
    selection_name = (
        "frozen_selection.json" if selection_frozen else "provisional_selection.json"
    )
    selection_path = root / selection_name
    if not selection_path.is_file() or selection_path.stat().st_size <= 0:
        raise AssertionError(f"Missing selector selection record: {selection_path}")
    if require_frozen and not selection_frozen:
        raise AssertionError(
            "Candidate expansion is required; provisional selection cannot enter test evaluation"
        )
    selected = _csv(root / "selected_top20.csv")
    if [int(row["rank"]) for row in selected] != list(range(1, 21)):
        raise AssertionError("selected_top20 ranks differ from 1..20")
    candidate_ids = [row["candidate_id"] for row in selected]
    fragments = [row["fragment"] for row in selected]
    if len(set(candidate_ids)) != 20 or len(set(fragments)) != 20:
        raise AssertionError("selected_top20 contains duplicate actions")
    selection = _json(selection_path)
    if selection.get("selection_frozen") is not selection_frozen:
        raise AssertionError("Selection record and summary freeze states differ")
    if selection.get("test_used") is not False or selection.get("gcf_result_used") is not False:
        raise AssertionError("Selector provenance admits test/GCF input")
    selected_sha = _sha256(root / "selected_top20.csv")
    if selection.get("selected_sequence_sha256") != selected_sha:
        raise AssertionError("Selection SHA does not identify selected_top20.csv")
    run = _json(root / "run_manifest.json")
    if run.get("selection_split") != "calibration" or run.get("test_loaded") is not False:
        raise AssertionError("Selector was not calibration-only")
    if run.get("selection_performed_in_eval") is not False:
        raise AssertionError("Selection is marked as occurring in evaluation")
    if require_connected:
        connected_required = (
            "candidate_connectivity_stats.csv",
            "selector_manifest.json",
            "selector_audit.json",
        )
        for name in connected_required:
            if not (root / name).is_file():
                raise AssertionError(f"Missing connected selector artifact: {root / name}")
        connected_audit = _json(root / "selector_audit.json")
        for payload in (selection, run, connected_audit):
            if payload.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
                raise AssertionError("Selector lacks connected action semantics")
            if payload.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
                raise AssertionError("Selector match policy changed")
        if connected_audit.get("disconnected_residual_used_count") != 0:
            raise AssertionError("Selector used a disconnected residual")
        if connected_audit.get("all_calibration_winners_connected") is not True:
            raise AssertionError("Selector has a disconnected calibration winner")
        if connected_audit.get(
            "all_selected_have_connected_valid_calibration_action"
        ) is not True:
            raise AssertionError("A selected action lacks connected calibration support")
    rank = _json(root / "rank_preservation_audit.json")
    if rank.get("rank_preservation_pass") is not True:
        raise AssertionError("Rank preservation failed")
    return {
        "mode": "selector",
        "pass": True,
        "candidate_count": 20,
        "selected_sequence_sha256": selected_sha,
        "selection_frozen": selection_frozen,
        "test_used": False,
        "gcf_result_used": False,
        "action_semantics_version": run.get("action_semantics_version"),
        "match_selection_policy": run.get("match_selection_policy"),
    }


def audit_final(
    root: Path,
    selector_root: Path,
    *,
    require_connected: bool = False,
) -> dict[str, Any]:
    required = (
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "table2_ours_k10.csv",
        "summary.json",
        "run_manifest.json",
        "final_artifact_audit.json",
    )
    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"Missing final artifact: {path}")
    selector_audit = audit_selector(
        selector_root,
        require_frozen=True,
        require_connected=require_connected,
    )
    figure3 = _csv(root / "figure3_coverage_vs_k.csv")
    figure4 = _csv(root / "figure4_coverage_vs_threshold.csv")
    table2 = _csv(root / "table2_ours_k10.csv")
    if list(figure3[0]) != ["method", "k", "coverage", "cost"]:
        raise AssertionError("Figure 3 schema changed")
    if list(figure4[0]) != ["method", "threshold", "coverage"]:
        raise AssertionError("Figure 4 schema changed")
    if list(table2[0]) != ["method", "k", "coverage", "cost", "flip_rate", "cf_drop"]:
        raise AssertionError("Table 2 schema changed")
    if [int(row["k"]) for row in figure3] != list(range(1, 21)):
        raise AssertionError("Figure 3 K grid changed")
    _assert_monotone([float(row["coverage"]) for row in figure3], "Figure 3")
    _assert_monotone([float(row["coverage"]) for row in figure4], "Figure 4")
    summary = _json(root / "summary.json")
    manifest = _json(root / "run_manifest.json")
    artifact_audit = _json(root / "final_artifact_audit.json")
    if summary.get("cf_mode") != "strict_flip":
        raise AssertionError("strict_flip changed")
    if summary.get("distance_line") != "MolCLR-Node-Wasserstein":
        raise AssertionError("distance line changed")
    if manifest.get("selection_performed_in_eval") is not False:
        raise AssertionError("Final run selected candidates in evaluation")
    if manifest.get("threshold_fitted_on_test") is not False:
        raise AssertionError("Final run fitted threshold on test")
    if require_connected:
        for payload in (summary, manifest, artifact_audit):
            if payload.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
                raise AssertionError("Final artifact lacks connected action semantics")
            if payload.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
                raise AssertionError("Final artifact match policy changed")
        if summary.get("disconnected_residual_used_count") != 0:
            raise AssertionError("Final artifact used a disconnected residual")
        if summary.get("covered_residual_connected_rate") not in {None, 1.0}:
            raise AssertionError("Final covered residual connected rate is not 1")
    if int(manifest.get("test_evaluation_count") or 0) != 1:
        raise AssertionError("Final run must record exactly one test evaluation")
    if manifest.get("selected_sequence_sha256") != selector_audit["selected_sequence_sha256"]:
        raise AssertionError("Final run did not use the frozen selected sequence")
    required_identity_flags = (
        "selected_candidate_ids_exact",
        "teacher_identity_exact",
        "molclr_identity_exact",
        "threshold_identity_exact",
        "same_test_parents",
        "same_theta",
        "same_cost_definition",
        "same_reference_teacher",
        "same_reference_molclr",
    )
    for field in required_identity_flags:
        source = summary if field in summary else artifact_audit
        if source.get(field) is not True:
            raise AssertionError(f"Final identity/protocol gate failed: {field}")
    frozen = _json(selector_root / "frozen_selection.json")
    if not math.isclose(
        float(summary["theta_star"]),
        float(frozen["theta_star"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Final theta differs from frozen selection")
    if manifest.get("thresholds_json_sha256") != frozen.get(
        "threshold_manifest_sha256"
    ):
        raise AssertionError("Final threshold manifest differs from frozen selection")
    if not math.isclose(
        float(table2[0]["coverage"]),
        float(figure3[9]["coverage"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Table 2 does not equal Figure 3 K=10")
    return {
        "mode": "final",
        "pass": True,
        "rank_preservation_pass": True,
        "strict_flip": True,
        "selection_performed_in_eval": False,
        "threshold_fitted_on_test": False,
        "test_evaluation_count": 1,
        "same_teacher": True,
        "same_molclr": True,
        "same_theta": True,
        "same_test_parents": True,
        "same_cost_definition": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("selector", "final"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--selector-root")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--require-connected", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if args.mode == "selector":
        result = audit_selector(
            root,
            require_frozen=not args.allow_provisional,
            require_connected=bool(args.require_connected),
        )
    else:
        if not args.selector_root:
            raise ValueError("--mode final requires --selector-root")
        result = audit_final(
            root,
            Path(args.selector_root).expanduser().resolve(),
            require_connected=bool(args.require_connected),
        )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    print("[BACE_OURS_WNODE_PREFIX_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
