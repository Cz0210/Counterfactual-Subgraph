#!/usr/bin/env python3
"""Fail-closed common audit for corrected BACE Ours and GCF artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return dict(payload)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(root: Path, method: str) -> None:
    for name in (
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        f"table2_{method}_k10.csv",
        "summary.json",
        "run_manifest.json",
        "final_artifact_audit.json",
    ):
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"Missing BACE common artifact: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--ours-root", required=True)
    parser.add_argument("--gcf-root", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--gcf-candidate-audit", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    ours_root = Path(args.ours_root).expanduser().resolve()
    gcf_root = Path(args.gcf_root).expanduser().resolve()
    output = Path(args.output_root).expanduser().resolve()
    if ours_root.parent != output or gcf_root.parent != output:
        raise AssertionError("Ours/GCF roots must be direct children of the common root.")
    _required(ours_root, "ours")
    _required(gcf_root, "gcfexplainer")
    threshold_path = Path(args.thresholds_json).expanduser().resolve()
    threshold_sha = _sha256(threshold_path)
    threshold = _json(threshold_path)
    ours_summary = _json(ours_root / "summary.json")
    gcf_summary = _json(gcf_root / "summary.json")
    ours_manifest = _json(ours_root / "run_manifest.json")
    gcf_manifest = _json(gcf_root / "run_manifest.json")
    ours_audit = _json(ours_root / "final_artifact_audit.json")
    gcf_audit = _json(gcf_root / "final_artifact_audit.json")
    gcf_candidates = _json(Path(args.gcf_candidate_audit).expanduser().resolve())

    same_threshold = (
        ours_manifest.get("thresholds_json_sha256")
        == gcf_manifest.get("thresholds_json_sha256")
        == threshold_sha
    )
    same_parent_cohort = (
        ours_summary.get("test_parent_ids_sha256")
        == gcf_summary.get("test_parent_ids_sha256")
        and ours_summary.get("test_parent_count") == gcf_summary.get("test_parent_count")
    )
    same_teacher = _sha256(Path(ours_manifest["teacher_path"])) == _sha256(
        Path(gcf_manifest["teacher_path"])
    )
    same_molclr = _sha256(Path(ours_manifest["molclr_checkpoint"])) == _sha256(
        Path(gcf_manifest["molclr_checkpoint"])
    )
    common_checks = {
        "same_threshold": same_threshold,
        "same_parent_cohort": same_parent_cohort,
        "same_teacher": same_teacher,
        "same_molclr": same_molclr,
        "same_cf_mode": ours_summary.get("cf_mode")
        == gcf_summary.get("cf_mode")
        == "strict_flip",
        "same_distance_line": ours_summary.get("distance_line")
        == gcf_summary.get("distance_line")
        == "MolCLR-Node-Wasserstein",
        "threshold_fitted_on_test": False,
        "ours_disconnected_residual_used_count": ours_summary.get(
            "disconnected_residual_used_count"
        ),
        "ours_covered_residual_connected_rate": ours_summary.get(
            "covered_residual_connected_rate"
        ),
        "gcf_all_candidates_connected": gcf_candidates.get(
            "all_candidates_connected"
        ),
    }
    passed = bool(
        all(
            common_checks[field]
            for field in (
                "same_threshold",
                "same_parent_cohort",
                "same_teacher",
                "same_molclr",
                "same_cf_mode",
                "same_distance_line",
                "gcf_all_candidates_connected",
            )
        )
        and common_checks["ours_disconnected_residual_used_count"] == 0
        and common_checks["ours_covered_residual_connected_rate"] in {None, 1.0}
        and ours_audit.get("passed") is True
        and gcf_audit.get("passed") is True
        and threshold.get("action_semantics_version") == CONNECTED_ACTION_SEMANTICS
        and threshold.get("match_selection_policy") == CONNECTED_MATCH_SELECTION_POLICY
        and threshold.get("threshold_fitted_on_test") is False
    )
    if not passed:
        raise AssertionError(f"BACE connected common protocol audit failed: {common_checks}")

    comparisons: list[dict[str, Any]] = []
    for method, root in (("Ours", ours_root), ("GCFExplainer", gcf_root)):
        figure3 = _csv(root / "figure3_coverage_vs_k.csv")
        table = _csv(
            root
            / (
                "table2_ours_k10.csv"
                if method == "Ours"
                else "table2_gcfexplainer_k10.csv"
            )
        )[0]
        comparisons.append(
            {
                "method": method,
                "k1_coverage": figure3[0]["coverage"],
                "k10_coverage": figure3[9]["coverage"],
                "k20_coverage": figure3[19]["coverage"],
                "k10_cost": table["cost"],
                "k10_cf_drop": table["cf_drop"],
            }
        )
    with (output / "method_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    protocol = {
        "schema_version": "bace_connected_common_protocol_audit_v3",
        "passed": True,
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        **common_checks,
        "ours_root": str(ours_root),
        "gcf_root": str(gcf_root),
    }
    _write_json(output / "bace_connected_protocol_audit.json", protocol)
    _write_json(
        output / "threshold_parity_audit.json",
        {
            "passed": True,
            "same_threshold": True,
            "threshold_manifest": str(threshold_path),
            "threshold_manifest_sha256": threshold_sha,
            "theta_star": threshold["theta_star"],
            "thresholds": threshold["thresholds"],
            "threshold_fitted_on_test": False,
        },
    )
    print(json.dumps(protocol, sort_keys=True), flush=True)
    print("[BACE_CONNECTED_COMMON_ARTIFACT_AUDIT_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
