"""Machine-readable engineering gates for COMRECGC recovery stages."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import GenerationParameters, RecourseParameters, UPSTREAM_COMMIT, sha256_file, write_json


EXPECTED_MUT_TEACHER_SHA256 = "af213aa766626decaf99876b43ede725412a355adf37f1aa0d56233d8653e204"
EXPECTED_MOLCLR_SHA256 = "93bc4f02ea8847cd44fa21ec3f65600ff2f4a7ae6d3a85e8519a5bcc56afc20a"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def gate_aids_native_full(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_project_commit: str | None = None,
) -> dict[str, Any]:
    source = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load(source / "run_manifest.json")
    common = _load(source / "native_common_recourse.json")
    failures: list[str] = []
    expected_generation = GenerationParameters.for_mode("full").__dict__
    expected_recourse = RecourseParameters.for_mode("full").__dict__
    if manifest.get("run_complete") is not True:
        failures.append("run_complete")
    if manifest.get("mode") != "full" or manifest.get("full_parent_universe") is not True:
        failures.append("full_parent_universe")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        failures.append("upstream_commit")
    if (
        expected_project_commit is not None
        and manifest.get("project_commit") != expected_project_commit
    ):
        failures.append("project_commit")
    if manifest.get("parameters") != expected_generation:
        failures.append("generation_parameters")
    if common.get("parameters") != expected_recourse:
        failures.append("common_recourse_parameters")
    empty = bool(manifest.get("scientific_output_empty"))
    if empty and manifest.get("native_cost") is not None:
        failures.append("empty_cost_must_be_na")
    for name in ("counterfactuals.pt", "native_common_recourse.json", "native_representative_counterfactuals.pt"):
        path = source / name
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"missing:{name}")
    result = {
        "schema_version": 1,
        "stage": "aids_native_full_gate",
        "audit_passed": not failures,
        "run_complete": not failures,
        "failed_hard_checks": failures,
        "scientific_output_empty": empty,
        "status": (
            "AIDS_FULL_PASS_EMPTY"
            if not failures and empty
            else "AIDS_FULL_PASS_NONEMPTY"
            if not failures
            else "BLOCKED"
        ),
        "source_run_dir": str(source),
        "source_manifest_sha256": sha256_file(source / "run_manifest.json"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "gate_result.json", result)
    if failures:
        write_json(output / "_RUN_FAILED.json", result)
        raise ValueError("AIDS native full gate failed: " + ", ".join(failures))
    write_json(output / "_RUN_COMPLETE.json", result)
    return result


def gate_mutagenicity_chemistry_smoke(
    input_dir: str | Path,
    output_dir: str | Path,
    eval_dir: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = _load(source / "audit.json")
    failures: list[str] = []
    exact = {
        "audit_passed": True,
        "engineering_smoke_pass": True,
        "source_parent_count": 64,
        "source_roundtrip_pass_count": 64,
        "noop_roundtrip_pass_count": 64,
        "trace_parity": True,
        "raw_candidate_count": 164,
        "repair_provenance_count": 164,
        "official_medoid_count": 4,
        "repair_deterministic_count": 164,
        "one_raw_candidate_max_one_repaired_candidate": True,
        "official_cluster_rank_unchanged": True,
        "invalid_slot_backfill": False,
        "rank_compaction": False,
        "rf_used_in_repair": False,
        "wnode_used_in_repair": False,
        "strict_flip_used_in_repair": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    for field, expected in exact.items():
        if audit.get(field) != expected:
            failures.append(f"{field}:actual={audit.get(field)!r}:expected={expected!r}")
    required = (
        "source_roundtrip.csv",
        "noop_roundtrip.csv",
        "raw_candidates.csv",
        "candidate_validity.csv",
        "action_replay.jsonl",
        "repaired_candidates.pt",
        "repaired_official_medoids.pt",
        "run_manifest.json",
    )
    for name in required:
        path = source / name
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"missing:{name}")
    eval_audit: dict[str, Any] | None = None
    if eval_dir is not None:
        evaluation = Path(eval_dir).expanduser().resolve()
        eval_audit = _load(evaluation / "final_artifact_audit.json")
        expected_eval = {
            "audit_passed": True,
            "parent_count": 16,
            "distance_line": "MolCLR-Node-Wasserstein",
            "distance_type": "node_wasserstein",
            "cf_mode": "strict_flip",
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "invalid_slot_backfill": False,
            "rank_compaction": False,
            "invalid_candidates_sent_to_rf_or_wnode": False,
            "rf_callable": True,
            "wnode_callable": True,
            "calibration_loaded": False,
            "test_used_for_selection": False,
        }
        for field, expected in expected_eval.items():
            if eval_audit.get(field) != expected:
                failures.append(
                    f"eval:{field}:actual={eval_audit.get(field)!r}:expected={expected!r}"
                )
        if not (evaluation / "_SMOKE_AUDIT_COMPLETE.json").is_file():
            failures.append("eval:missing:_SMOKE_AUDIT_COMPLETE.json")
    result = {
        "schema_version": 1,
        "stage": "mutagenicity_chemistry_smoke_gate",
        "audit_passed": not failures,
        "run_complete": not failures,
        "failed_hard_checks": failures,
        "status": "MUT_REPAIR_SMOKE_PASS" if not failures else "BLOCKED",
        "project_feasibility_status": audit.get("project_feasibility_status"),
        "strict_flip_status": audit.get("strict_flip_status"),
        "repaired_candidate_count": audit.get("repaired_candidate_count"),
        "repaired_official_medoid_count": audit.get("repaired_official_medoid_count"),
        "source_run_dir": str(source),
        "source_audit_sha256": sha256_file(source / "audit.json"),
        "unified_eval_audit": eval_audit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "gate_result.json", result)
    if failures:
        write_json(output / "_RUN_FAILED.json", result)
        raise ValueError("Mutagenicity chemistry smoke gate failed: " + ", ".join(failures))
    write_json(output / "_RUN_COMPLETE.json", result)
    return result


def _csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def gate_project_full(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    expected_parent_count: int,
    expected_teacher_sha256: str,
    expected_project_commit: str | None = None,
) -> dict[str, Any]:
    source = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load(source / "run_manifest.json")
    audit = _load(source / "final_artifact_audit.json")
    failures: list[str] = []
    exact = {
        "run_complete": True,
        "mode": "full",
        "dataset_key": dataset,
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "parent_count": int(expected_parent_count),
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "candidate_order_unchanged": True,
        "invalid_candidates_sent_to_rf_or_wnode": False,
        "invalid_slot_backfill": False,
        "rank_compaction": False,
        "distance_calculation_reimplemented": False,
        "teacher_calculation_reimplemented": False,
        "calibration_loaded": False,
        "test_loaded_for_selection": False,
    }
    for field, expected in exact.items():
        if manifest.get(field) != expected:
            failures.append(f"{field}:actual={manifest.get(field)!r}:expected={expected!r}")
    if manifest.get("teacher_sha256") != expected_teacher_sha256:
        failures.append("teacher_sha256")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        failures.append("upstream_commit")
    if (
        expected_project_commit is not None
        and manifest.get("project_commit") != expected_project_commit
    ):
        failures.append("project_commit")
    if manifest.get("molclr_checkpoint_sha256") != EXPECTED_MOLCLR_SHA256:
        failures.append("molclr_checkpoint_sha256")
    if audit.get("audit_passed") is not True:
        failures.append("final_artifact_audit")
    required = (
        "pair_matrix.jsonl",
        "selected_sequence.jsonl",
        "selected_common_recourses.json",
        "representative_counterfactuals.jsonl",
        "parent_best_distances.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "table2_comrecgc_k10.csv",
        "table2_comrecgc_k20.csv",
        "summary.json",
        "run_manifest.json",
        "final_artifact_audit.json",
        "_RUN_COMPLETE.json",
    )
    empty_allowed = {
        "pair_matrix.jsonl",
        "selected_sequence.jsonl",
        "representative_counterfactuals.jsonl",
    }
    for name in required:
        path = source / name
        if not path.is_file() or (
            path.stat().st_size <= 0 and name not in empty_allowed
        ):
            failures.append(f"missing:{name}")
    prefixes = _csv_rows(source / "prefix_metrics.csv") if (source / "prefix_metrics.csv").is_file() else []
    if [int(row["k"]) for row in prefixes] != list(range(1, 21)):
        failures.append("prefix_k_grid")
    coverage = [float(row["close_cf_coverage"]) for row in prefixes]
    if any(right + 1e-12 < left for left, right in zip(coverage, coverage[1:])):
        failures.append("coverage_vs_k_monotonic")
    figure4 = _csv_rows(source / "figure4_coverage_vs_threshold.csv") if (source / "figure4_coverage_vs_threshold.csv").is_file() else []
    threshold_rows = sorted(figure4, key=lambda row: float(row["threshold"]))
    threshold_coverage = [float(row["close_cf_coverage"]) for row in threshold_rows]
    if any(right + 1e-12 < left for left, right in zip(threshold_coverage, threshold_coverage[1:])):
        failures.append("coverage_vs_threshold_monotonic")
    for row in prefixes:
        for field in ("close_cf_coverage", "applicable_coverage", "fixed_capped_mean_cost"):
            value = float(row[field])
            if not math.isfinite(value):
                failures.append(f"nonfinite:{field}:k={row['k']}")
    if prefixes and float(prefixes[-1]["close_cf_coverage"]) == 0.0:
        cost = str(prefixes[-1].get("conditional_median_cost") or "").strip()
        if cost:
            failures.append("empty_cost_must_be_na")
    result = {
        "schema_version": 1,
        "stage": f"{dataset}_project_full_gate",
        "audit_passed": not failures,
        "run_complete": not failures,
        "failed_hard_checks": failures,
        "status": "FULL_EXECUTION_PASS" if not failures else "BLOCKED",
        "dataset": dataset,
        "scientific_output_empty": bool(audit.get("scientific_output_empty")),
        "scientific_output_status": (
            "SCIENTIFIC_OUTPUT_EMPTY"
            if bool(audit.get("scientific_output_empty"))
            else "SCIENTIFIC_OUTPUT_NONEMPTY"
        ),
        "figure_artifact_status": (
            "FIGURE_ARTIFACTS_READY" if not failures else "BLOCKED"
        ),
        "strict_flip_status": audit.get("strict_flip_status"),
        "valid_k20": manifest.get("valid_k20"),
        "k20_coverage": manifest.get("k20_coverage"),
        "source_run_dir": str(source),
        "source_manifest_sha256": sha256_file(source / "run_manifest.json"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "gate_result.json", result)
    if failures:
        write_json(output / "_RUN_FAILED.json", result)
        raise ValueError(f"{dataset} project full gate failed: " + ", ".join(failures))
    write_json(output / "_RUN_COMPLETE.json", result)
    return result


def gate_mutagenicity_full(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    return gate_project_full(
        input_dir,
        output_dir,
        dataset="mutagenicity",
        expected_parent_count=217,
        expected_teacher_sha256=EXPECTED_MUT_TEACHER_SHA256,
    )
