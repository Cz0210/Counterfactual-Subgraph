from __future__ import annotations

import json
from pathlib import Path

from src.baselines.comrecgc.contracts import GenerationParameters, RecourseParameters, UPSTREAM_COMMIT
from src.baselines.comrecgc.recovery_gate import (
    EXPECTED_MOLCLR_SHA256,
    EXPECTED_MUT_TEACHER_SHA256,
    gate_aids_native_full,
    gate_mutagenicity_full,
    gate_mutagenicity_chemistry_smoke,
)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_empty_cluster_is_valid_scientific_result_and_cost_is_na(tmp_path: Path) -> None:
    source = tmp_path / "aids"
    source.mkdir()
    write(
        source / "run_manifest.json",
        {
            "run_complete": True,
            "mode": "full",
            "full_parent_universe": True,
            "upstream_commit": UPSTREAM_COMMIT,
            "parameters": GenerationParameters.for_mode("full").__dict__,
            "scientific_output_empty": True,
            "native_cost": None,
        },
    )
    write(
        source / "native_common_recourse.json",
        {"parameters": RecourseParameters.for_mode("full").__dict__},
    )
    (source / "counterfactuals.pt").write_bytes(b"payload")
    (source / "native_representative_counterfactuals.pt").write_bytes(b"empty-list-payload")

    result = gate_aids_native_full(source, tmp_path / "gate")

    assert result["audit_passed"] is True
    assert result["status"] == "AIDS_FULL_PASS_EMPTY"


def test_mut_smoke_gate_does_not_require_positive_scientific_yield(tmp_path: Path) -> None:
    source = tmp_path / "mut"
    source.mkdir()
    write(
        source / "audit.json",
        {
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
            "project_feasibility_status": "PROJECT_FEASIBILITY_NOT_OBSERVED",
            "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
            "repaired_candidate_count": 0,
            "repaired_official_medoid_count": 0,
        },
    )
    for name in (
        "source_roundtrip.csv",
        "noop_roundtrip.csv",
        "raw_candidates.csv",
        "candidate_validity.csv",
        "action_replay.jsonl",
        "repaired_candidates.pt",
        "repaired_official_medoids.pt",
        "run_manifest.json",
    ):
        (source / name).write_text("x", encoding="utf-8")

    result = gate_mutagenicity_chemistry_smoke(source, tmp_path / "gate")

    assert result["audit_passed"] is True
    assert result["status"] == "MUT_REPAIR_SMOKE_PASS"


def test_mut_full_gate_accepts_audited_empty_scientific_output(tmp_path: Path) -> None:
    source = tmp_path / "full"
    source.mkdir()
    write(
        source / "run_manifest.json",
        {
            "run_complete": True,
            "mode": "full",
            "distance_line": "MolCLR-Node-Wasserstein",
            "cf_mode": "strict_flip",
            "parent_count": 217,
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
            "teacher_sha256": EXPECTED_MUT_TEACHER_SHA256,
            "molclr_checkpoint_sha256": EXPECTED_MOLCLR_SHA256,
            "valid_k20": 0,
            "k20_coverage": 0.0,
        },
    )
    write(
        source / "final_artifact_audit.json",
        {
            "audit_passed": True,
            "scientific_output_empty": True,
            "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        },
    )
    prefix_fields = (
        "k,close_cf_coverage,applicable_coverage,fixed_capped_mean_cost,"
        "conditional_median_cost\n"
    )
    prefix_rows = "".join(f"{k},0,0,0.1,\n" for k in range(1, 21))
    (source / "prefix_metrics.csv").write_text(
        prefix_fields + prefix_rows, encoding="utf-8"
    )
    (source / "figure4_coverage_vs_threshold.csv").write_text(
        "threshold,close_cf_coverage\n0.01,0\n0.02,0\n", encoding="utf-8"
    )
    for name in (
        "pair_matrix.jsonl",
        "selected_sequence.jsonl",
        "parent_best_distances.csv",
        "prefix_metrics.json",
        "figure3_coverage_vs_k.csv",
        "table2_comrecgc_k10.csv",
        "table2_comrecgc_k20.csv",
        "summary.json",
        "_RUN_COMPLETE.json",
    ):
        (source / name).write_text("x", encoding="utf-8")

    result = gate_mutagenicity_full(source, tmp_path / "full_gate")

    assert result["audit_passed"] is True
    assert result["scientific_output_empty"] is True
    assert result["status"] == "MUT_FULL_PASS"
