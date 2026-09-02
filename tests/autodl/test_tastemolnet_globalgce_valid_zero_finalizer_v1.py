from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.baselines.globalgce_mutagenicity_adapter import (
    OFFICIAL_AFFINE_EDGE_HARD_DECODE,
)
from src.baselines.tastemolnet_globalgce_full import (
    BRANCH_MANIFEST_SCHEMA,
    CHECKPOINT_SCHEMA,
    STAGE,
    TasteGlobalGCEFullConfig,
    stable_sha256,
)
from src.data.tastemolnet_ppo import TASTEMOLNET_PREPARED_FIELDS
from src.eval import tastemolnet_globalgce_valid_zero as zero
from src.eval import tastemolnet_matrix_append as matrix


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _branch(root: Path, *, gnn_model: Path) -> list[dict[str, Any]]:
    root.mkdir(parents=True)
    (root / "native_rule_catalog.jsonl").write_bytes(b"")
    rejected = [
        {
            "native_rule_index": 0,
            "candidate_id": "native-index-0",
            "reason": "GlobalGCENativeRuleError:RHS adjacency is asymmetric",
        }
    ]
    (root / "native_rule_rejections.jsonl").write_text(
        json.dumps(rejected[0], sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (root / "globalgce_rules.pt").write_bytes(b"rules")
    (root / "branch_manifest.json").write_text("{}\n", encoding="utf-8")
    summary = {
        "native_rule_count": 1,
        "valid_native_rule_count": 0,
        "rejected_native_rule_count": 1,
        "native_rule_edge_score_contract": (
            "pinned_official_unbounded_affine_class_scores"
        ),
        "native_rule_edge_score_hard_decode": OFFICIAL_AFFINE_EDGE_HARD_DECODE,
        "native_rule_catalog": str(root / "native_rule_catalog.jsonl"),
        "native_rule_catalog_sha256": hashlib.sha256(b"").hexdigest(),
        "native_rule_rejections": str(root / "native_rule_rejections.jsonl"),
        "gnn_checkpoint": str(gnn_model),
        "codec_metadata": {
            "node_label_mapping": {"0": "padding", "1": "C"},
            "edge_label_mapping": {"0": "no_edge", "1": "single"},
        },
    }
    _write_json(root / "training_core_summary.json", summary)
    return rejected


def _source(tmp_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]], Path]:
    source = tmp_path / "source"
    (source / "raw").mkdir(parents=True)
    gnn = tmp_path / "t3"
    gnn.mkdir()
    model = gnn / "model.pt"
    model.write_bytes(b"gine")
    _write_json(gnn / "temperature_scaling.json", {"temperature": 1.25})
    checkpoint_hash = hashlib.sha256(b"gine").hexdigest()
    config = TasteGlobalGCEFullConfig().to_dict()
    resume = {
        "schema_version": "tastemolnet_t13_resume_identity_v1",
        "dataset": "TasteMolNet",
        "method": "GlobalGCE",
        "stage": STAGE,
        "config": config,
        "train_sha256": "1" * 64,
        "calibration_sha256": "2" * 64,
        "declared_test_sha256": "3" * 64,
        "checkpoint_id": checkpoint_hash,
        "dataset_hash": "4" * 64,
        "split_manifest_sha256": "5" * 64,
        "molclr_checkpoint_sha256": "6" * 64,
        "t8_pass_sha256": "7" * 64,
        "t8_oracle_checkpoint_hash": checkpoint_hash,
        "threshold_config_hash": "8" * 64,
    }
    _write_json(
        source / "checkpoint.json",
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "stage": STAGE,
            "phase": "TARGET_2_COMPLETE",
            "resume_identity": resume,
            "resume_identity_sha256": stable_sha256(resume),
        },
    )
    cohort_sha = "9" * 64
    _write_json(
        source / "raw/train_cohort_manifest.json",
        {
            "selected_count": 2,
            "ordered_parent_cohort_sha256": cohort_sha,
            "train_only": True,
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    rejected = _branch(source / "raw/target_0", gnn_model=model)
    _branch(source / "raw/target_2", gnn_model=model)
    proc = tmp_path / "proc"
    proc.mkdir()
    return source, resume, rejected, proc


def _branch_validator(**kwargs: Any) -> dict[str, Any]:
    target = kwargs["target_label"]
    return {
        "schema_version": BRANCH_MANIFEST_SCHEMA,
        "valid_native_rule_count": 0,
        "target_label": target,
        "oracle_resume_identity_sha256": "a" * 64,
        "native_train_cohort_sha256": "b" * 64,
        "source_train_cohort_sha256": "c" * 64,
        "official_source_identity_sha256": "d" * 64,
    }


def test_one_attempt_receipt_and_authorization_are_exact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    attempt_path = tmp_path / "attempt.json"
    _write_json(
        attempt_path,
        {
            "schema_version": zero.ATTEMPT_SCHEMA,
            "status": "CONSUMED",
            "attempt_id": "18675079-382b-41a9-b6ac-f5aa6e79babf",
            "attempt_ordinal": 1,
            "max_attempts": 1,
            "seed": 7,
            "epochs": 100,
            "gpu_index": 1,
            "output_root": str(source),
            "controller_root": str(tmp_path / "controller"),
        },
    )
    attempt = zero.validate_attempt_receipt(attempt_path, source_root=source)
    receipt = zero.build_authorization_receipt(
        source_root=source, attempt_receipt=attempt, execution_commit="e" * 40
    )
    auth_path = tmp_path / "authorization.json"
    _write_json(auth_path, receipt)
    reopened = zero.validate_authorization_receipt(
        auth_path, source_root=source, attempt_receipt=attempt
    )
    assert reopened["allow_second_recovery_attempt"] is False
    tampered = json.loads(attempt_path.read_text())
    tampered["max_attempts"] = 2
    _write_json(attempt_path, tampered)
    with pytest.raises(zero.TasteGlobalGCEValidZeroError, match="sole"):
        zero.validate_attempt_receipt(attempt_path, source_root=source)


def test_typed_zero_rejections_are_independently_replayed(tmp_path: Path) -> None:
    gnn = tmp_path / "model.pt"
    gnn.write_bytes(b"gine")
    branch = tmp_path / "target_0"
    rejected = _branch(branch, gnn_model=gnn)
    result = zero.validate_zero_branch_rejections(
        branch,
        oracle_checkpoint_hash=hashlib.sha256(b"gine").hexdigest(),
        rules_loader=lambda _path: {"frozen": True},
        rematerializer=lambda **_kwargs: ([], rejected),
    )
    assert result["valid_native_rule_count"] == 0
    assert result["typed_scientific_rejections_only"] is True


def test_engineering_exception_cannot_be_published_as_scientific_zero(
    tmp_path: Path,
) -> None:
    gnn = tmp_path / "model.pt"
    gnn.write_bytes(b"gine")
    branch = tmp_path / "target_0"
    rejected = _branch(branch, gnn_model=gnn)
    rejected[0]["reason"] = "OSError:disk read failed"
    (branch / "native_rule_rejections.jsonl").write_text(
        json.dumps(rejected[0]) + "\n", encoding="utf-8"
    )
    with pytest.raises(zero.TasteGlobalGCEValidZeroError, match="engineering"):
        zero.validate_zero_branch_rejections(
            branch,
            oracle_checkpoint_hash="a" * 64,
            rules_loader=lambda _path: {},
            rematerializer=lambda **_kwargs: ([], rejected),
        )


def test_source_gate_requires_both_completed_zero_branches(tmp_path: Path) -> None:
    source, _resume, rejected, proc = _source(tmp_path)
    audit = zero.validate_valid_zero_source(
        source,
        proc_root=proc,
        branch_validator=_branch_validator,
        rules_loader=lambda _path: {},
        rematerializer=lambda **_kwargs: ([], rejected),
    )
    assert audit["valid_unique_rule_count"] == 0
    assert audit["both_target_branches_complete"] is True
    assert audit["writer_audit"]["writers"] == []
    checkpoint = json.loads((source / "checkpoint.json").read_text())
    checkpoint["phase"] = "TARGET_0_COMPLETE"
    _write_json(source / "checkpoint.json", checkpoint)
    with pytest.raises(zero.TasteGlobalGCEValidZeroError, match="completed"):
        zero.validate_valid_zero_source(
            source,
            proc_root=proc,
            branch_validator=_branch_validator,
            rules_loader=lambda _path: {},
            rematerializer=lambda **_kwargs: ([], rejected),
        )


def _test_csv(path: Path) -> None:
    row = {field: "x" for field in TASTEMOLNET_PREPARED_FIELDS}
    row.update(
        {
            "molecule_id": "test-sweet-1",
            "raw_smiles": "CC",
            "canonical_smiles": "CC",
            "model_smiles": "CC",
            "label": "1",
            "label_name": "Sweet",
            "split": "test",
            "exclusion_reason": "",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASTEMOLNET_PREPARED_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def test_valid_zero_overlay_has_no_fake_rules_and_zero_plateau(tmp_path: Path) -> None:
    source, resume, rejected, proc = _source(tmp_path)
    test_csv = tmp_path / "test.csv"
    _test_csv(test_csv)
    thresholds = [0.1, 0.2]
    threshold_path = tmp_path / "threshold.json"
    _write_json(
        threshold_path,
        {
            "dataset": "TasteMolNet",
            "thresholds": thresholds,
            "theta_star": 0.1,
            "cost_cap": 1.0,
            "threshold_source": "frozen T3 calibration",
            "threshold_source_split": "calibration",
            "test_used_for_selection": False,
            "threshold_config_hash": stable_sha256(thresholds),
        },
    )
    resume["declared_test_sha256"] = hashlib.sha256(test_csv.read_bytes()).hexdigest()
    resume["threshold_config_hash"] = stable_sha256(thresholds)
    checkpoint = json.loads((source / "checkpoint.json").read_text())
    checkpoint["resume_identity"] = resume
    checkpoint["resume_identity_sha256"] = stable_sha256(resume)
    _write_json(source / "checkpoint.json", checkpoint)
    source_audit = zero.validate_valid_zero_source(
        source,
        proc_root=proc,
        branch_validator=_branch_validator,
        rules_loader=lambda _path: {},
        rematerializer=lambda **_kwargs: ([], rejected),
    )
    attempt = {
        "attempt_id": "18675079-382b-41a9-b6ac-f5aa6e79babf",
        "receipt_sha256": "a" * 64,
    }
    authorization = {"receipt_file_sha256": "b" * 64}
    observation = {
        "root_completed_count": 100,
        "root_total_count": 100,
        "patterns_seen": 123,
        "patterns_delta": 10,
        "patterns_per_minute": 2.5,
        "rss_bytes": 1024,
        "cpu_percent": 99.0,
        "last_progress_time": "2026-09-03T01:00:00Z",
        "observation_sha256": "c" * 64,
    }
    output = tmp_path / "overlay"
    result = zero.publish_valid_zero_result(
        source_audit=source_audit,
        attempt_receipt=attempt,
        authorization=authorization,
        observation=observation,
        test_csv=test_csv,
        threshold_contract=threshold_path,
        output_root=output,
        execution_commit="d" * 40,
    )
    assert result["valid_zero_result"] is True
    assert (output / "raw/merged_rules.jsonl").read_bytes() == b""
    assert (output / "raw/selected_rules.jsonl").read_bytes() == b""
    figure3 = list(csv.DictReader((output / "figure3_coverage_vs_k.csv").open()))
    assert [int(row["k"]) for row in figure3] == list(range(1, 21))
    assert all(float(row["coverage"]) == 0.0 for row in figure3)
    assert all(row["cost"] == "N/A" for row in figure3)
    table2 = list(csv.DictReader((output / "table2_globalgce_k10.csv").open()))
    assert len(table2) == 1
    assert int(table2[0]["effective_rule_count"]) == 0
    assert (output / "PASS").read_bytes() == b"PASS\n"


def test_registry_reconciliation_is_limited_to_exact_na_cost_shape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "terminal"
    root.mkdir()
    _write_json(root / "run_manifest.json", {"schema_version": zero.RUN_MANIFEST_SCHEMA})
    _write_json(
        root / "terminal.json",
        {
            "schema_version": zero.TERMINAL_SCHEMA,
            "result_type": zero.RESULT_TYPE,
            "valid_zero_result": True,
            "no_valid_rule_generated": True,
            "effective_rule_count": 0,
            "coverage": 0.0,
            "CCRCOV": 0.0,
            "flip_count": 0,
            "cost": "N/A",
            "numeric_imputation_used": False,
        },
    )
    row = {
        "dataset": "TasteMolNet",
        "method": "GlobalGCE",
        "status": "STALE_METRIC",
        "k_max": 20,
        "table2_k": 10,
        "rerun_reason": "FIGURE3_INVALID:ValueError;TABLE2_INVALID:ValueError",
    }
    promoted = matrix._reconcile_globalgce_valid_zero_registry_row(
        row, terminal_root=root
    )
    assert promoted["status"] == "FROZEN_PASS"
    row["rerun_reason"] += ";FIGURE4_INVALID:ValueError"
    with pytest.raises(matrix.TasteMatrixAppendError, match="exact approved"):
        matrix._reconcile_globalgce_valid_zero_registry_row(row, terminal_root=root)
