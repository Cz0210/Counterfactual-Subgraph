"""Calibration-only B12 selection, frozen B13 metrics, and manifest-only B14."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.bace_frozen_gnn_contracts import (
    CF_MODE,
    CLASSIFIER_TYPE,
    DATASET,
    NUM_CLASSES,
    ORACLE_BACKEND,
    SOURCE_LABEL,
    assert_no_rf_provenance,
    atomic_json,
    atomic_marker,
    file_identity,
    fresh_output_dir,
    read_json,
    read_jsonl,
    sha256_file,
    stable_sha256,
    utc_now,
    validate_pass_manifest,
)
from src.eval.mutagenicity_wnode_selector import run_mutagenicity_wnode_selector


def _rule_hash(row: Mapping[str, Any]) -> str:
    return stable_sha256(
        {
            "candidate_id": str(row.get("candidate_id") or ""),
            "canonical_fragment": str(row.get("canonical_fragment") or ""),
        }
    )


def run_b12_selector(
    *,
    matrix_output: str | Path,
    output_dir: str | Path,
    seed: int = 13,
) -> dict[str, Any]:
    matrix_root = Path(matrix_output).expanduser().resolve(strict=True)
    matrix_manifest = validate_pass_manifest(
        matrix_root / "matrix_manifest.json",
        expected_stage="B11_CROSS_PARENT_VERIFIED",
        require_no_test=True,
    )
    assert_no_rf_provenance(matrix_manifest)
    output = Path(output_dir).expanduser()
    if not output.is_absolute():
        raise ValueError(f"B12 output root must be absolute: {output}")
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"B12 output root must be fresh: {output}")
    # The reused selector creates the fresh directory itself and independently
    # enforces calibration-only input paths and test_loaded=false.
    summary = run_mutagenicity_wnode_selector(
        matrix_run_dir=matrix_root,
        output_dir=output,
        top_k=20,
        table_k=10,
        seed=int(seed),
        forbid_test=True,
    )
    if summary.get("test_loaded") is not False:
        raise RuntimeError("B12 selector unexpectedly loaded held-out test data")
    decision = read_json(output / "calibration_decision.json")
    variant = str(decision.get("selected_variant") or "")
    if not variant:
        raise RuntimeError("B12 calibration selector did not choose a variant")
    selected_path = output / "variants" / variant / "selected_top20.json"
    selected = read_json(selected_path)
    candidates = [dict(row) for row in selected.get("candidates", [])]
    ordered_ids = [str(value) for value in selected.get("candidate_ids", [])]
    if (
        len(candidates) != 20
        or len(ordered_ids) != 20
        or len(set(ordered_ids)) != 20
        or ordered_ids != [str(row.get("candidate_id") or "") for row in candidates]
    ):
        raise RuntimeError("B12 must freeze exactly 20 unique ordered rules")
    thresholds = read_json(output / "thresholds.json")
    selector_manifest = read_json(output / "run_manifest.json")
    rule_hashes = [_rule_hash(row) for row in candidates]
    top20 = {
        "schema_version": "bace_frozen_gnn_selected_top20_v1",
        "dataset": DATASET,
        "stage": "B12_SELECTOR",
        "status": "FROZEN",
        "variant": variant,
        "K": 20,
        "candidate_ids": ordered_ids,
        "candidates": candidates,
        "rule_hashes": rule_hashes,
        "test_loaded": False,
    }
    atomic_json(output / "selected_top20.json", top20)
    frozen = {
        "schema_version": "bace_frozen_gnn_selection_manifest_v1",
        "dataset": DATASET,
        "stage": "B12_SELECTOR",
        "status": "FROZEN",
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "cf_mode": CF_MODE,
        "policy_checkpoint_hash": matrix_manifest["policy_checkpoint_hash"],
        "oracle_checkpoint_hash": matrix_manifest["oracle_checkpoint_hash"],
        "molclr_checkpoint_hash": matrix_manifest["molclr_checkpoint_hash"],
        "selector_fitted_on_calibration": True,
        "selection_frozen": True,
        "calibration_loaded": True,
        "test_loaded": False,
        "test_used": False,
        "K": 20,
        "ordered_rule_ids": ordered_ids,
        "ordered_rule_ids_sha256": stable_sha256(ordered_ids),
        "rule_hashes": rule_hashes,
        "prefixes": {
            str(k): ordered_ids[:k] for k in range(1, 21)
        },
        "thresholds": thresholds,
        "selector_config": selector_manifest.get("config"),
        "selected_variant": variant,
        "calibration_input_hash": sha256_file(matrix_root / "pair_matrix.jsonl"),
        "candidate_pool_hash": matrix_manifest["candidate_universe_hash"],
        "selected_top20_hash": sha256_file(output / "selected_top20.json"),
        "matrix_manifest_identity": file_identity(
            matrix_root / "matrix_manifest.json"
        ),
        "created_at": utc_now(),
    }
    atomic_json(output / "frozen_selection_manifest.json", frozen)
    atomic_json(
        output / "oracle_provenance.json",
        {
            key: frozen[key]
            for key in (
                "dataset",
                "stage",
                "status",
                "oracle_backend",
                "classifier_type",
                "rf_oracle_used",
                "source_label",
                "num_classes",
                "policy_checkpoint_hash",
                "oracle_checkpoint_hash",
                "calibration_loaded",
                "test_loaded",
            )
        },
    )
    atomic_marker(output / "PASS", "PASS")
    return frozen


def _merged_thresholds(frozen_selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = frozen_selection.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("B12 frozen manifest lacks thresholds")
    values = [dict(row) for row in thresholds.get("merged_thresholds", [])]
    if not values:
        raise ValueError("B12 frozen manifest has no calibrated thresholds")
    for row in values:
        value = float(row["threshold"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("B12 threshold is non-finite or negative")
    return values


def compute_frozen_test_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    frozen_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only the frozen ordered prefixes and calibration thresholds."""

    ordered_ids = [str(value) for value in frozen_selection["ordered_rule_ids"]]
    if len(ordered_ids) != 20 or len(set(ordered_ids)) != 20:
        raise ValueError("Frozen test evaluation requires exactly 20 ordered rules")
    thresholds = _merged_thresholds(frozen_selection)
    by_parent: dict[str, dict[str, float]] = {}
    for row in pair_rows:
        parent_id = str(row.get("parent_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id not in ordered_ids or not parent_id:
            raise ValueError("B13 pair matrix escaped the frozen B12 universe")
        value = row.get("wnode_distance")
        distance = math.inf
        if row.get("pair_strict_flip"):
            try:
                distance = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Strict B13 pair lacks a WNode distance") from exc
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError("Strict B13 pair has invalid WNode distance")
        by_parent.setdefault(parent_id, {})[candidate_id] = distance
    if not by_parent:
        raise ValueError("B13 final evaluation has no held-out source parents")
    expected_pairs = len(by_parent) * len(ordered_ids)
    if len(pair_rows) != expected_pairs or any(
        set(values) != set(ordered_ids) for values in by_parent.values()
    ):
        raise ValueError("B13 pair matrix is not the full frozen test Cartesian product")
    parent_ids = sorted(by_parent)
    distance_matrix = np.asarray(
        [[by_parent[parent][candidate] for candidate in ordered_ids] for parent in parent_ids],
        dtype=np.float64,
    )
    prefix_rows: list[dict[str, Any]] = []
    for k in range(1, 21):
        best = np.min(distance_matrix[:, :k], axis=1)
        row: dict[str, Any] = {
            "K": k,
            "strict_flip_any_rate": float(np.mean(np.isfinite(best))),
            "finite_best_count": int(np.count_nonzero(np.isfinite(best))),
        }
        for threshold in thresholds:
            threshold_id = str(threshold["threshold_id"])
            value = float(threshold["threshold"])
            row[f"ccrcov_{threshold_id}"] = float(np.mean(best <= value))
        prefix_rows.append(row)
    theta_star = float(frozen_selection["thresholds"]["theta_star"])
    theta_star_curve = [
        float(np.mean(np.min(distance_matrix[:, :k], axis=1) <= theta_star))
        for k in range(1, 21)
    ]
    return {
        "schema_version": "bace_frozen_gnn_test_metrics_v1",
        "dataset": DATASET,
        "stage": "B13_FINAL_EVAL",
        "status": "PASS",
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "cf_mode": CF_MODE,
        "parent_count": len(parent_ids),
        "ordered_rule_count": len(ordered_ids),
        "ordered_rule_ids": ordered_ids,
        "threshold_source": "B12_calibration_frozen",
        "theta_star": theta_star,
        "ccrcov_theta_star_by_k": theta_star_curve,
        "prefix_metrics": prefix_rows,
        "selector_refit_on_test": False,
        "threshold_refit_on_test": False,
        "test_loaded": True,
    }


def finalize_b13_output(
    *,
    b13_output: str | Path,
    b12_output: str | Path,
) -> dict[str, Any]:
    b13_root = Path(b13_output).expanduser().resolve(strict=True)
    b12_root = Path(b12_output).expanduser().resolve(strict=True)
    verification = validate_pass_manifest(
        b13_root / "matrix_manifest.json",
        expected_stage="B13_FINAL_EVAL",
    )
    frozen = validate_pass_manifest(
        b12_root / "frozen_selection_manifest.json",
        expected_stage="B12_SELECTOR",
        require_no_test=True,
    )
    if (
        verification.get("test_loaded") is not True
        or verification.get("test_used_only_after_freeze") is not True
        or verification.get("selector_fitted_on_calibration") is not True
    ):
        raise ValueError("B13 verification does not prove the frozen test boundary")
    for field in (
        "oracle_checkpoint_hash",
        "molclr_checkpoint_hash",
        "policy_checkpoint_hash",
    ):
        if verification.get(field) != frozen.get(field):
            raise ValueError(f"B12/B13 identity mismatch: {field}")
    pair_rows = read_jsonl(b13_root / "pair_matrix.jsonl")
    metrics = compute_frozen_test_metrics(pair_rows, frozen_selection=frozen)
    metrics.update(
        {
            "oracle_checkpoint_hash": verification["oracle_checkpoint_hash"],
            "molclr_checkpoint_hash": verification["molclr_checkpoint_hash"],
            "policy_checkpoint_hash": verification["policy_checkpoint_hash"],
            "frozen_selection_hash": sha256_file(
                b12_root / "frozen_selection_manifest.json"
            ),
            "pair_matrix_hash": sha256_file(b13_root / "pair_matrix.jsonl"),
            "created_at": utc_now(),
        }
    )
    atomic_json(b13_root / "final_metrics.json", metrics)
    manifest = {
        "schema_version": "bace_frozen_gnn_test_evaluation_manifest_v1",
        "dataset": DATASET,
        "stage": "B13_FINAL_EVAL",
        "status": "PASS",
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "cf_mode": CF_MODE,
        "policy_checkpoint_hash": verification["policy_checkpoint_hash"],
        "oracle_checkpoint_hash": verification["oracle_checkpoint_hash"],
        "molclr_checkpoint_hash": verification["molclr_checkpoint_hash"],
        "ordered_rule_ids": frozen["ordered_rule_ids"],
        "ordered_rule_ids_sha256": frozen["ordered_rule_ids_sha256"],
        "selector_fitted_on_calibration": True,
        "selection_frozen_before_test": True,
        "test_used_only_after_freeze": True,
        "selector_refit_on_test": False,
        "threshold_refit_on_test": False,
        "calibration_loaded": False,
        "test_loaded": True,
        "frozen_selection_manifest_identity": file_identity(
            b12_root / "frozen_selection_manifest.json"
        ),
        "verification_manifest_identity": file_identity(
            b13_root / "matrix_manifest.json"
        ),
        "pair_matrix_identity": file_identity(b13_root / "pair_matrix.jsonl"),
        "final_metrics_identity": file_identity(b13_root / "final_metrics.json"),
        "created_at": utc_now(),
    }
    atomic_json(b13_root / "test_evaluation_manifest.json", manifest)
    return manifest


def _verify_declared_identity(identity: Mapping[str, Any]) -> None:
    path = Path(str(identity.get("path") or "")).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Frozen artifact is not a file: {path}")
    if int(identity.get("size", -1)) != int(path.stat().st_size):
        raise ValueError(f"Frozen artifact size changed: {path}")
    if str(identity.get("sha256") or "") != sha256_file(path):
        raise ValueError(f"Frozen artifact SHA256 changed: {path}")


def run_b14_manifest_freeze(
    *,
    b12_output: str | Path,
    b13_output: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze B14 from manifests only; raw calibration/test are never reopened."""

    b12_root = Path(b12_output).expanduser().resolve(strict=True)
    b13_root = Path(b13_output).expanduser().resolve(strict=True)
    frozen = validate_pass_manifest(
        b12_root / "frozen_selection_manifest.json",
        expected_stage="B12_SELECTOR",
        require_no_test=True,
    )
    test_manifest = validate_pass_manifest(
        b13_root / "test_evaluation_manifest.json",
        expected_stage="B13_FINAL_EVAL",
    )
    metrics = validate_pass_manifest(
        b13_root / "final_metrics.json",
        expected_stage="B13_FINAL_EVAL",
    )
    assert_no_rf_provenance(frozen)
    assert_no_rf_provenance(test_manifest)
    assert_no_rf_provenance(metrics)
    for identity_name in (
        "frozen_selection_manifest_identity",
        "verification_manifest_identity",
        "pair_matrix_identity",
        "final_metrics_identity",
    ):
        _verify_declared_identity(test_manifest[identity_name])
    required = {
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "cf_mode": CF_MODE,
        "selector_fitted_on_calibration": True,
        "test_used_only_after_freeze": True,
    }
    failures = [
        f"{key}={test_manifest.get(key)!r}"
        for key, expected in required.items()
        if test_manifest.get(key) != expected
    ]
    if test_manifest.get("ordered_rule_ids") != frozen.get("ordered_rule_ids"):
        failures.append("ordered_rule_ids_changed_after_B12")
    for field in (
        "oracle_checkpoint_hash",
        "molclr_checkpoint_hash",
        "policy_checkpoint_hash",
    ):
        if test_manifest.get(field) != frozen.get(field):
            failures.append(f"{field}_changed_after_B12")
    if failures:
        raise ValueError("B14 final gate failed: " + ", ".join(failures))
    output = fresh_output_dir(output_dir)
    freeze_manifest = {
        "schema_version": "bace_frozen_gnn_final_freeze_v1",
        "dataset": DATASET,
        "stage": "B14_FROZEN",
        "status": "FROZEN",
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "cf_mode": CF_MODE,
        "policy_checkpoint_hash": frozen["policy_checkpoint_hash"],
        "oracle_checkpoint_hash": frozen["oracle_checkpoint_hash"],
        "molclr_checkpoint_hash": frozen["molclr_checkpoint_hash"],
        "selector_fitted_on_calibration": True,
        "test_used_only_after_freeze": True,
        "all_hashes_frozen": True,
        "manifest_only_gate": True,
        "raw_calibration_reopened": False,
        "raw_test_reopened": False,
        "ordered_rule_ids": frozen["ordered_rule_ids"],
        "rule_hashes": frozen["rule_hashes"],
        "K": 20,
        "b12_manifest_identity": file_identity(
            b12_root / "frozen_selection_manifest.json"
        ),
        "b13_manifest_identity": file_identity(
            b13_root / "test_evaluation_manifest.json"
        ),
        "final_metrics_identity": file_identity(b13_root / "final_metrics.json"),
        "created_at": utc_now(),
    }
    atomic_json(output / "freeze_manifest.json", freeze_manifest)
    final_pass = {
        **freeze_manifest,
        "status": "PASS",
        "final_gate_pass": True,
    }
    atomic_json(output / "FINAL_PASS.json", final_pass)
    atomic_json(
        output / "oracle_provenance.json",
        {
            key: final_pass[key]
            for key in (
                "dataset",
                "stage",
                "status",
                "oracle_backend",
                "classifier_type",
                "rf_oracle_used",
                "source_label",
                "num_classes",
                "policy_checkpoint_hash",
                "oracle_checkpoint_hash",
            )
        },
    )
    atomic_marker(output / "PASS", "PASS")
    return final_pass


__all__ = [
    "compute_frozen_test_metrics",
    "finalize_b13_output",
    "run_b12_selector",
    "run_b14_manifest_freeze",
]
