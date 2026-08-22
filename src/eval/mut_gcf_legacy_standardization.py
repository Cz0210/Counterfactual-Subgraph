"""Deterministically publish the frozen Mutagenicity GCF cell schema.

The upstream held-out route already computes and audits the complete WNode
matrix.  This module never opens a raw split, model, or candidate generator. It
only verifies that frozen export, selects the preregistered K=10 Figure-4 rows,
normalizes the public method identity, and publishes the common four-by-four
artifact closure in a fresh directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from src.eval.am_legacy_standardization import scan_live_writers


DATASET = "Mutagenicity"
METHOD = "GCFExplainer"
RAW_METHOD = "GCFExplainer-Top20"
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
CF_MODE = "strict_flip"
PARENT_COUNT = 217
CANDIDATE_COUNT = 20
PAIR_COUNT = 4340
TABLE2_K = 10
THRESHOLD_COUNT = 601
THETA_STAR = 0.05
COST_CAP = 0.0535
PASS_MARKER = "[MUT_GCF_LEGACY_STANDARDIZATION_PASS]"
HEX64 = re.compile(r"[0-9a-f]{64}")


class MutGcfStandardizationError(ValueError):
    """A frozen source cannot be promoted without weakening provenance."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MutGcfStandardizationError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise MutGcfStandardizationError(f"JSON artifact is not an object: {path}")
    return dict(payload)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if HEX64.fullmatch(text) is None:
        raise MutGcfStandardizationError(f"{label} is not one SHA256")
    return text


def _bool_false(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise MutGcfStandardizationError(f"Frozen source does not prove {key}=false")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as exc:
        raise MutGcfStandardizationError(f"Invalid CSV artifact: {path}") from exc
    if not fields or not rows:
        raise MutGcfStandardizationError(f"CSV artifact is empty: {path}")
    return fields, rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _normalized_rows(
    fields: Sequence[str], rows: Sequence[Mapping[str, Any]], *, require_dataset: bool = False
) -> tuple[list[str], list[dict[str, Any]]]:
    output_fields = list(fields)
    if "method" not in output_fields:
        output_fields.append("method")
    if require_dataset and "dataset" not in output_fields:
        output_fields.append("dataset")
    normalized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        observed = str(row.get("method") or RAW_METHOD).strip()
        if re.sub(r"[^a-z0-9]+", "", observed.lower()) not in {
            "gcfexplainer",
            "gcfexplainertop20",
        }:
            raise MutGcfStandardizationError(
                f"Unexpected frozen GCF method identity: {observed!r}"
            )
        row["method"] = METHOD
        if require_dataset:
            observed_dataset = str(row.get("dataset") or DATASET).strip().lower()
            if observed_dataset != DATASET.lower():
                raise MutGcfStandardizationError(
                    f"Unexpected frozen dataset identity: {observed_dataset!r}"
                )
            row["dataset"] = DATASET
        normalized.append(row)
    return output_fields, normalized


def _validate_source(source: Path, frozen: Path, *, proc_root: Path) -> dict[str, Any]:
    required = (
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "table2_gcfexplainer_k10.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "summary.json",
        "run_manifest.json",
        "final_artifact_audit.json",
        "artifact_manifest.json",
        "_FINALIZED.json",
        "_RUN_COMPLETE.json",
    )
    missing = [
        name
        for name in required
        if not (source / name).is_file()
        or (source / name).is_symlink()
        or (source / name).stat().st_size <= 0
    ]
    if missing:
        raise MutGcfStandardizationError(f"Frozen GCF final export is incomplete: {missing}")
    live_writer = scan_live_writers(source, proc_root=proc_root)
    manifest = _read_object(source / "run_manifest.json")
    summary = _read_object(source / "summary.json")
    audit = _read_object(source / "final_artifact_audit.json")
    complete = _read_object(source / "_RUN_COMPLETE.json")
    finalized = _read_object(source / "_FINALIZED.json")
    inventory = _read_object(source / "artifact_manifest.json")
    for payload, label in ((summary, "summary"),):
        if str(payload.get("dataset") or "") != DATASET:
            raise MutGcfStandardizationError(f"{label} dataset differs")
        if str(payload.get("method") or "") != RAW_METHOD:
            raise MutGcfStandardizationError(f"{label} native method differs")
    expected_summary = {
        "source_label": 1,
        "target_label": 0,
        "test_parent_count": PARENT_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "pair_count": PAIR_COUNT,
        "complete_cartesian": True,
        "run_complete": True,
        "candidate_selection_performed": False,
        "selection_used_test": False,
        "threshold_fitted_on_test": False,
        "test_used_for_selection": False,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise MutGcfStandardizationError(
                f"Frozen summary.{key}={summary.get(key)!r}; expected={expected!r}"
            )
    for key in ("selection_used_test", "threshold_fitted_on_test", "test_used_for_selection"):
        _bool_false(manifest, key)
    if complete.get("run_complete") is not True or complete.get("audit_passed") is not True:
        raise MutGcfStandardizationError("Frozen GCF completion marker did not PASS")
    if finalized.get("finalized") is not True:
        raise MutGcfStandardizationError("Frozen GCF export is not finalized")
    if audit.get("final_artifact_audit_passed") is not True:
        raise MutGcfStandardizationError("Frozen GCF final audit did not PASS")
    for key, expected in (
        ("parent_count", PARENT_COUNT),
        ("candidate_count", CANDIDATE_COUNT),
        ("pair_count", PAIR_COUNT),
        ("complete_cartesian", True),
        ("candidate_order_frozen", True),
        ("test_used_for_selection", False),
        ("threshold_fitted_on_test", False),
    ):
        if audit.get(key) != expected:
            raise MutGcfStandardizationError(f"Frozen final audit {key} differs")
    files = inventory.get("files")
    if not isinstance(files, Mapping) or not files:
        raise MutGcfStandardizationError("Frozen artifact hash inventory is missing")
    for relative, claimed in files.items():
        path = source / str(relative)
        if not path.is_file() or path.is_symlink() or _sha(path) != _hex(
            claimed, label=f"source artifact {relative}"
        ):
            raise MutGcfStandardizationError(f"Frozen artifact hash differs: {relative}")
    if _sha(source / "artifact_manifest.json") != _hex(
        finalized.get("artifact_manifest_sha256"), label="artifact manifest identity"
    ):
        raise MutGcfStandardizationError("Frozen artifact manifest hash differs")

    test_audit = manifest.get("test_evaluation_audit")
    if not isinstance(test_audit, Mapping):
        raise MutGcfStandardizationError("Frozen run lacks held-out audit provenance")
    expected_test = {
        "audit_passed": True,
        "cohort": "test",
        "parent_count": PARENT_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "pair_count": PAIR_COUNT,
        "complete_cartesian": True,
        "strict_flip": True,
        "distance_line": DISTANCE_LINE,
        "candidate_selection_performed": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
    }
    for key, expected in expected_test.items():
        if test_audit.get(key) != expected:
            raise MutGcfStandardizationError(f"Held-out audit {key} differs")
    dataset_hash = _hex(test_audit.get("dataset_csv_sha256"), label="dataset hash")
    split_hash = _hex(test_audit.get("test_cohort_hash"), label="test cohort hash")
    oracle_hash = _hex(test_audit.get("teacher_sha256"), label="RF oracle hash")
    molclr_hash = _hex(
        test_audit.get("molclr_checkpoint_sha256"), label="MolCLR checkpoint hash"
    )
    teacher = manifest.get("teacher")
    molclr = manifest.get("molclr_checkpoint")
    if not isinstance(teacher, Mapping) or not isinstance(molclr, Mapping):
        raise MutGcfStandardizationError("Frozen model identities are missing")
    if _hex(teacher.get("sha256"), label="manifest RF oracle hash") != oracle_hash:
        raise MutGcfStandardizationError("RF oracle identity differs across audits")
    if _hex(molclr.get("sha256"), label="manifest MolCLR hash") != molclr_hash:
        raise MutGcfStandardizationError("MolCLR identity differs across audits")

    threshold_path = frozen / "matched_thresholds.json"
    threshold = _read_object(threshold_path)
    values = threshold.get("thresholds")
    if not isinstance(values, list) or len(values) != THRESHOLD_COUNT:
        raise MutGcfStandardizationError("Matched threshold grid must contain 601 points")
    parsed = [float(value) for value in values]
    expected_grid = [COST_CAP * index / (THRESHOLD_COUNT - 1) for index in range(THRESHOLD_COUNT)]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(parsed, expected_grid)
    ):
        raise MutGcfStandardizationError("Matched threshold grid differs")
    expected_threshold = {
        "status": "PASS",
        "dataset": DATASET,
        "distance_line": DISTANCE_LINE,
        "cf_mode": CF_MODE,
        "test_used_for_selection": False,
    }
    for key, expected in expected_threshold.items():
        if threshold.get(key) != expected:
            raise MutGcfStandardizationError(f"Matched threshold {key} differs")
    if not math.isclose(float(threshold.get("theta_star")), THETA_STAR, abs_tol=1e-12):
        raise MutGcfStandardizationError("Matched theta_star differs")
    if not math.isclose(float(threshold.get("cost_cap")), COST_CAP, abs_tol=1e-12):
        raise MutGcfStandardizationError("Matched cost cap differs")
    threshold_hash = _hex(
        threshold.get("threshold_config_hash"), label="threshold config hash"
    )
    provenance = manifest.get("threshold_provenance")
    if not isinstance(provenance, Mapping):
        raise MutGcfStandardizationError("Frozen threshold provenance is missing")
    source_threshold = frozen / "schema_reference/thresholds.json"
    if not source_threshold.is_file() or _sha(source_threshold) != _hex(
        provenance.get("ours_thresholds_json_sha256"),
        label="source threshold artifact hash",
    ):
        raise MutGcfStandardizationError("Frozen threshold artifact hash differs")
    if _read_object(source_threshold) != threshold:
        raise MutGcfStandardizationError("Frozen threshold copies are not identical")
    return {
        "source_live_writer_audit": live_writer,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "oracle_hash": oracle_hash,
        "oracle_checkpoint": str(teacher.get("path") or ""),
        "molclr_hash": molclr_hash,
        "molclr_checkpoint": str(molclr.get("path") or ""),
        "threshold_hash": threshold_hash,
        "thresholds": parsed,
        "source_manifest_sha256": _sha(source / "run_manifest.json"),
        "source_final_audit_sha256": _sha(source / "final_artifact_audit.json"),
        "source_artifact_manifest_sha256": _sha(source / "artifact_manifest.json"),
        "source_threshold_artifact_sha256": _sha(source_threshold),
    }


def standardize_mut_gcf_legacy_cell(
    *,
    heldout_root: str | Path,
    frozen_root: str | Path,
    output_dir: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    heldout = Path(heldout_root).expanduser().resolve(strict=True)
    source = (heldout / "final").resolve(strict=True)
    frozen = Path(frozen_root).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Standardized output must be fresh: {destination}")
    provenance = _validate_source(
        source, frozen, proc_root=Path(proc_root).expanduser().resolve()
    )

    figure3_fields, figure3_rows = _read_csv(source / "figure3_coverage_vs_k.csv")
    figure3_fields, figure3_rows = _normalized_rows(figure3_fields, figure3_rows)
    if [int(row.get("k") or -1) for row in figure3_rows] != list(range(1, 21)):
        raise MutGcfStandardizationError("Frozen Figure 3 is not K=1..20")
    coverages = [float(row.get("close_cf_coverage") or row.get("coverage")) for row in figure3_rows]
    if any(right + 1e-12 < left for left, right in zip(coverages, coverages[1:])):
        raise MutGcfStandardizationError("Frozen Figure 3 coverage decreases")

    figure4_fields, all_figure4 = _read_csv(source / "figure4_coverage_vs_threshold.csv")
    figure4_fields, all_figure4 = _normalized_rows(figure4_fields, all_figure4)
    figure4_rows = [row for row in all_figure4 if int(row.get("k") or -1) == TABLE2_K]
    if len(figure4_rows) != THRESHOLD_COUNT:
        raise MutGcfStandardizationError("Frozen Figure 4 lacks the complete K=10 grid")
    observed_thresholds = [float(row["threshold"]) for row in figure4_rows]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(observed_thresholds, provenance["thresholds"])
    ):
        raise MutGcfStandardizationError("Frozen Figure 4 threshold order differs")
    figure4_coverages = [
        float(row.get("close_cf_coverage") or row.get("coverage"))
        for row in figure4_rows
    ]
    if any(
        right + 1e-12 < left
        for left, right in zip(figure4_coverages, figure4_coverages[1:])
    ):
        raise MutGcfStandardizationError("Frozen Figure 4 coverage decreases")

    table_fields, table_rows = _read_csv(source / "table2_gcfexplainer_k10.csv")
    table_fields, table_rows = _normalized_rows(
        table_fields, table_rows, require_dataset=True
    )
    if len(table_rows) != 1 or int(table_rows[0].get("k") or -1) != TABLE2_K:
        raise MutGcfStandardizationError("Frozen Table 2 is not exactly K=10")
    prefix_fields, prefix_rows = _read_csv(source / "prefix_metrics.csv")
    prefix_fields, prefix_rows = _normalized_rows(prefix_fields, prefix_rows)
    if [int(row.get("k") or -1) for row in prefix_rows] != list(range(1, 21)):
        raise MutGcfStandardizationError("Frozen prefix metrics are not K=1..20")
    parent_fields, parent_rows = _read_csv(source / "parent_best_distances.csv")
    parent_fields, parent_rows = _normalized_rows(parent_fields, parent_rows)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_csv(temporary / "figure3_coverage_vs_k.csv", figure3_fields, figure3_rows)
        _write_csv(temporary / "figure4_coverage_vs_threshold.csv", figure4_fields, figure4_rows)
        _write_csv(temporary / "table2_gcfexplainer_k10.csv", table_fields, table_rows)
        _write_csv(temporary / "prefix_metrics.csv", prefix_fields, prefix_rows)
        _atomic_json(
            temporary / "prefix_metrics.json",
            {
                "schema_version": "four_by_four_prefix_metrics_v1",
                "dataset": DATASET,
                "method": METHOD,
                "k_max": CANDIDATE_COUNT,
                "metrics": prefix_rows,
            },
        )
        _write_csv(temporary / "parent_best_distances.csv", parent_fields, parent_rows)
        _write_csv(
            temporary / "destination_distribution.csv",
            ("dataset", "method", "destination_label", "count", "rate", "reason"),
            (
                {
                    "dataset": DATASET,
                    "method": METHOD,
                    "destination_label": "N/A",
                    "count": "N/A",
                    "rate": "N/A",
                    "reason": "binary_1_to_0_task_destination_distribution_not_applicable",
                },
            ),
        )
        common = {
            "schema_version": "four_by_four_standardized_cell_v1",
            "dataset": DATASET,
            "method": METHOD,
            "status": "FROZEN_PASS",
            "frozen": True,
            "artifacts_frozen": True,
            "raw_output_root": str(source),
            "standardized_output_root": str(destination),
            "raw_output_complete": True,
            "generation_adopted": True,
            "generation_rerun": False,
            "distance_recomputed": False,
            "oracle_recomputed": False,
            "candidate_order_changed": False,
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "oracle_checkpoint": provenance["oracle_checkpoint"],
            "oracle_hash": provenance["oracle_hash"],
            "rf_oracle_used": True,
            "dataset_hash": provenance["dataset_hash"],
            "split_hash": provenance["split_hash"],
            "test_cohort_hash": provenance["split_hash"],
            "distance_line": DISTANCE_LINE,
            "molclr_checkpoint": provenance["molclr_checkpoint"],
            "molclr_checkpoint_hash": provenance["molclr_hash"],
            "cf_mode": CF_MODE,
            "source_label": 1,
            "target_label": 0,
            "k_max": CANDIDATE_COUNT,
            "table2_k": TABLE2_K,
            "theta_star": THETA_STAR,
            "cost_cap": COST_CAP,
            "threshold_config_hash": provenance["threshold_hash"],
            "threshold_source": "frozen_calibration_matched_existing_protocol",
            "threshold_source_split": "frozen_calibration",
            "selector_fitted_on_calibration": True,
            "selector_frozen_before_test": True,
            "selector_parameters_frozen": True,
            "test_loaded_only_after_freeze": True,
            "test_used_only_after_freeze": True,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "test_candidate_selection": False,
            "test_variant_selection": False,
            "raw_test_opened_by_standardizer": False,
            "paper_written": False,
        }
        summary = {
            **common,
            "schema_version": "mut_gcf_legacy_standardized_summary_v1",
            "test_parent_count": PARENT_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "pair_count": PAIR_COUNT,
            "complete_cartesian": True,
            "figure3_rows": CANDIDATE_COUNT,
            "figure4_rows": THRESHOLD_COUNT,
            "table2_rows": 1,
            "run_complete": True,
        }
        oracle_manifest = {
            **common,
            "schema_version": "four_by_four_oracle_manifest_v1",
            "retrained": False,
        }
        evaluation_manifest = {
            **common,
            "schema_version": "four_by_four_evaluation_manifest_v1",
            "distance_encoder_hash": provenance["molclr_hash"],
            "parent_count": PARENT_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "pair_count": PAIR_COUNT,
            "complete_cartesian": True,
            "threshold_count": THRESHOLD_COUNT,
        }
        run_manifest = {
            **common,
            "schema_version": "mut_gcf_legacy_standardization_run_v1",
            "standardization_mode": "deterministic_frozen_artifact_replay",
            "source_heldout_root": str(heldout),
            "source_frozen_root": str(frozen),
            "source_run_manifest_sha256": provenance["source_manifest_sha256"],
            "source_final_audit_sha256": provenance["source_final_audit_sha256"],
            "source_artifact_manifest_sha256": provenance[
                "source_artifact_manifest_sha256"
            ],
            "source_threshold_artifact_sha256": provenance[
                "source_threshold_artifact_sha256"
            ],
            "source_live_writer_audit": provenance["source_live_writer_audit"],
            "figure4_source_k_values": [10, 20],
            "figure4_export_k": TABLE2_K,
            "run_complete": True,
        }
        _atomic_json(temporary / "summary.json", summary)
        _atomic_json(temporary / "oracle_manifest.json", oracle_manifest)
        _atomic_json(temporary / "evaluation_manifest.json", evaluation_manifest)
        _atomic_json(temporary / "run_manifest.json", run_manifest)
        declared_names = (
            "figure3_coverage_vs_k.csv",
            "figure4_coverage_vs_threshold.csv",
            "table2_gcfexplainer_k10.csv",
            "prefix_metrics.csv",
            "prefix_metrics.json",
            "parent_best_distances.csv",
            "destination_distribution.csv",
            "summary.json",
            "run_manifest.json",
            "oracle_manifest.json",
            "evaluation_manifest.json",
        )
        hashes = {name: _sha(temporary / name) for name in declared_names}
        artifact_manifest = {
            "schema_version": "four_by_four_artifact_manifest_v1",
            "dataset": DATASET,
            "method": METHOD,
            "file_sha256": hashes,
            "file_count": len(hashes),
            "self_excluded": "artifact_manifest.json",
        }
        _atomic_json(temporary / "artifact_manifest.json", artifact_manifest)
        final_audit = {
            **common,
            "schema_version": "four_by_four_final_artifact_audit_v1",
            "passed": True,
            "audit_passed": True,
            "final_artifact_audit_passed": True,
            "file_sha256": hashes,
            "source_checksum_closure_passed": True,
            "source_read_only": True,
            "source_unchanged": True,
            "live_writer_audit_passed": True,
            "complete_cartesian": True,
            "parent_count": PARENT_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "pair_count": PAIR_COUNT,
            "figure3_k_grid": list(range(1, 21)),
            "figure4_k": TABLE2_K,
            "figure4_threshold_count": THRESHOLD_COUNT,
            "table2_k": TABLE2_K,
            "aggregation_only": True,
            "candidate_order_changed": False,
        }
        _atomic_json(temporary / "final_artifact_audit.json", final_audit)
        freeze_names = (*declared_names, "artifact_manifest.json", "final_artifact_audit.json")
        freeze_files = {
            name: {"bytes": (temporary / name).stat().st_size, "sha256": _sha(temporary / name)}
            for name in freeze_names
        }
        _atomic_json(
            temporary / "freeze_manifest.json",
            {
                **common,
                "schema_version": "four_by_four_freeze_manifest_v1",
                "finalized": True,
                "gate_passed": True,
                "files": freeze_files,
                "file_count": len(freeze_files),
            },
        )
        _atomic_json(
            temporary / "_FINALIZED.json",
            {
                **common,
                "finalized": True,
                "gate_passed": True,
                "freeze_manifest_sha256": _sha(temporary / "freeze_manifest.json"),
            },
        )
        (temporary / "PASS").write_text("PASS\n", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "FROZEN_PASS",
        "dataset": DATASET,
        "method": METHOD,
        "output_root": str(destination),
        "source_heldout_root": str(heldout),
        "raw_test_opened": False,
        "distance_recomputed": False,
        "oracle_recomputed": False,
        "candidate_order_changed": False,
    }


__all__ = [
    "MutGcfStandardizationError",
    "PASS_MARKER",
    "standardize_mut_gcf_legacy_cell",
]
