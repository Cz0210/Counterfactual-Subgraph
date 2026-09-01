"""Strict one-cell matrix publication for the remaining non-Taste routes.

This module is intentionally a small campaign-specific append operation.  It
does not run science, standardize a result, or infer a terminal from a marker.
Only the completed AIDS/ComRecGC recovery controller, either explicit
Mutagenicity/ComRecGC production terminal, and the two BACE frozen
standardization contracts are accepted.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import shutil
from typing import Any, Mapping

from scripts.autodl.append_bace_gcf_matrix_authority import (
    _git_identity,
    _json_bytes,
    _read_json,
    _verify_authority,
)
from scripts.autodl.run_comrecgc_standardized_continuation import (
    _verify_adopted_generation_integrity as verify_mut_adopted_generation_integrity,
    _validate_common_recourse_completion,
)
from scripts.autodl.run_mut_comrecgc_parity_standardization import (
    FAST_ACCURATE_RUN_SCHEMA as MUT_FAST_ACCURATE_RUN_SCHEMA,
    _validate_common_adoption as validate_mut_parity_common_adoption,
    _validate_historical_adoption as validate_mut_historical_adoption,
    _validate_parity as validate_mut_parity_standardization,
)
from src.baselines.comrecgc.contracts import (
    UPSTREAM_COMMIT as COMRECGC_UPSTREAM_COMMIT,
)
from src.eval.am_legacy_standardization import scan_live_writers
from src.eval.aids_comrecgc_terminal_reconciliation import (
    RECONCILIATION_SCHEMA as AIDS_RECONCILIATION_SCHEMA,
    science_terminal_projection as aids_science_terminal_projection,
    validate_missing_controller_terminal as validate_aids_missing_controller_terminal,
    validate_reconciliation_root as validate_aids_reconciliation_root,
)
from src.eval.bace_frozen_cell_standardization import (
    BACECellStandardizationError,
    _checkpoint_contract,
    _load_frozen_inputs,
    _split_contract,
    _threshold_contract,
    sha256_file,
)
from src.eval.four_by_four_registry import (
    AuditConfig,
    CellStatus,
    DATASETS,
    METHODS,
    PASS_STATUSES,
    audit_registry,
    stable_json_sha256,
    write_registry_outputs,
)
from src.eval.user_approved_frozen_v4 import APPROVAL_ID as FROZEN_V4_APPROVAL_ID
from src.train.molecular_gnn_resume import atomic_rename_directory_noreplace
from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (
    EXACT_STAGE,
    FINAL_STAGE,
    FINAL_STAGE_RECEIPT_SCHEMA,
    TERMINAL_SCHEMA as AIDS_CONTROLLER_TERMINAL_SCHEMA,
    load_bound_controller_manifest,
    validate_controller_terminal,
    validate_stage_terminal,
)
from src.utils.autodl_mut_comrecgc_exact_postprocess_v1 import (
    EXPECTED_COMMON_RECOURSES as MUT_EXPECTED_COMMON_RECOURSES,
    MATRIX_APPEND_SCHEMA as MUT_MATRIX_APPEND_SCHEMA,
    RUN_SCHEMA as MUT_RUN_SCHEMA,
    SOURCE_CANDIDATE_COUNT as MUT_SOURCE_CANDIDATE_COUNT,
    SOURCE_PAYLOAD_SHA256 as MUT_SOURCE_PAYLOAD_SHA256,
    _reopen_existing_matrix_append as _reopen_mut_matrix_append,
    validate_exact_adoption as validate_mut_exact_adoption,
)
from src.utils.autodl_mut_traceoff_parity_v1 import (
    SOURCE_PROJECT_COMMIT,
    _validate_parity_gate as validate_mut_trace_parity,
)


APPEND_SCHEMA = "non_taste_matrix_authority_append_v1"
TARGETS = {
    ("AIDS", "ComRecGC"),
    ("Mutagenicity", "ComRecGC"),
    ("BACE", "GlobalGCE"),
    ("BACE", "ComRecGC"),
}
PASS_BYTES = b"PASS\n"
MAX_HASH_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_AIDS_ZERO_CONDITIONAL_COST_REGISTRY_REASONS = frozenset(
    {
        "FIGURE3_INVALID:ValueError",
        "TABLE2_INVALID:ValueError",
    }
)
_BACE_SHARED_FIELDS = (
    "dataset_hash",
    "split_hash",
    "oracle_backend",
    "oracle_checkpoint",
    "oracle_hash",
    "molclr_checkpoint_hash",
    "distance_line",
    "cf_mode",
    "threshold_config_hash",
)
_RF_SHARED_FIELDS = _BACE_SHARED_FIELDS
_MUT_PARITY_RUN_SCHEMA = "mut_comrecgc_parity_standardization_v1"
_COMRECGC_REQUIRED_SOURCE_FILES = (
    "comrecgc.py",
    "common_recourse.py",
    "data.py",
    "gnn.py",
)


class NonTasteMatrixAppendError(RuntimeError):
    """A proposed non-Taste cell is not a strict terminal append."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _physical_directory(path_like: str | Path, *, label: str) -> Path:
    logical = Path(path_like).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise NonTasteMatrixAppendError(f"{label} must be an absolute physical directory")
    try:
        root = logical.resolve(strict=True)
    except OSError as exc:
        raise NonTasteMatrixAppendError(f"{label} is absent: {logical}") from exc
    if not root.is_dir():
        raise NonTasteMatrixAppendError(f"{label} is not a directory: {root}")
    return root


def _physical_file(path_like: str | Path, *, label: str, allow_empty: bool = False) -> Path:
    logical = Path(path_like).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise NonTasteMatrixAppendError(f"{label} must be an absolute physical file")
    try:
        path = logical.resolve(strict=True)
    except OSError as exc:
        raise NonTasteMatrixAppendError(f"{label} is absent: {logical}") from exc
    if not path.is_file() or (not allow_empty and path.stat().st_size <= 0):
        raise NonTasteMatrixAppendError(f"{label} is not a nonempty file: {path}")
    return path


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NonTasteMatrixAppendError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise NonTasteMatrixAppendError(f"{label} must be a JSON object: {path}")
    return dict(value)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _valid_sha(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise NonTasteMatrixAppendError(f"{label} is not one SHA256")
    return text


def _require_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    changed = [key for key, value in expected.items() if payload.get(key) != value]
    if changed:
        raise NonTasteMatrixAppendError(
            f"{label} terminal contract changed: {', '.join(changed)}"
        )


def _verify_inventory(root: Path, raw: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping) or not raw:
        raise NonTasteMatrixAppendError(f"{label} file inventory is absent")
    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_identity in sorted(raw.items()):
        name = str(raw_name)
        relative = Path(name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise NonTasteMatrixAppendError(f"{label} has an unsafe member: {name!r}")
        if not isinstance(raw_identity, Mapping):
            raise NonTasteMatrixAppendError(f"{label} identity is malformed: {name}")
        path = root / relative
        allow_empty = name in {
            "pair_matrix.jsonl",
            "selected_sequence.jsonl",
            "representative_counterfactuals.jsonl",
        }
        _physical_file(path, label=f"{label}[{name}]", allow_empty=allow_empty)
        try:
            expected_bytes = int(raw_identity.get("bytes", -1))
        except (TypeError, ValueError):
            expected_bytes = -1
        expected_sha = _valid_sha(raw_identity.get("sha256"), label=f"{label}[{name}].sha256")
        if path.stat().st_size != expected_bytes or _sha(path) != expected_sha:
            raise NonTasteMatrixAppendError(f"{label} member drifted: {name}")
        result[name] = dict(raw_identity)
    return result


def _writer_audit(root: Path, *, proc_root: str | Path, required: bool) -> dict[str, Any]:
    if not required:
        return {
            "procfs_verified": False,
            "scanned_process_count": 0,
            "writable_fd_count": 0,
            "writers": [],
        }
    result = scan_live_writers(root, proc_root=proc_root)
    if (
        result.get("procfs_verified") is not True
        or result.get("writable_fd_count") != 0
        or result.get("writers") != []
    ):
        raise NonTasteMatrixAppendError(f"Terminal root still has a live writer: {root}")
    return result


def _critical_inventory(root: Path, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = _physical_file(root / name, label=f"terminal artifact {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    return result


def _validate_bace_terminal(
    root_like: str | Path,
    *,
    method: str,
    proc_root: str | Path,
    require_writer_audit: bool,
) -> dict[str, Any]:
    root = _physical_directory(root_like, label=f"BACE/{method} standardized root")
    if any((root / name).exists() for name in ("FAILED", "FAILED.json", "FAIL.json")):
        raise NonTasteMatrixAppendError("BACE standardized root contains a failure sentinel")
    if _physical_file(root / "PASS", label="BACE PASS").read_bytes() != PASS_BYTES:
        raise NonTasteMatrixAppendError("BACE standardized PASS bytes changed")
    payloads = {
        name: _json(_physical_file(root / name, label=name), label=name)
        for name in (
            "summary.json",
            "run_manifest.json",
            "oracle_manifest.json",
            "evaluation_manifest.json",
            "artifact_manifest.json",
            "freeze_manifest.json",
            "_FINALIZED.json",
            "final_artifact_audit.json",
        )
    }
    run = payloads["run_manifest.json"]
    summary = payloads["summary.json"]
    oracle = payloads["oracle_manifest.json"]
    evaluation = payloads["evaluation_manifest.json"]
    artifact = payloads["artifact_manifest.json"]
    freeze = payloads["freeze_manifest.json"]
    finalized = payloads["_FINALIZED.json"]
    audit = payloads["final_artifact_audit.json"]
    schemas = {
        "summary.json": "bace_frozen_cell_standardization_v1",
        "run_manifest.json": "bace_frozen_cell_standardization_v1",
        "oracle_manifest.json": "bace_frozen_cell_oracle_manifest_v1",
        "evaluation_manifest.json": "bace_frozen_cell_evaluation_manifest_v1",
        "artifact_manifest.json": "bace_frozen_cell_artifact_manifest_v1",
        "freeze_manifest.json": "bace_frozen_cell_freeze_manifest_v1",
        "_FINALIZED.json": "bace_frozen_cell_finalized_v1",
        "final_artifact_audit.json": "bace_frozen_cell_final_artifact_audit_v1",
    }
    for name, schema in schemas.items():
        if payloads[name].get("schema_version") != schema:
            raise NonTasteMatrixAppendError(f"BACE terminal schema changed: {name}")
    common = {
        "dataset": "BACE",
        "method": method,
        "stage": "BACE_FROZEN_CELL_STANDARDIZATION",
        "status": "PASS",
        "frozen": True,
        "finalized": True,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "source_label": 1,
        "num_classes": 2,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "selector_fitted_on_calibration": True,
        "selection_frozen_before_test": True,
        "test_loaded_only_after_freeze": True,
        "test_used_only_after_freeze": True,
        "raw_test_opened": False,
        "candidate_order_changed": False,
        "selector_refit": False,
        "threshold_refit": False,
        "raw_output_complete": True,
        "source_artifacts_complete": True,
        "k_max": 20,
    }
    for name in (
        "summary.json",
        "run_manifest.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "freeze_manifest.json",
        "final_artifact_audit.json",
    ):
        _require_fields(payloads[name], common, label=name)
    try:
        effective_rules = int(run.get("effective_rule_count", -1))
    except (TypeError, ValueError) as exc:
        raise NonTasteMatrixAppendError("BACE effective rule count is invalid") from exc
    if not 10 <= effective_rules <= 20:
        raise NonTasteMatrixAppendError("BACE terminal has fewer than 10 or more than 20 rules")
    if summary.get("table2_k") != 10 or summary.get("effective_rule_count") != effective_rules:
        raise NonTasteMatrixAppendError("BACE summary K/effective-rule contract changed")
    _require_fields(
        finalized,
        {
            "dataset": "BACE",
            "method": method,
            "status": "PASS",
            "finalized": True,
            "gate_passed": True,
            "frozen": True,
            "raw_test_opened": False,
        },
        label="_FINALIZED.json",
    )
    _require_fields(
        audit,
        {
            "passed": True,
            "audit_passed": True,
            "final_artifact_audit_passed": True,
            "all_required_files_nonempty": True,
            "hash_closure_complete": True,
            "no_numeric_imputation": True,
            "n_a_fields_have_explicit_reason": True,
        },
        label="final_artifact_audit.json",
    )
    run_files = _verify_inventory(root, run.get("files"), label="BACE run manifest")
    if freeze.get("files") != run_files or audit.get("files") != run_files:
        raise NonTasteMatrixAppendError("BACE run/freeze/audit inventories differ")
    artifact_files = _verify_inventory(
        root, artifact.get("files"), label="BACE artifact manifest"
    )
    expected_artifact = {
        **run_files,
        "run_manifest.json": {
            "bytes": (root / "run_manifest.json").stat().st_size,
            "sha256": _sha(root / "run_manifest.json"),
        },
    }
    if artifact_files != expected_artifact:
        raise NonTasteMatrixAppendError("BACE artifact manifest closure changed")
    if (
        freeze.get("run_manifest_sha256") != _sha(root / "run_manifest.json")
        or freeze.get("artifact_manifest_sha256") != _sha(root / "artifact_manifest.json")
        or finalized.get("freeze_manifest_sha256") != _sha(root / "freeze_manifest.json")
        or audit.get("artifact_manifest_sha256") != _sha(root / "artifact_manifest.json")
        or audit.get("freeze_manifest_sha256") != _sha(root / "freeze_manifest.json")
        or audit.get("finalized_marker_sha256") != _sha(root / "_FINALIZED.json")
    ):
        raise NonTasteMatrixAppendError("BACE terminal hash chain changed")

    source = _physical_directory(run.get("raw_output_root", ""), label="BACE source final root")
    checkpoint = _checkpoint_contract(run.get("oracle_checkpoint", ""))
    try:
        inputs = _load_frozen_inputs(
            method_slug=method.lower(),
            source_root=source,
            checkpoint_id=str(checkpoint["checkpoint_id"]),
        )
        split = _split_contract(inputs, checkpoint)
        thresholds = _threshold_contract(inputs.selection_manifest)
    except BACECellStandardizationError as exc:
        raise NonTasteMatrixAppendError(f"BACE frozen source replay failed: {exc}") from exc
    identities = {
        "oracle_checkpoint_hash": str(checkpoint["checkpoint_id"]),
        "dataset_hash": str(checkpoint["dataset_hash"]),
        "split_hash": str(split["test_split_hash"]),
        "molclr_checkpoint_hash": str(inputs.selection_manifest["molclr_checkpoint_hash"]),
        "threshold_config_hash": str(thresholds["threshold_config_hash"]),
    }
    if any(str(run.get(key) or "") != value for key, value in identities.items()):
        raise NonTasteMatrixAppendError("BACE standardized identities differ from frozen source")
    if (
        oracle.get("temperature_scaling_sha256") != checkpoint["temperature_hash"]
        or oracle.get("feature_schema_sha256") != checkpoint["feature_schema_hash"]
        or evaluation.get("test_parent_set_digest") != split["test_parent_set_sha256"]
        or evaluation.get("scientific_metrics_recomputed") is not False
        or evaluation.get("deterministic_aggregation_replayed") is not True
    ):
        raise NonTasteMatrixAppendError("BACE oracle/split replay evidence changed")
    expected_sources = {
        "source_final_manifest": inputs.final_manifest_path,
        "source_selection_manifest": inputs.selection_manifest_path,
        "source_test_manifest": inputs.test_manifest_path,
        "source_pair_matrix": inputs.pair_matrix_path,
        "source_final_metrics": inputs.final_metrics_path,
    }
    for field, expected_path in expected_sources.items():
        identity = evaluation.get(field)
        if (
            not isinstance(identity, Mapping)
            or Path(str(identity.get("path") or "")).resolve(strict=True) != expected_path
            or identity.get("sha256") != _sha(expected_path)
        ):
            raise NonTasteMatrixAppendError(f"BACE evaluation source binding changed: {field}")
    writer = _writer_audit(root, proc_root=proc_root, required=require_writer_audit)
    source_writer = _writer_audit(
        source, proc_root=proc_root, required=require_writer_audit
    )
    return {
        "terminal_kind": "BACE_FROZEN_STANDARDIZATION",
        "root": str(root),
        "source_final_root": str(source),
        "run_manifest_sha256": _sha(root / "run_manifest.json"),
        "final_artifact_audit_sha256": _sha(root / "final_artifact_audit.json"),
        "freeze_manifest_sha256": _sha(root / "freeze_manifest.json"),
        "pass_sha256": _sha(root / "PASS"),
        "effective_rule_count": effective_rules,
        "identities": identities,
        "writer_audit": writer,
        "source_writer_audit": source_writer,
        "inventory": _critical_inventory(root, tuple(payloads) + ("PASS",)),
    }


def _validate_rf_standardized(
    science_root: Path, *, dataset: str, dataset_key: str
) -> dict[str, Any]:
    label = dataset
    standardized = _physical_directory(
        science_root / "standardized", label=f"{label} standardized root"
    )
    run = _json(
        _physical_file(standardized / "run_manifest.json", label=f"{label} standardized run"),
        label=f"{label} standardized run",
    )
    summary = _json(
        _physical_file(standardized / "summary.json", label=f"{label} summary"),
        label=f"{label} summary",
    )
    audit = _json(
        _physical_file(
            standardized / "final_artifact_audit.json", label=f"{label} final audit"
        ),
        label=f"{label} final audit",
    )
    freeze = _json(
        _physical_file(standardized / "freeze_manifest.json", label=f"{label} freeze"),
        label=f"{label} freeze",
    )
    finalized = _json(
        _physical_file(standardized / "_FINALIZED.json", label=f"{label} finalized"),
        label=f"{label} finalized",
    )
    common = {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_key": dataset_key,
        "method": "COMRECGC-Adapted-DeterministicChemRepair",
        "run_complete": True,
        "mode": "full",
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
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
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
    }
    # The historical evaluator puts dataset/schema/run_complete only in the run
    # manifest.  Summary/audit still carry the leakage and ordering guards.
    _require_fields(run, common, label=f"{label} standardized run_manifest.json")
    _require_fields(
        summary,
        {
            "dataset": dataset,
            "dataset_key": dataset_key,
            "method": "COMRECGC-Adapted-DeterministicChemRepair",
            "distance_line": "MolCLR-Node-Wasserstein",
            "cf_mode": "strict_flip",
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "calibration_loaded": False,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
        },
        label=f"{label} summary.json",
    )
    _require_fields(
        audit,
        {
            "schema_version": 1,
            "audit_passed": True,
            "run_complete": True,
            "method": "COMRECGC-Adapted-DeterministicChemRepair",
            "distance_line": "MolCLR-Node-Wasserstein",
            "cf_mode": "strict_flip",
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "calibration_loaded": False,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
        },
        label=f"{label} final_artifact_audit.json",
    )
    _require_fields(
        freeze,
        {
            "schema_version": 1,
            "dataset": dataset,
            "dataset_key": dataset_key,
            "method": "COMRECGC-Adapted-DeterministicChemRepair",
            "standardized_output_root": str(standardized),
            "gate_return_code": 0,
        },
        label=f"{label} freeze_manifest.json",
    )
    _require_fields(
        finalized,
        {"finalized": True, "gate_passed": True},
        label=f"{label} _FINALIZED",
    )
    files = _verify_inventory(
        standardized, freeze.get("files"), label=f"{label} freeze"
    )
    if (
        "run_manifest.json" not in files
        or "final_artifact_audit.json" not in files
        or freeze.get("source_run_manifest_sha256") != _sha(standardized / "run_manifest.json")
        or finalized.get("freeze_manifest_sha256") != _sha(standardized / "freeze_manifest.json")
    ):
        raise NonTasteMatrixAppendError(f"{label} standardized freeze closure changed")
    source_eval = _physical_directory(
        freeze.get("source_output_root", ""), label=f"{label} source evaluation root"
    )
    gate_path = _physical_file(
        freeze.get("source_gate_result_path", ""), label=f"{label} source gate result"
    )
    gate = _json(gate_path, label=f"{label} source gate result")
    _require_fields(
        gate,
        {
            "schema_version": 1,
            "stage": f"{dataset_key}_project_full_gate",
            "status": "FULL_EXECUTION_PASS",
            "audit_passed": True,
            "run_complete": True,
            "dataset": dataset_key,
            "source_run_dir": str(source_eval),
            "source_manifest_sha256": _sha(source_eval / "run_manifest.json"),
        },
        label=f"{label} source full gate",
    )
    if (
        freeze.get("source_gate_result_sha256") != _sha(gate_path)
        or freeze.get("teacher_sha256") != run.get("teacher_sha256")
        or freeze.get("molclr_checkpoint_sha256") != run.get("molclr_checkpoint_sha256")
        or freeze.get("dataset_fingerprint") != run.get("dataset_fingerprint")
    ):
        raise NonTasteMatrixAppendError(f"{label} freeze/source identities changed")
    identities = {
        "oracle_checkpoint": str(run.get("teacher_path") or ""),
        "oracle_hash": _valid_sha(
            run.get("teacher_sha256"), label=f"{label} teacher_sha256"
        ),
        "dataset_hash": _valid_sha(
            run.get("dataset_csv_sha256"), label=f"{label} dataset_csv_sha256"
        ),
        "split_hash": _valid_sha(
            run.get("parent_ids_sha256"), label=f"{label} parent_ids_sha256"
        ),
        "molclr_checkpoint_hash": _valid_sha(
            run.get("molclr_checkpoint_sha256"), label=f"{label} MolCLR SHA256"
        ),
        "threshold_config_hash": _valid_sha(
            run.get("thresholds_sha256"), label=f"{label} thresholds SHA256"
        ),
    }
    if not Path(identities["oracle_checkpoint"]).is_absolute():
        raise NonTasteMatrixAppendError(f"{label} RF teacher path is not absolute")
    for path_field, hash_field in (
        ("teacher_path", "teacher_sha256"),
        ("dataset_csv", "dataset_csv_sha256"),
        ("molclr_checkpoint", "molclr_checkpoint_sha256"),
        ("thresholds_path", "thresholds_sha256"),
    ):
        artifact = _physical_file(run.get(path_field, ""), label=f"{label} {path_field}")
        if _sha(artifact) != run.get(hash_field):
            raise NonTasteMatrixAppendError(
                f"{label} scientific input drifted: {path_field}"
            )
    return {
        "root": str(standardized),
        "source_evaluation_root": str(source_eval),
        "run_manifest_sha256": _sha(standardized / "run_manifest.json"),
        "final_artifact_audit_sha256": _sha(standardized / "final_artifact_audit.json"),
        "freeze_manifest_sha256": _sha(standardized / "freeze_manifest.json"),
        "identities": identities,
    }


def _validate_aids_standardized(science_root: Path) -> dict[str, Any]:
    return _validate_rf_standardized(science_root, dataset="AIDS", dataset_key="aids")


def _validate_aids_science_terminal(
    root_like: str | Path,
    *,
    controller_manifest_path: str | Path | None,
    proc_root: str | Path,
    require_writer_audit: bool,
    require_controller_terminal: bool,
) -> dict[str, Any]:
    if controller_manifest_path is None:
        raise NonTasteMatrixAppendError("AIDS append requires --aids-controller-manifest")
    science_root = _physical_directory(root_like, label="AIDS final-stage science root")
    if any((science_root / name).exists() for name in ("FAILED", "FAILED.json", "FAIL.json")):
        raise NonTasteMatrixAppendError("AIDS final-stage root contains a failure sentinel")
    if _physical_file(science_root / "PASS", label="AIDS science PASS").read_bytes() != PASS_BYTES:
        raise NonTasteMatrixAppendError("AIDS final-stage PASS bytes changed")
    controller_path = _physical_file(
        controller_manifest_path, label="AIDS controller manifest"
    )
    try:
        controller = load_bound_controller_manifest(controller_path)
        controller_terminal = (
            validate_controller_terminal(controller)
            if require_controller_terminal
            else None
        )
        exact = validate_stage_terminal(controller, stage_id=EXACT_STAGE)
        final = validate_stage_terminal(controller, stage_id=FINAL_STAGE)
    except Exception as exc:
        raise NonTasteMatrixAppendError(
            f"AIDS controller/exact/final terminal reopen failed: {exc}"
        ) from exc
    if (
        require_controller_terminal
        and (
            not isinstance(controller_terminal, Mapping)
            or controller_terminal.get("schema_version")
            != AIDS_CONTROLLER_TERMINAL_SCHEMA
        )
    ):
        raise NonTasteMatrixAppendError("AIDS controller terminal schema changed")
    final_manifest = final.get("manifest")
    final_path = _physical_file(final.get("path", ""), label="AIDS final-stage receipt")
    if (
        final_path.parent != science_root
        or not isinstance(final_manifest, Mapping)
        or final_manifest.get("schema_version") != FINAL_STAGE_RECEIPT_SCHEMA
        or final_manifest.get("status") != "PASS"
        or final_manifest.get("run_complete") is not True
        or final_manifest.get("dataset") != "aids"
        or final_manifest.get("method") != "COMRECGC"
        or _json(final_path, label="AIDS final-stage receipt") != dict(final_manifest)
    ):
        raise NonTasteMatrixAppendError("Supplied AIDS root is not the controller final stage")
    exact_manifest = exact.get("stage_receipt")
    exact_path = _physical_file(exact.get("path", ""), label="AIDS exact-stage receipt")
    if (
        not isinstance(exact_manifest, Mapping)
        or exact_manifest.get("status") != "PASS"
        or exact_manifest.get("run_complete") is not True
        or exact_manifest.get("dbscan_partition_proven") is not True
        or exact_manifest.get("ordinary_pass_dependency_eligible") is not False
        or _json(exact_path, label="AIDS exact-stage receipt") != dict(exact_manifest)
    ):
        raise NonTasteMatrixAppendError("AIDS exact stage is not a proven recovery terminal")
    continuation_path = _physical_file(
        science_root / "_RUN_COMPLETE.json", label="AIDS continuation terminal"
    )
    continuation = _json(continuation_path, label="AIDS continuation terminal")
    common_path = _physical_file(
        science_root / "common_recourse/_RUN_COMPLETE.json",
        label="AIDS common-recourse terminal",
    )
    common = _json(common_path, label="AIDS common-recourse terminal")
    _validate_common_recourse_completion(marker=common_path, terminal=common)
    _require_fields(
        continuation,
        {
            "schema_version": 1,
            "status": "PASS",
            "run_complete": True,
            "dataset": "aids",
            "method": "COMRECGC",
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "rf_oracle_used": True,
            "generation_adopted": True,
            "generation_rerun": False,
            "ordering_adopted": False,
            "evaluation_adopted": False,
            "cf_mode": "strict_flip",
            "distance_line": "MolCLR-Node-Wasserstein",
            "standardized_output_root": str(science_root / "standardized"),
        },
        label="AIDS continuation terminal",
    )
    if (
        final_manifest.get("continuation_terminal_sha256") != _sha(continuation_path)
        or final_manifest.get("common_terminal_sha256") != _sha(common_path)
        or final_manifest.get("freeze_manifest_sha256")
        != _sha(science_root / "standardized/freeze_manifest.json")
    ):
        raise NonTasteMatrixAppendError("AIDS final-stage receipt hashes changed")
    standardized = _validate_aids_standardized(science_root)
    run = _json(science_root / "run_manifest.json", label="AIDS container run manifest")
    final_gate = _json(science_root / "final_gate.json", label="AIDS container final gate")
    expected_outer = dict(continuation)
    expected_outer.pop("run_complete", None)
    if run != expected_outer or final_gate != expected_outer:
        raise NonTasteMatrixAppendError("AIDS container run/final gate diverged")
    if (
        run.get("standardized_run_manifest_sha256") != standardized["run_manifest_sha256"]
        or run.get("freeze_manifest_sha256") != standardized["freeze_manifest_sha256"]
        or run.get("teacher_sha256") != standardized["identities"]["oracle_hash"]
        or run.get("molclr_checkpoint_sha256")
        != standardized["identities"]["molclr_checkpoint_hash"]
        or run.get("dataset_csv_sha256") != standardized["identities"]["dataset_hash"]
    ):
        raise NonTasteMatrixAppendError("AIDS container/standardized identities changed")
    source_generation = _physical_directory(
        run.get("source_generation_root", ""), label="AIDS frozen generation root"
    )
    source_integrity = _physical_file(
        science_root / "source_integrity_final.json",
        label="AIDS final source-integrity receipt",
    )
    if run.get("source_integrity_final_sha256") != _sha(source_integrity):
        raise NonTasteMatrixAppendError("AIDS final source-integrity binding changed")
    writer = _writer_audit(
        science_root, proc_root=proc_root, required=require_writer_audit
    )
    result = {
        "terminal_kind": (
            "AIDS_EXACT_RECOVERY_CONTROLLER_FINAL"
            if require_controller_terminal
            else "AIDS_EXACT_RECOVERY_SCIENCE_FINAL"
        ),
        "root": str(science_root),
        "controller_manifest_path": str(controller_path),
        "controller_manifest_sha256": _sha(controller_path),
        "exact_stage_receipt_path": str(exact_path),
        "exact_stage_receipt_sha256": _sha(exact_path),
        "final_stage_receipt_path": str(final_path),
        "final_stage_receipt_sha256": _sha(final_path),
        "continuation_terminal_sha256": _sha(continuation_path),
        "common_terminal_sha256": _sha(common_path),
        "standardized": standardized,
        "source_generation_root": str(source_generation),
        "source_integrity_final_sha256": _sha(source_integrity),
        "writer_audit": writer,
        "inventory": _critical_inventory(
            science_root,
            (
                "PASS",
                "run_manifest.json",
                "final_gate.json",
                "_RUN_COMPLETE.json",
                "common_recourse/_RUN_COMPLETE.json",
                "standardized/run_manifest.json",
                "standardized/final_artifact_audit.json",
                "standardized/freeze_manifest.json",
                "standardized/_FINALIZED.json",
                "source_integrity_final.json",
            ),
        ),
    }
    if require_controller_terminal:
        assert isinstance(controller_terminal, Mapping)
        result.update(
            {
                "controller_terminal_schema": AIDS_CONTROLLER_TERMINAL_SCHEMA,
                "controller_terminal_sha256": stable_json_sha256(
                    controller_terminal
                ),
            }
        )
    return result


def _validate_aids_terminal(
    root_like: str | Path,
    *,
    controller_manifest_path: str | Path | None,
    proc_root: str | Path,
    require_writer_audit: bool,
) -> dict[str, Any]:
    """Dispatch between the ordinary controller terminal and its narrow wrapper."""

    supplied = _physical_directory(root_like, label="AIDS terminal root")
    run_path = supplied / "run_manifest.json"
    schema = (
        _json(run_path, label="AIDS supplied run manifest").get("schema_version")
        if run_path.is_file() and not run_path.is_symlink()
        else None
    )
    if schema != AIDS_RECONCILIATION_SCHEMA:
        return _validate_aids_science_terminal(
            supplied,
            controller_manifest_path=controller_manifest_path,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
            require_controller_terminal=True,
        )
    try:
        receipt = validate_aids_reconciliation_root(supplied, proc_root=proc_root)
        controller = receipt["controller_terminal_reconciliation"]
        if (
            controller_manifest_path is None
            or Path(controller_manifest_path).expanduser().resolve(strict=True)
            != Path(str(controller["controller_manifest_path"])).resolve(strict=True)
        ):
            raise NonTasteMatrixAppendError(
                "AIDS reconciliation/controller manifest identity changed"
            )
        reopened_controller = validate_aids_missing_controller_terminal(
            controller_manifest_path, proc_root=proc_root
        )
        if reopened_controller != controller:
            raise NonTasteMatrixAppendError(
                "AIDS missing-controller terminal evidence changed"
            )
        science = _validate_aids_science_terminal(
            receipt["source_science_root"],
            controller_manifest_path=controller_manifest_path,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
            require_controller_terminal=False,
        )
        if aids_science_terminal_projection(science) != receipt.get(
            "science_terminal_projection"
        ):
            raise NonTasteMatrixAppendError(
                "AIDS reconciled science-terminal projection changed"
            )
    except NonTasteMatrixAppendError:
        raise
    except Exception as exc:
        raise NonTasteMatrixAppendError(
            f"AIDS terminal reconciliation reopen failed: {exc}"
        ) from exc
    science.update(
        {
            "terminal_kind": "AIDS_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION",
            "reconciliation_root": str(supplied),
            "reconciliation_sha256": receipt["reconciliation_sha256"],
            "scientific_output_empty": True,
            "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
            "coverage": 0.0,
            "conditional_cost_available": False,
            "numeric_imputation_used": False,
            "scientific_metrics_recomputed": False,
            "controller_restart_performed": False,
            "conditional_cost_unavailable_by_science": True,
            "registry_numeric_imputation_used": False,
            "reconciliation_inventory": _critical_inventory(
                supplied,
                (
                    "PASS",
                    "terminal_reconciliation.json",
                    "run_manifest.json",
                    "final_artifact_audit.json",
                ),
            ),
        }
    )
    return science


def _reconcile_aids_zero_registry_row(
    target: Mapping[str, Any], *, terminal: Mapping[str, Any]
) -> dict[str, Any]:
    """Accept only the registry's known finite-conditional-cost mismatch.

    The generic registry predates the explicit expected-empty ComRecGC
    contract and therefore tries to parse the blank conditional-cost column as
    finite before considering the separately reported fixed-capped metric.  A
    reconciliation terminal has already reopened every zero-result export and
    proved that the conditional statistic is genuinely undefined.  No other
    registry reason is waived here.
    """

    row = dict(target)
    if terminal.get("terminal_kind") != (
        "AIDS_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION"
    ):
        return row
    if (
        terminal.get("scientific_output_empty") is not True
        or terminal.get("strict_flip_status") != "STRICT_FLIP_NOT_OBSERVED"
        or terminal.get("coverage") != 0.0
        or terminal.get("conditional_cost_available") is not False
        or terminal.get("numeric_imputation_used") is not False
        or terminal.get("registry_numeric_imputation_used") is not False
    ):
        raise NonTasteMatrixAppendError(
            "AIDS reconciliation does not prove the legitimate empty-result contract"
        )
    reasons = frozenset(
        reason
        for reason in str(row.get("rerun_reason") or "").split(";")
        if reason
    )
    if reasons != _AIDS_ZERO_CONDITIONAL_COST_REGISTRY_REASONS:
        return row
    if (
        row.get("dataset") != "AIDS"
        or row.get("method") != "ComRecGC"
        or row.get("status") != CellStatus.INCOMPLETE.value
        or row.get("k_max") != 20
        or row.get("table2_k") != 10
    ):
        raise NonTasteMatrixAppendError(
            "AIDS zero-result registry mismatch is not the exact known shape"
        )
    row["status"] = CellStatus.FROZEN_PASS.value
    row["adoption_reason"] = (
        "hash-closed strict-flip result is scientifically empty; conditional cost "
        "is unavailable and no numeric value was imputed"
    )
    row["rerun_reason"] = ""
    return row


def _validate_mut_parity_source_integrity(
    generation: Mapping[str, Any],
    final_integrity: Mapping[str, Any],
    *,
    source_root: Path,
) -> dict[str, Any]:
    _require_fields(
        generation,
        {
            "schema_version": 1,
            "status": "PASS",
            "dataset": "mutagenicity",
            "generation_adopted": True,
            "generation_mode": "adopted_read_only_cache",
            "generation_rerun": False,
            "source_generation_root": str(source_root),
            "counterfactuals_sha256_claimed": MUT_SOURCE_PAYLOAD_SHA256,
            "counterfactuals_sha256_actual": MUT_SOURCE_PAYLOAD_SHA256,
            "counterfactuals_sha256_verified": True,
            "counterfactuals_sha256_computation_count": 1,
            "counterfactual_candidate_count": MUT_SOURCE_CANDIDATE_COUNT,
            "source_project_commit": SOURCE_PROJECT_COMMIT,
            "upstream_commit": COMRECGC_UPSTREAM_COMMIT,
            "serialization_rerun": False,
            "lineage_resolution_rerun": False,
        },
        label="Mut parity generation adoption",
    )
    payload = _physical_file(
        generation.get("counterfactuals_path", ""), label="Mut frozen counterfactuals"
    )
    try:
        payload.relative_to(source_root)
    except ValueError as exc:
        raise NonTasteMatrixAppendError(
            "Mut frozen counterfactuals escaped the generation root"
        ) from exc
    try:
        reopened = verify_mut_adopted_generation_integrity(generation)
    except Exception as exc:
        raise NonTasteMatrixAppendError(
            f"Mut frozen generation integrity reopen failed: {exc}"
        ) from exc
    stable_fields = (
        "schema_version",
        "status",
        "payload_sha256_recomputed",
        "payload_stat_unchanged",
        "critical_manifest_stat_and_hash_unchanged",
        "critical_manifests",
        "payload",
    )
    changed = [
        field
        for field in stable_fields
        if final_integrity.get(field) != reopened.get(field)
    ]
    if changed:
        raise NonTasteMatrixAppendError(
            "Mut final/source integrity receipt changed: " + ", ".join(changed)
        )
    return {
        "source_generation_root": str(source_root),
        "source_payload_path": str(payload),
        "source_payload_sha256": MUT_SOURCE_PAYLOAD_SHA256,
        "source_payload_sha256_recomputed": False,
        "critical_manifests": dict(reopened["critical_manifests"]),
        "payload_snapshot": dict(reopened["payload"]),
        "final_integrity_sha256": None,
        "writer_audit_before": dict(
            reopened["live_writer_audit_before_snapshot"]
        ),
        "writer_audit_after": dict(reopened["live_writer_audit_after_snapshot"]),
    }


def _validate_mut_upstream_checkout(root: Path) -> dict[str, Any]:
    audit_path = _physical_file(
        root / "upstream_checkout_audit.json", label="Mut upstream checkout audit"
    )
    audit = _json(audit_path, label="Mut upstream checkout audit")
    _require_fields(
        audit,
        {
            "expected_commit": COMRECGC_UPSTREAM_COMMIT,
            "actual_commit": COMRECGC_UPSTREAM_COMMIT,
            "commit_match": True,
            "import_pass": True,
            "network_required": False,
            "passed": True,
        },
        label="Mut upstream checkout audit",
    )
    upstream = _physical_directory(audit.get("root", ""), label="Mut upstream root")
    files = audit.get("required_files")
    if not isinstance(files, Mapping) or set(files) != set(_COMRECGC_REQUIRED_SOURCE_FILES):
        raise NonTasteMatrixAppendError("Mut upstream source inventory changed")
    for name in _COMRECGC_REQUIRED_SOURCE_FILES:
        path = _physical_file(upstream / name, label=f"Mut upstream {name}")
        if _sha(path) != _valid_sha(files.get(name), label=f"Mut upstream {name} hash"):
            raise NonTasteMatrixAppendError(f"Mut upstream source drifted: {name}")
    return {
        "root": str(upstream),
        "audit_sha256": _sha(audit_path),
        "required_files": dict(files),
    }


def _validate_mut_parity_terminal(
    root_like: str | Path,
    *,
    proc_root: str | Path,
    require_writer_audit: bool,
) -> dict[str, Any]:
    """Reopen the independent parity-v2 chemistry/evaluation/freeze terminal."""

    root = _physical_directory(root_like, label="Mut parity-standardization root")
    if any((root / name).exists() for name in ("FAILED", "FAILED.json", "FAIL.json")):
        raise NonTasteMatrixAppendError(
            "Mut parity-standardization root contains a failure sentinel"
        )
    if _physical_file(root / "PASS", label="Mut parity PASS").read_bytes() != PASS_BYTES:
        raise NonTasteMatrixAppendError("Mut parity-standardization PASS bytes changed")
    payloads = {
        name: _json(_physical_file(root / name, label=f"Mut {name}"), label=f"Mut {name}")
        for name in (
            "run_manifest.json",
            "final_gate.json",
            "_RUN_COMPLETE.json",
            "generation_adoption_manifest.json",
            "common_recourse_adoption_manifest.json",
            "trace_parity_adoption_manifest.json",
            "source_integrity_final.json",
        )
    }
    run = payloads["run_manifest.json"]
    final_gate = payloads["final_gate.json"]
    complete = payloads["_RUN_COMPLETE.json"]
    generation = payloads["generation_adoption_manifest.json"]
    common_adoption = payloads["common_recourse_adoption_manifest.json"]
    parity_adoption = payloads["trace_parity_adoption_manifest.json"]
    source_integrity_final = payloads["source_integrity_final.json"]
    if final_gate != run or complete != {**run, "run_complete": True}:
        raise NonTasteMatrixAppendError("Mut parity outer terminal manifests diverged")
    _require_fields(
        run,
        {
            "schema_version": _MUT_PARITY_RUN_SCHEMA,
            "status": "PASS",
            "dataset": "mutagenicity",
            "method": "COMRECGC",
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "rf_oracle_used": True,
            "cf_mode": "strict_flip",
            "distance_line": "MolCLR-Node-Wasserstein",
            "generation_adopted": True,
            "generation_rerun": False,
            "traceoff_reference_rerun": True,
            "trace_parity_passed": True,
            "trace_fields_stripped": False,
            "common_recourse_adopted": True,
            "common_recourse_rerun": False,
            "chemistry_rerun": True,
            "evaluation_rerun": True,
            "source_payload_sha256": MUT_SOURCE_PAYLOAD_SHA256,
            "standardized_output_root": str(root / "standardized"),
            "calibration_loaded": False,
            "test_loaded_only_in_unified_evaluation": True,
        },
        label="Mut parity run terminal",
    )
    if re.fullmatch(r"[0-9a-f]{40}", str(run.get("project_commit") or "")) is None:
        raise NonTasteMatrixAppendError("Mut parity execution commit is invalid")

    source_root = _physical_directory(
        run.get("source_generation_root", ""), label="Mut parity frozen generation root"
    )
    if generation.get("source_generation_root") != str(source_root):
        raise NonTasteMatrixAppendError("Mut parity generation root binding changed")
    parity_path = _physical_file(
        parity_adoption.get("path", ""), label="Mut parity source receipt"
    )
    try:
        reopened_parity = validate_mut_parity_standardization(
            parity_path, source_root=source_root
        )
    except Exception as exc:
        raise NonTasteMatrixAppendError(f"Mut parity receipt reopen failed: {exc}") from exc
    if (
        parity_adoption != reopened_parity
        or run.get("trace_parity_path") != str(parity_path)
        or run.get("trace_parity_sha256") != _sha(parity_path)
    ):
        raise NonTasteMatrixAppendError("Mut parity receipt binding changed")
    common_path = _physical_file(
        common_adoption.get("path", ""), label="Mut common-adoption source receipt"
    )
    try:
        reopened_common = validate_mut_parity_common_adoption(
            common_path, parity=reopened_parity
        )
    except Exception as exc:
        raise NonTasteMatrixAppendError(
            f"Mut common-adoption receipt reopen failed: {exc}"
        ) from exc
    if common_adoption != reopened_common:
        raise NonTasteMatrixAppendError("Mut common-adoption receipt changed")
    common_root = _physical_directory(
        reopened_common.get("common_root", ""), label="Mut adopted common-recourse root"
    )
    if run.get("source_common_recourse_root") != str(common_root):
        raise NonTasteMatrixAppendError("Mut common-recourse root binding changed")

    standardized = _validate_rf_standardized(
        root, dataset="Mutagenicity", dataset_key="mutagenicity"
    )
    if (
        run.get("standardized_run_manifest_sha256")
        != standardized["run_manifest_sha256"]
        or run.get("freeze_manifest_sha256") != standardized["freeze_manifest_sha256"]
        or run.get("teacher_sha256") != standardized["identities"]["oracle_hash"]
    ):
        raise NonTasteMatrixAppendError("Mut parity outer/standardized identities changed")
    source_integrity = _validate_mut_parity_source_integrity(
        generation,
        source_integrity_final,
        source_root=source_root,
    )
    source_integrity["final_integrity_sha256"] = _sha(root / "source_integrity_final.json")
    upstream = _validate_mut_upstream_checkout(root)
    writer = _writer_audit(root, proc_root=proc_root, required=require_writer_audit)
    common_writer = _writer_audit(
        common_root, proc_root=proc_root, required=require_writer_audit
    )
    return {
        "terminal_kind": "MUT_PARITY_STANDARDIZATION_FINAL",
        "root": str(root),
        "run_manifest_sha256": _sha(root / "run_manifest.json"),
        "final_gate_sha256": _sha(root / "final_gate.json"),
        "run_complete_sha256": _sha(root / "_RUN_COMPLETE.json"),
        "standardized": standardized,
        "trace_parity_sha256": _sha(parity_path),
        "common_adoption_sha256": _sha(common_path),
        "source_integrity": source_integrity,
        "upstream_checkout": upstream,
        "writer_audit": writer,
        "common_writer_audit": common_writer,
        "inventory": _critical_inventory(
            root,
            (
                "PASS",
                "run_manifest.json",
                "final_gate.json",
                "_RUN_COMPLETE.json",
                "generation_adoption_manifest.json",
                "common_recourse_adoption_manifest.json",
                "trace_parity_adoption_manifest.json",
                "source_integrity_final.json",
                "upstream_checkout_audit.json",
                "standardized/run_manifest.json",
                "standardized/final_artifact_audit.json",
                "standardized/freeze_manifest.json",
                "standardized/_FINALIZED.json",
            ),
        ),
    }


def _validate_mut_fast_accurate_terminal(
    root_like: str | Path,
    *,
    proc_root: str | Path,
    require_writer_audit: bool,
) -> dict[str, Any]:
    """Reopen the historical trace-on 50k adoption terminal truthfully."""

    root = _physical_directory(root_like, label="Mut fast-accurate standardization root")
    if any((root / name).exists() for name in ("FAILED", "FAILED.json", "FAIL.json")):
        raise NonTasteMatrixAppendError(
            "Mut fast-accurate standardization root contains a failure sentinel"
        )
    if _physical_file(root / "PASS", label="Mut fast-accurate PASS").read_bytes() != PASS_BYTES:
        raise NonTasteMatrixAppendError("Mut fast-accurate PASS bytes changed")
    payloads = {
        name: _json(_physical_file(root / name, label=f"Mut {name}"), label=f"Mut {name}")
        for name in (
            "run_manifest.json",
            "final_gate.json",
            "_RUN_COMPLETE.json",
            "generation_adoption_manifest.json",
            "historical_adoption_manifest.json",
            "source_integrity_final.json",
        )
    }
    run = payloads["run_manifest.json"]
    final_gate = payloads["final_gate.json"]
    complete = payloads["_RUN_COMPLETE.json"]
    generation = payloads["generation_adoption_manifest.json"]
    historical = payloads["historical_adoption_manifest.json"]
    source_integrity_final = payloads["source_integrity_final.json"]
    if final_gate != run or complete != {**run, "run_complete": True}:
        raise NonTasteMatrixAppendError(
            "Mut fast-accurate outer terminal manifests diverged"
        )
    truthful_contract = {
        "schema_version": MUT_FAST_ACCURATE_RUN_SCHEMA,
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "rf_oracle_used": True,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "generation_adopted": True,
        "generation_rerun": False,
        "historical_artifact_adopted": True,
        "historical_source_trace_enabled": True,
        "full_50k_rerun_performed": False,
        "traceoff_reference_rerun": False,
        "trace_parity_passed": False,
        "500_step_semantic_equivalence_passed": True,
        "adoption_without_full_50k_parity_rerun_authorized": True,
        "trace_fields_stripped": False,
        "common_recourse_adopted": True,
        "common_recourse_rerun": False,
        "pair_store_reused": True,
        "dbscan_reused": True,
        "pair_store_rerun": False,
        "dbscan_rerun": False,
        "chemistry_rerun": True,
        "evaluation_rerun": True,
        "generation_steps": 50_000,
        "M_MAX": 50_000,
        "M_EFFECTIVE": 50_000,
        "early_stop_used": False,
        "stop_reason": "HISTORICAL_FULL_50K_ARTIFACT_ADOPTION",
        "candidate_capacity": 100_000,
        "candidate_universe_binding_state": "PASS",
        "transitive_binding_kind": (
            "transitive_generation_pair_store_vectors_dbscan_v1"
        ),
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
        "source_payload_sha256": MUT_SOURCE_PAYLOAD_SHA256,
        "trace_parity_path": None,
        "trace_parity_sha256": None,
        "standardized_output_root": str(root / "standardized"),
        "calibration_loaded": False,
        "test_loaded_only_in_unified_evaluation": True,
    }
    _require_fields(run, truthful_contract, label="Mut fast-accurate run terminal")
    if re.fullmatch(r"[0-9a-f]{40}", str(run.get("project_commit") or "")) is None:
        raise NonTasteMatrixAppendError("Mut fast-accurate execution commit is invalid")
    universe = run.get("candidate_universe_sha")
    if (
        _SHA256_RE.fullmatch(str(universe or "")) is None
        or run.get("source_native_candidate_universe_sha") != universe
        or run.get("pair_store_source_candidate_universe_sha") != universe
        or run.get("dbscan_native_candidate_universe_sha") is not None
        or run.get("dbscan_transitively_bound_candidate_universe_sha")
        != universe
    ):
        raise NonTasteMatrixAppendError(
            "Mut fast-accurate candidate-universe binding changed"
        )

    source_root = _physical_directory(
        run.get("source_generation_root", ""),
        label="Mut fast-accurate frozen generation root",
    )
    if generation.get("source_generation_root") != str(source_root):
        raise NonTasteMatrixAppendError(
            "Mut fast-accurate generation root binding changed"
        )
    adoption_path = _physical_file(
        run.get("historical_adoption_path", ""),
        label="Mut historical adoption source receipt",
    )
    try:
        reopened_historical = validate_mut_historical_adoption(
            adoption_path,
            source_root=source_root,
        )
    except Exception as exc:
        raise NonTasteMatrixAppendError(
            f"Mut historical adoption reopen failed: {exc}"
        ) from exc
    if (
        historical != reopened_historical
        or run.get("historical_adoption_sha256") != _sha(adoption_path)
        or historical.get("sha256") != _sha(adoption_path)
        or historical.get("candidate_universe_sha") != universe
        or historical.get("source_native_candidate_universe_sha") != universe
        or historical.get("pair_store_source_candidate_universe_sha") != universe
        or historical.get("dbscan_native_candidate_universe_sha") is not None
        or historical.get("dbscan_transitively_bound_candidate_universe_sha")
        != universe
        or historical.get("pair_candidate_graph_hashes_sha256") != universe
        or run.get("pair_candidate_graph_hashes_sha256") != universe
        or historical.get("dbscan_native_candidate_universe_field_present")
        is not False
        or historical.get("dbscan_universe_binding_via_pair_vectors") is not True
    ):
        raise NonTasteMatrixAppendError(
            "Mut historical adoption receipt or universe binding changed"
        )
    common_root = _physical_directory(
        reopened_historical.get("common_root", ""),
        label="Mut adopted common-recourse root",
    )
    if run.get("source_common_recourse_root") != str(common_root):
        raise NonTasteMatrixAppendError(
            "Mut fast-accurate common-recourse root binding changed"
        )
    for run_path_field, run_sha_field, adoption_path_field, adoption_sha_field in (
        (
            "source_pair_store_manifest_path",
            "source_pair_store_manifest_sha256",
            "source_pair_store_manifest_path",
            "source_pair_store_manifest_sha256",
        ),
        (
            "source_dbscan_manifest_path",
            "source_dbscan_manifest_sha256",
            "source_dbscan_manifest_path",
            "source_dbscan_manifest_sha256",
        ),
        (
            "500_step_semantic_equivalence_receipt_path",
            "500_step_semantic_equivalence_receipt_sha256",
            "500_step_semantic_equivalence_receipt_path",
            "500_step_semantic_equivalence_receipt_sha256",
        ),
    ):
        evidence_path = _physical_file(
            reopened_historical.get(adoption_path_field, ""),
            label=f"Mut {adoption_path_field}",
        )
        if (
            run.get(run_path_field) != str(evidence_path)
            or run.get(run_sha_field) != _sha(evidence_path)
            or reopened_historical.get(adoption_sha_field) != _sha(evidence_path)
        ):
            raise NonTasteMatrixAppendError(
                f"Mut fast-accurate evidence binding changed: {run_path_field}"
            )

    pair_manifest = _json(
        _physical_file(
            run.get("source_pair_store_manifest_path", ""),
            label="Mut historical pair-store manifest",
        ),
        label="Mut historical pair-store manifest",
    )
    dbscan_manifest = _json(
        _physical_file(
            run.get("source_dbscan_manifest_path", ""),
            label="Mut historical DBSCAN manifest",
        ),
        label="Mut historical DBSCAN manifest",
    )
    pair_identity = pair_manifest.get("scientific_identity")
    dbscan_identity = dbscan_manifest.get("scientific_identity")
    if (
        not isinstance(pair_identity, Mapping)
        or pair_identity.get("candidate_graph_hashes_sha256") != universe
    ):
        raise NonTasteMatrixAppendError(
            "Mut fast-accurate universe is not the pair-store strict-flip universe"
        )
    native_universe_fields = (
        "source_candidate_universe_sha256",
        "candidate_universe_sha256",
    )
    if any(field in dbscan_manifest for field in native_universe_fields) or (
        isinstance(dbscan_identity, Mapping)
        and any(field in dbscan_identity for field in native_universe_fields)
    ):
        raise NonTasteMatrixAppendError(
            "Mut historical DBSCAN unexpectedly claims a native candidate universe"
        )
    if (
        dbscan_manifest.get("approximation_used") is not False
        or
        not isinstance(dbscan_identity, Mapping)
        or dbscan_identity.get("vectors_path") != pair_manifest.get("vectors_path")
        or dbscan_identity.get("vectors_sha256")
        != pair_manifest.get("vectors_sha256")
    ):
        raise NonTasteMatrixAppendError(
            "Mut historical DBSCAN is not transitively bound through pair vectors"
        )

    standardized = _validate_rf_standardized(
        root, dataset="Mutagenicity", dataset_key="mutagenicity"
    )
    if (
        run.get("standardized_run_manifest_sha256")
        != standardized["run_manifest_sha256"]
        or run.get("freeze_manifest_sha256")
        != standardized["freeze_manifest_sha256"]
        or run.get("teacher_sha256") != standardized["identities"]["oracle_hash"]
    ):
        raise NonTasteMatrixAppendError(
            "Mut fast-accurate outer/standardized identities changed"
        )
    source_integrity = _validate_mut_parity_source_integrity(
        generation,
        source_integrity_final,
        source_root=source_root,
    )
    source_integrity["final_integrity_sha256"] = _sha(
        root / "source_integrity_final.json"
    )
    upstream = _validate_mut_upstream_checkout(root)
    writer = _writer_audit(root, proc_root=proc_root, required=require_writer_audit)
    common_writer = _writer_audit(
        common_root, proc_root=proc_root, required=require_writer_audit
    )
    return {
        "terminal_kind": "MUT_FAST_ACCURATE_STANDARDIZATION_FINAL",
        "root": str(root),
        "run_manifest_sha256": _sha(root / "run_manifest.json"),
        "final_gate_sha256": _sha(root / "final_gate.json"),
        "run_complete_sha256": _sha(root / "_RUN_COMPLETE.json"),
        "standardized": standardized,
        "historical_adoption_sha256": _sha(adoption_path),
        "candidate_universe_sha": str(universe),
        "source_integrity": source_integrity,
        "upstream_checkout": upstream,
        "writer_audit": writer,
        "common_writer_audit": common_writer,
        "inventory": _critical_inventory(
            root,
            (
                "PASS",
                "run_manifest.json",
                "final_gate.json",
                "_RUN_COMPLETE.json",
                "generation_adoption_manifest.json",
                "historical_adoption_manifest.json",
                "source_integrity_final.json",
                "upstream_checkout_audit.json",
                "standardized/run_manifest.json",
                "standardized/final_artifact_audit.json",
                "standardized/freeze_manifest.json",
                "standardized/_FINALIZED.json",
            ),
        ),
    }


def _validate_mut_exact_terminal(
    root_like: str | Path,
    *,
    proc_root: str | Path,
    require_writer_audit: bool,
) -> dict[str, Any]:
    """Reopen the full Mut exact-postprocess terminal and its original append.

    The postprocess predates the shared fast16 pointer and therefore published
    its own immutable one-cell matrix fork.  That fork is accepted only as
    terminal evidence; the caller still republishes the standardized cell from
    the current shared pointer.
    """

    science_root = _physical_directory(root_like, label="Mut exact-postprocess root")
    if any((science_root / name).exists() for name in ("FAILED", "FAILED.json", "FAIL.json")):
        raise NonTasteMatrixAppendError("Mut exact-postprocess root contains a failure sentinel")
    if _physical_file(science_root / "PASS", label="Mut postprocess PASS").read_bytes() != PASS_BYTES:
        raise NonTasteMatrixAppendError("Mut postprocess PASS bytes changed")

    payloads = {
        name: _json(_physical_file(science_root / name, label=f"Mut {name}"), label=f"Mut {name}")
        for name in (
            "run_manifest.json",
            "_RUN_COMPLETE.json",
            "science_manifest.json",
            "_SCIENCE_COMPLETE.json",
            "continuation_resume_contract.json",
            "matrix_append_receipt.json",
            "generation_adoption_manifest.json",
            "exact_common_adoption_manifest.json",
            "trace_parity_adoption_manifest.json",
            "source_generation_integrity_final.json",
            "source_exact_integrity_final.json",
        )
    }
    run = payloads["run_manifest.json"]
    complete = payloads["_RUN_COMPLETE.json"]
    science = payloads["science_manifest.json"]
    science_complete = payloads["_SCIENCE_COMPLETE.json"]
    contract = payloads["continuation_resume_contract.json"]
    matrix_receipt = payloads["matrix_append_receipt.json"]
    generation = payloads["generation_adoption_manifest.json"]
    exact_adoption = payloads["exact_common_adoption_manifest.json"]
    parity_adoption = payloads["trace_parity_adoption_manifest.json"]
    generation_final = payloads["source_generation_integrity_final.json"]
    exact_final = payloads["source_exact_integrity_final.json"]

    science_contract = {
        "schema_version": "mut_comrecgc_exact_postprocess_science_v1",
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "trace_parity_passed": True,
        "generation_adopted": True,
        "generation_rerun": False,
        "common_recourse_adopted": True,
        "common_recourse_rerun": False,
        "dbscan_rerun": False,
        "pair_store_rerun": False,
        "chemistry_rerun": True,
        "wnode_evaluation_rerun": True,
        "expected_common_recourse_count": MUT_EXPECTED_COMMON_RECOURSES,
        "standardized_output_root": str(science_root / "standardized"),
        "calibration_loaded": False,
        "test_loaded_only_in_unified_evaluation": True,
    }
    _require_fields(science, science_contract, label="Mut science_manifest.json")
    if science_complete != {**science, "run_complete": True}:
        raise NonTasteMatrixAppendError("Mut science terminal diverged from science manifest")
    if run != complete:
        raise NonTasteMatrixAppendError("Mut run_manifest and _RUN_COMPLETE diverged")
    _require_fields(
        run,
        {
            **{key: value for key, value in science_contract.items() if key != "schema_version"},
            "schema_version": MUT_RUN_SCHEMA,
            "matrix_append_status": "PASS",
            "matrix_total_cells": 16,
            "run_complete": True,
        },
        label="Mut run terminal",
    )
    for key, value in science.items():
        if key not in {"schema_version", "completed_at"} and run.get(key) != value:
            raise NonTasteMatrixAppendError(f"Mut science/run terminal differs at {key}")

    standardized = _validate_rf_standardized(
        science_root, dataset="Mutagenicity", dataset_key="mutagenicity"
    )
    if run.get("teacher_sha256") != standardized["identities"]["oracle_hash"]:
        raise NonTasteMatrixAppendError("Mut outer/standardized teacher identity changed")

    _require_fields(
        contract,
        {
            "schema_version": "mut_comrecgc_exact_postprocess_resume_v1",
            "expected_common_recourse_count": MUT_EXPECTED_COMMON_RECOURSES,
            "forbidden_reruns": ["pair_store", "DBSCAN", "common_recourse"],
        },
        label="Mut continuation resume contract",
    )
    generation_root = _physical_directory(
        generation.get("source_generation_root", ""), label="Mut frozen generation root"
    )
    _require_fields(
        generation,
        {
            "status": "PASS",
            "dataset": "mutagenicity",
            "generation_adopted": True,
            "generation_mode": "adopted_read_only_cache",
            "generation_rerun": False,
            "counterfactuals_sha256_claimed": MUT_SOURCE_PAYLOAD_SHA256,
            "counterfactuals_sha256_actual": MUT_SOURCE_PAYLOAD_SHA256,
            "counterfactuals_sha256_verified": True,
            "counterfactual_candidate_count": MUT_SOURCE_CANDIDATE_COUNT,
        },
        label="Mut generation adoption",
    )
    _require_fields(
        generation_final,
        {
            "schema_version": 1,
            "status": "PASS",
            "payload_sha256_recomputed": False,
            "payload_stat_unchanged": True,
            "critical_manifest_stat_and_hash_unchanged": True,
        },
        label="Mut final generation integrity",
    )

    exact_receipt_path = _physical_file(
        contract.get("exact_adoption_receipt_path", ""), label="Mut exact adoption receipt"
    )
    common_root = _physical_directory(
        contract.get("adopted_common_root", ""), label="Mut adopted exact common root"
    )
    try:
        reopened_exact = validate_mut_exact_adoption(
            adoption_receipt_path=exact_receipt_path,
            common_root=common_root,
            source_generation_root=generation_root,
            proc_root=proc_root,
        )
    except Exception as exc:
        raise NonTasteMatrixAppendError(f"Mut exact adoption reopen failed: {exc}") from exc
    exact_identity_fields = (
        "adoption_receipt_path",
        "adoption_receipt_sha256",
        "source_controller_state_path",
        "source_controller_state_sha256",
        "common_root",
        "common_terminal_sha256",
        "common_manifest_sha256",
        "source_generation_root",
        "source_generation_manifest_sha256",
        "source_counterfactuals_sha256",
        "common_recourse_count",
        "common_recourse_parameters",
        "dbscan_scientific_identity_sha256",
        "dbscan_next_offset",
    )
    for label, recorded in (
        ("entry", exact_adoption),
        ("final", exact_final),
    ):
        _require_fields(
            recorded,
            {
                "schema_version": "mut_comrecgc_exact_read_only_adoption_v1",
                "status": "PASS",
                "common_recourse_rerun": False,
                "dbscan_rerun": False,
                "pair_store_rerun": False,
            },
            label=f"Mut exact {label} adoption",
        )
        changed = [
            field
            for field in exact_identity_fields
            if recorded.get(field) != reopened_exact.get(field)
        ]
        if changed:
            raise NonTasteMatrixAppendError(
                f"Mut exact {label} adoption identity changed: {', '.join(changed)}"
            )
    if (
        contract.get("exact_adoption_receipt_sha256") != _sha(exact_receipt_path)
        or contract.get("adopted_common_terminal_sha256")
        != reopened_exact["common_terminal_sha256"]
        or contract.get("adopted_common_manifest_sha256")
        != reopened_exact["common_manifest_sha256"]
        or contract.get("dbscan_scientific_identity_sha256")
        != reopened_exact["dbscan_scientific_identity_sha256"]
    ):
        raise NonTasteMatrixAppendError("Mut resume contract/exact identity changed")

    parity_path = _physical_file(
        contract.get("trace_parity_path", ""), label="Mut trace parity receipt"
    )
    try:
        parity = validate_mut_trace_parity(parity_path)
    except Exception as exc:
        raise NonTasteMatrixAppendError(f"Mut trace parity reopen failed: {exc}") from exc
    if (
        parity_adoption != parity
        or contract.get("trace_parity_sha256") != _sha(parity_path)
        or Path(str(parity.get("traced_source_root") or "")).resolve(strict=True)
        != generation_root
    ):
        raise NonTasteMatrixAppendError("Mut trace parity binding changed")

    prior_matrix = _physical_directory(
        contract.get("prior_matrix_root", ""), label="Mut original prior matrix"
    )
    matrix_output = _physical_directory(
        contract.get("matrix_output_root", ""), label="Mut original published matrix"
    )
    try:
        original_append = _reopen_mut_matrix_append(
            prior_authority_root=prior_matrix,
            standardized_root=Path(standardized["root"]),
            output_root=matrix_output,
            proc_root=Path(proc_root),
            require_writer_audit=require_writer_audit,
        )
    except Exception as exc:
        raise NonTasteMatrixAppendError(f"Mut original strict append reopen failed: {exc}") from exc
    comparable = set(original_append) - {"adopted_after_interruption"}
    if set(matrix_receipt) != set(original_append) or any(
        matrix_receipt.get(key) != original_append.get(key) for key in comparable
    ):
        raise NonTasteMatrixAppendError("Mut matrix append receipt changed")
    if (
        matrix_receipt.get("status") != "PASS"
        or run.get("matrix_output_root") != str(matrix_output)
        or run.get("matrix_complete_cells") != original_append["matrix_complete_cells"]
        or run.get("matrix_status_sha256") != original_append["matrix_status_sha256"]
        or _json(matrix_output / "append_authority.json", label="Mut old append authority").get(
            "schema_version"
        )
        != MUT_MATRIX_APPEND_SCHEMA
    ):
        raise NonTasteMatrixAppendError("Mut outer/original matrix terminal changed")

    writer = _writer_audit(
        science_root, proc_root=proc_root, required=require_writer_audit
    )
    return {
        "terminal_kind": "MUT_EXACT_POSTPROCESS_FINAL",
        "root": str(science_root),
        "run_manifest_sha256": _sha(science_root / "run_manifest.json"),
        "science_manifest_sha256": _sha(science_root / "science_manifest.json"),
        "resume_contract_sha256": _sha(science_root / "continuation_resume_contract.json"),
        "original_matrix_authority_root": str(matrix_output),
        "original_matrix_status_sha256": original_append["matrix_status_sha256"],
        "original_matrix_used_as_terminal_evidence_only": True,
        "standardized": standardized,
        "exact_adoption_receipt_sha256": _sha(exact_receipt_path),
        "trace_parity_sha256": _sha(parity_path),
        "writer_audit": writer,
        "inventory": _critical_inventory(
            science_root,
            (
                "PASS",
                "run_manifest.json",
                "_RUN_COMPLETE.json",
                "science_manifest.json",
                "_SCIENCE_COMPLETE.json",
                "continuation_resume_contract.json",
                "matrix_append_receipt.json",
                "standardized/run_manifest.json",
                "standardized/final_artifact_audit.json",
                "standardized/freeze_manifest.json",
                "standardized/_FINALIZED.json",
            ),
        ),
    }


def _validate_mut_terminal(
    root_like: str | Path,
    *,
    proc_root: str | Path,
    require_writer_audit: bool,
) -> dict[str, Any]:
    """Dispatch only between the explicit Mut production terminals."""

    root = _physical_directory(root_like, label="Mut terminal root")
    run = _json(
        _physical_file(root / "run_manifest.json", label="Mut run manifest"),
        label="Mut run manifest",
    )
    schema = run.get("schema_version")
    if schema == _MUT_PARITY_RUN_SCHEMA:
        return _validate_mut_parity_terminal(
            root,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
        )
    if schema == MUT_FAST_ACCURATE_RUN_SCHEMA:
        return _validate_mut_fast_accurate_terminal(
            root,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
        )
    if schema == MUT_RUN_SCHEMA:
        return _validate_mut_exact_terminal(
            root,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
        )
    raise NonTasteMatrixAppendError(
        f"Unsupported Mut terminal schema: {schema!r}"
    )


def _ordered_rows(rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(rows[(dataset, method)]) for dataset in DATASETS for method in METHODS)


def _identity_compatibility(
    *,
    dataset: str,
    target: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    fields = _BACE_SHARED_FIELDS if dataset == "BACE" else _RF_SHARED_FIELDS
    passing = {status.value for status in PASS_STATUSES}
    if str(reference.get("status") or "") not in passing:
        raise NonTasteMatrixAppendError(f"{dataset} append requires a passing Ours reference")
    unavailable: list[str] = []
    for field in fields:
        observed = target.get(field)
        expected = reference.get(field)
        if observed in (None, ""):
            raise NonTasteMatrixAppendError(f"Target cell lacks shared identity: {field}")
        if expected in (None, ""):
            unavailable.append(field)
        elif observed != expected:
            raise NonTasteMatrixAppendError(
                f"{dataset} target differs from Ours identity: {field}"
            )
    if unavailable:
        approved_legacy = (
            dataset in {"AIDS", "Mutagenicity"}
            and reference.get("status") == CellStatus.ADOPTABLE_PASS.value
            and reference.get("registry_exception") == FROZEN_V4_APPROVAL_ID
            and reference.get("identity_evidence_status")
            == "USER_APPROVED_LEGACY_IDENTITIES_NOT_EMBEDDED"
        )
        if not approved_legacy:
            raise NonTasteMatrixAppendError(
                f"{dataset}/Ours reference lacks identities without an approved legacy exception: "
                + ", ".join(unavailable)
            )
    return {
        "fields": list(fields),
        "matched_fields": [field for field in fields if field not in unavailable],
        "reference_unavailable_fields": unavailable,
        "legacy_exception_used_for_missing_reference_identity": bool(unavailable),
    }


def append_non_taste_matrix_cell(
    *,
    prior_authority_root: str | Path,
    dataset: str,
    method: str,
    cell_terminal_root: str | Path,
    output_root: str | Path,
    aids_controller_manifest: str | Path | None = None,
    proc_root: str | Path = "/proc",
    require_writer_audit: bool = True,
    git_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Append exactly one strict non-Taste terminal to a hash-closed authority."""

    key = (dataset, method)
    if key not in TARGETS:
        raise NonTasteMatrixAppendError(f"Unsupported non-Taste matrix cell: {key}")
    if dataset != "AIDS" and aids_controller_manifest is not None:
        raise NonTasteMatrixAppendError("--aids-controller-manifest is AIDS-only")
    prior = _verify_authority(prior_authority_root)
    prior_rows = prior["rows"]
    passing = {status.value for status in PASS_STATUSES}
    if str(prior_rows[key].get("status") or "") in passing:
        raise NonTasteMatrixAppendError(f"Prior authority already passes {dataset}/{method}")
    reference = prior_rows[(dataset, "Ours")]
    destination_logical = Path(output_root).expanduser()
    if not destination_logical.is_absolute() or destination_logical.is_symlink():
        raise NonTasteMatrixAppendError("Matrix output must be an absolute non-symlink path")
    destination = destination_logical.resolve(strict=False)
    if destination.exists():
        raise NonTasteMatrixAppendError(f"Matrix output must be fresh: {destination}")
    if dataset == "BACE":
        terminal = _validate_bace_terminal(
            cell_terminal_root,
            method=method,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
        )
    elif dataset == "AIDS":
        terminal = _validate_aids_terminal(
            cell_terminal_root,
            controller_manifest_path=aids_controller_manifest,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
        )
    else:
        terminal = _validate_mut_terminal(
            cell_terminal_root,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
        )
    cell_root = Path(str(terminal["root"])).resolve(strict=True)
    registry_cell_root = (
        Path(str(terminal["standardized"]["root"])).resolve(strict=True)
        if dataset == "Mutagenicity"
        else cell_root
    )
    protected = {
        Path(prior["root"]).resolve(strict=True),
        cell_root,
        registry_cell_root,
    }
    reconciliation_root = terminal.get("reconciliation_root")
    if reconciliation_root is not None:
        protected.add(Path(str(reconciliation_root)).resolve(strict=True))
    if any(destination == root or destination in root.parents or root in destination.parents for root in protected):
        raise NonTasteMatrixAppendError("Matrix output overlaps a protected authority/science root")

    explicit_cells = {
        f"{old_dataset}/{old_method}": str(
            Path(str(row["standardized_output_root"])).resolve(strict=True)
        )
        for (old_dataset, old_method), row in prior_rows.items()
        if str(row.get("status") or "") in passing
    }
    explicit_cells[f"{dataset}/{method}"] = str(registry_cell_root)
    dataset_expectation = {
        field: reference.get(field)
        for field in (
            "oracle_backend",
            "oracle_checkpoint",
            "oracle_hash",
            "dataset_hash",
            "split_hash",
            "molclr_checkpoint_hash",
            "threshold_config_hash",
        )
        if reference.get(field) not in (None, "")
    }
    result = audit_registry(
        AuditConfig(
            scan_roots=(),
            output_root=destination,
            explicit_cells=explicit_cells,
            expectations={"datasets": {dataset: dataset_expectation}},
            max_hash_bytes=MAX_HASH_BYTES,
        )
    )
    proposed = {
        (str(row["dataset"]), str(row["method"])): dict(row)
        for row in result.matrix_rows
    }
    target = proposed[key]
    if key == ("AIDS", "ComRecGC"):
        target = _reconcile_aids_zero_registry_row(target, terminal=terminal)
        proposed[key] = target
    if (
        target.get("status") != CellStatus.FROZEN_PASS.value
        or Path(str(target.get("standardized_output_root") or "")).resolve(strict=True)
        != (
            cell_root / "standardized" if dataset == "AIDS" else registry_cell_root
        )
        or target.get("k_max") != 20
        or target.get("table2_k") != 10
    ):
        raise NonTasteMatrixAppendError(
            f"{dataset}/{method} failed the ordinary frozen registry gate: "
            f"{target.get('rerun_reason')}"
        )
    compatibility = _identity_compatibility(
        dataset=dataset, target=target, reference=reference
    )
    # A strict append changes exactly one row.  The predecessor is already a
    # hash-closed authority; re-discovery differences cannot rewrite any of its
    # rows (notably an older nested AIDS container whose row stores /standardized).
    for old_key, old_row in prior_rows.items():
        if old_key != key:
            proposed[old_key] = dict(old_row)
    expected_complete = int(prior["complete"]) + 1
    observed_complete = sum(
        str(row.get("status") or "") in passing for row in proposed.values()
    )
    if observed_complete != expected_complete:
        raise NonTasteMatrixAppendError(
            f"Append did not add exactly one cell: prior={prior['complete']} proposed={observed_complete}"
        )
    result = replace(
        result,
        matrix_rows=_ordered_rows(proposed),
        matrix_complete_cells=observed_complete,
    )
    execution = dict(git_identity or _git_identity())
    if set(execution) != {"commit", "tree"} or any(
        re.fullmatch(r"[0-9a-f]{40}", str(execution[field])) is None
        for field in ("commit", "tree")
    ):
        raise NonTasteMatrixAppendError("Execution Git identity is incomplete")
    marker = f"[MATRIX_{expected_complete}_OF_16_PASS]"
    receipt = {
        "schema_version": APPEND_SCHEMA,
        "status": "PASS",
        "created_at": _utc_now(),
        "execution": execution,
        "prior_authority_root": str(prior["root"]),
        "prior_matrix_complete_cells": prior["complete"],
        "prior_matrix_status_sha256": prior["matrix_sha256"],
        "prior_combined_audit_sha256": prior["combined_sha256"],
        "prior_rows_sha256": stable_json_sha256(list(prior["matrix"]["cells"])),
        "appended_cell": {
            "dataset": dataset,
            "method": method,
            "cell_terminal_root": str(cell_root),
            "registry_row": target,
            "terminal_evidence": terminal,
        },
        "reference_cell": dict(reference),
        "identity_compatibility": compatibility,
        "unchanged_non_target_rows": True,
        "new_matrix_complete_cells": expected_complete,
        "new_matrix_total_cells": 16,
        "new_authority_root": str(destination),
        "scientific_metrics_recomputed": False,
        "candidate_order_changed": False,
        "raw_test_opened": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "numeric_imputation_used": False,
        "gnn_ablation_started": False,
        "marker": marker,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{secrets.token_hex(16)}"
    try:
        write_registry_outputs(
            result,
            staging,
            supplemental_outputs={"append_authority.json": _json_bytes(receipt)},
        )
        staged = _verify_authority(staging, expected_complete=expected_complete)
        if staged["rows"] != proposed:
            raise NonTasteMatrixAppendError("Staged matrix rows changed")
        if _read_json(staging / "append_authority.json") != receipt:
            raise NonTasteMatrixAppendError("Staged append receipt changed")
        atomic_rename_directory_noreplace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    reopened = _verify_authority(destination, expected_complete=expected_complete)
    if reopened["rows"] != proposed:
        raise NonTasteMatrixAppendError("Published matrix rows changed on independent reopen")
    if _read_json(destination / "append_authority.json") != receipt:
        raise NonTasteMatrixAppendError("Published append receipt changed on reopen")
    return {
        "status": "PASS",
        "output_root": str(destination),
        "matrix_status_path": str(destination / "matrix_status.json"),
        "matrix_status_sha256": reopened["matrix_sha256"],
        "combined_audit_sha256": reopened["combined_sha256"],
        "matrix_complete_cells": reopened["complete"],
        "matrix_total_cells": 16,
        "appended_cell": f"{dataset}/{method}",
        "marker": marker,
    }


__all__ = [
    "APPEND_SCHEMA",
    "NonTasteMatrixAppendError",
    "TARGETS",
    "append_non_taste_matrix_cell",
]
