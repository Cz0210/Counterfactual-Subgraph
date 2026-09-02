"""Strict, read-only TasteMolNet paper-cell matrix append.

This module does not run a model, select a rule, or open a dataset split.  It
reopens already-published T11--T14 full-cell terminals, proves their common T3
classifier/temperature and evaluation identities, and appends only those rows
to an existing hash-closed 4x4 matrix authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
from typing import Any, Mapping

from scripts.autodl.append_bace_gcf_matrix_authority import (
    _git_identity,
    _json_bytes,
    _read_json,
    _verify_authority,
)
from src.eval.am_legacy_standardization import scan_live_writers
from src.eval.four_by_four_registry import (
    AuditConfig,
    CellStatus,
    DATASETS,
    METHODS,
    PASS_STATUSES,
    audit_registry,
    build_oracle_registry,
    stable_json_sha256,
    write_registry_outputs,
)
from src.eval.tastemolnet_t4_oracle_smoke_v2 import HeldPublishedT3
from src.eval.tastemolnet_t11_policy_path_relocation import (
    validate_t11_policy_path_relocation,
)
from src.train.molecular_gnn_resume import atomic_rename_directory_noreplace
from src.utils.tastemolnet_research_policy import (
    load_tastemolnet_research_policy,
    validate_tastemolnet_local_authority,
    validate_tastemolnet_policy_receipt,
)


APPEND_SCHEMA = "tastemolnet_matrix_authority_append_v1"
TARGET_DATASET = "TasteMolNet"
MIN_PRIOR_COMPLETE = 8
MAX_HASH_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MethodContract:
    stage: str
    pass_payload: bytes
    run_schema: str | re.Pattern[str]
    audit_schema: str | re.Pattern[str]


METHOD_CONTRACTS: dict[str, MethodContract] = {
    "Ours": MethodContract(
        stage="T11_OURS_FULL",
        pass_payload=b"[TASTE_OURS_PASS]\n",
        run_schema="tastemolnet_t11_final_run_manifest_v2",
        audit_schema="tastemolnet_t11_final_artifact_audit_v2",
    ),
    "GCFExplainer": MethodContract(
        stage="T12_GCF_FULL",
        pass_payload=b"[TASTE_GCF_PASS]\n",
        # Production T12 is not released yet.  These exact names form its
        # fail-closed consumer contract; no replay/canary schema can match.
        run_schema="tastemolnet_t12_final_run_manifest_v1",
        audit_schema="tastemolnet_t12_terminal_verification_v1",
    ),
    "GlobalGCE": MethodContract(
        stage="T13_GLOBALGCE_FULL",
        pass_payload=b"PASS\n",
        run_schema="tastemolnet_t13_run_manifest_v1",
        audit_schema="tastemolnet_t13_terminal_verification_v1",
    ),
    "ComRecGC": MethodContract(
        stage="T14_COMRECGC_FULL_POSTPROCESS",
        pass_payload=b"[TASTE_COMRECGC_PASS]\n",
        run_schema="tastemolnet_t14_postprocess_run_manifest_v1",
        audit_schema="tastemolnet_t14_postprocess_terminal_verification_v1",
    ),
}


SHARED_REGISTRY_FIELDS = (
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
SHARED_FULL_FIELDS = (
    "oracle_checkpoint_hash",
    "temperature_calibration_hash",
    "dataset_hash",
    "test_split_hash",
    "test_parent_ids_sha256",
    "molclr_checkpoint_hash",
    "threshold_config_hash",
)


class TasteMatrixAppendError(RuntimeError):
    """A proposed Taste paper-cell append is not scientifically closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_directory(path_like: str | Path, *, label: str) -> Path:
    logical = Path(path_like).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise TasteMatrixAppendError(f"{label} must be an absolute physical directory")
    try:
        path = logical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TasteMatrixAppendError(f"{label} is absent: {logical}") from exc
    if path != logical or not path.is_dir():
        raise TasteMatrixAppendError(f"{label} is not an exact physical directory: {logical}")
    return path


def _physical_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TasteMatrixAppendError(f"{label} must be one physical file: {path}")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise TasteMatrixAppendError(f"{label} is not a regular file: {path}")
    return path


def _schema_matches(value: Any, expected: str | re.Pattern[str]) -> bool:
    text = str(value or "")
    return text == expected if isinstance(expected, str) else expected.fullmatch(text) is not None


def _require_bool(payload: Mapping[str, Any], field: str, expected: bool, *, label: str) -> None:
    if payload.get(field) is not expected:
        raise TasteMatrixAppendError(f"{label}.{field} must be {expected}")


def _safe_inventory_path(root: Path, raw_name: Any, *, label: str) -> Path:
    relative = Path(str(raw_name))
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise TasteMatrixAppendError(f"{label} contains an unsafe path: {raw_name!r}")
    path = root / relative
    _physical_file(path, label=f"{label}:{relative.as_posix()}")
    try:
        if path.resolve(strict=True).relative_to(root) != relative:
            raise TasteMatrixAppendError(f"{label} path escaped its root: {raw_name!r}")
    except ValueError as exc:
        raise TasteMatrixAppendError(f"{label} path escaped its root: {raw_name!r}") from exc
    return path


def _validate_inventory(
    root: Path,
    inventory: Any,
    *,
    label: str,
    expected_digest: Any = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(inventory, Mapping) or not inventory:
        raise TasteMatrixAppendError(f"{label} must be one nonempty file inventory")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_identity in sorted(inventory.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_identity, Mapping):
            raise TasteMatrixAppendError(f"{label} identity is malformed: {raw_name}")
        path = _safe_inventory_path(root, raw_name, label=label)
        expected_sha = str(raw_identity.get("sha256") or "").lower()
        expected_bytes = raw_identity.get("bytes")
        if (
            not _SHA256_RE.fullmatch(expected_sha)
            or type(expected_bytes) is not int
            or expected_bytes < 0
            or path.stat().st_size != expected_bytes
            or _sha256_file(path) != expected_sha
        ):
            raise TasteMatrixAppendError(f"{label} member drifted: {raw_name}")
        normalized[str(raw_name)] = {
            "bytes": expected_bytes,
            "sha256": expected_sha,
        }
    if expected_digest not in (None, ""):
        if (
            not _SHA256_RE.fullmatch(str(expected_digest))
            or stable_json_sha256(normalized) != expected_digest
        ):
            raise TasteMatrixAppendError(f"{label} aggregate digest changed")
    return normalized


def _validate_freeze(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    terminal_freeze = _physical_file(
        root / "freeze_manifest.json", label="terminal freeze manifest"
    )
    inventory_root = root
    freeze_path = terminal_freeze
    if method == "Ours":
        science_root = _physical_directory(
            str(manifest.get("science_root") or ""), label="T11 science root"
        )
        freeze_path = _physical_file(
            science_root / "freeze_manifest.json", label="T11 science freeze manifest"
        )
        expected = str(manifest.get("science_freeze_manifest_sha256") or "")
        if (
            not _SHA256_RE.fullmatch(expected)
            or _sha256_file(freeze_path) != expected
            or _sha256_file(terminal_freeze) != expected
        ):
            raise TasteMatrixAppendError("T11 final did not bind its science freeze")
        inventory_root = science_root
    freeze = _read_json(freeze_path)
    files = _validate_inventory(
        inventory_root,
        freeze.get("files"),
        label="freeze.files",
        expected_digest=freeze.get("inventory_sha256"),
    )
    source_files: dict[str, dict[str, Any]] = {}
    raw_source = freeze.get("source_evidence_files")
    if raw_source not in (None, {}):
        source_root = inventory_root
        try:
            source_files = _validate_inventory(
                source_root,
                raw_source,
                label="freeze.source_evidence_files",
                expected_digest=freeze.get("source_evidence_inventory_sha256"),
            )
        except TasteMatrixAppendError:
            science_value = manifest.get("science_root")
            if not science_value:
                raise
            source_root = _physical_directory(science_value, label="terminal science root")
            science_freeze = _physical_file(
                source_root / "freeze_manifest.json", label="science freeze manifest"
            )
            if (
                not _SHA256_RE.fullmatch(str(manifest.get("science_freeze_manifest_sha256") or ""))
                or _sha256_file(science_freeze)
                != manifest.get("science_freeze_manifest_sha256")
            ):
                raise TasteMatrixAppendError("terminal does not bind its science freeze")
            source_files = _validate_inventory(
                source_root,
                raw_source,
                label="freeze.source_evidence_files",
                expected_digest=freeze.get("source_evidence_inventory_sha256"),
            )
    return {
        "freeze_manifest_sha256": _sha256_file(freeze_path),
        "frozen_file_count": len(files),
        "source_evidence_file_count": len(source_files),
        "inventory_sha256": stable_json_sha256(files),
        "source_evidence_inventory_sha256": (
            stable_json_sha256(source_files) if source_files else None
        ),
    }


def _terminal_inventory(root: Path) -> dict[str, Any]:
    required = {
        "PASS",
        "summary.json",
        "run_manifest.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "freeze_manifest.json",
        "final_artifact_audit.json",
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "destination_distribution.csv",
    }
    table2 = sorted(root.glob("table2_*_k10.csv"))
    if len(table2) != 1:
        raise TasteMatrixAppendError("Taste final root requires exactly one Table-2 K=10 CSV")
    required.add(table2[0].name)
    files: dict[str, dict[str, Any]] = {}
    recursive_file_count = 0
    recursive_bytes = 0
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in tuple(directory_names):
            if (current_path / name).is_symlink():
                raise TasteMatrixAppendError("Taste final root contains a directory symlink")
        for name in file_names:
            path = current_path / name
            if path.is_symlink():
                raise TasteMatrixAppendError("Taste final root contains a file symlink")
            recursive_file_count += 1
            recursive_bytes += path.stat().st_size
    for name in sorted(required):
        path = _physical_file(root / name, label=f"Taste terminal {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    return {
        "terminal_files": files,
        "terminal_files_sha256": stable_json_sha256(files),
        "recursive_file_count": recursive_file_count,
        "recursive_bytes": recursive_bytes,
    }


def _temperature_evidence(
    manifest: Mapping[str, Any],
    oracle_manifest: Mapping[str, Any],
) -> tuple[str, float | None]:
    explicit = str(manifest.get("temperature_calibration_hash") or "").lower()
    checkpoint = _physical_directory(
        str(manifest.get("oracle_checkpoint") or ""), label="Taste oracle checkpoint"
    )
    temperature_path = _physical_file(
        checkpoint / "temperature_scaling.json", label="Taste temperature scaling"
    )
    observed = _sha256_file(temperature_path)
    if explicit and (not _SHA256_RE.fullmatch(explicit) or explicit != observed):
        raise TasteMatrixAppendError("terminal temperature calibration hash changed")
    payload = _read_json(temperature_path)
    value = payload.get("temperature")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise TasteMatrixAppendError("T3 temperature value is invalid")
    recorded = oracle_manifest.get("temperature")
    if recorded is not None and (
        isinstance(recorded, bool)
        or not isinstance(recorded, (int, float))
        or float(recorded) != float(value)
    ):
        raise TasteMatrixAppendError("oracle manifest temperature differs from T3")
    return observed, float(value)


def _validate_taste_cell(
    root_like: str | Path,
    *,
    method: str,
    t3_binding: Mapping[str, Any],
    proc_root: str | Path,
    require_writer_audit: bool,
) -> dict[str, Any]:
    contract = METHOD_CONTRACTS[method]
    root = _physical_directory(root_like, label=f"Taste {method} final root")
    marker = _physical_file(root / "PASS", label=f"Taste {method} PASS").read_bytes()
    if marker != contract.pass_payload:
        raise TasteMatrixAppendError(f"Taste {method} method-cell PASS marker changed")
    manifest = _read_json(_physical_file(root / "run_manifest.json", label="run manifest"))
    audit = _read_json(
        _physical_file(root / "final_artifact_audit.json", label="final artifact audit")
    )
    oracle = _read_json(
        _physical_file(root / "oracle_manifest.json", label="oracle manifest")
    )
    evaluation = _read_json(
        _physical_file(root / "evaluation_manifest.json", label="evaluation manifest")
    )
    summary = _read_json(_physical_file(root / "summary.json", label="summary"))
    destination_labels = evaluation.get("destination_labels")
    if destination_labels is None and method == "Ours":
        # T11 v2 records the branch set in summary.json; its evaluation
        # manifest binds the full Cartesian replay but predates this common
        # field on T13/T14.
        destination_labels = summary.get("destination_labels")
    if (
        not _schema_matches(manifest.get("schema_version"), contract.run_schema)
        or not _schema_matches(audit.get("schema_version"), contract.audit_schema)
        or manifest.get("dataset") != TARGET_DATASET
        or manifest.get("method") != method
        or manifest.get("stage") != contract.stage
        or manifest.get("status") != "PASS"
        or manifest.get("state") != "PASS"
        or manifest.get("run_complete") is not True
        or manifest.get("raw_output_complete") is not True
        or manifest.get("frozen") is not True
        or audit.get("dataset") != TARGET_DATASET
        or audit.get("method") != method
        or audit.get("stage") != contract.stage
        or audit.get("status") != "PASS"
        or audit.get("passed") is not True
        or audit.get("audit_passed") is not True
    ):
        raise TasteMatrixAppendError(f"Taste {method} terminal receipts are not a full PASS")
    independent = (
        manifest.get("independent_terminal_verification_passed") is True
        or audit.get("independent_verifier") is True
        or manifest.get("terminal_verifier") == "separate_verify_only_invocation"
    )
    if not independent or manifest.get("worker_wrote_pass") is not False:
        raise TasteMatrixAppendError(f"Taste {method} independent verifier is not proven")
    # Run/evaluation fields are common across T11--T14.  The independent
    # audits intentionally have method-specific layouts: T11 stores these
    # fields directly, while T13 records selector replay under ``checks``.
    for payload, label in (
        (manifest, "run_manifest"),
        (evaluation, "evaluation_manifest"),
    ):
        _require_bool(payload, "selection_frozen_before_test", True, label=label)
        _require_bool(payload, "test_used_for_selection", False, label=label)
        _require_bool(payload, "threshold_fitted_on_test", False, label=label)
    checks = audit.get("checks") if isinstance(audit.get("checks"), Mapping) else {}
    if (
        audit.get("selection_frozen_before_test") is not True
        and checks.get("selection_frozen_before_test") is not True
    ):
        raise TasteMatrixAppendError(
            f"Taste {method} audit did not independently prove calibration freeze"
        )
    _require_bool(
        audit, "test_used_for_selection", False, label="final_artifact_audit"
    )
    if (
        "threshold_fitted_on_test" in audit
        and audit.get("threshold_fitted_on_test") is not False
    ):
        raise TasteMatrixAppendError(
            "final_artifact_audit.threshold_fitted_on_test must be False"
        )
    if method == "Ours" and (
        audit.get("calibration_pair_chunks_replayed") is not True
        or audit.get("recomputed_metrics") is not True
    ):
        raise TasteMatrixAppendError("T11 audit did not replay calibration and metrics")
    if method == "GlobalGCE" and checks.get("calibration_only_selector") is not True:
        raise TasteMatrixAppendError("T13 audit did not replay its calibration selector")
    if method in {"GCFExplainer", "ComRecGC"} and (
        checks.get("calibration_only_selector_replayed") is not True
    ):
        raise TasteMatrixAppendError(
            f"Taste {method} audit did not replay its calibration selector"
        )
    _require_bool(evaluation, "full_cartesian_test_pairs", True, label="evaluation_manifest")
    if (
        manifest.get("oracle_backend") != "gnn"
        or manifest.get("classifier_family") != "gine"
        or manifest.get("rf_oracle_used") is not False
        or manifest.get("num_classes") != 3
        or manifest.get("source_label") != 1
        or oracle.get("same_frozen_gine_for_generation_calibration_test") is not True
        or oracle.get("rf_oracle_used") is not False
        or oracle.get("num_classes") != 3
        or oracle.get("source_label") != 1
        or sorted(destination_labels or []) != [0, 2]
    ):
        raise TasteMatrixAppendError(f"Taste {method} GINE/multiclass contract changed")
    checkpoint_path = _physical_directory(
        str(manifest.get("oracle_checkpoint") or ""), label="Taste oracle checkpoint"
    )
    expected_checkpoint = _physical_directory(
        str(t3_binding.get("checkpoint_dir") or ""), label="held T3 checkpoint"
    )
    checkpoint_hash = str(manifest.get("oracle_checkpoint_hash") or "").lower()
    if (
        checkpoint_path != expected_checkpoint
        or checkpoint_hash != t3_binding.get("checkpoint_id")
        or manifest.get("oracle_hash") != checkpoint_hash
    ):
        raise TasteMatrixAppendError(f"Taste {method} did not use the held T3 GINE")
    temperature_hash, temperature = _temperature_evidence(manifest, oracle)
    if (
        temperature_hash != t3_binding.get("temperature_scaling_sha256")
        or temperature != float(t3_binding.get("temperature"))
    ):
        raise TasteMatrixAppendError(f"Taste {method} did not use the held T3 temperature")
    identities: dict[str, str] = {
        "oracle_checkpoint_hash": checkpoint_hash,
        "temperature_calibration_hash": temperature_hash,
    }
    for field in SHARED_FULL_FIELDS[2:]:
        value = str(manifest.get(field) or "").lower()
        if not _SHA256_RE.fullmatch(value):
            raise TasteMatrixAppendError(f"Taste {method} missing identity: {field}")
        identities[field] = value
    freeze = _validate_freeze(root, manifest, method=method)
    if method == "Ours":
        _validate_inventory(
            root,
            audit.get("files"),
            label="T11 final audit files",
        )
    terminal = _terminal_inventory(root)
    writer_audit = (
        scan_live_writers(root, proc_root=proc_root)
        if require_writer_audit
        else {
            "procfs_verified": False,
            "scanned_process_count": 0,
            "writable_fd_count": 0,
            "writers": [],
        }
    )
    if require_writer_audit and (
        writer_audit.get("procfs_verified") is not True
        or writer_audit.get("writable_fd_count") != 0
        or writer_audit.get("writers") != []
    ):
        raise TasteMatrixAppendError(f"Taste {method} final root still has a live writer")
    return {
        "method": method,
        "root": str(root),
        "run_manifest_sha256": _sha256_file(root / "run_manifest.json"),
        "final_artifact_audit_sha256": _sha256_file(root / "final_artifact_audit.json"),
        "oracle_manifest_sha256": _sha256_file(root / "oracle_manifest.json"),
        "evaluation_manifest_sha256": _sha256_file(root / "evaluation_manifest.json"),
        "summary_sha256": _sha256_file(root / "summary.json"),
        "pass_sha256": _sha256_file(root / "PASS"),
        "identities": identities,
        "freeze": freeze,
        "terminal_inventory": terminal,
        "writer_audit": writer_audit,
    }


def _load_t3_binding(t3_root: str | Path) -> dict[str, Any]:
    held = HeldPublishedT3(t3_root)
    try:
        held.verify()
        return dict(held.binding)
    finally:
        held.close()


def _load_policy_binding(
    *,
    policy_path: str | Path,
    policy_receipt: str | Path,
    prepared_root: str | Path,
    graph_cache_root: str | Path,
) -> dict[str, Any]:
    policy = load_tastemolnet_research_policy(policy_path)
    authority = validate_tastemolnet_local_authority(
        policy,
        prepared_root=prepared_root,
        graph_cache_root=graph_cache_root,
    )
    receipt = validate_tastemolnet_policy_receipt(
        policy_receipt,
        policy=policy,
        authority=authority,
        require_active=True,
        require_policy_version=2,
    )
    if (
        receipt.payload.get("paper_reporting_authorized") is not True
        or receipt.payload.get("dataset_redistribution_authorized") is not False
        or receipt.payload.get("license_conclusion") != "NOT_GRANTED_OR_INFERRED"
    ):
        raise TasteMatrixAppendError("Taste scoped reporting policy changed")
    return {
        "policy_receipt_path": str(receipt.path),
        "policy_receipt_sha256": receipt.sha256,
        "policy_id": policy.policy_id,
        "policy_version": policy.version,
        "paper_reporting_authorized": True,
        "dataset_redistribution_authorized": False,
        "license_conclusion": "NOT_GRANTED_OR_INFERRED",
        "legacy_license_pass_claimed": False,
    }


def _taste_gate(policy_binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "passed": True,
        "reuse_basis": (
            "scoped private-research and aggregate-reporting authorization; "
            f"receipt_sha256={policy_binding['policy_receipt_sha256']}; "
            "upstream licence not inferred"
        ),
    }


def _ordered_rows(rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(rows[(dataset, method)]) for dataset in DATASETS for method in METHODS)


def append_tastemolnet_cells(
    *,
    prior_authority_root: str | Path,
    taste_cells: Mapping[str, str | Path],
    output_root: str | Path,
    t3_root: str | Path,
    policy_path: str | Path,
    policy_receipt: str | Path,
    policy_path_relocation_receipt: str | Path | None = None,
    prepared_root: str | Path,
    graph_cache_root: str | Path,
    proc_root: str | Path = "/proc",
    require_writer_audit: bool = True,
    git_identity: Mapping[str, str] | None = None,
    t3_binding: Mapping[str, Any] | None = None,
    policy_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one or more completed Taste cells to an immutable predecessor."""

    prior = _verify_authority(prior_authority_root)
    if prior["complete"] < MIN_PRIOR_COMPLETE:
        raise TasteMatrixAppendError(
            f"Taste append requires at least {MIN_PRIOR_COMPLETE}/16 predecessor cells"
        )
    if not taste_cells:
        raise TasteMatrixAppendError("At least one Taste method cell is required")
    unknown = set(taste_cells) - set(METHOD_CONTRACTS)
    if unknown:
        raise TasteMatrixAppendError(f"Unsupported Taste methods: {sorted(unknown)}")
    prior_rows = prior["rows"]
    passing = {status.value for status in PASS_STATUSES}
    for method in taste_cells:
        if str(prior_rows[(TARGET_DATASET, method)].get("status") or "") in passing:
            raise TasteMatrixAppendError(f"Prior authority already passes Taste/{method}")
    destination = Path(output_root).expanduser()
    if not destination.is_absolute() or destination.is_symlink():
        raise TasteMatrixAppendError("Matrix append output must be an absolute non-symlink path")
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise TasteMatrixAppendError(f"Matrix append output must be absent: {destination}")
    held_t3 = dict(t3_binding or _load_t3_binding(t3_root))
    if policy_binding is not None:
        scoped_policy = dict(policy_binding)
    elif policy_path_relocation_receipt is not None:
        scoped_policy = validate_t11_policy_path_relocation(
            policy_path_relocation_receipt,
            current_policy_path=policy_path,
            policy_receipt=policy_receipt,
            prepared_root=prepared_root,
            graph_cache_root=graph_cache_root,
        ).matrix_policy_binding()
    else:
        scoped_policy = _load_policy_binding(
            policy_path=policy_path,
            policy_receipt=policy_receipt,
            prepared_root=prepared_root,
            graph_cache_root=graph_cache_root,
        )
    if (
        not _SHA256_RE.fullmatch(str(held_t3.get("checkpoint_id") or ""))
        or not _SHA256_RE.fullmatch(
            str(held_t3.get("temperature_scaling_sha256") or "")
        )
        or scoped_policy.get("paper_reporting_authorized") is not True
        or scoped_policy.get("dataset_redistribution_authorized") is not False
        or scoped_policy.get("license_conclusion") != "NOT_GRANTED_OR_INFERRED"
        or scoped_policy.get("legacy_license_pass_claimed") is not False
    ):
        raise TasteMatrixAppendError("T3/policy binding is malformed")
    explicit_cells = {
        f"{dataset}/{method}": str(
            Path(str(row["standardized_output_root"])).resolve(strict=True)
        )
        for (dataset, method), row in prior_rows.items()
        if str(row.get("status") or "") in passing
    }
    for method, root in taste_cells.items():
        explicit_cells[f"{TARGET_DATASET}/{method}"] = str(
            _physical_directory(root, label=f"Taste {method} final root")
        )
    result = audit_registry(
        AuditConfig(
            scan_roots=(),
            output_root=destination,
            explicit_cells=explicit_cells,
            taste_license_gate=_taste_gate(scoped_policy),
            max_hash_bytes=MAX_HASH_BYTES,
        )
    )
    proposed = {
        (str(row["dataset"]), str(row["method"])): dict(row)
        for row in result.matrix_rows
    }
    target_keys = {(TARGET_DATASET, method) for method in taste_cells}
    for key, prior_row in prior_rows.items():
        if key in target_keys:
            continue
        # The predecessor is already a hash-closed matrix authority.  Its rows
        # may encode an explicitly reviewed publication decision that the
        # generic scanner cannot reconstruct (notably a scientifically valid
        # zero-result cell).  Re-auditing remains useful for discovering the
        # new Taste target, but it must never reinterpret or rewrite any
        # non-target row in an append-only authority chain.
        proposed[key] = dict(prior_row)
    expected_complete = int(prior["complete"]) + len(taste_cells)
    observed_complete = sum(
        str(row.get("status") or "") in passing for row in proposed.values()
    )
    if observed_complete != expected_complete:
        raise TasteMatrixAppendError(
            f"Taste append must add exactly {len(taste_cells)} cells; "
            f"prior={prior['complete']}, proposed={observed_complete}"
        )
    for method in taste_cells:
        row = proposed[(TARGET_DATASET, method)]
        if row.get("status") != CellStatus.FROZEN_PASS.value:
            raise TasteMatrixAppendError(
                f"Taste/{method} did not pass the ordinary frozen registry gate: "
                f"{row.get('rerun_reason')}"
            )
    result = replace(
        result,
        matrix_rows=_ordered_rows(proposed),
        oracle_registry=build_oracle_registry(_ordered_rows(proposed), {}),
        matrix_complete_cells=observed_complete,
    )
    taste_evidence: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        row = proposed[(TARGET_DATASET, method)]
        if str(row.get("status") or "") not in passing:
            continue
        taste_evidence[method] = _validate_taste_cell(
            row["standardized_output_root"],
            method=method,
            t3_binding=held_t3,
            proc_root=proc_root,
            require_writer_audit=require_writer_audit,
        )
    identity_sets = {
        field: {evidence["identities"][field] for evidence in taste_evidence.values()}
        for field in SHARED_FULL_FIELDS
    }
    conflict = {field: values for field, values in identity_sets.items() if len(values) != 1}
    if conflict:
        raise TasteMatrixAppendError(
            "Taste full cells do not share one T3/split/evaluation identity: "
            + ", ".join(sorted(conflict))
        )
    reference = proposed[(TARGET_DATASET, next(iter(taste_evidence)))]
    for method in taste_evidence:
        row = proposed[(TARGET_DATASET, method)]
        if any(row.get(field) != reference.get(field) for field in SHARED_REGISTRY_FIELDS):
            raise TasteMatrixAppendError(f"Taste/{method} registry identity differs")
    execution = dict(git_identity or _git_identity())
    if set(execution) != {"commit", "tree"} or any(
        not re.fullmatch(r"[0-9a-f]{40}", str(execution[field]))
        for field in ("commit", "tree")
    ):
        raise TasteMatrixAppendError("Execution Git identity is incomplete")
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
        "appended_methods": sorted(taste_cells, key=METHODS.index),
        "appended_cells": {
            method: {
                "registry_row": proposed[(TARGET_DATASET, method)],
                "terminal_evidence": taste_evidence[method],
            }
            for method in sorted(taste_cells, key=METHODS.index)
        },
        "all_passing_taste_cells": taste_evidence,
        "shared_taste_identity": {
            field: next(iter(values)) for field, values in identity_sets.items()
        },
        "shared_registry_identity_fields": list(SHARED_REGISTRY_FIELDS),
        "shared_full_identity_fields": list(SHARED_FULL_FIELDS),
        "t3_binding": held_t3,
        "scoped_policy": scoped_policy,
        "unchanged_non_target_rows": True,
        "new_matrix_complete_cells": expected_complete,
        "new_matrix_total_cells": 16,
        "new_authority_root": str(destination),
        "scientific_metrics_recomputed": False,
        "candidate_order_changed": False,
        "test_split_opened": False,
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
            raise TasteMatrixAppendError("Staged Taste matrix rows changed")
        if _read_json(staging / "append_authority.json") != receipt:
            raise TasteMatrixAppendError("Staged Taste append receipt changed")
        atomic_rename_directory_noreplace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    reopened = _verify_authority(destination, expected_complete=expected_complete)
    if reopened["rows"] != proposed:
        raise TasteMatrixAppendError("Published Taste matrix rows changed on reopen")
    return {
        "status": "PASS",
        "output_root": str(destination),
        "matrix_status_path": str(destination / "matrix_status.json"),
        "matrix_status_sha256": reopened["matrix_sha256"],
        "combined_audit_sha256": reopened["combined_sha256"],
        "matrix_complete_cells": reopened["complete"],
        "matrix_total_cells": 16,
        "appended_methods": receipt["appended_methods"],
        "marker": marker,
    }


__all__ = [
    "APPEND_SCHEMA",
    "METHOD_CONTRACTS",
    "TasteMatrixAppendError",
    "append_tastemolnet_cells",
]
