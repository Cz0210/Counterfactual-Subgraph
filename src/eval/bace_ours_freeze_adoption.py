"""Receipt-only adoption of the already-frozen BACE Ours paper cell."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from src.eval.am_legacy_standardization import scan_live_writers
from src.eval.four_by_four_registry import CellStatus, audit_explicit_candidate
from src.utils.terminal_publisher_v2 import _atomic_rename_noreplace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = (
    PROJECT_ROOT / "configs/autodl/bace_ours_freeze_adoption_v1.json"
)
POLICY_SCHEMA = "bace_ours_freeze_adoption_policy_v1"
ADOPTION_SCHEMA = "bace_ours_frozen_adoption_manifest_v1"
VERIFICATION_SCHEMA = "bace_ours_frozen_adoption_verification_v1"
PASS_MARKER = "[BACE_OURS_FREEZE_ADOPTION_PASS]"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BACEOursFreezeAdoptionError(RuntimeError):
    """The frozen candidate cannot be adopted without recomputation."""


@dataclass(frozen=True)
class AdoptionPolicy:
    path: Path
    sha256: str
    source_root: Path
    writer_guard_root: Path
    expected_identity: dict[str, Any]
    source_files: dict[str, str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BACEOursFreezeAdoptionError(f"JSON object required: {path}")
    return dict(value)


def _physical_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise BACEOursFreezeAdoptionError(f"{label} may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BACEOursFreezeAdoptionError(f"{label} is absent: {path}") from exc
    if not resolved.is_dir():
        raise BACEOursFreezeAdoptionError(f"{label} is not a directory")
    return resolved


def load_policy(path: str | Path = DEFAULT_POLICY) -> AdoptionPolicy:
    source = Path(path).expanduser().resolve(strict=True)
    payload_bytes = source.read_bytes()
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict) or payload.get("schema_version") != POLICY_SCHEMA:
        raise BACEOursFreezeAdoptionError("BACE Ours adoption policy schema changed")
    if payload.get("dataset") != "BACE" or payload.get("method") != "Ours":
        raise BACEOursFreezeAdoptionError("adoption policy scope is not BACE/Ours")
    expected = payload.get("expected_identity")
    files = payload.get("source_files")
    if not isinstance(expected, dict) or not isinstance(files, dict) or not files:
        raise BACEOursFreezeAdoptionError("adoption policy closure is incomplete")
    normalized_files = {str(name): str(digest) for name, digest in files.items()}
    if any(
        not name or Path(name).name != name or not _SHA256_RE.fullmatch(digest)
        for name, digest in normalized_files.items()
    ):
        raise BACEOursFreezeAdoptionError("adoption source inventory is malformed")
    required_identity = {
        "cf_mode",
        "classifier_family",
        "dataset_hash",
        "feature_schema_sha256",
        "k_max",
        "method",
        "molclr_checkpoint_hash",
        "num_classes",
        "oracle_backend",
        "oracle_checkpoint",
        "oracle_hash",
        "rf_oracle_used",
        "source_label",
        "split_hash",
        "table2_k",
        "temperature_scaling_sha256",
        "threshold_config_hash",
    }
    if set(expected) != required_identity:
        raise BACEOursFreezeAdoptionError("pinned identity field set changed")
    fixed_identity = {
        "cf_mode": "strict_flip",
        "classifier_family": "gine",
        "k_max": 20,
        "method": "Ours",
        "num_classes": 2,
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "source_label": 1,
        "table2_k": 10,
    }
    if any(expected.get(field) != value for field, value in fixed_identity.items()):
        raise BACEOursFreezeAdoptionError("fixed BACE Ours identity changed")
    hash_fields = {
        "dataset_hash",
        "feature_schema_sha256",
        "molclr_checkpoint_hash",
        "oracle_hash",
        "split_hash",
        "temperature_scaling_sha256",
        "threshold_config_hash",
    }
    if any(
        not isinstance(expected.get(field), str)
        or not _SHA256_RE.fullmatch(expected[field])
        for field in hash_fields
    ):
        raise BACEOursFreezeAdoptionError("pinned BACE Ours hash identity is malformed")
    if not Path(str(expected.get("oracle_checkpoint") or "")).is_absolute():
        raise BACEOursFreezeAdoptionError("pinned BACE oracle checkpoint is not absolute")
    source_root = Path(str(payload.get("source_root") or ""))
    writer_guard_root = Path(str(payload.get("writer_guard_root") or ""))
    if not source_root.is_absolute() or not writer_guard_root.is_absolute():
        raise BACEOursFreezeAdoptionError("pinned BACE source roots must be absolute")
    return AdoptionPolicy(
        path=source,
        sha256=_sha256_bytes(payload_bytes),
        source_root=source_root,
        writer_guard_root=writer_guard_root,
        expected_identity=dict(expected),
        source_files=normalized_files,
    )


def _stat_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    stat = path.stat(follow_symlinks=False)
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_nlink,
    )


def _source_inventory(policy: AdoptionPolicy) -> dict[str, dict[str, Any]]:
    root = _physical_directory(policy.source_root, label="BACE Ours source root")
    observed_names = sorted(entry.name for entry in root.iterdir())
    if observed_names != sorted(policy.source_files):
        raise BACEOursFreezeAdoptionError("BACE Ours source file set changed")
    result: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in sorted(policy.source_files.items()):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise BACEOursFreezeAdoptionError(f"source file is not physical: {name}")
        before = _stat_identity(path)
        digest = _sha256_file(path)
        after = _stat_identity(path)
        if before != after or digest != expected_sha256:
            raise BACEOursFreezeAdoptionError(f"source identity changed: {name}")
        result[name] = {
            "bytes": before[2],
            "sha256": digest,
            "device": before[0],
            "inode": before[1],
            "mtime_ns": before[3],
            "ctime_ns": before[4],
            "link_count": before[5],
        }
    if (root / "PASS").read_bytes() != b"PASS\n":
        raise BACEOursFreezeAdoptionError("source PASS marker changed")
    return result


def _expectations(policy: AdoptionPolicy) -> dict[str, Any]:
    expected = policy.expected_identity
    return {
        "datasets": {
            "BACE": {
                "oracle_backend": expected["oracle_backend"],
                "classifier_family": expected["classifier_family"],
                "oracle_checkpoint": expected["oracle_checkpoint"],
                "oracle_hash": expected["oracle_hash"],
                "dataset_hash": expected["dataset_hash"],
                "split_hash": expected["split_hash"],
                "molclr_checkpoint_hash": expected["molclr_checkpoint_hash"],
                "threshold_config_hash": expected["threshold_config_hash"],
                "num_classes": expected["num_classes"],
                "source_label": expected["source_label"],
                "rf_oracle_used": expected["rf_oracle_used"],
            }
        }
    }


def validate_source_candidate(
    policy: AdoptionPolicy,
    *,
    proc_root: str | Path = "/proc",
    require_writer_audit: bool = True,
) -> dict[str, Any]:
    source = _physical_directory(policy.source_root, label="BACE Ours source root")
    writer_root = _physical_directory(policy.writer_guard_root, label="BACE Ours writer root")
    inventory = _source_inventory(policy)
    writer_audits: list[dict[str, Any]] = []
    if require_writer_audit:
        writer_audits = [
            scan_live_writers(source, proc_root=proc_root),
            scan_live_writers(writer_root, proc_root=proc_root),
        ]
    candidate = audit_explicit_candidate(
        source,
        dataset="BACE",
        method="Ours",
        expectations=_expectations(policy),
    )
    if candidate.status is not CellStatus.FROZEN_PASS or candidate.reason_codes:
        raise BACEOursFreezeAdoptionError(
            "ordinary registry gate rejected BACE Ours: "
            + ";".join(candidate.reason_codes)
        )
    row = candidate.row
    expected = policy.expected_identity
    row_fields = {
        "cf_mode": "cf_mode",
        "classifier_family": "classifier_family",
        "dataset_hash": "dataset_hash",
        "k_max": "k_max",
        "method": "method",
        "molclr_checkpoint_hash": "molclr_checkpoint_hash",
        "oracle_backend": "oracle_backend",
        "oracle_checkpoint": "oracle_checkpoint",
        "oracle_hash": "oracle_hash",
        "split_hash": "split_hash",
        "table2_k": "table2_k",
        "threshold_config_hash": "threshold_config_hash",
    }
    for expected_field, row_field in row_fields.items():
        if row.get(row_field) != expected[expected_field]:
            raise BACEOursFreezeAdoptionError(
                f"pinned candidate identity changed: {expected_field}"
            )
    oracle = _read_json(source / "oracle_manifest.json")
    summary = _read_json(source / "summary.json")
    evaluation = _read_json(source / "evaluation_manifest.json")
    audit = _read_json(source / "final_artifact_audit.json")
    if (
        oracle.get("feature_schema_sha256") != expected["feature_schema_sha256"]
        or oracle.get("temperature_scaling_sha256")
        != expected["temperature_scaling_sha256"]
        or summary.get("num_classes") != expected["num_classes"]
        or summary.get("source_label") != expected["source_label"]
        or summary.get("rf_oracle_used") is not False
        or summary.get("selection_frozen_before_test") is not True
        or summary.get("selector_fitted_on_calibration") is not True
        or summary.get("test_loaded_only_after_freeze") is not True
        or summary.get("test_used_for_selection") is not False
        or summary.get("threshold_fitted_on_test") is not False
        or evaluation.get("candidate_order_changed") is not False
        or evaluation.get("scientific_metrics_recomputed") is not False
        or audit.get("final_artifact_audit_passed") is not True
        or audit.get("hash_closure_complete") is not True
        or audit.get("no_numeric_imputation") is not True
    ):
        raise BACEOursFreezeAdoptionError("BACE Ours freeze/test boundary changed")
    after = _source_inventory(policy)
    if inventory != after:
        raise BACEOursFreezeAdoptionError("BACE Ours source drifted during verification")
    return {
        "source_root": str(source),
        "writer_guard_root": str(writer_root),
        "source_inventory": inventory,
        "source_inventory_sha256": _stable_sha256(inventory),
        "writer_audits": writer_audits,
        "registry_status": candidate.status.value,
        "registry_row": dict(row),
        "registry_reason_codes": list(candidate.reason_codes),
        "scientific_recomputation_performed": False,
        "numeric_values_changed": False,
        "raw_test_opened": False,
    }


def _git_identity(project_root: Path) -> dict[str, str]:
    head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if not re.fullmatch(r"[0-9a-f]{40}", head) or dirty:
        raise BACEOursFreezeAdoptionError("adoption requires a clean committed worktree")
    return {"commit": head, "tree": tree}


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_recorded_writer_audits(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise BACEOursFreezeAdoptionError("recorded writer audit closure changed")
    for audit in value:
        if (
            not isinstance(audit, dict)
            or audit.get("procfs_verified") is not True
            or not isinstance(audit.get("scanned_process_count"), int)
            or audit["scanned_process_count"] < 0
            or audit.get("writable_fd_count") != 0
            or audit.get("writers") != []
        ):
            raise BACEOursFreezeAdoptionError("recorded writer audit closure changed")


def adopt_bace_ours_frozen_cell(
    *,
    matrix_root: str | Path,
    output_root: str | Path,
    policy: AdoptionPolicy | None = None,
    proc_root: str | Path = "/proc",
    require_clean_git: bool = True,
) -> dict[str, Any]:
    policy = policy or load_policy()
    matrix = _physical_directory(Path(matrix_root).expanduser(), label="matrix root")
    raw_destination = Path(output_root).expanduser()
    if not raw_destination.is_absolute():
        raise BACEOursFreezeAdoptionError("BACE Ours adoption root must be absolute")
    adoptions = matrix / "adoptions"
    if adoptions.is_symlink():
        raise BACEOursFreezeAdoptionError("matrix adoptions root may not be a symlink")
    adoptions.mkdir(parents=True, exist_ok=True)
    expected_parent = _physical_directory(adoptions, label="matrix adoptions root")
    destination = raw_destination.resolve(strict=False)
    if destination.parent != expected_parent or not destination.name.startswith(
        "bace_ours_frozen_"
    ):
        raise BACEOursFreezeAdoptionError(
            "BACE Ours adoption must be one fresh bace_ours_frozen_* child"
        )
    if destination.exists() or destination.is_symlink():
        raise BACEOursFreezeAdoptionError("BACE Ours adoption root must be absent")
    evidence = validate_source_candidate(
        policy,
        proc_root=proc_root,
        require_writer_audit=True,
    )
    git_identity = (
        _git_identity(PROJECT_ROOT)
        if require_clean_git
        else {"commit": "TEST_ONLY", "tree": "TEST_ONLY"}
    )
    created_at = datetime.now(timezone.utc).isoformat()
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=expected_parent)
    )
    try:
        adoption_manifest = {
            "schema_version": ADOPTION_SCHEMA,
            "status": "PASS",
            "dataset": "BACE",
            "method": "Ours",
            "adoption_kind": "RECEIPT_ONLY_EXISTING_FROZEN_CELL",
            "source_standardized_root": evidence["source_root"],
            "source_inventory": evidence["source_inventory"],
            "source_inventory_sha256": evidence["source_inventory_sha256"],
            "policy_path": str(policy.path),
            "policy_sha256": policy.sha256,
            "execution_git": git_identity,
            "created_at": created_at,
            "scientific_recomputation_performed": False,
            "numeric_values_changed": False,
            "matrix_cell_count_increment": 1,
        }
        adoption_bytes = _json_bytes(adoption_manifest)
        _write_new(temporary / "adoption_manifest.json", adoption_bytes)
        verification = {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "PASS",
            "independent_registry_verifier": True,
            "dataset": "BACE",
            "method": "Ours",
            "ordinary_registry_status": evidence["registry_status"],
            "ordinary_registry_reason_codes": evidence["registry_reason_codes"],
            "ordinary_registry_row": evidence["registry_row"],
            "writer_audits": evidence["writer_audits"],
            "source_inventory_sha256": evidence["source_inventory_sha256"],
            "adoption_manifest_sha256": _sha256_bytes(adoption_bytes),
            "policy_sha256": policy.sha256,
            "rf_oracle_used": False,
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "scientific_recomputation_performed": False,
            "numeric_values_changed": False,
        }
        _write_new(temporary / "verification.json", _json_bytes(verification))
        _write_new(temporary / "PASS", f"{PASS_MARKER}\n".encode("ascii"))
        _fsync_directory(temporary)
        parent_descriptor = os.open(expected_parent, os.O_RDONLY)
        try:
            _atomic_rename_noreplace(
                source_parent_descriptor=parent_descriptor,
                source_name=temporary.name,
                destination_parent_descriptor=parent_descriptor,
                destination_name=destination.name,
            )
            _fsync_directory(expected_parent)
        finally:
            os.close(parent_descriptor)
        validate_adoption_receipt(
            destination,
            policy=policy,
            proc_root=proc_root,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "PASS",
        "output_root": str(destination),
        "source_standardized_root": evidence["source_root"],
        "source_inventory_sha256": evidence["source_inventory_sha256"],
        "marker": PASS_MARKER,
    }


def validate_adoption_receipt(
    root: str | Path,
    *,
    policy: AdoptionPolicy | None = None,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    policy = policy or load_policy()
    output = _physical_directory(Path(root).expanduser(), label="adoption root")
    if sorted(entry.name for entry in output.iterdir()) != [
        "PASS",
        "adoption_manifest.json",
        "verification.json",
    ]:
        raise BACEOursFreezeAdoptionError("adoption receipt file set changed")
    if any(entry.is_symlink() or not entry.is_file() for entry in output.iterdir()):
        raise BACEOursFreezeAdoptionError("adoption receipt files must be physical")
    if (output / "PASS").read_bytes() != f"{PASS_MARKER}\n".encode("ascii"):
        raise BACEOursFreezeAdoptionError("adoption PASS marker changed")
    manifest_bytes = (output / "adoption_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    verification = _read_json(output / "verification.json")
    evidence = validate_source_candidate(
        policy,
        proc_root=proc_root,
        require_writer_audit=True,
    )
    execution_git = manifest.get("execution_git") if isinstance(manifest, dict) else None
    if not isinstance(execution_git, dict) or set(execution_git) != {"commit", "tree"}:
        raise BACEOursFreezeAdoptionError("adoption execution identity changed")
    if not all(
        value == "TEST_ONLY" or bool(re.fullmatch(r"[0-9a-f]{40}", str(value)))
        for value in execution_git.values()
    ):
        raise BACEOursFreezeAdoptionError("adoption execution identity changed")
    _validate_recorded_writer_audits(verification.get("writer_audits"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != ADOPTION_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("dataset") != "BACE"
        or manifest.get("method") != "Ours"
        or manifest.get("adoption_kind")
        != "RECEIPT_ONLY_EXISTING_FROZEN_CELL"
        or manifest.get("policy_sha256") != policy.sha256
        or manifest.get("policy_path") != str(policy.path)
        or manifest.get("source_standardized_root") != evidence["source_root"]
        or manifest.get("source_inventory") != evidence["source_inventory"]
        or manifest.get("source_inventory_sha256")
        != evidence["source_inventory_sha256"]
        or manifest.get("scientific_recomputation_performed") is not False
        or manifest.get("numeric_values_changed") is not False
        or manifest.get("matrix_cell_count_increment") != 1
        or verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("status") != "PASS"
        or verification.get("independent_registry_verifier") is not True
        or verification.get("dataset") != "BACE"
        or verification.get("method") != "Ours"
        or verification.get("ordinary_registry_status") != "FROZEN_PASS"
        or verification.get("ordinary_registry_reason_codes") != []
        or verification.get("ordinary_registry_row") != evidence["registry_row"]
        or verification.get("source_inventory_sha256")
        != evidence["source_inventory_sha256"]
        or verification.get("adoption_manifest_sha256")
        != _sha256_bytes(manifest_bytes)
        or verification.get("policy_sha256") != policy.sha256
        or verification.get("rf_oracle_used") is not False
        or verification.get("selection_frozen_before_test") is not True
        or verification.get("test_used_for_selection") is not False
        or verification.get("threshold_fitted_on_test") is not False
        or verification.get("scientific_recomputation_performed") is not False
        or verification.get("numeric_values_changed") is not False
    ):
        raise BACEOursFreezeAdoptionError("adoption receipt closure changed")
    return {
        "status": "PASS",
        "output_root": str(output),
        "source_standardized_root": evidence["source_root"],
        "source_inventory_sha256": evidence["source_inventory_sha256"],
        "marker": PASS_MARKER,
    }


__all__ = [
    "ADOPTION_SCHEMA",
    "AdoptionPolicy",
    "BACEOursFreezeAdoptionError",
    "DEFAULT_POLICY",
    "PASS_MARKER",
    "VERIFICATION_SCHEMA",
    "adopt_bace_ours_frozen_cell",
    "load_policy",
    "validate_adoption_receipt",
    "validate_source_candidate",
]
