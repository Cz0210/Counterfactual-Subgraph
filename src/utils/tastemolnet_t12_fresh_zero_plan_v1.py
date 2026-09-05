"""Strict predeployment plan for fresh-from-zero T12 10k/20k production.

This is intentionally not an executor.  Every stage remains non-dispatchable
until an independently written diagnostic parity receipt is present and
validated by the future owner.  The plan reserves no GPU and creates no
science output root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence
import uuid

from src.eval.tastemolnet_t3_calibration_v2 import (
    CANDIDATE_CHECKPOINT_FILES as T3_CHECKPOINT_FILES,
    TasteT3CalibrationError,
    _parse_hash_manifest as _parse_t3_hash_manifest,
)
from src.train.tastemolnet_clean_policy_init import (
    TasteCleanPolicyError,
    _hash_regular as _hash_content_file,
    _inventory_directory as _inventory_content_tree,
    _read_regular as _read_content_file,
)

from .final16_owner_registry_v1 import (
    Final16OwnerRegistryError,
    validate_owner_registry,
)
from .main_ready_task_specs import atomic_json, stable_sha256
from .tastemolnet_t12_formal_profile_v1 import (
    FORMAL_PRODUCTION_CHECKPOINT_CURSORS,
)


SCHEMA = "tastemolnet_t12_fresh_zero_predeployment_v1"
STAGE_SCHEMA = "tastemolnet_t12_fresh_zero_stage_spec_v1"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class T12FreshZeroPlanError(RuntimeError):
    """A production predeployment would be runnable early or underspecified."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_tree_sha256(root: Path, *, field: str) -> str:
    """Use the project's existing deterministic physical-tree hash contract."""

    try:
        _inventory, digest = _inventory_content_tree(root, label=field)
    except (OSError, TasteCleanPolicyError) as exc:
        raise T12FreshZeroPlanError(f"{field} content tree is invalid") from exc
    return digest


def _gnn_checkpoint_sha256(path: Path) -> str:
    """Bind either one legacy checkpoint file or the sealed T3 checkpoint tree."""

    try:
        info = path.lstat()
        if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
            _size, digest = _hash_content_file(path, label="gnn_checkpoint")
            return digest
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise T12FreshZeroPlanError(
                "gnn_checkpoint must be a regular file or sealed T3 directory"
            )

        inventory, digest = _inventory_content_tree(
            path, label="sealed T3 gnn_checkpoint"
        )
        if set(inventory) != set(T3_CHECKPOINT_FILES):
            raise T12FreshZeroPlanError(
                "sealed T3 gnn_checkpoint inventory differs from the T3 contract"
            )
        declared = _parse_t3_hash_manifest(
            _read_content_file(
                path / "sha256sums.txt",
                label="sealed T3 gnn_checkpoint sha256sums.txt",
            ),
            expected=T3_CHECKPOINT_FILES - {"sha256sums.txt"},
            label="sealed T3 gnn_checkpoint sha256sums.txt",
        )
        if any(
            inventory[name]["sha256"] != expected
            for name, expected in declared.items()
        ):
            raise T12FreshZeroPlanError(
                "sealed T3 gnn_checkpoint file differs from sha256sums.txt"
            )
        inventory_after, digest_after = _inventory_content_tree(
            path, label="sealed T3 gnn_checkpoint revalidation"
        )
        if inventory_after != inventory or digest_after != digest:
            raise T12FreshZeroPlanError(
                "sealed T3 gnn_checkpoint changed while it was bound"
            )
        return digest
    except T12FreshZeroPlanError:
        raise
    except (OSError, TasteCleanPolicyError, TasteT3CalibrationError) as exc:
        raise T12FreshZeroPlanError("gnn_checkpoint binding failed closed") from exc


def _absolute(path: Path, *, field: str, must_exist: bool = False) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise T12FreshZeroPlanError(f"{field} must be an absolute non-symlink path")
    resolved = path.resolve(strict=must_exist)
    if must_exist and not resolved.exists():
        raise T12FreshZeroPlanError(f"{field} is absent")
    return resolved


def _stage(
    *, name: str, command: Sequence[str], cwd: Path, gpu: int | None,
    inputs: Mapping[str, str], outputs: Mapping[str, str], predecessor: str | None,
) -> dict[str, Any]:
    if not command or not all(type(item) is str and item for item in command):
        raise T12FreshZeroPlanError(f"{name} command is incomplete")
    value = {
        "schema_version": STAGE_SCHEMA,
        "stage_id": name,
        "status": "BLOCKED_WAITING_DIAGNOSTIC_PARITY",
        "dispatchable": False,
        "cwd": str(cwd),
        "command": list(command),
        "gpu": gpu,
        "inputs": dict(inputs),
        "outputs": dict(outputs),
        "required_predecessor": predecessor,
        "science_output_created": False,
        "matrix_write_allowed": False,
        "stage_sha256": "0" * 64,
    }
    value["stage_sha256"] = stable_sha256(
        {key: item for key, item in value.items() if key != "stage_sha256"}
    )
    return value


def build_fresh_zero_plan(
    *,
    repo_root: Path,
    python: Path,
    config: Path,
    execution_commit: str,
    attempt_id: str,
    generation_token: str,
    gpu_index: int,
    gpu_uuid: str,
    diagnostic_terminal: Path,
    required_parity_receipt: Path,
    managed_neurosed_root: Path,
    t3_root: Path,
    official_root: Path,
    threshold_authority: Path,
    replay_gate: Path,
    production_root: Path,
    postprocess_root: Path,
    train_csv: Path,
    calibration_csv: Path,
    test_csv: Path,
    gnn_checkpoint: Path,
    molclr_root: Path,
    molclr_checkpoint: Path,
    threshold_contract: Path,
    wnode_cache_db: Path,
    node_embedding_cache_dir: Path,
    verification_root: Path,
    publisher_id: str,
    publisher_locator: Path,
    owner_registry: Path,
    expected_owner_registry_sha256: str,
    expected_owner_registry_file_sha256: str,
    matrix_authority_root: Path,
    diagnostic_bridge_history_root: Path,
    nvme_disposable_index_root: Path,
) -> dict[str, Any]:
    root = _absolute(repo_root, field="repo_root", must_exist=True)
    executable = _absolute(python, field="python", must_exist=True)
    cfg = _absolute(config, field="config", must_exist=True)
    diagnostic = _absolute(diagnostic_terminal, field="diagnostic_terminal", must_exist=True)
    try:
        diagnostic_value = json.loads(diagnostic.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T12FreshZeroPlanError("diagnostic terminal is unreadable") from exc
    if (
        diagnostic_value.get("status")
        != "DIAGNOSTIC_510_RECONCILED_ENGINEERING_FAILURE"
        or diagnostic_value.get("diagnostic_only") is not True
        or diagnostic_value.get("promotion_allowed") is not False
        or diagnostic_value.get("paper_cell_pass") is not False
        or diagnostic_value.get("terminal_sha256")
        != stable_sha256(
            {
                key: value
                for key, value in diagnostic_value.items()
                if key != "terminal_sha256"
            }
        )
    ):
        raise T12FreshZeroPlanError("diagnostic terminal cannot authorize production")
    if required_parity_receipt.exists() or required_parity_receipt.is_symlink():
        raise T12FreshZeroPlanError(
            "predeployment expects a future fresh parity receipt, not an unvalidated file"
        )
    for path, label in (
        (production_root, "production_root"),
        (postprocess_root, "postprocess_root"),
        (verification_root, "verification_root"),
    ):
        _absolute(path, field=label)
        if path.exists() or path.is_symlink():
            raise T12FreshZeroPlanError(f"{label} must remain fresh before dispatch")
    for path, label in (
        (managed_neurosed_root, "managed_neurosed_root"),
        (t3_root, "t3_root"),
        (official_root, "official_root"),
        (threshold_authority, "threshold_authority"),
        (replay_gate, "replay_gate"),
        (train_csv, "train_csv"),
        (calibration_csv, "calibration_csv"),
        (test_csv, "test_csv"),
        (gnn_checkpoint, "gnn_checkpoint"),
        (molclr_root, "molclr_root"),
        (molclr_checkpoint, "molclr_checkpoint"),
        (threshold_contract, "threshold_contract"),
        (matrix_authority_root, "matrix_authority_root"),
    ):
        _absolute(path, field=label, must_exist=True)
    for path, label in (
        (required_parity_receipt, "required_parity_receipt"),
        (wnode_cache_db, "wnode_cache_db"),
        (node_embedding_cache_dir, "node_embedding_cache_dir"),
        (publisher_locator, "publisher_locator"),
    ):
        _absolute(path, field=label)
    registry_path = _absolute(owner_registry, field="owner_registry", must_exist=True)
    if (
        _SHA256.fullmatch(expected_owner_registry_file_sha256) is None
        or file_sha256(registry_path) != expected_owner_registry_file_sha256
    ):
        raise T12FreshZeroPlanError("canonical owner registry file binding changed")
    try:
        registry_raw = json.loads(registry_path.read_text(encoding="utf-8"))
        registry = validate_owner_registry(
            registry_raw, check_processes=False
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, Final16OwnerRegistryError) as exc:
        raise T12FreshZeroPlanError("canonical owner registry is invalid") from exc
    if (
        registry.get("self_sha256") != expected_owner_registry_sha256
        or registry.get("matrix_authority_root") != str(matrix_authority_root)
    ):
        raise T12FreshZeroPlanError("canonical owner registry binding changed")
    cell_claims = [
        row
        for row in registry["publishers"]
        if row["cell_id"] == "TasteMolNet/GCFExplainer"
        and row["claim_enabled"] is True
    ]
    if (
        len(cell_claims) != 1
        or cell_claims[0]["publisher_id"] != publisher_id
        or cell_claims[0]["locator"] != str(publisher_locator)
    ):
        raise T12FreshZeroPlanError("T12 canonical publisher is not unique or changed")
    locator = _absolute(publisher_locator, field="publisher_locator")
    if locator.exists() and not locator.is_file():
        raise T12FreshZeroPlanError("publisher locator must be a regular file when present")
    locator_present = locator.exists()
    locator_sha256 = file_sha256(locator) if locator_present else None
    bridge_history = _absolute(
        diagnostic_bridge_history_root,
        field="diagnostic_bridge_history_root",
        must_exist=True,
    )
    if not bridge_history.is_dir():
        raise T12FreshZeroPlanError("diagnostic bridge history is not a directory")
    gnn_checkpoint_sha256 = _gnn_checkpoint_sha256(gnn_checkpoint)
    disposable_index = _absolute(
        nvme_disposable_index_root,
        field="nvme_disposable_index_root",
    )
    try:
        disposable_index.relative_to(Path("/root/autodl-tmp"))
    except ValueError as exc:
        raise T12FreshZeroPlanError(
            "T12 disposable index must be staged under /root/autodl-tmp"
        ) from exc
    if gpu_index < 0 or not gpu_uuid.startswith("GPU-"):
        raise T12FreshZeroPlanError("production GPU identity is invalid")
    try:
        parsed_attempt = uuid.UUID(attempt_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise T12FreshZeroPlanError("production attempt ID is invalid") from exc
    if (
        parsed_attempt.version != 4
        or str(parsed_attempt) != attempt_id
        or _GIT_SHA.fullmatch(execution_commit) is None
        or _SHA256.fullmatch(generation_token) is None
    ):
        raise T12FreshZeroPlanError("production identity is invalid")
    generation_entrypoint = [
        str(executable), "-I", "-B", str(root / "scripts/run_tastemolnet_gcf_full.py"),
    ]
    common = [
        "--config", str(cfg), "--set", "inference.fallback_to_heuristic=false",
        "--output-root", str(production_root), "--attempt-id", attempt_id,
        "--generation-token", generation_token, "--gpu-uuid", gpu_uuid,
        "--managed-neurosed-root", str(managed_neurosed_root),
        "--t3-root", str(t3_root), "--official-root", str(official_root),
        "--neurosed-threshold-authority", str(threshold_authority),
        "--exact-replay-gate", str(replay_gate),
        "--formal-checkpoint-cadence",
        "--disposable-index-root", str(disposable_index),
    ]
    cursors = tuple(FORMAL_PRODUCTION_CHECKPOINT_CURSORS)
    checkpoints = {
        cursor: production_root
        / "checkpoints"
        / f"checkpoint-{cursor:08d}.manifest.json"
        for cursor in cursors
    }
    checkpoint_20k = checkpoints[cursors[-1]]
    generation_verification = production_root / "generation_verification"
    shared_inputs = {
        "diagnostic_terminal": str(diagnostic),
        "required_future_parity_receipt": str(required_parity_receipt),
        "replay_gate": str(replay_gate),
    }
    stages: list[dict[str, Any]] = []
    predecessor: str | None = None
    previous_cursor = 0
    for cursor in cursors:
        stage_id = (
            f"T12_FRESH_FROM_ZERO_TO_{cursor:08d}"
            if previous_cursor == 0
            else f"T12_RESUME_{previous_cursor:08d}_TO_{cursor:08d}"
        )
        command = [*generation_entrypoint, "--mode", "fresh", *common]
        stage_inputs = dict(shared_inputs)
        if previous_cursor:
            command = [
                *generation_entrypoint,
                "--mode",
                "resume",
                "--checkpoint-manifest",
                str(checkpoints[previous_cursor]),
                *common,
            ]
            stage_inputs[f"checkpoint_{previous_cursor}"] = str(
                checkpoints[previous_cursor]
            )
        outputs = {"checkpoint": str(checkpoints[cursor])}
        if cursor == cursors[-1]:
            outputs["candidate_manifest"] = str(
                production_root
                / "native_candidates/native-candidates-00020000.manifest.json"
            )
        stages.append(
            _stage(
                name=stage_id,
                command=command,
                cwd=root,
                gpu=gpu_index,
                inputs=stage_inputs,
                outputs=outputs,
                predecessor=predecessor,
            )
        )
        predecessor = stage_id
        previous_cursor = cursor
    stages.extend([
        _stage(
            name="T12_GENERATION_VERIFY",
            command=[str(executable), "-I", "-B", str(root / "scripts/verify_tastemolnet_gcf_full_generation.py"), "--config", str(cfg), "--set", "inference.fallback_to_heuristic=false", "--formal-checkpoint-cadence", "--production-root", str(production_root), "--output-root", str(generation_verification)],
            cwd=root, gpu=None, inputs={"checkpoint_20k": str(checkpoint_20k)},
            outputs={"pass": str(generation_verification / "GENERATION_PASS")},
            predecessor=predecessor,
        ),
        _stage(
            name="T12_POSTPROCESS",
            command=[str(executable), "-I", "-B", str(root / "scripts/run_tastemolnet_gcf_full_postprocess.py"), "--config", str(cfg), "--set", "inference.fallback_to_heuristic=false", "--formal-checkpoint-cadence", "--generation-root", str(production_root), "--generation-verification-root", str(generation_verification), "--train-csv", str(train_csv), "--calibration-csv", str(calibration_csv), "--test-csv", str(test_csv), "--gnn-checkpoint", str(gnn_checkpoint), "--molclr-root", str(molclr_root), "--molclr-checkpoint", str(molclr_checkpoint), "--threshold-contract", str(threshold_contract), "--output-root", str(postprocess_root), "--wnode-cache-db", str(wnode_cache_db), "--node-embedding-cache-dir", str(node_embedding_cache_dir), "--device", "cuda:0"],
            cwd=root, gpu=gpu_index,
            inputs={"generation_pass": str(generation_verification / "GENERATION_PASS")},
            outputs={"postprocess_root": str(postprocess_root)}, predecessor="T12_GENERATION_VERIFY",
        ),
        _stage(
            name="T12_TERMINAL_VERIFY",
            command=[str(executable), "-I", "-B", str(root / "scripts/verify_tastemolnet_gcf_full.py"), "--config", str(cfg), "--set", "inference.fallback_to_heuristic=false", "--formal-checkpoint-cadence", "--generation-root", str(production_root), "--generation-verification-root", str(generation_verification), "--train-csv", str(train_csv), "--calibration-csv", str(calibration_csv), "--test-csv", str(test_csv), "--gnn-checkpoint", str(gnn_checkpoint), "--molclr-root", str(molclr_root), "--molclr-checkpoint", str(molclr_checkpoint), "--threshold-contract", str(threshold_contract), "--output-root", str(postprocess_root), "--verification-root", str(verification_root), "--device", "cuda:0"],
            cwd=root, gpu=gpu_index, inputs={"postprocess_root": str(postprocess_root)},
            outputs={"pass": str(verification_root / "PASS")}, predecessor="T12_POSTPROCESS",
        ),
    ])
    plan = {
        "schema_version": SCHEMA,
        "status": "BLOCKED_WAITING_DIAGNOSTIC_PARITY",
        "dispatchable": False,
        "execution_commit": execution_commit,
        "attempt_id": attempt_id,
        "generation_token": generation_token,
        "fresh_from_zero": True,
        "source_checkpoint": None,
        "production_steps": list(cursors),
        "formal_checkpoint_cadence": True,
        "checkpoint_cadence_complete": True,
        "nvme_staging": {
            "disposable_index_root": str(disposable_index),
            "disposable_index_authoritative": False,
            "diagnostic_bridge_history_source": str(bridge_history),
            "diagnostic_bridge_history_source_sha256": _content_tree_sha256(
                bridge_history,
                field="diagnostic bridge history",
            ),
            "diagnostic_bridge_history_staging_root": str(
                disposable_index / "diagnostic_bridge_history"
            ),
            "diagnostic_bridge_history_use": "PARITY_EVIDENCE_ONLY",
            "production_bridge_history_root": str(production_root / "bridge_history"),
            "production_bridge_history_authoritative": True,
            "production_reuses_diagnostic_history": False,
        },
        "input_hash_bindings": {
            "config": file_sha256(cfg),
            "diagnostic_terminal": file_sha256(diagnostic),
            "owner_registry_file": expected_owner_registry_file_sha256,
            "owner_registry_self": expected_owner_registry_sha256,
            "publisher_locator": locator_sha256,
            "publisher_locator_path": str(publisher_locator),
            "publisher_locator_present": locator_present,
            "publisher_locator_binding_state": (
                "PRESENT_SHA256_BOUND"
                if locator_present
                else "ABSENT_EXACT_PATH_BOUND"
            ),
            "replay_gate": file_sha256(replay_gate),
            "threshold_authority": file_sha256(threshold_authority),
            "train_csv": file_sha256(train_csv),
            "calibration_csv": file_sha256(calibration_csv),
            "test_csv": file_sha256(test_csv),
            "gnn_checkpoint": gnn_checkpoint_sha256,
            "molclr_checkpoint": file_sha256(molclr_checkpoint),
            "threshold_contract": file_sha256(threshold_contract),
        },
        "source_sha_binding": execution_commit,
        "diagnostic_steps_never_promoted": [250, 500, 510],
        "diagnostic_terminal": str(diagnostic),
        "diagnostic_terminal_sha256": file_sha256(diagnostic),
        "required_future_parity_receipt": str(required_parity_receipt),
        "required_future_parity_status": "T12_DIAGNOSTIC_PARITY_PASS",
        "parity_receipt_present": False,
        "stages": stages,
        "publisher_handoff": {
            "status": "BLOCKED_WAITING_PARITY_AND_EXCLUSIVE_OWNER_TRANSFER",
            "publisher_id": publisher_id,
            "canonical_locator": str(publisher_locator),
            "canonical_locator_present": locator_present,
            "canonical_locator_sha256": locator_sha256,
            "canonical_locator_creation_allowed": False,
            "matrix_authority_root": str(matrix_authority_root),
            "owner_registry": str(registry_path),
            "owner_registry_sha256": expected_owner_registry_sha256,
            "canonical_publisher_snapshot": dict(cell_claims[0]),
            "new_publisher_created": False,
            "dispatchable": False,
            "required_terminal_pass": str(verification_root / "PASS"),
            "cell": "TasteMolNet/GCFExplainer",
            "requires_exact_existing_owner_retirement": True,
            "requires_active_writer_count_zero": True,
            "requires_atomic_registry_owner_transfer": True,
        },
        "gpu_lease_acquired": False,
        "science_started": False,
        "plan_sha256": "0" * 64,
    }
    plan["plan_sha256"] = stable_sha256(
        {key: item for key, item in plan.items() if key != "plan_sha256"}
    )
    return plan


def write_plan_bundle(root: Path, plan: Mapping[str, Any]) -> None:
    if root.exists() or root.is_symlink():
        raise T12FreshZeroPlanError("predeployment spec root must be fresh")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    for stage in plan["stages"]:
        atomic_json(root / f"{stage['stage_id'].lower()}.json", stage)
    atomic_json(root / "fresh_zero_plan.json", plan)


__all__ = ["SCHEMA", "T12FreshZeroPlanError", "build_fresh_zero_plan", "write_plan_bundle"]
