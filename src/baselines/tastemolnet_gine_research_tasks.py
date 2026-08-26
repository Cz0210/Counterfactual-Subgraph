"""Typed, fresh TasteMolNet GINE research task fragment.

The scoped project-owner policy permits private research computation and
aggregate paper reporting while preserving the upstream licence uncertainty
and a strict no-data-redistribution boundary.  A typed local-authority receipt
is required before this builder emits an executable task.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from src.utils.tastemolnet_research_policy import (
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    validate_tastemolnet_local_authority,
    validate_tastemolnet_policy_receipt,
)


FRAGMENT_SCHEMA = "tastemolnet_gine_research_controller_fragment_v1"
TASK_SCHEMA = "tastemolnet_gine_full_research_task_v1"
TASK_ID = "tastemolnet_gine_full_research_v1"
STAGE = "TASTEMOLNET_GINE_FULL_RESEARCH_V1"
PENDING_REASON = "TASTEMOLNET_POLICY_PENDING_ROOT_ACTIVATION"
REQUIRED_OUTPUT_FILES = (
    "model.pt",
    "model_card.json",
    "feature_schema.json",
    "training_metrics.json",
    "test_evaluation_status.json",
    "temperature_scaling.json",
    "sha256sums.txt",
    "data_use_policy_binding.json",
    "graph_cache_usage.json",
    "oracle_manifest.json",
)


def _absolute(value: str | Path | None, *, field: str, required: bool) -> Path | None:
    if value is None:
        if required:
            raise TasteResearchPolicyError(f"{field} is required")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise TasteResearchPolicyError(f"{field} must be absolute")
    unresolved = Path(os.path.abspath(path))
    current = Path(unresolved.anchor)
    for part in unresolved.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise TasteResearchPolicyError(f"{field} may not contain symlink components")
    return path.resolve(strict=False)


def build_tastemolnet_gine_research_fragment(
    *,
    policy_path: str | Path,
    expected_output_root: str | Path,
    prepared_root: str | Path | None = None,
    graph_cache_root: str | Path | None = None,
    policy_receipt: str | Path | None = None,
    expected_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a disabled template or an authority-closed runnable fragment."""

    policy = load_tastemolnet_research_policy(
        policy_path, expected_file_sha256=expected_policy_sha256
    )
    output_root = _absolute(
        expected_output_root, field="expected_output_root", required=True
    )
    assert output_root is not None
    if output_root.exists():
        raise TasteResearchPolicyError("expected_output_root must be fresh and absent")
    prepared = _absolute(prepared_root, field="prepared_root", required=policy.active)
    cache = _absolute(graph_cache_root, field="graph_cache_root", required=policy.active)
    receipt_path = _absolute(
        policy_receipt, field="policy_receipt", required=policy.active
    )
    authority_evidence = None
    receipt_evidence = None
    if policy.active:
        assert prepared is not None and cache is not None and receipt_path is not None
        for private_root in (prepared, cache):
            if (
                output_root == private_root
                or output_root in private_root.parents
                or private_root in output_root.parents
            ):
                raise TasteResearchPolicyError(
                    "expected_output_root must be disjoint from private data/cache"
                )
        authority = validate_tastemolnet_local_authority(
            policy, prepared_root=prepared, graph_cache_root=cache
        )
        receipt = validate_tastemolnet_policy_receipt(
            receipt_path,
            policy=policy,
            authority=authority,
            require_active=True,
        )
        authority_evidence = authority.evidence()
        receipt_evidence = {
            "path": str(receipt.path),
            "sha256": receipt.sha256,
        }
    elif receipt_path is not None or prepared is not None or cache is not None:
        raise TasteResearchPolicyError(
            "inactive template may not claim live data or policy-receipt authority"
        )

    task_command = None
    if policy.active:
        task_command = [
            "bash",
            "scripts/autodl/run_tastemolnet_gine_controller.sh",
        ]
    expected = str(output_root)
    training_state_root = f"{expected}.training_state"
    cid_suffix = hashlib.sha256(
        f"{expected}\0{policy.file_sha256}".encode("utf-8")
    ).hexdigest()[:8]
    controller_cid = f"tastemolnet_gine_v1_20260825T000000Z_{cid_suffix}"
    controller_root = str(
        output_root.parent / f".{output_root.name}.controller-{controller_cid}"
    )
    task = {
        "schema_version": TASK_SCHEMA,
        "id": TASK_ID,
        "dataset": "tastemolnet",
        "stage": STAGE,
        "enabled": policy.active,
        "blocked_reason": None if policy.active else PENDING_REASON,
        "resource": "gpu",
        "physical_gpu_index": 2,
        "gpu_lock_mode": "exclusive",
        "gpu_memory_reservation_mb": 0,
        "gpu_shared_workload_class": None,
        "run_tastemolnet": 1 if policy.active else 0,
        "environment": {
            "RUN_TASTEMOLNET": "1" if policy.active else "0",
            "TASTEMOLNET_GPU_INDEX": "2",
            "TASTEMOLNET_POLICY_FILE": str(policy.path),
            "TASTEMOLNET_POLICY_SHA256": policy.file_sha256,
            "TASTEMOLNET_POLICY_RECEIPT": (
                str(receipt_path) if receipt_path is not None else ""
            ),
            "TASTEMOLNET_SPLIT_ROOT": str(prepared / "splits") if prepared else "",
            "TASTEMOLNET_PREPARED_ROOT": str(prepared) if prepared else "",
            "TASTEMOLNET_GRAPH_CACHE_ROOT": str(cache) if cache else "",
            "TASTEMOLNET_GNN_TRAINING_STATE_ROOT": (
                training_state_root if policy.active else ""
            ),
            "TASTEMOLNET_GNN_FULL_OUTPUT": expected if policy.active else "",
            "TASTEMOLNET_GINE_CONTROLLER_CID": (
                controller_cid if policy.active else ""
            ),
            "TASTEMOLNET_GINE_CONTROLLER_ROOT": (
                controller_root if policy.active else ""
            ),
            "TASTE_RESEARCH_COMPUTE_ALLOWED": "1" if policy.active else "0",
            "TASTE_PAPER_RESULTS_ALLOWED": "1" if policy.active else "0",
            "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
            "TASTE_UPSTREAM_LICENSE_STATUS": "NOT_EXPLICITLY_STATED",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "command": task_command,
        "command_template": [
            "bash",
            "scripts/autodl/run_tastemolnet_gine_controller.sh",
        ],
        "config_files": [
            "configs/hpc.yaml",
            "configs/gnn/gine.yaml",
            "configs/autodl/tastemolnet_gine_research_v1.yaml",
        ],
        "expected_output": expected,
        "persistent_training_state_root": training_state_root,
        "persistent_controller_root": controller_root,
        "controller_cid": controller_cid,
        "fresh_output_required": True,
        "epoch_checkpoint_required": True,
        "same_root_resume_supported": True,
        "retry_policy": "resume_same_training_state_after_process_loss",
        "required_output_files": list(REQUIRED_OUTPUT_FILES),
        "required_log_marker": "[TASTE_GINE_THREE_CLASS_PASS]",
        "data_splits_loaded": ["train", "validation"],
        "calibration_loaded": False,
        "test_loaded": False,
        "test_metadata_hash_only": True,
        "classifier_contract": {
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "num_classes": 3,
            "label_map": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
            "source_label": 1,
            "source_label_name": "Sweet",
            "counterfactual_mode": "untargeted_strict_flip",
            "strict_flip_condition": "pred_before == 1 and pred_after != 1",
        },
        "data_contract": {
            "reuse_existing_prepared_only": True,
            "reuse_existing_graph_cache_only": True,
            "data_reprepared": False,
            "graph_cache_rebuilt": False,
            "prepared_root": str(prepared) if prepared else None,
            "graph_cache_root": str(cache) if cache else None,
            "authority": authority_evidence,
        },
        "policy_receipt": receipt_evidence,
        "public_artifact_audit_required": True,
        "dataset_redistribution_allowed": False,
        "upstream_terms_status": "NOT_EXPLICITLY_STATED",
        "hpc_execution_allowed": False,
    }
    return {
        "schema_version": FRAGMENT_SCHEMA,
        "dataset": "tastemolnet",
        "status": policy.status,
        "authorization_status": policy.authorization_status,
        "policy": policy.evidence(),
        "policy_active": policy.active,
        "controller_contract": {
            "dedicated_tastemolnet_controller_required": True,
            "generic_four_gpu_controller_eligible": False,
            "exact_physical_gpu_index": 2,
            "exclusive_gpu_lock_required": True,
            "fresh_controller_root_required": True,
            "persistent_process_loss_supervision": True,
            "durable_exec_startup_barrier_required": True,
            "same_cid_worker_adoption_required": True,
            "terminal_babysit_required": True,
            "fresh_science_output_required": True,
        },
        "tasks": [task],
    }


def validate_tastemolnet_gine_research_fragment(
    payload: dict[str, Any], *, require_active: bool
) -> dict[str, Any]:
    if payload.get("schema_version") != FRAGMENT_SCHEMA or payload.get("dataset") != "tastemolnet":
        raise TasteResearchPolicyError("Taste GINE fragment schema changed")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise TasteResearchPolicyError("Taste GINE fragment must contain one task")
    task = tasks[0]
    active = payload.get("policy_active") is True
    if require_active and not active:
        raise TasteResearchPolicyError("TASTEMOLNET_POLICY_NOT_ACTIVATED")
    if (
        task.get("schema_version") != TASK_SCHEMA
        or task.get("id") != TASK_ID
        or task.get("stage") != STAGE
        or task.get("physical_gpu_index") != 2
        or task.get("gpu_lock_mode") != "exclusive"
        or task.get("gpu_memory_reservation_mb") != 0
        or task.get("classifier_contract", {}).get("num_classes") != 3
        or task.get("classifier_contract", {}).get("source_label") != 1
        or task.get("classifier_contract", {}).get("rf_oracle_used") is not False
        or task.get("test_loaded") is not False
        or task.get("calibration_loaded") is not False
        or task.get("dataset_redistribution_allowed") is not False
        or task.get("hpc_execution_allowed") is not False
        or task.get("epoch_checkpoint_required") is not True
        or task.get("same_root_resume_supported") is not True
        or task.get("retry_policy")
        != "resume_same_training_state_after_process_loss"
    ):
        raise TasteResearchPolicyError("Taste GINE task contract changed")
    if active:
        if (
            task.get("enabled") is not True
            or task.get("run_tastemolnet") != 1
            or task.get("command") != task.get("command_template")
            or not isinstance(task.get("policy_receipt"), dict)
            or not isinstance(task.get("data_contract", {}).get("authority"), dict)
            or task.get("environment", {}).get(
                "TASTEMOLNET_GNN_TRAINING_STATE_ROOT"
            )
            != task.get("persistent_training_state_root")
            or task.get("environment", {}).get("TASTEMOLNET_GNN_FULL_OUTPUT")
            != task.get("expected_output")
            or task.get("environment", {}).get("TASTEMOLNET_GINE_CONTROLLER_ROOT")
            != task.get("persistent_controller_root")
            or task.get("environment", {}).get("TASTEMOLNET_GINE_CONTROLLER_CID")
            != task.get("controller_cid")
            or task.get("command")
            != ["bash", "scripts/autodl/run_tastemolnet_gine_controller.sh"]
        ):
            raise TasteResearchPolicyError("active Taste GINE task is not authority-closed")
    else:
        if (
            task.get("enabled") is not False
            or task.get("run_tastemolnet") != 0
            or task.get("command") is not None
            or task.get("blocked_reason") != PENDING_REASON
            or task.get("policy_receipt") is not None
            or task.get("data_contract", {}).get("authority") is not None
        ):
            raise TasteResearchPolicyError("inactive Taste GINE template became runnable")
    return payload


__all__ = [
    "FRAGMENT_SCHEMA",
    "PENDING_REASON",
    "REQUIRED_OUTPUT_FILES",
    "STAGE",
    "TASK_ID",
    "TASK_SCHEMA",
    "build_tastemolnet_gine_research_fragment",
    "validate_tastemolnet_gine_research_fragment",
]
