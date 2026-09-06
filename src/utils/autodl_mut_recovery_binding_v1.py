"""Seal this Mut resource-stop continuation; never start a new scheduler.

The original observer is incomplete. A is therefore a new prefix from the
identical initialization, not a resume of the incomplete legacy250 boundary.
Large immutable inputs reuse the original sealed hashes at preparation time;
the existing owner reopens them when actual resource admission permits science.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from .autodl_mut_common_boundary_v1 import checkpoint_binding
from .autodl_mut_first_divergence_v1 import atomic_json, file_sha256, stable_sha256
from .autodl_mut_same_contract_ab_v1 import (
    INSTRUMENTATION_COMMIT, SOURCE_COMMIT, same_contract_ab_command,
    validate_same_contract_ab_spec,
)
from .autodl_mut_next_stage_executor_v1 import validate_successor_spec


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Physical JSON required: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Object required: {path}")
    return value


def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {k: v for k, v in value.items() if k != field}
    value[field] = stable_sha256(value)
    return value


def resource_status(path: Path) -> dict[str, Any]:
    while not path.exists():
        path = path.parent
    fs = os.statvfs(path)
    return {"state": "ADMISSION_PASS" if fs.f_favail >= 100160 and
            fs.f_bavail * fs.f_frsize >= 50 * 1024**3 and
            fs.f_bavail / max(1, fs.f_blocks) >= .02 else "SEALED_WAITING_INODE_OR_BYTES",
            "path": str(path), "free_inodes": fs.f_favail,
            "guard": 100000, "known_new_compact_peak": 160,
            "fixed_required_free": 100160,
            "fixed_inode_shortfall": max(0, 100160 - fs.f_favail),
            "unknown_dynamic_peak": "UNKNOWN_NOT_ZERO",
            "free_bytes": fs.f_bavail * fs.f_frsize,
            "minimum_free_bytes": 50 * 1024**3, "minimum_free_ratio": .02,
            "recorded_at": datetime.now(timezone.utc).isoformat()}


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old in sorted(replacements, key=len, reverse=True):
            if value == old or value.startswith(old + "/"):
                return replacements[old] + value[len(old):]
        return value
    if isinstance(value, list):
        return [_replace_strings(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: _replace_strings(v, replacements) for k, v in value.items()}
    return value


def seal_recovery(*, old_ab_path: Path, old_executor_path: Path,
                  driver_root: Path, output: Path, gpu_lock_root: Path,
                  gpu_uuid: str) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Recovery binding root must be fresh")
    old_ab = validate_same_contract_ab_spec(read_json(old_ab_path), check_files=False)
    old_executor = validate_successor_spec(read_json(old_executor_path), check_files=False)
    if old_executor["predecessor_task_spec"] != str(old_ab_path):
        raise ValueError("Executor does not consume this exact legacy A/B")
    commit = subprocess.check_output(["git", "-C", str(driver_root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(driver_root), "status", "--porcelain"], text=True).strip():
        raise ValueError("Recovery execution checkout must be clean")
    token = str(uuid4())
    original_arm = Path(old_ab["run_root"]) / "trace_on"
    checkpoint = original_arm / "generation_checkpoints/step-000000000250"
    bound = checkpoint_binding(checkpoint, step=250)
    # One bounded ledger pass confirms the known1-based gap. It neither loads
    # checkpoint tensors nor opens the old graph database.
    observer = original_arm / "common_step_state.jsonl"
    steps = []
    with observer.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("phase") != "continuous" or row.get("next_step") != row.get("step", -1) + 1:
                raise ValueError("Old observer phase/cursor changed")
            steps.append(row["step"])
    if steps != list(range(1, 250)):
        raise ValueError("Old observer is no longer exactly1..249; re-audit before binding")
    output.mkdir(parents=True)
    recovery = sealed({"schema_version": "mut_resource_replay_250_v1",
        "original_ab_spec": str(old_ab_path), "original_ab_spec_sha256": file_sha256(old_ab_path),
        "original_checkpoint": str(checkpoint), "original_checkpoint_manifest_sha256": bound["manifest_sha256"],
        "original_observer": str(observer), "original_observer_sha256": file_sha256(observer),
        "replay_range": [1, 250], "observed_event_range": [1, 249], "last_joint_completed_step": 0,
        "skip_event_250": False, "route_b_on_resource_failure": False,
        "source_algorithm_commit": SOURCE_COMMIT, "instrumentation_commit": INSTRUMENTATION_COMMIT,
        "pythonhashseed": "0", "resource_stop_is_scientific_divergence": False,
        "comparison_required_before251": True, "old_checkpoint_promoted": False,
        "created_at": datetime.now(timezone.utc).isoformat()}, "contract_sha256")
    contract_path = output / "replay-contract.json"
    atomic_json(contract_path, recovery)
    ab = copy.deepcopy(old_ab)
    ab.update(task_id=f"mut-resource-replay-{token}", attempt_uuid=token,
        controller_project_root=str(driver_root), controller_commit=commit,
        runner_path=str(driver_root / "scripts/autodl/run_mut_trace_mode_equivalence.py"),
        run_root=str(output / "science"), output_dir=str(output / "equivalence-audit"),
        control_root=str(output / "ab-owner"), gpu_lock_root=str(gpu_lock_root), gpu_uuid=gpu_uuid,
        recovery_contract=str(contract_path), recovery_contract_sha256=file_sha256(contract_path),
        created_at=datetime.now(timezone.utc).isoformat())
    # Reuse the same owner namespace/lease, not a new independent authority.
    ab["bound_file_sha256s"]["runner"] = file_sha256(Path(ab["runner_path"]))
    ab = sealed(ab, "spec_sha256")
    validate_same_contract_ab_spec(ab, check_files=False)
    ab_path = output / "ab-task-spec.json"
    atomic_json(ab_path, ab)
    for mode in ("on", "off"):
        arm = {"schema_version": "mut_bound_recovery_arm_v1", "trace_mode": mode,
               "ab_task_spec": str(ab_path), "ab_spec_sha256": ab["spec_sha256"],
               "arm_root": str(Path(ab["run_root"]) / f"trace_{mode}"),
               "initialization": "FRESH_SAME_PINNED_INITIAL_INPUT_AND_RNG",
               "comparison_steps": 500, "post_reload_steps": [501, 510],
               "sequence_index": 0 if mode == "on" else 1,
               "resume_parity_separate": True,
               "replay_reconciliation": str(contract_path) if mode == "on" else None,
               "segments": [[1, 250], [251, 510], [501, 510]] if mode == "on" else [[1, 510], [501, 510]],
               "dispatch_owner": str(Path(ab["control_root"]) / "owner_pid.json"),
               "state": "SEALED_WAITING_INODE"}
        atomic_json(output / f"arm-{mode}-spec.json", sealed(arm, "spec_sha256"))
    replacements = {
        old_ab["controller_project_root"]: str(driver_root),
        old_executor["adoption_pipeline"][0]["cwd"]: str(driver_root),
        str(old_ab_path): str(ab_path), old_ab["control_root"]: ab["control_root"],
        old_ab["output_dir"]: ab["output_dir"],
        str(Path(old_executor["next_action_path"]).parent): str(output / "post-ab"),
        old_executor["runtime_root"]: str(output / "executor"),
    }
    for index, stage in enumerate(old_executor["adoption_pipeline"] + old_executor["route_b_pipeline"]):
        replacements[stage["output_root"]] = str(output / f"stage-{index:02d}-{stage['stage'].lower()}")
    executor = _replace_strings(old_executor, replacements)
    executor.update(task_id=f"mut-next-stage-resource-replay-{token}", execution_commit=commit,
        predecessor_task_id=ab["task_id"], predecessor_task_spec_sha256=file_sha256(ab_path),
        created_at=datetime.now(timezone.utc).isoformat())
    for stage in executor["adoption_pipeline"] + executor["route_b_pipeline"]:
        stage["argv_sha256"] = stable_sha256(stage["argv"])
    executor = sealed(executor, "spec_sha256")
    validate_successor_spec(executor, check_files=False)
    executor_path = output / "executor-task-spec.json"
    atomic_json(executor_path, executor)
    python = ab["python"]
    post_command = [python, "-I", "-B", str(driver_root / "scripts/autodl/run_mut_post_ab_continuation_v1.py"),
                    "--config", "configs/hpc.yaml", "--ab-task-spec", str(ab_path),
                    "--output-root", str(output / "post-ab"), "--poll-seconds", "60"]
    owner_command = [python, "-I", "-B", str(driver_root / "scripts/autodl/run_mut_same_contract_ab_owner_v1.py"),
                     "--config", "configs/hpc.yaml", "--task-spec", str(ab_path)]
    executor_command = [python, "-I", "-B", str(driver_root / "scripts/autodl/run_mut_next_stage_executor_v1.py"),
                        "--config", "configs/hpc.yaml", "--task-spec", str(executor_path), "--poll-seconds", "60"]
    atomic_json(output / "post-ab-spec.json", sealed({"schema_version": "mut_post_ab_command_binding_v1",
        "ab_task_spec": str(ab_path), "ab_spec_sha256": ab["spec_sha256"],
        "argv": post_command, "cwd": str(driver_root), "environment": ab["required_environment"],
        "output_root": str(output / "post-ab")}, "spec_sha256"))
    status = {"schema_version": "mut_resource_recovery_binding_v1", "state": "SEALED_WAITING_INODE",
        "driver_commit": commit, "driver_root": str(driver_root), "ab_task_spec": str(ab_path),
        "post_ab_spec": str(output / "post-ab-spec.json"), "executor_task_spec": str(executor_path),
        "replay_contract": str(contract_path), "replay_range": [1, 250],
        "owner_command": owner_command, "ab_science_command": same_contract_ab_command(ab),
        "post_ab_command": post_command, "executor_command": executor_command,
        "old_executor_spec": str(old_executor_path), "old_executor_dynamic_reload": False,
        "old_executor_lease_preserved": old_executor["lease_path"],
        "activation_requires": ["resource_admission", "old_idle_executor_identity_and_no_child_no_claim",
             "graceful_old_executor_exit", "canonical_registry_stage_boundary_rebind"],
        "science_started": False, "old_executor_signaled": False,
        "resource": resource_status(output),
        "fallback_state": "BLOCKED_ADAPTER_MISSING_IF_TRUE_SCIENTIFIC_DIVERGENCE",
        "fallback_on_resource_failure": False, "matrix_authority": old_executor["matrix_authority_root"]}
    atomic_json(output / "binding.json", sealed(status, "binding_sha256"))
    return status
