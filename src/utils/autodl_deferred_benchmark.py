"""Persistently defer the BACE GINE batch matrix until GPU2 is truly free.

The controller is intentionally narrower than a general scheduler.  It binds
one upstream ComRecGC pair run, one physical GPU UUID, one immutable checkout,
and one fresh benchmark run/output root.  Resource contention is retried with
a 60-second heartbeat; provenance mismatch or any partial target is terminal
and fail closed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from src.utils.autodl_runtime import (
    GPUFileLock,
    GPULockError,
    ProjectGPUSlotLock,
    append_jsonl_locked,
    atomic_write_json,
    available_project_gpu_slots,
    fsync_directory,
    gpu_lock_available,
    query_gpu_inventory,
    read_json_object,
    sha256_file,
    utc_now,
)


TERMINAL_RUN_STATES = frozenset({"PASS", "FAILED", "BLOCKED"})
CONTROLLER_TERMINALS = frozenset({"PASS", "BLOCKED"})
SCHEMA_VERSION = "bace_gnn_inference_deferred_controller_v1"
REQUIRED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


@dataclass(frozen=True)
class DeferredObservation:
    launch_spec_valid: bool
    launch_spec_reason: str | None
    pair_state: str | None
    pair_worker_identity: str
    uuid_lock_available: bool
    project_slot_available: bool
    gpu_identity_valid: bool
    gpu_process_count: int
    gpu_free_memory_mb: int
    gpu_utilization_percent: int
    minimum_free_memory_mb: int
    maximum_utilization_percent: int
    benchmark_output_exists: bool
    benchmark_run_state_exists: bool
    immutable_checkout_valid: bool
    input_manifest_valid: bool


def classify_readiness(observation: DeferredObservation) -> tuple[str, str]:
    """Return ``(state, reason)`` without mutating any controller state."""

    if observation.benchmark_output_exists:
        return "BLOCKED", "PARTIAL_OR_PREEXISTING_BENCHMARK_OUTPUT_ROOT"
    if observation.benchmark_run_state_exists:
        return "BLOCKED", "PARTIAL_OR_PREEXISTING_BENCHMARK_RUN_STATE"
    if not observation.launch_spec_valid:
        return "BLOCKED", observation.launch_spec_reason or "PAIR_LAUNCH_SPEC_INVALID"
    if not observation.immutable_checkout_valid:
        return "BLOCKED", "IMMUTABLE_EXECUTION_CHECKOUT_INVALID"
    if not observation.input_manifest_valid:
        return "BLOCKED", "BENCHMARK_INPUT_MANIFEST_INVALID"
    if observation.pair_state is None:
        return "BLOCKED", "PAIR_REGISTRY_STATE_MISSING_OR_INVALID"
    if (
        observation.pair_worker_identity == "PID_REUSED"
        and observation.pair_state not in TERMINAL_RUN_STATES
    ):
        return "BLOCKED", "PAIR_WORKER_PID_REUSED_BEFORE_REGISTRY_TERMINAL"
    if observation.pair_state not in TERMINAL_RUN_STATES:
        return "WAITING_RESOURCE", f"PAIR_REGISTRY_{observation.pair_state}"
    if observation.pair_worker_identity == "MATCHING_ALIVE":
        return "WAITING_RESOURCE", "PAIR_WORKER_STILL_ALIVE_AFTER_TERMINAL"
    if not observation.uuid_lock_available:
        return "WAITING_RESOURCE", "GPU_UUID_EXCLUSIVE_LOCK_STILL_HELD"
    if not observation.project_slot_available:
        return "WAITING_RESOURCE", "NO_PROJECT_GPU_SLOT_AVAILABLE"
    if not observation.gpu_identity_valid:
        return "BLOCKED", "GPU_INDEX_UUID_IDENTITY_MISMATCH"
    if observation.gpu_process_count:
        return "WAITING_RESOURCE", "GPU_UUID_HAS_COMPUTE_PROCESS"
    if observation.gpu_free_memory_mb < observation.minimum_free_memory_mb:
        return "WAITING_RESOURCE", "GPU_FREE_MEMORY_BELOW_GATE"
    if observation.gpu_utilization_percent > observation.maximum_utilization_percent:
        return "WAITING_RESOURCE", "GPU_UTILIZATION_ABOVE_GATE"
    return "READY", "ALL_DEPENDENCY_AND_RESOURCE_GATES_READY"


def _safe_component(value: str, *, label: str) -> str:
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not value or any(character not in safe for character in value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _proc_identity(pid: int) -> tuple[int, int] | None:
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return int(fields[1]), int(fields[19])
    except ValueError:
        return None


def _git_value(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _is_clean_commit(project_root: Path, expected_commit: str) -> bool:
    try:
        return (
            _git_value(project_root, "rev-parse", "HEAD") == expected_commit
            and _git_value(project_root, "status", "--porcelain") == ""
        )
    except (OSError, RuntimeError):
        return False


def _path_exists_or_symlink(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _fresh_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def _controller_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "controller_id": args.controller_id,
        "project_root": str(args.project_root),
        "execution_commit": args.execution_commit,
        "pair_run_id": args.pair_run_id,
        "pair_run_state_root": str(args.pair_run_state_root),
        "pair_launch_spec_sha256": args.pair_launch_spec_sha256,
        "pair_worker_pid": args.pair_worker_pid,
        "pair_worker_start_ticks": args.pair_worker_start_ticks,
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "lock_root": str(args.lock_root),
        "benchmark_run_id": args.benchmark_run_id,
        "benchmark_run_state_root": str(args.benchmark_run_state_root),
        "benchmark_output_root": str(args.benchmark_output_root),
        "benchmark_input_manifest": str(args.benchmark_input_manifest),
        "benchmark_input_manifest_sha256": args.benchmark_input_manifest_sha256,
        "benchmark_log_path": str(args.benchmark_log_path),
        "dataset_dir": str(args.dataset_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "poll_seconds": args.poll_seconds,
        "stable_ready_seconds": args.stable_ready_seconds,
        "minimum_free_memory_mb": args.minimum_free_memory_mb,
        "maximum_utilization_percent": args.maximum_utilization_percent,
        "config": str(args.config),
        "set_values": list(args.set),
        "batch_sizes": [1, 8, 32, 128, 512],
        "warmups": args.warmups,
        "repeats": args.repeats,
        "thread_environment": REQUIRED_THREAD_ENVIRONMENT,
        "diagnostic_only": True,
        "paper_eligible": False,
        "authorizes_vrrw_replacement": False,
    }


def _validate_pair_launch_spec(args: argparse.Namespace) -> tuple[bool, str | None]:
    path = args.pair_run_state_root / "launch_spec.json"
    if not path.is_file() or path.is_symlink():
        return False, "PAIR_LAUNCH_SPEC_MISSING_OR_SYMLINK"
    if sha256_file(path) != args.pair_launch_spec_sha256:
        return False, "PAIR_LAUNCH_SPEC_SHA256_MISMATCH"
    try:
        payload = read_json_object(path)
    except Exception:
        return False, "PAIR_LAUNCH_SPEC_INVALID_JSON"
    expected = {
        "run_id": args.pair_run_id,
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "gpu_lock_mode": "exclusive",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return False, f"PAIR_LAUNCH_SPEC_BINDING_MISMATCH:{key}"
    return True, None


def _pair_state(args: argparse.Namespace) -> str | None:
    path = args.pair_run_state_root / "state.json"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = read_json_object(path)
    except Exception:
        return None
    if payload.get("run_id") != args.pair_run_id:
        return None
    value = payload.get("state")
    return str(value) if isinstance(value, str) and value else None


def _pair_worker_identity(args: argparse.Namespace) -> str:
    identity = _proc_identity(args.pair_worker_pid)
    if identity is None:
        return "GONE"
    _parent, start_ticks = identity
    if start_ticks == args.pair_worker_start_ticks:
        return "MATCHING_ALIVE"
    return "PID_REUSED"


def _validate_input_manifest(args: argparse.Namespace) -> bool:
    path = args.benchmark_input_manifest
    if not path.is_file() or path.is_symlink():
        return False
    try:
        if sha256_file(path) != args.benchmark_input_manifest_sha256:
            return False
        payload = read_json_object(path)
    except Exception:
        return False
    expected = {
        "schema_version": "bace_gnn_inference_deferred_input_v1",
        "controller_id": args.controller_id,
        "benchmark_run_id": args.benchmark_run_id,
        "execution_commit": args.execution_commit,
        "project_root": str(args.project_root),
        "benchmark_output_root": str(args.benchmark_output_root),
        "benchmark_run_state_root": str(args.benchmark_run_state_root),
        "dataset_dir": str(args.dataset_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "pair_run_id": args.pair_run_id,
        "pair_launch_spec_sha256": args.pair_launch_spec_sha256,
        "pair_worker_pid": args.pair_worker_pid,
        "pair_worker_start_ticks": args.pair_worker_start_ticks,
        "batch_sizes": [1, 8, 32, 128, 512],
        "thread_environment": REQUIRED_THREAD_ENVIRONMENT,
    }
    return all(payload.get(key) == value for key, value in expected.items())


def _gpu_observation(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    try:
        matches = [
            item
            for item in query_gpu_inventory()
            if item.index == args.gpu_index and item.uuid == args.gpu_uuid
        ]
    except Exception:
        return False, 0, 0, 100
    if len(matches) != 1:
        return False, 0, 0, 100
    item = matches[0]
    return (
        True,
        item.process_count,
        item.memory_free_mb,
        item.utilization_gpu_percent,
    )


def observe(args: argparse.Namespace, *, ignore_owned_locks: bool = False) -> DeferredObservation:
    launch_valid, launch_reason = _validate_pair_launch_spec(args)
    identity_valid, process_count, free_memory, utilization = _gpu_observation(args)
    try:
        uuid_available = (
            True
            if ignore_owned_locks
            else gpu_lock_available(args.lock_root, args.gpu_uuid)
        )
    except Exception:
        uuid_available = False
    try:
        project_available = (
            True
            if ignore_owned_locks
            else available_project_gpu_slots(
                args.lock_root, 4, hard_limit=4
            )
            > 0
        )
    except Exception:
        project_available = False
    input_valid = _validate_input_manifest(args)
    return DeferredObservation(
        launch_spec_valid=launch_valid,
        launch_spec_reason=launch_reason,
        pair_state=_pair_state(args),
        pair_worker_identity=_pair_worker_identity(args),
        uuid_lock_available=uuid_available,
        project_slot_available=project_available,
        gpu_identity_valid=identity_valid,
        gpu_process_count=process_count,
        gpu_free_memory_mb=free_memory,
        gpu_utilization_percent=utilization,
        minimum_free_memory_mb=args.minimum_free_memory_mb,
        maximum_utilization_percent=args.maximum_utilization_percent,
        benchmark_output_exists=_path_exists_or_symlink(args.benchmark_output_root),
        benchmark_run_state_exists=_path_exists_or_symlink(
            args.benchmark_run_state_root
        ),
        immutable_checkout_valid=_is_clean_commit(
            args.project_root, args.execution_commit
        ),
        input_manifest_valid=input_valid,
    )


def _state_document(
    args: argparse.Namespace,
    *,
    state: str,
    reason: str,
    observation: DeferredObservation | None,
    ready_seconds: float,
    child_pid: int | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "controller_id": args.controller_id,
        "controller_pid": os.getpid(),
        "controller_start_ticks": (_proc_identity(os.getpid()) or (None, None))[1],
        "state": state,
        "reason": reason,
        "heartbeat_at": utc_now(),
        "poll_seconds": args.poll_seconds,
        "ready_stability_seconds": ready_seconds,
        "required_ready_stability_seconds": args.stable_ready_seconds,
        "pair_run_id": args.pair_run_id,
        "pair_launch_spec_sha256": args.pair_launch_spec_sha256,
        "thread_environment": REQUIRED_THREAD_ENVIRONMENT,
        "pair_worker_pid": args.pair_worker_pid,
        "pair_worker_start_ticks": args.pair_worker_start_ticks,
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "benchmark_run_id": args.benchmark_run_id,
        "benchmark_output_root": str(args.benchmark_output_root),
        "benchmark_child_pid": child_pid,
        "benchmark_exit_code": exit_code,
        "observation": asdict(observation) if observation is not None else None,
    }


def _publish_state(
    args: argparse.Namespace,
    *,
    state: str,
    reason: str,
    observation: DeferredObservation | None,
    ready_seconds: float,
    child_pid: int | None = None,
    exit_code: int | None = None,
) -> None:
    payload = _state_document(
        args,
        state=state,
        reason=reason,
        observation=observation,
        ready_seconds=ready_seconds,
        child_pid=child_pid,
        exit_code=exit_code,
    )
    atomic_write_json(args.state_root / "state.json", payload)
    append_jsonl_locked(args.state_root / "status_updates.jsonl", payload)


def _publish_blocked(
    args: argparse.Namespace,
    *,
    reason: str,
    observation: DeferredObservation | None,
) -> int:
    _publish_state(
        args,
        state="BLOCKED",
        reason=reason,
        observation=observation,
        ready_seconds=0.0,
    )
    marker = args.state_root / "BLOCKED"
    if not marker.exists():
        _fresh_write_json(
            marker,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "BLOCKED",
                "reason": reason,
                "published_at": utc_now(),
            },
        )
    return 2


def _benchmark_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        "scripts/autodl/benchmark_bace_gnn_inference_matrix.py",
        "--config",
        str(args.config),
    ]
    for value in args.set:
        command.extend(["--set", value])
    command.extend(
        [
            "--dataset-dir",
            str(args.dataset_dir),
            "--checkpoint-dir",
            str(args.checkpoint_dir),
            "--output-dir",
            str(args.benchmark_output_root),
            "--batch-sizes",
            "1,8,32,128,512",
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
        ]
    )
    return command


def _run_spec(args: argparse.Namespace, command: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": args.benchmark_run_id,
        "created_at": utc_now(),
        "project_root": str(args.project_root),
        "git_commit": args.execution_commit,
        "command": list(command),
        "dataset": "bace-gine-inference-matrix",
        "stage": "BACE_GNN_INFERENCE_BATCH_MATRIX",
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "gpu_lock_mode": "exclusive",
        "gpu_memory_reservation_mb": 16000,
        "input_manifest": str(args.benchmark_input_manifest),
        "input_hash": args.benchmark_input_manifest_sha256,
        "expected_output": str(args.benchmark_output_root),
        "log_path": str(args.benchmark_log_path),
        "controller_id": args.controller_id,
        "pair_run_id": args.pair_run_id,
        "pair_launch_spec_sha256": args.pair_launch_spec_sha256,
        "thread_environment": REQUIRED_THREAD_ENVIRONMENT,
    }


def _registry_event(
    spec: Mapping[str, Any], *, state: str, exit_code: int | None, pid: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "timestamp": utc_now(),
        "pid": pid,
        "tmux_session": None,
        "command": spec["command"],
        "dataset": spec["dataset"],
        "stage": spec["stage"],
        "gpu_index": spec["gpu_index"],
        "gpu_uuid": spec["gpu_uuid"],
        "gpu_lock_mode": "exclusive",
        "gpu_memory_reservation_mb": 16000,
        "gpu_shared_workload_class": None,
        "gpu_colocation_gate": None,
        "gpu_colocation_gate_sha256": None,
        "git_commit": spec["git_commit"],
        "config_hash": None,
        "input_hash": spec["input_hash"],
        "expected_output": spec["expected_output"],
        "state": state,
        "exit_code": exit_code,
        "backend": "autodl",
        "slurm_job_id": None,
        "log_path": spec["log_path"],
        "deferred_controller_id": spec["controller_id"],
    }


def _write_run_state(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    *,
    state: str,
    child_pid: int | None,
    exit_code: int | None,
    failures: Sequence[str] = (),
) -> None:
    atomic_write_json(
        args.benchmark_run_state_root / "state.json",
        {
            "schema_version": 1,
            "run_id": args.benchmark_run_id,
            "dataset": spec["dataset"],
            "stage": spec["stage"],
            "state": state,
            "pid": os.getpid(),
            "child_pid": child_pid,
            "gpu_index": args.gpu_index,
            "gpu_uuid": args.gpu_uuid,
            "gpu_lock_mode": "exclusive",
            "log_path": str(args.benchmark_log_path),
            "updated_at": utc_now(),
            "exit_code": exit_code,
            "failures": list(failures),
            "deferred_controller_id": args.controller_id,
        },
    )


def validate_benchmark_result(args: argparse.Namespace, exit_code: int) -> list[str]:
    failures: list[str] = []
    if exit_code != 0:
        failures.append(f"benchmark_exit_code:{exit_code}")
    result_path = args.benchmark_output_root / "bace_gnn_inference_benchmark.json"
    complete_path = args.benchmark_output_root / "_BENCHMARK_COMPLETE.json"
    for path in (result_path, complete_path):
        if not path.is_file() or path.is_symlink():
            failures.append(f"missing_or_symlink:{path.name}")
    if failures:
        return failures
    try:
        result = read_json_object(result_path)
        complete = read_json_object(complete_path)
    except Exception as exc:
        return [f"invalid_result_json:{type(exc).__name__}"]
    if result.get("status") != "PASS":
        failures.append("benchmark_result_not_pass")
    if result.get("batch_sizes") != [1, 8, 32, 128, 512]:
        failures.append("batch_matrix_mismatch")
    if result.get("all_argmax_and_allclose_checks_pass") is not True:
        failures.append("argmax_or_allclose_failed")
    if result.get("all_calibrated_probability_checks_pass") is not True:
        failures.append("calibrated_probability_checks_failed")
    if result.get("thread_environment") != REQUIRED_THREAD_ENVIRONMENT:
        failures.append("benchmark_thread_environment_mismatch")
    best_end_to_end = result.get("best_end_to_end")
    if (
        not isinstance(best_end_to_end, dict)
        or not isinstance(best_end_to_end.get("overall"), dict)
        or best_end_to_end["overall"].get("device") not in {"cpu", "gpu"}
        or best_end_to_end["overall"].get("batch_size")
        not in {1, 8, 32, 128, 512}
    ):
        failures.append("best_end_to_end_summary_missing_or_invalid")
    if result.get("cpu_raw_byte_repeat_exact_all_batches") is not True:
        failures.append("cpu_raw_byte_repeat_failed")
    if result.get("authorizes_vrrw_replacement") is not False:
        failures.append("replacement_authority_must_be_false")
    cohort = result.get("cohort", {})
    if not isinstance(cohort, dict) or cohort.get("test_loaded") is not False:
        failures.append("heldout_test_must_not_be_loaded")
    if complete.get("status") != "PASS":
        failures.append("completion_not_pass")
    if complete.get("result_sha256") != sha256_file(result_path):
        failures.append("completion_result_sha256_mismatch")
    try:
        log_text = args.benchmark_log_path.read_text(encoding="utf-8")
    except OSError:
        failures.append("benchmark_log_missing")
    else:
        if "[BACE_GNN_INFERENCE_MATRIX_BENCHMARK_PASS]" not in log_text:
            failures.append("benchmark_log_marker_missing")
    return failures


def _launch_with_owned_locks(args: argparse.Namespace) -> int:
    observation = observe(args)
    state, reason = classify_readiness(observation)
    if state != "READY":
        if state == "BLOCKED":
            return _publish_blocked(args, reason=reason, observation=observation)
        _publish_state(
            args,
            state="WAITING_RESOURCE",
            reason=f"RESOURCE_RACE:{reason}",
            observation=observation,
            ready_seconds=0.0,
        )
        return 75

    slot = ProjectGPUSlotLock(
        args.lock_root,
        max_slots=4,
        hard_limit=4,
        owner={
            "run_id": args.benchmark_run_id,
            "stage": "BACE_GNN_INFERENCE_BATCH_MATRIX",
            "controller_id": args.controller_id,
        },
    )
    gpu = GPUFileLock(
        args.lock_root,
        gpu_index=args.gpu_index,
        gpu_uuid=args.gpu_uuid,
        owner={
            "run_id": args.benchmark_run_id,
            "stage": "BACE_GNN_INFERENCE_BATCH_MATRIX",
            "controller_id": args.controller_id,
        },
    )
    try:
        slot.acquire()
        gpu.acquire()
    except GPULockError:
        gpu.release()
        slot.release()
        _publish_state(
            args,
            state="WAITING_RESOURCE",
            reason="RESOURCE_RACE_DURING_ATOMIC_LOCK_ACQUIRE",
            observation=observation,
            ready_seconds=0.0,
        )
        return 75

    try:
        recheck = observe(args, ignore_owned_locks=True)
        recheck_state, recheck_reason = classify_readiness(recheck)
        if recheck_state != "READY":
            if recheck_state == "BLOCKED":
                return _publish_blocked(
                    args, reason=f"POST_LOCK:{recheck_reason}", observation=recheck
                )
            _publish_state(
                args,
                state="WAITING_RESOURCE",
                reason=f"POST_LOCK_RESOURCE_RACE:{recheck_reason}",
                observation=recheck,
                ready_seconds=0.0,
            )
            return 75

        # This is the only point at which the single benchmark run state may
        # be created.  Both target roots were rechecked while both locks held.
        args.benchmark_run_state_root.mkdir(parents=True, exist_ok=False)
        command = _benchmark_command(args)
        spec = _run_spec(args, command)
        _fresh_write_json(
            args.benchmark_run_state_root / "launch_spec.json", spec
        )
        _write_run_state(
            args, spec, state="STARTING", child_pid=None, exit_code=None
        )
        append_jsonl_locked(
            args.registry_path,
            _registry_event(spec, state="STARTING", exit_code=None, pid=os.getpid()),
        )

        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": str(args.project_root),
                "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
                "AUTODL_PHYSICAL_GPU_INDEX": str(args.gpu_index),
                "AUTODL_PHYSICAL_GPU_UUID": args.gpu_uuid,
                "AUTODL_MPS_ENABLED": "0",
                **REQUIRED_THREAD_ENVIRONMENT,
                "PYTHONHASHSEED": "0",
                "RUN_TASTEMOLNET": "0",
            }
        )
        for key in list(environment):
            if key.startswith("CUDA_MPS"):
                environment.pop(key, None)
        args.benchmark_log_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handles = [slot._handle, gpu._handle]
        lock_fds = tuple(
            handle.fileno() for handle in lock_handles if handle is not None
        )
        for descriptor in lock_fds:
            os.set_inheritable(descriptor, True)
        with args.benchmark_log_path.open("x", encoding="utf-8", buffering=1) as log:
            child = subprocess.Popen(
                command,
                cwd=args.project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=lock_fds,
            )
            _write_run_state(
                args, spec, state="RUNNING", child_pid=child.pid, exit_code=None
            )
            append_jsonl_locked(
                args.registry_path,
                _registry_event(
                    spec, state="RUNNING", exit_code=None, pid=os.getpid()
                ),
            )
            while child.poll() is None:
                _publish_state(
                    args,
                    state="RUNNING",
                    reason="BENCHMARK_CHILD_RUNNING_WITH_INHERITED_UUID_LOCK",
                    observation=recheck,
                    ready_seconds=args.stable_ready_seconds,
                    child_pid=child.pid,
                )
                time.sleep(args.poll_seconds)
            exit_code = int(child.returncode)
            log.flush()
            os.fsync(log.fileno())
        fsync_directory(args.benchmark_log_path.parent)
        failures = validate_benchmark_result(args, exit_code)
        final_run_state = "PASS" if not failures else "FAILED"
        _write_run_state(
            args,
            spec,
            state=final_run_state,
            child_pid=child.pid,
            exit_code=exit_code,
            failures=failures,
        )
        append_jsonl_locked(
            args.registry_path,
            _registry_event(
                spec, state=final_run_state, exit_code=exit_code, pid=os.getpid()
            ),
        )
        if failures:
            return _publish_blocked(
                args,
                reason="BENCHMARK_TERMINAL_CONTRACT_FAILED:" + ";".join(failures),
                observation=recheck,
            )
        _publish_state(
            args,
            state="PASS",
            reason="BENCHMARK_TERMINAL_CONTRACT_PASS",
            observation=recheck,
            ready_seconds=args.stable_ready_seconds,
            child_pid=child.pid,
            exit_code=exit_code,
        )
        _fresh_write_json(
            args.state_root / "PASS",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "benchmark_run_id": args.benchmark_run_id,
                "benchmark_result": str(
                    args.benchmark_output_root
                    / "bace_gnn_inference_benchmark.json"
                ),
                "published_at": utc_now(),
            },
        )
        return 0
    finally:
        gpu.release()
        slot.release()


def _validate_and_initialize_controller(args: argparse.Namespace) -> None:
    spec = _controller_spec(args)
    manifest = args.state_root / "controller_manifest.json"
    if args.resume:
        if not manifest.is_file() or manifest.is_symlink():
            raise RuntimeError("resume requires an existing regular controller manifest")
        existing = read_json_object(manifest)
        if existing != spec:
            raise RuntimeError("resume controller manifest differs from current arguments")
        current = args.state_root / "state.json"
        if current.is_file():
            state = read_json_object(current).get("state")
            if state in CONTROLLER_TERMINALS:
                raise RuntimeError(f"controller is already terminal: {state}")
    else:
        if args.state_root.exists() or args.state_root.is_symlink():
            raise RuntimeError("fresh controller state root already exists")
        args.state_root.mkdir(parents=True, exist_ok=False)
        _fresh_write_json(manifest, spec)


def run(args: argparse.Namespace) -> int:
    _validate_and_initialize_controller(args)
    lock_path = args.state_root / "controller.lock"
    with lock_path.open("a+", encoding="utf-8") as controller_lock:
        try:
            fcntl.flock(
                controller_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise RuntimeError("deferred controller singleton lock is held") from exc

        ready_started: float | None = None
        while True:
            observation = observe(args)
            state, reason = classify_readiness(observation)
            if state == "BLOCKED":
                return _publish_blocked(
                    args, reason=reason, observation=observation
                )
            if state == "READY":
                if ready_started is None:
                    ready_started = time.monotonic()
                ready_seconds = max(0.0, time.monotonic() - ready_started)
                if ready_seconds >= args.stable_ready_seconds:
                    result = _launch_with_owned_locks(args)
                    if result == 75:
                        ready_started = None
                    else:
                        return result
                _publish_state(
                    args,
                    state="WAITING_RESOURCE",
                    reason="GPU_READY_STABILITY_WINDOW",
                    observation=observation,
                    ready_seconds=ready_seconds,
                )
            else:
                ready_started = None
                _publish_state(
                    args,
                    state="WAITING_RESOURCE",
                    reason=reason,
                    observation=observation,
                    ready_seconds=0.0,
                )
            time.sleep(args.poll_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hpc.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pair-run-id", required=True)
    parser.add_argument("--pair-run-state-root", type=Path, required=True)
    parser.add_argument("--pair-launch-spec-sha256", required=True)
    parser.add_argument("--pair-worker-pid", type=int, required=True)
    parser.add_argument("--pair-worker-start-ticks", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--benchmark-run-id", required=True)
    parser.add_argument("--benchmark-run-state-root", type=Path, required=True)
    parser.add_argument("--benchmark-output-root", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest-sha256", required=True)
    parser.add_argument("--benchmark-log-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stable-ready-seconds", type=float, default=60.0)
    parser.add_argument("--minimum-free-memory-mb", type=int, default=16000)
    parser.add_argument("--maximum-utilization-percent", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    args.controller_id = _safe_component(args.controller_id, label="controller ID")
    args.pair_run_id = _safe_component(args.pair_run_id, label="pair run ID")
    args.benchmark_run_id = _safe_component(
        args.benchmark_run_id, label="benchmark run ID"
    )
    for label in (
        "execution_commit",
        "pair_launch_spec_sha256",
        "benchmark_input_manifest_sha256",
    ):
        value = str(getattr(args, label))
        expected_length = 40 if label == "execution_commit" else 64
        if len(value) != expected_length or any(
            character not in "0123456789abcdef" for character in value
        ):
            parser.error(f"--{label.replace('_', '-')} must be lowercase hex")
    if args.poll_seconds <= 0 or args.stable_ready_seconds < 60:
        parser.error("poll must be positive and stable-ready at least 60 seconds")
    if args.pair_worker_pid <= 0 or args.pair_worker_start_ticks <= 0:
        parser.error("pair worker PID/start ticks must be positive")
    if args.gpu_index < 0:
        parser.error("GPU index must be non-negative")
    if args.minimum_free_memory_mb <= 0:
        parser.error("minimum free GPU memory must be positive")
    if not 0 <= args.maximum_utilization_percent <= 100:
        parser.error("maximum GPU utilization must be in [0,100]")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("warmups must be non-negative and repeats positive")
    for name in (
        "state_root",
        "project_root",
        "python",
        "pair_run_state_root",
        "lock_root",
        "registry_path",
        "benchmark_run_state_root",
        "benchmark_output_root",
        "benchmark_input_manifest",
        "benchmark_log_path",
        "dataset_dir",
        "checkpoint_dir",
        "config",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=False))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        reason = f"UNHANDLED_CONTROLLER_ERROR:{type(exc).__name__}:{exc}"
        if args.state_root.is_dir() and not args.state_root.is_symlink():
            try:
                _publish_blocked(args, reason=reason, observation=None)
                run_state = args.benchmark_run_state_root / "state.json"
                if run_state.is_file() and not run_state.is_symlink():
                    payload = read_json_object(run_state)
                    payload.update(
                        {
                            "state": "BLOCKED",
                            "updated_at": utc_now(),
                            "failures": [reason],
                        }
                    )
                    atomic_write_json(run_state, payload)
            except Exception:
                # Preserve the original exception and never pretend cleanup
                # succeeded when the filesystem itself is unhealthy.
                pass
        print(
            f"[BACE_GNN_INFERENCE_DEFERRED_CONTROLLER_ERROR] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
