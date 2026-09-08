"""T13-only adapter for the existing gpu_lock owner and resource provider.

No registry writes or second lock: the canonical T14 release evidence is
reopened before and after the original physical UUID lease is held.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from src.baselines.t13_real_batch_performance import GIB, validate_plan
from src.eval.bace_frozen_gnn_contracts import atomic_json

SCHEMA = "t13_real_batch_existing_owner_dispatch_v1"


def bound_json(descriptor):
    path = Path(descriptor["path"])
    if not path.is_absolute() or path.is_symlink() or path.stat().st_size > 2 * 1024 ** 2:
        raise ValueError("T13_DISPATCH_SMALL_ABSOLUTE_INPUT_REQUIRED")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("T13_DISPATCH_INPUT_HASH_CHANGED:" + str(path))
    return json.loads(raw)


def validate_dispatch(spec, *, project_root, gpu_index, gpu_uuid, lock_root):
    if (spec.get("schema") != SCHEMA or spec.get("task_family") != "t13_performance_diagnostic"
            or spec.get("science_started") is not False or spec.get("borrow_enabled") is not False
            or spec.get("registry_write_authorized_by_this_command") is not False
            or spec.get("main_matrix_write") is not False or spec.get("max_full_starts_consumed") != 0
            or spec.get("gpu_index") != 2 or gpu_index != 2 or spec.get("gpu_uuid") != gpu_uuid):
        raise ValueError("T13_PERFORMANCE_DISPATCH_SCOPE_CHANGED")
    if Path(spec["execution_root"]).resolve() != Path(project_root).resolve():
        raise ValueError("T13_PERFORMANCE_IMMUTABLE_CHECKOUT_CHANGED")
    plan = validate_plan(bound_json(spec["performance_plan"]))
    config = bound_json(spec["resource_config"])
    if Path(config["gpu_lock_root"]).resolve() != Path(lock_root).resolve():
        raise ValueError("T13_PERFORMANCE_MUST_USE_ORIGINAL_LOCK_ROOT")
    if (config["minimum_memory_headroom_bytes"] != 448 * GIB
            or spec["other_tasks_headroom_reserve_bytes"] != 384 * GIB
            or config["minimum_persistent_free_bytes"] != 100 * GIB
            or plan["process_peak_budget_bytes"] != 64 * GIB):
        raise ValueError("T13_PERFORMANCE_JOINT_MEMORY_CONTRACT_CHANGED")
    expected = [spec["python"], "-I", "-B", str(Path(project_root) / "scripts/benchmarks/benchmark_t13_real_batch.py"),
                "--config", str(Path(project_root) / "configs/hpc.yaml"), "--set", "inference.fallback_to_heuristic=false",
                "--plan", spec["performance_plan"]["path"], "--action", "run"]
    if spec["science_command_without_owner_fds"] != expected:
        raise ValueError("T13_PERFORMANCE_DATASET_SPECIFIC_ENTRYPOINT_CHANGED")
    return plan, config


def decision(spec, observation):
    blockers = list(observation.get("source_blockers", []))
    if not observation.get("memory_safe"):
        blockers.append("JOINT_HEADROOM_BELOW_BOUND_OTHER_RESERVE_PLUS_CANARY64_GIB")
    if not observation.get("storage_safe"):
        blockers.append("PERSISTENT_NEXT_STAGE_SPACE_OR_FILE_BOUND_NOT_ADMITTED")
    if observation.get("gpu_main_reservation"):
        blockers.append("MAIN_GPU2_RESERVATION_PRESENT")
    if observation.get("main_ready_waiting_gpu"):
        blockers.append("MAIN_READY_GPU_TASK_PRESENT")
    gpu = observation.get("actual_gpu_observation", {})
    if gpu.get("process_count", len(gpu.get("processes", [None]))) and not observation.get('gpu_child_pid'):
        blockers.append("GPU2_HAS_ACTUAL_PROCESS")
    if not spec.get("concurrent_file_peak_fully_bound", False):
        blockers.append("CONCURRENT_NEXT_BOUNDARY_FILE_PEAK_REQUIRES_CANONICAL_STAGE_BINDING")
    claim = spec.get("canonical_gpu2_diagnostic_claim")
    if not isinstance(claim, dict) or claim != {
            'task_id': spec.get('task_id'), 'gpu_uuid': spec.get('gpu_uuid'),
            'scope': 'RELEASED_GPU2_T13_DIAGNOSTIC_ONLY',
            'terminal_release': spec.get('terminal_release_binding')} or not spec.get('terminal_release_binding'):
        blockers.append("CANONICAL_GPU2_DIAGNOSTIC_CLAIM_NOT_BOUND")
    if observation.get('active_early_ablation_gpus', 0) != 0:
        blockers.append('ANOTHER_EARLY_GPU_OWNER_ACTIVE')
    return dict(state="ADMITTED_TO_EXISTING_OWNER_LEASE_ATTEMPT" if not blockers else "BLOCKED_RESOURCE_OR_BINDING",
                allowed=not blockers, blockers=blockers,
                science_started=False, science_pid=None, gpu_lease_acquired=False,
                automatic_waiting_owner_started=False, registry_modified=False,
                max_full_starts_consumed=0, safe_handover_performed=False,
                required_headroom_bytes=observation.get('required_headroom_bytes', 448 * GIB),
                actual_headroom_bytes=observation.get("memory_headroom_bytes"),
                next_binding_point="ResourceSampler.sample -> verify_terminal_dependency: canonical diagnostic claim and real inherited FD required before same-lock resampling",
                condition_for_reconsideration="Supply defensible next-boundary remaining-peak evidence or wait for genuine resource release; re-read source and canonical stage bindings")


def preflight(*, spec_descriptor, project_root, gpu_index, gpu_uuid, lock_root, output_root,
              sampler_factory=None):
    spec = bound_json(spec_descriptor)
    plan, config = validate_dispatch(spec, project_root=project_root, gpu_index=gpu_index,
                                    gpu_uuid=gpu_uuid, lock_root=lock_root)
    if sampler_factory is None:
        from src.ablations.llm.existing_gpu_owner import ResourceSampler
        sampler_factory = ResourceSampler
    output = Path(output_root)
    if output.exists():
        raise ValueError("T13_PREFLIGHT_FRESH_RECEIPT_ROOT_REQUIRED")
    # Sampling before the real canary must never claim checkpoint/resume PASS.
    sampler = sampler_factory(config, gpu_index, gpu_uuid, observation_only=True)
    observed = sampler.sample()
    result = decision(spec, observed)
    result['source_observation'] = observed
    result.update(observed_at_epoch_seconds=time.time(), inspector_pid=os.getpid(),
                  performance_plan=spec["performance_plan"], resource_config=spec["resource_config"],
                  execution_root=str(project_root), task_id=spec["task_id"],
                  future_science_command=spec["science_command_without_owner_fds"],
                  terminal_verifier_semantics="Unheld released T14 lease tested by existing independent flock; no held-child sample is claimed",
                  file_budget_scope=spec["file_budget_scope"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "preflight.json", result)
    print(json.dumps(result, sort_keys=True))
    return 75


def run(*, spec_descriptor, project_root, gpu_index, gpu_uuid, lock_root, output_root,
        wait_seconds=0, refresh_seconds=60):
    """Use the original bounded owner; absent bindings remain non-running."""
    from src.ablations.llm.existing_gpu_owner import ResourceSampler, run_owned_child
    from src.utils.autodl_runtime import sanitized_environment
    spec = bound_json(spec_descriptor)
    plan, config = validate_dispatch(spec, project_root=project_root, gpu_index=gpu_index,
                                    gpu_uuid=gpu_uuid, lock_root=lock_root)
    dependencies = config.get('terminal_resource_dependencies', [])
    if dependencies != [spec.get('terminal_release_binding')]:
        raise ValueError('T13_ORIGINAL_TERMINAL_RELEASE_NOT_BOUND')
    backend = json.loads(Path(plan['source_runtime_backend_receipt']).read_text())
    from src.utils.t13_deterministic_execution import BACKEND
    if (backend.get('observed_backend') != BACKEND
            or backend.get('torch_num_threads') != plan['torch_num_threads']
            or backend.get('torch_num_interop_threads') != plan['torch_num_interop_threads']):
        raise ValueError('T13_SOURCE_BACKEND_CHANGED')
    env = dict(sanitized_environment(), **backend['thread_environment'])
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(Path(output_root).parent), CUDA_VISIBLE_DEVICES=gpu_uuid)
    sampler = ResourceSampler(config, gpu_index, gpu_uuid, observation_only=True,
        task_family='t13_performance_diagnostic', reach_contract=spec_descriptor)
    return run_owned_child(command=spec['science_command_without_owner_fds'],
        environment=env, sampler=sampler, output_root=output_root, lock_root=lock_root,
        run_id=spec['task_id'], interval=refresh_seconds, max_wait_seconds=wait_seconds)


def child_evidence(value, *, plan_sha):
    """Adapt actual inherited-provider fields, never invent a resource PASS."""
    from datetime import datetime
    if (value.get('task_family') != 't13_performance_diagnostic'
            or value.get('plan_sha256') != plan_sha or value.get('source_error')
            or value.get('pause_requested')):
        raise ValueError('T13_CHILD_PROVIDER_SCOPE_OR_PAUSE')
    status = value.get('t13_admission', {})
    return dict(value, resource_admission='PASS' if status.get('allowed') else 'BLOCKED',
        observed_at_epoch_seconds=datetime.fromisoformat(value['observed_at'].replace('Z', '+00:00')).timestamp(),
        owner_pid=value['gpu_owner_pid'], child_pid=value['gpu_child_pid'],
        owner_start_ticks=value['gpu_owner_start_ticks'], child_start_ticks=value['gpu_child_start_ticks'],
        other_task_headroom_required_bytes=value['other_tasks_headroom_reserve_bytes'])
