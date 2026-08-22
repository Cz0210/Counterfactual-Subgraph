#!/usr/bin/env python3
"""Persistent, fail-closed four-GPU AutoDL recovery scheduler.

The controller is deliberately a control-plane component.  Scientific work is
declared as argv arrays in a frozen manifest and is launched through
``scripts/autodl/exp_run.py``.  Consequently GPU UUID locks, detached workers,
stage gates, result contracts, and the canonical experiment registry retain a
single implementation.

This file does not implement MUT/AIDS recovery, PPO, candidate generation,
verification, or selector science.  Those entrypoints are integration inputs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO

from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    BACE_STAGES,
    FOUR_GPU_RECOVERY_LIMIT,
    GPUObservation,
    append_jsonl_locked,
    assert_bace_stage_can_start,
    atomic_write_json,
    build_runtime_layout,
    command_digest,
    fsync_directory,
    gpu_lock_available,
    observe_stable_idle_gpus,
    query_gpu_inventory,
    read_bace_stage,
    read_json_object,
    resolve_project_root,
    sanitized_environment,
    select_data_root,
    sha256_file,
    sha256_paths,
    stage_paths,
    update_bace_stage_state,
    utc_now,
    validate_max_gpus,
    verify_required_absolute_outputs,
    verify_required_output_alternatives,
    verify_required_outputs,
)
from src.train.stable_ppo_resume import (
    find_latest_stable_ppo_resume_checkpoint,
    read_stable_ppo_resume_manifest,
)
from src.utils.autodl_bace_continuation import (
    BaceContinuationError,
    PredecessorControllerGuard,
    build_bace_continuation_payload,
    validate_continuation_policy,
)


SCHEMA_VERSION = 1
CONTROLLER_NAME = "four_gpu_recovery"
DEFAULT_AUTODL_PYTHON = "/root/miniconda3/envs/smiles_pip118/bin/python"
RECOVERY_BACE_STAGES = ("B6_PPO_SMOKE_V2", *BACE_STAGES[7:])
RECOVERY_BACE_PREDECESSOR = {
    stage: ("B5_ORACLE_SMOKE" if index == 0 else RECOVERY_BACE_STAGES[index - 1])
    for index, stage in enumerate(RECOVERY_BACE_STAGES)
}
# Base and high-temperature pools are independent fixed-shard consumers of B7;
# their join occurs only at B10.
RECOVERY_BACE_PREDECESSOR["B9_POOL_HIGHTEMP"] = "B7_PPO_FULL"
CONTROLLER_STATES = {
    "NOT_STARTED",
    "WAITING_DEPENDENCY",
    "READY",
    "WAITING_RESOURCE",
    "STARTING",
    "RUNNING",
    "PASS",
    "FAILED",
    "BLOCKED",
    "SKIPPED",
}
ACTIVE_STATES = {"STARTING", "RUNNING"}
TERMINAL_STATES = {"PASS", "FAILED", "BLOCKED", "SKIPPED"}
FAILURE_STATES = {"FAILED", "BLOCKED", "SKIPPED"}
OOM_MARKERS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cudnn_status_alloc_failed",
    "hip out of memory",
)
TRANSIENT_IO_MARKERS = (
    "input/output error",
    "errno 5",
    "stale file handle",
    "transport endpoint is not connected",
    "connection reset by peer",
    "connection timed out",
    "resource temporarily unavailable",
)
TRANSIENT_FAILURE_CLASSES = frozenset({"TRANSIENT_IO", "TRANSIENT_PROCESS_LOSS"})
ALLOCATION_SAFE_GPU_AUDITS = frozenset({"AVAILABLE", "STALE_METADATA"})
DEFAULT_LAUNCH_GRACE_SECONDS = 180
DEFAULT_MAX_TRANSIENT_RETRIES = 1
THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
DEFAULT_SEMANTIC_MARKERS = (
    "provenance gate failed",
    "semantic gate failed",
    "test leakage",
    "rf_oracle_used=true",
    "oracle_backend=rf",
    "strict flip gate failed",
)
SECRET_KEY = re.compile(
    r"(?i)(password|passwd|secret|token|authorization|api[_-]?key|"
    r"credential|private[_-]?key)"
)
ALLOWED_SECRET_LIKE_ENV_KEYS = frozenset({"TOKENIZERS_PARALLELISM"})
SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
TOKEN = re.compile(r"\{([A-Za-z0-9_]+)\}")
TEST_PATH = re.compile(r"(?i)(?:^|[/_.-])test(?:[/_.-]|$)")
POST_FREEZE_TEST_STAGES = {
    "B13_TEST_PARENT_MANIFEST",
    "B13_FINAL_EVAL_SHARDS",
    "B13_FINAL_EVAL",
}


class ControllerError(AutoDLRuntimeError):
    """The controller cannot proceed without weakening an invariant."""


@dataclass(frozen=True)
class OOMRetry:
    enabled: bool = False
    batch_env: str = "BATCH_SIZE"
    initial_batch_size: int | None = None
    retry_batch_size: int | None = None


@dataclass(frozen=True)
class FixedShardSpec:
    count: int
    parent_manifest: str
    split: str
    parent_id_key: str = "parent_ids"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset: str
    stage: str
    command: tuple[str, ...] | None
    depends_on: tuple[str, ...]
    resource: str
    priority: int
    enabled: bool
    blocked_reason: str | None
    skip_reason: str | None
    data_splits: tuple[str, ...]
    manifest_only: bool
    runner_dataset: str
    runner_stage: str
    external_bace_stage: str | None
    adopt_existing_run_id: str | None
    adopt_gpu_index: int | None
    adopt_gpu_uuid: str | None
    adopt_project_root: str | None
    adopt_git_commit: str | None
    adopt_max_gpus: int | None
    adopt_heavy: bool | None
    config_files: tuple[str, ...]
    input_manifest: str | None
    expected_output: str | None
    required_output_files: tuple[str, ...]
    required_output_any: tuple[tuple[str, ...], ...]
    required_absolute_output_files: tuple[str, ...]
    required_log_marker: str | None
    environment: Mapping[str, str]
    semantic_failure_markers: tuple[str, ...]
    oom_retry: OOMRetry
    shards: FixedShardSpec | None
    publish_bace_stage: bool
    freezes_selector: bool
    selector_parameters_frozen: bool
    read_only_test: bool

    @property
    def instance_ids(self) -> tuple[str, ...]:
        if self.shards is None:
            return ("main",)
        return tuple(f"shard-{index:03d}" for index in range(self.shards.count))


@dataclass(frozen=True)
class ControllerManifest:
    path: Path
    sha256: str
    controller_id: str
    tasks: tuple[TaskSpec, ...]
    runtime: Mapping[str, Any]
    resource_gates: Mapping[str, Any]

    @property
    def by_id(self) -> dict[str, TaskSpec]:
        return {task.task_id: task for task in self.tasks}


@dataclass(frozen=True)
class HostResources:
    cpu_count: int
    load_1m: float
    available_ram_gb: float
    free_disk_gb: float


def _safe_id(value: str, *, label: str) -> str:
    normalized = SAFE_ID.sub("-", value).strip("._-")
    if not normalized:
        raise ControllerError(f"{label} is empty after normalization")
    if normalized != value:
        raise ControllerError(f"{label} contains unsafe characters: {value!r}")
    return normalized[:120]


def controller_safety_environment(
    *, cpu_count: int | None = None
) -> dict[str, str]:
    """Return the controller-owned CPU/tokenizer limits for one task.

    Four concurrent GPU workers must not each inherit the full host thread
    pool.  These values are controller policy, not user-tunable scientific
    inputs, and are frozen into every newly launched task's evidence.
    """

    total = os.cpu_count() if cpu_count is None else cpu_count
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        total = 1
    threads = max(1, total // FOUR_GPU_RECOVERY_LIMIT - 1)
    return {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
    }


def _effective_launch_environment(task: TaskSpec) -> dict[str, str]:
    """Merge immutable task variables with non-overridable safety defaults."""

    environment = dict(task.environment)
    if task.adopt_existing_run_id is not None:
        # Adoption verifies the exact historical launch environment and never
        # retroactively claims that controller defaults were present.
        return environment
    safety = controller_safety_environment()
    conflicts = {
        key: (environment[key], value)
        for key, value in safety.items()
        if key in environment and environment[key] != value
    }
    if conflicts:
        raise ControllerError(
            f"{task.task_id} may not override controller safety environment: "
            f"{conflicts}"
        )
    environment.update(safety)
    return environment


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ControllerError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ControllerError(f"Controller manifest must be one object: {path}")
    return payload


def _as_string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ControllerError(f"{label} must be a list of nonempty strings")
    return tuple(value)


def _parse_oom_retry(raw: Any, *, task_id: str) -> OOMRetry:
    if raw is None:
        return OOMRetry()
    if not isinstance(raw, dict):
        raise ControllerError(f"{task_id}.oom_retry must be an object")
    enabled = bool(raw.get("enabled", False))
    batch_env = str(raw.get("batch_env", "BATCH_SIZE"))
    initial = raw.get("initial_batch_size")
    retry = raw.get("retry_batch_size")
    if SECRET_KEY.search(batch_env) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", batch_env):
        raise ControllerError(f"{task_id}.oom_retry.batch_env is unsafe")
    if enabled:
        if not isinstance(initial, int) or not isinstance(retry, int):
            raise ControllerError(
                f"{task_id} OOM retry requires integer initial/retry batch sizes"
            )
        if initial <= 0 or retry <= 0 or retry >= initial:
            raise ControllerError(
                f"{task_id} OOM retry must lower a positive batch size exactly once"
            )
    return OOMRetry(enabled, batch_env, initial, retry)


def _parse_shards(raw: Any, *, task_id: str) -> FixedShardSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ControllerError(f"{task_id}.shards must be an object")
    count = raw.get("count")
    parent_manifest = raw.get("parent_manifest")
    split = raw.get("split")
    parent_id_key = raw.get("parent_id_key", "parent_ids")
    if not isinstance(count, int) or count < 2 or count > FOUR_GPU_RECOVERY_LIMIT:
        raise ControllerError(f"{task_id}.shards.count must be in [2, 4]")
    if not isinstance(parent_manifest, str) or not parent_manifest:
        raise ControllerError(f"{task_id}.shards.parent_manifest is required")
    if not isinstance(split, str) or not split:
        raise ControllerError(f"{task_id}.shards.split is required")
    if not isinstance(parent_id_key, str) or not parent_id_key:
        raise ControllerError(f"{task_id}.shards.parent_id_key is required")
    return FixedShardSpec(count, parent_manifest, split.lower(), parent_id_key)


def _parse_output_alternatives(raw: Any, *, task_id: str) -> tuple[tuple[str, ...], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ControllerError(f"{task_id}.required_output_any must be a list")
    groups: list[tuple[str, ...]] = []
    for index, group in enumerate(raw):
        values = _as_string_tuple(
            group, label=f"{task_id}.required_output_any[{index}]"
        )
        if len(values) < 2:
            raise ControllerError(
                f"{task_id}.required_output_any[{index}] needs at least two alternatives"
            )
        groups.append(values)
    return tuple(groups)


def _parse_task(raw: Any) -> TaskSpec:
    if not isinstance(raw, dict):
        raise ControllerError("Each task must be an object")
    task_id = _safe_id(str(raw.get("id", "")), label="task id")
    dataset = str(raw.get("dataset", "")).strip().lower()
    stage = str(raw.get("stage", "")).strip()
    if not dataset or not stage:
        raise ControllerError(f"{task_id} requires dataset and stage")
    command_value = raw.get("command")
    if command_value is None:
        command: tuple[str, ...] | None = None
    else:
        command = _as_string_tuple(command_value, label=f"{task_id}.command")
        if not command:
            raise ControllerError(f"{task_id}.command cannot be empty")
    environment = raw.get("environment", {})
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, (str, int, float, bool))
        for key, value in environment.items()
    ):
        raise ControllerError(f"{task_id}.environment must contain scalar values")
    for key in environment:
        if SECRET_KEY.search(key) and key not in ALLOWED_SECRET_LIKE_ENV_KEYS:
            raise ControllerError(f"{task_id} has credential-like environment key")
    if str(environment.get("PYTHONDONTWRITEBYTECODE", "1")) != "1":
        raise ControllerError(
            f"{task_id}.environment must keep PYTHONDONTWRITEBYTECODE=1"
        )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    blocked_reason = raw.get("blocked_reason")
    skip_reason = raw.get("skip_reason")
    if blocked_reason is not None and not isinstance(blocked_reason, str):
        raise ControllerError(f"{task_id}.blocked_reason must be a string")
    if skip_reason is not None and not isinstance(skip_reason, str):
        raise ControllerError(f"{task_id}.skip_reason must be a string")
    resource = str(raw.get("resource", "gpu")).lower()
    if resource not in {"gpu", "cpu"}:
        raise ControllerError(f"{task_id}.resource must be gpu or cpu")
    priority = raw.get("priority", 100)
    if not isinstance(priority, int):
        raise ControllerError(f"{task_id}.priority must be an integer")
    shards = _parse_shards(raw.get("shards"), task_id=task_id)
    runner_dataset = str(raw.get("runner_dataset", dataset)).strip().lower()
    runner_stage = str(raw.get("runner_stage", stage)).strip()
    if shards is not None and "{shard_id}" not in runner_stage:
        raise ControllerError(
            f"{task_id}.runner_stage must include {{shard_id}} for isolated stage state"
        )
    if shards is not None and runner_dataset == "bace" and stage in BACE_STAGES:
        raise ControllerError(
            f"{task_id} shards must use a non-primary runner_dataset; the controller "
            "publishes the aggregate BACE stage only after every shard passes"
        )
    oom_retry = _parse_oom_retry(raw.get("oom_retry"), task_id=task_id)
    expected_output = raw.get("expected_output")
    if expected_output is not None and not isinstance(expected_output, str):
        raise ControllerError(f"{task_id}.expected_output must be a string")
    if oom_retry.enabled and expected_output and "{attempt}" not in expected_output:
        raise ControllerError(
            f"{task_id} OOM retry output must include {{attempt}} to remain immutable"
        )
    if shards is not None and expected_output and "{shard_id}" not in expected_output:
        raise ControllerError(
            f"{task_id} sharded output must include {{shard_id}}"
        )
    external = raw.get("external_bace_stage")
    if external is not None and external not in BACE_STAGES:
        raise ControllerError(f"{task_id} has invalid external_bace_stage={external}")
    adopted_run = raw.get("adopt_existing_run_id")
    if adopted_run is not None:
        if not isinstance(adopted_run, str):
            raise ControllerError(f"{task_id}.adopt_existing_run_id must be a string")
        adopted_run = _safe_id(adopted_run, label="adopt_existing_run_id")
        if shards is not None:
            raise ControllerError(f"{task_id} cannot adopt one run into a sharded task")
    adopt_gpu_index = raw.get("adopt_gpu_index")
    adopt_gpu_uuid = raw.get("adopt_gpu_uuid")
    if (adopt_gpu_index is None) != (adopt_gpu_uuid is None):
        raise ControllerError(
            f"{task_id}.adopt_gpu_index/adopt_gpu_uuid must be provided together"
        )
    if adopt_gpu_index is not None:
        if (
            isinstance(adopt_gpu_index, bool)
            or not isinstance(adopt_gpu_index, int)
            or adopt_gpu_index < 0
        ):
            raise ControllerError(f"{task_id}.adopt_gpu_index is invalid")
        if not isinstance(adopt_gpu_uuid, str) or not adopt_gpu_uuid:
            raise ControllerError(f"{task_id}.adopt_gpu_uuid is invalid")
    if adopted_run is None and adopt_gpu_index is not None:
        raise ControllerError(f"{task_id} declares an adopt GPU without an adopted run")
    if adopted_run is not None and resource == "gpu" and adopt_gpu_index is None:
        raise ControllerError(
            f"{task_id} GPU adoption requires exact adopt_gpu_index/adopt_gpu_uuid"
        )
    adopt_project_root = raw.get("adopt_project_root")
    adopt_git_commit = raw.get("adopt_git_commit")
    adopt_max_gpus = raw.get("adopt_max_gpus")
    adopt_heavy = raw.get("adopt_heavy")
    adoption_closure = (
        adopt_project_root,
        adopt_git_commit,
        adopt_max_gpus,
        adopt_heavy,
    )
    if adopted_run is None and any(value is not None for value in adoption_closure):
        raise ControllerError(f"{task_id} declares adoption closure without a run ID")
    if adopted_run is not None:
        if not isinstance(adopt_project_root, str) or not Path(
            adopt_project_root
        ).expanduser().is_absolute():
            raise ControllerError(f"{task_id}.adopt_project_root must be absolute")
        if not isinstance(adopt_git_commit, str) or re.fullmatch(
            r"[0-9a-f]{40}", adopt_git_commit
        ) is None:
            raise ControllerError(f"{task_id}.adopt_git_commit must be a full SHA")
        if isinstance(adopt_max_gpus, bool) or not isinstance(adopt_max_gpus, int):
            raise ControllerError(f"{task_id}.adopt_max_gpus is required")
        validate_max_gpus(
            adopt_max_gpus, hard_limit=FOUR_GPU_RECOVERY_LIMIT
        )
        if not isinstance(adopt_heavy, bool):
            raise ControllerError(f"{task_id}.adopt_heavy must be boolean")
    data_splits = tuple(
        value.lower()
        for value in _as_string_tuple(
            raw.get("data_splits", []), label=f"{task_id}.data_splits"
        )
    )
    manifest_only = bool(raw.get("manifest_only", False))
    if manifest_only and data_splits:
        raise ControllerError(
            f"{task_id} manifest_only task may not declare raw data_splits"
        )
    return TaskSpec(
        task_id=task_id,
        dataset=dataset,
        stage=stage,
        command=command,
        depends_on=_as_string_tuple(raw.get("depends_on", []), label=f"{task_id}.depends_on"),
        resource=resource,
        priority=priority,
        enabled=bool(raw.get("enabled", True)),
        blocked_reason=blocked_reason,
        skip_reason=skip_reason,
        data_splits=data_splits,
        manifest_only=manifest_only,
        runner_dataset=runner_dataset,
        runner_stage=runner_stage,
        external_bace_stage=external,
        adopt_existing_run_id=adopted_run,
        adopt_gpu_index=adopt_gpu_index,
        adopt_gpu_uuid=adopt_gpu_uuid,
        adopt_project_root=adopt_project_root,
        adopt_git_commit=adopt_git_commit,
        adopt_max_gpus=adopt_max_gpus,
        adopt_heavy=adopt_heavy,
        config_files=_as_string_tuple(raw.get("config_files", []), label=f"{task_id}.config_files"),
        input_manifest=(
            str(raw["input_manifest"]) if raw.get("input_manifest") is not None else None
        ),
        expected_output=expected_output,
        required_output_files=_as_string_tuple(
            raw.get("required_output_files", []),
            label=f"{task_id}.required_output_files",
        ),
        required_output_any=_parse_output_alternatives(
            raw.get("required_output_any"), task_id=task_id
        ),
        required_absolute_output_files=_as_string_tuple(
            raw.get("required_absolute_output_files", []),
            label=f"{task_id}.required_absolute_output_files",
        ),
        required_log_marker=(
            str(raw["required_log_marker"])
            if raw.get("required_log_marker") is not None
            else None
        ),
        environment={str(key): str(value) for key, value in environment.items()},
        semantic_failure_markers=tuple(
            value.lower()
            for value in _as_string_tuple(
                raw.get("semantic_failure_markers", list(DEFAULT_SEMANTIC_MARKERS)),
                label=f"{task_id}.semantic_failure_markers",
            )
        ),
        oom_retry=oom_retry,
        shards=shards,
        publish_bace_stage=bool(raw.get("publish_bace_stage", False)),
        freezes_selector=bool(raw.get("freezes_selector", False)),
        selector_parameters_frozen=bool(raw.get("selector_parameters_frozen", False)),
        read_only_test=bool(raw.get("read_only_test", False)),
    )


def dependency_order(tasks: Sequence[TaskSpec]) -> list[str]:
    """Return deterministic topological order or fail on an unknown/cyclic edge."""

    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ControllerError("Task IDs must be unique")
    unknown = sorted(
        dependency
        for task in tasks
        for dependency in task.depends_on
        if dependency not in by_id
    )
    if unknown:
        raise ControllerError(f"Unknown task dependencies: {unknown}")
    indegree = {task.task_id: len(set(task.depends_on)) for task in tasks}
    children: dict[str, list[str]] = {task.task_id: [] for task in tasks}
    for task in tasks:
        if len(set(task.depends_on)) != len(task.depends_on):
            raise ControllerError(f"{task.task_id} repeats a dependency")
        for dependency in task.depends_on:
            children[dependency].append(task.task_id)
    ready = sorted(
        (task for task in tasks if indegree[task.task_id] == 0),
        key=lambda task: (task.priority, task.task_id),
    )
    result: list[str] = []
    while ready:
        task = ready.pop(0)
        result.append(task.task_id)
        for child_id in sorted(children[task.task_id]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(by_id[child_id])
                ready.sort(key=lambda value: (value.priority, value.task_id))
    if len(result) != len(tasks):
        cycle = sorted(task_id for task_id, degree in indegree.items() if degree)
        raise ControllerError(f"Task dependency cycle: {cycle}")
    return result


def _transitive_dependencies(task_id: str, tasks: Mapping[str, TaskSpec]) -> set[str]:
    result: set[str] = set()
    pending = list(tasks[task_id].depends_on)
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(tasks[dependency].depends_on)
    return result


def validate_no_test_before_freeze(tasks: Sequence[TaskSpec]) -> None:
    """Keep held-out test bytes unreachable until the B12 selector is frozen."""

    by_id = {task.task_id: task for task in tasks}
    selector_ids = {
        task.task_id
        for task in tasks
        if task.stage == "B12_SELECTOR" and task.freezes_selector
    }
    for task in tasks:
        uses_test = "test" in task.data_splits
        path_fields = [
            *(task.command or ()),
            *task.config_files,
            *(task.environment.values()),
            task.input_manifest or "",
        ]
        if task.shards is not None:
            path_fields.append(task.shards.parent_manifest)
        path_mentions_test = any(TEST_PATH.search(value) for value in path_fields)
        if path_mentions_test and not uses_test:
            raise ControllerError(
                f"{task.task_id} references a test-looking input without declaring test access"
            )
        if not uses_test:
            continue
        if task.stage not in POST_FREEZE_TEST_STAGES:
            raise ControllerError(
                f"{task.task_id} accesses test before/after the one-shot B13 boundary"
            )
        ancestors = _transitive_dependencies(task.task_id, by_id)
        if not selector_ids.intersection(ancestors):
            raise ControllerError(
                f"{task.task_id} test access requires a frozen B12 selector dependency"
            )
        if not task.selector_parameters_frozen or not task.read_only_test:
            raise ControllerError(
                f"{task.task_id} must declare selector_parameters_frozen and read_only_test"
            )


def _validate_bace_order(tasks: Sequence[TaskSpec]) -> None:
    recovery_tasks = [
        task
        for task in tasks
        if task.dataset == "bace" and task.stage in RECOVERY_BACE_STAGES
    ]
    by_stage = {task.stage: task for task in recovery_tasks}
    if len(by_stage) != len(recovery_tasks):
        raise ControllerError("Recovery BACE stages must be unique")
    by_id = {task.task_id: task for task in tasks}
    for stage, task in by_stage.items():
        predecessor = RECOVERY_BACE_PREDECESSOR[stage]
        if stage == "B6_PPO_SMOKE_V2" and predecessor == task.external_bace_stage:
            continue
        predecessor_task = by_stage.get(predecessor)
        if predecessor_task is None:
            if predecessor != task.external_bace_stage:
                raise ControllerError(
                    f"{task.task_id} must bind predecessor {predecessor} as a task "
                    "dependency or external_bace_stage"
                )
            continue
        if predecessor_task.task_id not in _transitive_dependencies(task.task_id, by_id):
            raise ControllerError(
                f"{task.task_id} does not depend on predecessor {predecessor_task.task_id}"
            )


def load_controller_manifest(path: Path) -> ControllerManifest:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ControllerError(f"Manifest must be a physical file: {resolved}")
    payload = _load_json_or_yaml(resolved)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ControllerError("Unsupported four-GPU controller manifest schema")
    if payload.get("paper_frozen") is not True:
        raise ControllerError("paper_frozen=true is mandatory")
    controller_id = _safe_id(
        str(payload.get("controller_id", "")), label="controller_id"
    )
    try:
        validate_continuation_policy(
            payload.get("continuation"), controller_id=controller_id
        )
    except BaceContinuationError as exc:
        raise ControllerError(str(exc)) from exc
    runtime = payload.get("runtime", {})
    resources = payload.get("resource_gates", {})
    if not isinstance(runtime, dict) or not isinstance(resources, dict):
        raise ControllerError("runtime/resource_gates must be objects")
    max_gpus = runtime.get("max_gpus", FOUR_GPU_RECOVERY_LIMIT)
    validate_max_gpus(int(max_gpus), hard_limit=FOUR_GPU_RECOVERY_LIMIT)
    if int(max_gpus) != FOUR_GPU_RECOVERY_LIMIT:
        raise ControllerError("The recovery controller must retain a four-GPU ceiling")
    if float(runtime.get("stable_idle_seconds", 60)) < 60:
        raise ControllerError("stable_idle_seconds must be at least 60")
    if float(runtime.get("poll_seconds", 60)) < 60:
        raise ControllerError("poll_seconds must be at least 60")
    if float(runtime.get("sample_interval_seconds", 5)) <= 0:
        raise ControllerError("sample_interval_seconds must be positive")
    launch_grace_seconds = int(
        runtime.get("launch_grace_seconds", DEFAULT_LAUNCH_GRACE_SECONDS)
    )
    if launch_grace_seconds < 30 or launch_grace_seconds > 900:
        raise ControllerError("launch_grace_seconds must be in [30, 900]")
    max_transient_retries = int(
        runtime.get("max_transient_retries", DEFAULT_MAX_TRANSIENT_RETRIES)
    )
    if max_transient_retries not in {0, DEFAULT_MAX_TRANSIENT_RETRIES}:
        raise ControllerError("max_transient_retries must be 0 or 1")
    keep_alive_when_blocked = runtime.get("keep_alive_when_blocked", False)
    if not isinstance(keep_alive_when_blocked, bool):
        raise ControllerError("runtime.keep_alive_when_blocked must be boolean")
    tasks_value = payload.get("tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise ControllerError("Manifest requires a nonempty tasks list")
    tasks = tuple(_parse_task(value) for value in tasks_value)
    dependency_order(tasks)
    validate_no_test_before_freeze(tasks)
    _validate_bace_order(tasks)
    for task in tasks:
        # Evaluate safety-owned launch defaults while the manifest is loaded so
        # a conflicting user environment fails before any state is created.
        _effective_launch_environment(task)
        if task.dataset in {"paper", "manuscript"}:
            raise ControllerError("The controller may not schedule paper work")
        strings: Iterable[str] = (
            *(task.command or ()),
            *task.config_files,
            task.input_manifest or "",
            task.expected_output or "",
            *task.required_absolute_output_files,
            *(task.environment.values()),
            task.adopt_gpu_uuid or "",
            task.adopt_project_root or "",
            task.adopt_git_commit or "",
        )
        if any("paper/" in value.replace("\\", "/").lower() for value in strings):
            raise ControllerError(f"{task.task_id} attempts to access frozen paper files")
        runnable = (
            task.enabled
            and task.skip_reason is None
            and task.blocked_reason is None
            and task.dataset not in {"tastemolnet", "taste"}
        )
        if runnable:
            if any("__CONFIGURE_" in value for value in strings):
                raise ControllerError(
                    f"Runnable task {task.task_id} still contains configuration placeholders"
                )
            if task.command is None:
                raise ControllerError(f"Runnable task {task.task_id} has no command")
            if not task.input_manifest:
                raise ControllerError(
                    f"Runnable task {task.task_id} requires an input_manifest"
                )
            if not task.expected_output or (
                not task.required_output_files
                and not task.required_output_any
                and not task.required_absolute_output_files
            ):
                raise ControllerError(
                    f"Runnable task {task.task_id} requires an immutable output contract"
                )
            if (
                max_transient_retries
                and task.adopt_existing_run_id is None
                and "{attempt}" not in task.expected_output
            ):
                raise ControllerError(
                    f"Runnable task {task.task_id} transient retry output must include "
                    "{attempt} to remain immutable"
                )
            if (
                max_transient_retries
                and task.adopt_existing_run_id is None
                and any(
                    "{attempt}" not in value
                    for value in task.required_absolute_output_files
                )
            ):
                raise ControllerError(
                    f"Runnable task {task.task_id} transient retry absolute outputs must "
                    "include {attempt}"
                )
            if not task.required_log_marker:
                raise ControllerError(
                    f"Runnable task {task.task_id} requires a PASS log marker"
                )
            if not task.data_splits and not task.manifest_only:
                raise ControllerError(
                    f"Runnable task {task.task_id} must declare data_splits or manifest_only"
                )
    return ControllerManifest(
        path=resolved,
        sha256=sha256_file(resolved),
        controller_id=controller_id,
        tasks=tasks,
        runtime=dict(runtime),
        resource_gates=dict(resources),
    )


def _controller_root(layout: Any, controller_id: str) -> Path:
    return layout.control_root / CONTROLLER_NAME / controller_id


def _task_root(root: Path, task_id: str) -> Path:
    return root / "tasks" / task_id


def _task_paths(root: Path, task_id: str) -> dict[str, Path]:
    task_root = _task_root(root, task_id)
    return {
        "root": task_root,
        "state": task_root / "state.json",
        "manifest": task_root / "manifest.json",
        "gate": task_root / "gate.json",
        "shards": task_root / "shards",
    }


def _task_manifest_payload(task: TaskSpec, manifest_sha256: str) -> dict[str, Any]:
    effective_environment = _effective_launch_environment(task)
    return {
        "schema_version": SCHEMA_VERSION,
        "controller_manifest_sha256": manifest_sha256,
        "task_id": task.task_id,
        "dataset": task.dataset,
        "stage": task.stage,
        "depends_on": list(task.depends_on),
        "resource": task.resource,
        "priority": task.priority,
        "enabled": task.enabled,
        "blocked_reason": task.blocked_reason,
        "skip_reason": task.skip_reason,
        "command": list(task.command) if task.command else None,
        "runner_dataset": task.runner_dataset,
        "runner_stage": task.runner_stage,
        "external_bace_stage": task.external_bace_stage,
        "adopt_existing_run_id": task.adopt_existing_run_id,
        "adopt_gpu_index": task.adopt_gpu_index,
        "adopt_gpu_uuid": task.adopt_gpu_uuid,
        "adopt_project_root": task.adopt_project_root,
        "adopt_git_commit": task.adopt_git_commit,
        "adopt_max_gpus": task.adopt_max_gpus,
        "adopt_heavy": task.adopt_heavy,
        "input_manifest": task.input_manifest,
        "expected_output": task.expected_output,
        "required_output_files": list(task.required_output_files),
        "required_output_any": [list(group) for group in task.required_output_any],
        "required_absolute_output_files": list(
            task.required_absolute_output_files
        ),
        "required_log_marker": task.required_log_marker,
        "environment": dict(sorted(task.environment.items())),
        "controller_safety_environment": (
            {}
            if task.adopt_existing_run_id is not None
            else controller_safety_environment()
        ),
        "effective_launch_environment": dict(sorted(effective_environment.items())),
        "config_files": list(task.config_files),
        "semantic_failure_markers": list(task.semantic_failure_markers),
        "data_splits": list(task.data_splits),
        "manifest_only": task.manifest_only,
        "publish_bace_stage": task.publish_bace_stage,
        "freezes_selector": task.freezes_selector,
        "selector_parameters_frozen": task.selector_parameters_frozen,
        "read_only_test": task.read_only_test,
        "shards": (
            {
                "count": task.shards.count,
                "parent_manifest": task.shards.parent_manifest,
                "split": task.shards.split,
                "parent_id_key": task.shards.parent_id_key,
            }
            if task.shards
            else None
        ),
        "oom_retry": {
            "enabled": task.oom_retry.enabled,
            "batch_env": task.oom_retry.batch_env,
            "initial_batch_size": task.oom_retry.initial_batch_size,
            "retry_batch_size": task.oom_retry.retry_batch_size,
        },
        "status": "FROZEN",
    }


def _initial_instance(instance_id: str) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "state": "NOT_STARTED",
        "attempt": 0,
        "run_id": None,
        "adopted": False,
        "launcher_pid": None,
        "worker_pid": None,
        "child_pid": None,
        "worker_identity": None,
        "launcher_identity": None,
        "gpu_index": None,
        "gpu_uuid": None,
        "heartbeat_at": None,
        "failure_class": None,
        "failure_reason": None,
        "oom_retry_count": 0,
        "transient_retry_count": 0,
        "retry_kind": None,
        "resume_from_checkpoint": None,
        "resume_source_output": None,
    }


def initialize_controller_state(
    layout: Any, manifest: ControllerManifest
) -> tuple[Path, dict[str, dict[str, Any]]]:
    root = _controller_root(layout, manifest.controller_id)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "controller_manifest.json"
    snapshot_payload = _load_json_or_yaml(manifest.path)
    snapshot_payload["source_manifest"] = str(manifest.path)
    snapshot_payload["source_manifest_sha256"] = manifest.sha256
    if snapshot.exists():
        previous = read_json_object(snapshot)
        if previous.get("source_manifest_sha256") != manifest.sha256:
            raise ControllerError(
                "Persistent controller manifest is already frozen with different bytes"
            )
    else:
        atomic_write_json(snapshot, snapshot_payload)

    states: dict[str, dict[str, Any]] = {}
    for task in manifest.tasks:
        paths = _task_paths(root, task.task_id)
        paths["root"].mkdir(parents=True, exist_ok=True)
        task_manifest = _task_manifest_payload(task, manifest.sha256)
        if paths["manifest"].exists():
            if read_json_object(paths["manifest"]) != task_manifest:
                raise ControllerError(f"Frozen task manifest changed: {task.task_id}")
        else:
            atomic_write_json(paths["manifest"], task_manifest)
        if not paths["gate"].exists():
            atomic_write_json(
                paths["gate"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": task.task_id,
                    "status": "NOT_EVALUATED",
                    "checked_at": None,
                    "reason": None,
                    "runs": [],
                },
            )
        if paths["state"].exists():
            state = read_json_object(paths["state"])
            if set((state.get("instances") or {}).keys()) != set(task.instance_ids):
                raise ControllerError(
                    f"Persistent instance topology changed: {task.task_id}"
                )
        else:
            initial_state = "NOT_STARTED"
            reason: str | None = None
            if task.dataset in {"tastemolnet", "taste"}:
                initial_state = "BLOCKED"
                reason = "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW"
            elif task.skip_reason is not None:
                initial_state = "SKIPPED"
                reason = task.skip_reason
            elif not task.enabled or task.blocked_reason is not None:
                initial_state = "BLOCKED"
                reason = task.blocked_reason or "TASK_DISABLED_IN_MANIFEST"
            state = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task.task_id,
                "dataset": task.dataset,
                "stage": task.stage,
                "state": initial_state,
                "reason": reason,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "instances": {
                    instance_id: _initial_instance(instance_id)
                    for instance_id in task.instance_ids
                },
            }
            atomic_write_json(paths["state"], state)
            if initial_state in TERMINAL_STATES:
                _write_task_gate(root, state, status=initial_state, reason=reason)
        states[task.task_id] = state
    return root, states


def _write_task_state(root: Path, state: Mapping[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at"] = utc_now()
    atomic_write_json(_task_paths(root, str(payload["task_id"]))["state"], payload)


def _write_task_gate(
    root: Path,
    state: Mapping[str, Any],
    *,
    status: str,
    reason: str | None,
) -> None:
    instances = state.get("instances") or {}
    atomic_write_json(
        _task_paths(root, str(state["task_id"]))["gate"],
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": state["task_id"],
            "status": status,
            "checked_at": utc_now(),
            "reason": reason,
            "runs": [
                {
                    "instance_id": instance_id,
                    "run_id": instance.get("run_id"),
                    "state": instance.get("state"),
                    "attempt": instance.get("attempt"),
                    "gpu_index": instance.get("gpu_index"),
                    "gpu_uuid": instance.get("gpu_uuid"),
                    "expected_output": instance.get("expected_output"),
                    "retry_kind": instance.get("retry_kind"),
                    "resume_from_checkpoint": instance.get(
                        "resume_from_checkpoint"
                    ),
                }
                for instance_id, instance in sorted(instances.items())
            ],
        },
    )


def _append_event(
    root: Path,
    manifest: ControllerManifest,
    *,
    task: TaskSpec | None,
    state: str,
    reason: str | None = None,
    instance: Mapping[str, Any] | None = None,
) -> None:
    append_jsonl_locked(
        root / "registry" / "events.jsonl",
        {
            "schema_version": SCHEMA_VERSION,
            "controller_id": manifest.controller_id,
            "timestamp": utc_now(),
            "task_id": task.task_id if task else None,
            "dataset": task.dataset if task else None,
            "stage": task.stage if task else None,
            "state": state,
            "reason": reason,
            "run_id": instance.get("run_id") if instance else None,
            "pid": instance.get("worker_pid") if instance else os.getpid(),
            "tmux_session": instance.get("tmux_session") if instance else None,
            "command": list(task.command) if task and task.command else None,
            "gpu_index": instance.get("gpu_index") if instance else None,
            "gpu_uuid": instance.get("gpu_uuid") if instance else None,
            "retry_kind": instance.get("retry_kind") if instance else None,
            "resume_from_checkpoint": (
                instance.get("resume_from_checkpoint") if instance else None
            ),
            "manifest_sha256": manifest.sha256,
        },
    )


def _append_markdown_locked(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            new_file = not path.exists() or path.stat().st_size == 0
            with path.open("a", encoding="utf-8") as handle:
                if new_file:
                    handle.write("# AutoDL four-GPU experiment log\n\n")
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            fsync_directory(path.parent)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _registry_export_row(
    layout: Any,
    root: Path,
    manifest: ControllerManifest,
    task: TaskSpec,
    instance_id: str,
    instance: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = instance.get("run_id")
    run_state = (
        _read_run_state(layout, str(run_id))
        if isinstance(run_id, str)
        else None
    )
    spec_path = (
        layout.runs_root / str(run_id) / "launch_spec.json"
        if isinstance(run_id, str)
        else None
    )
    spec = read_json_object(spec_path) if spec_path and spec_path.is_file() else {}
    environment = spec.get("environment")
    if not isinstance(environment, dict):
        environment = dict(task.environment)
    checkpoints = {
        key: value
        for key, value in sorted(environment.items())
        if "CHECKPOINT" in str(key).upper()
        or "INITIALIZER" in str(key).upper()
    }
    input_manifest = spec.get("input_manifest") or task.input_manifest
    input_root = environment.get("SOURCE_GENERATION_DIR")
    if input_root is None and isinstance(input_manifest, str):
        input_root = str(Path(input_manifest).parent)
    output_root = (
        instance.get("expected_output")
        or spec.get("expected_output")
        or task.expected_output
    )
    checkpoint_hash = spec.get("checkpoint_hash")
    if checkpoint_hash is None:
        checkpoint_hash = next(
            (
                value
                for key, value in environment.items()
                if "CHECKPOINT" in str(key).upper()
                and str(key).upper().endswith(("_HASH", "_SHA256", "_ID"))
            ),
            None,
        )
    config_hash = spec.get("config_hash") or sha256_file(
        _task_paths(root, task.task_id)["manifest"]
    )
    state = str(instance.get("state") or "NOT_STARTED")
    terminal = state in TERMINAL_STATES
    start_time = instance.get("started_at") or spec.get("created_at")
    end_time = (
        (run_state or {}).get("completed_at") or instance.get("completed_at")
        if terminal
        else None
    )
    exit_code = (run_state or {}).get("exit_code")
    tmux_session = instance.get("tmux_session")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id
        or f"{manifest.controller_id}:{task.task_id}:{instance_id}",
        "controller_id": manifest.controller_id,
        "timestamp": utc_now(),
        "task_id": task.task_id,
        "instance_id": instance_id,
        "dataset": task.dataset,
        "stage": task.stage,
        "state": state,
        "pid": instance.get("worker_pid") or instance.get("launcher_pid"),
        "child_pid": instance.get("child_pid"),
        "tmux_session": tmux_session,
        "tmux": tmux_session,
        "command": spec.get("command") or list(task.command or ()),
        "gpu_index": instance.get("gpu_index"),
        "gpu_uuid": instance.get("gpu_uuid"),
        "gpu": {
            "index": instance.get("gpu_index"),
            "uuid": instance.get("gpu_uuid"),
        },
        "git_commit": spec.get("git_commit") or task.adopt_git_commit,
        "git": spec.get("git_commit") or task.adopt_git_commit,
        "input": {
            "manifest": input_manifest,
            "sha256": spec.get("input_hash"),
        },
        "input_root": input_root,
        "output": output_root,
        "output_root": output_root,
        "checkpoint": checkpoints,
        "checkpoint_hash": checkpoint_hash,
        "config": {
            "files": spec.get("config_files") or list(task.config_files),
            "sha256": config_hash,
            "controller_manifest_sha256": manifest.sha256,
        },
        "config_hash": config_hash,
        "start_time": start_time,
        "start": start_time,
        "end_time": end_time,
        "end": end_time,
        "exit_code": exit_code,
        "exit": exit_code,
        "retry": {
            "attempt": int(instance.get("attempt", 0)),
            "oom_retry_limit": 1 if task.oom_retry.enabled else 0,
            "transient_retry_limit": int(
                manifest.runtime.get(
                    "max_transient_retries", DEFAULT_MAX_TRANSIENT_RETRIES
                )
            ),
            "kind": instance.get("retry_kind"),
            "oom_retry_count": int(instance.get("oom_retry_count", 0)),
            "transient_retry_count": int(
                instance.get("transient_retry_count", 0)
            ),
            "failure_class": instance.get("failure_class"),
            "resume_from_checkpoint": instance.get("resume_from_checkpoint"),
        },
        "retry_count": int(instance.get("attempt", 0)),
        "dependencies": list(task.depends_on),
        "dependency_ids": list(task.depends_on),
        "adopted": bool(instance.get("adopted", False)),
    }


def publish_user_registry(
    layout: Any,
    root: Path,
    manifest: ControllerManifest,
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    """Mirror controller transitions to the user-specified runtime registry."""

    registry_root = layout.artifacts_dir / "autodl" / "experiment_registry"
    runs_path = registry_root / "runs.jsonl"
    updates_path = registry_root / "status_updates.jsonl"
    docs_path = layout.runtime_root / "docs" / "AUTODL_FOUR_GPU_EXPERIMENT_LOG.md"
    export_state_path = root / "registry_export_state.json"
    prior = (
        read_json_object(export_state_path)
        if export_state_path.is_file()
        else {"rows": {}}
    )
    prior_rows = prior.get("rows")
    if not isinstance(prior_rows, dict):
        raise ControllerError("Corrupt registry export state")
    next_rows: dict[str, str] = {}
    changed: list[dict[str, Any]] = []
    for task in manifest.tasks:
        task_state = states[task.task_id]
        for instance_id, instance in sorted(
            (task_state.get("instances") or {}).items()
        ):
            row = _registry_export_row(
                layout, root, manifest, task, instance_id, instance
            )
            stable_row = {key: value for key, value in row.items() if key != "timestamp"}
            digest = hashlib.sha256(
                json.dumps(stable_row, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            key = f"{task.task_id}:{instance_id}"
            next_rows[key] = digest
            if prior_rows.get(key) != digest:
                append_jsonl_locked(runs_path, row)
                changed.append(row)
    counts: dict[str, int] = {}
    for task_state in states.values():
        value = str(task_state.get("state"))
        counts[value] = counts.get(value, 0) + 1
    for task in manifest.tasks:
        task_state = states[task.task_id]
        for instance_id, instance in sorted(
            (task_state.get("instances") or {}).items()
        ):
            row = _registry_export_row(
                layout, root, manifest, task, instance_id, instance
            )
            row.update(
                {
                    "record_type": "status_update",
                    "controller_pid": os.getpid(),
                    "task_counts": counts,
                    "heartbeat": str(root / "heartbeat.json"),
                }
            )
            append_jsonl_locked(updates_path, row)
    if changed:
        lines = [f"## {utc_now()} — {manifest.controller_id}\n\n"]
        for row in changed:
            lines.append(
                f"- `{row['run_id']}` · {row['dataset']} · {row['stage']} · "
                f"**{row['state']}** · GPU `{row['gpu_index']}` / "
                f"`{row['gpu_uuid']}` · attempt {row['retry']['attempt']}\n"
            )
        lines.append("\n")
        _append_markdown_locked(docs_path, "".join(lines))
    atomic_write_json(
        export_state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "controller_id": manifest.controller_id,
            "updated_at": utc_now(),
            "rows": next_rows,
        },
    )


def _set_task_state(
    root: Path,
    manifest: ControllerManifest,
    task: TaskSpec,
    state: dict[str, Any],
    new_state: str,
    *,
    reason: str | None = None,
) -> None:
    if new_state not in CONTROLLER_STATES:
        raise ControllerError(f"Invalid controller task state: {new_state}")
    changed = state.get("state") != new_state or state.get("reason") != reason
    state["state"] = new_state
    state["reason"] = reason
    _write_task_state(root, state)
    if new_state in TERMINAL_STATES:
        _write_task_gate(root, state, status=new_state, reason=reason)
    if changed:
        _append_event(root, manifest, task=task, state=new_state, reason=reason)


def collect_host_resources(path: Path) -> HostResources:
    cpu_count = os.cpu_count() or 1
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        raise ControllerError("Cannot read CPU load average")
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        raise ControllerError("/proc/meminfo is required for fail-closed RAM gating")
    available_kib: int | None = None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    if available_kib is None:
        raise ControllerError("MemAvailable is absent from /proc/meminfo")
    free_disk = shutil.disk_usage(path).free / (1024**3)
    return HostResources(
        cpu_count=cpu_count,
        load_1m=load_1m,
        available_ram_gb=available_kib / (1024**2),
        free_disk_gb=free_disk,
    )


def resource_gate_failures(
    snapshot: HostResources, policy: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    min_ram = float(policy.get("min_available_ram_gb", 8))
    min_disk = float(policy.get("min_free_disk_gb", 20))
    max_load_fraction = float(policy.get("max_cpu_load_fraction", 0.90))
    if snapshot.available_ram_gb < min_ram:
        failures.append(
            f"available_ram_gb={snapshot.available_ram_gb:.2f} < {min_ram:.2f}"
        )
    if snapshot.free_disk_gb < min_disk:
        failures.append(
            f"free_disk_gb={snapshot.free_disk_gb:.2f} < {min_disk:.2f}"
        )
    load_fraction = snapshot.load_1m / max(1, snapshot.cpu_count)
    if not math.isfinite(load_fraction) or load_fraction > max_load_fraction:
        failures.append(
            f"cpu_load_fraction={load_fraction:.3f} > {max_load_fraction:.3f}"
        )
    return failures


def _parent_ids(payload: Mapping[str, Any], key: str) -> list[str]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise ControllerError(f"Frozen parent manifest has no list field {key!r}")
    values: list[str] = []
    for value in raw:
        if isinstance(value, dict):
            value = value.get("parent_id")
        if not isinstance(value, (str, int)) or not str(value):
            raise ControllerError("Frozen parent IDs must be nonempty scalars")
        values.append(str(value))
    if len(values) != len(set(values)):
        raise ControllerError("Frozen parent manifest contains duplicate parent IDs")
    if not values:
        raise ControllerError("Frozen parent manifest is empty")
    return sorted(values)


def materialize_fixed_parent_shards(
    *,
    source_manifest: Path,
    destination_root: Path,
    shard_count: int,
    expected_dataset: str,
    expected_split: str,
    parent_id_key: str = "parent_ids",
    allow_test: bool = False,
) -> list[Path]:
    """Freeze deterministic, mutually exclusive, exhaustive parent shards."""

    source = source_manifest.expanduser().resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ControllerError(f"Parent manifest must be a physical file: {source}")
    if shard_count < 2 or shard_count > FOUR_GPU_RECOVERY_LIMIT:
        raise ControllerError("Fixed parent shard count must be in [2, 4]")
    payload = read_json_object(source)
    if payload.get("status") not in {"FROZEN", "PASS"}:
        raise ControllerError("Parent manifest is not frozen")
    if str(payload.get("dataset", "")).lower() != expected_dataset.lower():
        raise ControllerError("Parent manifest dataset mismatch")
    if str(payload.get("split", "")).lower() != expected_split.lower():
        raise ControllerError("Parent manifest split mismatch")
    if expected_split.lower() == "test" and not allow_test:
        raise ControllerError("Parent sharding may not load held-out test data")
    parent_ids = _parent_ids(payload, parent_id_key)
    buckets = [parent_ids[index::shard_count] for index in range(shard_count)]
    source_sha = sha256_file(source)
    assignment_sha = hashlib.sha256(
        json.dumps(buckets, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, ids in enumerate(buckets):
        path = destination_root / f"shard-{index:03d}.json"
        document = {
            "schema_version": SCHEMA_VERSION,
            "status": "FROZEN",
            "dataset": expected_dataset,
            "split": expected_split,
            "shard_id": index,
            "shard_count": shard_count,
            "parent_ids": ids,
            "parent_count": len(ids),
            "source_manifest": str(source),
            "source_manifest_sha256": source_sha,
            "assignment_sha256": assignment_sha,
        }
        if path.exists():
            if read_json_object(path) != document:
                raise ControllerError(f"Frozen parent shard changed: {path}")
        else:
            atomic_write_json(path, document)
        paths.append(path)
    flattened = [value for bucket in buckets for value in bucket]
    if len(flattened) != len(parent_ids) or set(flattened) != set(parent_ids):
        raise ControllerError("Parent shard materialization is not exhaustive/disjoint")
    return paths


def classify_failure(
    log_text: str,
    *,
    semantic_markers: Sequence[str] = DEFAULT_SEMANTIC_MARKERS,
) -> str:
    normalized = log_text.lower()
    if any(marker.lower() in normalized for marker in semantic_markers):
        return "SEMANTIC"
    if any(marker in normalized for marker in OOM_MARKERS):
        return "OOM"
    if any(marker in normalized for marker in TRANSIENT_IO_MARKERS):
        return "TRANSIENT_IO"
    return "EXECUTION"


def oom_retry_allowed(
    failure_class: str, oom_retry_count: int, policy: OOMRetry
) -> bool:
    """Authorize only the first OOM retry; semantic/execution failures never retry."""

    return failure_class == "OOM" and policy.enabled and oom_retry_count == 0


def transient_retry_allowed(
    failure_class: str,
    transient_retry_count: int,
    *,
    max_transient_retries: int,
) -> bool:
    """Authorize one fresh-root retry for recognized transient failures."""

    return (
        failure_class in TRANSIENT_FAILURE_CLASSES
        and max_transient_retries == DEFAULT_MAX_TRANSIENT_RETRIES
        and transient_retry_count < max_transient_retries
    )


def _reset_instance_for_retry(
    instance: dict[str, Any],
    task: TaskSpec,
    *,
    retry_kind: str,
    retry_reason: str,
) -> None:
    attempt = int(instance.get("attempt", 0))
    resume_from_checkpoint: str | None = None
    resume_source_output: str | None = None
    # Preserve B7 progress only for a same-batch transient retry.  An OOM
    # retry deliberately starts from the clean initializer because its lower
    # batch changes the frozen dataloader trajectory.
    if retry_kind in TRANSIENT_FAILURE_CLASSES and task.stage == "B7_PPO_FULL":
        raw_source_output = instance.get("expected_output")
        if isinstance(raw_source_output, str):
            source_output = Path(raw_source_output)
            latest = find_latest_stable_ppo_resume_checkpoint(source_output)
            if latest is not None:
                resume_from_checkpoint = str(latest)
                resume_source_output = str(source_output.resolve(strict=True))
                retry_reason += f" from checkpoint {latest.name}"
    instance.update(
        {
            "state": "NOT_STARTED",
            "attempt": attempt + 1,
            "run_id": None,
            "launcher_pid": None,
            "launcher_identity": None,
            "worker_pid": None,
            "child_pid": None,
            "worker_identity": None,
            "tmux_session": None,
            "gpu_index": None,
            "gpu_uuid": None,
            "log_path": None,
            "started_at": None,
            "heartbeat_at": None,
            "failure_class": f"{retry_kind}_RETRY",
            "failure_reason": retry_reason,
            "oom_retry_count": int(instance.get("oom_retry_count", 0))
            + int(retry_kind == "OOM"),
            "transient_retry_count": int(
                instance.get("transient_retry_count", 0)
            )
            + int(retry_kind in TRANSIENT_FAILURE_CLASSES),
            "retry_kind": retry_kind,
            "resume_from_checkpoint": resume_from_checkpoint,
            "resume_source_output": resume_source_output,
        }
    )


def _fail_or_retry_process_loss(
    instance: dict[str, Any],
    task: TaskSpec,
    *,
    reason: str,
    max_transient_retries: int,
) -> None:
    transient_retry_count = int(instance.get("transient_retry_count", 0))
    failure_class = "TRANSIENT_PROCESS_LOSS"
    if not bool(instance.get("adopted")) and transient_retry_allowed(
        failure_class,
        transient_retry_count,
        max_transient_retries=max_transient_retries,
    ):
        _reset_instance_for_retry(
            instance,
            task,
            retry_kind=failure_class,
            retry_reason="one bounded dead-worker retry authorized: " + reason,
        )
        return
    instance["state"] = "FAILED"
    instance["failure_class"] = "STALE_PROCESS"
    instance["failure_reason"] = reason


def read_process_identity(pid: int) -> dict[str, Any] | None:
    """Read the Linux PID generation and command digest without signalling it."""

    if pid <= 0:
        return None
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        stat = stat_path.read_text(encoding="utf-8")
        close = stat.rfind(")")
        fields = stat[close + 2 :].split()
        # /proc/PID/stat field 22; fields starts at field 3.
        start_ticks = int(fields[19])
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").strip()
    except (OSError, ValueError, IndexError):
        return None
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "command_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def process_identity_matches(expected: Mapping[str, Any], pid: int) -> bool:
    current = read_process_identity(pid)
    return bool(
        current
        and int(expected.get("pid", -1)) == pid
        and int(expected.get("start_ticks", -1)) == current["start_ticks"]
        and str(expected.get("command_sha256", "")) == current["command_sha256"]
    )


def tmux_session_alive(session: str) -> bool:
    """Return whether one exact detached worker tmux session still exists."""

    executable = shutil.which("tmux")
    if executable is None or not session:
        return False
    completed = subprocess.run(
        [executable, "has-session", "-t", session],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    return completed.returncode == 0


def _timestamp_age_seconds(value: Any, *, now_epoch: float) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, now_epoch - parsed.timestamp())


def _launcher_evidence_alive(instance: dict[str, Any]) -> bool:
    launcher_pid = instance.get("launcher_pid")
    if isinstance(launcher_pid, int):
        expected = instance.get("launcher_identity")
        if isinstance(expected, Mapping):
            if process_identity_matches(expected, launcher_pid):
                return True
        else:
            identity = read_process_identity(launcher_pid)
            if identity is not None:
                instance["launcher_identity"] = identity
                return True
    session = instance.get("tmux_session")
    return isinstance(session, str) and tmux_session_alive(session)


def _gpu_lock_path(lock_root: Path, gpu_uuid: str) -> Path:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", gpu_uuid).strip("._")
    return lock_root / f"gpu-{component}.lock"


def audit_gpu_locks(
    lock_root: Path,
    observations: Sequence[GPUObservation],
    *,
    probe_advisory_lock: bool = True,
) -> list[dict[str, Any]]:
    """Audit UUID lock metadata against PID identity and GPU process rows.

    Stale metadata is reported, never deleted.  Only the owning ``exp_run``
    worker may release a lock.
    """

    rows: list[dict[str, Any]] = []
    for gpu in sorted(observations, key=lambda value: value.index)[:FOUR_GPU_RECOVERY_LIMIT]:
        path = _gpu_lock_path(lock_root, gpu.uuid)
        metadata: dict[str, Any] = {}
        if path.is_file() and not path.is_symlink():
            try:
                metadata = read_json_object(path)
            except AutoDLRuntimeError:
                metadata = {"state": "CORRUPT"}
        owner_pid = metadata.get("pid")
        owner_pid_valid = (
            isinstance(owner_pid, int)
            and not isinstance(owner_pid, bool)
            and owner_pid > 0
        )
        owner_alive = (
            owner_pid_valid and read_process_identity(owner_pid) is not None
        )
        available: bool | None = None
        if probe_advisory_lock:
            available = gpu_lock_available(lock_root, gpu.uuid)
        process_pids = sorted(process.pid for process in gpu.processes)
        if metadata.get("state") == "CORRUPT":
            audit = "CORRUPT_METADATA"
        elif metadata.get("state") == "LOCKED" and not owner_pid_valid:
            # Reclaiming LOCKED metadata requires an explicit PID whose death
            # can be proven. Missing, boolean, string, or non-positive owners
            # are malformed evidence and must never become STALE_METADATA.
            audit = "MALFORMED_LOCK_METADATA"
        elif process_pids:
            audit = (
                "GPU_PROCESS_PRESENT"
                if metadata.get("state") == "LOCKED"
                else "EXTERNAL_BUSY"
            )
        elif available is False:
            # Advisory ownership wins over stale/non-LOCKED JSON metadata.  A
            # worker can hold the UUID during setup before a compute process
            # appears, and that race must never be treated as AVAILABLE.
            audit = "HELD" if owner_alive else "INDETERMINATE"
        elif metadata.get("state") == "LOCKED":
            audit = (
                "STALE_METADATA"
                if available is True and not owner_alive
                else "INDETERMINATE"
            )
        elif available is True:
            audit = "AVAILABLE"
        else:
            audit = "UNVERIFIED_AVAILABLE"
        rows.append(
            {
                "gpu_index": gpu.index,
                "gpu_uuid": gpu.uuid,
                "utilization_gpu_percent": gpu.utilization_gpu_percent,
                "memory_total_mb": gpu.memory_total_mb,
                "memory_used_mb": gpu.memory_used_mb,
                "memory_free_mb": gpu.memory_free_mb,
                "compute_pids": process_pids,
                "lock_path": str(path),
                "lock_state": metadata.get("state", "ABSENT"),
                "lock_owner_pid": owner_pid,
                "lock_owner_pid_valid": owner_pid_valid,
                "lock_owner_alive": owner_alive,
                "advisory_lock_available": available,
                "audit": audit,
            }
        )
    return rows


def allocation_safe_gpu_uuids(
    audit_rows: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Return UUIDs whose lock metadata is safe to allocate right now.

    Advisory-lock availability alone is insufficient: a previous owner PID may
    still be alive after accidentally closing its file descriptor, and PID
    reuse must fail closed.  ``STALE_METADATA`` is the only reclaimable locked
    metadata state because the audit proves both that the owner PID is absent
    and that the UUID has no compute process.
    """

    return frozenset(
        str(row["gpu_uuid"])
        for row in audit_rows
        if row.get("audit") in ALLOCATION_SAFE_GPU_AUDITS
        and isinstance(row.get("gpu_uuid"), str)
    )


def _expand(value: str, context: Mapping[str, str], *, label: str) -> str:
    unknown = sorted(set(TOKEN.findall(value)) - set(context))
    if unknown:
        raise ControllerError(f"{label} has unknown template tokens: {unknown}")
    for key, replacement in context.items():
        value = value.replace("{" + key + "}", replacement)
    return value


def _absolute_expanded_path(
    value: str,
    context: Mapping[str, str],
    *,
    label: str,
    must_exist: bool,
) -> Path:
    expanded = Path(_expand(value, context, label=label)).expanduser()
    if not expanded.is_absolute():
        raise ControllerError(f"{label} must expand to an absolute path: {expanded}")
    return expanded.resolve(strict=must_exist)


def _runtime_context(
    layout: Any,
    task: TaskSpec,
    instance_id: str,
    attempt: int,
    *,
    python_executable: Path,
    shard_manifest: Path | None,
    retry_kind: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    batch_size = task.oom_retry.initial_batch_size
    if retry_kind == "OOM":
        batch_size = task.oom_retry.retry_batch_size
    shard_match = re.fullmatch(r"shard-(\d+)", instance_id)
    shard_index = str(int(shard_match.group(1))) if shard_match else ""
    context = {
        "project_root": str(layout.project_root),
        "data_root": str(layout.data_root),
        "runtime_root": str(layout.runtime_root),
        "artifact_root": str(layout.artifacts_dir),
        "control_root": str(layout.control_root),
        "python": str(python_executable),
        "task_id": task.task_id,
        "stage": task.stage,
        "instance_id": instance_id,
        "shard_id": instance_id,
        "shard_index": shard_index,
        "shard_manifest": str(shard_manifest) if shard_manifest else "",
        "attempt": str(attempt),
        "batch_size": str(batch_size) if batch_size is not None else "",
    }
    context.update(dict(extra or {}))
    return context


def _assert_persistent_input(path: Path, layout: Any, *, label: str) -> None:
    try:
        path.relative_to(layout.data_root)
    except ValueError as exc:
        raise ControllerError(
            f"{label} must live under persistent data root {layout.data_root}: {path}"
        ) from exc
    try:
        path.relative_to(layout.project_root)
    except ValueError:
        return
    raise ControllerError(f"{label} must not live in the fast code worktree: {path}")


def _resolve_python(explicit: Path | None) -> Path:
    raw = explicit or Path(os.environ.get("AUTODL_PYTHON", DEFAULT_AUTODL_PYTHON))
    if not raw.expanduser().is_absolute():
        raise ControllerError("AUTODL_PYTHON must be absolute")
    resolved = raw.expanduser().resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ControllerError(f"AUTODL_PYTHON is not executable: {resolved}")
    return resolved


def _prepare_shards(
    layout: Any,
    root: Path,
    task: TaskSpec,
    python_executable: Path,
    states: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    if task.shards is None:
        return {}
    context = _runtime_context(
        layout,
        task,
        "shard-000",
        0,
        python_executable=python_executable,
        shard_manifest=None,
        extra=(
            _dependency_output_context(root, task, states)
            if states is not None
            else None
        ),
    )
    source = _absolute_expanded_path(
        task.shards.parent_manifest,
        context,
        label=f"{task.task_id}.shards.parent_manifest",
        must_exist=True,
    )
    _assert_persistent_input(source, layout, label="parent shard manifest")
    paths = materialize_fixed_parent_shards(
        source_manifest=source,
        destination_root=_task_paths(root, task.task_id)["shards"],
        shard_count=task.shards.count,
        expected_dataset=task.dataset,
        expected_split=task.shards.split,
        parent_id_key=task.shards.parent_id_key,
        allow_test=(
            task.stage in POST_FREEZE_TEST_STAGES
            and task.selector_parameters_frozen
            and task.read_only_test
        ),
    )
    return {path.stem: path for path in paths}


def _read_run_state(layout: Any, run_id: str) -> dict[str, Any] | None:
    path = layout.runs_root / run_id / "state.json"
    if not path.is_file():
        return None
    return read_json_object(path)


def bind_adopted_runs(
    layout: Any,
    root: Path,
    manifest: ControllerManifest,
    states: dict[str, dict[str, Any]],
    *,
    python_executable: Path,
) -> None:
    """Bind already-launched canonical exp_run workers without a second writer."""

    for task in manifest.tasks:
        run_id = task.adopt_existing_run_id
        if run_id is None:
            continue
        if (
            not task.enabled
            or task.blocked_reason is not None
            or task.skip_reason is not None
            or task.dataset in {"taste", "tastemolnet"}
        ):
            raise ControllerError(
                f"{task.task_id} cannot adopt a run while disabled/BLOCKED/SKIPPED"
            )
        instance = states[task.task_id]["instances"]["main"]
        existing_run_id = instance.get("run_id")
        if existing_run_id not in {None, run_id}:
            raise ControllerError(
                f"{task.task_id} is already bound to a different run: {existing_run_id}"
            )
        spec_path = layout.runs_root / run_id / "launch_spec.json"
        run_state = _read_run_state(layout, run_id)
        if not spec_path.is_file() or run_state is None:
            raise ControllerError(f"Adopted exp_run is absent/incomplete: {run_id}")
        spec = read_json_object(spec_path)
        declared_project_root = Path(str(task.adopt_project_root)).expanduser().resolve(
            strict=True
        )
        context = _runtime_context(
            layout,
            task,
            "main",
            int(instance.get("attempt", 0)),
            python_executable=python_executable,
            shard_manifest=None,
            extra=_dependency_output_context(root, task, states),
        )
        # Adoption validates the immutable worktree which launched the existing
        # writer.  That worktree is intentionally allowed to differ from the
        # controller's newer integration worktree.
        context["project_root"] = str(declared_project_root)
        if task.command is None or task.input_manifest is None or task.expected_output is None:
            raise ControllerError(f"Adopted task lacks a complete contract: {task.task_id}")
        expected_output = _absolute_expanded_path(
            task.expected_output,
            context,
            label=f"{task.task_id}.expected_output",
            must_exist=False,
        )
        context["task_output"] = str(expected_output)
        expected_command = [
            _expand(value, context, label=f"{task.task_id}.command")
            for value in task.command
        ]
        expected_input = _absolute_expanded_path(
            task.input_manifest,
            context,
            label=f"{task.task_id}.input_manifest",
            must_exist=True,
        )
        expected_config_files = [
            _absolute_expanded_path(
                value,
                context,
                label=f"{task.task_id}.config_file",
                must_exist=True,
            )
            for value in task.config_files
        ]
        expected_stage = _expand(task.runner_stage, context, label="runner_stage")
        failures: list[str] = []
        exact = {
            "run_id": run_id,
            "dataset": task.runner_dataset,
            "stage": expected_stage,
            "command": expected_command,
            "input_manifest": str(expected_input),
            "expected_output": str(expected_output),
            "required_output_files": list(task.required_output_files),
            "required_output_any": [
                list(group) for group in task.required_output_any
            ],
            "required_absolute_output_files": [
                str(
                    _absolute_expanded_path(
                        value,
                        context,
                        label=f"{task.task_id}.required_absolute_output_file",
                        must_exist=False,
                    )
                )
                for value in task.required_absolute_output_files
            ],
            "required_log_marker": task.required_log_marker,
            "python_executable": str(python_executable),
            "project_root": str(declared_project_root),
            "data_root": str(layout.data_root),
            "control_root": str(layout.control_root),
            "git_commit": task.adopt_git_commit,
            "max_gpus": task.adopt_max_gpus,
            "heavy": task.adopt_heavy,
            "config_files": [str(path) for path in expected_config_files],
            "config_hash": sha256_paths(expected_config_files),
        }
        list_keys = {
            "required_output_any",
            "required_absolute_output_files",
        }
        for key, value in exact.items():
            if spec.get(key, [] if key in list_keys else None) != value:
                failures.append(f"{key} mismatch")
        expected_environment = {
            key: _expand(value, context, label=f"{task.task_id}.environment")
            for key, value in task.environment.items()
        }
        actual_environment = spec.get("environment")
        if actual_environment != expected_environment:
            failures.append("environment mismatch")
        run_state_exact = {
            "run_id": run_id,
            "dataset": task.runner_dataset,
            "stage": expected_stage,
            "gpu_index": task.adopt_gpu_index,
            "gpu_uuid": task.adopt_gpu_uuid,
        }
        for key, value in run_state_exact.items():
            if run_state.get(key) != value:
                failures.append(f"run state {key} mismatch")
        if task.resource == "cpu":
            if spec.get("gpu_index") is not None or spec.get("gpu_uuid") is not None:
                failures.append("CPU task was launched with a GPU assignment")
        elif (
            spec.get("gpu_index") != task.adopt_gpu_index
            or spec.get("gpu_uuid") != task.adopt_gpu_uuid
        ):
            failures.append("adopted GPU index/UUID mismatch")
        if spec.get("input_hash") != sha256_file(expected_input):
            failures.append("input hash mismatch")
        if failures:
            raise ControllerError(
                f"Adopted run {run_id} does not match frozen task {task.task_id}: "
                + ", ".join(failures)
            )
        instance.update(
            {
                "state": "STARTING",
                "run_id": run_id,
                "adopted": True,
                "launcher_pid": None,
                "worker_pid": run_state.get("pid"),
                "child_pid": run_state.get("child_pid"),
                "tmux_session": run_state.get("tmux_session"),
                "gpu_index": run_state.get("gpu_index"),
                "gpu_uuid": run_state.get("gpu_uuid"),
                "log_path": run_state.get("log_path"),
                "expected_output": str(expected_output),
                "started_at": spec.get("created_at"),
                "required_absolute_output_files": exact[
                    "required_absolute_output_files"
                ],
                "command_sha256": command_digest(expected_command),
                "heartbeat_at": utc_now(),
            }
        )
        states[task.task_id]["state"] = "STARTING"
        states[task.task_id]["reason"] = "ADOPTED_EXISTING_EXP_RUN"
        _write_task_state(root, states[task.task_id])
        _append_event(
            root,
            manifest,
            task=task,
            state="STARTING",
            reason="ADOPTED_EXISTING_EXP_RUN",
            instance=instance,
        )


def _run_log_text(layout: Any, instance: Mapping[str, Any]) -> str:
    run_id = instance.get("run_id")
    if not isinstance(run_id, str):
        return ""
    state = _read_run_state(layout, run_id)
    raw_path = state.get("log_path") if state else None
    if not isinstance(raw_path, str):
        raw_path = instance.get("log_path")
    if not isinstance(raw_path, str):
        return ""
    path = Path(raw_path)
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - 2_000_000))
        return handle.read().decode("utf-8", errors="replace")


def _verify_passed_instance_contract(
    layout: Any, task: TaskSpec, instance: Mapping[str, Any]
) -> list[str]:
    raw_output = instance.get("expected_output")
    if not isinstance(raw_output, str):
        return ["instance has no frozen expected_output"]
    output = Path(raw_output)
    failures = verify_required_outputs(output, task.required_output_files)
    failures.extend(
        verify_required_output_alternatives(output, task.required_output_any)
    )
    absolute_outputs = instance.get("required_absolute_output_files") or []
    failures.extend(
        verify_required_absolute_outputs(
            [Path(str(value)) for value in absolute_outputs],
            allowed_root=layout.artifacts_dir,
        )
    )
    marker = task.required_log_marker
    if marker and marker not in _run_log_text(layout, instance):
        failures.append(f"required log marker is absent: {marker}")
    return failures


def _reconcile_instance(
    layout: Any,
    task: TaskSpec,
    instance: dict[str, Any],
    *,
    launch_grace_seconds: int = DEFAULT_LAUNCH_GRACE_SECONDS,
    max_transient_retries: int = DEFAULT_MAX_TRANSIENT_RETRIES,
    now_epoch: float | None = None,
) -> None:
    if instance.get("state") not in ACTIVE_STATES:
        return
    run_id = instance.get("run_id")
    if not isinstance(run_id, str):
        instance["state"] = "FAILED"
        instance["failure_class"] = "CONTROLLER"
        instance["failure_reason"] = "active instance has no run_id"
        return
    current_epoch = time.time() if now_epoch is None else float(now_epoch)
    age = _timestamp_age_seconds(instance.get("started_at"), now_epoch=current_epoch)
    within_launch_grace = (
        age is not None and age <= float(launch_grace_seconds)
    )
    run_state = _read_run_state(layout, run_id)
    if run_state is None:
        if within_launch_grace and _launcher_evidence_alive(instance):
            instance["state"] = "STARTING"
            instance["heartbeat_at"] = utc_now()
            return
        if within_launch_grace:
            # Preserve the last worker-originated activity timestamp.  The
            # controller must not manufacture a fresh heartbeat when it has no
            # live process evidence.
            instance["failure_reason"] = "waiting within detached launch grace"
            return
        _fail_or_retry_process_loss(
            instance,
            task,
            reason="run state absent after bounded launch grace",
            max_transient_retries=max_transient_retries,
        )
        return
    observed = str(run_state.get("state", ""))
    instance["worker_pid"] = run_state.get("pid")
    instance["child_pid"] = run_state.get("child_pid")
    if observed == "STARTING":
        pid = run_state.get("pid")
        worker_alive = False
        if isinstance(pid, int) and not isinstance(pid, bool):
            identity = read_process_identity(pid)
            if identity is not None:
                expected_identity = instance.get("worker_identity")
                if expected_identity is not None and not process_identity_matches(
                    expected_identity, pid
                ):
                    _fail_or_retry_process_loss(
                        instance,
                        task,
                        reason="worker PID generation/command changed",
                        max_transient_retries=max_transient_retries,
                    )
                    return
                instance["worker_identity"] = identity
                worker_alive = True
        if worker_alive:
            instance["state"] = "STARTING"
            instance["heartbeat_at"] = utc_now()
            return
        if within_launch_grace:
            instance["state"] = "STARTING"
            if _launcher_evidence_alive(instance):
                instance["heartbeat_at"] = utc_now()
            else:
                instance["heartbeat_at"] = run_state.get(
                    "updated_at"
                ) or instance.get("heartbeat_at")
            instance["failure_reason"] = "waiting within detached launch grace"
            return
        _fail_or_retry_process_loss(
            instance,
            task,
            reason="STARTING worker has no live PID after bounded launch grace",
            max_transient_retries=max_transient_retries,
        )
        return
    if observed == "RUNNING":
        pid = run_state.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            _fail_or_retry_process_loss(
                instance,
                task,
                reason="RUNNING exp_run state has no worker PID",
                max_transient_retries=max_transient_retries,
            )
            return
        identity = read_process_identity(pid)
        if identity is None:
            _fail_or_retry_process_loss(
                instance,
                task,
                reason="RUNNING exp_run worker PID is absent",
                max_transient_retries=max_transient_retries,
            )
            return
        expected_identity = instance.get("worker_identity")
        if expected_identity is not None and not process_identity_matches(
            expected_identity, pid
        ):
            _fail_or_retry_process_loss(
                instance,
                task,
                reason="worker PID generation/command changed",
                max_transient_retries=max_transient_retries,
            )
            return
        instance["worker_identity"] = identity
        instance["state"] = "RUNNING"
        # This heartbeat is backed by a live PID generation check.
        instance["heartbeat_at"] = utc_now()
        instance["failure_reason"] = None
        return
    if observed == "PASS":
        contract_failures = _verify_passed_instance_contract(layout, task, instance)
        if contract_failures:
            instance["state"] = "FAILED"
            instance["failure_class"] = "SEMANTIC"
            instance["failure_reason"] = "; ".join(contract_failures)
            return
        instance["state"] = "PASS"
        instance["heartbeat_at"] = utc_now()
        return
    if observed == "BLOCKED":
        instance["state"] = "BLOCKED"
        instance["failure_class"] = "SEMANTIC"
        instance["failure_reason"] = "; ".join(run_state.get("failures") or []) or "scientific blocker"
        return
    if observed != "FAILED":
        instance["state"] = "FAILED"
        instance["failure_class"] = "CONTROLLER"
        instance["failure_reason"] = f"unknown exp_run state: {observed}"
        return
    failure_class = classify_failure(
        _run_log_text(layout, instance),
        semantic_markers=task.semantic_failure_markers,
    )
    attempt = int(instance.get("attempt", 0))
    oom_retry_count = int(instance.get("oom_retry_count", 0))
    transient_retry_count = int(instance.get("transient_retry_count", 0))
    retry_kind: str | None = None
    retry_reason: str | None = None
    if not bool(instance.get("adopted")) and oom_retry_allowed(
        failure_class, oom_retry_count, task.oom_retry
    ):
        retry_kind = "OOM"
        retry_reason = "one bounded lower-batch OOM retry authorized"
    elif not bool(instance.get("adopted")) and transient_retry_allowed(
        failure_class,
        transient_retry_count,
        max_transient_retries=max_transient_retries,
    ):
        retry_kind = "TRANSIENT_IO"
        retry_reason = "one bounded transient I/O retry authorized"
    if retry_kind is not None:
        _reset_instance_for_retry(
            instance,
            task,
            retry_kind=retry_kind,
            retry_reason=str(retry_reason),
        )
        return
    instance["state"] = "FAILED"
    instance["failure_class"] = (
        "OOM_RETRY_EXHAUSTED" if failure_class == "OOM" else failure_class
    )
    instance["failure_reason"] = (
        "; ".join(run_state.get("failures") or []) or f"{failure_class} failure"
    )


def _external_bace_gate_ready(layout: Any, task: TaskSpec) -> tuple[bool, str | None]:
    if task.external_bace_stage is None:
        return True, None
    try:
        documents = read_bace_stage(layout, task.external_bace_stage)
    except AutoDLRuntimeError as exc:
        return False, str(exc)
    if (
        documents["state"].get("state") == "PASS"
        and documents["gate"].get("status") == "PASS"
    ):
        return True, None
    return False, f"waiting for external {task.external_bace_stage} state/gate PASS"


def scheduler_candidates(
    tasks: Sequence[TaskSpec],
    states: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """Return all launchable instances in work-conserving priority order."""

    order = dependency_order(tasks)
    rank = {task_id: index for index, task_id in enumerate(order)}
    by_id = {task.task_id: task for task in tasks}
    result: list[tuple[str, str]] = []
    for task in sorted(
        tasks, key=lambda value: (value.priority, rank[value.task_id], value.task_id)
    ):
        state = states[task.task_id]
        if state.get("state") not in {"READY", "WAITING_RESOURCE", "RUNNING"}:
            continue
        instance_values = list((state.get("instances") or {}).values())
        if any(
            instance.get("state") in {"FAILED", "BLOCKED"}
            for instance in instance_values
        ):
            # Drain already-running siblings after one shard fails, but never
            # spend another GPU on a shard whose aggregate can no longer pass.
            continue
        if any(states[dependency].get("state") != "PASS" for dependency in task.depends_on):
            continue
        for instance_id, instance in sorted((state.get("instances") or {}).items()):
            if instance.get("state") == "NOT_STARTED":
                result.append((task.task_id, instance_id))
    return result


def _aggregate_task_state(
    root: Path,
    layout: Any,
    manifest: ControllerManifest,
    task: TaskSpec,
    state: dict[str, Any],
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    persisted_instances = list((state.get("instances") or {}).values())
    if state.get("state") in TERMINAL_STATES and not any(
        instance.get("state") in ACTIVE_STATES for instance in persisted_instances
    ):
        return
    dependency_states = {
        dependency: str(states[dependency].get("state"))
        for dependency in task.depends_on
    }
    failed_dependencies = {
        dependency: value
        for dependency, value in dependency_states.items()
        if value in FAILURE_STATES
    }
    if failed_dependencies:
        _set_task_state(
            root,
            manifest,
            task,
            state,
            "BLOCKED",
            reason=f"dependency terminal failure: {failed_dependencies}",
        )
        return
    if any(value != "PASS" for value in dependency_states.values()):
        _set_task_state(
            root,
            manifest,
            task,
            state,
            "WAITING_DEPENDENCY",
            reason="waiting for declared task dependencies",
        )
        return
    external_ready, external_reason = _external_bace_gate_ready(layout, task)
    if not external_ready:
        _set_task_state(
            root,
            manifest,
            task,
            state,
            "WAITING_DEPENDENCY",
            reason=external_reason,
        )
        return
    instances = list((state.get("instances") or {}).values())
    instance_states = [str(instance.get("state")) for instance in instances]
    has_active = any(value in ACTIVE_STATES for value in instance_states)
    has_failed = any(value == "FAILED" for value in instance_states)
    has_blocked = any(value == "BLOCKED" for value in instance_states)
    if has_active and (has_failed or has_blocked):
        terminal_siblings = [
            f"{instance.get('instance_id')}:{instance.get('state')}"
            for instance in instances
            if instance.get("state") in {"FAILED", "BLOCKED"}
        ]
        _set_task_state(
            root,
            manifest,
            task,
            state,
            "RUNNING",
            reason=(
                "DRAINING_ACTIVE_SIBLINGS_AFTER_TERMINAL_INSTANCE: "
                + ", ".join(terminal_siblings)
            ),
        )
    elif has_failed:
        failures = [
            f"{instance.get('instance_id')}:{instance.get('failure_class')}"
            for instance in instances
            if instance.get("state") == "FAILED"
        ]
        _set_task_state(
            root, manifest, task, state, "FAILED", reason=", ".join(failures)
        )
    elif has_blocked:
        _set_task_state(
            root,
            manifest,
            task,
            state,
            "BLOCKED",
            reason="one or more scientific instances are BLOCKED",
        )
    elif instances and all(value == "PASS" for value in instance_states):
        if task.publish_bace_stage and task.stage in BACE_STAGES and task.shards:
            _publish_sharded_bace_pass(layout, root, manifest, task, state)
        _set_task_state(root, manifest, task, state, "PASS")
    elif has_active:
        _set_task_state(root, manifest, task, state, "RUNNING")
    elif any(value == "NOT_STARTED" for value in instance_states):
        _set_task_state(root, manifest, task, state, "READY")
    else:
        _set_task_state(
            root,
            manifest,
            task,
            state,
            "FAILED",
            reason=f"unsupported instance states: {instance_states}",
        )


def _publish_sharded_bace_pass(
    layout: Any,
    root: Path,
    manifest: ControllerManifest,
    task: TaskSpec,
    state: Mapping[str, Any],
) -> None:
    """Publish the official BACE gate only after every immutable shard passes."""

    documents = read_bace_stage(layout, task.stage)
    if (
        documents["state"].get("state") == "PASS"
        and documents["gate"].get("status") == "PASS"
    ):
        return
    assert_bace_stage_can_start(layout, task.stage)
    runs = [
        {
            "instance_id": instance_id,
            "run_id": instance.get("run_id"),
            "gpu_uuid": instance.get("gpu_uuid"),
            "attempt": instance.get("attempt"),
        }
        for instance_id, instance in sorted((state.get("instances") or {}).items())
    ]
    now = utc_now()
    paths = stage_paths(layout, task.stage)
    atomic_write_json(
        paths["manifest"],
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": "bace",
            "stage": task.stage,
            "status": "FROZEN",
            "controller_id": manifest.controller_id,
            "controller_manifest_sha256": manifest.sha256,
            "task_manifest": str(_task_paths(root, task.task_id)["manifest"]),
            "task_manifest_sha256": sha256_file(
                _task_paths(root, task.task_id)["manifest"]
            ),
            "shard_runs": runs,
            "published_at": now,
        },
    )
    atomic_write_json(
        paths["gate"],
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": "bace",
            "stage": task.stage,
            "status": "PASS",
            "checked_at": now,
            "evidence": runs,
            "reason": None,
        },
    )
    update_bace_stage_state(
        layout,
        task.stage,
        "PASS",
        completed_at=now,
        controller_id=manifest.controller_id,
        shard_runs=runs,
    )


def _validated_b7_resume_checkpoint(
    layout: Any,
    task: TaskSpec,
    instance: Mapping[str, Any],
) -> Path | None:
    raw = instance.get("resume_from_checkpoint")
    if raw is None:
        return None
    if (
        task.stage != "B7_PPO_FULL"
        or instance.get("retry_kind") not in TRANSIENT_FAILURE_CLASSES
        or int(instance.get("transient_retry_count", 0)) != 1
    ):
        raise ControllerError(
            "Only one recognized transient B7 retry may consume a checkpoint"
        )
    unresolved = Path(str(raw)).expanduser()
    if unresolved.is_symlink():
        raise ControllerError("B7 resume checkpoint may not be a symlink")
    checkpoint = unresolved.resolve(strict=True)
    try:
        checkpoint.relative_to(layout.artifacts_dir)
    except ValueError as exc:
        raise ControllerError(
            f"B7 resume checkpoint escapes persistent artifacts: {checkpoint}"
        ) from exc
    source_raw = instance.get("resume_source_output")
    if not isinstance(source_raw, str):
        raise ControllerError("B7 resume checkpoint has no frozen source output")
    source_output = Path(source_raw).expanduser().resolve(strict=True)
    if checkpoint.parent != source_output:
        raise ControllerError("B7 resume checkpoint escaped its failed attempt root")
    try:
        resume_manifest = read_stable_ppo_resume_manifest(checkpoint)
    except (OSError, TypeError, ValueError) as exc:
        raise ControllerError(f"B7 resume checkpoint failed validation: {exc}") from exc
    contract = resume_manifest.get("resume_contract")
    if not isinstance(contract, dict) or contract.get("stage") != "B7_PPO_FULL":
        raise ControllerError("B7 resume checkpoint has the wrong stage contract")
    completed = int(resume_manifest["completed_steps"])
    max_steps = contract.get("max_steps")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or completed >= max_steps
    ):
        raise ControllerError("B7 resume checkpoint is not an incomplete periodic state")
    return checkpoint


def _launch_instance(
    layout: Any,
    root: Path,
    manifest: ControllerManifest,
    task: TaskSpec,
    state: dict[str, Any],
    instance_id: str,
    *,
    python_executable: Path,
    gpu: GPUObservation | None,
    shard_manifest: Path | None,
    extra_context: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    if task.command is None or task.input_manifest is None or task.expected_output is None:
        raise ControllerError(f"Task {task.task_id} has no launch contract")
    instance = state["instances"][instance_id]
    attempt = int(instance.get("attempt", 0))
    context = _runtime_context(
        layout,
        task,
        instance_id,
        attempt,
        python_executable=python_executable,
        shard_manifest=shard_manifest,
        retry_kind=(
            str(instance.get("retry_kind"))
            if instance.get("retry_kind") is not None
            else None
        ),
        extra=extra_context,
    )
    expected_output = _absolute_expanded_path(
        task.expected_output,
        context,
        label=f"{task.task_id}.expected_output",
        must_exist=False,
    )
    context["task_output"] = str(expected_output)
    command = [
        _expand(value, context, label=f"{task.task_id}.command")
        for value in task.command
    ]
    if any(value in {"python", "python3"} for value in task.command):
        raise ControllerError(
            f"{task.task_id} uses a bare Python; use the {{python}} token"
        )
    input_manifest = _absolute_expanded_path(
        task.input_manifest,
        context,
        label=f"{task.task_id}.input_manifest",
        must_exist=True,
    )
    _assert_persistent_input(input_manifest, layout, label="input manifest")
    try:
        expected_output.relative_to(layout.artifacts_dir)
    except ValueError as exc:
        raise ControllerError(
            f"Expected output must be under {layout.artifacts_dir}: {expected_output}"
        ) from exc
    config_files = [
        _absolute_expanded_path(
            value,
            context,
            label=f"{task.task_id}.config_file",
            must_exist=True,
        )
        for value in task.config_files
    ]
    frozen_task_manifest = _task_paths(root, task.task_id)["manifest"].resolve(
        strict=True
    )
    frozen_task_document = read_json_object(frozen_task_manifest)
    frozen_environment = frozen_task_document.get("effective_launch_environment")
    if not isinstance(frozen_environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in frozen_environment.items()
    ):
        raise ControllerError(
            f"Frozen task evidence lacks effective launch environment: {task.task_id}"
        )
    current_effective_environment = _effective_launch_environment(task)
    if frozen_environment != current_effective_environment:
        raise ControllerError(
            f"Controller safety environment changed after task freeze: {task.task_id}"
        )
    if frozen_task_manifest not in config_files:
        config_files.append(frozen_task_manifest)
    runner_stage = _expand(task.runner_stage, context, label="runner_stage")
    run_id = _safe_id(
        f"{manifest.controller_id}-{task.task_id}-{instance_id}-a{attempt}",
        label="run_id",
    )
    log_path = layout.logs_dir / CONTROLLER_NAME / manifest.controller_id / f"{run_id}.log"
    launch_command = [
        str(python_executable),
        str(layout.project_root / "scripts/autodl/exp_run.py"),
        "--project-root",
        str(layout.project_root),
        "--data-root",
        str(layout.data_root),
        "launch",
        "--dataset",
        task.runner_dataset,
        "--stage",
        runner_stage,
        "--run-id",
        run_id,
        "--max-gpus",
        str(FOUR_GPU_RECOVERY_LIMIT),
        "--gpu-hard-limit",
        str(FOUR_GPU_RECOVERY_LIMIT),
        "--min-free-memory-mb",
        str(int(manifest.runtime.get("min_free_memory_mb", 16000))),
        "--idle-util-threshold",
        str(int(manifest.runtime.get("idle_util_threshold", 10))),
        "--input-manifest",
        str(input_manifest),
        "--expected-output",
        str(expected_output),
        "--required-log-marker",
        str(task.required_log_marker),
        "--log-path",
        str(log_path),
        "--launcher",
        str(manifest.runtime.get("worker_launcher", "auto")),
    ]
    for config_file in config_files:
        launch_command.extend(("--config-file", str(config_file)))
    for value in task.required_output_files:
        launch_command.extend(("--required-output-file", value))
    for group in task.required_output_any:
        launch_command.extend(("--required-output-any", "|".join(group)))
    required_absolute_outputs: list[Path] = []
    for value in task.required_absolute_output_files:
        path = _absolute_expanded_path(
            value,
            context,
            label=f"{task.task_id}.required_absolute_output_file",
            must_exist=False,
        )
        try:
            path.relative_to(layout.artifacts_dir)
        except ValueError as exc:
            raise ControllerError(
                f"Required absolute output escapes {layout.artifacts_dir}: {path}"
            ) from exc
        required_absolute_outputs.append(path)
        launch_command.extend(("--required-absolute-output-file", str(path)))
    environment = dict(frozen_environment)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if "BACE_PPO_RESUME_FROM_CHECKPOINT" in task.environment:
        raise ControllerError(
            "BACE_PPO_RESUME_FROM_CHECKPOINT is controller-owned retry state"
        )
    resume_checkpoint = _validated_b7_resume_checkpoint(layout, task, instance)
    if resume_checkpoint is not None:
        environment["BACE_PPO_RESUME_FROM_CHECKPOINT"] = str(resume_checkpoint)
    if task.oom_retry.enabled:
        batch_size = (
            task.oom_retry.retry_batch_size
            if instance.get("retry_kind") == "OOM"
            else task.oom_retry.initial_batch_size
        )
        environment[task.oom_retry.batch_env] = str(batch_size)
    if shard_manifest is not None:
        environment["PARENT_SHARD_MANIFEST"] = str(shard_manifest)
    for key, value in sorted(environment.items()):
        launch_command.extend(
            ("--env", f"{key}={_expand(value, context, label=f'{task.task_id}.environment')}")
        )
    if gpu is not None:
        launch_command.extend(
            (
                "--gpu-index",
                str(gpu.index),
                "--gpu-uuid",
                gpu.uuid,
                "--gpu-required",
                "--heavy",
            )
        )
    launch_command.extend(("--", *command))
    environment_for_runner = sanitized_environment()
    environment_for_runner.update(
        {
            "PYTHONPATH": str(layout.project_root),
            "AUTODL_CONTROL_ROOT": str(layout.control_root),
            "AUTODL_PYTHON": str(python_executable),
        }
    )
    completed = runner(
        launch_command,
        cwd=layout.project_root,
        env=environment_for_runner,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        launch_failure = (
            completed.stderr.strip() or completed.stdout.strip() or "exp_run launch failed"
        )
        failure_class = classify_failure(
            launch_failure,
            semantic_markers=task.semantic_failure_markers,
        )
        if transient_retry_allowed(
            failure_class,
            int(instance.get("transient_retry_count", 0)),
            max_transient_retries=int(
                manifest.runtime.get(
                    "max_transient_retries", DEFAULT_MAX_TRANSIENT_RETRIES
                )
            ),
        ):
            instance.update(
                {
                    "state": "NOT_STARTED",
                    "attempt": attempt + 1,
                    "transient_retry_count": int(
                        instance.get("transient_retry_count", 0)
                    )
                    + 1,
                    "retry_kind": "TRANSIENT_IO",
                    "resume_from_checkpoint": None,
                    "resume_source_output": None,
                    "failure_class": "TRANSIENT_IO_RETRY",
                    "failure_reason": "one bounded transient launch retry authorized",
                }
            )
        else:
            instance["state"] = "FAILED"
            instance["failure_class"] = "LAUNCH"
            instance["failure_reason"] = launch_failure
        _write_task_state(root, state)
        return
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ControllerError(
            f"exp_run returned non-JSON for {task.task_id}: {completed.stdout!r}"
        ) from exc
    instance.update(
        {
            "state": "STARTING",
            "run_id": response.get("run_id"),
            "launcher_pid": response.get("launcher_pid"),
            "launcher_identity": (
                read_process_identity(int(response["launcher_pid"]))
                if isinstance(response.get("launcher_pid"), int)
                and not isinstance(response.get("launcher_pid"), bool)
                else None
            ),
            "tmux_session": response.get("tmux_session"),
            "gpu_index": gpu.index if gpu else None,
            "gpu_uuid": gpu.uuid if gpu else None,
            "log_path": str(log_path),
            "expected_output": str(expected_output),
            "required_absolute_output_files": [
                str(path) for path in required_absolute_outputs
            ],
            "command_sha256": command_digest(command),
            "started_at": utc_now(),
            "heartbeat_at": utc_now(),
            "failure_class": None,
            "failure_reason": None,
        }
    )
    _write_task_state(root, state)
    _append_event(root, manifest, task=task, state="STARTING", instance=instance)


def _dependency_output_context(
    root: Path,
    task: TaskSpec,
    states: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    context: dict[str, str] = {}
    for dependency in task.depends_on:
        instances_by_id = states[dependency].get("instances") or {}
        instances = list(instances_by_id.values())
        outputs = [
            str(instance["expected_output"])
            for instance in instances
            if instance.get("state") == "PASS" and instance.get("expected_output")
        ]
        key = "dep_" + re.sub(r"[^A-Za-z0-9_]", "_", dependency) + "_output"
        for instance_id, instance in sorted(instances_by_id.items()):
            if instance.get("state") != "PASS" or not instance.get("expected_output"):
                continue
            instance_key = (
                key.removesuffix("_output")
                + "_"
                + re.sub(r"[^A-Za-z0-9_]", "_", str(instance_id))
                + "_output"
            )
            context[instance_key] = str(instance["expected_output"])
        if len(instances) == 1 and len(outputs) == 1:
            context[key] = outputs[0]
        else:
            # Sharded dependencies are consumed via the frozen aggregate gate,
            # which lists every run and expected output.
            context[key] = str(_task_paths(root, dependency)["root"])
    return context


def _summarize_controller_state(counts: Mapping[str, int]) -> str:
    """Return the dashboard state for one explicit set of task counts."""

    if counts.get("FAILED"):
        return "FAILED"
    if counts.get("RUNNING") or counts.get("STARTING"):
        return "RUNNING"
    if counts.get("READY") or counts.get("WAITING_RESOURCE"):
        return "WAITING_RESOURCE"
    if counts.get("WAITING_DEPENDENCY") or counts.get("BLOCKED"):
        return "BLOCKED"
    if counts.get("PASS"):
        return "PASS"
    return "NOT_STARTED"


def keep_alive_after_all_terminal(
    runtime: Mapping[str, Any], states: Sequence[Mapping[str, Any]]
) -> bool:
    """Keep polling terminal state without manufacturing a placeholder task."""

    if runtime.get("keep_alive_when_blocked", False) is not True or not states:
        return False
    instances = [
        instance
        for state in states
        for instance in (state.get("instances") or {}).values()
    ]
    return all(state.get("state") in TERMINAL_STATES for state in states) and not any(
        instance.get("state") in ACTIVE_STATES for instance in instances
    )


def _write_controller_snapshot(
    root: Path,
    manifest: ControllerManifest,
    states: Mapping[str, Mapping[str, Any]],
    *,
    resources: HostResources | None,
    resource_failures: Sequence[str],
    gpu_audit: Sequence[Mapping[str, Any]],
) -> None:
    counts: dict[str, int] = {}
    workload_counts: dict[str, int] = {}
    for task_id, state in states.items():
        value = str(state.get("state"))
        counts[value] = counts.get(value, 0) + 1
        task = manifest.by_id[task_id]
        if task.dataset not in {"taste", "tastemolnet"}:
            workload_counts[value] = workload_counts.get(value, 0) + 1
    overall = _summarize_controller_state(counts)
    workload_state = _summarize_controller_state(workload_counts)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "controller_id": manifest.controller_id,
        "manifest_sha256": manifest.sha256,
        "state": overall,
        # Keep the raw state (including the immutable Taste license gate), but
        # expose the executable science workload separately so a successful
        # run is not presented as a failed experiment solely due to Taste.
        "workload_state": workload_state,
        "pid": os.getpid(),
        "process_identity": read_process_identity(os.getpid()),
        "heartbeat_at": utc_now(),
        "task_counts": counts,
        "workload_task_counts": workload_counts,
        "tasks": {
            task_id: {
                "state": state.get("state"),
                "stage": state.get("stage"),
                "dataset": state.get("dataset"),
                "reason": state.get("reason"),
            }
            for task_id, state in sorted(states.items())
        },
        "host_resources": (
            {
                "cpu_count": resources.cpu_count,
                "load_1m": resources.load_1m,
                "available_ram_gb": resources.available_ram_gb,
                "free_disk_gb": resources.free_disk_gb,
            }
            if resources
            else None
        ),
        "resource_gate_failures": list(resource_failures),
        "gpu_lock_audit": list(gpu_audit),
        "tastemolnet": {
            "state": "BLOCKED",
            "reason": "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW",
        },
        "paper": {"state": "FROZEN", "mutations_allowed": False},
    }
    atomic_write_json(root / "controller_state.json", payload)
    atomic_write_json(
        root / "heartbeat.json",
        {
            "schema_version": SCHEMA_VERSION,
            "controller_id": manifest.controller_id,
            "pid": os.getpid(),
            "process_identity": payload["process_identity"],
            "heartbeat_at": payload["heartbeat_at"],
            "state": overall,
            "workload_state": workload_state,
        },
    )


def _write_heartbeat_pulse(
    root: Path, manifest: ControllerManifest, *, state: str
) -> None:
    atomic_write_json(
        root / "heartbeat.json",
        {
            "schema_version": SCHEMA_VERSION,
            "controller_id": manifest.controller_id,
            "manifest_sha256": manifest.sha256,
            "pid": os.getpid(),
            "process_identity": read_process_identity(os.getpid()),
            "heartbeat_at": utc_now(),
            "state": state,
        },
    )


class _ControllerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: TextIO | None = None

    def __enter__(self) -> "_ControllerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ControllerError("A four-GPU recovery controller is already active") from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {"pid": os.getpid(), "started_at": utc_now()}, handle, sort_keys=True
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self.handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def controller_tick(
    layout: Any,
    root: Path,
    manifest: ControllerManifest,
    states: dict[str, dict[str, Any]],
    *,
    python_executable: Path,
    dry_run: bool,
    host_resource_reader: Callable[[Path], HostResources] = collect_host_resources,
    gpu_sampler: Callable[[], list[GPUObservation]] = query_gpu_inventory,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    _write_heartbeat_pulse(root, manifest, state="RECONCILING")
    by_id = manifest.by_id
    for task in manifest.tasks:
        state = states[task.task_id]
        for instance in (state.get("instances") or {}).values():
            _reconcile_instance(
                layout,
                task,
                instance,
                launch_grace_seconds=int(
                    manifest.runtime.get(
                        "launch_grace_seconds", DEFAULT_LAUNCH_GRACE_SECONDS
                    )
                ),
                max_transient_retries=int(
                    manifest.runtime.get(
                        "max_transient_retries", DEFAULT_MAX_TRANSIENT_RETRIES
                    )
                ),
            )
        _write_task_state(root, state)
    # Repeated topological passes let a newly completed parent unlock children
    # in this same work-conserving tick.
    for task_id in dependency_order(manifest.tasks):
        _aggregate_task_state(
            root, layout, manifest, by_id[task_id], states[task_id], states
        )

    resources = host_resource_reader(layout.runtime_root)
    resource_failures = resource_gate_failures(resources, manifest.resource_gates)
    candidates = scheduler_candidates(manifest.tasks, states)
    shard_paths: dict[str, dict[str, Path]] = {}
    for task_id, _instance_id in candidates:
        task = by_id[task_id]
        if task.shards and task_id not in shard_paths:
            try:
                shard_paths[task_id] = _prepare_shards(
                    layout, root, task, python_executable, states
                )
            except Exception as exc:
                _set_task_state(
                    root,
                    manifest,
                    task,
                    states[task_id],
                    "FAILED",
                    reason=f"shard preparation failed: {type(exc).__name__}: {exc}",
                )

    candidates = scheduler_candidates(manifest.tasks, states)
    cpu_candidates = [value for value in candidates if by_id[value[0]].resource == "cpu"]
    gpu_candidates = [value for value in candidates if by_id[value[0]].resource == "gpu"]
    max_cpu_tasks = int(manifest.runtime.get("max_cpu_tasks", 2))
    if max_cpu_tasks < 1:
        raise ControllerError("runtime.max_cpu_tasks must be positive")
    active_cpu_tasks = sum(
        1
        for task in manifest.tasks
        if task.resource == "cpu"
        for instance in (states[task.task_id].get("instances") or {}).values()
        if instance.get("state") in ACTIVE_STATES
    )
    cpu_capacity = max(0, max_cpu_tasks - active_cpu_tasks)
    deferred_cpu_candidates = cpu_candidates[cpu_capacity:]
    cpu_candidates = cpu_candidates[:cpu_capacity]
    observations: Sequence[GPUObservation] = ()
    selected: list[GPUObservation] = []
    gpu_audit: list[dict[str, Any]] = []
    if gpu_candidates and not dry_run and not resource_failures:
        def sampling_sleep(seconds: float) -> None:
            _write_heartbeat_pulse(root, manifest, state="SAMPLING_GPU_IDLE")
            sleep(seconds)

        stable = observe_stable_idle_gpus(
            stable_seconds=float(manifest.runtime.get("stable_idle_seconds", 60)),
            sample_interval_seconds=float(
                manifest.runtime.get("sample_interval_seconds", 5)
            ),
            min_free_memory_mb=int(
                manifest.runtime.get("min_free_memory_mb", 16000)
            ),
            max_utilization_percent=int(
                manifest.runtime.get("idle_util_threshold", 10)
            ),
            sampler=gpu_sampler,
            sleep=sampling_sleep,
        )
        observations = stable.observations
        gpu_audit = audit_gpu_locks(
            layout.locks_dir,
            observations,
            probe_advisory_lock=True,
        )
        selected = stable.selected(
            max_gpus=FOUR_GPU_RECOVERY_LIMIT,
            hard_limit=FOUR_GPU_RECOVERY_LIMIT,
            lock_root=layout.locks_dir,
            eligible_uuids=allocation_safe_gpu_uuids(gpu_audit),
        )
    else:
        try:
            observations = gpu_sampler()
        except AutoDLRuntimeError:
            observations = ()
        gpu_audit = audit_gpu_locks(
            layout.locks_dir,
            observations,
            probe_advisory_lock=not dry_run,
        )

    if resource_failures:
        for task_id, _instance_id in candidates:
            state = states[task_id]
            _set_task_state(
                root,
                manifest,
                by_id[task_id],
                state,
                "WAITING_RESOURCE",
                reason="; ".join(resource_failures),
            )
    elif not dry_run:
        for task_id, instance_id in cpu_candidates:
            task = by_id[task_id]
            _launch_instance(
                layout,
                root,
                manifest,
                task,
                states[task_id],
                instance_id,
                python_executable=python_executable,
                gpu=None,
                shard_manifest=shard_paths.get(task_id, {}).get(instance_id),
                extra_context=_dependency_output_context(root, task, states),
            )
        for (task_id, instance_id), gpu in zip(gpu_candidates, selected, strict=False):
            task = by_id[task_id]
            _launch_instance(
                layout,
                root,
                manifest,
                task,
                states[task_id],
                instance_id,
                python_executable=python_executable,
                gpu=gpu,
                shard_manifest=shard_paths.get(task_id, {}).get(instance_id),
                extra_context=_dependency_output_context(root, task, states),
            )
        if len(selected) < len(gpu_candidates):
            for task_id, _instance_id in gpu_candidates[len(selected) :]:
                state = states[task_id]
                if state.get("state") not in ACTIVE_STATES:
                    _set_task_state(
                        root,
                        manifest,
                        by_id[task_id],
                        state,
                        "WAITING_RESOURCE",
                        reason="WAITING_FOR_60S_STABLE_IDLE_GPU",
                    )
        for task_id, _instance_id in deferred_cpu_candidates:
            state = states[task_id]
            if state.get("state") not in ACTIVE_STATES:
                _set_task_state(
                    root,
                    manifest,
                    by_id[task_id],
                    state,
                    "WAITING_RESOURCE",
                    reason="WAITING_FOR_CONTROLLER_CPU_SLOT",
                )

    for task_id in dependency_order(manifest.tasks):
        _aggregate_task_state(
            root, layout, manifest, by_id[task_id], states[task_id], states
        )

    _write_controller_snapshot(
        root,
        manifest,
        states,
        resources=resources,
        resource_failures=resource_failures,
        gpu_audit=gpu_audit,
    )
    publish_user_registry(layout, root, manifest, states)
    return {
        "candidates": candidates,
        "selected_gpu_uuids": [gpu.uuid for gpu in selected],
        "resource_gate_failures": resource_failures,
        "dry_run": dry_run,
    }


def _build_layout(args: argparse.Namespace) -> Any:
    project_root = resolve_project_root(args.project_root)
    data_root = select_data_root(project_root, explicit=args.data_root)
    return build_runtime_layout(
        project_root=project_root,
        data_root=data_root,
        control_root=args.control_root,
    ).ensure()


def _fresh_manifest_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish validated manifest bytes without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ControllerError(f"Continuation manifest target already exists: {path}")
    candidate = path.with_name(f".{path.name}.candidate-{os.getpid()}")
    if candidate.exists() or candidate.is_symlink():
        raise ControllerError(f"Stale continuation manifest candidate exists: {candidate}")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with candidate.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        load_controller_manifest(candidate)
        try:
            os.link(candidate, path)
        except FileExistsError as exc:
            raise ControllerError(
                f"Continuation manifest target appeared concurrently: {path}"
            ) from exc
        fsync_directory(path.parent)
    finally:
        candidate.unlink(missing_ok=True)


def build_bace_continuation(args: argparse.Namespace) -> int:
    """Freeze one continuation manifest after B10 and the MolCLR repair pass."""

    layout = _build_layout(args)
    source = load_controller_manifest(args.source_manifest)
    controller_id = _safe_id(str(args.controller_id), label="controller_id")
    if controller_id == source.controller_id:
        raise ControllerError("Continuation requires a new controller ID")
    source_root = _controller_root(layout, source.controller_id)
    controller_root = _controller_root(layout, controller_id)
    if controller_root.exists() or controller_root.is_symlink():
        raise ControllerError(
            f"Fresh continuation controller root already exists: {controller_root}"
        )
    output_root = (
        args.output_root.expanduser().resolve(strict=False)
        if args.output_root is not None
        else (
            layout.artifacts_dir
            / "autodl"
            / "four_gpu_recovery"
            / controller_id
            / "bace"
            / "frozen_gnn_downstream"
        ).resolve(strict=False)
    )
    wnode_cache_db = (
        args.wnode_cache_db.expanduser().resolve(strict=False)
        if args.wnode_cache_db is not None
        else (
            layout.cache_dir
            / "bace"
            / "frozen_gnn_downstream"
            / controller_id
            / "wnode"
            / "wnode_cache.sqlite3"
        ).resolve(strict=False)
    )
    output_manifest = (
        args.output_manifest.expanduser().resolve(strict=False)
        if args.output_manifest is not None
        else (
            layout.control_root
            / CONTROLLER_NAME
            / "manifests"
            / f"{controller_id}.json"
        ).resolve(strict=False)
    )
    for path, parent, label in (
        (output_root, layout.artifacts_dir, "continuation output root"),
        (wnode_cache_db, layout.cache_dir, "continuation WNode cache"),
        (output_manifest, layout.control_root, "continuation manifest"),
    ):
        try:
            path.relative_to(parent.resolve(strict=True))
        except ValueError as exc:
            raise ControllerError(f"{label} escapes persistent runtime: {path}") from exc
    if output_root.exists() or output_root.is_symlink():
        raise ControllerError(f"Continuation output root is not fresh: {output_root}")
    if wnode_cache_db.exists() or wnode_cache_db.is_symlink():
        raise ControllerError(f"Continuation WNode cache is not fresh: {wnode_cache_db}")

    build_kwargs = {
        "source_manifest": source.path,
        "source_controller_root": source_root,
        "runs_root": layout.runs_root,
        "molclr_repair_run_id": str(args.molclr_repair_run_id),
        "controller_id": controller_id,
        "output_root": output_root,
        "wnode_cache_db": wnode_cache_db,
    }
    preliminary = build_bace_continuation_payload(**build_kwargs)
    policy = validate_continuation_policy(
        preliminary.get("continuation"), controller_id=controller_id
    )
    if policy is None:
        raise ControllerError("Continuation builder omitted its safety policy")
    persistent_inputs = (
        (Path(str(policy["molclr_repair_output"])), layout.artifacts_dir, "MolCLR repair output"),
        (
            Path(str(policy["molclr_node_embedding_cache"])),
            layout.cache_dir,
            "MolCLR repair cache",
        ),
    )
    for path, parent, label in persistent_inputs:
        try:
            path.resolve(strict=True).relative_to(parent.resolve(strict=True))
        except (FileNotFoundError, ValueError) as exc:
            raise ControllerError(f"{label} escapes its persistent runtime root") from exc
    for task in preliminary["tasks"]:
        if not isinstance(task, dict) or task.get("adopt_existing_run_id") is None:
            continue
        adopted_output = Path(str(task.get("expected_output", ""))).expanduser()
        try:
            adopted_output.resolve(strict=True).relative_to(
                layout.artifacts_dir.resolve(strict=True)
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ControllerError(
                f"Adopted output escapes persistent artifacts: {adopted_output}"
            ) from exc
    # Hold the predecessor lock while re-reading every PASS run and publishing
    # the manifest.  A later controller launch reacquires and retains this same
    # lock for its full lifetime, so the v2 writer can never run in parallel.
    with PredecessorControllerGuard(policy, require_fresh_targets=True):
        payload = build_bace_continuation_payload(**build_kwargs)
        if payload.get("continuation") != preliminary.get("continuation"):
            raise ControllerError("Continuation evidence changed while acquiring lock")
        _fresh_manifest_write(output_manifest, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "controller_id": controller_id,
                "manifest": str(output_manifest),
                "manifest_sha256": sha256_file(output_manifest),
                "source_controller_id": source.controller_id,
                "fresh_output_root": str(output_root),
                "fresh_wnode_cache_db": str(wnode_cache_db),
                "adopted_run_count": len(policy["adopted_run_ids"]),
                "b8_b9_adoption": "8_FLATTENED_MAIN_TASKS",
                "next_fresh_stage": "B11_CROSS_PARENT_VERIFIED",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_controller(args: argparse.Namespace) -> int:
    layout = _build_layout(args)
    manifest = load_controller_manifest(args.manifest)
    python_executable = _resolve_python(args.python)
    raw_manifest = _load_json_or_yaml(manifest.path)
    policy = validate_continuation_policy(
        raw_manifest.get("continuation"), controller_id=manifest.controller_id
    )
    root = _controller_root(layout, manifest.controller_id)
    first_launch = not (root / "controller_manifest.json").is_file()
    with ExitStack() as stack:
        if policy is not None:
            stack.enter_context(
                PredecessorControllerGuard(
                    policy, require_fresh_targets=first_launch
                )
            )
        root, states = initialize_controller_state(layout, manifest)
        stack.enter_context(_ControllerLock(root / "controller.lock"))
        bind_adopted_runs(
            layout,
            root,
            manifest,
            states,
            python_executable=python_executable,
        )
        _append_event(root, manifest, task=None, state="RUNNING")
        while True:
            result = controller_tick(
                layout,
                root,
                manifest,
                states,
                python_executable=python_executable,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            if args.once or args.dry_run:
                return 0
            non_taste = [
                state
                for task_id, state in states.items()
                if manifest.by_id[task_id].dataset not in {"taste", "tastemolnet"}
            ]
            active_instances = [
                instance
                for state in non_taste
                for instance in (state.get("instances") or {}).values()
                if instance.get("state") in ACTIVE_STATES
            ]
            all_terminal = non_taste and all(
                state.get("state") in TERMINAL_STATES for state in non_taste
            ) and not active_instances
            if all_terminal and not keep_alive_after_all_terminal(
                manifest.runtime, non_taste
            ):
                return 0 if all(
                    state.get("state") in {"PASS", "SKIPPED"} for state in non_taste
                ) else 4
            remaining = float(manifest.runtime.get("poll_seconds", 60))
            while remaining > 0:
                _write_heartbeat_pulse(root, manifest, state="WAITING_NEXT_TICK")
                interval = min(10.0, remaining)
                time.sleep(interval)
                remaining -= interval


def validate_manifest(args: argparse.Namespace) -> int:
    manifest = load_controller_manifest(args.manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "controller_id": manifest.controller_id,
                "manifest": str(manifest.path),
                "manifest_sha256": manifest.sha256,
                "task_order": dependency_order(manifest.tasks),
                "max_gpus": FOUR_GPU_RECOVERY_LIMIT,
                "taste_policy": "BLOCKED_LICENSE_REVIEW",
                "paper": "FROZEN",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--config", action="append", default=[])
    commands = parser.add_subparsers(dest="action", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    continuation = commands.add_parser("build-bace-continuation")
    continuation.add_argument("--source-manifest", type=Path, required=True)
    continuation.add_argument("--molclr-repair-run-id", required=True)
    continuation.add_argument("--controller-id", required=True)
    continuation.add_argument("--output-manifest", type=Path)
    continuation.add_argument("--output-root", type=Path)
    continuation.add_argument("--wnode-cache-db", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--once", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "validate":
            return validate_manifest(args)
        if args.action == "build-bace-continuation":
            return build_bace_continuation(args)
        if args.action == "run":
            return run_controller(args)
        raise ControllerError(f"Unknown action: {args.action}")
    except (
        BaceContinuationError,
        ControllerError,
        AutoDLRuntimeError,
        OSError,
        ValueError,
    ) as exc:
        print(f"FOUR_GPU_RECOVERY_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
