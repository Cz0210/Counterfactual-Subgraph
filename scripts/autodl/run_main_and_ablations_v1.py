#!/usr/bin/env python3
"""Minimal priority scheduler for the remaining main cells and ablations.

The scheduler deliberately delegates scientific work to dataset-specific
launchers.  Its only responsibilities are single-owner dispatch, durable
heartbeat/status, and enforcing MAIN > LLM > GNN admission gates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
from uuid import uuid4


SCHEMA = "main_and_ablations_controller_v1"
HEARTBEAT_SCHEMA = "main_and_ablations_heartbeat_v1"
DEFAULT_POLL_SECONDS = 30
LAUNCH_GRACE_SECONDS = 60
RETRY_BACKOFF_SECONDS = (60, 120, 300)
MAX_LAUNCH_ATTEMPTS = len(RETRY_BACKOFF_SECONDS)
TASK_SPEC_SCHEMA = "main_and_ablations_task_spec_v1"
DISPATCH_SCHEMA = "main_and_ablations_dispatch_state_v2"
RESOLVED_TASK_SPEC_SCHEMA = "main_and_ablations_resolved_task_spec_v1"
LAUNCH_EXIT_SCHEMA = "main_and_ablations_launcher_exit_v1"
COMPONENTS = (
    "mut_continuation",
    "t14_resume",
    "t8_valid_zero_finalizer",
    "llm_ablation",
    "gnn_ablation",
)
DEFAULT_LAUNCHERS = {
    "mut_continuation": "scripts/autodl/launch_mut_throttled_continuation_v1.sh",
    "t14_resume": "scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh",
    "t8_valid_zero_finalizer": "scripts/autodl/launch_tastemolnet_globalgce_valid_zero_relay_v1.sh",
    "llm_ablation": "scripts/autodl/launch_llm_ablation_core_v1.sh",
    "gnn_ablation": "scripts/autodl/launch_gnn_five_backbone_ablation_v1.sh",
}
TASK_SPEC_ENV = {
    "mut_continuation": "MUT_CONTINUATION_TASK_SPEC",
    "t14_resume": "T14_RESUME_TASK_SPEC",
    "t8_valid_zero_finalizer": "T8_ZERO_FINALIZER_TASK_SPEC",
    "llm_ablation": "LLM_ABLATION_TASK_SPEC",
    "gnn_ablation": "GNN_ABLATION_TASK_SPEC",
}
COMPONENT_REQUIRED_ENV = {
    "mut_continuation": frozenset(
        {
            "AUTODL_DATA_ROOT",
            "AUTODL_RUNTIME_ROOT",
            "AUTODL_CONTROL_ROOT",
            "AUTODL_PYTHON",
            "RUN_GNN_ABLATION",
            "MUT_FAST_SPEC",
            "MUT_TRACE_AUTHORIZATION_RECEIPT",
            "MUT_TRACE_PROTECTED_MANIFEST",
            "MUT_TRACE_HISTORICAL_PROJECT_ROOT",
            "MUT_TRACE_INSTRUMENTATION_PROJECT_ROOT",
            "MUT_TRACE_SEMANTIC_FINALIZER_PROJECT_ROOT",
            "MUT_TRACE_CONTROLLER_PID",
            "MUT_TRACE_CONTROLLER_START_TICKS",
            "MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE",
            "MUT_COMPLETED_A_ARM_ROOT",
            "MUT_TRACE_OUTPUT_ROOT",
        }
    ),
    "t14_resume": frozenset(
        {
            "AUTODL_DATA_ROOT",
            "AUTODL_RUNTIME_ROOT",
            "AUTODL_CONTROL_ROOT",
            "AUTODL_PYTHON",
            "RUN_TASTEMOLNET",
            "TASTE_RESEARCH_COMPUTE_ALLOWED",
            "TASTE_PAPER_RESULTS_ALLOWED",
            "TASTE_DATA_REDISTRIBUTION_ALLOWED",
            "RUN_GNN_ABLATION",
            "T14_AUDITOR_REPO_ROOT",
            "T14_CHECKPOINT_ROOT",
            "T14_RESUME_SPEC",
            "TASTEMOLNET_T14_OUTPUT",
            "TASTEMOLNET_T14_RUN_ID",
            "TASTEMOLNET_T14_RESUME",
            "TASTEMOLNET_T14_GPU_INDEX",
            "TASTEMOLNET_T2_ADOPTION_ROOT",
            "TASTEMOLNET_T2_ADOPTION_GATE_SHA256",
            "TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256",
            "TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256",
            "TASTEMOLNET_T3_OUTPUT_ROOT",
            "TASTEMOLNET_T4_OUTPUT_ROOT",
            "TASTEMOLNET_TRAIN_CSV",
            "COMRECGC_OFFICIAL_ROOT",
        }
    ),
}
COMPONENT_OUTPUT_ENV = {
    "mut_continuation": "MUT_TRACE_OUTPUT_ROOT",
    "t14_resume": "TASTEMOLNET_T14_OUTPUT",
    "t8_valid_zero_finalizer": "TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _required_env(name: str, expected: str) -> None:
    observed = os.environ.get(name, expected)
    if observed != expected:
        raise ValueError(f"{name} must equal {expected!r}, observed {observed!r}")


def validate_policy() -> dict[str, Any]:
    for name, expected in (
        ("RUN_MAIN_TABLE", "1"),
        ("RUN_LLM_ABLATION", "1"),
        ("RUN_GNN_ABLATION", "1"),
        ("MAIN_TABLE_PRIORITY", "100"),
        ("LLM_ABLATION_PRIORITY", "50"),
        ("GNN_ABLATION_PRIORITY", "20"),
        ("ALLOW_MUT_CONTINUATION_RECOVERY", "1"),
        ("MUT_CPU_WORKERS", "2"),
        ("MUT_BASELINE_WINDOW_SECONDS", "1800"),
        ("MUT_SLOWDOWN_THRESHOLD", "0.15"),
        ("MUT_SLOWDOWN_SUSTAIN_SECONDS", "1200"),
        ("ALLOW_T14_EXTERNAL_CONVERGENCE_AUDITOR", "1"),
        ("ALLOW_TASTE_GLOBALGCE_VALID_ZERO_RULE_RESULT", "1"),
        ("T8_RECOVERY_MAX_ATTEMPTS", "1"),
        ("LLM_EARLY_START_MIN_MATRIX", "13"),
        ("LLM_EARLY_START_IDLE_SECONDS", "1200"),
        ("LLM_MAX_EARLY_GPUS", "1"),
        ("RUN_MATCHED_SFT_AUXILIARY_STUDY", "0"),
        ("GNN_START_AFTER_MATRIX", "16"),
        ("GNN_PRIMARY_SEEDS", "7"),
        ("GNN_MAX_CONCURRENT_GPUS", "2"),
        ("RUN_GRAPH_MAMBA", "0"),
    ):
        _required_env(name, expected)
    llm_variants = os.environ.get(
        "LLM_CORE_VARIANTS",
        "BRICS_FIXED,CHEMLLM_7B_OFF_THE_SHELF,CHEMLLM_7B_PPO_LORA_MAIN,CHEMLLM_2B_OFF_THE_SHELF",
    )
    gnn_backbones = os.environ.get(
        "GNN_BACKBONES", "gine,gin,gcn,gatv2,gatedgcn_plus"
    )
    if llm_variants.split(",") != [
        "BRICS_FIXED",
        "CHEMLLM_7B_OFF_THE_SHELF",
        "CHEMLLM_7B_PPO_LORA_MAIN",
        "CHEMLLM_2B_OFF_THE_SHELF",
    ]:
        raise ValueError("LLM_CORE_VARIANTS differs from the authorized core rows")
    if gnn_backbones.split(",") != [
        "gine",
        "gin",
        "gcn",
        "gatv2",
        "gatedgcn_plus",
    ]:
        raise ValueError("GNN_BACKBONES differs from the authorized five rows")
    return {
        "main_priority": 100,
        "llm_priority": 50,
        "gnn_priority": 20,
        "llm_variants": llm_variants.split(","),
        "gnn_backbones": gnn_backbones.split(","),
        "graph_mamba_run_enabled": False,
        "matched_sft_auxiliary_run_enabled": False,
    }


def _matrix(path: Path) -> dict[str, Any]:
    pointer = _load_json(path)
    if pointer.get("schema_version") != "fast16_matrix_authority_pointer_v1":
        raise ValueError("matrix authority pointer schema changed")
    count = pointer.get("latest_count")
    cells = pointer.get("applied_cells")
    if not isinstance(count, int) or not 0 <= count <= 16:
        raise ValueError("matrix latest_count is invalid")
    if not isinstance(cells, list) or len(cells) != count or len(set(cells)) != count:
        raise ValueError("matrix applied_cells do not match latest_count")
    return {
        "count": count,
        "cells": tuple(str(item) for item in cells),
        "pointer": pointer,
    }


def _cell_present(matrix: Mapping[str, Any], dataset: str, method: str) -> bool:
    wanted = f"{dataset}/{method}".casefold()
    return any(str(item).casefold() == wanted for item in matrix["cells"])


def _process_identity(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        return {"pid": pid, "alive": False}
    try:
        raw = (proc / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        ticks = int(fields[19])
        command_bytes = (proc / "cmdline").read_bytes()
        command = command_bytes.replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        cwd = str((proc / "cwd").resolve(strict=True))
    except (OSError, ValueError, IndexError):
        return {"pid": pid, "alive": False, "unreadable": True}
    return {
        "pid": pid,
        "alive": True,
        "start_ticks": ticks,
        "command": command,
        "command_sha256": hashlib.sha256(command_bytes).hexdigest(),
        "cwd": cwd,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a nonempty string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _seconds_since(value: object, *, now_epoch: float) -> float:
    return now_epoch - _timestamp(value).timestamp()


def _physical_path(
    raw: object,
    *,
    label: str,
    kind: str | None = None,
    must_exist: bool = True,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a nonempty path string")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{label} must be one absolute non-symlink path")
    if must_exist:
        if kind == "file" and not path.is_file():
            raise ValueError(f"{label} must be an existing file")
        if kind == "dir" and not path.is_dir():
            raise ValueError(f"{label} must be an existing directory")
        if kind is None and not path.exists():
            raise ValueError(f"{label} must exist")
    return path


def _replace_attempt_tokens(value: str, *, attempt_uuid: str, attempt: int) -> str:
    return value.replace("{attempt_uuid}", attempt_uuid).replace(
        "{attempt_number}", str(attempt)
    )


def _resolve_launcher(project_root: Path, component: str, spec: Mapping[str, Any]) -> Path:
    configured = spec.get("launcher")
    if not isinstance(configured, str) or not configured:
        configured = DEFAULT_LAUNCHERS[component]
    candidate = Path(configured)
    launcher = candidate if candidate.is_absolute() else project_root / candidate
    if not launcher.is_file() or launcher.is_symlink():
        raise ValueError(f"{component} launcher must be one physical file")
    try:
        launcher.resolve(strict=True).relative_to(project_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{component} launcher escapes its immutable repo") from exc
    return launcher


def _load_task_spec(
    project_root: Path, component: str
) -> tuple[Path, dict[str, Any], str]:
    variable = TASK_SPEC_ENV[component]
    configured = os.environ.get(variable)
    if not configured:
        raise ValueError(f"{variable} is required; ambient launcher env is forbidden")
    path = _physical_path(configured, label=variable, kind="file")
    value = _load_json(path)
    if value.get("schema_version") != TASK_SPEC_SCHEMA:
        raise ValueError(f"{component} task spec schema changed")
    for field in (
        "task_id",
        "task_type",
        "repo_root",
        "execution_commit",
        "python",
        "config",
        "manifest",
        "input_root",
        "output_root",
        "gpu_request",
        "cpu_request",
        "memory_request",
        "required_environment",
        "owner",
        "terminal",
    ):
        if field not in value:
            raise ValueError(f"{component} task spec is missing {field}")
    if value.get("task_type") != component:
        raise ValueError(f"{component} task_type differs from dispatch component")
    task_id = value.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"{component} task_id must be nonempty")
    repo_root = _physical_path(
        value["repo_root"], label=f"{component}.repo_root", kind="dir"
    ).resolve(strict=True)
    if repo_root != project_root.resolve(strict=True):
        raise ValueError(f"{component} repo_root differs from the sidecar checkout")
    commit = value.get("execution_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(
        char not in "0123456789abcdef" for char in commit
    ):
        raise ValueError(f"{component} execution_commit must be one lowercase SHA-1")
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != commit:
        raise ValueError(f"{component} execution_commit differs from repo HEAD")
    python_raw = value["python"]
    if not isinstance(python_raw, str) or not Path(python_raw).is_absolute():
        raise ValueError(f"{component}.python must be absolute")
    python = Path(python_raw).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"{component}.python must be executable")
    config = _physical_path(
        value["config"], label=f"{component}.config", kind="file"
    ).resolve(strict=True)
    if config != (repo_root / "configs/hpc.yaml").resolve(strict=True):
        raise ValueError(f"{component}.config must bind repo configs/hpc.yaml")
    _physical_path(value["manifest"], label=f"{component}.manifest")
    _physical_path(value["input_root"], label=f"{component}.input_root")
    output_root = value.get("output_root")
    if not isinstance(output_root, str) or not output_root.startswith("/"):
        raise ValueError(f"{component}.output_root must be absolute")
    output_mode = value.get("output_mode", "fresh")
    if output_mode not in {"fresh", "resume"}:
        raise ValueError(f"{component}.output_mode must be fresh or resume")
    for label in ("gpu_request", "cpu_request", "memory_request", "owner", "terminal"):
        if not isinstance(value[label], Mapping):
            raise ValueError(f"{component}.{label} must be an object")
    workers = value["cpu_request"].get("workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError(f"{component}.cpu_request.workers must be positive")
    gpu_required = value["gpu_request"].get("required")
    if not isinstance(gpu_required, bool):
        raise ValueError(f"{component}.gpu_request.required must be boolean")
    if gpu_required:
        index = value["gpu_request"].get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in range(4):
            raise ValueError(f"{component}.gpu_request.index must be in [0,3]")
        lease_path = value["gpu_request"].get("lease_path")
        if not isinstance(lease_path, str) or not Path(lease_path).is_absolute():
            raise ValueError(f"{component}.gpu_request.lease_path must be absolute")
    headroom = value["memory_request"].get("required_headroom_bytes")
    if not isinstance(headroom, int) or isinstance(headroom, bool) or headroom < 0:
        raise ValueError(
            f"{component}.memory_request.required_headroom_bytes must be nonnegative"
        )
    required_environment = value["required_environment"]
    if not isinstance(required_environment, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(environment_value, str)
        for key, environment_value in required_environment.items()
    ):
        raise ValueError(f"{component}.required_environment must map strings to strings")
    missing = sorted(
        COMPONENT_REQUIRED_ENV.get(component, frozenset())
        - set(required_environment)
    )
    if missing:
        raise ValueError(f"{component}.required_environment is missing {missing}")
    output_environment_variable = COMPONENT_OUTPUT_ENV.get(component)
    if output_environment_variable is not None:
        configured_output = required_environment.get(output_environment_variable)
        if configured_output != output_root:
            raise ValueError(
                f"{component}.{output_environment_variable} must equal task output_root"
            )
    configured_python = required_environment.get("AUTODL_PYTHON", str(python_raw))
    if Path(configured_python).resolve(strict=True) != python:
        raise ValueError(f"{component} AUTODL_PYTHON differs from task spec python")
    if required_environment.get("RUN_GNN_ABLATION", "0") != "0" and component in {
        "mut_continuation",
        "t14_resume",
        "t8_valid_zero_finalizer",
    }:
        raise ValueError(f"{component} must keep RUN_GNN_ABLATION=0")
    if component == "t14_resume":
        for name in ("T14_AUDITOR_REPO_ROOT", "T14_CHECKPOINT_ROOT", "T14_RESUME_SPEC"):
            _physical_path(
                required_environment[name], label=f"{component}.{name}"
            )
        if required_environment.get("TASTEMOLNET_T14_RESUME", "1") != "1":
            raise ValueError("t14_resume must set TASTEMOLNET_T14_RESUME=1")
        if (
            value["gpu_request"].get("required") is not True
            or value["gpu_request"].get("index") != 2
            or required_environment.get("TASTEMOLNET_T14_GPU_INDEX") != "2"
        ):
            raise ValueError("t14_resume must request physical GPU2")
        if headroom <= 0:
            raise ValueError("t14_resume requires an evidence-based memory headroom")
        serial_auditor = value.get("serial_auditor")
        if not isinstance(serial_auditor, Mapping) or not isinstance(
            serial_auditor.get("active"), bool
        ):
            raise ValueError("t14_resume.serial_auditor must declare active boolean")
        if serial_auditor["active"]:
            _physical_path(
                serial_auditor.get("heartbeat_path"),
                label="t14_resume.serial_auditor.heartbeat_path",
                kind="file",
            )
            relay_ticks = serial_auditor.get("controller_start_ticks")
            relay_pid = serial_auditor.get("controller_pid")
            if (
                not isinstance(relay_ticks, int)
                or isinstance(relay_ticks, bool)
                or relay_ticks <= 0
                or not isinstance(relay_pid, int)
                or isinstance(relay_pid, bool)
                or relay_pid <= 0
            ):
                raise ValueError(
                    "t14_resume.serial_auditor PID/start ticks must be positive"
                )
    if component == "mut_continuation":
        if (
            value["gpu_request"].get("required") is not True
            or value["gpu_request"].get("index") != 0
            or workers != 2
        ):
            raise ValueError("mut_continuation must request GPU0 and two CPU workers")
    owner = value["owner"]
    fixed_command_sha = owner.get("command_sha256")
    command_argv = owner.get("command_argv")
    fixed_command_valid = (
        isinstance(fixed_command_sha, str)
        and len(fixed_command_sha) == 64
        and all(char in "0123456789abcdef" for char in fixed_command_sha)
    )
    command_argv_valid = isinstance(command_argv, list) and bool(command_argv) and all(
        isinstance(argument, str) and bool(argument) for argument in command_argv
    )
    if not fixed_command_valid and not command_argv_valid:
        raise ValueError(
            f"{component}.owner requires command_sha256 or deterministic command_argv"
        )
    _resolve_launcher(project_root, component, value)
    return path, value, _sha256_file(path)


def _adopt_live_mut_owner(
    path_text: str | None, *, max_age_seconds: int = 120
) -> dict[str, Any] | None:
    """Adopt an already-running Mut owner instead of launching a duplicate."""

    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        return None
    try:
        heartbeat = _load_json(path)
        worker_pid = heartbeat.get("worker_pid")
        worker_start_ticks = heartbeat.get("worker_start_ticks")
        written = datetime.fromisoformat(
            str(heartbeat["heartbeat_at"]).replace("Z", "+00:00")
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    age = (datetime.now(timezone.utc) - written).total_seconds()
    if age < 0 or age > max_age_seconds or not isinstance(worker_pid, int):
        return None
    identity = _process_identity(worker_pid)
    if not identity.get("alive"):
        return None
    if (
        not isinstance(worker_start_ticks, int)
        or identity.get("start_ticks") != worker_start_ticks
    ):
        return None
    state = str(heartbeat.get("state", ""))
    if state in {"FAILED", "PASS", "BLOCKED", "STOPPED"}:
        return None
    return {
        "state": "ADOPTED_LIVE_OWNER",
        "heartbeat": str(path.resolve(strict=True)),
        "heartbeat_age_seconds": age,
        "worker_pid": worker_pid,
        "worker_start_ticks": worker_start_ticks,
        "worker_state": state,
    }


def _adopt_live_t14_relay(
    path_text: str | None,
    start_ticks_text: str | None,
    *,
    max_age_seconds: int = 180,
) -> dict[str, Any] | None:
    """Adopt the exact persistent T14 relay without launching a duplicate."""

    if not path_text or not start_ticks_text:
        return None
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        return None
    try:
        heartbeat = _load_json(path)
        controller_pid = heartbeat.get("controller_pid")
        expected_start_ticks = int(start_ticks_text)
        written = datetime.fromisoformat(
            str(heartbeat["written_at"]).replace("Z", "+00:00")
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        heartbeat.get("schema_version")
        != "tastemolnet_t14_external_convergence_relay_heartbeat_v1"
    ):
        return None
    age = (datetime.now(timezone.utc) - written).total_seconds()
    if not isinstance(controller_pid, int):
        return None
    identity = _process_identity(controller_pid)
    if (
        not identity.get("alive")
        or identity.get("start_ticks") != expected_start_ticks
    ):
        return None
    return {
        "state": (
            "ADOPTED_LIVE_RELAY"
            if 0 <= age <= max_age_seconds
            else "LIVE_RELAY_STALE_HEARTBEAT"
        ),
        "heartbeat": str(path.resolve(strict=True)),
        "heartbeat_age_seconds": age,
        "controller_pid": controller_pid,
        "controller_start_ticks": expected_start_ticks,
        "relay_phase": heartbeat.get("phase"),
        "audited_through_step": heartbeat.get("audited_through_step"),
        "converged": heartbeat.get("converged") is True,
    }


def _task_spec_t14_relay(project_root: Path) -> dict[str, Any] | None:
    """Read the conflicting full-state auditor only from the T14 task spec."""

    try:
        _, spec, _ = _load_task_spec(project_root, "t14_resume")
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
        return None
    auditor = spec.get("serial_auditor")
    if not isinstance(auditor, Mapping) or auditor.get("active") is not True:
        return None
    configured_pid = int(auditor["controller_pid"])
    configured_ticks = int(auditor["controller_start_ticks"])
    identity = _process_identity(configured_pid)
    if identity.get("alive") and identity.get("start_ticks") == configured_ticks:
        observed = _adopt_live_t14_relay(
            str(auditor.get("heartbeat_path") or ""),
            str(configured_ticks),
        )
        return observed or {
            "state": "LIVE_RELAY_UNREADABLE_HEARTBEAT",
            "controller_pid": configured_pid,
            "controller_start_ticks": configured_ticks,
            "heartbeat": str(auditor.get("heartbeat_path") or ""),
        }
    return _adopt_live_t14_relay(
        str(auditor.get("heartbeat_path") or ""),
        str(configured_ticks),
    )


def _t8_zero_attempt_receipt_blocker() -> dict[str, Any] | None:
    """Fail closed until the sole recovery attempt has an authoritative receipt."""

    configured = os.environ.get("TASTE_GLOBALGCE_ATTEMPT_RECEIPT")
    if not configured:
        return {
            "state": "BLOCKED_MISSING_AUTHORITATIVE_ATTEMPT_RECEIPT",
            "required_env": "TASTE_GLOBALGCE_ATTEMPT_RECEIPT",
        }
    path = Path(configured)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        return {
            "state": "BLOCKED_INVALID_AUTHORITATIVE_ATTEMPT_RECEIPT_PATH",
            "attempt_receipt": str(path),
        }
    return None


def _nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(dotted)
        current = current[component]
    return current


def _resolved_output_root(
    spec: Mapping[str, Any], *, attempt_uuid: str, attempt: int
) -> str:
    raw = str(spec["output_root"])
    resolved = _replace_attempt_tokens(
        raw, attempt_uuid=attempt_uuid, attempt=attempt
    )
    if not Path(resolved).is_absolute():
        raise ValueError("resolved output root must remain absolute")
    return resolved


def _resolved_environment(
    spec: Mapping[str, Any], *, attempt_uuid: str, attempt: int
) -> dict[str, str]:
    return {
        key: _replace_attempt_tokens(
            str(value), attempt_uuid=attempt_uuid, attempt=attempt
        )
        for key, value in spec["required_environment"].items()
    }


def _latest_attempt(state: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(state, Mapping):
        return None
    attempts = state.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    latest = attempts[-1]
    return latest if isinstance(latest, Mapping) else None


def _expected_owner_output(
    spec: Mapping[str, Any], state: Mapping[str, Any] | None
) -> str | None:
    attempt = _latest_attempt(state)
    if attempt is not None and isinstance(attempt.get("output_root"), str):
        return str(attempt["output_root"])
    owner = spec["owner"]
    configured = owner.get("expected_output_root")
    if isinstance(configured, str) and Path(configured).is_absolute():
        return configured
    raw = str(spec["output_root"])
    return None if "{attempt_" in raw else raw


def _command_sha256_from_argv(argv: list[str]) -> str:
    return hashlib.sha256(b"\0".join(item.encode("utf-8") for item in argv) + b"\0").hexdigest()


def _resolved_owner_command_sha256(
    spec: Mapping[str, Any],
    state: Mapping[str, Any] | None,
    *,
    attempt_uuid: str | None = None,
    attempt: int | None = None,
    output_root: str | None = None,
) -> str | None:
    latest = _latest_attempt(state)
    if latest is not None:
        frozen = latest.get("expected_owner_command_sha256")
        if isinstance(frozen, str):
            return frozen
    owner = spec["owner"]
    fixed = owner.get("command_sha256")
    if isinstance(fixed, str) and len(fixed) == 64:
        return fixed
    template = owner.get("command_argv")
    if (
        not isinstance(template, list)
        or any(not isinstance(item, str) or not item for item in template)
        or attempt_uuid is None
        or attempt is None
        or output_root is None
    ):
        return None
    resolved = [
        _replace_attempt_tokens(item, attempt_uuid=attempt_uuid, attempt=attempt).replace(
            "{output_root}", output_root
        )
        for item in template
    ]
    return _command_sha256_from_argv(resolved)


def _terminal_probe(
    spec: Mapping[str, Any], *, expected_output: str | None
) -> dict[str, Any] | None:
    contract = spec["terminal"]
    raw_path = contract.get("receipt_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("terminal.receipt_path is required")
    if expected_output is not None:
        raw_path = raw_path.replace("{output_root}", expected_output)
    if "{" in raw_path or "}" in raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("terminal receipt path must be absolute and non-symlink")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("terminal receipt must be a file")
    receipt = _load_json(path)
    expected_schema = contract.get("schema_version")
    if expected_schema and receipt.get("schema_version") != expected_schema:
        raise ValueError("terminal receipt schema changed")
    self_hash_field = contract.get("self_hash_field")
    if self_hash_field:
        claimed = receipt.get(self_hash_field)
        unsigned = {key: value for key, value in receipt.items() if key != self_hash_field}
        digest = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if claimed != digest:
            raise ValueError("terminal receipt self hash changed")
    state_field = str(contract.get("state_field", "state"))
    terminal_states = contract.get("terminal_states", ["PASS", "FAILED", "BLOCKED"])
    if (
        not isinstance(terminal_states, list)
        or _nested(receipt, state_field) not in terminal_states
    ):
        raise ValueError("terminal receipt does not contain an allowed terminal state")
    output_field = contract.get("output_root_field")
    if expected_output is not None and output_field:
        if _nested(receipt, str(output_field)) != expected_output:
            raise ValueError("terminal receipt output root changed")
    return {
        "state": "TERMINAL",
        "receipt": str(path.resolve(strict=True)),
        "receipt_sha256": _sha256_file(path),
        "terminal_state": _nested(receipt, state_field),
        "output_root": expected_output,
    }


def _owner_probe(
    spec: Mapping[str, Any],
    *,
    expected_output: str | None,
    expected_command_sha256: str | None,
    expected_start_ticks: int | None,
    now_epoch: float,
) -> dict[str, Any]:
    owner = spec["owner"]
    raw_path = owner.get("heartbeat_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("owner.heartbeat_path is required")
    if expected_output is not None:
        raw_path = raw_path.replace("{output_root}", expected_output)
    if "{" in raw_path or "}" in raw_path:
        return {"state": "ABSENT", "reason": "OWNER_OUTPUT_NOT_RESOLVED"}
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("owner heartbeat path must be absolute and non-symlink")
    if not path.exists():
        return {"state": "ABSENT", "heartbeat": str(path)}
    if not path.is_file():
        return {"state": "INVALID", "reason": "OWNER_HEARTBEAT_NOT_FILE"}
    try:
        heartbeat = _load_json(path)
        expected_schema = owner.get("heartbeat_schema")
        if expected_schema and heartbeat.get("schema_version") != expected_schema:
            raise ValueError("owner heartbeat schema changed")
        pid = _nested(heartbeat, str(owner.get("pid_field", "pid")))
        start_ticks_field = owner.get("start_ticks_field", "start_ticks")
        ticks = (
            _nested(heartbeat, str(start_ticks_field))
            if start_ticks_field is not None
            else None
        )
        output_root_field = owner.get("output_root_field", "output_root")
        if output_root_field is None:
            if owner.get("output_root_from_heartbeat_parent") is not True:
                raise ValueError("owner output-root derivation is not authorized")
            output = str(path.parent)
        else:
            output = _nested(heartbeat, str(output_root_field))
        heartbeat_at = _nested(
            heartbeat, str(owner.get("timestamp_field", "heartbeat_at"))
        )
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError("owner PID is invalid")
        if expected_output is None or output != expected_output:
            raise ValueError("owner output root changed")
        identity = _process_identity(pid)
        if not identity.get("alive"):
            return {
                "state": "ABSENT",
                "reason": "RECORDED_OWNER_PROCESS_EXITED",
                "heartbeat": str(path.resolve(strict=True)),
                "recorded_pid": pid,
            }
        if start_ticks_field is None:
            observed_ticks = identity.get("start_ticks")
            if (
                not isinstance(observed_ticks, int)
                or isinstance(observed_ticks, bool)
                or observed_ticks <= 0
            ):
                raise ValueError("owner process start ticks are unavailable")
            if expected_start_ticks is None:
                ticks = observed_ticks
            elif observed_ticks != expected_start_ticks:
                raise ValueError("owner PID start ticks changed")
            else:
                ticks = expected_start_ticks
        elif not isinstance(ticks, int) or isinstance(ticks, bool) or ticks <= 0:
            raise ValueError("owner start ticks are invalid")
        if identity.get("start_ticks") != ticks:
            raise ValueError("owner PID start ticks changed")
        expected_cwd = str(owner.get("cwd", spec["repo_root"]))
        if identity.get("cwd") != expected_cwd:
            raise ValueError("owner cwd changed")
        if not isinstance(expected_command_sha256, str) or len(expected_command_sha256) != 64:
            raise ValueError("resolved owner command SHA256 is unavailable")
        if identity.get("command_sha256") != expected_command_sha256:
            raise ValueError("owner command SHA256 changed")
        tokens = owner.get("command_contains", [])
        if not isinstance(tokens, list) or any(
            not isinstance(token, str) or not token for token in tokens
        ):
            raise ValueError("owner.command_contains must be a string list")
        if any(token not in str(identity.get("command", "")) for token in tokens):
            raise ValueError("owner command tokens changed")
        age = _seconds_since(heartbeat_at, now_epoch=now_epoch)
        max_age = owner.get("max_age_seconds", 120)
        if not isinstance(max_age, (int, float)) or isinstance(max_age, bool) or max_age <= 0:
            raise ValueError("owner.max_age_seconds must be positive")
        if age < -30 or age > float(max_age):
            raise ValueError(f"live owner heartbeat is stale: {age:.1f}s")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "reason": f"{type(exc).__name__}: {exc}",
            "heartbeat": str(path),
        }
    return {
        "state": (
            "OWNER_CANDIDATE"
            if start_ticks_field is None and expected_start_ticks is None
            else "OWNER_CONFIRMED"
        ),
        "heartbeat": str(path.resolve(strict=True)),
        "heartbeat_age_seconds": age,
        "science_pid": pid,
        "science_start_ticks": ticks,
        "science_command_sha256": expected_command_sha256,
        "output_root": expected_output,
    }


def _read_dispatch_state(component_root: Path) -> dict[str, Any] | None:
    path = component_root / "dispatch_state.json"
    if path.is_file() and not path.is_symlink():
        value = _load_json(path)
        if value.get("schema_version") != DISPATCH_SCHEMA:
            raise ValueError("dispatch state schema changed")
        return value
    legacy = component_root / "launch.json"
    if not legacy.is_file() or legacy.is_symlink():
        return None
    launch = _load_json(legacy)
    return {
        "schema_version": DISPATCH_SCHEMA,
        "component": launch.get("component"),
        "state": "LAUNCHING",
        "attempt_count": 1,
        "attempts": [
            {
                "attempt": 1,
                "attempt_uuid": "legacy-v1-launch",
                "launcher_pid": launch.get("launcher_pid"),
                "launcher_start_ticks": launch.get("launcher_start_ticks"),
                "started_at": launch.get("dispatched_at", _utc_now()),
                "output_root": None,
                "legacy_launch_receipt": str(legacy),
            }
        ],
        "migrated_from_v1": True,
        "updated_at": _utc_now(),
    }


def _writer_pids(output_root: str) -> list[int]:
    needle = output_root.encode("utf-8")
    matches: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if needle in command:
            matches.append(int(entry.name))
    return sorted(matches)


def _resource_blocker(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    request = spec["memory_request"]
    required = int(request["required_headroom_bytes"])
    if required:
        limit_path = _physical_path(
            request.get("limit_path"), label="memory_request.limit_path", kind="file"
        )
        current_path = _physical_path(
            request.get("current_path"),
            label="memory_request.current_path",
            kind="file",
        )
        limit = int(limit_path.read_text(encoding="utf-8").strip())
        current = int(current_path.read_text(encoding="utf-8").strip())
        headroom = limit - current
        if headroom < required:
            return {
                "state": "WAITING_MEMORY_HEADROOM",
                "required_headroom_bytes": required,
                "observed_headroom_bytes": headroom,
            }
    gpu = spec["gpu_request"]
    lease_text = gpu.get("lease_path")
    if gpu.get("required") and lease_text:
        lease = Path(str(lease_text))
        if not lease.is_absolute() or lease.is_symlink():
            raise ValueError("gpu_request.lease_path must be absolute and non-symlink")
        if lease.exists():
            if not lease.is_file():
                raise ValueError("gpu lease path must be a file")
            handle = lease.open("r", encoding="utf-8")
            try:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return {
                        "state": "WAITING_GPU_LEASE",
                        "gpu_index": gpu.get("index"),
                        "lease_path": str(lease),
                    }
                finally:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
            finally:
                handle.close()
    return None


def _clean_launcher_environment(required: Mapping[str, str]) -> dict[str, str]:
    allowed = ("HOME", "PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "USER", "LOGNAME", "SHELL")
    environment = {
        key: value for key in allowed if isinstance((value := os.environ.get(key)), str)
    }
    environment.update(required)
    return environment


def _run_launch_supervisor(resolved_spec_path: Path, exit_receipt: Path) -> int:
    resolved = _load_json(resolved_spec_path)
    if resolved.get("schema_version") != RESOLVED_TASK_SPEC_SCHEMA:
        raise ValueError("resolved launch spec schema changed")
    launcher = _physical_path(
        resolved.get("launcher"), label="resolved launcher", kind="file"
    )
    repo_root = _physical_path(
        resolved.get("repo_root"), label="resolved repo root", kind="dir"
    )
    environment = resolved.get("required_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("resolved required_environment must be an object")
    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=repo_root,
        env=_clean_launcher_environment(
            {str(key): str(value) for key, value in environment.items()}
        ),
        check=False,
    )
    _atomic_json(
        exit_receipt,
        {
            "schema_version": LAUNCH_EXIT_SCHEMA,
            "returncode": completed.returncode,
            "resolved_task_spec": str(resolved_spec_path),
            "resolved_task_spec_sha256": _sha256_file(resolved_spec_path),
            "written_at": _utc_now(),
        },
    )
    return int(completed.returncode)


def _dispatch(
    state_root: Path,
    project_root: Path,
    component: str,
    *,
    reason: str,
    dry_run: bool,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now_epoch is None else float(now_epoch)
    try:
        spec_path, spec, spec_sha = _load_task_spec(project_root, component)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        return {
            "state": "DISPATCH_CONFIG_INVALID",
            "error": f"{type(exc).__name__}: {exc}",
            "task_spec_env": TASK_SPEC_ENV[component],
        }
    component_root = state_root / "components" / component
    try:
        state = _read_dispatch_state(component_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"state": "DISPATCH_CONFIG_INVALID", "error": str(exc)}
    if state is not None and state.get("task_spec_sha256") not in {None, spec_sha}:
        return {
            "state": "DISPATCH_CONFIG_INVALID",
            "error": "task spec changed after first dispatch",
        }
    expected_output = _expected_owner_output(spec, state)
    expected_owner_command_sha256 = _resolved_owner_command_sha256(
        spec, state
    )
    latest_before_probe = _latest_attempt(state)
    expected_start_ticks = (
        latest_before_probe.get("observed_science_start_ticks")
        if latest_before_probe is not None
        and isinstance(latest_before_probe.get("observed_science_start_ticks"), int)
        else None
    )
    try:
        terminal = _terminal_probe(spec, expected_output=expected_output)
        owner = _owner_probe(
            spec,
            expected_output=expected_output,
            expected_command_sha256=expected_owner_command_sha256,
            expected_start_ticks=expected_start_ticks,
            now_epoch=now,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"state": "DISPATCH_CONFIG_INVALID", "error": str(exc)}
    if terminal is not None:
        if not dry_run:
            terminal_state = {
                "schema_version": DISPATCH_SCHEMA,
                "component": component,
                "task_id": spec["task_id"],
                "task_spec_path": str(spec_path),
                "task_spec_sha256": spec_sha,
                "state": "TERMINAL",
                "attempt_count": int((state or {}).get("attempt_count", 0)),
                "attempts": list((state or {}).get("attempts", [])),
                "terminal": terminal,
                "updated_at": _utc_now(),
            }
            _atomic_json(component_root / "dispatch_state.json", terminal_state)
        return terminal
    if owner["state"] == "OWNER_CONFIRMED":
        confirmed = {
            "schema_version": DISPATCH_SCHEMA,
            "component": component,
            "task_id": spec["task_id"],
            "task_spec_path": str(spec_path),
            "task_spec_sha256": spec_sha,
            "state": "OWNER_CONFIRMED",
            "attempt_count": int((state or {}).get("attempt_count", 0)),
            "attempts": list((state or {}).get("attempts", [])),
            "owner": owner,
            "updated_at": _utc_now(),
        }
        if not dry_run:
            _atomic_json(component_root / "dispatch_state.json", confirmed)
        return owner
    if owner["state"] == "OWNER_CANDIDATE":
        if state is None or _latest_attempt(state) is None:
            return {
                "state": "OWNER_EVIDENCE_INVALID",
                "owner": owner,
                "reason": "unbound owner candidate has no dispatch attempt",
            }
        candidate_state = dict(state)
        candidate_attempts = list(candidate_state.get("attempts", []))
        candidate_latest = dict(candidate_attempts[-1])
        candidate_latest["observed_science_start_ticks"] = owner[
            "science_start_ticks"
        ]
        candidate_latest["owner_candidate_observed_at"] = _utc_now()
        candidate_attempts[-1] = candidate_latest
        candidate_state["attempts"] = candidate_attempts
        candidate_state["state"] = "LAUNCHING"
        candidate_state["updated_at"] = _utc_now()
        if not dry_run:
            _atomic_json(component_root / "dispatch_state.json", candidate_state)
        return {
            "state": "LAUNCHING",
            "owner_confirmation_pending": True,
            "owner_candidate": owner,
        }
    if owner["state"] == "INVALID":
        return {"state": "OWNER_EVIDENCE_INVALID", "owner": owner}

    if state is None:
        state = {
            "schema_version": DISPATCH_SCHEMA,
            "component": component,
            "task_id": spec["task_id"],
            "task_spec_path": str(spec_path),
            "task_spec_sha256": spec_sha,
            "state": "READY",
            "attempt_count": 0,
            "attempts": [],
            "updated_at": _utc_now(),
        }
    else:
        state["task_id"] = spec["task_id"]
        state["task_spec_path"] = str(spec_path)
        state["task_spec_sha256"] = spec_sha

    latest = _latest_attempt(state)
    if state.get("state") in {"LAUNCHING", "OWNER_CONFIRMED"} and latest is not None:
        launcher_pid = latest.get("launcher_pid")
        launcher_alive = (
            isinstance(launcher_pid, int)
            and _process_identity(launcher_pid).get("alive") is True
        )
        age = _seconds_since(latest.get("started_at"), now_epoch=now)
        if launcher_alive:
            return {
                "state": "LAUNCHING",
                "attempt": dict(latest),
                "owner_confirmation_pending": True,
            }
        if age < LAUNCH_GRACE_SECONDS:
            return {
                "state": "LAUNCHING",
                "attempt": dict(latest),
                "owner_confirmation_pending": True,
                "launch_grace_remaining_seconds": LAUNCH_GRACE_SECONDS - age,
            }
        exit_path_text = latest.get("exit_receipt")
        exit_value = None
        if isinstance(exit_path_text, str) and Path(exit_path_text).is_file():
            exit_value = _load_json(Path(exit_path_text))
        failure_reason = "launcher exited without a confirmed science owner"
        if isinstance(exit_value, Mapping):
            failure_reason += f"; returncode={exit_value.get('returncode')}"
        state["state"] = "FAILED_TO_START"
        state["failure_reason"] = failure_reason
        attempt_number = int(state.get("attempt_count", 0))
        backoff = RETRY_BACKOFF_SECONDS[max(0, min(attempt_number, MAX_LAUNCH_ATTEMPTS) - 1)]
        state["next_retry_at_unix"] = now + backoff
        state["next_retry_at"] = datetime.fromtimestamp(
            now + backoff, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z")
        state["updated_at"] = _utc_now()
        if not dry_run:
            _atomic_json(component_root / "dispatch_state.json", state)
        return {
            "state": "FAILED_TO_START",
            "attempt_count": attempt_number,
            "failure_reason": failure_reason,
            "retry_after_seconds": backoff,
        }

    if state.get("state") in {"FAILED_TO_START", "BACKOFF"}:
        retry_at = float(state.get("next_retry_at_unix", now))
        if now < retry_at:
            state["state"] = "BACKOFF"
            state["updated_at"] = _utc_now()
            if not dry_run:
                _atomic_json(component_root / "dispatch_state.json", state)
            return {
                "state": "BACKOFF",
                "attempt_count": int(state.get("attempt_count", 0)),
                "retry_in_seconds": retry_at - now,
                "failure_reason": state.get("failure_reason"),
            }
        if int(state.get("attempt_count", 0)) >= MAX_LAUNCH_ATTEMPTS:
            state["state"] = "TERMINAL"
            state["terminal_reason"] = "BLOCKED_LAUNCHER_RETRY_EXHAUSTED"
            state["updated_at"] = _utc_now()
            if not dry_run:
                _atomic_json(component_root / "dispatch_state.json", state)
            return {
                "state": "BLOCKED_LAUNCHER_RETRY_EXHAUSTED",
                "attempt_count": int(state.get("attempt_count", 0)),
            }
        state["state"] = "READY"

    next_attempt = int(state.get("attempt_count", 0)) + 1
    prior_writers: dict[str, list[int]] = {}
    for prior_attempt in state.get("attempts", []):
        if not isinstance(prior_attempt, Mapping):
            continue
        prior_output = prior_attempt.get("output_root")
        if not isinstance(prior_output, str):
            continue
        writers = _writer_pids(prior_output)
        if writers:
            prior_writers[prior_output] = writers
    if prior_writers:
        return {
            "state": "BLOCKED_DUPLICATE_TASK_OWNER",
            "task_id": spec["task_id"],
            "prior_output_writers": prior_writers,
        }
    prior_attempt_uuids = {
        str(row.get("attempt_uuid"))
        for row in state.get("attempts", [])
        if isinstance(row, Mapping) and isinstance(row.get("attempt_uuid"), str)
    }
    attempt_uuid: str | None = None
    for _ in range(8):
        candidate_uuid = str(uuid4())
        if candidate_uuid not in prior_attempt_uuids:
            attempt_uuid = candidate_uuid
            break
    if attempt_uuid is None:
        return {
            "state": "DISPATCH_CONFIG_INVALID",
            "error": "could not allocate a unique attempt UUID",
        }
    output_root = _resolved_output_root(
        spec, attempt_uuid=attempt_uuid, attempt=next_attempt
    )
    expected_owner_command_sha256 = _resolved_owner_command_sha256(
        spec,
        state,
        attempt_uuid=attempt_uuid,
        attempt=next_attempt,
        output_root=output_root,
    )
    if expected_owner_command_sha256 is None:
        return {
            "state": "DISPATCH_CONFIG_INVALID",
            "error": "owner command SHA256 cannot be resolved for this attempt",
        }
    output_writers = _writer_pids(output_root)
    if output_writers:
        return {
            "state": "BLOCKED_DUPLICATE_OUTPUT_WRITER",
            "output_root": output_root,
            "writer_pids": output_writers,
        }
    output_path = Path(output_root)
    if spec.get("output_mode", "fresh") == "fresh" and (
        output_path.exists() or output_path.is_symlink()
    ):
        return {
            "state": "DISPATCH_CONFIG_INVALID",
            "error": "fresh resolved output root already exists",
            "output_root": output_root,
        }
    if spec.get("output_mode") == "resume" and (
        not output_path.is_dir() or output_path.is_symlink()
    ):
        return {
            "state": "DISPATCH_CONFIG_INVALID",
            "error": "resume resolved output root must be an existing directory",
            "output_root": output_root,
        }
    try:
        blocker = _resource_blocker(spec)
    except (OSError, ValueError) as exc:
        return {"state": "DISPATCH_CONFIG_INVALID", "error": str(exc)}
    if blocker is not None:
        return blocker
    launcher = _resolve_launcher(project_root, component, spec)
    if dry_run:
        return {
            "state": "WOULD_DISPATCH",
            "launcher": str(launcher.resolve(strict=True)),
            "reason": reason,
            "attempt": next_attempt,
            "output_root": output_root,
        }
    component_root.mkdir(parents=True, exist_ok=True)
    attempt_root = component_root / "attempts" / f"attempt-{next_attempt:02d}-{attempt_uuid}"
    attempt_root.mkdir(parents=True)
    resolved_environment = _resolved_environment(
        spec, attempt_uuid=attempt_uuid, attempt=next_attempt
    )
    resolved_environment.setdefault("AUTODL_PYTHON", str(spec["python"]))
    resolved = {
        "schema_version": RESOLVED_TASK_SPEC_SCHEMA,
        "component": component,
        "task_id": spec["task_id"],
        "attempt": next_attempt,
        "attempt_uuid": attempt_uuid,
        "repo_root": spec["repo_root"],
        "execution_commit": spec["execution_commit"],
        "launcher": str(launcher.resolve(strict=True)),
        "output_root": output_root,
        "required_environment": resolved_environment,
        "source_task_spec": str(spec_path),
        "source_task_spec_sha256": spec_sha,
    }
    resolved_path = attempt_root / "resolved_task_spec.json"
    exit_receipt = attempt_root / "launcher_exit.json"
    log_path = attempt_root / "launcher.log"
    _atomic_json(resolved_path, resolved)
    log_handle = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            [
                str(spec["python"]),
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--internal-launch-spec",
                str(resolved_path),
                "--internal-launch-exit-receipt",
                str(exit_receipt),
            ],
            cwd=project_root,
            env=_clean_launcher_environment({}),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    attempt_receipt = {
        "schema_version": "main_and_ablations_component_launch_v2",
        "component": component,
        "task_id": spec["task_id"],
        "attempt": next_attempt,
        "attempt_uuid": attempt_uuid,
        "launcher": str(launcher.resolve(strict=True)),
        "launcher_pid": process.pid,
        "launcher_start_ticks": _process_identity(process.pid).get("start_ticks"),
        "exit_receipt": str(exit_receipt),
        "resolved_task_spec": str(resolved_path),
        "resolved_task_spec_sha256": _sha256_file(resolved_path),
        "log_path": str(log_path),
        "output_root": output_root,
        "expected_owner_command_sha256": expected_owner_command_sha256,
        "reason": reason,
        "started_at": _utc_now(),
    }
    state["state"] = "LAUNCHING"
    state["attempt_count"] = next_attempt
    state.setdefault("attempts", []).append(attempt_receipt)
    state["updated_at"] = _utc_now()
    _atomic_json(component_root / "launch.json", attempt_receipt)
    _atomic_json(component_root / "dispatch_state.json", state)
    return {"state": "LAUNCHING", "attempt": attempt_receipt}


def _explicit_pass(path_text: str | None) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        return False
    try:
        value = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return value.get("status") == "PASS" or value.get("state") == "PASS"


def _main_ready_waiting(matrix: Mapping[str, Any]) -> tuple[bool, str]:
    configured = os.environ.get("MAIN_READY_QUEUE")
    if configured:
        path = Path(configured)
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            return True, "MAIN_READY_QUEUE_UNAVAILABLE"
        value = _load_json(path)
        rows = value.get("ready_waiting_gpu", value.get("tasks", []))
        if isinstance(rows, int):
            return rows > 0, f"ready_waiting_gpu={rows}"
        if isinstance(rows, list):
            waiting = [
                row
                for row in rows
                if not isinstance(row, Mapping)
                or row.get("state") == "READY_WAITING_GPU"
            ]
            return bool(waiting), f"ready_waiting_gpu={len(waiting)}"
        return True, "MAIN_READY_QUEUE_SCHEMA_UNKNOWN"

    owner_manifest = os.environ.get("MAIN_OWNER_MANIFEST")
    if not owner_manifest:
        return True, "MAIN_READY_QUEUE_AND_OWNER_MANIFEST_UNBOUND"
    path = Path(owner_manifest)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        return True, "MAIN_OWNER_MANIFEST_UNAVAILABLE"
    value = _load_json(path)
    if value.get("schema_version") != "main_live_owner_manifest_v1":
        return True, "MAIN_OWNER_MANIFEST_SCHEMA_UNKNOWN"
    rows = value.get("owners")
    if not isinstance(rows, list):
        return True, "MAIN_OWNER_MANIFEST_ROWS_INVALID"
    owners: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return True, "MAIN_OWNER_MANIFEST_ROW_INVALID"
        cell = row.get("cell")
        if not isinstance(cell, str) or cell in owners:
            return True, "MAIN_OWNER_MANIFEST_CELL_INVALID"
        owners[cell] = row
    missing_without_owner: list[str] = []
    for cell in (
        "Mutagenicity/ComRecGC",
        "TasteMolNet/GCFExplainer",
        "TasteMolNet/GlobalGCE",
        "TasteMolNet/ComRecGC",
    ):
        dataset, method = cell.split("/", 1)
        if _cell_present(matrix, dataset, method):
            continue
        row = owners.get(cell)
        if row is None:
            missing_without_owner.append(cell)
            continue
        pid = row.get("pid")
        ticks = row.get("start_ticks")
        if not isinstance(pid, int) or not isinstance(ticks, int):
            missing_without_owner.append(cell)
            continue
        identity = _process_identity(pid)
        if not identity.get("alive") or identity.get("start_ticks") != ticks:
            missing_without_owner.append(cell)
    return (
        bool(missing_without_owner),
        "main_missing_cells_without_live_owner="
        + (",".join(missing_without_owner) if missing_without_owner else "0"),
    )


def observe_and_dispatch(
    *,
    state_root: Path,
    project_root: Path,
    matrix_path: Path,
    policy: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    matrix = _matrix(matrix_path)
    main_ready, main_ready_reason = _main_ready_waiting(matrix)
    components: dict[str, Any] = {}

    if _cell_present(matrix, "Mutagenicity", "ComRecGC"):
        components["mut_continuation"] = {"state": "MAIN_CELL_PASS"}
    else:
        live_mut = _adopt_live_mut_owner(
            os.environ.get("MUT_CONTINUATION_HEARTBEAT")
        )
        components["mut_continuation"] = live_mut or _dispatch(
            state_root,
            project_root,
            "mut_continuation",
            reason="P0_MUT_CELL_MISSING_NO_LIVE_OWNER",
            dry_run=dry_run,
        )
    if _cell_present(matrix, "TasteMolNet", "ComRecGC"):
        components["t14_resume"] = {"state": "MAIN_CELL_PASS"}
        components["t14_convergence_auditor"] = {"state": "MAIN_CELL_PASS"}
    else:
        live_t14_relay = _task_spec_t14_relay(project_root) or _adopt_live_t14_relay(
            os.environ.get("T14_AUDITOR_RELAY_HEARTBEAT"),
            os.environ.get("T14_AUDITOR_RELAY_START_TICKS"),
        )
        components["t14_convergence_auditor"] = live_t14_relay or {
            "state": "NOT_RUNNING_SERIAL_ONLY"
        }
        if live_t14_relay is not None:
            # The retained relay can deserialize the same very large checkpoint
            # as T14 science.  It is diagnostic evidence, never a science owner,
            # and SERIAL_ONLY forbids launching the resume while it is live.
            components["t14_resume"] = {
                "state": "BLOCKED_SERIAL_AUDITOR_ACTIVE",
                "auditor": live_t14_relay,
            }
        else:
            components["t14_resume"] = _dispatch(
                state_root,
                project_root,
                "t14_resume",
                reason="P0_TASTE_COMRECGC_CELL_MISSING_NO_SCIENCE_OWNER",
                dry_run=dry_run,
            )
    if _cell_present(matrix, "TasteMolNet", "GlobalGCE"):
        components["t8_valid_zero_finalizer"] = {"state": "MAIN_CELL_PASS"}
    else:
        components["t8_valid_zero_finalizer"] = (
            _t8_zero_attempt_receipt_blocker()
            or _dispatch(
                state_root,
                project_root,
                "t8_valid_zero_finalizer",
                reason="P0_TASTE_GLOBALGCE_CELL_MISSING",
                dry_run=dry_run,
            )
        )

    llm_blockers: list[str] = []
    if matrix["count"] < 13:
        llm_blockers.append("MATRIX_LT_13")
    if not _cell_present(matrix, "Mutagenicity", "ComRecGC"):
        llm_blockers.append("MUT_STILL_REQUIRES_PRIORITY")
    if main_ready:
        llm_blockers.append(main_ready_reason)
    llm_gate_receipt = os.environ.get("LLM_EARLY_GATE_RECEIPT")
    if not _explicit_pass(llm_gate_receipt):
        llm_blockers.append("LLM_EARLY_GATE_RECEIPT_NOT_PASS")
    if llm_blockers:
        components["llm_ablation"] = {
            "state": "BLOCKED_MAIN_PRIORITY",
            "blockers": llm_blockers,
        }
    else:
        components["llm_ablation"] = _dispatch(
            state_root,
            project_root,
            "llm_ablation",
            reason="P1_EARLY_GATE_PASS",
            dry_run=dry_run,
        )

    gnn_blockers: list[str] = []
    if matrix["count"] != 16:
        gnn_blockers.append("MATRIX_NOT_16")
    for label, variable in (
        ("FINAL_MATRIX_AUDIT", "FINAL_MATRIX_AUDIT_RECEIPT"),
        ("FINAL_FIGURE3", "FINAL_FIGURE3_RECEIPT"),
        ("FINAL_FIGURE4", "FINAL_FIGURE4_RECEIPT"),
        ("FINAL_TABLE2", "FINAL_TABLE2_RECEIPT"),
    ):
        if not _explicit_pass(os.environ.get(variable)):
            gnn_blockers.append(f"{label}_NOT_PASS")
    if gnn_blockers:
        components["gnn_ablation"] = {
            "state": "BLOCKED_WAITING_FINAL_MAIN",
            "blockers": gnn_blockers,
        }
    else:
        components["gnn_ablation"] = _dispatch(
            state_root,
            project_root,
            "gnn_ablation",
            reason="P2_MAIN_16_AND_FINAL_ARTIFACTS_PASS",
            dry_run=dry_run,
        )

    return {
        "schema_version": HEARTBEAT_SCHEMA,
        "written_at": _utc_now(),
        "controller_pid": os.getpid(),
        "matrix_authority": str(matrix_path),
        "matrix_complete_cells": matrix["count"],
        "matrix_total_cells": 16,
        "matrix_cells": list(matrix["cells"]),
        "priorities": dict(policy),
        "main_ready_waiting_gpu": main_ready,
        "main_ready_queue_reason": main_ready_reason,
        "components": components,
        "dry_run": dry_run,
    }


def run(args: argparse.Namespace) -> int:
    policy = validate_policy()
    project_root = Path(__file__).resolve().parents[2]
    expected_config = project_root / "configs/hpc.yaml"
    if args.config.resolve(strict=True) != expected_config.resolve(strict=True):
        raise ValueError("--config must bind this checkout's configs/hpc.yaml")
    state_root = args.state_root.absolute()
    matrix_path = args.matrix_authority.absolute()
    if args.poll_seconds != DEFAULT_POLL_SECONDS:
        raise ValueError("scheduler poll interval must remain 30 seconds")
    state_root.mkdir(parents=True, exist_ok=True)
    lock = (state_root / "controller.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another main-and-ablations controller owns this root") from exc
    controller_id = os.environ.get(
        "MAIN_AND_ABLATIONS_CONTROLLER_ID", f"main_and_ablations_v1_{uuid4()}"
    )
    receipt_path = state_root / "controller_receipt.json"
    if not receipt_path.exists():
        _atomic_json(
            receipt_path,
            {
                "schema_version": SCHEMA,
                "controller_id": controller_id,
                "controller_pid": os.getpid(),
                "execution_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "created_at": _utc_now(),
                "matrix_authority": str(matrix_path),
                "policy": policy,
            },
        )
    else:
        controller_id = str(_load_json(receipt_path)["controller_id"])

    stopped = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    sequence = 0
    try:
        while True:
            sequence += 1
            payload = observe_and_dispatch(
                state_root=state_root,
                project_root=project_root,
                matrix_path=matrix_path,
                policy=policy,
                dry_run=args.dry_run,
            )
            payload.update({"controller_id": controller_id, "sequence": sequence})
            _atomic_json(state_root / "heartbeat.json", payload)
            if args.once or stopped:
                return 0
            time.sleep(args.poll_seconds)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--matrix-authority", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--internal-launch-spec" in raw_argv:
            internal = argparse.ArgumentParser(add_help=False)
            internal.add_argument("--internal-launch-spec", type=Path, required=True)
            internal.add_argument(
                "--internal-launch-exit-receipt", type=Path, required=True
            )
            parsed = internal.parse_args(raw_argv)
            return _run_launch_supervisor(
                parsed.internal_launch_spec.absolute(),
                parsed.internal_launch_exit_receipt.absolute(),
            )
        return run(build_parser().parse_args(raw_argv))
    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"[MAIN_AND_ABLATIONS_BLOCKED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
