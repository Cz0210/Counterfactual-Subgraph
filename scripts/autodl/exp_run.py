#!/usr/bin/env python3
"""Launch and register one detached AutoDL frozen-GNN experiment."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    BACE_STAGES,
    GPUFileLock,
    ProjectGPUSlotLock,
    StageTransitionError,
    append_jsonl_locked,
    assert_bace_stage_can_start,
    assert_tastemolnet_launch_allowed,
    atomic_write_json,
    build_runtime_layout,
    initialize_bace_stage_tree,
    latest_registry_events,
    query_gpu_inventory,
    read_registry,
    resolve_passed_bace_stage_output,
    resolve_project_root,
    sanitized_environment,
    select_data_root,
    sha256_file,
    sha256_paths,
    stage_paths,
    update_bace_stage_state,
    utc_now,
    validate_max_gpus,
    verify_required_outputs,
    mark_bace_stage_pass,
)


SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")
_SECRET = re.compile(
    r"(?i)(password|passwd|secret|token|authorization|api[_-]?key|"
    r"credential|private[_-]?key)"
)


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _safe_id(value: str, *, label: str) -> str:
    normalized = _SAFE_ID.sub("-", value).strip(".-_")
    if not normalized:
        raise AutoDLRuntimeError(f"{label} is empty after normalization")
    return normalized[:120]


def _new_run_id(dataset: str, stage: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _safe_id(f"{timestamp}-{dataset}-{stage}-{os.getpid()}", label="run_id")


def _parse_environment(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise AutoDLRuntimeError(f"--env requires KEY=VALUE, got {value!r}")
        key, content = value.split("=", 1)
        if not key or _SECRET.search(key):
            raise AutoDLRuntimeError(f"Unsafe environment key: {key!r}")
        result[key] = content
    return result


def _assert_command_safe(command: Sequence[str]) -> None:
    if not command:
        raise AutoDLRuntimeError("A scientific command is required after --")
    for value in command:
        if _SECRET.search(value):
            raise AutoDLRuntimeError(
                "Scientific command contains a credential-like option; use a "
                "credential provider outside experiment manifests"
            )


def _registry_event(spec: Mapping[str, Any], *, state: str, exit_code: int | None, pid: int | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": spec["run_id"],
        "timestamp": utc_now(),
        "pid": pid,
        "tmux_session": spec.get("tmux_session"),
        "command": spec["command"],
        "dataset": spec["dataset"],
        "stage": spec["stage"],
        "gpu_index": spec.get("gpu_index"),
        "gpu_uuid": spec.get("gpu_uuid"),
        "git_commit": spec.get("git_commit"),
        "config_hash": spec.get("config_hash"),
        "input_hash": spec.get("input_hash"),
        "expected_output": spec.get("expected_output"),
        "state": state,
        "exit_code": exit_code,
        "backend": "autodl",
        "slurm_job_id": None,
        "log_path": spec["log_path"],
    }


def _generic_stage_paths(layout: Any, dataset: str, stage: str) -> dict[str, Path]:
    root = layout.control_root / _safe_id(dataset, label="dataset") / "stages" / _safe_id(stage, label="stage")
    return {
        "root": root,
        "state": root / "state.json",
        "manifest": root / "manifest.json",
        "gate": root / "gate.json",
    }


def _scientific_stage_paths(layout: Any, dataset: str, stage: str) -> dict[str, Path]:
    if dataset == "bace" and stage in BACE_STAGES:
        return stage_paths(layout, stage)
    return _generic_stage_paths(layout, dataset, stage)


def _write_stage_manifest(layout: Any, spec: Mapping[str, Any]) -> None:
    paths = _scientific_stage_paths(layout, str(spec["dataset"]), str(spec["stage"]))
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        paths["manifest"],
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": spec["dataset"],
            "stage": spec["stage"],
            "status": "FROZEN",
            "run_id": spec["run_id"],
            "command": spec["command"],
            "config_files": spec["config_files"],
            "config_hash": spec["config_hash"],
            "input_manifest": spec["input_manifest"],
            "input_hash": spec["input_hash"],
            "expected_output": spec["expected_output"],
            "required_output_files": spec["required_output_files"],
            "required_log_marker": spec["required_log_marker"],
            "git_commit": spec["git_commit"],
            "gpu_index": spec["gpu_index"],
            "gpu_uuid": spec["gpu_uuid"],
            "created_at": spec["created_at"],
        },
    )


def _write_stage_state(
    layout: Any,
    spec: Mapping[str, Any],
    state: str,
    **fields: Any,
) -> None:
    dataset = str(spec["dataset"])
    stage = str(spec["stage"])
    if dataset == "bace" and stage in BACE_STAGES:
        update_bace_stage_state(layout, stage, state, **fields)
        return
    paths = _scientific_stage_paths(layout, dataset, stage)
    paths["root"].mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if paths["state"].is_file():
        try:
            value = json.loads(paths["state"].read_text(encoding="utf-8"))
            if isinstance(value, dict):
                current = value
        except json.JSONDecodeError:
            current = {}
    current.update(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "stage": stage,
            "state": state,
            "updated_at": utc_now(),
            **fields,
        }
    )
    atomic_write_json(paths["state"], current)


def _write_stage_gate(
    layout: Any,
    spec: Mapping[str, Any],
    *,
    status: str,
    failures: Sequence[str],
) -> None:
    paths = _scientific_stage_paths(layout, str(spec["dataset"]), str(spec["stage"]))
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        paths["gate"],
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": spec["dataset"],
            "stage": spec["stage"],
            "run_id": spec["run_id"],
            "status": status,
            "checked_at": utc_now(),
            "evidence": {
                "log_path": spec["log_path"],
                "required_log_marker": spec["required_log_marker"],
                "expected_output": spec["expected_output"],
                "required_output_files": spec["required_output_files"],
            },
            "failures": list(failures),
        },
    )


def _write_run_state(layout: Any, spec: Mapping[str, Any], state: str, **fields: Any) -> None:
    path = layout.runs_root / str(spec["run_id"]) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": spec["run_id"],
            "dataset": spec["dataset"],
            "stage": spec["stage"],
            "state": state,
            "updated_at": utc_now(),
            "pid": fields.pop("pid", None),
            "child_pid": fields.pop("child_pid", None),
            "tmux_session": spec.get("tmux_session"),
            "gpu_index": spec.get("gpu_index"),
            "gpu_uuid": spec.get("gpu_uuid"),
            "log_path": spec["log_path"],
            **fields,
        },
    )


def _assert_gpu_identity(spec: Mapping[str, Any]) -> None:
    if spec.get("gpu_index") is None:
        return
    observations = query_gpu_inventory()
    matches = [gpu for gpu in observations if gpu.index == int(spec["gpu_index"])]
    if len(matches) != 1 or matches[0].uuid != spec.get("gpu_uuid"):
        raise AutoDLRuntimeError(
            "Assigned physical GPU index/UUID no longer match nvidia-smi"
        )
    gpu = matches[0]
    if not gpu.is_idle(
        min_free_memory_mb=int(spec["min_free_memory_mb"]),
        max_utilization_percent=int(spec["idle_util_threshold"]),
    ):
        raise AutoDLRuntimeError(
            f"Assigned GPU {gpu.index} ({gpu.uuid}) became busy before worker start"
        )


def _validate_success(spec: Mapping[str, Any], *, exit_code: int) -> list[str]:
    failures: list[str] = []
    if exit_code != 0:
        failures.append(f"scientific command exited {exit_code}")
    log_path = Path(str(spec["log_path"]))
    marker = spec.get("required_log_marker")
    if marker:
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        if str(marker) not in log_text:
            failures.append(f"required log marker is absent: {marker}")
    expected = spec.get("expected_output")
    required = [str(value) for value in spec.get("required_output_files", [])]
    if expected and required:
        failures.extend(verify_required_outputs(Path(str(expected)), required))
    return failures


def run_worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != SCHEMA_VERSION:
        print(f"Invalid launch spec: {spec_path}", file=sys.stderr)
        return 2
    if not isinstance(spec.get("python_executable"), str) or not isinstance(
        spec.get("control_root"), str
    ):
        print(
            "Launch spec omits the frozen Python executable or control root",
            file=sys.stderr,
        )
        return 2
    expected_python = Path(str(spec["python_executable"])).resolve(strict=True)
    current_python = Path(sys.executable).resolve(strict=True)
    if current_python != expected_python:
        print(
            "Detached worker interpreter mismatch: "
            f"expected={expected_python}, current={current_python}",
            file=sys.stderr,
        )
        return 2
    project_root = Path(str(spec["project_root"])).resolve(strict=True)
    data_root = Path(str(spec["data_root"])).resolve(strict=True)
    control_root = Path(str(spec["control_root"]))
    layout = build_runtime_layout(
        project_root=project_root,
        data_root=data_root,
        control_root=control_root,
    ).ensure()
    environment = sanitized_environment()
    environment.update({str(key): str(value) for key, value in spec.get("environment", {}).items()})
    environment["PYTHONPATH"] = str(project_root) + (
        f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""
    )
    lock_context: Any = nullcontext()
    slot_context: Any = nullcontext()
    if spec.get("gpu_index") is not None:
        slot_context = ProjectGPUSlotLock(
            layout.locks_dir,
            max_slots=int(spec["max_gpus"]),
            owner={"run_id": spec["run_id"], "stage": spec["stage"]},
        )
        lock_context = GPUFileLock(
            layout.locks_dir,
            gpu_index=int(spec["gpu_index"]),
            gpu_uuid=str(spec["gpu_uuid"]),
            owner={"run_id": spec["run_id"], "stage": spec["stage"]},
        )
        environment["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_index"])
        environment["AUTODL_PHYSICAL_GPU_INDEX"] = str(spec["gpu_index"])
        environment["AUTODL_PHYSICAL_GPU_UUID"] = str(spec["gpu_uuid"])

    exit_code = 125
    failures: list[str] = []
    child_pid: int | None = None
    try:
        with slot_context, lock_context:
            _assert_gpu_identity(spec)
            _write_stage_state(layout, spec, "RUNNING", run_id=spec["run_id"], pid=os.getpid(), started_at=utc_now())
            _write_run_state(layout, spec, "RUNNING", pid=os.getpid())
            append_jsonl_locked(
                layout.registry_path,
                _registry_event(spec, state="RUNNING", exit_code=None, pid=os.getpid()),
            )
            log_path = Path(str(spec["log_path"]))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(
                    "[AUTODL_RUN_START] "
                    + json.dumps(
                        {
                            "run_id": spec["run_id"],
                            "dataset": spec["dataset"],
                            "stage": spec["stage"],
                            "gpu_index": spec.get("gpu_index"),
                            "gpu_uuid": spec.get("gpu_uuid"),
                            "command": spec["command"],
                            "timestamp": utc_now(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                child = subprocess.Popen(
                    list(spec["command"]),
                    cwd=project_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                child_pid = child.pid
                _write_stage_state(layout, spec, "RUNNING", run_id=spec["run_id"], pid=os.getpid(), child_pid=child_pid)
                _write_run_state(layout, spec, "RUNNING", pid=os.getpid(), child_pid=child_pid)
                exit_code = int(child.wait())
                log.write(f"[AUTODL_RUN_EXIT] exit_code={exit_code} timestamp={utc_now()}\n")
                log.flush()
                os.fsync(log.fileno())
            failures = _validate_success(spec, exit_code=exit_code)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")

    success = not failures
    final_state = "PASS" if success else "FAILED"
    _write_stage_gate(layout, spec, status=final_state, failures=failures)
    _write_stage_state(
        layout,
        spec,
        final_state,
        run_id=spec["run_id"],
        pid=os.getpid(),
        child_pid=child_pid,
        completed_at=utc_now(),
        exit_code=exit_code,
        failures=failures,
    )
    _write_run_state(
        layout,
        spec,
        final_state,
        pid=os.getpid(),
        child_pid=child_pid,
        completed_at=utc_now(),
        exit_code=exit_code,
        failures=failures,
    )
    append_jsonl_locked(
        layout.registry_path,
        _registry_event(spec, state=final_state, exit_code=exit_code, pid=os.getpid()),
    )
    if failures:
        print("AUTODL_RUN_FAILED: " + "; ".join(failures), file=sys.stderr)
        return exit_code if exit_code != 0 else 4
    return 0


def _layout_from_args(args: argparse.Namespace) -> Any:
    project_root = resolve_project_root(args.project_root)
    data_root = select_data_root(project_root, explicit=args.data_root)
    return build_runtime_layout(project_root=project_root, data_root=data_root).ensure()


def launch(args: argparse.Namespace) -> int:
    layout = _layout_from_args(args)
    validate_max_gpus(args.max_gpus)
    initialize_bace_stage_tree(layout)
    dataset = args.dataset.strip().lower()
    stage = args.stage.strip()
    assert_tastemolnet_launch_allowed(
        dataset=dataset,
        heavy=args.heavy,
        run_tastemolnet=os.environ.get("RUN_TASTEMOLNET", "0"),
    )
    if dataset == "bace":
        if stage not in BACE_STAGES:
            raise StageTransitionError(f"Primary BACE run requires a B0--B14 stage, got {stage}")
        assert_bace_stage_can_start(layout, stage)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    _assert_command_safe(command)
    if (args.gpu_index is None) != (args.gpu_uuid is None):
        raise AutoDLRuntimeError("--gpu-index and --gpu-uuid must be provided together")
    if args.gpu_required and args.gpu_index is None:
        raise AutoDLRuntimeError("This stage requires a physical GPU assignment")
    environment = _parse_environment(args.env)
    config_paths = [path.expanduser().resolve(strict=True) for path in args.config_file]
    input_manifest = args.input_manifest.expanduser().resolve(strict=True) if args.input_manifest else None
    expected_output = args.expected_output.expanduser().resolve(strict=False) if args.expected_output else None
    if expected_output is not None and expected_output.exists() and any(expected_output.iterdir() if expected_output.is_dir() else [expected_output]):
        raise AutoDLRuntimeError(f"Expected output must be fresh/absent: {expected_output}")
    run_id = _safe_id(args.run_id or _new_run_id(dataset, stage), label="run_id")
    launcher = args.launcher
    if launcher == "auto":
        launcher = "tmux" if shutil.which("tmux") else "nohup"
    if launcher == "tmux" and not shutil.which("tmux"):
        raise AutoDLRuntimeError("tmux launcher requested but tmux is unavailable")
    if launcher == "nohup" and not shutil.which("nohup"):
        raise AutoDLRuntimeError("nohup launcher requested but nohup is unavailable")
    tmux_session = _safe_id(f"cf-{run_id}", label="tmux_session") if launcher == "tmux" else None
    run_root = layout.runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    spec_path = run_root / "launch_spec.json"
    log_path = Path(args.log_path).expanduser().resolve(strict=False) if args.log_path else layout.logs_dir / f"{run_id}.log"
    spec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "project_root": str(layout.project_root),
        "data_root": str(layout.data_root),
        "control_root": str(layout.control_root),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "dataset": dataset,
        "stage": stage,
        "command": command,
        "environment": environment,
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "min_free_memory_mb": args.min_free_memory_mb,
        "idle_util_threshold": args.idle_util_threshold,
        "max_gpus": args.max_gpus,
        "git_commit": _git_value(layout.project_root, "rev-parse", "HEAD"),
        "config_files": [str(path) for path in config_paths],
        "config_hash": sha256_paths(config_paths),
        "input_manifest": str(input_manifest) if input_manifest else None,
        "input_hash": sha256_file(input_manifest) if input_manifest else None,
        "expected_output": str(expected_output) if expected_output else None,
        "required_output_files": list(args.required_output_file),
        "required_log_marker": args.required_log_marker,
        "log_path": str(log_path),
        "launcher": launcher,
        "tmux_session": tmux_session,
        "heavy": bool(args.heavy),
    }
    atomic_write_json(spec_path, spec)
    _write_stage_manifest(layout, spec)
    _write_stage_gate(layout, spec, status="NOT_EVALUATED", failures=[])
    _write_stage_state(layout, spec, "STARTING", run_id=run_id, tmux_session=tmux_session, gpu_index=args.gpu_index, gpu_uuid=args.gpu_uuid)
    _write_run_state(layout, spec, "STARTING", pid=None)
    append_jsonl_locked(
        layout.registry_path,
        _registry_event(spec, state="STARTING", exit_code=None, pid=None),
    )
    if args.foreground:
        return run_worker(spec_path)
    worker_command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--launch-spec", str(spec_path)]
    bootstrap_log = run_root / "launcher.log"
    try:
        if launcher == "tmux":
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", str(tmux_session), *worker_command],
                cwd=layout.project_root,
                check=True,
            )
            launcher_pid = None
        else:
            with bootstrap_log.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    [str(shutil.which("nohup")), *worker_command],
                    cwd=layout.project_root,
                    env=sanitized_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            launcher_pid = process.pid
    except Exception as exc:
        failures = [f"detached launcher failed: {type(exc).__name__}: {exc}"]
        _write_stage_gate(layout, spec, status="FAILED", failures=failures)
        _write_stage_state(layout, spec, "FAILED", run_id=run_id, failures=failures)
        _write_run_state(layout, spec, "FAILED", pid=None, failures=failures)
        append_jsonl_locked(
            layout.registry_path,
            _registry_event(spec, state="FAILED", exit_code=126, pid=None),
        )
        raise AutoDLRuntimeError(failures[0]) from exc
    print(
        json.dumps(
            {
                "run_id": run_id,
                "state": "STARTING",
                "launcher": launcher,
                "launcher_pid": launcher_pid,
                "tmux_session": tmux_session,
                "gpu_index": args.gpu_index,
                "gpu_uuid": args.gpu_uuid,
                "log_path": str(log_path),
                "control_root": str(layout.control_root),
                "python_executable": str(Path(sys.executable).resolve(strict=True)),
                "status_command": (
                    f"{Path(sys.executable).resolve(strict=True)} "
                    "scripts/autodl/status.py "
                    f"--data-root {layout.data_root}"
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", action="append", default=[])
    commands = parser.add_subparsers(dest="action", required=True)

    commands.add_parser("init-bace")
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--limit", type=int, default=20)

    stage_output = commands.add_parser("stage-output")
    stage_output.add_argument("--stage", choices=BACE_STAGES, required=True)
    stage_output.add_argument(
        "--required-output-file", action="append", default=[]
    )

    mark = commands.add_parser("mark-stage")
    mark.add_argument("--stage", choices=("B0_AUDIT", "B1_DATA_READY"), required=True)
    mark.add_argument("--evidence", type=Path, action="append", required=True)
    mark.add_argument("--note", required=True)

    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--dataset", required=True)
    launch_parser.add_argument("--stage", required=True)
    launch_parser.add_argument("--run-id")
    launch_parser.add_argument("--gpu-index", type=int)
    launch_parser.add_argument("--gpu-uuid")
    launch_parser.add_argument("--gpu-required", action="store_true")
    launch_parser.add_argument("--heavy", action="store_true")
    launch_parser.add_argument("--config-file", type=Path, action="append", default=[])
    launch_parser.add_argument("--input-manifest", type=Path)
    launch_parser.add_argument("--expected-output", type=Path)
    launch_parser.add_argument("--required-output-file", action="append", default=[])
    launch_parser.add_argument("--required-log-marker")
    launch_parser.add_argument("--log-path", type=Path)
    launch_parser.add_argument("--env", action="append", default=[])
    launch_parser.add_argument("--launcher", choices=("auto", "tmux", "nohup"), default="auto")
    launch_parser.add_argument("--foreground", action="store_true")
    launch_parser.add_argument(
        "--max-gpus",
        type=int,
        default=int(os.environ.get("AUTODL_MAX_GPUS", "2")),
    )
    launch_parser.add_argument(
        "--min-free-memory-mb",
        type=int,
        default=int(os.environ.get("AUTODL_MIN_FREE_MEMORY_MB", "16000")),
    )
    launch_parser.add_argument(
        "--idle-util-threshold",
        type=int,
        default=int(os.environ.get("AUTODL_IDLE_UTIL_THRESHOLD", "10")),
    )
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)

    worker = commands.add_parser("_worker")
    worker.add_argument("--launch-spec", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "_worker":
            return run_worker(args.launch_spec)
        layout = _layout_from_args(args)
        if args.action == "init-bace":
            initialize_bace_stage_tree(layout)
            print(layout.stages_root)
            return 0
        if args.action == "mark-stage":
            initialize_bace_stage_tree(layout)
            mark_bace_stage_pass(
                layout,
                stage=args.stage,
                evidence=args.evidence,
                note=args.note,
            )
            print(f"{args.stage}=PASS")
            return 0
        if args.action == "status":
            initialize_bace_stage_tree(layout)
            rows = latest_registry_events(read_registry(layout.registry_path))[: args.limit]
            print(json.dumps({"runs": rows}, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.action == "stage-output":
            initialize_bace_stage_tree(layout)
            output = resolve_passed_bace_stage_output(
                layout,
                args.stage,
                required_relative=args.required_output_file,
            )
            print(output)
            return 0
        if args.action == "launch":
            return launch(args)
        raise AutoDLRuntimeError(f"Unknown action: {args.action}")
    except (AutoDLRuntimeError, OSError, ValueError) as exc:
        print(f"AUTODL_EXP_RUN_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
