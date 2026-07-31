#!/usr/bin/env python3
"""State-machine control plane for local, HPC, and Slurm experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.gate_runner import build_gate_json, evaluate_gate
from scripts.ops.git_ops import (
    GitSafetyError,
    commit_allowed,
    commits_changed_paths,
    head_commit,
    inspect_status,
    push_head,
    stage_allowed_changes,
)
from scripts.ops.report import write_blocked_report, write_final_report
from scripts.ops.slurm_ops import build_exp_sbatch_argv, parse_exp_sbatch_job_id
from scripts.ops.spec import TaskSpec, dump_spec_snapshot, load_task_spec
from scripts.ops.ssh_ops import (
    SSHConfig,
    build_deploy_argv,
    build_preflight_argv,
    build_status_argv,
)
from scripts.ops.state import RunStatus, RunStore, append_jsonl_fsync, atomic_write_json
from scripts.ops.subprocess_utils import (
    CommandResult,
    CommandRunner,
    environment_audit,
    inherited_environment,
    write_command_streams,
)


class AutomationBlocked(RuntimeError):
    """A safety gate intentionally prevented further execution."""


def _json_print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _stage_plan(spec: TaskSpec) -> list[dict[str, Any]]:
    stage_by_id = spec.stage_by_id
    order = spec.topological_stage_ids()
    planned: list[dict[str, Any]] = []
    stop_before = spec.data["execution"].get("stop_before")
    for index, stage_id in enumerate(order, start=1):
        stage = stage_by_id[stage_id]
        planned.append(
            {
                "index": index,
                "stage_id": stage_id,
                "kind": stage["kind"],
                "dependencies": list(stage["dependencies"]),
                "automatic": stage_id != stop_before,
                "stop_before": stage_id == stop_before,
                "requires_approval": _requires_approval(stage),
            }
        )
    return planned


def build_plan(spec: TaskSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": spec.task_id,
        "spec_path": str(spec.path),
        "local_root": str(spec.local_root),
        "remote_root": spec.data["project"]["remote_root"],
        "branch": spec.data["project"]["branch"],
        "auto_until": spec.data["execution"]["auto_until"],
        "stop_before": spec.data["execution"].get("stop_before"),
        "permissions": dict(spec.data["permissions"]),
        "stages": _stage_plan(spec),
        "long_polling": False,
        "submission_entrypoint": "scripts/exp_sbatch.sh",
    }


def _reports_root(spec: TaskSpec) -> Path:
    return spec.local_root / "ops/reports"


def _new_store(spec: TaskSpec, run_id: str | None = None) -> RunStore:
    store = RunStore.create(
        _reports_root(spec),
        spec.task_id,
        run_id=run_id,
        spec_path=str(spec.path),
    )
    dump_spec_snapshot(spec, store.run_dir / "spec.snapshot.yaml")
    atomic_write_json(store.run_dir / "plan.json", build_plan(spec))
    store.transition(RunStatus.VALIDATED)
    return store


def _replace_python(command: list[str]) -> list[str]:
    if command and command[0] in {"python", "python3"}:
        return [sys.executable, *command[1:]]
    return command


def _resolve_cwd(spec: TaskSpec, stage: Mapping[str, Any]) -> Path:
    value = stage.get("cwd")
    if not value:
        return spec.local_root
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = spec.local_root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(spec.local_root)
    except ValueError as exc:
        raise AutomationBlocked(f"Stage cwd escapes local root: {resolved}") from exc
    return resolved


def _record_command(
    store: RunStore,
    stage_id: str,
    attempt: int,
    result: CommandResult,
) -> tuple[Path, Path]:
    stage_log_dir = store.run_dir / "logs" / stage_id
    stdout_path = stage_log_dir / f"attempt_{attempt}.stdout.log"
    stderr_path = stage_log_dir / f"attempt_{attempt}.stderr.log"
    write_command_streams(
        result, stdout_path=stdout_path, stderr_path=stderr_path
    )
    store.append_command(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage_id": stage_id,
            "attempt": attempt,
            "argv": result.argv,
            "cwd": result.cwd,
            "return_code": result.returncode,
            "timed_out": result.timed_out,
            "dry_run": result.dry_run,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    )
    return stdout_path, stderr_path


def _next_stage_id(spec: TaskSpec, current: str) -> str | None:
    order = spec.topological_stage_ids()
    index = order.index(current)
    return order[index + 1] if index + 1 < len(order) else None


def _run_local_stage(
    spec: TaskSpec,
    store: RunStore,
    stage: Mapping[str, Any],
    runner: CommandRunner,
) -> bool:
    stage_id = str(stage["id"])
    previous = store.load().get("stages", {}).get(stage_id, {})
    attempt = int(previous.get("attempt", 0)) + 1
    command = _replace_python([str(value) for value in stage["command"]])
    cwd = _resolve_cwd(spec, stage)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = runner.run(
        command,
        cwd=cwd,
        timeout_seconds=stage.get("timeout_seconds"),
        environment=inherited_environment(
            preserve_proxy_environment=spec.data["permissions"][
                "preserve_proxy_environment"
            ]
        ),
    )
    stdout_path, stderr_path = _record_command(
        store, stage_id, attempt, result
    )
    evaluation = evaluate_gate(
        task_id=spec.task_id,
        run_id=store.load()["run_id"],
        stage_id=stage_id,
        gate_spec=stage["gate"],
        expected_artifacts=list(stage["expected_artifacts"]),
        root=spec.local_root,
        stdout=result.stdout,
    )
    if result.returncode != 0:
        failures = list(evaluation.failed_hard_checks)
        failures.append(f"return_code:{result.returncode}")
        evaluation = type(evaluation)(
            passed=False,
            failed_hard_checks=tuple(failures),
            checks=evaluation.checks,
        )
    gate_payload = build_gate_json(
        task_id=spec.task_id,
        run_id=store.load()["run_id"],
        stage_id=stage_id,
        evaluation=evaluation,
        artifacts={
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        provenance={
            "git_commit": head_commit(runner, spec.local_root),
            "job_id": None,
            "dataset": None,
            "teacher_path": None,
            "candidate_path": None,
            "thresholds": [],
            "cf_mode": None,
        },
        next_stage=_next_stage_id(spec, stage_id),
        message="Local stage passed." if evaluation.passed else "Local stage failed.",
    )
    gate_path = store.run_dir / "gates" / f"{stage_id}.json"
    atomic_write_json(gate_path, gate_payload)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "stage_id": stage_id,
        "attempt": attempt,
        "start_time": started,
        "end_time": finished,
        "command_argv": command,
        "cwd": str(cwd),
        "return_code": result.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "artifacts": list(stage["expected_artifacts"]),
        "gate_result": str(gate_path),
        "job_id": None,
        "git_commit": head_commit(runner, spec.local_root),
        "remote_git_commit": None,
        "status": "PASSED" if evaluation.passed else "FAILED",
    }
    store.record_stage(stage_id, record)
    return evaluation.passed


def _block_after_stage_failure(
    store: RunStore,
    stage: Mapping[str, Any],
    *,
    error_class: str,
) -> None:
    stage_id = str(stage["id"])
    record = store.load()["stages"][stage_id]
    stderr_path = Path(str(record["stderr_path"]))
    stderr = (
        stderr_path.read_text(encoding="utf-8")
        if stderr_path.is_file()
        else ""
    )
    reason = f"{stage_id} exhausted its permitted attempts."
    store.transition(RunStatus.LOCAL_GATE_FAILED, reason=reason)
    store.transition(RunStatus.BLOCKED, reason=reason)
    write_blocked_report(
        store.run_dir,
        state=store.load(),
        failed_stage=stage_id,
        error_class=error_class,
        return_code=record.get("return_code"),
        stderr=stderr,
        artifacts=stage["expected_artifacts"],
        retry_count=max(0, int(record.get("attempt", 1)) - 1),
        recommended_action=(
            f"Inspect {stderr_path} and {record.get('gate_result')}."
        ),
        scientific_semantics_risk=False,
    )


def run_local(
    spec: TaskSpec,
    *,
    run_id: str | None = None,
    existing_store: RunStore | None = None,
) -> RunStore:
    store = existing_store or _new_store(spec, run_id)
    if store.load()["status"] == RunStatus.COMPLETED.value:
        store.append_event("resume_noop_completed")
        return store
    runner = CommandRunner()
    store.transition(RunStatus.LOCAL_PREFLIGHT)
    environment = inherited_environment(
        preserve_proxy_environment=spec.data["permissions"][
            "preserve_proxy_environment"
        ]
    )
    atomic_write_json(
        store.run_dir / "environment_audit.json", environment_audit(environment)
    )
    store.transition(RunStatus.LOCAL_GATE_RUNNING)
    stop_before = spec.data["execution"].get("stop_before")
    auto_until = spec.data["execution"]["auto_until"]
    reached_auto_until = False
    for stage_id in spec.topological_stage_ids():
        if stage_id == stop_before:
            store.transition(
                RunStatus.WAITING_APPROVAL,
                reason=f"Stopped before stage {stage_id}.",
            )
            break
        stage = spec.stage_by_id[stage_id]
        if stage["kind"] != "local_command":
            store.transition(RunStatus.LOCAL_GATE_PASSED)
            store.transition(
                RunStatus.WAITING_APPROVAL,
                reason=(
                    f"Local execution stopped before non-local stage {stage_id}; "
                    "use a separately permitted deploy or submit action."
                ),
            )
            break
        if store.stage_succeeded(stage_id):
            store.append_event("stage_skipped_resume", stage_id=stage_id)
        else:
            maximum_attempts = (
                int(spec.data["execution"]["max_auto_retries"]) + 1
            )
            passed = False
            while not passed:
                passed = _run_local_stage(spec, store, stage, runner)
                attempt = int(
                    store.load()["stages"][stage_id].get("attempt", 1)
                )
                if passed or attempt >= maximum_attempts:
                    break
                store.append_event(
                    "stage_retry_scheduled",
                    stage_id=stage_id,
                    next_attempt=attempt + 1,
                )
            if not passed:
                _block_after_stage_failure(
                    store, stage, error_class="LocalGateFailure"
                )
                return store
        if stage_id == auto_until:
            reached_auto_until = True
            break
    if store.load()["status"] == RunStatus.WAITING_APPROVAL.value:
        write_final_report(
            store.run_dir,
            state=store.load(),
            gate_summary="Local gates passed; approval boundary reached.",
            output_roots=[],
            provenance={"test_used": False, "calibration_used": False},
            next_allowed_stage=stop_before,
            stop_reason=f"Stopped before {stop_before}.",
        )
        return store
    if reached_auto_until or all(
        stage["kind"] != "local_command"
        or store.stage_succeeded(str(stage["id"]))
        for stage in spec.data["stages"]
        if str(stage["id"]) != stop_before
    ):
        store.transition(RunStatus.LOCAL_GATE_PASSED)
        store.transition(RunStatus.COMPLETED)
        state = store.load()
        state["stop_reason"] = "Configured local automation completed."
        store.save(state)
        write_final_report(
            store.run_dir,
            state=store.load(),
            gate_summary="All configured local command gates passed.",
            output_roots=[],
            provenance={"test_used": False, "calibration_used": False},
            next_allowed_stage=None,
            stop_reason="Configured local automation completed.",
        )
    return store


def _requires_approval(stage: Mapping[str, Any]) -> bool:
    if stage["kind"] == "approval":
        return True
    text = " ".join(
        [
            str(stage.get("id") or ""),
            str((stage.get("resources") or {}).get("tags") or ""),
            str((stage.get("resources") or {}).get("notes") or ""),
        ]
    ).lower()
    boundaries = (
        "calibration",
        "selector_freeze",
        "freeze_selector",
        "test_evaluation",
        "finalization",
        "overwrite",
        "threshold_change",
        "metric_change",
        "cohort_change",
        "teacher_change",
        "label_change",
    )
    if any(token in text for token in boundaries):
        return True
    return "full" in text.replace("-", "_").split("_")


def _ssh_config(spec: TaskSpec) -> SSHConfig:
    remote = spec.data["remote"]
    host_value = str(remote["host"])
    if "@" in host_value:
        user, host = host_value.split("@", maxsplit=1)
    else:
        user, host = getpass.getuser(), host_value
    return SSHConfig(
        host=host,
        port=int(remote["port"]),
        user=user,
        remote_root=spec.data["project"]["remote_root"],
        conda_env=remote["conda_env"],
        control_socket=remote.get("control_socket"),
    )


def deploy(
    spec: TaskSpec,
    *,
    dry_run: bool,
    store: RunStore | None = None,
) -> dict[str, Any]:
    if not spec.data["permissions"]["allow_remote_write"] and not dry_run:
        raise AutomationBlocked("Remote writes are disabled by the task spec.")
    runner = CommandRunner(default_timeout_seconds=120)
    git_spec = spec.data["git"]
    status = inspect_status(runner, spec.local_root, git_spec["allowed_paths"])
    if status.allowed_modified_paths and not git_spec["allow_commit"]:
        raise AutomationBlocked(
            "Allowed-path changes are uncommitted and allow_commit=false."
        )
    if git_spec["allow_commit"] and (
        status.allowed_modified_paths or status.staged_paths
    ):
        stage_allowed_changes(
            runner,
            spec.local_root,
            list(git_spec["allowed_paths"]),
            dry_run=dry_run,
        )
        commit = commit_allowed(
            runner,
            spec.local_root,
            branch=spec.data["project"]["branch"],
            message=git_spec["commit_message"],
            dry_run=dry_run,
        )
    else:
        commit = head_commit(runner, spec.local_root)
    if git_spec["allow_push"]:
        commit = push_head(
            runner,
            spec.local_root,
            branch=spec.data["project"]["branch"],
            dry_run=dry_run,
        )
    affected_paths = commits_changed_paths(
        runner, spec.local_root, spec.data["project"]["branch"]
    )
    config = _ssh_config(spec)
    output_roots = [
        str(stage["resources"]["expected_output_root"])
        for stage in spec.data["stages"]
        if (stage.get("resources") or {}).get("expected_output_root")
    ]
    preflight_argv = build_preflight_argv(
        config,
        protected_output_roots=output_roots,
        allow_overwrite=spec.data["permissions"]["allow_overwrite"],
    )
    deploy_argv = build_deploy_argv(
        config,
        branch=spec.data["project"]["branch"],
        expected_commit=commit,
        affected_paths=affected_paths,
        dynamic_paths=git_spec.get("dynamic_remote_paths") or [],
    )
    if dry_run:
        return {
            "dry_run": True,
            "preflight_argv": preflight_argv,
            "deploy_argv": deploy_argv,
            "commit": commit,
            "affected_paths": affected_paths,
        }
    if store:
        store.transition(RunStatus.REMOTE_PREFLIGHT)
    preflight = runner.run(preflight_argv, cwd=spec.local_root)
    if preflight.returncode != 0:
        raise AutomationBlocked(preflight.stderr or "Remote preflight failed.")
    result = runner.run(deploy_argv, cwd=spec.local_root)
    if result.returncode != 0:
        raise AutomationBlocked(result.stderr or "Remote deploy failed.")
    if store:
        state = store.load()
        state["local_commit"] = commit
        state["remote_commit"] = commit
        store.save(state)
        store.transition(RunStatus.PUSHED)
    return {
        "dry_run": False,
        "commit": commit,
        "remote_stdout": result.stdout,
    }


def submit(
    spec: TaskSpec,
    *,
    dry_run: bool,
    store: RunStore | None = None,
) -> dict[str, Any]:
    if not spec.data["permissions"]["allow_sbatch"] and not dry_run:
        raise AutomationBlocked("Slurm submission is disabled by the task spec.")
    runner = CommandRunner()
    job_ids: dict[str, str] = {}
    commands: list[list[str]] = []
    if not dry_run and store is None:
        raise AutomationBlocked(
            "A persisted --run-dir is required for real submission."
        )
    for stage_id in spec.topological_stage_ids():
        stage = spec.stage_by_id[stage_id]
        if stage["kind"] not in {"slurm_job", "audit"} or not stage.get("script"):
            continue
        if _requires_approval(stage) and (
            store is None or not store.is_approved(stage_id)
        ):
            raise AutomationBlocked(
                f"Stage {stage_id} requires an explicit approval event."
            )
        automation_environment: dict[str, str] = {}
        if store is not None:
            relative_run = store.run_dir.relative_to(spec.local_root)
            remote_run = (
                Path(spec.data["project"]["remote_root"]) / relative_run
            )
            relative_spec = spec.path.relative_to(spec.local_root)
            remote_spec = (
                Path(spec.data["project"]["remote_root"]) / relative_spec
            )
            automation_environment = {
                "AUTOMATION_RUN_DIR": str(remote_run),
                "AUTOMATION_RUN_ID": str(store.load()["run_id"]),
                "AUTOMATION_SPEC": str(remote_spec),
                "AUTOMATION_STAGE_ID": stage_id,
            }
            if stage["kind"] == "audit" and stage["dependencies"]:
                dependency_id = str(stage["dependencies"][0])
                if dependency_id in job_ids:
                    automation_environment["AUTOMATION_UPSTREAM_JOB_ID"] = (
                        job_ids[dependency_id]
                    )
        argv = build_exp_sbatch_argv(
            stage,
            spec.stage_by_id,
            job_ids,
            automation_environment=automation_environment,
        )
        commands.append(argv)
        if dry_run:
            job_ids[stage_id] = f"DRY_{len(job_ids) + 1}"
            continue
        result = runner.run(argv, cwd=spec.local_root)
        if result.returncode != 0:
            raise AutomationBlocked(result.stderr or "exp_sbatch failed.")
        job_ids[stage_id] = parse_exp_sbatch_job_id(result.stdout)
        if store:
            previous = store.load().get("stages", {}).get(stage_id, {})
            store.record_stage(
                stage_id,
                {
                    **previous,
                    "stage_id": stage_id,
                    "attempt": int(previous.get("attempt", 0)) + 1,
                    "start_time": None,
                    "end_time": None,
                    "command_argv": argv,
                    "cwd": str(spec.local_root),
                    "return_code": result.returncode,
                    "stdout_path": None,
                    "stderr_path": None,
                    "artifacts": list(stage["expected_artifacts"]),
                    "gate_result": None,
                    "job_id": job_ids[stage_id],
                    "git_commit": head_commit(runner, spec.local_root),
                    "remote_git_commit": store.load().get("remote_commit"),
                    "status": "SUBMITTED",
                },
            )
    return {"dry_run": dry_run, "commands": commands, "job_ids": job_ids}


def _load_snapshot(run_dir: Path) -> TaskSpec:
    return load_task_spec(run_dir / "spec.snapshot.yaml")


def _runtime_spec(spec: TaskSpec, runtime_root: Path) -> TaskSpec:
    payload = deepcopy(spec.data)
    payload["project"]["local_root"] = str(runtime_root.resolve())
    return TaskSpec(path=spec.path, data=payload)


def execute_stage(
    store: RunStore, stage_id: str, *, runtime_root: Path
) -> bool:
    spec = _runtime_spec(_load_snapshot(store.run_dir), runtime_root)
    if stage_id not in spec.stage_by_id:
        raise AutomationBlocked(f"Unknown stage: {stage_id}")
    stage = spec.stage_by_id[stage_id]
    if stage["kind"] not in {"local_command", "remote_command"}:
        raise AutomationBlocked(
            f"Stage {stage_id} is not directly executable: {stage['kind']}"
        )
    for dependency in stage["dependencies"]:
        if not store.stage_succeeded(str(dependency)):
            raise AutomationBlocked(
                f"Stage {stage_id} dependency is not passed: {dependency}"
            )
    if store.stage_succeeded(stage_id):
        store.append_event("stage_skipped_resume", stage_id=stage_id)
        return True
    maximum_attempts = int(spec.data["execution"]["max_auto_retries"]) + 1
    passed = False
    while not passed:
        passed = _run_local_stage(spec, store, stage, CommandRunner())
        attempt = int(store.load()["stages"][stage_id].get("attempt", 1))
        if passed or attempt >= maximum_attempts:
            break
        store.append_event(
            "stage_retry_scheduled",
            stage_id=stage_id,
            next_attempt=attempt + 1,
        )
    if not passed:
        _block_after_stage_failure(
            store, stage, error_class="RemoteStageFailure"
        )
    return passed


def execute_gate(
    store: RunStore,
    stage_id: str,
    *,
    runtime_root: Path,
    slurm_exit_code: str,
    marker_log: Path | None,
) -> bool:
    spec = _runtime_spec(_load_snapshot(store.run_dir), runtime_root)
    if stage_id not in spec.stage_by_id:
        raise AutomationBlocked(f"Unknown gate stage: {stage_id}")
    stage = spec.stage_by_id[stage_id]
    if stage["kind"] != "audit":
        raise AutomationBlocked(f"Stage is not an audit gate: {stage_id}")
    stdout = (
        marker_log.read_text(encoding="utf-8")
        if marker_log and marker_log.is_file()
        else ""
    )
    evaluation = evaluate_gate(
        task_id=spec.task_id,
        run_id=store.load()["run_id"],
        stage_id=stage_id,
        gate_spec=stage["gate"],
        expected_artifacts=list(stage["expected_artifacts"]),
        root=runtime_root,
        stdout=stdout,
        slurm_exit_code=slurm_exit_code,
    )
    gate_payload = build_gate_json(
        task_id=spec.task_id,
        run_id=store.load()["run_id"],
        stage_id=stage_id,
        evaluation=evaluation,
        artifacts={
            value: str((runtime_root / value).resolve())
            for value in stage["expected_artifacts"]
        },
        provenance={
            "git_commit": head_commit(CommandRunner(), runtime_root),
            "job_id": None,
            "dataset": None,
            "teacher_path": None,
            "candidate_path": None,
            "thresholds": [],
            "cf_mode": "strict_flip",
        },
        next_stage=_next_stage_id(spec, stage_id),
        message="Audit gate passed." if evaluation.passed else "Audit gate failed.",
    )
    destination = store.run_dir / "gates" / f"{stage_id}.json"
    atomic_write_json(destination, gate_payload)
    previous = store.load().get("stages", {}).get(stage_id, {})
    store.record_stage(
        stage_id,
        {
            **previous,
            "stage_id": stage_id,
            "attempt": int(previous.get("attempt", 0)) + 1,
            "start_time": None,
            "end_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command_argv": [],
            "cwd": str(runtime_root),
            "return_code": 0 if evaluation.passed else 1,
            "stdout_path": str(marker_log) if marker_log else None,
            "stderr_path": None,
            "artifacts": list(stage["expected_artifacts"]),
            "gate_result": str(destination),
            "job_id": None,
            "git_commit": gate_payload["provenance"]["git_commit"],
            "remote_git_commit": gate_payload["provenance"]["git_commit"],
            "status": "PASSED" if evaluation.passed else "FAILED",
        },
    )
    store.transition(
        RunStatus.AUDITING if evaluation.passed else RunStatus.BLOCKED,
        reason=gate_payload["message"],
    )
    return evaluation.passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safe state-machine experiment automation control plane."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-spec", "plan"):
        child = subparsers.add_parser(name)
        child.add_argument("spec")
    local = subparsers.add_parser("run-local")
    local.add_argument("spec")
    local.add_argument("--run-id")
    local.add_argument("--dry-run", action="store_true")
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("spec")
    deploy_parser.add_argument("--dry-run", action="store_true")
    deploy_parser.add_argument("--run-dir")
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("spec")
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--run-dir")
    for name in ("report", "resume"):
        child = subparsers.add_parser(name)
        child.add_argument("--run-dir", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", required=True)
    status.add_argument("--refresh", action="store_true")
    approve = subparsers.add_parser("approve")
    approve.add_argument("--run-dir", required=True)
    approve.add_argument("--stage", required=True)
    approve.add_argument("--reason", required=True)
    initialize = subparsers.add_parser("initialize-run")
    initialize.add_argument("--spec", required=True)
    initialize.add_argument("--run-dir", required=True)
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--project-root", required=True)
    execute = subparsers.add_parser("execute-stage")
    execute.add_argument("--run-dir", required=True)
    execute.add_argument("--stage", required=True)
    execute.add_argument("--project-root", required=True)
    gate = subparsers.add_parser("run-gate")
    gate.add_argument("--run-dir", required=True)
    gate.add_argument("--stage", required=True)
    gate.add_argument("--project-root", required=True)
    gate.add_argument("--slurm-exit-code", default="0:0")
    gate.add_argument("--marker-log")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-spec":
            spec = load_task_spec(args.spec)
            _json_print(
                {
                    "valid": True,
                    "task_id": spec.task_id,
                    "stage_order": spec.topological_stage_ids(),
                }
            )
        elif args.command == "plan":
            _json_print(build_plan(load_task_spec(args.spec)))
        elif args.command == "run-local":
            spec = load_task_spec(args.spec)
            if args.dry_run:
                _json_print(
                    {
                        "dry_run": True,
                        "side_effects": False,
                        "plan": build_plan(spec),
                    }
                )
            else:
                store = run_local(spec, run_id=args.run_id)
                _json_print(
                    {
                        "run_dir": str(store.run_dir),
                        "status": store.load()["status"],
                    }
                )
        elif args.command == "deploy":
            store = RunStore.open(args.run_dir) if args.run_dir else None
            _json_print(
                deploy(
                    load_task_spec(args.spec),
                    dry_run=args.dry_run,
                    store=store,
                )
            )
        elif args.command == "submit":
            store = RunStore.open(args.run_dir) if args.run_dir else None
            _json_print(
                submit(
                    load_task_spec(args.spec),
                    dry_run=args.dry_run,
                    store=store,
                )
            )
        elif args.command == "approve":
            store = RunStore.open(args.run_dir)
            spec = _load_snapshot(store.run_dir)
            if args.stage not in spec.stage_by_id:
                raise AutomationBlocked(f"Unknown stage: {args.stage}")
            store.approve(args.stage, args.reason, getpass.getuser())
            _json_print({"approved": True, "stage": args.stage})
        elif args.command == "initialize-run":
            spec = _runtime_spec(
                load_task_spec(args.spec), Path(args.project_root)
            )
            requested = Path(args.run_dir).expanduser().resolve()
            expected = (
                _reports_root(spec) / spec.task_id / args.run_id
            ).resolve()
            if requested != expected:
                raise AutomationBlocked(
                    f"Run directory does not match task/run ID: {requested}"
                )
            if requested.exists():
                store = RunStore.open(requested)
            else:
                store = RunStore.create(
                    _reports_root(spec),
                    spec.task_id,
                    run_id=args.run_id,
                    spec_path=str(spec.path),
                )
                dump_spec_snapshot(spec, store.run_dir / "spec.snapshot.yaml")
                atomic_write_json(store.run_dir / "plan.json", build_plan(spec))
                store.transition(RunStatus.VALIDATED)
            _json_print({"run_dir": str(store.run_dir), "initialized": True})
        elif args.command == "resume":
            store = RunStore.open(args.run_dir)
            spec = _load_snapshot(store.run_dir)
            resumed = run_local(spec, existing_store=store)
            _json_print(
                {"run_dir": str(resumed.run_dir), "status": resumed.load()["status"]}
            )
        elif args.command == "status":
            store = RunStore.open(args.run_dir)
            state = store.load()
            payload: dict[str, Any] = {"state": state}
            if args.refresh:
                spec = _load_snapshot(store.run_dir)
                job_ids = [
                    str(record["job_id"])
                    for record in state.get("stages", {}).values()
                    if str(record.get("job_id") or "").isdigit()
                ]
                if job_ids:
                    runner = CommandRunner(default_timeout_seconds=120)
                    result = runner.run(
                        build_status_argv(_ssh_config(spec), job_ids),
                        cwd=spec.local_root,
                    )
                    payload["slurm_query"] = {
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                else:
                    payload["slurm_query"] = {"jobs": [], "skipped": True}
            _json_print(payload)
        elif args.command == "report":
            store = RunStore.open(args.run_dir)
            state = store.load()
            path = (
                store.run_dir / "FINAL_REPORT.json"
                if (store.run_dir / "FINAL_REPORT.json").is_file()
                else store.run_dir / "BLOCKED_REPORT.json"
            )
            if not path.is_file():
                write_final_report(
                    store.run_dir,
                    state=state,
                    gate_summary="Report requested before terminal completion.",
                    output_roots=[],
                    provenance={},
                    next_allowed_stage=None,
                    stop_reason=state.get("stop_reason") or "No stop reason recorded.",
                )
                path = store.run_dir / "FINAL_REPORT.json"
            _json_print(json.loads(path.read_text(encoding="utf-8")))
        elif args.command == "execute-stage":
            store = RunStore.open(args.run_dir)
            passed = execute_stage(
                store, args.stage, runtime_root=Path(args.project_root)
            )
            print(
                "[AUTOMATION_STAGE_EXECUTION_PASS]"
                if passed
                else "[AUTOMATION_STAGE_EXECUTION_FAIL]"
            )
            return 0 if passed else 1
        elif args.command == "run-gate":
            store = RunStore.open(args.run_dir)
            passed = execute_gate(
                store,
                args.stage,
                runtime_root=Path(args.project_root),
                slurm_exit_code=args.slurm_exit_code,
                marker_log=Path(args.marker_log) if args.marker_log else None,
            )
            print(
                "[AUTOMATION_STAGE_GATE_PASS]"
                if passed
                else "[AUTOMATION_STAGE_GATE_FAIL]"
            )
            return 0 if passed else 1
        elif args.command == "finalize":
            store = RunStore.open(args.run_dir)
            state = store.load()
            if state["status"] not in {
                RunStatus.BLOCKED.value,
                RunStatus.FAILED.value,
            }:
                store.transition(RunStatus.COMPLETED)
            write_final_report(
                store.run_dir,
                state=store.load(),
                gate_summary="Pipeline finalizer collected persisted stage gates.",
                output_roots=[],
                provenance={},
                next_allowed_stage=None,
                stop_reason="Slurm dependency chain reached finalizer.",
            )
            print("[AUTOMATION_PIPELINE_REPORT_READY]")
        return 0
    except (AutomationBlocked, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[AUTOMATION_BLOCKED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
