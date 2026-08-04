#!/usr/bin/env python3
"""State-machine control plane for local, HPC, and Slurm experiments."""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.gate_runner import (
    build_gate_json,
    evaluate_gate,
    evaluate_gate_payload,
)
from scripts.ops.adopt_existing import (
    build_remote_script as build_adopt_remote_script,
    build_verification_argv as build_adopt_verification_argv,
    parse_evidence as parse_adopt_evidence,
)
from scripts.ops.git_ops import (
    GitSafetyError,
    commit_allowed,
    commits_changed_paths,
    head_commit,
    inspect_status,
    push_head,
    stage_allowed_changes,
)
from scripts.ops.preflight_policy import (
    evaluate_remote_dirty_policy,
    proxy_is_ready,
)
from scripts.ops.report import write_blocked_report, write_final_report
from scripts.ops.slurm_ops import (
    build_exp_sbatch_argv,
    is_active_status,
    is_failure_status,
    is_success_status,
    parse_exp_sbatch_job_id,
    parse_slurm_status_output,
)
from scripts.ops.spec import TaskSpec, dump_spec_snapshot, load_task_spec
from scripts.ops.ssh_ops import (
    SSHConfig,
    build_deploy_argv,
    build_preflight_argv,
    build_remote_project_command_argv,
    build_remote_submit_argv,
    build_status_argv,
    parse_preflight_output,
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
        approval_dependency = any(
            stage_by_id[str(dependency)]["kind"] == "approval"
            for dependency in stage["dependencies"]
        )
        planned.append(
            {
                "index": index,
                "stage_id": stage_id,
                "kind": stage["kind"],
                "dependencies": list(stage["dependencies"]),
                "automatic": (
                    stage_id != stop_before and stage["kind"] != "approval"
                ),
                "stop_before": stage_id == stop_before,
                "requires_approval": (
                    _requires_approval(stage) or approval_dependency
                ),
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


def _deploy_store(
    spec: TaskSpec, run_dir: str | Path | None
) -> RunStore:
    if run_dir is None:
        return _new_store(spec)
    requested = Path(run_dir).expanduser().resolve()
    if requested.exists():
        if (requested / "state.json").is_file():
            store = RunStore.open(requested)
            state = store.load()
            if state["task_id"] != spec.task_id:
                raise AutomationBlocked(
                    f"Run directory belongs to task {state['task_id']!r}, "
                    f"not {spec.task_id!r}."
                )
            return store
        if any(requested.iterdir()):
            raise AutomationBlocked(
                f"Existing run directory is nonempty but has no state: {requested}"
            )
    store = RunStore.create_at(
        requested,
        task_id=spec.task_id,
        spec_path=str(spec.path),
    )
    dump_spec_snapshot(spec, store.run_dir / "spec.snapshot.yaml")
    atomic_write_json(store.run_dir / "plan.json", build_plan(spec))
    store.transition(RunStatus.VALIDATED)
    return store


def _set_commits(
    store: RunStore, *, local_commit: str, remote_commit: str | None
) -> None:
    state = store.load()
    state["local_commit"] = local_commit
    state["remote_commit"] = remote_commit
    store.save(state)


def _set_stop_reason(store: RunStore, reason: str) -> None:
    state = store.load()
    state["stop_reason"] = reason
    store.save(state)


def _existing_deploy_report(store: RunStore) -> Path | None:
    for name in ("FINAL_REPORT.json", "BLOCKED_REPORT.json"):
        path = store.run_dir / name
        if path.is_file():
            return path
    return None


def deploy(
    spec: TaskSpec,
    *,
    dry_run: bool,
    preflight_only: bool = False,
    store: RunStore | None = None,
    run_dir: str | Path | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    if dry_run and preflight_only:
        raise AutomationBlocked(
            "--dry-run and --preflight-only are mutually exclusive."
        )
    if (
        spec.data["permissions"]["preserve_proxy_environment"] is not True
    ):
        raise AutomationBlocked(
            "Deploy preflight requires preserve_proxy_environment=true."
        )
    if (
        not spec.data["permissions"]["allow_remote_write"]
        and not dry_run
        and not preflight_only
    ):
        raise AutomationBlocked("Remote writes are disabled by the task spec.")
    if store is not None and run_dir is not None:
        raise AutomationBlocked("Pass either store or run_dir, not both.")
    store = store or _deploy_store(spec, run_dir)
    command_runner = runner or CommandRunner(default_timeout_seconds=120)
    previous_report = _existing_deploy_report(store)
    if (
        dry_run
        and store.load()["status"] == RunStatus.DRY_RUN_COMPLETED.value
        and previous_report
    ):
        return {
            "run_dir": str(store.run_dir),
            "status": RunStatus.DRY_RUN_COMPLETED.value,
            "report": str(previous_report),
            "return_code": 0,
            "resumed": True,
        }
    previous_preflight = store.load().get("stages", {}).get(
        "remote_preflight", {}
    )
    current_head = head_commit(command_runner, spec.local_root)
    if (
        preflight_only
        and previous_preflight.get("status")
        in {
            "PASSED",
            RunStatus.REMOTE_PREFLIGHT_PASSED.value,
            RunStatus.REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS.value,
            RunStatus.NEEDS_DEPLOY.value,
            RunStatus.NEEDS_PROXY_SETUP.value,
        }
        and previous_report
        and store.load().get("local_commit") == current_head
        and store.load().get("remote_commit") == current_head
    ):
        return {
            "run_dir": str(store.run_dir),
            "status": store.load()["status"],
            "report": str(previous_report),
            "return_code": 0,
            "resumed": True,
        }
    git_spec = spec.data["git"]
    status = inspect_status(
        command_runner, spec.local_root, git_spec["allowed_paths"]
    )
    if (
        not dry_run
        and not preflight_only
        and status.allowed_modified_paths
        and not git_spec["allow_commit"]
    ):
        raise AutomationBlocked(
            "Allowed-path changes are uncommitted and allow_commit=false."
        )
    if (
        not dry_run
        and not preflight_only
        and git_spec["allow_commit"]
        and (
        status.allowed_modified_paths or status.staged_paths
        )
    ):
        stage_allowed_changes(
            command_runner,
            spec.local_root,
            list(git_spec["allowed_paths"]),
            dry_run=dry_run,
        )
        commit = commit_allowed(
            command_runner,
            spec.local_root,
            branch=spec.data["project"]["branch"],
            message=git_spec["commit_message"],
            dry_run=dry_run,
        )
    else:
        commit = head_commit(command_runner, spec.local_root)
    if (
        not dry_run
        and not preflight_only
        and git_spec["allow_push"]
    ):
        commit = push_head(
            command_runner,
            spec.local_root,
            branch=spec.data["project"]["branch"],
            dry_run=dry_run,
        )
    affected_paths = commits_changed_paths(
        command_runner, spec.local_root, spec.data["project"]["branch"]
    )
    config = _ssh_config(spec)
    output_roots = list(
        dict.fromkeys(
            str(stage["resources"]["expected_output_root"])
            for stage in spec.data["stages"]
            if (stage.get("resources") or {}).get("expected_output_root")
        )
    )
    preflight_argv = build_preflight_argv(
        config,
        protected_output_roots=output_roots,
        allow_overwrite=(
            spec.data["permissions"]["allow_overwrite"]
            and not preflight_only
            and not dry_run
        ),
        patched_submodules=spec.data["remote_dirty_policy"][
            "allowed_patched_submodules"
        ],
    )
    deploy_argv = build_deploy_argv(
        config,
        branch=spec.data["project"]["branch"],
        expected_commit=commit,
        affected_paths=affected_paths,
        dynamic_paths=git_spec.get("dynamic_remote_paths") or [],
    )
    if dry_run:
        _set_commits(store, local_commit=commit, remote_commit=None)
        for stage_id, argv in (
            ("deploy_preflight_dry_run", preflight_argv),
            ("deploy_sync_dry_run", deploy_argv),
        ):
            store.append_command(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "stage_id": stage_id,
                    "attempt": 1,
                    "argv": argv,
                    "cwd": str(spec.local_root),
                    "return_code": None,
                    "timed_out": False,
                    "dry_run": True,
                    "executed": False,
                    "stdout_path": None,
                    "stderr_path": None,
                }
            )
        store.record_stage(
            "deploy_dry_run",
            {
                "stage_id": "deploy_dry_run",
                "attempt": 1,
                "start_time": None,
                "end_time": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "command_argv": [],
                "cwd": str(spec.local_root),
                "return_code": 0,
                "stdout_path": None,
                "stderr_path": None,
                "artifacts": [],
                "gate_result": None,
                "job_id": None,
                "git_commit": commit,
                "remote_git_commit": None,
                "status": "PASSED",
            },
        )
        store.transition(
            RunStatus.DRY_RUN_COMPLETED,
            reason="Deploy dry-run generated commands without SSH.",
        )
        _set_stop_reason(
            store, "Dry-run only; neither SSH nor remote Git ran."
        )
        report, _ = write_final_report(
            store.run_dir,
            state=store.load(),
            gate_summary="Deploy dry-run completed without remote execution.",
            output_roots=output_roots,
            provenance={"remote_contacted": False},
            next_allowed_stage="deploy --preflight-only",
            stop_reason="Dry-run only; neither SSH nor remote Git ran.",
            details={
                "preflight_command_record": "commands.jsonl:1",
                "deploy_command_record": "commands.jsonl:2",
                "affected_paths": affected_paths,
                "remote_contacted": False,
                "proxy_variables_present": environment_audit(
                    inherited_environment()
                )["proxy_present"],
            },
        )
        return {
            "run_dir": str(store.run_dir),
            "status": RunStatus.DRY_RUN_COMPLETED.value,
            "report": str(report),
            "return_code": 0,
            "dry_run": True,
            "preflight_argv": preflight_argv,
            "deploy_argv": deploy_argv,
            "commit": commit,
            "affected_paths": affected_paths,
        }
    if preflight_only:
        previous = previous_preflight
        store.transition(RunStatus.REMOTE_PREFLIGHT_RUNNING)
        environment = inherited_environment(
            preserve_proxy_environment=True
        )
        atomic_write_json(
            store.run_dir / "environment_audit.json",
            environment_audit(environment),
        )
        attempt = int(previous.get("attempt", 0)) + 1
        result = command_runner.run(
            preflight_argv,
            cwd=spec.local_root,
            environment=environment,
        )
        stdout_path, stderr_path = _record_command(
            store, "remote_preflight", attempt, result
        )
        parsed = parse_preflight_output(result.stdout, result.stderr)
        parsed_details = parsed.to_dict()
        dirty_result = evaluate_remote_dirty_policy(
            parsed, spec.data["remote_dirty_policy"]
        )
        proxy_policy = spec.data["proxy_policy"]
        proxy_ready = proxy_is_ready(parsed.proxy_present, proxy_policy)
        proxy_required_for_deploy = (
            "deploy_git_sync" in proxy_policy["required_for_stages"]
            and proxy_policy["require_any_present_for_git_network"]
        )
        commits_equal = parsed.commit == commit
        parsed_details.update(
            {
                "local_commit": commit,
                "remote_commit": parsed.commit,
                "expected_branch": spec.data["project"]["branch"],
                "remote_branch": parsed.branch,
                "commits_equal": commits_equal,
                "remote_dirty_summary": dirty_result.to_dict(),
                "remote_tracked_dirty_paths": list(
                    dirty_result.remote_tracked_dirty_paths
                ),
                "allowed_remote_tracked_dirty_paths": list(
                    dirty_result.allowed_remote_tracked_dirty_paths
                ),
                "disallowed_remote_tracked_dirty_paths": list(
                    dirty_result.disallowed_remote_tracked_dirty_paths
                ),
                "configured_allowed_remote_tracked_dirty_paths": list(
                    spec.data["remote_dirty_policy"][
                        "allowed_tracked_paths"
                    ]
                ),
                "configured_allowed_remote_untracked_paths": list(
                    spec.data["remote_dirty_policy"][
                        "allowed_untracked_paths"
                    ]
                ),
                "allowed_remote_untracked_paths": list(
                    dirty_result.allowed_remote_untracked_paths
                ),
                "remote_untracked_paths": list(
                    dirty_result.remote_untracked_paths
                ),
                "disallowed_remote_untracked_paths": list(
                    dirty_result.disallowed_remote_untracked_paths
                ),
                "dynamic_dirty": list(dirty_result.dynamic_tracked),
                "allowed_dynamic_dirty": list(
                    dirty_result.dynamic_tracked
                ),
                "verified_patched_submodules": list(
                    dirty_result.verified_patched_submodules
                ),
                "blocked_remote_dirty": list(dirty_result.blocked),
                "patched_submodule_audits": list(
                    dirty_result.submodule_audits
                ),
                "patched_submodule_verified": (
                    bool(
                        spec.data["remote_dirty_policy"][
                            "allowed_patched_submodules"
                        ]
                    )
                    and len(dirty_result.verified_patched_submodules)
                    == len(
                        spec.data["remote_dirty_policy"][
                            "allowed_patched_submodules"
                        ]
                    )
                ),
                "proxy_ready": proxy_ready,
                "proxy_required_for_deploy_git_sync": (
                    proxy_required_for_deploy
                ),
            }
        )
        evidence_path = (
            store.run_dir / "evidence" / "remote_preflight_evidence.json"
        )
        atomic_write_json(
            evidence_path,
            {
                **parsed_details,
                "remote_write_performed": False,
            },
        )
        hard_failures: list[str] = []
        if result.returncode != 0:
            hard_failures.append(f"ssh_return_code={result.returncode}")
        if parsed.branch != spec.data["project"]["branch"]:
            hard_failures.append("remote_branch_mismatch")
        if not parsed.conda_ready:
            hard_failures.append("conda_not_ready")
        if not parsed.sbatch_ready:
            hard_failures.append("sbatch_not_ready")
        if not parsed.sacct_ready:
            hard_failures.append("sacct_not_ready")
        if parsed.finalized_output_blocked:
            hard_failures.append("finalized_output_blocked")
        if dirty_result.tracked_blocked and dirty_result.untracked_blocked:
            hard_failures.append("remote_dirty_files_blocked")
        elif dirty_result.tracked_blocked:
            hard_failures.append("remote_tracked_files_dirty")
        elif dirty_result.untracked_blocked:
            hard_failures.append("remote_untracked_files_dirty")
        if parsed.commit is None:
            hard_failures.append("remote_commit_missing")
        _set_commits(
            store, local_commit=commit, remote_commit=parsed.commit
        )
        command_status = "PASSED" if result.returncode == 0 else "FAILED"
        if hard_failures:
            gate_status = "BLOCKED"
            if result.returncode == 0:
                stage_status = RunStatus.REMOTE_PREFLIGHT_BLOCKED.value
                terminal_status = RunStatus.REMOTE_PREFLIGHT_BLOCKED
            else:
                stage_status = "FAILED"
                terminal_status = RunStatus.BLOCKED
            store.record_stage(
                "remote_preflight",
                {
                    "stage_id": "remote_preflight",
                    "attempt": attempt,
                    "start_time": None,
                    "end_time": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "command_argv": preflight_argv,
                    "cwd": str(spec.local_root),
                    "return_code": result.returncode,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "artifacts": [str(evidence_path)],
                    "gate_result": None,
                    "job_id": None,
                    "git_commit": commit,
                    "remote_git_commit": parsed.commit,
                    "command_status": command_status,
                    "gate_status": gate_status,
                    "remote_tracked_dirty_paths": list(
                        dirty_result.remote_tracked_dirty_paths
                    ),
                    "allowed_remote_tracked_dirty_paths": list(
                        dirty_result.allowed_remote_tracked_dirty_paths
                    ),
                    "disallowed_remote_tracked_dirty_paths": list(
                        dirty_result.disallowed_remote_tracked_dirty_paths
                    ),
                    "configured_allowed_remote_untracked_paths": list(
                        spec.data["remote_dirty_policy"][
                            "allowed_untracked_paths"
                        ]
                    ),
                    "allowed_remote_untracked_paths": list(
                        dirty_result.allowed_remote_untracked_paths
                    ),
                    "remote_untracked_paths": list(
                        dirty_result.remote_untracked_paths
                    ),
                    "disallowed_remote_untracked_paths": list(
                        dirty_result.disallowed_remote_untracked_paths
                    ),
                    "status": stage_status,
                },
            )
            store.transition(terminal_status, reason="; ".join(hard_failures))
            _set_stop_reason(store, "; ".join(hard_failures))
            report, _ = write_blocked_report(
                store.run_dir,
                state=store.load(),
                failed_stage="remote_preflight",
                error_class="RemotePreflightFailure",
                return_code=result.returncode,
                stderr=result.stderr,
                artifacts=[str(evidence_path)],
                retry_count=attempt - 1,
                recommended_action=(
                    "Resolve the reported read-only preflight condition; "
                    "do not pull automatically."
                ),
                scientific_semantics_risk=False,
                details={
                    **parsed_details,
                    "failed_checks": hard_failures,
                    "command_status": command_status,
                    "gate_status": gate_status,
                    "next_action": "manual_preflight_remediation",
                    "remote_write_performed": False,
                },
            )
            return {
                "run_dir": str(store.run_dir),
                "status": terminal_status.value,
                "report": str(report),
                "return_code": result.returncode or 2,
            }
        allowlisted_tracked_dirty = bool(
            dirty_result.allowed_remote_tracked_dirty_paths
        )
        allowlisted_untracked = bool(
            dirty_result.allowed_remote_untracked_paths
        )
        if not commits_equal and proxy_required_for_deploy and not proxy_ready:
            selected_status = RunStatus.NEEDS_PROXY_SETUP
            next_action = "proxy_or_ssh_tunnel_required_before_deploy"
            preflight_status = RunStatus.NEEDS_PROXY_SETUP.value
            gate_status = "BLOCKED"
            stop_reason = (
                "Remote commit differs and Git network proxy readiness is absent."
            )
        elif not commits_equal:
            selected_status = RunStatus.NEEDS_DEPLOY
            next_action = "deploy"
            preflight_status = RunStatus.NEEDS_DEPLOY.value
            gate_status = "PASSED"
            stop_reason = "Remote commit differs; deploy was not run."
        elif (
            proxy_required_for_deploy and not proxy_ready
        ) or allowlisted_tracked_dirty or allowlisted_untracked:
            selected_status = RunStatus.REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS
            if proxy_required_for_deploy and not proxy_ready:
                next_action = (
                    "configure_proxy_before_future_git_sync_or_request_"
                    "remote_write_approval"
                )
            else:
                next_action = "remote_write_approval_required"
            preflight_status = (
                RunStatus.REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS.value
            )
            gate_status = "PASSED_WITH_WARNINGS"
            warning_reasons: list[str] = []
            if proxy_required_for_deploy and not proxy_ready:
                warning_reasons.append(
                    "proxy is required before a future Git synchronization"
                )
            if allowlisted_tracked_dirty:
                warning_reasons.append(
                    "remote tracked dirt was accepted by exact path allowlist"
                )
            if allowlisted_untracked:
                warning_reasons.append(
                    "remote untracked files were accepted by exact path "
                    "allowlist"
                )
            stop_reason = (
                "Read-only preflight passed with warnings: "
                + "; ".join(warning_reasons)
                + "."
            )
        else:
            selected_status = RunStatus.REMOTE_PREFLIGHT_PASSED
            next_action = "remote_write_approval_required"
            preflight_status = RunStatus.REMOTE_PREFLIGHT_PASSED.value
            gate_status = "PASSED"
            stop_reason = (
                "Read-only preflight passed; remote write awaits approval."
            )
        store.record_stage(
            "remote_preflight",
            {
                "stage_id": "remote_preflight",
                "attempt": attempt,
                "start_time": None,
                "end_time": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "command_argv": preflight_argv,
                "cwd": str(spec.local_root),
                "return_code": result.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "artifacts": [str(evidence_path)],
                "gate_result": None,
                "job_id": None,
                "git_commit": commit,
                "remote_git_commit": parsed.commit,
                "command_status": command_status,
                "gate_status": gate_status,
                "remote_tracked_dirty_paths": list(
                    dirty_result.remote_tracked_dirty_paths
                ),
                "allowed_remote_tracked_dirty_paths": list(
                    dirty_result.allowed_remote_tracked_dirty_paths
                ),
                "disallowed_remote_tracked_dirty_paths": list(
                    dirty_result.disallowed_remote_tracked_dirty_paths
                ),
                "configured_allowed_remote_untracked_paths": list(
                    spec.data["remote_dirty_policy"][
                        "allowed_untracked_paths"
                    ]
                ),
                "allowed_remote_untracked_paths": list(
                    dirty_result.allowed_remote_untracked_paths
                ),
                "remote_untracked_paths": list(
                    dirty_result.remote_untracked_paths
                ),
                "disallowed_remote_untracked_paths": list(
                    dirty_result.disallowed_remote_untracked_paths
                ),
                "status": selected_status.value,
            },
        )
        store.transition(selected_status, reason=stop_reason)
        _set_stop_reason(store, stop_reason)
        report, _ = write_final_report(
            store.run_dir,
            state=store.load(),
            gate_summary=f"Read-only SSH preflight: {preflight_status}.",
            output_roots=output_roots,
            provenance={"remote_write_performed": False},
            next_allowed_stage=next_action,
            stop_reason=stop_reason,
            details={
                **parsed_details,
                "preflight_status": preflight_status,
                "command_status": command_status,
                "gate_status": gate_status,
                "next_action": next_action,
                "remote_write_performed": False,
            },
        )
        return {
            "run_dir": str(store.run_dir),
            "status": store.load()["status"],
            "report": str(report),
            "return_code": 0,
        }
    if store:
        store.transition(RunStatus.REMOTE_PREFLIGHT)
    preflight = command_runner.run(preflight_argv, cwd=spec.local_root)
    if preflight.returncode != 0:
        raise AutomationBlocked(preflight.stderr or "Remote preflight failed.")
    parsed_preflight = parse_preflight_output(
        preflight.stdout, preflight.stderr
    )
    dirty_preflight = evaluate_remote_dirty_policy(
        parsed_preflight, spec.data["remote_dirty_policy"]
    )
    if not dirty_preflight.passed:
        raise AutomationBlocked(
            "Remote dirty policy blocked deploy: "
            + ", ".join(dirty_preflight.blocked)
        )
    proxy_policy = spec.data["proxy_policy"]
    proxy_required = (
        "deploy_git_sync" in proxy_policy["required_for_stages"]
        and proxy_policy["require_any_present_for_git_network"]
    )
    if proxy_required and not proxy_is_ready(
        parsed_preflight.proxy_present, proxy_policy
    ):
        raise AutomationBlocked(
            "Git-network proxy readiness is absent; "
            "Git fetch/pull was not executed."
        )
    result = command_runner.run(deploy_argv, cwd=spec.local_root)
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


def _require_adoption_permissions(spec: TaskSpec) -> None:
    forbidden = (
        "allow_remote_write",
        "allow_sbatch",
        "allow_overwrite",
        "allow_gpu_smoke",
        "allow_full",
        "allow_calibration",
        "allow_test",
        "allow_finalization",
    )
    enabled = [key for key in forbidden if spec.data["permissions"][key]]
    if enabled:
        raise AutomationBlocked(
            "adopt-existing requires all execution permissions false; enabled="
            + ",".join(enabled)
        )
    adopt = spec.data.get("adopt_existing")
    if not isinstance(adopt, Mapping) or adopt.get("enabled") is not True:
        raise AutomationBlocked("adopt_existing is not enabled by the task spec.")


def _adoption_verification_config(
    spec: TaskSpec, *, local_commit: str
) -> dict[str, Any]:
    adopt = spec.data["adopt_existing"]
    adopted_stages = [str(value) for value in adopt["stages"]]
    stage_gates = []
    for stage_id in adopted_stages:
        stage = spec.stage_by_id[stage_id]
        stage_gates.append(
            {
                "stage_id": stage_id,
                "json_path": stage["gate"].get("json_path"),
                "required_fields": dict(
                    stage["gate"].get("required_fields") or {}
                ),
                "forbidden_fields": dict(
                    stage["gate"].get("forbidden_fields") or {}
                ),
            }
        )
    next_stage = _next_stage_id(spec, adopted_stages[-1])
    return {
        "mode": adopt["mode"],
        "output_root": adopt["output_root"],
        "completion_marker": adopt["completion_marker"],
        "manifest_path": adopt["manifest_path"],
        "finalized_marker": adopt["finalized_marker"],
        "expected_generation_commit": adopt["expected_generation_commit"],
        "artifact_aliases": dict(adopt["artifact_aliases"]),
        "allowed_external_manifest_artifacts": list(
            adopt["allowed_external_manifest_artifacts"]
        ),
        "jsonl_row_counts": dict(adopt["jsonl_row_counts"]),
        "allow_missing_current_markers": adopt[
            "allow_missing_current_markers"
        ],
        "adopted_stages": adopted_stages,
        "next_stage": next_stage,
        "stage_gates": stage_gates,
        "current_local_commit": local_commit,
    }


def _append_adoption_report_contract(
    report_path: Path, *, commits_differ: bool
) -> None:
    statements = [
        "",
        "## Legacy Adoption",
        "",
        "- Phase A was adopted from verified legacy artifacts.",
        "- No remote artifact was modified.",
        "- No Slurm job was submitted.",
        "- Current code commit and legacy generation commit differ: "
        f"{str(commits_differ).lower()}.",
        "- All manifest artifacts had their size and SHA256 verified.",
        "- Missing current markers were accepted only through legacy manifest "
        "integrity verification.",
        "- Execution stopped before phase_b_gpu_smoke pending explicit approval.",
        "",
    ]
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(statements))


def adopt_existing(
    spec: TaskSpec,
    *,
    dry_run: bool,
    run_dir: str | Path | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    store = _deploy_store(spec, run_dir)
    try:
        _require_adoption_permissions(spec)
    except AutomationBlocked as exc:
        reason = str(exc)
        store.transition(RunStatus.BLOCKED, reason=reason)
        _set_stop_reason(store, reason)
        report, _ = write_blocked_report(
            store.run_dir,
            state=store.load(),
            failed_stage="adopt_existing_permissions",
            error_class="AdoptExistingPermissionBoundary",
            return_code=None,
            stderr=reason,
            artifacts=[],
            retry_count=0,
            recommended_action=(
                "Disable all execution permissions before legacy adoption."
            ),
            scientific_semantics_risk=False,
            details={"remote_write_performed": False, "slurm_jobs": []},
        )
        return {
            "status": RunStatus.BLOCKED.value,
            "return_code": 2,
            "run_dir": str(store.run_dir),
            "report": str(report),
            "remote_write_performed": False,
            "slurm_jobs": [],
        }
    command_runner = runner or CommandRunner(default_timeout_seconds=600)
    local_commit = head_commit(command_runner, spec.local_root)
    verification_config = _adoption_verification_config(
        spec, local_commit=local_commit
    )
    ssh_config = _ssh_config(spec)
    verification_argv = build_adopt_verification_argv(
        ssh_config, verification_config
    )
    remote_script = build_adopt_remote_script(
        ssh_config, verification_config
    )
    adopted_stages = list(verification_config["adopted_stages"])
    next_stage = verification_config["next_stage"]
    output_root = str(verification_config["output_root"])

    if dry_run:
        _set_commits(store, local_commit=local_commit, remote_commit=None)
        store.append_command(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "stage_id": "adopt_existing_verification_dry_run",
                "attempt": 1,
                "argv": verification_argv,
                "cwd": str(spec.local_root),
                "return_code": None,
                "timed_out": False,
                "dry_run": True,
                "executed": False,
                "stdout_path": None,
                "stderr_path": None,
            }
        )
        store.record_stage(
            "adopt_existing_dry_run",
            {
                "stage_id": "adopt_existing_dry_run",
                "attempt": 1,
                "start_time": None,
                "end_time": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "command_argv": verification_argv,
                "cwd": str(spec.local_root),
                "return_code": None,
                "status": RunStatus.ADOPT_EXISTING_DRY_RUN.value,
                "command_status": "NOT_EXECUTED",
                "gate_status": "NOT_EVALUATED",
                "job_id": None,
            },
        )
        store.transition(
            RunStatus.ADOPT_EXISTING_DRY_RUN,
            reason="Adoption dry-run generated a read-only verification command.",
        )
        _set_stop_reason(store, "Dry-run only; SSH and local tests were not run.")
        report, _ = write_final_report(
            store.run_dir,
            state=store.load(),
            gate_summary="Adopt-existing dry-run only.",
            output_roots=[output_root],
            provenance={"remote_write_performed": False, "slurm_jobs": []},
            next_allowed_stage="adopt-existing without --dry-run",
            stop_reason="Dry-run only; SSH and local tests were not run.",
            details={
                "adopted_stages": adopted_stages,
                "next_stage": next_stage,
                "remote_write_performed": False,
                "slurm_jobs": [],
            },
        )
        return {
            "status": RunStatus.ADOPT_EXISTING_DRY_RUN.value,
            "return_code": 0,
            "dry_run": True,
            "run_dir": str(store.run_dir),
            "report": str(report),
            "verification_argv": verification_argv,
            "remote_script": remote_script,
            "adopted_stages": adopted_stages,
            "next_stage": next_stage,
            "remote_write_performed": False,
            "slurm_jobs": [],
        }

    local_stage = spec.stage_by_id.get("local_phase_a_tests")
    if local_stage is None or local_stage["kind"] != "local_command":
        raise AutomationBlocked(
            "adopt-existing requires local_phase_a_tests local_command."
        )
    store.transition(RunStatus.LOCAL_GATE_RUNNING)
    local_passed = _run_local_stage(
        spec, store, local_stage, command_runner
    )
    if not local_passed:
        _block_after_stage_failure(
            store, local_stage, error_class="AdoptExistingLocalGateFailure"
        )
        report = store.run_dir / "BLOCKED_REPORT.md"
        return {
            "status": RunStatus.BLOCKED.value,
            "return_code": 2,
            "run_dir": str(store.run_dir),
            "report": str(report),
        }

    store.transition(RunStatus.ADOPT_EXISTING_VERIFYING)
    result = command_runner.run(
        verification_argv,
        cwd=spec.local_root,
        environment=inherited_environment(preserve_proxy_environment=True),
    )
    stdout_path, stderr_path = _record_command(
        store, "adopt_existing_verification", 1, result
    )
    evidence: dict[str, Any]
    parse_error: str | None = None
    try:
        evidence = parse_adopt_evidence(result.stdout)
    except (ValueError, RuntimeError) as exc:
        evidence = {}
        parse_error = str(exc)
    failures = list(evidence.get("failed_hard_checks") or [])
    if parse_error:
        failures.append(f"evidence_parse_error:{parse_error}")
    if result.returncode != 0:
        failures.append(f"ssh_return_code:{result.returncode}")
    if evidence.get("verification_passed") is not True:
        failures.append("verification_passed_not_true")
    if evidence.get("current_local_commit") not in (None, local_commit):
        failures.append("evidence_local_commit_mismatch")

    if failures:
        store.record_stage(
            "adopt_existing_verification",
            {
                "stage_id": "adopt_existing_verification",
                "attempt": 1,
                "start_time": None,
                "end_time": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "command_argv": verification_argv,
                "cwd": str(spec.local_root),
                "return_code": result.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "status": "FAILED",
                "command_status": (
                    "PASSED" if result.returncode == 0 else "FAILED"
                ),
                "gate_status": "BLOCKED",
                "job_id": None,
            },
        )
        store.transition(RunStatus.BLOCKED, reason="; ".join(failures))
        _set_stop_reason(store, "; ".join(failures))
        report, _ = write_blocked_report(
            store.run_dir,
            state=store.load(),
            failed_stage="adopt_existing_verification",
            error_class="AdoptExistingVerificationFailure",
            return_code=result.returncode,
            stderr=result.stderr or parse_error or "",
            artifacts=[output_root],
            retry_count=0,
            recommended_action="Inspect legacy manifest evidence; do not modify it.",
            scientific_semantics_risk=False,
            details={
                "failed_checks": failures,
                "evidence": evidence,
                "remote_write_performed": False,
            },
        )
        return {
            "status": RunStatus.BLOCKED.value,
            "return_code": result.returncode or 2,
            "run_dir": str(store.run_dir),
            "report": str(report),
        }

    evidence_path = store.run_dir / "evidence/adopt_existing_evidence.json"
    atomic_write_json(evidence_path, evidence)
    remote_commit = str(evidence["current_remote_commit"])
    _set_commits(
        store, local_commit=local_commit, remote_commit=remote_commit
    )
    store.record_stage(
        "adopt_existing_verification",
        {
            "stage_id": "adopt_existing_verification",
            "attempt": 1,
            "start_time": None,
            "end_time": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "command_argv": verification_argv,
            "cwd": str(spec.local_root),
            "return_code": result.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "artifacts": [str(evidence_path)],
            "gate_result": str(evidence_path),
            "job_id": None,
            "git_commit": local_commit,
            "remote_git_commit": remote_commit,
            "status": "PASSED",
            "command_status": "PASSED",
            "gate_status": "PASSED",
        },
    )
    for stage_id in adopted_stages:
        stage = spec.stage_by_id[stage_id]
        store.record_stage(
            stage_id,
            {
                "stage_id": stage_id,
                "attempt": 1,
                "start_time": None,
                "end_time": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "command_argv": [],
                "cwd": str(spec.local_root),
                "return_code": None,
                "stdout_path": None,
                "stderr_path": None,
                "artifacts": list(stage["expected_artifacts"]),
                "gate_result": str(evidence_path),
                "job_id": None,
                "git_commit": local_commit,
                "remote_git_commit": remote_commit,
                "status": RunStatus.ADOPTED_EXISTING.value,
                "command_status": "NOT_EXECUTED",
                "gate_status": "PASSED",
                "provenance": {
                    "source": "legacy_manifest_sha256",
                    "legacy_generation_commit": evidence[
                        "legacy_generation_commit"
                    ],
                },
            },
        )
    store.transition(RunStatus.ADOPTED_EXISTING)
    store.transition(
        RunStatus.STOPPED_BEFORE_APPROVAL,
        reason=f"Stopped before {next_stage} pending explicit approval.",
    )
    _set_stop_reason(
        store, f"Stopped before {next_stage} pending explicit approval."
    )
    report, _ = write_final_report(
        store.run_dir,
        state=store.load(),
        gate_summary="Phase A adopted through verified legacy manifest integrity.",
        output_roots=[output_root],
        provenance={
            "source": "legacy_manifest_sha256",
            "current_local_commit": local_commit,
            "current_remote_commit": remote_commit,
            "legacy_generation_commit": evidence["legacy_generation_commit"],
            "remote_write_performed": False,
            "slurm_jobs": [],
        },
        next_allowed_stage=next_stage,
        stop_reason=f"Stopped before {next_stage} pending explicit approval.",
        details={
            "evidence_path": str(evidence_path),
            "artifact_count": evidence["artifact_count"],
            "current_required_marker_present": evidence[
                "current_required_marker_present"
            ],
            "accepted_via_legacy_manifest_integrity": evidence[
                "accepted_via_legacy_manifest_integrity"
            ],
            "remote_write_performed": False,
            "slurm_jobs": [],
        },
    )
    _append_adoption_report_contract(
        report,
        commits_differ=(
            local_commit != str(evidence["legacy_generation_commit"])
        ),
    )
    return {
        "status": RunStatus.STOPPED_BEFORE_APPROVAL.value,
        "return_code": 0,
        "dry_run": False,
        "run_dir": str(store.run_dir),
        "report": str(report),
        "evidence": str(evidence_path),
        "adopted_stages": adopted_stages,
        "next_stage": next_stage,
        "remote_write_performed": False,
        "slurm_jobs": [],
    }


_CLEAR_PHASE_B_GATE_MARKER = (
    "[AUTOMATION_CLEAR_PHASE_B_GATE_EVIDENCE_B64]"
)


def _render_runtime_template(
    value: str,
    *,
    run_id: str,
    output_root: str | None = None,
    job_id: str | None = None,
    commit: str | None = None,
    remote_root: str | None = None,
) -> str:
    replacements = {
        "{run_id}": run_id,
        "{output_root}": output_root,
        "{job_id}": job_id,
        "{commit}": commit,
        "{remote_root}": remote_root,
    }
    rendered = str(value)
    for token, replacement in replacements.items():
        if token in rendered:
            if replacement is None:
                raise AutomationBlocked(
                    f"Runtime template {token} is unavailable for {value!r}."
                )
            rendered = rendered.replace(token, str(replacement))
    if "{" in rendered or "}" in rendered:
        raise AutomationBlocked(f"Unsupported runtime template: {value!r}")
    return rendered


def _smoke_submission_permissions(spec: TaskSpec) -> None:
    permissions = spec.data["permissions"]
    required_true = ("allow_remote_write", "allow_sbatch", "allow_gpu_smoke")
    required_false = (
        "allow_full",
        "allow_calibration",
        "allow_test",
        "allow_finalization",
        "allow_overwrite",
    )
    failures = [key for key in required_true if permissions.get(key) is not True]
    failures.extend(
        key for key in required_false if permissions.get(key) is not False
    )
    if failures:
        raise AutomationBlocked(
            "Phase B GPU smoke permission contract failed: "
            + ", ".join(failures)
        )


def _execution_snapshot_path(store: RunStore) -> Path:
    return store.run_dir / "spec.execution.snapshot.yaml"


def _load_execution_spec(store: RunStore) -> TaskSpec:
    snapshot = _execution_snapshot_path(store)
    if snapshot.is_file():
        return load_task_spec(snapshot)
    source = Path(str(store.load().get("spec_path") or "")).expanduser()
    if source.is_file():
        return load_task_spec(source)
    return _load_snapshot(store.run_dir)


def _slurm_compute_stages(spec: TaskSpec) -> list[dict[str, Any]]:
    return [
        spec.stage_by_id[stage_id]
        for stage_id in spec.topological_stage_ids()
        if spec.stage_by_id[stage_id]["kind"] == "slurm_job"
        and spec.stage_by_id[stage_id].get("script")
    ]


def _approval_dependency(
    spec: TaskSpec, stage: Mapping[str, Any]
) -> str | None:
    approvals = [
        str(stage_id)
        for stage_id in stage["dependencies"]
        if spec.stage_by_id[str(stage_id)]["kind"] == "approval"
    ]
    if len(approvals) > 1:
        raise AutomationBlocked(
            f"Stage {stage['id']} has multiple approval dependencies."
        )
    return approvals[0] if approvals else None


def _render_output_root(
    stage: Mapping[str, Any], *, run_id: str
) -> str:
    template = str((stage.get("resources") or {}).get("expected_output_root") or "")
    if not template:
        raise AutomationBlocked(
            f"Slurm stage {stage['id']} lacks expected_output_root."
        )
    rendered = _render_runtime_template(template, run_id=run_id)
    path = Path(rendered)
    if path.is_absolute() or ".." in path.parts or rendered in {"", "."}:
        raise AutomationBlocked(f"Unsafe Slurm output root: {rendered!r}")
    return path.as_posix()


def _write_submission_block(
    store: RunStore,
    *,
    stage_id: str,
    result: CommandResult,
    output_root: str,
    attempt: int,
) -> None:
    stdout_path, stderr_path = _record_command(store, stage_id, attempt, result)
    reason = result.stderr.strip() or result.stdout.strip() or "Remote submission failed."
    store.transition(RunStatus.BLOCKED, reason=reason)
    _set_stop_reason(store, reason)
    write_blocked_report(
        store.run_dir,
        state=store.load(),
        failed_stage=stage_id,
        error_class="SlurmSubmissionFailure",
        return_code=result.returncode,
        stderr=result.stderr or result.stdout,
        artifacts=[output_root],
        retry_count=max(0, attempt - 1),
        recommended_action=(
            "Inspect the guarded exp_sbatch output. Do not resubmit until it "
            "is known whether a job ID was created."
        ),
        scientific_semantics_risk=False,
        details={
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "remote_write_performed": result.returncode == 0,
        },
    )


def submit(
    spec: TaskSpec,
    *,
    dry_run: bool,
    store: RunStore | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    compute_stages = _slurm_compute_stages(spec)
    if len(compute_stages) != 1:
        raise AutomationBlocked(
            "Phase B contract permits exactly one controlled Slurm smoke job."
        )
    stage = compute_stages[0]
    stage_id = str(stage["id"])
    if not dry_run:
        _smoke_submission_permissions(spec)
        if store is None:
            raise AutomationBlocked(
                "A persisted --run-dir is required for real submission."
            )
    command_runner = runner or CommandRunner(default_timeout_seconds=120)
    run_id = store.load()["run_id"] if store is not None else "DRY_RUN"
    output_root = _render_output_root(stage, run_id=str(run_id))
    commit = head_commit(command_runner, spec.local_root)
    approval_stage = _approval_dependency(spec, stage)
    if not dry_run:
        assert store is not None
        if store.load()["status"] in {
            RunStatus.BLOCKED.value,
            RunStatus.FAILED.value,
        }:
            raise AutomationBlocked(
                "Blocked/failed runs cannot be resubmitted automatically; "
                "review the report and create a new run if appropriate."
            )
        existing_jobs = list(store.load().get("slurm_jobs") or [])
        if existing_jobs:
            if len(existing_jobs) != 1:
                raise AutomationBlocked("More than one Slurm job is persisted.")
            return {
                "dry_run": False,
                "commands": [],
                "job_ids": {stage_id: existing_jobs[0]["job_id"]},
                "slurm_jobs": existing_jobs,
                "resumed": True,
            }
        if approval_stage is None or not store.is_approved(approval_stage):
            raise AutomationBlocked(
                f"Stage {stage_id} requires approval of {approval_stage or 'a boundary stage'}."
            )
        approval = store.load()["approvals"][approval_stage]
        if approval.get("git_commit") != commit:
            raise AutomationBlocked(
                "Approval commit differs from current HEAD; approve again after review."
            )
        for dependency in stage["dependencies"]:
            dependency_id = str(dependency)
            dependency_stage = spec.stage_by_id[dependency_id]
            if dependency_stage["kind"] == "approval":
                continue
            if not store.stage_succeeded(dependency_id):
                raise AutomationBlocked(
                    f"Stage dependency is not passed: {dependency_id}"
                )
        state = store.load()
        if state.get("local_commit") != commit or state.get("remote_commit") != commit:
            raise AutomationBlocked(
                "Checkpoint commit changed or remote preflight is stale; "
                "require matching local/remote commits before submission."
            )

    automation_environment = {
        "AUTOMATION_OUTPUT_ROOT": output_root,
        "AUTOMATION_SUBMITTED_COMMIT": commit,
        "PROJECT_ROOT": str(spec.data["project"]["remote_root"]),
    }
    runtime_stage = deepcopy(stage)
    runtime_stage["resources"] = {
        **dict(runtime_stage.get("resources") or {}),
        "expected_output_root": output_root,
    }
    runtime_stages = {**spec.stage_by_id, stage_id: runtime_stage}
    sbatch_argv = build_exp_sbatch_argv(
        runtime_stage,
        runtime_stages,
        {},
        automation_environment=automation_environment,
    )
    ssh_argv = build_remote_submit_argv(
        _ssh_config(spec),
        sbatch_argv,
        expected_commit=commit,
        output_root=output_root,
    )
    if dry_run:
        return {
            "dry_run": True,
            "commands": [ssh_argv],
            "sbatch_argv": sbatch_argv,
            "job_ids": {},
            "slurm_jobs": [],
            "output_root": output_root,
            "remote_write_performed": False,
        }

    assert store is not None
    dump_spec_snapshot(spec, _execution_snapshot_path(store))
    previous = store.load().get("stages", {}).get(stage_id, {})
    attempt = int(previous.get("attempt", 0)) + 1
    result = command_runner.run(
        ssh_argv,
        cwd=spec.local_root,
        environment=inherited_environment(preserve_proxy_environment=True),
    )
    if result.returncode != 0:
        _write_submission_block(
            store,
            stage_id=stage_id,
            result=result,
            output_root=output_root,
            attempt=attempt,
        )
        raise AutomationBlocked(result.stderr or "Remote exp_sbatch failed.")
    try:
        job_id = parse_exp_sbatch_job_id(result.stdout)
    except RuntimeError as exc:
        _write_submission_block(
            store,
            stage_id=stage_id,
            result=result,
            output_root=output_root,
            attempt=attempt,
        )
        raise AutomationBlocked(str(exc)) from exc
    submission_stdout, submission_stderr = _record_command(
        store, stage_id, attempt, result
    )
    submitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    remote_stdout = f"logs/clear_phase_b_smoke_{job_id}.out"
    remote_stderr = f"logs/clear_phase_b_smoke_{job_id}.err"
    completion_marker = (
        f"{output_root}/_AUTOMATION_PHASE_B_GPU_SMOKE_COMPLETE.json"
    )
    record = {
        "stage_id": stage_id,
        "job_id": job_id,
        "slurm_state": "SUBMITTED",
        "exit_code": None,
        "submitted_at": submitted_at,
        "submit_argv": sbatch_argv,
        "ssh_argv": ssh_argv,
        "submitted_commit": commit,
        "remote_commit": store.load().get("remote_commit"),
        "output_root": output_root,
        "stdout_path": remote_stdout,
        "stderr_path": remote_stderr,
        "completion_marker": completion_marker,
        "gate_result": None,
    }
    store.record_slurm_job(record)
    store.record_stage(
        stage_id,
        {
            **previous,
            "stage_id": stage_id,
            "attempt": attempt,
            "start_time": submitted_at,
            "end_time": None,
            "command_argv": ssh_argv,
            "submit_argv": sbatch_argv,
            "cwd": str(spec.local_root),
            "return_code": result.returncode,
            "submission_stdout_path": str(submission_stdout),
            "submission_stderr_path": str(submission_stderr),
            "stdout_path": remote_stdout,
            "stderr_path": remote_stderr,
            "artifacts": [output_root],
            "gate_result": None,
            "job_id": job_id,
            "git_commit": commit,
            "remote_git_commit": store.load().get("remote_commit"),
            "status": "SUBMITTED",
        },
    )
    store.transition(RunStatus.SUBMITTED, reason=f"Submitted Slurm job {job_id}.")
    _set_stop_reason(store, "Waiting for on-demand Slurm status refresh.")
    write_final_report(
        store.run_dir,
        state=store.load(),
        gate_summary="CLEAR Phase B smoke submitted; scientific gate has not run.",
        output_roots=[output_root],
        provenance={
            "submitted_commit": commit,
            "remote_commit": store.load().get("remote_commit"),
            "calibration_used": False,
            "test_used": False,
            "remote_write_performed": True,
        },
        next_allowed_stage="status --refresh",
        stop_reason="Waiting for on-demand Slurm status refresh.",
        details={"approval": store.load()["approvals"].get(approval_stage)},
    )
    return {
        "dry_run": False,
        "commands": [ssh_argv],
        "job_ids": {stage_id: job_id},
        "slurm_jobs": list(store.load()["slurm_jobs"]),
        "output_root": output_root,
        "remote_write_performed": True,
    }


def _gate_stage_for_compute(
    spec: TaskSpec, compute_stage_id: str
) -> dict[str, Any]:
    matches = [
        stage
        for stage in spec.data["stages"]
        if stage["kind"] == "audit"
        and list(stage["dependencies"]) == [compute_stage_id]
        and stage.get("command")
    ]
    if len(matches) != 1:
        raise AutomationBlocked(
            f"Expected one read-only audit stage after {compute_stage_id}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _parse_clear_phase_b_gate_evidence(stdout: str) -> dict[str, Any]:
    encoded_values = [
        line.split(maxsplit=1)[1]
        for line in stdout.splitlines()
        if line.startswith(_CLEAR_PHASE_B_GATE_MARKER + " ")
        and len(line.split(maxsplit=1)) == 2
    ]
    if len(encoded_values) != 1:
        raise ValueError(
            "Remote gate must emit exactly one Phase B evidence marker."
        )
    try:
        payload = json.loads(
            base64.b64decode(encoded_values[0], validate=True).decode("utf-8")
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Remote Phase B gate evidence is invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Remote Phase B gate evidence must be a JSON object.")
    return payload


def _update_persisted_job(
    store: RunStore,
    current: Mapping[str, Any],
    *,
    slurm_state: str,
    exit_code: str | None,
    gate_result: str | None = None,
) -> dict[str, Any]:
    updated = {
        **dict(current),
        "slurm_state": slurm_state,
        "exit_code": exit_code,
        "last_refreshed_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    if gate_result is not None:
        updated["gate_result"] = gate_result
    store.record_slurm_job(updated)
    return updated


def _block_slurm_run(
    store: RunStore,
    *,
    stage_id: str,
    error_class: str,
    return_code: int | None,
    stderr: str,
    output_root: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    reason = str(details.get("stop_reason") or error_class)
    store.transition(RunStatus.BLOCKED, reason=reason)
    _set_stop_reason(store, reason)
    report, _ = write_blocked_report(
        store.run_dir,
        state=store.load(),
        failed_stage=stage_id,
        error_class=error_class,
        return_code=return_code,
        stderr=stderr,
        artifacts=[output_root],
        retry_count=0,
        recommended_action=(
            "Inspect the persisted Slurm logs and Gate evidence; do not "
            "overwrite or automatically retry the output directory."
        ),
        scientific_semantics_risk=False,
        details={**dict(details), "remote_write_performed": True},
    )
    return {
        "state": store.load(),
        "status": RunStatus.BLOCKED.value,
        "report": str(report),
    }


def refresh_status(
    store: RunStore,
    spec: TaskSpec,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Perform one bounded Slurm refresh and, when ready, one read-only Gate."""

    state = store.load()
    if state["status"] in {RunStatus.BLOCKED.value, RunStatus.FAILED.value}:
        store.append_event("status_refresh_skipped_blocked_run")
        return {
            "state": state,
            "status": state["status"],
            "slurm_query": {"skipped": True, "reason": "run_is_blocked"},
        }
    gate_records = [
        record
        for stage_id, record in state.get("stages", {}).items()
        if stage_id == "phase_b_gpu_smoke_gate"
        and record.get("status") == "PASSED"
    ]
    if gate_records:
        store.append_event("status_refresh_skipped_completed_gate")
        return {"state": store.load(), "resumed": True, "slurm_query": {"skipped": True}}
    jobs = list(state.get("slurm_jobs") or [])
    if not jobs:
        return {"state": state, "slurm_query": {"jobs": [], "skipped": True}}
    if len(jobs) != 1:
        raise AutomationBlocked("Phase B permits exactly one persisted Slurm job.")
    job = jobs[0]
    stage_id = str(job["stage_id"])
    job_id = str(job["job_id"])
    command_runner = runner or CommandRunner(default_timeout_seconds=120)
    current_commit = head_commit(command_runner, spec.local_root)
    submitted_commit = str(job.get("submitted_commit") or "")
    if current_commit != submitted_commit:
        return _block_slurm_run(
            store,
            stage_id=stage_id,
            error_class="CheckpointCommitChanged",
            return_code=None,
            stderr=(
                f"Current commit {current_commit} differs from submitted "
                f"commit {submitted_commit}."
            ),
            output_root=str(job["output_root"]),
            details={
                "stop_reason": "checkpoint_commit_changed",
                "current_commit": current_commit,
                "submitted_commit": submitted_commit,
            },
        )
    status_argv = build_status_argv(_ssh_config(spec), [job_id])
    result = command_runner.run(
        status_argv,
        cwd=spec.local_root,
        environment=inherited_environment(preserve_proxy_environment=True),
    )
    attempt = int(job.get("status_refresh_count", 0)) + 1
    status_stdout, status_stderr = _record_command(
        store, f"{stage_id}_status", attempt, result
    )
    if result.returncode != 0:
        return _block_slurm_run(
            store,
            stage_id=stage_id,
            error_class="SlurmStatusQueryFailure",
            return_code=result.returncode,
            stderr=result.stderr,
            output_root=str(job["output_root"]),
            details={
                "stop_reason": "slurm_status_query_failed",
                "status_stdout_path": str(status_stdout),
                "status_stderr_path": str(status_stderr),
            },
        )
    try:
        slurm_status = parse_slurm_status_output(result.stdout, job_id)
    except ValueError as exc:
        return _block_slurm_run(
            store,
            stage_id=stage_id,
            error_class="SlurmStatusParseFailure",
            return_code=result.returncode,
            stderr=str(exc),
            output_root=str(job["output_root"]),
            details={"stop_reason": "slurm_status_unavailable"},
        )
    job = {
        **job,
        "status_refresh_count": attempt,
        "status_query_argv": status_argv,
        "status_stdout_path": str(status_stdout),
        "status_stderr_path": str(status_stderr),
    }
    job = _update_persisted_job(
        store,
        job,
        slurm_state=slurm_status.state,
        exit_code=slurm_status.exit_code,
    )
    previous_stage = store.load().get("stages", {}).get(stage_id, {})
    stage_status = "RUNNING" if slurm_status.state == "RUNNING" else slurm_status.state
    store.record_stage(
        stage_id,
        {
            **previous_stage,
            "status": stage_status,
            "slurm_state": slurm_status.state,
            "slurm_exit_code": slurm_status.exit_code,
            "return_code": (
                0 if slurm_status.exit_code == "0:0" else previous_stage.get("return_code")
            ),
        },
    )
    if is_active_status(slurm_status):
        run_status = (
            RunStatus.RUNNING
            if slurm_status.state in {"RUNNING", "COMPLETING"}
            else RunStatus.SUBMITTED
        )
        store.transition(run_status, reason=f"Slurm job is {slurm_status.state}.")
        _set_stop_reason(store, "Waiting for on-demand Slurm status refresh.")
        write_final_report(
            store.run_dir,
            state=store.load(),
            gate_summary=f"Slurm job {job_id} is {slurm_status.state}; Gate not run.",
            output_roots=[str(job["output_root"])],
            provenance={
                "submitted_commit": submitted_commit,
                "remote_commit": state.get("remote_commit"),
                "calibration_used": False,
                "test_used": False,
            },
            next_allowed_stage="status --refresh",
            stop_reason="Waiting for on-demand Slurm status refresh.",
            details={
                "poll_interval_seconds": int(spec.data["execution"]["poll_interval_seconds"]),
                "gate_executed": False,
            },
        )
        return {
            "state": store.load(),
            "slurm_query": {
                "job_id": slurm_status.job_id,
                "state": slurm_status.state,
                "exit_code": slurm_status.exit_code,
            },
            "next_refresh_after_seconds": int(
                spec.data["execution"]["poll_interval_seconds"]
            ),
        }
    if is_failure_status(slurm_status):
        return _block_slurm_run(
            store,
            stage_id=stage_id,
            error_class=f"Slurm{slurm_status.state.title().replace('_', '')}",
            return_code=1,
            stderr=(
                f"Slurm state={slurm_status.state} "
                f"exit_code={slurm_status.exit_code}"
            ),
            output_root=str(job["output_root"]),
            details={
                "stop_reason": f"slurm_{slurm_status.state.lower()}",
                "slurm_state": slurm_status.state,
                "slurm_exit_code": slurm_status.exit_code,
                "stdout_path": job.get("stdout_path"),
                "stderr_path": job.get("stderr_path"),
                "gate_executed": False,
            },
        )
    if not is_success_status(slurm_status):
        return _block_slurm_run(
            store,
            stage_id=stage_id,
            error_class="UnsupportedSlurmState",
            return_code=None,
            stderr=f"Unsupported Slurm state: {slurm_status.state}",
            output_root=str(job["output_root"]),
            details={"stop_reason": "unsupported_slurm_state"},
        )

    store.transition(
        RunStatus.AUDITING,
        reason=f"Slurm job {job_id} completed; running read-only smoke Gate.",
    )
    gate_stage = _gate_stage_for_compute(spec, stage_id)
    gate_stage_id = str(gate_stage["id"])
    rendered_command = [
        _render_runtime_template(
            str(value),
            run_id=str(state["run_id"]),
            output_root=str(job["output_root"]),
            job_id=job_id,
            commit=submitted_commit,
            remote_root=str(spec.data["project"]["remote_root"]),
        )
        for value in gate_stage["command"]
    ]
    gate_argv = build_remote_project_command_argv(
        _ssh_config(spec), rendered_command
    )
    gate_result = command_runner.run(
        gate_argv,
        cwd=spec.local_root,
        environment=inherited_environment(preserve_proxy_environment=True),
    )
    gate_attempt = int(
        store.load().get("stages", {}).get(gate_stage_id, {}).get("attempt", 0)
    ) + 1
    gate_stdout, gate_stderr = _record_command(
        store, gate_stage_id, gate_attempt, gate_result
    )
    try:
        evidence = _parse_clear_phase_b_gate_evidence(gate_result.stdout)
    except ValueError as exc:
        evidence = {
            "audit_passed": False,
            "run_complete": False,
            "failed_hard_checks": [f"evidence_parse_failure:{exc}"],
            "checks": {},
            "artifacts": {},
            "provenance": {},
        }
    evidence.update(
        {
            "task_id": spec.task_id,
            "run_id": state["run_id"],
            "stage_id": gate_stage_id,
            "job_id": job_id,
            "slurm_state": slurm_status.state,
            "slurm_exit_code": slurm_status.exit_code,
            "submit_argv": job.get("submit_argv"),
            "submitted_commit": submitted_commit,
            "remote_commit": state.get("remote_commit"),
            "stdout_path": job.get("stdout_path"),
            "stderr_path": job.get("stderr_path"),
            "completion_marker": job.get("completion_marker"),
            "approval": state.get("approvals", {}).get(
                _approval_dependency(spec, spec.stage_by_id[stage_id]) or ""
            ),
            "approved_stage": _approval_dependency(
                spec, spec.stage_by_id[stage_id]
            ),
            "remote_write_performed": True,
            "gate_remote_write_performed": False,
        }
    )
    reported_failures = list(evidence.get("failed_hard_checks") or [])
    evaluation = evaluate_gate_payload(
        evidence,
        gate_spec=gate_stage["gate"],
        slurm_exit_code=slurm_status.exit_code,
    )
    combined_failures = [*reported_failures, *evaluation.failed_hard_checks]
    if gate_result.returncode != 0:
        combined_failures.append(f"gate_return_code:{gate_result.returncode}")
    if combined_failures:
        evaluation = type(evaluation)(
            passed=False,
            failed_hard_checks=tuple(dict.fromkeys(combined_failures)),
            checks=evaluation.checks,
        )
    evidence["audit_passed"] = evaluation.passed
    evidence["run_complete"] = evaluation.passed
    evidence["failed_hard_checks"] = list(evaluation.failed_hard_checks)
    evidence_path = store.run_dir / "evidence" / "phase_b_gpu_smoke_gate.json"
    atomic_write_json(evidence_path, evidence)
    store.record_stage(
        gate_stage_id,
        {
            "stage_id": gate_stage_id,
            "attempt": gate_attempt,
            "start_time": None,
            "end_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command_argv": gate_argv,
            "cwd": str(spec.local_root),
            "return_code": gate_result.returncode,
            "stdout_path": str(gate_stdout),
            "stderr_path": str(gate_stderr),
            "artifacts": [
                _render_runtime_template(
                    str(value),
                    run_id=str(state["run_id"]),
                    output_root=str(job["output_root"]),
                    job_id=job_id,
                    commit=submitted_commit,
                    remote_root=str(spec.data["project"]["remote_root"]),
                )
                for value in gate_stage["expected_artifacts"]
            ],
            "gate_result": str(evidence_path),
            "job_id": None,
            "git_commit": current_commit,
            "remote_git_commit": state.get("remote_commit"),
            "command_status": "PASSED" if gate_result.returncode == 0 else "FAILED",
            "gate_status": "PASSED" if evaluation.passed else "BLOCKED",
            "status": "PASSED" if evaluation.passed else "FAILED",
        },
    )
    job = _update_persisted_job(
        store,
        job,
        slurm_state=slurm_status.state,
        exit_code=slurm_status.exit_code,
        gate_result=str(evidence_path),
    )
    if not evaluation.passed:
        return _block_slurm_run(
            store,
            stage_id=gate_stage_id,
            error_class="PhaseBGpuSmokeGateFailure",
            return_code=gate_result.returncode,
            stderr=gate_result.stderr or "\n".join(evaluation.failed_hard_checks),
            output_root=str(job["output_root"]),
            details={
                "stop_reason": "phase_b_gpu_smoke_gate_failed",
                "gate_result": str(evidence_path),
                "failed_hard_checks": list(evaluation.failed_hard_checks),
                "slurm_state": slurm_status.state,
                "slurm_exit_code": slurm_status.exit_code,
            },
        )

    next_stage = _next_stage_id(spec, gate_stage_id)
    store.transition(
        RunStatus.STOPPED_BEFORE_APPROVAL,
        reason=f"Stopped before {next_stage} pending explicit approval.",
    )
    _set_stop_reason(
        store, f"Stopped before {next_stage} pending explicit approval."
    )
    report, _ = write_final_report(
        store.run_dir,
        state=store.load(),
        gate_summary="CLEAR Phase B GPU smoke and scientific artifact Gate passed.",
        output_roots=[str(job["output_root"])],
        provenance={
            **dict(evidence.get("provenance") or {}),
            "submitted_commit": submitted_commit,
            "remote_commit": state.get("remote_commit"),
            "job_id": job_id,
            "remote_write_performed": True,
        },
        next_allowed_stage=next_stage,
        stop_reason=f"Stopped before {next_stage} pending explicit approval.",
        details={
            "approval": state.get("approvals", {}).get(
                _approval_dependency(spec, spec.stage_by_id[stage_id]) or ""
            ),
            "gate_result": str(evidence_path),
            "slurm_state": slurm_status.state,
            "slurm_exit_code": slurm_status.exit_code,
            "completion_marker": job.get("completion_marker"),
        },
    )
    return {
        "state": store.load(),
        "status": RunStatus.STOPPED_BEFORE_APPROVAL.value,
        "report": str(report),
        "gate_result": str(evidence_path),
        "next_stage": next_stage,
    }


def resume_run(
    store: RunStore, *, runner: CommandRunner | None = None
) -> dict[str, Any]:
    """Advance at most one resumable action; never poll in a loop."""

    spec = _load_execution_spec(store)
    if store.load().get("slurm_jobs"):
        return refresh_status(store, spec, runner=runner)
    compute_stages = _slurm_compute_stages(spec)
    if compute_stages:
        approval_stage = _approval_dependency(spec, compute_stages[0])
        if approval_stage and store.is_approved(approval_stage):
            result = submit(spec, dry_run=False, store=store, runner=runner)
            return {"state": store.load(), "submission": result}
    store.append_event("resume_stopped_no_approved_remote_action")
    return {"state": store.load(), "resumed": True, "action": "none"}


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
    deploy_mode = deploy_parser.add_mutually_exclusive_group()
    deploy_mode.add_argument("--dry-run", action="store_true")
    deploy_mode.add_argument("--preflight-only", action="store_true")
    deploy_parser.add_argument("--run-dir")
    adopt_parser = subparsers.add_parser("adopt-existing")
    adopt_parser.add_argument("spec")
    adopt_parser.add_argument("--dry-run", action="store_true")
    adopt_parser.add_argument("--run-dir")
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
            result = deploy(
                load_task_spec(args.spec),
                dry_run=args.dry_run,
                preflight_only=args.preflight_only,
                run_dir=args.run_dir,
            )
            _json_print(result)
            return int(result.get("return_code", 0))
        elif args.command == "adopt-existing":
            result = adopt_existing(
                load_task_spec(args.spec),
                dry_run=args.dry_run,
                run_dir=args.run_dir,
            )
            _json_print(result)
            return int(result.get("return_code", 0))
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
            spec = _load_execution_spec(store)
            if args.stage not in spec.stage_by_id:
                raise AutomationBlocked(f"Unknown stage: {args.stage}")
            stage = spec.stage_by_id[args.stage]
            if stage["kind"] != "approval":
                raise AutomationBlocked(
                    f"Stage is not an approval boundary: {args.stage}"
                )
            for dependency in stage["dependencies"]:
                if not store.stage_succeeded(str(dependency)):
                    raise AutomationBlocked(
                        f"Approval dependency is not passed: {dependency}"
                    )
            commit = head_commit(CommandRunner(), spec.local_root)
            store.approve(
                args.stage,
                args.reason,
                getpass.getuser(),
                git_commit=commit,
            )
            previous = store.load().get("stages", {}).get(args.stage, {})
            store.record_stage(
                args.stage,
                {
                    **previous,
                    "stage_id": args.stage,
                    "attempt": int(previous.get("attempt", 0)) + 1,
                    "start_time": None,
                    "end_time": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "command_argv": [],
                    "cwd": str(spec.local_root),
                    "return_code": 0,
                    "stdout_path": None,
                    "stderr_path": None,
                    "artifacts": [],
                    "gate_result": None,
                    "job_id": None,
                    "git_commit": commit,
                    "remote_git_commit": store.load().get("remote_commit"),
                    "command_status": "NOT_EXECUTED",
                    "gate_status": "PASSED",
                    "status": "APPROVED",
                },
            )
            store.transition(
                RunStatus.WAITING_APPROVAL,
                reason=f"{args.stage} approved; submission remains separate.",
            )
            _set_stop_reason(
                store, f"{args.stage} approved; invoke submit or resume."
            )
            _json_print(
                {"approved": True, "stage": args.stage, "git_commit": commit}
            )
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
            result = resume_run(store)
            _json_print(
                {
                    "run_dir": str(store.run_dir),
                    "status": store.load()["status"],
                    **{key: value for key, value in result.items() if key != "state"},
                }
            )
        elif args.command == "status":
            store = RunStore.open(args.run_dir)
            state = store.load()
            payload: dict[str, Any] = {"state": state}
            if args.refresh:
                result = refresh_status(
                    store,
                    _load_execution_spec(store),
                )
                payload = result
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
