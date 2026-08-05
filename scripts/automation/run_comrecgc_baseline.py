#!/usr/bin/env python3
"""Recoverable COMRECGC Slurm-chain driver.

The local ``--remote`` mode delegates to this same script on HPC.  Slurm jobs
are always submitted through ``scripts/exp_sbatch.sh`` and are connected by
dependencies; this process never polls for hours.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import append_jsonl, write_json  # noqa: E402

DEFAULT_REMOTE_HOST = "u20526@logini.tongji.edu.cn"
DEFAULT_REMOTE_PORT = 10022
DEFAULT_REMOTE_ROOT = "/share/home/u20526/czx/counterfactual-subgraph"
DEFAULT_CONTROL_SOCKET = "/tmp/tongji-codex.sock"
REGISTERED_JOB_ID_RE = re.compile(r"^job_id=(\d+)$", re.MULTILINE)


@dataclass(frozen=True)
class JobSpec:
    stage: str
    dataset: str
    script: str
    output_root: str
    dependency: str | None
    environment: dict[str, str]


class AutomationError(RuntimeError):
    pass


class RunState:
    def __init__(self, root: Path, *, run_id: str, datasets: Sequence[str], mode: str) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        if self.state_path.exists():
            self.data = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 1,
                "run_id": run_id,
                "datasets": list(datasets),
                "mode": mode,
                "status": "CREATED",
                "jobs": [],
                "completed_stages": [],
                "failed_stage": None,
                "created_at": now(),
            }
            self.save()
            self.event("CREATED", {"datasets": list(datasets), "mode": mode})

    def save(self) -> None:
        self.data["updated_at"] = now()
        write_json(self.state_path, self.data)

    def event(self, event: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.events_path,
            {"time": now(), "event": event, "payload": payload},
        )

    def transition(self, status: str, **payload: Any) -> None:
        self.data["status"] = status
        self.data.update(payload)
        self.save()
        self.event("STATE_CHANGED", {"status": status, **payload})

    def find_job(self, stage: str, dataset: str) -> dict[str, Any] | None:
        return next(
            (
                record
                for record in self.data["jobs"]
                if record["stage"] == stage and record["dataset"] == dataset
            ),
            None,
        )

    def add_job(self, spec: JobSpec, job_id: str, argv: Sequence[str]) -> dict[str, Any]:
        existing = self.find_job(spec.stage, spec.dataset)
        if existing is not None:
            return existing
        record = {
            "job_id": str(job_id),
            "stage": spec.stage,
            "dataset": spec.dataset,
            "script": spec.script,
            "output_root": spec.output_root,
            "dependency": spec.dependency,
            "environment": dict(spec.environment),
            "submit_argv": list(argv),
            "submitted_at": now(),
            "slurm_state": "SUBMITTED",
            "exit_code": None,
        }
        self.data["jobs"].append(record)
        self.save()
        self.event("JOB_SUBMITTED", record)
        return record


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 600,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ssh_argv(args: argparse.Namespace, remote_script: str) -> list[str]:
    values = ["ssh"]
    if args.control_socket:
        values.extend(["-S", args.control_socket])
    values.extend(
        [
            "-p",
            str(args.remote_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ClearAllForwardings=yes",
            args.remote_host,
            "--",
            "bash",
            "-lc",
            remote_script,
        ]
    )
    return values


def run_remote(args: argparse.Namespace, remote_argv: Sequence[str], *, timeout: int = 1800) -> str:
    script = "cd " + shlex.quote(args.remote_root) + " && " + shlex.join(list(remote_argv))
    result = run_command(ssh_argv(args, script), cwd=PROJECT_ROOT, timeout=timeout)
    return result.stdout


def git_commit() -> str:
    return run_command(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).stdout.strip()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.remote:
        script = (
            "cd "
            + shlex.quote(args.remote_root)
            + " && printf '[COMRECGC_REMOTE_PREFLIGHT]\\n' && hostname && git branch --show-current && git rev-parse HEAD && command -v sbatch && command -v sacct"
        )
        output = run_command(ssh_argv(args, script), cwd=PROJECT_ROOT, timeout=60).stdout
        return {"remote": True, "output": output.splitlines(), "local_commit": git_commit()}
    for command in ([sys.executable, "-m", "py_compile", __file__],):
        run_command(command, cwd=PROJECT_ROOT)
    return {"remote": False, "hostname": socket.gethostname(), "commit": git_commit()}


def exact_git_commit_and_push(message: str) -> str:
    allowed = [
        ".gitignore",
        "src/baselines/comrecgc",
        "scripts/baselines/comrecgc",
        "scripts/slurm/comrecgc_dataset_identity.sh",
        "scripts/slurm/comrecgc_native_smoke.sh",
        "scripts/slurm/comrecgc_project_generate.sh",
        "scripts/slurm/comrecgc_common_recourse.sh",
        "scripts/slurm/comrecgc_export.sh",
        "scripts/slurm/comrecgc_unified_eval.sh",
        "scripts/slurm/comrecgc_gate.sh",
        "scripts/automation/run_comrecgc_baseline.py",
        "tests/baselines/comrecgc",
        "docs/baselines/COMRECGC.md",
        "docs/decisions.md",
    ]
    run_command(["git", "add", "--", *allowed], cwd=PROJECT_ROOT)
    run_command(["git", "diff", "--cached", "--check"], cwd=PROJECT_ROOT)
    staged = run_command(["git", "diff", "--cached", "--name-only"], cwd=PROJECT_ROOT).stdout
    if not staged.strip():
        return git_commit()
    run_command(["git", "commit", "-m", message], cwd=PROJECT_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=PROJECT_ROOT).stdout.strip()
    run_command(["git", "push", "origin", f"HEAD:{branch}"], cwd=PROJECT_ROOT, timeout=600)
    return git_commit()


def submit_job(state: RunState, spec: JobSpec, *, dry_run: bool) -> dict[str, Any]:
    existing = state.find_job(spec.stage, spec.dataset)
    if existing is not None:
        return existing
    # exp_sbatch.py records and parses Slurm's standard
    # ``Submitted batch job <id>`` output.  Passing --parsable changes that
    # output to a bare number and causes the registry to record UNKNOWN.
    sbatch_args: list[str] = []
    if spec.dependency:
        sbatch_args.append(f"--dependency={spec.dependency}")
    exports = {"PROJECT_ROOT": str(PROJECT_ROOT), **spec.environment}
    sbatch_args.extend(
        [
            "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
            spec.script,
        ]
    )
    argv = [
        "scripts/exp_sbatch.sh",
        "--name",
        f"comrecgc_{spec.dataset}_{spec.stage}",
        "--tags",
        f"COMRECGC,{spec.dataset},{spec.stage}",
        "--notes",
        "Pinned COMRECGC project adaptation; no selection in unified eval",
        "--expected-output-root",
        spec.output_root,
        "--dataset",
        spec.dataset,
        "--method",
        "COMRECGC",
        "--metric",
        "MolCLR-Node-Wasserstein strict_flip CCRCOV",
    ]
    if dry_run:
        argv.append("--dry-run")
    argv.extend(["--", *sbatch_args])
    result = run_command(argv, cwd=PROJECT_ROOT, timeout=120)
    if dry_run:
        job_id = f"DRYRUN_{spec.dataset}_{spec.stage}"
    else:
        match = REGISTERED_JOB_ID_RE.search(result.stdout)
        if match is None:
            raise AutomationError(
                f"Could not parse Slurm job ID for {spec.dataset}/{spec.stage}: "
                f"stdout={result.stdout[-500:]!r}, stderr={result.stderr[-500:]!r}"
            )
        job_id = match.group(1)
    return state.add_job(spec, job_id, argv)


def _chain_for_dataset(
    state: RunState,
    *,
    dataset: str,
    profile: str,
    run_id: str,
    initial_dependency: str,
    dry_run: bool,
) -> str:
    base = f"outputs/hpc/baselines/comrecgc/{dataset}/{profile}_{run_id}"
    previous = initial_dependency
    stages = [
        ("generation", "scripts/slurm/comrecgc_project_generate.sh", {}),
        ("common_recourse", "scripts/slurm/comrecgc_common_recourse.sh", {}),
        (
            "export",
            "scripts/slurm/comrecgc_export.sh",
            {"REQUIRE_TOP_K": "true" if profile == "full" else "false"},
        ),
        ("eval", "scripts/slurm/comrecgc_unified_eval.sh", {}),
    ]
    all_ids: list[str] = []
    for stage, script, extra in stages:
        dependency = f"afterok:{previous}" if previous else None
        spec = JobSpec(
            stage=f"{profile}_{stage}",
            dataset=dataset,
            script=script,
            output_root=f"{base}/{stage if stage != 'eval' else 'eval'}",
            dependency=dependency,
            environment={"DATASET": dataset, "MODE": profile, "BASE_ROOT": base, **extra},
        )
        record = submit_job(state, spec, dry_run=dry_run)
        previous = record["job_id"]
        all_ids.append(previous)
    gate_spec = JobSpec(
        stage=f"{profile}_gate",
        dataset=dataset,
        script="scripts/slurm/comrecgc_gate.sh",
        output_root=base,
        dependency="afterany:" + ":".join(all_ids),
        environment={"DATASET": dataset, "MODE": profile, "BASE_ROOT": base},
    )
    return submit_job(state, gate_spec, dry_run=dry_run)["job_id"]


def submit_chains(args: argparse.Namespace, state: RunState) -> None:
    identity = submit_job(
        state,
        JobSpec(
            stage="dataset_identity",
            dataset="shared",
            script="scripts/slurm/comrecgc_dataset_identity.sh",
            output_root="outputs/hpc/baselines/comrecgc/audits",
            dependency=None,
            environment={},
        ),
        dry_run=args.dry_run,
    )
    identity_id = identity["job_id"]
    for dataset in args.datasets:
        native_root = (
            f"outputs/hpc/baselines/comrecgc/native_smoke/{dataset}/"
            f"{state.data['run_id']}"
        )
        submit_job(
            state,
            JobSpec(
                stage="native_smoke",
                dataset=dataset,
                script="scripts/slurm/comrecgc_native_smoke.sh",
                output_root=native_root,
                dependency=f"afterok:{identity_id}",
                environment={"NATIVE_DATASET": dataset, "OUTPUT_DIR": native_root},
            ),
            dry_run=args.dry_run,
        )
        smoke_gate = _chain_for_dataset(
            state,
            dataset=dataset,
            profile="smoke",
            run_id=state.data["run_id"],
            initial_dependency=identity_id,
            dry_run=args.dry_run,
        )
        if args.mode in {"full", "all"}:
            if not args.after_smoke_pass:
                raise AutomationError("Full submission requires --after-smoke-pass.")
            _chain_for_dataset(
                state,
                dataset=dataset,
                profile="full",
                run_id=state.data["run_id"],
                initial_dependency=smoke_gate,
                dry_run=args.dry_run,
            )
    state.transition("FULL_SUBMITTED" if args.mode in {"full", "all"} else "SMOKE_SUBMITTED")


def refresh(args: argparse.Namespace, state: RunState) -> None:
    job_ids = [record["job_id"] for record in state.data["jobs"] if str(record["job_id"]).isdigit()]
    if not job_ids:
        return
    result = run_command(
        ["sacct", "-n", "-P", "-j", ",".join(job_ids), "--format=JobIDRaw,State,ExitCode"],
        cwd=PROJECT_ROOT,
        timeout=60,
    )
    statuses: dict[str, tuple[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] in job_ids:
            statuses[parts[0]] = (parts[1], parts[2])
    for record in state.data["jobs"]:
        if record["job_id"] in statuses:
            record["slurm_state"], record["exit_code"] = statuses[record["job_id"]]
    state.save()
    run_command([sys.executable, "scripts/sync_experiment_status.py"], cwd=PROJECT_ROOT, timeout=120)
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    states = {record["slurm_state"].split("+")[0] for record in state.data["jobs"]}
    if states and states <= terminal:
        failed = [record for record in state.data["jobs"] if not (record["slurm_state"] == "COMPLETED" and record["exit_code"] == "0:0")]
        state.transition("BLOCKED" if failed else "JOBS_COMPLETED", failed_jobs=failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="aids,mutagenicity")
    parser.add_argument("--mode", choices=("smoke", "full", "all", "status"), default="smoke")
    parser.add_argument("--after-smoke-pass", action="store_true")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--control-socket", default=DEFAULT_CONTROL_SOCKET)
    parser.add_argument("--run-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-git-write", action="store_true")
    parser.add_argument("--skip-remote-sync", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    if not args.datasets or any(value not in {"aids", "mutagenicity"} for value in args.datasets):
        raise AutomationError("Datasets must be a comma-separated subset of aids,mutagenicity.")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    preflight_payload = preflight(args)
    if args.allow_git_write:
        preflight_payload["committed_sha"] = exact_git_commit_and_push(
            "feat: add pinned COMRECGC AIDS and Mutagenicity baseline"
        )
    if args.remote:
        if not args.skip_remote_sync:
            run_remote(args, ["git", "pull", "--ff-only", "origin", "baseline/comrecgc-aids-mut"])
        remote_args = [
            sys.executable,
            "scripts/automation/run_comrecgc_baseline.py",
            "--datasets",
            ",".join(args.datasets),
            "--mode",
            args.mode,
            "--run-id",
            run_id,
        ]
        if args.after_smoke_pass:
            remote_args.append("--after-smoke-pass")
        if args.dry_run:
            remote_args.append("--dry-run")
        output = run_remote(args, remote_args)
        print(output, end="")
        return 0
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else PROJECT_ROOT / "outputs/hpc/automation/comrecgc" / run_id
    )
    state = RunState(run_dir, run_id=run_id, datasets=args.datasets, mode=args.mode)
    state.event("PREFLIGHT", preflight_payload)
    if args.mode == "status":
        refresh(args, state)
    else:
        submit_chains(args, state)
    write_json(
        run_dir / "final_report.json",
        {
            "run_id": run_id,
            "status": state.data["status"],
            "jobs": state.data["jobs"],
            "state_path": str(state.state_path),
            "events_path": str(state.events_path),
        },
    )
    (run_dir / "final_report.md").write_text(
        "# COMRECGC Automation\n\n"
        f"- Run ID: {run_id}\n"
        f"- Status: {state.data['status']}\n"
        f"- Jobs: {len(state.data['jobs'])}\n"
        f"- State: {state.state_path}\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "status": state.data["status"], "jobs": state.data["jobs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error_class": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        raise
