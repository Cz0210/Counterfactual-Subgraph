#!/usr/bin/env python3
"""Idempotent local/HPC driver for the pinned COMRECGC recovery protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
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

from src.baselines.comrecgc.artifact_resolution import resolve_recovery_artifacts  # noqa: E402
from src.baselines.comrecgc.contracts import (  # noqa: E402
    UPSTREAM_COMMIT,
    append_jsonl,
    write_json,
)
from src.baselines.comrecgc.upstream import validate_upstream_checkout  # noqa: E402

DEFAULT_REMOTE_HOST = "u20526@logini.tongji.edu.cn"
DEFAULT_REMOTE_PORT = 10022
DEFAULT_CONTROL_SOCKET = "/tmp/tongji-codex.sock"
DEFAULT_REMOTE_ROOT = "/share/home/u20526/czx/worktrees/comrecgc-recovery-20260806"
JOB_ID_RE = re.compile(r"^job_id=(\d+)$", re.MULTILINE)
ALLOWED_DYNAMIC_DIRTY_PATHS = frozenset({"docs/EXPERIMENT_LOG.md"})


@dataclass(frozen=True)
class Stage:
    stage_id: str
    dataset: str
    script: str
    output_root: str
    dependency_type: str | None
    dependency_stages: tuple[str, ...]
    environment: dict[str, str]
    defer_until_dependencies_complete: bool = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(
    argv: Sequence[str], *, cwd: Path, timeout: int = 600, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class RecoveryState:
    def __init__(
        self,
        root: Path,
        run_id: str,
        *,
        requested_mode: str,
        datasets: Sequence[str],
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "evidence").mkdir(exist_ok=True)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.jobs_path = self.root / "jobs.json"
        if self.state_path.exists():
            self.data = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 1,
                "run_id": run_id,
                "requested_mode": requested_mode,
                "datasets": sorted(str(value) for value in datasets),
                "status": "CREATED",
                "project_commit": git_commit(PROJECT_ROOT),
                "upstream_commit": UPSTREAM_COMMIT,
                "jobs": [],
                "adopted_stages": [],
                "completed_stages": [],
                "failed_stages": [],
                "created_at": now(),
            }
            self.save()
            self.event("RUN_CREATED", {"run_id": run_id})
        self.data.setdefault("adopted_stages", [])

    def save(self) -> None:
        self.data["updated_at"] = now()
        write_json(self.state_path, self.data)
        write_json(self.jobs_path, {"jobs": self.data["jobs"]})

    def event(self, name: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.events_path, {"time": now(), "event": name, "payload": payload})

    def transition(self, status: str, **values: Any) -> None:
        self.data.update(values)
        self.data["status"] = status
        self.save()
        self.event("STATE_CHANGED", {"status": status, **values})

    def job(self, stage_id: str) -> dict[str, Any] | None:
        return next(
            (row for row in self.data["jobs"] if row["stage_id"] == stage_id),
            None,
        )

    def add_job(self, stage: Stage, job_id: str, argv: Sequence[str]) -> dict[str, Any]:
        existing = self.job(stage.stage_id)
        if existing is not None:
            return existing
        value = {
            "stage_id": stage.stage_id,
            "dataset": stage.dataset,
            "job_id": str(job_id),
            "script": stage.script,
            "expected_output_root": stage.output_root,
            "dependency_type": stage.dependency_type,
            "dependency_stages": list(stage.dependency_stages),
            "submit_argv": list(argv),
            "submitted_at": now(),
            "slurm_state": "SUBMITTED",
            "exit_code": None,
        }
        self.data["jobs"].append(value)
        self.save()
        self.event("JOB_SUBMITTED", value)
        return value


def git_commit(root: Path) -> str:
    return command(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()


def partition_recovery_dirty(lines: Sequence[str]) -> tuple[list[str], list[str]]:
    """Allow only the registry-generated experiment log in a recovery worktree."""

    allowed: list[str] = []
    blocked: list[str] = []
    for line in lines:
        relative = line[3:] if len(line) >= 4 else ""
        target = allowed if relative in ALLOWED_DYNAMIC_DIRTY_PATHS else blocked
        target.append(line)
    return allowed, blocked


def ssh_argv(args: argparse.Namespace, script: str) -> list[str]:
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
            script,
        ]
    )
    return values


def delegate_remote(args: argparse.Namespace) -> int:
    remote = [
        "python",
        "scripts/automation/run_comrecgc_recovery.py",
        "--mode",
        args.mode,
        "--datasets",
        args.datasets,
        "--run-id",
        args.run_id,
    ]
    if args.dry_run:
        remote.append("--dry-run")
    script = "cd " + shlex.quote(args.remote_root) + " && " + shlex.join(remote)
    result = command(ssh_argv(args, script), cwd=PROJECT_ROOT, timeout=1800, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return int(result.returncode)


def preflight(state: RecoveryState) -> dict[str, Any]:
    branch = command(["git", "branch", "--show-current"], cwd=PROJECT_ROOT).stdout.strip()
    commit = git_commit(PROJECT_ROOT)
    dirty = command(["git", "status", "--short"], cwd=PROJECT_ROOT).stdout.splitlines()
    allowed_dynamic_dirty, blocked_dirty = partition_recovery_dirty(dirty)
    if blocked_dirty:
        raise RuntimeError(
            "Recovery HPC worktree is dirty outside the exact dynamic allowlist: "
            f"{blocked_dirty}"
        )
    if branch != "baseline/comrecgc-recovery-20260806":
        raise RuntimeError(f"Unexpected recovery branch: {branch}")
    upstream = validate_upstream_checkout(PROJECT_ROOT / "external/COMRECGC")
    artifact_resolution = resolve_recovery_artifacts(
        outputs_root=PROJECT_ROOT / "outputs/hpc/baselines/comrecgc",
        output_path=state.root / "evidence/artifact_resolution.json",
    )
    adopt_resolved_artifacts(state, artifact_resolution)
    evidence = {
        "branch": branch,
        "project_commit": commit,
        "upstream_root": str(upstream),
        "upstream_commit": UPSTREAM_COMMIT,
        "artifact_resolution_passed": artifact_resolution["resolution_passed"],
        "adopted_stage_count": len(state.data["adopted_stages"]),
        "allowed_dynamic_dirty": allowed_dynamic_dirty,
        "blocked_dirty": blocked_dirty,
        "sbatch": shutil.which("sbatch") or "",
        "sacct": shutil.which("sacct") or "",
    }
    if not evidence["sbatch"] or not evidence["sacct"]:
        raise RuntimeError("HPC preflight requires both sbatch and sacct on PATH.")
    write_json(state.root / "evidence/preflight.json", evidence)
    state.transition("PREFLIGHT_PASSED", preflight=evidence)
    return evidence


def adopt_resolved_artifacts(
    state: RecoveryState, artifact_resolution: dict[str, Any]
) -> list[dict[str, Any]]:
    """Record frozen successful inputs without submitting or regenerating them."""

    selected = dict(artifact_resolution.get("selected") or {})
    aids = dict(selected.get("aids_native") or {})
    mut_generation = dict(selected.get("mutagenicity_generation") or {})
    mut_common = dict(selected.get("mutagenicity_common_recourse") or {})
    if not aids or not mut_generation or not mut_common:
        raise RuntimeError("Resolved COMRECGC artifacts are incomplete and cannot be adopted.")
    records = [
        {
            "stage_id": "aids_native_candidate_artifact",
            "dataset": "aids",
            "status": "ADOPT_EXISTING",
            "artifact_path": aids["counterfactuals_path"],
            "artifact_sha256": aids["counterfactuals_sha256"],
            "artifact_bytes": int(aids["counterfactuals_bytes"]),
            "evidence_path": aids["evidence_path"],
            "algorithm_rerun": False,
        },
        {
            "stage_id": "mut_smoke_generation_artifact",
            "dataset": "mutagenicity",
            "status": "ADOPT_EXISTING",
            "artifact_path": mut_generation["counterfactuals_path"],
            "artifact_sha256": mut_generation["counterfactuals_sha256"],
            "artifact_bytes": int(mut_generation["counterfactuals_bytes"]),
            "evidence_path": mut_generation["manifest_path"],
            "algorithm_rerun": False,
        },
        {
            "stage_id": "mut_smoke_common_recourse_artifact",
            "dataset": "mutagenicity",
            "status": "ADOPT_EXISTING",
            "artifact_path": mut_common["common_recourse_dir"],
            "selected_common_recourses_sha256": mut_common[
                "selected_common_recourses_sha256"
            ],
            "representative_counterfactuals_sha256": mut_common[
                "representative_counterfactuals_sha256"
            ],
            "evidence_path": mut_common["manifest_path"],
            "algorithm_rerun": False,
        },
    ]
    if state.data.get("adopted_stages") != records:
        state.data["adopted_stages"] = records
        state.save()
        state.event("STAGES_ADOPTED", {"stages": records})
    return records


def _dependency(stage: Stage, state: RecoveryState) -> str | None:
    if not stage.dependency_stages:
        return None
    ids: list[str] = []
    for dependency_stage in stage.dependency_stages:
        record = state.job(dependency_stage)
        if record is None:
            raise RuntimeError(f"Dependency is not submitted: {dependency_stage}")
        ids.append(str(record["job_id"]))
    return f"{stage.dependency_type}:" + ":".join(ids)


def submit(stage: Stage, state: RecoveryState, *, dry_run: bool) -> dict[str, Any]:
    existing = state.job(stage.stage_id)
    if existing is not None:
        return existing
    dependency = _dependency(stage, state)
    exports = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "RECOVERY_RUN_ID": state.data["run_id"],
        **stage.environment,
    }
    sbatch_args = [
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    ]
    if dependency:
        sbatch_args.append(f"--dependency={dependency}")
    sbatch_args.append(stage.script)
    argv = [
        "scripts/exp_sbatch.sh",
        "--name",
        f"comrecgc_recovery_{stage.stage_id}",
        "--tags",
        f"COMRECGC,recovery,{stage.dataset},{stage.stage_id}",
        "--notes",
        (
            f"upstream={UPSTREAM_COMMIT}; project={state.data['project_commit']}; "
            "fixed protocol; deterministic chemistry repair; no selection in eval"
        ),
        "--expected-output-root",
        stage.output_root,
        "--dataset",
        stage.dataset,
        "--method",
        "COMRECGC-Adapted-DeterministicChemRepair"
        if stage.dataset == "mutagenicity"
        else "COMRECGC-Native",
        "--metric",
        "MolCLR-Node-Wasserstein strict_flip CCRCOV"
        if stage.dataset == "mutagenicity"
        else "official native common recourse",
    ]
    if dry_run:
        argv.append("--dry-run")
    argv.extend(["--", *sbatch_args])
    result = command(argv, cwd=PROJECT_ROOT, timeout=180, check=True)
    if dry_run:
        job_id = f"DRYRUN_{stage.stage_id}"
    else:
        match = JOB_ID_RE.search(result.stdout)
        if match is None:
            raise RuntimeError(
                f"Could not parse registered job ID for {stage.stage_id}: {result.stdout[-500:]!r}"
            )
        job_id = match.group(1)
    return state.add_job(stage, job_id, argv)


def submit_ready_stages(
    stage_values: Sequence[Stage],
    state: RecoveryState,
    *,
    dry_run: bool,
) -> list[str]:
    """Submit only stages whose dependency records exist and whose hard gate passed."""

    submitted: list[str] = []
    completed = set(state.data.get("completed_stages") or [])
    for stage in stage_values:
        if state.job(stage.stage_id) is not None:
            continue
        if any(state.job(dependency) is None for dependency in stage.dependency_stages):
            continue
        if stage.defer_until_dependencies_complete and any(
            dependency not in completed for dependency in stage.dependency_stages
        ):
            continue
        submit(stage, state, dry_run=dry_run)
        submitted.append(stage.stage_id)
    return submitted


def stages(run_id: str, datasets: set[str], mode: str) -> list[Stage]:
    values: list[Stage] = []
    if "aids" in datasets:
        values.extend(
            [
                Stage(
                    "aids_existing_audit",
                    "aids",
                    "scripts/slurm/comrecgc_aids_existing_audit.sh",
                    "outputs/hpc/baselines/comrecgc/aids_native_dbscan_audit_v1",
                    None,
                    (),
                    {},
                ),
                Stage(
                    "aids_density_retry",
                    "aids",
                    "scripts/slurm/comrecgc_aids_density_retry.sh",
                    "outputs/hpc/baselines/comrecgc/aids_native_parent_density_retry_v1",
                    "afterok",
                    ("aids_existing_audit",),
                    {},
                ),
            ]
        )
        if mode == "all":
            values.extend(
                [
                    Stage(
                        "aids_native_full",
                        "aids",
                        "scripts/slurm/comrecgc_aids_native_full.sh",
                        "outputs/hpc/baselines/comrecgc/native_full/aids/native_full_v1",
                        "afterok",
                        ("aids_density_retry",),
                        {},
                        True,
                    ),
                    Stage(
                        "aids_native_full_gate",
                        "aids",
                        "scripts/slurm/comrecgc_aids_native_full_gate.sh",
                        "outputs/hpc/baselines/comrecgc/native_full/aids/native_full_v1/gate",
                        "afterany",
                        ("aids_native_full",),
                        {},
                    ),
                ]
            )
    if "mutagenicity" in datasets:
        smoke_base = f"outputs/hpc/baselines/comrecgc/mutagenicity/recovery_smoke_{run_id}"
        values.extend(
            [
                Stage(
                    "mut_trace_adopt",
                    "mutagenicity",
                    "scripts/slurm/comrecgc_mut_trace_adopt.sh",
                    smoke_base + "/generation",
                    None,
                    (),
                    {
                        "SOURCE_FAILED_GENERATION_DIR": (
                            "outputs/hpc/baselines/comrecgc/mutagenicity/"
                            "recovery_smoke_comrecgc_recovery_20260806_mut_retry1/"
                            "generation"
                        ),
                        "OUTPUT_DIR": smoke_base + "/generation",
                    },
                ),
                Stage(
                    "mut_chemistry_audit",
                    "mutagenicity",
                    "scripts/slurm/comrecgc_mut_chemistry_audit.sh",
                    smoke_base + "/chemistry",
                    "afterok",
                    ("mut_trace_adopt",),
                    {
                        "GENERATION_DIR": smoke_base + "/generation",
                        "OUTPUT_DIR": smoke_base + "/chemistry",
                    },
                ),
                Stage(
                    "mut_unified_eval_smoke",
                    "mutagenicity",
                    "scripts/slurm/comrecgc_mut_unified_eval.sh",
                    smoke_base + "/unified_eval",
                    "afterok",
                    ("mut_chemistry_audit",),
                    {
                        "MODE": "smoke",
                        "CHEMISTRY_DIR": smoke_base + "/chemistry",
                        "OUTPUT_DIR": smoke_base + "/unified_eval",
                    },
                ),
                Stage(
                    "mut_chemrepair_smoke_gate",
                    "mutagenicity",
                    "scripts/slurm/comrecgc_mut_chemrepair_smoke_gate.sh",
                    smoke_base + "/gate",
                    "afterany",
                    ("mut_unified_eval_smoke",),
                    {
                        "INPUT_DIR": smoke_base + "/chemistry",
                        "EVAL_DIR": smoke_base + "/unified_eval",
                        "OUTPUT_DIR": smoke_base + "/gate",
                    },
                ),
            ]
        )
        if mode == "all":
            full_base = f"outputs/hpc/baselines/comrecgc/mutagenicity/full_chemrepair_{run_id}"
            values.extend(
                [
                    Stage(
                        "mut_full_generation",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_mut_full.sh",
                        full_base + "/generation",
                        "afterok",
                        ("mut_chemrepair_smoke_gate",),
                        {"BASE_ROOT": full_base},
                        True,
                    ),
                    Stage(
                        "mut_full_common_recourse",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_common_recourse.sh",
                        full_base + "/common_recourse",
                        "afterok",
                        ("mut_full_generation",),
                        {"BASE_ROOT": full_base, "DATASET": "mutagenicity", "MODE": "full"},
                    ),
                    Stage(
                        "mut_full_chemistry",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_mut_full_chemistry.sh",
                        full_base + "/chemistry",
                        "afterok",
                        ("mut_full_common_recourse",),
                        {
                            "BASE_ROOT": full_base,
                            "TRACE_PARITY_PATH": smoke_base + "/generation/trace_parity.json",
                        },
                    ),
                    Stage(
                        "mut_full_unified_eval",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_mut_unified_eval.sh",
                        full_base + "/unified_eval",
                        "afterok",
                        ("mut_full_chemistry",),
                        {"BASE_ROOT": full_base, "MODE": "full"},
                    ),
                    Stage(
                        "mut_full_gate",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_mut_full_gate.sh",
                        full_base + "/full_gate",
                        "afterany",
                        ("mut_full_unified_eval",),
                        {"BASE_ROOT": full_base},
                    ),
                    Stage(
                        "mut_freeze",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_mut_freeze.sh",
                        "outputs/hpc/eval/paper/mutagenicity_common4_comrecgc_standardized_v1/comrecgc",
                        "afterok",
                        ("mut_full_gate",),
                        {
                            "BASE_ROOT": full_base,
                            "STANDARDIZED_ROOT": "outputs/hpc/eval/paper/mutagenicity_common4_comrecgc_standardized_v1/comrecgc",
                            "AUTOMATION_STATE": f"outputs/hpc/automation/comrecgc_recovery/{run_id}/state.json",
                        },
                    ),
                ]
            )
    return values


def refresh(state: RecoveryState) -> None:
    active = False
    failed: list[str] = []
    completed: list[str] = []
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    for record in state.data["jobs"]:
        if str(record["job_id"]).startswith("DRYRUN_"):
            continue
        result = command(
            [
                "sacct",
                "-j",
                str(record["job_id"]),
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,ExitCode",
            ],
            cwd=PROJECT_ROOT,
            timeout=60,
            check=False,
        )
        rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
        primary = next((row for row in rows if row[0] == str(record["job_id"])), None)
        if primary is None:
            active = True
            continue
        slurm_state = primary[1].split()[0].split("+")[0]
        record["slurm_state"] = slurm_state
        record["exit_code"] = primary[2]
        record["refreshed_at"] = now()
        if slurm_state == "COMPLETED" and primary[2] == "0:0":
            completed.append(record["stage_id"])
        elif slurm_state in terminal:
            failed.append(record["stage_id"])
        else:
            active = True
    state.data["completed_stages"] = sorted(set(completed))
    state.data["failed_stages"] = sorted(set(failed))
    if failed:
        status = "BLOCKED"
    elif active:
        status = "RUNNING"
    elif state.data["jobs"]:
        status = "JOBS_COMPLETED"
    else:
        status = "PREFLIGHT_PASSED"
    state.transition(status)


def report(state: RecoveryState) -> None:
    payload = {
        "run_id": state.data["run_id"],
        "status": state.data["status"],
        "project_commit": state.data["project_commit"],
        "upstream_commit": state.data["upstream_commit"],
        "jobs": state.data["jobs"],
        "adopted_stages": state.data.get("adopted_stages") or [],
        "completed_stages": state.data["completed_stages"],
        "failed_stages": state.data["failed_stages"],
        "state_path": str(state.state_path),
    }
    write_json(state.root / "final_report.json", payload)
    lines = [
        "# COMRECGC Recovery Report",
        "",
        f"- Run ID: {payload['run_id']}",
        f"- Status: {payload['status']}",
        f"- Project commit: {payload['project_commit']}",
        f"- Upstream commit: {payload['upstream_commit']}",
        "- Adopted stages: "
        + ", ".join(row["stage_id"] for row in payload["adopted_stages"])
        if payload["adopted_stages"]
        else "- Adopted stages: none",
        f"- Completed stages: {', '.join(payload['completed_stages']) or 'none'}",
        f"- Failed stages: {', '.join(payload['failed_stages']) or 'none'}",
        "",
        "## Jobs",
        "",
    ]
    lines.extend(
        f"- {row['stage_id']}: {row['job_id']} {row['slurm_state']} {row['exit_code']}"
        for row in payload["jobs"]
    )
    (state.root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "all", "refresh"), default="smoke")
    parser.add_argument("--datasets", default="aids,mutagenicity")
    parser.add_argument("--run-id", default="comrecgc_recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--control-socket", default=DEFAULT_CONTROL_SOCKET)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.remote:
        return delegate_remote(args)
    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    if not datasets or not datasets <= {"aids", "mutagenicity"}:
        raise ValueError("--datasets must contain aids and/or mutagenicity.")
    state = RecoveryState(
        PROJECT_ROOT / f"outputs/hpc/automation/comrecgc_recovery/{args.run_id}",
        args.run_id,
        requested_mode=args.mode,
        datasets=sorted(datasets),
    )
    if args.mode == "refresh":
        refresh(state)
        requested_mode = str(state.data.get("requested_mode") or "smoke")
        requested_datasets = set(state.data.get("datasets") or datasets)
        if requested_mode == "all" and not state.data.get("failed_stages"):
            submitted = submit_ready_stages(
                stages(args.run_id, requested_datasets, requested_mode),
                state,
                dry_run=False,
            )
            if submitted:
                state.transition("FULL_SUBMITTED", newly_submitted_stages=submitted)
        report(state)
    else:
        if state.data["status"] == "CREATED":
            preflight(state)
        submit_ready_stages(
            stages(args.run_id, datasets, args.mode),
            state,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            submitted_status = "DRY_RUN_COMPLETED"
        elif args.mode == "all" and state.job("mut_full_generation") is None:
            submitted_status = "SMOKE_SUBMITTED_AWAITING_GATE"
        elif args.mode == "all":
            submitted_status = "FULL_SUBMITTED"
        else:
            submitted_status = "SMOKE_SUBMITTED"
        state.transition(submitted_status)
        if not args.dry_run:
            command(
                [sys.executable, "scripts/sync_experiment_status.py"],
                cwd=PROJECT_ROOT,
                timeout=180,
                check=False,
            )
        report(state)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": state.data["status"],
                "state_path": str(state.state_path),
                "jobs": [
                    {"stage_id": row["stage_id"], "job_id": row["job_id"]}
                    for row in state.data["jobs"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
