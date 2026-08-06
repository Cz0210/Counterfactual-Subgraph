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
from src.baselines.comrecgc.cache_trust import audit_aids_pyg_cache  # noqa: E402
from src.baselines.comrecgc.contracts import (  # noqa: E402
    GenerationParameters,
    RecourseParameters,
    UPSTREAM_COMMIT,
    append_jsonl,
    sha256_file,
    write_json,
)
from src.baselines.comrecgc.upstream import validate_upstream_checkout  # noqa: E402

DEFAULT_REMOTE_HOST = "u20526@logini.tongji.edu.cn"
DEFAULT_REMOTE_PORT = 10022
DEFAULT_CONTROL_SOCKET = "/tmp/tongji-codex.sock"
DEFAULT_REMOTE_ROOT = "/share/home/u20526/czx/worktrees/comrecgc-recovery-20260806"
JOB_ID_RE = re.compile(r"^job_id=(\d+)$", re.MULTILINE)
ALLOWED_DYNAMIC_DIRTY_PATHS = frozenset({"docs/EXPERIMENT_LOG.md"})
RETRY3_SCOPE = "RETRY3_SMOKE_ONLY"
END_TO_END_SCOPE = "COMRECGC_END_TO_END_AIDS_MUTAGENICITY"
AUTHORIZED_SMOKE_STAGES = frozenset(
    {
        "aids_existing_audit",
        "aids_density_retry",
        "mut_trace_adopt",
        "mut_chemistry_audit",
        "mut_unified_eval_smoke",
        "mut_chemrepair_smoke_gate",
    }
)
END_TO_END_RUN_PREFIX = "comrecgc_end_to_end_"
FULL_STAGE_IDS = frozenset(
    {
        "aids_native_full",
        "aids_native_full_gate",
        "aids_project_full_generation",
        "aids_project_full_common_recourse",
        "aids_project_full_chemistry",
        "aids_project_full_unified_eval",
        "aids_project_full_gate",
        "aids_project_freeze",
        "mut_full_generation",
        "mut_full_common_recourse",
        "mut_full_chemistry",
        "mut_full_unified_eval",
        "mut_full_gate",
        "mut_freeze",
    }
)
SMOKE_GATE_STAGE_IDS = {
    "aids": frozenset({"aids_project_smoke_gate"}),
    "mutagenicity": frozenset({"mut_chemrepair_smoke_gate"}),
}


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
        authorization: dict[str, Any] | None = None,
        authorization_sha256: str | None = None,
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
                "authorization": authorization,
                "authorization_sha256": authorization_sha256,
                "created_at": now(),
            }
            self.save()
            self.event("RUN_CREATED", {"run_id": run_id})
        self.data.setdefault("adopted_stages", [])
        self.data.setdefault("completed_stages", [])
        self.data.setdefault("failed_stages", [])
        self.data.setdefault("completed_stage_adoptions", [])
        self.data.setdefault("adopted_stage_outputs", {})
        if authorization is not None:
            existing_sha = self.data.get("authorization_sha256")
            if existing_sha not in {None, authorization_sha256}:
                raise RuntimeError("Recovery state authorization SHA256 changed.")
            self.data["authorization"] = authorization
            self.data["authorization_sha256"] = authorization_sha256
            self.save()

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


def retry3_authorization_template(project_commit: str) -> dict[str, Any]:
    return {
        "authorization_status": "AUTHORIZED",
        "authorized_scope": RETRY3_SCOPE,
        "project_commit": str(project_commit),
        "upstream_commit": UPSTREAM_COMMIT,
        "phase_c_full_approved": False,
        "full_submission_allowed": False,
        "auto_promote_to_full": False,
        "candidate_regeneration_allowed": False,
        "random_walk_rerun_allowed": False,
        "max_aids_jobs": 2,
        "max_mutagenicity_jobs": 4,
        "max_total_jobs": 6,
    }


def end_to_end_authorization_template(project_commit: str) -> dict[str, Any]:
    return {
        "authorization_status": "AUTHORIZED",
        "authorized_scope": END_TO_END_SCOPE,
        "project_commit": str(project_commit),
        "upstream_commit": UPSTREAM_COMMIT,
        "phase_c_full_approved": True,
        "full_submission_allowed": True,
        "auto_promote_to_full": True,
        "candidate_regeneration_in_smoke": False,
        "candidate_generation_in_full": True,
        "random_walk_rerun_in_smoke": False,
        "random_walk_full_allowed": True,
        "scientific_parameter_sweep_allowed": False,
        "rank_backfill_allowed": False,
        "rf_guided_repair_allowed": False,
        "wnode_guided_repair_allowed": False,
    }


def validate_retry3_authorization(
    authorization: dict[str, Any], *, project_commit: str
) -> None:
    expected = retry3_authorization_template(project_commit)
    mismatches = {
        field: {"actual": authorization.get(field), "expected": expected_value}
        for field, expected_value in expected.items()
        if authorization.get(field) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Invalid retry3 authorization: {mismatches}")


def initialize_retry3_authorization(root: Path, *, project_commit: str) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "authorization.json"
    expected = retry3_authorization_template(project_commit)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(f"Existing retry3 authorization differs: {path}")
    else:
        write_json(path, expected)
    return path, sha256_file(path)


def load_retry3_authorization(root: Path, *, project_commit: str) -> tuple[dict[str, Any], str]:
    path = root / "authorization.json"
    if not path.is_file():
        raise RuntimeError(f"Retry3 authorization is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Retry3 authorization must be a JSON object: {path}")
    validate_retry3_authorization(value, project_commit=project_commit)
    return value, sha256_file(path)


def initialize_end_to_end_authorization(
    root: Path, *, project_commit: str
) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "authorization.json"
    expected = end_to_end_authorization_template(project_commit)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(f"Existing end-to-end authorization differs: {path}")
    else:
        write_json(path, expected)
    return path, sha256_file(path)


def load_end_to_end_authorization(
    root: Path, *, project_commit: str
) -> tuple[dict[str, Any], str]:
    path = root / "authorization.json"
    if not path.is_file():
        raise RuntimeError(f"End-to-end authorization is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"End-to-end authorization must be a JSON object: {path}")
    expected = end_to_end_authorization_template(project_commit)
    mismatches = {
        field: {"actual": value.get(field), "expected": expected_value}
        for field, expected_value in expected.items()
        if value.get(field) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Invalid end-to-end authorization: {mismatches}")
    return value, sha256_file(path)


def assert_full_authorized(authorization: dict[str, Any] | None) -> None:
    if authorization is None:
        raise RuntimeError("Full submission requires an explicit authorization object.")
    if authorization.get("phase_c_full_approved") is not True:
        raise RuntimeError("Full submission blocked: phase_c_full_approved is not true.")
    if authorization.get("full_submission_allowed") is not True:
        raise RuntimeError("Full submission blocked: full_submission_allowed is not true.")
    if authorization.get("auto_promote_to_full") is not True:
        raise RuntimeError("Full submission blocked: auto_promote_to_full is not true.")


def assert_smoke_engineering_gate(state: RecoveryState, dataset: str) -> None:
    required = SMOKE_GATE_STAGE_IDS[str(dataset)]
    completed = set(state.data.get("completed_stages") or [])
    missing = sorted(required - completed)
    if missing:
        raise RuntimeError(
            f"Full submission blocked before {dataset} smoke engineering Gate: {missing}"
        )


def _stage_payload(stage: Stage, state: RecoveryState) -> dict[str, Any]:
    profile = "full" if stage.stage_id in FULL_STAGE_IDS else "smoke"
    scientific_parameters: dict[str, Any] = {
        key: value
        for key, value in stage.environment.items()
        if key in {"DATASET", "MODE", "PARENT_LIMIT"}
    }
    if "generation" in stage.stage_id:
        scientific_parameters["generation"] = GenerationParameters.for_mode(
            profile
        ).__dict__
    if "common_recourse" in stage.stage_id or stage.stage_id == "aids_native_full":
        scientific_parameters["common_recourse"] = RecourseParameters.for_mode(
            profile
        ).__dict__
    return {
        "stage": stage.stage_id,
        "dataset": stage.dataset,
        "slurm_script": stage.script,
        "dependency_type": stage.dependency_type,
        "dependency_stages": list(stage.dependency_stages),
        "input_roots": sorted(
            value
            for key, value in stage.environment.items()
            if key.endswith("_DIR") or key.endswith("_ROOT") or key.endswith("_PATH")
        ),
        "output_root": stage.output_root,
        "project_commit": state.data["project_commit"],
        "upstream_commit": UPSTREAM_COMMIT,
        "scientific_parameters": scientific_parameters,
        "expected_artifacts": ["run_manifest.json", "completion_marker"],
        "profile": profile,
        "retry_policy": "one_engineering_or_transient_retry_no_scientific_retry",
    }


def write_authorized_job_dag(
    state: RecoveryState, stage_values: Sequence[Stage]
) -> tuple[Path, str]:
    path = state.root / "authorized_job_dag.json"
    payload = {
        "schema_version": 1,
        "authorization_sha256": state.data.get("authorization_sha256"),
        "project_commit": state.data["project_commit"],
        "upstream_commit": UPSTREAM_COMMIT,
        "nodes": [_stage_payload(stage, state) for stage in stage_values],
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Authorized COMRECGC job DAG changed: {path}")
    else:
        write_json(path, payload)
    digest = sha256_file(path)
    state.data["job_dag_sha256"] = digest
    state.save()
    return path, digest


def validate_stage_in_authorized_dag(stage: Stage, state: RecoveryState) -> None:
    authorization = state.data.get("authorization")
    if not isinstance(authorization, dict):
        return
    if authorization.get("authorized_scope") != END_TO_END_SCOPE:
        return
    path = state.root / "authorized_job_dag.json"
    if not path.is_file():
        raise RuntimeError("End-to-end submission requires authorized_job_dag.json.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = _stage_payload(stage, state)
    matches = [row for row in payload.get("nodes") or [] if row.get("stage") == stage.stage_id]
    if matches != [expected]:
        raise RuntimeError(f"Stage is absent or differs from authorized DAG: {stage.stage_id}")


def validate_job_caps(state: RecoveryState, *, next_stage: Stage | None = None) -> None:
    authorization = state.data.get("authorization")
    if not isinstance(authorization, dict):
        raise RuntimeError("Retry3 job submission requires validated authorization.")
    stage_ids = [str(row["stage_id"]) for row in state.data["jobs"]]
    if next_stage is not None:
        stage_ids.append(next_stage.stage_id)
    if any(stage_id not in AUTHORIZED_SMOKE_STAGES for stage_id in stage_ids):
        raise RuntimeError(f"Retry3 authorization refuses non-smoke stages: {stage_ids}")
    aids_count = sum(stage_id.startswith("aids_") for stage_id in stage_ids)
    mut_count = sum(stage_id.startswith("mut_") for stage_id in stage_ids)
    if aids_count > int(authorization["max_aids_jobs"]):
        raise RuntimeError("Retry3 AIDS job cap exceeded.")
    if mut_count > int(authorization["max_mutagenicity_jobs"]):
        raise RuntimeError("Retry3 Mutagenicity job cap exceeded.")
    if len(stage_ids) > int(authorization["max_total_jobs"]):
        raise RuntimeError("Retry3 total job cap exceeded.")


def recover_registered_jobs(
    stage_values: Sequence[Stage], state: RecoveryState
) -> list[str]:
    """Adopt exact retry3 jobs recorded before a client interruption."""

    registry_path = PROJECT_ROOT / "outputs/hpc/experiment_registry/jobs.jsonl"
    registry_rows: list[dict[str, Any]] = []
    if registry_path.is_file():
        for line in registry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                registry_rows.append(value)
    adopted: list[str] = []
    completed_adoptions = set(state.data.get("adopted_stage_outputs") or {})
    for stage in stage_values:
        if stage.stage_id in completed_adoptions:
            continue
        if state.job(stage.stage_id) is not None:
            continue
        matches = [
            row
            for row in registry_rows
            if row.get("experiment_name") == f"comrecgc_recovery_{stage.stage_id}"
            and row.get("expected_output_root") == stage.output_root
            and row.get("git_commit") == state.data["project_commit"]
            and str(row.get("job_id") or "").isdigit()
        ]
        identifiers = sorted({str(row["job_id"]) for row in matches})
        if len(identifiers) > 1:
            raise RuntimeError(
                f"Multiple retry3 registry jobs match stage {stage.stage_id}: {identifiers}"
            )
        if not identifiers:
            continue
        record = state.add_job(stage, identifiers[0], matches[0].get("sbatch_args") or [])
        record["adoption_mode"] = "ADOPT_EXISTING_REGISTRY"
        record["adopted_at"] = now()
        state.save()
        state.event("JOB_ADOPTED", record)
        adopted.append(stage.stage_id)

    queue = command(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j|%T"],
        cwd=PROJECT_ROOT,
        timeout=60,
        check=False,
    )
    accounting = command(
        [
            "sacct",
            "-S",
            "2026-08-01",
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName,State,ExitCode",
            "-X",
        ],
        cwd=PROJECT_ROOT,
        timeout=90,
        check=False,
    )
    write_json(
        state.root / "evidence/submission_recovery_scan.json",
        {
            "registry_path": str(registry_path),
            "registry_match_stage_ids": adopted,
            "state_job_ids": [str(row["job_id"]) for row in state.data["jobs"]],
            "squeue_comrecgc_rows": [
                line for line in queue.stdout.splitlines() if "comrecgc" in line.lower()
            ],
            "sacct_comrecgc_rows": [
                line for line in accounting.stdout.splitlines() if "comrecgc" in line.lower()
            ],
            "duplicate_submission_count": 0,
        },
    )
    if "_retry3" in str(state.data.get("run_id")):
        validate_job_caps(state)
    return adopted


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
    if args.initialize_authorization:
        remote.append("--initialize-authorization")
    for value in args.adopt_completed_stage:
        remote.extend(["--adopt-completed-stage", value])
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
    input_manifest = write_retry3_input_manifest(state, artifact_resolution)
    cache_trust: dict[str, Any] | None = None
    if "aids" in set(state.data.get("datasets") or []):
        cache_trust = audit_aids_pyg_cache(
            upstream_root=PROJECT_ROOT / "external/COMRECGC",
            output_path=state.root / "evidence/aids_cache_trust_before.json",
        )
        state.data["aids_cache_trust"] = cache_trust
        state.save()
        if cache_trust.get("cache_trust_passed") is not True:
            raise RuntimeError(
                "AIDS cache trust gate blocked submission: group/world writable or "
                "symlinked cache content was detected."
            )
    evidence = {
        "branch": branch,
        "project_commit": commit,
        "upstream_root": str(upstream),
        "upstream_commit": UPSTREAM_COMMIT,
        "artifact_resolution_passed": artifact_resolution["resolution_passed"],
        "adopted_stage_count": len(state.data["adopted_stages"]),
        "allowed_dynamic_dirty": allowed_dynamic_dirty,
        "blocked_dirty": blocked_dirty,
        "authorization_sha256": state.data.get("authorization_sha256"),
        "input_artifact_set_sha256": input_manifest["input_artifact_set_sha256"],
        "aids_cache_trust_passed": (
            None if cache_trust is None else cache_trust["cache_trust_passed"]
        ),
        "sbatch": shutil.which("sbatch") or "",
        "sacct": shutil.which("sacct") or "",
    }
    if not evidence["sbatch"] or not evidence["sacct"]:
        raise RuntimeError("HPC preflight requires both sbatch and sacct on PATH.")
    write_json(state.root / "evidence/preflight.json", evidence)
    state.transition("PREFLIGHT_PASSED", preflight=evidence)
    return evidence


def write_retry3_input_manifest(
    state: RecoveryState, artifact_resolution: dict[str, Any]
) -> dict[str, Any]:
    selected = dict(artifact_resolution.get("selected") or {})
    records: list[dict[str, Any]] = []
    datasets = set(state.data.get("datasets") or [])
    if "aids" in datasets:
        value = dict(selected["aids_native"])
        artifact = Path(value["counterfactuals_path"]).resolve()
        evidence = Path(value["evidence_path"]).resolve()
        records.append(
            {
                "dataset": "aids",
                "artifact_path": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "manifest_path": str(evidence),
                "manifest_sha256": sha256_file(evidence),
                "parent_count": 64,
                "candidate_count": 31,
                "seed": 0,
                "algorithm_rerun": False,
                "adoption_mode": "ADOPT_EXISTING",
            }
        )
        project_smoke = (
            PROJECT_ROOT
            / "outputs/hpc/baselines/comrecgc/aids/"
            "smoke_comrecgc_smoke_budget_retry_20260806_v4"
        )
        project_generation = project_smoke / "generation/counterfactuals.pt"
        project_manifest = project_smoke / "generation/run_manifest.json"
        project_gate = project_smoke / "gate.json"
        if not all(path.is_file() for path in (project_generation, project_manifest, project_gate)):
            raise RuntimeError("Frozen project AIDS/HIV smoke adoption inputs are missing.")
        project_value = json.loads(project_manifest.read_text(encoding="utf-8"))
        records.append(
            {
                "dataset": "aids_project_smoke",
                "artifact_path": str(project_generation.resolve()),
                "artifact_sha256": sha256_file(project_generation),
                "artifact_bytes": project_generation.stat().st_size,
                "manifest_path": str(project_manifest.resolve()),
                "manifest_sha256": sha256_file(project_manifest),
                "gate_path": str(project_gate.resolve()),
                "gate_sha256": sha256_file(project_gate),
                "dataset_fingerprint": project_value["dataset_audit"][
                    "dataset_fingerprint"
                ],
                "parent_count": 64,
                "candidate_count": int(project_value["counterfactual_candidate_count"]),
                "seed": int(project_value["parameters"]["seed"]),
                "algorithm_rerun": False,
                "adoption_mode": "ADOPT_EXISTING",
            }
        )
    if "mutagenicity" in datasets:
        generation = dict(selected["mutagenicity_generation"])
        common = dict(selected["mutagenicity_common_recourse"])
        artifact = Path(generation["counterfactuals_path"]).resolve()
        generation_manifest = Path(generation["manifest_path"]).resolve()
        common_manifest = Path(common["manifest_path"]).resolve()
        generation_value = json.loads(generation_manifest.read_text(encoding="utf-8"))
        records.append(
            {
                "dataset": "mutagenicity",
                "artifact_path": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "manifest_path": str(generation_manifest),
                "manifest_sha256": sha256_file(generation_manifest),
                "common_recourse_manifest_path": str(common_manifest),
                "common_recourse_manifest_sha256": sha256_file(common_manifest),
                "dataset_fingerprint": generation_value["dataset_audit"][
                    "dataset_fingerprint"
                ],
                "generation_parent_ids_sha256": generation_value[
                    "generation_parent_ids_sha256"
                ],
                "parent_count": 64,
                "candidate_count": 164,
                "model_counterfactual_count": 70,
                "cluster_count": 4,
                "seed": 0,
                "algorithm_rerun": False,
                "adoption_mode": "ADOPT_EXISTING",
            }
        )
    manifest = {
        "schema_version": 1,
        "project_commit": state.data["project_commit"],
        "upstream_commit": UPSTREAM_COMMIT,
        "authorization_sha256": state.data.get("authorization_sha256"),
        "artifacts": records,
        "input_artifact_set_sha256": _artifact_set_sha256(records),
        "input_sha256_before": _artifact_set_sha256(records),
    }
    write_json(state.root / "retry3_input_manifest.json", manifest)
    write_json(state.root / "evidence/retry3_input_manifest.json", manifest)
    state.data["input_artifact_set_sha256_before"] = manifest[
        "input_artifact_set_sha256"
    ]
    state.save()
    return manifest


def _artifact_set_sha256(records: Sequence[dict[str, Any]]) -> str:
    from src.baselines.comrecgc.contracts import stable_json_sha256

    return stable_json_sha256(
        [
            {
                key: value
                for key, value in record.items()
                if key.endswith("sha256") or key in {"artifact_path", "manifest_path"}
            }
            for record in records
        ]
    )


def verify_retry3_inputs(state: RecoveryState) -> str | None:
    path = state.root / "retry3_input_manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = list(manifest.get("artifacts") or [])
    for record in records:
        if sha256_file(record["artifact_path"]) != record["artifact_sha256"]:
            raise RuntimeError("Retry3 frozen input artifact changed after submission.")
        if sha256_file(record["manifest_path"]) != record["manifest_sha256"]:
            raise RuntimeError("Retry3 frozen input manifest changed after submission.")
        common_path = record.get("common_recourse_manifest_path")
        if common_path and sha256_file(common_path) != record[
            "common_recourse_manifest_sha256"
        ]:
            raise RuntimeError("Retry3 frozen common-recourse manifest changed.")
        gate_path = record.get("gate_path")
        if gate_path and sha256_file(gate_path) != record["gate_sha256"]:
            raise RuntimeError("Frozen project AIDS/HIV smoke gate changed.")
    after = _artifact_set_sha256(records)
    if after != manifest["input_artifact_set_sha256"]:
        raise RuntimeError("Retry3 frozen input inventory changed after submission.")
    state.data["input_artifact_set_sha256_after"] = after
    state.save()
    return after


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
    if "aids" in set(state.data.get("datasets") or []):
        project_smoke = (
            PROJECT_ROOT
            / "outputs/hpc/baselines/comrecgc/aids/"
            "smoke_comrecgc_smoke_budget_retry_20260806_v4"
        )
        project_counterfactuals = project_smoke / "generation/counterfactuals.pt"
        if project_counterfactuals.is_file():
            records.insert(
                1,
                {
                    "stage_id": "aids_project_smoke_artifact",
                    "dataset": "aids",
                    "status": "ADOPT_EXISTING",
                    "artifact_path": str(project_smoke),
                    "artifact_sha256": sha256_file(project_counterfactuals),
                    "evidence_path": str(project_smoke / "gate.json"),
                    "algorithm_rerun": False,
                },
            )
    if state.data.get("adopted_stages") != records:
        state.data["adopted_stages"] = records
        state.save()
        state.event("STAGES_ADOPTED", {"stages": records})
    return records


def _dependency(stage: Stage, state: RecoveryState) -> str | None:
    if not stage.dependency_stages:
        return None
    ids: list[str] = []
    completed = set(state.data.get("completed_stages") or [])
    for dependency_stage in stage.dependency_stages:
        record = state.job(dependency_stage)
        if record is None:
            if dependency_stage in completed:
                continue
            raise RuntimeError(f"Dependency is not submitted: {dependency_stage}")
        ids.append(str(record["job_id"]))
    if not ids:
        return None
    return f"{stage.dependency_type}:" + ":".join(ids)


def submit(stage: Stage, state: RecoveryState, *, dry_run: bool) -> dict[str, Any]:
    existing = state.job(stage.stage_id)
    if existing is not None:
        return existing
    if stage.stage_id in FULL_STAGE_IDS:
        assert_full_authorized(state.data.get("authorization"))
        assert_smoke_engineering_gate(state, stage.dataset)
    validate_stage_in_authorized_dag(stage, state)
    if "_retry3" in str(state.data.get("run_id")):
        validate_job_caps(state, next_stage=stage)
    dependency = _dependency(stage, state)
    exports = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "RECOVERY_RUN_ID": state.data["run_id"],
        "EXPECTED_PROJECT_COMMIT": state.data["project_commit"],
        "EXPECTED_UPSTREAM_COMMIT": UPSTREAM_COMMIT,
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
        (
            "COMRECGC-Native"
            if stage.stage_id.startswith("aids_native")
            or stage.stage_id in {"aids_existing_audit", "aids_density_retry"}
            else "COMRECGC-Adapted-DeterministicChemRepair"
        ),
        "--metric",
        (
            "official native common recourse"
            if stage.stage_id.startswith("aids_native")
            or stage.stage_id in {"aids_existing_audit", "aids_density_retry"}
            else "MolCLR-Node-Wasserstein strict_flip CCRCOV"
        ),
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
    completed = set(state.data.get("completed_stages") or []) | set(
        state.data.get("adopted_stage_outputs") or {}
    )
    for stage in stage_values:
        if stage.stage_id in set(state.data.get("adopted_stage_outputs") or {}):
            continue
        if state.job(stage.stage_id) is not None:
            continue
        if any(
            state.job(dependency) is None and dependency not in completed
            for dependency in stage.dependency_stages
        ):
            continue
        if stage.defer_until_dependencies_complete and any(
            dependency not in completed for dependency in stage.dependency_stages
        ):
            continue
        submit(stage, state, dry_run=dry_run)
        submitted.append(stage.stage_id)
    return submitted


def validate_new_output_roots(
    stage_values: Sequence[Stage], state: RecoveryState
) -> None:
    """Refuse an unregistered retry output instead of guessing its provenance."""

    collisions: list[str] = []
    adopted = set(state.data.get("adopted_stage_outputs") or {})
    for stage in stage_values:
        if stage.stage_id in adopted:
            continue
        if state.job(stage.stage_id) is not None:
            continue
        output = (PROJECT_ROOT / stage.output_root).resolve()
        if output.exists():
            collisions.append(str(output))
    if collisions:
        raise RuntimeError(f"Retry3 output collision: {collisions}")


def _validate_completed_stage_artifacts(stage_id: str, output: Path) -> list[Path]:
    required = [output / "_RUN_COMPLETE.json", output / "run_manifest.json"]
    if stage_id == "mut_trace_adopt":
        required.extend([output / "trace_parity.json", output / "counterfactuals.pt"])
    elif stage_id == "mut_chemistry_audit":
        required.extend(
            [output / "audit.json", output / "audit.txt", output / "final_artifact_audit.json"]
        )
    else:
        raise RuntimeError(f"Completed-stage adoption is not authorized for {stage_id}.")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"Completed-stage adoption artifacts are missing: {missing}")
    marker = json.loads(required[0].read_text(encoding="utf-8"))
    manifest = json.loads(required[1].read_text(encoding="utf-8"))
    if marker.get("run_complete") is not True or manifest.get("run_complete") is not True:
        raise RuntimeError(f"Completed-stage adoption is not complete: {stage_id}")
    if stage_id == "mut_trace_adopt":
        parity = json.loads((output / "trace_parity.json").read_text(encoding="utf-8"))
        exact_fields = (
            "trace_parity_passed",
            "importance_threshold_mask_exact",
            "model_cf_id_set_exact",
            "model_cf_order_exact",
            "dbscan_input_id_set_exact",
            "dbscan_input_order_exact",
        )
        if any(parity.get(field) is not True for field in exact_fields):
            raise RuntimeError("Completed trace adoption does not pass exact discrete parity.")
        compared = set(parity.get("compared_fields") or [])
        if not {"stable_graph_sha256", "frequency", "order"} <= compared:
            raise RuntimeError("Completed trace adoption lacks exact graph/order/frequency evidence.")
    else:
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        audit_text = (output / "audit.txt").read_text(encoding="utf-8")
        if (
            audit.get("audit_passed") is not True
            or float(audit.get("source_roundtrip_rate", 0.0)) != 1.0
            or float(audit.get("noop_roundtrip_rate", 0.0)) != 1.0
            or "[COMRECGC_PROJECT_CHEMISTRY_ENGINEERING_PASS]" not in audit_text
        ):
            raise RuntimeError("Completed chemistry adoption does not pass its engineering Gate.")
    return required


def register_completed_stage_adoptions(
    state: RecoveryState,
    values: Sequence[str],
    *,
    known_stage_ids: set[str],
) -> dict[str, str]:
    mappings = dict(state.data.get("adopted_stage_outputs") or {})
    records = list(state.data.get("completed_stage_adoptions") or [])
    records_by_stage = {str(row["stage_id"]): row for row in records}
    allowed_root = (PROJECT_ROOT / "outputs/hpc/baselines/comrecgc").resolve()
    for value in values:
        if "=" not in value:
            raise RuntimeError("--adopt-completed-stage requires STAGE_ID=OUTPUT_PATH.")
        stage_id, raw_path = value.split("=", 1)
        if stage_id not in known_stage_ids:
            raise RuntimeError(f"Unknown completed stage adoption: {stage_id}")
        output = Path(raw_path).expanduser()
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output = output.resolve(strict=True)
        if not output.is_relative_to(allowed_root):
            raise RuntimeError("Completed-stage adoption must remain in COMRECGC outputs.")
        relative = (
            Path("outputs/hpc/baselines/comrecgc")
            / output.relative_to(allowed_root)
        ).as_posix()
        existing = mappings.get(stage_id)
        if existing is not None and existing != relative:
            raise RuntimeError(f"Completed-stage adoption path changed for {stage_id}.")
        paths = _validate_completed_stage_artifacts(stage_id, output)
        files = {
            path.name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        }
        record = {
            "stage_id": stage_id,
            "status": "ADOPT_EXISTING",
            "output_root": relative,
            "files": files,
            "adopted_at": now(),
        }
        previous = records_by_stage.get(stage_id)
        if previous is not None:
            comparable_previous = {key: previous.get(key) for key in ("stage_id", "status", "output_root", "files")}
            comparable_record = {key: record.get(key) for key in comparable_previous}
            if comparable_previous != comparable_record:
                raise RuntimeError(f"Completed-stage adoption evidence changed for {stage_id}.")
            record = previous
        else:
            records.append(record)
            records_by_stage[stage_id] = record
            state.event("COMPLETED_STAGE_ADOPTED", record)
        mappings[stage_id] = relative
    state.data["adopted_stage_outputs"] = mappings
    state.data["completed_stage_adoptions"] = records
    state.data["completed_stages"] = sorted(
        set(state.data.get("completed_stages") or []) | set(mappings)
    )
    state.save()
    write_json(state.root / "evidence/completed_stage_adoptions.json", {"stages": records})
    return mappings


def verify_completed_stage_adoptions(state: RecoveryState) -> None:
    for record in state.data.get("completed_stage_adoptions") or []:
        output = (PROJECT_ROOT / str(record["output_root"])).resolve(strict=True)
        paths = _validate_completed_stage_artifacts(str(record["stage_id"]), output)
        current = {path.name: sha256_file(path) for path in paths}
        expected = {
            name: str(value["sha256"])
            for name, value in dict(record.get("files") or {}).items()
        }
        if current != expected:
            raise RuntimeError(
                f"Completed-stage adoption changed after freeze: {record['stage_id']}"
            )


def stages(
    run_id: str,
    datasets: set[str],
    mode: str,
    *,
    adopted_stage_outputs: dict[str, str] | None = None,
) -> list[Stage]:
    adopted_outputs = dict(adopted_stage_outputs or {})
    values: list[Stage] = []
    if "aids" in datasets:
        smoke_base = f"outputs/hpc/baselines/comrecgc/aids/recovery_smoke_{run_id}"
        existing_audit = smoke_base + "/existing_audit"
        density_audit = smoke_base + "/parent_density"
        project_smoke = smoke_base + "/project_adapter_adoption"
        values.extend(
            [
                Stage(
                    "aids_existing_audit",
                    "aids",
                    "scripts/slurm/comrecgc_aids_existing_audit.sh",
                    existing_audit,
                    None,
                    (),
                    {"OUTPUT_DIR": existing_audit},
                ),
                Stage(
                    "aids_density_retry",
                    "aids",
                    "scripts/slurm/comrecgc_aids_density_retry.sh",
                    density_audit,
                    "afterok",
                    ("aids_existing_audit",),
                    {
                        "EXISTING_AUDIT": existing_audit + "/audit.json",
                        "PREREGISTRATION": (
                            f"outputs/hpc/automation/comrecgc_recovery/{run_id}/"
                            "aids_parent_density_preregistration.json"
                        ),
                        "OUTPUT_DIR": density_audit,
                    },
                ),
                Stage(
                    "aids_project_smoke_gate",
                    "aids",
                    "scripts/slurm/comrecgc_aids_project_smoke_adopt.sh",
                    project_smoke,
                    "afterok",
                    ("aids_density_retry",),
                    {
                        "DATASET": "aids",
                        "MODE": "smoke",
                        "SOURCE_ROOT": (
                            "outputs/hpc/baselines/comrecgc/aids/"
                            "smoke_comrecgc_smoke_budget_retry_20260806_v4"
                        ),
                        "OUTPUT_DIR": project_smoke,
                    },
                ),
            ]
        )
        if mode == "all":
            native_full = (
                f"outputs/hpc/baselines/comrecgc/native_full/aids/{run_id}"
            )
            project_full = (
                f"outputs/hpc/baselines/comrecgc/aids/project_full_{run_id}"
            )
            standardized = (
                "outputs/hpc/eval/paper/"
                "aids_common4_comrecgc_standardized_v1/comrecgc"
            )
            values.extend(
                [
                    Stage(
                        "aids_native_full",
                        "aids",
                        "scripts/slurm/comrecgc_aids_native_full.sh",
                        native_full,
                        "afterok",
                        ("aids_project_smoke_gate",),
                        {
                            "OUTPUT_DIR": native_full,
                            "PREREGISTRATION": native_full + "/preregistration.json",
                        },
                        True,
                    ),
                    Stage(
                        "aids_native_full_gate",
                        "aids",
                        "scripts/slurm/comrecgc_aids_native_full_gate.sh",
                        native_full + "/gate",
                        "afterok",
                        ("aids_native_full",),
                        {"INPUT_DIR": native_full, "OUTPUT_DIR": native_full + "/gate"},
                    ),
                    Stage(
                        "aids_project_full_generation",
                        "aids",
                        "scripts/slurm/comrecgc_project_generate.sh",
                        project_full + "/generation",
                        "afterok",
                        ("aids_project_smoke_gate",),
                        {
                            "DATASET": "aids",
                            "MODE": "full",
                            "BASE_ROOT": project_full,
                            "PARENT_LIMIT": "1283",
                        },
                        True,
                    ),
                    Stage(
                        "aids_project_full_common_recourse",
                        "aids",
                        "scripts/slurm/comrecgc_common_recourse.sh",
                        project_full + "/common_recourse",
                        "afterok",
                        ("aids_project_full_generation",),
                        {
                            "DATASET": "aids",
                            "MODE": "full",
                            "BASE_ROOT": project_full,
                            "PARENT_LIMIT": "1283",
                        },
                    ),
                    Stage(
                        "aids_project_full_chemistry",
                        "aids",
                        "scripts/slurm/comrecgc_project_chemistry.sh",
                        project_full + "/chemistry",
                        "afterok",
                        ("aids_project_full_common_recourse",),
                        {
                            "DATASET": "aids",
                            "MODE": "full",
                            "BASE_ROOT": project_full,
                            "PARENT_LIMIT": "1283",
                        },
                    ),
                    Stage(
                        "aids_project_full_unified_eval",
                        "aids",
                        "scripts/slurm/comrecgc_project_slot_eval.sh",
                        project_full + "/unified_eval",
                        "afterok",
                        ("aids_project_full_chemistry",),
                        {"DATASET": "aids", "MODE": "full", "BASE_ROOT": project_full},
                    ),
                    Stage(
                        "aids_project_full_gate",
                        "aids",
                        "scripts/slurm/comrecgc_project_full_gate.sh",
                        project_full + "/full_gate",
                        "afterok",
                        ("aids_project_full_unified_eval",),
                        {"DATASET": "aids", "BASE_ROOT": project_full},
                    ),
                    Stage(
                        "aids_project_freeze",
                        "aids",
                        "scripts/slurm/comrecgc_project_freeze.sh",
                        standardized,
                        "afterok",
                        ("aids_project_full_gate",),
                        {
                            "DATASET": "aids",
                            "BASE_ROOT": project_full,
                            "STANDARDIZED_ROOT": standardized,
                            "AUTOMATION_STATE": (
                                f"outputs/hpc/automation/comrecgc_recovery/{run_id}/state.json"
                            ),
                        },
                    ),
                ]
            )
    if "mutagenicity" in datasets:
        smoke_base = f"outputs/hpc/baselines/comrecgc/mutagenicity/recovery_smoke_{run_id}"
        generation_output = adopted_outputs.get(
            "mut_trace_adopt", smoke_base + "/generation"
        )
        chemistry_output = adopted_outputs.get(
            "mut_chemistry_audit", smoke_base + "/chemistry"
        )
        unified_eval_output = adopted_outputs.get(
            "mut_unified_eval_smoke", smoke_base + "/unified_eval"
        )
        gate_output = adopted_outputs.get(
            "mut_chemrepair_smoke_gate", smoke_base + "/gate"
        )
        values.extend(
            [
                Stage(
                    "mut_trace_adopt",
                    "mutagenicity",
                    "scripts/slurm/comrecgc_mut_trace_adopt.sh",
                    generation_output,
                    None,
                    (),
                    {
                        "SOURCE_FAILED_GENERATION_DIR": (
                            "outputs/hpc/baselines/comrecgc/mutagenicity/"
                            "recovery_smoke_comrecgc_recovery_20260806_mut_retry1/"
                            "generation"
                        ),
                        "OUTPUT_DIR": generation_output,
                    },
                ),
                Stage(
                    "mut_chemistry_audit",
                    "mutagenicity",
                    "scripts/slurm/comrecgc_mut_chemistry_audit.sh",
                    chemistry_output,
                    "afterok",
                    ("mut_trace_adopt",),
                    {
                        "GENERATION_DIR": generation_output,
                        "OUTPUT_DIR": chemistry_output,
                    },
                ),
                Stage(
                    "mut_unified_eval_smoke",
                    "mutagenicity",
                    "scripts/slurm/comrecgc_mut_unified_eval.sh",
                    unified_eval_output,
                    "afterok",
                    ("mut_chemistry_audit",),
                    {
                        "MODE": "smoke",
                        "CHEMISTRY_DIR": chemistry_output,
                        "OUTPUT_DIR": unified_eval_output,
                    },
                ),
                Stage(
                    "mut_chemrepair_smoke_gate",
                    "mutagenicity",
                    "scripts/slurm/comrecgc_mut_chemrepair_smoke_gate.sh",
                    gate_output,
                    "afterok",
                    ("mut_unified_eval_smoke",),
                    {
                        "INPUT_DIR": chemistry_output,
                        "EVAL_DIR": unified_eval_output,
                        "OUTPUT_DIR": gate_output,
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
                        {"BASE_ROOT": full_base, "PARENT_LIMIT": "1448"},
                        True,
                    ),
                    Stage(
                        "mut_full_common_recourse",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_common_recourse.sh",
                        full_base + "/common_recourse",
                        "afterok",
                        ("mut_full_generation",),
                        {
                            "BASE_ROOT": full_base,
                            "DATASET": "mutagenicity",
                            "MODE": "full",
                            "PARENT_LIMIT": "1448",
                        },
                    ),
                    Stage(
                        "mut_full_chemistry",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_project_chemistry.sh",
                        full_base + "/chemistry",
                        "afterok",
                        ("mut_full_common_recourse",),
                        {
                            "BASE_ROOT": full_base,
                            "DATASET": "mutagenicity",
                            "MODE": "full",
                            "PARENT_LIMIT": "1448",
                            "TRACE_EVIDENCE_PATH": generation_output + "/trace_parity.json",
                        },
                    ),
                    Stage(
                        "mut_full_unified_eval",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_project_slot_eval.sh",
                        full_base + "/unified_eval",
                        "afterok",
                        ("mut_full_chemistry",),
                        {"BASE_ROOT": full_base, "DATASET": "mutagenicity", "MODE": "full"},
                    ),
                    Stage(
                        "mut_full_gate",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_project_full_gate.sh",
                        full_base + "/full_gate",
                        "afterok",
                        ("mut_full_unified_eval",),
                        {"BASE_ROOT": full_base, "DATASET": "mutagenicity"},
                    ),
                    Stage(
                        "mut_freeze",
                        "mutagenicity",
                        "scripts/slurm/comrecgc_project_freeze.sh",
                        "outputs/hpc/eval/paper/mutagenicity_common4_comrecgc_standardized_v1/comrecgc",
                        "afterok",
                        ("mut_full_gate",),
                        {
                            "BASE_ROOT": full_base,
                            "DATASET": "mutagenicity",
                            "STANDARDIZED_ROOT": "outputs/hpc/eval/paper/mutagenicity_common4_comrecgc_standardized_v1/comrecgc",
                            "AUTOMATION_STATE": f"outputs/hpc/automation/comrecgc_recovery/{run_id}/state.json",
                        },
                    ),
                ]
            )
    return values


def refresh(state: RecoveryState) -> None:
    verify_retry3_inputs(state)
    verify_completed_stage_adoptions(state)
    active = False
    failed: list[str] = []
    completed: list[str] = list(state.data.get("adopted_stage_outputs") or {})
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
    datasets = set(state.data.get("datasets") or [])
    terminal_stages = set()
    if "aids" in datasets:
        terminal_stages.update({"aids_native_full_gate", "aids_project_freeze"})
    if "mutagenicity" in datasets:
        terminal_stages.add("mut_freeze")
    if terminal_stages and terminal_stages <= set(completed):
        status = "END_TO_END_COMPLETED"
    elif failed and active:
        status = "RUNNING_WITH_BLOCKED_DATASET"
    elif failed:
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
        "authorization_sha256": state.data.get("authorization_sha256"),
        "job_dag_sha256": state.data.get("job_dag_sha256"),
        "input_sha256_before": state.data.get("input_artifact_set_sha256_before"),
        "input_sha256_after": state.data.get("input_artifact_set_sha256_after"),
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
    parser.add_argument(
        "--initialize-authorization",
        action="store_true",
        help="Create the exact run-scoped authorization and exit.",
    )
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--control-socket", default=DEFAULT_CONTROL_SOCKET)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--adopt-completed-stage",
        action="append",
        default=[],
        metavar="STAGE_ID=OUTPUT_PATH",
        help="Adopt an exact completed smoke-stage output without resubmitting it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.remote:
        return delegate_remote(args)
    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    if not datasets or not datasets <= {"aids", "mutagenicity"}:
        raise ValueError("--datasets must contain aids and/or mutagenicity.")
    state_root = PROJECT_ROOT / f"outputs/hpc/automation/comrecgc_recovery/{args.run_id}"
    current_commit = git_commit(PROJECT_ROOT)
    end_to_end = args.run_id.startswith(END_TO_END_RUN_PREFIX)
    if args.initialize_authorization:
        initializer = (
            initialize_end_to_end_authorization
            if end_to_end
            else initialize_retry3_authorization
        )
        path, digest = initializer(state_root, project_commit=current_commit)
        print(
            json.dumps(
                {
                    "authorization_path": str(path),
                    "authorization_sha256": digest,
                    "project_commit": current_commit,
                },
                sort_keys=True,
            )
        )
        return 0

    authorization: dict[str, Any] | None = None
    authorization_sha256: str | None = None
    if end_to_end:
        authorization, authorization_sha256 = load_end_to_end_authorization(
            state_root, project_commit=current_commit
        )
        if args.mode not in {"all", "refresh"}:
            raise RuntimeError("End-to-end authorization requires --mode all or refresh.")
        assert_full_authorized(authorization)
    elif "_retry3" in args.run_id:
        authorization, authorization_sha256 = load_retry3_authorization(
            state_root, project_commit=current_commit
        )
        if args.mode == "all":
            assert_full_authorized(authorization)

    state = RecoveryState(
        state_root,
        args.run_id,
        requested_mode=args.mode,
        datasets=sorted(datasets),
        authorization=authorization,
        authorization_sha256=authorization_sha256,
    )
    if state.data.get("project_commit") != current_commit:
        raise RuntimeError(
            "Recovery state project commit differs from current HEAD: "
            f"state={state.data.get('project_commit')}, current={current_commit}."
        )
    try:
        requested_mode = str(state.data.get("requested_mode") or args.mode)
        requested_datasets = set(state.data.get("datasets") or datasets)
        default_stage_values = stages(args.run_id, requested_datasets, requested_mode)
        register_completed_stage_adoptions(
            state,
            args.adopt_completed_stage,
            known_stage_ids={stage.stage_id for stage in default_stage_values},
        )
        stage_values = stages(
            args.run_id,
            requested_datasets,
            requested_mode,
            adopted_stage_outputs=dict(state.data.get("adopted_stage_outputs") or {}),
        )
        if end_to_end:
            write_authorized_job_dag(state, stage_values)
        if args.mode == "refresh":
            recover_registered_jobs(stage_values, state)
            refresh(state)
            if requested_mode == "all":
                assert_full_authorized(state.data.get("authorization"))
                validate_new_output_roots(stage_values, state)
                submitted = submit_ready_stages(
                    stage_values,
                    state,
                    dry_run=False,
                )
                if submitted:
                    status = (
                        "FULL_SUBMITTED"
                        if any(stage_id in FULL_STAGE_IDS for stage_id in submitted)
                        else "SMOKE_SUBMITTED"
                    )
                    state.transition(status, newly_submitted_stages=submitted)
            report(state)
        else:
            if state.data["status"] == "CREATED":
                preflight(state)
            recover_registered_jobs(stage_values, state)
            validate_new_output_roots(stage_values, state)
            submit_ready_stages(
                stage_values,
                state,
                dry_run=args.dry_run,
            )
            submitted_status = "DRY_RUN_COMPLETED" if args.dry_run else "SMOKE_SUBMITTED"
            state.transition(submitted_status)
            if not args.dry_run:
                command(
                    [sys.executable, "scripts/sync_experiment_status.py"],
                    cwd=PROJECT_ROOT,
                    timeout=180,
                    check=False,
                )
            report(state)
    except Exception as exc:
        state.transition(
            "BLOCKED",
            stop_reason=str(exc),
            error_class=type(exc).__name__,
        )
        report(state)
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "BLOCKED",
                    "error_class": type(exc).__name__,
                    "stop_reason": str(exc),
                    "state_path": str(state.state_path),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
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
