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
FULL_STAGE_IDS = frozenset(
    {
        "aids_native_full",
        "aids_native_full_gate",
        "mut_full_generation",
        "mut_full_common_recourse",
        "mut_full_chemistry",
        "mut_full_unified_eval",
        "mut_full_gate",
        "mut_freeze",
    }
)


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


def assert_full_authorized(authorization: dict[str, Any] | None) -> None:
    if authorization is None:
        raise RuntimeError("Full submission requires an explicit authorization object.")
    if authorization.get("phase_c_full_approved") is not True:
        raise RuntimeError("Full submission blocked: phase_c_full_approved is not true.")
    if authorization.get("full_submission_allowed") is not True:
        raise RuntimeError("Full submission blocked: full_submission_allowed is not true.")


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
    for stage in stage_values:
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
    if stage.stage_id in FULL_STAGE_IDS:
        assert_full_authorized(state.data.get("authorization"))
    if "_retry3" in str(state.data.get("run_id")):
        validate_job_caps(state, next_stage=stage)
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


def validate_new_output_roots(
    stage_values: Sequence[Stage], state: RecoveryState
) -> None:
    """Refuse an unregistered retry output instead of guessing its provenance."""

    collisions: list[str] = []
    for stage in stage_values:
        if state.job(stage.stage_id) is not None:
            continue
        output = (PROJECT_ROOT / stage.output_root).resolve()
        if output.exists():
            collisions.append(str(output))
    if collisions:
        raise RuntimeError(f"Retry3 output collision: {collisions}")


def stages(run_id: str, datasets: set[str], mode: str) -> list[Stage]:
    values: list[Stage] = []
    if "aids" in datasets:
        smoke_base = f"outputs/hpc/baselines/comrecgc/aids/recovery_smoke_{run_id}"
        existing_audit = smoke_base + "/existing_audit"
        density_audit = smoke_base + "/parent_density"
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
                    "afterok",
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
    verify_retry3_inputs(state)
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
        "authorization_sha256": state.data.get("authorization_sha256"),
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
        help="Create the exact smoke-only retry3 authorization and exit.",
    )
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
    state_root = PROJECT_ROOT / f"outputs/hpc/automation/comrecgc_recovery/{args.run_id}"
    current_commit = git_commit(PROJECT_ROOT)
    if args.initialize_authorization:
        path, digest = initialize_retry3_authorization(
            state_root, project_commit=current_commit
        )
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
    if "_retry3" in args.run_id:
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
        if args.mode == "refresh":
            recover_registered_jobs(
                stages(
                    args.run_id,
                    set(state.data.get("datasets") or datasets),
                    str(state.data.get("requested_mode") or "smoke"),
                ),
                state,
            )
            refresh(state)
            requested_mode = str(state.data.get("requested_mode") or "smoke")
            requested_datasets = set(state.data.get("datasets") or datasets)
            if requested_mode == "all" and not state.data.get("failed_stages"):
                assert_full_authorized(state.data.get("authorization"))
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
            stage_values = stages(args.run_id, datasets, args.mode)
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
