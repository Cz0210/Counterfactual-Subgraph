"""Fail-closed manifest builder for the BACE equivalence sidecar.

The main four-by-four manifests are immutable after first launch.  This module
therefore creates a separate controller which shares the audited global GPU
UUID locks.  It never signals or rewrites one of the protected long-running
jobs.  The already-running ComRecGC M=500 pair is recorded as protected
evidence instead of being launched a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from scripts.autodl.build_four_by_four_manifest import (
    DEFAULT_RESOURCE_GATES,
    DEFAULT_RUNTIME,
)
from scripts.autodl.run_four_gpu_recovery_controller import (
    load_controller_manifest,
    read_process_identity,
)


HEX40 = re.compile(r"[0-9a-f]{40}")
ACTIVE_RUN_STATES = {"STARTING", "RUNNING"}
TERMINAL_RUN_STATES = {"PASS", "FAILED", "BLOCKED"}


class BaceEquivalenceSidecarError(RuntimeError):
    """Raised when the sidecar cannot be frozen without changing science."""


@dataclass(frozen=True)
class ProtectedRun:
    run_id: str
    role: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BaceEquivalenceSidecarError(f"Required physical JSON is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BaceEquivalenceSidecarError(f"Expected one JSON object: {path}")
    return payload


def _publish_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one new JSON file without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise BaceEquivalenceSidecarError(f"Fresh output already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    candidate = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(candidate, path)
        except FileExistsError as exc:
            raise BaceEquivalenceSidecarError(
                f"Fresh output appeared concurrently: {path}"
            ) from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        candidate.unlink(missing_ok=True)


def _git_head(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    commit = result.stdout.strip()
    if HEX40.fullmatch(commit) is None:
        raise BaceEquivalenceSidecarError("Execution worktree has no full Git commit")
    return commit


def audit_protected_run(
    *,
    runtime_root: Path,
    run: ProtectedRun,
) -> dict[str, Any]:
    """Read and bind one existing exp-run and its exclusive UUID lock."""

    run_root = runtime_root / "control/experiment_registry/run_state" / run.run_id
    spec_path = run_root / "launch_spec.json"
    state_path = run_root / "state.json"
    spec = _read_object(spec_path)
    state = _read_object(state_path)
    failures: list[str] = []
    for payload_name, payload in (("spec", spec), ("state", state)):
        if payload.get("run_id") != run.run_id:
            failures.append(f"{payload_name}.run_id")
    observed = str(state.get("state", ""))
    if observed not in ACTIVE_RUN_STATES | TERMINAL_RUN_STATES:
        failures.append(f"state={observed}")
    if spec.get("gpu_lock_mode", "exclusive") != "exclusive":
        failures.append("gpu_lock_mode")
    if spec.get("gpu_uuid") != state.get("gpu_uuid"):
        failures.append("gpu_uuid")
    if spec.get("gpu_index") != state.get("gpu_index"):
        failures.append("gpu_index")
    worker_pid = state.get("pid")
    lock_path: Path | None = None
    lock_payload: dict[str, Any] | None = None
    process_identity: dict[str, Any] | None = None
    if observed in ACTIVE_RUN_STATES:
        if isinstance(worker_pid, bool) or not isinstance(worker_pid, int):
            failures.append("worker_pid")
        else:
            process_identity = read_process_identity(worker_pid)
            if process_identity is None:
                failures.append("worker_process_missing")
        gpu_uuid = str(state.get("gpu_uuid") or "")
        lock_path = runtime_root / "locks" / f"gpu-{gpu_uuid}.lock"
        try:
            lock_payload = _read_object(lock_path)
        except BaceEquivalenceSidecarError:
            failures.append("exclusive_lock_missing")
        else:
            if lock_payload.get("run_id") != run.run_id:
                failures.append("exclusive_lock_run_id")
            if lock_payload.get("pid") != worker_pid:
                failures.append("exclusive_lock_pid")
            if lock_payload.get("gpu_uuid") != gpu_uuid:
                failures.append("exclusive_lock_uuid")
    if failures:
        raise BaceEquivalenceSidecarError(
            f"Protected run {run.run_id} failed audit: {', '.join(failures)}"
        )
    return {
        "role": run.role,
        "run_id": run.run_id,
        "state": observed,
        "worker_pid": worker_pid,
        "child_pid": state.get("child_pid"),
        "gpu_index": state.get("gpu_index"),
        "gpu_uuid": state.get("gpu_uuid"),
        "project_root": spec.get("project_root"),
        "git_commit": spec.get("git_commit"),
        "expected_output": spec.get("expected_output"),
        "launch_spec": str(spec_path),
        "launch_spec_sha256": sha256_file(spec_path),
        "state_path": str(state_path),
        "process_identity": process_identity,
        "exclusive_lock": str(lock_path) if lock_path else None,
        "exclusive_lock_payload": lock_payload,
        "mutation_performed": False,
    }


def _gcf_task(
    *,
    task_id: str,
    stage: str,
    budget: int,
    replay_class: str,
    dependency: str | None,
    output_root: Path,
    input_manifest: Path,
    project_root: Path,
    python: Path,
    dataset_dir: Path,
    official_root: Path,
    gine_checkpoint: Path,
    neurosed_checkpoint: Path,
    neurosed_manifest: Path,
    priority: int,
) -> dict[str, Any]:
    marker = (
        f"[BACE_GCF_QUICK_{budget}_PASS]"
        if replay_class == "quick"
        else f"[BACE_GCF_EQUIVALENCE_M{budget}_PASS]"
    )
    return {
        "id": task_id,
        "dataset": "bace",
        "stage": stage,
        "runner_dataset": "bace-equivalence-sidecar",
        "runner_stage": stage,
        "depends_on": [dependency] if dependency else [],
        "resource": "gpu",
        "gpu_lock_mode": "exclusive",
        "priority": priority,
        "command": [
            "bash",
            "{project_root}/scripts/autodl/run_bace_gcf_equivalence_replay.sh",
        ],
        "environment": {
            "AUTODL_PYTHON": str(python),
            "BACE_GCF_DATASET_DIR": str(dataset_dir),
            "GCF_OFFICIAL_ROOT": str(official_root),
            "BACE_GINE_CHECKPOINT": str(gine_checkpoint),
            "BACE_NEUROSED_CHECKPOINT": str(neurosed_checkpoint),
            "BACE_NEUROSED_MANIFEST": str(neurosed_manifest),
            "BACE_GCF_REPLAY_OUTPUT": "{task_output}",
            "BACE_GCF_REPLAY_BUDGET": str(budget),
            "BACE_GCF_REPLAY_CLASS": replay_class,
            "BACE_GCF_CPU_NEIGHBOR_WORKERS": "4",
            "BACE_GCF_GINE_BATCH_SIZE": "256",
            "BACE_GCF_GRAPH_CACHE_CAPACITY": "100000",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "{project_root}",
            "RUN_TASTEMOLNET": "0",
        },
        "config_files": [str(project_root / "configs/hpc.yaml")],
        "input_manifest": str(input_manifest),
        "expected_output": str(output_root / task_id / "attempt-{attempt}"),
        "required_output_files": [
            "PASS",
            "replay_manifest.json",
            f"equivalence-m{budget}.json",
            "legacy/run_manifest.json",
            "ordered_v2/run_manifest.json",
        ],
        "required_log_marker": marker,
        "data_splits": ["train"],
        "manifest_only": False,
    }


def build_sidecar_manifest(
    *,
    controller_id: str,
    project_root: Path,
    runtime_root: Path,
    python: Path,
    output_root: Path,
    output_manifest: Path,
    build_audit: Path,
    protected_runs: Sequence[ProtectedRun],
    comrec_run_id: str,
    dataset_dir: Path,
    gcf_official_root: Path,
    gine_checkpoint: Path,
    neurosed_checkpoint: Path,
    neurosed_manifest: Path,
) -> dict[str, Any]:
    """Freeze a fresh sidecar manifest and its protected-task audit."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", controller_id):
        raise BaceEquivalenceSidecarError("Unsafe controller_id")
    roots = {
        "project_root": project_root.resolve(strict=True),
        "runtime_root": runtime_root.resolve(strict=True),
        "python": python.resolve(strict=True),
        "dataset_dir": dataset_dir.resolve(strict=True),
        "gcf_official_root": gcf_official_root.resolve(strict=True),
        "gine_checkpoint": gine_checkpoint.resolve(strict=True),
        "neurosed_checkpoint": neurosed_checkpoint.resolve(strict=True),
        "neurosed_manifest": neurosed_manifest.resolve(strict=True),
    }
    for path in (output_root, output_manifest, build_audit):
        if path.exists() or path.is_symlink():
            raise BaceEquivalenceSidecarError(f"Sidecar target is not fresh: {path}")
    run_rows = [
        audit_protected_run(runtime_root=roots["runtime_root"], run=run)
        for run in protected_runs
    ]
    if len(run_rows) != 4 or len({row["run_id"] for row in run_rows}) != 4:
        raise BaceEquivalenceSidecarError("Exactly four distinct protected runs required")
    comrec_rows = [row for row in run_rows if row["run_id"] == comrec_run_id]
    if len(comrec_rows) != 1:
        raise BaceEquivalenceSidecarError("ComRecGC M=500 protected run is absent")
    comrec = comrec_rows[0]
    commit = _git_head(roots["project_root"])
    audit_payload = {
        "schema_version": "bace_equivalence_sidecar_build_audit_v1",
        "status": "PASS",
        "controller_id": controller_id,
        "execution_commit": commit,
        "execution_project_root": str(roots["project_root"]),
        "protected_run_count": 4,
        "protected_runs": run_rows,
        "existing_controller_append_supported": False,
        "append_block_reason": "controller manifest SHA and task topology are frozen",
        "old_full_stop_requested": False,
        "old_full_signal_sent": False,
        "gpu_lock_protocol": "global_exclusive_uuid_lock",
        "comrecgc_m500_policy": "observe_existing_no_duplicate",
    }
    # The manifest loader does not need the input file bytes at schema time,
    # but exp_run hashes this exact physical audit before launching each task.
    tasks = [
        _gcf_task(
            task_id="bace_gcf_quick_50",
            stage="BACE_GCF_QUICK_EQUIVALENCE_50",
            budget=50,
            replay_class="quick",
            dependency=None,
            output_root=output_root,
            input_manifest=build_audit,
            project_root=roots["project_root"],
            python=roots["python"],
            dataset_dir=roots["dataset_dir"],
            official_root=roots["gcf_official_root"],
            gine_checkpoint=roots["gine_checkpoint"],
            neurosed_checkpoint=roots["neurosed_checkpoint"],
            neurosed_manifest=roots["neurosed_manifest"],
            priority=10,
        ),
        _gcf_task(
            task_id="bace_gcf_quick_100",
            stage="BACE_GCF_QUICK_EQUIVALENCE_100",
            budget=100,
            replay_class="quick",
            dependency="bace_gcf_quick_50",
            output_root=output_root,
            input_manifest=build_audit,
            project_root=roots["project_root"],
            python=roots["python"],
            dataset_dir=roots["dataset_dir"],
            official_root=roots["gcf_official_root"],
            gine_checkpoint=roots["gine_checkpoint"],
            neurosed_checkpoint=roots["neurosed_checkpoint"],
            neurosed_manifest=roots["neurosed_manifest"],
            priority=20,
        ),
        _gcf_task(
            task_id="bace_gcf_equivalence_500",
            stage="BACE_GCF_FORMAL_EQUIVALENCE_500",
            budget=500,
            replay_class="formal",
            dependency="bace_gcf_quick_100",
            output_root=output_root,
            input_manifest=build_audit,
            project_root=roots["project_root"],
            python=roots["python"],
            dataset_dir=roots["dataset_dir"],
            official_root=roots["gcf_official_root"],
            gine_checkpoint=roots["gine_checkpoint"],
            neurosed_checkpoint=roots["neurosed_checkpoint"],
            neurosed_manifest=roots["neurosed_manifest"],
            priority=30,
        ),
    ]
    runtime = dict(DEFAULT_RUNTIME)
    runtime.update(
        {
            "max_gpus": 4,
            "max_cpu_tasks": 1,
            "stable_idle_seconds": 60,
            "sample_interval_seconds": 5,
            "poll_seconds": 60,
            "max_transient_retries": 1,
            "keep_alive_when_blocked": True,
        }
    )
    manifest_payload = {
        "schema_version": 1,
        "controller_id": controller_id,
        "paper_frozen": True,
        "runtime": runtime,
        "resource_gates": dict(DEFAULT_RESOURCE_GATES),
        "sidecar_policy": audit_payload,
        "tasks": tasks,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    candidate = output_manifest.with_name(
        f".{output_manifest.name}.candidate-{os.getpid()}"
    )
    try:
        _publish_new_json(candidate, manifest_payload)
        load_controller_manifest(candidate)
        _publish_new_json(build_audit, audit_payload)
        try:
            os.link(candidate, output_manifest)
        except FileExistsError as exc:
            raise BaceEquivalenceSidecarError(
                f"Manifest appeared concurrently: {output_manifest}"
            ) from exc
    except Exception:
        build_audit.unlink(missing_ok=True)
        output_manifest.unlink(missing_ok=True)
        raise
    finally:
        candidate.unlink(missing_ok=True)
    frozen = load_controller_manifest(output_manifest)
    return {
        "status": "PASS",
        "controller_id": controller_id,
        "manifest": str(output_manifest),
        "manifest_sha256": frozen.sha256,
        "build_audit": str(build_audit),
        "build_audit_sha256": sha256_file(build_audit),
        "output_root": str(output_root),
        "task_order": [task.task_id for task in frozen.tasks],
        "protected_run_count": 4,
        "existing_comrecgc_m500_launch_spec_sha256": comrec[
            "launch_spec_sha256"
        ],
        "existing_comrecgc_m500_scheduled_again": False,
        "old_full_stop_requested": False,
    }


__all__ = [
    "BaceEquivalenceSidecarError",
    "ProtectedRun",
    "audit_protected_run",
    "build_sidecar_manifest",
    "sha256_file",
]
