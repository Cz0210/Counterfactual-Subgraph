"""Build one fresh, throttled Mut adoption successor task specification.

This module is intentionally dataset specific.  It converts one terminal
operationally-failed Mut owner specification into a fresh-vs-fresh successor
without changing any scientific input, adoption hash, or matrix authority.
It never launches science by itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from src.utils.autodl_mut_throttled_continuation_v1 import find_live_mut_writers
from src.utils.main_ready_task_specs import (
    MainReadyTaskSpecError,
    atomic_json,
    file_sha256,
    manifest_for_specs,
    materialize_task_spec_path,
    process_identity,
    seal_spec,
    load_spec,
)


SCHEMA = "mut_clean_successor_v2_preflight"
ALLOWED_PREDECESSOR_TERMINALS = frozenset(
    {
        "FAILED_REVIEWED_WORKER_EXIT_1",
        "FAILED_REVIEWED_WORKER_TERMINAL",
    }
)


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MainReadyTaskSpecError(f"expected one physical JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MainReadyTaskSpecError(f"expected one JSON object: {path}")
    return value


def validate_predecessor_terminal(
    *,
    prior_spec: Mapping[str, Any],
    terminal_path: Path,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Prove the predecessor is terminal and that no Mut writer remains."""

    terminal = _load_object(terminal_path)
    status = str(terminal.get("status") or "")
    if status not in ALLOWED_PREDECESSOR_TERMINALS:
        raise MainReadyTaskSpecError(
            f"predecessor terminal is not an allowed operational failure: {status}"
        )
    if (
        terminal.get("task_id") != prior_spec.get("task_id")
        or terminal.get("output_root") != prior_spec.get("output_root")
        or not isinstance(terminal.get("owner_pid"), int)
        or not isinstance(terminal.get("owner_start_ticks"), int)
    ):
        raise MainReadyTaskSpecError("predecessor terminal is not bound to its task spec")
    identity = process_identity(int(terminal["owner_pid"]), proc_root=proc_root)
    if identity is not None and identity.get("alive"):
        if int(identity.get("start_ticks", -1)) == int(terminal["owner_start_ticks"]):
            raise MainReadyTaskSpecError("predecessor owner is still alive")
    writers = find_live_mut_writers(proc_root)
    if writers:
        raise MainReadyTaskSpecError(
            "a Mut scientific writer already exists: "
            + ",".join(str(row["pid"]) for row in writers)
        )
    return {
        "schema_version": SCHEMA,
        "status": "PASS_PREDECESSOR_OPERATIONAL_FAILURE_NO_LIVE_WRITER",
        "predecessor_task_id": prior_spec["task_id"],
        "predecessor_owner_pid": terminal["owner_pid"],
        "predecessor_owner_start_ticks": terminal["owner_start_ticks"],
        "predecessor_terminal_status": status,
        "predecessor_output_root": terminal["output_root"],
        "live_mut_writers": [],
        "science_failure_claimed": False,
        "fresh_successor_required": True,
    }


def _git_head(repo_root: Path) -> str:
    if not repo_root.is_absolute() or repo_root.is_symlink() or not repo_root.is_dir():
        raise MainReadyTaskSpecError("successor repo root must be one physical directory")
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_successor_spec(
    *,
    prior_spec: Mapping[str, Any],
    repo_root: Path,
    task_id: str,
    attempt_uuid: str,
    spec_path: Path,
    output_root: Path,
    owner_runtime_root: Path,
    gpu_index: int,
    gpu_uuid: str,
    lease_path: Path,
) -> dict[str, Any]:
    """Return a sealed spec that changes ownership only, not science inputs."""

    entrypoint = repo_root / "scripts/autodl/run_mut_clean_trace_equivalence_v1.py"
    config = repo_root / "configs/hpc.yaml"
    execution_commit = _git_head(repo_root)
    if any(path.exists() or path.is_symlink() for path in (output_root, owner_runtime_root)):
        raise FileExistsError("successor output and owner roots must both be fresh")
    value = dict(prior_spec)
    value.update(
        {
            "task_id": task_id,
            "attempt_uuid": attempt_uuid,
            "repo_root": str(repo_root),
            "execution_commit": execution_commit,
            "entrypoint": str(entrypoint),
            "config_path": str(config),
            "config_sha256": file_sha256(config),
            "output_root": str(output_root),
            "gpu_request": {
                "index": int(gpu_index),
                "uuid": gpu_uuid,
                "lease_path": str(lease_path),
                "lease_scope": "MAIN_READY_DISPATCH_OWNER",
                "selection_policy": "reviewed_worker_revalidates_first_idle_main_gpu",
            },
            "cpu_request": {
                "workers": 2,
                "protected_task_baseline_seconds": 1800,
                "nice": 10,
                "ionice_class": 2,
                "ionice_priority": 7,
            },
            "required_environment": {
                **dict(prior_spec["required_environment"]),
                "MUT_PROTECTED_BASELINE_MAX_WAIT_SECONDS": "1800",
                "MUT_EXACT_WORKERS": "2",
                "MUT_CPU_WORKERS": "2",
                "MUT_PREFETCH": "1",
                "MUT_PREFETCH_FACTOR": "1",
                "RUN_GNN_ABLATION": "0",
                "RUN_LLM_ABLATION": "0",
            },
            "expected_heartbeat_path": str(owner_runtime_root / "heartbeat.json"),
            "expected_pid_file": str(owner_runtime_root / "owner_pid.json"),
            "expected_terminal_path": str(owner_runtime_root / "terminal.json"),
            "owner_probe": {
                **dict(prior_spec.get("owner_probe") or {}),
                "expected_cwd": str(repo_root),
                "max_age_seconds": 120,
            },
            "arguments": ["--task-spec", str(spec_path)],
            "resume_policy": "fresh_trace_on_and_trace_off_then_separate_reload_parity",
            "single_writer_policy": "fail_if_live_owner_or_output_writer",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    contract = dict(prior_spec["science_contract"])
    contract.update(
        {
            "trace_on_root": str(output_root / "equivalence/trace_on"),
            "trace_off_root": str(output_root / "equivalence/trace_off"),
            "predecessor_task_id": prior_spec["task_id"],
            "predecessor_output_root": prior_spec["output_root"],
            "predecessor_result_adopted": False,
            "fresh_vs_fresh": True,
        }
    )
    value["science_contract"] = contract
    value.pop("expected_owner_command_sha256", None)
    value.pop("spec_sha256", None)
    return seal_spec(materialize_task_spec_path(value, spec_path))


def publish_successor_bundle(
    *, output_root: Path, spec: Mapping[str, Any], spec_path: Path
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"successor spec bundle must be fresh: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".tmp", dir=output_root.parent
        )
    )
    published = False
    try:
        staged_spec = staging / spec_path.name
        atomic_json(staged_spec, spec)
        manifest = manifest_for_specs([staged_spec], published_paths=[spec_path])
        atomic_json(staging / "task_specs_manifest.json", manifest)
        directory = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.rename(staging, output_root)
        published = True
        directory = os.open(output_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return {
        "status": "PASS",
        "spec_path": str(spec_path),
        "spec_sha256": spec["spec_sha256"],
        "manifest_path": str(output_root / "task_specs_manifest.json"),
        "execution_commit": spec["execution_commit"],
        "task_id": spec["task_id"],
        "attempt_uuid": spec["attempt_uuid"],
        "output_root": spec["output_root"],
    }


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-spec", type=_absolute, required=True)
    parser.add_argument("--prior-terminal", type=_absolute, required=True)
    parser.add_argument("--repo-root", type=_absolute, required=True)
    parser.add_argument("--spec-root", type=_absolute, required=True)
    parser.add_argument("--owner-runtime-root", type=_absolute, required=True)
    parser.add_argument("--science-output-root", type=_absolute, required=True)
    parser.add_argument("--lease-path", type=_absolute, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--attempt-uuid", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--preflight-output", type=_absolute, required=True)
    args = parser.parse_args(argv)

    prior = load_spec(args.prior_spec)
    preflight = validate_predecessor_terminal(
        prior_spec=prior,
        terminal_path=args.prior_terminal,
    )
    spec_path = args.spec_root / f"{args.task_id}.json"
    spec = build_successor_spec(
        prior_spec=prior,
        repo_root=args.repo_root,
        task_id=args.task_id,
        attempt_uuid=args.attempt_uuid,
        spec_path=spec_path,
        output_root=args.science_output_root,
        owner_runtime_root=args.owner_runtime_root,
        gpu_index=args.gpu_index,
        gpu_uuid=args.gpu_uuid,
        lease_path=args.lease_path,
    )
    result = publish_successor_bundle(
        output_root=args.spec_root,
        spec=spec,
        spec_path=spec_path,
    )
    atomic_json(
        args.preflight_output,
        {
            **preflight,
            "successor_task_id": args.task_id,
            "successor_attempt_uuid": args.attempt_uuid,
            "successor_spec_path": str(spec_path),
            "successor_spec_sha256": spec["spec_sha256"],
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
