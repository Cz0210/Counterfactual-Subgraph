from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid

import pytest

from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from src.utils.autodl_bace_equivalence_sidecar import (
    BaceEquivalenceSidecarError,
    ProtectedRun,
    audit_protected_run,
    build_sidecar_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _protected(
    runtime: Path,
    run_id: str,
    *,
    state: str = "PASS",
    gpu_index: int = 0,
) -> ProtectedRun:
    gpu_uuid = f"GPU-test-{gpu_index}"
    root = runtime / "control/experiment_registry/run_state" / run_id
    _write(
        root / "launch_spec.json",
        {
            "run_id": run_id,
            "gpu_lock_mode": "exclusive",
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "project_root": str(PROJECT_ROOT),
            "git_commit": "a" * 40,
            "expected_output": str(runtime / "outputs" / run_id),
        },
    )
    _write(
        root / "state.json",
        {
            "run_id": run_id,
            "state": state,
            "pid": os.getpid() if state == "RUNNING" else None,
            "child_pid": None,
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
        },
    )
    if state == "RUNNING":
        _write(
            runtime / "locks" / f"gpu-{gpu_uuid}.lock",
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "gpu_index": gpu_index,
                "gpu_uuid": gpu_uuid,
            },
        )
    return ProtectedRun(run_id, f"role-{gpu_index}")


def _scientific_inputs(tmp_path: Path) -> dict[str, Path]:
    values = {
        "dataset_dir": tmp_path / "dataset",
        "gcf_official_root": tmp_path / "official",
        "gine_checkpoint": tmp_path / "gine",
        "neurosed_checkpoint": tmp_path / "neurosed.pt",
        "neurosed_manifest": tmp_path / "neurosed.json",
    }
    values["dataset_dir"].mkdir()
    values["gcf_official_root"].mkdir()
    values["gine_checkpoint"].mkdir()
    values["neurosed_checkpoint"].write_bytes(b"weights")
    values["neurosed_manifest"].write_text("{}\n", encoding="utf-8")
    return values


def test_audit_protected_running_run_requires_matching_uuid_lock(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    run = _protected(runtime, "live-run", state="RUNNING", gpu_index=2)
    evidence = audit_protected_run(runtime_root=runtime, run=run)
    assert evidence["state"] == "RUNNING"
    assert evidence["worker_pid"] == os.getpid()
    assert evidence["mutation_performed"] is False

    lock = runtime / "locks/gpu-GPU-test-2.lock"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["run_id"] = "another-run"
    _write(lock, payload)
    with pytest.raises(BaceEquivalenceSidecarError, match="exclusive_lock_run_id"):
        audit_protected_run(runtime_root=runtime, run=run)


def test_builds_gcf_only_immutable_queue_and_never_duplicates_comrec(
    tmp_path: Path,
) -> None:
    # The production controller intentionally rejects any path segment named
    # ``test``.  Keep the fixture below pytest's parent but outside its
    # function-named ``test_*`` directory so the real leakage guard runs.
    safe_root = tmp_path.parent / f"bace_sidecar_fixture_{uuid.uuid4().hex}"
    runtime = safe_root / "runtime"
    runs = [
        _protected(runtime, f"protected-{index}", gpu_index=index)
        for index in range(4)
    ]
    inputs = _scientific_inputs(safe_root)
    manifest = safe_root / "manifest.json"
    audit = safe_root / "build-audit.json"
    output = safe_root / "outputs"
    result = build_sidecar_manifest(
        controller_id="bace_equivalence_sidecar_test",
        project_root=PROJECT_ROOT,
        runtime_root=runtime,
        python=Path(sys.executable),
        output_root=output,
        output_manifest=manifest,
        build_audit=audit,
        protected_runs=runs,
        comrec_run_id="protected-2",
        **inputs,
    )
    assert result["status"] == "PASS"
    assert result["existing_comrecgc_m500_scheduled_again"] is False
    frozen = load_controller_manifest(manifest)
    assert [task.task_id for task in frozen.tasks] == [
        "bace_gcf_quick_50",
        "bace_gcf_quick_100",
        "bace_gcf_equivalence_500",
    ]
    assert frozen.by_id["bace_gcf_quick_50"].depends_on == ()
    assert frozen.by_id["bace_gcf_quick_100"].depends_on == (
        "bace_gcf_quick_50",
    )
    assert frozen.by_id["bace_gcf_equivalence_500"].depends_on == (
        "bace_gcf_quick_100",
    )
    assert all(task.resource == "gpu" for task in frozen.tasks)
    assert all(task.gpu_lock_mode == "exclusive" for task in frozen.tasks)
    assert all("{attempt}" in str(task.expected_output) for task in frozen.tasks)
    assert not any("comrec" in task.task_id for task in frozen.tasks)
    build_evidence = json.loads(audit.read_text(encoding="utf-8"))
    assert build_evidence["protected_run_count"] == 4
    assert build_evidence["old_full_signal_sent"] is False
    assert build_evidence["existing_controller_append_supported"] is False

    with pytest.raises(BaceEquivalenceSidecarError, match="not fresh"):
        build_sidecar_manifest(
            controller_id="bace_equivalence_sidecar_test",
            project_root=PROJECT_ROOT,
            runtime_root=runtime,
            python=Path(sys.executable),
            output_root=output,
            output_manifest=manifest,
            build_audit=audit,
            protected_runs=runs,
            comrec_run_id="protected-2",
            **inputs,
        )


def test_builder_requires_exactly_four_distinct_protected_runs(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runs = [
        _protected(runtime, f"protected-{index}", gpu_index=index)
        for index in range(3)
    ]
    inputs = _scientific_inputs(tmp_path)
    with pytest.raises(BaceEquivalenceSidecarError, match="Exactly four"):
        build_sidecar_manifest(
            controller_id="bace_equivalence_sidecar_short",
            project_root=PROJECT_ROOT,
            runtime_root=runtime,
            python=Path(sys.executable),
            output_root=tmp_path / "outputs",
            output_manifest=tmp_path / "manifest.json",
            build_audit=tmp_path / "audit.json",
            protected_runs=runs,
            comrec_run_id="protected-2",
            **inputs,
        )
