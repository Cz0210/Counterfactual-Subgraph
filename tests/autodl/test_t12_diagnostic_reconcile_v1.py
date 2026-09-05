from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from src.baselines.tastemolnet_gcf_full import run_t12_generation_segment
from src.eval.tastemolnet_t3_calibration_v2 import CANDIDATE_CHECKPOINT_FILES
from src.train.tastemolnet_clean_policy_init import _inventory_directory
from src.utils.final16_owner_registry_v1 import build_owner_registry
from src.utils.main_ready_task_specs import stable_sha256
from src.utils.tastemolnet_t12_diagnostic_reconcile_v1 import (
    ENGINEERING_ERROR,
    T12DiagnosticReconcileError,
    file_sha256,
    reconcile_diagnostic_510,
    reconcile_registry_after_diagnostic,
    reconcile_registry_file_after_diagnostic,
)
from src.utils.tastemolnet_t12_formal_profile_v1 import (
    FORMAL_PRODUCTION_CHECKPOINT_CURSORS,
)
from src.utils.tastemolnet_t12_fresh_zero_plan_v1 import (
    T12FreshZeroPlanError,
    build_fresh_zero_plan,
)


SHA = "a" * 64
COMMIT = "b" * 40


def _proc(proc: Path, pid: int, ticks: int) -> None:
    target = proc / str(pid)
    target.mkdir(parents=True)
    fields = ["0"] * 20
    fields[0] = "S"
    fields[19] = str(ticks)
    (target / "stat").write_text(
        f"{pid} (t12 fixture) " + " ".join(fields) + "\n", encoding="utf-8"
    )


def _diagnostic_fixture(tmp_path: Path) -> dict[str, Path | int]:
    science = tmp_path / "science"
    checkpoints = science / "checkpoints"
    checkpoints.mkdir(parents=True)
    (science / "run_identity.json").write_text(
        json.dumps({"identity_template": {"sample_size": 10000}}), encoding="utf-8"
    )
    contract: dict[str, str] = {}
    for cursor in (500, 510):
        payload = checkpoints / f"checkpoint-{cursor:08d}.pt"
        payload.write_bytes(f"payload-{cursor}".encode())
        import hashlib

        manifest = {
            "schema_version": "tastemolnet_t12_checkpoint_manifest_v1",
            "status": "COMMITTED",
            "stage": "T12_GCF_FULL_GENERATION",
            "payload_file": payload.name,
            "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            "payload_bytes": payload.stat().st_size,
            "checkpoint_cursor": cursor,
            "total_steps": 510,
            "purpose": "production",
            "attempt_id": str(uuid.uuid4()),
            "generation_token": SHA,
            "identity_sha256": SHA,
            "state_sha256": SHA,
            "rng_sha256": SHA,
            "written_at": "2026-09-05T00:00:00+00:00",
            "immutable_no_replace": True,
        }
        manifest_path = checkpoints / f"checkpoint-{cursor:08d}.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        contract[f"accelerated_checkpoint_{cursor}"] = str(manifest_path)
    receipt500 = {
        "status": "GENERATION_CHECKPOINT_COMMITTED",
        "checkpoint_cursor": 500,
        "candidate_manifest": None,
        "candidate_manifest_sha256": None,
        "calibration_loaded": False,
        "test_loaded": False,
        "paper_cell_pass": False,
    }
    (science / "generation_receipt_00000500.json").write_text(
        json.dumps(receipt500), encoding="utf-8"
    )
    spec = {
        "task_id": "t12-accelerated-fixture",
        "task_kind": "T12_ACCELERATED_FROM_CHECKPOINT250",
        "output_root": str(science),
        "science_contract": contract,
        "spec_sha256": "0" * 64,
    }
    spec["spec_sha256"] = stable_sha256(
        {key: value for key, value in spec.items() if key != "spec_sha256"}
    )
    spec_path = tmp_path / "task-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    terminal = {
        "task_id": spec["task_id"],
        "status": "FAILED_AT_510",
        "exit_code": 1,
        "completed_step": 500,
        "owner_pid": 218411,
        "owner_start_ticks": 555,
        "output_root": str(science),
        "gpu_lock_held": False,
        "reference_signaled": False,
    }
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    log_path = tmp_path / "segment-510.log"
    log_path.write_text(f"RuntimeError: {ENGINEERING_ERROR}\n", encoding="utf-8")
    proc = tmp_path / "proc"
    proc.mkdir()
    return {
        "science": science,
        "spec": spec_path,
        "terminal": terminal_path,
        "log": log_path,
        "proc": proc,
        "pid": 218411,
        "ticks": 555,
    }


def test_accelerated_segment_disables_diagnostic_candidate_materialization() -> None:
    assert inspect.signature(run_t12_generation_segment).parameters[
        "materialize_terminal_candidates"
    ].default is True
    source = Path("scripts/autodl/run_t12_accelerated_from250_v1.py").read_text()
    assert "materialize_terminal_candidates=False" in source
    assert "diagnostic_only=True" in source


def test_formal_checkpoint_profile_reaches_every_declared_boundary() -> None:
    source = """
import json
from src.utils.tastemolnet_t12_formal_profile_v1 import (
    FORMAL_PRODUCTION_CHECKPOINT_CURSORS,
    configure_t12_formal_production_profile,
)
configure_t12_formal_production_profile()
from src.baselines.tastemolnet_gcf_full_resume import production_segment_bounds
cursor = 0
observed = []
for expected in FORMAL_PRODUCTION_CHECKPOINT_CURSORS:
    start, end = production_segment_bounds(cursor)
    observed.append([start, end])
    assert end == expected
    cursor = end
print(json.dumps(observed))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", source], text=True, cwd=Path.cwd()
    )
    observed = json.loads(output)
    assert observed[0] == [1, 100]
    assert observed[-1] == [17_501, 20_000]


def test_completed_510_is_reconciled_without_rerun_or_candidates(tmp_path: Path) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    seen: list[int] = []

    def reopen(_path: Path, *, expected_identity: dict) -> dict:
        seen.append(expected_identity["checkpoint_cursor"])
        return {"identity_sha256": SHA, "state_sha256": SHA, "rng_sha256": SHA}

    result = reconcile_diagnostic_510(
        task_spec_path=fixture["spec"],
        source_terminal_path=fixture["terminal"],
        segment_510_log_path=fixture["log"],
        overlay_root=tmp_path / "overlay",
        expected_owner_pid=fixture["pid"],
        expected_owner_start_ticks=fixture["ticks"],
        proc_root=fixture["proc"],
        checkpoint_reopener=reopen,
    )
    assert seen == [500, 510]
    assert result["checkpoint_510_rerun"] is False
    terminal = json.loads(Path(result["terminal"]).read_text())
    assert terminal["completed_step"] == 510
    assert terminal["native_candidates_materialized"] is False
    assert terminal["diagnostic_only"] is True
    assert terminal["promotion_allowed"] is False
    assert not (fixture["science"] / "generation_receipt_00000510.json").exists()


def test_reconcile_refuses_a_live_or_reused_owner_pid(tmp_path: Path) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    _proc(fixture["proc"], fixture["pid"], fixture["ticks"] + 1)
    with pytest.raises(T12DiagnosticReconcileError, match="PID is still present"):
        reconcile_diagnostic_510(
            task_spec_path=fixture["spec"], source_terminal_path=fixture["terminal"],
            segment_510_log_path=fixture["log"], overlay_root=tmp_path / "overlay",
            expected_owner_pid=fixture["pid"], expected_owner_start_ticks=fixture["ticks"],
            proc_root=fixture["proc"], checkpoint_reopener=lambda *a, **k: {},
        )


def test_registry_retires_only_dead_accelerated_owner_and_releases_its_lease(tmp_path: Path) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    authority = tmp_path / "matrix"
    authority.mkdir()
    lease = tmp_path / "gpu1.lock"
    lease.write_text("")
    publisher = {
        "publisher_id": "t12-paper",
        "cell_id": "TasteMolNet/GCFExplainer",
        "owner_state": "PREDEPLOYED", "owner_pid": None, "owner_start_ticks": None,
        "heartbeat": None, "locator": str(tmp_path / "locator.json"),
        "lease_path": str(tmp_path / "publisher.lock"), "execution_commit": COMMIT,
        "claim_enabled": True, "active_writer_count": 0,
    }
    task = {
        "task_id": "t12-accelerated-fixture", "dataset": "TasteMolNet",
        "method": "GCFExplainer", "stage": "DIAGNOSTIC_510",
        "owner_state": "RUNNING", "owner_pid": fixture["pid"],
        "owner_start_ticks": fixture["ticks"], "heartbeat": str(tmp_path / "old-heartbeat.json"),
        "input_root": str(tmp_path / "input"), "output_root": str(fixture["science"]),
        "execution_commit": COMMIT, "task_spec_sha": SHA, "gpu": 1,
        "successor_task_id": None, "publisher_id": "t12-paper",
    }
    registry = build_owner_registry(
        registry_id="fixture", matrix_authority_root=authority, tasks=[task],
        publishers=[publisher], gpu_leases=[{"gpu": 1, "task_id": task["task_id"], "state": "HELD", "lease_path": str(lease)}],
        proc_root=fixture["proc"], check_processes=False,
    )
    terminal = {
        "status": "DIAGNOSTIC_510_RECONCILED_ENGINEERING_FAILURE",
        "task_id": task["task_id"], "native_candidates_materialized": False,
        "diagnostic_only": True, "promotion_allowed": False, "paper_cell_pass": False,
    }
    terminal["terminal_sha256"] = stable_sha256(terminal)
    updated = reconcile_registry_after_diagnostic(
        registry=registry, expected_registry_sha256=registry["self_sha256"],
        task_id=task["task_id"], expected_owner_pid=fixture["pid"],
        expected_owner_start_ticks=fixture["ticks"], diagnostic_terminal=terminal,
        proc_root=fixture["proc"],
    )
    assert updated["tasks"][0]["owner_state"] == "TERMINAL_FAILED_ENGINEERING"
    assert updated["tasks"][0]["owner_pid"] is None
    assert updated["gpu_leases"][0]["state"] == "RELEASED"
    assert updated["publishers"] == registry["publishers"]


def test_registry_file_reconcile_is_sha_cas_under_publish_lock(tmp_path: Path) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    authority = tmp_path / "matrix"
    authority.mkdir()
    lease = tmp_path / "gpu1.lock"
    lease.write_text("")
    task = {
        "task_id": "t12-accelerated-fixture", "dataset": "TasteMolNet",
        "method": "GCFExplainer", "stage": "DIAGNOSTIC_510",
        "owner_state": "RUNNING", "owner_pid": fixture["pid"],
        "owner_start_ticks": fixture["ticks"], "heartbeat": str(tmp_path / "heartbeat"),
        "input_root": str(tmp_path / "input"), "output_root": str(fixture["science"]),
        "execution_commit": COMMIT, "task_spec_sha": SHA, "gpu": 1,
        "successor_task_id": None, "publisher_id": "t12-paper",
    }
    publisher = {
        "publisher_id": "t12-paper", "cell_id": "TasteMolNet/GCFExplainer",
        "owner_state": "PREDEPLOYED", "owner_pid": None, "owner_start_ticks": None,
        "heartbeat": None, "locator": str(tmp_path / "locator.json"),
        "lease_path": str(tmp_path / "publisher.lock"), "execution_commit": COMMIT,
        "claim_enabled": True, "active_writer_count": 0,
    }
    registry = build_owner_registry(
        registry_id="cas", matrix_authority_root=authority, tasks=[task],
        publishers=[publisher], gpu_leases=[{
            "gpu": 1, "task_id": task["task_id"], "state": "HELD",
            "lease_path": str(lease),
        }], proc_root=fixture["proc"], check_processes=False,
    )
    registry_path = tmp_path / "current.json"
    registry_path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
    terminal = {
        "status": "DIAGNOSTIC_510_RECONCILED_ENGINEERING_FAILURE",
        "task_id": task["task_id"], "diagnostic_only": True,
        "native_candidates_materialized": False, "promotion_allowed": False,
        "paper_cell_pass": False,
    }
    terminal["terminal_sha256"] = stable_sha256(terminal)
    updated = reconcile_registry_file_after_diagnostic(
        registry_path=registry_path,
        expected_registry_file_sha256=file_sha256(registry_path),
        expected_registry_sha256=registry["self_sha256"], task_id=task["task_id"],
        expected_owner_pid=fixture["pid"], expected_owner_start_ticks=fixture["ticks"],
        diagnostic_terminal=terminal, proc_root=fixture["proc"],
    )
    assert json.loads(registry_path.read_text()) == updated
    assert updated["tasks"][0]["owner_state"] == "TERMINAL_FAILED_ENGINEERING"


def test_registry_file_reconcile_rejects_stale_file_sha(tmp_path: Path) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    authority = tmp_path / "matrix"
    authority.mkdir()
    registry = build_owner_registry(
        registry_id="cas-stale", matrix_authority_root=authority, tasks=[],
        publishers=[], gpu_leases=[], check_processes=False,
    )
    path = tmp_path / "current.json"
    path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
    with pytest.raises(T12DiagnosticReconcileError, match="registry file changed"):
        reconcile_registry_file_after_diagnostic(
            registry_path=path, expected_registry_file_sha256="0" * 64,
            expected_registry_sha256=registry["self_sha256"], task_id="missing",
            expected_owner_pid=1, expected_owner_start_ticks=1,
            diagnostic_terminal={}, proc_root=fixture["proc"],
        )


@pytest.mark.parametrize("locator_exists", [False, True])
@pytest.mark.parametrize(
    "gnn_checkpoint_mode",
    ["regular_file", "sealed_directory", "tampered_sealed_directory"],
)
def test_fresh_zero_plan_is_strictly_blocked_and_preserves_unique_publisher(
    tmp_path: Path, locator_exists: bool, gnn_checkpoint_mode: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "configs/hpc.yaml"
    config.parent.mkdir()
    config.write_text("fixture: true\n")
    diagnostic = tmp_path / "diagnostic_terminal.json"
    diagnostic_value = {
        "status": "DIAGNOSTIC_510_RECONCILED_ENGINEERING_FAILURE",
        "diagnostic_only": True, "promotion_allowed": False, "paper_cell_pass": False,
    }
    diagnostic_value["terminal_sha256"] = stable_sha256(diagnostic_value)
    diagnostic.write_text(json.dumps(diagnostic_value))
    existing: dict[str, Path] = {}
    for name in ("neurosed", "t3", "official", "replay", "train", "cal", "test", "gnn", "molclr", "molclr.pt", "threshold-authority", "threshold-contract", "matrix"):
        path = tmp_path / name
        if "." in name or name in {"replay", "train", "cal", "test", "gnn", "threshold-authority", "threshold-contract"}:
            path.write_text("fixture")
        else:
            path.mkdir()
        existing[name] = path
    if gnn_checkpoint_mode != "regular_file":
        existing["gnn"].unlink()
        existing["gnn"].mkdir()
        for name in sorted(CANDIDATE_CHECKPOINT_FILES - {"sha256sums.txt"}):
            (existing["gnn"] / name).write_bytes(f"sealed:{name}\n".encode())
        (existing["gnn"] / "sha256sums.txt").write_text(
            "".join(
                f"{file_sha256(existing['gnn'] / name)}  {name}\n"
                for name in sorted(CANDIDATE_CHECKPOINT_FILES - {"sha256sums.txt"})
            ),
            encoding="utf-8",
        )
        if gnn_checkpoint_mode == "tampered_sealed_directory":
            (existing["gnn"] / "model.pt").write_bytes(b"tampered-after-seal\n")
    registry = build_owner_registry(
        registry_id="fresh-plan", matrix_authority_root=existing["matrix"], tasks=[],
        publishers=[{
            "publisher_id": "existing-t12-paper", "cell_id": "TasteMolNet/GCFExplainer",
            "owner_state": "PREDEPLOYED", "owner_pid": None, "owner_start_ticks": None,
            "heartbeat": None, "locator": str(tmp_path / "locator.json"),
            "lease_path": str(tmp_path / "publisher.lock"), "execution_commit": COMMIT,
            "claim_enabled": True, "active_writer_count": 0,
        }], gpu_leases=[], check_processes=False,
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    locator = tmp_path / "locator.json"
    if locator_exists:
        locator.write_text('{"publisher":"fixture"}\n', encoding="utf-8")
    bridge_history = tmp_path / "diagnostic-bridge-history"
    bridge_history.mkdir()
    (bridge_history / "history.bin").write_bytes(b"sealed-history")
    kwargs = dict(
        repo_root=repo, python=Path(sys.executable).resolve(), config=config,
        execution_commit=COMMIT, attempt_id=str(uuid.uuid4()), generation_token=SHA,
        gpu_index=1, gpu_uuid="GPU-fixture", diagnostic_terminal=diagnostic,
        required_parity_receipt=tmp_path / "future-parity.json",
        managed_neurosed_root=existing["neurosed"], t3_root=existing["t3"], official_root=existing["official"],
        threshold_authority=existing["threshold-authority"], replay_gate=existing["replay"],
        production_root=tmp_path / "production", postprocess_root=tmp_path / "postprocess",
        train_csv=existing["train"], calibration_csv=existing["cal"], test_csv=existing["test"],
        gnn_checkpoint=existing["gnn"], molclr_root=existing["molclr"], molclr_checkpoint=existing["molclr.pt"],
        threshold_contract=existing["threshold-contract"], wnode_cache_db=tmp_path / "cache.sqlite",
        node_embedding_cache_dir=tmp_path / "embeddings", verification_root=tmp_path / "verify",
        publisher_id="existing-t12-paper", publisher_locator=tmp_path / "locator.json",
        owner_registry=registry_path,
        expected_owner_registry_sha256=registry["self_sha256"],
        expected_owner_registry_file_sha256=file_sha256(registry_path),
        matrix_authority_root=existing["matrix"],
        diagnostic_bridge_history_root=bridge_history,
        nvme_disposable_index_root=Path("/root/autodl-tmp/t12-fixture-index"),
    )
    if gnn_checkpoint_mode == "tampered_sealed_directory":
        with pytest.raises(T12FreshZeroPlanError, match="differs from sha256sums.txt"):
            build_fresh_zero_plan(**kwargs)
        return
    plan = build_fresh_zero_plan(**kwargs)
    assert plan["dispatchable"] is False
    assert plan["fresh_from_zero"] is True and plan["source_checkpoint"] is None
    assert plan["production_steps"] == list(FORMAL_PRODUCTION_CHECKPOINT_CURSORS)
    assert len(plan["stages"]) == len(FORMAL_PRODUCTION_CHECKPOINT_CURSORS) + 3
    assert all(stage["dispatchable"] is False for stage in plan["stages"])
    assert plan["publisher_handoff"]["new_publisher_created"] is False
    assert plan["publisher_handoff"]["canonical_locator"] == str(locator)
    assert plan["publisher_handoff"]["canonical_locator_present"] is locator_exists
    assert plan["publisher_handoff"]["canonical_locator_creation_allowed"] is False
    assert locator.exists() is locator_exists
    assert plan["input_hash_bindings"]["publisher_locator_present"] is locator_exists
    assert plan["input_hash_bindings"]["publisher_locator_path"] == str(locator)
    if locator_exists:
        assert plan["input_hash_bindings"]["publisher_locator"] == file_sha256(locator)
        assert plan["input_hash_bindings"]["publisher_locator_binding_state"] == "PRESENT_SHA256_BOUND"
    else:
        assert plan["input_hash_bindings"]["publisher_locator"] is None
        assert plan["input_hash_bindings"]["publisher_locator_binding_state"] == "ABSENT_EXACT_PATH_BOUND"
    assert "fresh" in plan["stages"][0]["command"]
    assert "resume" in plan["stages"][1]["command"]
    assert all(
        "--formal-checkpoint-cadence" in stage["command"]
        for stage in plan["stages"]
    )
    assert plan["nvme_staging"]["production_reuses_diagnostic_history"] is False
    assert plan["input_hash_bindings"]["owner_registry_file"] == file_sha256(
        registry_path
    )
    if gnn_checkpoint_mode == "sealed_directory":
        _inventory, expected_gnn_sha256 = _inventory_directory(
            existing["gnn"], label="test sealed T3 checkpoint"
        )
    else:
        expected_gnn_sha256 = file_sha256(existing["gnn"])
    assert plan["input_hash_bindings"]["gnn_checkpoint"] == expected_gnn_sha256
