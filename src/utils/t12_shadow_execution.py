"""Finite T12 diagnostic segments using the existing official production kernel.

No owner is created here. The canonical owner must admit the task, pass its
exclusive descriptor and bind the source-equivalence/regression receipts.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.utils.main_ready_task_specs import atomic_json, load_spec, stable_sha256
from src.utils.t12_shadow_recovery import JointLedger, require_natural_510, validate_plan
from src.utils.t12_raw_evidence import BoundSelectedStepObserver, RawEvidenceResolver


def run_live_tail(*, plan: dict, arm: str, observer: Any, resolver: RawEvidenceResolver,
                  ledger_root: Path, live: dict, walk=None) -> dict:
    """Seal 500, then use the SAME walker/bridge/RNG objects for 501..510."""
    validate_plan(plan)
    stages = [s for s in plan["stages"] if s["stage_id"] == f"{arm}_continuous_501_510"]
    if len(stages) != 1 or stages[0]["kind"] != "SAME_PROCESS_TAIL":
        raise ValueError("T12_UNBUDGETED_CONTINUOUS_TAIL")
    if live["checkpoint_cursor"] != 500 or len(live["vrrw"].traversed_hashes) != 500:
        raise ValueError("T12_LIVE_TAIL_REQUIRES_ACTUAL_CURSOR500")
    head = observer.ledger.seal(Path(live["checkpoint_manifest"]))
    resolver.save(ledger_root / "raw-evidence-at-500.jsonl.gz")
    observer.ledger = JointLedger(ledger_root, start=501, end=510,
                                 binding_sha=plan["plan_sha256"])
    if walk is None:
        from src.baselines.tastemolnet_gcf_smoke import _run_official_walk_segment
        walk = _run_official_walk_segment
    from src.baselines.tastemolnet_gcf_full import PRODUCTION_TELEPORT
    # Official summary writes relative native-result files. A fresh directory
    # prevents the tail from overwriting the already sealed500 result.
    runtime = ledger_root / "continuous-native-runtime"
    runtime.mkdir(exist_ok=False)
    old_cwd = Path.cwd()
    try:
        os.chdir(runtime)
        result = walk(vrrw=live["vrrw"], input_graphs=live["input_graphs"],
            importance_args=live["importance_args"], teleport_probability=PRODUCTION_TELEPORT,
            start_step=501, end_step=510, resume_graph_hash=live["current_graph_identity"])
    finally:
        os.chdir(old_cwd)
    live["sources"].revalidate()
    manifest = live["orchestrator"].commit(completed_steps=510, vrrw=live["vrrw"],
        bridge=live["bridge"], adapter=live["adapter"], action_counts=live["action_counts"],
        current_graph_identity=result.current_graph_hash, np=live["np"], torch=live["torch"])
    tail = observer.ledger.seal(manifest)
    resolver.save(ledger_root / "raw-evidence-at-510.jsonl.gz")
    receipt = {"status": "DIAGNOSTIC_TAIL_COMMITTED_NOT_PARITY", "head": head, "tail": tail,
               "new_transitions": 10, "same_process_live_state": True,
               "checkpoint_promotion_allowed": False, "test_loaded": False}
    atomic_json(ledger_root / "continuous-tail.json", receipt)
    return receipt


def run_shadow_segment(*, plan: dict, task_spec: Path, stage_id: str,
                       process_alive, raw_input: Path | None = None) -> dict:
    """Execute one already bound 251..500(+tail) or independent 501..510 segment."""
    import fcntl
    require_natural_510(plan, process_alive=process_alive)
    stages = [s for s in plan["stages"] if s["stage_id"] == stage_id]
    if len(stages) != 1 or stages[0]["kind"] not in {"SHADOW", "INDEPENDENT_RELOAD"}:
        raise ValueError("T12_STAGE_NOT_AN_INDEPENDENT_EXECUTION")
    stage = stages[0]
    spec = load_spec(task_spec)
    binding = spec["science_contract"].get("shadow_binding", {})
    if binding.get("plan_sha256") != plan["plan_sha256"] or binding.get("stage_id") != stage_id:
        raise ValueError("T12_SHADOW_SPEC_BINDING_MISSING")
    # No implicit GPU claim. This is the actual descriptor inherited from the
    # existing owner; checking a JSON integer alone is insufficient.
    fd = int(os.environ.get("T12_OWNER_HELD_GPU_FD", "-1"))
    if fd < 0:
        raise ValueError("T12_CANONICAL_OWNER_FD_NOT_PASSED")
    gpu = spec["gpu_request"]
    lease = Path(gpu["lease_path"])
    opened, named = os.fstat(fd), lease.lstat()
    if lease.is_symlink() or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError("T12_SHADOW_LEASE_IDENTITY_CHANGED")
    probe = subprocess.run([sys.executable, "-I", "-c",
        "import fcntl,sys\nf=open(sys.argv[1], 'rb')\ntry:\n"
        " fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n sys.exit(73)\n", str(lease)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if probe.returncode != 73:
        raise ValueError("T12_SHADOW_LEASE_NOT_EXCLUSIVE")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu["uuid"]:
        raise ValueError("T12_SHADOW_GPU_UUID_CHANGED")
    owner = binding["owner_identity"]
    if os.getppid() != owner["pid"] or not process_alive(owner["pid"], owner["start_ticks"]):
        raise ValueError("T12_SHADOW_CANONICAL_PARENT_CHANGED")
    resource = subprocess.run(binding["resource_provider_command"], check=True,
                              capture_output=True, text=True)
    measured = json.loads(resource.stdout)
    if (measured.get("actual_resources_resampled") is not True
            or measured.get("stage_id") != stage_id or measured.get("gpu_uuid") != gpu["uuid"]
            or not 0 <= time.time() - measured.get("measured_at_unix_seconds", 0) <= 120
            or measured.get("allowed") is not True):
        raise ValueError("T12_SHADOW_RESOURCE_ADMISSION_NOT_PASS")
    from scripts.autodl.run_t12_accelerated_from250_v1 import _configure_profile, _validate_source_equivalence
    _validate_source_equivalence(spec)
    regression = json.loads(Path(binding["observer_regression_receipt"]).read_text())
    if (regression.get("status") != "PASS" or regression.get("observer_changes_science") is not False
            or regression.get("raw_binding_tested") is not True):
        raise ValueError("T12_OBSERVER_REAL_REGRESSION_MISSING")
    _configure_profile()
    import numpy as np
    import torch
    from src.baselines.tastemolnet_gcf_full import run_t12_generation_segment
    output = Path(spec["output_root"])
    if not output.exists():
        from src.utils.tastemolnet_t12_accelerated_from250 import fork_committed_diagnostic_prefix
        from src.baselines.tastemolnet_gcf_full_resume import production_checkpoint_identity
        source_root = Path(binding["fork_source_root"])
        source_checkpoint = Path(binding["fork_source_checkpoint"])
        source_run = json.loads((source_root / "run_identity.json").read_text())
        expected = production_checkpoint_identity(source_run["identity_template"],
                                                  checkpoint_cursor=stage["restore_cursor"])
        fork_committed_diagnostic_prefix(source_root=source_root, target_root=output,
            source_checkpoint_manifest=source_checkpoint, expected_identity=expected,
            torch=torch, checkpoint_cursor=stage["restore_cursor"])
    checkpoint = output / "checkpoints" / f"checkpoint-{stage['restore_cursor']:08d}.manifest.json"
    if json.loads(checkpoint.read_text()).get("checkpoint_cursor") != stage["restore_cursor"]:
        raise ValueError("T12_SHADOW_RESTORE_CURSOR_CHANGED")
    ledger_root = output / "shadow-ledger" / stage_id
    ledger_root.mkdir(parents=True, exist_ok=False)
    claim = {"plan_sha256": plan["plan_sha256"], "stage_id": stage_id,
             "pid": os.getpid(), "budgeted_transitions": stage["transitions"],
             "status": "STARTED_NOT_COMPLETE", "fresh_generation": False}
    atomic_json(ledger_root / "attempt.json", claim)
    resolver = RawEvidenceResolver(binding["raw_contract_sha256"])
    if raw_input is not None:
        resolver.load(raw_input)
    ledger = JointLedger(ledger_root, start=stage["start"], end=stage["end"],
                         binding_sha=plan["plan_sha256"])
    observer = BoundSelectedStepObserver(ledger, np=np, torch=torch, resolver=resolver)
    contract = spec["science_contract"]
    tail_enabled = any(s["stage_id"] == f"{stage['arm']}_continuous_501_510" for s in plan["stages"])
    callback = None
    if stage["kind"] == "SHADOW" and tail_enabled:
        callback = lambda **live: run_live_tail(plan=plan, arm=stage["arm"],
            observer=observer, resolver=resolver, ledger_root=ledger_root, live=live)
    try:
        with observer.installed():
            result = run_t12_generation_segment(mode="resume", output_root=output,
                checkpoint_manifest=checkpoint, attempt_id=contract["source_science_attempt_id"],
                generation_token=contract["generation_token"], gpu_uuid=gpu["uuid"],
                managed_neurosed_root=contract["managed_neurosed_root"], t3_root=contract["t3_root"],
                official_root=contract["official_root"], threshold_authority_path=contract["threshold_authority"],
                replay_gate_path=contract["replay_gate"], resume_run_identity_authority=output / "run_identity.json",
                disposable_index_root=contract["disposable_index_root"],
                scientific_source_equivalence_receipt_path=contract["scientific_source_equivalence_receipt"],
                materialize_terminal_candidates=False, diagnostic_only=True,
                diagnostic_after_checkpoint=callback)
        if callback is None:
            ledger.seal(Path(result["checkpoint_manifest"]))
            resolver.save(ledger_root / "raw-evidence.jsonl.gz")
        atomic_json(ledger_root / "terminal.json", {"status": "DIAGNOSTIC_COMMITTED_NOT_PARITY",
            "stage_id": stage_id, "result": result, "matrix_written": False})
        return result
    except BaseException as exc:
        if not observer.ledger.raw.closed:
            observer.ledger.close_failed()
        atomic_json(ledger_root / "terminal.json", {"status": "FAILED", "stage_id": stage_id,
            "error_type": type(exc).__name__, "error": str(exc), "matrix_written": False})
        raise
