"""T12's one committed-500 recovery tail under the existing owner.

This is a child adapter, not a launcher/controller. It never claims an owner,
creates a GPU lock, changes a matrix, or runs the first 500 transitions.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from src.utils.main_ready_task_specs import atomic_json, load_spec
from src.utils.final16_owner_registry_v1 import process_start_ticks


def _read(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("T12_RECOVERY_PHYSICAL_ABSOLUTE_RECEIPT_REQUIRED")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("T12_RECOVERY_RECEIPT_NOT_OBJECT")
    return value


def _sha(path: str | Path) -> str:
    # Only small manifests here; adopted payload/segment checks stay in the
    # existing checkpoint/history implementation, not another full hash scan.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_inherited_owner(spec: dict, binding: dict, *, now=None) -> dict:
    """Prove real FD inheritance/exclusion and freshly measured stage resources."""
    fd = int(os.environ.get("T12_OWNER_HELD_GPU_FD", "-1"))
    if fd < 0:
        raise ValueError("T12_RECOVERY_OWNER_FD_NOT_INHERITED")
    gpu = spec["gpu_request"]
    lease = Path(gpu["lease_path"])
    if not lease.is_absolute() or lease.is_symlink():
        raise ValueError("T12_RECOVERY_INVALID_LEASE_PATH")
    held, named = os.fstat(fd), lease.lstat()
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError("T12_RECOVERY_LEASE_INODE_CHANGED")
    # A separate open in a separate process must fail. fstat alone says nothing
    # about flock; do not acquire a new lock and pretend it was inherited.
    probe = subprocess.run([
        sys.executable, "-I", "-c",
        "import fcntl,sys\nf=open(sys.argv[1],'rb')\ntry:\n"
        " fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n sys.exit(73)\n", str(lease)],
        close_fds=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=20, check=False)
    if probe.returncode != 73:
        raise ValueError("T12_RECOVERY_INHERITED_LOCK_NOT_EXCLUSIVE")
    owner = binding["owner_identity"]
    if (os.getppid() != owner["pid"] or
            process_start_ticks("/proc", owner["pid"]) != owner["start_ticks"]):
        raise ValueError("T12_RECOVERY_PARENT_IDENTITY_CHANGED")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu["uuid"]:
        raise ValueError("T12_RECOVERY_VISIBLE_GPU_MUST_BE_FULL_UUID")
    provider = binding["resource_provider_command"]
    if (not isinstance(provider, list) or not provider or
            any(not isinstance(x, str) for x in provider) or
            not Path(provider[0]).is_absolute()):
        raise ValueError("T12_RECOVERY_PROVIDER_COMMAND_UNBOUND")
    sampled = subprocess.run(provider, check=True, close_fds=True, timeout=30,
                             capture_output=True, text=True)
    measured = json.loads(sampled.stdout)
    age = (time.time() if now is None else now) - measured.get("measured_at_unix_seconds", 0)
    if (measured.get("actual_resources_resampled") is not True or
            measured.get("allowed") is not True or not 0 <= age <= 120 or
            measured.get("stage_id") != "T12_RECOVERY_501_510" or
            measured.get("gpu_uuid") != gpu["uuid"] or
            measured.get("owner_identity") != owner or
            measured.get("task_id") != spec["task_id"]):
        raise ValueError("T12_RECOVERY_FRESH_RESOURCES_NOT_ADMITTED")
    return measured


def require_recovery_binding(spec: dict, binding: dict) -> dict:
    if (binding.get("resume_cursor") != 500 or binding.get("end_cursor") != 510 or
            binding.get("maximum_new_transitions") != 10 or
            binding.get("diagnostic_checkpoint_promotion_allowed") is not False):
        raise ValueError("T12_RECOVERY_ONLY_COMMITTED500_TO510")
    source = Path(binding["source_root"])
    output = Path(spec["output_root"])
    if source == output or source.resolve() == output.resolve():
        raise ValueError("T12_RECOVERY_ORIGINAL_ROOT_IS_READ_ONLY")
    source_manifest = source / "checkpoints/checkpoint-00000500.manifest.json"
    target_manifest = output / "checkpoints/checkpoint-00000500.manifest.json"
    source_record, target_record = _read(source_manifest), _read(target_manifest)
    if (source_record.get("checkpoint_cursor") != 500 or
            target_record.get("checkpoint_cursor") != 500 or
            source_record.get("payload_sha256") != binding["source_payload_sha256"] or
            source_record.get("rng_sha256") != target_record.get("rng_sha256") or
            source_record.get("identity_sha256") != target_record.get("identity_sha256")):
        raise ValueError("T12_RECOVERY_CHECKPOINT500_BINDING_CHANGED")
    fork_path = Path(binding["relocation_receipt"])
    if _sha(fork_path) != binding["relocation_receipt_sha256"]:
        raise ValueError("T12_RECOVERY_RELOCATION_RECEIPT_CHANGED")
    fork = _read(fork_path)
    if (fork.get("status") != "PASS" or fork.get("source_root") != str(source) or
            fork.get("target_root") != str(output) or
            fork.get("source_checkpoint_manifest_sha256") != _sha(source_manifest) or
            fork.get("target_checkpoint_manifest_sha256") != _sha(target_manifest) or
            fork.get("scientific_state_mutated") is not False or
            fork.get("storage_roots_relocated") is not True or
            fork.get("first_seen_embedding_record_bytes_copied_exactly") is not True or
            fork.get("rng_sha256") != source_record.get("rng_sha256")):
        raise ValueError("T12_RECOVERY_RELOCATION_NOT_CLOSED")
    # The admission producer must cite actual two-round storage evidence. A
    # df snapshot or successful read is not a storage-write acceptance.
    admission_path = Path(binding["storage_admission_receipt"])
    if _sha(admission_path) != binding["storage_admission_receipt_sha256"]:
        raise ValueError("T12_RECOVERY_STORAGE_RECEIPT_CHANGED")
    storage = _read(admission_path)
    if (storage.get("status") != "STORAGE_RECOVERY_ADMITTED" or
            storage.get("file_fsync_pass") is not True or
            storage.get("rename_reopen_pass") is not True or
            storage.get("consistent_snapshot_reopen_pass") is not True or
            storage.get("two_rounds_pass") is not True or
            storage.get("writer_set_empty") is not True or
            storage.get("output_root") != str(output) or
            storage.get("disposable_index_root") != spec["science_contract"]["disposable_index_root"] or
            storage.get("local_safety_reserve_bytes", 0) < 2 * 1024**3 or
            storage.get("joint_peak_capacity_admitted") is not True):
        raise ValueError("T12_RECOVERY_STORAGE_NOT_ADMITTED")
    if (output / "segment-00501-00510").exists():
        raise ValueError("T12_RECOVERY_TAIL_ALREADY_ATTEMPTED")
    return {"checkpoint_manifest": str(target_manifest), "fork": fork, "storage": storage}


def run_recovery_segment(task_spec: Path) -> dict[str, Any]:
    """Execute only a pre-relocated diagnostic tail; existing owner retains FD."""
    spec = load_spec(task_spec)
    contract = spec["science_contract"]
    binding = contract["eio_recovery_binding"]
    admitted = require_recovery_binding(spec, binding)
    resources = require_inherited_owner(spec, binding)
    from scripts.autodl.run_t12_accelerated_from250_v1 import (
        _configure_profile, _validate_source_equivalence,
    )
    # Existing four-file review remains enforced. A fresh driver/storage delta
    # needs its explicit reviewed receipt; this adapter cannot waive it.
    _validate_source_equivalence(spec)
    _configure_profile()
    from src.baselines.tastemolnet_gcf_full import run_t12_generation_segment
    root = Path(spec["output_root"]) / "eio-recovery-tail"
    root.mkdir(mode=0o700, exist_ok=False)
    atomic_json(root / "attempt.json", {
        "state": "RECOVERY_STARTED_NOT_VERIFIED", "task_id": spec["task_id"],
        "science_pid": os.getpid(), "original_attempt_id": contract["source_science_attempt_id"],
        "resume_cursor": 500, "end_cursor": 510, "maximum_new_transitions": 10,
        "new_fresh_run": False, "resources": resources,
    })
    try:
        result = run_t12_generation_segment(
            mode="resume", output_root=spec["output_root"],
            checkpoint_manifest=admitted["checkpoint_manifest"],
            attempt_id=contract["source_science_attempt_id"],
            generation_token=contract["generation_token"], gpu_uuid=spec["gpu_request"]["uuid"],
            managed_neurosed_root=contract["managed_neurosed_root"], t3_root=contract["t3_root"],
            official_root=contract["official_root"], threshold_authority_path=contract["threshold_authority"],
            replay_gate_path=contract["replay_gate"],
            resume_run_identity_authority=Path(spec["output_root"]) / "run_identity.json",
            disposable_index_root=contract["disposable_index_root"],
            scientific_source_equivalence_receipt_path=contract["scientific_source_equivalence_receipt"],
            materialize_terminal_candidates=False, diagnostic_only=True,
            durable_recovery_index=True,
        )
        final_manifest = Path(spec["output_root"]) / "checkpoints/checkpoint-00000510.manifest.json"
        final = _read(final_manifest)
        if (result.get("status") != "GENERATION_CHECKPOINT_COMMITTED" or
                result.get("checkpoint_cursor") != 510 or
                result.get("diagnostic_only") is not True or
                result.get("candidate_manifest") is not None or
                result.get("terminal_candidate_materialization_requested") is not False or
                result.get("checkpoint_manifest") != str(final_manifest) or
                result.get("checkpoint_manifest_sha256") != _sha(final_manifest) or
                final.get("status") != "COMMITTED" or final.get("checkpoint_cursor") != 510):
            raise ValueError("T12_RECOVERY_REAL510_CHECKPOINT_NOT_COMMITTED")
        receipt = {"state": "RECOVERY_TAIL_COMMITTED_NOT_FULL_PARITY", "result": result,
                   "newly_committed_unique_steps": 10, "replayed_steps_0_to_500": 0,
                   "diagnostic_checkpoint_promotion_allowed": False,
                   "matrix_written": False, "automatic_retry": False}
        atomic_json(root / "terminal.json", receipt)
        return receipt
    except BaseException as exc:
        atomic_json(root / "terminal.json", {"state": "FAILED", "error_type": type(exc).__name__,
            "error": str(exc), "automatic_retry": False, "matrix_written": False})
        raise
