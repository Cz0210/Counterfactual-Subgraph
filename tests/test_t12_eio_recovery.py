from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from src.baselines import tastemolnet_gcf_production_state as production
from src.utils import t12_eio_recovery as recovery


def _journal(tmp_path, *, snapshot=None, durable=False, index="idx", writer=True):
    return production.T12CompactHistoryJournal(
        root=tmp_path / "history", index_root=tmp_path / index,
        bounds=production.T12ProductionBounds.pinned(parent_count=2),
        contract_sha256="a" * 64, generation_token="b" * 64,
        attempt_id="c6fdce9f-7e01-4481-b97e-372ae851a0f8",
        resume_snapshot=snapshot, open_writer=writer, durable_recovery_index=durable)


def test_delete_full_rebuild_preserves_authenticated_values_and_old_index(tmp_path):
    old = _journal(tmp_path)
    row = old.append_observation(graph_identity_sha256="c" * 64,
        probabilities=(0.1, 0.2, 0.7), prediction=2, candidate=True, valid_fullgraph=True,
        coverage_vector=(1, 0), embedding_sha256="d" * 64, failure_reason="",
        lineage_sha256="e" * 64, neurosed_query_sha256="f" * 64)
    snapshot = old.checkpoint_state()
    old.close()
    original = old._index_path.read_bytes()
    history_file = tmp_path / "history" / snapshot["segments"][0]["segment_file"]
    original_history = history_file.read_bytes()
    fresh = _journal(tmp_path, snapshot=snapshot, durable=True, index="new-index", writer=False)
    assert fresh._connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    assert fresh._connection.execute("PRAGMA synchronous").fetchone() == (2,)
    assert fresh.lookup_first("c" * 64) == row
    assert fresh.checkpoint_state() == snapshot
    assert fresh._connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    fresh.close()
    assert old._index_path.read_bytes() == original
    assert history_file.read_bytes() == original_history


def test_durable_mode_still_rejects_corrupt_scientific_segment(tmp_path):
    old = _journal(tmp_path)
    old.append_observation(graph_identity_sha256="c" * 64,
        probabilities=(0.1, 0.2, 0.7), prediction=2, candidate=True, valid_fullgraph=True,
        coverage_vector=(1, 0), embedding_sha256="d" * 64, failure_reason="",
        lineage_sha256="e" * 64, neurosed_query_sha256="f" * 64)
    snapshot = old.checkpoint_state()
    old.close()
    p = tmp_path / "history" / snapshot["segments"][0]["segment_file"]
    raw = bytearray(p.read_bytes()); raw[-1] ^= 1; p.write_bytes(raw)
    with pytest.raises(production.TasteT12ProductionStateError, match="hash chain"):
        _journal(tmp_path, snapshot=snapshot, durable=True, index="new-index", writer=False)


def _write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value)); return hashlib.sha256(p.read_bytes()).hexdigest()


def _gate_fixture(tmp_path):
    source, output = tmp_path / "old", tmp_path / "fresh"
    base = {"checkpoint_cursor": 500, "payload_sha256": "a" * 64,
            "rng_sha256": "b" * 64, "identity_sha256": "c" * 64}
    source_manifest = source / "checkpoints/checkpoint-00000500.manifest.json"
    target_manifest = output / "checkpoints/checkpoint-00000500.manifest.json"
    a, b = _write(source_manifest, base), _write(target_manifest, base)
    fork_path = output / "step500_fork_receipt.json"
    fork_sha = _write(fork_path, {"status": "PASS", "source_root": str(source),
        "target_root": str(output), "source_checkpoint_manifest_sha256": a,
        "target_checkpoint_manifest_sha256": b, "rng_sha256": base["rng_sha256"],
        "scientific_state_mutated": False, "storage_roots_relocated": True,
        "first_seen_embedding_record_bytes_copied_exactly": True})
    storage_path = output / "storage-admission.json"
    storage = {"status": "STORAGE_RECOVERY_ADMITTED", "file_fsync_pass": True,
        "rename_reopen_pass": True, "consistent_snapshot_reopen_pass": True,
        "two_rounds_pass": True, "writer_set_empty": True, "output_root": str(output),
        "disposable_index_root": str(tmp_path / "local-index"),
        "local_safety_reserve_bytes": 2 * 1024**3, "joint_peak_capacity_admitted": True}
    storage_sha = _write(storage_path, storage)
    spec = {"output_root": str(output), "science_contract": {
        "disposable_index_root": str(tmp_path / "local-index")}}
    binding = {"resume_cursor": 500, "end_cursor": 510, "maximum_new_transitions": 10,
        "diagnostic_checkpoint_promotion_allowed": False, "source_root": str(source),
        "source_payload_sha256": "a" * 64, "relocation_receipt": str(fork_path),
        "relocation_receipt_sha256": fork_sha, "storage_admission_receipt": str(storage_path),
        "storage_admission_receipt_sha256": storage_sha}
    return spec, binding, storage


def test_admitted500_gate_does_not_require_prior510(tmp_path):
    spec, binding, _ = _gate_fixture(tmp_path)
    assert recovery.require_recovery_binding(spec, binding)["checkpoint_manifest"].endswith("00000500.manifest.json")


@pytest.mark.parametrize("field", ["file_fsync_pass", "two_rounds_pass", "writer_set_empty",
                                 "consistent_snapshot_reopen_pass", "joint_peak_capacity_admitted"])
def test_each_missing_storage_property_blocks(tmp_path, field):
    spec, binding, storage = _gate_fixture(tmp_path)
    storage[field] = False
    binding["storage_admission_receipt_sha256"] = _write(Path(binding["storage_admission_receipt"]), storage)
    with pytest.raises(ValueError, match="STORAGE_NOT_ADMITTED"):
        recovery.require_recovery_binding(spec, binding)


def test_no_original_root_or_fresh0_or_repeated_tail(tmp_path):
    spec, binding, _ = _gate_fixture(tmp_path)
    binding["resume_cursor"] = 0
    with pytest.raises(ValueError, match="COMMITTED500_TO510"):
        recovery.require_recovery_binding(spec, binding)
    binding["resume_cursor"] = 500
    (Path(spec["output_root"]) / "segment-00501-00510").mkdir()
    with pytest.raises(ValueError, match="ALREADY_ATTEMPTED"):
        recovery.require_recovery_binding(spec, binding)
    spec["output_root"] = binding["source_root"]
    with pytest.raises(ValueError, match="ORIGINAL_ROOT_IS_READ_ONLY"):
        recovery.require_recovery_binding(spec, binding)


def test_fd_must_be_really_inherited_and_lock_exclusive(tmp_path):
    lease = tmp_path / "gpu.lock"
    root = str(Path(__file__).resolve().parents[1])
    with lease.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd = stream.fileno()
        # macOS lacks /proc. Stub only Linux start-tick reading in this CPU
        # test; real FD transfer, parent PID, UUID and exclusion are OS tests.
        child = (
            "import os,sys,json,time;sys.path.insert(0,sys.argv[1]);"
            "from src.utils import t12_eio_recovery as r;"
            "r.process_start_ticks=lambda root,pid:77;"
            "owner={'pid':os.getppid(),'start_ticks':77};"
            "gpu={'uuid':'GPU-test-full-uuid','lease_path':sys.argv[2]};"
            "provider=[sys.executable,'-I','-c',"
            "'import json,time;print(json.dumps('+repr({'actual_resources_resampled':True,"
            "'allowed':True,'stage_id':'T12_RECOVERY_501_510','gpu_uuid':gpu['uuid'],"
            "'owner_identity':owner,'task_id':'test','measured_at_unix_seconds':time.time()})+'))'];"
            "r.require_inherited_owner({'gpu_request':gpu,'task_id':'test'},"
            "{'owner_identity':owner,'resource_provider_command':provider});print('EXCLUSIVE')")
        env = dict(os.environ, T12_OWNER_HELD_GPU_FD=str(fd), CUDA_VISIBLE_DEVICES="GPU-test-full-uuid")
        yes = subprocess.run([sys.executable, "-I", "-c", child, root, str(lease)],
            pass_fds=(fd,), env=env, capture_output=True, text=True)
        assert yes.returncode == 0, yes.stderr
        assert "EXCLUSIVE" in yes.stdout
        no = subprocess.run([sys.executable, "-I", "-c", child, root, str(lease)],
            env=env, capture_output=True, text=True)
        assert no.returncode != 0
        fcntl.flock(fd, fcntl.LOCK_UN)
        unlocked = subprocess.run([sys.executable, "-I", "-c", child, root, str(lease)],
            pass_fds=(fd,), env=env, capture_output=True, text=True)
        assert unlocked.returncode != 0
        assert "INHERITED_LOCK_NOT_EXCLUSIVE" in unlocked.stderr


def test_generation_adapter_limits_durable_recovery_to_resume_with_local_index():
    from src.baselines.tastemolnet_gcf_full import run_t12_generation_segment
    from src.baselines.tastemolnet_gcf_full_resume import TasteGCFFullResumeError
    with pytest.raises(TasteGCFFullResumeError, match="durable recovery index"):
        run_t12_generation_segment(mode="fresh", output_root="/unused", checkpoint_manifest=None,
            attempt_id=str(uuid.uuid4()), generation_token="b" * 64, gpu_uuid="GPU-test",
            managed_neurosed_root="/unused", t3_root="/unused", official_root="/unused",
            threshold_authority_path="/unused", replay_gate_path="/unused",
            durable_recovery_index=True)


@pytest.mark.parametrize("age,visible,expected", [
    (121, "GPU-test-full-uuid", "FRESH_RESOURCES_NOT_ADMITTED"),
    (0, "3", "VISIBLE_GPU_MUST_BE_FULL_UUID"),
])
def test_stale_measurement_and_ordinal_mapping_rejected(tmp_path, monkeypatch, age, visible, expected):
    lease = tmp_path / "gpu.lock"
    owner = {"pid": os.getppid(), "start_ticks": 77}
    measured = {"actual_resources_resampled": True, "allowed": True,
        "stage_id": "T12_RECOVERY_501_510", "gpu_uuid": "GPU-test-full-uuid",
        "owner_identity": owner, "task_id": "test", "measured_at_unix_seconds": time.time() - age}
    with lease.open("a+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("T12_OWNER_HELD_GPU_FD", str(held.fileno()))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        monkeypatch.setattr(recovery, "process_start_ticks", lambda root, pid: 77)
        spec = {"task_id": "test", "gpu_request": {"uuid": "GPU-test-full-uuid", "lease_path": str(lease)}}
        provider = [sys.executable, "-I", "-c", "print(" + repr(json.dumps(measured)) + ")"]
        with pytest.raises(ValueError, match=expected):
            recovery.require_inherited_owner(spec, {"owner_identity": owner,
                                                   "resource_provider_command": provider})
