from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.autodl import run_t14_low_memory_resume_owner as owner
from src.baselines.comrecgc import generation_checkpoint as checkpoints
from src.baselines.comrecgc.generation_checkpoint import scientific_command_sha256
from src.baselines.comrecgc.transition_cache import CompactMoveScopedTransitionMap
from src.baselines.tastemolnet_t14_resume import (
    CANARY_RECEIPT_SCHEMA,
    T14ResumeError,
    assert_auditor_serialized,
    bind_resume_identity,
    build_resume_spec,
    evaluate_memory_admission,
    write_resume_spec,
)
from src.baselines.tastemolnet_comrecgc_full import TasteComRecGCFullBridge


GIB = 1024**3
SOURCE_COMMIT = "1" * 40
RESUME_COMMIT = "2" * 40


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _identity(commit: str) -> dict[str, object]:
    argv = (
        "tastemolnet_t14_comrecgc_full_v1",
        "train_sha256=" + "a" * 64,
        f"execution_commit={commit}",
    )
    command = scientific_command_sha256(argv)
    provenance = {
        "dataset": "tastemolnet",
        "method": "comrecgc",
        "stage": "T14_COMRECGC_FULL",
        "train_csv_sha256": "a" * 64,
        "execution_commit": commit,
        "scientific_command_sha256": command,
        "total_steps": "25000",
    }
    return {
        "schema_version": "tastemolnet_t14_checkpoint_provenance_v1",
        "status": "FROZEN",
        "provenance": provenance,
        "scientific_argv": list(argv),
        "command_sha256": command,
        "total_steps": 25_000,
        "checkpoint_interval": 2_500,
        "transition_expanded_capacity": 5,
        "raw_neighbor_graphs_retained_unbounded": False,
    }


def _checkpoint_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path.resolve() / "t14"
    root.mkdir()
    identity = _identity(SOURCE_COMMIT)
    database = sqlite3.connect(tmp_path / "live.sqlite3")
    database.execute("CREATE TABLE graphs (id INTEGER PRIMARY KEY)")
    database.commit()
    checkpoints.save_generation_checkpoint(
        root / "checkpoints",
        completed_step=12_500,
        step_complete=True,
        algorithm_state={"science": [1, 2, 3]},
        trace_state={"enabled": False},
        sqlite_source=database,
        provenance_fingerprints=identity["provenance"],
        scientific_argv=identity["scientific_argv"],
        command_sha256=identity["command_sha256"],
        total_steps=25_000,
    )
    database.close()
    (root / "checkpoint_identity.json").write_bytes(_canonical_bytes(identity))
    spec = build_resume_spec(
        output_root=root,
        checkpoint_dir=root / "checkpoints" / "step-000000012500",
        resume_execution_commit=RESUME_COMMIT,
        historical_process_peak_bytes=386_452_664_320,
        historical_checkpoint_peak_bytes=481_224_437_760,
    )
    return root, spec


def _canary(spec: dict[str, object], **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": CANARY_RECEIPT_SCHEMA,
        "status": "PASS",
        "checkpoint_digest": spec["checkpoint_digest"],
        "resume_spec_sha256": spec["spec_sha256"],
        "generation_state_sha256": spec["generation_state_sha256"],
        "resume_execution_commit": RESUME_COMMIT,
        "source_completed_step": 12_500,
        "start_step": 12_501,
        "end_step": 12_550,
        "steps": 50,
        "semantic_parity_pass": True,
        "checkpoint_reload_pass": True,
        "checkpoint_save_pass": True,
        "forced_checkpoint_step": 12_550,
        "first_semantic_divergence_step": None,
        "step_state_digests_equal": True,
        "rng_state_equal": True,
        "candidate_registry_equal": True,
        "lineage_equal": True,
        "scientific_config_equal": True,
        "cgroup_failcnt_delta": 0,
        "cgroup_oom_kill_delta": 0,
        "cgroup_limit_path": spec["memory"]["cgroup_limit_path"],
        "cgroup_current_path": spec["memory"]["cgroup_current_path"],
        "cgroup_failcnt_path": spec["memory"]["cgroup_failcnt_path"],
        "cgroup_limit_bytes": 480 * GIB,
        "cgroup_baseline_current_bytes": 10 * GIB,
        "cgroup_peak_current_bytes": 350 * GIB,
        "science_pid": 101,
        "science_start_ticks": 1001,
        "reload_verifier_pid": 202,
        "reload_verifier_start_ticks": 2002,
        "reference_final_state_sha256": "a" * 64,
        "optimized_final_state_sha256": "a" * 64,
        "forced_checkpoint_state_sha256": "a" * 64,
        "torch_version": "2.0.1+cu118",
        "mmap_effective": False,
        "resume_peak_bytes": 340 * GIB,
        "checkpoint_peak_bytes": 350 * GIB,
    }
    value.update(overrides)
    return value


def _write_receipt(path: Path, value: dict[str, object]) -> None:
    value["receipt_sha256"] = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    path.write_bytes(_canonical_bytes(value))


def test_t14_checkpoint_12500_adoption_binds_only_transport_commit(
    tmp_path: Path,
) -> None:
    root, spec = _checkpoint_fixture(tmp_path)
    spec_path = (tmp_path / "resume.json").resolve()
    write_resume_spec(spec_path, spec)

    frozen, receipt = bind_resume_identity(
        spec_path=spec_path,
        output_root=root,
        current_execution_commit=RESUME_COMMIT,
        current_checkpoint_identity=_identity(RESUME_COMMIT),
    )

    assert frozen == _identity(SOURCE_COMMIT)
    assert receipt["status"] == "PASS"
    assert receipt["scientific_state_changes"] is False
    changed = _identity(RESUME_COMMIT)
    changed["total_steps"] = 20_000
    with pytest.raises(T14ResumeError, match="scientific identity changed"):
        bind_resume_identity(
            spec_path=spec_path,
            output_root=root,
            current_execution_commit=RESUME_COMMIT,
            current_checkpoint_identity=changed,
        )


def test_t14_memory_admission_requires_measured_parity_canary(
    tmp_path: Path,
) -> None:
    _root, spec = _checkpoint_fixture(tmp_path)
    historical = evaluate_memory_admission(
        spec,
        cgroup_limit_bytes=480 * GIB,
        cgroup_current_bytes=0,
    )
    assert historical.state == "WAITING_MEMORY_HEADROOM"
    assert historical.required_headroom_bytes > 480 * GIB

    measured = evaluate_memory_admission(
        spec,
        cgroup_limit_bytes=480 * GIB,
        cgroup_current_bytes=10 * GIB,
        optimized_canary_receipt=_canary(spec),
    )
    assert measured.state == "PASS"
    assert measured.required_headroom_bytes == 414 * GIB
    assert measured.available_headroom_bytes == 470 * GIB

    with pytest.raises(T14ResumeError, match="receipt is invalid"):
        evaluate_memory_admission(
            spec,
            cgroup_limit_bytes=480 * GIB,
            cgroup_current_bytes=0,
            optimized_canary_receipt=_canary(spec, rng_state_equal=False),
        )


def test_t14_checkpoint_save_does_not_reload_live_state_and_loads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = sqlite3.connect(tmp_path / "live.sqlite3")
    database.execute("CREATE TABLE graphs (id INTEGER PRIMARY KEY)")
    database.commit()
    identity = _identity(SOURCE_COMMIT)
    calls = 0
    original = checkpoints._torch_load

    def counted(path: Path, *, mmap: bool = False):
        nonlocal calls
        calls += 1
        return original(path, mmap=mmap)

    monkeypatch.setattr(checkpoints, "_torch_load", counted)
    root = tmp_path / "checkpoints"
    checkpoints.save_generation_checkpoint(
        root,
        completed_step=12_500,
        step_complete=True,
        algorithm_state={"science": np.arange(32, dtype=np.float32)},
        trace_state={"enabled": False},
        sqlite_source=database,
        provenance_fingerprints=identity["provenance"],
        scientific_argv=identity["scientific_argv"],
        command_sha256=identity["command_sha256"],
        total_steps=25_000,
        reload_after_write=False,
    )
    assert calls == 0
    loaded = checkpoints.load_generation_checkpoint(
        root / "step-000000012500",
        expected_completed_step=12_500,
        single_pass=True,
    )
    database.close()
    assert calls == 1
    np.testing.assert_array_equal(
        loaded.algorithm_state["science"], np.arange(32, dtype=np.float32)
    )


def test_checkpoint_root_falls_back_before_repairing_latest_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = sqlite3.connect(tmp_path / "live.sqlite3")
    database.execute("CREATE TABLE graphs (id INTEGER PRIMARY KEY)")
    database.commit()
    identity = _identity(SOURCE_COMMIT)
    root = tmp_path / "checkpoints"
    for step in (100, 200):
        checkpoints.save_generation_checkpoint(
            root,
            completed_step=step,
            step_complete=True,
            algorithm_state={"step": step},
            trace_state={"enabled": False},
            sqlite_source=database,
            provenance_fingerprints=identity["provenance"],
            scientific_argv=identity["scientific_argv"],
            command_sha256=identity["command_sha256"],
            total_steps=25_000,
        )
    database.close()
    original = checkpoints._torch_load

    def reject_latest(path: Path, *, mmap: bool = False):
        if path.parent.name == "step-000000000200":
            raise checkpoints.GenerationCheckpointError("payload cannot be loaded")
        return original(path, mmap=mmap)

    monkeypatch.setattr(checkpoints, "_torch_load", reject_latest)
    loaded = checkpoints.load_generation_checkpoint(root)
    latest = json.loads((root / "LATEST").read_text())
    assert loaded.completed_step == 100
    assert latest["checkpoint_dir"] == "step-000000000100"


def test_t14_pending_checkpoint_requires_reload_before_latest_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = sqlite3.connect(tmp_path / "live.sqlite3")
    database.execute("CREATE TABLE graphs (id INTEGER PRIMARY KEY)")
    database.commit()
    identity = _identity(SOURCE_COMMIT)
    root = tmp_path / "checkpoints"
    for step, strict in ((100, True), (200, False)):
        checkpoints.save_generation_checkpoint(
            root,
            completed_step=step,
            step_complete=True,
            algorithm_state={"step": step},
            trace_state={"enabled": False},
            sqlite_source=database,
            provenance_fingerprints=identity["provenance"],
            scientific_argv=identity["scientific_argv"],
            command_sha256=identity["command_sha256"],
            total_steps=25_000,
            reload_after_write=strict,
        )
    database.close()
    assert json.loads((root / "LATEST").read_text())["completed_step"] == 100
    assert json.loads((root / "PENDING_LATEST.json").read_text())[
        "completed_step"
    ] == 200

    original = checkpoints._torch_load

    def reject(_path: Path, *, mmap: bool = False):
        raise checkpoints.GenerationCheckpointError("independent reload failed")

    monkeypatch.setattr(checkpoints, "_torch_load", reject)
    with pytest.raises(checkpoints.GenerationCheckpointError):
        checkpoints.promote_generation_checkpoint(root / "step-000000000200")
    assert json.loads((root / "LATEST").read_text())["completed_step"] == 100
    monkeypatch.setattr(checkpoints, "_torch_load", original)
    promoted = checkpoints.promote_generation_checkpoint(
        root / "step-000000000200"
    )
    assert promoted.completed_step == 200
    assert json.loads((root / "LATEST").read_text())["completed_step"] == 200


def test_t14_resume_parity_tensor_checkpoint_transport_is_zero_copy() -> None:
    class Graph:
        def __init__(self, value: int) -> None:
            self.value = value

    source = Graph(10)
    module = SimpleNamespace(
        graph_map={"source": [source, None, None]},
        graph_index_map={"source": 0},
    )
    cache = CompactMoveScopedTransitionMap(
        module,
        {},
        seed=7,
        expanded_capacity=1,
        rebuild_target=lambda graph, action: Graph(graph.value + int(action[1])),
    )
    target = Graph(11)
    cache.record_enumerated(target, ("ADD", 1))
    cache["source"] = (
        ["target"],
        [target],
        [np.asarray([0.25, 1.0], dtype=np.float64)],
        [np.asarray([3.0, 4.0], dtype=np.float32)],
    )
    exported = cache.export_checkpoint_state(tensor_storage=True)
    original = cache._entries["source"]
    tensor = exported["entries"][0]["embeddings"]
    assert tensor.data_ptr() == original.embeddings.ctypes.data

    restored = CompactMoveScopedTransitionMap(
        module,
        {},
        seed=7,
        expanded_capacity=1,
        rebuild_target=lambda graph, action: Graph(graph.value + int(action[1])),
    )
    restored.restore_checkpoint_state(exported, consume=True)
    assert exported == {}
    np.testing.assert_array_equal(restored["source"][2], cache["source"][2])
    np.testing.assert_array_equal(restored["source"][3], cache["source"][3])


def test_t14_restored_embedding_arrays_are_read_only() -> None:
    bridge = object.__new__(TasteComRecGCFullBridge)
    restored = bridge._restored_embedding_values(
        np.asarray([1.0, 2.0], dtype=np.float32)
    )
    assert isinstance(restored, np.ndarray)
    assert restored.flags.writeable is False
    with pytest.raises(ValueError):
        restored[0] = 9.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resume_spec_sha256", "f" * 64),
        ("checkpoint_save_pass", False),
        ("forced_checkpoint_state_sha256", "b" * 64),
        ("cgroup_limit_bytes", 479 * GIB),
        ("cgroup_failcnt_delta", 1),
        ("reload_verifier_pid", 101),
    ],
)
def test_t14_canary_receipt_rejects_bound_evidence_tamper(
    tmp_path: Path, field: str, value: object
) -> None:
    _root, spec = _checkpoint_fixture(tmp_path)
    with pytest.raises(T14ResumeError, match="receipt is invalid"):
        evaluate_memory_admission(
            spec,
            cgroup_limit_bytes=480 * GIB,
            cgroup_current_bytes=10 * GIB,
            optimized_canary_receipt=_canary(spec, **{field: value}),
        )


def test_t14_owner_passes_spec_bound_lock_fd_to_science() -> None:
    source = inspect.getsource(owner.main)
    assert '"T14_FULL_STATE_CONSUMER_LOCK_FD"' in source
    assert "pass_fds=(lock_handle.fileno(),)" in source


def test_t14_inherited_lock_survives_owner_descriptor_close(tmp_path: Path) -> None:
    lock_path = tmp_path / "full-state.lock"
    owner_lock = lock_path.open("a+b")
    fcntl.flock(owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,sys,time; os.fstat(int(sys.argv[1])); print('READY', flush=True); time.sleep(30)",
            str(owner_lock.fileno()),
        ],
        pass_fds=(owner_lock.fileno(),),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        owner_lock.close()
        contender = lock_path.open("a+b")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            contender.close()
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_t14_auditor_serialization_uses_pid_and_start_ticks(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    live = proc / "77"
    live.mkdir(parents=True)
    # fields after comm: state through starttime (field 22)
    live.joinpath("stat").write_text(
        "77 (full state auditor) S " + " ".join(["0"] * 18 + ["12345"]) + "\n"
    )
    with pytest.raises(T14ResumeError, match="auditor is still live"):
        assert_auditor_serialized(
            auditor_pid=77,
            auditor_start_ticks=12_345,
            proc_root=proc,
        )
    observed = assert_auditor_serialized(
        auditor_pid=77,
        auditor_start_ticks=99_999,
        proc_root=proc,
    )
    assert observed.start_ticks == 12_345


def test_t14_owner_dry_run_requires_480g_cap_plus_64g_admission(
    tmp_path: Path,
) -> None:
    root, spec = _checkpoint_fixture(tmp_path)
    config = (tmp_path / "config.yaml").resolve()
    config.write_text("runtime: {}\n")
    wrapper = (tmp_path / "science.sh").resolve()
    wrapper.write_text("#!/usr/bin/env bash\nexit 99\n")
    wrapper.chmod(0o755)
    limit = (tmp_path / "memory.limit").resolve()
    current = (tmp_path / "memory.current").resolve()
    limit.write_text(str(480 * GIB))
    current.write_text(str(10 * GIB))
    spec["memory"]["cgroup_limit_path"] = str(limit)
    spec["memory"]["cgroup_current_path"] = str(current)
    unsigned = {key: value for key, value in spec.items() if key != "spec_sha256"}
    spec["spec_sha256"] = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    spec_path = (tmp_path / "resume.json").resolve()
    write_resume_spec(spec_path, spec)
    receipt_path = (tmp_path / "canary.json").resolve()
    _write_receipt(receipt_path, _canary(spec))
    proc = (tmp_path / "proc").resolve()
    proc.mkdir()

    assert owner.main(
        [
            "--config",
            str(config),
            "--resume-spec",
            str(spec_path),
            "--canary-receipt",
            str(receipt_path),
            "--owner-root",
            str((tmp_path / "owner").resolve()),
            "--science-wrapper",
            str(wrapper),
            "--cgroup-limit-file",
            str(limit),
            "--cgroup-current-file",
            str(current),
            "--auditor-pid",
            "77",
            "--auditor-start-ticks",
            "12345",
            "--proc-root",
            str(proc),
            "--dry-run",
        ]
    ) == 0
    admission = json.loads(
        (tmp_path / "owner" / "admission.json").read_text()
    )
    assert admission["status"] == "PASS"
    assert admission["science_started"] is False
    assert not (tmp_path / "owner" / "heartbeat.json").exists()
    assert root.exists()
