from __future__ import annotations

import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.contracts import sha256_file
from src.utils import autodl_aids_greed_full_scan_supervisor as supervisor


COMMIT = "a" * 40


def _terminal(science: Path) -> dict[str, object]:
    return {
        "status": "PASS",
        "science_root": str(science),
        "science_root_identity": supervisor._directory_identity(science),
        "close_pair_contract": str(science / "close_pair_contract.json"),
        "close_pair_contract_sha256": "close-sha",
        "pair_semantics_audit_sha256": "audit-sha",
        "distance_scan_manifest_sha256": "scan-sha",
        "terminal_array_audit": {
            "status": "PASS",
            "physical_pair_count": 3,
            "finite_count": 3,
            "nonfinite_count": 0,
            "logical_close_pair_count": 2,
            "theta": 0.1,
            "predicate": "normalized_distance_float32 <= theta_float32",
            "streaming_block_size": 1000000,
        },
        "terminal_files": {},
    }


def _patch_small_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    science: Path,
    child_results: list[tuple[int, str, bool]],
) -> list[bool]:
    resumes: list[bool] = []
    monkeypatch.setattr(supervisor, "_git_head", lambda _root: COMMIT)
    monkeypatch.setattr(
        supervisor,
        "validate_child_command",
        lambda *_args, **_kwargs: {
            "pair_store_manifest": "/source/pairs.json",
            "expected_pair_store_manifest_sha256": "pair-sha",
            "generation_dir": "/source/generation",
            "distance_checkpoint": "/source/model.pt",
        },
    )
    monkeypatch.setattr(
        supervisor,
        "read_process_identity",
        lambda pid, **_kwargs: {
            "pid": pid,
            "start_ticks": pid + 10,
            "cmdline_sha256": f"cmd-{pid}",
        },
    )
    monkeypatch.setattr(supervisor, "process_identity_matches", lambda *_a, **_k: False)
    monkeypatch.setattr(
        supervisor,
        "validate_resume_checkpoint",
        lambda **_kwargs: {
            "checkpoint_sha256": "checkpoint-sha",
            "checkpoint_identity_sha256": "identity-sha",
            "rows_processed": 3,
        },
    )
    monkeypatch.setattr(
        supervisor,
        "validate_terminal_science",
        lambda **_kwargs: _terminal(science),
    )

    def fake_child(**kwargs: object) -> tuple[int, str, bool]:
        resume = bool(kwargs["resume"])
        resumes.append(resume)
        science.mkdir(parents=True, exist_ok=True)
        result = child_results.pop(0)
        if result[0] == 0:
            (science / "PASS").write_text("PASS\n", encoding="utf-8")
        return result

    monkeypatch.setattr(supervisor, "_run_child", fake_child)
    return resumes


def test_host_restart_uses_fresh_receipt_attempt_and_authenticated_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    science = campaign / "pair_semantics_science"
    resumes = _patch_small_run(
        monkeypatch,
        science=science,
        child_results=[(143, "", True), (0, "", False)],
    )
    common = {
        "project_root": Path(__file__).resolve().parents[2],
        "execution_commit": COMMIT,
        "campaign_root": campaign,
        "science_root": science,
        "proc_root": proc,
        "max_same_root_resumes": 1,
        "semantic_failure_markers": (),
        "child_argv": ("python", "reviewed-child"),
    }
    with pytest.raises(SystemExit, match="143"):
        supervisor.run_supervisor(
            **common, receipt_output=campaign / "pair_semantics_receipt/attempt-0"
        )
    receipt = supervisor.run_supervisor(
        **common, receipt_output=campaign / "pair_semantics_receipt/attempt-1"
    )
    assert resumes == [False, True]
    assert receipt["science_root"] == str(science)
    assert not (campaign / "pair_semantics_receipt/attempt-0/PASS").exists()
    assert (campaign / "pair_semantics_receipt/attempt-1/PASS").read_bytes() == b"PASS\n"
    assert json.loads(
        (campaign / supervisor.CONTROL_DIRECTORY / supervisor.STATE_NAME).read_text()
    )["resume_count"] == 1


def test_semantic_failure_never_authorizes_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    science = campaign / "pair_semantics_science"
    resumes = _patch_small_run(
        monkeypatch,
        science=science,
        child_results=[(70, "AIDSPairSemanticsError: identity differs", False)],
    )
    with pytest.raises(SystemExit, match="70"):
        supervisor.run_supervisor(
            project_root=Path(__file__).resolve().parents[2],
            execution_commit=COMMIT,
            campaign_root=campaign,
            science_root=science,
            receipt_output=campaign / "pair_semantics_receipt/attempt-0",
            proc_root=proc,
            max_same_root_resumes=1,
            semantic_failure_markers=(),
            child_argv=("python", "reviewed-child"),
        )
    state = json.loads(
        (campaign / supervisor.CONTROL_DIRECTORY / supervisor.STATE_NAME).read_text()
    )
    assert resumes == [False]
    assert state["status"] == "FAILED_SEMANTIC_OR_PROVENANCE"
    assert state["resume_count"] == 0


def test_hard_loss_cannot_adopt_replaced_science_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    science = campaign / "pair_semantics_science"
    _patch_small_run(
        monkeypatch,
        science=science,
        child_results=[(143, "", True)],
    )
    common = {
        "project_root": Path(__file__).resolve().parents[2],
        "execution_commit": COMMIT,
        "campaign_root": campaign,
        "science_root": science,
        "proc_root": proc,
        "max_same_root_resumes": 1,
        "semantic_failure_markers": (),
        "child_argv": ("python", "reviewed-child"),
    }
    with pytest.raises(SystemExit, match="143"):
        supervisor.run_supervisor(
            **common, receipt_output=campaign / "receipt/attempt-0"
        )
    science.rename(campaign / "displaced-science")
    science.mkdir()
    with pytest.raises(
        supervisor.AIDSGreedFullScanSupervisorError, match="inode differs"
    ):
        supervisor.run_supervisor(
            **common, receipt_output=campaign / "receipt/attempt-1"
        )


def test_science_initialization_recovers_both_crash_boundaries(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir()
    science = tmp_path / "science"
    # Crash after control mkdir but before any initialization record.
    frozen = supervisor._reserve_science_root(
        control_root=control, science_root=science
    )
    assert frozen["status"] == "INIT_FROZEN"
    assert frozen["science_root_identity"] == supervisor._directory_identity(science)

    control_two = tmp_path / "control-two"
    control_two.mkdir()
    science_two = tmp_path / "science-two"
    temporary = tmp_path / ".science-two.init-fixture"
    temporary.mkdir()
    identity = supervisor._directory_identity(temporary)
    supervisor._atomic_json(
        control_two / supervisor.INIT_NAME,
        {
            "schema_version": supervisor.SUPERVISOR_SCHEMA,
            "status": "INIT_PENDING",
            "science_root": str(science_two),
            "temporary_root": str(temporary),
            "science_root_identity": identity,
            "reserved_at": "fixture",
        },
    )
    # Crash after INIT_PENDING but before promotion.
    recovered = supervisor._reserve_science_root(
        control_root=control_two, science_root=science_two
    )
    assert recovered["status"] == "INIT_FROZEN"
    assert supervisor._directory_identity(science_two) == identity
    # Crash after promotion but before INIT_FROZEN publication.
    supervisor._atomic_json(
        control_two / supervisor.INIT_NAME,
        {**recovered, "status": "INIT_PENDING", "frozen_at": None},
    )
    assert supervisor._reserve_science_root(
        control_root=control_two, science_root=science_two
    )["status"] == "INIT_FROZEN"


def test_contract_only_initialization_replays_before_first_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    science = campaign / "science"
    resumes = _patch_small_run(
        monkeypatch, science=science, child_results=[(0, "", False)]
    )
    original_atomic = supervisor._atomic_json

    def crash_before_state(path: Path, payload: object) -> None:
        if path.name == supervisor.STATE_NAME:
            raise RuntimeError("fixture crash after contract")
        original_atomic(path, payload)

    monkeypatch.setattr(supervisor, "_atomic_json", crash_before_state)
    common = {
        "project_root": Path(__file__).resolve().parents[2],
        "execution_commit": COMMIT,
        "campaign_root": campaign,
        "science_root": science,
        "receipt_output": campaign / "receipt/attempt-0",
        "proc_root": proc,
        "max_same_root_resumes": 1,
        "semantic_failure_markers": (),
        "child_argv": ("python", "reviewed-child"),
    }
    with pytest.raises(RuntimeError, match="after contract"):
        supervisor.run_supervisor(**common)
    control = campaign / supervisor.CONTROL_DIRECTORY
    assert (control / supervisor.CONTRACT_NAME).is_file()
    assert not (control / supervisor.STATE_NAME).exists()
    monkeypatch.setattr(supervisor, "_atomic_json", original_atomic)
    supervisor.run_supervisor(**common)
    assert resumes == [False]


def test_pdeathsig_resets_inherited_sigterm_before_parent_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(supervisor.sys, "platform", "linux")
    monkeypatch.setattr(
        supervisor.signal,
        "signal",
        lambda signum, handler: events.append(("signal", signum, handler)),
    )
    monkeypatch.setattr(
        supervisor.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(
            prctl=lambda *_args: events.append(("prctl",)) or 0
        ),
    )
    monkeypatch.setattr(supervisor.os, "getppid", lambda: 41)
    monkeypatch.setattr(supervisor.os, "getpid", lambda: 42)
    monkeypatch.setattr(
        supervisor.os,
        "kill",
        lambda pid, signum: events.append(("kill", pid, signum)),
    )
    supervisor._pdeathsig(40)
    assert events[0] == ("signal", signal.SIGTERM, signal.SIG_DFL)
    assert events[1] == ("prctl",)
    assert events[2] == ("kill", 42, signal.SIGTERM)


def test_terminal_array_audit_rejects_forged_bitmap_count_and_negative(
    tmp_path: Path,
) -> None:
    distances = tmp_path / "distances.npy"
    bitmap = tmp_path / "bitmap.npy"
    np.save(distances, np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    np.save(bitmap, np.asarray([1, 1, 0], dtype=np.uint8))
    audit = supervisor._validate_terminal_arrays(
        distance_path=distances,
        bitmap_path=bitmap,
        expected_count=3,
        theta=0.1,
        block_size=2,
    )
    assert audit["logical_close_pair_count"] == 2
    with pytest.raises(supervisor.AIDSGreedFullScanSupervisorError, match="schema"):
        supervisor._validate_terminal_arrays(
            distance_path=distances,
            bitmap_path=bitmap,
            expected_count=4,
            theta=0.1,
        )
    np.save(bitmap, np.asarray([1, 0, 0], dtype=np.uint8))
    with pytest.raises(supervisor.AIDSGreedFullScanSupervisorError, match="distance <= theta"):
        supervisor._validate_terminal_arrays(
            distance_path=distances,
            bitmap_path=bitmap,
            expected_count=3,
            theta=0.1,
        )
    np.save(distances, np.asarray([-0.1, 0.1, 0.2], dtype=np.float32))
    with pytest.raises(supervisor.AIDSGreedFullScanSupervisorError, match="negative"):
        supervisor._validate_terminal_arrays(
            distance_path=distances,
            bitmap_path=bitmap,
            expected_count=3,
            theta=0.1,
        )


def test_held_flock_rejects_unlink_recreate_race(tmp_path: Path) -> None:
    lock = tmp_path / "science.lock"
    with supervisor._HeldFlock(lock) as held:
        original = held.identity
        lock.unlink()
        lock.write_bytes(b"")
        assert supervisor._file_identity(lock)["inode"] != original["inode"]
        with pytest.raises(
            supervisor.AIDSGreedFullScanSupervisorError, match="lock path/inode"
        ):
            held.assert_path_identity()


@pytest.mark.parametrize("adopted_terminal", [False, True])
def test_terminal_validation_lock_replacement_never_publishes_receipt_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adopted_terminal: bool,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    science = campaign / "science"
    child_results = [(143, "", True)] if adopted_terminal else [(0, "", False)]
    _patch_small_run(
        monkeypatch, science=science, child_results=child_results
    )
    common = {
        "project_root": Path(__file__).resolve().parents[2],
        "execution_commit": COMMIT,
        "campaign_root": campaign,
        "science_root": science,
        "proc_root": proc,
        "max_same_root_resumes": 1,
        "semantic_failure_markers": (),
        "child_argv": ("python", "reviewed-child"),
    }
    if adopted_terminal:
        with pytest.raises(SystemExit, match="143"):
            supervisor.run_supervisor(
                **common, receipt_output=campaign / "receipt/attempt-0"
            )
        (science / "PASS").write_text("PASS\n", encoding="utf-8")
    receipt = campaign / f"receipt/attempt-{int(adopted_terminal)}"
    lock = science.parent / f".{science.name}{supervisor.LOCK_NAME}"

    def replace_named_lock(**_kwargs: object) -> dict[str, object]:
        lock.unlink()
        lock.write_bytes(b"")
        return _terminal(science)

    monkeypatch.setattr(
        supervisor, "validate_terminal_science", replace_named_lock
    )
    with pytest.raises(
        supervisor.AIDSGreedFullScanSupervisorError, match="lock path/inode"
    ):
        supervisor.run_supervisor(**common, receipt_output=receipt)
    assert not (receipt / "PASS").exists()
    assert not (
        campaign
        / supervisor.CONTROL_DIRECTORY
        / supervisor.TERMINAL_PASS_NAME
    ).exists()


def test_post_receipt_lock_replacement_revokes_receipt_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    science = campaign / "science"
    _patch_small_run(
        monkeypatch,
        science=science,
        child_results=[(0, "", False)],
    )
    receipt = campaign / "receipt/attempt-0"
    lock = science.parent / f".{science.name}{supervisor.LOCK_NAME}"
    original_publish = supervisor._publish_receipt

    def publish_then_replace(**kwargs: object) -> dict[str, object]:
        payload = original_publish(**kwargs)
        lock.unlink()
        lock.write_bytes(b"")
        return payload

    monkeypatch.setattr(supervisor, "_publish_receipt", publish_then_replace)
    with pytest.raises(
        supervisor.AIDSGreedFullScanSupervisorError, match="lock path/inode"
    ):
        supervisor.run_supervisor(
            project_root=Path(__file__).resolve().parents[2],
            execution_commit=COMMIT,
            campaign_root=campaign,
            science_root=science,
            receipt_output=receipt,
            proc_root=proc,
            max_same_root_resumes=1,
            semantic_failure_markers=(),
            child_argv=("python", "reviewed-child"),
        )
    assert (receipt / supervisor.RECEIPT_NAME).is_file()
    assert not (receipt / "PASS").exists()
    with pytest.raises(
        supervisor.AIDSGreedFullScanSupervisorError, match="receipt PASS is absent"
    ):
        supervisor.validate_receipt(
            receipt_path=receipt / supervisor.RECEIPT_NAME,
            expected_science_root=science,
        )


def test_live_writer_scan_matches_exact_science_argument(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    science = tmp_path / "campaign/science"
    for pid, command in {
        11: ["python", "run_aids_comrecgc_pair_semantics.py", "--output-dir", str(science)],
        12: ["python", "other.py", "--output-dir", str(science)],
    }.items():
        root = proc / str(pid)
        root.mkdir(parents=True)
        (root / "cmdline").write_bytes(b"\0".join(value.encode() for value in command) + b"\0")
        fields = ["S", *(["0"] * 18), str(1000 + pid), *(["0"] * 8)]
        (root / "stat").write_text(f"{pid} (worker) " + " ".join(fields), encoding="utf-8")
    writers = supervisor.find_live_science_writers(
        science_root=science, proc_root=proc
    )
    assert [value["pid"] for value in writers] == [11]


def test_inode_writer_audit_rejects_fd_without_science_path_in_cmdline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    science = tmp_path / "science"
    science.mkdir()
    artifact = science / "checkpoint.json"
    artifact.write_text("{}", encoding="utf-8")
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(
        supervisor,
        "_find_writable_process_references",
        lambda paths, **_kwargs: [
            {"pid": 44, "kind": "fd", "path": str(paths[0])}
        ],
    )
    with pytest.raises(
        supervisor.AIDSGreedFullScanSupervisorError,
        match="writable process references",
    ):
        supervisor._assert_no_writable_science_references(
            science, proc_root=proc
        )


def test_receipt_tamper_and_lock_inode_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    science = tmp_path / "science"
    science.mkdir()
    control = tmp_path / "control"
    control.mkdir()
    lock = tmp_path / "science.lock"
    lock.write_text("", encoding="utf-8")
    lock_identity = {
        "device": int(os.lstat(lock).st_dev),
        "inode": int(os.lstat(lock).st_ino),
    }
    contract = control / supervisor.CONTRACT_NAME
    supervisor._atomic_json(
        contract,
        {
            "schema_version": supervisor.SUPERVISOR_SCHEMA,
            "status": "FROZEN",
            "science_root": str(science),
            "science_root_identity_at_contract_creation": supervisor._directory_identity(
                science
            ),
            "execution_commit": COMMIT,
        },
    )
    terminal = _terminal(science)
    terminal_path, terminal_manifest = supervisor._freeze_terminal_supervisor_manifest(
        control_root=control,
        contract_path=contract,
        terminal=terminal,
        resume_count=0,
        resume_reasons=(),
        owner_identity={"pid": 1, "start_ticks": 2, "cmdline_sha256": "x"},
        lock_path=lock,
        lock_identity=lock_identity,
    )
    receipt_root = tmp_path / "receipt"
    supervisor._publish_receipt(
        receipt_output=receipt_root,
        contract_path=contract,
        terminal_manifest_path=terminal_path,
        terminal_manifest=terminal_manifest,
        terminal=terminal,
    )
    monkeypatch.setattr(
        supervisor,
        "validate_terminal_science",
        lambda **_kwargs: terminal,
    )
    receipt_path = receipt_root / supervisor.RECEIPT_NAME
    assert supervisor.validate_receipt(
        receipt_path=receipt_path,
        expected_science_root=science,
        expected_execution_commit=COMMIT,
    )["receipt_sha256"] == sha256_file(receipt_path)
    payload = json.loads(receipt_path.read_text())
    assert payload["same_root_resume_count"] == 0
    payload["same_root_resume_count"] = 1
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(supervisor.AIDSGreedFullScanSupervisorError, match="closure"):
        supervisor.validate_receipt(
            receipt_path=receipt_path,
            expected_science_root=science,
            expected_execution_commit=COMMIT,
        )
    payload["same_root_resume_count"] = 0
    payload["close_pair_contract_sha256"] = "tampered"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(supervisor.AIDSGreedFullScanSupervisorError, match="closure"):
        supervisor.validate_receipt(
            receipt_path=receipt_path,
            expected_science_root=science,
            expected_execution_commit=COMMIT,
        )
