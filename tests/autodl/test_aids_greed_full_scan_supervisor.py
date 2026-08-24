from __future__ import annotations

import json
import os
from pathlib import Path

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
            "science_root_identity_at_contract_creation": None,
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
        resume_count=0,
        resume_reasons=(),
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
    payload["close_pair_contract_sha256"] = "tampered"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(supervisor.AIDSGreedFullScanSupervisorError, match="closure"):
        supervisor.validate_receipt(
            receipt_path=receipt_path,
            expected_science_root=science,
            expected_execution_commit=COMMIT,
        )
