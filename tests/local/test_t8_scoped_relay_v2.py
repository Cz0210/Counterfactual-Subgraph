from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scripts/local/run_t8_scoped_relay_v2.sh"
STATUS = ROOT / "scripts/local/status_t8_scoped_relay_v2.sh"
EXPECTED_BYTES = 6_103_923_589
EXPECTED_SHA = "06702fdc97ae2bb3661855497a336d19c6ceb33fd53f2304f41471781629346e"


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_relay_pins_canonical_hpc_contract_and_exact_result_identity() -> None:
    source = RUN.read_text(encoding="utf-8")
    assert "/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/continuations/" in source
    assert "hierarchical-08a63955-20260904T181200Z" in source
    assert "HIERARCHICAL_PACKAGE_READY.json" in source
    assert "result_manifest.json" in source
    assert f"EXPECTED_ARCHIVE_BYTES={EXPECTED_BYTES}" in source
    assert f"EXPECTED_ARCHIVE_SHA256={EXPECTED_SHA}" in source
    assert 'package = (root / "artifacts" / "package").resolve(strict=True)' in source
    assert 'archive = (package / manifest["archive_name"]).resolve(strict=True)' in source


def test_relay_is_append_verified_atomic_and_never_deletes_hpc_source() -> None:
    source = RUN.read_text(encoding="utf-8")
    assert "--append-verify" in source
    assert "--protect-args" in source
    assert "--info=progress2" in source
    assert source.count('"${rsync_common_args[@]}"') == 10
    assert '\"$RSYNC_BIN\" \"$required_rsync_option\" --version' in source
    assert "rsync lacks required $required_rsync_option support" in source
    assert ".partial-$RELAY_ATTEMPT_ID" in source
    assert 'mv "$local_partial" "$local_final"' in source
    assert "os.rename(partial, final)" in source
    forbidden = (
        "--delete",
        "--remove-source-files",
        "rm -rf",
        "ssh $HPC_ALIAS rm",
        "matrix_authority",
        "while true",
    )
    for token in forbidden:
        assert token not in source


def test_relay_has_heartbeat_independent_sha_and_terminal_exit() -> None:
    source = RUN.read_text(encoding="utf-8")
    assert '"heartbeat_at"' in source
    assert "COPYING_HPC_TO_MAC" in source
    assert "COPYING_MAC_TO_AUTODL" in source
    assert "VERIFYING_MAC_SHA256" in source
    assert "VERIFYING_AUTODL_SHA256" in source
    assert "independent_autodl_sha256_verified" in source
    assert "write_state FAILED" in source
    assert "state=PASS" in source
    assert '"matrix_write_enabled": False' in source
    assert '"hpc_source_delete_enabled": False' in source


def test_relay_re_adopts_verified_mac_final_after_network_interruption() -> None:
    source = RUN.read_text(encoding="utf-8")
    assert "VERIFYING_EXISTING_MAC_FINAL" in source
    assert '[[ -f "$local_final/MAC_RELAY_READY.json" ]]' in source
    assert 'digest(archive) != expected_sha' in source
    assert 'digest(evidence_archive) != evidence_sha' in source
    assert 'if [[ "$adopt_existing_mac_final" != true ]]; then' in source
    assert "re-adopted verified content-addressed Mac final" in source


def test_status_is_read_only_and_reports_scoped_paths(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir()
    payload = {
        "state": "COPYING_HPC_TO_MAC",
        "heartbeat_at": "2026-09-05T00:00:00+00:00",
        "detail": "result archive",
        "pid": 99999999,
        "attempt_id": "attempt-test",
        "hpc_package_root": "/share/test/package",
        "hpc_archive_path": "/share/test/package/result.tar.gz",
        "mac_partial_root": "/Volumes/Test/.partial",
        "mac_final_root": "/Volumes/Test/final",
        "autodl_partial_root": "/autodl/.partial",
        "autodl_final_root": "/autodl/final",
        "expected_archive_bytes": EXPECTED_BYTES,
        "expected_archive_sha256": EXPECTED_SHA,
        "current_partial_bytes": 1234,
        "log_path": "/tmp/relay.log",
        "matrix_write_enabled": False,
        "hpc_source_delete_enabled": False,
    }
    (control / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = subprocess.run(
        ["bash", str(STATUS)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MAC_RELAY_ROOT": str(tmp_path), "RELAY_CONTROL_ROOT": str(control)},
    )
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert "relay_state=COPYING_HPC_TO_MAC" in result.stdout
    assert "relay_current_partial_bytes=1234" in result.stdout
    assert "relay_hpc_source_delete_enabled=False" in result.stdout
    assert "status_side_effects=NONE" in result.stdout


def test_status_source_contains_no_mutating_remote_operation() -> None:
    source = STATUS.read_text(encoding="utf-8")
    for token in ("ssh ", "rsync", "rm ", "mv ", "mkdir", "touch ", "kill -TERM", "scancel", "sbatch"):
        assert token not in source


def test_relay_fails_closed_before_transfer_on_wrong_pinned_size(tmp_path: Path) -> None:
    hpc_root = tmp_path / "hpc" / "hierarchical"
    package = hpc_root / "artifacts" / "package"
    package.mkdir(parents=True)
    archive = package / "t8_exact_result_bundle.tar.gz"
    archive.write_bytes(b"not-the-6.1GB-bundle")
    evidence = package / "t8_hierarchical_evidence.tar.gz"
    evidence.write_bytes(b"evidence")
    evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    (package / "result_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "archive_name": archive.name,
                "archive_bytes": EXPECTED_BYTES,
                "archive_sha256": EXPECTED_SHA,
            }
        ),
        encoding="utf-8",
    )
    (package / "hierarchical_evidence_manifest.json").write_text(
        json.dumps({"status": "PASS", "archive_sha256": evidence_sha}), encoding="utf-8"
    )
    (package / "HIERARCHICAL_PACKAGE_READY.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "result_archive_sha256": EXPECTED_SHA,
                "evidence_archive_sha256": evidence_sha,
            }
        ),
        encoding="utf-8",
    )
    fake_ssh = tmp_path / "fake-ssh"
    _executable(
        fake_ssh,
        "#!/usr/bin/env bash\nset -eu\ncommand=${!#}\nexec bash -c \"$command\"\n",
    )
    fake_rsync = tmp_path / "fake-rsync"
    marker = tmp_path / "rsync-called"
    _executable(
        fake_rsync,
        f"#!/bin/sh\n"
        "case \"${1:-}\" in --append-verify|--protect-args|--info=progress2) exit 0;; esac\n"
        f"touch '{marker}'\nexit 99\n",
    )
    env = {
        **os.environ,
        "HPC_ALIAS": "fake-hpc",
        "AUTODL_ALIAS": "fake-autodl",
        "HPC_HIERARCHICAL_ROOT": str(hpc_root),
        "MAC_RELAY_ROOT": str(tmp_path / "mac"),
        "AUTODL_IMPORT_PARENT": str(tmp_path / "autodl"),
        "AUTODL_PYTHON": "/usr/bin/python3",
        "RELAY_CONTROL_ROOT": str(tmp_path / "control"),
        "RELAY_ATTEMPT_ID": "wrong-size-test",
        "SSH_BIN": str(fake_ssh),
        "RSYNC_BIN": str(fake_rsync),
    }
    result = subprocess.run(["bash", str(RUN)], capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert not marker.exists()
    state = json.loads((tmp_path / "control" / "state.json").read_text(encoding="utf-8"))
    assert state["state"] == "FAILED"
