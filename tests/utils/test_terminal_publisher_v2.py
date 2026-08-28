from __future__ import annotations

import os
from pathlib import Path
import shutil
import threading
from typing import Iterator
import uuid

import pytest

from src.utils.managed_execution_v2 import (
    ManagedExecutionV2Error,
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
    write_worker_exit,
    write_worker_metadata,
    write_worker_raw_evidence,
)
from src.utils.terminal_publisher_v2 import (
    HeldSealedArtifactV2,
    SealedWorkerArtifactV2,
    TerminalPublisherV2Error,
    open_sealed_worker_artifact,
    seal_worker_staging,
    verify_and_publish_sealed_attempt,
)


def _sealed(tmp_path: Path) -> tuple[object, object, SealedWorkerArtifactV2]:
    attempt = create_managed_attempt(
        stage_root=tmp_path,
        controller_id="controller-v2",
        task_id="task-v2",
        git_commit="1" * 40,
        config_hash="2" * 64,
        input_hashes={"predecessor": "3" * 64},
        boot_id="boot-generation-1",
    )
    staging = create_worker_staging(attempt)
    artifact = staging.artifact_root / "result.json"
    artifact.write_text('{"scientific":true}\n', encoding="utf-8")
    artifact.chmod(0o600)
    raw = write_worker_raw_evidence(
        staging,
        {
            "expected_final_path": str(tmp_path / "published" / "result"),
            "predecessors": [
                {
                    "kind": "TASTE_GCF_NEUROSED_PASS",
                    "path": str(tmp_path / "neurosed" / "PASS"),
                    "sha256": "3" * 64,
                }
            ],
        },
    )
    raw.close()
    worker_exit = write_worker_exit(
        staging,
        {"exit_code": 0, "worker_closed_artifact_writers": True},
    )
    worker_exit.close()
    sealed = seal_worker_staging(staging)
    staging.close()
    return attempt, staging, sealed


def _close_attempt(attempt: object) -> None:
    getattr(attempt, "close")()


def test_terminal_no_mutable_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt, _staging, sealed = _sealed(tmp_path)
    calls: list[object] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("file linking is forbidden")

    monkeypatch.setattr(os, "link", forbidden)
    final = tmp_path / "published" / "same-fs"
    final.parent.mkdir()
    try:
        with open_sealed_worker_artifact(sealed.seal_path) as held:
            verify_and_publish_sealed_attempt(
                held,
                final_path=final,
                verification={"status": "PASS", "method": "test"},
            )
        assert calls == []
        assert final.is_dir()
    finally:
        _close_attempt(attempt)


def test_terminal_atomic_publish_same_fs(tmp_path: Path) -> None:
    attempt, _staging, sealed = _sealed(tmp_path)
    final = tmp_path / "published" / "same-fs"
    final.parent.mkdir()
    try:
        held = open_sealed_worker_artifact(sealed.seal_path)
        publication = verify_and_publish_sealed_attempt(
            held,
            final_path=final,
            verification={"status": "PASS", "method": "test"},
        )
        held.close()
        assert publication.publish_mode == "SAME_FILESYSTEM_ATOMIC_RENAME"
        assert final.is_dir()
        assert not sealed.staging_path.exists()
        assert (final / "PASS").read_text(encoding="utf-8") == (
            "[MANAGED_EXECUTION_V2_PASS]\n"
        )
        assert load_verified_gate(final)["status"] == "PASS"
    finally:
        _close_attempt(attempt)


def test_terminal_atomic_publish_cross_fs(tmp_path: Path) -> None:
    attempt, _staging, sealed = _sealed(tmp_path)
    final = tmp_path / "published" / "cross-fs"
    final.parent.mkdir()
    try:
        with open_sealed_worker_artifact(sealed.seal_path) as held:
            publication = verify_and_publish_sealed_attempt(
                held,
                final_path=final,
                verification={"status": "PASS", "method": "test"},
                force_cross_filesystem=True,
            )
        assert publication.publish_mode == (
            "CROSS_FILESYSTEM_COPY_REHASH_ATOMIC_RENAME"
        )
        assert sealed.staging_path.is_dir()
        assert (final / "artifacts" / "result.json").read_bytes() == (
            sealed.artifact_root / "result.json"
        ).read_bytes()
        assert load_verified_gate(final)["status"] == "PASS"
    finally:
        _close_attempt(attempt)


def test_terminal_rejects_symlink(tmp_path: Path) -> None:
    attempt = create_managed_attempt(
        stage_root=tmp_path,
        controller_id="controller-v2",
        task_id="task-v2",
        git_commit="1" * 40,
        config_hash="2" * 64,
        input_hashes={"input": "3" * 64},
        boot_id="boot-generation-1",
    )
    staging = create_worker_staging(attempt)
    raw = write_worker_raw_evidence(staging, {})
    raw.close()
    worker_exit = write_worker_exit(staging, {"exit_code": 0})
    worker_exit.close()
    (staging.artifact_root / "outside.txt").write_text("outside", encoding="utf-8")
    (staging.artifact_root / "bad-link").symlink_to(
        staging.artifact_root / "outside.txt"
    )
    try:
        with pytest.raises(TerminalPublisherV2Error, match="symlink"):
            seal_worker_staging(staging)
    finally:
        staging.close()
        attempt.close()


def test_terminal_rejects_inode_swap(tmp_path: Path) -> None:
    attempt, _staging, sealed = _sealed(tmp_path)
    artifact = sealed.artifact_root / "result.json"
    replacement = sealed.artifact_root / "replacement.json"
    replacement.write_bytes(artifact.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, artifact)
    try:
        with pytest.raises(TerminalPublisherV2Error, match="inventory|identity|file"):
            open_sealed_worker_artifact(sealed.seal_path)
    finally:
        _close_attempt(attempt)


def test_terminal_rejects_aba_directory(tmp_path: Path) -> None:
    attempt, _staging, sealed = _sealed(tmp_path)
    original = sealed.staging_path.with_name(sealed.staging_path.name + ".old")
    sealed.staging_path.rename(original)
    shutil.copytree(original, sealed.staging_path)
    try:
        with pytest.raises(TerminalPublisherV2Error, match="inode|ABA"):
            open_sealed_worker_artifact(sealed.seal_path)
    finally:
        _close_attempt(attempt)


def test_terminal_rejects_modified_after_seal(tmp_path: Path) -> None:
    attempt, _staging, sealed = _sealed(tmp_path)
    with (sealed.artifact_root / "result.json").open("a", encoding="utf-8") as handle:
        handle.write("modified\n")
    try:
        with pytest.raises(TerminalPublisherV2Error, match="inventory|file"):
            open_sealed_worker_artifact(sealed.seal_path)
    finally:
        _close_attempt(attempt)


def test_terminal_pass_written_only_by_verifier(tmp_path: Path) -> None:
    attempt, _staging, sealed = _sealed(tmp_path)
    assert not (sealed.staging_path / "PASS").exists()
    assert not (sealed.staging_path / "gate.json").exists()
    final = tmp_path / "published" / "verified"
    final.parent.mkdir()
    try:
        with open_sealed_worker_artifact(sealed.seal_path) as held:
            verify_and_publish_sealed_attempt(
                held,
                final_path=final,
                verification={"status": "PASS", "method": "test"},
            )
        assert (final / "PASS").is_file()
        assert (final / "gate.json").is_file()
        assert (final / "verification.json").is_file()
    finally:
        _close_attempt(attempt)


def test_terminal_fifty_round_concurrent_aba_race(tmp_path: Path) -> None:
    for round_index in range(50):
        root = tmp_path / f"race-{round_index}"
        root.mkdir()
        attempt, _staging, sealed = _sealed(root)
        artifact = sealed.artifact_root / "result.json"
        original = artifact.read_bytes()
        barrier = threading.Barrier(2)

        def swap() -> None:
            replacement = artifact.with_name("replacement.json")
            replacement.write_bytes(original)
            replacement.chmod(0o600)
            barrier.wait()
            os.replace(replacement, artifact)

        thread = threading.Thread(target=swap)
        thread.start()
        barrier.wait()
        held: HeldSealedArtifactV2 | None = None
        rejected = False
        try:
            try:
                held = open_sealed_worker_artifact(sealed.seal_path)
            except TerminalPublisherV2Error:
                rejected = True
            thread.join()
            if held is not None:
                with pytest.raises(TerminalPublisherV2Error):
                    held.revalidate()
                rejected = True
            assert rejected
        finally:
            if held is not None:
                held.close()
            if thread.is_alive():
                thread.join()
            _close_attempt(attempt)
