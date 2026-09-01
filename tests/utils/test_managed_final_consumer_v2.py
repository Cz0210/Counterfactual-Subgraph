from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.managed_final_consumer_v2 import (
    ManagedFinalConsumerV2Error,
    hold_verified_managed_final,
)
from src.utils.terminal_publisher_v2 import (
    DirectoryEvidenceV2,
    FileEvidenceV2,
    TerminalPublisherV2Error,
    open_sealed_worker_artifact,
    seal_worker_staging,
    verify_and_publish_sealed_attempt,
)


def _published_final(
    tmp_path: Path,
    *,
    cross_filesystem: bool = False,
    extra_checkpoint_files: int = 0,
):
    (tmp_path / "stages").mkdir()
    attempt = create_managed_attempt(
        stage_root=tmp_path / "stages",
        controller_id="taste-main-v2",
        task_id="T3_CALIBRATION",
        git_commit="1" * 40,
        config_hash="2" * 64,
        input_hashes={"t2_receipt_gate": "3" * 64},
        boot_id="boot-generation-1",
    )
    staging = create_worker_staging(attempt)
    checkpoint = staging.artifact_root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "split_manifest.json").write_text(
        '{"dataset":"tastemolnet"}\n', encoding="utf-8"
    )
    for index in range(extra_checkpoint_files):
        (checkpoint / f"checkpoint-{index:04d}.json").write_text(
            f'{{"index":{index}}}\n', encoding="utf-8"
        )
    raw = write_worker_raw_evidence(staging, {"stage": "T3_CALIBRATION"})
    raw.close()
    worker_exit = write_worker_exit(
        staging, {"exit_code": 0, "worker_closed_artifact_writers": True}
    )
    worker_exit.close()
    sealed = seal_worker_staging(staging)
    staging.close()
    final = tmp_path / "published" / "calibrated-fixture"
    final.parent.mkdir()
    with open_sealed_worker_artifact(sealed.seal_path) as held:
        publication = verify_and_publish_sealed_attempt(
            held,
            final_path=final,
            verification={
                "schema_version": "tastemolnet_t3_calibration_v2",
                "status": "PASS",
                "stage": "T3_GINE_CALIBRATED",
            },
            force_cross_filesystem=cross_filesystem,
        )
    return attempt, final, publication


@pytest.mark.parametrize("cross_filesystem", [False, True])
def test_holds_actual_published_final_with_verifier_trio(
    tmp_path: Path, cross_filesystem: bool
) -> None:
    attempt, final, publication = _published_final(
        tmp_path, cross_filesystem=cross_filesystem
    )
    try:
        assert {path.name for path in final.iterdir()} == {
            ".generation_token.json",
            "artifacts",
            "raw_evidence.json",
            "worker_exit.json",
            "SEALED.json",
            "verification.json",
            "gate.json",
            "PASS",
        }
        with pytest.raises(
            TerminalPublisherV2Error, match="verifier-only|staging inode"
        ):
            open_sealed_worker_artifact(final)
        with hold_verified_managed_final(
            final,
            required_relative_paths=(
                "artifacts/checkpoint/split_manifest.json",
            ),
        ) as held:
            reopened = held.revalidate()
            assert reopened["gate_sha256"] == publication.gate_sha256
            assert held.mount_session_remount_detected is False
            assert held.mount_session_device_drift == ()
            assert held.file("artifacts/checkpoint/split_manifest.json").read_bytes()
            assert held.file("PASS").read_bytes() == b"[MANAGED_EXECUTION_V2_PASS]\n"
    finally:
        attempt.close()


def test_published_final_rejects_uninventoried_file(tmp_path: Path) -> None:
    attempt, final, _publication = _published_final(tmp_path)
    try:
        with hold_verified_managed_final(final) as held:
            (final / "worker-authored-second-pass.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with pytest.raises(ManagedFinalConsumerV2Error, match="exact inventory"):
                held.revalidate()
    finally:
        attempt.close()


def test_published_final_does_not_hold_one_ancestor_chain_per_checkpoint(
    tmp_path: Path,
) -> None:
    attempt, final, _publication = _published_final(
        tmp_path, extra_checkpoint_files=250
    )
    descriptor_root = Path("/dev/fd")
    before = len(list(descriptor_root.iterdir()))
    try:
        with hold_verified_managed_final(final) as held:
            during = len(list(descriptor_root.iterdir()))
            assert during - before < 128
            assert held.revalidate()["attempt_id"]
    finally:
        attempt.close()


def test_published_final_accepts_and_reports_mount_session_device_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt, final, _publication = _published_final(tmp_path)
    original_file_from_mapping = FileEvidenceV2.from_mapping
    original_directory_from_mapping = DirectoryEvidenceV2.from_mapping

    def remounted_file(raw):
        evidence = original_file_from_mapping(raw)
        return replace(evidence, st_dev=evidence.st_dev + 50)

    def remounted_directory(raw):
        evidence = original_directory_from_mapping(raw)
        return replace(evidence, st_dev=evidence.st_dev + 50)

    monkeypatch.setattr(
        FileEvidenceV2,
        "from_mapping",
        classmethod(lambda _cls, raw: remounted_file(raw)),
    )
    monkeypatch.setattr(
        DirectoryEvidenceV2,
        "from_mapping",
        classmethod(lambda _cls, raw: remounted_directory(raw)),
    )
    try:
        with hold_verified_managed_final(final) as held:
            stable_evidence = held.revalidate()
            assert set(stable_evidence) == {
                "attempt_id",
                "generation_token",
                "seal_sha256",
                "source_inventory_sha256",
                "published_inventory_sha256",
                "verification_sha256",
                "gate_sha256",
                "pass_sha256",
            }
            assert held.mount_session_remount_detected is True
            drift = held.mount_session_device_drift
            assert drift
            assert {item["kind"] for item in drift} == {"file", "directory"}
            assert all(
                item["sealed_st_dev"] == item["mount_session_st_dev"] + 50
                for item in drift
            )
    finally:
        attempt.close()


def test_published_final_rejects_inode_swap_with_same_bytes_and_metadata(
    tmp_path: Path,
) -> None:
    attempt, final, _publication = _published_final(tmp_path)
    target = final / "artifacts/checkpoint/split_manifest.json"
    original = target.stat()
    replacement = target.with_name("replacement.json")
    replacement.write_bytes(target.read_bytes())
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(target)
    assert target.stat().st_ino != original.st_ino
    assert target.stat().st_size == original.st_size
    assert target.stat().st_mtime_ns == original.st_mtime_ns
    try:
        with pytest.raises(
            ManagedFinalConsumerV2Error, match="published file identity changed"
        ):
            hold_verified_managed_final(final)
    finally:
        attempt.close()


def test_published_final_rejects_same_identity_byte_tamper(tmp_path: Path) -> None:
    attempt, final, _publication = _published_final(tmp_path)
    target = final / "artifacts/checkpoint/split_manifest.json"
    original = target.stat()
    contents = bytearray(target.read_bytes())
    contents[-2] ^= 1
    target.write_bytes(contents)
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    current = target.stat()
    assert current.st_ino == original.st_ino
    assert current.st_size == original.st_size
    assert current.st_mtime_ns == original.st_mtime_ns
    try:
        with pytest.raises(
            ManagedFinalConsumerV2Error, match="published file identity changed"
        ):
            hold_verified_managed_final(final)
    finally:
        attempt.close()


def test_published_final_rejects_symlink_even_when_target_bytes_match(
    tmp_path: Path,
) -> None:
    attempt, final, _publication = _published_final(tmp_path)
    target = final / "artifacts/checkpoint/split_manifest.json"
    replacement = target.with_name("replacement.json")
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(replacement.name)
    try:
        with pytest.raises((ManagedFinalConsumerV2Error, OSError)):
            hold_verified_managed_final(final)
    finally:
        attempt.close()
