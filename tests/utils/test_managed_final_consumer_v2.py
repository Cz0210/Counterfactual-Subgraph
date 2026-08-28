from __future__ import annotations

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
            assert held.revalidate()["gate_sha256"] == publication.gate_sha256
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
