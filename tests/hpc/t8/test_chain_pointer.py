from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from scripts.hpc.t8 import run_stress_followup as followup
from src.utils.hpc_t8_chain_pointer import (
    T8ChainPointerError,
    canonical_sha256,
    chain_lock,
    load_current_pointer,
    parse_array_sacct,
    reconcile_full_chain_pointer,
    sha256_file,
    write_current_pointer,
)


def _refinement_payload(
    *, canary: str = "101", followup_job: str = "102", depth: int = 5
) -> dict[str, object]:
    return {
        "schema_version": "t8_hpc_current_chain_v1",
        "state": "REFINEMENT_CANARY_SUBMITTED",
        "active_stage": "REFINEMENT_CANARY",
        "updated_at": "2026-09-04T00:00:00+00:00",
        "output_root": "/runtime/continuation/control",
        "continuation_root": "/runtime/continuation/artifacts",
        "decision_root": "/runtime/continuation/control/upstream-100",
        "upstream_job_id": "100",
        "controller_commit": "a" * 40,
        "science_commit": "b" * 40,
        "refinement_depth": depth,
        "canary_job_id": canary,
        "followup_job_id": followup_job,
        "followup_dependency": f"afterany:{canary}",
        "canary_root": "/runtime/continuation/artifacts/canary",
        "matrix_write_enabled": False,
        "gpu_requested": False,
    }


def test_current_pointer_is_atomic_self_hashed_and_monotonic(tmp_path: Path) -> None:
    path = tmp_path / "control" / "current.json"
    first = write_current_pointer(path, _refinement_payload())

    assert load_current_pointer(path) == first
    assert path.with_suffix(".json.sha256").is_file()
    assert not list(path.parent.glob("*.tmp"))

    second = write_current_pointer(
        path, _refinement_payload(canary="201", followup_job="202", depth=6)
    )
    assert second["previous_current_sha256"] == first["current_sha256"]
    assert load_current_pointer(path) == second

    stale = write_current_pointer(path, _refinement_payload(depth=5))
    assert stale == second
    assert load_current_pointer(path) == second


def test_one_canary_cannot_be_bound_to_two_followups(tmp_path: Path) -> None:
    path = tmp_path / "current.json"
    write_current_pointer(path, _refinement_payload())
    with pytest.raises(T8ChainPointerError, match="two follow-up"):
        write_current_pointer(path, _refinement_payload(followup_job="999"))


def test_chain_lock_serializes_the_stable_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "chain" / "current.json"
    with chain_lock(pointer):
        with (pointer.parent / ".chain.lock").open("a+b") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_submitted_refinement_publishes_dynamic_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "runtime" / "control" / "current.json"
    args = type(
        "Args",
        (),
        {
            "submit": True,
            "current_pointer": pointer,
            "output_root": tmp_path / "decision",
            "continuation_root": tmp_path / "artifacts",
            "expected_controller_commit": "a" * 40,
            "expected_science_commit": "b" * 40,
        },
    )()
    followup._publish_submitted_chain_pointer(
        args,
        {"job_id": "100"},
        tmp_path / "decision" / "upstream-100",
        {
            "state": "REFINEMENT_CANARY_SUBMITTED",
            "refinement_level": 3,
            "refinement_canary_job_id": "201",
            "afterany_followup_job_id": "202",
            "fresh_canary_root": str(tmp_path / "artifacts" / "depth-6"),
        },
    )
    current = load_current_pointer(pointer)
    assert current["refinement_depth"] == 6
    assert current["canary_job_id"] == "201"
    assert current["followup_job_id"] == "202"
    assert current["followup_dependency"] == "afterany:201"


def test_refinement_limit_is_exact_depth_eight() -> None:
    assert followup.INITIAL_STRESS_DEPTH == 3
    assert followup.MAX_REFINEMENT_DEPTH == 8
    assert followup.MAX_REFINEMENT_LEVELS == 5


def _write_receipt(path: Path, payload: dict[str, object], hash_field: str) -> None:
    body = dict(payload)
    body[hash_field] = canonical_sha256(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
    )
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n")


def test_parse_array_sacct_expands_pending_range() -> None:
    status = parse_array_sacct(
        "\n".join(
            [
                "2536781_0|COMPLETED|0:0",
                "2536781_1|RUNNING|0:0",
                "2536781_[2-3%2]|PENDING|0:0",
                "2536781_0.batch|COMPLETED|0:0",
            ]
        ),
        array_job_id="2536781",
        array_range="0-3",
    )
    assert status["completed_shard_ids"] == [0]
    assert status["running_shard_ids"] == [1]
    assert status["pending_shard_ids"] == [2, 3]
    assert status["failed_shard_ids"] == []


def test_reconcile_full_chain_pointer_uses_receipts_and_live_state(tmp_path: Path) -> None:
    decision = tmp_path / "continuation" / "control" / "upstream-2536771"
    full_root = tmp_path / "continuation" / "artifacts" / "full"
    manifest = full_root / "partition_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    admission_path = decision / "admission_receipt.json"
    inventory_path = decision / "slurm_inventory.json"
    submission_path = decision / "submission_receipt.json"
    _write_receipt(
        admission_path,
        {
            "schema_version": "t8_hpc_full_admission_v1",
            "state": "PASS",
            "admission_pass": True,
            "full_manifest_path": str(manifest),
            "full_manifest_sha256": "a" * 64,
            "full_manifest_file_sha256": sha256_file(manifest),
        },
        "admission_receipt_sha256",
    )
    _write_receipt(
        inventory_path,
        {
            "schema_version": "t8_hpc_slurm_inventory_v1",
            "state": "PASS",
            "array_job_id": "2536781",
            "merge_job_id": "2536786",
            "package_job_id": "2536787",
            "array_range": "0-1",
            "dependency_chain": [
                {"dependency": "afterok:2536781", "job_id": "2536786"},
                {"dependency": "afterok:2536786", "job_id": "2536787"},
            ],
        },
        "slurm_inventory_sha256",
    )
    _write_receipt(
        submission_path,
        {
            "schema_version": "t8_hpc_stress_followup_v1",
            "state": "FULL_CHAIN_SUBMITTED",
            "action": "SUBMIT_ARRAY_AFTEROK_MERGE_AFTEROK_PACKAGE",
            "array_job_id": "2536781",
            "merge_job_id": "2536786",
            "package_job_id": "2536787",
            "controller_commit": "a" * 40,
            "science_commit": "b" * 40,
            "full_root": str(full_root),
            "full_manifest": str(manifest),
            "upstream_terminal": {"job_id": "2536771"},
            "array_sbatch_argv": [
                "sbatch",
                "--export",
                "ALL,T8_EXPECTED_COMMIT="
                + "b" * 40
                + ",T8_EXPECTED_INPUT_MANIFEST_SHA256="
                + "c" * 64
                + ",T8_CANARY_PARITY_RECEIPT=/parity.json",
            ],
        },
        "submission_receipt_sha256",
    )
    pointer = tmp_path / "runtime" / "control" / "t8-production-chain" / "current.json"
    result = reconcile_full_chain_pointer(
        current_pointer=pointer,
        submission_receipt=submission_path,
        slurm_inventory=inventory_path,
        admission_receipt=admission_path,
        sacct_text="2536781_0|COMPLETED|0:0\n2536781_1|RUNNING|0:0\n",
        updated_at="2026-09-04T04:00:00+00:00",
    )
    assert result["state"] == "FULL_CHAIN_RUNNING"
    assert result["completed_shards"] == 1
    assert result["running_shards"] == 1
    assert result["merge_dependency"] == "afterok:2536781"
    assert load_current_pointer(pointer) == result

    advanced = reconcile_full_chain_pointer(
        current_pointer=pointer,
        submission_receipt=submission_path,
        slurm_inventory=inventory_path,
        admission_receipt=admission_path,
        sacct_text="2536781_0|COMPLETED|0:0\n2536781_1|COMPLETED|0:0\n",
        updated_at="2026-09-04T05:00:00+00:00",
    )
    assert advanced["state"] == "FULL_CHAIN_ARRAY_PASS"
    assert advanced["completed_shards"] == 2
    assert advanced["previous_current_sha256"] == result["current_sha256"]

    with pytest.raises(T8ChainPointerError, match="completed.*regressed"):
        reconcile_full_chain_pointer(
            current_pointer=pointer,
            submission_receipt=submission_path,
            slurm_inventory=inventory_path,
            admission_receipt=admission_path,
            sacct_text=(
                "2536781_0|COMPLETED|0:0\n2536781_1|RUNNING|0:0\n"
            ),
            updated_at="2026-09-04T05:01:00+00:00",
        )
