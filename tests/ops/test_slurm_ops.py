from __future__ import annotations

import pytest

from scripts.ops.slurm_ops import (
    SlurmSubmissionError,
    build_exp_sbatch_argv,
    dependency_argument,
    parse_exp_sbatch_job_id,
)


def stage(stage_id, kind, dependencies=(), script="scripts/slurm/job.sh"):
    return {
        "id": stage_id,
        "kind": kind,
        "dependencies": list(dependencies),
        "script": script,
        "command": [],
        "resources": {
            "name": stage_id,
            "tags": "automation",
            "notes": "",
            "expected_output_root": "outputs/run",
        },
    }


def test_exp_sbatch_job_id_protocol() -> None:
    assert (
        parse_exp_sbatch_job_id("[EXP_SUBMIT_OK]\njob_id=2018086\n")
        == "2018086"
    )
    with pytest.raises(SlurmSubmissionError):
        parse_exp_sbatch_job_id("Submitted batch job 123")


def test_audit_depends_afterany_on_compute() -> None:
    compute = stage("smoke", "slurm_job")
    audit = stage("smoke_audit", "audit", ["smoke"])
    mapping = {"smoke": compute, "smoke_audit": audit}
    assert dependency_argument(audit, mapping, {"smoke": "11"}) == (
        "--dependency=afterany:11"
    )


def test_downstream_compute_depends_afterok_on_audit() -> None:
    audit = stage("smoke_audit", "audit")
    full = stage("next_compute", "slurm_job", ["smoke_audit"])
    mapping = {"smoke_audit": audit, "next_compute": full}
    assert dependency_argument(full, mapping, {"smoke_audit": "12"}) == (
        "--dependency=afterok:12"
    )


def test_final_report_uses_afterany() -> None:
    audit = stage("full_audit", "audit")
    final = stage("final_report", "audit", ["full_audit"])
    mapping = {"full_audit": audit, "final_report": final}
    assert dependency_argument(final, mapping, {"full_audit": "13"}) == (
        "--dependency=afterany:13"
    )


def test_submission_always_uses_exp_sbatch_wrapper() -> None:
    compute = stage("smoke", "slurm_job")
    argv = build_exp_sbatch_argv(compute, {"smoke": compute}, {})
    assert argv[0] == "scripts/exp_sbatch.sh"
    assert "sbatch" not in argv
