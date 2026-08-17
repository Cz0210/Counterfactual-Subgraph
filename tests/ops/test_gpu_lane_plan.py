from __future__ import annotations

from scripts.ops.audit_project_gpu_lanes import audit_records
from scripts.ops.validate_two_line_gpu_plan import validate_plan


def _stage(name, dataset, method, gpus, lane=None, data=(), resource=()):
    return {
        "stage": name,
        "dataset": dataset,
        "method": method,
        "resource_type": "gpu" if gpus else "cpu",
        "gpus": gpus,
        "gpu_lane": lane,
        "data_dependencies": list(data),
        "resource_dependencies": list(resource),
        "script": f"scripts/slurm/{name}.sh",
        "expected_output_root": f"outputs/{name}",
    }


def test_current_usage_separates_protected_gpu_from_project_lanes() -> None:
    records = [
        {
            "JobId": "2261163",
            "JobName": "long-dense-distill",
            "JobState": "RUNNING",
            "WorkDir": "/share/home/u20526/other",
            "Command": "/share/home/u20526/other/run.sh",
            "AllocTRES": "cpu=16,gres/gpu=2,gres/gpu:a800=2",
        },
        {
            "JobId": "3000001",
            "JobName": "mut_generation_retry8",
            "JobState": "RUNNING",
            "WorkDir": "/share/home/u20526/czx/worktrees/fix",
            "Command": "/share/home/u20526/czx/worktrees/fix/scripts/slurm/mut.sh",
            "AllocTRES": "cpu=7,gres/gpu=1,gres/gpu:a800=1",
        },
        {
            "JobId": "3000002",
            "JobName": "bace_eval",
            "JobState": "PENDING",
            "WorkDir": "/share/home/u20526/czx/counterfactual-subgraph",
            "Command": "/share/home/u20526/czx/counterfactual-subgraph/scripts/slurm/bace.sh",
            "ReqTRES": "cpu=7,gres/gpu:a800=1",
        },
    ]

    audit = audit_records(records)

    assert audit["active_mut_gpus"] == 1
    assert audit["active_bace_gpus"] == 0
    assert audit["protected_other_project_gpus"] == 2
    assert len(audit["pending_bace_gpu_requests"]) == 1


def test_valid_two_lane_plan_includes_cpu_stages_and_serial_gpu_stages() -> None:
    plan = {
        "stages": [
            _stage("mut_gen", "mutagenicity", "ComRecGC", 1, "mut"),
            _stage(
                "mut_integrity",
                "mutagenicity",
                "ComRecGC",
                0,
                data=("afterok:mut_gen",),
            ),
            _stage(
                "mut_eval",
                "mutagenicity",
                "ComRecGC",
                1,
                "mut",
                data=("afterok:mut_integrity",),
            ),
            _stage("bace_gcf", "bace", "GCFExplainer", 1, "bace"),
            _stage(
                "bace_lane_barrier",
                "bace",
                "lane-barrier",
                0,
                resource=("afterany:bace_gcf",),
            ),
            _stage(
                "bace_comrecgc",
                "bace",
                "ComRecGC",
                1,
                "bace",
                data=("afterok:bace_lane_barrier",),
            ),
            _stage("globalgce", "bace", "GlobalGCE", 0),
        ]
    }
    current = {"active_mut_gpus": 0, "active_bace_gpus": 0}

    audit = validate_plan(
        plan,
        current=current,
        mut_lane_limit=1,
        bace_lane_limit=1,
        total_limit=2,
    )

    assert audit["PLAN_VALID"] is True
    assert audit["CURRENT_OLD_JOBS_INCLUDED"] is True
    assert audit["MAX_MUT_PLUS_BACE_CONCURRENT_GPUS"] == 2


def test_parallel_or_two_gpu_stage_is_rejected() -> None:
    plan = {
        "stages": [
            _stage("bace_a", "bace", "GCFExplainer", 1, "bace"),
            _stage("bace_b", "bace", "ComRecGC", 1, "bace"),
            _stage("mut_bad", "mutagenicity", "ComRecGC", 2, "mut"),
        ]
    }
    audit = validate_plan(
        plan,
        current=None,
        mut_lane_limit=1,
        bace_lane_limit=1,
        total_limit=2,
    )
    assert audit["PLAN_VALID"] is False
    assert any(value.startswith("single_job_gpu_limit:mut_bad") for value in audit["failures"])
    assert any(value.startswith("parallel_gpu_stages_in_bace_lane") for value in audit["failures"])


def test_existing_lane_head_requires_resource_serialization() -> None:
    plan = {"stages": [_stage("bace_new", "bace", "ComRecGC", 1, "bace")]}
    audit = validate_plan(
        plan,
        current={"active_mut_gpus": 0, "active_bace_gpus": 1},
        mut_lane_limit=1,
        bace_lane_limit=1,
        total_limit=2,
    )
    assert audit["PLAN_VALID"] is False
    assert "existing_bace_lane_head_not_serialized" in audit["failures"]
