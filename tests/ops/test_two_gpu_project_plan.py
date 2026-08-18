from __future__ import annotations

from scripts.ops.validate_two_gpu_project_plan import validate_project_plan


def _gpu(stage, lane, resource=(), data=(), method="ComRecGC"):
    return {
        "stage": stage,
        "dataset": "BACE" if lane == "bace" else "AIDS",
        "method": method,
        "gpus": 1,
        "gpu_lane": lane,
        "resource_dependencies": list(resource),
        "data_dependencies": list(data),
    }


def test_live_two_gpu_heads_and_serial_future_jobs_pass() -> None:
    plan = {
        "external_jobs": ["2282077", "2284866"],
        "existing_lane_heads": {"mut_aids": "2282077", "bace": "2284866"},
        "stages": [
            _gpu("aids_wnode", "mut_aids", resource=("afterany:2282077",)),
            _gpu("bace_cont1", "bace", resource=("afterany:2284866",)),
            _gpu("bace_cont2", "bace", resource=("afterany:bace_cont1",)),
            _gpu(
                "globalgce_wnode",
                "bace",
                resource=("afterany:bace_cont2",),
                method="GlobalGCE",
            ),
        ],
    }
    current = {
        "active_mut_aids_gpus": 1,
        "active_bace_gpus": 1,
        "active_project_gpus": 2,
    }
    result = validate_project_plan(plan, current)
    assert result["PLAN_VALID"] is True
    assert result["MAX_CONCURRENT_GPUS"] == 2
    assert result["NEW_GPU_JOB_STARTABLE_NOW"] == 0


def test_parallel_bace_gpu_or_two_gpu_job_fails() -> None:
    plan = {
        "stages": [
            _gpu("bace_a", "bace"),
            _gpu("bace_b", "bace"),
            {**_gpu("bad", "mut_aids"), "gpus": 2},
        ]
    }
    result = validate_project_plan(
        plan,
        {"active_mut_aids_gpus": 0, "active_bace_gpus": 0, "active_project_gpus": 0},
    )
    assert result["PLAN_VALID"] is False
    assert any(value.startswith("single_job_gpu_limit") for value in result["failures"])
