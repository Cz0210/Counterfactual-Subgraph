#!/usr/bin/env python3
"""Fail-closed static validation for one-GPU MUT and BACE Slurm lanes."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


PROTECTED_JOB_IDS = {
    "2261163",
    "2261164",
    "2227575",
    "2227578",
    "2227584",
    "2227585",
    "2227586",
    "2227587",
    "2227588",
    "2227589",
    "2227590",
    "2227591",
    "2227592",
    "2229392",
    "2229482",
}


def _dependency_stage(value: str) -> str:
    return value.split(":", 1)[1] if ":" in value else ""


def validate_plan(
    plan: dict[str, Any],
    *,
    current: dict[str, Any] | None,
    mut_lane_limit: int,
    bace_lane_limit: int,
    total_limit: int,
) -> dict[str, Any]:
    current_included = current is not None
    stages = list(plan.get("stages") or [])
    by_name = {str(row.get("stage")): row for row in stages}
    failures: list[str] = []
    if len(by_name) != len(stages) or "" in by_name:
        failures.append("stage_names_must_be_unique_and_nonempty")
    edges: dict[str, set[str]] = {name: set() for name in by_name}
    for name, row in by_name.items():
        dataset = str(row.get("dataset") or "").lower()
        method = str(row.get("method") or "").lower()
        gpus = int(row.get("gpus") or 0)
        lane = row.get("gpu_lane")
        if gpus > 1:
            failures.append(f"single_job_gpu_limit:{name}")
        if dataset == "aids" and gpus:
            failures.append(f"aids_gpu_forbidden:{name}")
        if "globalgce" in method and gpus:
            failures.append(f"globalgce_gpu_forbidden:{name}")
        if lane == "mut" and gpus != 1:
            failures.append(f"mut_gpu_stage_must_request_one:{name}")
        if lane == "bace" and gpus != 1:
            failures.append(f"bace_gpu_stage_must_request_one:{name}")
        if gpus and lane not in {"mut", "bace"}:
            failures.append(f"gpu_stage_missing_lane:{name}")
        for dependency in row.get("data_dependencies") or []:
            dependency = str(dependency)
            if not dependency.startswith("afterok:"):
                failures.append(f"scientific_dependency_not_afterok:{name}:{dependency}")
            upstream = _dependency_stage(dependency)
            if upstream in by_name:
                edges[name].add(upstream)
            elif upstream and upstream not in {str(value) for value in plan.get("external_completed_jobs", [])}:
                failures.append(f"unknown_scientific_dependency:{name}:{upstream}")
        for dependency in row.get("resource_dependencies") or []:
            dependency = str(dependency)
            if not dependency.startswith("afterany:"):
                failures.append(f"resource_dependency_not_afterany:{name}:{dependency}")
            upstream = _dependency_stage(dependency)
            if upstream in by_name:
                edges[name].add(upstream)
        dependency_text = json.dumps(
            [row.get("data_dependencies"), row.get("resource_dependencies")]
        )
        if any(job_id in dependency_text for job_id in PROTECTED_JOB_IDS):
            failures.append(f"protected_job_dependency:{name}")

    def ancestors(name: str, stack: set[str] | None = None) -> set[str]:
        stack = set(stack or ())
        if name in stack:
            failures.append(f"dependency_cycle:{name}")
            return set()
        stack.add(name)
        result = set(edges.get(name, set()))
        for parent in tuple(result):
            result.update(ancestors(parent, stack))
        return result

    for lane in ("mut", "bace"):
        gpu_stages = [
            name
            for name, row in by_name.items()
            if row.get("gpu_lane") == lane and int(row.get("gpus") or 0) == 1
        ]
        for index, left in enumerate(gpu_stages):
            for right in gpu_stages[index + 1 :]:
                if left not in ancestors(right) and right not in ancestors(left):
                    failures.append(f"parallel_gpu_stages_in_{lane}_lane:{left}:{right}")

    current = current or {}
    existing_mut = int(current.get("active_mut_gpus") or 0)
    existing_bace = int(current.get("active_bace_gpus") or 0)
    if existing_mut > mut_lane_limit:
        failures.append("existing_mut_gpu_limit")
    if existing_bace > bace_lane_limit:
        failures.append("existing_bace_gpu_limit")
    if existing_mut + existing_bace > total_limit:
        failures.append("existing_total_gpu_limit")
    for lane, existing in (("mut", existing_mut), ("bace", existing_bace)):
        if not existing:
            continue
        lane_head = str(plan.get("existing_lane_heads", {}).get(lane) or "")
        first_gpu = next(
            (
                row
                for row in stages
                if row.get("gpu_lane") == lane and int(row.get("gpus") or 0) == 1
            ),
            None,
        )
        dependencies = [str(value) for value in (first_gpu or {}).get("resource_dependencies") or []]
        if not lane_head or f"afterany:{lane_head}" not in dependencies:
            failures.append(f"existing_{lane}_lane_head_not_serialized")

    planned_mut = any(
        row.get("gpu_lane") == "mut" and int(row.get("gpus") or 0) == 1
        for row in stages
    )
    planned_bace = any(
        row.get("gpu_lane") == "bace" and int(row.get("gpus") or 0) == 1
        for row in stages
    )
    max_mut = max(existing_mut, int(planned_mut))
    max_bace = max(existing_bace, int(planned_bace))
    max_total = max_mut + max_bace
    if max_mut > mut_lane_limit:
        failures.append("planned_mut_gpu_limit")
    if max_bace > bace_lane_limit:
        failures.append("planned_bace_gpu_limit")
    if max_total > total_limit:
        failures.append("planned_total_gpu_limit")

    result = {
        "schema_version": "two_line_gpu_budget_plan_audit_v2",
        "PLAN_VALID": not failures,
        "failures": sorted(set(failures)),
        "MUT_MAX_CONCURRENT_GPUS": max_mut,
        "BACE_MAX_CONCURRENT_GPUS": max_bace,
        "TOTAL_MAX_CONCURRENT_GPUS": max_total,
        "SINGLE_JOB_MAX_GPUS": max((int(row.get("gpus") or 0) for row in stages), default=0),
        "MAX_MUT_CONCURRENT_GPUS": max_mut,
        "MAX_BACE_CONCURRENT_GPUS": max_bace,
        "MAX_MUT_PLUS_BACE_CONCURRENT_GPUS": max_total,
        "CURRENT_OLD_JOBS_INCLUDED": current_included,
        "existing_active_mut_gpus": existing_mut,
        "existing_active_bace_gpus": existing_bace,
        "planned_stage_count": len(stages),
    }
    return result


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--mut-lane-limit", type=int, default=1)
    parser.add_argument("--bace-lane-limit", type=int, default=1)
    parser.add_argument("--total-limit", type=int, default=2)
    parser.add_argument("--include-current-squeue", action="store_true")
    parser.add_argument("--current-usage-json")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    plan_path = Path(args.plan_json).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    current = None
    if args.include_current_squeue:
        if not args.current_usage_json:
            parser.error("--include-current-squeue requires --current-usage-json")
        current = json.loads(
            Path(args.current_usage_json).expanduser().resolve().read_text(encoding="utf-8")
        )
    result = validate_plan(
        plan,
        current=current,
        mut_lane_limit=args.mut_lane_limit,
        bace_lane_limit=args.bace_lane_limit,
        total_limit=args.total_limit,
    )
    output = Path(args.output_json).expanduser().resolve() if args.output_json else plan_path.parent / "gpu_budget_plan_audit.json"
    _write(output, result)
    for key in (
        "PLAN_VALID",
        "SINGLE_JOB_MAX_GPUS",
        "MAX_MUT_CONCURRENT_GPUS",
        "MAX_BACE_CONCURRENT_GPUS",
        "MAX_MUT_PLUS_BACE_CONCURRENT_GPUS",
        "CURRENT_OLD_JOBS_INCLUDED",
    ):
        print(f"{key}={str(result[key]).lower() if isinstance(result[key], bool) else result[key]}")
    if result["failures"]:
        print("FAILURES=" + ",".join(result["failures"]))
    return 0 if result["PLAN_VALID"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
