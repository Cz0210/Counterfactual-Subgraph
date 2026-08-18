#!/usr/bin/env python3
"""Validate the live MUT/AIDs and BACE one-GPU lanes as one project plan."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


LANES = ("mut_aids", "bace")


def _upstream(value: str) -> str:
    return value.split(":", 1)[1] if ":" in value else ""


def validate_project_plan(
    plan: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    rows = list(plan.get("stages") or [])
    by_name = {str(row.get("stage") or ""): row for row in rows}
    failures: list[str] = []
    if "" in by_name or len(by_name) != len(rows):
        failures.append("stage_names_must_be_unique_and_nonempty")
    edges: dict[str, set[str]] = {name: set() for name in by_name}
    external = {str(value) for value in plan.get("external_jobs") or []}
    heads = {
        str(key): str(value)
        for key, value in (plan.get("existing_lane_heads") or {}).items()
    }
    for name, row in by_name.items():
        gpus = int(row.get("gpus") or 0)
        lane = str(row.get("gpu_lane") or "")
        method = str(row.get("method") or "").lower()
        stage_text = name.lower()
        if gpus > 1:
            failures.append(f"single_job_gpu_limit:{name}")
        if gpus and lane not in LANES:
            failures.append(f"gpu_stage_missing_lane:{name}")
        if lane in LANES and gpus != 1:
            failures.append(f"gpu_lane_stage_must_request_one:{name}")
        if "globalgce" in method and gpus and "wnode" not in stage_text:
            failures.append(f"globalgce_non_wnode_gpu_forbidden:{name}")
        for dependency in row.get("data_dependencies") or []:
            dependency = str(dependency)
            if not dependency.startswith("afterok:"):
                failures.append(f"scientific_dependency_not_afterok:{name}:{dependency}")
            upstream = _upstream(dependency)
            if upstream in by_name:
                edges[name].add(upstream)
            elif upstream and upstream not in external:
                failures.append(f"unknown_scientific_dependency:{name}:{upstream}")
        for dependency in row.get("resource_dependencies") or []:
            dependency = str(dependency)
            if not dependency.startswith("afterany:"):
                failures.append(f"resource_dependency_not_afterany:{name}:{dependency}")
            upstream = _upstream(dependency)
            if upstream in by_name:
                edges[name].add(upstream)
            elif upstream and upstream not in external and upstream not in heads.values():
                failures.append(f"unknown_resource_dependency:{name}:{upstream}")

    ancestor_cache: dict[str, set[str]] = {}

    def ancestors(name: str, visiting: set[str] | None = None) -> set[str]:
        if name in ancestor_cache:
            return ancestor_cache[name]
        visiting = set(visiting or ())
        if name in visiting:
            failures.append(f"dependency_cycle:{name}")
            return set()
        visiting.add(name)
        result = set(edges.get(name, set()))
        for parent in tuple(result):
            result.update(ancestors(parent, visiting))
        ancestor_cache[name] = result
        return result

    for lane in LANES:
        gpu = [
            name
            for name, row in by_name.items()
            if row.get("gpu_lane") == lane and int(row.get("gpus") or 0) == 1
        ]
        for index, left in enumerate(gpu):
            for right in gpu[index + 1 :]:
                if left not in ancestors(right) and right not in ancestors(left):
                    failures.append(f"parallel_gpu_stages:{lane}:{left}:{right}")
        head = heads.get(lane)
        if head and gpu:
            first = gpu[0]
            dependencies = {
                str(value) for value in by_name[first].get("resource_dependencies") or []
            }
            if f"afterany:{head}" not in dependencies:
                failures.append(f"existing_lane_head_not_serialized:{lane}:{first}")

    existing_mut_aids = int(current.get("active_mut_aids_gpus") or 0)
    existing_bace = int(current.get("active_bace_gpus") or 0)
    existing_total = int(current.get("active_project_gpus") or 0)
    if existing_mut_aids > 1:
        failures.append("existing_mut_aids_lane_limit")
    if existing_bace > 1:
        failures.append("existing_bace_lane_limit")
    if existing_total > 2:
        failures.append("existing_project_gpu_limit")
    planned_mut_aids = int(
        any(row.get("gpu_lane") == "mut_aids" and int(row.get("gpus") or 0) for row in rows)
    )
    planned_bace = int(
        any(row.get("gpu_lane") == "bace" and int(row.get("gpus") or 0) for row in rows)
    )
    max_mut_aids = max(existing_mut_aids, planned_mut_aids)
    max_bace = max(existing_bace, planned_bace)
    max_total = max_mut_aids + max_bace
    if max_mut_aids > 1 or max_bace > 1 or max_total > 2:
        failures.append("planned_project_gpu_limit")
    return {
        "schema_version": "two_gpu_project_plan_v7",
        "PLAN_VALID": not failures,
        "failures": sorted(set(failures)),
        "CURRENT_JOBS_INCLUDED": True,
        "MAX_CONCURRENT_GPUS": max_total,
        "MAX_SINGLE_JOB_GPUS": max((int(row.get("gpus") or 0) for row in rows), default=0),
        "MUT_AIDS_LANE_MAX": max_mut_aids,
        "BACE_LANE_MAX": max_bace,
        "ACTIVE_PROJECT_GPUS": existing_total,
        "NEW_GPU_JOB_STARTABLE_NOW": max(0, 2 - existing_total),
        "planned_stage_count": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--current-usage-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plan_path = Path(args.plan_json).expanduser().resolve()
    current_path = Path(args.current_usage_json).expanduser().resolve()
    result = validate_project_plan(
        json.loads(plan_path.read_text(encoding="utf-8")),
        json.loads(current_path.read_text(encoding="utf-8")),
    )
    output = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else plan_path.parent / "audit.json"
    )
    if not args.validate_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["PLAN_VALID"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
