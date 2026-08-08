"""Validated, plan-only BBBP experiment DAGs.

The planner deliberately has no submit API.  It can render future
``scripts/exp_sbatch.sh`` commands, but rendering and execution are separate
operations so local validation cannot mutate the experiment registry or Slurm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.molecular_split import validate_repository_relative_path


CONFIG_PATHS = {
    "common4": "configs/experiments/bbbp_common4_v1.yaml",
    "cross-scaffold": "configs/experiments/bbbp_generalization_v1.yaml",
    "heldout": "configs/experiments/bbbp_generalization_v1.yaml",
    "candidate-source-ablation": "configs/experiments/bbbp_ablations_v1.yaml",
    "selector-ablation": "configs/experiments/bbbp_ablations_v1.yaml",
    "candidate-budget": "configs/experiments/bbbp_ablations_v1.yaml",
}
ALL_PLAN_NAMES = tuple(CONFIG_PATHS)


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - HPC has PyYAML.
            raise RuntimeError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable."
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"BBBP plan config must be a mapping: {path}")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class Stage:
    stage_id: str
    method: str
    wrapper: str
    depends_on: tuple[str, ...]
    output_root: str
    stage_type: str
    status: str
    resources: Mapping[str, Any]
    required_inputs: tuple[str, ...]
    expected_artifacts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Stage":
        required = {
            "id",
            "method",
            "wrapper",
            "depends_on",
            "output_root",
            "stage_type",
            "status",
            "resources",
            "required_inputs",
            "expected_artifacts",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"BBBP stage is missing fields {missing}: {payload}")
        return cls(
            stage_id=str(payload["id"]),
            method=str(payload["method"]),
            wrapper=validate_repository_relative_path(
                str(payload["wrapper"]), field="stage.wrapper"
            ),
            depends_on=tuple(str(value) for value in payload["depends_on"]),
            output_root=str(payload["output_root"]),
            stage_type=str(payload["stage_type"]),
            status=str(payload["status"]),
            resources=dict(payload["resources"]),
            required_inputs=tuple(str(value) for value in payload["required_inputs"]),
            expected_artifacts=tuple(str(value) for value in payload["expected_artifacts"]),
        )


@dataclass(frozen=True, slots=True)
class Plan:
    name: str
    source_path: Path
    dataset: str
    protocol: Mapping[str, Any]
    seeds: tuple[int, ...]
    stages: tuple[Stage, ...]

    def stage(self, stage_id: str) -> Stage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise ValueError(f"Unknown stage {stage_id!r} in BBBP plan {self.name!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "bbbp_validated_plan_v1",
            "plan": self.name,
            "dataset": self.dataset,
            "source_config": str(self.source_path),
            "protocol": dict(self.protocol),
            "seeds": list(self.seeds),
            "status": "FRAMEWORK_ONLY_NOT_RUN",
            "submission_performed": False,
            "registry_written": False,
            "formal_output_written": False,
            "stages": [
                {
                    "id": stage.stage_id,
                    "method": stage.method,
                    "wrapper": stage.wrapper,
                    "depends_on": list(stage.depends_on),
                    "output_root": stage.output_root,
                    "stage_type": stage.stage_type,
                    "status": stage.status,
                    "resources": dict(stage.resources),
                    "required_inputs": list(stage.required_inputs),
                    "expected_artifacts": list(stage.expected_artifacts),
                }
                for stage in self.stages
            ],
        }


def _validate_protocol(payload: Mapping[str, Any], *, source: Path) -> None:
    expected = {
        "dataset": "BBBP",
        "source_label": 1,
        "target_label": 0,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "threshold_source": "calibration",
        "selection_performed_in_eval": False,
        "threshold_fitted_on_test": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"BBBP protocol mismatch in {source}: {field}={payload.get(field)!r}, "
                f"expected={value!r}."
            )
    if payload.get("candidate_source_splits") != ["train", "val"]:
        raise ValueError("BBBP candidate discovery must remain train/val only.")
    if payload.get("selector_source_splits") != ["calibration"]:
        raise ValueError("BBBP selector tuning must remain calibration only.")
    if payload.get("test_usage") != "final_evaluation_only":
        raise ValueError("BBBP test split must remain final-evaluation only.")


def _validate_dag(stages: Sequence[Stage], *, project_root: Path) -> None:
    if not stages:
        raise ValueError("BBBP plan cannot have an empty stage DAG.")
    ids = [stage.stage_id for stage in stages]
    if len(set(ids)) != len(ids):
        raise ValueError(f"BBBP plan has duplicate stage IDs: {ids}")
    by_id = {stage.stage_id: stage for stage in stages}
    for stage in stages:
        unknown = sorted(set(stage.depends_on) - set(by_id))
        if unknown:
            raise ValueError(f"Stage {stage.stage_id} has unknown dependencies: {unknown}")
        wrapper = project_root / stage.wrapper
        if not wrapper.is_file():
            raise FileNotFoundError(f"BBBP stage wrapper is missing: {wrapper}")
        if stage.status != "NOT_RUN":
            raise ValueError(f"Framework stage status must be NOT_RUN: {stage.stage_id}")
        resources = dict(stage.resources)
        for field in (
            "wall_time_seconds",
            "gpu_time_seconds",
            "cpu_time_seconds",
            "peak_rss_mb",
            "peak_gpu_memory_mb",
            "num_parents",
            "num_candidates",
            "num_pairs",
            "cache_hit_rate",
            "candidates_per_second",
            "pairs_per_second",
            "successful_cf_per_gpu_hour",
        ):
            if field not in resources or resources[field] is not None:
                raise ValueError(
                    f"Unrun BBBP stage {stage.stage_id} must record {field}=null."
                )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visiting:
            raise ValueError(f"BBBP stage DAG contains a cycle at {stage_id}.")
        if stage_id in visited:
            return
        visiting.add(stage_id)
        for dependency in by_id[stage_id].depends_on:
            visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id in ids:
        visit(stage_id)


def load_plan(
    name: str,
    *,
    project_root: str | Path,
) -> Plan:
    if name not in ALL_PLAN_NAMES:
        raise ValueError(f"Unknown BBBP plan {name!r}; expected {ALL_PLAN_NAMES}.")
    root = Path(project_root).expanduser().resolve()
    config_path = root / CONFIG_PATHS[name]
    document = _load_document(config_path)
    if document.get("schema_version") != "bbbp_experiment_plans_v1":
        raise ValueError(f"Unsupported BBBP plan schema: {config_path}")
    protocol = document.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"BBBP config has no protocol mapping: {config_path}")
    _validate_protocol(protocol, source=config_path)
    plans = document.get("plans")
    if not isinstance(plans, dict) or name not in plans:
        raise ValueError(f"BBBP config does not define plan {name!r}: {config_path}")
    selected = plans[name]
    if not isinstance(selected, dict):
        raise ValueError(f"BBBP plan {name!r} must be a mapping.")
    resource_template = document.get("runtime_metrics_template")
    if not isinstance(resource_template, dict):
        raise ValueError(f"BBBP config has no runtime_metrics_template: {config_path}")
    stage_payloads: list[dict[str, Any]] = []
    for raw_stage in selected.get("stages", ()):
        if not isinstance(raw_stage, dict):
            raise ValueError(f"BBBP stage must be a mapping: {raw_stage!r}")
        stage_payload = dict(raw_stage)
        overrides = stage_payload.get("resources") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"BBBP stage resources must be a mapping: {raw_stage!r}")
        stage_payload["resources"] = {**resource_template, **overrides}
        stage_payloads.append(stage_payload)
    stages = tuple(Stage.from_mapping(value) for value in stage_payloads)
    _validate_dag(stages, project_root=root)
    seeds = tuple(int(value) for value in document.get("seeds", ()))
    if seeds != (0, 1, 2):
        raise ValueError(f"BBBP plans must freeze multi-seed protocol [0,1,2], got {seeds}.")
    return Plan(
        name=name,
        source_path=config_path,
        dataset=str(protocol["dataset"]),
        protocol=dict(protocol),
        seeds=seeds,
        stages=stages,
    )


def load_plans(selection: str, *, project_root: str | Path) -> tuple[Plan, ...]:
    names: Iterable[str] = ALL_PLAN_NAMES if selection == "all" else (selection,)
    return tuple(load_plan(name, project_root=project_root) for name in names)


def render_plan_text(plans: Sequence[Plan]) -> str:
    lines = ["BBBP FRAMEWORK PLAN", "status=FRAMEWORK_ONLY_NOT_RUN"]
    for plan in plans:
        lines.append(f"plan={plan.name} config={plan.source_path}")
        for stage in plan.stages:
            dependencies = ",".join(stage.depends_on) or "none"
            lines.append(
                f"  {stage.stage_id} <- {dependencies} "
                f"[{stage.method}; {stage.status}] {stage.wrapper}"
            )
    lines.extend(
        [
            "submission_performed=false",
            "registry_written=false",
            "formal_output_written=false",
        ]
    )
    return "\n".join(lines) + "\n"


def render_future_shell(plans: Sequence[Plan]) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "DO_NOT_EXECUTE_WITHOUT_SEPARATE_EXPERIMENT_AUTHORIZATION=1",
        "# Generated plan only. This file was not executed by the planner.",
    ]
    emitted: set[tuple[str, str]] = set()
    for plan in plans:
        lines.append(f"\n# plan={plan.name}")
        variables: dict[str, str] = {}
        for stage in plan.stages:
            key = (plan.name, stage.stage_id)
            if key in emitted:
                continue
            emitted.add(key)
            variable = stage.stage_id.upper().replace("-", "_") + "_JOB_ID"
            dependency_values = [variables[value] for value in stage.depends_on if value in variables]
            dependency = (
                f" --dependency=afterok:{':'.join('${' + value + '}' for value in dependency_values)}"
                if dependency_values
                else ""
            )
            lines.extend(
                [
                    f"# stage={stage.stage_id} status=NOT_RUN",
                    f"{variable}=$(scripts/exp_sbatch.sh \\",
                    f"  --name 'BBBP {plan.name} {stage.stage_id}' \\",
                    f"  --tags 'bbbp,{plan.name},{stage.method},{stage.stage_type}' \\",
                    "  --dataset BBBP \\",
                    f"  --method '{stage.method}' \\",
                    f"  --expected-output-root '{stage.output_root}'{dependency} \\",
                    f"  -- {stage.wrapper})",
                ]
            )
            variables[stage.stage_id] = variable
    return "\n".join(lines) + "\n"


def plans_json(plans: Sequence[Plan]) -> dict[str, Any]:
    return {
        "schema_version": "bbbp_plan_bundle_v1",
        "status": "FRAMEWORK_ONLY_NOT_RUN",
        "submission_performed": False,
        "registry_written": False,
        "formal_output_written": False,
        "plans": [plan.to_dict() for plan in plans],
    }


__all__ = [
    "ALL_PLAN_NAMES",
    "CONFIG_PATHS",
    "Plan",
    "Stage",
    "load_plan",
    "load_plans",
    "plans_json",
    "render_future_shell",
    "render_plan_text",
]
