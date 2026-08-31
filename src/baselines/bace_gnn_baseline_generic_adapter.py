"""Adapt native BACE baseline task fragments to the generic AutoDL controller.

The native fragment is intentionally method-facing: it records ``argv``,
``output_root`` and rich resource metadata.  The persistent four-GPU
controller has a different retry and dependency contract.  This module is the
single, fail-closed translation boundary between those two schemas; the native
fragment remains unchanged and useful for direct/manual inspection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from src.baselines.bace_gnn_baseline_contracts import baseline_spec
from src.baselines.bace_gnn_baseline_tasks import (
    build_bace_baseline_controller_fragment,
)


GENERIC_FRAGMENT_SCHEMA = "bace_baseline_generic_controller_fragment_v1"
B11_SHARD_PRIORITY = 90


def _dependency_token(task_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in task_id
    )
    return "{dep_" + safe + "_output}"


def _replace_path_prefix(value: str, prefix: str, replacement: str) -> str | None:
    if value == prefix:
        return replacement
    normalized = prefix.rstrip("/")
    if value.startswith(normalized + "/"):
        return replacement.rstrip("/") + value[len(normalized) :]
    return None


def _native_output_map(tasks: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        output = str(task.get("output_root") or "")
        if not task_id or not output or not Path(output).is_absolute():
            raise ValueError(f"Native task has no absolute output contract: {task_id!r}")
        if task_id in result:
            raise ValueError(f"Native fragment repeats task id: {task_id}")
        result[task_id] = output
    return result


def _rewrite_argument(
    value: str,
    *,
    task_id: str,
    dependencies: Sequence[str],
    outputs: Mapping[str, str],
    native_root: str,
) -> str:
    candidates = [
        (outputs[dependency], _dependency_token(dependency))
        for dependency in dependencies
    ]
    candidates.append((outputs[task_id], "{task_output}"))
    for prefix, replacement in sorted(
        candidates, key=lambda item: len(item[0]), reverse=True
    ):
        rewritten = _replace_path_prefix(value, prefix, replacement)
        if rewritten is not None:
            return rewritten

    # Native COMRECGC checkpoint/trace/cache paths are siblings of its primary
    # output.  They must not remain shared across retries.  Fold any such
    # method-root path into this task's immutable attempt directory.
    rewritten = _replace_path_prefix(value, native_root, "{task_output}/_native_aux")
    if rewritten is not None:
        return rewritten
    return value


def _required_output_files(task_id: str) -> list[str]:
    if task_id.endswith("_preflight"):
        files = [
            "route_contract.json",
            "oracle_provenance.json",
            "state.json",
            "READY",
        ]
        if task_id.startswith("bace_globalgce_"):
            files.extend(
                [
                    "official_source_audit.json",
                    "official_tensor_parity.json",
                    "NATIVE_ACTION_READY",
                ]
            )
        return files
    if task_id.endswith("_train_vrrw"):
        return ["run_manifest.json", "counterfactuals.pt", "_RUN_COMPLETE.json"]
    if task_id.endswith("_bridge_smoke"):
        return [
            "bridge_gradient_audit.json",
            "oracle_provenance.json",
            "run_manifest.json",
            "state.json",
            "PASS",
            "BRIDGE_PASS",
        ]
    if task_id.endswith("_train_summary"):
        return [
            "run_manifest.json",
            "native_summary_rank.jsonl",
            "_RUN_COMPLETE.json",
        ]
    if task_id.endswith("_train_generation"):
        return ["run_manifest.json", "counterfactuals.pt", "_RUN_COMPLETE.json"]
    if task_id.endswith("_train_common_recourse"):
        return [
            "run_manifest.json",
            "selected_common_recourses.json",
            "representative_counterfactuals.pt",
            "_RUN_COMPLETE.json",
        ]
    if task_id.endswith("_train_candidates"):
        return [
            "candidate_universe.jsonl",
            "candidate_filter_audit.jsonl",
            "oracle_provenance.json",
            "run_manifest.json",
            "PASS",
        ]
    if "_calibration_shard_" in task_id or "_test_shard_" in task_id:
        return [
            "pair_details.jsonl",
            "pair_details.csv",
            "oracle_provenance.json",
            "run_manifest.json",
            "PASS",
        ]
    if task_id.endswith("_calibration_merge") or task_id.endswith("_test_merge"):
        return [
            "pair_matrix.jsonl",
            "selected_candidate_universe.jsonl",
            "summary.json",
            "run_manifest.json",
            "PASS",
        ]
    if task_id.endswith("_selection"):
        return [
            "selected_top20.json",
            "frozen_selection_manifest.json",
            "thresholds.json",
            "PASS",
        ]
    if task_id.endswith("_final_freeze"):
        return [
            "final_metrics.json",
            "prefix_metrics.csv",
            "FINAL_PASS.json",
            "run_manifest.json",
            "PASS",
        ]
    if task_id.endswith("_standardized"):
        method_id = task_id.removeprefix("bace_").removesuffix("_standardized")
        if method_id not in {"ours", "gcfexplainer", "globalgce", "comrecgc"}:
            raise ValueError(f"Unknown standardized BACE method task: {task_id}")
        return [
            "figure3_coverage_vs_k.csv",
            "figure4_coverage_vs_threshold.csv",
            f"table2_{method_id}_k10.csv",
            "prefix_metrics.csv",
            "prefix_metrics.json",
            "parent_best_distances.csv",
            "destination_distribution.csv",
            "summary.json",
            "run_manifest.json",
            "oracle_manifest.json",
            "evaluation_manifest.json",
            "artifact_manifest.json",
            "freeze_manifest.json",
            "_FINALIZED.json",
            "final_artifact_audit.json",
            "PASS",
        ]
    raise ValueError(f"No generic output contract for native task: {task_id}")


def _required_log_marker(task_id: str) -> str:
    if task_id.endswith("_preflight"):
        return '"status": "READY"'
    if task_id.endswith("_train_vrrw"):
        return "[BACE_GCFEXPLAINER_VRRW_OK]"
    if task_id.endswith("_train_summary"):
        return "[BACE_GCFEXPLAINER_NATIVE_SUMMARY_OK]"
    if task_id.endswith("_bridge_smoke"):
        return "[BACE_GLOBALGCE_BRIDGE_PASS]"
    if task_id.endswith(("_train_generation", "_train_common_recourse")):
        return '"run_complete": true'
    if task_id.endswith("_selection"):
        return '"status": "FROZEN"'
    if task_id.endswith("_standardized"):
        return "[BACE_FROZEN_CELL_STANDARDIZATION_PASS]"
    return '"status": "PASS"'


def _stage_contract(task_id: str) -> tuple[str, list[str], bool, bool, bool]:
    """Return stage, splits, freezes-selector, frozen-selector, read-only-test."""

    if task_id.endswith("_preflight"):
        return "BACE_BASELINE_PREFLIGHT", [], False, False, False
    if task_id.endswith(("_train_vrrw", "_train_generation")):
        return "BACE_BASELINE_TRAIN_GENERATION", ["train"], False, False, False
    if task_id.endswith("_bridge_smoke"):
        return "BACE_GLOBALGCE_FROZEN_GINE_BRIDGE_SMOKE", [], False, False, False
    if task_id.endswith("_train_summary"):
        return "BACE_BASELINE_TRAIN_SUMMARY", ["train"], False, False, False
    if task_id.endswith("_train_common_recourse"):
        return "BACE_BASELINE_TRAIN_RECOURSE", ["train"], False, False, False
    if task_id.endswith("_train_candidates"):
        return "BACE_BASELINE_TRAIN_CANDIDATES", ["train"], False, False, False
    if "_calibration_shard_" in task_id:
        return "BACE_BASELINE_CALIBRATION_VERIFY", ["calibration"], False, False, False
    if task_id.endswith("_calibration_merge"):
        return "BACE_BASELINE_CALIBRATION_MERGE", ["calibration"], False, False, False
    if task_id.endswith("_selection"):
        return "BACE_BASELINE_SELECTOR", ["calibration"], True, False, False
    if "_test_shard_" in task_id:
        return "BACE_BASELINE_TEST_VERIFY", ["test"], False, True, True
    if task_id.endswith("_test_merge"):
        return "BACE_BASELINE_TEST_MERGE", ["test"], False, True, True
    if task_id.endswith("_final_freeze"):
        return "BACE_BASELINE_FINAL_FREEZE", ["test"], False, True, True
    if task_id.endswith("_standardized"):
        return "BACE_BASELINE_STANDARDIZED", ["test"], False, True, True
    raise ValueError(f"No generic stage contract for native task: {task_id}")


def _priority(task_id: str, *, method_id: str) -> int:
    # The two native train routes are the two protected baseline lanes.  Their
    # READY GPU work sorts before priority-90 B11 instances; downstream
    # four-shard verification waits behind B11 so it cannot consume all cards.
    method_offset = 0 if method_id == "gcfexplainer" else 1
    if task_id.endswith("_preflight"):
        return 60 + method_offset
    if task_id.endswith(("_train_vrrw", "_train_generation")):
        return 80 + method_offset
    if task_id.endswith("_bridge_smoke"):
        return 78 + method_offset
    if task_id.endswith(("_train_summary", "_train_common_recourse")):
        return 82 + method_offset
    if task_id.endswith("_train_candidates"):
        return 84 + method_offset
    if "_calibration_shard_" in task_id:
        shard = int(task_id.rsplit("_", 1)[1])
        return 100 + shard * 2 + method_offset
    if task_id.endswith("_calibration_merge"):
        return 110 + method_offset
    if task_id.endswith("_selection"):
        return 120 + method_offset
    if "_test_shard_" in task_id:
        shard = int(task_id.rsplit("_", 1)[1])
        return 130 + shard * 2 + method_offset
    if task_id.endswith("_test_merge"):
        return 140 + method_offset
    if task_id.endswith("_final_freeze"):
        return 150 + method_offset
    if task_id.endswith("_standardized"):
        return 160 + method_offset
    raise ValueError(f"No generic priority contract for native task: {task_id}")


def _dependency_manifest(
    dependency: str, required_by_id: Mapping[str, Sequence[str]]
) -> str:
    files = set(required_by_id[dependency])
    if "frozen_selection_manifest.json" in files:
        name = "frozen_selection_manifest.json"
    elif "run_manifest.json" in files:
        name = "run_manifest.json"
    elif "route_contract.json" in files:
        name = "route_contract.json"
    else:  # pragma: no cover - every supported native task has one of these.
        raise ValueError(f"Dependency has no authoritative manifest: {dependency}")
    return f"{_dependency_token(dependency)}/{name}"


def adapt_bace_baseline_controller_fragment(
    fragment: Mapping[str, Any], *, output_root: str | Path
) -> dict[str, Any]:
    """Translate one native fragment into composer/production-loader schema."""

    if fragment.get("schema_version") != "bace_baseline_controller_fragment_v1":
        raise ValueError("Unsupported native BACE baseline fragment schema")
    spec = baseline_spec(str(fragment.get("method") or ""))
    if str(fragment.get("method_id") or "") != spec.method_id:
        raise ValueError("Native BACE baseline fragment method identity changed")
    native_root_path = Path(output_root).expanduser()
    if not native_root_path.is_absolute():
        raise ValueError("output_root must be absolute")
    native_root = str(native_root_path.resolve(strict=False))
    runner_dataset = f"bace-baseline-{spec.method_id}"

    if not spec.native_route_available:
        terminal = fragment.get("static_terminal")
        if not isinstance(terminal, Mapping):
            raise ValueError("Blocked native route lacks static terminal evidence")
        blocker = str(terminal.get("blocker_code") or spec.blocker_code or "")
        if not blocker.startswith("BLOCKED_"):
            raise ValueError("Blocked native route lacks a stable blocker code")
        task_id = str(terminal.get("task_id") or f"bace_{spec.method_id}_terminal")
        generic_tasks: list[dict[str, Any]] = []
        native_tasks = fragment.get("tasks")
        if native_tasks:
            if spec.method_id != "globalgce" or not isinstance(native_tasks, list):
                raise ValueError("Blocked route unexpectedly contains runnable tasks")
            if len(native_tasks) != 1 or not isinstance(native_tasks[0], Mapping):
                raise ValueError("GlobalGCE blocked route requires one native preflight")
            preflight = dict(native_tasks[0])
            preflight_id = str(preflight.get("task_id") or "")
            output = str(preflight.get("output_root") or "")
            argv = preflight.get("argv")
            if (
                not preflight_id.endswith("_preflight")
                or not Path(output).is_absolute()
                or not isinstance(argv, list)
                or not argv
            ):
                raise ValueError("GlobalGCE native preflight contract is malformed")
            command = [
                _rewrite_argument(
                    str(value),
                    task_id=preflight_id,
                    dependencies=[],
                    outputs={preflight_id: output},
                    native_root=native_root,
                )
                for value in argv
            ]
            inputs = preflight.get("inputs")
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("GlobalGCE native preflight has no frozen input")
            checkpoint = Path(str(inputs[0]))
            if not checkpoint.is_absolute():
                raise ValueError("GlobalGCE native preflight checkpoint is not absolute")
            generic_tasks.append(
                {
                    "id": preflight_id,
                    "dataset": "bace",
                    "stage": "BACE_GLOBALGCE_NATIVE_ACTION_PREFLIGHT",
                    "runner_dataset": runner_dataset,
                    "runner_stage": "BACE_GLOBALGCE_NATIVE_ACTION_PREFLIGHT",
                    "depends_on": [],
                    "resource": "cpu",
                    "priority": 61,
                    "enabled": True,
                    "data_splits": [],
                    "manifest_only": True,
                    "command": command,
                    "input_manifest": str(checkpoint / "model_card.json"),
                    "expected_output": output.rstrip("/") + "/attempt-{attempt}",
                    "required_output_files": [
                        "route_contract.json",
                        "oracle_provenance.json",
                        "official_source_audit.json",
                        "official_tensor_parity.json",
                        "state.json",
                        "NATIVE_ACTION_READY",
                        "BLOCKED_CODE.json",
                        "BLOCKED_CODE",
                    ],
                    "required_log_marker": '"native_action_status": "PASS"',
                    "environment": {
                        "PYTHONPATH": "{project_root}",
                        "RUN_TASTEMOLNET": "0",
                        "PYTHONHASHSEED": "0",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    "native_action_kind": spec.action_kind,
                    "native_action_semantics": spec.action_semantics,
                }
            )
        terminal_dependencies = [
            str(value) for value in terminal.get("dependencies", [])
        ]
        generic_tasks.append(
            {
                "id": task_id,
                "dataset": "bace",
                "stage": "BACE_BASELINE_BLOCKED_CODE",
                "runner_dataset": runner_dataset,
                "runner_stage": "BACE_BASELINE_BLOCKED_CODE",
                "depends_on": terminal_dependencies,
                "resource": "cpu",
                # Reserve priority 82 for the first GPU rule-training stage if
                # a reviewed exact-gradient adapter later releases the block.
                "priority": 82,
                "enabled": True,
                "blocked_reason": blocker,
                "blocker_code": blocker,
                "blocker_detail": str(terminal.get("reason") or ""),
                "data_splits": [],
                "manifest_only": True,
                "command": None,
            }
        )
        return {
            "schema_version": GENERIC_FRAGMENT_SCHEMA,
            "dataset": "bace",
            "method": spec.method,
            "method_id": spec.method_id,
            "tasks": generic_tasks,
            "root_task_ids": list(fragment.get("root_task_ids") or [task_id]),
            "terminal_task_ids": [task_id],
            "native_fragment_preserved": True,
        }

    raw_tasks = fragment.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Runnable native route has no tasks")
    if not all(isinstance(task, Mapping) for task in raw_tasks):
        raise ValueError("Native task list contains a non-object")
    tasks = [dict(task) for task in raw_tasks]
    outputs = _native_output_map(tasks)
    required_by_id = {
        task_id: _required_output_files(task_id) for task_id in outputs
    }
    generic: list[dict[str, Any]] = []
    for native in tasks:
        task_id = str(native["task_id"])
        dependencies = [str(value) for value in native.get("dependencies", [])]
        unknown = sorted(set(dependencies) - set(outputs))
        if unknown:
            raise ValueError(f"{task_id} has unknown dependencies: {unknown}")
        resource_value = native.get("resource")
        if not isinstance(resource_value, Mapping):
            raise ValueError(f"{task_id} has no native resource object")
        resource = str(resource_value.get("kind") or "")
        if resource not in {"cpu", "gpu"}:
            raise ValueError(f"{task_id} has unsupported resource: {resource}")
        argv = native.get("argv")
        if not isinstance(argv, list) or not argv or any(
            not isinstance(value, str) or not value for value in argv
        ):
            raise ValueError(f"{task_id} has no argv array")
        command = [
            _rewrite_argument(
                value,
                task_id=task_id,
                dependencies=dependencies,
                outputs=outputs,
                native_root=native_root,
            )
            for value in argv
        ]
        stale_outputs = [
            output
            for output in outputs.values()
            if any(
                value == output or value.startswith(output.rstrip("/") + "/")
                for value in command
            )
        ]
        if stale_outputs:
            raise ValueError(
                f"{task_id} retained mutable native output paths: {stale_outputs}"
            )
        stage, splits, freezes, selector_frozen, read_only_test = _stage_contract(
            task_id
        )
        if dependencies:
            input_manifest = _dependency_manifest(dependencies[0], required_by_id)
        else:
            inputs = native.get("inputs")
            if not isinstance(inputs, list) or not inputs:
                raise ValueError(f"Root task {task_id} has no frozen input")
            checkpoint = str(inputs[0])
            if not Path(checkpoint).is_absolute():
                raise ValueError(f"Root task {task_id} input is not absolute")
            input_manifest = str(Path(checkpoint) / "model_card.json")
        environment = {
            "PYTHONPATH": "{project_root}",
            "RUN_TASTEMOLNET": "0",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        row: dict[str, Any] = {
            "id": task_id,
            "dataset": "bace",
            "stage": stage,
            "runner_dataset": runner_dataset,
            "runner_stage": stage,
            "depends_on": dependencies,
            "resource": resource,
            "priority": _priority(task_id, method_id=spec.method_id),
            "enabled": True,
            "data_splits": splits,
            "manifest_only": not splits,
            "command": command,
            "input_manifest": input_manifest,
            "expected_output": outputs[task_id].rstrip("/") + "/attempt-{attempt}",
            "required_output_files": required_by_id[task_id],
            "required_log_marker": _required_log_marker(task_id),
            "environment": environment,
            "native_action_kind": spec.action_kind,
            "native_action_semantics": spec.action_semantics,
        }
        if freezes:
            row["freezes_selector"] = True
        if selector_frozen:
            row["selector_parameters_frozen"] = True
        if read_only_test:
            row["read_only_test"] = True
        generic.append(row)

    for task in generic:
        for dependency in task["depends_on"]:
            token = _dependency_token(str(dependency))
            if token not in json.dumps(task, sort_keys=True):
                raise ValueError(
                    f"{task['id']} does not bind dependency output token {token}"
                )
    return {
        "schema_version": GENERIC_FRAGMENT_SCHEMA,
        "dataset": "bace",
        "method": spec.method,
        "method_id": spec.method_id,
        "root_task_ids": list(fragment.get("root_task_ids") or []),
        "terminal_task_ids": list(fragment.get("terminal_task_ids") or []),
        "tasks": generic,
        "native_fragment_preserved": True,
        "b11_shard_priority_reference": B11_SHARD_PRIORITY,
    }


def build_bace_baseline_generic_controller_fragment(**kwargs: Any) -> dict[str, Any]:
    """Build the native fragment, then adapt a copy to generic schema."""

    native = build_bace_baseline_controller_fragment(**kwargs)
    return adapt_bace_baseline_controller_fragment(
        native, output_root=kwargs["output_root"]
    )


def atomic_write_generic_fragment(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        raise ValueError("fragment output must be absolute")
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Generic fragment output must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "B11_SHARD_PRIORITY",
    "GENERIC_FRAGMENT_SCHEMA",
    "adapt_bace_baseline_controller_fragment",
    "atomic_write_generic_fragment",
    "build_bace_baseline_generic_controller_fragment",
]
