"""Generic AutoDL controller contract for the final four-by-four exporter."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from src.eval.four_by_four_main_results import (
    DATASET_ORDER,
    METHOD_ORDER,
    expected_final_output_files,
)
from src.eval.four_by_four_registry import canonical_dataset, canonical_method


FRAGMENT_SCHEMA_VERSION = "four_by_four_main_results_export_fragment_v1"
EXPORT_TASK_ID = "four_by_four_main_results_export"
EXPORT_STAGE = "FOUR_BY_FOUR_MAIN_RESULTS_EXPORT"
EXPORT_LOG_MARKER = "[FOUR_BY_FOUR_MAIN_RESULTS_PASS]"


def _safe_task_id(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError(f"{label} is not one safe controller task ID: {value!r}")
    return text


def _dependency_token(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", task_id)
    return "{dep_" + safe + "_output}"


def _canonical_cell_map(raw: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for key, value in raw.items():
        if "/" not in str(key):
            raise ValueError(f"Cell dependency key must be '<dataset>/<method>': {key}")
        raw_dataset, raw_method = str(key).split("/", 1)
        dataset = canonical_dataset(raw_dataset)
        method = canonical_method(raw_method)
        if dataset not in DATASET_ORDER or method not in METHOD_ORDER:
            raise ValueError(f"Unsupported cell dependency identity: {key}")
        identity = (dataset, method)
        if identity in result:
            raise ValueError(f"Duplicate canonical cell dependency: {identity}")
        result[identity] = _safe_task_id(value, label=f"dependency for {key}")
    expected = {(dataset, method) for dataset in DATASET_ORDER for method in METHOD_ORDER}
    if set(result) != expected:
        raise ValueError(
            f"Dependency map must contain exactly all 16 cells; "
            f"missing={sorted(expected - set(result))}, extra={sorted(set(result) - expected)}"
        )
    task_ids = list(result.values())
    if len(set(task_ids)) != 16:
        raise ValueError("Every matrix cell must bind one distinct terminal PASS task")
    if EXPORT_TASK_ID in task_ids:
        raise ValueError("Final export task cannot depend on itself")
    return result


def load_dependency_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dependency contract must be one JSON object")
    matrix_task_id = _safe_task_id(payload.get("matrix_task_id"), label="matrix_task_id")
    cells = payload.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("Dependency contract requires a cells object")
    canonical = _canonical_cell_map(cells)
    if matrix_task_id in canonical.values() or matrix_task_id == EXPORT_TASK_ID:
        raise ValueError("matrix_task_id must be distinct from all cell/export tasks")
    declared_template = str(payload.get("matrix_status") or "").strip()
    expected_template = _dependency_token(matrix_task_id) + "/matrix_status.json"
    if declared_template and declared_template != expected_template:
        raise ValueError(
            "matrix_status must be the exact passing matrix-task output token: "
            f"{expected_template}"
        )
    return {
        "source": str(source),
        "matrix_task_id": matrix_task_id,
        "matrix_status": expected_template,
        "cells": canonical,
    }


def build_export_task_fragment(
    *,
    controller_id: str,
    dependency_contract: str | Path,
    output_root: str | Path,
    priority: int = 2000,
) -> dict[str, Any]:
    controller = _safe_task_id(controller_id, label="controller_id")
    contract = load_dependency_contract(dependency_contract)
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        raise ValueError("output_root must be absolute")
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ValueError("priority must be an integer")
    ordered_cell_dependencies = [
        contract["cells"][(dataset, method)]
        for dataset in DATASET_ORDER
        for method in METHOD_ORDER
    ]
    dependencies = [*ordered_cell_dependencies, contract["matrix_task_id"]]
    expected_output = str(root.resolve(strict=False) / "attempt-{attempt}")
    matrix_status = str(contract["matrix_status"])
    task = {
        "id": EXPORT_TASK_ID,
        "dataset": "paper-main-results",
        "stage": EXPORT_STAGE,
        "runner_dataset": "paper-main-results",
        "runner_stage": EXPORT_STAGE,
        "depends_on": dependencies,
        "resource": "cpu",
        "priority": priority,
        "enabled": True,
        "data_splits": [],
        "manifest_only": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/export_four_by_four_main_results.py",
            "export",
            "--matrix-status",
            matrix_status,
            "--output-root",
            "{task_output}",
            "--project-root",
            "{project_root}",
            "--require-complete",
        ],
        "input_manifest": matrix_status,
        "expected_output": expected_output,
        "required_output_files": expected_final_output_files(),
        "required_log_marker": EXPORT_LOG_MARKER,
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "license_gate",
        },
        "semantic_failure_markers": [
            "blocked_incomplete_matrix",
            "standardized closure",
            "matrix cell set mismatch",
            "clear is not comrecgc",
            "numeric imputation",
            "test selection exclusion",
        ],
    }
    return {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        "controller_id": controller,
        "paper_frozen": True,
        "matrix_dependency_task_id": contract["matrix_task_id"],
        "cell_dependency_count": 16,
        "tasks": [task],
    }


def atomic_write_fragment(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Task fragment output must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "EXPORT_LOG_MARKER",
    "EXPORT_STAGE",
    "EXPORT_TASK_ID",
    "FRAGMENT_SCHEMA_VERSION",
    "atomic_write_fragment",
    "build_export_task_fragment",
    "load_dependency_contract",
]
