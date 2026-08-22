#!/usr/bin/env python3
"""Build the post-cell matrix-audit task and final-export dependency contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from src.eval.four_by_four_registry import canonical_dataset, canonical_method


DATASETS = ("AIDS", "Mutagenicity", "BACE", "TasteMolNet")
METHODS = ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC")
MATRIX_TASK_ID = "final_matrix_audit"
BACE_STANDARDIZED_CELL_TASKS = {
    "Ours": "bace_ours_standardized",
    "GCFExplainer": "bace_gcfexplainer_standardized",
    "ComRecGC": "bace_comrecgc_standardized",
}
BACE_SCIENCE_TERMINALS = {
    "Ours": "bace_b14_frozen",
    "GCFExplainer": "bace_gcfexplainer_final_freeze",
    "ComRecGC": "bace_comrecgc_final_freeze",
}
MUT_GCF_STANDARDIZED_TASK = "mut_gcf_legacy_standardized"
MUT_GCF_RAW_TERMINAL = "mut_gcf_legacy_heldout"


def _safe_id(value: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"Unsafe task/controller ID: {value!r}")
    return text


def _absolute(value: str, *, existing: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Absolute path required: {value}")
    resolved = path.resolve(strict=existing)
    return resolved


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output must be fresh: {destination}")
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


def _parse_cells(values: list[str]) -> dict[tuple[str, str], str]:
    cells: dict[tuple[str, str], str] = {}
    for declaration in values:
        if "=" not in declaration or "/" not in declaration.split("=", 1)[0]:
            raise ValueError("--cell-task must use DATASET/METHOD=TASK_ID")
        identity, task_id = declaration.split("=", 1)
        raw_dataset, raw_method = identity.split("/", 1)
        key = (canonical_dataset(raw_dataset), canonical_method(raw_method))
        if key in cells:
            raise ValueError(f"Duplicate cell task: {key}")
        cells[key] = _safe_id(task_id)
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if set(cells) != expected:
        raise ValueError(
            f"Exactly 16 fixed cells are required; missing={sorted(expected - set(cells))}, "
            f"extra={sorted(set(cells) - expected)}"
        )
    if len(set(cells.values())) != 16:
        raise ValueError("Each cell requires one distinct terminal task")
    for method, expected_task in BACE_STANDARDIZED_CELL_TASKS.items():
        task_id = cells[("BACE", method)]
        if task_id != expected_task:
            science_task = BACE_SCIENCE_TERMINALS[method]
            raise ValueError(
                f"BACE/{method} must bind standardized task "
                f"{expected_task!r}; received {task_id!r}. Raw science terminal "
                f"{science_task!r} is not one registry-complete cell"
            )
    mut_gcf = cells[("Mutagenicity", "GCFExplainer")]
    if mut_gcf != MUT_GCF_STANDARDIZED_TASK:
        detail = (
            "raw held-out output is not a standardized cell"
            if mut_gcf == MUT_GCF_RAW_TERMINAL
            else f"expected {MUT_GCF_STANDARDIZED_TASK!r}"
        )
        raise ValueError(
            "Mutagenicity/GCFExplainer must bind the deterministic standardized "
            f"closure, not {mut_gcf!r}: {detail}"
        )
    return cells


def _token(task_id: str) -> str:
    return "{dep_" + re.sub(r"[^A-Za-z0-9_]", "_", task_id) + "_output}"


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    controller_id = _safe_id(args.controller_id)
    cells = _parse_cells(args.cell_task)
    license_task = _safe_id(args.taste_license_task_id)
    expectations = _absolute(args.expectations_json, existing=True)
    scan_roots = [_absolute(value, existing=True) for value in args.scan_root]
    output_root = _absolute(args.output_root, existing=False)
    dependencies = [
        cells[(dataset, method)] for dataset in DATASETS for method in METHODS
    ]
    if license_task not in dependencies:
        dependencies.append(license_task)
    command = [
        "{python}",
        "{project_root}/scripts/autodl/audit_four_methods_four_datasets.py",
        "--runtime-root",
        "{runtime_root}",
        "--output-root",
        "{task_output}",
        "--expectations-json",
        str(expectations),
        "--taste-license-gate-json",
        _token(license_task) + "/taste_license_gate.json",
    ]
    for root in scan_roots:
        command.extend(["--scan-root", str(root)])
    for dataset in DATASETS:
        for method in METHODS:
            task_id = cells[(dataset, method)]
            command.extend(
                ["--explicit-cell", f"{dataset}/{method}={_token(task_id)}"]
            )
    expected_output = str(output_root / "final-matrix-audit" / "attempt-{attempt}")
    task = {
        "id": MATRIX_TASK_ID,
        "dataset": "paper-matrix-audit",
        "stage": "FOUR_BY_FOUR_FINAL_MATRIX_AUDIT",
        "runner_dataset": "paper-matrix-audit",
        "runner_stage": "FOUR_BY_FOUR_FINAL_MATRIX_AUDIT",
        "depends_on": dependencies,
        "resource": "cpu",
        "priority": 1900,
        "data_splits": [],
        "manifest_only": True,
        "command": command,
        "input_manifest": str(expectations),
        "expected_output": expected_output,
        "required_output_files": [
            "matrix_status.csv",
            "matrix_status.json",
            "oracle_registry.json",
            "evaluation_contract.json",
            "artifact_inventory.csv",
            "stale_artifacts.csv",
            "adoption_report.md",
            "threshold_contracts/aids.json",
            "threshold_contracts/mutagenicity.json",
            "threshold_contracts/bace.json",
            "threshold_contracts/tastemolnet.json",
        ],
        "required_log_marker": "[MATRIX_AUDIT_PASS]",
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    fragment = {
        "schema_version": "four_by_four_final_matrix_audit_fragment_v1",
        "controller_id": controller_id,
        "tasks": [task],
    }
    dependency_contract = {
        "schema_version": "four_by_four_export_dependencies_v1",
        "matrix_task_id": MATRIX_TASK_ID,
        "matrix_status": _token(MATRIX_TASK_ID) + "/matrix_status.json",
        "bace_standardized_cell_tasks": dict(BACE_STANDARDIZED_CELL_TASKS),
        "cells": {
            f"{dataset}/{method}": cells[(dataset, method)]
            for dataset in DATASETS
            for method in METHODS
        },
    }
    return fragment, dependency_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--cell-task", action="append", required=True)
    parser.add_argument("--taste-license-task-id", default="tastemolnet_license_audit")
    parser.add_argument("--expectations-json", required=True)
    parser.add_argument("--scan-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fragment-output", required=True)
    parser.add_argument("--dependency-contract-output", required=True)
    args = parser.parse_args()
    fragment, dependency_contract = build(args)
    _atomic_json(Path(args.fragment_output), fragment)
    _atomic_json(Path(args.dependency_contract_output), dependency_contract)
    print(json.dumps({"status": "PASS", "task_id": MATRIX_TASK_ID}, sort_keys=True))
    print("[FOUR_BY_FOUR_FINAL_TASKS_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
