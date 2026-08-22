#!/usr/bin/env python3
"""Build fresh CPU controller tasks for frozen BACE cell standardization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


CONTROLLER_TASKS = {
    "Ours": ("bace_ours_standardized", "bace_b14_frozen", 160),
    "GCFExplainer": (
        "bace_gcfexplainer_standardized",
        "bace_gcfexplainer_final_freeze",
        161,
    ),
    "GlobalGCE": (
        "bace_globalgce_standardized",
        "bace_globalgce_final_freeze",
        162,
    ),
    "ComRecGC": (
        "bace_comrecgc_standardized",
        "bace_comrecgc_final_freeze",
        163,
    ),
}
METHOD_SLUGS = {
    "Ours": "ours",
    "GCFExplainer": "gcfexplainer",
    "GlobalGCE": "globalgce",
    "ComRecGC": "comrecgc",
}
REQUIRED_FILES = [
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
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


def _safe(value: str, *, label: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError(f"{label} is not a safe ID: {value!r}")
    return text


def _absolute(value: str, *, existing: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Absolute path required: {value}")
    return path.resolve(strict=existing)


def _token(task_id: str) -> str:
    return "{dep_" + re.sub(r"[^A-Za-z0-9_]", "_", task_id) + "_output}"


def build_fragment(
    *,
    controller_id: str,
    output_root: str | Path,
    gnn_checkpoint: str | Path,
    expected_dataset_hash: str | None = None,
    expected_split_hash: str | None = None,
    expected_molclr_hash: str | None = None,
    expected_threshold_hash: str | None = None,
) -> dict[str, Any]:
    controller = _safe(controller_id, label="controller_id")
    root = _absolute(str(output_root), existing=False)
    checkpoint = _absolute(str(gnn_checkpoint), existing=True)
    expected_flags = (
        ("--expected-dataset-hash", expected_dataset_hash),
        ("--expected-split-hash", expected_split_hash),
        ("--expected-molclr-hash", expected_molclr_hash),
        ("--expected-threshold-hash", expected_threshold_hash),
    )
    tasks: list[dict[str, Any]] = []
    for method, (task_id, dependency, priority) in CONTROLLER_TASKS.items():
        slug = METHOD_SLUGS[method]
        command = [
            "{python}",
            "{project_root}/scripts/autodl/standardize_bace_frozen_cell.py",
            "--method",
            method,
            "--source-final-root",
            _token(dependency),
            "--gnn-checkpoint",
            str(checkpoint),
            "--output-dir",
            "{task_output}",
        ]
        for flag, value in expected_flags:
            if value is not None:
                command.extend((flag, str(value)))
        tasks.append(
            {
                "id": task_id,
                "dataset": "bace",
                "stage": "BACE_FROZEN_CELL_STANDARDIZATION",
                "runner_dataset": f"paper-cell-bace-{slug}",
                "runner_stage": "BACE_FROZEN_CELL_STANDARDIZATION",
                "depends_on": [dependency],
                "resource": "cpu",
                "priority": priority,
                "enabled": True,
                "data_splits": [],
                "manifest_only": True,
                "command": command,
                "input_manifest": _token(dependency) + "/FINAL_PASS.json",
                "expected_output": str(
                    root / "bace" / slug / "standardized" / "attempt-{attempt}"
                ),
                "required_output_files": [
                    *REQUIRED_FILES,
                    f"table2_{slug}_k10.csv",
                ],
                "required_log_marker": (
                    f"[BACE_FROZEN_CELL_STANDARDIZATION_PASS] method={method}"
                ),
                "environment": {
                    "PYTHONPATH": "{project_root}",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "TOKENIZERS_PARALLELISM": "false",
                    "RUN_TASTEMOLNET": "license_gate",
                },
                "semantic_failure_markers": [
                    "frozen contract differs",
                    "identity sha256 changed",
                    "raw test opened",
                    "threshold_config_hash differs",
                    "rf-free bace gine contract",
                    "artifact-only replay",
                ],
            }
        )
    return {
        "schema_version": "bace_frozen_cell_standardization_fragment_v1",
        "controller_id": controller,
        "paper_frozen": True,
        "raw_test_access": False,
        "globalgce_policy": "native_frozen_gine_lhs_rhs_route_only",
        "tasks": tasks,
    }


def atomic_write(path: str | Path, payload: dict[str, Any]) -> Path:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--expected-dataset-hash")
    parser.add_argument("--expected-split-hash")
    parser.add_argument("--expected-molclr-hash")
    parser.add_argument("--expected-threshold-hash")
    parser.add_argument("--fragment-output", required=True)
    args = parser.parse_args()
    fragment = build_fragment(
        controller_id=args.controller_id,
        output_root=args.output_root,
        gnn_checkpoint=args.gnn_checkpoint,
        expected_dataset_hash=args.expected_dataset_hash,
        expected_split_hash=args.expected_split_hash,
        expected_molclr_hash=args.expected_molclr_hash,
        expected_threshold_hash=args.expected_threshold_hash,
    )
    destination = atomic_write(args.fragment_output, fragment)
    print(
        json.dumps(
            {
                "status": "PASS",
                "fragment_output": str(destination),
                "task_ids": [task["id"] for task in fragment["tasks"]],
            },
            sort_keys=True,
        )
    )
    print("[BACE_CELL_STANDARDIZATION_TASKS_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
