#!/usr/bin/env python3
"""Build exact AutoDL task fragments for Taste licensing and A/M COMRECGC.

This builder performs no scientific work.  It resolves every input before it
publishes a fresh JSON fragment, leaving the generic controller to launch the
foreground payloads and record their run identities.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


METHODS = ("ours", "gcfexplainer", "globalgce", "comrecgc")


def _absolute_existing(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise argparse.ArgumentTypeError(f"path does not exist: {value}") from exc


def _absolute_fresh(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _atomic_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Task fragment output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _comrecgc_task(
    *,
    dataset: str,
    source_generation_root: Path,
    upstream_root: Path,
    dataset_dir: Path,
    dataset_csv: Path,
    teacher_path: Path,
    distance_checkpoint: Path,
    molclr_root: Path,
    molclr_checkpoint: Path,
    thresholds_path: Path,
    output_root: Path,
    source_csv: Path | None,
    priority: int,
) -> dict[str, Any]:
    task_id = f"{dataset}_comrecgc_standardized"
    threshold_task_id = f"{dataset}_comrecgc_threshold_freeze"
    expected = str(output_root / dataset / "comrecgc" / "attempt-{attempt}")
    environment = {
        "AUTODL_PYTHON": "{python}",
        "DATASET": dataset,
        "SOURCE_GENERATION_ROOT": str(source_generation_root),
        "COMRECGC_UPSTREAM_ROOT": str(upstream_root),
        "DATASET_DIR": str(dataset_dir),
        "DATASET_CSV": str(dataset_csv),
        "TEACHER_PATH": str(teacher_path),
        "DISTANCE_CHECKPOINT": str(distance_checkpoint),
        "MOLCLR_ROOT": str(molclr_root),
        "MOLCLR_CHECKPOINT": str(molclr_checkpoint),
        "THRESHOLDS_PATH": (
            f"{{dep_{threshold_task_id}_output}}/frozen_threshold_contract.json"
        ),
        "OUTPUT_ROOT": expected,
        "DEVICE": "cuda:0",
    }
    if source_csv is not None:
        environment["SOURCE_CSV"] = str(source_csv)
    return {
        "id": task_id,
        "dataset": dataset,
        "stage": "AM_COMRECGC_HELDOUT_EVAL",
        "runner_dataset": f"paper-cell-{dataset}-comrecgc",
        "runner_stage": "AM_COMRECGC_HELDOUT_EVAL",
        "depends_on": [threshold_task_id],
        "resource": "gpu",
        "priority": priority,
        "data_splits": ["test"],
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            "bash",
            "{project_root}/scripts/autodl/run_comrecgc_standardized_continuation.sh",
        ],
        "input_manifest": str(source_generation_root / "run_manifest.json"),
        "expected_output": expected,
        "required_output_files": [
            "adoption_manifest.json",
            "standardized/_FINALIZED.json",
            "standardized/run_manifest.json",
            "run_manifest.json",
            "final_gate.json",
            "_RUN_COMPLETE.json",
            "PASS",
        ],
        "required_log_marker": (
            f"[COMRECGC_STANDARDIZED_CONTINUATION_PASS] dataset={dataset}"
        ),
        "environment": environment,
    }


def _threshold_freeze_task(
    *,
    dataset: str,
    thresholds_path: Path,
    output_root: Path,
    priority: int,
) -> dict[str, Any]:
    task_id = f"{dataset}_comrecgc_threshold_freeze"
    expected = str(output_root / dataset / "threshold-freeze" / "attempt-{attempt}")
    return {
        "id": task_id,
        "dataset": dataset,
        "stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "runner_dataset": f"paper-threshold-{dataset}",
        "runner_stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "depends_on": [],
        "resource": "cpu",
        "priority": priority,
        "data_splits": [],
        "manifest_only": True,
        "freezes_selector": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/verify_frozen_threshold_contract.py",
            "--config",
            "configs/hpc.yaml",
            "--dataset",
            dataset,
            "--source",
            str(thresholds_path),
            "--output",
            "{task_output}",
        ],
        "input_manifest": str(thresholds_path),
        "expected_output": expected,
        "required_output_files": [
            "frozen_threshold_contract.json",
            "threshold_adoption_audit.json",
            "PASS",
        ],
        "required_log_marker": (
            f"[FROZEN_THRESHOLD_CONTRACT_PASS] dataset={dataset}"
        ),
        "environment": {"AUTODL_PYTHON": "{python}"},
    }


def build_tasks(args: argparse.Namespace) -> dict[str, Any]:
    runtime_root = args.runtime_root
    run_root = (
        runtime_root
        / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/runs"
        / args.controller_id
    )
    license_output = str(run_root / "taste-license-audit/attempt-{attempt}")
    license_environment = {
        "AUTODL_PYTHON": "{python}",
        "TASTEMOLNET_PREPARED_ROOT": str(args.taste_prepared_root),
        "OUTPUT_ROOT": license_output,
    }
    if args.taste_approval_file is not None:
        license_environment["TASTEMOLNET_LICENSE_APPROVAL_FILE"] = str(
            args.taste_approval_file
        )
    if args.taste_upstream_checkout is not None:
        license_environment["TASTEMOLNET_UPSTREAM_CHECKOUT"] = str(
            args.taste_upstream_checkout
        )

    tasks: list[dict[str, Any]] = [
        {
            "id": "tastemolnet_license_audit",
            "dataset": "taste-license-audit",
            "stage": "TASTEMOLNET_LICENSE_AUDIT",
            "runner_dataset": "taste-license-audit",
            "runner_stage": "TASTEMOLNET_LICENSE_AUDIT",
            "depends_on": [],
            "resource": "cpu",
            "priority": 5,
            "data_splits": [],
            "manifest_only": True,
            "command": [
                "bash",
                "{project_root}/scripts/autodl/run_tastemolnet_license_audit.sh",
            ],
            "input_manifest": str(args.taste_prepared_root / "provenance_manifest.json"),
            "expected_output": license_output,
            "required_output_files": [
                "taste_license_audit.md",
                "taste_license_evidence.json",
                "taste_license_gate.json",
            ],
            "required_output_any": [["PASS", "BLOCKED_LICENSE_REVIEW"]],
            "required_log_marker": "[TASTE_LICENSE_AUDIT_COMPLETE]",
            "environment": license_environment,
        },
        _threshold_freeze_task(
            dataset="mutagenicity",
            thresholds_path=args.mut_thresholds_path,
            output_root=run_root / "cells",
            priority=190,
        ),
        _comrecgc_task(
            dataset="mutagenicity",
            source_generation_root=args.mut_source_generation_root,
            upstream_root=args.comrecgc_upstream_root,
            dataset_dir=args.mut_dataset_dir,
            dataset_csv=args.mut_dataset_csv,
            teacher_path=args.mut_teacher_path,
            distance_checkpoint=args.mut_distance_checkpoint,
            molclr_root=args.molclr_root,
            molclr_checkpoint=args.molclr_checkpoint,
            thresholds_path=args.mut_thresholds_path,
            output_root=run_root / "cells",
            source_csv=None,
            priority=200,
        ),
        _threshold_freeze_task(
            dataset="aids",
            thresholds_path=args.aids_thresholds_path,
            output_root=run_root / "cells",
            priority=191,
        ),
        _comrecgc_task(
            dataset="aids",
            source_generation_root=args.aids_source_generation_root,
            upstream_root=args.comrecgc_upstream_root,
            dataset_dir=args.aids_dataset_dir,
            dataset_csv=args.aids_dataset_csv,
            teacher_path=args.aids_teacher_path,
            distance_checkpoint=args.aids_distance_checkpoint,
            molclr_root=args.molclr_root,
            molclr_checkpoint=args.molclr_checkpoint,
            thresholds_path=args.aids_thresholds_path,
            output_root=run_root / "cells",
            source_csv=args.aids_source_csv,
            priority=201,
        ),
    ]
    for offset, method in enumerate(METHODS):
        tasks.append(
            {
                "id": f"tastemolnet_{method}",
                "dataset": "tastemolnet",
                "stage": f"TASTEMOLNET_{method.upper()}",
                "depends_on": ["tastemolnet_license_audit"],
                "resource": "gpu",
                "priority": 1000 + offset,
                "data_splits": [],
                "manifest_only": True,
                "command": None,
                "blocked_reason": "BLOCKED_LICENSE_REVIEW",
            }
        )
    payload = {
        "schema_version": 1,
        "controller_id": args.controller_id,
        "paper_frozen": True,
        "tasks": tasks,
    }
    _atomic_json(args.output, payload)
    return {
        "status": "PASS",
        "output": str(args.output),
        "task_count": len(tasks),
        "task_ids": [task["id"] for task in tasks],
        "taste_heavy_authorized": False,
        "paper_frozen": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--runtime-root", type=_absolute_existing, required=True)
    parser.add_argument("--output", type=_absolute_fresh, required=True)
    parser.add_argument("--taste-prepared-root", type=_absolute_existing, required=True)
    parser.add_argument("--taste-approval-file", type=_absolute_existing)
    parser.add_argument("--taste-upstream-checkout", type=_absolute_existing)
    parser.add_argument("--comrecgc-upstream-root", type=_absolute_existing, required=True)
    parser.add_argument("--molclr-root", type=_absolute_existing, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute_existing, required=True)
    for prefix in ("mut", "aids"):
        parser.add_argument(
            f"--{prefix}-source-generation-root", type=_absolute_existing, required=True
        )
        parser.add_argument(f"--{prefix}-dataset-dir", type=_absolute_existing, required=True)
        parser.add_argument(f"--{prefix}-dataset-csv", type=_absolute_existing, required=True)
        parser.add_argument(f"--{prefix}-teacher-path", type=_absolute_existing, required=True)
        parser.add_argument(
            f"--{prefix}-distance-checkpoint", type=_absolute_existing, required=True
        )
        parser.add_argument(
            f"--{prefix}-thresholds-path", type=_absolute_existing, required=True
        )
    parser.add_argument("--aids-source-csv", type=_absolute_existing, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_tasks(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[FOUR_BY_FOUR_CORE_TASKS_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
