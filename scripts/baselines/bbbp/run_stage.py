#!/usr/bin/env python3
"""Validate one BBBP stage and optionally run its explicitly wired entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.bbbp_framework import load_plan  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def _resolve_inputs(root: Path, values: tuple[str, ...]) -> tuple[list[str], list[str]]:
    existing: list[str] = []
    missing: list[str] = []
    for raw in values:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        rendered = str(path.resolve())
        (existing if path.exists() else missing).append(rendered)
    return existing, missing


def _builtin_command(stage_id: str, root: Path) -> list[str] | None:
    python = sys.executable
    mappings: dict[str, list[str]] = {
        "prepare": [
            python,
            str(root / "scripts/data/prepare_bbbp.py"),
            "--config",
            "configs/hpc.yaml",
            "--set",
            "inference.fallback_to_heuristic=false",
            "--raw-csv",
            os.environ.get("RAW_CSV", "data/raw/BBBP/bbbp.csv"),
            "--output-dir",
            os.environ.get("OUTPUT_DIR", "data/processed/BBBP"),
            "--seed",
            os.environ.get("SEED", "13"),
        ],
        "teacher": [
            python,
            str(root / "scripts/train_bbbp_teacher.py"),
            "--config",
            "configs/hpc.yaml",
            "--set",
            "inference.fallback_to_heuristic=false",
            "--data-dir",
            os.environ.get("DATA_DIR", "data/processed/BBBP"),
            "--output-dir",
            os.environ.get("OUTPUT_DIR", "outputs/hpc/oracle/bbbp"),
            "--seed",
            os.environ.get("SEED", "13"),
            "--n-jobs",
            os.environ.get("SLURM_CPUS_PER_TASK", "7"),
        ],
        "cross_prepare": [
            python,
            str(root / "scripts/data/prepare_scaffold_split.py"),
            "--config",
            "configs/hpc.yaml",
            "--set",
            "inference.fallback_to_heuristic=false",
            "--input-csv",
            os.environ.get("INPUT_CSV", "data/processed/BBBP/all.csv"),
            "--output-dir",
            os.environ.get("OUTPUT_DIR", "data/processed/BBBP/cross_scaffold_v1"),
            "--dataset",
            "BBBP",
            "--seed",
            os.environ.get("SEED", "13"),
            "--acyclic-policy",
            os.environ.get("ACYCLIC_POLICY", "canonical-smiles"),
        ],
    }
    return mappings.get(stage_id)


def _explicit_command() -> list[str] | None:
    raw = os.environ.get("BBBP_STAGE_EXEC_ARGV_JSON")
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload or any(not isinstance(value, str) or not value for value in payload):
        raise ValueError("BBBP_STAGE_EXEC_ARGV_JSON must be a non-empty JSON string array.")
    return list(payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    plan = load_plan(args.plan, project_root=root)
    stage = plan.stage(args.stage)
    existing, missing = _resolve_inputs(root, stage.required_inputs)
    report: dict[str, Any] = {
        "schema_version": "bbbp_stage_validation_v1",
        "status": "VALIDATED_NOT_RUN" if not args.execute else "READY_TO_EXECUTE",
        "dataset": "BBBP",
        "plan": plan.name,
        "stage": stage.stage_id,
        "method": stage.method,
        "wrapper": stage.wrapper,
        "depends_on": list(stage.depends_on),
        "output_root": stage.output_root,
        "existing_inputs": existing,
        "missing_inputs": missing,
        "expected_artifacts": list(stage.expected_artifacts),
        "cf_mode": plan.protocol["cf_mode"],
        "distance_line": plan.protocol["distance_line"],
        "threshold_source": plan.protocol["threshold_source"],
        "test_usage": plan.protocol["test_usage"],
        "selection_performed_in_eval": False,
        "threshold_fitted_on_test": False,
        "submission_performed": False,
        "formal_output_written": False,
    }
    if args.validate_only or args.dry_run:
        report["mode"] = "validate_only" if args.validate_only else "dry_run"
        print(json.dumps(report, sort_keys=True))
        print("[BBBP_STAGE_VALIDATED_NOT_RUN]")
        return 0
    if missing:
        raise FileNotFoundError(
            f"BBBP stage {stage.stage_id} is missing frozen inputs: {missing}"
        )
    output = Path(stage.output_root).expanduser()
    if not output.is_absolute():
        output = root / output
    if output.exists() and any(output.iterdir() if output.is_dir() else (output,)):
        raise FileExistsError(f"BBBP stage refuses non-empty output: {output}")
    command = _explicit_command() or _builtin_command(stage.stage_id, root)
    if command is None:
        raise RuntimeError(
            f"BBBP stage {stage.stage_id} has a validated framework contract but its "
            "dataset-specific native command is INPUT_REQUIRED. Set the documented "
            "BBBP_STAGE_EXEC_ARGV_JSON after freezing the baseline checkpoint/input contract."
        )
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"BBBP stage {stage.stage_id} failed with return code {completed.returncode}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
