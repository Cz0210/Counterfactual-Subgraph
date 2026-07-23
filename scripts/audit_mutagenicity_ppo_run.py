#!/usr/bin/env python3
"""Audit a completed Mutagenicity stable PPO run without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity_continued_sft import write_json_atomic  # noqa: E402
from src.train.mutagenicity_stable_ppo import (  # noqa: E402
    SOURCE_LABEL,
    TARGET_LABEL,
    validate_candidate_pool_schema,
)
from src.utils.io import read_jsonl  # noqa: E402


REQUIRED_ARTIFACTS = (
    "resolved_config.json",
    "dataset_manifest.json",
    "model_audit.json",
    "teacher_audit.json",
    "parent_coverage.json",
    "ppo_metrics.jsonl",
    "candidate_pool.jsonl",
    "validation_metrics.jsonl",
    "validation_samples.csv",
    "checkpoint_manifest.json",
    "best_checkpoint.json",
    "training_report.md",
    "_RUN_COMPLETE.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--report-txt", type=Path, default=None)
    parser.add_argument("--require-full-coverage", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_run(
    run_dir: str | Path,
    *,
    require_full_coverage: bool,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    missing = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (root / name).is_file() or (root / name).stat().st_size <= 0
    ]
    if missing:
        raise FileNotFoundError(f"Mutagenicity PPO run is missing artifacts: {missing}")

    resolved = _read_json(root / "resolved_config.json")
    manifest = _read_json(root / "dataset_manifest.json")
    model = _read_json(root / "model_audit.json")
    teacher = _read_json(root / "teacher_audit.json")
    coverage = _read_json(root / "parent_coverage.json")
    completion = _read_json(root / "_RUN_COMPLETE.json")
    candidates = read_jsonl(root / "candidate_pool.jsonl")
    metrics = read_jsonl(root / "ppo_metrics.jsonl")
    validation_metrics = read_jsonl(root / "validation_metrics.jsonl")
    validate_candidate_pool_schema(candidates)

    checks = {
        "source_target_ok": (
            int(resolved.get("source_label", -1)) == SOURCE_LABEL
            and int(resolved.get("target_label", -1)) == TARGET_LABEL
        ),
        "strict_flip_definition_ok": (
            resolved.get("strict_flip_definition")
            == "pred_before==1_and_pred_after==0"
            and teacher.get("strict_flip_definition")
            == "pred_before==1_and_pred_after==0"
        ),
        "cf_drop_definition_ok": (
            resolved.get("cf_drop_definition") == "p1_before_minus_p1_after"
            and teacher.get("cf_drop_definition") == "p1_before_minus_p1_after"
        ),
        "model_audit_passed": bool(model.get("model_audit_passed")),
        "reference_frozen": int(model.get("reference_trainable_params", -1)) == 0,
        "base_frozen": int(model.get("base_params_trainable", -1)) == 0,
        "teacher_audit_passed": bool(teacher.get("teacher_audit_passed")),
        "train_val_isolated": bool(
            manifest.get("isolation_audit", {}).get("isolation_passed")
        ),
        "calibration_not_loaded": not bool(manifest.get("calibration_loaded")),
        "test_not_loaded": not bool(manifest.get("test_loaded")),
        "metrics_present": bool(metrics),
        "validation_present": bool(validation_metrics),
        "candidate_count_matches_samples": (
            len(candidates) == int(coverage.get("num_samples_processed", -1))
        ),
        "sampling_without_replacement": (
            coverage.get("sampling_with_replacement") is False
        ),
        "completion_marker": bool(completion.get("completed")),
    }
    if require_full_coverage:
        checks["full_parent_coverage"] = abs(
            float(coverage.get("unique_parent_coverage", 0.0)) - 1.0
        ) <= 1e-12

    failures = sorted(key for key, passed in checks.items() if not passed)
    summary = {
        "run_dir": str(root),
        "audit_passed": not failures,
        "failed_checks": failures,
        "checks": checks,
        "num_candidate_rows": len(candidates),
        "num_update_rows": len(metrics),
        "num_validation_evaluations": len(validation_metrics),
        "num_dataset_rows": coverage.get("num_dataset_rows"),
        "num_samples_processed": coverage.get("num_samples_processed"),
        "num_unique_parents_seen": coverage.get("num_unique_parents_seen"),
        "unique_parent_coverage": coverage.get("unique_parent_coverage"),
        "samples_per_update": coverage.get("samples_per_update"),
        "updates_per_epoch": coverage.get("updates_per_epoch"),
        "global_step": coverage.get("global_step"),
        "distance_or_evaluation_recomputed": False,
    }
    if failures:
        raise ValueError(
            "Mutagenicity PPO run audit failed: " + ", ".join(failures)
        )
    return summary


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Mutagenicity Stable PPO Audit",
            "",
            f"- Run: `{summary['run_dir']}`",
            f"- Audit passed: {str(summary['audit_passed']).lower()}",
            f"- Dataset rows: {summary['num_dataset_rows']}",
            f"- Samples processed: {summary['num_samples_processed']}",
            f"- Unique parents seen: {summary['num_unique_parents_seen']}",
            f"- Unique parent coverage: {summary['unique_parent_coverage']}",
            f"- Samples per update: {summary['samples_per_update']}",
            f"- Updates per epoch: {summary['updates_per_epoch']}",
            f"- Global step: {summary['global_step']}",
            f"- Candidate rows: {summary['num_candidate_rows']}",
            f"- Validation evaluations: {summary['num_validation_evaluations']}",
            "- Calibration/test loaded: false",
            "- strict flip: pred_before == 1 and pred_after == 0",
            "- cf_drop: p1_before - p1_after",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit_run(
        args.run_dir,
        require_full_coverage=bool(args.require_full_coverage),
    )
    root = Path(args.run_dir).expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else root / "ppo_run_audit.json"
    )
    report_txt = (
        args.report_txt.expanduser().resolve()
        if args.report_txt
        else root / "ppo_run_audit.md"
    )
    write_json_atomic(output_json, summary)
    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report_txt.write_text(_report(summary), encoding="utf-8")
    print("[MUTAGENICITY_PPO_RUN_AUDIT_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
