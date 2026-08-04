#!/usr/bin/env python3
"""Read-only scientific gate for the CLEAR Mutagenicity Phase B smoke."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


EVIDENCE_MARKER = "[AUTOMATION_CLEAR_PHASE_B_GATE_EVIDENCE_B64]"
SUCCESS_MARKERS = (
    "[MUTAGENICITY_CLEAR_GRAPHPRED_SMOKE_OK]",
    "[MUTAGENICITY_CLEAR_GRAPHCFE_SMOKE_OK]",
    "[MUTAGENICITY_CLEAR_GENERATION_SMOKE_OK]",
    "[MUTAGENICITY_CLEAR_TRAIN_POOL_AUDIT_OK]",
    "[MUTAGENICITY_CLEAR_TRAIN_POOL_SMOKE_OK]",
    "[AUTOMATION_CLEAR_PHASE_B_GPU_SMOKE_COMPLETE]",
)
FATAL_LOG_TOKENS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OUT_OF_MEMORY",
    "DUE TO TIME LIMIT",
    "slurmstepd: error: Detected",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-job-id", required=True)
    return parser


def _resolve_under(root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes project root: {resolved}") from exc
    return resolved


def _load_json(path: Path, failures: list[str]) -> dict[str, Any]:
    if not path.is_file():
        failures.append(f"missing_file:{path.name}")
        return {}
    if path.stat().st_size <= 0:
        failures.append(f"empty_file:{path.name}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid_json:{path.name}:{type(exc).__name__}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path.name}")
        return {}
    return payload


def _assert_finite(value: Any, path: str, failures: list[str]) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            failures.append(f"non_finite_json_value:{path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]", failures)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}", failures)


def _require(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    source: str,
    failures: list[str],
) -> Any:
    actual = payload.get(key)
    if isinstance(expected, bool):
        matched = isinstance(actual, bool) and actual is expected
    elif isinstance(expected, int):
        matched = isinstance(actual, int) and not isinstance(actual, bool)
        matched = matched and actual == expected
    else:
        matched = actual == expected
    if not matched:
        failures.append(
            f"field_mismatch:{source}.{key}:actual={actual!r}:expected={expected!r}"
        )
    return actual


def evaluate(
    *,
    project_root: Path,
    output_dir: Path,
    stdout_log: Path,
    stderr_log: Path,
    expected_commit: str,
    expected_job_id: str,
) -> dict[str, Any]:
    failures: list[str] = []
    required_files = (
        "raw_generated_candidates.jsonl",
        "candidate_pool.jsonl",
        "candidate_universe.jsonl",
        "generation_progress.json",
        "summary.json",
        "run_manifest.json",
        "train_pool_audit.json",
        "_RUN_COMPLETE.json",
        "_AUTOMATION_PHASE_B_GPU_SMOKE_COMPLETE.json",
    )
    output_prefix = (
        project_root
        / "outputs/hpc/mutagenicity/baselines/clear/automation_phase_b_smoke"
    ).resolve()
    try:
        output_dir.relative_to(output_prefix)
    except ValueError:
        failures.append(f"unexpected_output_root:{output_dir}")

    artifacts: dict[str, dict[str, Any]] = {}
    for name in required_files:
        path = output_dir / name
        exists = path.is_file()
        nonempty = exists and path.stat().st_size > 0
        artifacts[name] = {"path": str(path), "exists": exists, "nonempty": nonempty}
        if not exists:
            failures.append(f"missing_file:{name}")
        elif not nonempty:
            failures.append(f"empty_file:{name}")

    summary = _load_json(output_dir / "summary.json", failures)
    manifest = _load_json(output_dir / "run_manifest.json", failures)
    progress = _load_json(output_dir / "generation_progress.json", failures)
    audit = _load_json(output_dir / "train_pool_audit.json", failures)
    complete = _load_json(output_dir / "_RUN_COMPLETE.json", failures)
    automation_marker = _load_json(
        output_dir / "_AUTOMATION_PHASE_B_GPU_SMOKE_COMPLETE.json", failures
    )
    for name, payload in (
        ("summary", summary),
        ("manifest", manifest),
        ("generation_progress", progress),
        ("train_pool_audit", audit),
        ("run_complete_marker", complete),
        ("automation_marker", automation_marker),
    ):
        _assert_finite(payload, name, failures)

    summary_contract = {
        "model_train_rows": 2885,
        "model_val_rows": 355,
        "generation_source_parent_rows": 1448,
        "selected_generation_parents": 64,
        "parent_limit": 64,
        "generation_chunk_size": 16,
        "graphpred_epochs": 5,
        "cfe_epochs": 5,
        "batch_size": 8,
        "seed": 13,
        "generation_profile": "smoke",
        "generation_only": False,
        "model_training_performed": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "calibration_rows_loaded": 0,
        "test_rows_loaded": 0,
        "run_complete": True,
    }
    for key, expected in summary_contract.items():
        _require(summary, key, expected, source="summary", failures=failures)
    for key in ("candidate_pool_rows", "canonical_unique_candidates"):
        value = summary.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            failures.append(f"field_not_positive:summary.{key}:actual={value!r}")

    manifest_contract = {
        "parent_limit": 64,
        "generation_chunk_size": 16,
        "seed": 13,
        "generation_profile": "smoke",
        "generation_only": False,
        "model_training_performed": True,
        "generation_input_split": "train",
        "model_train_split": "train",
        "model_validation_split": "val",
        "candidate_selection_performed": False,
        "source_label": 1,
        "target_label": 0,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
        "git_commit": expected_commit,
    }
    for key, expected in manifest_contract.items():
        _require(manifest, key, expected, source="manifest", failures=failures)
    parent_ids = manifest.get("generation_parent_ids")
    if not isinstance(parent_ids, list) or len(parent_ids) != 64:
        failures.append(
            "field_mismatch:manifest.generation_parent_ids_count:"
            f"actual={len(parent_ids) if isinstance(parent_ids, list) else None}:expected=64"
        )
    elif len({str(value) for value in parent_ids}) != 64:
        failures.append("duplicate_generation_parent_ids")
    for name, value in dict(manifest.get("inputs") or {}).items():
        lowered = str(value).replace("\\", "/").lower()
        if any(token in lowered for token in ("calibration_source", "calibration_target", "test_source", "test_target")):
            failures.append(f"forbidden_input:{name}")
    expected_input_suffixes = {
        "phase_a_root": (
            "outputs/hpc/mutagenicity/final/clear_phase_a_dataset_codec_best",
            "outputs/hpc/mutagenicity/baselines/clear/phase_a_dataset_codec_v2",
        ),
        "generation_csv": ((
            "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/"
            "train_source_label1_teacher_correct.csv"
        ),),
        "teacher_path": ((
            "outputs/hpc/oracle/mutagenicity_rf_v1/"
            "mutagenicity_rf_model.pkl"
        ),),
        "official_root": ("baselines/clear_official",),
    }
    manifest_inputs = dict(manifest.get("inputs") or {})
    for name, suffixes in expected_input_suffixes.items():
        actual = str(manifest_inputs.get(name) or "").replace("\\", "/")
        if not any(actual.endswith(suffix) for suffix in suffixes):
            failures.append(
                f"provenance_path_mismatch:manifest.inputs.{name}:actual={actual!r}"
            )

    audit_contract = {
        "model_train_rows": 2885,
        "model_val_rows": 355,
        "generation_source_rows": 1448,
        "selected_generation_parents": 64,
        "generation_profile": "smoke",
        "generation_only": False,
        "model_training_performed": True,
        "completed_chunk_count": 4,
        "calibration_rows_loaded": 0,
        "test_rows_loaded": 0,
        "chunk_resume_duplicate_rows": 0,
        "run_complete": True,
        "audit_passed": True,
    }
    for key, expected in audit_contract.items():
        _require(audit, key, expected, source="audit", failures=failures)
    for key in ("candidate_pool_rows", "candidate_universe_rows"):
        value = audit.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            failures.append(f"field_not_positive:audit.{key}:actual={value!r}")

    _require(progress, "selected_parent_count", 64, source="progress", failures=failures)
    _require(progress, "completed_chunk_count", 4, source="progress", failures=failures)
    _require(progress, "generation_profile", "smoke", source="progress", failures=failures)
    _require(progress, "run_complete", True, source="progress", failures=failures)
    _require(complete, "run_complete", True, source="complete", failures=failures)
    _require(complete, "generation_profile", "smoke", source="complete", failures=failures)
    _require(automation_marker, "run_complete", True, source="automation_marker", failures=failures)
    _require(automation_marker, "git_commit", expected_commit, source="automation_marker", failures=failures)
    _require(automation_marker, "job_id", expected_job_id, source="automation_marker", failures=failures)
    _require(automation_marker, "calibration_loaded", False, source="automation_marker", failures=failures)
    _require(automation_marker, "test_loaded", False, source="automation_marker", failures=failures)
    if Path(str(automation_marker.get("output_dir", ""))).resolve() != output_dir:
        failures.append("field_mismatch:automation_marker.output_dir")

    log_checks: dict[str, Any] = {}
    stdout_text = ""
    stderr_text = ""
    for label, path in (("stdout", stdout_log), ("stderr", stderr_log)):
        exists = path.is_file()
        nonempty = exists and path.stat().st_size > 0
        log_checks[f"{label}_exists"] = exists
        log_checks[f"{label}_nonempty"] = nonempty
        if not exists:
            failures.append(f"missing_{label}_log")
        elif label == "stdout" and not nonempty:
            failures.append("empty_stdout_log")
    if stdout_log.is_file():
        stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace")
    if stderr_log.is_file():
        stderr_text = stderr_log.read_text(encoding="utf-8", errors="replace")
    for marker in SUCCESS_MARKERS:
        present = marker in stdout_text
        log_checks[f"marker:{marker}"] = present
        if not present:
            failures.append(f"stdout_marker_missing:{marker}")
    combined_logs = stdout_text + "\n" + stderr_text
    for token in FATAL_LOG_TOKENS:
        present = token in combined_logs
        log_checks[f"fatal_token:{token}"] = present
        if present:
            failures.append(f"fatal_log_token:{token}")

    checks = {
        "model_train_rows": audit.get("model_train_rows"),
        "model_val_rows": audit.get("model_val_rows"),
        "generation_source_rows": audit.get("generation_source_rows"),
        "selected_generation_parents": audit.get("selected_generation_parents"),
        "generation_profile": audit.get("generation_profile"),
        "candidate_pool_rows": audit.get("candidate_pool_rows"),
        "candidate_universe_rows": audit.get("candidate_universe_rows"),
        "completed_chunk_count": audit.get("completed_chunk_count"),
        "calibration_loaded": summary.get("calibration_loaded"),
        "test_loaded": summary.get("test_loaded"),
        "model_training_performed": manifest.get("model_training_performed"),
        "git_commit": manifest.get("git_commit"),
        "job_id": automation_marker.get("job_id"),
        "output_dir": str(output_dir),
        **log_checks,
    }
    return {
        "schema_version": 1,
        "audit_passed": not failures,
        "run_complete": not failures,
        "failed_hard_checks": failures,
        "checks": checks,
        "artifacts": artifacts,
        "provenance": {
            "dataset": "Mutagenicity",
            "source_label": 1,
            "target_label": 0,
            "cf_mode": "strict_flip",
            "git_commit": manifest.get("git_commit"),
            "generation_input_split": manifest.get("generation_input_split"),
            "calibration_loaded": summary.get("calibration_loaded"),
            "test_loaded": summary.get("test_loaded"),
            "output_dir": str(output_dir),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    try:
        if not project_root.is_dir():
            raise ValueError(f"Project root does not exist: {project_root}")
        output_dir = _resolve_under(project_root, args.output_dir, label="output-dir")
        stdout_log = _resolve_under(project_root, args.stdout_log, label="stdout-log")
        stderr_log = _resolve_under(project_root, args.stderr_log, label="stderr-log")
        evidence = evaluate(
            project_root=project_root,
            output_dir=output_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            expected_commit=str(args.expected_commit),
            expected_job_id=str(args.expected_job_id),
        )
    except Exception as exc:
        evidence = {
            "schema_version": 1,
            "audit_passed": False,
            "run_complete": False,
            "failed_hard_checks": [f"gate_exception:{type(exc).__name__}:{exc}"],
            "checks": {},
            "artifacts": {},
            "provenance": {},
        }
    encoded = base64.b64encode(
        json.dumps(evidence, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    print(f"{EVIDENCE_MARKER} {encoded}", flush=True)
    return 0 if evidence["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
