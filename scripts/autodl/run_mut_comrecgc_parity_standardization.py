#!/usr/bin/env python3
"""Resume Mutagenicity ComRecGC at chemistry after true trace parity.

The immutable repair-v2 common-recourse output is consumed read-only.  This
entry point creates a fresh root and runs only chemistry, unified evaluation,
the recovery gate, and freeze.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autodl.run_comrecgc_standardized_continuation import (  # noqa: E402
    ContinuationInputs,
    _git_head,
    _load_object,
    _require_directory,
    _require_file,
    _run_stage,
    _utc_now,
    _verify_adopted_generation_integrity,
    validate_adopted_generation,
)
from scripts.verify_comrecgc_checkout import verify_checkout  # noqa: E402
from src.baselines.comrecgc.contracts import (  # noqa: E402
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    UPSTREAM_COMMIT,
    atomic_write_bytes,
    sha256_file,
    write_json,
)
from src.utils.autodl_mut_traceoff_parity_v1 import (  # noqa: E402
    SOURCE_CANDIDATE_COUNT,
    SOURCE_PAYLOAD_SHA256,
    SOURCE_PROJECT_COMMIT,
    SOURCE_STEPS,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _object(path: Path, *, label: str) -> dict[str, Any]:
    value = _load_object(_require_file(path))
    if not isinstance(value, dict):  # defensive; _load_object already enforces
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_parity(path: Path, *, source_root: Path) -> dict[str, Any]:
    value = _object(path, label="trace parity")
    failures: list[str] = []
    for key, expected in {
        "schema_version": "mut_trace_on_off_parity_v1",
        "status": "PASS",
        "trace_parity_passed": True,
        "candidate_count": SOURCE_CANDIDATE_COUNT,
        "reference_trace_enabled": False,
        "traced_source_trace_enabled": True,
        "self_comparison": False,
        "trace_fields_stripped": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "traced_payload_sha256": SOURCE_PAYLOAD_SHA256,
    }.items():
        if value.get(key) != expected:
            failures.append(key)
    if Path(str(value.get("traced_source_root") or "")).resolve(strict=True) != source_root:
        failures.append("traced_source_root")
    reference_root = Path(str(value.get("reference_root") or "")).resolve(strict=True)
    if reference_root == source_root:
        failures.append("self_comparison_root")
    evidence = value.get("reference_evidence")
    if not isinstance(evidence, Mapping):
        failures.append("reference_evidence")
    else:
        checkpoint = evidence.get("checkpoint_evidence")
        if (
            evidence.get("status") != "PASS"
            or evidence.get("reference_root") != str(reference_root)
            or evidence.get("source_algorithm_commit") != SOURCE_PROJECT_COMMIT
            or evidence.get("reference_trace_enabled") is not False
            or evidence.get("reference_generation_rerun") is not True
            or evidence.get("calibration_loaded") is not False
            or evidence.get("test_loaded") is not False
            or not isinstance(checkpoint, Mapping)
            or int(checkpoint.get("completed_step", -1)) != SOURCE_STEPS
        ):
            failures.append("reference_evidence_contract")
        payload = Path(str(evidence.get("reference_payload") or "")).resolve(
            strict=True
        )
        if (
            payload.parent != reference_root
            or sha256_file(payload) != evidence.get("reference_payload_sha256")
            or evidence.get("reference_payload_sha256")
            != value.get("reference_payload_sha256")
        ):
            failures.append("reference_payload")
    if failures:
        raise ValueError(f"Trace parity gate is invalid: {failures}")
    return {**value, "path": str(path), "sha256": sha256_file(path)}


def _validate_common_adoption(path: Path, *, parity: Mapping[str, Any]) -> dict[str, Any]:
    value = _object(path, label="common-recourse adoption")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Common-recourse adoption has no evidence object")
    failures: list[str] = []
    if value.get("schema_version") != "mut_common_recourse_adoption_gate_v1":
        failures.append("schema_version")
    if value.get("status") != "PASS" or value.get("trace_parity_passed") is not True:
        failures.append("status")
    if value.get("trace_parity_sha256") != parity["sha256"]:
        failures.append("trace_parity_sha256")
    if evidence.get("status") != "PASS" or evidence.get("common_recourse_adopted") is not True:
        failures.append("evidence")
    common_root = Path(str(evidence.get("source_common_recourse_root") or "")).resolve(
        strict=True
    )
    if not common_root.is_dir():
        failures.append("common_root")
    source_files = evidence.get("source_files")
    if not isinstance(source_files, Mapping):
        failures.append("source_files")
    else:
        for name, record in source_files.items():
            if not isinstance(record, Mapping):
                failures.append(f"source_files.{name}")
                continue
            source = Path(str(record.get("path") or "")).resolve(strict=True)
            if source.parent != common_root or sha256_file(source) != record.get("sha256"):
                failures.append(f"source_files.{name}")
    if failures:
        raise ValueError(f"Common-recourse adoption gate is invalid: {failures}")
    return {
        **value,
        "path": str(path),
        "sha256": sha256_file(path),
        "common_root": str(common_root),
    }


def _commands(
    inputs: ContinuationInputs,
    *,
    common_root: Path,
    parity_path: Path,
    project_commit: str,
    teacher_sha256: str,
) -> list[tuple[str, list[str], Path, str]]:
    chemistry = inputs.output_root / "chemistry"
    evaluation = inputs.output_root / "unified_eval"
    gate = inputs.output_root / "full_gate"
    standardized = inputs.output_root / "standardized"
    trace = inputs.source_generation_root / "trace"
    chemistry_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--project-root",
        str(PROJECT_ROOT),
        "--dataset",
        "mutagenicity",
        "--dataset-dir",
        str(inputs.dataset_dir),
        "--generation-dir",
        str(inputs.source_generation_root),
        "--trace-lineage-path",
        str(trace / "candidate_action_lineage.json"),
        "--trace-evidence-path",
        str(parity_path),
        "--common-recourse-dir",
        str(common_root),
        "--output-dir",
        str(chemistry),
        "--preregistration-path",
        str(inputs.output_root / "preregistration/deterministic_chem_repair.json"),
        "--parent-limit",
        "1448",
        "--expected-candidate-count",
        str(SOURCE_CANDIDATE_COUNT),
        "--expected-counterfactuals-sha256",
        SOURCE_PAYLOAD_SHA256,
    ]
    evaluation_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/run_slot_unified_eval.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        "mutagenicity",
        "--mode",
        "full",
        "--chemistry-dir",
        str(chemistry),
        "--dataset-csv",
        str(inputs.dataset_csv),
        "--teacher-path",
        str(inputs.teacher_path),
        "--molclr-root",
        str(inputs.molclr_root),
        "--molclr-checkpoint",
        str(inputs.molclr_checkpoint),
        "--thresholds-json",
        str(inputs.thresholds_path),
        "--output-dir",
        str(evaluation),
        "--expected-parent-count",
        "217",
        "--max-k",
        "20",
        "--device",
        "cpu",
    ]
    gate_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/gate_recovery.py"),
        "--stage",
        "project-full",
        "--dataset",
        "mutagenicity",
        "--expected-parent-count",
        "217",
        "--expected-teacher-sha256",
        teacher_sha256,
        "--expected-project-commit",
        project_commit,
        "--input-dir",
        str(evaluation),
        "--output-dir",
        str(gate),
    ]
    freeze_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/freeze_recovery_result.py"),
        "--dataset",
        "mutagenicity",
        "--source-dir",
        str(evaluation),
        "--gate-dir",
        str(gate),
        "--output-dir",
        str(standardized),
    ]
    return [
        ("chemistry", chemistry_argv, chemistry / "_RUN_COMPLETE.json", "run_complete"),
        ("unified_eval", evaluation_argv, evaluation / "_RUN_COMPLETE.json", "run_complete"),
        ("full_gate", gate_argv, gate / "gate_result.json", "audit_passed"),
        ("freeze", freeze_argv, standardized / "_FINALIZED.json", "finalized"),
    ]


def run(
    inputs: ContinuationInputs,
    *,
    common_adoption_path: Path,
    parity_path: Path,
) -> dict[str, Any]:
    if inputs.dataset != "mutagenicity" or inputs.device != "cpu":
        raise ValueError("This continuation is Mutagenicity CPU-only")
    if inputs.output_root.exists() or inputs.output_root.is_symlink():
        raise FileExistsError(f"Fresh OUTPUT_ROOT already exists: {inputs.output_root}")
    inputs.output_root.parent.mkdir(parents=True, exist_ok=True)
    inputs.output_root.mkdir(mode=0o755)
    try:
        parity = _validate_parity(parity_path, source_root=inputs.source_generation_root)
        common = _validate_common_adoption(common_adoption_path, parity=parity)
        common_root = _require_directory(Path(common["common_root"]))
        adoption = validate_adopted_generation(inputs)
        if int(adoption["counterfactual_candidate_count"]) != SOURCE_CANDIDATE_COUNT:
            raise ValueError("Frozen generation candidate count changed")
        checkout = verify_checkout(
            inputs.upstream_root,
            expected_commit=UPSTREAM_COMMIT,
            validate_imports=True,
        )
        write_json(inputs.output_root / "generation_adoption_manifest.json", adoption)
        write_json(inputs.output_root / "common_recourse_adoption_manifest.json", common)
        write_json(inputs.output_root / "trace_parity_adoption_manifest.json", parity)
        write_json(inputs.output_root / "upstream_checkout_audit.json", checkout)
        project_commit = _git_head()
        teacher_sha256 = sha256_file(inputs.teacher_path)
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        for stage, argv, marker, field in _commands(
            inputs,
            common_root=common_root,
            parity_path=parity_path,
            project_commit=project_commit,
            teacher_sha256=teacher_sha256,
        ):
            _run_stage(
                stage=stage,
                argv=argv,
                marker=marker,
                required_field=field,
                environment=environment,
                output_root=inputs.output_root,
            )
        standardized = inputs.output_root / "standardized"
        standardized_manifest = _object(
            standardized / "run_manifest.json", label="standardized manifest"
        )
        freeze_manifest = _object(
            standardized / "freeze_manifest.json", label="freeze manifest"
        )
        failures: list[str] = []
        if standardized_manifest.get("dataset_key") != "mutagenicity":
            failures.append("dataset")
        if standardized_manifest.get("cf_mode") != CF_MODE:
            failures.append("cf_mode")
        if standardized_manifest.get("distance_line") != DISTANCE_LINE:
            failures.append("distance_line")
        if standardized_manifest.get("teacher_sha256") != teacher_sha256:
            failures.append("teacher")
        if freeze_manifest.get("dataset_key") != "mutagenicity":
            failures.append("freeze_dataset")
        if failures:
            raise ValueError(f"Standardized output identity mismatch: {failures}")
        source_integrity = _verify_adopted_generation_integrity(adoption)
        write_json(inputs.output_root / "source_integrity_final.json", source_integrity)
        final = {
            "schema_version": "mut_comrecgc_parity_standardization_v1",
            "status": "PASS",
            "dataset": "mutagenicity",
            "method": METHOD,
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "rf_oracle_used": True,
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "generation_adopted": True,
            "generation_rerun": False,
            "traceoff_reference_rerun": True,
            "trace_parity_passed": True,
            "trace_fields_stripped": False,
            "common_recourse_adopted": True,
            "common_recourse_rerun": False,
            "chemistry_rerun": True,
            "evaluation_rerun": True,
            "source_generation_root": str(inputs.source_generation_root),
            "source_common_recourse_root": str(common_root),
            "trace_parity_path": str(parity_path),
            "trace_parity_sha256": parity["sha256"],
            "standardized_output_root": str(standardized),
            "project_commit": project_commit,
            "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
            "standardized_run_manifest_sha256": sha256_file(
                standardized / "run_manifest.json"
            ),
            "freeze_manifest_sha256": sha256_file(standardized / "freeze_manifest.json"),
            "teacher_sha256": teacher_sha256,
            "calibration_loaded": False,
            "test_loaded_only_in_unified_evaluation": True,
            "completed_at": _utc_now(),
        }
        write_json(inputs.output_root / "run_manifest.json", final)
        write_json(inputs.output_root / "final_gate.json", final)
        write_json(inputs.output_root / "_RUN_COMPLETE.json", {**final, "run_complete": True})
        atomic_write_bytes(inputs.output_root / "PASS", b"PASS\n")
        print("[MUT_COMRECGC_PARITY_STANDARDIZATION_PASS]", flush=True)
        return final
    except Exception as exc:
        write_json(
            inputs.output_root / "FAILED.json",
            {
                "schema_version": "mut_comrecgc_parity_standardization_failure_v1",
                "status": "FAILED",
                "dataset": "mutagenicity",
                "error_class": type(exc).__name__,
                "message": str(exc),
                "output_root": str(inputs.output_root),
                "failed_at": _utc_now(),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--source-generation-root", type=_absolute, required=True)
    parser.add_argument("--upstream-root", type=_absolute, required=True)
    parser.add_argument("--dataset-dir", type=_absolute, required=True)
    parser.add_argument("--distance-checkpoint", type=_absolute, required=True)
    parser.add_argument("--dataset-csv", type=_absolute, required=True)
    parser.add_argument("--teacher-path", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--thresholds-path", type=_absolute, required=True)
    parser.add_argument("--common-adoption", type=_absolute, required=True)
    parser.add_argument("--trace-parity", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = ContinuationInputs(
        dataset="mutagenicity",
        source_generation_root=_require_directory(args.source_generation_root),
        upstream_root=_require_directory(args.upstream_root),
        dataset_dir=_require_directory(args.dataset_dir),
        source_csv=None,
        distance_checkpoint=_require_file(args.distance_checkpoint),
        dataset_csv=_require_file(args.dataset_csv),
        teacher_path=_require_file(args.teacher_path),
        molclr_root=_require_directory(args.molclr_root),
        molclr_checkpoint=_require_file(args.molclr_checkpoint),
        thresholds_path=_require_file(args.thresholds_path),
        output_root=args.output_root,
        device=str(args.device),
        theta_star=None,
        cost_cap=None,
    )
    run(
        inputs,
        common_adoption_path=_require_file(args.common_adoption),
        parity_path=_require_file(args.trace_parity),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
