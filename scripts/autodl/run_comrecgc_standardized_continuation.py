#!/usr/bin/env python3
"""Continue a frozen COMRECGC generation into one standardized paper cell.

The completed generation is adopted read-only.  Every downstream stage writes
below a fresh output root and the PASS marker is published last.  This entry
point intentionally does not regenerate random walks or modify the recovery
root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_comrecgc_checkout import verify_checkout  # noqa: E402
from src.baselines.comrecgc.contracts import (  # noqa: E402
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    UPSTREAM_COMMIT,
    atomic_write_bytes,
    sha256_file,
    stable_json_sha256,
    write_json,
)


DATASET_CONTRACTS: dict[str, dict[str, int]] = {
    "aids": {"generation_parent_limit": 1283, "evaluation_parent_count": 1283},
    "mutagenicity": {
        "generation_parent_limit": 1448,
        "evaluation_parent_count": 217,
    },
}


@dataclass(frozen=True)
class ContinuationInputs:
    dataset: str
    source_generation_root: Path
    upstream_root: Path
    dataset_dir: Path
    source_csv: Path | None
    distance_checkpoint: Path
    dataset_csv: Path
    teacher_path: Path
    molclr_root: Path
    molclr_checkpoint: Path
    thresholds_path: Path
    output_root: Path
    device: str
    theta_star: float | None
    cost_cap: float | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(resolved)
    return resolved


def _require_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _git_head(project_root: Path = PROJECT_ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()


def validate_adopted_generation(inputs: ContinuationInputs) -> dict[str, Any]:
    """Validate small frozen gates without rehashing the multi-GB payload twice."""

    if inputs.dataset not in DATASET_CONTRACTS:
        raise ValueError(f"Unsupported dataset: {inputs.dataset}")
    source = _require_directory(inputs.source_generation_root)
    contract = DATASET_CONTRACTS[inputs.dataset]
    manifest_path = _require_file(source / "run_manifest.json")
    complete_path = _require_file(source / "_RUN_COMPLETE.json")
    recovery_path = _require_file(source / "freeze_only_recovery.json")
    closure_path = _require_file(source / "frozen_payload_closure_audit.json")
    original_adoption_path = _require_file(source / "adoption_manifest.json")
    manifest = _load_object(manifest_path)
    complete = _load_object(complete_path)
    recovery = _load_object(recovery_path)
    closure = _load_object(closure_path)
    original_adoption = _load_object(original_adoption_path)

    failures: list[str] = []
    expected = {
        "dataset": inputs.dataset,
        "mode": "full",
        "parent_limit": contract["generation_parent_limit"],
        "run_complete": True,
        "freeze_only_recovery": True,
        "algorithm_rerun": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "generation_mode": "adopted_read_only_cache",
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            failures.append(
                f"run_manifest.{field}:actual={manifest.get(field)!r}:"
                f"expected={expected_value!r}"
            )
    if complete.get("run_complete") is not True:
        failures.append("_RUN_COMPLETE.run_complete")
    if complete.get("freeze_only_recovery") is not True:
        failures.append("_RUN_COMPLETE.freeze_only_recovery")
    if recovery.get("recovery_completed") is not True:
        failures.append("freeze_only_recovery.recovery_completed")
    if int(recovery.get("completed_steps", -1)) != 50_000:
        failures.append("freeze_only_recovery.completed_steps")
    if recovery.get("algorithm_rerun") is not False:
        failures.append("freeze_only_recovery.algorithm_rerun")
    if closure.get("closure_complete") is not True:
        failures.append("frozen_payload_closure.closure_complete")
    if closure.get("post_write_reload_verified") is not True:
        failures.append("frozen_payload_closure.post_write_reload_verified")
    if original_adoption.get("generation_mode") != "adopted_read_only_cache":
        failures.append("adoption_manifest.generation_mode")

    payload = Path(str(manifest.get("counterfactuals_path") or "")).expanduser()
    try:
        payload = payload.resolve(strict=True)
        payload.relative_to(source)
    except (FileNotFoundError, ValueError):
        failures.append("counterfactuals_path_not_inside_frozen_source")
    claimed_payload_sha = str(manifest.get("counterfactuals_sha256") or "")
    if len(claimed_payload_sha) != 64:
        failures.append("counterfactuals_sha256")
    if complete.get("counterfactuals_sha256") != claimed_payload_sha:
        failures.append("counterfactuals_sha256_gate_disagreement")
    if recovery.get("counterfactuals_sha256") != claimed_payload_sha:
        failures.append("counterfactuals_sha256_recovery_disagreement")
    candidate_count = int(manifest.get("counterfactual_candidate_count", -1))
    if candidate_count <= 0:
        failures.append("counterfactual_candidate_count")
    if failures:
        raise ValueError("Frozen generation adoption failed: " + "; ".join(failures))

    return {
        "schema_version": 1,
        "status": "PASS",
        "dataset": inputs.dataset,
        "generation_adopted": True,
        "generation_mode": "adopted_read_only_cache",
        "generation_rerun": False,
        "source_generation_root": str(source),
        "source_run_manifest_sha256": sha256_file(manifest_path),
        "source_complete_sha256": sha256_file(complete_path),
        "source_recovery_sha256": sha256_file(recovery_path),
        "source_closure_sha256": sha256_file(closure_path),
        "source_adoption_manifest_sha256": sha256_file(original_adoption_path),
        "counterfactuals_path": str(payload),
        "counterfactuals_sha256_claimed": claimed_payload_sha,
        "counterfactual_candidate_count": candidate_count,
        "source_project_commit": manifest.get("project_commit"),
        "upstream_commit": manifest.get("upstream_commit"),
        "serialization_rerun": False,
        "lineage_resolution_rerun": False,
        "downstream_common_recourse_rerun": True,
        "downstream_chemistry_rerun": True,
        "downstream_unified_evaluation_rerun": True,
        "source_checksums": original_adoption.get("source_checksums"),
        "validated_at": _utc_now(),
    }


def build_stage_commands(
    inputs: ContinuationInputs,
    *,
    project_commit: str,
    candidate_count: int,
    teacher_sha256: str,
) -> list[tuple[str, list[str], Path, str]]:
    """Return ordered stage commands and their required completion markers."""

    contract = DATASET_CONTRACTS[inputs.dataset]
    python = sys.executable
    source_args: list[str] = []
    if inputs.source_csv is not None:
        source_args = ["--source-csv", str(inputs.source_csv)]
    common = inputs.output_root / "common_recourse"
    chemistry = inputs.output_root / "chemistry"
    evaluation = inputs.output_root / "unified_eval"
    gate = inputs.output_root / "full_gate"
    standardized = inputs.output_root / "standardized"
    trace = inputs.source_generation_root / "trace"
    counterfactuals_sha = _load_object(
        inputs.source_generation_root / "run_manifest.json"
    )["counterfactuals_sha256"]

    common_argv = [
        python,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/run_common_recourse.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        inputs.dataset,
        "--mode",
        "full",
        "--upstream-root",
        str(inputs.upstream_root),
        "--dataset-dir",
        str(inputs.dataset_dir),
        *source_args,
        "--generation-dir",
        str(inputs.source_generation_root),
        "--distance-checkpoint",
        str(inputs.distance_checkpoint),
        "--output-dir",
        str(common),
        "--parent-limit",
        str(contract["generation_parent_limit"]),
        "--device",
        inputs.device,
    ]
    chemistry_argv = [
        python,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--project-root",
        str(PROJECT_ROOT),
        "--dataset",
        inputs.dataset,
        "--dataset-dir",
        str(inputs.dataset_dir),
        *source_args,
        "--generation-dir",
        str(inputs.source_generation_root),
        "--trace-lineage-path",
        str(trace / "candidate_action_lineage.json"),
        "--trace-evidence-path",
        str(trace / "trace_summary.json"),
        "--common-recourse-dir",
        str(common),
        "--output-dir",
        str(chemistry),
        "--preregistration-path",
        str(inputs.output_root / "preregistration/deterministic_chem_repair.json"),
        "--parent-limit",
        str(contract["generation_parent_limit"]),
        "--expected-candidate-count",
        str(candidate_count),
        "--expected-counterfactuals-sha256",
        str(counterfactuals_sha),
    ]
    evaluation_argv = [
        python,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/run_slot_unified_eval.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        inputs.dataset,
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
        str(contract["evaluation_parent_count"]),
        "--max-k",
        "20",
        "--device",
        inputs.device,
    ]
    if inputs.theta_star is not None:
        evaluation_argv.extend(["--theta-star", format(inputs.theta_star, ".17g")])
    if inputs.cost_cap is not None:
        evaluation_argv.extend(["--cost-cap", format(inputs.cost_cap, ".17g")])
    gate_argv = [
        python,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/gate_recovery.py"),
        "--stage",
        "project-full",
        "--dataset",
        inputs.dataset,
        "--expected-parent-count",
        str(contract["evaluation_parent_count"]),
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
        python,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/freeze_recovery_result.py"),
        "--dataset",
        inputs.dataset,
        "--source-dir",
        str(evaluation),
        "--gate-dir",
        str(gate),
        "--output-dir",
        str(standardized),
    ]
    return [
        ("common_recourse", common_argv, common / "_RUN_COMPLETE.json", "run_complete"),
        ("chemistry", chemistry_argv, chemistry / "_RUN_COMPLETE.json", "run_complete"),
        ("unified_eval", evaluation_argv, evaluation / "_RUN_COMPLETE.json", "run_complete"),
        ("full_gate", gate_argv, gate / "gate_result.json", "audit_passed"),
        ("freeze", freeze_argv, standardized / "_FINALIZED.json", "finalized"),
    ]


def _run_stage(
    *,
    stage: str,
    argv: Sequence[str],
    marker: Path,
    required_field: str,
    environment: Mapping[str, str],
    output_root: Path,
) -> None:
    write_json(
        output_root / "stage_state.json",
        {
            "schema_version": 1,
            "status": "RUNNING",
            "stage": stage,
            "argv_sha256": stable_json_sha256(list(argv)),
            "started_at": _utc_now(),
        },
    )
    subprocess.run(
        list(argv),
        cwd=PROJECT_ROOT,
        env=dict(environment),
        check=True,
    )
    payload = _load_object(_require_file(marker))
    if payload.get(required_field) is not True:
        raise ValueError(
            f"Stage {stage} completion field {required_field!r} is not true: {marker}"
        )
    write_json(
        output_root / "stage_state.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "stage": stage,
            "marker": str(marker),
            "marker_sha256": sha256_file(marker),
            "completed_at": _utc_now(),
        },
    )


def run_continuation(inputs: ContinuationInputs) -> dict[str, Any]:
    if inputs.output_root.exists():
        raise FileExistsError(f"Fresh OUTPUT_ROOT already exists: {inputs.output_root}")
    inputs.output_root.parent.mkdir(parents=True, exist_ok=True)
    inputs.output_root.mkdir(mode=0o755)
    try:
        adoption = validate_adopted_generation(inputs)
        checkout = verify_checkout(
            inputs.upstream_root,
            expected_commit=UPSTREAM_COMMIT,
            validate_imports=True,
        )
        write_json(inputs.output_root / "generation_adoption_manifest.json", adoption)
        write_json(inputs.output_root / "upstream_checkout_audit.json", checkout)
        project_commit = _git_head()
        # The frozen teacher is hashed exactly once here.  The shared evaluator
        # performs its own scientific input check; this driver reuses the same
        # identity for the downstream gate and final provenance instead of
        # rescanning a potentially large model repeatedly.
        teacher_sha256 = sha256_file(inputs.teacher_path)
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        commands = build_stage_commands(
            inputs,
            project_commit=project_commit,
            candidate_count=int(adoption["counterfactual_candidate_count"]),
            teacher_sha256=teacher_sha256,
        )
        for stage, argv, marker, field in commands:
            _run_stage(
                stage=stage,
                argv=argv,
                marker=marker,
                required_field=field,
                environment=environment,
                output_root=inputs.output_root,
            )

        standardized = inputs.output_root / "standardized"
        source_manifest = _load_object(standardized / "run_manifest.json")
        freeze_manifest = _load_object(standardized / "freeze_manifest.json")
        if source_manifest.get("dataset_key") != inputs.dataset:
            raise ValueError("Standardized dataset identity mismatch")
        if source_manifest.get("cf_mode") != CF_MODE:
            raise ValueError("Standardized counterfactual mode mismatch")
        if source_manifest.get("distance_line") != DISTANCE_LINE:
            raise ValueError("Standardized distance line mismatch")
        if source_manifest.get("teacher_sha256") != teacher_sha256:
            raise ValueError("Standardized frozen teacher identity mismatch")
        if freeze_manifest.get("dataset_key") != inputs.dataset:
            raise ValueError("Freeze dataset identity mismatch")

        final = {
            "schema_version": 1,
            "status": "PASS",
            "dataset": inputs.dataset,
            "method": METHOD,
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "rf_oracle_used": True,
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "generation_adopted": True,
            "generation_rerun": False,
            "ordering_adopted": False,
            "evaluation_adopted": False,
            "source_generation_root": str(inputs.source_generation_root),
            "standardized_output_root": str(standardized),
            "project_commit": project_commit,
            "source_generation_manifest_sha256": adoption[
                "source_run_manifest_sha256"
            ],
            "standardized_run_manifest_sha256": sha256_file(
                standardized / "run_manifest.json"
            ),
            "freeze_manifest_sha256": sha256_file(
                standardized / "freeze_manifest.json"
            ),
            "teacher_sha256": source_manifest.get("teacher_sha256"),
            "molclr_checkpoint_sha256": source_manifest.get(
                "molclr_checkpoint_sha256"
            ),
            "dataset_csv_sha256": source_manifest.get("dataset_csv_sha256"),
            "completed_at": _utc_now(),
        }
        write_json(inputs.output_root / "run_manifest.json", final)
        write_json(inputs.output_root / "final_gate.json", final)
        write_json(inputs.output_root / "_RUN_COMPLETE.json", {**final, "run_complete": True})
        atomic_write_bytes(inputs.output_root / "PASS", b"PASS\n")
        print(f"[COMRECGC_STANDARDIZED_CONTINUATION_PASS] dataset={inputs.dataset}")
        return final
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "FAILED",
            "dataset": inputs.dataset,
            "error_class": type(exc).__name__,
            "message": str(exc),
            "output_root": str(inputs.output_root),
            "failed_at": _utc_now(),
        }
        write_json(inputs.output_root / "FAILED.json", failure)
        raise


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONTRACTS), required=True)
    parser.add_argument("--source-generation-root", type=_absolute, required=True)
    parser.add_argument("--upstream-root", type=_absolute, required=True)
    parser.add_argument("--dataset-dir", type=_absolute, required=True)
    parser.add_argument("--source-csv", type=_absolute)
    parser.add_argument("--distance-checkpoint", type=_absolute, required=True)
    parser.add_argument("--dataset-csv", type=_absolute, required=True)
    parser.add_argument("--teacher-path", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--thresholds-path", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--theta-star", type=float)
    parser.add_argument("--cost-cap", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "aids" and args.source_csv is None:
        raise SystemExit("AIDS requires --source-csv")
    if args.dataset == "mutagenicity" and args.source_csv is not None:
        raise SystemExit("Mutagenicity does not accept --source-csv")
    values = ContinuationInputs(
        dataset=args.dataset,
        source_generation_root=args.source_generation_root,
        upstream_root=_require_directory(args.upstream_root),
        dataset_dir=_require_directory(args.dataset_dir),
        source_csv=_require_file(args.source_csv) if args.source_csv else None,
        distance_checkpoint=_require_file(args.distance_checkpoint),
        dataset_csv=_require_file(args.dataset_csv),
        teacher_path=_require_file(args.teacher_path),
        molclr_root=_require_directory(args.molclr_root),
        molclr_checkpoint=_require_file(args.molclr_checkpoint),
        thresholds_path=_require_file(args.thresholds_path),
        output_root=args.output_root,
        device=str(args.device),
        theta_star=args.theta_star,
        cost_cap=args.cost_cap,
    )
    run_continuation(values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
