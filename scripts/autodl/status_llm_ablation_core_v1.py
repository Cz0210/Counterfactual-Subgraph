#!/usr/bin/env python3
"""Read-only, evidence-bound launch/status decision for one core LLM variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.llm.contracts import canonical_json_sha256  # noqa: E402
from src.ablations.llm.core_execution import (  # noqa: E402
    CoreLLMVariant,
    derive_core_reference,
    load_core_run_spec,
    status_core_run,
    validate_variant_artifact_bindings,
)
from src.ablations.llm.early_launch_gate import (  # noqa: E402
    EarlyLaunchSnapshot,
    EarlyRunAuthorizationReceipt,
    evaluate_early_launch_gate,
)
from src.ablations.llm.final16_owner_evidence import (  # noqa: E402
    assert_snapshot_matches_owner_coverage,
    evaluate_final16_owner_coverage,
)
from src.ablations.llm.model_scale_registry import load_model_scale_registry  # noqa: E402
from src.ablations.llm.runtime_evidence import (  # noqa: E402
    evaluate_runtime_model_evidence,
    load_bace_reference_v2,
    sha256_file,
    validate_off_the_shelf_7b_parameter_report,
)
from src.ablations.launch_gate import validate_matrix_authority_pointer  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    payload = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix.lower() in {".yaml", ".yml"}
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected one object: {path}")
    return payload


def _identity(path_like: Path, *, role: str) -> dict[str, Any]:
    if not path_like.is_absolute() or path_like.is_symlink():
        raise ValueError(f"{role} must be an absolute physical file")
    path = path_like.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{role} is not a regular file")
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def _optional_pair(path: Path | None, sha: str | None, *, role: str):
    if (path is None) != (sha is None):
        raise ValueError(f"{role} requires path and SHA256 together")
    return None if path is None else (str(path), str(sha))


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    value = result.stdout.strip().lower()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("checkout does not resolve to a full Git commit")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--run-spec-sha256", required=True)
    parser.add_argument("--main-snapshot", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--reference-contract-sha256", required=True)
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=PROJECT_ROOT / "configs/ablations/llm/chemllm_model_scale_registry_v2.yaml",
    )
    parser.add_argument("--two-b-snapshot-manifest", type=Path)
    parser.add_argument("--two-b-snapshot-manifest-sha256")
    parser.add_argument("--two-b-parameter-report", type=Path)
    parser.add_argument("--two-b-parameter-report-sha256")
    parser.add_argument("--seven-b-parameter-report", type=Path)
    parser.add_argument("--seven-b-parameter-report-sha256")
    parser.add_argument("--twenty-b-metadata-manifest", type=Path)
    parser.add_argument("--twenty-b-metadata-manifest-sha256")
    parser.add_argument("--early-run-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    commit = _commit()
    spec = load_core_run_spec(args.run_spec, args.run_spec_sha256)
    run_spec_identity = _identity(args.run_spec, role="core run spec")
    if spec.execution_commit != commit:
        raise ValueError("run spec execution commit differs from deployed checkout")
    reference = load_bace_reference_v2(
        args.reference_contract, args.reference_contract_sha256
    )
    core_reference = derive_core_reference(reference)
    validate_variant_artifact_bindings(spec, reference)
    if (
        spec.reference_contract.path != reference.path
        or spec.reference_contract.sha256 != reference.file_sha256
    ):
        raise ValueError("run spec reference binding differs from reopened reference")
    snapshot_identity = _identity(args.main_snapshot, role="main snapshot")
    snapshot = EarlyLaunchSnapshot.from_mapping(_load(args.main_snapshot))
    matrix_identity = _identity(Path(snapshot.matrix_authority_path), role="matrix authority")
    if matrix_identity["sha256"] != snapshot.matrix_authority_sha256:
        raise ValueError("snapshot matrix authority SHA changed")
    matrix_authority = validate_matrix_authority_pointer(
        _load(Path(snapshot.matrix_authority_path))
    )
    matrix_authority["pointer_root"] = str(
        Path(snapshot.matrix_authority_path).resolve().parent
    )
    owner_registry_identity = _identity(
        Path(snapshot.main_owner_registry_path), role="canonical final16 owner registry"
    )
    if owner_registry_identity["sha256"] != snapshot.main_owner_registry_sha256:
        raise ValueError("snapshot canonical owner-registry file SHA changed")
    owner_coverage = evaluate_final16_owner_coverage(
        authority=matrix_authority,
        owner_registry=_load(Path(snapshot.main_owner_registry_path)),
    )
    assert_snapshot_matches_owner_coverage(snapshot, owner_coverage)
    if spec.matrix_authority.sha256 != matrix_identity["sha256"]:
        raise ValueError("run spec matrix authority differs from live authority")

    registry = load_model_scale_registry(_load(args.model_registry))
    evidence = evaluate_runtime_model_evidence(
        registry,
        two_b_snapshot=_optional_pair(
            args.two_b_snapshot_manifest,
            args.two_b_snapshot_manifest_sha256,
            role="2B snapshot",
        ),
        two_b_parameter_report=_optional_pair(
            args.two_b_parameter_report,
            args.two_b_parameter_report_sha256,
            role="2B parameter report",
        ),
        # The legacy 7B row is base+PPO.  The core scale row needs a distinct
        # base-only report and is validated immediately below.
        seven_b_parameter_report=None,
        twenty_b_metadata=_optional_pair(
            args.twenty_b_metadata_manifest,
            args.twenty_b_metadata_manifest_sha256,
            role="20B metadata",
        ),
    )
    seven_b_ots_state = "BLOCKED_MISSING_7B_OFF_THE_SHELF_PARAMETER_REPORT"
    seven_b_pair = _optional_pair(
        args.seven_b_parameter_report,
        args.seven_b_parameter_report_sha256,
        role="7B off-the-shelf parameter report",
    )
    if seven_b_pair is not None:
        validate_off_the_shelf_7b_parameter_report(*seven_b_pair)
        seven_b_ots_state = "RUNTIME_EVIDENCE_PASS"
    target_state = {
        CoreLLMVariant.BRICS_FIXED: "ARTIFACT_BOUND",
        CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN: "MAIN_RESULT_BOUND",
        CoreLLMVariant.CHEMLLM_7B_OFF_THE_SHELF: seven_b_ots_state,
        CoreLLMVariant.CHEMLLM_2B_OFF_THE_SHELF: evidence["states"]["chemllm_2b_1_5"],
    }[spec.variant]
    target_ready = target_state in {
        "ARTIFACT_BOUND",
        "MAIN_RESULT_BOUND",
        "RUNTIME_EVIDENCE_PASS",
    }
    runner_identity = _identity(
        PROJECT_ROOT / "scripts/autodl/run_llm_ablation_variant.py", role="science entrypoint"
    )
    runtime_evidence_files: dict[str, Any] = {}
    for name, path in (
        ("two_b_snapshot", args.two_b_snapshot_manifest),
        ("two_b_parameter_report", args.two_b_parameter_report),
        ("seven_b_off_the_shelf_parameter_report", args.seven_b_parameter_report),
        ("twenty_b_metadata", args.twenty_b_metadata_manifest),
    ):
        if path is not None:
            runtime_evidence_files[name] = _identity(path, role=name)
    run_contract_sha = canonical_json_sha256(
        {
            "schema_version": "llm_core_runtime_contract_v1",
            "execution_commit": commit,
            "run_spec_file_sha256": run_spec_identity["sha256"],
            "run_spec_self_sha256": spec.run_spec_sha256,
            "reference_file_sha256": reference.file_sha256,
            "core_reference_sha256": core_reference["core_reference_sha256"],
            "science_entrypoint_sha256": runner_identity["sha256"],
            "target_runtime_state": target_state,
            "runtime_evidence_files": runtime_evidence_files,
            "canonical_owner_registry_file_sha256": owner_registry_identity["sha256"],
            "canonical_owner_registry_self_sha256": owner_coverage.registry_self_sha256,
            "owner_coverage_sha256": owner_coverage.to_dict()["coverage_sha256"],
        }
    )
    receipt = (
        EarlyRunAuthorizationReceipt(**_load(args.early_run_receipt))
        if args.early_run_receipt is not None
        else None
    )
    gate = evaluate_early_launch_gate(
        snapshot,
        receipt=receipt,
        matrix_authority_sha256=matrix_identity["sha256"],
        run_contract_sha256=run_contract_sha,
        execution_commit=commit,
        runtime_evidence_ready=target_ready,
        science_entrypoint_available=True,
    )
    authorization_identity = (
        _identity(args.early_run_receipt, role="authorization receipt")
        if args.early_run_receipt is not None
        else None
    )
    payload: dict[str, Any] = {
        "schema_version": "llm_core_launch_decision_v1",
        "variant": spec.variant.value,
        "run_id": spec.run_id,
        "run_spec_sha256": spec.run_spec_sha256,
        "run_spec": run_spec_identity,
        "execution_commit": commit,
        "science_entrypoint": runner_identity,
        "science_entrypoint_available": True,
        "checkpoint_resume_supported": True,
        "target_runtime_state": target_state,
        "core_reference": core_reference,
        "main_snapshot": snapshot_identity,
        "canonical_owner_registry": owner_registry_identity,
        "main_owner_coverage": owner_coverage.to_dict(),
        "matrix_authority": matrix_identity,
        "reference_contract": {
            "path": reference.path,
            "sha256": reference.file_sha256,
            "size": Path(reference.path).stat().st_size,
        },
        "runtime_run_contract_sha256": run_contract_sha,
        "runtime_evidence_files": runtime_evidence_files,
        "authorization_receipt": authorization_identity,
        "early_launch": gate.to_dict(),
        "science_launch_allowed": gate.science_launch_allowed,
        "assigned_gpu": gate.assigned_gpu,
        "run_status": status_core_run(spec),
        "sft_auxiliary_state": core_reference["sft_auxiliary"]["state"],
        "sft_auxiliary_reason": core_reference["sft_auxiliary"]["reason"],
        "science_started_by_status": False,
    }
    payload["launch_decision_sha256"] = canonical_json_sha256(payload)
    if args.output is not None:
        _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
