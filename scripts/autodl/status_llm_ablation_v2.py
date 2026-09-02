#!/usr/bin/env python3
"""Evaluate LLM stage/scale framework and early-GPU gates without science."""

from __future__ import annotations

import argparse
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

from src.ablations.llm.early_launch_gate import (  # noqa: E402
    EarlyLaunchSnapshot,
    EarlyRunAuthorizationReceipt,
    evaluate_early_launch_gate,
)
from src.ablations.llm.model_scale_registry import load_model_scale_registry  # noqa: E402
from src.ablations.llm.runtime_evidence import (  # noqa: E402
    evaluate_runtime_model_evidence,
    load_bace_reference_v2,
    runtime_run_contract_sha256,
    sha256_file,
    validate_stage_config_against_reference,
)
from src.ablations.llm.stage_scale import (  # noqa: E402
    LLMScaleVariant,
    LLMStageVariant,
    validate_non_factorial_design,
)


def _load(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _physical_identity(path_like: Path, *, role: str) -> dict[str, Any]:
    if not path_like.is_absolute() or path_like.is_symlink():
        raise ValueError(f"{role} must be an absolute physical file")
    path = path_like.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{role} is not a regular file")
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def _optional_pair(path: Path | None, sha256: str | None, *, role: str):
    if (path is None) != (sha256 is None):
        raise ValueError(f"{role} requires both path and SHA256")
    return None if path is None else (str(path), str(sha256))


def _execution_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("deployed checkout did not resolve to one Git commit")
    return commit


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
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=PROJECT_ROOT / "configs/ablations/llm/bace_ours_stage_ablation_v2.yaml",
    )
    parser.add_argument(
        "--scale-config",
        type=Path,
        default=PROJECT_ROOT / "configs/ablations/llm/bace_ours_scale_ablation_v2.yaml",
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=PROJECT_ROOT / "configs/ablations/llm/chemllm_model_scale_registry_v2.yaml",
    )
    parser.add_argument("--main-snapshot", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--reference-contract-sha256", required=True)
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

    stage = _load(args.stage_config)
    scale = _load(args.scale_config)
    registry = load_model_scale_registry(_load(args.model_registry))
    if stage.get("schema_version") != "bace_ours_llm_stage_ablation_v2":
        raise ValueError("stage config schema changed")
    if scale.get("schema_version") != "bace_ours_llm_scale_ablation_v2":
        raise ValueError("scale config schema changed")
    validate_non_factorial_design(
        stage_variants=[row["variant"] for row in stage["design"]["variants"]],
        scale_variants=[item.value for item in LLMScaleVariant],
        scale_stage_full_factorial=bool(scale["scale_stage_full_factorial"]),
    )
    reference = load_bace_reference_v2(
        args.reference_contract, args.reference_contract_sha256
    )
    validate_stage_config_against_reference(stage, reference)
    snapshot = EarlyLaunchSnapshot.from_mapping(_load(args.main_snapshot))
    matrix_identity = _physical_identity(
        Path(snapshot.matrix_authority_path), role="matrix_authority_evidence"
    )
    if matrix_identity["sha256"] != snapshot.matrix_authority_sha256:
        raise ValueError("main snapshot matrix-authority SHA changed")
    model_evidence = evaluate_runtime_model_evidence(
        registry,
        two_b_snapshot=_optional_pair(
            args.two_b_snapshot_manifest,
            args.two_b_snapshot_manifest_sha256,
            role="2B snapshot evidence",
        ),
        two_b_parameter_report=_optional_pair(
            args.two_b_parameter_report,
            args.two_b_parameter_report_sha256,
            role="2B actual parameter report",
        ),
        seven_b_parameter_report=_optional_pair(
            args.seven_b_parameter_report,
            args.seven_b_parameter_report_sha256,
            role="7B actual parameter report",
        ),
        twenty_b_metadata=_optional_pair(
            args.twenty_b_metadata_manifest,
            args.twenty_b_metadata_manifest_sha256,
            role="20B metadata evidence",
        ),
    )
    commit = _execution_commit()
    contract_files = {
        "stage_config": _physical_identity(args.stage_config.resolve(), role="stage config"),
        "scale_config": _physical_identity(args.scale_config.resolve(), role="scale config"),
        "model_registry": _physical_identity(
            args.model_registry.resolve(), role="model registry"
        ),
        "reference_contract": {
            "path": reference.path,
            "sha256": reference.file_sha256,
            "size": Path(reference.path).stat().st_size,
        },
    }
    for name, path in (
        ("two_b_snapshot", args.two_b_snapshot_manifest),
        ("two_b_parameter_report", args.two_b_parameter_report),
        ("seven_b_parameter_report", args.seven_b_parameter_report),
        ("twenty_b_metadata", args.twenty_b_metadata_manifest),
    ):
        if path is not None:
            contract_files[name] = _physical_identity(path, role=name)
    run_contract_sha = runtime_run_contract_sha256(
        file_identities=contract_files, execution_commit=commit
    )
    receipt = (
        EarlyRunAuthorizationReceipt(**_load(args.early_run_receipt))
        if args.early_run_receipt is not None
        else None
    )
    decision = evaluate_early_launch_gate(
        snapshot,
        receipt=receipt,
        matrix_authority_sha256=matrix_identity["sha256"],
        run_contract_sha256=run_contract_sha,
        execution_commit=commit,
        runtime_evidence_ready=bool(model_evidence["runtime_science_ready"]),
        science_entrypoint_available=False,
    )
    payload = {
        "schema_version": "llm_stage_scale_status_v2",
        "framework_build_allowed": True,
        "framework_build_only": True,
        "launcher_state": "CONFIG_ONLY_BLOCKED_NO_SCIENCE_ENTRYPOINT",
        "science_started": False,
        "gpu_lock_acquired": False,
        "stage_variants": [item.value for item in LLMStageVariant],
        "stage_availability": {
            row["id"]: {
                "variant": row["variant"],
                "availability": row["availability"],
                "blocker": row.get("blocker"),
                "observed_main_stage": row.get("observed_main_stage"),
            }
            for row in stage["design"]["variants"]
        },
        "scale_variants": [item.value for item in LLMScaleVariant],
        "scale_primary_state": scale["primary_comparison"]["state"],
        "scale_fallback_state": scale["fallback_comparison"]["state"],
        "model_registry_states": {key: value.status for key, value in registry.items()},
        "model_runtime_evidence": model_evidence,
        "reference_contract": {
            "path": reference.path,
            "file_sha256": reference.file_sha256,
            "self_sha256": reference.self_sha256,
        },
        "matrix_authority_evidence": matrix_identity,
        "execution_commit": commit,
        "runtime_run_contract_sha256": run_contract_sha,
        "early_launch": decision.to_dict(),
        "gnn_science_started": False,
    }
    if args.output is not None:
        _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
