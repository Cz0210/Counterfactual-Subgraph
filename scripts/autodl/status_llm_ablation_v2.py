#!/usr/bin/env python3
"""Evaluate LLM stage/scale framework and early-GPU gates without science."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    snapshot = EarlyLaunchSnapshot.from_mapping(_load(args.main_snapshot))
    receipt = (
        EarlyRunAuthorizationReceipt(**_load(args.early_run_receipt))
        if args.early_run_receipt is not None
        else None
    )
    decision = evaluate_early_launch_gate(snapshot, receipt=receipt)
    payload = {
        "schema_version": "llm_stage_scale_status_v2",
        "framework_build_allowed": True,
        "framework_build_only": True,
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
        "early_launch": decision.to_dict(),
        "gnn_science_started": False,
    }
    if args.output is not None:
        _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
