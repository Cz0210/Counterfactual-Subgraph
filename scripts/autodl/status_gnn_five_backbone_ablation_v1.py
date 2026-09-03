#!/usr/bin/env python3
"""Report the strict post-16/16 five-backbone GNN launch gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.gnn.five_backbone import (  # noqa: E402
    build_five_backbone_plan,
    load_five_backbone_config,
)
from src.ablations.gnn.five_backbone_launch import (  # noqa: E402
    evaluate_five_backbone_launch,
)
from src.ablations.launch_gate import (  # noqa: E402
    evaluate_launch_gate,
    load_json_object,
)
from src.models.gatedgcn_plus_backbone import (  # noqa: E402
    gatedgcn_plus_runtime_capabilities,
)


DEFAULT_ABLATION_CONFIG = (
    PROJECT_ROOT
    / "configs/ablations/gnn/bace_ours_proposal_fixed_five_backbones_v1.yaml"
)


def _optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return load_json_object(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime_config = Path(args.config).expanduser()
    if not runtime_config.is_absolute():
        runtime_config = (PROJECT_ROOT / runtime_config).resolve(strict=False)
    if not runtime_config.is_file():
        raise ValueError(f"runtime config is absent: {runtime_config}")
    config = load_five_backbone_config(
        args.ablation_config, project_root=PROJECT_ROOT
    )
    authority = load_json_object(args.matrix_authority)
    main_gate = evaluate_launch_gate(
        family="gnn",
        matrix_authority=authority,
        final_audit=_optional(args.final_audit),
        figure3=_optional(args.figure3_pass),
        figure4=_optional(args.figure4_pass),
        table2=_optional(args.table2_pass),
        authorization_receipt=_optional(args.authorization_receipt),
        # The current project-owner instruction authorizes GNN after 16/16;
        # its exact environment flag is evaluated by the dedicated gate below.
        run_requested=bool(args.run_requested),
    )
    capabilities = gatedgcn_plus_runtime_capabilities()
    decision = evaluate_five_backbone_launch(
        config=config,
        main_gate=main_gate,
        allow_after_16=bool(args.allow_after_16),
        run_requested=bool(args.run_requested),
        main_ready_gpu_tasks=_optional(args.main_ready_gpu_tasks),
        proposal_manifest=_optional(args.proposal_manifest),
        gatedgcn_plus_runtime_capabilities=capabilities,
    )
    plan = build_five_backbone_plan(config)
    payload = {
        **decision.to_dict(),
        "main_gate": main_gate.to_dict(),
        "ablation_config": config.source_path,
        "ablation_config_sha256": config.source_sha256,
        "plan_sha256": plan["plan_sha256"],
        "gatedgcn_plus_runtime_capabilities": capabilities,
        "authorization_source": (
            "USER_DIRECTIVE_ALLOW_GNN_ABLATION_RUN_AFTER_16"
            if args.allow_after_16
            else None
        ),
        "launcher_executes_science": False,
        "launcher_role": "GATE_AND_TWO_LANE_SCHEDULE_EMITTER",
        "graph_mamba_registered": True,
        "graph_mamba_run_enabled": False,
        "gpu_lock_acquired": False,
        "main_matrix_modified": False,
    }
    if args.output is not None:
        _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--ablation-config", type=Path, default=DEFAULT_ABLATION_CONFIG)
    parser.add_argument("--matrix-authority", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path)
    parser.add_argument("--figure3-pass", type=Path)
    parser.add_argument("--figure4-pass", type=Path)
    parser.add_argument("--table2-pass", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--main-ready-gpu-tasks", type=Path)
    parser.add_argument("--proposal-manifest", type=Path)
    parser.add_argument("--allow-after-16", action="store_true")
    parser.add_argument("--run-requested", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
