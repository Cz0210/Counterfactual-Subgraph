"""Post-main launch decision for the five-backbone BACE ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.ablations.gnn.five_backbone import (
    FIVE_BACKBONES,
    FiveBackboneConfig,
    validate_proposal_fixed_runtime_manifest,
)
from src.ablations.launch_gate import LaunchGateDecision


@dataclass(frozen=True, slots=True)
class FiveBackboneLaunchDecision:
    state: str
    science_launch_allowed: bool
    main_gate_pass: bool
    user_authorized_after_16: bool
    run_requested: bool
    no_main_task_waiting_for_gpu: bool
    proposal_fixed_manifest_pass: bool
    gps_runtime_pass: bool
    max_concurrent_gpus: int
    phase1_seed: int
    backbones: tuple[str, ...]
    schedule: Mapping[str, tuple[str, ...]]
    blockers: tuple[str, ...]
    graph_mamba_run_enabled: bool = False
    schema_version: str = "gnn_five_backbone_launch_decision_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backbones"] = list(self.backbones)
        payload["schedule"] = {
            lane: list(backbones) for lane, backbones in self.schedule.items()
        }
        payload["blockers"] = list(self.blockers)
        return payload


def _main_queue_empty(payload: Mapping[str, Any] | None) -> bool:
    if payload is None:
        # Exact 16/16 plus all final publication receipts means no main method
        # cell remains eligible to request a GPU.  A supplied live queue still
        # takes precedence and can block launch.
        return True
    tasks = payload.get("ready_waiting_gpu", payload.get("ready_gpu_tasks"))
    return (
        payload.get("status") in {"PASS", "READY"}
        and isinstance(tasks, list)
        and not tasks
    )


def evaluate_five_backbone_launch(
    *,
    config: FiveBackboneConfig,
    main_gate: LaunchGateDecision,
    allow_after_16: bool,
    run_requested: bool,
    main_ready_gpu_tasks: Mapping[str, Any] | None,
    proposal_manifest: Mapping[str, Any] | None,
    gps_runtime_capabilities: Mapping[str, Any],
) -> FiveBackboneLaunchDecision:
    """Combine the hash-closed main gate with five-backbone runtime gates."""

    main_gate_pass = bool(
        main_gate.authority_verified
        and main_gate.main_matrix_complete_cells == 16
        and main_gate.main_matrix_total_cells == 16
        and main_gate.final_audit_pass
        and main_gate.figure3_pass
        and main_gate.figure4_pass
        and main_gate.table2_pass
        and main_gate.artifact_receipts_bound
        and not main_gate.evidence_errors
    )
    no_main_waiter = _main_queue_empty(main_ready_gpu_tasks)
    proposal_pass = False
    if proposal_manifest is not None:
        try:
            validate_proposal_fixed_runtime_manifest(proposal_manifest)
            proposal_pass = True
        except ValueError:
            proposal_pass = False
    gps_pass = bool(
        gps_runtime_capabilities.get("torch_available")
        and gps_runtime_capabilities.get("torch_geometric_available")
        and gps_runtime_capabilities.get("gpsconv_available")
        and gps_runtime_capabilities.get("add_random_walk_pe_available")
    )
    blockers: list[str] = []
    if not main_gate_pass:
        blockers.append("WAITING_HASH_CLOSED_MAIN_16_OF_16_AND_FINAL_EXPORTS")
    if not allow_after_16:
        blockers.append("ALLOW_GNN_ABLATION_RUN_AFTER_16_NOT_SET")
    if not run_requested:
        blockers.append("RUN_GNN_ABLATION_NOT_SET")
    if not no_main_waiter:
        blockers.append("MAIN_TASK_READY_WAITING_GPU")
    if not proposal_pass:
        blockers.append("BACE_OURS_PROPOSAL_FIXED_MANIFEST_MISSING_OR_INVALID")
    if not gps_pass:
        blockers.append("PYG_GPSCONV_OR_RANDOM_WALK_PE_UNAVAILABLE")
    if config.backbones != FIVE_BACKBONES or config.max_concurrent_gpus != 2:
        blockers.append("FIVE_BACKBONE_CONFIG_CHANGED")
    if config.graph_mamba_metadata.get("run_enabled") is not False:
        blockers.append("GRAPH_MAMBA_MUST_REMAIN_DISABLED")
    allowed = not blockers
    return FiveBackboneLaunchDecision(
        state=(
            "AUTHORIZED_TO_LAUNCH_FIVE_BACKBONE_PHASE1"
            if allowed
            else "BLOCKED_GNN_FIVE_BACKBONE_GATE"
        ),
        science_launch_allowed=allowed,
        main_gate_pass=main_gate_pass,
        user_authorized_after_16=bool(allow_after_16),
        run_requested=bool(run_requested),
        no_main_task_waiting_for_gpu=no_main_waiter,
        proposal_fixed_manifest_pass=proposal_pass,
        gps_runtime_pass=gps_pass,
        max_concurrent_gpus=config.max_concurrent_gpus,
        phase1_seed=config.primary_seed,
        backbones=config.backbones,
        schedule={
            "lane0": ("gine", "gin", "gps"),
            "lane1": ("gcn", "gatv2"),
        },
        blockers=tuple(blockers),
    )


__all__ = [
    "FiveBackboneLaunchDecision",
    "evaluate_five_backbone_launch",
]
