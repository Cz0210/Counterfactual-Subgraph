"""2026-09-05 project-owner scheduling policy; no process or matrix writes."""
from __future__ import annotations

CORE = ("gine", "gin", "gcn", "gatv2", "gatedgcn_plus")


def hpc_cpu_allowed(*, main_cells: int, bace_reference_pass: bool, active_jobs: int):
    blockers = []
    if main_cells < 12:
        blockers.append("MAIN_MATRIX_BELOW_12")
    if not bace_reference_pass:
        blockers.append("BACE_REFERENCE_UNVERIFIED")
    if active_jobs >= 2:
        blockers.append("MAX_TWO_HPC_GNN_JOBS")
    return {"allowed": not blockers, "blockers": blockers, "gpu_requested": False,
            "main_matrix_write_allowed": False, "priority": 70}


def gpu_allowed(evidence, *, family):
    if family not in ("gnn", "llm", "ours_reach", "globalgce_chemaligned"):
        raise ValueError("unknown ablation family")
    blockers = []
    if family == "gnn" and evidence.get("main_cells", 0) < 12:
        blockers.append("MAIN_MATRIX_BELOW_12")
    for field in ("owners_healthy", "registry_healthy", "memory_safe", "storage_safe", "checkpoint_resume_pass"):
        if evidence.get(field) is not True:
            blockers.append(field.upper())
    if evidence.get("main_ready_waiting_gpu") is not False:
        blockers.append("MAIN_READY_WAITING_GPU")
    if evidence.get("gpu_main_reservation") is not False:
        blockers.append("MAIN_GPU_RESERVATION")
    if family not in ("ours_reach", "globalgce_chemaligned") and evidence.get("gpu_idle_seconds", 0) < 1200:
        blockers.append("GPU_IDLE_BELOW_1200_SECONDS")
    if family == "ours_reach":
        if evidence.get("ours_reach_contract_verified") is not True:
            blockers.append("OURS_REACH_TRAIN_CONTRACT_UNVERIFIED")
        if (evidence.get("gpu_index") != 0
            or (evidence.get("actual_gpu_observation", {}).get("process_count") != 0
                and not evidence.get("gpu_child_pid"))):
            blockers.append("OURS_REACH_REQUIRES_EXCLUSIVE_GPU0")
    if evidence.get("active_early_ablation_gpus") != 0:
        blockers.append("MAX_ONE_EARLY_ABLATION_GPU")
    if family == "globalgce_chemaligned":
        if evidence.get("chemaligned_contract_verified") is not True:
            blockers.append("CHEMALIGNED_TRAIN_CONTRACT_UNVERIFIED")
        if (evidence.get("gpu_index") != 0 or
            (evidence.get("actual_gpu_observation", {}).get("process_count") != 0
             and not evidence.get("gpu_child_pid"))):
            blockers.append("CHEMALIGNED_REQUIRES_EXCLUSIVE_GPU0")
    if family == "llm" and evidence.get("gnn_core_seed7_audit") != "PASS":
        blockers.append("WAITING_GNN_CORE_SEED7")
    return {"allowed": not blockers, "blockers": blockers, "pause_at_next_checkpoint_on_main_waiter": True}


def core_complete(audit):
    return (audit.get("status") == "PASS" and audit.get("seed") == 7
            and all(audit.get("backbones", {}).get(b) == "PASS" for b in CORE)
            and all(audit.get(k) == "PASS" for k in ("classifier_table", "native_cohort", "common_cohort", "explanation_table")))
