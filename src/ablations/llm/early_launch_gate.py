"""Main-table-priority gate for at most one early LLM-ablation GPU."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import LLMAblationContractError, canonical_json_sha256, require_sha256


MIN_MATRIX_CELLS = 13
MIN_IDLE_GPU_SECONDS = 1200
MAX_EARLY_GPUS = 1


@dataclass(frozen=True, slots=True)
class EarlyLaunchSnapshot:
    matrix_complete_cells: int
    t8_t13_state: str
    t8_t13_science_pid: int | None
    t12_healthy: bool
    t14_healthy: bool
    mut_passed_or_gpu_released: bool
    main_ready_waiting_gpu: tuple[str, ...]
    main_publishers_waiting_gpu: tuple[str, ...]
    idle_gpu: int | None
    idle_gpu_seconds: int
    persistent_free_gb: float
    minimum_persistent_free_gb: float
    memory_available_gb: float
    minimum_memory_available_gb: float
    checkpoint_resume_supported: bool
    requested_early_gpus: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "main_ready_waiting_gpu", tuple(self.main_ready_waiting_gpu))
        object.__setattr__(
            self, "main_publishers_waiting_gpu", tuple(self.main_publishers_waiting_gpu)
        )
        if not 0 <= self.matrix_complete_cells <= 16:
            raise LLMAblationContractError("matrix count must be in [0, 16]")
        if self.requested_early_gpus < 0:
            raise LLMAblationContractError("requested early GPU count must be non-negative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EarlyLaunchSnapshot":
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class EarlyRunAuthorizationReceipt:
    authorization_id: str
    authorized_by: str
    matrix_authority_sha256: str
    snapshot_sha256: str
    run_contract_sha256: str
    execution_commit: str
    allow_early_llm_ablation: bool
    max_gpus: int
    authorization_sha256: str
    schema_version: str = "early_llm_ablation_authorization_receipt_v1"

    def __post_init__(self) -> None:
        if self.authorized_by != "user_project_owner" or not self.authorization_id.strip():
            raise LLMAblationContractError("early receipt requires project-owner identity")
        for field in (
            "matrix_authority_sha256",
            "snapshot_sha256",
            "run_contract_sha256",
        ):
            object.__setattr__(self, field, require_sha256(getattr(self, field), field=field))
        if len(self.execution_commit) != 40 or any(c not in "0123456789abcdef" for c in self.execution_commit):
            raise LLMAblationContractError("execution_commit must be one Git SHA")
        if not self.allow_early_llm_ablation or self.max_gpus != 1:
            raise LLMAblationContractError("early receipt scope must allow exactly one GPU")
        claimed = require_sha256(self.authorization_sha256, field="authorization_sha256")
        body = asdict(self)
        body.pop("authorization_sha256")
        if canonical_json_sha256(body) != claimed:
            raise LLMAblationContractError("early authorization receipt self-hash changed")


@dataclass(frozen=True, slots=True)
class EarlyLaunchDecision:
    state: str
    eligible_for_authorization_receipt: bool
    science_launch_allowed: bool
    assigned_gpu: int | None
    blockers: tuple[str, ...]
    schema_version: str = "early_llm_ablation_gate_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload


def snapshot_sha256(snapshot: EarlyLaunchSnapshot) -> str:
    return canonical_json_sha256(asdict(snapshot))


def evaluate_early_launch_gate(
    snapshot: EarlyLaunchSnapshot,
    *,
    receipt: EarlyRunAuthorizationReceipt | None,
) -> EarlyLaunchDecision:
    blockers: list[str] = []
    if snapshot.matrix_complete_cells < MIN_MATRIX_CELLS:
        blockers.append("MATRIX_BELOW_13")
    if snapshot.t8_t13_state not in {"RUNNING", "PASS"}:
        blockers.append("T8_T13_HAS_NO_SCIENCE_OWNER")
    elif snapshot.t8_t13_state == "RUNNING" and not snapshot.t8_t13_science_pid:
        blockers.append("T8_T13_SCIENCE_PID_MISSING")
    if not snapshot.t12_healthy:
        blockers.append("T12_NOT_HEALTHY")
    if not snapshot.t14_healthy:
        blockers.append("T14_NOT_HEALTHY")
    if not snapshot.mut_passed_or_gpu_released:
        blockers.append("MUT_STILL_NEEDS_GPU")
    if snapshot.main_ready_waiting_gpu:
        blockers.append("MAIN_TASK_READY_WAITING_GPU")
    if snapshot.main_publishers_waiting_gpu:
        blockers.append("MAIN_PUBLISHER_OR_EVALUATOR_WAITING_GPU")
    if snapshot.idle_gpu is None or snapshot.idle_gpu_seconds < MIN_IDLE_GPU_SECONDS:
        blockers.append("NO_GPU_IDLE_FOR_1200_SECONDS")
    if snapshot.persistent_free_gb < snapshot.minimum_persistent_free_gb:
        blockers.append("PERSISTENT_STORAGE_HEADROOM_UNSAFE")
    if snapshot.memory_available_gb < snapshot.minimum_memory_available_gb:
        blockers.append("MEMORY_HEADROOM_UNSAFE")
    if not snapshot.checkpoint_resume_supported:
        blockers.append("ABLATION_CHECKPOINT_RESUME_NOT_READY")
    if snapshot.requested_early_gpus != MAX_EARLY_GPUS:
        blockers.append("EARLY_GPU_REQUEST_MUST_EQUAL_ONE")
    eligible = not blockers
    if receipt is None:
        if eligible:
            blockers.append("EXPLICIT_EARLY_RUN_RECEIPT_REQUIRED")
        state = "READY_FOR_EARLY_RUN_RECEIPT" if eligible else "BLOCKED_MAIN_PRIORITY"
        return EarlyLaunchDecision(state, eligible, False, None, tuple(blockers))
    if receipt.snapshot_sha256 != snapshot_sha256(snapshot):
        blockers.append("EARLY_RECEIPT_SNAPSHOT_CHANGED")
    allowed = eligible and not blockers
    return EarlyLaunchDecision(
        state="AUTHORIZED_TO_START_ONE_LLM_GPU" if allowed else "BLOCKED_MAIN_PRIORITY",
        eligible_for_authorization_receipt=eligible,
        science_launch_allowed=allowed,
        assigned_gpu=snapshot.idle_gpu if allowed else None,
        blockers=tuple(blockers),
    )


def main_priority_runtime_action(
    snapshot: EarlyLaunchSnapshot,
    *,
    ablation_running: bool,
    at_safe_checkpoint: bool,
) -> str:
    """Return a checkpoint-first, non-destructive response to new main work."""

    main_waiting = bool(
        snapshot.main_ready_waiting_gpu or snapshot.main_publishers_waiting_gpu
    )
    if not ablation_running or not main_waiting:
        return "NO_ACTION"
    if at_safe_checkpoint:
        return "GRACEFUL_PAUSE_AND_RELEASE_GPU"
    return "REQUEST_CHECKPOINT_THEN_PAUSE"


__all__ = [
    "EarlyLaunchDecision",
    "EarlyLaunchSnapshot",
    "EarlyRunAuthorizationReceipt",
    "MAX_EARLY_GPUS",
    "MIN_IDLE_GPU_SECONDS",
    "MIN_MATRIX_CELLS",
    "evaluate_early_launch_gate",
    "main_priority_runtime_action",
    "snapshot_sha256",
]
