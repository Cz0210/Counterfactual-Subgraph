"""Auditable, post-main ablation framework contracts.

The package is deliberately import-only: importing it never loads a model,
claims a GPU, or mutates the 4x4 main-matrix authority.
"""

from src.ablations.contracts import (
    AblationRunContract,
    AblationStatus,
    ContractError,
)
from src.ablations.launch_gate import LaunchGateDecision, evaluate_launch_gate

__all__ = [
    "AblationRunContract",
    "AblationStatus",
    "ContractError",
    "LaunchGateDecision",
    "evaluate_launch_gate",
]
