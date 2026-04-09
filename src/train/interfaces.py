"""Typed interfaces for future SFT and RL runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class TrainStage(str, Enum):
    """Training stages supported by the v3 roadmap."""

    FORMAT_SFT = "format_sft"
    WEAK_SUPERVISION_SFT = "weak_supervision_sft"
    COUNTERFACTUAL_RL = "counterfactual_rl"


@dataclass(frozen=True, slots=True)
class TrainingRunRequest:
    """Minimal run contract for a training entrypoint."""

    stage: TrainStage
    run_name: str
    output_dir: Path
    max_steps: int
    resume_from: Path | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingStatus:
    """Small status object that future train loops can return or log."""

    step: int
    epoch: int
    examples_seen: int
    notes: tuple[str, ...] = ()


class Trainer(Protocol):
    """Interface for SFT and RL stage runners."""

    def run(self, request: TrainingRunRequest) -> TrainingStatus:
        """Run one training stage and return terminal status."""
