"""Minimal evaluation contracts for checkpoint assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.rewards.types import RewardBreakdown


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """One saved qualitative example from evaluation."""

    record_id: str | int
    parent_smiles: str
    label: int
    generated_fragment: str
    residual_smiles: str | None = None
    valid_substructure: bool | None = None
    reward_breakdown: RewardBreakdown | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Machine-readable evaluation outputs for reporting and checkpoint selection."""

    metric_values: dict[str, float]
    example_count: int
    notes: tuple[str, ...] = field(default_factory=tuple)


class Evaluator(Protocol):
    """Interface for standalone evaluation runners."""

    def evaluate(self) -> EvaluationSummary:
        """Run evaluation and return summary metrics."""
