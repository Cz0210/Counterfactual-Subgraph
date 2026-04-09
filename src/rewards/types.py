"""Typed reward containers that keep reward semantics explicit and testable."""

from __future__ import annotations

from dataclasses import dataclass, field


JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Configurable weights for structural and counterfactual reward terms."""

    parseable: float = 1.0
    chemically_valid: float = 1.0
    connected: float = 1.0
    substructure: float = 1.0
    counterfactual_effect: float = 1.0
    compactness: float = 1.0
    anti_collapse: float = 1.0
    kl_penalty: float = 0.0


@dataclass(frozen=True, slots=True)
class RewardTerm:
    """One named reward component before and after weighting."""

    name: str
    value: float
    weight: float
    weighted_value: float
    passed: bool | None = None


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Structured reward output for logging, debugging, and evaluation."""

    total_reward: float
    terms: tuple[RewardTerm, ...]
    metadata: dict[str, JsonScalar] = field(default_factory=dict)
