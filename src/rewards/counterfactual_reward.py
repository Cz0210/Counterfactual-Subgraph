"""Minimal reward wiring that preserves the v3 counterfactual objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.chem.types import FragmentValidationResult
from src.rewards.aggregation import aggregate_reward_terms
from src.rewards.types import RewardBreakdown, RewardTerm, RewardWeights


@dataclass(frozen=True, slots=True)
class RewardContext:
    """Inputs needed to build a reward breakdown for one fragment candidate."""

    parent_smiles: str
    generated_fragment: str
    original_label: int
    validation: FragmentValidationResult
    residual_smiles: str | None = None
    counterfactual_score: float | None = None
    compactness_score: float | None = None
    anti_collapse_penalty: float | None = None
    extra_metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)


class CounterfactualScorer(Protocol):
    """Backend contract for deletion-based label-flip scoring."""

    def score(
        self,
        *,
        parent_smiles: str,
        fragment_smiles: str,
        original_label: int,
        residual_smiles: str | None = None,
    ) -> float:
        """Return a scalar that measures deletion-induced label change."""


def _binary_term(name: str, passed: bool, weight: float) -> RewardTerm:
    value = 1.0 if passed else 0.0
    return RewardTerm(
        name=name,
        value=value,
        weight=weight,
        weighted_value=value * weight,
        passed=passed,
    )


def build_reward_breakdown(
    context: RewardContext,
    weights: RewardWeights,
) -> RewardBreakdown:
    """Build a structured reward object without hard-coding model logic."""

    terms = (
        _binary_term("parseable", context.validation.parseable, weights.parseable),
        _binary_term(
            "chemically_valid",
            context.validation.chemically_valid,
            weights.chemically_valid,
        ),
        _binary_term("connected", context.validation.connected, weights.connected),
        _binary_term("substructure", context.validation.is_substructure, weights.substructure),
        RewardTerm(
            name="counterfactual_effect",
            value=context.counterfactual_score or 0.0,
            weight=weights.counterfactual_effect,
            weighted_value=(context.counterfactual_score or 0.0)
            * weights.counterfactual_effect,
            passed=None,
        ),
        RewardTerm(
            name="compactness",
            value=context.compactness_score or 0.0,
            weight=weights.compactness,
            weighted_value=(context.compactness_score or 0.0) * weights.compactness,
            passed=None,
        ),
        RewardTerm(
            name="anti_collapse",
            value=context.anti_collapse_penalty or 0.0,
            weight=weights.anti_collapse,
            weighted_value=(context.anti_collapse_penalty or 0.0) * weights.anti_collapse,
            passed=None,
        ),
    )
    metadata = {
        "original_label": context.original_label,
        "has_residual_smiles": context.residual_smiles is not None,
    }
    metadata.update(context.extra_metadata)
    return RewardBreakdown(
        total_reward=aggregate_reward_terms(terms),
        terms=terms,
        metadata=metadata,
    )
