"""Reward aggregation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from src.rewards.types import RewardTerm


def aggregate_reward_terms(terms: Sequence[RewardTerm]) -> float:
    """Sum weighted reward terms into one scalar."""

    return float(sum(term.weighted_value for term in terms))
