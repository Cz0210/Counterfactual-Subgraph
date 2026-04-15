"""Reward terms and aggregation aligned to deletion-based counterfactual scoring."""

from src.rewards.aggregation import aggregate_reward_terms
from src.rewards.anti_collapse import (
    CollapseDiagnostics,
    analyze_batch_collapse,
    collapse_penalty_from_diagnostics,
)
from src.rewards.chem_rules import ChemRewardEngine
from src.rewards.counterfactual_reward import (
    CounterfactualScorer,
    RewardContext,
    build_reward_breakdown,
)
from src.rewards.reward_calculator import (
    CounterfactualReward,
    load_oracle_bundle,
    prepare_smiles_for_oracle,
    smiles_to_morgan_array,
)
from src.rewards.reward_wrapper import ChemRLRewarder, RewardTrace, shape_probability_reward
from src.rewards.types import RewardBreakdown, RewardTerm, RewardWeights

__all__ = [
    "ChemRewardEngine",
    "ChemRLRewarder",
    "CollapseDiagnostics",
    "CounterfactualReward",
    "CounterfactualScorer",
    "RewardTrace",
    "RewardBreakdown",
    "RewardContext",
    "RewardTerm",
    "RewardWeights",
    "aggregate_reward_terms",
    "analyze_batch_collapse",
    "build_reward_breakdown",
    "collapse_penalty_from_diagnostics",
    "load_oracle_bundle",
    "prepare_smiles_for_oracle",
    "shape_probability_reward",
    "smiles_to_morgan_array",
]
