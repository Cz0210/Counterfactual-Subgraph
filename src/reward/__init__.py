"""兼容层：保留 `src.reward.*` 导入路径，同时复用 `src.rewards.*` 实现。"""

from src.rewards.chem_rules import ChemRewardEngine
from src.rewards.counterfactual_oracle import (
    CounterfactualTeacherResult,
    CounterfactualTeacherScorer,
    delete_one_substructure,
)
from src.rewards.reward_calculator import (
    CounterfactualReward,
    load_oracle_bundle,
    prepare_smiles_for_oracle,
    smiles_to_morgan_array,
)
from src.rewards.teacher_semantic import (
    TeacherSemanticResult,
    TeacherSemanticScorer,
    require_teacher_semantic_scorer,
)
from src.rewards.reward_wrapper import ChemRLRewarder, RewardTrace, shape_probability_reward

__all__ = [
    "ChemRewardEngine",
    "ChemRLRewarder",
    "CounterfactualReward",
    "CounterfactualTeacherResult",
    "CounterfactualTeacherScorer",
    "RewardTrace",
    "TeacherSemanticResult",
    "TeacherSemanticScorer",
    "delete_one_substructure",
    "load_oracle_bundle",
    "prepare_smiles_for_oracle",
    "require_teacher_semantic_scorer",
    "shape_probability_reward",
    "smiles_to_morgan_array",
]
