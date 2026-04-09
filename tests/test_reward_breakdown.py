from src.chem.types import FragmentValidationResult
from src.rewards.counterfactual_reward import RewardContext, build_reward_breakdown
from src.rewards.types import RewardWeights


def test_reward_breakdown_keeps_counterfactual_term_explicit() -> None:
    validation = FragmentValidationResult(
        parent_smiles="CCO",
        fragment_smiles="CO",
        parseable=True,
        chemically_valid=True,
        connected=True,
        is_substructure=True,
        deletion_supported=False,
    )
    context = RewardContext(
        parent_smiles="CCO",
        generated_fragment="CO",
        original_label=1,
        validation=validation,
        counterfactual_score=0.5,
        compactness_score=0.2,
        anti_collapse_penalty=-0.1,
    )

    breakdown = build_reward_breakdown(context, RewardWeights())

    assert breakdown.total_reward == 4.6
    assert any(term.name == "counterfactual_effect" for term in breakdown.terms)
