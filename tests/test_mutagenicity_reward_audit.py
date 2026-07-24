from __future__ import annotations

import math

import pytest

from scripts.audit_mutagenicity_ppo_reward_components import recommend_config
from scripts.train_ppo_stable import (
    build_parser as build_stable_parser,
    resolve_flip_dominant_reward_config,
)
from src.rewards.reward_wrapper_stable import (
    FlipDominantRewardConfig,
    apply_flip_dominant_profile_to_reward_logs,
)
from src.train.mutagenicity_stable_ppo import build_parent_coverage_plan


def _config() -> FlipDominantRewardConfig:
    return FlipDominantRewardConfig(
        enabled=True,
        strict_flip_bonus=2.0,
        non_flip_penalty=-0.5,
        cf_drop_weight=1.0,
        validity_weight=0.2,
        substructure_weight=0.2,
        size_weight=0.2,
        projection_penalty=0.2,
        non_flip_aux_reward_cap=0.6,
        strict_flip_reward_margin=0.5,
        reward_clip_min=-5.0,
        reward_clip_max=8.0,
        profile_name="mutagenicity_flip_dominant",
    )


def _row(*, strict: bool, projected: bool = False, raw: str = "CCCCCCCC") -> dict:
    return {
        "pred_before": 1,
        "pred_after": 0 if strict else 1,
        "cf_drop": 0.4,
        "raw_fragment": raw,
        "final_fragment": "CC",
        "final_substructure": True,
        "projection_used": projected,
        "atom_ratio": 0.25,
        "reward_total": 3.0,
        "reward_components": {
            "format_r": 1.0,
            "valid_r": 1.0,
            "subgraph_r": 1.0,
            "subdist_contribution": 0.2,
            "size_window_r": 0.5,
            "length_r": 0.0,
            "dummy_r": 0.0,
        },
    }


def test_strict_flip_reward_dominates_equivalent_non_flip() -> None:
    rows, rewards = apply_flip_dominant_profile_to_reward_logs(
        [_row(strict=True), _row(strict=False)],
        config=_config(),
    )
    assert rewards[0] > rewards[1]
    assert rows[0]["cf_flip"] is True
    assert rows[1]["cf_flip"] is False


def test_invalid_candidate_reward_is_below_legal_non_flip() -> None:
    legal = _row(strict=False)
    invalid = {
        **_row(strict=False),
        "final_fragment": "",
        "final_substructure": False,
        "cf_drop": 0.0,
        "reward_components": {
            "format_r": -1.0,
            "valid_r": -1.0,
            "subgraph_r": -1.0,
            "subdist_contribution": 0.0,
            "size_window_r": -0.5,
            "length_r": -0.2,
            "dummy_r": 0.0,
        },
    }
    _rows, rewards = apply_flip_dominant_profile_to_reward_logs(
        [legal, invalid],
        config=_config(),
    )
    assert rewards[1] < rewards[0]


def test_non_flip_aux_cap_and_strict_exemption() -> None:
    rows, _ = apply_flip_dominant_profile_to_reward_logs(
        [_row(strict=False), _row(strict=True)],
        config=_config(),
    )
    assert rows[0]["positive_aux_reward_after_cap"] <= 0.6
    assert rows[0]["non_flip_aux_cap_applied"] is True
    assert rows[1]["positive_aux_reward_after_cap"] > 0.6
    assert rows[1]["non_flip_aux_cap_applied"] is False


def test_projected_strict_flip_keeps_bonus_and_final_fragment_semantics() -> None:
    direct, direct_reward = apply_flip_dominant_profile_to_reward_logs(
        [_row(strict=True, projected=False, raw="N" * 100)],
        config=_config(),
    )
    projected, projected_reward = apply_flip_dominant_profile_to_reward_logs(
        [_row(strict=True, projected=True, raw="N" * 100)],
        config=_config(),
    )
    assert projected[0]["strict_flip_bonus"] == 2.0
    assert projected_reward[0] == pytest.approx(direct_reward[0] - 0.2)
    assert projected_reward[0] > 0.0
    assert projected[0]["final_fragment"] == "CC"


def test_raw_fragment_does_not_change_flip_dominant_reward() -> None:
    _rows, rewards = apply_flip_dominant_profile_to_reward_logs(
        [
            _row(strict=True, raw="N"),
            _row(strict=True, raw="N" * 500),
        ],
        config=_config(),
    )
    assert rewards[0] == pytest.approx(rewards[1])


def test_legacy_profile_is_disabled_and_does_not_require_new_values() -> None:
    args = build_stable_parser().parse_args([])
    config = resolve_flip_dominant_reward_config(args)
    assert config.enabled is False
    rows, rewards = apply_flip_dominant_profile_to_reward_logs(
        [{"reward_total": 1.25, "total": 1.25}],
        config=config,
    )
    assert rewards == [1.25]
    assert rows[0]["reward_total"] == 1.25


def test_all_flip_dominant_rewards_are_finite() -> None:
    rows, rewards = apply_flip_dominant_profile_to_reward_logs(
        [_row(strict=True), _row(strict=False, projected=True)],
        config=_config(),
    )
    assert all(math.isfinite(value) for value in rewards)
    assert all(math.isfinite(float(row["reward_total"])) for row in rows)


def test_flip_dominant_rejects_positive_non_flip_penalty() -> None:
    config = _config()
    invalid = FlipDominantRewardConfig(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field != "non_flip_penalty"
        },
        non_flip_penalty=0.5,
    )
    with pytest.raises(ValueError, match="non-positive"):
        invalid.validate()


def test_reward_recommendation_satisfies_derived_margin() -> None:
    rows = [_row(strict=True) for _ in range(20)] + [
        _row(strict=False) for _ in range(20)
    ]
    recommended = recommend_config(rows, margin=0.5)
    derivation = recommended["derivation"]
    assert (
        derivation["projected_strict_p10_after_bonus"]
        >= derivation["non_flip_p90"] + 0.5 - 1e-12
    )


def test_batch_16_full_epoch_is_91_updates() -> None:
    plan = build_parent_coverage_plan(
        num_dataset_rows=1448,
        rollout_batch_size=16,
        sampler_seed=7,
    )
    assert plan.samples_per_update == 16
    assert plan.updates_per_epoch == 91
    assert plan.max_updates == 91
    assert plan.sampling_with_replacement is False
