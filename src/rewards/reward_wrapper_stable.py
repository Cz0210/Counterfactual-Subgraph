"""Stable PPO-specific reward post-processing without changing default reward behavior."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

from src.reward.reward_wrapper import ChemRLRewarder


_LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric


@dataclass(frozen=True, slots=True)
class StableTeacherConfidenceGateConfig:
    """Resolved teacher-confidence gate settings for stable PPO."""

    enabled: bool = False
    min_teacher_p_before: float = 0.5
    low_conf_cf_weight: float = 0.3


def apply_teacher_confidence_gate_to_reward_logs(
    reward_logs: Sequence[dict[str, Any]],
    *,
    config: StableTeacherConfidenceGateConfig,
    step_index: int | None = None,
    logger: Any | None = None,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Apply a conservative counterfactual reward discount on low-confidence parents.

    The default reward wrapper stays unchanged. This function only post-processes
    stable PPO rewards by scaling the counterfactual component toward zero when
    the teacher/oracle is not confident about the parent's original label.
    """

    resolved_logger = logger or _LOGGER
    updated_logs: list[dict[str, Any]] = []
    adjusted_rewards: list[float] = []

    for reward_log in reward_logs:
        updated = dict(reward_log)
        reward_before_gate = _safe_float(
            updated.get("reward_total", updated.get("total"))
        ) or 0.0
        reward_after_gate = reward_before_gate
        p_before = _safe_float(updated.get("p_before"))
        counterfactual_component = _safe_float(
            updated.get("counterfactual_sem", updated.get("cf_r"))
        )
        applied = False

        if (
            config.enabled
            and p_before is not None
            and p_before < float(config.min_teacher_p_before)
            and counterfactual_component is not None
        ):
            gated_counterfactual = float(counterfactual_component) * float(
                config.low_conf_cf_weight
            )
            reward_after_gate = (
                float(reward_before_gate)
                - float(counterfactual_component)
                + float(gated_counterfactual)
            )
            updated["counterfactual_sem"] = float(gated_counterfactual)
            updated["semantic"] = float(gated_counterfactual)
            updated["semantic_component"] = float(gated_counterfactual)
            updated["teacher_sem"] = float(gated_counterfactual)
            updated["total"] = float(reward_after_gate)
            updated["reward_total"] = float(reward_after_gate)
            applied = True

        updated["stable_teacher_conf_gate_applied"] = bool(applied)
        updated["stable_teacher_conf_gate_weight"] = float(
            config.low_conf_cf_weight
        )
        updated["stable_teacher_conf_gate_min_p_before"] = float(
            config.min_teacher_p_before
        )
        updated["stable_teacher_conf_gate_reward_before"] = float(reward_before_gate)
        updated["stable_teacher_conf_gate_reward_after"] = float(reward_after_gate)
        adjusted_rewards.append(float(reward_after_gate))
        updated_logs.append(updated)

        if config.enabled:
            resolved_logger.info(
                "[STABLE_PPO_TEACHER_CONF_GATE] step=%s p_before=%s min_p_before=%s applied=%s low_conf_cf_weight=%s reward_before_gate=%s reward_after_gate=%s",
                step_index,
                p_before,
                config.min_teacher_p_before,
                applied,
                config.low_conf_cf_weight,
                reward_before_gate,
                reward_after_gate,
            )

    return updated_logs, adjusted_rewards


class StableChemRLRewardWrapper:
    """Thin wrapper that keeps stable-only reward adjustments out of base PPO."""

    def __init__(
        self,
        *,
        base_rewarder: ChemRLRewarder,
        teacher_conf_gate: StableTeacherConfidenceGateConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        self.base_rewarder = base_rewarder
        self.teacher_conf_gate = teacher_conf_gate or StableTeacherConfidenceGateConfig()
        self.logger = logger or _LOGGER

    def compute_rewards_from_decoded(
        self,
        *,
        parent_smiles: Sequence[str],
        generated_fragments: Sequence[str],
        raw_outputs: Sequence[str] | None = None,
        labels: Sequence[int] | None = None,
        metas: Sequence[dict[str, Any]] | None = None,
        device: Any = None,
        step_index: int | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        reward_tensor, reward_logs = self.base_rewarder.compute_rewards_from_decoded(
            parent_smiles=parent_smiles,
            generated_fragments=generated_fragments,
            raw_outputs=raw_outputs,
            labels=labels,
            metas=metas,
            device=device,
        )
        updated_logs, adjusted_rewards = apply_teacher_confidence_gate_to_reward_logs(
            reward_logs,
            config=self.teacher_conf_gate,
            step_index=step_index,
            logger=self.logger,
        )
        if adjusted_rewards:
            reward_tensor = reward_tensor.clone()
            for index, reward_value in enumerate(adjusted_rewards):
                reward_tensor[index] = float(reward_value)
        return reward_tensor, updated_logs

