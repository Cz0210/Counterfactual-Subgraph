"""Stable PPO-specific reward post-processing without changing default reward behavior."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

from src.chem import is_parent_substructure, parse_smiles
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


def _heavy_atom_count(smiles: Any, *, allow_capped_fragments: bool = False) -> int | None:
    normalized = str(smiles or "").strip()
    if not normalized:
        return None
    parsed = parse_smiles(
        normalized,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=allow_capped_fragments,
    )
    if not parsed.sanitized or parsed.mol is None:
        return None
    try:
        return int(parsed.mol.GetNumHeavyAtoms())
    except Exception:
        return None


def _resolve_actual_final_fragment(row: dict[str, Any]) -> tuple[str | None, bool]:
    projection_used = bool(
        row.get("used_projected_subgraph_for_reward")
        or row.get("projection_used")
    )
    if projection_used:
        projected = row.get("projected_fragment") or row.get(
            "nearest_parent_subgraph_smiles"
        )
        normalized = str(projected or "").strip()
        return (normalized or None), True
    if bool(row.get("direct_substructure") or row.get("direct_substructure_success")):
        direct = row.get("core_fragment") or row.get("fragment")
        normalized = str(direct or "").strip()
        return (normalized or None), False
    return None, False


def _reward_components_with_size(
    row: dict[str, Any],
    *,
    size_window_reward: float,
) -> dict[str, Any]:
    components = dict(row.get("reward_components") or row.get("breakdown") or {})
    components["size_window_r"] = float(size_window_reward)
    return components


def apply_final_fragment_atom_ratio_to_reward_logs(
    reward_logs: Sequence[dict[str, Any]],
    *,
    base_rewarder: ChemRLRewarder,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Align stable-PPO size reward and diagnostics with the deleted fragment."""

    updated_logs: list[dict[str, Any]] = []
    adjusted_rewards: list[float] = []
    for reward_log in reward_logs:
        updated = dict(reward_log)
        parent_smiles = str(updated.get("parent_smiles") or "").strip()
        raw_fragment = str(updated.get("raw_fragment") or "").strip()
        final_fragment, projection_used = _resolve_actual_final_fragment(updated)

        parent_heavy_atoms = _heavy_atom_count(parent_smiles)
        raw_fragment_heavy_atoms = _heavy_atom_count(
            raw_fragment,
            allow_capped_fragments=True,
        )
        final_fragment_heavy_atoms = _heavy_atom_count(final_fragment)
        raw_atom_ratio = (
            float(raw_fragment_heavy_atoms) / float(parent_heavy_atoms)
            if raw_fragment_heavy_atoms is not None
            and parent_heavy_atoms is not None
            and parent_heavy_atoms > 0
            else None
        )

        final_substructure = False
        atom_ratio: float | None = None
        if (
            final_fragment
            and final_fragment_heavy_atoms is not None
            and final_fragment_heavy_atoms > 0
            and parent_heavy_atoms is not None
            and parent_heavy_atoms > 0
        ):
            try:
                final_substructure = bool(
                    is_parent_substructure(parent_smiles, final_fragment)
                )
            except Exception:
                final_substructure = False
            if final_substructure:
                atom_ratio = float(final_fragment_heavy_atoms) / float(
                    parent_heavy_atoms
                )
                if not (0.0 < atom_ratio <= 1.0 + 1e-12):
                    raise RuntimeError(
                        "[STABLE_PPO_ATOM_RATIO_AUDIT_FAILED] "
                        f"parent={parent_smiles!r} final_fragment={final_fragment!r} "
                        f"parent_heavy_atoms={parent_heavy_atoms} "
                        f"final_fragment_heavy_atoms={final_fragment_heavy_atoms} "
                        f"atom_ratio={atom_ratio}"
                    )
                atom_ratio = min(atom_ratio, 1.0)

        old_size_reward = _safe_float(updated.get("size_window_reward"))
        if old_size_reward is None:
            old_components = dict(
                updated.get("reward_components") or updated.get("breakdown") or {}
            )
            old_size_reward = _safe_float(old_components.get("size_window_r")) or 0.0
        new_size_reward, size_window_bucket = base_rewarder._compute_size_window_reward(
            atom_ratio=atom_ratio
        )
        reward_delta = float(new_size_reward) - float(old_size_reward)
        reward_before = _safe_float(updated.get("reward_total", updated.get("total"))) or 0.0
        reward_after = float(reward_before) + reward_delta

        components = _reward_components_with_size(
            updated,
            size_window_reward=float(new_size_reward),
        )
        updated.update(
            {
                "final_fragment": final_fragment,
                "projection_used": bool(projection_used),
                "final_substructure": bool(final_substructure),
                "parent_heavy_atoms": parent_heavy_atoms,
                "raw_fragment_heavy_atoms": raw_fragment_heavy_atoms,
                "final_fragment_heavy_atoms": final_fragment_heavy_atoms,
                "raw_atom_ratio": raw_atom_ratio,
                "atom_ratio": atom_ratio,
                "atom_ratio_source": "final_fragment",
                "final_fragment_atom_count": final_fragment_heavy_atoms or 0,
                "final_fragment_atom_ratio": atom_ratio,
                "size_window_reward": float(new_size_reward),
                "size_window_bucket": size_window_bucket,
                "total": reward_after,
                "reward_total": reward_after,
                "reward_components": components,
                "breakdown": components,
            }
        )
        for field_name in (
            "reward_before_projection_penalty",
            "reward_after_projection_penalty",
        ):
            numeric = _safe_float(updated.get(field_name))
            if numeric is not None:
                updated[field_name] = float(numeric) + reward_delta

        updated_logs.append(updated)
        adjusted_rewards.append(reward_after)

    return updated_logs, adjusted_rewards


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
        reward_logs, ratio_adjusted_rewards = (
            apply_final_fragment_atom_ratio_to_reward_logs(
                reward_logs,
                base_rewarder=self.base_rewarder,
            )
        )
        if ratio_adjusted_rewards:
            reward_tensor = reward_tensor.clone()
            for index, reward_value in enumerate(ratio_adjusted_rewards):
                reward_tensor[index] = float(reward_value)
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
