"""Declarative registry for the two post-main ablation families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AblationVariantSpec:
    family: str
    name: str
    dataset: str
    method: str
    requires_gpu_for_science: bool
    replaces: str


_VARIANTS = {
    "llm": {
        name: AblationVariantSpec(
            family="llm",
            name=name,
            dataset="bace",
            method="ours",
            requires_gpu_for_science=name != "BRICS_FIXED",
            replaces="proposal_generator_only",
        )
        for name in (
            "BRICS_FIXED",
            "CHEMLLM_PRETRAINED",
            "CHEMLLM_SFT",
            "CHEMLLM_SFT_PPO",
        )
    },
    "gnn": {
        name: AblationVariantSpec(
            family="gnn",
            name=name,
            dataset="bace",
            method="ours",
            requires_gpu_for_science=True,
            replaces="frozen_classifier_backbone",
        )
        for name in ("gine", "gin", "gcn", "gatv2")
    },
}


def available_ablation_variants(family: str) -> tuple[str, ...]:
    key = str(family).strip().lower()
    if key not in _VARIANTS:
        raise KeyError(f"unknown ablation family: {family}")
    return tuple(_VARIANTS[key])


def get_ablation_variant(family: str, name: str) -> AblationVariantSpec:
    key = str(family).strip().lower()
    try:
        return _VARIANTS[key][str(name).strip()]
    except KeyError as exc:
        raise KeyError(f"unknown {family} ablation variant: {name}") from exc


__all__ = [
    "AblationVariantSpec",
    "available_ablation_variants",
    "get_ablation_variant",
]
