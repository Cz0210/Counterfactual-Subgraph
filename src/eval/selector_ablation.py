"""Configuration-only selector component ablation registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SelectorAblationConfig:
    alpha_cf: float
    beta_coverage: float
    gamma_structural_redundancy: float
    gamma_coverage_redundancy: float
    eta_size: float
    selection_seed: int = 13
    selection_mode: str = "weighted_greedy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FULL_SELECTOR = SelectorAblationConfig(1.0, 1.0, 0.7, 0.7, 0.3)
SELECTOR_VARIANTS = {
    "full_selector": FULL_SELECTOR,
    "no_cf_term": SelectorAblationConfig(0.0, 1.0, 0.7, 0.7, 0.3),
    "no_coverage_term": SelectorAblationConfig(1.0, 0.0, 0.7, 0.7, 0.3),
    "no_structural_redundancy": SelectorAblationConfig(1.0, 1.0, 0.0, 0.7, 0.3),
    "no_coverage_redundancy": SelectorAblationConfig(1.0, 1.0, 0.7, 0.0, 0.3),
    "no_size_penalty": SelectorAblationConfig(1.0, 1.0, 0.7, 0.7, 0.0),
    "cfdrop_only": SelectorAblationConfig(1.0, 0.0, 0.0, 0.0, 0.0),
    "coverage_only": SelectorAblationConfig(0.0, 1.0, 0.0, 0.0, 0.0),
    "random_topk": SelectorAblationConfig(0.0, 0.0, 0.0, 0.0, 0.0, selection_mode="random_topk"),
}


def selector_variant(name: str, *, seed: int = 13) -> SelectorAblationConfig:
    try:
        selected = SELECTOR_VARIANTS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown selector ablation {name!r}; expected {sorted(SELECTOR_VARIANTS)}"
        ) from exc
    return SelectorAblationConfig(
        alpha_cf=selected.alpha_cf,
        beta_coverage=selected.beta_coverage,
        gamma_structural_redundancy=selected.gamma_structural_redundancy,
        gamma_coverage_redundancy=selected.gamma_coverage_redundancy,
        eta_size=selected.eta_size,
        selection_seed=int(seed),
        selection_mode=selected.selection_mode,
    )


__all__ = ["FULL_SELECTOR", "SELECTOR_VARIANTS", "SelectorAblationConfig", "selector_variant"]
