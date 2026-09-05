"""Process-local configuration for the authorized T12 production cadence.

The historic T12 implementation defaults to two durable process boundaries
(10k and 20k).  The final-16 recovery contract requires earlier recovery
points without changing any walk, model, RNG, or candidate semantics.  This
module changes only the process-local checkpoint schedule and the matching
journal segment limits.  Callers must opt in explicitly in every generation
and verification process.
"""

from __future__ import annotations

from typing import Any


FORMAL_PRODUCTION_TOTAL_STEPS = 20_000
FORMAL_PRODUCTION_CHECKPOINT_CURSORS = (
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    7_500,
    10_000,
    12_500,
    15_000,
    17_500,
    20_000,
)


def configure_t12_formal_production_profile() -> dict[str, Any]:
    """Install the frozen cadence in all modules that cache these constants."""

    from src.baselines import tastemolnet_gcf_full as full
    from src.baselines import tastemolnet_gcf_full_resume as resume
    from src.baselines import tastemolnet_gcf_full_verify as verify
    from src.baselines import tastemolnet_gcf_production_state as state
    from src.baselines import tastemolnet_gcf_transition_store as transitions

    cursors = tuple(FORMAL_PRODUCTION_CHECKPOINT_CURSORS)
    frozen = frozenset(cursors)
    state.PINNED_TOTAL_STEPS = FORMAL_PRODUCTION_TOTAL_STEPS
    state.PINNED_CHECKPOINT_CURSORS = cursors
    state.HISTORY_MAX_SEGMENTS = len(cursors)
    transitions.TRANSITION_MAX_SEGMENTS = len(cursors)
    resume.PRODUCTION_TOTAL_STEPS = FORMAL_PRODUCTION_TOTAL_STEPS
    resume.PRODUCTION_CHECKPOINT_CURSORS = frozen
    full.PRODUCTION_TOTAL_STEPS = FORMAL_PRODUCTION_TOTAL_STEPS
    full.PRODUCTION_CHECKPOINT_CURSORS = frozen
    verify.PRODUCTION_TOTAL_STEPS = FORMAL_PRODUCTION_TOTAL_STEPS
    verify.PRODUCTION_CHECKPOINT_CURSORS = frozen
    return {
        "profile": "T12_FORMAL_RECOVERY_CADENCE_V1",
        "total_steps": FORMAL_PRODUCTION_TOTAL_STEPS,
        "checkpoint_cursors": list(cursors),
        "scientific_transition_changed": False,
        "checkpoint_schedule_only": True,
    }


__all__ = [
    "FORMAL_PRODUCTION_CHECKPOINT_CURSORS",
    "FORMAL_PRODUCTION_TOTAL_STEPS",
    "configure_t12_formal_production_profile",
]
