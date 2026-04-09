"""Diagnostic payloads that future train loops should emit regularly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticsSnapshot:
    """Minimal logging payload for instability and collapse monitoring."""

    step: int
    reward_mean: float | None = None
    parseable_rate: float | None = None
    chemically_valid_rate: float | None = None
    connected_rate: float | None = None
    substructure_rate: float | None = None
    counterfactual_flip_rate: float | None = None
    output_length_mean: float | None = None
    duplicate_output_fraction: float | None = None
    longest_repeated_run: int | None = None
