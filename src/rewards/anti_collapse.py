"""Collapse diagnostics for repeated-token and duplicate-output failure modes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CollapseDiagnostics:
    """Simple text-level diagnostics that can be logged before RL is wired up."""

    longest_run: int
    dominant_character_fraction: float
    duplicate_output_fraction: float


def _longest_run(text: str) -> int:
    longest = 0
    current = 0
    previous = None
    for character in text:
        if character == previous:
            current += 1
        else:
            current = 1
            previous = character
        longest = max(longest, current)
    return longest


def _dominant_character_fraction(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    dominant_count = counts.most_common(1)[0][1]
    return dominant_count / len(text)


def analyze_batch_collapse(outputs: Sequence[str]) -> CollapseDiagnostics:
    """Compute a small set of collapse signals for generated outputs."""

    if not outputs:
        return CollapseDiagnostics(
            longest_run=0,
            dominant_character_fraction=0.0,
            duplicate_output_fraction=0.0,
        )

    normalized = [output.strip() for output in outputs]
    longest_run = max(_longest_run(output) for output in normalized)
    dominant_fraction = max(_dominant_character_fraction(output) for output in normalized)
    unique_outputs = len(set(normalized))
    duplicate_fraction = 1.0 - (unique_outputs / len(normalized))
    return CollapseDiagnostics(
        longest_run=longest_run,
        dominant_character_fraction=dominant_fraction,
        duplicate_output_fraction=duplicate_fraction,
    )


def collapse_penalty_from_diagnostics(diagnostics: CollapseDiagnostics) -> float:
    """Convert collapse signals into a simple negative penalty."""

    run_penalty = max(0.0, diagnostics.longest_run - 6) / 10.0
    dominance_penalty = max(0.0, diagnostics.dominant_character_fraction - 0.6)
    duplicate_penalty = max(0.0, diagnostics.duplicate_output_fraction)
    return -(run_penalty + dominance_penalty + duplicate_penalty)
