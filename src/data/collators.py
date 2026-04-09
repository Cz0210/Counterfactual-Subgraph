"""Batch collation helpers for prompt-based fragment generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import FragmentExample, MoleculeRecord


@dataclass(frozen=True, slots=True)
class PromptBatch:
    """Minimal batch contract shared by SFT, RL rollouts, and evaluation."""

    records: tuple[MoleculeRecord, ...]
    prompts: tuple[str, ...]
    target_fragments: tuple[str | None, ...]


class CounterfactualPromptCollator:
    """Convert examples into prompt-only batches without training logic."""

    def __init__(self, *, include_label: bool = False) -> None:
        self.include_label = include_label

    def __call__(self, examples: Sequence[FragmentExample]) -> PromptBatch:
        records = tuple(example.record for example in examples)
        prompts = tuple(
            build_counterfactual_prompt(example.record, include_label=self.include_label)
            for example in examples
        )
        targets = tuple(example.target_fragment for example in examples)
        return PromptBatch(records=records, prompts=prompts, target_fragments=targets)
