"""Minimal model interfaces shared by inference, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One fragment generation request from a parent molecule prompt."""

    parent_smiles: str
    label: int | None = None
    prompt: str | None = None
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One model output after post-processing to a fragment candidate."""

    fragment_smiles: str
    raw_text: str
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FragmentGenerator(Protocol):
    """Interface for any model that generates one fragment per request."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one fragment candidate for a parent molecule."""
