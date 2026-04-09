"""Planned end-to-end fragment validation contract."""

from __future__ import annotations

from typing import Protocol

from src.chem.types import FragmentValidationResult


class FragmentValidator(Protocol):
    """Validate one generated fragment against the v3 structural constraints."""

    def validate(
        self,
        parent_smiles: str,
        fragment_smiles: str,
    ) -> FragmentValidationResult:
        """Run parseability, validity, connectivity, and substructure checks."""


def validate_fragment_candidate(
    parent_smiles: str,
    fragment_smiles: str,
) -> FragmentValidationResult:
    """Placeholder entrypoint for future structural validation."""

    raise NotImplementedError(
        "Validation will be implemented once parsing and deletion helpers exist."
    )
