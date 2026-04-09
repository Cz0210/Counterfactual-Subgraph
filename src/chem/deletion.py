"""Planned fragment deletion interfaces for counterfactual scoring."""

from __future__ import annotations

from typing import Protocol

from src.chem.types import DeletionResult


class FragmentDeletionEngine(Protocol):
    """Backend contract for deletion-based counterfactual construction."""

    def delete(self, parent_smiles: str, fragment_smiles: str) -> DeletionResult:
        """Remove one fragment candidate from the parent molecule."""


def delete_fragment_from_parent(parent_smiles: str, fragment_smiles: str) -> DeletionResult:
    """Placeholder entrypoint for future deletion logic."""

    raise NotImplementedError(
        "Deletion logic will be implemented after the chemistry interfaces settle."
    )
