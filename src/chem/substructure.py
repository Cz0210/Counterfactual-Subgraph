"""Planned substructure and connectivity checks."""

from __future__ import annotations

from typing import Protocol


class SubstructureMatcher(Protocol):
    """Backend contract for parent-substructure and connectivity checks."""

    def is_substructure(self, parent_smiles: str, fragment_smiles: str) -> bool:
        """Return whether the fragment is a genuine substructure of the parent."""

    def is_connected(self, fragment_smiles: str) -> bool:
        """Return whether the fragment corresponds to one connected component."""


def is_parent_substructure(parent_smiles: str, fragment_smiles: str) -> bool:
    """Placeholder entrypoint for future substructure matching."""

    raise NotImplementedError(
        "Substructure matching will be implemented in the chemistry phase."
    )


def is_connected_fragment(fragment_smiles: str) -> bool:
    """Placeholder entrypoint for future connectivity checks."""

    raise NotImplementedError(
        "Connectivity checks will be implemented in the chemistry phase."
    )
