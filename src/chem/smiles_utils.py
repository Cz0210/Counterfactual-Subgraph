"""Planned SMILES parsing and canonicalization hooks."""

from __future__ import annotations

from typing import Protocol

from src.chem.types import ParsedMolecule


class MoleculeParser(Protocol):
    """Backend contract for RDKit-based parsing and canonicalization."""

    def parse(self, smiles: str) -> ParsedMolecule:
        """Parse and sanitize a SMILES string."""

    def canonicalize(self, smiles: str) -> str:
        """Return a canonical SMILES representation when available."""


def parse_smiles(smiles: str) -> ParsedMolecule:
    """Placeholder entrypoint for future RDKit-backed parsing."""

    raise NotImplementedError(
        "RDKit-backed parsing belongs to the chemistry implementation phase."
    )


def canonicalize_smiles(smiles: str) -> str:
    """Placeholder entrypoint for future canonicalization support."""

    raise NotImplementedError(
        "Canonicalization will be implemented with the chemistry backend."
    )
