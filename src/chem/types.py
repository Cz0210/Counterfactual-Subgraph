"""Shared chemistry result types used by reward, train, and eval code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedMolecule:
    """Result of parsing and optionally sanitizing a SMILES string."""

    smiles: str
    parseable: bool
    canonical_smiles: str | None = None
    atom_count: int | None = None
    sanitized: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DeletionResult:
    """Result of removing a fragment candidate from a parent molecule."""

    parent_smiles: str
    fragment_smiles: str
    residual_smiles: str | None
    success: bool
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FragmentValidationResult:
    """Structured structural checks for one generated fragment."""

    parent_smiles: str
    fragment_smiles: str
    parseable: bool
    chemically_valid: bool
    connected: bool
    is_substructure: bool
    deletion_supported: bool
    failure_reasons: tuple[str, ...] = ()
