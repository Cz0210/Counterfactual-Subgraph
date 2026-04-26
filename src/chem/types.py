"""Shared chemistry result types used by reward, train, and eval code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChemistryFailureType(str, Enum):
    """Normalized failure categories for chemistry utilities."""

    INVALID_INPUT_TYPE = "invalid_input_type"
    EMPTY_SMILES = "empty_smiles"
    RDKIT_UNAVAILABLE = "rdkit_unavailable"
    PARSE_FAILED = "parse_failed"
    SANITIZE_FAILED = "sanitize_failed"
    DISCONNECTED_FRAGMENT = "disconnected_fragment"
    NOT_SUBSTRUCTURE = "not_substructure"
    INVALID_CAPPED_FRAGMENT = "invalid_capped_fragment"
    CAPPED_SUBGRAPH_MISMATCH = "capped_subgraph_mismatch"
    NO_SUBSTRUCTURE_MATCH = "no_substructure_match"
    RESIDUAL_SANITIZE_FAILED = "residual_sanitize_failed"


@dataclass(frozen=True, slots=True)
class ParsedMolecule:
    """Result of parsing and optionally sanitizing a SMILES string."""

    smiles: str
    parseable: bool
    canonical_smiles: str | None = None
    atom_count: int | None = None
    sanitized: bool = False
    contains_dummy_atoms: bool = False
    used_relaxed_sanitization: bool = False
    failure_type: ChemistryFailureType | None = None
    failure_reason: str | None = None
    mol: object | None = None


@dataclass(frozen=True, slots=True)
class DeletionResult:
    """Result of removing a fragment candidate from a parent molecule."""

    parent_smiles: str
    fragment_smiles: str
    residual_smiles: str | None
    success: bool
    failure_type: ChemistryFailureType | None = None
    failure_reason: str | None = None
    match_count: int = 0
    selected_match: tuple[int, ...] = ()
    residual_atom_count: int | None = None


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
    parent_parseable: bool = False
    parent_chemically_valid: bool = False
    residual_smiles: str | None = None
    failure_types: tuple[ChemistryFailureType, ...] = ()
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FragmentRepairResult:
    """Result of attempting to repair one decoded fragment to a parent-derived subgraph."""

    parent_smiles: str
    raw_fragment_smiles: str
    attempted: bool
    success: bool
    repaired_fragment_smiles: str | None = None
    repair_source: str | None = None
    repair_similarity: float | None = None
    reason: str | None = None
    candidate_count: int = 0


@dataclass(frozen=True, slots=True)
class FragmentProjectionResult:
    """Result of projecting one decoded fragment onto a parent-derived candidate."""

    parent_smiles: str
    raw_fragment_smiles: str
    attempted: bool
    success: bool
    projection_method: str | None = None
    projected_fragment_smiles: str | None = None
    projection_source: str | None = None
    projection_score: float | None = None
    reason: str | None = None
    candidate_count: int = 0
    projected_atom_count: int | None = None
    projected_atom_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class FragmentSyntaxRepairCandidate:
    """One local syntax-repair candidate before parent/projection validation."""

    fragment_smiles: str
    repair_method: str | None = None
    reason: str | None = None
    edit_distance: int = 0
    suffix_trim_count: int = 0
    added_parentheses: int = 0
    added_ring_closures: int = 0


@dataclass(frozen=True, slots=True)
class FragmentSyntaxRepairResult:
    """Result of a strictly local syntax repair attempt."""

    raw_fragment_smiles: str
    attempted: bool
    success: bool
    repaired_fragment_smiles: str | None = None
    repair_method: str | None = None
    reason: str | None = None
    edit_distance: int = 0
    suffix_trim_count: int = 0
    added_parentheses: int = 0
    added_ring_closures: int = 0
    repaired_atom_count: int | None = None
    failure_reason: str | None = None
    failure_stage: str | None = None
    candidate_count: int = 0
    candidates_parse_ok: int = 0
    candidates_core_ok: int = 0
    candidates_parent_ok: int = 0
    candidates_projection_ok: int = 0
    best_candidate: str | None = None
    accept_stage: str | None = None
    candidate_accepted: bool = False
    candidate_rejected_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FragmentComponentSalvageResult:
    """Result of extracting one connected component from a decoded fragment."""

    raw_fragment_smiles: str
    attempted: bool
    success: bool
    component_count: int = 0
    raw_component_count: int = 0
    core_component_count: int = 0
    salvage_stage: str | None = None
    salvage_method: str | None = None
    salvaged_fragment_smiles: str | None = None
    salvaged_atom_count: int | None = None
    candidate_count: int = 0
    best_candidate: str | None = None
    reason: str | None = None
    failure_reason: str | None = None
    failure_stage: str | None = None
