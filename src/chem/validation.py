"""End-to-end fragment validation contract."""

from __future__ import annotations

from typing import Protocol

from src.chem.deletion import delete_fragment_from_parent
from src.chem.smiles_utils import parse_smiles
from src.chem.substructure import (
    is_connected_fragment,
    is_parent_substructure,
    is_valid_capped_subgraph,
)
from src.chem.types import ChemistryFailureType, FragmentValidationResult


class FragmentValidator(Protocol):
    """Validate one generated fragment against the v3 structural constraints."""

    def validate(
        self,
        parent_smiles: str,
        fragment_smiles: str,
    ) -> FragmentValidationResult:
        """Run parseability, validity, connectivity, and substructure checks."""


def _append_failure(
    failure_types: list[ChemistryFailureType],
    failure_reasons: list[str],
    *,
    failure_type: ChemistryFailureType | None,
    failure_reason: str | None,
) -> None:
    if failure_type is not None and failure_type not in failure_types:
        failure_types.append(failure_type)
    if failure_reason:
        failure_reasons.append(failure_reason)


def validate_fragment_candidate(
    parent_smiles: str,
    fragment_smiles: str,
) -> FragmentValidationResult:
    """Validate one fragment candidate against the structural v3 constraints."""

    failure_types: list[ChemistryFailureType] = []
    failure_reasons: list[str] = []

    parent = parse_smiles(parent_smiles, sanitize=True, canonicalize=False)
    fragment = parse_smiles(
        fragment_smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )

    if not parent.parseable:
        _append_failure(
            failure_types,
            failure_reasons,
            failure_type=parent.failure_type,
            failure_reason=f"Parent parse failed: {parent.failure_reason}",
        )
    elif not parent.sanitized:
        _append_failure(
            failure_types,
            failure_reasons,
            failure_type=parent.failure_type,
            failure_reason=f"Parent sanitization failed: {parent.failure_reason}",
        )

    if not fragment.parseable:
        _append_failure(
            failure_types,
            failure_reasons,
            failure_type=fragment.failure_type,
            failure_reason=f"Fragment parse failed: {fragment.failure_reason}",
        )
    elif not fragment.sanitized:
        _append_failure(
            failure_types,
            failure_reasons,
            failure_type=fragment.failure_type,
            failure_reason=f"Fragment sanitization failed: {fragment.failure_reason}",
        )

    connected = fragment.sanitized and is_connected_fragment(fragment_smiles)
    if fragment.sanitized and not connected:
        _append_failure(
            failure_types,
            failure_reasons,
            failure_type=ChemistryFailureType.DISCONNECTED_FRAGMENT,
            failure_reason="Fragment contains more than one connected component.",
        )

    is_substructure = False
    deletion_supported = False
    residual_smiles = None
    if parent.sanitized and fragment.sanitized and connected:
        if fragment.contains_dummy_atoms:
            is_substructure = is_valid_capped_subgraph(parent_smiles, fragment_smiles)
            substructure_failure_type = ChemistryFailureType.CAPPED_SUBGRAPH_MISMATCH
            substructure_failure_reason = (
                "Capped fragment does not correspond to a fully capped parent subgraph."
            )
        else:
            is_substructure = is_parent_substructure(parent_smiles, fragment_smiles)
            substructure_failure_type = ChemistryFailureType.NOT_SUBSTRUCTURE
            substructure_failure_reason = (
                "Fragment is not a substructure of the parent molecule."
            )
        if not is_substructure:
            _append_failure(
                failure_types,
                failure_reasons,
                failure_type=substructure_failure_type,
                failure_reason=substructure_failure_reason,
            )
        else:
            deletion_result = delete_fragment_from_parent(parent_smiles, fragment_smiles)
            deletion_supported = deletion_result.success
            residual_smiles = deletion_result.residual_smiles
            if not deletion_result.success:
                _append_failure(
                    failure_types,
                    failure_reasons,
                    failure_type=deletion_result.failure_type,
                    failure_reason=f"Deletion failed: {deletion_result.failure_reason}",
                )

    return FragmentValidationResult(
        parent_smiles=parent_smiles,
        fragment_smiles=fragment_smiles,
        parseable=fragment.parseable,
        chemically_valid=fragment.sanitized,
        connected=connected,
        is_substructure=is_substructure,
        deletion_supported=deletion_supported,
        parent_parseable=parent.parseable,
        parent_chemically_valid=parent.sanitized,
        residual_smiles=residual_smiles,
        failure_types=tuple(failure_types),
        failure_reasons=tuple(failure_reasons),
    )
