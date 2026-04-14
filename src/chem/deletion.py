"""Fragment deletion interfaces for counterfactual scoring."""

from __future__ import annotations

from typing import Protocol

from src.chem.smiles_utils import (
    is_rdkit_available,
    parse_smiles,
    sanitize_molecule,
)
from src.chem.substructure import (
    find_parent_substructure_matches,
    is_connected_fragment,
    is_valid_capped_subgraph,
)
from src.chem.types import ChemistryFailureType, DeletionResult

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


class FragmentDeletionEngine(Protocol):
    """Backend contract for deletion-based counterfactual construction."""

    def delete(self, parent_smiles: str, fragment_smiles: str) -> DeletionResult:
        """Remove one fragment candidate from the parent molecule."""


def _failure(
    parent_smiles: str,
    fragment_smiles: str,
    *,
    failure_type: ChemistryFailureType,
    failure_reason: str,
    match_count: int = 0,
    selected_match: tuple[int, ...] = (),
) -> DeletionResult:
    return DeletionResult(
        parent_smiles=parent_smiles,
        fragment_smiles=fragment_smiles,
        residual_smiles=None,
        success=False,
        failure_type=failure_type,
        failure_reason=failure_reason,
        match_count=match_count,
        selected_match=selected_match,
    )


def _clear_broken_aromatic_flags(mol: object) -> None:
    """Repair a common post-deletion failure mode for aromatic residues."""

    if Chem is None:
        return

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() and not bond.IsInRing():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)


def delete_fragment_from_parent(parent_smiles: str, fragment_smiles: str) -> DeletionResult:
    """Delete one connected fragment match from the parent molecule."""

    if not is_rdkit_available() or Chem is None:
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.RDKIT_UNAVAILABLE,
            failure_reason="RDKit is required for fragment deletion but is not installed.",
        )

    parent = parse_smiles(parent_smiles, sanitize=True, canonicalize=False)
    if not parent.sanitized or parent.mol is None:
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=parent.failure_type or ChemistryFailureType.SANITIZE_FAILED,
            failure_reason=f"Parent molecule is not usable for deletion: {parent.failure_reason}",
        )

    fragment = parse_smiles(
        fragment_smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not fragment.sanitized or fragment.mol is None:
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=fragment.failure_type or ChemistryFailureType.SANITIZE_FAILED,
            failure_reason=f"Fragment molecule is not usable for deletion: {fragment.failure_reason}",
        )

    if not is_connected_fragment(fragment_smiles):
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.DISCONNECTED_FRAGMENT,
            failure_reason="Fragment deletion expects one connected fragment.",
        )

    if fragment.contains_dummy_atoms:
        return _delete_capped_fragment_from_parent(parent, fragment)

    matches = find_parent_substructure_matches(parent_smiles, fragment_smiles)
    if not matches:
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.NO_SUBSTRUCTURE_MATCH,
            failure_reason="Fragment is not a substructure of the parent molecule.",
        )

    selected_match = tuple(matches[0])
    rw_mol = Chem.RWMol(parent.mol)
    for atom_index in sorted(selected_match, reverse=True):
        rw_mol.RemoveAtom(atom_index)
    residual_mol = rw_mol.GetMol()

    if residual_mol.GetNumAtoms() == 0:
        return DeletionResult(
            parent_smiles=parent_smiles,
            fragment_smiles=fragment_smiles,
            residual_smiles="",
            success=True,
            match_count=len(matches),
            selected_match=selected_match,
            residual_atom_count=0,
        )

    try:
        Chem.SanitizeMol(residual_mol)
    except Exception:
        _clear_broken_aromatic_flags(residual_mol)
        try:
            Chem.SanitizeMol(residual_mol)
        except Exception as exc:
            return _failure(
                parent_smiles,
                fragment_smiles,
                failure_type=ChemistryFailureType.RESIDUAL_SANITIZE_FAILED,
                failure_reason=f"Residual molecule could not be sanitized after deletion: {exc}",
                match_count=len(matches),
                selected_match=selected_match,
            )

    residual_smiles = Chem.MolToSmiles(residual_mol, canonical=True)
    return DeletionResult(
        parent_smiles=parent_smiles,
        fragment_smiles=fragment_smiles,
        residual_smiles=residual_smiles,
        success=True,
        match_count=len(matches),
        selected_match=selected_match,
        residual_atom_count=residual_mol.GetNumAtoms(),
    )


def get_remainder_graph(parent_smiles: str, capped_fragment_smiles: str) -> str:
    """Return the capped remainder graph or raise a clear error."""

    result = delete_fragment_from_parent(parent_smiles, capped_fragment_smiles)
    if not result.success or result.residual_smiles is None:
        failure_type = result.failure_type or ChemistryFailureType.RESIDUAL_SANITIZE_FAILED
        detail = result.failure_reason or "unknown deletion failure"
        raise ValueError(
            "Could not construct remainder graph "
            f"({failure_type.value}): {detail}"
        )
    return result.residual_smiles


def _delete_capped_fragment_from_parent(parent: object, fragment: object) -> DeletionResult:
    parent_smiles = parent.smiles
    fragment_smiles = fragment.smiles

    if not is_valid_capped_subgraph(parent_smiles, fragment_smiles):
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.CAPPED_SUBGRAPH_MISMATCH,
            failure_reason=(
                "Capped fragment did not match a fully capped subgraph of the parent."
            ),
        )

    matches = find_parent_substructure_matches(parent_smiles, fragment_smiles)
    if not matches:
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.CAPPED_SUBGRAPH_MISMATCH,
            failure_reason="No capped subgraph match was found in the parent molecule.",
        )

    try:
        query = Chem.MolFromSmarts(fragment_smiles)
    except Exception as exc:  # pragma: no cover - depends on RDKit internals
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.INVALID_CAPPED_FRAGMENT,
            failure_reason=f"RDKit could not build a capped fragment query: {exc}",
        )

    if query is None:
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.INVALID_CAPPED_FRAGMENT,
            failure_reason="RDKit could not build a capped fragment query.",
        )

    selected_match = tuple(matches[0])
    remainder_mol = Chem.ReplaceCore(
        parent.mol,
        query,
        selected_match,
        False,
        False,
        True,
    )
    if remainder_mol is None:
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.RESIDUAL_SANITIZE_FAILED,
            failure_reason=(
                "RDKit could not construct a capped remainder graph from the matched fragment."
            ),
            match_count=len(matches),
            selected_match=selected_match,
        )

    _clear_dummy_atom_isotopes(remainder_mol)
    if remainder_mol.GetNumAtoms() == 0:
        return DeletionResult(
            parent_smiles=parent_smiles,
            fragment_smiles=fragment_smiles,
            residual_smiles="",
            success=True,
            match_count=len(matches),
            selected_match=selected_match,
            residual_atom_count=0,
        )

    sanitized_remainder, _, _, failure_reason = sanitize_molecule(
        remainder_mol,
        allow_capped_fragments=True,
    )
    if sanitized_remainder is None:
        return _failure(
            parent_smiles,
            fragment_smiles,
            failure_type=ChemistryFailureType.RESIDUAL_SANITIZE_FAILED,
            failure_reason=(
                "Residual molecule could not be sanitized after capped deletion: "
                f"{failure_reason}"
            ),
            match_count=len(matches),
            selected_match=selected_match,
        )

    _clear_dummy_atom_isotopes(sanitized_remainder)
    residual_smiles = Chem.MolToSmiles(sanitized_remainder, canonical=True)
    return DeletionResult(
        parent_smiles=parent_smiles,
        fragment_smiles=fragment_smiles,
        residual_smiles=residual_smiles,
        success=True,
        match_count=len(matches),
        selected_match=selected_match,
        residual_atom_count=sanitized_remainder.GetNumAtoms(),
    )


def _clear_dummy_atom_isotopes(mol: object) -> None:
    if Chem is None:
        return
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope():
            atom.SetIsotope(0)
