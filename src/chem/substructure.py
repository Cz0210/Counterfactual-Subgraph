"""Substructure and connectivity checks."""

from __future__ import annotations

from typing import Protocol

from src.chem.smiles_utils import is_rdkit_available, parse_smiles

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None
    rdmolops = None


class SubstructureMatcher(Protocol):
    """Backend contract for parent-substructure and connectivity checks."""

    def is_substructure(self, parent_smiles: str, fragment_smiles: str) -> bool:
        """Return whether the fragment is a genuine substructure of the parent."""

    def is_connected(self, fragment_smiles: str) -> bool:
        """Return whether the fragment corresponds to one connected component."""


def find_parent_substructure_matches(
    parent_smiles: str,
    fragment_smiles: str,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic atom-index matches for a fragment in the parent.

    For ordinary fragments this returns the RDKit match vector directly.
    For capped fragments containing dummy atoms, the match vector also includes
    the parent-side attachment atoms matched by each ``*``.
    """

    if not is_rdkit_available() or rdmolops is None or Chem is None:
        return ()

    parent = parse_smiles(parent_smiles, sanitize=True, canonicalize=False)
    fragment = parse_smiles(
        fragment_smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not parent.sanitized or parent.mol is None:
        return ()
    if not fragment.sanitized or fragment.mol is None:
        return ()
    if fragment.contains_dummy_atoms:
        return _find_capped_subgraph_matches(parent.mol, fragment.smiles, fragment.mol)

    matches = parent.mol.GetSubstructMatches(
        fragment.mol,
        useChirality=True,
        uniquify=True,
    )
    return tuple(tuple(match) for match in matches)


def is_parent_substructure(parent_smiles: str, fragment_smiles: str) -> bool:
    """Return whether the fragment is a sanitized substructure of the parent."""

    return len(find_parent_substructure_matches(parent_smiles, fragment_smiles)) > 0


def is_valid_capped_subgraph(parent_smiles: str, capped_fragment_smiles: str) -> bool:
    """Return whether a capped fragment matches one fully capped parent subgraph.

    The dummy atoms in ``capped_fragment_smiles`` represent cut-bond attachment
    points and are not themselves considered fragment atoms to be deleted.
    """

    if not is_rdkit_available() or Chem is None:
        return False

    parent = parse_smiles(parent_smiles, sanitize=True, canonicalize=False)
    fragment = parse_smiles(
        capped_fragment_smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not parent.sanitized or parent.mol is None:
        return False
    if not fragment.sanitized or fragment.mol is None:
        return False

    return len(_find_capped_subgraph_matches(parent.mol, fragment.smiles, fragment.mol)) > 0


def is_connected_fragment(fragment_smiles: str) -> bool:
    """Return whether the fragment has exactly one connected component."""

    if not is_rdkit_available() or rdmolops is None:
        return False

    fragment = parse_smiles(
        fragment_smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not fragment.sanitized or fragment.mol is None:
        return False

    if fragment.mol.GetNumAtoms() == 0:
        return False

    components = rdmolops.GetMolFrags(fragment.mol)
    return len(components) == 1


def _dummy_atom_indices(mol: object) -> tuple[int, ...]:
    return tuple(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


def _non_dummy_atom_indices(mol: object) -> tuple[int, ...]:
    return tuple(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0)


def _is_well_formed_capped_fragment(fragment_mol: object) -> bool:
    real_atom_indices = _non_dummy_atom_indices(fragment_mol)
    if not real_atom_indices:
        return False

    for atom in fragment_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        if atom.GetDegree() != 1:
            return False
        neighbor = atom.GetNeighbors()[0]
        if neighbor.GetAtomicNum() == 0:
            return False
    return True


def _build_capped_query(capped_fragment_smiles: str) -> object | None:
    if Chem is None:
        return None
    try:
        return Chem.MolFromSmarts(capped_fragment_smiles)
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None


def _find_capped_subgraph_matches(
    parent_mol: object,
    capped_fragment_smiles: str,
    fragment_mol: object,
) -> tuple[tuple[int, ...], ...]:
    if Chem is None:
        return ()
    if not _is_well_formed_capped_fragment(fragment_mol):
        return ()

    query = _build_capped_query(capped_fragment_smiles)
    if query is None:
        return ()

    matches = parent_mol.GetSubstructMatches(
        query,
        useChirality=True,
        uniquify=True,
    )
    if not matches:
        return ()

    valid_matches: list[tuple[int, ...]] = []
    dummy_indices = set(_dummy_atom_indices(fragment_mol))
    real_query_indices = _non_dummy_atom_indices(fragment_mol)

    for match in matches:
        real_parent_indices = {match[index] for index in real_query_indices}
        query_boundary_pairs: list[tuple[int, int]] = []
        valid_match = True

        for dummy_index in dummy_indices:
            dummy_atom = fragment_mol.GetAtomWithIdx(dummy_index)
            parent_attachment_index = match[dummy_index]
            if parent_attachment_index in real_parent_indices:
                valid_match = False
                break
            neighbor = dummy_atom.GetNeighbors()[0]
            parent_internal_index = match[neighbor.GetIdx()]
            query_boundary_pairs.append((parent_internal_index, parent_attachment_index))

        if not valid_match:
            continue

        actual_boundary_pairs: list[tuple[int, int]] = []
        for real_parent_index in real_parent_indices:
            parent_atom = parent_mol.GetAtomWithIdx(real_parent_index)
            for neighbor in parent_atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index not in real_parent_indices:
                    actual_boundary_pairs.append((real_parent_index, neighbor_index))

        if sorted(query_boundary_pairs) != sorted(actual_boundary_pairs):
            continue

        valid_matches.append(tuple(match))

    return tuple(valid_matches)
