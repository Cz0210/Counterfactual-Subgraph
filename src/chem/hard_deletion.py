"""Versioned hard-deletion actions for molecular counterfactual evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

try:  # pragma: no cover - deployment dependency
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


CONNECTED_ACTION_SEMANTICS = "connected_sanitized_residual_v1"
CONNECTED_MATCH_SELECTION_POLICY = (
    "existential_min_wnode_among_valid_connected_strict_flips_v1"
)
CONNECTED_WNODE_CACHE_NAMESPACE = "molclr_node_wasserstein_connected_residual_v3"


@dataclass(frozen=True, slots=True)
class HardDeletionOutcome:
    """One exact atom-match deletion and its molecular-validity audit."""

    parent_id: str | None
    candidate_id: str | None
    match_id: int
    match_atom_indices: tuple[int, ...]
    removed_atom_symbols: tuple[str, ...]
    removed_atom_count: int
    removed_bond_count: int
    boundary_bond_count: int
    residual_smiles: str | None
    residual_heavy_atom_count: int
    residual_num_components: int
    residual_connected: bool
    sanitize_ok: bool
    contains_dot: bool
    valid: bool
    invalid_reason: str | None
    atom_delete_ratio: float | None
    bond_delete_ratio: float | None
    residual_atom_count: int
    residual_bond_count: int
    action_semantics_version: str = CONNECTED_ACTION_SEMANTICS

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "match_index": self.match_id,
                "match_atoms": list(self.match_atom_indices),
                "num_match_atoms": self.removed_atom_count,
                "num_removed_atoms": self.removed_atom_count,
                "num_removed_bonds": self.removed_bond_count,
                "num_components": self.residual_num_components,
                "delete_valid": self.valid,
                "error": self.invalid_reason,
            }
        )
        return payload


def _invalid_outcome(
    *,
    parent_id: str | None,
    candidate_id: str | None,
    match_id: int,
    match_atom_indices: tuple[int, ...],
    removed_atom_symbols: tuple[str, ...] = (),
    removed_bond_count: int = 0,
    boundary_bond_count: int = 0,
    residual_smiles: str | None = None,
    residual_heavy_atom_count: int = 0,
    residual_num_components: int = 0,
    residual_connected: bool = False,
    sanitize_ok: bool = False,
    contains_dot: bool = False,
    atom_delete_ratio: float | None = None,
    bond_delete_ratio: float | None = None,
    residual_atom_count: int = 0,
    residual_bond_count: int = 0,
    reason: str,
) -> HardDeletionOutcome:
    return HardDeletionOutcome(
        parent_id=parent_id,
        candidate_id=candidate_id,
        match_id=int(match_id),
        match_atom_indices=match_atom_indices,
        removed_atom_symbols=removed_atom_symbols,
        removed_atom_count=len(match_atom_indices),
        removed_bond_count=int(removed_bond_count),
        boundary_bond_count=int(boundary_bond_count),
        residual_smiles=residual_smiles,
        residual_heavy_atom_count=int(residual_heavy_atom_count),
        residual_num_components=int(residual_num_components),
        residual_connected=bool(residual_connected),
        sanitize_ok=bool(sanitize_ok),
        contains_dot=bool(contains_dot),
        valid=False,
        invalid_reason=reason,
        atom_delete_ratio=atom_delete_ratio,
        bond_delete_ratio=bond_delete_ratio,
        residual_atom_count=int(residual_atom_count),
        residual_bond_count=int(residual_bond_count),
    )


def apply_hard_deletion_match(
    parent_mol: Any,
    match_atom_indices: Sequence[int],
    *,
    parent_id: str | None = None,
    candidate_id: str | None = None,
    match_id: int = 0,
    require_nonempty: bool = True,
    require_sanitized: bool = True,
    require_single_component: bool = True,
) -> HardDeletionOutcome:
    """Apply one exact deletion without repairing or dropping components."""

    indices = tuple(sorted({int(value) for value in match_atom_indices}))
    if Chem is None or parent_mol is None:
        return _invalid_outcome(
            parent_id=parent_id,
            candidate_id=candidate_id,
            match_id=match_id,
            match_atom_indices=indices,
            reason="rdkit_unavailable_or_parent_invalid",
        )
    parent_atom_count = int(parent_mol.GetNumAtoms())
    parent_bond_count = int(parent_mol.GetNumBonds())
    if not indices:
        return _invalid_outcome(
            parent_id=parent_id,
            candidate_id=candidate_id,
            match_id=match_id,
            match_atom_indices=indices,
            reason="empty_match",
        )
    if indices[0] < 0 or indices[-1] >= parent_atom_count:
        return _invalid_outcome(
            parent_id=parent_id,
            candidate_id=candidate_id,
            match_id=match_id,
            match_atom_indices=indices,
            reason="match_atom_index_out_of_range",
        )

    selected = set(indices)
    symbols = tuple(parent_mol.GetAtomWithIdx(index).GetSymbol() for index in indices)
    removed_bonds = 0
    boundary_bonds = 0
    for bond in parent_mol.GetBonds():
        left = int(bond.GetBeginAtomIdx()) in selected
        right = int(bond.GetEndAtomIdx()) in selected
        if left or right:
            removed_bonds += 1
        if left != right:
            boundary_bonds += 1
    atom_ratio = len(indices) / parent_atom_count if parent_atom_count else None
    bond_ratio = removed_bonds / parent_bond_count if parent_bond_count else 0.0

    try:
        editable = Chem.RWMol(Chem.Mol(parent_mol))
        for atom_index in reversed(indices):
            editable.RemoveAtom(atom_index)
        residual = editable.GetMol()
    except Exception as exc:
        return _invalid_outcome(
            parent_id=parent_id,
            candidate_id=candidate_id,
            match_id=match_id,
            match_atom_indices=indices,
            removed_atom_symbols=symbols,
            removed_bond_count=removed_bonds,
            boundary_bond_count=boundary_bonds,
            atom_delete_ratio=atom_ratio,
            bond_delete_ratio=bond_ratio,
            reason=f"delete_failed:{exc}",
        )

    residual_atom_count = int(residual.GetNumAtoms())
    residual_bond_count = int(residual.GetNumBonds())
    if require_nonempty and residual_atom_count == 0:
        return _invalid_outcome(
            parent_id=parent_id,
            candidate_id=candidate_id,
            match_id=match_id,
            match_atom_indices=indices,
            removed_atom_symbols=symbols,
            removed_bond_count=removed_bonds,
            boundary_bond_count=boundary_bonds,
            atom_delete_ratio=atom_ratio,
            bond_delete_ratio=bond_ratio,
            residual_atom_count=residual_atom_count,
            residual_bond_count=residual_bond_count,
            reason="empty_residual_after_deletion",
        )

    sanitize_ok = False
    try:
        Chem.SanitizeMol(residual)
        sanitize_ok = True
    except Exception as exc:
        if require_sanitized:
            return _invalid_outcome(
                parent_id=parent_id,
                candidate_id=candidate_id,
                match_id=match_id,
                match_atom_indices=indices,
                removed_atom_symbols=symbols,
                removed_bond_count=removed_bonds,
                boundary_bond_count=boundary_bonds,
                atom_delete_ratio=atom_ratio,
                bond_delete_ratio=bond_ratio,
                residual_atom_count=residual_atom_count,
                residual_bond_count=residual_bond_count,
                reason=f"residual_sanitize_failed:{exc}",
            )

    components = tuple(Chem.GetMolFrags(residual))
    component_count = len(components)
    connected = component_count == 1
    heavy_atom_count = sum(atom.GetAtomicNum() > 1 for atom in residual.GetAtoms())
    try:
        residual_smiles = Chem.MolToSmiles(residual, canonical=True)
    except Exception as exc:
        return _invalid_outcome(
            parent_id=parent_id,
            candidate_id=candidate_id,
            match_id=match_id,
            match_atom_indices=indices,
            removed_atom_symbols=symbols,
            removed_bond_count=removed_bonds,
            boundary_bond_count=boundary_bonds,
            residual_heavy_atom_count=heavy_atom_count,
            residual_num_components=component_count,
            residual_connected=connected,
            sanitize_ok=sanitize_ok,
            atom_delete_ratio=atom_ratio,
            bond_delete_ratio=bond_ratio,
            residual_atom_count=residual_atom_count,
            residual_bond_count=residual_bond_count,
            reason=f"residual_smiles_failed:{exc}",
        )
    contains_dot = "." in residual_smiles
    invalid_reason: str | None = None
    if require_nonempty and (not residual_smiles or heavy_atom_count <= 0):
        invalid_reason = "empty_or_no_heavy_atom_residual"
    elif require_sanitized and not sanitize_ok:
        invalid_reason = "residual_not_sanitized"
    elif require_single_component and (not connected or contains_dot):
        invalid_reason = "disconnected_residual"
    return HardDeletionOutcome(
        parent_id=parent_id,
        candidate_id=candidate_id,
        match_id=int(match_id),
        match_atom_indices=indices,
        removed_atom_symbols=symbols,
        removed_atom_count=len(indices),
        removed_bond_count=removed_bonds,
        boundary_bond_count=boundary_bonds,
        residual_smiles=residual_smiles,
        residual_heavy_atom_count=heavy_atom_count,
        residual_num_components=component_count,
        residual_connected=connected,
        sanitize_ok=sanitize_ok,
        contains_dot=contains_dot,
        valid=invalid_reason is None,
        invalid_reason=invalid_reason,
        atom_delete_ratio=atom_ratio,
        bond_delete_ratio=bond_ratio,
        residual_atom_count=residual_atom_count,
        residual_bond_count=residual_bond_count,
    )


def enumerate_connected_hard_deletions(
    parent_smiles: str,
    fragment_smiles: str,
    *,
    parent_id: str | None = None,
    candidate_id: str | None = None,
) -> list[HardDeletionOutcome]:
    """Enumerate exact unique RDKit matches under the connected policy."""

    if Chem is None:
        return []
    try:
        parent = Chem.MolFromSmiles(str(parent_smiles or "").strip(), sanitize=True)
        query = Chem.MolFromSmiles(str(fragment_smiles or "").strip(), sanitize=True)
    except Exception:
        return []
    if parent is None or query is None or query.GetNumAtoms() <= 0:
        return []
    if any(atom.GetAtomicNum() == 0 for atom in query.GetAtoms()):
        return []
    try:
        raw_matches = parent.GetSubstructMatches(
            query,
            useChirality=True,
            uniquify=True,
        )
    except Exception:
        return []
    canonical_matches: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw in raw_matches:
        atom_set = tuple(sorted(int(value) for value in raw))
        if atom_set not in seen:
            seen.add(atom_set)
            canonical_matches.append(atom_set)
    return [
        apply_hard_deletion_match(
            parent,
            match,
            parent_id=parent_id,
            candidate_id=candidate_id,
            match_id=index,
            require_nonempty=True,
            require_sanitized=True,
            require_single_component=True,
        )
        for index, match in enumerate(canonical_matches)
    ]


__all__ = [
    "CONNECTED_ACTION_SEMANTICS",
    "CONNECTED_MATCH_SELECTION_POLICY",
    "CONNECTED_WNODE_CACHE_NAMESPACE",
    "HardDeletionOutcome",
    "apply_hard_deletion_match",
    "enumerate_connected_hard_deletions",
]
