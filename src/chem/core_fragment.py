"""Core-fragment normalization and parent-match recovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.chem.smiles_utils import is_rdkit_available, parse_smiles, sanitize_molecule

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


@dataclass(frozen=True, slots=True)
class CoreFragmentNormalizationResult:
    """Normalized core-fragment view derived from one raw fragment string."""

    raw_fragment_smiles: str
    raw_parse_ok: bool
    raw_sanitize_ok: bool
    raw_has_dummy: bool
    raw_dummy_count: int
    core_fragment_smiles: str | None
    core_parse_ok: bool
    core_connected: bool
    core_atom_count: int
    component_count_before_selection: int
    kept_largest_component: bool
    failure_tag: str | None
    failure_reason: str | None
    raw_canonical_smiles: str | None = None
    core_mol: object | None = None


@dataclass(frozen=True, slots=True)
class BoundaryBondRecord:
    """One parent boundary bond around a matched core fragment."""

    parent_bond_index: int
    parent_atom_index: int
    outside_atom_index: int
    fragment_atom_position: int
    bond_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_bond_index": self.parent_bond_index,
            "parent_atom_index": self.parent_atom_index,
            "outside_atom_index": self.outside_atom_index,
            "fragment_atom_position": self.fragment_atom_position,
            "bond_type": self.bond_type,
        }


@dataclass(frozen=True, slots=True)
class ParentFragmentMatchResult:
    """Strict parent-subgraph match information for a connected core fragment."""

    parent_smiles: str
    fragment_smiles: str
    matched: bool
    match_atom_indices: tuple[int, ...]
    boundary_bonds: tuple[BoundaryBondRecord, ...]
    attachment_points: tuple[int, ...]
    explanation_fragment_with_dummy: str | None
    atom_count: int
    atom_ratio: float | None
    full_parent: bool
    reason: str | None = None

    def boundary_bonds_as_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.boundary_bonds]


def normalize_core_fragment(
    raw_fragment_smiles: str,
    *,
    keep_largest_component: bool = True,
) -> CoreFragmentNormalizationResult:
    """Normalize a raw fragment into one no-dummy connected core fragment."""

    normalized = str(raw_fragment_smiles or "").strip()
    if not normalized:
        return CoreFragmentNormalizationResult(
            raw_fragment_smiles=normalized,
            raw_parse_ok=False,
            raw_sanitize_ok=False,
            raw_has_dummy=False,
            raw_dummy_count=0,
            core_fragment_smiles=None,
            core_parse_ok=False,
            core_connected=False,
            core_atom_count=0,
            component_count_before_selection=0,
            kept_largest_component=False,
            failure_tag="empty_fragment",
            failure_reason="Fragment string is empty.",
        )

    if not is_rdkit_available() or Chem is None:
        return CoreFragmentNormalizationResult(
            raw_fragment_smiles=normalized,
            raw_parse_ok=False,
            raw_sanitize_ok=False,
            raw_has_dummy="*" in normalized,
            raw_dummy_count=normalized.count("*"),
            core_fragment_smiles=None,
            core_parse_ok=False,
            core_connected=False,
            core_atom_count=0,
            component_count_before_selection=0,
            kept_largest_component=False,
            failure_tag="rdkit_unavailable",
            failure_reason="RDKit is required for core-fragment normalization.",
        )

    parsed_raw = parse_smiles(
        normalized,
        sanitize=False,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not parsed_raw.parseable or parsed_raw.mol is None:
        return CoreFragmentNormalizationResult(
            raw_fragment_smiles=normalized,
            raw_parse_ok=False,
            raw_sanitize_ok=False,
            raw_has_dummy="*" in normalized,
            raw_dummy_count=normalized.count("*"),
            core_fragment_smiles=None,
            core_parse_ok=False,
            core_connected=False,
            core_atom_count=0,
            component_count_before_selection=0,
            kept_largest_component=False,
            failure_tag="parse_failed",
            failure_reason=parsed_raw.failure_reason or "RDKit could not parse the fragment.",
        )

    raw_sanitized, _, _, raw_failure_reason = sanitize_molecule(
        parsed_raw.mol,
        allow_capped_fragments=True,
    )
    working_raw = raw_sanitized if raw_sanitized is not None else parsed_raw.mol
    raw_has_dummy = _has_dummy_atoms(working_raw) or "*" in normalized
    raw_dummy_count = max(_dummy_atom_count(working_raw), normalized.count("*"))
    raw_canonical_smiles = _mol_to_smiles(working_raw)

    if raw_has_dummy:
        core_candidate = _remove_dummy_atoms(working_raw)
    else:
        core_candidate = Chem.Mol(working_raw)

    if core_candidate is None or core_candidate.GetNumAtoms() == 0:
        return CoreFragmentNormalizationResult(
            raw_fragment_smiles=normalized,
            raw_parse_ok=True,
            raw_sanitize_ok=raw_sanitized is not None,
            raw_has_dummy=raw_has_dummy,
            raw_dummy_count=raw_dummy_count,
            core_fragment_smiles=None,
            core_parse_ok=False,
            core_connected=False,
            core_atom_count=0,
            component_count_before_selection=0,
            kept_largest_component=False,
            failure_tag="sanitize_failed",
            failure_reason="Fragment core vanished after dummy removal.",
            raw_canonical_smiles=raw_canonical_smiles,
        )

    component_mols = Chem.GetMolFrags(core_candidate, asMols=True, sanitizeFrags=False)
    component_count = len(component_mols)
    sanitized_components: list[tuple[int, int, str, object]] = []
    for component in component_mols or (core_candidate,):
        if component.GetNumAtoms() == 0:
            continue
        sanitized_component, _, _, _ = sanitize_molecule(
            component,
            allow_capped_fragments=False,
        )
        if sanitized_component is None:
            continue
        component_smiles = _mol_to_smiles(sanitized_component)
        if not component_smiles:
            continue
        sanitized_components.append(
            (
                _non_dummy_atom_count(sanitized_component),
                int(sanitized_component.GetNumAtoms()),
                component_smiles,
                sanitized_component,
            )
        )

    if not sanitized_components:
        return CoreFragmentNormalizationResult(
            raw_fragment_smiles=normalized,
            raw_parse_ok=True,
            raw_sanitize_ok=raw_sanitized is not None,
            raw_has_dummy=raw_has_dummy,
            raw_dummy_count=raw_dummy_count,
            core_fragment_smiles=None,
            core_parse_ok=False,
            core_connected=False,
            core_atom_count=0,
            component_count_before_selection=component_count,
            kept_largest_component=False,
            failure_tag="sanitize_failed",
            failure_reason=raw_failure_reason or "No sanitized core component remained.",
            raw_canonical_smiles=raw_canonical_smiles,
        )

    sanitized_components.sort(key=lambda item: (-item[0], -item[1], item[2]))
    if component_count > 1 and not keep_largest_component:
        return CoreFragmentNormalizationResult(
            raw_fragment_smiles=normalized,
            raw_parse_ok=True,
            raw_sanitize_ok=raw_sanitized is not None,
            raw_has_dummy=raw_has_dummy,
            raw_dummy_count=raw_dummy_count,
            core_fragment_smiles=None,
            core_parse_ok=False,
            core_connected=False,
            core_atom_count=0,
            component_count_before_selection=component_count,
            kept_largest_component=False,
            failure_tag="fragment_not_connected",
            failure_reason="Core fragment is disconnected.",
            raw_canonical_smiles=raw_canonical_smiles,
        )

    _, _, selected_smiles, selected_mol = sanitized_components[0]
    selected_atom_count = _non_dummy_atom_count(selected_mol)
    return CoreFragmentNormalizationResult(
        raw_fragment_smiles=normalized,
        raw_parse_ok=True,
        raw_sanitize_ok=raw_sanitized is not None,
        raw_has_dummy=raw_has_dummy,
        raw_dummy_count=raw_dummy_count,
        core_fragment_smiles=selected_smiles,
        core_parse_ok=True,
        core_connected=True,
        core_atom_count=selected_atom_count,
        component_count_before_selection=component_count,
        kept_largest_component=bool(component_count > 1 and keep_largest_component),
        failure_tag=None,
        failure_reason=None,
        raw_canonical_smiles=raw_canonical_smiles,
        core_mol=selected_mol,
    )


def match_core_fragment_to_parent(
    parent_smiles: str,
    fragment_smiles: str,
) -> ParentFragmentMatchResult:
    """Return one strict parent-subgraph match plus boundary recovery metadata."""

    normalized_parent = str(parent_smiles or "").strip()
    normalized_fragment = str(fragment_smiles or "").strip()
    if not normalized_parent or not normalized_fragment:
        return ParentFragmentMatchResult(
            parent_smiles=normalized_parent,
            fragment_smiles=normalized_fragment,
            matched=False,
            match_atom_indices=(),
            boundary_bonds=(),
            attachment_points=(),
            explanation_fragment_with_dummy=None,
            atom_count=0,
            atom_ratio=None,
            full_parent=False,
            reason="missing_parent_or_fragment",
        )

    if not is_rdkit_available() or Chem is None:
        return ParentFragmentMatchResult(
            parent_smiles=normalized_parent,
            fragment_smiles=normalized_fragment,
            matched=False,
            match_atom_indices=(),
            boundary_bonds=(),
            attachment_points=(),
            explanation_fragment_with_dummy=None,
            atom_count=0,
            atom_ratio=None,
            full_parent=False,
            reason="rdkit_unavailable",
        )

    parent = parse_smiles(
        normalized_parent,
        sanitize=True,
        canonicalize=True,
        allow_capped_fragments=False,
    )
    fragment = parse_smiles(
        normalized_fragment,
        sanitize=True,
        canonicalize=True,
        allow_capped_fragments=False,
    )
    if not parent.sanitized or parent.mol is None:
        return ParentFragmentMatchResult(
            parent_smiles=normalized_parent,
            fragment_smiles=normalized_fragment,
            matched=False,
            match_atom_indices=(),
            boundary_bonds=(),
            attachment_points=(),
            explanation_fragment_with_dummy=None,
            atom_count=0,
            atom_ratio=None,
            full_parent=False,
            reason="parent_parse_failed",
        )
    if not fragment.sanitized or fragment.mol is None:
        return ParentFragmentMatchResult(
            parent_smiles=normalized_parent,
            fragment_smiles=normalized_fragment,
            matched=False,
            match_atom_indices=(),
            boundary_bonds=(),
            attachment_points=(),
            explanation_fragment_with_dummy=None,
            atom_count=0,
            atom_ratio=None,
            full_parent=False,
            reason="fragment_parse_failed",
        )

    matches = parent.mol.GetSubstructMatches(
        fragment.mol,
        useChirality=True,
        uniquify=True,
    )
    if not matches:
        return ParentFragmentMatchResult(
            parent_smiles=normalized_parent,
            fragment_smiles=normalized_fragment,
            matched=False,
            match_atom_indices=(),
            boundary_bonds=(),
            attachment_points=(),
            explanation_fragment_with_dummy=None,
            atom_count=int(fragment.mol.GetNumAtoms()),
            atom_ratio=None,
            full_parent=False,
            reason="not_substructure",
        )

    selected_match = tuple(int(index) for index in matches[0])
    boundary_bonds = _collect_boundary_bonds(parent.mol, selected_match)
    attachment_points = tuple(
        sorted({record.parent_atom_index for record in boundary_bonds})
    )
    explanation_fragment = build_dummy_fragment_from_parent_match(
        parent_mol=parent.mol,
        match_atom_indices=selected_match,
        boundary_bonds=boundary_bonds,
    )
    atom_count = int(fragment.mol.GetNumAtoms())
    parent_atom_count = int(parent.mol.GetNumAtoms())
    fragment_canonical = fragment.canonical_smiles or normalized_fragment
    parent_canonical = parent.canonical_smiles or normalized_parent
    return ParentFragmentMatchResult(
        parent_smiles=normalized_parent,
        fragment_smiles=fragment_canonical,
        matched=True,
        match_atom_indices=selected_match,
        boundary_bonds=boundary_bonds,
        attachment_points=attachment_points,
        explanation_fragment_with_dummy=explanation_fragment,
        atom_count=atom_count,
        atom_ratio=(atom_count / parent_atom_count) if parent_atom_count > 0 else None,
        full_parent=bool(fragment_canonical == parent_canonical),
        reason="ok",
    )


def build_dummy_fragment_from_parent_match(
    *,
    parent_mol: object,
    match_atom_indices: tuple[int, ...],
    boundary_bonds: tuple[BoundaryBondRecord, ...] | None = None,
) -> str | None:
    """Recover a capped explanation fragment by re-attaching dummy atoms."""

    if Chem is None or parent_mol is None or not match_atom_indices:
        return None

    boundary_records = (
        boundary_bonds
        if boundary_bonds is not None
        else _collect_boundary_bonds(parent_mol, match_atom_indices)
    )
    atom_position_map = {
        int(parent_atom_index): fragment_position
        for fragment_position, parent_atom_index in enumerate(match_atom_indices)
    }

    rw_mol = Chem.RWMol()
    for parent_atom_index in match_atom_indices:
        parent_atom = parent_mol.GetAtomWithIdx(int(parent_atom_index))
        rw_mol.AddAtom(Chem.Atom(parent_atom))

    for bond in parent_mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin not in atom_position_map or end not in atom_position_map:
            continue
        rw_mol.AddBond(
            atom_position_map[begin],
            atom_position_map[end],
            bond.GetBondType(),
        )

    for boundary in boundary_records:
        dummy_index = rw_mol.AddAtom(Chem.Atom(0))
        rw_mol.AddBond(
            int(boundary.fragment_atom_position),
            int(dummy_index),
            _bond_type_from_string(boundary.bond_type),
        )

    fragment_mol = rw_mol.GetMol()
    sanitized_fragment, _, _, _ = sanitize_molecule(
        fragment_mol,
        allow_capped_fragments=True,
    )
    return _mol_to_smiles(sanitized_fragment or fragment_mol)


def _collect_boundary_bonds(
    parent_mol: object,
    match_atom_indices: tuple[int, ...],
) -> tuple[BoundaryBondRecord, ...]:
    selected = {int(index) for index in match_atom_indices}
    atom_position_map = {
        int(parent_atom_index): fragment_position
        for fragment_position, parent_atom_index in enumerate(match_atom_indices)
    }
    boundary_bonds: list[BoundaryBondRecord] = []
    for parent_atom_index in match_atom_indices:
        atom = parent_mol.GetAtomWithIdx(int(parent_atom_index))
        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtom(atom)
            neighbor_index = int(neighbor.GetIdx())
            if neighbor_index in selected:
                continue
            boundary_bonds.append(
                BoundaryBondRecord(
                    parent_bond_index=int(bond.GetIdx()),
                    parent_atom_index=int(parent_atom_index),
                    outside_atom_index=neighbor_index,
                    fragment_atom_position=int(atom_position_map[int(parent_atom_index)]),
                    bond_type=str(bond.GetBondType()),
                )
            )
    boundary_bonds.sort(
        key=lambda record: (
            record.parent_atom_index,
            record.outside_atom_index,
            record.parent_bond_index,
        )
    )
    return tuple(boundary_bonds)


def _remove_dummy_atoms(mol: object) -> object | None:
    if Chem is None or mol is None:
        return None
    editable = Chem.RWMol(Chem.Mol(mol))
    dummy_indices = sorted(
        (atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    )
    for atom_index in dummy_indices:
        editable.RemoveAtom(int(atom_index))
    result = editable.GetMol()
    return result if result.GetNumAtoms() > 0 else None


def _has_dummy_atoms(mol: object | None) -> bool:
    return _dummy_atom_count(mol) > 0


def _dummy_atom_count(mol: object | None) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return 0


def _non_dummy_atom_count(mol: object | None) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0)
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return 0


def _mol_to_smiles(mol: object | None) -> str | None:
    if Chem is None or mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None


def _bond_type_from_string(value: str) -> object:
    if Chem is None:
        return value
    mapping = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
    }
    return mapping.get(str(value), Chem.BondType.SINGLE)


__all__ = [
    "BoundaryBondRecord",
    "CoreFragmentNormalizationResult",
    "ParentFragmentMatchResult",
    "build_dummy_fragment_from_parent_match",
    "match_core_fragment_to_parent",
    "normalize_core_fragment",
]
