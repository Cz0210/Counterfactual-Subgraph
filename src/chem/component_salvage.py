"""Connected-component salvage for disconnected decoded fragments."""

from __future__ import annotations

from src.chem.projection import project_fragment_to_parent_subgraph
from src.chem.smiles_utils import is_rdkit_available, parse_smiles, sanitize_molecule
from src.chem.substructure import is_parent_substructure
from src.chem.types import FragmentComponentSalvageResult

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


def salvage_connected_component(
    parent_smiles: str,
    raw_fragment: str,
    *,
    method: str = "largest_then_best_parent_match",
    min_atoms: int = 3,
    max_components: int = 16,
    projection_min_score: float = 0.35,
    projection_max_candidates: int = 128,
    projection_max_atom_ratio: float = 0.70,
    projection_enable_khop3: bool = False,
    projection_mcs_timeout: int = 1,
) -> FragmentComponentSalvageResult:
    """Extract one connected component from a disconnected fragment."""

    normalized_parent = str(parent_smiles or "").strip()
    normalized_fragment = str(raw_fragment or "").strip()
    if not normalized_fragment:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="empty_fragment",
            failure_reason="empty_fragment",
            failure_stage="input",
        )
    if not is_rdkit_available() or Chem is None:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="rdkit_unavailable",
            failure_reason="rdkit_unavailable",
            failure_stage="runtime",
        )

    parsed = parse_smiles(
        normalized_fragment,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not parsed.parseable or parsed.mol is None:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="parse_failed",
            failure_reason="component_salvage_parse_failed",
            failure_stage="parse",
        )

    try:
        component_mols = tuple(
            Chem.GetMolFrags(parsed.mol, asMols=True, sanitizeFrags=False)
        )
    except Exception:  # pragma: no cover - depends on RDKit internals
        component_mols = ()
    component_count = len(component_mols)
    if component_count <= 1:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            component_count=component_count,
            reason="no_disconnected_components_detected",
            failure_reason="no_disconnected_components_detected",
            failure_stage="component_detection",
        )
    if component_count > int(max_components):
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            component_count=component_count,
            reason="too_many_components",
            failure_reason="too_many_components",
            failure_stage="component_detection",
        )

    components: list[tuple[str, int, float, bool]] = []
    for component_mol in component_mols:
        sanitized, _, _, _ = sanitize_molecule(
            component_mol,
            allow_capped_fragments=True,
        )
        if sanitized is None:
            continue
        smiles = _mol_to_smiles(sanitized)
        atom_count = _non_dummy_atom_count(sanitized)
        if not smiles or atom_count < int(min_atoms):
            continue
        strict_match = bool(
            normalized_parent and is_parent_substructure(normalized_parent, smiles)
        )
        projection_score = -1.0
        if method in {"best_parent_match", "largest_then_best_parent_match"}:
            projection = project_fragment_to_parent_subgraph(
                normalized_parent,
                smiles,
                min_score=projection_min_score,
                max_candidates=projection_max_candidates,
                min_atoms=min_atoms,
                max_atom_ratio=projection_max_atom_ratio,
                enable_khop3=projection_enable_khop3,
                mcs_timeout=projection_mcs_timeout,
            )
            projection_score = float(projection.projection_score or 0.0)
            if projection.success:
                projection_score = max(projection_score, 1.0)
        components.append((smiles, atom_count, projection_score, strict_match))

    if not components:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            component_count=component_count,
            reason="no_component_meets_min_atoms",
            failure_reason="no_component_meets_min_atoms",
            failure_stage="component_filter",
        )

    if method == "largest":
        selected = max(components, key=lambda item: (item[1], item[0]))
        salvage_method = "largest"
    elif method == "best_parent_match":
        selected = max(components, key=lambda item: (item[3], item[2], item[1], item[0]))
        salvage_method = "best_parent_match"
    else:
        largest = max(components, key=lambda item: (item[1], item[3], item[2], item[0]))
        if largest[3] or largest[2] >= float(projection_min_score):
            selected = largest
        else:
            selected = max(
                components,
                key=lambda item: (item[3], item[2], item[1], item[0]),
            )
        salvage_method = "largest_then_best_parent_match"

    return FragmentComponentSalvageResult(
        raw_fragment_smiles=normalized_fragment,
        attempted=True,
        success=True,
        component_count=component_count,
        salvage_method=salvage_method,
        salvaged_fragment_smiles=selected[0],
        salvaged_atom_count=selected[1],
        reason="component_salvage_success",
    )


def _mol_to_smiles(mol: object | None) -> str | None:
    if Chem is None or mol is None:
        return None
    try:
        return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) or "").strip()
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None


def _non_dummy_atom_count(mol: object | None) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0)
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return 0


__all__ = ["salvage_connected_component"]
