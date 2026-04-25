"""Parent-aware fragment repair helpers for decoded PPO generations."""

from __future__ import annotations

from collections.abc import Iterable

from src.chem.smiles_utils import is_rdkit_available, parse_smiles, sanitize_molecule
from src.chem.substructure import is_connected_fragment, is_parent_substructure
from src.chem.types import FragmentRepairResult

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import BRICS, rdMolDescriptors
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None
    DataStructs = None
    BRICS = None
    rdMolDescriptors = None


_FUNCTIONAL_GROUP_SMARTS: tuple[tuple[str, str], ...] = (
    ("smarts_carboxyl", "[CX3](=O)[OX2H1,O-]"),
    ("smarts_amide", "[NX3][CX3](=[OX1])[#6]"),
    ("smarts_ester", "[CX3](=O)[OX2][#6]"),
    ("smarts_sulfonamide", "[SX4](=[OX1])(=[OX1])[NX3]"),
    ("smarts_nitrile", "[CX2]#N"),
    ("smarts_halogen", "[F,Cl,Br,I]"),
    ("smarts_alcohol", "[OX2H][#6]"),
)

_REPAIR_SOURCE_PRIORITY = {
    "ring": 3,
    "brics": 2,
    "smarts": 1,
}


def repair_fragment_to_parent_subgraph(
    parent_smiles: str,
    raw_fragment: str,
    *,
    min_similarity: float = 0.35,
    max_candidates: int = 24,
) -> FragmentRepairResult:
    """Try to replace one invalid decoded fragment with a similar parent-derived subgraph."""

    normalized_parent = str(parent_smiles or "").strip()
    normalized_fragment = str(raw_fragment or "").strip()
    if not normalized_parent or not normalized_fragment:
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="missing_parent_or_fragment",
        )
    if not is_rdkit_available() or Chem is None:
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="rdkit_unavailable",
        )
    if max_candidates <= 0:
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="invalid_max_candidates",
        )

    parent = parse_smiles(
        normalized_parent,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=False,
    )
    if not parent.sanitized or parent.mol is None:
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="parent_parse_failed",
        )

    fragment = parse_smiles(
        normalized_fragment,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not fragment.sanitized or fragment.mol is None:
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="parse_failed",
        )
    if not is_connected_fragment(normalized_fragment):
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="fragment_not_connected",
        )
    if is_parent_substructure(normalized_parent, normalized_fragment):
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=True,
            repaired_fragment_smiles=normalized_fragment,
            repair_source="identity",
            repair_similarity=1.0,
            reason="already_parent_substructure",
        )

    query_core = _core_mol_from_parsed(fragment)
    if query_core is None:
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            reason="query_core_unavailable",
        )

    best_candidate: tuple[float, int, int, str, str] | None = None
    candidate_count = 0
    for candidate_smiles, repair_source in _iter_parent_repair_candidates(
        parent.mol,
        max_candidates=max_candidates,
    ):
        candidate_count += 1
        similarity = _tanimoto_similarity(query_core, candidate_smiles)
        if similarity is None:
            continue
        ranking = (
            float(similarity),
            int(_REPAIR_SOURCE_PRIORITY.get(repair_source, 0)),
            -len(candidate_smiles),
            candidate_smiles,
            repair_source,
        )
        if best_candidate is None or ranking > best_candidate:
            best_candidate = ranking

    if best_candidate is None:
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            reason="no_parent_repair_candidate",
            candidate_count=candidate_count,
        )

    best_similarity, _, _, repaired_fragment, repair_source = best_candidate
    if best_similarity < float(min_similarity):
        return FragmentRepairResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            repaired_fragment_smiles=repaired_fragment,
            repair_source=repair_source,
            repair_similarity=best_similarity,
            reason="repair_similarity_below_threshold",
            candidate_count=candidate_count,
        )

    return FragmentRepairResult(
        parent_smiles=normalized_parent,
        raw_fragment_smiles=normalized_fragment,
        attempted=True,
        success=True,
        repaired_fragment_smiles=repaired_fragment,
        repair_source=repair_source,
        repair_similarity=best_similarity,
        reason="repair_success",
        candidate_count=candidate_count,
    )


def _iter_parent_repair_candidates(
    parent_mol: object,
    *,
    max_candidates: int,
) -> Iterable[tuple[str, str]]:
    if Chem is None:
        return ()

    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []

    def add_candidate(candidate_smiles: str | None, repair_source: str) -> None:
        if candidate_smiles is None:
            return
        normalized_candidate = str(candidate_smiles or "").strip()
        if not normalized_candidate:
            return
        if normalized_candidate == parent_canonical:
            return
        if normalized_candidate in seen:
            return
        if not is_connected_fragment(normalized_candidate):
            return
        if not is_parent_substructure(parent_canonical, normalized_candidate):
            return
        seen.add(normalized_candidate)
        candidates.append((normalized_candidate, repair_source))

    parent_canonical = Chem.MolToSmiles(parent_mol, canonical=True)
    ring_info = parent_mol.GetRingInfo()
    for atom_ring in ring_info.AtomRings():
        ring_atom_indices = tuple(sorted(set(atom_ring)))
        add_candidate(_fragment_smiles_from_atoms(parent_mol, ring_atom_indices), "ring")
        add_candidate(
            _fragment_smiles_from_atoms(parent_mol, _expand_one_hop(parent_mol, ring_atom_indices)),
            "ring",
        )
        if len(candidates) >= max_candidates:
            break

    if len(candidates) < max_candidates and BRICS is not None:
        try:
            brics_mol = BRICS.BreakBRICSBonds(Chem.Mol(parent_mol))
            brics_frags = Chem.GetMolFrags(brics_mol, asMols=True, sanitizeFrags=False)
        except Exception:  # pragma: no cover - depends on RDKit internals
            brics_frags = ()
        for frag in brics_frags:
            candidate = _mol_to_smiles(frag)
            if candidate:
                add_candidate(candidate, "brics")
            if len(candidates) >= max_candidates:
                break

    if len(candidates) < max_candidates:
        for repair_source, smarts in _FUNCTIONAL_GROUP_SMARTS:
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                continue
            try:
                matches = parent_mol.GetSubstructMatches(query, useChirality=True, uniquify=True)
            except Exception:  # pragma: no cover - depends on RDKit internals
                matches = ()
            for match in matches:
                expanded = _expand_one_hop(parent_mol, tuple(match))
                add_candidate(_fragment_smiles_from_atoms(parent_mol, expanded), "smarts")
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

    return tuple(candidates[:max_candidates])


def _fragment_smiles_from_atoms(parent_mol: object, atom_indices: tuple[int, ...]) -> str | None:
    if Chem is None:
        return None
    try:
        fragment_smiles = Chem.MolFragmentToSmiles(
            parent_mol,
            atomsToUse=list(atom_indices),
            canonical=True,
        )
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None
    candidate = str(fragment_smiles or "").strip()
    return candidate or None


def _expand_one_hop(parent_mol: object, atom_indices: tuple[int, ...]) -> tuple[int, ...]:
    expanded = set(atom_indices)
    for atom_index in atom_indices:
        atom = parent_mol.GetAtomWithIdx(int(atom_index))
        expanded.update(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
    return tuple(sorted(expanded))


def _mol_to_smiles(mol: object | None) -> str | None:
    if Chem is None or mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None


def _core_mol_from_parsed(parsed_fragment: object) -> object | None:
    if Chem is None:
        return None
    mol = getattr(parsed_fragment, "mol", None)
    if mol is None:
        return None
    if not bool(getattr(parsed_fragment, "contains_dummy_atoms", False)):
        sanitized, _, _, _ = sanitize_molecule(mol, allow_capped_fragments=False)
        return sanitized

    editable = Chem.RWMol(Chem.Mol(mol))
    dummy_indices = sorted(
        (atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    )
    for atom_index in dummy_indices:
        editable.RemoveAtom(atom_index)
    core = editable.GetMol()
    if core.GetNumAtoms() == 0:
        return None
    sanitized, _, _, _ = sanitize_molecule(core, allow_capped_fragments=False)
    return sanitized


def _core_mol_from_smiles(smiles: str) -> object | None:
    parsed = parse_smiles(
        smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not parsed.sanitized or parsed.mol is None:
        return None
    return _core_mol_from_parsed(parsed)


def _tanimoto_similarity(query_core: object, candidate_smiles: str) -> float | None:
    if (
        Chem is None
        or DataStructs is None
        or rdMolDescriptors is None
        or query_core is None
    ):
        return None
    candidate_core = _core_mol_from_smiles(candidate_smiles)
    if candidate_core is None:
        return None
    try:
        query_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(query_core, 2, 1024)
        candidate_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(candidate_core, 2, 1024)
        return float(DataStructs.TanimotoSimilarity(query_fp, candidate_fp))
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None
