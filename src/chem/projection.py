"""Parent-constrained candidate retrieval projection for decoded fragments."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from src.chem.deletion import delete_fragment_from_parent
from src.chem.smiles_utils import is_rdkit_available, parse_smiles, sanitize_molecule
from src.chem.substructure import is_connected_fragment, is_parent_substructure
from src.chem.types import FragmentProjectionResult

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import BRICS, rdFMCS, rdMolDescriptors
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None
    DataStructs = None
    BRICS = None
    rdFMCS = None
    rdMolDescriptors = None


@dataclass(frozen=True, slots=True)
class ParentProjectionCandidate:
    """One strict connected parent-derived projection candidate."""

    smiles: str
    source: str
    atom_indices: tuple[int, ...]
    atom_count: int
    atom_ratio: float
    mol: object


_FUNCTIONAL_GROUP_SMARTS: tuple[tuple[str, str, int], ...] = (
    ("fg_carboxyl", "C(=O)O", 1),
    ("fg_amide", "C(=O)N", 1),
    ("fg_sulfonic_acid", "S(=O)(=O)O", 1),
    ("fg_sulfonamide", "S(=O)(=O)N", 1),
    ("fg_azo", "N=N", 1),
    ("fg_nitro", "[N+](=O)[O-]", 1),
    ("fg_disulfide", "SS", 1),
    ("fg_halogen", "[F,Cl,Br,I]", 2),
    ("fg_aromatic_oh", "[c][OX2H]", 1),
)

_SOURCE_PRIORITY = {
    "ring_system_r0": 90,
    "ring_system_r1": 85,
    "fg_carboxyl": 80,
    "fg_amide": 80,
    "fg_sulfonic_acid": 80,
    "fg_sulfonamide": 80,
    "fg_azo": 80,
    "fg_nitro": 80,
    "fg_disulfide": 80,
    "fg_halogen": 75,
    "fg_aromatic_oh": 75,
    "atom_k1": 60,
    "atom_k2": 55,
    "atom_k3": 50,
    "bond_k1": 45,
    "bond_k2": 40,
    "brics_component": 35,
}


def project_fragment_to_parent_subgraph(
    parent_smiles: str,
    raw_fragment: str,
    *,
    min_score: float = 0.35,
    max_candidates: int = 128,
    min_atoms: int = 3,
    max_atom_ratio: float = 0.70,
    enable_khop3: bool = False,
    mcs_timeout: int = 1,
) -> FragmentProjectionResult:
    """Retrieve the closest strict connected parent-derived candidate.

    This function intentionally refuses parse-failed raw fragments. It is meant
    for the decoded PPO failure bucket where the model output is parseable but
    does not match the parent as a strict usable subgraph.
    """

    normalized_parent = str(parent_smiles or "").strip()
    normalized_fragment = str(raw_fragment or "").strip()
    if not normalized_parent or not normalized_fragment:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="missing_parent_or_fragment",
        )
    if not is_rdkit_available() or Chem is None:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="rdkit_unavailable",
        )
    if max_candidates <= 0:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="invalid_max_candidates",
        )
    if min_atoms <= 0:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="invalid_min_atoms",
        )
    if max_atom_ratio <= 0.0:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="invalid_max_atom_ratio",
        )

    parent = parse_smiles(
        normalized_parent,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=False,
    )
    if not parent.sanitized or parent.mol is None:
        return FragmentProjectionResult(
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
    if not fragment.parseable or fragment.mol is None:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="parse_failed",
        )
    if not is_connected_fragment(normalized_fragment):
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            reason="fragment_not_connected",
        )

    query_core = _core_mol_from_parsed(fragment)
    query_smiles = _mol_to_smiles(query_core)
    if query_core is None or not query_smiles:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            reason="query_core_unavailable",
        )

    identity = _identity_projection_if_strict(
        normalized_parent=normalized_parent,
        raw_fragment_smiles=normalized_fragment,
        parent_mol=parent.mol,
        query_smiles=query_smiles,
        query_core=query_core,
        min_atoms=int(min_atoms),
        max_atom_ratio=float(max_atom_ratio),
    )
    if identity is not None:
        return identity

    candidates = build_parent_projection_candidates(
        parent.mol,
        parent_smiles=normalized_parent,
        max_candidates=int(max_candidates),
        min_atoms=int(min_atoms),
        max_atom_ratio=float(max_atom_ratio),
        enable_khop3=bool(enable_khop3),
    )
    if not candidates:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            reason="no_projection_candidates",
            candidate_count=0,
        )

    query_fg = _functional_group_names(query_core)
    query_atom_count = max(1, _non_dummy_atom_count(query_core))
    best: tuple[float, int, float, int, ParentProjectionCandidate] | None = None
    for candidate in candidates:
        score = _projection_score(
            query_core=query_core,
            query_functional_groups=query_fg,
            query_atom_count=query_atom_count,
            candidate=candidate,
            max_atom_ratio=float(max_atom_ratio),
            mcs_timeout=int(mcs_timeout),
        )
        ranking = (
            float(score),
            int(_SOURCE_PRIORITY.get(candidate.source, 0)),
            -abs(candidate.atom_count - query_atom_count),
            -candidate.atom_count,
            candidate,
        )
        if best is None or ranking[:4] > best[:4]:
            best = ranking

    if best is None:
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            reason="no_scored_projection_candidates",
            candidate_count=len(candidates),
        )

    best_score, _, _, _, best_candidate = best
    if best_score < float(min_score):
        return FragmentProjectionResult(
            parent_smiles=normalized_parent,
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            projection_method="retrieval",
            projected_fragment_smiles=best_candidate.smiles,
            projection_source=best_candidate.source,
            projection_score=float(best_score),
            reason="projection_failed_low_score",
            candidate_count=len(candidates),
            projected_atom_count=best_candidate.atom_count,
            projected_atom_ratio=best_candidate.atom_ratio,
        )

    return FragmentProjectionResult(
        parent_smiles=normalized_parent,
        raw_fragment_smiles=normalized_fragment,
        attempted=True,
        success=True,
        projection_method="retrieval",
        projected_fragment_smiles=best_candidate.smiles,
        projection_source=best_candidate.source,
        projection_score=float(best_score),
        reason="projection_success",
        candidate_count=len(candidates),
        projected_atom_count=best_candidate.atom_count,
        projected_atom_ratio=best_candidate.atom_ratio,
    )


def build_parent_projection_candidates(
    parent_mol: object,
    *,
    parent_smiles: str | None = None,
    max_candidates: int = 128,
    min_atoms: int = 3,
    max_atom_ratio: float = 0.70,
    enable_khop3: bool = False,
) -> tuple[ParentProjectionCandidate, ...]:
    """Construct filtered strict connected candidates from parent atom indices."""

    if Chem is None or parent_mol is None or max_candidates <= 0:
        return ()

    parent_atom_count = int(parent_mol.GetNumAtoms())
    if parent_atom_count <= 1:
        return ()
    parent_canonical = (
        str(parent_smiles or "").strip()
        or Chem.MolToSmiles(parent_mol, canonical=True, isomericSmiles=True)
    )

    seen_smiles: set[str] = set()
    candidates: list[ParentProjectionCandidate] = []

    def add_atom_set(atom_indices: Iterable[int], source: str) -> None:
        if len(candidates) >= int(max_candidates):
            return
        normalized_indices = tuple(sorted({int(index) for index in atom_indices}))
        candidate = _candidate_from_atom_indices(
            parent_mol=parent_mol,
            parent_smiles=parent_canonical,
            atom_indices=normalized_indices,
            source=source,
            parent_atom_count=parent_atom_count,
            min_atoms=int(min_atoms),
            max_atom_ratio=float(max_atom_ratio),
            seen_smiles=seen_smiles,
        )
        if candidate is None:
            return
        seen_smiles.add(candidate.smiles)
        candidates.append(candidate)

    for ring_system in _ring_system_atom_sets(parent_mol):
        add_atom_set(ring_system, "ring_system_r0")
        add_atom_set(_expand_atoms(parent_mol, ring_system, radius=1), "ring_system_r1")
        if len(candidates) >= int(max_candidates):
            return tuple(candidates[:max_candidates])

    for source, smarts, radius in _FUNCTIONAL_GROUP_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            continue
        try:
            matches = parent_mol.GetSubstructMatches(
                query,
                useChirality=True,
                uniquify=True,
            )
        except Exception:  # pragma: no cover - defensive around RDKit internals
            matches = ()
        for match in matches:
            add_atom_set(_expand_atoms(parent_mol, match, radius=radius), source)
            if len(candidates) >= int(max_candidates):
                return tuple(candidates[:max_candidates])

    for radius in (1, 2):
        for atom in parent_mol.GetAtoms():
            add_atom_set(
                _expand_atoms(parent_mol, (atom.GetIdx(),), radius=radius),
                f"atom_k{radius}",
            )
            if len(candidates) >= int(max_candidates):
                return tuple(candidates[:max_candidates])

    if enable_khop3 and len(candidates) < int(max_candidates):
        for atom in parent_mol.GetAtoms():
            add_atom_set(
                _expand_atoms(parent_mol, (atom.GetIdx(),), radius=3),
                "atom_k3",
            )
            if len(candidates) >= int(max_candidates):
                return tuple(candidates[:max_candidates])

    for radius in (1, 2):
        for bond in parent_mol.GetBonds():
            add_atom_set(
                _expand_atoms(
                    parent_mol,
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    radius=radius,
                ),
                f"bond_k{radius}",
            )
            if len(candidates) >= int(max_candidates):
                return tuple(candidates[:max_candidates])

    for component in _brics_parent_components(parent_mol):
        add_atom_set(component, "brics_component")
        if len(candidates) >= int(max_candidates):
            return tuple(candidates[:max_candidates])

    return tuple(candidates[:max_candidates])


def _identity_projection_if_strict(
    *,
    normalized_parent: str,
    raw_fragment_smiles: str,
    parent_mol: object,
    query_smiles: str,
    query_core: object,
    min_atoms: int,
    max_atom_ratio: float,
) -> FragmentProjectionResult | None:
    query_atom_count = _non_dummy_atom_count(query_core)
    parent_atom_count = int(parent_mol.GetNumAtoms())
    query_atom_ratio = query_atom_count / max(1, parent_atom_count)
    if query_atom_count < int(min_atoms):
        return None
    if query_atom_count >= parent_atom_count:
        return None
    if query_atom_ratio > float(max_atom_ratio):
        return None
    if not is_parent_substructure(normalized_parent, query_smiles):
        return None
    deletion = delete_fragment_from_parent(normalized_parent, query_smiles, max_matches=1)
    if not deletion.success or not deletion.residual_atom_count:
        return None
    return FragmentProjectionResult(
        parent_smiles=normalized_parent,
        raw_fragment_smiles=raw_fragment_smiles,
        attempted=False,
        success=True,
        projection_method="identity",
        projected_fragment_smiles=query_smiles,
        projection_source="identity",
        projection_score=1.0,
        reason="already_strict_parent_substructure",
        candidate_count=0,
        projected_atom_count=query_atom_count,
        projected_atom_ratio=query_atom_ratio,
    )


def _candidate_from_atom_indices(
    *,
    parent_mol: object,
    parent_smiles: str,
    atom_indices: tuple[int, ...],
    source: str,
    parent_atom_count: int,
    min_atoms: int,
    max_atom_ratio: float,
    seen_smiles: set[str],
) -> ParentProjectionCandidate | None:
    atom_count = len(atom_indices)
    if atom_count < int(min_atoms):
        return None
    if atom_count >= int(parent_atom_count):
        return None
    atom_ratio = atom_count / max(1, int(parent_atom_count))
    if atom_ratio > float(max_atom_ratio):
        return None
    if not _atom_set_is_connected(parent_mol, atom_indices):
        return None

    smiles = _fragment_smiles_from_atoms(parent_mol, atom_indices)
    if not smiles:
        return None
    parsed = parse_smiles(
        smiles,
        sanitize=True,
        canonicalize=True,
        allow_capped_fragments=False,
    )
    if not parsed.sanitized or parsed.mol is None:
        return None
    canonical_smiles = str(parsed.canonical_smiles or smiles).strip()
    if not canonical_smiles or canonical_smiles in seen_smiles:
        return None
    if canonical_smiles == parent_smiles:
        return None
    if not is_connected_fragment(canonical_smiles):
        return None
    if not is_parent_substructure(parent_smiles, canonical_smiles):
        return None
    deletion = delete_fragment_from_parent(parent_smiles, canonical_smiles, max_matches=1)
    if not deletion.success or not deletion.residual_atom_count:
        return None
    return ParentProjectionCandidate(
        smiles=canonical_smiles,
        source=source,
        atom_indices=atom_indices,
        atom_count=atom_count,
        atom_ratio=atom_ratio,
        mol=parsed.mol,
    )


def _fragment_smiles_from_atoms(
    parent_mol: object,
    atom_indices: tuple[int, ...],
) -> str | None:
    if Chem is None or not atom_indices:
        return None
    try:
        return str(
            Chem.MolFragmentToSmiles(
                parent_mol,
                atomsToUse=list(atom_indices),
                canonical=True,
                isomericSmiles=True,
            )
            or ""
        ).strip()
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None


def _ring_system_atom_sets(parent_mol: object) -> tuple[tuple[int, ...], ...]:
    ring_info = parent_mol.GetRingInfo()
    atom_rings = [set(int(index) for index in ring) for ring in ring_info.AtomRings()]
    systems: list[set[int]] = []
    for ring in atom_rings:
        merged = set(ring)
        changed = True
        while changed:
            changed = False
            remaining: list[set[int]] = []
            for system in systems:
                if merged.intersection(system):
                    merged.update(system)
                    changed = True
                else:
                    remaining.append(system)
            systems = remaining
        systems.append(merged)
    return tuple(tuple(sorted(system)) for system in systems)


def _expand_atoms(
    parent_mol: object,
    atom_indices: Iterable[int],
    *,
    radius: int,
) -> tuple[int, ...]:
    visited = {int(index) for index in atom_indices}
    frontier = set(visited)
    for _ in range(max(0, int(radius))):
        next_frontier: set[int] = set()
        for atom_index in frontier:
            atom = parent_mol.GetAtomWithIdx(int(atom_index))
            for neighbor in atom.GetNeighbors():
                neighbor_index = int(neighbor.GetIdx())
                if neighbor_index not in visited:
                    visited.add(neighbor_index)
                    next_frontier.add(neighbor_index)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(sorted(visited))


def _atom_set_is_connected(parent_mol: object, atom_indices: tuple[int, ...]) -> bool:
    atom_set = set(atom_indices)
    if not atom_set:
        return False
    start = next(iter(atom_set))
    visited = {start}
    queue: deque[int] = deque([start])
    while queue:
        atom_index = queue.popleft()
        atom = parent_mol.GetAtomWithIdx(int(atom_index))
        for neighbor in atom.GetNeighbors():
            neighbor_index = int(neighbor.GetIdx())
            if neighbor_index in atom_set and neighbor_index not in visited:
                visited.add(neighbor_index)
                queue.append(neighbor_index)
    return visited == atom_set


def _brics_parent_components(parent_mol: object) -> tuple[tuple[int, ...], ...]:
    if BRICS is None:
        return ()
    try:
        brics_bond_records = tuple(BRICS.FindBRICSBonds(parent_mol))
    except Exception:  # pragma: no cover - depends on RDKit internals
        return ()
    cut_bonds: set[frozenset[int]] = set()
    for record in brics_bond_records:
        try:
            atom_pair = record[0]
            begin_atom = int(atom_pair[0])
            end_atom = int(atom_pair[1])
        except Exception:
            continue
        cut_bonds.add(frozenset((begin_atom, end_atom)))
    if not cut_bonds:
        return ()

    all_atoms = {int(atom.GetIdx()) for atom in parent_mol.GetAtoms()}
    components: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for start in sorted(all_atoms):
        if start in visited:
            continue
        component: set[int] = set()
        queue: deque[int] = deque([start])
        visited.add(start)
        while queue:
            atom_index = queue.popleft()
            component.add(atom_index)
            atom = parent_mol.GetAtomWithIdx(int(atom_index))
            for neighbor in atom.GetNeighbors():
                neighbor_index = int(neighbor.GetIdx())
                if frozenset((atom_index, neighbor_index)) in cut_bonds:
                    continue
                if neighbor_index not in visited:
                    visited.add(neighbor_index)
                    queue.append(neighbor_index)
        components.append(tuple(sorted(component)))
    return tuple(components)


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


def _projection_score(
    *,
    query_core: object,
    query_functional_groups: frozenset[str],
    query_atom_count: int,
    candidate: ParentProjectionCandidate,
    max_atom_ratio: float,
    mcs_timeout: int,
) -> float:
    tanimoto = _morgan_tanimoto(query_core, candidate.mol)
    mcs_coverage = _mcs_atom_coverage(
        query_core,
        candidate.mol,
        query_atom_count=query_atom_count,
        timeout=max(1, int(mcs_timeout)),
    )
    functional_overlap = _functional_group_overlap(
        query_functional_groups,
        _functional_group_names(candidate.mol),
    )
    atom_count_diff = abs(candidate.atom_count - query_atom_count) / max(
        1,
        query_atom_count,
    )
    too_large_penalty = _too_large_penalty(candidate.atom_ratio, max_atom_ratio)
    return float(
        1.0 * tanimoto
        + 1.0 * mcs_coverage
        + 0.5 * functional_overlap
        - 0.1 * atom_count_diff
        - 0.5 * too_large_penalty
    )


def _morgan_tanimoto(query_mol: object, candidate_mol: object) -> float:
    if DataStructs is None or rdMolDescriptors is None:
        return 0.0
    try:
        query_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(query_mol, 2, 2048)
        candidate_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            candidate_mol,
            2,
            2048,
        )
        return float(DataStructs.TanimotoSimilarity(query_fp, candidate_fp))
    except Exception:  # pragma: no cover - depends on RDKit internals
        return 0.0


def _mcs_atom_coverage(
    query_mol: object,
    candidate_mol: object,
    *,
    query_atom_count: int,
    timeout: int,
) -> float:
    if rdFMCS is None:
        return 0.0
    try:
        result = rdFMCS.FindMCS(
            [query_mol, candidate_mol],
            timeout=max(1, int(timeout)),
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            matchValences=False,
        )
    except Exception:  # pragma: no cover - depends on RDKit internals
        return 0.0
    if getattr(result, "canceled", False):
        return 0.0
    return float(max(0, int(getattr(result, "numAtoms", 0))) / max(1, query_atom_count))


def _functional_group_names(mol: object | None) -> frozenset[str]:
    if Chem is None or mol is None:
        return frozenset()
    names: set[str] = set()
    for name, smarts, _radius in _FUNCTIONAL_GROUP_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            continue
        try:
            if mol.HasSubstructMatch(query, useChirality=True):
                names.add(name)
        except Exception:  # pragma: no cover - depends on RDKit internals
            continue
    return frozenset(names)


def _functional_group_overlap(
    query_groups: frozenset[str],
    candidate_groups: frozenset[str],
) -> float:
    union = query_groups | candidate_groups
    if not union:
        return 0.0
    return float(len(query_groups & candidate_groups) / len(union))


def _too_large_penalty(atom_ratio: float, max_atom_ratio: float) -> float:
    soft_limit = 0.5 * float(max_atom_ratio)
    if atom_ratio <= soft_limit:
        return 0.0
    span = max(1e-6, float(max_atom_ratio) - soft_limit)
    return float(min(1.0, max(0.0, (atom_ratio - soft_limit) / span)))


__all__ = [
    "ParentProjectionCandidate",
    "build_parent_projection_candidates",
    "project_fragment_to_parent_subgraph",
]
