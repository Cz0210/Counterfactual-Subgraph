"""Connected-component salvage for disconnected decoded fragments."""

from __future__ import annotations

from dataclasses import dataclass

from src.chem.projection import project_fragment_to_parent_subgraph
from src.chem.smiles_utils import is_rdkit_available, parse_smiles, sanitize_molecule
from src.chem.substructure import is_parent_substructure
from src.chem.types import FragmentComponentSalvageResult

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


@dataclass(frozen=True, slots=True)
class _ComponentCandidate:
    raw_fragment_smiles: str
    atom_count: int
    strict_parent_ok: bool
    projected_fragment_smiles: str | None
    projection_score: float | None
    projection_reason: str | None
    projection_success: bool

    @property
    def accepted_fragment_smiles(self) -> str | None:
        if self.strict_parent_ok:
            return self.raw_fragment_smiles
        if self.projection_success:
            return self.raw_fragment_smiles
        return None

    @property
    def acceptable(self) -> bool:
        return bool(self.accepted_fragment_smiles)


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
    salvage_stage: str | None = None,
    raw_component_count: int = 0,
    core_component_count: int = 0,
) -> FragmentComponentSalvageResult:
    """Extract one usable connected component from a disconnected fragment."""

    normalized_parent = str(parent_smiles or "").strip()
    normalized_fragment = str(raw_fragment or "").strip()
    normalized_stage = str(salvage_stage or "").strip() or None
    if not normalized_fragment:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            raw_component_count=int(raw_component_count),
            core_component_count=int(core_component_count),
            salvage_stage=normalized_stage,
            reason="empty_fragment",
            failure_reason="salvaged_component_other",
            failure_stage="input",
        )
    if normalized_stage not in {"raw", "core"}:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            raw_component_count=int(raw_component_count),
            core_component_count=int(core_component_count),
            salvage_stage=normalized_stage,
            reason="unsupported_salvage_stage",
            failure_reason="core_unusable_not_salvageable",
            failure_stage="input",
        )
    if not is_rdkit_available() or Chem is None:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            raw_component_count=int(raw_component_count),
            core_component_count=int(core_component_count),
            salvage_stage=normalized_stage,
            reason="rdkit_unavailable",
            failure_reason="salvaged_component_other",
            failure_stage="runtime",
        )

    parsed = parse_smiles(
        normalized_fragment,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not parsed.parseable or parsed.mol is None:
        failure_reason = "core_unusable_not_salvageable" if normalized_stage == "core" else "salvaged_component_other"
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=False,
            success=False,
            raw_component_count=int(raw_component_count),
            core_component_count=int(core_component_count),
            salvage_stage=normalized_stage,
            reason="parse_failed",
            failure_reason=failure_reason,
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
            raw_component_count=int(raw_component_count or component_count),
            core_component_count=int(core_component_count or component_count),
            salvage_stage=normalized_stage,
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
            raw_component_count=int(raw_component_count or component_count),
            core_component_count=int(core_component_count or component_count),
            salvage_stage=normalized_stage,
            reason="too_many_components",
            failure_reason="salvaged_component_other",
            failure_stage="component_detection",
        )

    candidates: list[_ComponentCandidate] = []
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
        projection_result = project_fragment_to_parent_subgraph(
            normalized_parent,
            smiles,
            min_score=projection_min_score,
            max_candidates=projection_max_candidates,
            min_atoms=min_atoms,
            max_atom_ratio=projection_max_atom_ratio,
            enable_khop3=projection_enable_khop3,
            mcs_timeout=projection_mcs_timeout,
        )
        candidates.append(
            _ComponentCandidate(
                raw_fragment_smiles=smiles,
                atom_count=atom_count,
                strict_parent_ok=strict_match,
                projected_fragment_smiles=projection_result.projected_fragment_smiles,
                projection_score=projection_result.projection_score,
                projection_reason=projection_result.reason,
                projection_success=bool(projection_result.success),
            )
        )

    candidate_count = len(candidates)
    if not candidates:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            component_count=component_count,
            raw_component_count=int(raw_component_count or component_count),
            core_component_count=int(core_component_count or component_count),
            salvage_stage=normalized_stage,
            reason="no_component_meets_min_atoms",
            failure_reason="no_component_meets_min_atoms",
            failure_stage="component_filter",
        )

    selected = _select_component(candidates, method=method)
    if selected is None:
        largest = max(candidates, key=lambda item: (item.atom_count, item.raw_fragment_smiles))
        failure_reason = (
            "largest_component_rejected"
            if method == "largest"
            else "best_parent_match_rejected"
            if method == "best_parent_match"
            else "salvaged_component_projection_failed"
        )
        if all(not candidate.acceptable for candidate in candidates):
            failure_reason = "salvaged_component_projection_failed"
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            component_count=component_count,
            raw_component_count=int(raw_component_count or component_count),
            core_component_count=int(core_component_count or component_count),
            salvage_stage=normalized_stage,
            candidate_count=candidate_count,
            best_candidate=largest.raw_fragment_smiles,
            reason=failure_reason,
            failure_reason=failure_reason,
            failure_stage="selection",
        )

    accepted_fragment = selected.accepted_fragment_smiles
    if not accepted_fragment:
        return FragmentComponentSalvageResult(
            raw_fragment_smiles=normalized_fragment,
            attempted=True,
            success=False,
            component_count=component_count,
            raw_component_count=int(raw_component_count or component_count),
            core_component_count=int(core_component_count or component_count),
            salvage_stage=normalized_stage,
            candidate_count=candidate_count,
            best_candidate=selected.raw_fragment_smiles,
            reason="salvaged_component_other",
            failure_reason="salvaged_component_other",
            failure_stage="selection",
        )

    return FragmentComponentSalvageResult(
        raw_fragment_smiles=normalized_fragment,
        attempted=True,
        success=True,
        component_count=component_count,
        raw_component_count=int(raw_component_count or component_count),
        core_component_count=int(core_component_count or component_count),
        salvage_stage=normalized_stage,
        candidate_count=candidate_count,
        best_candidate=selected.raw_fragment_smiles,
        salvage_method=str(method),
        salvaged_fragment_smiles=accepted_fragment,
        salvaged_atom_count=selected.atom_count,
        reason="component_salvage_success",
    )


def _select_component(
    candidates: list[_ComponentCandidate],
    *,
    method: str,
) -> _ComponentCandidate | None:
    acceptable = [candidate for candidate in candidates if candidate.acceptable]
    if not acceptable:
        return None
    if method == "largest":
        return max(acceptable, key=lambda item: (item.atom_count, item.raw_fragment_smiles))
    if method == "best_parent_match":
        return max(
            acceptable,
            key=lambda item: (
                int(item.strict_parent_ok),
                float(item.projection_score or 0.0),
                item.atom_count,
                item.raw_fragment_smiles,
            ),
        )

    largest = max(candidates, key=lambda item: (item.atom_count, item.raw_fragment_smiles))
    if largest.acceptable:
        return largest
    return max(
        acceptable,
        key=lambda item: (
            int(item.strict_parent_ok),
            float(item.projection_score or 0.0),
            item.atom_count,
            item.raw_fragment_smiles,
        ),
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
