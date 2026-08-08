"""Deterministic random connected induced-subgraph control generator."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from src.candidates.base_generator import CandidateBatch, CandidateRequest, candidate_row


class RandomConnectedSubgraphGenerator:
    source_name = "random_connected_size_matched"

    @staticmethod
    def _rdkit() -> Any:
        try:
            from rdkit import Chem
        except ImportError as exc:  # pragma: no cover - runtime dependency.
            raise RuntimeError("Random connected candidates require RDKit.") from exc
        return Chem

    @staticmethod
    def _connected_subset(molecule: Any, target_size: int, rng: random.Random) -> tuple[int, ...] | None:
        num_atoms = int(molecule.GetNumAtoms())
        if not 0 < int(target_size) < num_atoms:
            return None
        selected = {rng.randrange(num_atoms)}
        while len(selected) < int(target_size):
            frontier = sorted(
                {
                    int(neighbor.GetIdx())
                    for atom_index in selected
                    for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors()
                    if int(neighbor.GetIdx()) not in selected
                }
            )
            if not frontier:
                return None
            selected.add(frontier[rng.randrange(len(frontier))])
        return tuple(sorted(selected))

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        request.validate()
        Chem = self._rdkit()
        molecule = Chem.MolFromSmiles(request.parent_smiles)
        if molecule is None:
            raise ValueError(f"Invalid random-control parent: {request.parent_smiles!r}")
        Chem.SanitizeMol(molecule)
        rng = random.Random(int(request.seed))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        reasons: Counter[str] = Counter()
        attempts = 0
        while len(rows) < int(request.candidates_per_parent) and attempts < int(request.max_attempts):
            target = int(request.size_targets[len(rows) % len(request.size_targets)])
            subset = self._connected_subset(molecule, target, rng)
            attempts += 1
            if subset is None:
                reasons["target_size_unavailable"] += 1
                continue
            raw = str(
                Chem.MolFragmentToSmiles(
                    molecule,
                    atomsToUse=list(subset),
                    canonical=True,
                    isomericSmiles=True,
                )
            )
            fragment = Chem.MolFromSmiles(raw)
            if fragment is None:
                reasons["fragment_parse_failed"] += 1
                continue
            Chem.SanitizeMol(fragment)
            canonical = str(
                Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
            )
            if canonical in seen:
                reasons["canonical_duplicate"] += 1
                continue
            if not molecule.HasSubstructMatch(fragment):
                reasons["substructure_check_failed"] += 1
                continue
            seen.add(canonical)
            rows.append(
                candidate_row(
                    request=request,
                    source=self.source_name,
                    source_variant="induced_connected_v1",
                    rank=len(rows) + 1,
                    raw_fragment=raw,
                    canonical_fragment=canonical,
                    num_fragment_atoms=int(fragment.GetNumAtoms()),
                    num_parent_atoms=int(molecule.GetNumAtoms()),
                )
            )
        shortfall = int(request.candidates_per_parent) - len(rows)
        if shortfall:
            reasons["max_attempts_exhausted"] += shortfall
        return CandidateBatch(
            rows=tuple(rows),
            requested_count=int(request.candidates_per_parent),
            generated_count=len(rows),
            shortfall_count=shortfall,
            shortfall_reason_counts=dict(sorted(reasons.items())),
        )


__all__ = ["RandomConnectedSubgraphGenerator"]
