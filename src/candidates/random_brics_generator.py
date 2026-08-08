"""Deterministic size-matched random BRICS component generator."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from src.candidates.base_generator import CandidateBatch, CandidateRequest, candidate_row


class RandomBRICSGenerator:
    source_name = "random_brics_size_matched"

    @staticmethod
    def _rdkit() -> tuple[Any, Any]:
        try:
            from rdkit import Chem
            from rdkit.Chem import BRICS
        except ImportError as exc:  # pragma: no cover - runtime dependency.
            raise RuntimeError("Random BRICS candidates require RDKit.") from exc
        return Chem, BRICS

    @staticmethod
    def _components(molecule: Any, broken_edges: set[tuple[int, int]]) -> list[tuple[int, ...]]:
        adjacency: dict[int, set[int]] = {
            index: set() for index in range(int(molecule.GetNumAtoms()))
        }
        for bond in molecule.GetBonds():
            left = int(bond.GetBeginAtomIdx())
            right = int(bond.GetEndAtomIdx())
            if tuple(sorted((left, right))) in broken_edges:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
        remaining = set(adjacency)
        components: list[tuple[int, ...]] = []
        while remaining:
            start = min(remaining)
            stack = [start]
            component: set[int] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(sorted(adjacency[current] - component, reverse=True))
            remaining -= component
            components.append(tuple(sorted(component)))
        return components

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        request.validate()
        Chem, BRICS = self._rdkit()
        molecule = Chem.MolFromSmiles(request.parent_smiles)
        if molecule is None:
            raise ValueError(f"Invalid BRICS-control parent: {request.parent_smiles!r}")
        Chem.SanitizeMol(molecule)
        brics_bonds = list(BRICS.FindBRICSBonds(molecule))
        reasons: Counter[str] = Counter()
        if not brics_bonds:
            return CandidateBatch(
                rows=(),
                requested_count=int(request.candidates_per_parent),
                generated_count=0,
                shortfall_count=int(request.candidates_per_parent),
                shortfall_reason_counts={"no_brics_bonds": int(request.candidates_per_parent)},
            )
        broken = {
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair, _labels in brics_bonds
        }
        components = [
            component
            for component in self._components(molecule, broken)
            if 0 < len(component) < int(molecule.GetNumAtoms())
        ]
        rng = random.Random(int(request.seed))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        available = list(components)
        while len(rows) < int(request.candidates_per_parent) and available:
            target = int(request.size_targets[len(rows) % len(request.size_targets)])
            best_delta = min(abs(len(component) - target) for component in available)
            matched = [
                component
                for component in available
                if abs(len(component) - target) == best_delta
            ]
            component = matched[rng.randrange(len(matched))]
            available.remove(component)
            raw = str(
                Chem.MolFragmentToSmiles(
                    molecule,
                    atomsToUse=list(component),
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
                    source_variant="brics_components_v1",
                    rank=len(rows) + 1,
                    raw_fragment=raw,
                    canonical_fragment=canonical,
                    num_fragment_atoms=int(fragment.GetNumAtoms()),
                    num_parent_atoms=int(molecule.GetNumAtoms()),
                )
            )
        shortfall = int(request.candidates_per_parent) - len(rows)
        if shortfall:
            reasons["insufficient_unique_brics_components"] += shortfall
        return CandidateBatch(
            rows=tuple(rows),
            requested_count=int(request.candidates_per_parent),
            generated_count=len(rows),
            shortfall_count=shortfall,
            shortfall_reason_counts=dict(sorted(reasons.items())),
        )


__all__ = ["RandomBRICSGenerator"]
