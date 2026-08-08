"""Dataset-agnostic Bemis-Murcko scaffold split construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.molecular_split import (
    DEFAULT_SPLIT_RATIOS,
    SPLIT_NAMES,
    audit_split_overlap,
    hashed_group_split,
    stable_json_sha256,
)


ACYCLIC_POLICIES = ("canonical-smiles", "group")
DEFAULT_ACYCLIC_POLICY = "canonical-smiles"


def _rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - runtime dependency.
        raise RuntimeError("Scaffold splitting requires RDKit.") from exc
    return Chem, MurckoScaffold


def bemis_murcko_scaffold(
    smiles: str,
    *,
    acyclic_policy: str = DEFAULT_ACYCLIC_POLICY,
) -> tuple[str, str]:
    """Return a stable grouping key and canonical molecular SMILES."""

    if acyclic_policy not in ACYCLIC_POLICIES:
        raise ValueError(
            f"Unsupported acyclic policy {acyclic_policy!r}; "
            f"expected one of {ACYCLIC_POLICIES}."
        )
    Chem, MurckoScaffold = _rdkit()
    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        raise ValueError(f"Cannot compute scaffold for invalid SMILES: {smiles!r}")
    Chem.SanitizeMol(molecule)
    canonical = str(
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    )
    scaffold = str(
        MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule,
            includeChirality=True,
        )
        or ""
    )
    if scaffold:
        return scaffold, canonical
    if acyclic_policy == "group":
        return "__ACYCLIC__", canonical
    return f"__ACYCLIC__:{canonical}", canonical


def assign_scaffold_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    smiles_field: str = "canonical_smiles",
    seed: int = 13,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    acyclic_policy: str = DEFAULT_ACYCLIC_POLICY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign complete scaffold groups to deterministic four-way splits."""

    assigned: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        smiles = str(row.get(smiles_field) or "").strip()
        if not smiles:
            raise ValueError(
                f"Scaffold split row {index} lacks {smiles_field!r}."
            )
        scaffold, canonical = bemis_murcko_scaffold(
            smiles,
            acyclic_policy=acyclic_policy,
        )
        split = hashed_group_split(scaffold, seed=seed, ratios=ratios)
        row["canonical_smiles"] = canonical
        row["scaffold_smiles"] = scaffold
        row["split"] = split
        assigned.append(row)
    rows_by_split = {
        split: [row for row in assigned if row["split"] == split]
        for split in SPLIT_NAMES
    }
    leakage = audit_split_overlap(
        rows_by_split,
        require_scaffold_disjoint=True,
    )
    scaffold_counts = Counter(row["scaffold_smiles"] for row in assigned)
    audit = {
        "schema_version": "bemis_murcko_scaffold_split_v1",
        "passed": True,
        "split_seed": int(seed),
        "split_strategy": "sha256_scaffold_group_v1",
        "acyclic_policy": acyclic_policy,
        "num_rows": len(assigned),
        "num_scaffolds": len(scaffold_counts),
        "split_sizes": {
            split: len(rows_by_split[split]) for split in SPLIT_NAMES
        },
        "split_scaffold_counts": {
            split: len({row["scaffold_smiles"] for row in rows_by_split[split]})
            for split in SPLIT_NAMES
        },
        "scaffold_overlap_count": 0,
        "unseen_test_scaffold_rate": 1.0,
        "scaffold_hash": stable_json_sha256(
            sorted((row["molecule_id"], row["scaffold_smiles"]) for row in assigned)
        ),
        "leakage_audit": leakage,
    }
    return assigned, audit


__all__ = [
    "ACYCLIC_POLICIES",
    "DEFAULT_ACYCLIC_POLICY",
    "assign_scaffold_splits",
    "bemis_murcko_scaffold",
]
