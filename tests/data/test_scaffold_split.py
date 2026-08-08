from __future__ import annotations

from collections import Counter

import pytest

from src.data.molecular_split import SPLIT_NAMES
from src.data.scaffold_split import assign_scaffold_splits, bemis_murcko_scaffold


SCAFFOLD_MOLECULES = (
    "c1ccccc1O",
    "c1ccncc1N",
    "Cn1ccnc1",
    "C1CCCCC1O",
    "C1CCCC1N",
    "C1CCC1F",
    "c1ccc2ccccc2c1O",
    "c1ccc2[nH]ccc2c1N",
    "c1nnc2n1CCC2",
    "O=C1NCCCN1",
    "O=C1NC=CC=C1",
    "c1ccc2ncccc2c1F",
    "C1COCCN1",
    "C1CNCCN1",
    "C1CC2CCC1C2",
    "c1ccc(-c2ccccc2)cc1O",
    "c1ccc2occc2c1N",
    "c1ccc2sccc2c1F",
    "C1CCC2(CC1)CCCC2",
    "C1=CC2=CC=CC=C2C=C1N",
)


def _rows(seed: int) -> list[dict[str, object]]:
    rows = [
        {"molecule_id": f"m{index}", "canonical_smiles": smiles, "label": index % 2}
        for index, smiles in enumerate(SCAFFOLD_MOLECULES)
    ]
    assigned, _audit = assign_scaffold_splits(rows, seed=seed)
    counts = Counter(row["split"] for row in assigned)
    if set(counts) != set(SPLIT_NAMES):
        pytest.skip(f"fixture scaffold hashes do not cover all splits for seed={seed}: {counts}")
    return rows


def test_scaffold_groups_are_disjoint_and_deterministic() -> None:
    rows = _rows(13)
    first, audit = assign_scaffold_splits(rows, seed=13)
    second, _ = assign_scaffold_splits(rows, seed=13)
    assert first == second
    assert audit["passed"] is True
    assert audit["scaffold_overlap_count"] == 0
    split_by_scaffold: dict[str, set[str]] = {}
    for row in first:
        split_by_scaffold.setdefault(str(row["scaffold_smiles"]), set()).add(str(row["split"]))
    assert all(len(values) == 1 for values in split_by_scaffold.values())


def test_acyclic_policy_is_explicit() -> None:
    grouped, canonical = bemis_murcko_scaffold("CCO", acyclic_policy="group")
    isolated, canonical_again = bemis_murcko_scaffold(
        "CCO", acyclic_policy="canonical-smiles"
    )
    assert grouped == "__ACYCLIC__"
    assert isolated == f"__ACYCLIC__:{canonical}"
    assert canonical_again == canonical
    with pytest.raises(ValueError, match="Unsupported acyclic"):
        bemis_murcko_scaffold("CCO", acyclic_policy="guess")
