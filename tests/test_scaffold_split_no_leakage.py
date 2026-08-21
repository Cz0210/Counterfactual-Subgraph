from collections import defaultdict

from scripts.prepare_tastemolnet import scaffold_disjoint_split


def test_scaffold_group_never_crosses_split_and_result_is_deterministic() -> None:
    records = []
    for scaffold_index in range(24):
        scaffold = f"scaffold-{scaffold_index}"
        for member_index in range(2 if scaffold_index % 5 == 0 else 1):
            records.append(
                {
                    "molecule_id": f"mol-{scaffold_index}-{member_index}",
                    "model_smiles": f"C{'C' * scaffold_index}N{member_index}",
                    "scaffold": scaffold,
                    "label": scaffold_index % 3,
                }
            )

    first, first_summary = scaffold_disjoint_split(records, seed=7)
    second, second_summary = scaffold_disjoint_split(records, seed=7)
    assert first == second
    assert first_summary == second_summary
    assert first_summary["scaffold_overlap_audit"]["passed"] is True
    assert all(
        first_summary["split_statistics"][split]["rows"] > 0
        for split in ("train", "validation", "calibration", "test")
    )

    scaffold_splits = defaultdict(set)
    for row in first:
        scaffold_splits[row["scaffold"]].add(row["split"])
    assert all(len(splits) == 1 for splits in scaffold_splits.values())


def test_empty_scaffolds_are_grouped_by_canonical_molecule_not_globally() -> None:
    rows, audit = scaffold_disjoint_split(
        [
            {"molecule_id": f"m-{index}", "model_smiles": f"C{'C' * index}O", "scaffold": "", "label": index % 3}
            for index in range(12)
        ],
        seed=11,
    )
    assert len(rows) == 12
    assert audit["empty_scaffold_policy"] == "canonical_smiles_specific_acyclic_group"
    assert audit["scaffold_overlap_audit"]["passed"] is True
