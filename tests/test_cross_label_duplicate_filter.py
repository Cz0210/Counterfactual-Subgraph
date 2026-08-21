from scripts.prepare_tastemolnet import filter_cross_label_conflicts


def _row(source_row_id: str, smiles: str, label: int) -> dict:
    return {
        "source_row_id": source_row_id,
        "raw_smiles": smiles,
        "canonical_smiles": smiles,
        "model_smiles": smiles,
        "label": label,
        "label_name": {0: "Bitter", 1: "Sweet", 2: "Tasteless"}[label],
    }


def test_cross_label_duplicates_are_all_excluded_without_majority_vote() -> None:
    kept, excluded, conflicts = filter_cross_label_conflicts(
        [
            _row("a", "CCO", 0),
            _row("b", "CCO", 1),
            _row("c", "CCO", 1),
            _row("d", "CCN", 2),
        ]
    )
    assert [row["source_row_id"] for row in kept] == ["d"]
    assert {row["source_row_id"] for row in excluded} == {"a", "b", "c"}
    assert {row["exclusion_reason"] for row in excluded} == {"CROSS_LABEL_DUPLICATE"}
    assert len(conflicts) == 1
    assert conflicts[0]["labels"] == "[0,1]"


def test_same_label_duplicates_keep_only_first_deterministic_row() -> None:
    kept, excluded, conflicts = filter_cross_label_conflicts(
        [_row("first", "CCO", 1), _row("later", "CCO", 1)]
    )
    assert [row["source_row_id"] for row in kept] == ["first"]
    assert conflicts == []
    assert excluded[0]["exclusion_reason"] == "DUPLICATE_SAME_LABEL"
