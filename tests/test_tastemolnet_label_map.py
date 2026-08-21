import pytest

from scripts.prepare_tastemolnet import (
    TasteLabelError,
    detect_schema,
    normalize_taste_label,
)
from src.data.dataset_registry import get_dataset_spec


def test_registry_freezes_three_class_label_map_and_source_class() -> None:
    spec = get_dataset_spec("taste", allow_historical=False)
    assert spec.num_classes == 3
    assert spec.label_map == {0: "Bitter", 1: "Sweet", 2: "Tasteless"}
    assert spec.source_label == 1
    assert spec.source_label_name == "Sweet"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bitter", (0, "Bitter")),
        (" sweet ", (1, "Sweet")),
        ("TASTELESS", (2, "Tasteless")),
        ("0", (0, "Bitter")),
        (1.0, (1, "Sweet")),
    ],
)
def test_label_normalization(raw, expected) -> None:
    assert normalize_taste_label(raw) == expected


@pytest.mark.parametrize("raw", ["unknown", "ambiguous", "Sweet/Bitter", "Sweet and Tasteless", None])
def test_ambiguous_or_multitaste_labels_are_not_collapsed(raw) -> None:
    with pytest.raises(TasteLabelError) as error:
        normalize_taste_label(raw)
    assert error.value.reason == "AMBIGUOUS_LABEL"


def test_schema_detection_supports_upstream_names_without_hard_coding_one_layout() -> None:
    schema = detect_schema(["COMPOUND_ID", "PROCESSED_SMILES", "TARGET", "group"])
    assert schema.id_column == "COMPOUND_ID"
    assert schema.smiles_column == "PROCESSED_SMILES"
    assert schema.label_column == "TARGET"


def test_explicit_schema_override_is_case_insensitive() -> None:
    schema = detect_schema(
        ["mol_text", "Taste_Class", "row_key"],
        smiles_column="MOL_TEXT",
        label_column="taste-class",
        id_column="ROW KEY",
    )
    assert schema.smiles_column == "mol_text"
    assert schema.label_column == "Taste_Class"
    assert schema.id_column == "row_key"
