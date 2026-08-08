from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.bbbp_adapter import load_bbbp_records, validate_bbbp_source


def _write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_bbbp_aliases_invalid_and_duplicate_are_audited(tmp_path: Path) -> None:
    source = tmp_path / "bbp.csv"
    _write(
        source,
        ["mol", "p_np"],
        [
            {"mol": "c1ccccc1O", "p_np": 1},
            {"mol": "Oc1ccccc1", "p_np": 1},
            {"mol": "not-a-smiles", "p_np": 0},
            {"mol": "CCN", "p_np": 0},
        ],
    )
    records, audit = load_bbbp_records(source)
    assert [row.canonical_smiles for row in records] == ["Oc1ccccc1", "CCN"]
    assert audit["raw_smiles_col"] == "mol"
    assert audit["raw_label_col"] == "p_np"
    assert audit["duplicate_canonical_smiles_count"] == 1
    assert audit["invalid_smiles"][0]["reason_code"]
    assert len({row.molecule_id for row in records}) == 2
    assert all(row.source_graph_hash for row in records)


def test_bbbp_alias_ambiguity_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.csv"
    _write(
        source,
        ["smiles", "mol", "label"],
        [{"smiles": "CC", "mol": "CCC", "label": 1}],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        load_bbbp_records(source)
    records, _audit = load_bbbp_records(
        source, raw_smiles_col="smiles", raw_label_col="label"
    )
    assert records[0].smiles == "CC"


def test_bbbp_conflicting_duplicate_labels_fail(tmp_path: Path) -> None:
    source = tmp_path / "conflict.csv"
    _write(
        source,
        ["smiles", "label"],
        [{"smiles": "CCO", "label": 0}, {"smiles": "OCC", "label": 1}],
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        load_bbbp_records(source)


def test_validate_missing_bbbp_is_input_required_without_output(tmp_path: Path) -> None:
    result = validate_bbbp_source(tmp_path / "missing.csv")
    assert result["status"] == "INPUT_REQUIRED"
    assert result["formal_output_written"] is False
