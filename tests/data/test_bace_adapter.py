from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.data import bace_adapter


def _write_raw(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles", "label"])
        writer.writeheader()
        for smiles, label in rows:
            writer.writerow({"smiles": smiles, "label": label})


def test_load_bace_records_preserves_graph_and_stable_ids(tmp_path: Path) -> None:
    raw = tmp_path / "bace.csv"
    _write_raw(raw, [("CCO", 0), ("c1ccccc1N", 1), ("not-smiles", 1)])
    records, audit = bace_adapter.load_bace_records(raw)
    assert len(records) == 2
    assert audit["valid_unique_rows"] == 2
    assert len(audit["invalid_smiles"]) == 1
    assert all(record.molecule_id.startswith("BACE_") for record in records)
    assert all(record.smiles and record.graph.num_nodes > 0 for record in records)
    assert all(record.graph.num_edges >= 0 for record in records)
    assert len({record.source_graph_hash for record in records}) == 2


def test_prepare_bace_dataset_writes_required_summary_and_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "bace.csv"
    rows = [
        ("C", 0),
        ("CC", 0),
        ("CCC", 0),
        ("CCCC", 0),
        ("N", 1),
        ("NN", 1),
        ("NNN", 1),
        ("NNNN", 1),
    ]
    _write_raw(raw, rows)
    assignments: dict[str, str] = {}
    sequence = iter(["train", "val", "calibration", "test"] * 2)

    def assign(scaffold: str, **_kwargs: object) -> str:
        if scaffold not in assignments:
            assignments[scaffold] = next(sequence)
        return assignments[scaffold]

    monkeypatch.setattr(bace_adapter, "_split_for_scaffold", assign)
    output = tmp_path / "processed"
    summary = bace_adapter.prepare_bace_dataset(raw_csv=raw, output_dir=output)
    assert summary["num_graphs"] == 8
    assert summary["label_distribution"] == {"0": 4, "1": 4}
    assert summary["invalid_smiles_count"] == 0
    assert summary["avg_atoms"] > 0
    assert summary["avg_bonds"] >= 0
    assert (output / "graphs.jsonl").is_file()
    for split in bace_adapter.SPLIT_NAMES:
        with (output / f"{split}.csv").open(encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        assert {int(row["label"]) for row in parsed} == {0, 1}
        assert all(row["molecule_id"] == row["parent_id"] for row in parsed)
    persisted = json.loads((output / "bace_dataset_summary.json").read_text())
    assert persisted["smiles_col"] == "smiles"
    assert persisted["label_col"] == "label"


def test_duplicate_canonical_smiles_with_conflicting_labels_is_rejected(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "bace.csv"
    _write_raw(raw, [("CCO", 0), ("OCC", 1)])
    with pytest.raises(ValueError, match="conflicting labels"):
        bace_adapter.load_bace_records(raw)
