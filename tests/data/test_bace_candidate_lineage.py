from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.data.bace_candidate_lineage import attach_bace_candidate_lineage


def _write_parents(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["molecule_id", "smiles", "label", "source_graph_hash"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "molecule_id": "BACE_a",
                    "smiles": "CCO",
                    "label": 1,
                    "source_graph_hash": "a" * 64,
                },
                {
                    "molecule_id": "BACE_b",
                    "smiles": "CCN",
                    "label": 1,
                    "source_graph_hash": "b" * 64,
                },
            ]
        )


def _raw_rows() -> list[dict[str, object]]:
    return [
        {
            "parent_index": parent_index,
            "candidate_index": candidate_index,
            "parent_smiles": smiles,
            "label": 1,
            "final_fragment": "C",
            "cf_flip": True,
        }
        for parent_index, smiles in enumerate(("CCO", "CCN"))
        for candidate_index in range(2)
    ]


def test_bace_candidate_lineage_adds_ids_without_reordering(tmp_path: Path) -> None:
    parents = tmp_path / "parents.csv"
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / "candidate_pool.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_parents(parents)
    source_rows = _raw_rows()
    raw.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )
    result = attach_bace_candidate_lineage(
        raw_pool_jsonl=raw,
        parent_csv=parents,
        output_jsonl=output,
        manifest_path=manifest,
        expected_candidates_per_parent=2,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["candidate_order_unchanged"] is True
    assert [row["parent_index"] for row in rows] == [0, 0, 1, 1]
    assert [row["candidate_index"] for row in rows] == [0, 1, 0, 1]
    assert [row["parent_id"] for row in rows] == [
        "BACE_a",
        "BACE_a",
        "BACE_b",
        "BACE_b",
    ]
    for source, enriched in zip(source_rows, rows, strict=True):
        assert all(enriched[key] == value for key, value in source.items())


def test_bace_candidate_lineage_rejects_parent_smiles_drift(tmp_path: Path) -> None:
    parents = tmp_path / "parents.csv"
    raw = tmp_path / "raw.jsonl"
    _write_parents(parents)
    rows = _raw_rows()
    rows[0]["parent_smiles"] = "CCC"
    raw.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SMILES lineage mismatch"):
        attach_bace_candidate_lineage(
            raw_pool_jsonl=raw,
            parent_csv=parents,
            output_jsonl=tmp_path / "out.jsonl",
            manifest_path=tmp_path / "manifest.json",
            expected_candidates_per_parent=2,
        )
