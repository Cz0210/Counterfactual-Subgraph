from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.mutagenicity import (
    DEFAULT_SPLIT_RATIOS,
    build_scaffold_splits,
    evaluate_tu_label_mappings,
    load_csv_source,
    parse_sha256_manifest,
    preprocess_curated_source,
    stable_molecule_id,
)


def _write_source(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("smiles", "mutagenicity"))
        writer.writerows(rows)


@pytest.fixture()
def processed_fixture(tmp_path: Path) -> dict[str, object]:
    source_path = tmp_path / "curated.csv"
    _write_source(
        source_path,
        [
            ("CCO", 1),
            ("OCC", 1),
            ("CCN", 0),
            ("NCC", 1),
            ("CC.O", 0),
            ("this-is-not-smiles", 0),
            ("c1ccccc1", 0),
        ],
    )
    source = load_csv_source(source_path)
    return preprocess_curated_source(source, seed=42)


def test_label_mapping_detects_tu_inverse_encoding() -> None:
    csv_labels = [1, 0, 1, 1, 0]
    tu_labels = [0, 1, 0, 0, 1]
    rows, selected, mismatches = evaluate_tu_label_mappings(csv_labels, tu_labels)
    assert selected == "inverted_0_1"
    assert mismatches == []
    selected_row = next(row for row in rows if row["mapping"] == selected)
    assert selected_row["match_rate"] == 1.0


def test_canonical_same_label_duplicates_are_merged(processed_fixture: dict[str, object]) -> None:
    clean = processed_fixture["clean"]
    ethanol = next(row for row in clean if row["smiles"] == "CCO")
    assert ethanol["label"] == 1
    assert len(str(ethanol["source_row_ids"]).split(";")) == 2
    assert len(processed_fixture["duplicates"]) == 1


def test_conflicting_labels_are_excluded(processed_fixture: dict[str, object]) -> None:
    assert len(processed_fixture["conflicts"]) == 1
    clean_smiles = {row["smiles"] for row in processed_fixture["clean"]}
    assert "CCN" not in clean_smiles
    conflict_drops = [
        row for row in processed_fixture["dropped"] if row["drop_reason"] == "label_conflict"
    ]
    assert len(conflict_drops) == 2


def test_multicomponent_is_excluded(processed_fixture: dict[str, object]) -> None:
    drops = [row["drop_reason"] for row in processed_fixture["dropped"]]
    assert "multicomponent" in drops
    assert all("." not in str(row["smiles"]) for row in processed_fixture["clean"])


def test_invalid_smiles_is_excluded(processed_fixture: dict[str, object]) -> None:
    drops = [row["drop_reason"] for row in processed_fixture["dropped"]]
    assert "parse_failed" in drops


def _synthetic_benchmark() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scaffold_index in range(16):
        scaffold = f"scaffold_{scaffold_index}"
        for label in (0, 1):
            smiles = f"synthetic_{scaffold_index}_{label}"
            rows.append(
                {
                    "molecule_id": stable_molecule_id(smiles),
                    "smiles": smiles,
                    "label": label,
                    "semantic_label": "mutagenic" if label else "non_mutagenic",
                    "scaffold_smiles": scaffold,
                }
            )
    return rows


def test_scaffold_does_not_cross_splits() -> None:
    manifest, summary = build_scaffold_splits(_synthetic_benchmark(), seed=42)
    split_by_scaffold: dict[str, set[str]] = {}
    for row in manifest:
        split_by_scaffold.setdefault(str(row["scaffold_smiles"]), set()).add(str(row["split"]))
    assert all(len(splits) == 1 for splits in split_by_scaffold.values())
    assert summary["scaffold_overlap_count"] == 0


def test_canonical_smiles_does_not_cross_splits() -> None:
    _manifest, summary = build_scaffold_splits(_synthetic_benchmark(), seed=42)
    assert summary["canonical_smiles_overlap_count"] == 0
    assert summary["molecule_id_overlap_count"] == 0


def test_molecule_id_is_stable_and_order_independent() -> None:
    first = stable_molecule_id("CCO")
    second = stable_molecule_id("CCO")
    other = stable_molecule_id("CCN")
    assert first == second
    assert first.startswith("MUT_")
    assert first != other


def test_split_union_is_complete_and_unique() -> None:
    benchmark = _synthetic_benchmark()
    manifest, summary = build_scaffold_splits(
        benchmark,
        seed=42,
        ratios=DEFAULT_SPLIT_RATIOS,
    )
    benchmark_ids = {str(row["molecule_id"]) for row in benchmark}
    manifest_ids = [str(row["molecule_id"]) for row in manifest]
    assert set(manifest_ids) == benchmark_ids
    assert len(manifest_ids) == len(set(manifest_ids))
    assert summary["missing_molecule_count"] == 0
    assert summary["duplicate_molecule_count"] == 0
    assert summary["split_validation_passed"] is True


def test_checksum_manifest_parser_supports_gnu_and_binary_markers(tmp_path: Path) -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{digest_a}  smiles/a.csv\n{digest_b} *tudataset/b.txt\n",
        encoding="utf-8",
    )
    assert parse_sha256_manifest(manifest) == [
        (digest_a, "smiles/a.csv"),
        (digest_b, "tudataset/b.txt"),
    ]

