import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_tastemolnet import prepare_tastemolnet_dataset


pytest.importorskip("rdkit")


def _write_source(path: Path) -> None:
    rows = [
        ("b0", "CCO", "Bitter"),
        ("b1", "CC(=O)O", "Bitter"),
        ("b2", "c1ccccc1O", "Bitter"),
        ("b3", "CCCl", "Bitter"),
        ("b4", "CCBr", "Bitter"),
        ("s-conflict", "OCC", "Sweet"),
        ("s1", "CCN", "Sweet"),
        ("s2", "CCC", "Sweet"),
        ("s3", "c1ccccc1N", "Sweet"),
        ("s4", "CCS", "Sweet"),
        ("s5", "CCF", "Sweet"),
        ("t1", "COC", "Tasteless"),
        ("t2", "CC#N", "Tasteless"),
        ("t3", "c1ccncc1", "Tasteless"),
        ("t4", "O=C=O", "Tasteless"),
        ("t5", "C1CCCCC1", "Tasteless"),
        ("amb", "CCP", "Sweet/Bitter"),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["COMPOUND_ID", "PROCESSED_SMILES", "TARGET"])
        writer.writerows(rows)


def test_offline_prepare_writes_governance_and_split_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "processed"
    _write_source(source)

    summary = prepare_tastemolnet_dataset(
        source_csv=source,
        output_dir=output,
        source_mode="local",
        component_policy="canonical_only",
        split_seed=7,
        require_all_classes_per_split=False,
    )

    assert summary["status"] == "READY_NOT_RUN"
    assert summary["run_tastemolnet"] is False
    assert summary["license_status"] == "LICENSE_REVIEW_REQUIRED"
    assert summary["scaffold_overlap_gate_passed"] is True
    assert (output / "LICENSE_REVIEW_REQUIRED").is_file()
    assert not (output / source.name).exists()

    with (output / "cross_label_conflicts.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        conflicts = list(csv.DictReader(handle))
    assert len(conflicts) == 1
    assert conflicts[0]["conflict_identity"] == "CCO"
    assert conflicts[0]["labels"] == "[0,1]"

    with (output / "splits" / "excluded_rows.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        exclusions = list(csv.DictReader(handle))
    assert {
        (row["source_row_id"], row["exclusion_reason"]) for row in exclusions
    } >= {
        ("b0", "CROSS_LABEL_DUPLICATE"),
        ("s-conflict", "CROSS_LABEL_DUPLICATE"),
        ("amb", "AMBIGUOUS_LABEL"),
    }

    split_audit = json.loads(
        (output / "splits" / "scaffold_overlap_audit.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = json.loads(
        (output / "provenance_manifest.json").read_text(encoding="utf-8")
    )
    component_audit = json.loads(
        (output / "component_strategy_audit.json").read_text(encoding="utf-8")
    )
    assert split_audit["passed"] is True
    assert provenance["download_performed"] is False
    assert provenance["raw_data_copied_into_output"] is False
    assert provenance["schema_detection"] == {
        "id_column": "COMPOUND_ID",
        "label_column": "TARGET",
        "smiles_column": "PROCESSED_SMILES",
    }
    assert set(component_audit["strategies"]) == {
        "canonical_only",
        "largest_organic_fragment",
    }
    for strategy in component_audit["strategies"].values():
        assert "retention_rate" in strategy
        assert "cross_label_conflict_identities" in strategy
        assert "atom_count" in strategy
        assert "molclr_compatible_rate" in strategy


def test_upstream_mode_requires_commit_url_and_fresh_output(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(source)
    with pytest.raises(ValueError, match="upstream_commit"):
        prepare_tastemolnet_dataset(
            source_csv=source,
            output_dir=tmp_path / "out",
            source_mode="upstream_processed",
            source_url="https://example.invalid/data.csv",
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="fresh absent path"):
        prepare_tastemolnet_dataset(
            source_csv=source,
            output_dir=existing,
            source_mode="local",
        )
