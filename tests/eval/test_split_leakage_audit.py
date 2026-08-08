from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.eval.split_leakage_audit import audit_split_files


def _split_files(tmp_path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for index, split in enumerate(("train", "val", "calibration", "test")):
        path = tmp_path / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["molecule_id", "canonical_smiles", "scaffold_smiles", "label"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "molecule_id": f"m{index}",
                    "canonical_smiles": ["CC", "CCC", "CCCC", "CCCCC"][index],
                    "scaffold_smiles": f"s{index}",
                    "label": index % 2,
                }
            )
        result[split] = path
    return result


def test_split_leakage_audit_passes_frozen_roles(tmp_path: Path) -> None:
    audit = audit_split_files(
        _split_files(tmp_path),
        protocol="heldout",
        require_scaffold_disjoint=True,
        candidate_source_splits=("train", "val"),
        selector_source_splits=("calibration",),
        threshold_source_split="calibration",
    )
    assert audit["passed"] is True
    assert audit["threshold_fitted_on_test"] is False


@pytest.mark.parametrize(
    ("candidate_splits", "selector_splits", "threshold", "message"),
    [
        (("train", "test"), ("calibration",), "calibration", "test_used_for_candidate"),
        (("train",), ("test",), "calibration", "test_used_for_selector"),
        (("train",), ("calibration",), "test", "threshold_fitted_on_test"),
    ],
)
def test_test_or_threshold_leakage_fails_closed(
    tmp_path: Path,
    candidate_splits: tuple[str, ...],
    selector_splits: tuple[str, ...],
    threshold: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_split_files(
            _split_files(tmp_path),
            protocol="heldout",
            require_scaffold_disjoint=True,
            candidate_source_splits=candidate_splits,
            selector_source_splits=selector_splits,
            threshold_source_split=threshold,
        )


def test_molecule_overlap_fails_closed(tmp_path: Path) -> None:
    paths = _split_files(tmp_path)
    text = paths["test"].read_text(encoding="utf-8").replace("m3", "m0")
    paths["test"].write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        audit_split_files(
            paths,
            protocol="heldout",
            require_scaffold_disjoint=True,
            candidate_source_splits=("train",),
            selector_source_splits=("calibration",),
            threshold_source_split="calibration",
        )
