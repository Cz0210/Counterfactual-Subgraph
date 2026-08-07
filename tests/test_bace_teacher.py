from __future__ import annotations

import csv
import json
from pathlib import Path

from src.models.bace_rf_teacher import train_bace_teacher


MOLECULES = {
    "train": [("C", 0), ("CC", 0), ("N", 1), ("NN", 1)],
    "val": [("CCC", 0), ("CO", 0), ("NNN", 1), ("CN", 1)],
    "calibration": [("CCCC", 0), ("CCO", 0), ("NNNN", 1), ("CCN", 1)],
    "test": [("CCCCC", 0), ("CCCO", 0), ("NNNNN", 1), ("CCCN", 1)],
}


def _write_splits(root: Path) -> None:
    root.mkdir()
    index = 0
    for split, rows in MOLECULES.items():
        with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["molecule_id", "parent_id", "smiles", "label"]
            )
            writer.writeheader()
            for smiles, label in rows:
                writer.writerow(
                    {
                        "molecule_id": f"BACE_{index:04d}",
                        "parent_id": f"BACE_{index:04d}",
                        "smiles": smiles,
                        "label": label,
                    }
                )
                index += 1


def test_bace_teacher_is_independent_and_uses_val_selection(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "teacher"
    _write_splits(data)
    summary = train_bace_teacher(
        data_dir=data,
        output_dir=output,
        n_estimators_grid="5",
        max_depth_grid="none",
        min_samples_leaf_grid="1",
        n_jobs=1,
    )
    assert (output / "bace_teacher.pkl").is_file()
    assert summary["dataset"] == "BACE"
    assert summary["selection"]["selection_split"] == "val"
    assert summary["calibration_used_for_fit_or_selection"] is False
    assert summary["test_used_for_fit_or_selection"] is False
    assert set(("accuracy", "f1", "auc", "dataset_split")) <= set(summary)
    persisted = json.loads((output / "teacher_summary.json").read_text())
    assert persisted["teacher_path"].endswith("bace_teacher.pkl")
    assert (output / "teacher_consistent/test_source_label1_teacher_correct.csv").is_file()
