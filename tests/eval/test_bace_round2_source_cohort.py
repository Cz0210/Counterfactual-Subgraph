from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_bace_round2_source_cohort import build_round2_cohort


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_round2_cohort_uses_calibration_hard_groups_and_train_only(tmp_path: Path) -> None:
    matrix = tmp_path / "pairs.jsonl"
    matrix.write_text(
        json.dumps(
            {
                "parent_id": "c1",
                "num_connected_valid_matches": 1,
                "pair_strict_flip": True,
                "wnode_distance": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"theta_star": 0.1}), encoding="utf-8")
    calibration = tmp_path / "calibration.csv"
    _csv(
        calibration,
        [{"molecule_id": "c1", "smiles": "CCO", "scaffold": "", "split": "calibration"}],
    )
    train = tmp_path / "train.csv"
    _csv(
        train,
        [
            {"molecule_id": "t1", "smiles": "CCN", "scaffold": "", "split": "train"},
            {"molecule_id": "t2", "smiles": "c1ccccc1", "scaffold": "c1ccccc1", "split": "train"},
        ],
    )
    output = tmp_path / "round2.csv"
    manifest = tmp_path / "manifest.json"

    payload = build_round2_cohort(
        pair_matrix=matrix,
        thresholds_json=thresholds,
        calibration_csv=calibration,
        train_csv=train,
        output_csv=output,
        manifest_path=manifest,
        nearest_per_hard_parent=1,
    )

    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    assert rows[0]["split"] == "train"
    assert payload["hard_group_counts"] == {"B_only_high_threshold": 1}
    assert payload["test_loaded"] is False
