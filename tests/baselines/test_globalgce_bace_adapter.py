from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.baselines.globalgce.freeze_bace_frequency_top20 import (
    freeze_frequency_top20,
)
from src.baselines.globalgce_mutagenicity_adapter import stable_candidate_id
from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pool(root: Path, teacher: Path, *, count: int) -> None:
    root.mkdir()
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "BACE",
                "run_complete": True,
                "calibration_used": False,
                "test_used": False,
                "inputs": {"teacher_path": {"sha256": _sha(teacher)}},
            }
        ),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps({"canonical_unique_candidates": count}), encoding="utf-8"
    )
    rows = []
    for index in range(1, count + 1):
        smiles = "C" * index
        rows.append(
            {
                "candidate_id": stable_candidate_id(smiles, dataset_name="BACE"),
                "canonical_smiles": smiles,
                "teacher_target_ok": True,
                "teacher_pred": 0,
                "source_parent_count": count - index + 1,
                "source_occurrence_count": 2 * (count - index + 1),
            }
        )
    (root / "candidate_universe.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_bace_candidate_ids_do_not_change_mutagenicity_defaults() -> None:
    smiles = "CC"
    assert stable_candidate_id(smiles).startswith("MUT_GLOBALGCE_")
    assert stable_candidate_id(smiles, dataset_name="BACE").startswith(
        "BACE_GLOBALGCE_"
    )


def test_bace_frequency_top20_is_train_only_connected_and_deterministic(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher.pkl"
    molclr = tmp_path / "model.pth"
    thresholds = tmp_path / "thresholds.json"
    teacher.write_bytes(b"teacher")
    molclr.write_bytes(b"molclr")
    thresholds.write_text("{}", encoding="utf-8")
    pool = tmp_path / "pool"
    _write_pool(pool, teacher, count=22)

    output = tmp_path / "selector"
    result = freeze_frequency_top20(
        run_dir=pool,
        teacher_path=teacher,
        molclr_checkpoint=molclr,
        thresholds_json=thresholds,
        output_dir=output,
    )

    assert result["passed"] is True
    assert result["selection_split"] == "train"
    assert result["test_used"] is False
    assert result["action_semantics_version"] == CONNECTED_ACTION_SEMANTICS
    assert result["match_selection_policy"] == CONNECTED_MATCH_SELECTION_POLICY
    with (output / "selected_top20_for_eval.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["rank"]) for row in rows] == list(range(1, 21))
    assert len({row["candidate_id"] for row in rows}) == 20
    assert all(row["connected"] == "True" for row in rows)
    assert all(row["source_split"] == "train" for row in rows)


def test_bace_frequency_top20_fails_closed_when_pool_is_too_small(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher.pkl"
    molclr = tmp_path / "model.pth"
    thresholds = tmp_path / "thresholds.json"
    teacher.write_bytes(b"teacher")
    molclr.write_bytes(b"molclr")
    thresholds.write_text("{}", encoding="utf-8")
    pool = tmp_path / "pool"
    _write_pool(pool, teacher, count=19)

    with pytest.raises(RuntimeError, match="INSUFFICIENT_VALID_CONNECTED"):
        freeze_frequency_top20(
            run_dir=pool,
            teacher_path=teacher,
            molclr_checkpoint=molclr,
            thresholds_json=thresholds,
            output_dir=tmp_path / "selector",
        )
