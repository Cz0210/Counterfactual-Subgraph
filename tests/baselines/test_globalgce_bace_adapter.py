from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.baselines.globalgce.freeze_bace_frequency_top20 import (
    freeze_frequency_top20,
)
from src.baselines.globalgce_bace_adapter import (
    audit_bace_globalgce_train_contract,
)
from src.baselines.globalgce_mutagenicity_adapter import stable_candidate_id
from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stable(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_train_contract(root: Path) -> tuple[Path, Path]:
    processed = root / "processed" / "BACE"
    prepared = root / "prepared"
    processed.mkdir(parents=True)
    prepared.mkdir()
    native_rows = []
    for index in range(959):
        native_rows.append(
            {
                "molecule_id": f"p{index:04d}",
                "smiles": "CC" if index % 2 == 0 else "CO",
                "label": 1 if index < 360 or 869 <= index < 895 else 0,
                "split": "train",
            }
        )
    native_csv = processed / "train.csv"
    with native_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(native_rows[0]))
        writer.writeheader()
        writer.writerows(native_rows)
    native_summary = {
        "schema_version": "bace_processed_v1",
        "dataset": "BACE",
        "dataset_fingerprint": "dataset-fingerprint",
        "split_seed": 13,
        "split_counts": {
            "train": 959,
            "val": 187,
            "calibration": 129,
            "test": 238,
        },
    }
    native_summary_path = processed / "bace_dataset_summary.json"
    native_summary_path.write_text(json.dumps(native_summary), encoding="utf-8")
    (processed / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bace_processed_manifest_v1",
                "dataset": "BACE",
                "dataset_fingerprint": "dataset-fingerprint",
                "files": {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "sha256": _sha(path),
                    }
                    for path in (native_csv, native_summary_path)
                },
            }
        ),
        encoding="utf-8",
    )

    source_rows = []
    for index in range(869):
        label = 1 if index < 360 else 0
        source_rows.append(
            {
                "molecule_id": f"p{index:04d}",
                "canonical_smiles": "CC" if index % 2 == 0 else "CO",
                "split": "train",
                "label": label,
                "gnn_label": 0 if label == 1 else 1,
                "source_label": 1,
                "target_label": 0,
            }
        )
    for index in range(162):
        label = 1 if index < 92 else 0
        source_rows.append(
            {
                "molecule_id": f"v{index:04d}",
                "canonical_smiles": "CN",
                "split": "val",
                "label": label,
                "gnn_label": 0 if label == 1 else 1,
                "source_label": 1,
                "target_label": 0,
            }
        )
    source_path = prepared / "source_graph_manifest.jsonl"
    _write_jsonl(source_path, source_rows)
    summary = {
        "dataset": "BACE",
        "adapter": "official_gcfexplainer_bace_project_data",
        "train_rows": 869,
        "train_source_rows": 360,
        "train_target_rows": 509,
        "val_rows": 162,
        "val_source_rows": 92,
        "val_target_rows": 70,
        "generation_source_rows": 360,
        "gnn_label_mapping": {"project_1": 0, "project_0": 1},
        "train_ids_hash": _stable(
            [row["molecule_id"] for row in source_rows if row["split"] == "train"]
        ),
        "val_ids_hash": _stable(
            [row["molecule_id"] for row in source_rows if row["split"] == "val"]
        ),
        "generation_source_cohort_hash": _stable(
            [
                row["molecule_id"]
                for row in source_rows
                if row["split"] == "train" and row["label"] == 1
            ]
        ),
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    summary_path = prepared / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (prepared / "run_manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "artifacts": {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "sha256": _sha(path),
                    }
                    for path in (source_path, summary_path)
                },
            }
        ),
        encoding="utf-8",
    )
    return source_path, native_csv


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


def test_bace_globalgce_contract_maps_959_to_exact_frozen_869_and_360(
    tmp_path: Path,
) -> None:
    source, native = _write_train_contract(tmp_path)

    contract = audit_bace_globalgce_train_contract(
        source_manifest=source,
        native_train_csv=native,
    )

    assert len(contract.native_train_parent_ids) == 869
    assert len(contract.source_parents) == 360
    assert {row.parent_id for row in contract.source_parents} == {
        f"p{index:04d}" for index in range(360)
    }
    assert contract.audit["native_input_train_rows"] == 959
    assert contract.audit["native_excluded_train_rows"] == 90
    assert contract.audit["validation_manifest_rows_audited_not_loaded"] == 162
    assert contract.audit["calibration_loaded"] is False
    assert contract.audit["test_loaded"] is False


def test_bace_globalgce_contract_rejects_calibration_or_test_manifest_rows(
    tmp_path: Path,
) -> None:
    source, native = _write_train_contract(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[-1]["split"] = "test"
    _write_jsonl(source, rows)
    run_path = source.parent / "run_manifest.json"
    run = json.loads(run_path.read_text())
    run["artifacts"][source.name] = {
        "bytes": source.stat().st_size,
        "sha256": _sha(source),
    }
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration/test"):
        audit_bace_globalgce_train_contract(
            source_manifest=source,
            native_train_csv=native,
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
