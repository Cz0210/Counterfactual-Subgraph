from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl import bace_gnn_stage
from scripts.autodl.bace_gnn_stage import (
    _real_connected_deletions,
    require_every_parent_has_connected_deletion,
    select_correctly_predicted_source_indices,
    validate_b3_validation_provenance,
    validate_bace_model_card,
)


def test_b4_calibrates_fresh_copy_without_mutating_b3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "b3"
    source.mkdir()
    source_temperature = {
        "status": "not_fit",
        "selection_split": "validation",
        "test_used_for_fit": False,
        "temperature": 1.0,
    }
    (source / "temperature_scaling.json").write_text(
        json.dumps(source_temperature), encoding="utf-8"
    )
    (source / "sha256sums.txt").write_text("source\n", encoding="utf-8")
    validation = tmp_path / "val.csv"
    validation.write_text("molecule_id,smiles,label,split\nv,CC,1,val\n", encoding="utf-8")
    (source / "split_manifest.json").write_text(
        json.dumps(
            {
                "roles": {
                    "validation": "checkpoint_selection_and_temperature_calibration"
                },
                "files": {
                    "validation": {
                        "path": str(validation),
                        "sha256": bace_gnn_stage.sha256_file(validation),
                    }
                },
                "test_used_for_checkpoint_selection": False,
                "calibration_loaded_for_training": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "b4"

    monkeypatch.setattr(
        bace_gnn_stage,
        "verify_checkpoint_bundle",
        lambda _path: {
            "model_card": {
                "checkpoint_id": "checkpoint-id",
                "dataset": "bace",
                "selection_split": "validation",
                "temperature_calibration_split": "validation",
            }
        },
    )

    def fake_calibrate(arguments: list[str]) -> int:
        checkpoint = Path(arguments[arguments.index("--checkpoint-dir") + 1])
        validation_path = Path(arguments[arguments.index("--validation-csv") + 1])
        payload = {
            "status": "fit",
            "selection_split": "validation",
            "test_used_for_fit": False,
            "argmax_invariant": True,
            "temperature": 1.5,
            "nll_before": 0.7,
            "nll_after": 0.6,
            "ece_before": 0.2,
            "ece_after": 0.1,
            "brier_before": 0.3,
            "brier_after": 0.2,
            "validation_csv_sha256": bace_gnn_stage.sha256_file(validation_path),
        }
        (checkpoint / "temperature_scaling.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(bace_gnn_stage, "calibrate_main", fake_calibrate)
    assert bace_gnn_stage.main(
        [
            "calibrate",
            "--source-checkpoint",
            str(source),
            "--output-checkpoint",
            str(output),
            "--validation-csv",
            str(validation),
        ]
    ) == 0
    assert json.loads(
        (source / "temperature_scaling.json").read_text(encoding="utf-8")
    ) == source_temperature
    assert json.loads(
        (output / "temperature_scaling.json").read_text(encoding="utf-8")
    )["status"] == "fit"
    assert (output / "b4_calibration.json").is_file()
    assert "b4_calibration.json" in (
        output / "sha256sums.txt"
    ).read_text(encoding="utf-8")


def test_b4_validation_must_match_b3_path_and_sha(tmp_path: Path) -> None:
    checkpoint = tmp_path / "b3"
    checkpoint.mkdir()
    frozen = tmp_path / "frozen_val.csv"
    frozen.write_text("id,smiles,label,split\na,CC,1,val\n", encoding="utf-8")
    other = tmp_path / "other_val.csv"
    other.write_text("id,smiles,label,split\na,CCC,1,val\n", encoding="utf-8")
    manifest = {
        "roles": {
            "validation": "checkpoint_selection_and_temperature_calibration"
        },
        "files": {
            "validation": {
                "path": str(frozen),
                "sha256": bace_gnn_stage.sha256_file(frozen),
            }
        },
        "test_used_for_checkpoint_selection": False,
        "calibration_loaded_for_training": False,
    }
    (checkpoint / "split_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    evidence = validate_b3_validation_provenance(checkpoint, frozen.resolve())
    assert evidence["validation_csv_sha256"] == bace_gnn_stage.sha256_file(frozen)
    with pytest.raises(ValueError, match="validation path differs"):
        validate_b3_validation_provenance(checkpoint, other.resolve())
    manifest["files"]["validation"]["sha256"] = "0" * 64
    (checkpoint / "split_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="validation SHA differs"):
        validate_b3_validation_provenance(checkpoint, frozen.resolve())


def test_b5_freezes_exactly_sixteen_correct_source_calibration_parents() -> None:
    labels = [1] * 12 + [0] * 3 + [1] * 12
    predictions = [
        {"predicted_label": 1 if label == 1 and index not in {2, 20} else 0}
        for index, label in enumerate(labels)
    ]
    selected = select_correctly_predicted_source_indices(
        labels, predictions, source_label=1, count=16
    )
    assert len(selected) == 16
    assert all(labels[index] == 1 for index in selected)
    assert all(predictions[index]["predicted_label"] == 1 for index in selected)


def test_b5_source_cohort_fails_closed_when_fewer_than_sixteen() -> None:
    labels = [1] * 15 + [0] * 5
    predictions = [{"predicted_label": label} for label in labels]
    with pytest.raises(ValueError, match="exactly 16"):
        select_correctly_predicted_source_indices(
            labels, predictions, source_label=1, count=16
        )


def test_b5_requires_exact_bace_model_contract_and_one_deletion_per_parent() -> None:
    card = {
        "dataset": "bace",
        "num_classes": 2,
        "source_label": 1,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
    }
    validate_bace_model_card(card)
    for key, bad in (("dataset", "tastemolnet"), ("num_classes", 3), ("source_label", 0)):
        candidate = {**card, key: bad}
        with pytest.raises(ValueError, match="model-card contract"):
            validate_bace_model_card(candidate)

    counts = {f"parent-{index}": 1 for index in range(16)}
    require_every_parent_has_connected_deletion(counts)
    counts["parent-7"] = 0
    with pytest.raises(ValueError, match="each of 16 parents"):
        require_every_parent_has_connected_deletion(counts)


def test_b5_real_parent_deletions_are_connected_and_sanitized() -> None:
    actions = _real_connected_deletions(
        "CCOC(=O)N", parent_id="real-calibration-parent", maximum=4
    )
    assert actions
    assert len(actions) <= 4
    assert all(fragment for fragment, _outcome in actions)
    assert all(outcome.valid for _fragment, outcome in actions)
    assert all(outcome.residual_connected for _fragment, outcome in actions)
    assert all(outcome.sanitize_ok for _fragment, outcome in actions)
