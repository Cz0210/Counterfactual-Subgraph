from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("rdkit")

from scripts.evaluate_mutagenicity_rf_teacher import evaluate_teacher
from scripts.train_mutagenicity_rf_teacher import (
    DEFAULT_DATA_DIR,
    build_parser,
    train_teacher,
)
from src.models.mutagenicity_rf_teacher import (
    PREDICTION_FIELDS,
    FeatureConfig,
    classification_metrics,
    load_all_splits,
    load_bundle,
)
from src.rewards.reward_calculator import load_oracle_bundle


MOLECULES = (
    "C",
    "CC",
    "CCC",
    "CCCC",
    "CCO",
    "CCN",
    "COC",
    "CNC",
    "CCCl",
    "CCBr",
    "CCF",
    "CCS",
    "c1ccccc1",
    "c1ccncc1",
    "C1CCCCC1",
    "C1CCOCC1",
)


def _write_split(path: Path, smiles_values: tuple[str, ...], split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("molecule_id", "smiles", "label", "semantic_label", "split"),
        )
        writer.writeheader()
        for index, smiles in enumerate(smiles_values):
            label = index % 2
            writer.writerow(
                {
                    "molecule_id": f"{split}_{index}",
                    "smiles": smiles,
                    "label": label,
                    "semantic_label": "mutagenic" if label else "non_mutagenic",
                    "split": split,
                }
            )


@pytest.fixture()
def split_dir(tmp_path: Path) -> Path:
    for split_index, split in enumerate(("train", "val", "calibration", "test")):
        start = split_index * 4
        _write_split(tmp_path / f"{split}.csv", MOLECULES[start : start + 4], split)
    return tmp_path


def _training_args(data_dir: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=data_dir,
        output_dir=output_dir,
        smiles_col="smiles",
        label_col="label",
        radius=2,
        n_bits=128,
        n_estimators_grid="10",
        max_depth_grid="none",
        min_samples_leaf_grid="1",
        selection_metric="balanced_accuracy",
        class_weight="balanced_subsample",
        random_seed=7,
        n_jobs=1,
    )


def test_default_data_path_points_to_processed_subdirectory() -> None:
    assert DEFAULT_DATA_DIR == Path(
        "outputs/hpc/datasets/final/mutagenicity_v1_processed"
    )
    assert "mutagenicity_v1/train.csv" not in str(DEFAULT_DATA_DIR)


def test_load_all_splits_preserves_fixed_roles_and_disjoint_ids(split_dir: Path) -> None:
    datasets = load_all_splits(split_dir, feature_config=FeatureConfig(n_bits=128))
    assert set(datasets) == {"train", "val", "calibration", "test"}
    all_ids = [molecule_id for dataset in datasets.values() for molecule_id in dataset.molecule_ids]
    assert len(all_ids) == len(set(all_ids)) == 16
    assert all(set(dataset.labels.tolist()) == {0, 1} for dataset in datasets.values())


def test_classification_metrics_contains_probability_and_per_class_metrics() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray(
        [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]], dtype=np.float64
    )
    metrics = classification_metrics(labels, probabilities)
    for key in (
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "average_precision",
        "brier_score",
        "per_class",
        "confusion_matrix",
    ):
        assert key in metrics
    assert set(metrics["per_class"]) == {"0", "1"}
    assert set(metrics["per_class"]["1"]) == {"precision", "recall", "f1", "support"}


def test_training_writes_complete_artifact_contract(split_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "teacher"
    model_path = train_teacher(_training_args(split_dir, output_dir))
    required = {
        "mutagenicity_rf_model.pkl",
        "config.json",
        "feature_config.json",
        "metrics.json",
        "predictions_train.csv",
        "predictions_val.csv",
        "predictions_calibration.csv",
        "predictions_test.csv",
        "confusion_matrix.csv",
        "split_manifest.json",
        "teacher_report.md",
    }
    assert required.issubset({path.name for path in output_dir.iterdir()})
    assert model_path == output_dir / "mutagenicity_rf_model.pkl"
    with (output_dir / "predictions_test.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == PREDICTION_FIELDS
        assert len(list(reader)) == 4
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["splits"]) == {"train", "val", "calibration", "test"}
    assert metrics["source_label"] == 1
    assert metrics["target_label"] == 0


def test_bundle_is_compatible_and_records_no_calibration_or_test_leakage(
    split_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "teacher"
    model_path = train_teacher(_training_args(split_dir, output_dir))
    bundle = load_bundle(model_path)
    shared_bundle = load_oracle_bundle(model_path)
    assert shared_bundle["model"] is not None
    assert bundle["fit_split_names"] == ["train"]
    assert bundle["selection_split_names"] == ["val"]
    assert bundle["calibration_used_for_fit_or_selection"] is False
    assert bundle["test_used_for_fit_or_selection"] is False
    assert bundle["selection"]["selection_split"] == "val"


def test_standalone_evaluator_reloads_model_and_rewrites_predictions(
    split_dir: Path, tmp_path: Path
) -> None:
    train_output = tmp_path / "train_output"
    model_path = train_teacher(_training_args(split_dir, train_output))
    eval_output = tmp_path / "eval_output"
    evaluate_teacher(
        argparse.Namespace(
            data_dir=split_dir,
            model_path=model_path,
            output_dir=eval_output,
            smiles_col="smiles",
            label_col="label",
        )
    )
    assert (eval_output / "metrics.json").is_file()
    assert (eval_output / "evaluation_config.json").is_file()
    assert (eval_output / "predictions_calibration.csv").is_file()


def test_cli_parser_accepts_small_smoke_grid() -> None:
    args = build_parser().parse_args(
        [
            "--n-estimators-grid",
            "10",
            "--max-depth-grid",
            "none,5",
            "--min-samples-leaf-grid",
            "1",
        ]
    )
    assert args.n_estimators_grid == "10"
    assert args.max_depth_grid == "none,5"

