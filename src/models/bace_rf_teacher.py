"""BACE-specific Morgan-RF teacher built from the shared validated primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.models.mutagenicity_rf_teacher import (
    FeatureConfig,
    PREDICTION_FIELDS,
    SPLIT_NAMES,
    atomic_pickle,
    evaluate_model,
    load_all_splits,
    parse_int_grid,
    parse_optional_int_grid,
    select_random_forest,
)


SOURCE_LABEL = 1
TARGET_LABEL = 0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _split_manifest(datasets: Mapping[str, Any]) -> dict[str, Any]:
    roles = {
        "train": "model_fitting",
        "val": "model_and_hyperparameter_selection",
        "calibration": "distance_threshold_calibration_only",
        "test": "final_teacher_and_counterfactual_evaluation",
    }
    return {
        "schema_version": "bace_teacher_split_manifest_v1",
        "dataset": "BACE",
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "splits": {
            split: {
                "path": str(datasets[split].path),
                "sha256": sha256_file(datasets[split].path),
                "role": roles[split],
                "num_examples": int(datasets[split].size),
                "label_counts": {
                    "0": int((datasets[split].labels == 0).sum()),
                    "1": int((datasets[split].labels == 1).sum()),
                },
            }
            for split in SPLIT_NAMES
        },
        "fit_splits": ["train"],
        "selection_splits": ["val"],
        "calibration_used_for_fit_or_selection": False,
        "test_used_for_fit_or_selection": False,
    }


def _bundle(
    *,
    model: Any,
    feature_config: FeatureConfig,
    selection: Mapping[str, Any],
    metrics: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "task_name": "bace_binary",
        "dataset_name": "BACE",
        "dataset_version": "processed_v1",
        "feature_type": "rdkit_morgan",
        "fingerprint_radius": int(feature_config.radius),
        "fingerprint_bits": int(feature_config.n_bits),
        "positive_label": 1,
        "negative_label": 0,
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "class_labels": [int(label) for label in model.classes_],
        "fit_split_names": ["train"],
        "selection_split_names": ["val"],
        "calibration_used_for_fit_or_selection": False,
        "test_used_for_fit_or_selection": False,
        "selection": dict(selection),
        "metrics": dict(metrics),
        "split_manifest": dict(split_manifest),
    }


def _teacher_consistent_views(
    *,
    data_dir: Path,
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    root = output_dir / "teacher_consistent"
    inventory: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        rows = _read_csv(data_dir / f"{split}.csv")
        by_id = {str(row["molecule_id"]): dict(row) for row in rows}
        predicted = {str(row["molecule_id"]): dict(row) for row in predictions[split]}
        if set(by_id) != set(predicted):
            raise ValueError(f"BACE teacher prediction lineage mismatch for split={split}")
        for role, label in (("source", SOURCE_LABEL), ("target", TARGET_LABEL)):
            selected: list[dict[str, Any]] = []
            for row in rows:
                molecule_id = str(row["molecule_id"])
                prediction = predicted[molecule_id]
                if int(row["label"]) != label:
                    continue
                if int(prediction["teacher_pred"]) != label:
                    continue
                selected.append(
                    {
                        **row,
                        "teacher_pred": int(prediction["teacher_pred"]),
                        "teacher_prob_0": prediction["teacher_prob_0"],
                        "teacher_prob_1": prediction["teacher_prob_1"],
                        "teacher_correct": True,
                    }
                )
            path = root / f"{split}_{role}_label{label}_teacher_correct.csv"
            fields = list(rows[0]) + [
                "teacher_pred",
                "teacher_prob_0",
                "teacher_prob_1",
                "teacher_correct",
            ]
            _write_csv(path, selected, fields)
            inventory[path.name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "num_rows": len(selected),
                "split": split,
                "role": role,
                "label": label,
            }
    return inventory


def train_bace_teacher(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    radius: int = 2,
    n_bits: int = 2048,
    n_estimators_grid: str = "300,600",
    max_depth_grid: str = "none,20,40",
    min_samples_leaf_grid: str = "1,2",
    selection_metric: str = "balanced_accuracy",
    class_weight: str | None = "balanced_subsample",
    random_seed: int = 13,
    n_jobs: int = 7,
) -> dict[str, Any]:
    source = Path(data_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"BACE teacher output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    feature_config = FeatureConfig(radius=int(radius), n_bits=int(n_bits))
    datasets = load_all_splits(
        source,
        feature_config=feature_config,
        smiles_col="smiles",
        label_col="label",
    )
    model, selection, search_results = select_random_forest(
        datasets["train"],
        datasets["val"],
        n_estimators_grid=parse_int_grid(n_estimators_grid),
        max_depth_grid=parse_optional_int_grid(max_depth_grid),
        min_samples_leaf_grid=parse_int_grid(min_samples_leaf_grid),
        random_seed=int(random_seed),
        n_jobs=int(n_jobs),
        class_weight=class_weight,
        selection_metric=selection_metric,
    )
    metrics, predictions = evaluate_model(model, datasets)
    manifest = _split_manifest(datasets)
    model_path = destination / "bace_teacher.pkl"
    atomic_pickle(
        model_path,
        _bundle(
            model=model,
            feature_config=feature_config,
            selection=selection,
            metrics=metrics,
            split_manifest=manifest,
        ),
    )
    prediction_inventory: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        path = destination / f"predictions_{split}.csv"
        _write_csv(path, list(predictions[split]), PREDICTION_FIELDS)
        prediction_inventory[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "num_rows": len(predictions[split]),
        }
    consistent = _teacher_consistent_views(
        data_dir=source,
        predictions=predictions,
        output_dir=destination,
    )
    test_metrics = metrics["test"]
    summary = {
        "schema_version": "bace_teacher_summary_v1",
        "dataset": "BACE",
        "teacher_path": str(model_path),
        "teacher_sha256": sha256_file(model_path),
        "accuracy": float(test_metrics["accuracy"]),
        "f1": float(test_metrics["macro_f1"]),
        "auc": float(test_metrics["auroc"]),
        "dataset_split": manifest,
        "metrics": metrics,
        "selection": selection,
        "search_results": search_results,
        "feature_config": feature_config.to_dict(),
        "prediction_inventory": prediction_inventory,
        "teacher_consistent_views": consistent,
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "fit_split": "train",
        "selection_split": "val",
        "calibration_used_for_fit_or_selection": False,
        "test_used_for_fit_or_selection": False,
    }
    _write_json(destination / "teacher_summary.json", summary)
    _write_json(destination / "split_manifest.json", manifest)
    return summary


__all__ = ["SOURCE_LABEL", "TARGET_LABEL", "sha256_file", "train_bace_teacher"]
