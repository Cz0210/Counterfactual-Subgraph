"""BBBP-specific Morgan-RF teacher built from the shared validated primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.molecular_split import audit_split_overlap, stable_json_sha256

from src.models.mutagenicity_rf_teacher import (
    FeatureConfig,
    PREDICTION_FIELDS,
    SPLIT_NAMES,
    atomic_pickle,
    evaluate_model,
    load_split_dataset,
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
        "schema_version": "bbbp_teacher_split_manifest_v1",
        "dataset": "BBBP",
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
        "threshold_source": "calibration",
        "test_usage": "final_evaluation_only",
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
        "task_name": "bbbp_binary",
        "dataset_name": "BBBP",
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
    split_paths: Mapping[str, Path],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    root = output_dir / "teacher_consistent"
    inventory: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        rows = _read_csv(split_paths[split])
        by_id = {str(row["molecule_id"]): dict(row) for row in rows}
        predicted = {str(row["molecule_id"]): dict(row) for row in predictions[split]}
        if set(by_id) != set(predicted):
            raise ValueError(f"BBBP teacher prediction lineage mismatch for split={split}")
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


def train_bbbp_teacher(
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
    split_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    source = Path(data_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"BBBP teacher output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    feature_config = FeatureConfig(radius=int(radius), n_bits=int(n_bits))
    resolved_paths = {
        split: Path(split_paths[split]).expanduser().resolve()
        if split_paths is not None
        else source / f"{split}.csv"
        for split in SPLIT_NAMES
    }
    datasets = {
        split: load_split_dataset(
            resolved_paths[split],
            split_name=split,
            feature_config=feature_config,
            smiles_col="smiles",
            label_col="label",
        )
        for split in SPLIT_NAMES
    }
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
    leakage_audit = validate_bbbp_teacher_paths(resolved_paths)
    model_path = destination / "bbbp_teacher.pkl"
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
        split_paths=resolved_paths,
        predictions=predictions,
        output_dir=destination,
    )
    test_metrics = metrics["test"]
    summary = {
        "schema_version": "bbbp_teacher_summary_v1",
        "dataset": "BBBP",
        "teacher_path": str(model_path),
        "teacher_sha256": sha256_file(model_path),
        "accuracy": float(test_metrics["accuracy"]),
        "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
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
        "random_seed": int(random_seed),
        "git_commit": _git_commit(),
        "config_hash": stable_json_sha256(
            {
                "feature_config": feature_config.to_dict(),
                "n_estimators_grid": n_estimators_grid,
                "max_depth_grid": max_depth_grid,
                "min_samples_leaf_grid": min_samples_leaf_grid,
                "selection_metric": selection_metric,
                "class_weight": class_weight,
                "random_seed": int(random_seed),
            }
        ),
        "teacher_leakage_audit": leakage_audit,
    }
    _write_json(destination / "teacher_summary.json", summary)
    _write_json(destination / "teacher_split_manifest.json", manifest)
    _write_json(destination / "teacher_leakage_audit.json", leakage_audit)
    return summary


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def validate_bbbp_teacher_inputs(data_dir: str | Path) -> dict[str, Any]:
    """Validate split isolation and RF feature availability without fitting."""

    source = Path(data_dir).expanduser().resolve()
    from src.rewards.reward_calculator import smiles_to_morgan_array

    if not callable(smiles_to_morgan_array):
        raise RuntimeError("BBBP Morgan feature extractor is unavailable.")
    return validate_bbbp_teacher_paths(
        {split: source / f"{split}.csv" for split in SPLIT_NAMES}
    )


def validate_bbbp_teacher_paths(
    split_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    from src.rewards.reward_calculator import smiles_to_morgan_array

    if not callable(smiles_to_morgan_array):
        raise RuntimeError("BBBP Morgan feature extractor is unavailable.")
    missing_keys = sorted(set(SPLIT_NAMES) - set(split_paths))
    if missing_keys:
        raise ValueError(f"BBBP teacher split mapping is missing {missing_keys}.")
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split in SPLIT_NAMES:
        path = Path(split_paths[split]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing BBBP teacher split: {path}")
        rows = _read_csv(path)
        if not rows:
            raise ValueError(f"BBBP teacher split is empty: {path}")
        required = {"molecule_id", "canonical_smiles", "smiles", "label"}
        missing = sorted(required - set(rows[0]))
        if missing:
            raise ValueError(f"BBBP teacher split {path} is missing {missing}.")
        rows_by_split[split] = rows
    audit = audit_split_overlap(
        rows_by_split,
        require_scaffold_disjoint=True,
    )
    return {
        **audit,
        "schema_version": "bbbp_teacher_leakage_audit_v1",
        "dataset": "BBBP",
        "fit_splits": ["train"],
        "selection_splits": ["val"],
        "calibration_loaded_for_fit_or_selection": False,
        "test_loaded_for_fit_or_selection": False,
        "feature_extractor": "rdkit_morgan",
    }


__all__ = [
    "SOURCE_LABEL",
    "TARGET_LABEL",
    "sha256_file",
    "train_bbbp_teacher",
    "validate_bbbp_teacher_inputs",
    "validate_bbbp_teacher_paths",
]
