"""Training and evaluation utilities for the Mutagenicity RF teacher."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.rewards.reward_calculator import smiles_to_morgan_array

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
except ImportError:  # pragma: no cover - checked explicitly at runtime.
    RandomForestClassifier = None
    accuracy_score = None
    average_precision_score = None
    balanced_accuracy_score = None
    brier_score_loss = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    roc_auc_score = None


SPLIT_NAMES = ("train", "val", "calibration", "test")
SPLIT_ROLES = {
    "train": "model_fitting",
    "val": "model_and_hyperparameter_selection",
    "calibration": "reserved_probability_or_threshold_calibration",
    "test": "final_teacher_quality_evaluation",
}
PREDICTION_FIELDS = (
    "molecule_id",
    "smiles",
    "label",
    "teacher_pred",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_correct",
)


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = False
    clean_dummy_atoms: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_type": "rdkit_morgan_bit_vector",
            "fingerprint_radius": int(self.radius),
            "fingerprint_bits": int(self.n_bits),
            "use_chirality": bool(self.use_chirality),
            "clean_dummy_atoms": bool(self.clean_dummy_atoms),
            "dtype": "float32",
        }


@dataclass(frozen=True, slots=True)
class SplitDataset:
    name: str
    path: Path
    molecule_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    labels: np.ndarray
    features: np.ndarray

    @property
    def size(self) -> int:
        return len(self.smiles)


def require_dependencies() -> None:
    if RandomForestClassifier is None or accuracy_score is None:
        raise RuntimeError(
            "Mutagenicity RF teacher requires scikit-learn and RDKit. "
            "Activate the smiles_pip118 environment before running."
        )


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Mutagenicity split CSV does not exist: {target}")
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Mutagenicity split CSV is empty: {target}")
    return rows


def load_split_dataset(
    path: str | Path,
    *,
    split_name: str,
    feature_config: FeatureConfig,
    smiles_col: str = "smiles",
    label_col: str = "label",
) -> SplitDataset:
    require_dependencies()
    rows = read_csv_rows(path)
    required = {"molecule_id", smiles_col, label_col}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    molecule_ids: list[str] = []
    smiles_values: list[str] = []
    labels: list[int] = []
    features: list[np.ndarray] = []
    invalid: list[str] = []
    for row_index, row in enumerate(rows, 1):
        molecule_id = str(row.get("molecule_id") or "").strip()
        smiles = str(row.get(smiles_col) or "").strip()
        try:
            label = int(str(row.get(label_col)).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label at {path}:{row_index}: {row.get(label_col)!r}") from exc
        if not molecule_id:
            raise ValueError(f"Missing molecule_id at {path}:{row_index}")
        if not smiles:
            raise ValueError(f"Missing SMILES at {path}:{row_index}")
        if label not in (0, 1):
            raise ValueError(f"Label must be 0/1 at {path}:{row_index}: {label}")
        fingerprint = smiles_to_morgan_array(
            smiles,
            radius=feature_config.radius,
            n_bits=feature_config.n_bits,
            clean_dummy_atoms=feature_config.clean_dummy_atoms,
        )
        if fingerprint is None:
            invalid.append(f"{row_index}:{molecule_id}")
            continue
        molecule_ids.append(molecule_id)
        smiles_values.append(smiles)
        labels.append(label)
        features.append(fingerprint)
    if invalid:
        raise ValueError(
            f"Processed Mutagenicity split contains invalid SMILES: {path}; "
            f"count={len(invalid)}, sample={invalid[:10]}"
        )
    if len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError(f"Duplicate molecule_id values within split: {path}")
    if len(set(smiles_values)) != len(smiles_values):
        raise ValueError(f"Duplicate canonical SMILES values within split: {path}")
    if set(labels) != {0, 1}:
        raise ValueError(f"Split must contain both labels 0 and 1: {path}")
    return SplitDataset(
        name=split_name,
        path=Path(path),
        molecule_ids=tuple(molecule_ids),
        smiles=tuple(smiles_values),
        labels=np.asarray(labels, dtype=np.int64),
        features=np.asarray(features, dtype=np.float32),
    )


def load_all_splits(
    data_dir: str | Path,
    *,
    feature_config: FeatureConfig,
    smiles_col: str = "smiles",
    label_col: str = "label",
) -> dict[str, SplitDataset]:
    root = Path(data_dir)
    datasets = {
        split: load_split_dataset(
            root / f"{split}.csv",
            split_name=split,
            feature_config=feature_config,
            smiles_col=smiles_col,
            label_col=label_col,
        )
        for split in SPLIT_NAMES
    }
    seen_ids: dict[str, str] = {}
    seen_smiles: dict[str, str] = {}
    for split, dataset in datasets.items():
        for molecule_id in dataset.molecule_ids:
            if molecule_id in seen_ids:
                raise ValueError(
                    f"molecule_id crosses splits: {molecule_id} in {seen_ids[molecule_id]} and {split}"
                )
            seen_ids[molecule_id] = split
        for smiles in dataset.smiles:
            if smiles in seen_smiles:
                raise ValueError(
                    f"canonical SMILES crosses splits: {smiles} in {seen_smiles[smiles]} and {split}"
                )
            seen_smiles[smiles] = split
    return datasets


def class_probability_matrix(model: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    output = np.zeros((features.shape[0], 2), dtype=np.float64)
    for index, label in enumerate(model.classes_):
        integer_label = int(label)
        if integer_label not in (0, 1):
            raise ValueError(f"RF teacher exposes unsupported class label: {label}")
        output[:, integer_label] = raw[:, index]
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("RF teacher probabilities do not sum to one")
    return output


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        labels=[0, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    metrics: dict[str, Any] = {
        "num_examples": int(labels.shape[0]),
        "label_counts": {
            "0": int(np.sum(labels == 0)),
            "1": int(np.sum(labels == 1)),
        },
        "prediction_counts": {
            "0": int(np.sum(predicted == 0)),
            "1": int(np.sum(predicted == 1)),
        },
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "auroc": float(roc_auc_score(labels, probabilities[:, 1])),
        "average_precision": float(average_precision_score(labels, probabilities[:, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities[:, 1])),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            str(label): {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in (0, 1)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    return metrics


def prediction_rows(
    dataset: SplitDataset,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    return [
        {
            "molecule_id": molecule_id,
            "smiles": smiles,
            "label": int(label),
            "teacher_pred": int(prediction),
            "teacher_prob_0": float(probability[0]),
            "teacher_prob_1": float(probability[1]),
            "teacher_correct": bool(prediction == label),
        }
        for molecule_id, smiles, label, prediction, probability in zip(
            dataset.molecule_ids,
            dataset.smiles,
            dataset.labels,
            predicted,
            probabilities,
        )
    ]


def evaluate_model(
    model: Any,
    datasets: Mapping[str, SplitDataset],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    metrics: dict[str, Any] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_NAMES:
        dataset = datasets[split]
        probabilities = class_probability_matrix(model, dataset.features)
        metrics[split] = classification_metrics(dataset.labels, probabilities)
        predictions[split] = prediction_rows(dataset, probabilities)
    return metrics, predictions


def parse_int_grid(value: str) -> list[int]:
    parsed = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"Expected positive integer grid, received: {value!r}")
    return parsed


def parse_optional_int_grid(value: str) -> list[int | None]:
    parsed: list[int | None] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        candidate = None if token in {"none", "null", "unlimited"} else int(token)
        if candidate is not None and candidate <= 0:
            raise ValueError(f"max_depth must be positive or none: {value!r}")
        if candidate not in parsed:
            parsed.append(candidate)
    if not parsed:
        raise ValueError(f"Empty max-depth grid: {value!r}")
    return parsed


def _finite_for_sort(value: Any) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else -math.inf


def select_random_forest(
    train: SplitDataset,
    val: SplitDataset,
    *,
    n_estimators_grid: Sequence[int],
    max_depth_grid: Sequence[int | None],
    min_samples_leaf_grid: Sequence[int],
    random_seed: int,
    n_jobs: int,
    class_weight: str | None = "balanced_subsample",
    selection_metric: str = "balanced_accuracy",
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    require_dependencies()
    candidates: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    for n_estimators in n_estimators_grid:
        for max_depth in max_depth_grid:
            for min_samples_leaf in min_samples_leaf_grid:
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=max_depth,
                    min_samples_leaf=int(min_samples_leaf),
                    random_state=int(random_seed),
                    n_jobs=int(n_jobs),
                    class_weight=class_weight,
                )
                model.fit(train.features, train.labels)
                val_probabilities = class_probability_matrix(model, val.features)
                val_metrics = classification_metrics(val.labels, val_probabilities)
                parameters = {
                    "n_estimators": int(n_estimators),
                    "max_depth": max_depth,
                    "min_samples_leaf": int(min_samples_leaf),
                    "random_state": int(random_seed),
                    "n_jobs": int(n_jobs),
                    "class_weight": class_weight,
                }
                candidates.append((model, parameters, val_metrics))
    if selection_metric not in {"balanced_accuracy", "auroc", "average_precision"}:
        raise ValueError(f"Unsupported RF selection metric: {selection_metric}")

    def ranking(item: tuple[Any, dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
        _model, parameters, metrics = item
        max_depth = parameters["max_depth"]
        return (
            -_finite_for_sort(metrics[selection_metric]),
            -_finite_for_sort(metrics["balanced_accuracy"]),
            -_finite_for_sort(metrics["auroc"]),
            -_finite_for_sort(metrics["average_precision"]),
            int(parameters["n_estimators"]),
            math.inf if max_depth is None else int(max_depth),
            int(parameters["min_samples_leaf"]),
        )

    selected_model, selected_parameters, selected_metrics = min(candidates, key=ranking)
    search_rows = [
        {
            "parameters": parameters,
            "val_metrics": metrics,
            "selected": parameters == selected_parameters,
        }
        for _model, parameters, metrics in candidates
    ]
    selected = {
        "selection_split": "val",
        "selection_metric": selection_metric,
        "selected_parameters": selected_parameters,
        "selected_val_metrics": selected_metrics,
        "num_candidates": len(candidates),
    }
    return selected_model, selected, search_rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(datasets: Mapping[str, SplitDataset]) -> dict[str, Any]:
    return {
        "dataset": "Mutagenicity",
        "dataset_version": "v1",
        "source_label": 1,
        "target_label": 0,
        "splits": {
            split: {
                "path": str(datasets[split].path),
                "sha256": sha256_file(datasets[split].path),
                "role": SPLIT_ROLES[split],
                "num_examples": datasets[split].size,
                "label_counts": {
                    "0": int(np.sum(datasets[split].labels == 0)),
                    "1": int(np.sum(datasets[split].labels == 1)),
                },
            }
            for split in SPLIT_NAMES
        },
        "fit_splits": ["train"],
        "selection_splits": ["val"],
        "calibration_used_for_fit_or_selection": False,
        "test_used_for_fit_or_selection": False,
    }


def build_bundle(
    model: Any,
    *,
    feature_config: FeatureConfig,
    selection: Mapping[str, Any],
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "task_name": "mutagenicity_binary",
        "dataset_name": "Mutagenicity",
        "dataset_version": "v1",
        "feature_type": "rdkit_morgan",
        "fingerprint_radius": int(feature_config.radius),
        "fingerprint_bits": int(feature_config.n_bits),
        "positive_label": 1,
        "negative_label": 0,
        "source_label": 1,
        "target_label": 0,
        "class_labels": [int(label) for label in model.classes_],
        "fit_split_names": ["train"],
        "selection_split_names": ["val"],
        "calibration_used_for_fit_or_selection": False,
        "test_used_for_fit_or_selection": False,
        "selection": dict(selection),
        "metrics": dict(metrics),
        "split_manifest": dict(manifest),
    }


def atomic_pickle(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_bundle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    required = {"model", "fingerprint_radius", "fingerprint_bits"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid Mutagenicity RF bundle: {path}")
    if not hasattr(bundle["model"], "predict_proba"):
        raise ValueError("Mutagenicity RF bundle model lacks predict_proba")
    return bundle


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: int(value) if isinstance(value := row.get(field), bool) else value
                    for field in fieldnames
                }
            )


def write_evaluation_artifacts(
    output_dir: str | Path,
    *,
    metrics: Mapping[str, Any],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: Mapping[str, Any],
    model_path: str | Path,
    selection: Mapping[str, Any] | None = None,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "dataset": "Mutagenicity",
        "label_semantics": {"0": "non_mutagenic", "1": "mutagenic"},
        "source_label": 1,
        "target_label": 0,
        "model_path": str(model_path),
        "model_selection": dict(selection or {}),
        "splits": dict(metrics),
    }
    write_json(root / "metrics.json", metrics_payload)
    write_json(root / "split_manifest.json", manifest)
    confusion_rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        write_csv(
            root / f"predictions_{split}.csv",
            predictions[split],
            PREDICTION_FIELDS,
        )
        matrix = metrics[split]["confusion_matrix"]
        for true_label in (0, 1):
            confusion_rows.append(
                {
                    "split": split,
                    "true_label": true_label,
                    "predicted_0": int(matrix[true_label][0]),
                    "predicted_1": int(matrix[true_label][1]),
                }
            )
    write_csv(
        root / "confusion_matrix.csv",
        confusion_rows,
        ("split", "true_label", "predicted_0", "predicted_1"),
    )
    report_lines = [
        "# Mutagenicity RF Teacher Report",
        "",
        "- Label 1: mutagenic",
        "- Label 0: non-mutagenic",
        "- Main recourse direction: 1 -> 0",
        "- Fit split: train only",
        "- Selection split: validation only",
        "- Calibration and test are not used for fitting or model selection.",
        "",
        "## Split metrics",
        "",
        "| Split | Accuracy | Balanced accuracy | AUROC | Average precision | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLIT_NAMES:
        values = metrics[split]
        report_lines.append(
            f"| {split} | {values['accuracy']:.6f} | "
            f"{values['balanced_accuracy']:.6f} | {values['auroc']:.6f} | "
            f"{values['average_precision']:.6f} | {values['brier_score']:.6f} |"
        )
    report_lines.extend(("", "## Per-class metrics", ""))
    for split in SPLIT_NAMES:
        report_lines.append(f"### {split}")
        report_lines.append("")
        report_lines.append("| Label | Precision | Recall | F1 | Support |")
        report_lines.append("|---:|---:|---:|---:|---:|")
        for label in (0, 1):
            values = metrics[split]["per_class"][str(label)]
            report_lines.append(
                f"| {label} | {values['precision']:.6f} | {values['recall']:.6f} | "
                f"{values['f1']:.6f} | {values['support']} |"
            )
        report_lines.append("")
    (root / "teacher_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


__all__ = [
    "FeatureConfig",
    "PREDICTION_FIELDS",
    "SPLIT_NAMES",
    "SplitDataset",
    "atomic_pickle",
    "build_bundle",
    "class_probability_matrix",
    "classification_metrics",
    "evaluate_model",
    "load_all_splits",
    "load_bundle",
    "load_split_dataset",
    "parse_int_grid",
    "parse_optional_int_grid",
    "prediction_rows",
    "select_random_forest",
    "split_manifest",
    "write_evaluation_artifacts",
    "write_json",
]
