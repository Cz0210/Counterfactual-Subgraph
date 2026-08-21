#!/usr/bin/env python3
"""Train and freeze one task-specific molecular GNN classifier."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_registry import get_dataset_spec, normalize_dataset_id  # noqa: E402
from src.data.molecular_graph_dataset import (  # noqa: E402
    MolecularGraphData,
    MolecularGraphDataset,
    build_molecular_data_loader,
)
from src.data.molecular_graph_featurizer import (  # noqa: E402
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)
from src.chem.hard_deletion import enumerate_connected_hard_deletions  # noqa: E402
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig  # noqa: E402
from src.oracles.gnn_oracle import (  # noqa: E402
    GNNOracle,
    classification_metrics,
    save_gnn_checkpoint_bundle,
    sha256_file,
)
from src.utils.env import (  # noqa: E402
    apply_dotlist_overrides,
    load_and_merge_config_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Merge config files in order; pass hpc.yaml before a GNN config.",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--train-csv")
    parser.add_argument("--validation-csv")
    parser.add_argument("--calibration-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--device")
    parser.add_argument("--backbone")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--source-label", type=int)
    parser.add_argument("--label-map-json")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL runtime dependency.
        raise RuntimeError("train_molecular_gnn.py requires PyTorch.") from exc
    return torch


def _config(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(path) for path in args.config]
    if not paths:
        paths = [PROJECT_ROOT / "configs" / "gnn" / "gine.yaml"]
    config = load_and_merge_config_files(paths)
    return apply_dotlist_overrides(config, args.set)


def _nested(config: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    return value.get(key, default) if isinstance(value, Mapping) else default


def _first_existing(root: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of the required split files exist: {names} under {root}")


def _resolve_split_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Molecular GNN data directory does not exist: {root}")

    def explicit_or(names: Sequence[str], explicit: str | None) -> Path:
        if explicit:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        return _first_existing(root, names)

    return {
        "train": explicit_or(("train.csv",), args.train_csv),
        "validation": explicit_or(("validation.csv", "val.csv", "valid.csv"), args.validation_csv),
        "calibration": explicit_or(("calibration.csv",), args.calibration_csv),
        "test": explicit_or(("test.csv",), args.test_csv),
    }


def _set_seed(seed: int) -> None:
    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(requested: str | None, config: Mapping[str, Any]) -> str:
    torch = _require_torch()
    value = requested or str(_nested(config, "runtime", "device", "auto"))
    if value in {"auto", "cuda"}:
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return str(value)


def _class_weights(labels: Sequence[int], num_classes: int) -> list[float]:
    counts = Counter(int(label) for label in labels)
    if set(counts) != set(range(num_classes)):
        raise ValueError(
            f"Training split must contain every class: counts={dict(sorted(counts.items()))}"
        )
    total = len(labels)
    return [total / (num_classes * counts[label]) for label in range(num_classes)]


def _prediction_rows(
    dataset: MolecularGraphDataset,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, row_logits, row_probabilities in zip(
        dataset.records, logits, probabilities, strict=True
    ):
        rows.append(
            {
                "molecule_id": record.molecule_id,
                "smiles": record.graph.canonical_smiles,
                "split": record.split,
                "label": int(record.label),
                "predicted_label": int(row_probabilities.argmax()),
                "logits": json.dumps([float(value) for value in row_logits]),
                "probabilities": json.dumps(
                    [float(value) for value in row_probabilities]
                ),
                "source_graph_hash": record.graph.graph_sha256,
            }
        )
    return rows


def _numeric_values(payload: Any) -> list[float]:
    values: list[float] = []
    if isinstance(payload, Mapping):
        for value in payload.values():
            values.extend(_numeric_values(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            values.extend(_numeric_values(value))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        values.append(float(payload))
    return values


def _classifier_health_gate(
    *,
    metrics: Mapping[str, Any],
    probabilities: np.ndarray,
    source_label: int,
    profile: str,
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    raw_config = training_config.get("health_gate", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("training.health_gate must be a mapping.")
    enabled = bool(raw_config.get("enabled", False))
    apply_profile = str(raw_config.get("apply_profile", "full")).strip().lower()
    if not enabled or (apply_profile and profile != apply_profile):
        return {
            "status": "NOT_APPLIED",
            "profile": profile,
            "apply_profile": apply_profile,
            "failures": [],
        }

    failures: list[str] = []
    primary_metric = str(raw_config.get("primary_metric", "roc_auc"))
    minimum = float(raw_config.get("minimum_primary_metric", 0.0))
    observed = metrics.get(primary_metric)
    if observed is None or not math.isfinite(float(observed)):
        failures.append(f"{primary_metric}=unavailable_or_nonfinite")
    elif float(observed) < minimum:
        failures.append(
            f"{primary_metric}={float(observed):.6f}<minimum={minimum:.6f}"
        )

    predictions = np.asarray(probabilities, dtype=np.float64).argmax(axis=1)
    if bool(raw_config.get("require_multiple_predicted_classes", True)) and len(
        np.unique(predictions)
    ) < 2:
        failures.append("validation_predictions_are_single_class")
    if bool(raw_config.get("require_source_class_recall", True)):
        source_metrics = metrics.get("per_class", {}).get(str(int(source_label)), {})
        recall = source_metrics.get("recall") if isinstance(source_metrics, Mapping) else None
        if recall is None or not math.isfinite(float(recall)) or float(recall) <= 0.0:
            failures.append("source_class_recall_is_not_positive")
    if bool(raw_config.get("require_finite", True)):
        if not np.isfinite(probabilities).all() or not all(
            math.isfinite(value) for value in _numeric_values(metrics)
        ):
            failures.append("nonfinite_validation_output")

    return {
        "status": "PASS" if not failures else "FAIL",
        "profile": profile,
        "apply_profile": apply_profile,
        "primary_metric": primary_metric,
        "minimum_primary_metric": minimum,
        "observed_primary_metric": observed,
        "predicted_classes": sorted(int(value) for value in np.unique(predictions)),
        "failures": failures,
    }


def _reload_oracle_smoke(
    checkpoint_dir: Path,
    dataset: MolecularGraphDataset,
    *,
    device: str,
) -> dict[str, Any]:
    """Exercise the persisted bundle and calibrated-probability API once."""

    oracle = GNNOracle.from_checkpoint(
        checkpoint_dir,
        device=device,
        batch_size=min(8, len(dataset)),
    )
    graphs = [dataset[index] for index in range(min(8, len(dataset)))]
    batched = oracle.predict_proba(graphs)
    singles = np.vstack([oracle.predict_proba([graph]) for graph in graphs])
    if not np.isfinite(batched).all() or not np.allclose(
        batched, singles, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            "Reloaded GNN oracle failed finite batch/single probability equivalence."
        )
    records = oracle.predict_records(graphs)
    if len(records) != len(graphs) or any(
        len(record["probabilities"]) != oracle.num_classes for record in records
    ):
        raise RuntimeError("Reloaded GNN oracle prediction-record contract failed.")

    deletion = next(
        (
            outcome
            for outcome in enumerate_connected_hard_deletions("CCO", "O")
            if outcome.valid and outcome.residual_smiles
        ),
        None,
    )
    empty_deletion = enumerate_connected_hard_deletions("CC", "CC")
    invalid_deletion = enumerate_connected_hard_deletions("CC", "not-a-smiles")
    if deletion is None:
        raise RuntimeError("Hard-deletion smoke did not produce a connected residual.")
    if not empty_deletion or any(outcome.valid for outcome in empty_deletion):
        raise RuntimeError("Empty-residual deletion did not fail closed.")
    if invalid_deletion:
        raise RuntimeError("Invalid-fragment deletion did not fail closed.")

    featurizer = MolecularGraphFeaturizer(dataset.feature_schema)

    def graph_from_smiles(smiles: str, molecule_id: str) -> MolecularGraphData:
        features = featurizer.featurize(smiles)
        return MolecularGraphData(
            x=features.node_features,
            edge_index=features.edge_index,
            edge_attr=features.edge_features,
            y=-1,
            molecule_id=molecule_id,
            smiles=features.canonical_smiles,
            split="smoke",
            graph_sha256=features.graph_sha256,
        )

    deletion_records = oracle.predict_records(
        [
            graph_from_smiles("CCO", "deletion-parent"),
            graph_from_smiles(str(deletion.residual_smiles), "deletion-residual"),
        ]
    )
    pred_before = int(deletion_records[0]["predicted_label"])
    pred_after = int(deletion_records[1]["predicted_label"])
    source_probability_before = float(deletion_records[0]["source_probability"])
    source_probability_after = float(deletion_records[1]["source_probability"])
    return {
        "checkpoint_id": oracle.checkpoint_id,
        "num_examples": len(graphs),
        "batch_single_max_abs_difference": float(np.max(np.abs(batched - singles))),
        "temperature": oracle.temperature,
        "deletion_valid": True,
        "deletion_residual_smiles": deletion.residual_smiles,
        "pred_before": pred_before,
        "pred_after": pred_after,
        "cf_flip": pred_before != pred_after,
        "cf_drop": source_probability_before - source_probability_after,
        "empty_deletion_failed_closed": True,
        "invalid_deletion_failed_closed": True,
    }


def _evaluate(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: str,
    num_classes: int,
) -> dict[str, Any]:
    torch = _require_torch()
    model.eval()
    logits_parts: list[Any] = []
    labels_parts: list[Any] = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y.long())
            count = int(batch.y.numel())
            total_loss += float(loss.item()) * count
            total_examples += count
            logits_parts.append(logits.detach().cpu())
            labels_parts.append(batch.y.detach().cpu())
    if not logits_parts:
        raise RuntimeError("Molecular GNN evaluation loader produced no examples.")
    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(labels_parts, dim=0)
    probabilities = torch.softmax(logits, dim=1)
    metrics = classification_metrics(
        labels.numpy(), probabilities.numpy(), num_classes=num_classes
    )
    metrics["loss"] = total_loss / total_examples
    return {
        "metrics": metrics,
        "logits": logits.numpy().astype(np.float64),
        "probabilities": probabilities.numpy().astype(np.float64),
        "labels": labels.numpy().astype(np.int64),
    }


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_short": command("status", "--short").splitlines(),
    }


def _environment(device: str) -> dict[str, Any]:
    torch = _require_torch()
    try:
        import rdkit
    except ImportError:  # pragma: no cover
        rdkit_version = None
    else:
        rdkit_version = getattr(rdkit, "__version__", None)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": device,
        "rdkit": rdkit_version,
    }


def _label_map(args: argparse.Namespace, spec: Any) -> dict[str, str]:
    if not args.label_map_json:
        return {str(key): value for key, value in spec.label_map.items()}
    candidate = Path(args.label_map_json).expanduser()
    payload = (
        json.loads(candidate.read_text(encoding="utf-8"))
        if candidate.is_file()
        else json.loads(args.label_map_json)
    )
    return {str(int(key)): str(value) for key, value in payload.items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch = _require_torch()
    config = _config(args)
    dataset_id = normalize_dataset_id(args.dataset, allow_historical=False)
    spec = get_dataset_spec(dataset_id, allow_historical=False)
    num_classes = int(args.num_classes or spec.num_classes)
    source_label = spec.source_label if args.source_label is None else int(args.source_label)
    if num_classes != spec.num_classes or source_label != spec.source_label:
        raise ValueError("CLI class semantics conflict with the active dataset registry.")
    split_paths = _resolve_split_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Molecular GNN output must be fresh: {output_dir}")

    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, Mapping):
        raise ValueError("training config must be a mapping.")
    seed = int(args.seed if args.seed is not None else training_cfg.get("primary_seed", 7))
    _set_seed(seed)
    device = _resolve_device(args.device, config)
    profile = args.profile
    smoke = profile == "smoke"
    max_epochs = int(
        args.max_epochs
        if args.max_epochs is not None
        else min(int(training_cfg.get("max_epochs", 200)), 2) if smoke
        else int(training_cfg.get("max_epochs", 200))
    )
    patience = int(
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else training_cfg.get("early_stopping_patience", 20)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 64)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else training_cfg.get("learning_rate", 0.001)
    )
    weight_decay = float(
        args.weight_decay
        if args.weight_decay is not None
        else training_cfg.get("weight_decay", 0.00001)
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else _nested(config, "runtime", "num_workers", 0)
    )
    train_limit = args.train_limit if args.train_limit is not None else (64 if smoke else None)
    validation_limit = (
        args.validation_limit if args.validation_limit is not None else (32 if smoke else None)
    )
    test_limit = args.test_limit if args.test_limit is not None else (32 if smoke else None)
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("Epoch, patience, and batch size values must be positive.")
    class_weighted = bool(training_cfg.get("class_weighted_loss", True))
    weighted_sampler = bool(training_cfg.get("weighted_sampler", False))
    if class_weighted and weighted_sampler:
        raise ValueError(
            "class_weighted_loss and weighted_sampler are mutually exclusive."
        )

    schema = default_molecular_feature_schema()
    featurizer = MolecularGraphFeaturizer(schema)
    train_dataset = MolecularGraphDataset.from_csv(
        split_paths["train"],
        num_classes=num_classes,
        featurizer=featurizer,
        expected_split="train",
        limit=train_limit,
        stratified_limit=smoke,
    )
    validation_dataset = MolecularGraphDataset.from_csv(
        split_paths["validation"],
        num_classes=num_classes,
        featurizer=featurizer,
        expected_split="val",
        limit=validation_limit,
        stratified_limit=smoke,
    )
    # Test is loaded only after model-selection inputs are fixed; it is never
    # consulted by the early-stopping loop below.
    weights = _class_weights(train_dataset.labels, num_classes)
    sampler = None
    if weighted_sampler:
        sample_weights = torch.tensor(
            [weights[label] for label in train_dataset.labels], dtype=torch.double
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
    train_loader = build_molecular_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
    )
    validation_loader = build_molecular_data_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    gnn_values = dict(config.get("gnn", {}))
    if args.backbone:
        gnn_values["backbone"] = args.backbone
    gnn_values["num_classes"] = num_classes
    model_config = MolecularGNNConfig.from_mapping(gnn_values)
    model = MolecularGNN(
        model_config,
        node_cardinalities=schema.node_cardinalities,
        edge_cardinalities=schema.edge_cardinalities,
    ).to(device)
    criterion_weights = (
        torch.tensor(weights, dtype=torch.float32, device=device)
        if class_weighted
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=criterion_weights)
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError("The frozen molecular GNN route currently requires optimizer=adamw.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    clip_norm = float(training_cfg.get("gradient_clip_norm", 5.0))
    selection_metric = str(training_cfg.get("selection_metric", "macro_f1"))
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_value = -math.inf
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            count = int(batch.y.numel())
            total_loss += float(loss.item()) * count
            total_examples += count
        validation = _evaluate(
            model,
            validation_loader,
            criterion,
            device=device,
            num_classes=num_classes,
        )
        metric_value = validation["metrics"].get(selection_metric)
        if metric_value is None:
            raise ValueError(
                f"Selection metric {selection_metric!r} is unavailable on validation."
            )
        epoch_row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_examples),
            "validation": validation["metrics"],
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
        if float(metric_value) > best_value + 1e-12:
            best_value = float(metric_value)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Molecular GNN training did not produce a best checkpoint.")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    final_validation = _evaluate(
        model,
        validation_loader,
        criterion,
        device=device,
        num_classes=num_classes,
    )

    test_dataset = MolecularGraphDataset.from_csv(
        split_paths["test"],
        num_classes=num_classes,
        featurizer=featurizer,
        expected_split="test",
        limit=test_limit,
        stratified_limit=smoke,
    )
    test_loader = build_molecular_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    frozen_test = _evaluate(
        model, test_loader, criterion, device=device, num_classes=num_classes
    )
    health_gate = _classifier_health_gate(
        metrics=final_validation["metrics"],
        probabilities=final_validation["probabilities"],
        source_label=source_label,
        profile=profile,
        training_config=training_cfg,
    )
    split_manifest = {
        "schema_version": "molecular_gnn_split_manifest_v1",
        "dataset": dataset_id,
        "roles": {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        },
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in split_paths.items()
        },
        "train_manifest": train_dataset.manifest(),
        "validation_manifest": validation_dataset.manifest(),
        "test_manifest": test_dataset.manifest(),
        "calibration_loaded_for_training": False,
        "test_used_for_checkpoint_selection": False,
    }
    training_metrics = {
        "schema_version": "molecular_gnn_training_metrics_v1",
        "profile": profile,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_validation_selection_value": best_value,
        "epochs_completed": len(history),
        "history": history,
        "final_validation": final_validation["metrics"],
        "frozen_test": frozen_test["metrics"],
        "class_weights": weights,
        "class_weighted_loss": class_weighted,
        "weighted_sampler": weighted_sampler,
        "health_gate": health_gate,
    }
    resolved_config = copy.deepcopy(config)
    resolved_config["gnn"] = model_config.to_dict()
    resolved_config["training"] = {
        **dict(training_cfg),
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "primary_seed": seed,
        "class_weighted_loss": class_weighted,
        "weighted_sampler": weighted_sampler,
    }
    git_state = _git_state()
    model_card = {
        "dataset": dataset_id,
        "backbone": model_config.backbone,
        "num_classes": num_classes,
        "source_label": source_label,
        "seed": seed,
        "training_commit": git_state["commit"],
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "selection_split": "validation",
        "temperature_calibration_split": "validation",
        "calibration_used_for_model_fit_or_selection": False,
        "test_used_for_model_fit_or_selection": False,
        "profile": profile,
        "health_gate": health_gate,
    }
    bundle = save_gnn_checkpoint_bundle(
        model=model,
        checkpoint_dir=output_dir,
        feature_schema=schema,
        config=resolved_config,
        model_card=model_card,
        label_map=_label_map(args, spec),
        split_manifest=split_manifest,
        training_metrics=training_metrics,
        validation_predictions=_prediction_rows(
            validation_dataset,
            final_validation["logits"],
            final_validation["probabilities"],
        ),
        test_predictions=_prediction_rows(
            test_dataset, frozen_test["logits"], frozen_test["probabilities"]
        ),
        environment=_environment(device),
        git_state=git_state,
    )
    if smoke:
        reload_smoke = _reload_oracle_smoke(
            output_dir, validation_dataset, device=device
        )
        print(json.dumps({"oracle_reload_smoke": reload_smoke}, sort_keys=True), flush=True)
    print(json.dumps(bundle, sort_keys=True), flush=True)
    if health_gate["status"] == "FAIL":
        print(json.dumps({"health_gate": health_gate}, sort_keys=True), flush=True)
        if dataset_id == "bace":
            print("[BACE_GNN_HEALTH_GATE_FAILED]", flush=True)
        return 3
    if dataset_id == "bace" and profile == "smoke":
        print("[BACE_GNN_SMOKE_PASS]", flush=True)
    if dataset_id == "bace" and profile == "full":
        print("[BACE_GNN_TRAIN_PASS]", flush=True)
    print("[MOLECULAR_GNN_TRAIN_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
