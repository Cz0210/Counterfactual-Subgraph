#!/usr/bin/env python3
"""Evaluate one frozen molecular GNN without fitting or selecting anything."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_registry import normalize_dataset_id  # noqa: E402
from src.data.molecular_graph_dataset import MolecularGraphDataset  # noqa: E402
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer  # noqa: E402
from src.oracles.gnn_oracle import (  # noqa: E402
    GNNOracle,
    classification_metrics,
    load_gnn_checkpoint_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=("train", "validation", "val", "calibration", "test"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"GNN evaluation output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _model, metadata = load_gnn_checkpoint_bundle(
        args.checkpoint_dir, device=args.device
    )
    card = metadata["model_card"]
    dataset_id = normalize_dataset_id(args.dataset or str(card["dataset"]))
    if dataset_id != normalize_dataset_id(str(card["dataset"])):
        raise ValueError("Evaluation dataset conflicts with the frozen model card.")
    expected_split = "val" if args.split in {"val", "validation"} else args.split
    featurizer = MolecularGraphFeaturizer(metadata["feature_schema"])
    dataset = MolecularGraphDataset.from_csv(
        args.dataset_csv,
        num_classes=int(card["num_classes"]),
        featurizer=featurizer,
        expected_split=expected_split,
        limit=args.limit,
    )
    oracle = GNNOracle.from_checkpoint(
        args.checkpoint_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    records = oracle.predict_records(dataset, batch_size=args.batch_size)
    probabilities = np.asarray(
        [record["probabilities"] for record in records], dtype=np.float64
    )
    labels = np.asarray(dataset.labels, dtype=np.int64)
    metrics = classification_metrics(
        labels, probabilities, num_classes=oracle.num_classes
    )
    rows: list[dict[str, Any]] = []
    for source, prediction in zip(dataset.records, records, strict=True):
        rows.append(
            {
                "molecule_id": source.molecule_id,
                "smiles": source.graph.canonical_smiles,
                "split": source.split,
                "label": int(source.label),
                "predicted_label": prediction["predicted_label"],
                "probabilities": json.dumps(prediction["probabilities"]),
                "logits": json.dumps(prediction["logits"]),
                "source_probability": prediction["source_probability"],
                "confidence": prediction["confidence"],
                "checkpoint_id": prediction["checkpoint_id"],
                "temperature": prediction["temperature"],
            }
        )
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_label = int(card["source_label"])
    correctly_predicted_source = [
        row["molecule_id"]
        for row in rows
        if int(row["label"]) == source_label
        and int(row["predicted_label"]) == source_label
    ]
    all_predicted_source = [
        row["molecule_id"]
        for row in rows
        if int(row["predicted_label"]) == source_label
    ]
    summary = {
        "schema_version": "molecular_gnn_evaluation_v1",
        "dataset": dataset_id,
        "split": args.split,
        "selection_performed": False,
        "temperature_fitted": False,
        "checkpoint_id": oracle.checkpoint_id,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "backbone": oracle.backbone,
        "num_classes": oracle.num_classes,
        "source_label": source_label,
        "temperature": oracle.temperature,
        "metrics": metrics,
        "main_source_cohort_definition": "true_label_equals_source_and_pred_before_equals_source",
        "correctly_predicted_source_cohort": correctly_predicted_source,
        "all_predicted_source_cohort": all_predicted_source,
    }
    _atomic_json(output / "metrics.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[MOLECULAR_GNN_EVALUATION_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
