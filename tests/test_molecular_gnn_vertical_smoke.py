from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.calibrate_gnn_classifier import main as calibrate_main
from scripts.evaluate_molecular_gnn import main as evaluate_main
from scripts.train_molecular_gnn import main as train_main
from src.data.molecular_graph_dataset import MolecularGraphDataset
from src.oracles.gnn_oracle import GNNOracle, sha256_file, verify_checkpoint_bundle


def _write_split(path: Path, split: str) -> None:
    smiles_by_label = {
        1: ("CC", "CCC", "CCCC", "CCCCC"),
        0: ("CN", "CCN", "CCCN", "CCCCN"),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("molecule_id", "smiles", "label", "split")
        )
        writer.writeheader()
        # Deliberately group labels to regress the real BACE prefix ordering.
        for label in (1, 0):
            for index, smiles in enumerate(smiles_by_label[label]):
                writer.writerow(
                    {
                        "molecule_id": f"{split}-{label}-{index}",
                        "smiles": smiles,
                        "label": label,
                        "split": split,
                    }
                )


def test_train_calibrate_evaluate_vertical_smoke(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "splits"
    data.mkdir()
    _write_split(data / "train.csv", "train")
    _write_split(data / "val.csv", "val")
    _write_split(data / "calibration.csv", "calibration")
    _write_split(data / "test.csv", "test")
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "gnn:\n"
        "  backbone: gine\n"
        "  num_layers: 1\n"
        "  hidden_dim: 16\n"
        "  dropout: 0.0\n"
        "  pooling: mean\n"
        "  readout_layers: 1\n"
        "  normalization: layer_norm\n"
        "  residual: true\n"
        "training:\n"
        "  optimizer: adamw\n"
        "  learning_rate: 0.001\n"
        "  weight_decay: 0.0\n"
        "  max_epochs: 1\n"
        "  early_stopping_patience: 1\n"
        "  batch_size: 4\n"
        "  primary_seed: 7\n"
        "  selection_metric: roc_auc\n"
        "  class_weighted_loss: true\n"
        "  weighted_sampler: false\n"
        "  gradient_clip_norm: 5.0\n"
        "runtime:\n"
        "  device: cpu\n"
        "  num_workers: 0\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    loaded_split_paths: list[Path] = []
    original_from_csv = MolecularGraphDataset.from_csv.__func__

    def tracked_from_csv(cls, csv_path, *args, **kwargs):
        resolved = Path(csv_path).resolve()
        loaded_split_paths.append(resolved)
        if resolved == (data / "test.csv").resolve():
            raise AssertionError("training must not load the held-out test split")
        return original_from_csv(cls, csv_path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            MolecularGraphDataset, "from_csv", classmethod(tracked_from_csv)
        )
        assert (
            train_main(
                [
                    "--config",
                    str(config),
                    "--dataset",
                    "bace",
                    "--data-dir",
                    str(data),
                    "--output-dir",
                    str(checkpoint),
                    "--profile",
                    "smoke",
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )
    assert loaded_split_paths == [
        (data / "train.csv").resolve(),
        (data / "val.csv").resolve(),
    ]
    assert not (checkpoint / "test_predictions.csv").exists()
    test_status = json.loads(
        (checkpoint / "test_evaluation_status.json").read_text(encoding="utf-8")
    )
    assert test_status == {
        "path": str((data / "test.csv").resolve()),
        "reason": "held_out_until_frozen_final_evaluation",
        "schema_version": "molecular_gnn_test_evaluation_status_v1",
        "sha256": sha256_file(data / "test.csv"),
        "status": "NOT_EVALUATED",
        "test_loaded": False,
    }
    assert len(test_status["sha256"]) == 64
    assert calibrate_main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--validation-csv",
            str(data / "val.csv"),
        ]
    ) == 0
    evaluation = tmp_path / "evaluation"
    assert evaluate_main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--dataset-csv",
            str(data / "test.csv"),
            "--dataset",
            "bace",
            "--split",
            "test",
            "--output-dir",
            str(evaluation),
            "--device",
            "cpu",
        ]
    ) == 0

    audit = verify_checkpoint_bundle(checkpoint)
    oracle = GNNOracle.from_checkpoint(checkpoint, device="cpu")
    temperature = json.loads(
        (checkpoint / "temperature_scaling.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    assert audit["model_card"]["oracle_backend"] == "gnn"
    assert temperature["status"] == "fit"
    assert oracle.temperature == temperature["temperature"]
    assert metrics["selection_performed"] is False
    assert metrics["temperature_fitted"] is False
