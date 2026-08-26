from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.molecular_graph_dataset import (
    MolecularGraphData,
    MolecularGraphDataset,
    collate_molecular_graphs,
)
from src.data.molecular_graph_featurizer import default_molecular_feature_schema
from src.models.gnn_backbone_registry import available_gnn_backbones, get_gnn_backbone_spec
from src.models.molecular_gnn import build_molecular_gnn
from src.oracles.gnn_oracle import (
    GNNOracle,
    fit_temperature_scaling,
    load_gnn_checkpoint_bundle,
    save_gnn_checkpoint_bundle,
    verify_checkpoint_bundle,
)
from src.oracles.oracle_factory import build_oracle
from scripts.train_molecular_gnn import _classifier_health_gate, _selection_improves


def _graph(name: str, label: int, *, atom_offset: int = 0) -> MolecularGraphData:
    return MolecularGraphData(
        x=(
            (6 + atom_offset, 1, 5, 0, 3, 3, 0, 0),
            (8, 1, 5, 0, 1, 2, 0, 0),
        ),
        edge_index=((0, 1), (1, 0)),
        edge_attr=((0, 0, 0, 0), (0, 0, 0, 0)),
        y=label,
        molecule_id=name,
        smiles="CO",
        split="val",
        graph_sha256=f"hash-{name}",
    )


def _model(backbone: str = "gine"):
    torch.manual_seed(7)
    schema = default_molecular_feature_schema()
    model = build_molecular_gnn(
        backbone=backbone,
        num_classes=2,
        node_feature_schema=schema,
        edge_feature_schema=schema,
        num_layers=2,
        hidden_dim=16,
        dropout=0.0,
        pooling="mean",
        readout_layers=2,
        normalization="layer_norm",
        residual=True,
    )
    model.eval()
    return model, schema


def test_stratified_smoke_limit_covers_grouped_labels(tmp_path: Path) -> None:
    split = tmp_path / "train.csv"
    split.write_text(
        "molecule_id,smiles,label,split\n"
        + "".join(
            f"p{index},{'C' * (index + 2)},{1},train\n" for index in range(80)
        )
        + "".join(
            f"n{index},N{'C' * (index + 1)},{0},train\n" for index in range(20)
        ),
        encoding="utf-8",
    )
    dataset = MolecularGraphDataset.from_csv(
        split,
        num_classes=2,
        expected_split="train",
        limit=64,
        stratified_limit=True,
    )
    assert len(dataset) == 64
    assert set(dataset.labels) == {0, 1}
    assert dataset.labels.count(0) == 20
    assert dataset.labels.count(1) == 44


def test_full_classifier_health_gate_fails_single_class_predictions() -> None:
    gate = _classifier_health_gate(
        metrics={
            "roc_auc": 0.71,
            "per_class": {
                "0": {"recall": 1.0},
                "1": {"recall": 0.0},
            },
        },
        probabilities=np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float64),
        source_label=1,
        profile="full",
        training_config={
            "health_gate": {
                "enabled": True,
                "apply_profile": "full",
                "primary_metric": "roc_auc",
                "minimum_primary_metric": 0.65,
                "require_multiple_predicted_classes": True,
                "require_source_class_recall": True,
                "require_finite": True,
            }
        },
    )
    assert gate["status"] == "FAIL"
    assert "validation_predictions_are_single_class" in gate["failures"]
    assert "source_class_recall_is_not_positive" in gate["failures"]


def test_three_class_health_gate_requires_every_class_recall() -> None:
    gate = _classifier_health_gate(
        metrics={
            "macro_ovr_roc_auc": 0.72,
            "per_class": {
                "0": {"recall": 0.5},
                "1": {"recall": 0.25},
                "2": {"recall": 0.0},
            },
        },
        probabilities=np.asarray(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]],
            dtype=np.float64,
        ),
        source_label=1,
        profile="full",
        training_config={
            "health_gate": {
                "enabled": True,
                "apply_profile": "full",
                "primary_metric": "macro_ovr_roc_auc",
                "minimum_primary_metric": 0.0,
                "require_multiple_predicted_classes": True,
                "require_source_class_recall": True,
                "require_all_class_recall": True,
                "require_finite": True,
            }
        },
    )
    assert gate["status"] == "FAIL"
    assert gate["failures"] == ["class_2_recall_is_not_positive"]


def test_validation_selection_uses_macro_f1_only_for_primary_ties() -> None:
    assert _selection_improves(
        primary=0.8,
        tiebreak=0.4,
        best_primary=0.7,
        best_tiebreak=0.9,
    )
    assert _selection_improves(
        primary=0.8,
        tiebreak=0.6,
        best_primary=0.8,
        best_tiebreak=0.5,
    )
    assert not _selection_improves(
        primary=0.8,
        tiebreak=0.4,
        best_primary=0.8,
        best_tiebreak=0.5,
    )
    assert not _selection_improves(
        primary=0.79,
        tiebreak=1.0,
        best_primary=0.8,
        best_tiebreak=0.5,
    )


@pytest.mark.parametrize("backbone", ["gine", "gin", "gcn", "gatv2"])
def test_registered_backbones_share_input_and_output_contract(backbone: str) -> None:
    assert set(available_gnn_backbones()) == {"gine", "gin", "gcn", "gatv2"}
    assert "edge" in get_gnn_backbone_spec(backbone).edge_feature_mode
    model, _schema = _model(backbone)
    batch = collate_molecular_graphs([_graph("a", 0), _graph("b", 1)], edge_feature_dim=4)
    logits = model(batch)
    assert tuple(logits.shape) == (2, 2)
    assert torch.isfinite(logits).all()


def test_gnn_oracle_batched_and_single_predictions_are_equivalent() -> None:
    model, _schema = _model()
    oracle = GNNOracle(
        model,
        device="cpu",
        checkpoint_id="unit-checkpoint",
        backbone="gine",
        num_classes=2,
        source_label=1,
        temperature=1.7,
        edge_feature_dim=4,
        default_batch_size=8,
    )
    graphs = [_graph("a", 0), _graph("b", 1, atom_offset=1)]
    batched = oracle.predict_logits(graphs)
    singles = np.vstack([oracle.predict_logits([graph]) for graph in graphs])
    np.testing.assert_allclose(batched, singles, rtol=0.0, atol=1e-7)
    records = oracle.predict_records(graphs)
    assert len(records) == 2
    assert all(len(record["probabilities"]) == 2 for record in records)
    assert all(record["checkpoint_id"] == "unit-checkpoint" for record in records)
    assert all(record["temperature"] == pytest.approx(1.7) for record in records)


def test_checkpoint_bundle_roundtrip_and_hash_gate(tmp_path: Path) -> None:
    model, schema = _model()
    checkpoint = tmp_path / "checkpoint"
    result = save_gnn_checkpoint_bundle(
        model=model,
        checkpoint_dir=checkpoint,
        feature_schema=schema,
        config={"gnn": model.config.to_dict()},
        model_card={
            "dataset": "bace",
            "source_label": 1,
            "seed": 7,
            "training_commit": "unit",
            "best_epoch": 1,
            "selection_metric": "macro_f1",
        },
        label_map={0: "Inactive", 1: "Active"},
        split_manifest={"test_used_for_checkpoint_selection": False},
        training_metrics={"best_epoch": 1},
        test_evaluation_status={
            "status": "NOT_EVALUATED",
            "test_loaded": False,
            "reason": "held_out_until_frozen_final_evaluation",
            "path": "/frozen/test.csv",
            "sha256": "a" * 64,
        },
        validation_predictions=[
            {
                "molecule_id": "a",
                "label": 0,
                "predicted_label": 0,
                "logits": "[1.0, 0.0]",
                "probabilities": "[0.73, 0.27]",
            }
        ],
        environment={"python": "unit"},
        git_state={"commit": "unit"},
    )
    assert result["checkpoint_id"]
    audit = verify_checkpoint_bundle(checkpoint)
    assert audit["model_card"]["classifier_type"] == "gnn"
    assert audit["model_card"]["rf_oracle_used"] is False
    assert (checkpoint / "test_evaluation_status.json").is_file()
    assert not (checkpoint / "test_predictions.csv").exists()
    loaded, metadata = load_gnn_checkpoint_bundle(checkpoint, device="cpu")
    batch = collate_molecular_graphs([_graph("a", 0)], edge_feature_dim=4)
    with torch.no_grad():
        np.testing.assert_allclose(
            model(batch).numpy(), loaded(batch).numpy(), rtol=0.0, atol=0.0
        )
    assert metadata["checkpoint_id"] == result["checkpoint_id"]
    assert metadata["test_evaluation_status"]["status"] == "NOT_EVALUATED"
    (checkpoint / "model_card.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_checkpoint_bundle(checkpoint)


def test_temperature_scaling_is_validation_only_and_argmax_invariant() -> None:
    logits = np.asarray(
        [[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 3.0], [2.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    result = fit_temperature_scaling(logits, [0, 1, 2, 1], max_iter=25)
    assert result["selection_split"] == "validation"
    assert result["test_used_for_fit"] is False
    assert result["temperature"] > 0.0
    assert result["argmax_invariant"] is True


@pytest.mark.parametrize("dataset", ["bace", "TasteMolNet", "taste"])
def test_factory_rejects_rf_for_gnn_only_datasets(tmp_path: Path, dataset: str) -> None:
    with pytest.raises(ValueError, match="prohibited"):
        build_oracle(
            dataset=dataset,
            backend="rf",
            checkpoint=tmp_path / "does-not-exist.pkl",
        )


def test_checkpoint_model_card_contains_formal_gnn_provenance(tmp_path: Path) -> None:
    model, schema = _model()
    checkpoint = tmp_path / "checkpoint"
    save_gnn_checkpoint_bundle(
        model=model,
        checkpoint_dir=checkpoint,
        feature_schema=schema,
        config={"gnn": model.config.to_dict()},
        model_card={"dataset": "bace", "source_label": 1},
        label_map={0: "Inactive", 1: "Active"},
        split_manifest={},
        training_metrics={},
        test_evaluation_status={
            "status": "NOT_EVALUATED",
            "test_loaded": False,
            "reason": "held_out_until_frozen_final_evaluation",
            "path": "/frozen/test.csv",
            "sha256": "a" * 64,
        },
    )
    card = json.loads((checkpoint / "model_card.json").read_text(encoding="utf-8"))
    assert card["oracle_backend"] == "gnn"
    assert card["classifier_type"] == "gnn"
    assert card["rf_oracle_used"] is False


def test_tastemolnet_full_bundle_requires_scoped_policy_and_cache_closure(
    tmp_path: Path,
) -> None:
    model, schema = _model()
    with pytest.raises(ValueError, match="missing scoped policy/cache closure"):
        save_gnn_checkpoint_bundle(
            model=model,
            checkpoint_dir=tmp_path / "taste-full",
            feature_schema=schema,
            config={"gnn": model.config.to_dict()},
            model_card={
                "dataset": "tastemolnet",
                "source_label": 1,
                "profile": "full",
            },
            label_map={0: "Bitter", 1: "Sweet"},
            split_manifest={
                "files": {"test": {"path": "/frozen/test.csv", "sha256": "a" * 64}}
            },
            training_metrics={},
            test_evaluation_status={
                "status": "NOT_EVALUATED",
                "test_loaded": False,
                "reason": "held_out_until_frozen_final_evaluation",
                "path": "/frozen/test.csv",
                "sha256": "a" * 64,
            },
        )
