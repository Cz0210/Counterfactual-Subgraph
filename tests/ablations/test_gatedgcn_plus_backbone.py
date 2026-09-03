from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.gatedgcn_plus_backbone import (
    GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS,
    GATEDGCN_PLUS_OFFICIAL_COMMIT,
    GATEDGCN_PLUS_OFFICIAL_REPOSITORY,
    GATEDGCN_PLUS_LICENSE_SHA256,
    build_gatedgcn_plus_molecular_gnn,
    estimate_gatedgcn_plus_parameter_count,
    gatedgcn_plus_runtime_capabilities,
    match_gatedgcn_plus_hidden_dim,
)
from src.models.gnn_backbone_registry import (
    build_backbone,
    get_gnn_backbone_spec,
    normalize_gnn_backbone,
)
from src.models.graphgps_backbone import compute_topology_only_random_walk_pe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCH_RECEIPT = (
    PROJECT_ROOT
    / "configs/ablations/gnn/bace_gatedgcn_plus_parameter_match_v1.json"
)


def test_gatedgcn_plus_registry() -> None:
    assert normalize_gnn_backbone("GatedGCN+") == "gatedgcn_plus"
    spec = get_gnn_backbone_spec("gatedgcn_plus")
    assert spec.display_name == "GatedGCN+"
    assert "edge" in spec.edge_feature_mode
    assert "ffn" in spec.edge_feature_mode
    assert "rwpe" in spec.edge_feature_mode


def test_gatedgcn_plus_parameter_match() -> None:
    match = match_gatedgcn_plus_hidden_dim(1_432_583)
    assert match.allowed_hidden_dims == GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS
    assert match.selected_hidden_dim == 160
    assert match.selected_parameter_count == 1_219_138
    assert match.selected_relative_difference == pytest.approx(
        0.14899311244095456
    )
    assert match.validation_metrics_loaded is False
    assert match.test_metrics_loaded is False
    with pytest.raises(ValueError, match="exactly"):
        match_gatedgcn_plus_hidden_dim(
            1_432_583, allowed_hidden_dims=(128, 160)
        )


def test_gatedgcn_plus_receipt_matches_formula_and_actual_weights() -> None:
    pytest.importorskip("torch")
    from src.data.molecular_graph_featurizer import default_molecular_feature_schema

    receipt = json.loads(MATCH_RECEIPT.read_text(encoding="utf-8"))
    expected = {
        row["hidden_dim"]: row["parameter_count"]
        for row in receipt["candidates"]
    }
    assert {
        dim: estimate_gatedgcn_plus_parameter_count(dim)
        for dim in GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS
    } == expected
    schema = default_molecular_feature_schema()
    model = build_gatedgcn_plus_molecular_gnn(
        num_classes=2,
        node_feature_schema=schema,
        edge_feature_schema=schema,
        hidden_dim=160,
    )
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert actual == receipt["selected_parameter_count"] == 1_219_138
    assert receipt["official_commit"] == GATEDGCN_PLUS_OFFICIAL_COMMIT
    assert receipt["official_repository"] == (
        f"https://github.com/{GATEDGCN_PLUS_OFFICIAL_REPOSITORY}"
    )
    assert receipt["adapted_hyperparameters_not_official_bace_recipe"] is True
    source = (PROJECT_ROOT / "configs/ablations/gnn/gatedgcn_plus_source_v1.yaml").read_text(
        encoding="utf-8"
    )
    assert f"license_sha256: {GATEDGCN_PLUS_LICENSE_SHA256}" in source
    assert "adapted_hyperparameters_not_official_bace_recipe: true" in source


def _inputs() -> tuple[object, ...]:
    torch = pytest.importorskip("torch")
    from src.data.molecular_graph_featurizer import default_molecular_feature_schema

    schema = default_molecular_feature_schema()
    x = torch.zeros((3, len(schema.node_cardinalities)), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
    )
    edge_attr = torch.zeros(
        (4, len(schema.edge_cardinalities)), dtype=torch.long
    )
    batch = torch.zeros((3,), dtype=torch.long)
    rwpe = compute_topology_only_random_walk_pe(
        edge_index, num_nodes=3, walk_length=16
    )
    return schema, x, edge_index, edge_attr, batch, rwpe


def test_gatedgcn_plus_edge_features_residual_ffn() -> None:
    torch = pytest.importorskip("torch")
    schema, x, edge_index, edge_attr, batch, rwpe = _inputs()
    torch.manual_seed(7)
    model = build_gatedgcn_plus_molecular_gnn(
        num_classes=2,
        node_feature_schema=schema,
        edge_feature_schema=schema,
        hidden_dim=160,
    )
    assert model.config.residual is True
    assert model.config.ffn is True
    assert all(hasattr(layer, "ff_linear1") for layer in model.layers)
    model.eval()
    first = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        random_walk_pe=rwpe,
    )
    changed = edge_attr.clone()
    changed[:, 0] = 1
    second = model(
        x=x,
        edge_index=edge_index,
        edge_attr=changed,
        batch=batch,
        random_walk_pe=rwpe,
    )
    assert tuple(first.shape) == (1, 2)
    assert not torch.equal(first, second)
    first.sum().backward()
    assert model.edge_encoder.embeddings[0].weight.grad is not None
    assert model.layers[0].ff_linear1.weight.grad is not None


def test_gatedgcn_plus_rwpe() -> None:
    torch = pytest.importorskip("torch")
    schema, x, edge_index, edge_attr, batch, rwpe = _inputs()
    model = build_gatedgcn_plus_molecular_gnn(
        num_classes=2,
        node_feature_schema=schema,
        edge_feature_schema=schema,
        hidden_dim=160,
    )
    assert tuple(rwpe.shape) == (3, 16)
    with pytest.raises(ValueError, match="random_walk_pe"):
        model(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    with pytest.raises(TypeError):
        compute_topology_only_random_walk_pe(
            edge_index, num_nodes=3, walk_length=16, labels=[0, 1, 0]
        )


def test_registry_builds_gatedgcn_plus_and_runtime_is_dry_run_only() -> None:
    pytest.importorskip("torch")
    from src.data.molecular_graph_featurizer import default_molecular_feature_schema
    from src.utils.env import load_yaml_config

    schema = default_molecular_feature_schema()
    payload = load_yaml_config(PROJECT_ROOT / "configs/gnn/gatedgcn_plus.yaml")
    model = build_backbone(
        "gatedgcn_plus",
        payload,
        feature_schema=schema,
        expected_feature_schema_sha256=schema.to_dict()["schema_sha256"],
        num_classes=2,
    )
    assert model.config.backbone == "gatedgcn_plus"
    capabilities = gatedgcn_plus_runtime_capabilities()
    assert capabilities["model_build_pass"] is True
    assert capabilities["parameter_count_matches_receipt"] is True
    assert capabilities["validation_metrics_loaded"] is False
    assert capabilities["test_metrics_loaded"] is False
