from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from src.baselines.globalgce_frozen_gine_bridge import (  # noqa: E402
    FrozenGINEDifferentiableBridge,
    GlobalGCEClassZeroTargetAdapter,
)
from src.data.molecular_graph_featurizer import (  # noqa: E402
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig  # noqa: E402
from scripts.autodl.run_bace_baseline_gnn_route import build_parser  # noqa: E402


def _model() -> MolecularGNN:
    torch.manual_seed(17)
    schema = default_molecular_feature_schema()
    model = MolecularGNN(
        MolecularGNNConfig(
            backbone="gine",
            num_classes=2,
            num_layers=2,
            hidden_dim=16,
            dropout=0.0,
            pooling="mean",
            readout_layers=1,
            normalization="layer_norm",
            residual=True,
        ),
        node_cardinalities=schema.node_cardinalities,
        edge_cardinalities=schema.edge_cardinalities,
    )
    model.eval()
    return model


def _dense_ethanol() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Native GlobalGCE class zero is padding/no-atom and no-edge.
    features = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    adjacency = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    # lower-triangle order: (1,0), (2,0), (2,1)
    edges = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    return features, adjacency, edges


def _ordinary_logits(model: MolecularGNN) -> torch.Tensor:
    graph = MolecularGraphFeaturizer(default_molecular_feature_schema()).featurize(
        "CCO"
    )
    return model(
        x=torch.tensor(graph.node_features, dtype=torch.long),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(graph.edge_features, dtype=torch.long),
    )


def test_bridge_matches_hard_frozen_gine_and_only_transformation_gets_gradient() -> None:
    model = _model()
    bridge = FrozenGINEDifferentiableBridge(
        model,
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="a" * 64,
        temperature=1.0,
    )
    features, adjacency, edges = _dense_ethanol()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = bridge(features, adjacency, edges)
    expected = _ordinary_logits(model)
    assert torch.allclose(result["logits"], expected, rtol=0.0, atol=2e-6)
    assert result["bridge_audit"]["hard_graph_sanitized_count"] == 1

    torch.nn.functional.nll_loss(result["y_pred"], torch.tensor([0])).backward()
    transformation_gradient = sum(
        float(value.grad.detach().abs().sum())
        for value in (features, adjacency, edges)
        if value.grad is not None
    )
    assert transformation_gradient > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert all(
        torch.equal(before[name], value.detach())
        for name, value in model.state_dict().items()
    )
    assert bridge.checkpoint_id == "a" * 64


def test_bridge_keeps_frozen_model_in_eval_when_official_loop_calls_train() -> None:
    model = _model()
    bridge = FrozenGINEDifferentiableBridge(
        model,
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="b" * 64,
        temperature=1.3,
    )
    bridge.train(True)
    assert model.training is False
    assert all(module.training is False for module in model.modules())


def test_bridge_rejects_asymmetric_hard_adjacency() -> None:
    bridge = FrozenGINEDifferentiableBridge(
        _model(),
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="c" * 64,
        temperature=1.0,
    )
    features, adjacency, edges = _dense_ethanol()
    with torch.no_grad():
        adjacency[0, 0, 1] = 0.0
    with pytest.raises(ValueError, match="asymmetric"):
        bridge(features, adjacency, edges)


def test_bridge_smoke_cli_keeps_explicit_native_vocabularies() -> None:
    args = build_parser().parse_args(
        [
            "globalgce-bridge-smoke",
            "--method",
            "GlobalGCE",
            "--gnn-checkpoint",
            "/frozen/gine",
            "--output-dir",
            "/fresh/smoke",
            "--parent-smiles",
            "CCO",
            "--atom-symbol",
            "C",
            "--atom-symbol",
            "O",
            "--bond-name",
            "no_edge",
            "--bond-name",
            "single",
        ]
    )
    assert args.stage == "globalgce-bridge-smoke"
    assert args.atom_symbol == ["C", "O"]
    assert args.bond_name == ["no_edge", "single"]


def test_official_target_one_is_only_a_loss_view_of_frozen_class_zero() -> None:
    bridge = FrozenGINEDifferentiableBridge(
        _model(),
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="d" * 64,
        temperature=1.0,
    )
    features, adjacency, edges = _dense_ethanol()
    ordinary = bridge(features, adjacency, edges)
    adapted = GlobalGCEClassZeroTargetAdapter(bridge)(features, adjacency, edges)
    assert torch.equal(adapted["y_pred"][:, 1], ordinary["y_pred"][:, 0])
    assert torch.equal(adapted["y_pred"][:, 0], ordinary["y_pred"][:, 1])
    assert adapted["bridge_audit"]["frozen_bace_destination_label"] == 0
