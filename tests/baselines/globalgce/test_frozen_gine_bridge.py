from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from src.baselines.globalgce_frozen_gine_bridge import (  # noqa: E402
    EDGE_SCORE_RELAXATION,
    FrozenGINEDifferentiableBridge,
    GlobalGCEClassZeroTargetAdapter,
    GlobalGCETargetClassAdapter,
)
from src.data.molecular_graph_featurizer import (  # noqa: E402
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig  # noqa: E402
from scripts.autodl.run_bace_baseline_gnn_route import build_parser  # noqa: E402


def _model(*, num_classes: int = 2) -> MolecularGNN:
    torch.manual_seed(17)
    schema = default_molecular_feature_schema()
    model = MolecularGNN(
        MolecularGNNConfig(
            backbone="gine",
            num_classes=num_classes,
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


def test_bridge_from_payloads_uses_only_in_memory_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.baselines import globalgce_frozen_gine_bridge as bridge_module

    payloads = {
        name: f"held:{name}".encode("utf-8")
        for name in (
            "model.pt",
            "model_card.json",
            "feature_schema.json",
            "label_map.json",
            "split_manifest.json",
            "test_evaluation_status.json",
            "temperature_scaling.json",
        )
    }
    captured = {}

    def fake_loader(values, *, device):
        captured["payloads"] = values
        captured["device"] = device
        return _model(num_classes=3), {
            "checkpoint_id": "9" * 64,
            "feature_schema": default_molecular_feature_schema(),
            "temperature_scaling": {"temperature": 1.5},
        }

    monkeypatch.setattr(
        bridge_module,
        "load_gnn_checkpoint_payloads",
        fake_loader,
    )
    bridge = FrozenGINEDifferentiableBridge.from_payloads(
        payloads,
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        device="cpu",
        expected_num_classes=3,
    )
    assert captured == {"payloads": payloads, "device": "cpu"}
    assert bridge.checkpoint_id == "9" * 64
    assert bridge.temperature == 1.5
    assert bridge.num_classes == 3


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


def test_bridge_accepts_official_negative_edge_scores_with_hard_parity() -> None:
    model = _model()
    bridge = FrozenGINEDifferentiableBridge(
        model,
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="e" * 64,
        temperature=1.0,
    )
    features, adjacency, one_hot_edges = _dense_ethanol()
    offsets = torch.tensor([-1.25, -0.75, -0.25, 0.25]).view(1, 1, -1)
    # Same hard classes as the native one-hot graph, but with the unrestricted
    # finite affine score domain emitted by pinned official GlobalGCE.
    edge_scores = (offsets + 2.5 * one_hot_edges.detach()).requires_grad_(True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = bridge(features, adjacency, edge_scores)
    expected = _ordinary_logits(model)
    assert torch.allclose(result["logits"], expected, rtol=0.0, atol=2e-6)
    assert result["bridge_audit"]["edge_score_relaxation"] == EDGE_SCORE_RELAXATION
    graph_audit = result["bridge_audit"]["graphs"][0]
    assert graph_audit["edge_score_negative_value_count"] > 0
    assert graph_audit["edge_score_min"] < 0.0

    torch.nn.functional.nll_loss(result["y_pred"], torch.tensor([0])).backward()
    assert edge_scores.grad is not None
    assert float(edge_scores.grad.detach().abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert all(
        torch.equal(before[name], value.detach())
        for name, value in model.state_dict().items()
    )
    assert bridge.checkpoint_id == "e" * 64


def test_bridge_rejects_nonfinite_official_edge_scores() -> None:
    bridge = FrozenGINEDifferentiableBridge(
        _model(),
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="f" * 64,
        temperature=1.0,
    )
    features, adjacency, edges = _dense_ethanol()
    with torch.no_grad():
        edges[0, 0, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        bridge(features, adjacency, edges)


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


def test_bridge_smoke_cli_prints_controller_pass_marker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from scripts.autodl import run_bace_baseline_gnn_route as route_cli

    monkeypatch.setattr(
        route_cli,
        "run_frozen_gine_bridge_smoke",
        lambda **_kwargs: {"status": "PASS"},
    )
    assert route_cli.main(
        [
            "globalgce-bridge-smoke",
            "--method",
            "GlobalGCE",
            "--gnn-checkpoint",
            str(tmp_path / "gine"),
            "--output-dir",
            str(tmp_path / "fresh-smoke"),
            "--parent-smiles",
            "CCO",
            "--atom-symbol",
            "C",
            "--atom-symbol",
            "O",
            "--device",
            "cpu",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"status": "PASS"' in output
    assert "[BACE_GLOBALGCE_BRIDGE_PASS]" in output


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


@pytest.mark.parametrize(
    ("target_label", "expected_order"),
    ((0, (1, 0, 2)), (2, (1, 2, 0))),
)
def test_multiclass_target_adapter_uses_one_three_class_gine_without_projection(
    target_label: int,
    expected_order: tuple[int, int, int],
) -> None:
    bridge = FrozenGINEDifferentiableBridge(
        _model(num_classes=3),
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="1" * 64,
        temperature=1.0,
        expected_num_classes=3,
    )
    features, adjacency, edges = _dense_ethanol()
    ordinary = bridge(features, adjacency, edges)
    adapted = GlobalGCETargetClassAdapter(
        bridge,
        source_label=1,
        target_label=target_label,
    )(features, adjacency, edges)
    assert tuple(adapted["bridge_audit"]["frozen_class_order_seen_by_official"]) == (
        expected_order
    )
    for internal, frozen in enumerate(expected_order):
        assert torch.equal(
            adapted["y_pred"][:, internal], ordinary["y_pred"][:, frozen]
        )
    assert adapted["bridge_audit"]["num_classes"] == 3
    assert adapted["bridge_audit"]["frozen_source_label"] == 1
    assert adapted["bridge_audit"]["frozen_target_label"] == target_label


@pytest.mark.parametrize("target_label", (0, 2))
def test_multiclass_target_loss_keeps_every_frozen_logit_in_softmax_gradient(
    target_label: int,
) -> None:
    model = _model(num_classes=3)
    bridge = FrozenGINEDifferentiableBridge(
        model,
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="4" * 64,
        temperature=1.25,
        expected_num_classes=3,
    )
    captured_logits: list[torch.Tensor] = []

    def _capture_logits(_module, _inputs, output) -> None:
        output.retain_grad()
        captured_logits.append(output)

    hook = model.classifier.register_forward_hook(_capture_logits)
    try:
        features, adjacency, edges = _dense_ethanol()
        adapted = GlobalGCETargetClassAdapter(
            bridge,
            source_label=1,
            target_label=target_label,
        )(features, adjacency, edges)
        torch.nn.functional.nll_loss(
            adapted["y_pred"],
            torch.tensor([1]),
        ).backward()
    finally:
        hook.remove()

    assert len(captured_logits) == 1
    gradient = captured_logits[0].grad
    assert gradient is not None
    assert tuple(gradient.shape) == (1, 3)
    assert torch.all(gradient.abs() > 0.0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert sum(
        float(value.grad.detach().abs().sum())
        for value in (features, adjacency, edges)
        if value.grad is not None
    ) > 0.0


@pytest.mark.parametrize(
    ("source_label", "target_label"),
    ((True, 0), (1, False), (1, 1), (-1, 0), (1, 3)),
)
def test_multiclass_target_adapter_rejects_untyped_or_invalid_labels(
    source_label: object,
    target_label: object,
) -> None:
    bridge = FrozenGINEDifferentiableBridge(
        _model(num_classes=3),
        feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="2" * 64,
        temperature=1.0,
        expected_num_classes=3,
    )
    with pytest.raises(ValueError, match="distinct exact classes"):
        GlobalGCETargetClassAdapter(
            bridge,
            source_label=source_label,  # type: ignore[arg-type]
            target_label=target_label,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad", (True, 3.0, 1, 4))
def test_bridge_rejects_untyped_or_wrong_expected_class_count(bad: object) -> None:
    with pytest.raises(ValueError, match="class count|exact integer"):
        FrozenGINEDifferentiableBridge(
            _model(num_classes=3),
            feature_schema=default_molecular_feature_schema(),
            atom_symbols=("C", "O"),
            bond_names=("no_edge", "single", "double", "triple"),
            checkpoint_id="3" * 64,
            temperature=1.0,
            expected_num_classes=bad,  # type: ignore[arg-type]
        )
