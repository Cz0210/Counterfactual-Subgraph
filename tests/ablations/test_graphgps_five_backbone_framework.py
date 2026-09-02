from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ablations.gnn.five_backbone import (
    FIVE_BACKBONES,
    FiveBackboneConfigError,
    build_five_backbone_plan,
    load_five_backbone_config,
    validate_proposal_fixed_runtime_manifest,
)
from src.models.gnn_backbone_registry import (
    available_gnn_backbones,
    get_gnn_backbone_spec,
    normalize_gnn_backbone,
)
from src.models.graphgps_backbone import (
    GRAPHGPS_ALLOWED_HIDDEN_DIMS,
    compute_topology_only_random_walk_pe,
    estimate_graphgps_parameter_count,
    match_graphgps_hidden_dim,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    PROJECT_ROOT
    / "configs/ablations/gnn/bace_ours_proposal_fixed_five_backbones_v1.yaml"
)
REFERENCE = (
    PROJECT_ROOT
    / "configs/ablations/gnn/bace_gine_reference_parameter_receipt_v1.json"
)
MATCH = (
    PROJECT_ROOT / "configs/ablations/gnn/bace_graphgps_parameter_match_v1.json"
)


def test_gps_backbone_registry_discloses_edge_conditioning() -> None:
    assert set(available_gnn_backbones()) == set(FIVE_BACKBONES)
    assert normalize_gnn_backbone("GraphGPS") == "gps"
    spec = get_gnn_backbone_spec("gps")
    assert spec.display_name == "GraphGPS"
    assert "edge" in spec.edge_feature_mode
    assert "gine" in spec.edge_feature_mode


def test_graphgps_parameter_match_uses_only_five_allowed_dimensions() -> None:
    match = match_graphgps_hidden_dim(1_432_583)
    assert match.allowed_hidden_dims == GRAPHGPS_ALLOWED_HIDDEN_DIMS
    assert match.selected_hidden_dim == 160
    assert match.selected_parameter_count == 1_608_327
    assert match.selected_relative_difference == pytest.approx(
        0.12267631264645748
    )
    assert [candidate.hidden_dim for candidate in match.candidates] == [
        96,
        128,
        160,
        192,
        256,
    ]
    assert [candidate.within_tolerance for candidate in match.candidates] == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert match.validation_metrics_loaded is False
    assert match.test_metrics_loaded is False
    with pytest.raises(ValueError, match="exactly"):
        match_graphgps_hidden_dim(1_432_583, allowed_hidden_dims=(128, 160))


def test_graphgps_parameter_formula_matches_frozen_receipt() -> None:
    receipt = json.loads(MATCH.read_text(encoding="utf-8"))
    expected = {
        item["hidden_dim"]: item["parameter_count"]
        for item in receipt["candidates"]
    }
    assert {
        hidden: estimate_graphgps_parameter_count(hidden)
        for hidden in GRAPHGPS_ALLOWED_HIDDEN_DIMS
    } == expected
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert reference["source"] == "ACTUAL_LOADED_WEIGHTS"
    assert reference["total_parameters"] == 1_432_583
    assert reference["validation_metrics_loaded_for_parameter_count"] is False
    assert reference["test_metrics_loaded_for_parameter_count"] is False


def test_five_backbone_config_and_plan_keep_test_after_all_selectors() -> None:
    config = load_five_backbone_config(CONFIG, project_root=PROJECT_ROOT)
    assert config.backbones == FIVE_BACKBONES
    assert config.primary_seed == 7
    assert config.optional_seeds == (17, 27)
    assert config.max_concurrent_gpus == 2
    assert config.graphgps_receipts["selected_hidden_dim"] == 160
    plan = build_five_backbone_plan(config)
    assert plan["science_started"] is False
    assert plan["gpu_lock_acquired"] is False
    assert plan["main_matrix_modified"] is False
    assert plan["graph_mamba"]["run_enabled"] is False
    tasks = {task["task_id"]: task for task in plan["tasks"]}
    selectors = {f"selector:{name}:freeze" for name in FIVE_BACKBONES}
    assert set(tasks["selectors:all:freeze"]["depends_on"]) == selectors
    for name in FIVE_BACKBONES:
        test_task = tasks[f"test:{name}:native-common"]
        assert test_task["split_access"] == ["test"]
        assert "selectors:all:freeze" in test_task["depends_on"]
    for task in plan["tasks"]:
        if task["stage"] != "HELD_OUT_TEST_EVALUATION":
            assert "test" not in task["split_access"]


def test_graph_mamba_is_metadata_only_and_not_a_runnable_backbone() -> None:
    config = load_five_backbone_config(CONFIG, project_root=PROJECT_ROOT)
    metadata = config.graph_mamba_metadata
    assert metadata["official_commit"] == (
        "acb4a2321d46f4044cb5e073a9fadd47eb4f343f"
    )
    assert metadata["run_enabled"] is False
    assert metadata["science_started"] is False
    assert "mamba" not in available_gnn_backbones()


def test_proposal_fixed_manifest_is_hash_bound_and_train_only() -> None:
    sha = "a" * 64
    manifest = validate_proposal_fixed_runtime_manifest(
        {
            "dataset": "bace",
            "method": "ours",
            "source_split": "train",
            "candidate_universe_sha256": sha,
            "generation_per_backbone": False,
            "calibration_loaded": False,
            "test_loaded": False,
        }
    )
    assert manifest["candidate_universe_sha256"] == sha
    with pytest.raises(FiveBackboneConfigError, match="not closed"):
        validate_proposal_fixed_runtime_manifest(
            {**manifest, "test_loaded": True}
        )


def test_graphgps_rwpe_is_topology_only_and_has_fixed_shape() -> None:
    torch = pytest.importorskip("torch")
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
    )
    encoding = compute_topology_only_random_walk_pe(
        edge_index, num_nodes=3, walk_length=16
    )
    assert tuple(encoding.shape) == (3, 16)
    assert torch.isfinite(encoding).all()
    # The API intentionally has no label argument.
    with pytest.raises(TypeError):
        compute_topology_only_random_walk_pe(
            edge_index, num_nodes=3, walk_length=16, labels=[1, 0, 1]
        )


def test_graphgps_output_shape_edge_attr_and_gradient() -> None:
    torch = pytest.importorskip("torch")
    from src.data.molecular_graph_featurizer import default_molecular_feature_schema
    from src.models.graphgps_backbone import build_graphgps_molecular_gnn

    schema = default_molecular_feature_schema()
    model = build_graphgps_molecular_gnn(
        num_classes=2,
        node_feature_schema=schema,
        edge_feature_schema=schema,
        hidden_dim=160,
        backend="project_fallback",
    )
    x = torch.zeros((3, len(schema.node_cardinalities)), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
    )
    edge_attr = torch.zeros(
        (4, len(schema.edge_cardinalities)), dtype=torch.long
    )
    batch = torch.zeros((3,), dtype=torch.long)
    rwpe = compute_topology_only_random_walk_pe(edge_index, num_nodes=3)
    logits = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        random_walk_pe=rwpe,
    )
    assert tuple(logits.shape) == (1, 2)
    logits.sum().backward()
    assert model.edge_encoder.embeddings[0].weight.grad is not None
    with pytest.raises(ValueError, match="random_walk_pe"):
        model(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
