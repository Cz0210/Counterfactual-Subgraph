from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from src.baselines.bace_gine_native_adapter import BACEFrozenGINENativeGraphAdapter
from src.baselines.gcfexplainer_acceleration import (
    BufferedVRRWLogging,
    GCFAccelerationConfig,
    OrderedImportanceAcceleration,
    OrderedNeighbourAcceleration,
    build_acceleration_gate,
    compare_same_gpu_profiles,
    compare_vrrw_equivalence,
    canonical_graph_tensor_digest,
    ordered_parallel_map,
    validate_full_acceleration_gate,
)
from src.data.molecular_graph_featurizer import default_molecular_feature_schema


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_acceleration_config_is_opt_in_and_fingerprinted() -> None:
    legacy = GCFAccelerationConfig()
    optimized = GCFAccelerationConfig(
        mode="ordered_v2",
        gine_batch_size=64,
        graph_cache_capacity=1000,
        cpu_neighbor_workers=2,
        progress_every=500,
    )
    assert legacy.mode == "legacy"
    assert optimized.fingerprint != legacy.fingerprint
    with pytest.raises(ValueError, match="positive graph cache"):
        GCFAccelerationConfig(mode="ordered_v2")
    with pytest.raises(ValueError, match="legacy mode forbids"):
        GCFAccelerationConfig(mode="legacy", cpu_neighbor_workers=2)


def test_ordered_parallel_map_never_reorders_results() -> None:
    values = list(range(100))
    assert ordered_parallel_map(lambda value: value * value, values, workers=4) == [
        value * value for value in values
    ]


def test_buffered_logging_keeps_iteration_sequence() -> None:
    calls = []
    module = SimpleNamespace(tqdm=lambda values: calls.append("legacy") or values)
    with BufferedVRRWLogging(module):
        assert list(module.tqdm(range(5))) == [0, 1, 2, 3, 4]
        assert calls == []
    assert list(module.tqdm(range(1))) == [0]
    assert calls == ["legacy"]


def test_ordered_adapter_deduplicates_and_reuses_gine_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(hidden_dim=2)
            self.classifier = torch.nn.Linear(2, 2, bias=False)
            self.encode_calls = 0

        def encode_graph(self, batch: object) -> object:
            self.encode_calls += 1
            return torch.ones((batch.size, 2), dtype=self.anchor.dtype)

    class FakeBatch:
        def __init__(self, size: int) -> None:
            self.size = size

        def to(self, _device: object) -> "FakeBatch":
            return self

    model = FakeModel()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"model")
    monkeypatch.setattr(
        "src.baselines.bace_gine_native_adapter.load_gnn_checkpoint_bundle",
        lambda _root, device: (
            model,
            {
                "model_card": {
                    "dataset": "bace",
                    "backbone": "gine",
                    "num_classes": 2,
                    "source_label": 1,
                    "rf_oracle_used": False,
                },
                "checkpoint_id": "a" * 64,
                "temperature_scaling": {"temperature": 1.0},
                "feature_schema": default_molecular_feature_schema(),
            },
        ),
    )
    monkeypatch.setattr(
        "src.baselines.bace_gine_native_adapter.collate_molecular_graphs",
        lambda rows, edge_feature_dim: FakeBatch(len(rows)),
    )
    adapter = BACEFrozenGINENativeGraphAdapter(
        checkpoint,
        source_records=[{"molecule_id": "p0"}],
        graph_schema=object(),
        device="cpu",
        acceleration=GCFAccelerationConfig(
            mode="ordered_v2",
            gine_batch_size=2,
            graph_cache_capacity=10,
            cpu_neighbor_workers=2,
        ),
    )
    adapter._decode = MethodType(lambda self, graph: (graph.smiles, None), adapter)
    adapter._portable_graph = MethodType(
        lambda self, smiles, row_index: SimpleNamespace(smiles=smiles), adapter
    )
    graphs = [
        SimpleNamespace(smiles="CC", num_nodes=2),
        SimpleNamespace(smiles="CC", num_nodes=2),
    ]
    first = adapter(graphs)[2]
    second = adapter(graphs)[2]
    assert torch.equal(first, second)
    assert model.encode_calls == 1
    provenance = adapter.provenance()["acceleration"]
    assert provenance["unique_gine_graph_count"] == 1
    assert provenance["cache_hits"] == 2


def test_parallel_neighbour_builder_preserves_official_action_order() -> None:
    torch = pytest.importorskip("torch")

    class TorchUtils:
        @staticmethod
        def degree(_edges: object, *, num_nodes: int) -> object:
            assert num_nodes == 2
            return torch.tensor([1, 1])

    module = SimpleNamespace(
        edge_change=lambda *args, **kwargs: None,
        node_label_change=lambda *args, **kwargs: None,
        node_addition=lambda *args, **kwargs: None,
        node_removal=lambda *args, **kwargs: None,
        neighbor_graph_access=lambda _graph, action: action,
        torch_utils=TorchUtils(),
        nx=SimpleNamespace(),
    )
    graph = SimpleNamespace(
        x=torch.tensor([[1, 0, 0], [0, 1, 0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        num_nodes=2,
    )
    with OrderedNeighbourAcceleration(module, workers=2):
        label_actions, label_graphs = module.node_label_change(graph)
        add_actions, add_graphs = module.node_addition(graph)
        removal_actions, removal_graphs = module.node_removal(graph)
    assert label_actions == [
        ("NLC", 0, 1),
        ("NLC", 0, 2),
        ("NLC", 1, 0),
        ("NLC", 1, 2),
    ]
    assert label_graphs == label_actions
    assert add_actions == [
        ("NA", 0, 0),
        ("NA", 0, 1),
        ("NA", 0, 2),
        ("NA", 1, 0),
        ("NA", 1, 1),
        ("NA", 1, 2),
    ]
    assert add_graphs == add_actions
    assert removal_actions == [("NR", 0, 0), ("NR", 1, 1)]
    assert removal_graphs == removal_actions


def test_importance_cache_preserves_order_and_includes_lineage_identity() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    class Importance:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, graphs: object, _wargs: object) -> object:
            self.calls += 1
            size = len(graphs)
            return (
                np.asarray([[index, 1.0] for index in range(size)]),
                np.asarray([[index, index + 1] for index in range(size)]),
                torch.tensor([[index, 1] for index in range(size)]),
            )

    def graph(parent: int) -> object:
        return SimpleNamespace(
            num_nodes=2,
            x=torch.tensor([[1, 0], [0, 1]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            gcf_origin_index=torch.tensor([parent]),
            gcf_node_origin=torch.tensor([0, 1]),
        )

    first = graph(0)
    second = graph(1)
    assert canonical_graph_tensor_digest(first) != canonical_graph_tensor_digest(second)
    importance = Importance()
    with OrderedImportanceAcceleration(importance, capacity=10) as cache:
        expected = importance.call([first, second], {})
        actual = importance.call([second, first], {})
    assert importance.calls == 1
    assert actual[0].tolist() == [expected[0][1].tolist(), expected[0][0].tolist()]
    assert cache.report()["cache_hits"] == 2


def _equivalence_root(
    root: Path,
    *,
    mode: str,
    budget: int,
    fingerprint: str,
    trace_suffix: str = "same",
) -> None:
    manifest = {
        "dataset": "BACE",
        "dataset_name": "bace",
        "gnn_checkpoint_sha256": "g",
        "neurosed_checkpoint_sha256": "n",
        "neurosed_manifest_sha256": "nm",
        "parent_limit": 64,
        "generation_parent_ids_sha256": "parents",
        "generation_source_cohort_hash": "cohort",
        "M": budget,
        "alpha": 1.0,
        "theta": 0.05,
        "teleport": 0.1,
        "candidate_capacity": 100000,
        "sample": False,
        "sample_size": 10000,
        "seed": 13,
        "acceleration": {"mode": mode, "fingerprint": fingerprint},
    }
    trace = {
        "budget": budget,
        "traversed_count": budget,
        "traversed_canonical_sha256": f"walk-{trace_suffix}",
        "candidate_count": 3,
        "candidate_canonical_sha256": f"candidate-{trace_suffix}",
        "graph_map_count": 4,
        "graph_identity_multiset_sha256": f"graphs-{trace_suffix}",
        "python_random_sha256": "py",
        "numpy_random_sha256": "np",
        "torch_random_sha256": "torch",
        "torch_cuda_random_sha256": "cuda",
        "trace_sha256": f"trace-{trace_suffix}",
    }
    _write_json(root / "run_manifest.json", manifest)
    _write_json(root / "equivalence_trace.json", trace)


def test_equivalence_and_same_gpu_twenty_percent_gate(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    optimized = tmp_path / "optimized"
    _equivalence_root(legacy, mode="legacy", budget=500, fingerprint="legacy")
    _equivalence_root(
        optimized, mode="ordered_v2", budget=500, fingerprint="optimized"
    )
    comparison = compare_vrrw_equivalence(legacy, optimized, budget=500)
    assert comparison["status"] == "PASS"
    marker = tmp_path / "equivalence.json"
    _write_json(marker, comparison)
    for root, seconds in ((legacy, 100.0), (optimized, 75.0)):
        _write_json(
            root / "performance_profile.json",
            {
                "gpu_uuid": "GPU-same",
                "random_walk_seconds": seconds,
                "peak_vram_fraction": 0.4,
            },
        )
    benchmark = compare_same_gpu_profiles(
        legacy_root=legacy,
        optimized_root=optimized,
        equivalence_marker=marker,
    )
    assert benchmark["status"] == "PASS"
    assert benchmark["speedup_fraction"] == pytest.approx(1 / 3)


def test_full_gate_requires_both_budgets_and_same_config(tmp_path: Path) -> None:
    config = GCFAccelerationConfig(
        mode="ordered_v2",
        graph_cache_capacity=5000,
        cpu_neighbor_workers=2,
    )
    markers = []
    for budget in (500, 1000):
        marker = tmp_path / f"eq-{budget}.json"
        _write_json(
            marker,
            {
                "status": "PASS",
                "budget": budget,
                "optimized_config_fingerprint": config.fingerprint,
                "scientific_replay_contract_sha256": "scientific-contract",
            },
        )
        markers.append(marker)
    gate = build_acceleration_gate(
        equivalence_markers=markers,
        benchmark={
            "status": "PASS",
            "same_gpu_uuid": True,
            "speedup_fraction": 0.25,
            "peak_vram_fraction": 0.5,
        },
    )
    assert gate["status"] == "PASS"
    gate_path = tmp_path / "gate.json"
    _write_json(gate_path, gate)
    assert validate_full_acceleration_gate(gate_path, config=config)["status"] == "PASS"


def test_equivalence_fails_on_transition_sequence_difference(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    optimized = tmp_path / "optimized"
    _equivalence_root(legacy, mode="legacy", budget=1000, fingerprint="legacy")
    _equivalence_root(
        optimized,
        mode="ordered_v2",
        budget=1000,
        fingerprint="optimized",
        trace_suffix="different",
    )
    result = compare_vrrw_equivalence(legacy, optimized, budget=1000)
    assert result["status"] == "FAILED"
    assert "canonical_mismatch:traversed_canonical_sha256" in result["failures"]
