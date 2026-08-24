from __future__ import annotations

import json
from pathlib import Path
import random
from types import MethodType, SimpleNamespace

import pytest

from src.baselines.bace_gine_native_adapter import BACEFrozenGINENativeGraphAdapter
from src.baselines.gcfexplainer_bace_adapter import validate_bace_vrrw_profile
from src.baselines.gcfexplainer_acceleration import (
    BufferedVRRWLogging,
    GCFAccelerationConfig,
    LockstepVRRWTrace,
    OrderedImportanceAcceleration,
    OrderedNeighbourAcceleration,
    build_acceleration_gate,
    compare_same_gpu_profiles,
    compare_lockstep_traces,
    compare_vrrw_equivalence,
    canonical_graph_tensor_digest,
    ordered_parallel_map,
    validate_full_acceleration_gate,
)
from src.baselines.frozen_gine_batch_scorer import FrozenGINEBatchScorer
from src.data.molecular_graph_featurizer import default_molecular_feature_schema
from scripts.autodl.gate_bace_gcf_acceleration import parse_args as parse_gate_args
from scripts.autodl.benchmark_bace_frozen_gine_batch import (
    _record_smiles,
    parse_args as parse_benchmark_args,
)


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


@pytest.mark.parametrize("budget", [50, 100])
def test_equivalence_quick_profile_is_diagnostic_and_bounded(budget: int) -> None:
    validate_bace_vrrw_profile(
        "equivalence_quick",
        parent_limit=64,
        m=budget,
        alpha=1.0,
        theta=0.05,
        seed=13,
    )
    with pytest.raises(ValueError, match="expected"):
        validate_bace_vrrw_profile(
            "equivalence_quick",
            parent_limit=64,
            m=500,
            alpha=1.0,
            theta=0.05,
            seed=13,
        )


def test_quick_equivalence_gate_accepts_only_diagnostic_or_formal_budgets() -> None:
    for budget in (50, 100, 500, 1000):
        args = parse_gate_args(
            [
                "equivalence",
                "--legacy-root",
                "/legacy",
                "--optimized-root",
                "/optimized",
                "--budget",
                str(budget),
                "--output",
                "/gate.json",
            ]
        )
        assert args.budget == budget
    with pytest.raises(SystemExit):
        parse_gate_args(
            [
                "equivalence",
                "--legacy-root",
                "/legacy",
                "--optimized-root",
                "/optimized",
                "--budget",
                "49",
                "--output",
                "/gate.json",
            ]
        )


def test_quick_replay_shell_is_diagnostic_only() -> None:
    project_root = Path(__file__).resolve().parents[3]
    text = (
        project_root / "scripts/autodl/run_bace_gcf_quick_replay.sh"
    ).read_text(encoding="utf-8")
    assert "for budget in 50 100" in text
    assert "--profile equivalence_quick" in text
    assert '"eligible_for_full_acceleration_gate": False' in text
    assert "gate_bace_gcf_acceleration.py aggregate" not in text


def test_frozen_gine_benchmark_cli_and_paired_slurm_are_synchronized() -> None:
    args = parse_benchmark_args(
        [
            "--config",
            "configs/hpc.yaml",
            "--set",
            "inference.fallback_to_heuristic=false",
            "--dataset-dir",
            "/dataset",
            "--checkpoint-dir",
            "/checkpoint",
            "--output-dir",
            "/fresh",
        ]
    )
    assert args.rows == 64
    project_root = Path(__file__).resolve().parents[3]
    text = (
        project_root / "scripts/slurm/benchmark_bace_frozen_gine_batch.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
    ):
        assert required in text


def test_frozen_gine_benchmark_accepts_frozen_bace_record_schema() -> None:
    assert _record_smiles({"canonical_smiles": "CCO"}) == "CCO"
    assert _record_smiles({"original_smiles": "CCN"}) == "CCN"
    assert _record_smiles({"smiles": "CCC", "canonical_smiles": "CC"}) == "CCC"
    with pytest.raises(ValueError, match="no molecular SMILES"):
        _record_smiles({"molecule_id": "missing"})


def test_quick50_lockstep_wrapper_is_fail_closed_and_preserves_old_full() -> None:
    project_root = Path(__file__).resolve().parents[3]
    text = (
        project_root / "scripts/autodl/run_bace_gcf_lockstep_quick50.sh"
    ).read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "run_one legacy legacy_a" in text
    assert "run_one legacy legacy_b" in text
    assert "legacy_a_vs_legacy_b.json" in text
    assert "run_one ordered_v2 ordered_v2" in text
    assert "--m 50" in text
    assert "--profile equivalence_quick" in text
    assert "kill" not in text
    assert "139725" not in text


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


def test_ordered_adapter_preserves_duplicate_rows_and_legacy_batch_shape(
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
            self.batch_sizes = []

        def encode_graph(self, batch: object) -> object:
            self.encode_calls += 1
            self.batch_sizes.append(batch.size)
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
    assert model.encode_calls == 2
    assert model.batch_sizes == [2, 2]
    provenance = adapter.provenance()["acceleration"]
    assert provenance["unique_gine_graph_count"] == 4
    assert provenance["cache_hits"] == 0
    assert provenance["portable_cache_entries"] == 0
    assert provenance["prediction_cache_entries"] == 0
    assert provenance["batch_semantics"] == "legacy_full_valid_row_batch_v1"
    assert provenance["in_call_smiles_deduplication"] is False
    assert provenance["gine_chunking"] is False


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


def test_importance_cache_reuses_only_an_exact_complete_ordered_batch() -> None:
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
        wargs = {}
        expected = importance.call([first, second], wargs)
        exact_hit = importance.call([first, second], wargs)
        reversed_miss = importance.call([second, first], wargs)
    assert importance.calls == 2
    assert exact_hit[0].tolist() == expected[0].tolist()
    # A partial row cache would synthesize [row1,row0] here.  The safe cache
    # delegates the complete reversed batch, whose fake outputs are [row0,row1].
    assert reversed_miss[0].tolist() == expected[0].tolist()
    report = cache.report()
    assert report["cache_hits"] == 2
    assert report["cache_misses"] == 4
    assert report["cache_entries"] == 2
    assert report["cache_scope"] == "exact_ordered_full_batch_v1"
    assert report["partial_row_reuse"] is False


def test_frozen_gine_scorer_caches_only_complete_ordered_batches() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.classifier = torch.nn.Identity()
            self.calls = 0

        def encode_graph(self, batch: object) -> object:
            self.calls += 1
            return batch.x

    class Batch:
        def __init__(self, rows: object) -> None:
            self.x = torch.stack([row.x.reshape(-1) for row in rows])
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_attr = torch.empty((0, 1))
            self.batch = torch.arange(len(rows))

        def to(self, _device: object) -> "Batch":
            return self

    def graph(value: float) -> object:
        return SimpleNamespace(
            x=torch.tensor([[value, value + 1]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
            graph_sha256=str(value),
        )

    model = Model()
    scorer = FrozenGINEBatchScorer(
        model=model,
        device="cpu",
        temperature=1.0,
        checkpoint_id="checkpoint",
        collate_fn=Batch,
        cache_capacity=4,
        diagnostic_trace=True,
    )
    first, second = graph(1.0), graph(2.0)
    baseline = scorer.score([first, second], context={"rows": [0, 1]})
    exact = scorer.score([first, second], context={"rows": [0, 1]})
    reversed_rows = scorer.score([second, first], context={"rows": [0, 1]})
    assert model.calls == 2
    assert torch.equal(baseline.project_logits, exact.project_logits)
    assert reversed_rows.project_logits.tolist() == [[2.0, 3.0], [1.0, 2.0]]
    report = scorer.report()
    assert report["cache_hits"] == 2
    assert report["cache_misses"] == 4
    assert report["partial_row_reuse"] is False
    assert report["deduplication"] is False
    assert report["chunking"] is False


def test_lockstep_comparator_reports_exact_first_field(tmp_path: Path) -> None:
    left = {
        "budget": 50,
        "events": [
            {"event": "move", "step": 5, "selected_index": 3},
            {
                "event": "importance",
                "step": 6,
                "importance": {"values": [[0.25, 1.0]]},
            },
        ],
    }
    right = json.loads(json.dumps(left))
    right["events"][1]["importance"]["values"][0][0] = 0.25000003
    left_path = tmp_path / "legacy.json"
    right_path = tmp_path / "optimized.json"
    _write_json(left_path, left)
    _write_json(right_path, right)
    result = compare_lockstep_traces(left_path, right_path)
    assert result["status"] == "FAILED"
    assert result["first_divergence"] == {
        "event_index": 1,
        "step": 6,
        "event": "importance",
        "field": "importance.values[0][0]",
        "legacy": 0.25,
        "optimized": 0.25000003,
    }


def test_lockstep_tracer_does_not_consume_rng_or_change_result() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    graph = SimpleNamespace(
        num_nodes=2,
        x=torch.tensor([[1, 0], [0, 1]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )

    def execute(*, traced: bool) -> tuple[object, object, object]:
        importance = SimpleNamespace()

        def importance_call(graphs: object, _wargs: object) -> object:
            return (
                np.asarray([[0.5, 1.0]] * len(graphs), dtype=np.float32),
                np.asarray([[1.0, 2.0]] * len(graphs), dtype=np.float32),
                torch.ones((len(graphs), 1)),
            )

        importance.call = importance_call
        module = SimpleNamespace(
            graph_map={"g": graph},
            graph_index_map={"g": 0},
            transitions={},
            counterfactual_candidates=[{"importance_parts": [0.5, 1.0]}],
        )

        def restart(_graphs: object) -> str:
            importance.call([graph], {"gnn_model": None})
            return "g"

        def move(*, graph_hash: str, importance_args: object, teleport_probability: float) -> object:
            del importance_args, teleport_probability
            random.uniform(0, 1)
            importance.call([graph], {"gnn_model": None})
            module.transitions[graph_hash] = (["g"], [("NOTHING", None, None)])
            return "g", False

        module.restart_randomwalk = restart
        module.move_to_next_graph = move
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        if traced:
            with LockstepVRRWTrace(
                vrrw=module, importance=importance, torch=torch, np=np, budget=50
            ) as trace:
                result = module.restart_randomwalk([graph])
                moved = module.move_to_next_graph(
                    graph_hash=result,
                    importance_args={},
                    teleport_probability=0.1,
                )
            assert trace.payload()["event_count"] == 4
        else:
            result = module.restart_randomwalk([graph])
            moved = module.move_to_next_graph(
                graph_hash=result,
                importance_args={},
                teleport_probability=0.1,
            )
        return moved, random.getstate(), torch.get_rng_state().clone()

    plain = execute(traced=False)
    traced = execute(traced=True)
    assert plain[0] == traced[0]
    assert plain[1] == traced[1]
    assert torch.equal(plain[2], traced[2])


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


@pytest.mark.parametrize("budget", [50, 100])
def test_quick_equivalence_is_explicitly_diagnostic_only(
    tmp_path: Path, budget: int
) -> None:
    legacy = tmp_path / f"legacy-{budget}"
    optimized = tmp_path / f"optimized-{budget}"
    _equivalence_root(legacy, mode="legacy", budget=budget, fingerprint="legacy")
    _equivalence_root(
        optimized, mode="ordered_v2", budget=budget, fingerprint="optimized"
    )
    comparison = compare_vrrw_equivalence(legacy, optimized, budget=budget)
    assert comparison["status"] == "PASS"
    assert comparison["diagnostic_only"] is True
    assert comparison["eligible_for_full_acceleration_gate"] is False


def test_quick_markers_cannot_build_the_full_gate(tmp_path: Path) -> None:
    markers = []
    for budget in (50, 100):
        marker = tmp_path / f"quick-{budget}.json"
        _write_json(
            marker,
            {
                "status": "PASS",
                "budget": budget,
                "diagnostic_only": True,
                "eligible_for_full_acceleration_gate": False,
                "optimized_config_fingerprint": "same",
                "scientific_replay_contract_sha256": "same-science",
            },
        )
        markers.append(marker)
    gate = build_acceleration_gate(
        equivalence_markers=markers,
        benchmark={
            "status": "PASS",
            "same_gpu_uuid": True,
            "speedup_fraction": 1.0,
            "peak_vram_fraction": 0.1,
        },
    )
    assert gate["status"] == "FAILED"
    assert "requires_exact_500_and_1000_equivalence_markers" in gate["failures"]


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
