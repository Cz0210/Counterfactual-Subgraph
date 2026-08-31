from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from src.baselines import tastemolnet_gcf_full_resume as t12


GRAPH_A = "a" * 64
ROOT = Path(__file__).resolve().parents[2]


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.is_sparse = False

    @property
    def shape(self):
        return self.value.shape

    def all(self):
        return _Array(self.value.all())

    def item(self):
        return self.value.item()

    def cpu(self):
        return self

    def tolist(self):
        return self.value.tolist()

    def __getitem__(self, item):
        return _Array(self.value[item])

    def __eq__(self, other):
        return _Array(self.value == other)

    def __or__(self, other):
        return _Array(self.value | other.value)


class _FakeTorch:
    _cpu = b"cpu-rng"

    class cuda:
        _state = [b"cuda-rng"]

        @staticmethod
        def is_available():
            return True

        @classmethod
        def get_rng_state_all(cls):
            return list(cls._state)

        @classmethod
        def set_rng_state_all(cls, value):
            cls._state = list(value)

    @staticmethod
    def isfinite(value):
        raw = value.value if isinstance(value, _Array) else value
        return _Array(np.isfinite(raw))

    @classmethod
    def get_rng_state(cls):
        return cls._cpu

    @classmethod
    def set_rng_state(cls, value):
        cls._cpu = value

    @staticmethod
    def save(value, stream):
        pickle.dump(value, stream)

    @staticmethod
    def load(source, **_kwargs):
        if hasattr(source, "read"):
            return pickle.load(source)
        with Path(source).open("rb") as stream:
            return pickle.load(stream)


class _Scorer:
    cache_capacity = 0

    def __init__(self):
        self._cache = {}
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.scored_rows = 0
        self.last_trace = None


class _Adapter:
    def __init__(self):
        self.scorer = _Scorer()
        self.decode_failures = {}
        self.decode_success_count = 0
        self.empty_valid_batch_count = 0
        self.call_count = 0


def _bridge(*, adapter=None):
    return t12.T12StableGCFBridge(
        adapter=adapter or _Adapter(),
        vrrw=SimpleNamespace(
            torch=_FakeTorch,
            calculate_hash=lambda _row: -1,
            is_graph_counterfactual=lambda _row: False,
        ),
        importance=SimpleNamespace(
            call=lambda *_args: None,
            neurosed_threshold_coverage_estimation=lambda *_args: _Array([[1, 0]]),
        ),
        neurosed_model=object(),
        original_graph_element_counts=object(),
        distance_threshold=0.125,
        parent_count=2,
        feature_atomic_numbers=(6,),
    )


def _identity(*, cursor=1, total=2, purpose="gpu_replay_canary"):
    return {
        "schema_version": "tastemolnet_t12_checkpoint_identity_v1",
        "stage": t12.STAGE,
        "purpose": purpose,
        "attempt_id": str(uuid.uuid4()),
        "generation_token": "1" * 64,
        "total_steps": total,
        "checkpoint_cursor": cursor,
        "source_cohort_sha256": "2" * 64,
        "train_split_sha256": "3" * 64,
        "model_checkpoint_sha256": "4" * 64,
        "model_config_sha256": "5" * 64,
        "neurosed_checkpoint_sha256": "6" * 64,
        "neurosed_distance_threshold_hex": float(0.125).hex(),
        "neurosed_threshold_authority_sha256": "7" * 64,
        "official_source_inventory_sha256": "8" * 64,
        "execution_commit": "9" * 40,
        "execution_tree": "a" * 40,
        "runtime_identity_sha256": "b" * 64,
        "gpu_uuid": "GPU-test-123",
        "device": "cuda:0",
        "graph_identity_contract": t12.GRAPH_IDENTITY_CONTRACT,
        "seed": 7,
        "alpha_hex": float(1.0).hex(),
        "teleport_hex": float(0.1).hex(),
        "sample_size": 128,
        "candidate_capacity": 512,
        "train_loaded": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
    }


def _vrrw():
    return SimpleNamespace(
        graph_map={GRAPH_A: {"x": [1]}},
        graph_index_map={GRAPH_A: 0},
        counterfactual_candidates=[
            {
                "frequency": 2,
                "graph_hash": GRAPH_A,
                "importance_parts": [0.8, 0.5],
                "input_graphs_covering_list": [1, 0],
            }
        ],
        input_graphs_covered=[1, 0],
        covering_graphs={GRAPH_A},
        transitions={},
        traversed_hashes=[GRAPH_A],
        MAX_COUNTERFACTUAL_SIZE=512,
        starting_step=2,
        dataset_name="tastemolnet",
        alpha=1.0,
        sample_size=128,
        is_sample=True,
        importance_args={"alpha": 1.0},
    )


def test_stable_bridge_returns_structural_identity_not_embedding_hash(monkeypatch):
    collision = {
        "canonical_graph": "[C]",
        "num_nodes": 1,
        "num_edges": 0,
    }
    from src.baselines.tastemolnet_comrecgc_smoke import _identity_graph_sha256

    structural_hash = _identity_graph_sha256(collision)
    identity = SimpleNamespace(
        graph_identity_sha256=structural_hash,
        collision_payload=lambda: collision,
    )
    monkeypatch.setattr(t12, "canonical_attributed_graph", lambda *_args, **_kwargs: identity)
    adapter = _Adapter()
    adapter.score = lambda values: SimpleNamespace(
        probabilities=np.asarray([[0.1, 0.2, 0.7] for _ in values]),
        graph_embeddings=np.asarray([[1.0, 2.0] for _ in values], dtype=np.float32),
        valid_fullgraphs=tuple(True for _ in values),
        failure_reasons=tuple("" for _ in values),
    )
    bridge = _bridge(adapter=adapter)
    graph = SimpleNamespace(
        num_nodes=1,
        gcf_origin_index=[0],
        gcf_node_origin=[0],
    )
    parts, embeddings, _coverage = bridge.call([graph], {})
    assert parts.tolist() == [[0.8, 0.5]]
    assert bridge.calculate_hash(embeddings[0]) == structural_hash
    assert bridge.is_graph_counterfactual(structural_hash) is True
    report = bridge.report()
    assert report["graph_identity_contract"] == t12.GRAPH_IDENTITY_CONTRACT
    assert report["python_builtin_hash_used"] is False
    assert report["embedding_identity_used"] is False
    restored = _bridge(adapter=adapter)
    restored.restore_checkpoint_state(bridge.checkpoint_state())
    assert restored.checkpoint_state() == bridge.checkpoint_state()


def test_production_identity_allows_only_10k_and_20k():
    accepted = _identity(cursor=10_000, total=20_000, purpose="production")
    assert t12.validate_checkpoint_identity(accepted) == accepted
    rejected = dict(accepted, checkpoint_cursor=15_000)
    with pytest.raises(t12.TasteGCFFullResumeError, match="10k/20k"):
        t12.validate_checkpoint_identity(rejected)


def test_checkpoint_is_immutable_reopenable_and_restores_rng(tmp_path):
    root = tmp_path.resolve() / "checkpoints"
    identity = _identity()
    bridge = _bridge()
    adapter = bridge.adapter
    vrrw = _vrrw()
    payload = t12.capture_checkpoint_payload(
        identity=identity,
        vrrw=vrrw,
        bridge=bridge,
        adapter=adapter,
        action_counts={"NLC": 3},
        current_graph_identity=GRAPH_A,
        np=np,
        torch=_FakeTorch,
    )
    manifest = t12.write_checkpoint(root, payload, torch=_FakeTorch)
    loaded = t12.reopen_checkpoint(
        manifest, expected_identity=identity, torch=_FakeTorch
    )
    with pytest.raises(t12.TasteGCFFullResumeError, match="already exists"):
        t12.write_checkpoint(root, payload, torch=_FakeTorch)

    restored_vrrw = _vrrw()
    restored_vrrw.graph_map = {}
    restored_vrrw.graph_index_map = {}
    restored_vrrw.counterfactual_candidates = []
    restored_bridge = _bridge()
    restored_actions = {}
    current = t12.restore_checkpoint_payload(
        loaded,
        expected_identity=identity,
        vrrw=restored_vrrw,
        bridge=restored_bridge,
        adapter=restored_bridge.adapter,
        action_counts=restored_actions,
        np=np,
        torch=_FakeTorch,
    )
    assert current == GRAPH_A
    assert restored_vrrw.traversed_hashes == [GRAPH_A]
    assert restored_actions == {"NLC": 3}


def _observation(role, process, *, changed=False):
    scientific = {
        "completed_steps": 2,
        "traversed_graph_identities": [GRAPH_A, "b" * 64],
        "candidate_frequency_order": [[GRAPH_A, 2, "3" * 64, "4" * 64]],
        "graph_map_sha256": "5" * 64,
        "graph_index_map_sha256": "6" * 64,
        "transitions_sha256": "c" * 64,
        "current_graph_identity": "b" * 64,
        "bridge_state_sha256": "7" * 64,
        "adapter_state_sha256": "8" * 64,
        "action_counts_sha256": "9" * 64,
        "rng_state_sha256": "a" * 64,
        "generated_to_original_coverage_sha256": "d" * 64,
        "official_native_result_semantic_sha256": ("e" if not changed else "f") * 64,
    }
    return t12.build_canary_observation(
        role=role,
        canary_identity_sha256="1" * 64,
        gpu_uuid="GPU-test-123",
        process_identity={
            "pid": process,
            "start_ticks": process * 10,
            "command_sha256": "b" * 64,
            "executable_sha256": "c" * 64,
            "cwd_sha256": "d" * 64,
        },
        scientific_state=scientific,
        native_result_sha256="2" * 64,
        checkpoint_reloaded=role == "cross_process_resumed",
        generated_to_original_neurosed_assertion=True,
    )


def test_canary_gate_requires_new_process_and_exact_scientific_state(tmp_path):
    uninterrupted = _observation("uninterrupted", 100)
    resumed = _observation("cross_process_resumed", 200)
    output = tmp_path.resolve() / "gate.json"
    gate = t12.write_canary_gate(output, uninterrupted, resumed)
    assert gate["status"] == "PASS"
    assert gate["exact_equality"] is True
    assert gate["production_released"] is False
    with pytest.raises(t12.TasteGCFFullResumeError, match="scientific_state_sha256"):
        t12.compare_canary_observations(
            uninterrupted, _observation("cross_process_resumed", 300, changed=True)
        )
    same_process = dict(resumed, process_identity=dict(uninterrupted["process_identity"]))
    with pytest.raises(t12.TasteGCFFullResumeError, match="process boundary"):
        t12.compare_canary_observations(uninterrupted, same_process)


def test_canary_cli_has_paired_hpc_slurm_contract():
    cli = (ROOT / "scripts/run_tastemolnet_gcf_replay_canary.py").read_text()
    slurm = (
        ROOT / "scripts/slurm/run_tastemolnet_gcf_replay_canary.sh"
    ).read_text()
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
    ):
        assert token in slurm
    assert "write_canary_gate" in cli
    assert "CANARY_PASS_MARKER" in cli
