from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from src.baselines import tastemolnet_gcf_full_resume as t12
from src.baselines import tastemolnet_gcf_production_state as production


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


def _bridge(*, adapter=None, production_history=None):
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
        production_history=production_history,
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


def test_production_bridge_prunes_full_rows_and_reopens_compact_history(
    monkeypatch, tmp_path
):
    from src.baselines.tastemolnet_comrecgc_smoke import _identity_graph_sha256

    collisions = {
        "a": {"canonical_graph": "[C]", "num_nodes": 1, "num_edges": 0},
        "b": {"canonical_graph": "[N]", "num_nodes": 1, "num_edges": 0},
    }
    identities = {
        key: SimpleNamespace(
            graph_identity_sha256=_identity_graph_sha256(value),
            collision_payload=lambda value=value: value,
        )
        for key, value in collisions.items()
    }
    monkeypatch.setattr(
        t12,
        "canonical_attributed_graph",
        lambda graph, **_kwargs: identities[graph.token],
    )
    bounds = production.T12ProductionBounds.pinned(parent_count=2)
    attempt = str(uuid.uuid4())
    history = production.T12CompactHistoryJournal(
        root=(tmp_path / "history").resolve(),
        index_root=(tmp_path / "index-a").resolve(),
        bounds=bounds,
        contract_sha256="e" * 64,
        attempt_id=attempt,
        generation_token="f" * 64,
    )
    adapter = _Adapter()
    embedding = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    adapter.score = lambda values: SimpleNamespace(
        probabilities=np.asarray([[0.1, 0.2, 0.7] for _ in values]),
        graph_embeddings=np.asarray(
            [embedding[value.token] for value in values], dtype=np.float32
        ),
        valid_fullgraphs=tuple(True for _ in values),
        failure_reasons=tuple("" for _ in values),
    )
    bridge = _bridge(adapter=adapter, production_history=history)
    graphs = {
        key: SimpleNamespace(
            token=key,
            num_nodes=1,
            gcf_origin_index=[0],
            gcf_node_origin=[0],
        )
        for key in ("a", "b")
    }
    hashes = {}
    for key in ("a", "b"):
        _parts, rows, _coverage = bridge.call([graphs[key]], {})
        hashes[key] = bridge.calculate_hash(rows[0])
    bridge.vrrw.graph_map = {hashes["a"]: graphs["a"]}
    bridge.vrrw.graph_index_map = {hashes["a"]: 0}
    bridge.vrrw.counterfactual_candidates = [
        {
            "frequency": 2,
            "graph_hash": hashes["a"],
            "importance_parts": [0.8, 0.5],
            "input_graphs_covering_list": [1, 0],
        }
    ]
    bridge.vrrw.transitions = {
        hashes["a"]: (
            [hashes["a"], hashes["b"]],
            [("NOTHING", None, None), ("NLC", 0, 0)],
            [[0.8, 0.5], [0.8, 0.5]],
            [[1, 0], [1, 0]],
        )
    }
    audit = bridge.retain_official_live_domain(
        vrrw=bridge.vrrw, current_graph_identity=hashes["a"]
    )
    assert audit["live_complete_record_count"] == 1
    assert audit["evicted_this_boundary"] == 1
    assert set(bridge.records) == {hashes["a"]}
    assert bridge.is_graph_counterfactual(hashes["b"]) is True
    state = bridge.checkpoint_state()
    assert state["complete_records_are_live_domain_only"] is True
    assert state["history"]["observation_count"] == 2
    history.close()

    reopened_history = production.T12CompactHistoryJournal(
        root=(tmp_path / "history").resolve(),
        index_root=(tmp_path / "index-b").resolve(),
        bounds=bounds,
        contract_sha256="e" * 64,
        attempt_id=attempt,
        generation_token="f" * 64,
        resume_snapshot=state["history"],
    )
    restored = _bridge(production_history=reopened_history)
    restored.restore_checkpoint_state(state)
    assert set(restored.records) == {hashes["a"]}
    assert restored.is_graph_counterfactual(hashes["b"]) is True
    reopened_history.close()


def test_production_identity_allows_only_10k_and_20k():
    accepted = dict(
        _identity(cursor=10_000, total=20_000, purpose="production"),
        sample_size=t12.PINNED_SAMPLE_SIZE,
        candidate_capacity=t12.PINNED_CANDIDATE_CAPACITY,
    )
    assert t12.validate_checkpoint_identity(accepted) == accepted
    terminal = t12.production_checkpoint_identity(
        accepted, checkpoint_cursor=20_000
    )
    assert terminal["checkpoint_cursor"] == 20_000
    assert terminal["sample_size"] == 10_000
    assert terminal["candidate_capacity"] == 100_000
    rejected = dict(accepted, checkpoint_cursor=15_000)
    with pytest.raises(t12.TasteGCFFullResumeError, match="10k/20k"):
        t12.validate_checkpoint_identity(rejected)
    with pytest.raises(t12.TasteGCFFullResumeError, match="sample_size=10000"):
        t12.validate_checkpoint_identity(dict(accepted, sample_size=128))


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


def _process_identity(process):
    return {
        "pid": process,
        "start_ticks": process * 10,
        "command_sha256": "b" * 64,
        "executable_sha256": "c" * 64,
        "cwd_sha256": "d" * 64,
    }


def _prefix_receipt(tmp_path, process=150):
    manifest = {
        "schema_version": t12.CHECKPOINT_MANIFEST_SCHEMA,
        "status": "COMMITTED",
        "stage": t12.STAGE,
        "purpose": "gpu_replay_canary",
        "checkpoint_cursor": 8,
        "total_steps": 16,
        "identity_sha256": "f" * 64,
        "state_sha256": "1" * 64,
        "rng_sha256": "2" * 64,
        "immutable_no_replace": True,
    }
    manifest_path = tmp_path / "checkpoint-00000008.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "schema_version": t12.CANARY_PREFIX_RECEIPT_SCHEMA,
        "status": "CHECKPOINT_COMMITTED",
        "stage": t12.STAGE,
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "checkpoint_identity_sha256": "f" * 64,
        "checkpoint_state_sha256": "1" * 64,
        "checkpoint_rng_sha256": "2" * 64,
        "checkpoint_cursor": 8,
        "total_steps": 16,
        "canary_identity_sha256": "1" * 64,
        "gpu_uuid": "GPU-test-123",
        "process_identity": _process_identity(process),
        "calibration_loaded": False,
        "test_loaded": False,
        "production_released": False,
    }


def _observation(
    role, process, *, prefix=None, changed=False, native_result_sha256="2" * 64
):
    scientific = {
        "completed_steps": 16,
        "traversed_graph_identities": [GRAPH_A, *("b" * 64 for _ in range(15))],
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
        "official_state_sha256": "0" * 64,
        "official_native_result_semantic_sha256": ("e" if not changed else "f") * 64,
    }
    return t12.build_canary_observation(
        role=role,
        canary_identity_sha256="1" * 64,
        gpu_uuid="GPU-test-123",
        process_identity=_process_identity(process),
        scientific_state=scientific,
        native_result_sha256=native_result_sha256,
        checkpoint_reloaded=role == "cross_process_resumed",
        generated_to_original_neurosed_assertion=True,
        checkpoint_process_identity=(
            prefix["process_identity"] if prefix is not None else None
        ),
        checkpoint_manifest_sha256=(
            prefix["checkpoint_manifest_sha256"] if prefix is not None else None
        ),
        checkpoint_identity_sha256=(
            prefix["checkpoint_identity_sha256"] if prefix is not None else None
        ),
        checkpoint_state_sha256=(
            prefix["checkpoint_state_sha256"] if prefix is not None else None
        ),
        checkpoint_rng_sha256=(
            prefix["checkpoint_rng_sha256"] if prefix is not None else None
        ),
    )


def test_canary_gate_requires_new_process_and_exact_scientific_state(tmp_path):
    prefix = _prefix_receipt(tmp_path)
    uninterrupted = _observation("uninterrupted", 100)
    resumed = _observation("cross_process_resumed", 200, prefix=prefix)
    output = tmp_path.resolve() / "gate.json"
    gate = t12.write_canary_gate(output, uninterrupted, resumed, prefix)
    assert gate["status"] == "PASS"
    assert gate["exact_equality"] is True
    assert gate["exact_equality_scope"] == "canonical_scientific_state"
    assert gate["scientific_exact_equality"] is True
    assert gate["native_result_raw_bytes_equal"] is True
    assert gate["native_result_difference_classification"] == "RAW_BYTES_IDENTICAL"
    assert gate["production_released"] is False
    with pytest.raises(
        t12.TasteGCFFullResumeError,
        match="native-result scientific content diverged",
    ):
        t12.compare_canary_observations(
            uninterrupted,
            _observation(
                "cross_process_resumed", 300, prefix=prefix, changed=True
            ),
            prefix,
        )
    same_process_identity = dict(resumed["process_identity"])
    same_process_identity.update(
        pid=uninterrupted["process_identity"]["pid"],
        start_ticks=uninterrupted["process_identity"]["start_ticks"],
        cwd_sha256="9" * 64,
    )
    same_process = dict(resumed, process_identity=same_process_identity)
    with pytest.raises(t12.TasteGCFFullResumeError, match="process boundary"):
        t12.compare_canary_observations(uninterrupted, same_process, prefix)


def test_canary_gate_classifies_raw_archive_drift_as_nonsemantic(tmp_path):
    prefix = _prefix_receipt(tmp_path)
    uninterrupted = _observation(
        "uninterrupted",
        100,
        native_result_sha256=(
            "c636fb10ee796630f55050b15edf4970b020ab40c673c59f7b556a66cc64ce09"
        ),
    )
    resumed = _observation(
        "cross_process_resumed",
        200,
        prefix=prefix,
        native_result_sha256=(
            "3cec981dc116228f503b77dfb948d8208ec83a5c2306f3a679ef802532aa1b0d"
        ),
    )
    gate = t12.compare_canary_observations(uninterrupted, resumed, prefix)
    assert gate["status"] == "PASS"
    assert gate["native_result_semantic_sha256"] == "e" * 64
    assert gate["native_result_raw_bytes_equal"] is False
    assert gate["native_result_difference_classification"] == (
        "NON_SEMANTIC_SERIALIZATION_REPRESENTATION_ONLY"
    )
    assert gate["native_result_approximate_comparison_used"] is False
    assert gate["production_released"] is False


def test_native_result_contract_binds_content_dtype_shape_and_sequence_order():
    numpy = pytest.importorskip("numpy")
    base = {
        "graph_map": {"b": numpy.array([[1, 2]], dtype=numpy.int64), "a": 3},
        "counterfactual_candidates": [{"score": numpy.array([0.1, 0.2])}],
    }
    reordered_mapping = {
        "counterfactual_candidates": [{"score": numpy.array([0.1, 0.2])}],
        "graph_map": {"a": 3, "b": numpy.array([[1, 2]], dtype=numpy.int64)},
    }
    digest = t12.canonical_native_result_sha256(base)
    assert digest == t12.canonical_native_result_sha256(reordered_mapping)
    for changed in (
        {
            **base,
            "graph_map": {
                "b": numpy.array([[1, 3]], dtype=numpy.int64),
                "a": 3,
            },
        },
        {
            **base,
            "graph_map": {
                "b": numpy.array([[1, 2]], dtype=numpy.float64),
                "a": 3,
            },
        },
        {
            **base,
            "graph_map": {
                "b": numpy.array([1, 2], dtype=numpy.int64),
                "a": 3,
            },
        },
        {
            **base,
            "counterfactual_candidates": [
                {"score": numpy.array([0.2, 0.1])}
            ],
        },
    ):
        assert t12.canonical_native_result_sha256(changed) != digest


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
    assert "--checkpoint-prefix-receipt" in cli
