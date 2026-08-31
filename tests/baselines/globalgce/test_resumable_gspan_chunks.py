from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import src.baselines.globalgce_resumable as resumable_module
from src.baselines.globalgce_resumable import (
    GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V1,
    GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2,
    GLOBALGCE_TRAINING_RESUME_IDENTITY_SCHEMA_VERSION,
    normalize_globalgce_training_resume_identity,
    resumable_gspan_root_chunks,
    train_globalgce_resumable,
    validate_globalgce_epoch_checkpoint_identity,
)


Projected = list


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def PDFS(graph_id, edge, previous):
    return graph_id, edge, previous


def DFSedge(left, right, labels):
    return left, right, labels


class _NXGraph:
    def __init__(self, label: str) -> None:
        self.label = label

    def nodes(self, data=False):
        assert data is True
        return [(0, {"label": self.label}), (1, {"label": f"{self.label}x"})]

    def edges(self, data=False):
        assert data is True
        return [(0, 1, {"label": "single"})]


class _DFSCode:
    def __init__(self) -> None:
        self.values = []

    def append(self, value) -> None:
        self.values.append(value)

    def pop(self):
        return self.values.pop()

    def get_num_vertices(self) -> int:
        return 2

    def to_graph(self, *, gid, is_undirected):
        return (gid, is_undirected, tuple(self.values[-1][2]))


class _OfficialGraph:
    def __init__(self, source_label: str, target_label: str) -> None:
        target = SimpleNamespace(vid=1, vlb=target_label)
        self.vertices = {
            0: SimpleNamespace(vid=0, vlb=source_label),
            1: target,
        }
        self.edge = SimpleNamespace(to=1, elb="single")


class _FakeGSpan:
    def __init__(self) -> None:
        self._max_num_vertices = 4
        self._min_num_vertices = 2
        self._min_support = 1
        self._is_undirected = True
        self._counter = itertools.count()
        self._DFScode = _DFSCode()
        self._frequent_subgraphs = []
        self.fs_collection = []
        self.freq_collection = []
        self._support = 0
        self.graphs = {
            0: _OfficialGraph("A", "B"),
            1: _OfficialGraph("C", "D"),
        }
        self._nx_graph_list = [_NXGraph("A"), _NXGraph("C")]

    def _read_graphs(self) -> None:
        return None

    def _generate_1edge_frequent_subgraphs(self) -> None:
        self._counter = itertools.count()

    def _get_forward_root_edges(self, graph, vertex_id):
        return [graph.edge] if vertex_id == 0 else []

    def _subgraph_mining(self, projected) -> None:
        self._support = len(projected)
        self._report(projected)

    def _from_Graph_to_nx_Graph(self, graph):
        return graph

    def _report(self, projected) -> None:
        self._frequent_subgraphs.append(tuple(self._DFScode.values))
        graph = self._DFScode.to_graph(
            gid=next(self._counter), is_undirected=self._is_undirected
        )
        self.fs_collection.append(self._from_Graph_to_nx_Graph(graph))
        self.freq_collection.append(self._support)

    def run(self):
        self._read_graphs()
        self._generate_1edge_frequent_subgraphs()
        root = defaultdict(Projected)
        for graph_id, graph in self.graphs.items():
            for vertex_id, vertex in graph.vertices.items():
                for edge in self._get_forward_root_edges(graph, vertex_id):
                    root[(vertex.vlb, edge.elb, graph.vertices[edge.to].vlb)].append(
                        PDFS(graph_id, edge, None)
                    )
        for labels, projected in root.items():
            self._DFScode.append(DFSedge(0, 1, labels))
            self._subgraph_mining(projected)
            self._DFScode.pop()
        return self.fs_collection, self.freq_collection


def test_root_chunk_resume_matches_uninterrupted_reference(tmp_path) -> None:
    reference = _FakeGSpan().run()
    module = SimpleNamespace(gSpan=_FakeGSpan)

    first = _FakeGSpan()
    with resumable_gspan_root_chunks(module, checkpoint_root=tmp_path):
        first_result = first.run()

    second = _FakeGSpan()
    second._subgraph_mining = lambda _projected: (_ for _ in ()).throw(
        AssertionError("completed root should be loaded from checkpoint")
    )
    with resumable_gspan_root_chunks(module, checkpoint_root=tmp_path):
        resumed_result = second.run()

    assert first_result == reference
    assert resumed_result == reference
    databases = sorted(tmp_path.glob("support_*/frequent_patterns.sqlite3"))
    assert len(databases) == 1
    assert list(tmp_path.glob("support_*/checkpoint.json"))


def _training_resume_identity(*, target_label: int = 0) -> dict:
    cohort = {
        "count": 6,
        "ordered_sha256": "1" * 64,
        "train_count": 3,
        "train_ordered_sha256": "2" * 64,
        "val_count": 3,
        "val_ordered_sha256": "3" * 64,
    }
    inventory = [
        {"name": "model.pt", "bytes": 41, "sha256": "4" * 64},
        {
            "name": "temperature_scaling.json",
            "bytes": 42,
            "sha256": "5" * 64,
        },
    ]
    oracle_identity = {
        "schema_version": "globalgce_frozen_gine_resume_identity_v1",
        "backend": "frozen_gine",
        "checkpoint_root": "/frozen/tastemolnet/gine",
        "checkpoint_id": "4" * 64,
        "dataset": "tastemolnet",
        "num_classes": 3,
        "source_label": 1,
        "temperature_hex": float(1.25).hex(),
        "temperature_scaling_sha256": "5" * 64,
        "sha256sums_sha256": "6" * 64,
        "inventory": inventory,
        "inventory_sha256": _canonical_sha256({"files": inventory}),
    }
    oracle_identity["identity_sha256"] = _canonical_sha256(oracle_identity)
    official_source_identity = {
        "schema_version": "globalgce_official_source_resume_identity_v1",
        "root": "/frozen/official/globalgce",
        "files": {
            "models/GlobalGCE.py": {
                "bytes": 43,
                "sha256": "7" * 64,
            }
        },
    }
    official_source_identity["identity_sha256"] = _canonical_sha256(
        official_source_identity
    )
    return {
        "schema_version": GLOBALGCE_TRAINING_RESUME_IDENTITY_SCHEMA_VERSION,
        "dataset": "TasteMolNet",
        "num_classes": 3,
        "source_label": 1,
        "target_label": target_label,
        "oracle_identity": oracle_identity,
        "native_train_cohort": dict(cohort),
        "source_train_cohort": {
            **cohort,
            "ordered_sha256": "6" * 64,
            "train_ordered_sha256": "7" * 64,
            "val_ordered_sha256": "8" * 64,
        },
        "official_source_identity": official_source_identity,
        "training_config": {
            "seed": 7,
            "epochs": 1,
            "top_k_native": 20,
            "learning_rate_hex": float(0.001).hex(),
            "dropout_hex": float(0.5).hex(),
            "min_freq": 7,
            "gspan_flush_every": 256,
            "gspan_max_in_memory_candidates": 256,
            "gspan_exact_top_k_pruning": False,
            "gspan_adoption_identity": None,
        },
    }


def _identity_bound_checkpoint(identity: dict) -> dict:
    normalized, digest = normalize_globalgce_training_resume_identity(identity)
    return {
        "checkpoint_schema_version": GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2,
        "resume_identity": normalized,
        "resume_identity_sha256": digest,
    }


def test_partial_checkpoint_rejects_cross_target_resume() -> None:
    target_zero = _training_resume_identity(target_label=0)
    target_two = _training_resume_identity(target_label=2)
    checkpoint = _identity_bound_checkpoint(target_zero)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        validate_globalgce_epoch_checkpoint_identity(checkpoint, target_two)


def test_v2_identity_rejects_incomplete_frozen_gine_or_training_config() -> None:
    missing_temperature = _training_resume_identity()
    missing_temperature["oracle_identity"].pop("temperature_scaling_sha256")
    with pytest.raises(ValueError, match="frozen-GINE resume identity"):
        normalize_globalgce_training_resume_identity(missing_temperature)

    missing_gspan_limit = _training_resume_identity()
    missing_gspan_limit["training_config"].pop(
        "gspan_max_in_memory_candidates"
    )
    with pytest.raises(ValueError, match="configuration identity"):
        normalize_globalgce_training_resume_identity(missing_gspan_limit)


@pytest.mark.parametrize("drift", ("source_ids", "gine", "temperature"))
def test_partial_checkpoint_rejects_cohort_or_oracle_drift(drift: str) -> None:
    identity = _training_resume_identity(target_label=0)
    expected = deepcopy(identity)
    if drift == "source_ids":
        expected["source_train_cohort"]["train_ordered_sha256"] = "a" * 64
    elif drift == "gine":
        expected["oracle_identity"]["checkpoint_id"] = "b" * 64
        expected["oracle_identity"]["inventory"][0]["sha256"] = "b" * 64
        expected["oracle_identity"]["inventory_sha256"] = _canonical_sha256(
            {"files": expected["oracle_identity"]["inventory"]}
        )
    else:
        expected["oracle_identity"]["temperature_hex"] = float(2.0).hex()
    if drift in {"gine", "temperature"}:
        expected["oracle_identity"]["identity_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in expected["oracle_identity"].items()
                if key != "identity_sha256"
            }
        )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        validate_globalgce_epoch_checkpoint_identity(
            _identity_bound_checkpoint(identity),
            expected,
        )


class _TinyGlobalGCE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.device = torch.device("cpu")
        self.gt_gnn = SimpleNamespace(eval=lambda: None)
        self.fsg = SimpleNamespace(topk=1)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def get_rules(self, _frequent_subgraphs):
        return {"weight": self.weight.detach().clone()}

    def run_one_batch(self, _rules, _data):
        loss = (self.weight - 1.0).square()
        zero = loss * 0.0
        return loss, zero, zero, loss


def _tiny_eval(_expanded_val, model, _pred_model, _rules):
    loss = (model.weight - 1.0).square()
    zero = loss * 0.0
    return {"loss": loss, "loss_kl": zero, "loss_sim": zero, "loss_cfe": loss}


def _run_tiny_identity_training(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    identity: dict | None,
    resume: bool,
    expected_resume_checkpoint=None,
    on_resume_checkpoint=None,
    after_epoch_checkpoint=None,
) -> None:
    monkeypatch.setattr(
        resumable_module,
        "_get_fs_expanded_data_from_adoption",
        lambda **_kwargs: ((["fss"], ["train"], ["val"], ["test"]), {}),
    )
    train_globalgce_resumable(
        epochs=0,
        pred_model=SimpleNamespace(),
        model=_TinyGlobalGCE(),
        learning_rate=0.01,
        train_loader=SimpleNamespace(),
        val_loader=SimpleNamespace(),
        save_rule_path=tmp_path / "rules.pt",
        save_model_path=tmp_path / "model.pt",
        checkpoint_dir=tmp_path / "checkpoints",
        torch_module=torch,
        numpy_module=np,
        test_globalgce=_tiny_eval,
        gspan_module=SimpleNamespace(),
        resume=resume,
        gspan_adoption_proof=tmp_path / "adoption.json",
        resume_identity=identity,
        expected_resume_checkpoint=expected_resume_checkpoint,
        on_resume_checkpoint=on_resume_checkpoint,
        after_epoch_checkpoint=after_epoch_checkpoint,
    )


def test_epoch_callback_can_prove_planned_stop_and_exact_resume(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _training_resume_identity(target_label=0)
    checkpoint_events = []

    def planned_stop(event):
        checkpoint_events.append(event)
        raise RuntimeError("planned epoch boundary")

    with pytest.raises(RuntimeError, match="planned epoch boundary"):
        _run_tiny_identity_training(
            tmp_path,
            monkeypatch,
            identity=identity,
            resume=True,
            after_epoch_checkpoint=planned_stop,
        )
    assert len(checkpoint_events) == 1
    planned = checkpoint_events[0]
    assert planned["checkpoint_schema_version"] == (
        GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2
    )
    assert planned["epoch"] == 0
    assert planned["next_epoch"] == 1
    assert planned["checkpoint_and_heartbeat_durable"] is True

    resume_events = []
    _run_tiny_identity_training(
        tmp_path,
        monkeypatch,
        identity=identity,
        resume=True,
        expected_resume_checkpoint=planned["checkpoint_file"],
        on_resume_checkpoint=resume_events.append,
    )
    assert len(resume_events) == 1
    resumed = resume_events[0]
    assert resumed["checkpoint_sha256"] == planned["checkpoint_sha256"]
    assert resumed["checkpoint_file"] == planned["checkpoint_file"]
    assert planned["heartbeat_file"]["sha256"]
    assert resumed["resume_identity_sha256"] == planned["resume_identity_sha256"]
    assert resumed["next_epoch"] == 1
    assert resumed["rng_state_restored"] is True
    assert resumed["model_state_restored"] is True
    assert resumed["optimizer_state_restored"] is True
    assert resumed["scheduler_state_restored"] is True


class _DeviceLikeByteTensor(torch.Tensor):
    cpu_calls = 0

    @staticmethod
    def __new__(cls, base: torch.Tensor) -> "_DeviceLikeByteTensor":
        return torch.Tensor._make_subclass(cls, base, require_grad=False)

    def cpu(self) -> torch.Tensor:
        type(self).cpu_calls += 1
        return self.detach().clone().as_subclass(torch.Tensor)


def test_normalize_torch_rng_state_calls_cpu_on_device_like_tensor() -> None:
    original = torch.get_rng_state().clone()
    _DeviceLikeByteTensor.cpu_calls = 0
    value = _DeviceLikeByteTensor(original)
    normalized = resumable_module._normalize_torch_rng_state(
        torch,
        value,
        label="test torch rng state",
    )
    assert _DeviceLikeByteTensor.cpu_calls == 1
    assert isinstance(normalized, torch.Tensor)
    assert type(normalized) is torch.Tensor
    assert normalized.dtype == torch.uint8
    assert normalized.dim() == 1
    assert normalized.device.type == "cpu"
    assert normalized.is_contiguous()
    assert torch.equal(normalized, original.cpu())


def test_normalize_cuda_rng_state_all_normalizes_each_tensor() -> None:
    original = torch.get_rng_state().clone()
    _DeviceLikeByteTensor.cpu_calls = 0
    normalized = resumable_module._normalize_cuda_rng_state_all(
        torch,
        [_DeviceLikeByteTensor(original), _DeviceLikeByteTensor(original.clone())],
    )
    assert normalized is not None
    assert len(normalized) == 2
    assert _DeviceLikeByteTensor.cpu_calls == 2
    assert all(type(item) is torch.Tensor for item in normalized)
    assert all(item.dtype == torch.uint8 for item in normalized)
    assert all(item.dim() == 1 for item in normalized)
    assert all(item.device.type == "cpu" for item in normalized)
    assert all(item.is_contiguous() for item in normalized)


def test_normalize_torch_rng_state_rejects_wrong_dtype_and_dim() -> None:
    with pytest.raises(ValueError, match="dtype torch.uint8"):
        resumable_module._normalize_torch_rng_state(
            torch,
            torch.arange(8, dtype=torch.int64),
            label="bad dtype rng state",
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        resumable_module._normalize_torch_rng_state(
            torch,
            torch.zeros((2, 4), dtype=torch.uint8),
            label="bad dim rng state",
        )
    with pytest.raises(ValueError, match="must be a torch.Tensor"):
        resumable_module._normalize_torch_rng_state(
            torch,
            [1, 2, 3],
            label="bad type rng state",
        )


def test_expected_checkpoint_rejects_same_byte_physical_leaf_swap(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _training_resume_identity(target_label=0)
    checkpoint_events = []

    def planned_stop(event):
        checkpoint_events.append(event)
        raise RuntimeError("planned epoch boundary")

    with pytest.raises(RuntimeError, match="planned epoch boundary"):
        _run_tiny_identity_training(
            tmp_path,
            monkeypatch,
            identity=identity,
            resume=True,
            after_epoch_checkpoint=planned_stop,
        )
    planned = checkpoint_events[0]
    checkpoint = tmp_path / "checkpoints" / "training_checkpoint.pt"
    replacement = checkpoint.with_name(".same-bytes-replacement.pt")
    replacement.write_bytes(checkpoint.read_bytes())
    os.replace(replacement, checkpoint)
    assert checkpoint.stat().st_ino != planned["checkpoint_file"]["inode"]

    with pytest.raises(ValueError, match="planned physical leaf"):
        _run_tiny_identity_training(
            tmp_path,
            monkeypatch,
            identity=identity,
            resume=True,
            expected_resume_checkpoint=planned["checkpoint_file"],
        )


def test_legacy_no_identity_api_keeps_typed_v1_terminal_heartbeat(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_tiny_identity_training(
        tmp_path,
        monkeypatch,
        identity=None,
        resume=False,
    )
    heartbeat = json.loads(
        (tmp_path / "checkpoints" / "training_heartbeat.json").read_text()
    )
    assert heartbeat["stage"] == "complete"
    assert heartbeat["schema_version"] == GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V1
    assert "resume_identity" not in heartbeat
    assert "resume_identity_sha256" not in heartbeat


def test_epoch_and_terminal_identity_allow_same_branch_and_reject_other_target(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_zero = _training_resume_identity(target_label=0)
    _run_tiny_identity_training(
        tmp_path,
        monkeypatch,
        identity=target_zero,
        resume=False,
    )
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "training_checkpoint.pt",
        map_location="cpu",
    )
    validated = validate_globalgce_epoch_checkpoint_identity(checkpoint, target_zero)
    heartbeat = json.loads(
        (tmp_path / "checkpoints" / "training_heartbeat.json").read_text()
    )
    assert heartbeat["stage"] == "complete"
    assert heartbeat["schema_version"] == GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2
    assert heartbeat["resume_identity_sha256"] == validated["resume_identity_sha256"]

    _run_tiny_identity_training(
        tmp_path,
        monkeypatch,
        identity=target_zero,
        resume=True,
    )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run_tiny_identity_training(
            tmp_path,
            monkeypatch,
            identity=_training_resume_identity(target_label=2),
            resume=True,
        )
