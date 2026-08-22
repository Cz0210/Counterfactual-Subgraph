from __future__ import annotations

from concurrent.futures import Executor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.bace_gine_native_adapter import (
    BACEFrozenGINENativeGraphAdapter,
    LEGACY_PREPROCESS_ENGINE,
)
from src.baselines.comrecgc.bace_preprocessing import (
    PREPROCESS_ENGINE,
    ordered_bounded_submit,
)
from src.baselines.gcfexplainer_bace_adapter import (
    BACE_FEATURE_ATOMIC_NUMBERS,
    BACEGraphSchema,
)
from src.baselines.gcfexplainer_mutagenicity_adapter import (
    StrictMolecule,
    encode_source_graph,
)
from src.data.molecular_graph_featurizer import default_molecular_feature_schema


CHECKPOINT_ID = "d" * 64
GRAPH_SCHEMA = BACEGraphSchema(
    atom_vocabulary=BACE_FEATURE_ATOMIC_NUMBERS,
    feature_atomic_numbers=BACE_FEATURE_ATOMIC_NUMBERS,
    formal_charge_vocabulary=(-1, 0, 1),
    aromaticity_vocabulary=(False, True),
    bond_type_vocabulary=("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"),
    max_num_nodes=256,
)


@pytest.fixture()
def production_like_source_record() -> dict:
    """Small explicit-H BACE graph using the production codec and sidecars."""

    rdkit = pytest.importorskip("rdkit.Chem")
    molecule = rdkit.MolFromSmiles("CCO")
    assert molecule is not None
    canonical = rdkit.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    return encode_source_graph(
        StrictMolecule(
            molecule_id="BACE_PRODUCTION_LIKE_0001",
            smiles="CCO",
            canonical_smiles=canonical,
            label=1,
            split="train",
            semantic_label="active",
            source_row_index=0,
            source_path="production_like_fixture",
        ),
        GRAPH_SCHEMA,
    )


def _native_graph(record: dict, *, source_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        x=deepcopy(record["x"]),
        edge_index=deepcopy(record["edge_index"]),
        num_nodes=int(record["num_nodes"]),
        comrecgc_node_origin=list(range(int(record["num_nodes"]))),
        comrecgc_source_index=source_index,
    )


def _patch_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    torch = pytest.importorskip("torch")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(hidden_dim=3)
            self.classifier = torch.nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                self.classifier.weight.copy_(
                    torch.tensor([[0.25, 0.5, 0.75], [-0.5, 0.25, 0.125]])
                )

        def encode_graph(self, batch: object) -> object:
            return torch.tensor(
                [[1.0, 2.0, 3.0] for _ in range(batch.size)],
                dtype=self.anchor.dtype,
            )

    class FakeBatch:
        def __init__(self, size: int) -> None:
            self.size = size

        def to(self, _device: object) -> "FakeBatch":
            return self

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"frozen-gine")
    metadata = {
        "model_card": {
            "dataset": "bace",
            "backbone": "gine",
            "num_classes": 2,
            "source_label": 1,
            "rf_oracle_used": False,
        },
        "checkpoint_id": CHECKPOINT_ID,
        "temperature_scaling": {"temperature": 1.0},
        "feature_schema": default_molecular_feature_schema(),
    }
    monkeypatch.setattr(
        "src.baselines.bace_gine_native_adapter.load_gnn_checkpoint_bundle",
        lambda _root, device: (FakeModel(), metadata),
    )
    monkeypatch.setattr(
        "src.baselines.bace_gine_native_adapter.collate_molecular_graphs",
        lambda rows, edge_feature_dim: FakeBatch(len(rows)),
    )
    return checkpoint


def test_inline_optimized_preprocessing_is_exact_and_cacheable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    production_like_source_record: dict,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = _patch_checkpoint(monkeypatch, tmp_path)
    legacy = BACEFrozenGINENativeGraphAdapter(
        checkpoint,
        source_records=[production_like_source_record],
        graph_schema=GRAPH_SCHEMA,
        device="cpu",
        preprocess_engine=LEGACY_PREPROCESS_ENGINE,
    )
    optimized = BACEFrozenGINENativeGraphAdapter(
        checkpoint,
        source_records=[production_like_source_record],
        graph_schema=GRAPH_SCHEMA,
        device="cpu",
        preprocess_engine=PREPROCESS_ENGINE,
        preprocess_workers=0,
        source_cache_capacity=4,
        candidate_cache_capacity=4,
    )
    graph = _native_graph(production_like_source_record)
    legacy_output = legacy([graph])
    optimized_output = optimized([graph])
    assert all(
        torch.equal(left, right)
        for left, right in zip(legacy_output, optimized_output, strict=True)
    )
    second_output = optimized([graph])
    assert all(
        torch.equal(left, right)
        for left, right in zip(optimized_output, second_output, strict=True)
    )
    provenance = optimized.provenance()
    assert provenance["preprocess_order_preserved"] is True
    assert provenance["preprocess_rng_calls_added"] == 0
    assert provenance["preprocess_stats"]["source_cache_hit_count"] == 1
    assert provenance["preprocess_stats"]["unique_preprocess_count"] == 1


def test_cache_key_keeps_parent_metadata_out_of_graph_identity_but_bound_to_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    production_like_source_record: dict,
) -> None:
    checkpoint = _patch_checkpoint(monkeypatch, tmp_path)
    second = deepcopy(production_like_source_record)
    second["molecule_id"] = "BACE_PRODUCTION_LIKE_0002"
    adapter = BACEFrozenGINENativeGraphAdapter(
        checkpoint,
        source_records=[production_like_source_record, second],
        graph_schema=GRAPH_SCHEMA,
        device="cpu",
        preprocess_engine=PREPROCESS_ENGINE,
        source_cache_capacity=4,
    )
    first_request = adapter._request(_native_graph(production_like_source_record, source_index=0))
    second_request = adapter._request(_native_graph(second, source_index=1))
    first_content = adapter._content_sha256(
        x=first_request.x,
        edge_index=first_request.edge_index,
        num_nodes=first_request.num_nodes,
    )
    second_content = adapter._content_sha256(
        x=second_request.x,
        edge_index=second_request.edge_index,
        num_nodes=second_request.num_nodes,
    )
    assert first_content == second_content
    assert first_request.cache_key != second_request.cache_key


def test_spawn_process_pool_preserves_batch_outputs_and_coalesces_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    production_like_source_record: dict,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = _patch_checkpoint(monkeypatch, tmp_path)
    inline = BACEFrozenGINENativeGraphAdapter(
        checkpoint,
        source_records=[production_like_source_record],
        graph_schema=GRAPH_SCHEMA,
        device="cpu",
        preprocess_engine=PREPROCESS_ENGINE,
        preprocess_workers=0,
    )
    parallel = BACEFrozenGINENativeGraphAdapter(
        checkpoint,
        source_records=[production_like_source_record],
        graph_schema=GRAPH_SCHEMA,
        device="cpu",
        preprocess_engine=PREPROCESS_ENGINE,
        preprocess_workers=2,
        preprocess_max_inflight=2,
        source_cache_capacity=4,
    )
    graphs = [
        _native_graph(production_like_source_record),
        _native_graph(production_like_source_record),
    ]
    try:
        expected = inline(graphs)
        try:
            observed = parallel(graphs)
        except (PermissionError, NotImplementedError) as exc:
            pytest.skip(
                "Managed local sandbox cannot create multiprocessing "
                f"semaphores: {type(exc).__name__}: {exc}"
            )
    finally:
        parallel.close()
    assert all(
        torch.equal(left, right)
        for left, right in zip(expected, observed, strict=True)
    )
    stats = parallel.provenance()["preprocess_stats"]
    assert stats["process_pool_submitted_count"] == 1
    assert stats["within_batch_coalesced_count"] == 1


def test_ordered_bounded_submit_never_exposes_completion_order() -> None:
    class DeferredFuture:
        def __init__(self, owner: "TrackingExecutor", function: object, value: int) -> None:
            self.owner = owner
            self.function = function
            self.value = value

        def result(self) -> int:
            self.owner.outstanding -= 1
            return self.function(self.value)  # type: ignore[operator]

    class TrackingExecutor(Executor):
        def __init__(self) -> None:
            self.outstanding = 0
            self.max_outstanding = 0

        def submit(self, fn: object, /, *args: object, **kwargs: object) -> DeferredFuture:
            assert not kwargs
            self.outstanding += 1
            self.max_outstanding = max(self.max_outstanding, self.outstanding)
            return DeferredFuture(self, fn, int(args[0]))

    executor = TrackingExecutor()
    observed = list(
        ordered_bounded_submit(
            executor, lambda value: value * 10, [3, 1, 4, 2], max_inflight=2
        )
    )
    assert observed == [30, 10, 40, 20]
    assert executor.max_outstanding == 2


def test_legacy_engine_rejects_hidden_worker_or_cache_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    production_like_source_record: dict,
) -> None:
    checkpoint = _patch_checkpoint(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Legacy BACE preprocessing rejects"):
        BACEFrozenGINENativeGraphAdapter(
            checkpoint,
            source_records=[production_like_source_record],
            graph_schema=GRAPH_SCHEMA,
            device="cpu",
            preprocess_engine=LEGACY_PREPROCESS_ENGINE,
            preprocess_workers=1,
        )
