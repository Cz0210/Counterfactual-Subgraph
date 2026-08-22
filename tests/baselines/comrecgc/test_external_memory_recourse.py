from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.baselines.comrecgc import external_memory_recourse as external_recourse
from src.baselines.comrecgc.external_memory_dbscan import _rss_bytes
from src.baselines.comrecgc.external_memory_recourse import (
    ExternalPairStore,
    invoke_official_coverage_summary_external,
    trace_external_cluster_order,
)
from src.baselines.comrecgc.recourse import trace_official_cluster_order


def _greedy(counterfactual_covering, graphs_covered_by, k):
    selected = {}
    covered = set()
    for rank in range(1, k + 1):
        cluster, gains = max(
            counterfactual_covering.items(), key=lambda item: len(item[1])
        )
        covered.update(gains)
        counterfactual_covering.pop(cluster)
        for parent in gains:
            for other in graphs_covered_by[parent] - {cluster}:
                if other in counterfactual_covering:
                    counterfactual_covering[other].discard(parent)
        selected[rank] = (cluster, len(covered))
    return selected


def test_pair_store_resume_preserves_global_pair_order_and_hash(tmp_path: Path) -> None:
    identity = {"dataset": "aids", "theta": 0.1, "seed": 0}
    budget = _rss_bytes() + 256 * 1024**2
    first = ExternalPairStore(
        root=tmp_path / "resumed",
        scientific_identity=identity,
        max_rss_bytes=budget,
    )
    first.append(
        chunk_index=0,
        pairs=np.asarray([[0, 0], [2, 0]], dtype=np.int64),
        vectors=np.asarray([[0.1, 0.0], [0.2, 0.0]], dtype=np.float32),
        chunk_identity={"candidate_start": 0, "candidate_stop": 1},
    )

    resumed = ExternalPairStore(
        root=tmp_path / "resumed",
        scientific_identity=identity,
        max_rss_bytes=budget,
        resume=True,
    )
    assert resumed.next_chunk_index == 1
    assert resumed.verify_completed_chunk(
        chunk_index=0,
        chunk_identity={"candidate_start": 0, "candidate_stop": 1},
    ) == 2
    resumed.append(
        chunk_index=1,
        pairs=np.asarray([[1, 1], [3, 1]], dtype=np.int64),
        vectors=np.asarray([[0.3, 0.0], [0.4, 0.0]], dtype=np.float32),
        chunk_identity={"candidate_start": 1, "candidate_stop": 2},
    )
    result = resumed.finalize()

    one_shot = ExternalPairStore(
        root=tmp_path / "one-shot",
        scientific_identity=identity,
        max_rss_bytes=budget,
    )
    one_shot.append(
        chunk_index=0,
        pairs=np.asarray([[0, 0], [2, 0]], dtype=np.int64),
        vectors=np.asarray([[0.1, 0.0], [0.2, 0.0]], dtype=np.float32),
        chunk_identity={"candidate_start": 0, "candidate_stop": 1},
    )
    one_shot.append(
        chunk_index=1,
        pairs=np.asarray([[1, 1], [3, 1]], dtype=np.int64),
        vectors=np.asarray([[0.3, 0.0], [0.4, 0.0]], dtype=np.float32),
        chunk_identity={"candidate_start": 1, "candidate_stop": 2},
    )
    reference = one_shot.finalize()
    assert np.array_equal(
        np.load(result.pairs_path, allow_pickle=False),
        np.asarray([[0, 0], [2, 0], [1, 1], [3, 1]], dtype=np.int64),
    )
    assert result.pairs_sha256 == reference.pairs_sha256
    assert result.vectors_sha256 == reference.vectors_sha256
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_major_parent_minor_order"] is True


def test_external_trace_selected_rows_are_elementwise_legacy_exact(
    tmp_path: Path,
) -> None:
    vectors = np.asarray(
        [
            [0.010, 0.000],
            [0.012, 0.001],
            [0.014, 0.000],
            [0.050, 0.000],
            [0.052, 0.001],
            [0.054, 0.000],
            [0.500, 0.500],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1, -1], dtype=np.intp)
    pairs = np.asarray(
        [(0, 10), (1, 11), (2, 12), (1, 20), (3, 21), (4, 22), (5, 30)],
        dtype=np.int64,
    )
    expected = trace_official_cluster_order(
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=[tuple(map(int, row)) for row in pairs],
        radius=0.02,
        theta=0.1,
        recourse_size=100,
        official_greedy=_greedy,
    )
    actual, audit = trace_external_cluster_order(
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=pairs,
        radius=0.02,
        theta=0.1,
        recourse_size=100,
        official_greedy=_greedy,
        max_rss_bytes=_rss_bytes() + 256 * 1024**2,
    )
    assert actual == expected
    assert audit["legacy_numpy_reduction_order_preserved"] is True
    assert audit["official_greedy_invoked"] is True


def test_pair_store_rejects_parent_order_drift_within_candidate(tmp_path: Path) -> None:
    store = ExternalPairStore(
        root=tmp_path / "store",
        scientific_identity={"dataset": "aids"},
        max_rss_bytes=_rss_bytes() + 128 * 1024**2,
    )
    import pytest

    with pytest.raises(Exception, match="order"):
        store.append(
            chunk_index=0,
            pairs=np.asarray([[2, 0], [1, 0]], dtype=np.int64),
            vectors=np.asarray([[0.0], [0.1]], dtype=np.float32),
            chunk_identity={"candidate_start": 0},
        )


def test_pair_store_resume_rejects_dataset_content_fingerprint_drift(
    tmp_path: Path,
) -> None:
    budget = _rss_bytes() + 128 * 1024**2
    store = ExternalPairStore(
        root=tmp_path / "store",
        scientific_identity={
            "dataset": "aids",
            "dataset_fingerprint": "graphs-summary-source-hash-v1",
            "dataset_audit_sha256": "a" * 64,
        },
        max_rss_bytes=budget,
    )
    store.append(
        chunk_index=0,
        pairs=np.asarray([[0, 0]], dtype=np.int64),
        vectors=np.asarray([[0.0]], dtype=np.float32),
        chunk_identity={"candidate_start": 0},
    )
    with pytest.raises(Exception, match="identity mismatch"):
        ExternalPairStore(
            root=tmp_path / "store",
            scientific_identity={
                "dataset": "aids",
                "dataset_fingerprint": "graphs-summary-source-hash-v2",
                "dataset_audit_sha256": "b" * 64,
            },
            max_rss_bytes=budget,
            resume=True,
        )


def test_official_coverage_receives_zero_copy_vector_view(tmp_path: Path) -> None:
    import torch

    vectors_path = tmp_path / "vectors.npy"
    with vectors_path.open("wb") as handle:
        np.save(
            handle,
            np.asarray([[0.01, 0.0], [0.02, 0.0], [0.03, 0.0]], dtype=np.float32),
            allow_pickle=False,
        )
    vectors = np.load(vectors_path, mmap_mode="r+", allow_pickle=False)
    labels = np.asarray([0, 0, 0], dtype=np.intp)
    pairs = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)

    def official(**kwargs):
        received = kwargs["rec"]
        assert np.shares_memory(received.numpy(), vectors)
        assert kwargs["idxs"] is pairs
        return ([3], [0.1], [2])

    result, audit = invoke_official_coverage_summary_external(
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=pairs,
        radius=0.02,
        theta=0.1,
        recourse_size=100,
        official_coverage_summary=official,
        torch_module=torch,
        max_rss_bytes=_rss_bytes() + 256 * 1024**2,
    )
    assert result == ([3], [0.1], [2])
    assert audit["official_coverage_summary_invoked"] is True
    assert audit["full_vector_tensor_copy_created"] is False


def test_cluster_summary_rss_gate_covers_largest_cluster_copy() -> None:
    labels = np.zeros(100, dtype=np.intp)
    vectors = np.zeros((100, 32), dtype=np.float32)
    pairs = np.column_stack(
        (np.arange(100, dtype=np.int64), np.zeros(100, dtype=np.int64))
    )
    with pytest.raises(Exception, match="RSS"):
        trace_external_cluster_order(
            labels=labels,
            recourse_vectors=vectors,
            pair_indices=pairs,
            radius=0.02,
            theta=0.1,
            recourse_size=100,
            official_greedy=_greedy,
            max_rss_bytes=_rss_bytes() + 1024,
        )


def test_pair_consolidation_two_phase_recovers_after_first_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {"dataset": "aids", "dataset_fingerprint": "frozen"}
    budget = _rss_bytes() + 256 * 1024**2
    root = tmp_path / "crash"
    store = ExternalPairStore(
        root=root, scientific_identity=identity, max_rss_bytes=budget
    )
    store.append(
        chunk_index=0,
        pairs=np.asarray([[0, 0], [1, 0]], dtype=np.int64),
        vectors=np.asarray([[0.1], [0.2]], dtype=np.float32),
        chunk_identity={"candidate_start": 0, "candidate_stop": 1},
    )
    original_atomic = external_recourse._atomic_json

    def interrupt_after_ready(path, payload):
        original_atomic(path, payload)
        if path == store.state_path and payload.get("phase") == "consolidation_ready":
            raise RuntimeError("crash-after-consolidation-ready")

    monkeypatch.setattr(external_recourse, "_atomic_json", interrupt_after_ready)
    with pytest.raises(RuntimeError, match="crash-after-consolidation-ready"):
        store.finalize()
    state = json.loads((root / "checkpoint.json").read_text())
    assert state["phase"] == "consolidation_ready"
    os.replace(root / "pair_indices.partial.npy", root / "pair_indices.npy")

    monkeypatch.setattr(external_recourse, "_atomic_json", original_atomic)
    resumed = ExternalPairStore(
        root=root,
        scientific_identity=identity,
        max_rss_bytes=budget,
        resume=True,
    ).finalize()
    assert np.array_equal(
        np.load(resumed.pairs_path), np.asarray([[0, 0], [1, 0]], dtype=np.int64)
    )
    assert np.array_equal(
        np.load(resumed.vectors_path), np.asarray([[0.1], [0.2]], dtype=np.float32)
    )


def test_pair_consolidation_rejects_tampered_promoted_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {"dataset": "aids", "dataset_fingerprint": "frozen"}
    budget = _rss_bytes() + 256 * 1024**2
    root = tmp_path / "tampered"
    store = ExternalPairStore(
        root=root, scientific_identity=identity, max_rss_bytes=budget
    )
    store.append(
        chunk_index=0,
        pairs=np.asarray([[0, 0]], dtype=np.int64),
        vectors=np.asarray([[0.1]], dtype=np.float32),
        chunk_identity={"candidate_start": 0, "candidate_stop": 1},
    )
    original_atomic = external_recourse._atomic_json

    def interrupt_after_ready(path, payload):
        original_atomic(path, payload)
        if path == store.state_path and payload.get("phase") == "consolidation_ready":
            raise RuntimeError("crash-after-consolidation-ready")

    monkeypatch.setattr(external_recourse, "_atomic_json", interrupt_after_ready)
    with pytest.raises(RuntimeError, match="crash-after-consolidation-ready"):
        store.finalize()
    os.replace(root / "pair_indices.partial.npy", root / "pair_indices.npy")
    with (root / "pair_indices.npy").open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        byte = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([byte[0] ^ 1]))
    monkeypatch.setattr(external_recourse, "_atomic_json", original_atomic)
    resumed = ExternalPairStore(
        root=root,
        scientific_identity=identity,
        max_rss_bytes=budget,
        resume=True,
    )
    with pytest.raises(Exception, match="checksum mismatch"):
        resumed.finalize()
