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
    adopt_external_pair_store_read_only,
    invoke_official_coverage_summary_external,
    summarize_proven_one_cluster_external,
    trace_external_cluster_order,
    validate_adopted_pair_store_read_only,
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


def _official_one_cluster_coverage(
    *, labels, vectors, pairs, radius, theta, recourse_size
):
    import torch

    common_recourse = {}
    centroid_norms = {}
    graph_coverage_map = {}
    rec = torch.tensor(vectors)
    for cluster_label in range(max(labels) + 1):
        covered_graphs = set()
        covered_hashes = set()
        cluster_mask = labels == cluster_label
        cluster_points = rec[cluster_mask]
        cluster_indices = [
            index for index, is_in_cluster in enumerate(cluster_mask) if is_in_cluster
        ]
        centroid = torch.mean(cluster_points, dim=0)
        distances = torch.norm(cluster_points - centroid, dim=-1)
        for index, distance in enumerate(distances):
            if distance < radius:
                original_index, counterfactual_index = pairs[cluster_indices[index]]
                if int(original_index) not in covered_graphs:
                    covered_graphs.add(int(original_index))
                    covered_hashes.add(int(counterfactual_index))
        common_recourse[cluster_label] = covered_graphs
        centroid_norms[cluster_label] = torch.norm(centroid).item()
        graph_coverage_map[cluster_label] = covered_hashes
    filtered = {}
    covered_by = {}
    for label, parents in common_recourse.items():
        if centroid_norms[label] < theta:
            filtered[label] = parents
            for parent in parents:
                covered_by.setdefault(parent, set()).add(label)
    selected = _greedy(
        counterfactual_covering=filtered,
        graphs_covered_by=covered_by,
        k=min(recourse_size, len(filtered)),
    )
    covering, costs, sizes = [], [], []
    cumulative_cost = 0.0
    covered_hashes = set()
    for rank in selected:
        label, cumulative_covered = selected[rank]
        covering.append(cumulative_covered)
        covered_hashes.update(graph_coverage_map[label])
        cumulative_cost += centroid_norms[label]
        costs.append(cumulative_cost)
        sizes.append(len(covered_hashes))
    return covering, costs, sizes


def _one_cluster_fixture(
    tmp_path: Path, values: np.ndarray, pairs: np.ndarray, *, eps: float
):
    from src.baselines.comrecgc import external_memory_dbscan as external_dbscan
    import sklearn

    vectors_path = tmp_path / "vectors.npy"
    pairs_path = tmp_path / "pairs.npy"
    with vectors_path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    with pairs_path.open("wb") as handle:
        np.save(handle, pairs, allow_pickle=False)
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    pair_values = np.load(pairs_path, mmap_mode="r", allow_pickle=False)
    result = external_dbscan.fit_external_memory_dbscan(
        vectors_path=vectors_path,
        work_dir=tmp_path / "dbscan",
        contract=external_dbscan.ExternalDBSCANContract(
            eps=eps,
            min_samples=2,
            query_block_size=2,
            checkpoint_interval_blocks=1,
            max_rss_bytes=_rss_bytes() + 512 * 1024**2,
            expected_sklearn_version=sklearn.__version__,
            shortcut_mode=external_dbscan.ALL_CORE_ONE_COMPONENT_SHORTCUT,
            shortcut_anchor_count=len(values),
            shortcut_query_block_size=2,
            exact_fallback_max_samples=0,
        ),
    )
    assert result.shortcut_proof_path is not None
    return vectors, pair_values, result


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


def _completed_pair_store(tmp_path: Path):
    identity = {
        "dataset": "aids",
        "dataset_fingerprint": "fixture-dataset-sha",
        "dataset_audit_sha256": "a" * 64,
        "pair_order": "candidate_major_parent_minor",
    }
    store = ExternalPairStore(
        root=tmp_path / "source-pair-store",
        scientific_identity=identity,
        max_rss_bytes=_rss_bytes() + 128 * 1024**2,
    )
    store.append(
        chunk_index=0,
        pairs=np.asarray([[0, 0], [1, 0]], dtype=np.int64),
        vectors=np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
        chunk_identity={"candidate_start": 0, "candidate_stop": 1},
    )
    return identity, store.finalize()


def test_pair_store_physical_adoption_is_hash_bound_read_only_and_resumable(
    tmp_path: Path,
) -> None:
    identity, source = _completed_pair_store(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    before = {
        path: external_recourse._file_stat_identity(path)
        for path in (source.manifest_path, source.pairs_path, source.vectors_path)
    }
    adopted = adopt_external_pair_store_read_only(
        source_manifest_path=source.manifest_path,
        adoption_root=tmp_path / "fresh" / "pair_store_adoption",
        expected_scientific_identity=identity,
        proc_root=proc,
    )
    assert adopted.pair_store.manifest_path == source.manifest_path
    assert adopted.pair_store.pairs_path == source.pairs_path
    assert adopted.pair_store.vectors_path == source.vectors_path
    assert sorted(path.name for path in adopted.adoption_manifest_path.parent.iterdir()) == [
        "run_manifest.json"
    ]
    after = {
        path: external_recourse._file_stat_identity(path)
        for path in (source.manifest_path, source.pairs_path, source.vectors_path)
    }
    assert after == before
    resumed = adopt_external_pair_store_read_only(
        source_manifest_path=source.manifest_path,
        adoption_root=adopted.adoption_manifest_path.parent,
        expected_scientific_identity=identity,
        proc_root=proc,
        resume=True,
    )
    assert resumed.adoption_manifest_sha256 == adopted.adoption_manifest_sha256
    validated = validate_adopted_pair_store_read_only(
        adopted.adoption_manifest_path,
        expected_scientific_identity=identity,
        proc_root=proc,
    )
    assert validated.pair_store.pairs_sha256 == source.pairs_sha256


def test_pair_store_adoption_rejects_identity_writer_and_stat_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, source = _completed_pair_store(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    with pytest.raises(Exception, match="identity mismatch"):
        adopt_external_pair_store_read_only(
            source_manifest_path=source.manifest_path,
            adoption_root=tmp_path / "wrong-identity",
            expected_scientific_identity={**identity, "dataset_fingerprint": "drift"},
            proc_root=proc,
        )
    monkeypatch.setattr(
        external_recourse,
        "_find_writable_process_references",
        lambda *_args, **_kwargs: [
            {"pid": 7, "kind": "fd", "fd": 3, "path": str(source.vectors_path)}
        ],
    )
    with pytest.raises(Exception, match="LIVE_WRITER"):
        adopt_external_pair_store_read_only(
            source_manifest_path=source.manifest_path,
            adoption_root=tmp_path / "writer",
            expected_scientific_identity=identity,
            proc_root=proc,
        )
    monkeypatch.setattr(
        external_recourse,
        "_find_writable_process_references",
        lambda *_args, **_kwargs: [],
    )
    adopted = adopt_external_pair_store_read_only(
        source_manifest_path=source.manifest_path,
        adoption_root=tmp_path / "stat-drift",
        expected_scientific_identity=identity,
        proc_root=proc,
    )
    payload = json.loads(adopted.adoption_manifest_path.read_text(encoding="utf-8"))
    payload["source_files"][str(source.vectors_path)]["stat"]["size"] += 1
    adopted.adoption_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="stat drift"):
        validate_adopted_pair_store_read_only(
            adopted.adoption_manifest_path,
            expected_scientific_identity=identity,
            proc_root=proc,
        )


def test_pair_store_adoption_rejects_closed_chunks_without_terminal_promotion(
    tmp_path: Path,
) -> None:
    identity = {
        "dataset": "aids",
        "dataset_fingerprint": "fixture-dataset-sha",
    }
    store = ExternalPairStore(
        root=tmp_path / "chunks-only",
        scientific_identity=identity,
        max_rss_bytes=_rss_bytes() + 128 * 1024**2,
    )
    store.append(
        chunk_index=0,
        pairs=np.asarray([[0, 0], [1, 0]], dtype=np.int64),
        vectors=np.asarray([[0.0], [0.1]], dtype=np.float32),
        chunk_identity={"candidate_start": 0, "candidate_stop": 1},
    )
    checkpoint = tmp_path / "chunks-only/checkpoint.json"
    assert checkpoint.is_file()
    assert not (tmp_path / "chunks-only/run_manifest.json").exists()
    proc = tmp_path / "proc"
    proc.mkdir()
    with pytest.raises(Exception, match="NOT_TERMINALLY_PROMOTED"):
        adopt_external_pair_store_read_only(
            source_manifest_path=checkpoint,
            adoption_root=tmp_path / "fresh",
            expected_scientific_identity=identity,
            proc_root=proc,
        )
    assert not (tmp_path / "fresh").exists()


def test_pair_store_adoption_rejects_manifest_symlink(tmp_path: Path) -> None:
    identity, source = _completed_pair_store(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    alias = tmp_path / "pair-store-alias.json"
    alias.symlink_to(source.manifest_path)
    with pytest.raises(Exception, match="source manifest is a symlink"):
        adopt_external_pair_store_read_only(
            source_manifest_path=alias,
            adoption_root=tmp_path / "fresh",
            expected_scientific_identity=identity,
            proc_root=proc,
        )


def test_pair_store_adoption_rejects_owner_root_symlink(tmp_path: Path) -> None:
    identity, source = _completed_pair_store(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    owner_alias = tmp_path / "owner-alias"
    owner_alias.symlink_to(source.manifest_path.parent, target_is_directory=True)
    with pytest.raises(Exception, match="OWNER_ROOT_IS_SYMLINK"):
        adopt_external_pair_store_read_only(
            source_manifest_path=source.manifest_path,
            source_owner_root=owner_alias,
            adoption_root=tmp_path / "fresh",
            expected_scientific_identity=identity,
            proc_root=proc,
        )


def test_promoted_pair_store_adoption_rejects_any_partial_artifact(
    tmp_path: Path,
) -> None:
    identity, source = _completed_pair_store(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    partial = source.manifest_path.parent / "recourse_vectors.partial.npy"
    partial.write_bytes(b"unfinished-consolidation")

    with pytest.raises(Exception, match="PAIR_STORE_SOURCE_HAS_PARTIAL_ARTIFACTS"):
        adopt_external_pair_store_read_only(
            source_manifest_path=source.manifest_path,
            source_owner_root=source.manifest_path.parent,
            adoption_root=tmp_path / "fresh",
            expected_scientific_identity=identity,
            proc_root=proc,
        )
    assert not (tmp_path / "fresh").exists()


def test_promoted_pair_store_adoption_rejects_writable_sibling_inode(
    tmp_path: Path,
) -> None:
    identity, source = _completed_pair_store(tmp_path)
    sibling = source.manifest_path.parent / "consolidation.audit.json"
    sibling.write_text("{}\n", encoding="utf-8")
    proc = tmp_path / "proc"
    fd = proc / "701/fd/9"
    fd.parent.mkdir(parents=True)
    fd.symlink_to(sibling)
    fdinfo = proc / "701/fdinfo/9"
    fdinfo.parent.mkdir(parents=True)
    fdinfo.write_text("flags:\t02\n", encoding="utf-8")

    with pytest.raises(Exception, match="PAIR_STORE_SOURCE_TREE_HAS_LIVE_WRITER"):
        adopt_external_pair_store_read_only(
            source_manifest_path=source.manifest_path,
            source_owner_root=source.manifest_path.parent,
            adoption_root=tmp_path / "fresh",
            expected_scientific_identity=identity,
            proc_root=proc,
        )
    assert not (tmp_path / "fresh").exists()


def test_promoted_pair_store_adoption_rejects_live_owner_process(
    tmp_path: Path,
) -> None:
    identity, source = _completed_pair_store(tmp_path)
    proc = tmp_path / "proc"
    process = proc / "702"
    process.mkdir(parents=True)
    process.joinpath("cmdline").write_bytes(
        b"python\0run_common_recourse.py\0--output-dir\0"
        + os.fsencode(source.manifest_path.parent)
        + b"\0"
    )

    with pytest.raises(Exception, match="PAIR_STORE_SOURCE_OWNER_PROCESS_ACTIVE"):
        adopt_external_pair_store_read_only(
            source_manifest_path=source.manifest_path,
            source_owner_root=source.manifest_path.parent,
            adoption_root=tmp_path / "fresh",
            expected_scientific_identity=identity,
            proc_root=proc,
        )
    assert not (tmp_path / "fresh").exists()


def test_promoted_pair_store_adoption_allows_fresh_consumer_parent_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, source = _completed_pair_store(tmp_path)
    proc = tmp_path / "proc"
    process = proc / "703"
    process.mkdir(parents=True)
    process.joinpath("cmdline").write_bytes(
        b"python\0run_comrecgc_standardized_continuation.py\0"
        b"--external-pair-store-source-owner-root\0"
        + os.fsencode(source.manifest_path.parent)
        + b"\0"
    )
    monkeypatch.setattr(external_recourse.os, "getppid", lambda: 703)

    adopted = adopt_external_pair_store_read_only(
        source_manifest_path=source.manifest_path,
        source_owner_root=source.manifest_path.parent,
        adoption_root=tmp_path / "fresh",
        expected_scientific_identity=identity,
        proc_root=proc,
    )
    assert adopted.adoption_manifest_path.is_file()


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


def test_proven_one_cluster_stream_is_elementwise_legacy_exact(
    tmp_path: Path,
) -> None:
    import torch

    values = np.asarray(
        [[-0.030], [-0.015], [0.000], [0.015], [0.030]], dtype=np.float32
    )
    pairs = np.asarray(
        [[0, 10], [0, 11], [1, 11], [2, 12], [3, 13]], dtype=np.int64
    )
    radius = 0.020
    theta = 0.100
    vectors, pair_values, dbscan = _one_cluster_fixture(
        tmp_path, values, pairs, eps=radius
    )
    expected_official = _official_one_cluster_coverage(
        labels=np.zeros(len(values), dtype=np.intp),
        vectors=values,
        pairs=pairs,
        radius=radius,
        theta=theta,
        recourse_size=100,
    )
    expected_selected = trace_official_cluster_order(
        labels=np.zeros(len(values), dtype=np.intp),
        recourse_vectors=values,
        pair_indices=[tuple(map(int, row)) for row in pairs],
        radius=radius,
        theta=theta,
        recourse_size=100,
        official_greedy=_greedy,
    )
    actual = summarize_proven_one_cluster_external(
        work_dir=tmp_path / "summary",
        dbscan_manifest_path=dbscan.manifest_path,
        dbscan_manifest_sha256=dbscan.manifest_sha256,
        recourse_vectors=vectors,
        pair_indices=pair_values,
        pairs_sha256=external_recourse._sha256_file(tmp_path / "pairs.npy"),
        radius=radius,
        theta=theta,
        recourse_size=100,
        official_greedy=_greedy,
        torch_module=torch,
        max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        block_size=2,
    )
    assert actual.official_result == expected_official
    assert actual.selected == expected_selected
    manifest = json.loads(actual.manifest_path.read_text(encoding="utf-8"))
    assert manifest["exact_one_cluster_semantics_replayed"] is True
    assert manifest["approximation_used"] is False
    assert manifest["retained_count"] == 3
    assert actual.retained_positions_path is None
    assert actual.retained_vectors_path is None
    assert manifest["large_retained_arrays_materialized"] is False
    assert manifest["retained_vector_bytes_materialized"] == 0
    assert manifest["covered_parent_indices"] == [0, 1, 2]
    assert manifest["coverage_pair_orientation"] == (
        "col0_parent_col1_candidate"
    )
    assert manifest["official_covered_parent_indices"] == [0, 1, 2]
    assert manifest["official_first_counterfactual_indices"] == [11, 12]
    assert manifest["official_radius_counterfactual_indices"] == [11, 12]
    assert manifest["centroid_norm"] == manifest["torch_centroid_norm"]


def test_proven_one_cluster_trace_preserves_python_float_radius_boundary(
    tmp_path: Path,
) -> None:
    import torch

    values = np.asarray([[-0.020], [0.000], [0.020]], dtype=np.float32)
    pairs = np.asarray([[0, 10], [1, 11], [2, 12]], dtype=np.int64)
    vectors, pair_values, dbscan = _one_cluster_fixture(
        tmp_path, values, pairs, eps=0.020
    )
    expected = trace_official_cluster_order(
        labels=np.zeros(3, dtype=np.intp),
        recourse_vectors=values,
        pair_indices=[tuple(map(int, row)) for row in pairs],
        radius=0.020,
        theta=0.100,
        recourse_size=100,
        official_greedy=_greedy,
    )
    actual = summarize_proven_one_cluster_external(
        work_dir=tmp_path / "summary",
        dbscan_manifest_path=dbscan.manifest_path,
        dbscan_manifest_sha256=dbscan.manifest_sha256,
        recourse_vectors=vectors,
        pair_indices=pair_values,
        pairs_sha256=external_recourse._sha256_file(tmp_path / "pairs.npy"),
        radius=0.020,
        theta=0.100,
        recourse_size=100,
        official_greedy=_greedy,
        torch_module=torch,
        max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        block_size=2,
    )
    assert float(np.float32(0.020)) < 0.020
    assert actual.selected == expected
    assert actual.retained_positions_path is None
    assert actual.retained_vectors_path is None
    manifest = json.loads(actual.manifest_path.read_text(encoding="utf-8"))
    assert manifest["retained_count"] == 3
    assert manifest["float64_radius_membership_disagreement_count"] == 2


def test_proven_one_cluster_preserves_official_empty_coverage_corner(
    tmp_path: Path,
) -> None:
    import torch

    angles = np.arange(64, dtype=np.float64) * (2.0 * np.pi / 64.0)
    values = np.zeros((64, 64), dtype=np.float32)
    values[:, :2] = np.column_stack((np.cos(angles), np.sin(angles))).astype(
        np.float32
    )
    pairs = np.column_stack(
        (np.arange(64, dtype=np.int64), np.arange(64, dtype=np.int64))
    )
    radius = 0.11
    vectors, pair_values, dbscan = _one_cluster_fixture(
        tmp_path, values, pairs, eps=radius
    )
    expected = _official_one_cluster_coverage(
        labels=np.zeros(64, dtype=np.intp),
        vectors=values,
        pairs=pairs,
        radius=radius,
        theta=0.1,
        recourse_size=100,
    )
    actual = summarize_proven_one_cluster_external(
        work_dir=tmp_path / "summary",
        dbscan_manifest_path=dbscan.manifest_path,
        dbscan_manifest_sha256=dbscan.manifest_sha256,
        recourse_vectors=vectors,
        pair_indices=pair_values,
        pairs_sha256=external_recourse._sha256_file(tmp_path / "pairs.npy"),
        radius=radius,
        theta=0.1,
        recourse_size=100,
        official_greedy=_greedy,
        torch_module=torch,
        max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        block_size=7,
    )
    assert actual.official_result == expected
    assert actual.official_result[0] == [0]
    assert actual.official_result[2] == [0]
    assert actual.selected == []
    assert actual.retained_positions_path is None
    assert actual.retained_vectors_path is None


def test_proven_one_cluster_resume_is_exact_and_tamper_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    values = np.asarray(
        [[-0.030], [-0.015], [0.000], [0.015], [0.030]], dtype=np.float32
    )
    pairs = np.asarray(
        [[0, 10], [0, 11], [1, 11], [2, 12], [3, 13]], dtype=np.int64
    )
    vectors, pair_values, dbscan = _one_cluster_fixture(
        tmp_path, values, pairs, eps=0.020
    )
    kwargs = {
        "dbscan_manifest_path": dbscan.manifest_path,
        "dbscan_manifest_sha256": dbscan.manifest_sha256,
        "recourse_vectors": vectors,
        "pair_indices": pair_values,
        "pairs_sha256": external_recourse._sha256_file(tmp_path / "pairs.npy"),
        "radius": 0.020,
        "theta": 0.100,
        "recourse_size": 100,
        "official_greedy": _greedy,
        "torch_module": torch,
        "max_rss_bytes": _rss_bytes() + 512 * 1024**2,
        "block_size": 2,
    }
    original_checkpoint = external_recourse._summary_checkpoint
    interrupted = {"done": False}

    def interrupt(path, **checkpoint):
        result = original_checkpoint(path, **checkpoint)
        if (
            not interrupted["done"]
            and checkpoint["phase"] == "trace_mask"
            and checkpoint["next_offset"] == 2
        ):
            interrupted["done"] = True
            raise RuntimeError("summary interruption")
        return result

    monkeypatch.setattr(external_recourse, "_summary_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="summary interruption"):
        summarize_proven_one_cluster_external(
            work_dir=tmp_path / "resumed", **kwargs
        )
    monkeypatch.setattr(
        external_recourse, "_summary_checkpoint", original_checkpoint
    )
    checkpoint_path = tmp_path / "resumed/checkpoint.json"
    checkpoint_bytes = checkpoint_path.read_bytes()
    tampered = json.loads(checkpoint_bytes)
    assert tampered["phase"] == "trace_mask"
    tampered["next_offset"] = len(values)
    checkpoint_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(Exception, match="trace-mask committed prefix"):
        summarize_proven_one_cluster_external(
            work_dir=tmp_path / "resumed", resume=True, **kwargs
        )
    checkpoint_path.write_bytes(checkpoint_bytes)
    resumed = summarize_proven_one_cluster_external(
        work_dir=tmp_path / "resumed", resume=True, **kwargs
    )
    reference = summarize_proven_one_cluster_external(
        work_dir=tmp_path / "reference", **kwargs
    )
    assert resumed.official_result == reference.official_result
    assert resumed.selected == reference.selected
    assert external_recourse._sha256_file(resumed.retained_mask_path) == (
        external_recourse._sha256_file(reference.retained_mask_path)
    )
    assert resumed.retained_vectors_path is None
    assert reference.retained_vectors_path is None

    manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
    centroid_path = Path(manifest["numpy_centroid_path"])
    centroid = np.load(centroid_path, allow_pickle=False)
    with centroid_path.open("wb") as handle:
        np.save(handle, centroid + np.float32(0.1), allow_pickle=False)
    with pytest.raises(Exception, match="centroid checksum"):
        summarize_proven_one_cluster_external(
            work_dir=tmp_path / "resumed", resume=True, **kwargs
        )


def test_proven_one_cluster_rejects_nonproof_and_low_rss(tmp_path: Path) -> None:
    import torch

    values = np.zeros((4, 1), dtype=np.float32)
    pairs = np.column_stack(
        (np.arange(4, dtype=np.int64), np.arange(4, dtype=np.int64))
    )
    vectors, pair_values, dbscan = _one_cluster_fixture(
        tmp_path, values, pairs, eps=0.020
    )
    with pytest.raises(Exception, match="RSS"):
        summarize_proven_one_cluster_external(
            work_dir=tmp_path / "summary",
            dbscan_manifest_path=dbscan.manifest_path,
            dbscan_manifest_sha256=dbscan.manifest_sha256,
            recourse_vectors=vectors,
            pair_indices=pair_values,
            pairs_sha256=external_recourse._sha256_file(tmp_path / "pairs.npy"),
            radius=0.020,
            theta=0.100,
            recourse_size=100,
            official_greedy=_greedy,
            torch_module=torch,
            max_rss_bytes=_rss_bytes() + 1024,
            block_size=2,
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
