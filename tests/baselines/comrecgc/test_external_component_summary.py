from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from src.baselines.comrecgc import external_component_summary as summary
from src.baselines.comrecgc import external_memory_dbscan as dbscan
from src.baselines.comrecgc.close_pair_view import (
    ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
    FILTER_OPERATOR,
    NORMALIZED_DISTANCE_CONTRACT,
    PAIR_ORDER,
    PAIR_ORIENTATION,
    SCALE_CONTRACT,
    ThetaClosePairContract,
    materialize_theta_close_pair_view,
)
from src.baselines.comrecgc.recourse import trace_official_cluster_order


torch = pytest.importorskip("torch")
sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import DBSCAN  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _greedy(counterfactual_covering, graphs_covered_by, k):
    selected = {}
    covered = set()
    for rank in range(1, k + 1):
        label, gains = max(
            counterfactual_covering.items(), key=lambda item: len(item[1])
        )
        covered.update(gains)
        counterfactual_covering.pop(label)
        for parent in gains:
            for other in graphs_covered_by[parent] - {label}:
                if other in counterfactual_covering:
                    counterfactual_covering[other].discard(parent)
        selected[rank] = (label, len(covered))
    return selected


def _official_coverage(*, labels, vectors, pairs, radius, theta, recourse_size):
    rec = torch.from_numpy(vectors)
    common = {}
    norms = {}
    first_candidates = {}
    for label in range(int(labels.max()) + 1):
        mask = labels == label
        positions = np.flatnonzero(mask)
        points = rec[mask]
        center = torch.mean(points, dim=0)
        distances = torch.norm(points - center, dim=-1)
        parents = set()
        candidates = set()
        for local, distance in enumerate(distances):
            if distance < radius:
                parent, candidate = pairs[int(positions[local])]
                if int(parent) not in parents:
                    parents.add(int(parent))
                    candidates.add(int(candidate))
        common[label] = parents
        norms[label] = torch.norm(center).item()
        first_candidates[label] = candidates
    filtered = {
        label: set(parents)
        for label, parents in common.items()
        if norms[label] < theta
    }
    covered_by = {}
    for label, parents in filtered.items():
        for parent in parents:
            covered_by.setdefault(parent, set()).add(label)
    selected = _greedy(
        counterfactual_covering=filtered,
        graphs_covered_by=covered_by,
        k=min(recourse_size, len(filtered)),
    )
    covering = []
    costs = []
    sizes = []
    cost = 0.0
    candidates = set()
    for rank in selected:
        label, cumulative = selected[rank]
        covering.append(cumulative)
        cost += norms[label]
        costs.append(cost)
        candidates.update(first_candidates[label])
        sizes.append(len(candidates))
    return (covering, costs, sizes)


def _fixture(
    tmp_path: Path,
    values: np.ndarray,
    pairs: np.ndarray,
    *,
    eps: float = 0.02,
    block_size: int = 2,
):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        matrix = np.zeros((len(values), 64), dtype=np.float32)
        matrix[:, 0] = values
        values = matrix
    vector_path = tmp_path / "vectors.npy"
    pair_path = tmp_path / "pairs.npy"
    np.save(vector_path, values, allow_pickle=False)
    np.save(pair_path, np.asarray(pairs, dtype=np.int64), allow_pickle=False)
    result = dbscan.fit_external_memory_dbscan(
        vectors_path=vector_path,
        work_dir=tmp_path / "dbscan",
        contract=dbscan.ExternalDBSCANContract(
            eps=float(eps),
            min_samples=3,
            query_block_size=2,
            checkpoint_interval_blocks=1,
            max_rss_bytes=dbscan._rss_bytes() + 512 * 1024**2,
            expected_sklearn_version=sklearn.__version__,
            shortcut_mode=dbscan.ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
            shortcut_seed_count=3,
            shortcut_failure_cap=100,
            shortcut_query_block_size=block_size,
            exact_fallback_max_samples=0,
        ),
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["clustering_path"] == (
        dbscan.ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
    )
    return (
        np.load(vector_path, mmap_mode="r", allow_pickle=False),
        np.load(pair_path, mmap_mode="r", allow_pickle=False),
        result,
        pair_path,
    )


@pytest.mark.parametrize(
    "coordinates",
    [
        [0.000, 0.001, 0.002, 0.050, 0.051, 0.052],
        [
            0.000,
            0.001,
            0.002,
            0.050,
            0.051,
            0.052,
            0.100,
            0.101,
            0.102,
        ],
    ],
)
def test_all_core_multi_component_stream_is_official_and_trace_exact(
    tmp_path: Path, coordinates: list[float]
) -> None:
    pairs = np.asarray(
        [[index % 4, 10 + index // 2] for index in range(len(coordinates))],
        dtype=np.int64,
    )
    vectors, pair_values, result, pair_path = _fixture(
        tmp_path, np.asarray(coordinates), pairs
    )
    labels = np.load(result.labels_path, mmap_mode="r", allow_pickle=False)
    expected_labels = DBSCAN(eps=0.02, min_samples=3).fit_predict(vectors)
    assert np.array_equal(labels, expected_labels)
    expected_official = _official_coverage(
        labels=labels,
        vectors=np.asarray(vectors),
        pairs=pairs,
        radius=0.02,
        theta=0.2,
        recourse_size=100,
    )
    expected_selected = trace_official_cluster_order(
        labels=labels,
        recourse_vectors=np.asarray(vectors),
        pair_indices=[tuple(map(int, row)) for row in pairs],
        radius=0.02,
        theta=0.2,
        recourse_size=100,
        official_greedy=_greedy,
    )
    actual = summary.summarize_proven_all_core_components_external(
        work_dir=tmp_path / "summary",
        dbscan_manifest_path=result.manifest_path,
        dbscan_manifest_sha256=result.manifest_sha256,
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=pair_values,
        pairs_sha256=_sha(pair_path),
        radius=0.02,
        theta=0.2,
        recourse_size=100,
        official_greedy=_greedy,
        torch_module=torch,
        max_rss_bytes=dbscan._rss_bytes() + 512 * 1024**2,
        block_size=2,
    )
    assert actual.official_result == expected_official
    assert actual.selected == expected_selected
    manifest = json.loads(actual.manifest_path.read_text())
    assert manifest["cluster_count"] == len(set(expected_labels.tolist()))
    assert manifest["large_cluster_advanced_index_copy"] is False
    assert manifest["full_cluster_vector_bytes_materialized"] == 0
    assert manifest["terminal_full_replay_complete"] is True
    reopened = summary.validate_proven_all_core_component_summary(
        actual.manifest_path,
        torch_module=torch,
        full_replay=True,
    )
    assert reopened.manifest_sha256 == actual.manifest_sha256


def test_all_core_component_summary_strict_boundaries_and_plateau(
    tmp_path: Path,
) -> None:
    # Exact float32 +/- delta members are excluded.  The second cluster center
    # is exactly theta and is therefore ineligible under strict ``<``.
    coordinates = np.asarray(
        [
            -0.03125,
            -0.03125,
            0.0,
            0.03125,
            0.03125,
            0.09375,
            0.09375,
            0.125,
            0.15625,
            0.15625,
        ],
        dtype=np.float32,
    )
    pairs = np.asarray(
        [
            [0, 10],
            [0, 10],
            [1, 11],
            [2, 12],
            [2, 12],
            [3, 20],
            [3, 20],
            [4, 21],
            [5, 22],
            [5, 22],
        ],
        dtype=np.int64,
    )
    vectors, pair_values, result, pair_path = _fixture(
        tmp_path, coordinates, pairs, eps=0.03125
    )
    labels = np.load(result.labels_path, mmap_mode="r", allow_pickle=False)
    actual = summary.summarize_proven_all_core_components_external(
        work_dir=tmp_path / "summary",
        dbscan_manifest_path=result.manifest_path,
        dbscan_manifest_sha256=result.manifest_sha256,
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=pair_values,
        pairs_sha256=_sha(pair_path),
        radius=0.03125,
        theta=0.125,
        recourse_size=100,
        official_greedy=_greedy,
        torch_module=torch,
        max_rss_bytes=dbscan._rss_bytes() + 512 * 1024**2,
        block_size=3,
    )
    manifest = json.loads(actual.manifest_path.read_text())
    first, second = manifest["cluster_summaries"]
    assert first["count_exactly_at_delta"] == 4
    assert first["official_within_centroid_radius_count"] == 1
    assert second["count_exactly_at_theta"] == 1
    assert second["centroid_norm_lt_theta"] is False
    assert len(actual.selected) == 1
    assert actual.selected[0]["cluster_id"] == 0
    assert actual.selected[0]["representative_source_index"] == 1
    assert actual.selected[0]["representative_counterfactual_index"] == 11
    assert len(actual.selected) < 100


@pytest.mark.parametrize(
    "stable_center,radius_disagreements",
    [(0.2, 0), (0.1, 1)],
)
def test_all_core_component_summary_numeric_decision_disagreement_fails_closed(
    stable_center: float, radius_disagreements: int
) -> None:
    membership = (
        [{0: 0}],
        [{0}],
        [{0}],
        [{0}],
        np.asarray([3], dtype=np.int64),
        np.asarray([3], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([radius_disagreements], dtype=np.int64),
    )
    with pytest.raises(
        dbscan.ExternalMemoryDBSCANError,
        match="PROJECT_EXTENSION_NUMERIC_DECISION_DISAGREEMENT",
    ):
        summary._results_from_replay(
            cluster_counts=np.asarray([3], dtype=np.int64),
            official_centroids=np.asarray([[0.1]], dtype=np.float32),
            numpy_centroids=np.asarray([[0.1]], dtype=np.float32),
            stable_centroids=np.asarray([[stable_center]], dtype=np.float64),
            membership=membership,
            retained_counts=np.asarray([3], dtype=np.int64),
            medoid_positions=np.asarray([0], dtype=np.int64),
            medoid_distances=[0.0],
            pairs=np.asarray([[0, 0]], dtype=np.int64),
            radius=0.02,
            theta=0.15,
            recourse_size=1,
            official_greedy=_greedy,
            torch_module=torch,
        )


def test_all_core_component_summary_resume_tamper_and_implicit_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinates = np.asarray(
        [0.000, 0.001, 0.002, 0.050, 0.051, 0.052], dtype=np.float32
    )
    pairs = np.asarray(
        [[index % 3, index // 3] for index in range(6)], dtype=np.int64
    )
    vectors, _physical_pairs, result, _pair_path = _fixture(
        tmp_path, coordinates, pairs
    )
    labels = np.load(result.labels_path, mmap_mode="r", allow_pickle=False)
    distance_path = tmp_path / "normalized-distances.npy"
    np.save(
        distance_path,
        np.full(len(vectors), 0.01, dtype=np.float32),
        allow_pickle=False,
    )
    pair_semantics = tmp_path / "pair-semantics.json"
    pair_semantics.write_text(
        json.dumps(
            {
                "schema_version": "comrecgc_pair_semantics_receipt_v1",
                "status": "PASS",
                "pair_orientation": "col0=parent;col1=candidate",
            }
        )
    )
    contract = ThetaClosePairContract(
        theta=0.1,
        parent_count=3,
        candidate_count=2,
        distance_checkpoint_sha256="a" * 64,
        embedding_checkpoint_sha256="b" * 64,
        scale_contract=SCALE_CONTRACT,
        normalized_distance_contract=NORMALIZED_DISTANCE_CONTRACT,
    )
    view_root = tmp_path / "close-view"
    with pytest.raises(
        dbscan.ExternalMemoryDBSCANError,
        match="ALL_PAIRS_CLOSE_REVIEW_REQUIRED",
    ):
        materialize_theta_close_pair_view(
            physical_vectors_path=Path(vectors.filename),
            normalized_distances_path=distance_path,
            output_dir=view_root,
            contract=contract,
            pair_semantics_contract_path=pair_semantics,
            expected_pair_semantics_contract_sha256=_sha(pair_semantics),
            block_size=2,
        )
    checkpoint = json.loads((view_root / "checkpoint.json").read_text())
    view_identity = checkpoint["scientific_identity"]
    certificate = {
        "schema_version": ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
        "status": "PASS",
        "all_pairs_close_proven": True,
        "full_distance_scan_complete": True,
        "official_sample_comparison_pass": True,
        "normalization_audit_pass": True,
        "physical_store_rows": 6,
        "count_distance_le_theta": 6,
        "count_distance_gt_theta": 0,
        "count_distance_eq_theta": 0,
        "theta": contract.theta,
        "filter_operator": FILTER_OPERATOR,
        "pair_orientation": PAIR_ORIENTATION,
        "pair_order": PAIR_ORDER,
        "physical_vectors_sha256": view_identity["physical_vectors_sha256"],
        "normalized_distances_sha256": view_identity[
            "normalized_distances_sha256"
        ],
        "distance_checkpoint_sha256": contract.distance_checkpoint_sha256,
        "embedding_checkpoint_sha256": contract.embedding_checkpoint_sha256,
        "scale_contract": contract.scale_contract,
        "normalized_distance_contract": contract.normalized_distance_contract,
        "approximation_used": False,
    }
    certificate_path = tmp_path / "all-pairs-close-certificate.json"
    certificate_path.write_text(json.dumps(certificate, sort_keys=True))
    close_view = materialize_theta_close_pair_view(
        physical_vectors_path=Path(vectors.filename),
        normalized_distances_path=distance_path,
        output_dir=view_root,
        contract=contract,
        pair_semantics_contract_path=pair_semantics,
        expected_pair_semantics_contract_sha256=_sha(pair_semantics),
        all_pairs_close_certificate_path=certificate_path,
        block_size=2,
        resume=True,
    )
    implicit = close_view.open_pairs()
    logical_sha = close_view.pairs_sha256
    authority = close_view.manifest_path
    kwargs = {
        "dbscan_manifest_path": result.manifest_path,
        "dbscan_manifest_sha256": result.manifest_sha256,
        "labels": labels,
        "recourse_vectors": vectors,
        "pair_indices": implicit,
        "pairs_sha256": logical_sha,
        "pair_authority_manifest_path": authority,
        "pair_authority_manifest_sha256": close_view.manifest_sha256,
        "radius": 0.02,
        "theta": 0.2,
        "recourse_size": 100,
        "official_greedy": _greedy,
        "torch_module": torch,
        "max_rss_bytes": dbscan._rss_bytes() + 512 * 1024**2,
        "block_size": 2,
    }
    original = summary._write_checkpoint
    interrupted = {"done": False}

    def interrupt(path, **checkpoint):
        state = original(path, **checkpoint)
        if (
            not interrupted["done"]
            and checkpoint["phase"] == "membership_scan"
            and checkpoint["next_offset"] == 2
        ):
            interrupted["done"] = True
            raise RuntimeError("component summary interruption")
        return state

    monkeypatch.setattr(summary, "_write_checkpoint", interrupt)
    root = tmp_path / "summary"
    with pytest.raises(RuntimeError, match="component summary interruption"):
        summary.summarize_proven_all_core_components_external(
            work_dir=root, **kwargs
        )
    state_path = root / "checkpoint.json"
    pristine = state_path.read_bytes()
    state = json.loads(pristine)
    state["next_offset"] = 4
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(summary, "_write_checkpoint", original)
    with pytest.raises(
        dbscan.ExternalMemoryDBSCANError, match="checkpoint identity/closure"
    ):
        summary.summarize_proven_all_core_components_external(
            work_dir=root, resume=True, **kwargs
        )
    state_path.write_bytes(pristine)
    completed = summary.summarize_proven_all_core_components_external(
        work_dir=root, resume=True, **kwargs
    )
    reopened = summary.summarize_proven_all_core_components_external(
        work_dir=root, resume=True, **kwargs
    )
    assert reopened.manifest_sha256 == completed.manifest_sha256
    validated = summary.validate_proven_all_core_component_summary(
        completed.manifest_path,
        torch_module=torch,
        pair_indices=implicit,
        full_replay=True,
    )
    assert validated.selected == completed.selected
    standalone = summary.validate_proven_all_core_component_summary(
        completed.manifest_path,
        torch_module=torch,
        pair_indices=None,
        full_replay=True,
    )
    assert standalone.selected == completed.selected
    with pytest.raises(
        dbscan.ExternalMemoryDBSCANError,
        match="invocation identity drift",
    ):
        summary.summarize_proven_all_core_components_external(
            work_dir=root,
            resume=True,
            **{**kwargs, "theta": 0.19},
        )


def test_all_core_component_summary_partial_close_authority_reopens_selected_pairs(
    tmp_path: Path,
) -> None:
    physical = np.zeros((8, 64), dtype=np.float32)
    physical[:, 0] = np.asarray(
        [0.000, 0.001, 0.002, 0.050, 0.051, 0.052, 0.5, 0.6],
        dtype=np.float32,
    )
    distances = np.asarray(
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2],
        dtype=np.float32,
    )
    physical_path = tmp_path / "physical-vectors.npy"
    distance_path = tmp_path / "distances.npy"
    np.save(physical_path, physical, allow_pickle=False)
    np.save(distance_path, distances, allow_pickle=False)
    pair_semantics = tmp_path / "pair-semantics.json"
    pair_semantics.write_text(
        json.dumps(
            {
                "schema_version": "comrecgc_pair_semantics_receipt_v1",
                "status": "PASS",
                "pair_orientation": "col0=parent;col1=candidate",
            }
        )
    )
    close_view = materialize_theta_close_pair_view(
        physical_vectors_path=physical_path,
        normalized_distances_path=distance_path,
        output_dir=tmp_path / "partial-close-view",
        contract=ThetaClosePairContract(
            theta=0.1,
            parent_count=4,
            candidate_count=2,
            distance_checkpoint_sha256="c" * 64,
            embedding_checkpoint_sha256="d" * 64,
            scale_contract=SCALE_CONTRACT,
            normalized_distance_contract=NORMALIZED_DISTANCE_CONTRACT,
        ),
        pair_semantics_contract_path=pair_semantics,
        expected_pair_semantics_contract_sha256=_sha(pair_semantics),
        max_compact_bytes=1024**2,
        block_size=2,
    )
    assert close_view.view_storage == "materialized_selected_rows"
    exact = dbscan.fit_external_memory_dbscan(
        vectors_path=close_view.vectors_path,
        work_dir=tmp_path / "dbscan",
        contract=dbscan.ExternalDBSCANContract(
            eps=0.02,
            min_samples=3,
            query_block_size=2,
            checkpoint_interval_blocks=1,
            max_rss_bytes=dbscan._rss_bytes() + 512 * 1024**2,
            expected_sklearn_version=sklearn.__version__,
            shortcut_mode=dbscan.ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
            shortcut_seed_count=3,
            shortcut_failure_cap=100,
            shortcut_query_block_size=2,
            exact_fallback_max_samples=0,
        ),
    )
    labels = np.load(exact.labels_path, mmap_mode="r", allow_pickle=False)
    vectors = np.load(close_view.vectors_path, mmap_mode="r", allow_pickle=False)
    actual = summary.summarize_proven_all_core_components_external(
        work_dir=tmp_path / "summary",
        dbscan_manifest_path=exact.manifest_path,
        dbscan_manifest_sha256=exact.manifest_sha256,
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=close_view.open_pairs(),
        pairs_sha256=close_view.pairs_sha256,
        pair_authority_manifest_path=close_view.manifest_path,
        pair_authority_manifest_sha256=close_view.manifest_sha256,
        radius=0.02,
        theta=0.2,
        recourse_size=100,
        official_greedy=_greedy,
        torch_module=torch,
        max_rss_bytes=dbscan._rss_bytes() + 512 * 1024**2,
        block_size=2,
    )
    terminal = json.loads(actual.manifest_path.read_text())
    assert terminal["scientific_identity"]["pairs_storage"] == (
        "theta_close_view_v1"
    )
    assert terminal["scientific_identity"]["pairs_path"] == str(
        close_view.pairs_path
    )
    reopened = summary.validate_proven_all_core_component_summary(
        actual.manifest_path,
        torch_module=torch,
        pair_indices=None,
        full_replay=True,
    )
    assert reopened.official_result == actual.official_result


def test_all_core_component_summary_root_has_one_atomic_writer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "claimed-summary"
    barrier = threading.Barrier(2)

    def claim() -> str:
        barrier.wait()
        try:
            summary._claim_summary_root(
                root, identity_sha256="a" * 64, resume=False
            )
        except FileExistsError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = sorted(executor.map(lambda _value: claim(), range(2)))
    assert results == ["lost", "won"]
    receipt = json.loads((root / "owner_claim.json").read_text())
    assert receipt["root_stat_identity"] == summary._root_stat(root)
    symlink_root = tmp_path / "symlink-owner-root"
    symlink_root.mkdir()
    target = tmp_path / "outside-owner.json"
    target.write_text(json.dumps(receipt))
    (symlink_root / "owner_claim.json").symlink_to(target)
    with pytest.raises(
        dbscan.ExternalMemoryDBSCANError, match="has no owner claim"
    ):
        summary._validate_owner_claim(
            symlink_root, identity_sha256="a" * 64
        )


def test_all_core_component_summary_rejects_concurrent_resume_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinates = np.asarray(
        [0.000, 0.001, 0.002, 0.050, 0.051, 0.052], dtype=np.float32
    )
    pairs = np.asarray(
        [[index % 3, index // 3] for index in range(6)], dtype=np.int64
    )
    vectors, pair_values, result, pair_path = _fixture(
        tmp_path, coordinates, pairs
    )
    labels = np.load(result.labels_path, mmap_mode="r", allow_pickle=False)
    root = tmp_path / "summary"
    kwargs = {
        "work_dir": root,
        "dbscan_manifest_path": result.manifest_path,
        "dbscan_manifest_sha256": result.manifest_sha256,
        "labels": labels,
        "recourse_vectors": vectors,
        "pair_indices": pair_values,
        "pairs_sha256": _sha(pair_path),
        "radius": 0.02,
        "theta": 0.2,
        "recourse_size": 100,
        "official_greedy": _greedy,
        "torch_module": torch,
        "max_rss_bytes": dbscan._rss_bytes() + 512 * 1024**2,
        "block_size": 2,
    }
    summary.summarize_proven_all_core_components_external(**kwargs)
    with pytest.raises(FileExistsError, match="explicit resume adoption"):
        summary.summarize_proven_all_core_components_external(**kwargs)
    entered = threading.Event()
    release = threading.Event()
    original_validate = summary.validate_proven_all_core_component_summary

    def blocked_validate(*args, **validate_kwargs):
        entered.set()
        assert release.wait(timeout=10)
        return original_validate(*args, **validate_kwargs)

    monkeypatch.setattr(
        summary, "validate_proven_all_core_component_summary", blocked_validate
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            summary.summarize_proven_all_core_components_external,
            resume=True,
            **kwargs,
        )
        assert entered.wait(timeout=10)
        with pytest.raises(
            dbscan.ExternalMemoryDBSCANError,
            match="writer lock is already held",
        ):
            summary.summarize_proven_all_core_components_external(
                resume=True, **kwargs
            )
        release.set()
        assert first.result(timeout=10).manifest_path.is_file()


def _replace_writer_lock_inode(root: Path) -> None:
    lock_path = root / ".writer.lock"
    lock_path.unlink()
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        receipt = {
            "schema_version": summary._WRITER_LOCK_SCHEMA,
            "root": str(root),
            "root_stat_identity": summary._root_stat(root),
            "lock_stat_identity": {
                "device": int(lock_stat.st_dev),
                "inode": int(lock_stat.st_ino),
                "mode": int(lock_stat.st_mode),
            },
        }
        os.write(
            descriptor,
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "replace_stage",
    ["terminal_replay", "post_publish", "post_publish_hard_crash"],
)
def test_all_core_component_summary_revokes_pass_after_lock_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_stage: str,
) -> None:
    coordinates = np.asarray(
        [0.000, 0.001, 0.002, 0.050, 0.051, 0.052], dtype=np.float32
    )
    pairs = np.asarray(
        [[index % 3, index // 3] for index in range(6)], dtype=np.int64
    )
    vectors, pair_values, result, pair_path = _fixture(
        tmp_path, coordinates, pairs
    )
    labels = np.load(result.labels_path, mmap_mode="r", allow_pickle=False)
    root = tmp_path / "summary"
    kwargs = {
        "work_dir": root,
        "dbscan_manifest_path": result.manifest_path,
        "dbscan_manifest_sha256": result.manifest_sha256,
        "labels": labels,
        "recourse_vectors": vectors,
        "pair_indices": pair_values,
        "pairs_sha256": _sha(pair_path),
        "radius": 0.02,
        "theta": 0.2,
        "recourse_size": 100,
        "official_greedy": _greedy,
        "torch_module": torch,
        "max_rss_bytes": dbscan._rss_bytes() + 512 * 1024**2,
        "block_size": 2,
    }
    if replace_stage == "terminal_replay":
        original_results = summary._results_from_replay
        replaced = {"done": False}

        def replace_during_replay(*args, **result_kwargs):
            result_value = original_results(*args, **result_kwargs)
            if not replaced["done"]:
                replaced["done"] = True
                _replace_writer_lock_inode(root)
            return result_value

        monkeypatch.setattr(
            summary, "_results_from_replay", replace_during_replay
        )
    else:
        original_atomic_json = summary._atomic_json

        def replace_after_publish(path, payload, **atomic_kwargs):
            original_atomic_json(path, payload, **atomic_kwargs)
            if Path(path).name == "run_manifest.json":
                _replace_writer_lock_inode(root)

        monkeypatch.setattr(summary, "_atomic_json", replace_after_publish)
        if replace_stage == "post_publish_hard_crash":
            monkeypatch.setattr(
                summary, "_revoke_terminal_manifest", lambda _root: None
            )
    with pytest.raises(
        dbscan.ExternalMemoryDBSCANError,
        match="writer lock inode changed while held",
    ):
        summary.summarize_proven_all_core_components_external(**kwargs)
    if replace_stage == "post_publish_hard_crash":
        terminal = root / "run_manifest.json"
        assert terminal.is_file()
        with pytest.raises(
            dbscan.ExternalMemoryDBSCANError,
            match="writer lock identity mismatch",
        ):
            summary.validate_proven_all_core_component_summary(
                terminal, torch_module=torch, full_replay=False
            )
    else:
        assert not (root / "run_manifest.json").exists()


def test_completed_summary_remount_stat_exception_is_device_only() -> None:
    recorded = {"device": 126, "inode": 41, "mode": 0o40755}
    observed = {"device": 76, "inode": 41, "mode": 0o40755}
    evidence = summary._device_only_stat_reopen(
        label="fixture", recorded=recorded, observed=observed
    )
    assert evidence["device_changed"] is True
    assert evidence["stable_stat_fields_match"] is True
    with pytest.raises(
        dbscan.ExternalMemoryDBSCANError, match="stat identity drift"
    ):
        summary._device_only_stat_reopen(
            label="fixture",
            recorded=recorded,
            observed={**observed, "inode": 42},
        )


def test_terminal_remount_validation_holds_writer_lock_through_full_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinates = np.asarray(
        [0.000, 0.001, 0.002, 0.050, 0.051, 0.052], dtype=np.float32
    )
    pairs = np.asarray(
        [[index % 3, index // 3] for index in range(6)], dtype=np.int64
    )
    vectors, pair_values, dbscan_result, pair_path = _fixture(
        tmp_path, coordinates, pairs
    )
    labels = np.load(
        dbscan_result.labels_path, mmap_mode="r", allow_pickle=False
    )
    completed = summary.summarize_proven_all_core_components_external(
        work_dir=tmp_path / "summary",
        dbscan_manifest_path=dbscan_result.manifest_path,
        dbscan_manifest_sha256=dbscan_result.manifest_sha256,
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=pair_values,
        pairs_sha256=_sha(pair_path),
        radius=0.02,
        theta=0.2,
        recourse_size=100,
        official_greedy=_greedy,
        torch_module=torch,
        max_rss_bytes=dbscan._rss_bytes() + 512 * 1024**2,
        block_size=2,
    )

    entered_replay = threading.Event()
    release_replay = threading.Event()
    original_scan = summary._scan_centroid_range

    def blocked_scan(*args, **kwargs):
        entered_replay.set()
        assert release_replay.wait(timeout=10)
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(summary, "_scan_centroid_range", blocked_scan)
    with ThreadPoolExecutor(max_workers=1) as executor:
        validation = executor.submit(
            summary.validate_proven_all_core_component_summary,
            completed.manifest_path,
            torch_module=torch,
            full_replay=True,
            allow_remount_device_drift_for_aids_terminal_reconciliation=True,
        )
        try:
            assert entered_replay.wait(timeout=10)
            descriptor = os.open(
                completed.manifest_path.parent / ".writer.lock",
                os.O_RDONLY | os.O_NOFOLLOW,
            )
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
        finally:
            release_replay.set()
        reopened = validation.result(timeout=10)

    evidence = dict(reopened.source_reopen_evidence or {})
    writer_evidence = dict(evidence["writer_lock"])
    assert writer_evidence["exclusive_lock_held_for_terminal_validation"] is True
    assert writer_evidence["terminal_validation_completed_while_lock_held"] is True
    assert writer_evidence["session_stat_stable"] is True
    assert evidence["no_active_writer_verified"] is True
    assert evidence["stat_stable_during_reopen"] is True
