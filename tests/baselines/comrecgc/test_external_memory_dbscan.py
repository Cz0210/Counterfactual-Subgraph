from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.baselines.comrecgc import external_memory_dbscan as external


sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402


def _save(path: Path, values: np.ndarray) -> Path:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    return path


def _write_reauthenticated_checkpoint(path: Path, state: dict) -> None:
    """Model coordinated state corruption beyond the outer payload checksum."""

    unsigned = dict(state)
    unsigned.pop("checkpoint_payload_sha256", None)
    state["checkpoint_payload_sha256"] = external._stable_hash(unsigned)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _contract(*, block: int = 7, max_rss_bytes: int | None = None):
    return external.ExternalDBSCANContract(
        eps=0.37,
        min_samples=3,
        query_block_size=block,
        checkpoint_interval_blocks=1,
        max_rss_bytes=max_rss_bytes or external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
    )


def _shortcut_contract(
    *,
    block: int = 3,
    anchors: int = 4,
    fallback: int = 100,
    max_rss_bytes: int | None = None,
) -> external.ExternalDBSCANContract:
    return external.ExternalDBSCANContract(
        eps=1.0,
        min_samples=3,
        query_block_size=2,
        checkpoint_interval_blocks=1,
        max_rss_bytes=max_rss_bytes or external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
        shortcut_mode=external.ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_anchor_count=anchors,
        shortcut_query_block_size=block,
        exact_fallback_max_samples=fallback,
    )


def _all_core_values(num_samples: int = 12) -> np.ndarray:
    values = np.zeros((num_samples, 64), dtype=np.float32)
    # With four anchors and N=12 the deterministic anchor indices are
    # [0, 3, 7, 11].  Keep them duplicate-by-value but distinct-by-index.
    values[1, 0] = 1.0  # exactly eps from every anchor: sklearn uses <= eps
    values[2, 0] = 0.25
    values[4, 0] = -0.5
    values[5, 0] = 0.75
    values[6, 0] = -1.0
    values[8, 0] = 0.1
    values[9, 0] = -0.1
    values[10, 0] = 0.5
    return values


def _adaptive_values() -> np.ndarray:
    values = np.zeros((9, 64), dtype=np.float32)
    # The global minimum-norm seeds are indices 0, 1, 2.  The bridge at 0.55
    # sees only seed 1 and is therefore a first-pass failure; together with
    # the exact remote failures it connects and covers the final anchor graph.
    values[:, 0] = np.asarray(
        [0.0, 0.1, -0.1, 0.55, 1.0, 1.0, 1.0, 0.55, 1.0],
        dtype=np.float32,
    )
    return values


def _adaptive_contract(
    *,
    block: int = 3,
    failure_cap: int = 10,
    max_rss_bytes: int | None = None,
) -> external.ExternalDBSCANContract:
    return external.ExternalDBSCANContract(
        eps=0.5,
        min_samples=3,
        query_block_size=2,
        checkpoint_interval_blocks=1,
        max_rss_bytes=max_rss_bytes or external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
        shortcut_mode=external.ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_seed_count=3,
        shortcut_failure_cap=failure_cap,
        shortcut_query_block_size=block,
        exact_fallback_max_samples=0,
    )


def _component_recovery_contract(
    *, block: int = 2, fallback: int = 0
) -> external.ExternalDBSCANContract:
    return external.ExternalDBSCANContract(
        eps=0.02,
        min_samples=3,
        query_block_size=2,
        checkpoint_interval_blocks=1,
        max_rss_bytes=external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
        shortcut_mode=external.ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_seed_count=3,
        shortcut_failure_cap=100,
        shortcut_query_block_size=block,
        exact_fallback_max_samples=fallback,
    )


def _component_values(values: list[float]) -> np.ndarray:
    result = np.zeros((len(values), 64), dtype=np.float64)
    result[:, 0] = np.asarray(values, dtype=np.float64)
    return result


def test_boundary_reference_resolver_uses_inclusive_sklearn_float64() -> None:
    eps = 0.02
    left = np.zeros(64, dtype=np.float64)
    inside = np.zeros(64, dtype=np.float64)
    inside[0] = np.nextafter(eps, 0.0)
    outside = np.zeros(64, dtype=np.float64)
    outside[0] = np.nextafter(eps, np.inf)

    inside_membership, inside_distance = external.resolve_boundary_membership(
        left, inside, eps
    )
    outside_membership, outside_distance = external.resolve_boundary_membership(
        left, outside, eps
    )

    assert inside_membership is True
    assert inside_distance <= eps
    assert outside_membership is False
    assert outside_distance > eps


def test_boundary_reference_fallback_is_explicit_and_deterministic() -> None:
    eps = 0.02
    vectors = np.zeros((1, 64), dtype=np.float32)
    query = np.zeros((1, 64), dtype=np.float64)
    query[0, 0] = np.nextafter(eps, 0.0)

    class MissingInsideModel:
        @staticmethod
        def radius_neighbors(_values, *, return_distance):
            assert return_distance is False
            return np.asarray([np.asarray([], dtype=np.intp)], dtype=object)

    first = external._exact_query_membership(
        model=MissingInsideModel(),
        vectors=vectors,
        query_vectors64=query,
        start=0,
        stop=1,
        eps=eps,
    )
    replay = external._exact_query_membership(
        model=MissingInsideModel(),
        vectors=vectors,
        query_vectors64=query,
        start=0,
        stop=1,
        eps=eps,
    )

    within, distances, _near_count, rows = first
    assert bool(within[0, 0]) is True
    assert float(distances[0, 0]) <= eps
    assert len(rows) == 1
    assert rows[0]["row_id"] == 0
    assert rows[0]["fast_membership"] is False
    assert rows[0]["membership"] is True
    assert rows[0]["reference_semantics"] == "SKLEARN_FLOAT64"
    assert rows == replay[3]
    assert np.array_equal(within, replay[0])
    assert np.array_equal(distances, replay[1])


def test_production_boundary_pair_uses_reference_membership() -> None:
    vectors_path = Path(
        os.environ.get(
            "AIDS_BOUNDARY_PRODUCTION_VECTORS",
            "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/repairs/"
            "four_methods_four_datasets_aids_comrecgc_exact_route_v5_pair_order_v1/"
            "source_snapshot/attempt-0/pair_store/recourse_vectors.npy",
        )
    )
    if not vectors_path.is_file():
        pytest.skip("production boundary vector store is unavailable")
    row_id = 5_025_392
    anchor_id = 49_369_222
    vectors = np.load(vectors_path, mmap_mode="r")
    anchor64 = np.asarray(vectors[anchor_id : anchor_id + 1], dtype=np.float64)
    model = NearestNeighbors(radius=0.02, algorithm="brute", metric="euclidean")
    model.fit(np.asarray(vectors[anchor_id : anchor_id + 1]))

    first = external._exact_query_membership(
        model=model,
        vectors=vectors,
        query_vectors64=anchor64,
        start=row_id,
        stop=row_id + 1,
        eps=0.02,
    )
    replay = external._exact_query_membership(
        model=model,
        vectors=vectors,
        query_vectors64=anchor64,
        start=row_id,
        stop=row_id + 1,
        eps=0.02,
    )

    within, distances, _near_count, rows = first
    assert bool(within[0, 0]) is True
    assert len(rows) == 1
    assert rows[0]["row_id"] == row_id
    assert rows[0]["fast_membership"] is False
    assert rows[0]["membership"] is True
    assert rows[0]["sklearn_reference_distance"] <= 0.02
    assert rows == replay[3]
    assert np.array_equal(within, replay[0])
    assert np.array_equal(distances, replay[1])


def test_adaptive_disconnected_anchor_components_are_bridged_by_data(
    tmp_path: Path,
) -> None:
    values = _component_values(
        [0.0, 0.0, 0.001, 0.019, 0.030, 0.031, 0.032,
         -0.019, -0.030, -0.031, -0.032]
    )
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract()
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(values)
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "bridge",
        contract=contract,
    )
    manifest = json.loads(result.manifest_path.read_text())
    connectivity = json.loads(
        Path(manifest["connectivity_certificate_path"]).read_text()
    )
    boundary = json.loads(Path(manifest["boundary_certificate_path"]).read_text())
    assert manifest["clustering_path"] == external.ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert np.array_equal(
        np.flatnonzero(np.load(result.core_mask_path)), expected.core_sample_indices_
    )
    assert connectivity["initial_anchor_component_count"] == 3
    assert connectivity["final_exact_component_count"] == 1
    assert connectivity["single_epsilon_component_proven"] is True
    assert connectivity["exhaustive_all_anchor_scan_used"] is False
    assert boundary["recheck_dtype"] == "float64"
    assert boundary["uncertain_edges_accepted"] == 0
    reopened = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "bridge",
        contract=contract,
        resume=True,
    )
    assert reopened.manifest_sha256 == result.manifest_sha256


def test_adaptive_disconnected_true_multicluster_partition_is_exact(
    tmp_path: Path,
) -> None:
    values = _component_values([0.0, 0.0, 0.001, 0.050, 0.051, 0.052])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract()
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(values)
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "multi",
        contract=contract,
    )
    manifest = json.loads(result.manifest_path.read_text())
    connectivity = json.loads(
        Path(manifest["connectivity_certificate_path"]).read_text()
    )
    partition = json.loads(Path(manifest["cluster_partition_path"]).read_text())
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert result.cluster_count == 2
    assert result.noise_count == 0
    assert connectivity["exhaustive_all_anchor_scan_used"] is True
    assert connectivity["single_epsilon_component_proven"] is False
    assert connectivity["exact_multicomponent_partition_proven"] is True
    assert partition["cluster_order"] == "minimum_global_core_sample_index"
    assert partition["canonical_cluster_labels"] == [0, 1]


def test_component_recovery_rejects_nonanchor_to_nonanchor_bridge_when_seeds_split(
    tmp_path: Path,
) -> None:
    """A split seed set invalidates the component-attachment theorem.

    Each outer row has three seed neighbours in exactly one seed component,
    while the two outer rows connect those components only through each other.
    A same-row anchor-component scan cannot see that edge, so the recovery
    proof must fail closed instead of publishing sklearn's true one cluster.
    """

    values = np.zeros((8, 64), dtype=np.float64)
    values[:, :2] = np.asarray(
        [
            [-0.60, -0.03],
            [-0.60, 0.00],
            [-0.60, 0.03],
            [0.60, -0.03],
            [0.60, 0.00],
            [0.60, 0.03],
            [-0.30, 0.90],
            [0.30, 0.90],
        ],
        dtype=np.float64,
    )
    vectors = _save(tmp_path / "split-seed-vectors.npy", values)
    contract = external.ExternalDBSCANContract(
        eps=1.0,
        min_samples=3,
        query_block_size=2,
        checkpoint_interval_blocks=1,
        max_rss_bytes=external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
        shortcut_mode=external.ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_seed_count=6,
        shortcut_failure_cap=100,
        shortcut_query_block_size=2,
        exact_fallback_max_samples=0,
    )
    expected = DBSCAN(eps=1.0, min_samples=3).fit(values)
    assert set(expected.labels_.tolist()) == {0}
    assert len(expected.core_sample_indices_) == len(values)
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match=(
            "EXACT_DBSCAN_GENERAL_EXTERNAL_REQUIRED:"
            "reason=adaptive_seed_anchors_span_multiple_exact_components"
        ),
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / "split-seed-recovery",
            contract=contract,
        )
    failure = json.loads(
        (tmp_path / "split-seed-recovery/shortcut_failure.json").read_text()
    )
    assert failure["reason"] == (
        "adaptive_seed_anchors_span_multiple_exact_components_general_exact_required"
    )
    assert failure["details"]["unique_seed_component_proven"] is False
    assert failure["details"]["old_quadratic_route_started"] is False


def test_adaptive_noncore_failure_routes_small_noise_and_border_to_general_exact(
    tmp_path: Path,
) -> None:
    values = _component_values([0.0, 0.0, 0.001, 0.019, 0.039, 0.060])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract(fallback=100)
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(values)
    assert expected.labels_[-2:].tolist() == [0, -1]
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "border-noise",
        contract=contract,
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["clustering_path"] == "three_pass_exact_radius_graph_v1"
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert np.array_equal(
        np.flatnonzero(np.load(result.core_mask_path)), expected.core_sample_indices_
    )


def test_adaptive_noncore_failure_never_starts_large_quadratic_fallback(
    tmp_path: Path,
) -> None:
    values = _component_values([0.0, 0.0, 0.001, 0.019, 0.039, 0.060])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract(fallback=0)
    root = tmp_path / "general-required"
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="EXACT_DBSCAN_GENERAL_EXTERNAL_REQUIRED",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    failure = json.loads((root / "shortcut_failure.json").read_text())
    assert failure["reason"] == (
        "adaptive_failure_anchor_not_core_general_exact_required"
    )
    assert failure["details"]["old_quadratic_route_started"] is False
    assert not (root / "labels.npy").exists()


def test_adaptive_component_recovery_preserves_duplicates_and_exact_eps_boundary(
    tmp_path: Path,
) -> None:
    values = _component_values([0.0, 0.0, 0.0, 0.020, 0.040, 0.040, 0.040])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract()
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(values)
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "boundary-duplicates",
        contract=contract,
    )
    boundary = json.loads(
        Path(json.loads(result.manifest_path.read_text())["boundary_certificate_path"])
        .read_text()
    )
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert boundary["count_certifying_edges_exactly_at_eps"] > 0
    assert boundary["minimum_margin_to_eps_among_certifying_edges"] == 0.0


def test_adaptive_component_recovery_resume_replays_committed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _component_values(
        [0.0, 0.0, 0.001, 0.019, 0.030, 0.031, 0.032,
         -0.019, -0.030, -0.031, -0.032]
    )
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract(block=2)
    reference = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "reference",
        contract=contract,
    )
    reference_manifest = json.loads(reference.manifest_path.read_text())
    original = external._recovery_scan_block
    calls = {"count": 0}

    def interrupt(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("component recovery interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(external, "_recovery_scan_block", interrupt)
    root = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="component recovery interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    state = json.loads((root / "checkpoint.json").read_text())
    assert state["phase"] == external.COMPONENT_PRIMARY_PHASE
    assert state["next_offset"] == 2
    monkeypatch.setattr(external, "_recovery_scan_block", original)
    resumed = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract, resume=True
    )
    resumed_manifest = json.loads(resumed.manifest_path.read_text())
    assert resumed_manifest["labels_sha256"] == reference_manifest["labels_sha256"]
    assert resumed_manifest["core_mask_sha256"] == reference_manifest["core_mask_sha256"]


def test_adaptive_component_expansion_resume_preserves_promoted_array_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _component_values([0.0, 0.0, 0.001, 0.050, 0.051, 0.052])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract(block=2)
    reference = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "expansion-reference",
        contract=contract,
    )
    original = external._recovery_scan_block
    expansion_calls = {"count": 0}

    def interrupt(*args, **kwargs):
        if kwargs.get("verify_core_contract") is False:
            expansion_calls["count"] += 1
            if expansion_calls["count"] == 2:
                raise RuntimeError("component expansion interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(external, "_recovery_scan_block", interrupt)
    root = tmp_path / "expansion-resumed"
    with pytest.raises(RuntimeError, match="component expansion interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    state = json.loads((root / "checkpoint.json").read_text())
    assert state["phase"] == external.COMPONENT_EXPANSION_PHASE
    assert state["next_offset"] == 2
    assert state["component_core_lower_bounds_sha256"]
    assert state["component_attachments_sha256"]
    monkeypatch.setattr(external, "_recovery_scan_block", original)
    resumed = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract, resume=True
    )
    assert np.array_equal(
        np.load(resumed.labels_path), np.load(reference.labels_path)
    )
    assert resumed.manifest_sha256 == reference.manifest_sha256 or (
        json.loads(resumed.manifest_path.read_text())["labels_sha256"]
        == json.loads(reference.manifest_path.read_text())["labels_sha256"]
    )


@pytest.mark.parametrize("tamper_partial", [False, True])
def test_adaptive_component_recovery_labels_partial_is_replayed_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_partial: bool,
) -> None:
    values = _component_values([0.0, 0.0, 0.001, 0.050, 0.051, 0.052])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract(block=2)
    root = tmp_path / f"labels-partial-{tamper_partial}"
    original_replace = external.os.replace
    interrupted = {"done": False}

    def interrupt_labels_promotion(source, destination):
        if (
            not interrupted["done"]
            and Path(source).name == "labels.partial.npy"
            and Path(destination).name == "labels.npy"
        ):
            interrupted["done"] = True
            raise RuntimeError("labels promotion interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(external.os, "replace", interrupt_labels_promotion)
    with pytest.raises(RuntimeError, match="labels promotion interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    monkeypatch.setattr(external.os, "replace", original_replace)
    partial_path = root / "labels.partial.npy"
    assert partial_path.is_file()
    if tamper_partial:
        partial = np.load(partial_path, mmap_mode="r+")
        partial[0] = np.intp(99)
        partial.flush()
        del partial
        with pytest.raises(
            external.ExternalMemoryDBSCANError,
            match="labels resume content mismatch",
        ):
            external.fit_external_memory_dbscan(
                vectors_path=vectors,
                work_dir=root,
                contract=contract,
                resume=True,
            )
        assert partial_path.is_file()
        assert not (root / "labels.npy").exists()
    else:
        resumed = external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )
        expected = DBSCAN(
            eps=contract.eps, min_samples=contract.min_samples
        ).fit_predict(values)
        assert np.array_equal(np.load(resumed.labels_path), expected)
        assert not partial_path.exists()


def test_adaptive_component_recovery_resume_never_overwrites_tampered_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _component_values([0.0, 0.0, 0.001, 0.050, 0.051, 0.052])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract(block=2)
    root = tmp_path / "tampered-resume-certificate"
    original = external._ensure_exact_json
    interrupted = {"done": False}

    def interrupt_after_boundary(path, expected, **kwargs):
        result = original(path, expected, **kwargs)
        if (
            not interrupted["done"]
            and kwargs.get("label") == "component boundary certificate"
        ):
            interrupted["done"] = True
            raise RuntimeError("boundary certificate interruption")
        return result

    monkeypatch.setattr(external, "_ensure_exact_json", interrupt_after_boundary)
    with pytest.raises(RuntimeError, match="boundary certificate interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    monkeypatch.setattr(external, "_ensure_exact_json", original)
    boundary_path = root / "boundary_certificate.json"
    boundary = json.loads(boundary_path.read_text())
    boundary["uncertain_edges_accepted"] = 1
    boundary_path.write_text(json.dumps(boundary, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="boundary certificate resume content mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract, resume=True
        )
    assert json.loads(boundary_path.read_text())["uncertain_edges_accepted"] == 1


def test_adaptive_component_recovery_resume_rejects_tampered_core_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _component_values(
        [0.0, 0.0, 0.001, 0.019, 0.030, 0.031, 0.032,
         -0.019, -0.030, -0.031, -0.032]
    )
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract(block=2)
    original = external._recovery_scan_block
    calls = {"count": 0}

    def interrupt(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("component recovery tamper interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(external, "_recovery_scan_block", interrupt)
    root = tmp_path / "tampered-prefix"
    with pytest.raises(RuntimeError, match="tamper interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    lower = np.load(root / "component_core_lower_bounds.partial.npy", mmap_mode="r+")
    lower[0] = np.uint32(99)
    lower.flush()
    del lower
    monkeypatch.setattr(external, "_recovery_scan_block", original)
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="replay/core artifact mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract, resume=True
        )


def test_adaptive_component_recovery_terminal_rejects_tampered_bridge_receipt(
    tmp_path: Path,
) -> None:
    values = _component_values(
        [0.0, 0.0, 0.001, 0.019, 0.030, 0.031, 0.032,
         -0.019, -0.030, -0.031, -0.032]
    )
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract()
    root = tmp_path / "tampered-terminal"
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract
    )
    manifest = json.loads(result.manifest_path.read_text())
    connectivity = json.loads(
        Path(manifest["connectivity_certificate_path"]).read_text()
    )
    witness_path = Path(connectivity["bridge_witnesses_path"])
    receipt = json.loads(witness_path.read_text())
    receipt["witnesses"][0]["left_distance_float64_hex"] = float(0.0).hex()
    witness_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="connectivity artifact mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract, resume=True
        )


def test_adaptive_component_recovery_terminal_rejects_tampered_primary_query(
    tmp_path: Path,
) -> None:
    values = _component_values(
        [0.0, 0.0, 0.001, 0.019, 0.030, 0.031, 0.032]
    )
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract()
    root = tmp_path / "tampered-primary-query"
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract
    )
    manifest = json.loads(result.manifest_path.read_text())
    connectivity = json.loads(
        Path(manifest["connectivity_certificate_path"]).read_text()
    )
    query_path = Path(connectivity["primary_query_anchor_indices_path"])
    query = np.load(query_path, allow_pickle=False)
    with query_path.open("wb") as handle:
        np.save(handle, query[::-1], allow_pickle=False)
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="connectivity artifact mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract, resume=True
        )


def test_adaptive_component_recovery_terminal_rejects_coordinated_promoted_array_tamper(
    tmp_path: Path,
) -> None:
    values = _component_values([0.0, 0.0, 0.001, 0.050, 0.051, 0.052])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _component_recovery_contract(block=2)
    root = tmp_path / "coordinated-promoted-array-tamper"
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract
    )
    manifest = json.loads(result.manifest_path.read_text())
    proof_path = Path(manifest["shortcut_proof_path"])
    proof = json.loads(proof_path.read_text())
    lower_path = Path(proof["core_lower_bounds_path"])
    lower = np.load(lower_path, mmap_mode="r+")
    lower[0] = np.uint32(int(lower[0]) + 1)
    lower.flush()
    del lower
    lower_sha = external._sha256_file(lower_path)

    all_core_path = Path(proof["all_core_certificate_path"])
    all_core = json.loads(all_core_path.read_text())
    all_core["core_lower_bounds_sha256"] = lower_sha
    external._atomic_json(all_core_path, all_core)
    all_core_sha = external._sha256_file(all_core_path)

    connectivity_path = Path(proof["connectivity_certificate_path"])
    connectivity = json.loads(connectivity_path.read_text())
    connectivity["all_core_certificate_sha256"] = all_core_sha
    external._atomic_json(connectivity_path, connectivity)
    connectivity_sha = external._sha256_file(connectivity_path)

    partition_path = Path(proof["cluster_partition_path"])
    partition = json.loads(partition_path.read_text())
    partition["all_core_certificate_sha256"] = all_core_sha
    partition["connectivity_certificate_sha256"] = connectivity_sha
    external._atomic_json(partition_path, partition)
    partition_sha = external._sha256_file(partition_path)

    proof["core_lower_bounds_sha256"] = lower_sha
    proof["all_core_certificate_sha256"] = all_core_sha
    proof["connectivity_certificate_sha256"] = connectivity_sha
    proof["cluster_partition_sha256"] = partition_sha
    external._atomic_json(proof_path, proof)
    proof_sha = external._sha256_file(proof_path)

    manifest["shortcut_proof_sha256"] = proof_sha
    manifest["all_core_certificate_sha256"] = all_core_sha
    manifest["connectivity_certificate_sha256"] = connectivity_sha
    manifest["cluster_partition_sha256"] = partition_sha
    external._atomic_json(result.manifest_path, manifest)
    state_path = root / "checkpoint.json"
    checkpoint = json.loads(state_path.read_text())
    checkpoint["shortcut_proof_sha256"] = proof_sha
    _write_reauthenticated_checkpoint(state_path, checkpoint)

    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="promoted array ledger mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract, resume=True
        )


def test_production_anchor_disconnect_fixture_selects_expected_boundaries() -> None:
    fixture = json.loads(
        Path("tests/fixtures/comrecgc/aids_anchor_disconnected_failure_contract.json")
        .read_text()
    )
    rows_path = Path("/private/tmp/aids-adaptive-selected-anchor-rows.npy")
    indices_path = Path("/private/tmp/aids-shortcut-anchor-indices.npy")
    if not rows_path.is_file() or not indices_path.is_file():
        pytest.skip("read-only production failure fixture is not staged")
    assert external._sha256_file(rows_path) == fixture["anchor_rows_sha256"]
    assert external._sha256_file(indices_path) == fixture["anchor_indices_sha256"]
    rows = np.load(rows_path, allow_pickle=False)
    indices = np.load(indices_path, allow_pickle=False)
    model, version = external._fit_anchor_neighbors(rows, eps=fixture["eps"])
    assert version == sklearn.__version__
    edges, connected, reached, neighborhoods = external._anchor_graph(
        model=model, anchor_vectors=rows
    )
    components, groups = external._canonical_anchor_components(
        anchor_indices=indices, anchor_rows=neighborhoods
    )
    query, boundary = external._adaptive_primary_query_anchors(
        anchor_indices=indices,
        anchor_vectors64=np.asarray(rows, dtype=np.float64),
        component_by_anchor=components,
        seed_indices=fixture["seed_global_indices"],
    )
    assert connected is False
    assert reached in fixture["initial_component_sizes"]
    assert len(edges) == fixture["anchor_edge_count"]
    assert sorted(map(len, groups), reverse=True) == fixture["initial_component_sizes"]
    seed_locals = np.searchsorted(
        indices, np.asarray(fixture["seed_global_indices"], dtype=np.intp)
    )
    seed_components = {
        int(components[int(value)]) for value in seed_locals.tolist()
    }
    assert seed_components == {int(components[seed_locals[0]])}
    assert len(seed_components) == 1
    assert len(groups[next(iter(seed_components))]) == 3
    assert len(query) == 5
    assert [row["anchor_global_index"] for row in boundary] == fixture[
        "boundary_anchor_global_indices"
    ]
    actual = [
        float.fromhex(row["anchor_to_seed_distance_float64_hex"])
        for row in boundary
    ]
    assert np.allclose(
        actual,
        fixture["boundary_anchor_to_seed_distances_float64"],
        rtol=0.0,
        atol=5e-13,
    )


def test_adaptive_two_pass_witness_is_elementwise_sklearn_exact(
    tmp_path: Path,
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _adaptive_contract()
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(values)
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "adaptive",
        contract=contract,
    )
    manifest = json.loads(result.manifest_path.read_text())
    proof = json.loads(result.shortcut_proof_path.read_text())
    selection = json.loads(
        (tmp_path / "adaptive/adaptive_anchor_selection.json").read_text()
    )["selection_identity"]
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert np.array_equal(
        np.flatnonzero(np.load(result.core_mask_path)), expected.core_sample_indices_
    )
    assert manifest["clustering_path"] == (
        external.ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
    )
    assert manifest["neighbor_counts_available"] is False
    assert selection["seed_indices"] == [0, 1, 2]
    assert np.load(selection["failure_indices_path"]).tolist() == [3, 4, 5, 6, 7, 8]
    assert np.load(selection["anchor_indices_path"]).tolist() == list(range(9))
    assert selection["first_pass_complete"] is True
    assert selection["approximation_used"] is False
    assert proof["second_pass_complete"] is True
    assert proof["adaptive_selection_identity_sha256"]
    reopened = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "adaptive",
        contract=contract,
        resume=True,
    )
    assert reopened.manifest_sha256 == result.manifest_sha256


def test_adaptive_failure_scan_resume_preserves_selection_and_label_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _adaptive_contract(block=3)
    reference = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "reference",
        contract=contract,
    )
    reference_manifest = json.loads(reference.manifest_path.read_text())
    reference_selection = json.loads(
        (tmp_path / "reference/adaptive_anchor_selection.json").read_text()
    )
    original_fit = external._fit_anchor_neighbors
    calls = {"count": 0}

    class InterruptingSeedModel:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def radius_neighbors(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("adaptive failure-scan interruption")
            return self.wrapped.radius_neighbors(*args, **kwargs)

    def interrupted_fit(*args, **kwargs):
        model, version = original_fit(*args, **kwargs)
        return InterruptingSeedModel(model), version

    monkeypatch.setattr(external, "_fit_anchor_neighbors", interrupted_fit)
    root = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="adaptive failure-scan interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    assert checkpoint["phase"] == "adaptive_failure_scan"
    assert checkpoint["next_offset"] == 3

    monkeypatch.setattr(external, "_fit_anchor_neighbors", original_fit)
    resumed = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract, resume=True
    )
    resumed_manifest = json.loads(resumed.manifest_path.read_text())
    resumed_selection = json.loads((root / "adaptive_anchor_selection.json").read_text())
    assert resumed_manifest["labels_sha256"] == reference_manifest["labels_sha256"]
    for field in (
        "seed_indices_sha256",
        "failure_index_list_sha256",
        "selected_anchor_indices_sha256",
        "anchor_rows_sha256",
    ):
        assert resumed_selection["selection_identity"][field] == (
            reference_selection["selection_identity"][field]
        )


def test_adaptive_seed_next_offset_tamper_cannot_skip_unscanned_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _adaptive_contract(block=3)
    original = external._adaptive_seed_block_candidates
    calls = {"count": 0}

    def interrupt_second_block(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("adaptive seed interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        external, "_adaptive_seed_block_candidates", interrupt_second_block
    )
    root = tmp_path / "adaptive-seed-tamper"
    with pytest.raises(RuntimeError, match="adaptive seed interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["phase"] == "adaptive_seed_scan"
    assert checkpoint["next_offset"] == 3
    checkpoint["next_offset"] = len(values)
    _write_reauthenticated_checkpoint(checkpoint_path, checkpoint)

    monkeypatch.setattr(external, "_adaptive_seed_block_candidates", original)
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="offset/progress-ledger mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )
    assert not (root / "run_manifest.json").exists()


def test_adaptive_failure_next_offset_tamper_cannot_skip_unscanned_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _adaptive_contract(block=3)
    original_fit = external._fit_anchor_neighbors
    calls = {"count": 0}

    class InterruptingSeedModel:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def radius_neighbors(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("adaptive failure interruption")
            return self.wrapped.radius_neighbors(*args, **kwargs)

    def interrupted_fit(*args, **kwargs):
        model, version = original_fit(*args, **kwargs)
        return InterruptingSeedModel(model), version

    monkeypatch.setattr(external, "_fit_anchor_neighbors", interrupted_fit)
    root = tmp_path / "adaptive-failure-tamper"
    with pytest.raises(RuntimeError, match="adaptive failure interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["phase"] == "adaptive_failure_scan"
    assert checkpoint["next_offset"] == 3
    checkpoint["next_offset"] = len(values)
    _write_reauthenticated_checkpoint(checkpoint_path, checkpoint)

    monkeypatch.setattr(external, "_fit_anchor_neighbors", original_fit)
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="offset/progress-ledger mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )
    assert not (root / "run_manifest.json").exists()


def test_adaptive_selection_publish_crash_resumes_from_persisted_complete_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _adaptive_contract(block=3)
    original_atomic_json = external._atomic_json
    crashed = {"done": False}

    def crash_after_selection_publish(path, payload):
        original_atomic_json(path, payload)
        if path.name == "adaptive_anchor_selection.json" and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("selection publish interruption")

    monkeypatch.setattr(external, "_atomic_json", crash_after_selection_publish)
    root = tmp_path / "selection-publish-resume"
    with pytest.raises(RuntimeError, match="selection publish interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    assert checkpoint["phase"] == "adaptive_failure_scan"
    assert checkpoint["next_offset"] == len(values)
    assert checkpoint["progress_ledgers"]["adaptive_failure_scan"]["complete"]
    assert (root / "adaptive_anchor_selection.json").is_file()

    monkeypatch.setattr(external, "_atomic_json", original_atomic_json)
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=root,
        contract=contract,
        resume=True,
    )
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(values)
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert np.array_equal(
        np.flatnonzero(np.load(result.core_mask_path)), expected.core_sample_indices_
    )


def test_adaptive_selection_is_independent_of_resource_block_size(
    tmp_path: Path,
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    results: list[dict] = []
    for block in (2, 5):
        result = external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / f"adaptive-{block}",
            contract=_adaptive_contract(block=block),
        )
        selection = json.loads(
            (tmp_path / f"adaptive-{block}/adaptive_anchor_selection.json").read_text()
        )["selection_identity"]
        results.append(
            {
                "labels_sha256": json.loads(result.manifest_path.read_text())[
                    "labels_sha256"
                ],
                "seed_indices_sha256": selection["seed_indices_sha256"],
                "failure_index_list_sha256": selection[
                    "failure_index_list_sha256"
                ],
                "selected_anchor_indices_sha256": selection[
                    "selected_anchor_indices_sha256"
                ],
                "anchor_rows_sha256": selection["anchor_rows_sha256"],
            }
        )
    assert results[0] == results[1]


def test_adaptive_failure_cap_exceeded_is_terminal_and_never_falls_back(
    tmp_path: Path,
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    root = tmp_path / "blocked"
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="adaptive_failure_cap_exceeded",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=_adaptive_contract(failure_cap=2),
        )
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    failure = json.loads((root / "shortcut_failure.json").read_text())
    assert checkpoint["phase"] == "shortcut_blocked"
    assert failure["reason"] == "adaptive_failure_cap_exceeded"
    assert failure["details"]["first_pass_complete"] is False
    assert not (root / "neighbor_counts.npy").exists()


def test_reauthenticated_complete_failure_ledger_cannot_bypass_cap(
    tmp_path: Path,
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _adaptive_contract(block=3, failure_cap=5)
    root = tmp_path / "blocked-cap-tamper"
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="adaptive_failure_cap_exceeded",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    ledger = checkpoint["progress_ledgers"]["adaptive_failure_scan"]
    failures = checkpoint["adaptive_failure_indices"]
    assert ledger["committed_offset"] == len(values)
    assert len(failures) > contract.shortcut_failure_cap

    # Coordinate all self-authenticating state fields as an adversarial
    # reproducer.  Resume replay still observes the real >cap set and must not
    # allow selection merely because the ledger was marked complete.
    checkpoint["phase"] = "adaptive_failure_scan"
    ledger["complete"] = True
    ledger["result"] = {
        "first_pass_complete": True,
        "failure_indices": failures,
        "failure_indices_sha256": external._sample_indices_sha256(failures),
    }
    checkpoint["progress_ledgers_sha256"] = external._progress_ledgers_sha256(
        checkpoint["progress_ledgers"], identity=checkpoint["identity"]
    )
    _write_reauthenticated_checkpoint(checkpoint_path, checkpoint)

    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="exceeds the frozen cap",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )
    assert not (root / "adaptive_anchor_selection.json").exists()
    assert not (root / "run_manifest.json").exists()


def test_adaptive_terminal_rejects_tampered_first_pass_failure_set(
    tmp_path: Path,
) -> None:
    values = _adaptive_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _adaptive_contract()
    root = tmp_path / "adaptive"
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract
    )
    proof = json.loads(result.shortcut_proof_path.read_text())
    selection = json.loads(
        Path(proof["adaptive_selection_manifest_path"]).read_text()
    )["selection_identity"]
    failure_path = Path(selection["failure_indices_path"])
    with failure_path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        value = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="adaptive selection artifact mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract, resume=True
        )


def test_adaptive_seed_scan_rss_gate_precedes_full_scan(tmp_path: Path) -> None:
    vectors = _save(tmp_path / "vectors.npy", _adaptive_values())
    with pytest.raises(external.ExternalMemoryDBSCANError, match="RSS"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / "adaptive",
            contract=_adaptive_contract(
                max_rss_bytes=external._rss_bytes() + 1024
            ),
        )


def test_fixed_evenly_spaced_64_retains_fail_closed_negative(tmp_path: Path) -> None:
    values = np.zeros((100, 64), dtype=np.float32)
    fixed = set(external._deterministic_anchor_indices(100, 64).tolist())
    uncovered = next(index for index in range(100) if index not in fixed)
    values[uncovered, 0] = 10.0
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = external.ExternalDBSCANContract(
        eps=0.5,
        min_samples=3,
        query_block_size=2,
        checkpoint_interval_blocks=1,
        max_rss_bytes=external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
        shortcut_mode=external.ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_anchor_count=64,
        shortcut_query_block_size=11,
        exact_fallback_max_samples=99,
    )
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="EXACT_DBSCAN_COMPLEXITY_BLOCKED",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / "fixed64-blocked",
            contract=contract,
        )


def test_anchor_shortcut_is_elementwise_sklearn_exact_with_boundary_and_duplicates(
    tmp_path: Path,
) -> None:
    values = _all_core_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    expected = DBSCAN(eps=1.0, min_samples=3).fit(values)
    contract = _shortcut_contract()
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "shortcut",
        contract=contract,
    )

    labels = np.load(result.labels_path, allow_pickle=False)
    core = np.load(result.core_mask_path, allow_pickle=False)
    assert np.array_equal(labels, expected.labels_)
    assert np.array_equal(np.flatnonzero(core), expected.core_sample_indices_)
    assert labels.tolist() == [0] * len(values)
    assert core.tolist() == [True] * len(values)
    assert result.neighbor_counts_path is None
    assert result.shortcut_proof_path is not None

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    proof = json.loads(result.shortcut_proof_path.read_text(encoding="utf-8"))
    lower = np.load(proof["anchor_neighbor_lower_bounds_path"], allow_pickle=False)
    assert manifest["clustering_path"] == external.ALL_CORE_ONE_COMPONENT_SHORTCUT
    assert manifest["neighbor_counts_available"] is False
    assert manifest["neighbor_counts_path"] is None
    assert manifest["neighbor_counts_sha256"] is None
    assert manifest["approximation_used"] is False
    assert proof["exact_neighbor_counts_materialized"] is False
    assert proof["all_points_core_proven"] is True
    assert proof["single_epsilon_component_proven"] is True
    assert lower.tolist() == [3, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3]
    for name in (
        "all_core_certificate",
        "connectivity_certificate",
        "boundary_certificate",
        "cluster_partition",
    ):
        path = Path(manifest[f"{name}_path"])
        assert path.name == f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "PASS"
        assert payload["approximation_used"] is False
    boundary = json.loads(Path(manifest["boundary_certificate_path"]).read_text())
    assert boundary["comparison"] == "distance <= eps"
    assert boundary["recheck_dtype"] == "float64"
    assert boundary["float64_revalidated_row_count"] == len(values)
    all_core = json.loads(Path(manifest["all_core_certificate_path"]).read_text())
    assert all_core["min_samples"] == 3
    assert all_core["self_neighbor_counted_exactly_once"] is True
    connectivity = json.loads(
        Path(manifest["connectivity_certificate_path"]).read_text()
    )
    assert connectivity["attached_or_anchor_row_count"] == len(values)
    assert connectivity["every_close_row_attached_to_anchor_component"] is True
    reopened = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "shortcut",
        contract=contract,
        resume=True,
    )
    assert reopened.manifest_sha256 == result.manifest_sha256
    assert reopened.neighbor_counts_path is None


def test_anchor_shortcut_rechecks_identity_bound_self_after_gram_roundoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    num_samples = 2_000
    anchor_count = 273
    values = np.zeros((num_samples, 64), dtype=np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    anchor_indices = external._deterministic_anchor_indices(
        num_samples, anchor_count
    )
    injected_squared = np.linspace(2.0**-63, 2.0**-61, 113)
    real_matmul = np.matmul
    observed: dict[str, np.ndarray] = {}

    def inject_production_self_roundoff(left, right, *args, **kwargs):
        result = real_matmul(left, right, *args, **kwargs)
        if left.shape == (num_samples, 64) and right.shape == (64, anchor_count):
            for column, sample in enumerate(anchor_indices[:113].tolist()):
                result[int(sample), column] -= injected_squared[column] / 2.0
            block_squared = np.einsum("ij,ij->i", left, left)
            anchor_squared = np.einsum("ij,ij->i", right.T, right.T)
            observed["gram_squared"] = np.asarray(
                [
                    block_squared[int(sample)]
                    + anchor_squared[column]
                    - 2.0 * result[int(sample), column]
                    for column, sample in enumerate(anchor_indices[:113].tolist())
                ],
                dtype=np.float64,
            )
            observed["direct"] = np.asarray(
                [
                    np.linalg.norm(left[int(sample)] - right[:, column])
                    for column, sample in enumerate(anchor_indices[:113].tolist())
                ],
                dtype=np.float64,
            )
        return result

    monkeypatch.setattr(external.np, "matmul", inject_production_self_roundoff)
    contract = external.ExternalDBSCANContract(
        eps=0.02,
        min_samples=3,
        query_block_size=2_000,
        checkpoint_interval_blocks=1,
        max_rss_bytes=external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
        shortcut_mode=external.ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_anchor_count=anchor_count,
        shortcut_query_block_size=num_samples,
        exact_fallback_max_samples=num_samples,
    )
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "identity-self-roundoff",
        contract=contract,
    )

    assert observed["gram_squared"].min() == pytest.approx(2.0**-63)
    assert observed["gram_squared"].max() == pytest.approx(2.0**-61)
    assert np.array_equal(observed["direct"], np.zeros(113, dtype=np.float64))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    boundary = json.loads(
        Path(manifest["boundary_certificate_path"]).read_text(encoding="utf-8")
    )
    assert boundary["eps"] == 0.02
    assert boundary["near_boundary_direct_norm_recompute_count"] == 0
    assert (
        boundary["identity_bound_self_direct_norm_recompute_count"]
        == anchor_count
    )


def test_identity_self_recheck_does_not_accept_nonself_membership_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _all_core_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    real_matmul = np.matmul
    injected = False

    def inject_nonself_drift(left, right, *args, **kwargs):
        nonlocal injected
        result = real_matmul(left, right, *args, **kwargs)
        if not injected and left.shape == (3, 64) and right.shape == (64, 4):
            result[0, 1] = -0.500001
            injected = True
        return result

    monkeypatch.setattr(external.np, "matmul", inject_nonself_drift)
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="sklearn/float64 anchor counts differ",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / "nonself-drift",
            contract=_shortcut_contract(),
        )
    assert injected is True


def test_inconclusive_anchor_witness_falls_back_only_below_explicit_limit(
    tmp_path: Path,
) -> None:
    values = np.zeros((9, 64), dtype=np.float32)
    values[:, 0] = np.asarray([0.0, 0.01, 0.02, 10.0, 10.01, 10.02, 20.0, 20.01, 20.02])
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = external.ExternalDBSCANContract(
        **{**_shortcut_contract(anchors=3, fallback=9).__dict__, "eps": 0.05}
    )
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(values)
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=tmp_path / "fallback", contract=contract
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    failure = json.loads(
        (tmp_path / "fallback/shortcut_failure.json").read_text(encoding="utf-8")
    )
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert np.array_equal(
        np.flatnonzero(np.load(result.core_mask_path)), expected.core_sample_indices_
    )
    assert result.neighbor_counts_path is not None
    assert manifest["clustering_path"] == "three_pass_exact_radius_graph_v1"
    assert failure["status"] == "INCONCLUSIVE"
    assert failure["fallback_allowed"] is True
    assert failure["approximation_used"] is False


def test_shortcut_exact_fallback_resume_skips_retired_shortcut_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.zeros((9, 64), dtype=np.float32)
    values[:, 0] = np.asarray(
        [0.0, 0.01, 0.02, 10.0, 10.01, 10.02, 20.0, 20.01, 20.02]
    )
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = external.ExternalDBSCANContract(
        **{**_shortcut_contract(anchors=3, fallback=9).__dict__, "eps": 0.05}
    )
    original_checkpoint = external._checkpoint
    interrupted = {"done": False}

    def interrupt_after_core_union_checkpoint(*args, **kwargs):
        original_checkpoint(*args, **kwargs)
        if kwargs.get("phase") == "core_union" and not interrupted["done"]:
            interrupted["done"] = True
            raise RuntimeError("exact fallback interruption")

    monkeypatch.setattr(
        external, "_checkpoint", interrupt_after_core_union_checkpoint
    )
    root = tmp_path / "fallback-resume"
    with pytest.raises(RuntimeError, match="exact fallback interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    assert checkpoint["phase"] == "core_union"
    assert "progress_ledgers" not in checkpoint

    monkeypatch.setattr(external, "_checkpoint", original_checkpoint)
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=root,
        contract=contract,
        resume=True,
    )
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(values)
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert np.array_equal(
        np.flatnonzero(np.load(result.core_mask_path)), expected.core_sample_indices_
    )


def test_large_inconclusive_anchor_witness_is_explicit_complexity_block(
    tmp_path: Path,
) -> None:
    values = np.zeros((9, 64), dtype=np.float32)
    values[:, 0] = np.arange(9, dtype=np.float32) * 10.0
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = external.ExternalDBSCANContract(
        **{**_shortcut_contract(anchors=3, fallback=8).__dict__, "eps": 0.05}
    )
    root = tmp_path / "blocked"
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="EXACT_DBSCAN_COMPLEXITY_BLOCKED",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    failure = json.loads((root / "shortcut_failure.json").read_text())
    assert failure["fallback_allowed"] is False
    assert failure["approximation_used"] is False
    assert not (root / "labels.npy").exists()


def test_anchor_shortcut_resume_is_label_hash_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.zeros((16, 64), dtype=np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract(block=3, anchors=4)
    reference = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "reference",
        contract=contract,
    )
    reference_manifest = json.loads(reference.manifest_path.read_text())
    original_fit = external._fit_anchor_neighbors
    calls = {"count": 0}

    class InterruptingAnchorModel:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def radius_neighbors(self, *args, **kwargs):
            calls["count"] += 1
            # anchor graph, first committed sample block, then interruption
            if calls["count"] == 3:
                raise RuntimeError("anchor witness interruption")
            return self.wrapped.radius_neighbors(*args, **kwargs)

    def interrupted_fit(*args, **kwargs):
        model, version = original_fit(*args, **kwargs)
        return InterruptingAnchorModel(model), version

    monkeypatch.setattr(external, "_fit_anchor_neighbors", interrupted_fit)
    root = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="anchor witness interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    state = json.loads((root / "checkpoint.json").read_text())
    assert state["phase"] == "shortcut_anchor_scan"
    assert state["next_offset"] == 3

    monkeypatch.setattr(external, "_fit_anchor_neighbors", original_fit)
    resumed = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract, resume=True
    )
    resumed_manifest = json.loads(resumed.manifest_path.read_text())
    assert resumed_manifest["labels_sha256"] == reference_manifest["labels_sha256"]
    assert resumed_manifest["core_mask_sha256"] == reference_manifest["core_mask_sha256"]
    assert np.array_equal(np.load(resumed.labels_path), np.load(reference.labels_path))


def test_reviewer_fixed_next_offset_tamper_reproducer_cannot_publish_false_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _all_core_values()
    values[10] = 0.0
    values[10, 0] = 10.0
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract(block=3, anchors=4, fallback=0)
    original_fit = external._fit_anchor_neighbors
    calls = {"count": 0}

    class InterruptingAnchorModel:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def radius_neighbors(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("reviewer fixed interruption")
            return self.wrapped.radius_neighbors(*args, **kwargs)

    def interrupted_fit(*args, **kwargs):
        model, version = original_fit(*args, **kwargs)
        return InterruptingAnchorModel(model), version

    monkeypatch.setattr(external, "_fit_anchor_neighbors", interrupted_fit)
    root = tmp_path / "reviewer-fixed-tamper"
    with pytest.raises(RuntimeError, match="reviewer fixed interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["next_offset"] == 3
    checkpoint["next_offset"] = len(values)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")

    monkeypatch.setattr(external, "_fit_anchor_neighbors", original_fit)
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="checkpoint authentication mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )
    assert not (root / "run_manifest.json").exists()
    assert not (root / "labels.npy").exists()


def test_anchor_shortcut_resume_replays_and_rejects_tampered_committed_lower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.zeros((16, 64), dtype=np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract(block=3, anchors=4)
    original_fit = external._fit_anchor_neighbors
    calls = {"count": 0}

    class InterruptingAnchorModel:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def radius_neighbors(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("lower replay interruption")
            return self.wrapped.radius_neighbors(*args, **kwargs)

    def interrupted_fit(*args, **kwargs):
        model, version = original_fit(*args, **kwargs)
        return InterruptingAnchorModel(model), version

    monkeypatch.setattr(external, "_fit_anchor_neighbors", interrupted_fit)
    root = tmp_path / "lower-replay-tamper"
    with pytest.raises(RuntimeError, match="lower replay interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    lower_path = root / "shortcut_anchor_neighbor_lower_bounds.partial.npy"
    lower = np.load(lower_path, mmap_mode="r+")
    lower[0] = 0
    lower.flush()
    del lower

    monkeypatch.setattr(external, "_fit_anchor_neighbors", original_fit)
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="lower-bound resume replay mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )
    assert not (root / "run_manifest.json").exists()


def test_anchor_shortcut_validates_full_lower_array_before_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _all_core_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract()
    original_reconcile = external._reconcile_promoted_array

    def corrupt_after_reconcile(*args, **kwargs):
        path = original_reconcile(*args, **kwargs)
        if kwargs.get("label") == "anchor-neighbor lower bounds":
            lower = np.load(path, mmap_mode="r+")
            lower[-1] = 0
            lower.flush()
            del lower
        return path

    monkeypatch.setattr(
        external, "_reconcile_promoted_array", corrupt_after_reconcile
    )
    root = tmp_path / "final-lower-tamper"
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="final lower-bound ledger/array mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    assert not (root / "run_manifest.json").exists()
    assert not (root / "labels.npy").exists()


def test_source_mutation_during_shortcut_scan_is_rejected_before_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _all_core_values()
    vectors_path = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract()
    original_lower_block = external._anchor_lower_block
    mutated = {"done": False}

    def mutate_source_after_first_query(*args, **kwargs):
        result = original_lower_block(*args, **kwargs)
        if not mutated["done"]:
            mutated["done"] = True
            source = np.load(vectors_path, mmap_mode="r+")
            source[5, 1] = np.float32(0.001)
            source.flush()
            del source
        return result

    monkeypatch.setattr(external, "_anchor_lower_block", mutate_source_after_first_query)
    root = tmp_path / "source-mutation"
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="vector source .* identity changed before shortcut PASS",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors_path,
            work_dir=root,
            contract=contract,
        )
    assert not (root / "run_manifest.json").exists()
    assert not (root / "shortcut_proof.json").exists()
    assert not (root / "labels.npy").exists()


def test_anchor_shortcut_terminal_rejects_tampered_lower_bound_witness(
    tmp_path: Path,
) -> None:
    values = _all_core_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract()
    root = tmp_path / "shortcut"
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract
    )
    proof = json.loads(result.shortcut_proof_path.read_text())
    lower_path = Path(proof["anchor_neighbor_lower_bounds_path"])
    with lower_path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        value = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="proof artifact mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract, resume=True
        )


def test_anchor_shortcut_terminal_rejects_tampered_split_certificate(
    tmp_path: Path,
) -> None:
    values = _all_core_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract()
    root = tmp_path / "shortcut"
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract
    )
    manifest = json.loads(result.manifest_path.read_text())
    boundary_path = Path(manifest["boundary_certificate_path"])
    boundary = json.loads(boundary_path.read_text())
    boundary["uncertain_edges_accepted"] = 1
    boundary_path.write_text(json.dumps(boundary, sort_keys=True) + "\n")
    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="split certificate closure mismatch",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract, resume=True
        )


def test_anchor_shortcut_resume_rejects_resigned_self_recheck_count(
    tmp_path: Path,
) -> None:
    values = _all_core_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract()
    root = tmp_path / "resigned-self-recheck-count"
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    proof_path = Path(manifest["shortcut_proof_path"])
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    boundary_path = Path(manifest["boundary_certificate_path"])
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundary["identity_bound_self_direct_norm_recompute_count"] = (
        int(proof["anchor_count"]) - 1
    )
    boundary_path.write_text(
        json.dumps(boundary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    boundary_sha = external._sha256_file(boundary_path)
    proof["boundary_certificate_sha256"] = boundary_sha
    proof_path.write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["boundary_certificate_sha256"] = boundary_sha
    manifest["shortcut_proof_sha256"] = external._sha256_file(proof_path)
    result.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="terminal boundary certificate is incomplete",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )


def test_terminal_rejects_resigned_proof_threshold_drift(
    tmp_path: Path,
) -> None:
    values = _all_core_values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract()
    root = tmp_path / "resigned-proof-drift"
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors, work_dir=root, contract=contract
    )
    proof_path = result.shortcut_proof_path
    assert proof_path is not None
    proof = json.loads(proof_path.read_text())
    proof["min_samples"] = 1
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    manifest = json.loads(result.manifest_path.read_text())
    manifest["shortcut_proof_sha256"] = external._sha256_file(proof_path)
    result.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(
        external.ExternalMemoryDBSCANError,
        match="terminal shortcut proof is incomplete",
    ):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )


def test_anchor_shortcut_rss_gate_precedes_full_sample_scan(tmp_path: Path) -> None:
    values = _all_core_values(20)
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _shortcut_contract(
        block=20,
        max_rss_bytes=external._rss_bytes() + 1024,
    )
    with pytest.raises(external.ExternalMemoryDBSCANError, match="RSS"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / "shortcut",
            contract=contract,
        )


@pytest.mark.parametrize("seed", range(8))
def test_external_labels_are_elementwise_sklearn_exact(tmp_path: Path, seed: int) -> None:
    random = np.random.default_rng(seed)
    values = random.normal(size=(113, 4)).astype(np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "external",
        contract=_contract(block=5),
    )
    expected = DBSCAN(eps=0.37, min_samples=3).fit_predict(values)
    actual = np.load(result.labels_path, allow_pickle=False)
    assert actual.dtype == np.dtype(np.intp)
    assert np.array_equal(actual, expected)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sklearn_dbscan_label_semantics_preserved"] is True
    assert manifest["all_neighborhoods_materialized_simultaneously"] is False
    assert manifest["scientific_identity"]["sklearn_version"] == sklearn.__version__


def test_ambiguous_border_uses_earliest_core_component_like_sklearn(
    tmp_path: Path,
) -> None:
    values = np.asarray(
        [[0.0], [-0.05], [-0.10], [-0.15], [2.0], [2.05], [2.10], [2.15], [1.0]],
        dtype=np.float64,
    )
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = external.ExternalDBSCANContract(
        eps=1.0,
        min_samples=4,
        query_block_size=2,
        checkpoint_interval_blocks=1,
        max_rss_bytes=external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
    )
    expected_model = DBSCAN(eps=1.0, min_samples=4).fit(values)
    assert expected_model.labels_[-1] == 0
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "external",
        contract=contract,
    )
    assert np.array_equal(
        np.load(result.labels_path, allow_pickle=False), expected_model.labels_
    )


def test_component_labels_are_numbered_by_minimum_core_index(tmp_path: Path) -> None:
    values = np.asarray(
        [[10.0], [10.01], [9.99], [10.02], [50.0], [0.0], [0.01], [-0.01], [0.02]],
        dtype=np.float32,
    )
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = external.ExternalDBSCANContract(
        eps=0.05,
        min_samples=3,
        query_block_size=2,
        checkpoint_interval_blocks=1,
        max_rss_bytes=external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
    )
    expected = DBSCAN(eps=0.05, min_samples=3).fit_predict(values)
    assert expected.tolist() == [0, 0, 0, 0, -1, 1, 1, 1, 1]
    result = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "external",
        contract=contract,
    )
    assert np.array_equal(np.load(result.labels_path), expected)


def test_resume_replays_only_uncommitted_block_and_is_hash_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.random.default_rng(91).normal(size=(79, 3)).astype(np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _contract(block=4)
    reference = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "reference",
        contract=contract,
    )
    reference_labels = np.load(reference.labels_path, allow_pickle=False).copy()
    reference_hash = json.loads(reference.manifest_path.read_text())["labels_sha256"]

    original_fit = external._fit_neighbors
    calls = {"count": 0}

    class InterruptingModel:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def radius_neighbors(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("fixture interruption")
            return self.wrapped.radius_neighbors(*args, **kwargs)

    def interrupted_fit(*args, **kwargs):
        model, version, method = original_fit(*args, **kwargs)
        return InterruptingModel(model), version, method

    monkeypatch.setattr(external, "_fit_neighbors", interrupted_fit)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / "resumed",
            contract=contract,
        )
    checkpoint = json.loads(
        (tmp_path / "resumed/checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["phase"] == "neighbor_counts"
    assert checkpoint["next_offset"] == 8

    monkeypatch.setattr(external, "_fit_neighbors", original_fit)
    resumed = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "resumed",
        contract=contract,
        resume=True,
    )
    resumed_manifest = json.loads(resumed.manifest_path.read_text())
    assert np.array_equal(
        np.load(resumed.labels_path, allow_pickle=False), reference_labels
    )
    assert resumed_manifest["labels_sha256"] == reference_hash


def test_checkpoint_rejects_vector_or_contract_drift(tmp_path: Path) -> None:
    values = np.random.default_rng(3).normal(size=(20, 2)).astype(np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    root = tmp_path / "external"
    contract = _contract(block=2)

    original_fit = external._fit_neighbors

    class StopImmediately:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def radius_neighbors(self, *_args, **_kwargs):
            raise RuntimeError("stop")

    def stopping_fit(*args, **kwargs):
        model, version, method = original_fit(*args, **kwargs)
        return StopImmediately(model), version, method

    external._fit_neighbors = stopping_fit
    try:
        with pytest.raises(RuntimeError, match="stop"):
            external.fit_external_memory_dbscan(
                vectors_path=vectors, work_dir=root, contract=contract
            )
    finally:
        external._fit_neighbors = original_fit

    changed = external.ExternalDBSCANContract(
        **{**contract.__dict__, "eps": 0.38}
    )
    with pytest.raises(external.ExternalMemoryDBSCANError, match="identity mismatch"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=changed,
            resume=True,
        )


def test_union_checkpoint_replay_is_idempotent_and_label_hash_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.zeros((20, 2), dtype=np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _contract(block=4)
    reference = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "reference",
        contract=contract,
    )
    reference_hash = json.loads(reference.manifest_path.read_text())["labels_sha256"]

    original_fit = external._fit_neighbors
    calls = {"count": 0}

    class UnionInterrupt:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def radius_neighbors(self, *args, **kwargs):
            calls["count"] += 1
            # Five count blocks, two committed union blocks, then interrupt
            # inside the third union block.  Replaying from offset 8 sees a
            # partially compressed parent array and must remain idempotent.
            if calls["count"] == 8:
                raise RuntimeError("union interruption")
            return self.wrapped.radius_neighbors(*args, **kwargs)

    def interrupted_fit(*args, **kwargs):
        model, version, method = original_fit(*args, **kwargs)
        return UnionInterrupt(model), version, method

    monkeypatch.setattr(external, "_fit_neighbors", interrupted_fit)
    with pytest.raises(RuntimeError, match="union interruption"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / "resumed",
            contract=contract,
        )
    state = json.loads((tmp_path / "resumed/checkpoint.json").read_text())
    assert state["phase"] == "core_union"
    assert state["next_offset"] == 8

    monkeypatch.setattr(external, "_fit_neighbors", original_fit)
    resumed = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "resumed",
        contract=contract,
        resume=True,
    )
    assert json.loads(resumed.manifest_path.read_text())["labels_sha256"] == reference_hash


def test_rss_budget_fails_closed_before_unbounded_query(tmp_path: Path) -> None:
    values = np.zeros((300, 2), dtype=np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _contract(block=300, max_rss_bytes=external._rss_bytes() + 1024)
    with pytest.raises(external.ExternalMemoryDBSCANError, match="RSS"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=tmp_path / "external",
            contract=contract,
        )


@pytest.mark.parametrize(
    ("transition_phase", "partial_name", "final_name"),
    (
        (
            "neighbor_counts_finalize",
            "neighbor_counts.partial.npy",
            "neighbor_counts.npy",
        ),
        ("labels_finalize", "labels.partial.npy", "labels.npy"),
    ),
)
def test_two_phase_array_promotion_recovers_after_first_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transition_phase: str,
    partial_name: str,
    final_name: str,
) -> None:
    values = np.random.default_rng(121).normal(size=(67, 3)).astype(np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _contract(block=4)
    root = tmp_path / transition_phase
    original_checkpoint = external._checkpoint
    interrupted = {"done": False}

    def interrupt_after_ready(*args, **kwargs):
        original_checkpoint(*args, **kwargs)
        if kwargs.get("phase") == transition_phase and not interrupted["done"]:
            interrupted["done"] = True
            raise RuntimeError(f"crash-after-{transition_phase}")

    monkeypatch.setattr(external, "_checkpoint", interrupt_after_ready)
    with pytest.raises(RuntimeError, match=f"crash-after-{transition_phase}"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
        )
    state = json.loads((root / "checkpoint.json").read_text())
    assert state["phase"] == transition_phase
    os.replace(root / partial_name, root / final_name)

    monkeypatch.setattr(external, "_checkpoint", original_checkpoint)
    resumed = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=root,
        contract=contract,
        resume=True,
    )
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit_predict(
        values
    )
    assert np.array_equal(np.load(resumed.labels_path), expected)


def test_two_phase_array_promotion_rejects_tampered_renamed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.random.default_rng(122).normal(size=(31, 2)).astype(np.float32)
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _contract(block=3)
    root = tmp_path / "tampered"
    original_checkpoint = external._checkpoint

    def interrupt_after_ready(*args, **kwargs):
        original_checkpoint(*args, **kwargs)
        if kwargs.get("phase") == "neighbor_counts_finalize":
            raise RuntimeError("crash-after-ready")

    monkeypatch.setattr(external, "_checkpoint", interrupt_after_ready)
    with pytest.raises(RuntimeError, match="crash-after-ready"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=root, contract=contract
        )
    os.replace(root / "neighbor_counts.partial.npy", root / "neighbor_counts.npy")
    with (root / "neighbor_counts.npy").open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        byte = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([byte[0] ^ 1]))
    monkeypatch.setattr(external, "_checkpoint", original_checkpoint)
    with pytest.raises(external.ExternalMemoryDBSCANError, match="hash mismatch"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors,
            work_dir=root,
            contract=contract,
            resume=True,
        )
