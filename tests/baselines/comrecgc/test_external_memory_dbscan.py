from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.baselines.comrecgc import external_memory_dbscan as external


sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import DBSCAN  # noqa: E402


def _save(path: Path, values: np.ndarray) -> Path:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    return path


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
    reopened = external.fit_external_memory_dbscan(
        vectors_path=vectors,
        work_dir=tmp_path / "shortcut",
        contract=contract,
        resume=True,
    )
    assert reopened.manifest_sha256 == result.manifest_sha256
    assert reopened.neighbor_counts_path is None


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
