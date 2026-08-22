from __future__ import annotations

import json
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
