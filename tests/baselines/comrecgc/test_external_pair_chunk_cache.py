from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.baselines.comrecgc import external_pair_chunk_cache as chunk_cache
from src.baselines.comrecgc.external_memory_dbscan import ExternalMemoryDBSCANError
from src.baselines.comrecgc.external_memory_recourse import ExternalPairStore, _stable_hash
from src.baselines.comrecgc.external_memory_recourse import (
    summarize_proven_one_cluster_external,
)


def _portable_preallocate(path: Path, *, size: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(chunk_cache, "_preallocate_file", _portable_preallocate)
    monkeypatch.setattr(
        chunk_cache, "_allocated_bytes", lambda path: path.stat().st_size
    )
    monkeypatch.setattr(chunk_cache, "_statvfs_free_bytes", lambda _path: 10**12)
    identity = {"dataset": "aids", "seed": 0, "theta": 0.1}
    chunk_identities = [
        {"candidate_start": 0, "candidate_stop": 2},
        {"candidate_start": 2, "candidate_stop": 3},
    ]
    parent_count = 3
    candidate_count = 3
    store = ExternalPairStore(
        root=tmp_path / "source" / "pair_store",
        scientific_identity=identity,
        max_rss_bytes=10**12,
    )
    store.append(
        chunk_index=0,
        pairs=np.asarray(
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
            dtype=np.int64,
        ),
        vectors=np.arange(24, dtype=np.float32).reshape(6, 4),
        chunk_identity=chunk_identities[0],
    )
    store.append(
        chunk_index=1,
        pairs=np.asarray([[0, 2], [1, 2], [2, 2]], dtype=np.int64),
        vectors=np.arange(24, 36, dtype=np.float32).reshape(3, 4),
        chunk_identity=chunk_identities[1],
    )
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    owner_root = tmp_path / "source"
    return {
        "source_checkpoint_path": store.state_path,
        "source_owner_root": owner_root,
        "persistent_root": tmp_path / "fresh" / "adoption",
        "local_cache_root": tmp_path / "local" / "cache",
        "scratch_lock_path": tmp_path / "local" / "cache.lock",
        "expected_scientific_identity": identity,
        "expected_chunk_identities": chunk_identities,
        "parent_count": parent_count,
        "candidate_count": candidate_count,
        "min_local_free_bytes": 128,
        "proc_root": proc_root,
    }


def test_cartesian_chunk_cache_matches_materialized_arrays_and_survives_cache_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    result = chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)

    expected_pairs = np.asarray(
        [[parent, candidate] for candidate in range(3) for parent in range(3)],
        dtype=np.int64,
    )
    expected_vectors = np.arange(36, dtype=np.float32).reshape(9, 4)
    assert np.array_equal(result.pairs[:], expected_pairs)
    assert np.array_equal(np.load(result.vectors_path), expected_vectors)
    pairs_path = tmp_path / "pairs.npy"
    with pairs_path.open("wb") as handle:
        np.save(handle, expected_pairs, allow_pickle=False)
    assert result.pairs.logical_npy_sha256 == chunk_cache._sha256_file(pairs_path)

    reopened = chunk_cache.validate_cartesian_chunk_vector_cache(
        result.manifest_path,
        require_cache=True,
        proc_root=kwargs["proc_root"],
    )
    assert reopened.vectors_sha256 == result.vectors_sha256
    result.vectors_path.unlink()
    authority_only = chunk_cache.validate_cartesian_chunk_vector_cache(
        result.manifest_path,
        require_cache=False,
        proc_root=kwargs["proc_root"],
    )
    assert authority_only.vectors_sha256 == result.vectors_sha256
    with pytest.raises(ExternalMemoryDBSCANError, match="LOCAL_ARTIFACT_MISSING"):
        chunk_cache.validate_cartesian_chunk_vector_cache(
            result.manifest_path,
            require_cache=True,
            proc_root=kwargs["proc_root"],
        )
    with pytest.raises(
        ExternalMemoryDBSCANError,
        match="LOCAL_ARTIFACT_MISSING_FRESH_REBUILD_REQUIRED",
    ):
        chunk_cache.materialize_cartesian_chunk_vector_cache(
            **kwargs, resume=True
        )


def test_cartesian_chunk_cache_rejects_pair_order_even_with_resigned_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    checkpoint_path = Path(kwargs["source_checkpoint_path"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    row = checkpoint["chunks"][1]
    pair_path = Path(row["pairs_path"])
    pairs = np.load(pair_path)
    pairs[[0, 1]] = pairs[[1, 0]]
    with pair_path.open("wb") as handle:
        np.save(handle, pairs, allow_pickle=False)
    row["pairs_sha256"] = chunk_cache._sha256_file(pair_path)
    checkpoint_path.write_text(
        json.dumps(checkpoint, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ExternalMemoryDBSCANError, match="CARTESIAN_ORDER_MISMATCH"):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)


def test_cartesian_chunk_cache_rejects_active_source_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    proc_root = Path(kwargs["proc_root"])
    process = proc_root / "12345"
    process.mkdir()
    (process / "cmdline").write_bytes(
        b"python\0runner.py\0" + str(kwargs["source_owner_root"]).encode("utf-8")
    )

    with pytest.raises(ExternalMemoryDBSCANError, match="OWNER_PROCESS_ACTIVE"):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)
    audit = chunk_cache.audit_cartesian_chunk_source(
        source_checkpoint_path=kwargs["source_checkpoint_path"],
        source_owner_root=kwargs["source_owner_root"],
        output_path=tmp_path / "audit.json",
        expected_scientific_identity=kwargs["expected_scientific_identity"],
        expected_chunk_identities=kwargs["expected_chunk_identities"],
        parent_count=kwargs["parent_count"],
        candidate_count=kwargs["candidate_count"],
        proc_root=kwargs["proc_root"],
    )
    assert audit["status"] == "PASS"
    assert audit["scientific_source_closure_pass"] is True
    assert audit["diagnostic_only"] is True
    assert audit["eligible_for_adoption"] is False


def test_cartesian_chunk_cache_allows_fresh_consumer_parent_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    proc_root = Path(kwargs["proc_root"])
    process = proc_root / "12346"
    process.mkdir()
    (process / "cmdline").write_bytes(
        b"python\0run_comrecgc_standardized_continuation.py\0"
        + str(kwargs["source_owner_root"]).encode("utf-8")
    )
    monkeypatch.setattr(chunk_cache.os, "getppid", lambda: 12346)

    result = chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)
    assert result.manifest_path.is_file()


def test_cartesian_chunk_cache_headroom_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(chunk_cache, "_statvfs_free_bytes", lambda _path: 128)
    with pytest.raises(ExternalMemoryDBSCANError, match="LOCAL_HEADROOM_BLOCKED"):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)


def test_cartesian_chunk_cache_rejects_sparse_nominal_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(chunk_cache, "_allocated_bytes", lambda _path: 0)
    with pytest.raises(ExternalMemoryDBSCANError, match="ALLOCATION_EVIDENCE_MISMATCH"):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)


def test_cartesian_chunk_cache_replays_fallocate_after_precheckpoint_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    real_preallocate = chunk_cache._preallocate_file
    calls = 0

    def allocate_then_crash(path: Path, *, size: int) -> None:
        nonlocal calls
        calls += 1
        real_preallocate(path, size=size)
        if calls == 1:
            raise RuntimeError("allocation-before-checkpoint")

    monkeypatch.setattr(chunk_cache, "_preallocate_file", allocate_then_crash)
    with pytest.raises(RuntimeError, match="allocation-before-checkpoint"):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)
    state = chunk_cache._load_checkpoint(
        Path(kwargs["persistent_root"]) / "checkpoint.json"
    )
    assert state["phase"] == "allocate_cache"

    result = chunk_cache.materialize_cartesian_chunk_vector_cache(
        **kwargs, resume=True
    )
    assert calls == 2
    terminal = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert terminal["allocation_complete_authenticated"] is True
    allocation_state = chunk_cache._load_checkpoint(
        Path(terminal["allocation_checkpoint_path"])
    )
    assert allocation_state["phase"] == "cache_ready"
    assert allocation_state["allocation_evidence"]["allocation_complete"] is True


def test_cartesian_chunk_cache_rejects_resigned_allocation_evidence_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    original = chunk_cache._write_checkpoint

    def stop_after_allocation(path: Path, payload: object) -> dict[str, object]:
        result = original(path, payload)
        if result.get("phase") == "allocation_complete":
            raise RuntimeError("allocation checkpoint stop")
        return result

    monkeypatch.setattr(chunk_cache, "_write_checkpoint", stop_after_allocation)
    with pytest.raises(RuntimeError, match="allocation checkpoint stop"):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)
    state_path = Path(kwargs["persistent_root"]) / "checkpoint.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["allocation_evidence"]["physical_allocation_verified"] = False
    state["checkpoint_payload_sha256"] = chunk_cache._payload_sha256(state)
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(chunk_cache, "_write_checkpoint", original)
    with pytest.raises(
        ExternalMemoryDBSCANError, match="ALLOCATION_EVIDENCE_MISMATCH"
    ):
        chunk_cache.materialize_cartesian_chunk_vector_cache(
            **kwargs, resume=True
        )


def test_cartesian_chunk_cache_rejects_resigned_cursor_jump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    original = chunk_cache._write_checkpoint
    def interrupt_after_first_chunk(path: Path, payload: object) -> dict[str, object]:
        result = original(path, payload)
        if (
            result.get("phase") == "copy_chunks"
            and result.get("next_chunk_index") == 1
        ):
            raise RuntimeError("injected crash")
        return result

    monkeypatch.setattr(chunk_cache, "_write_checkpoint", interrupt_after_first_chunk)
    with pytest.raises(RuntimeError, match="injected crash"):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)

    state_path = Path(kwargs["persistent_root"]) / "checkpoint.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["next_chunk_index"] = 2
    state["next_row_offset"] = 9
    state["checkpoint_payload_sha256"] = chunk_cache._payload_sha256(state)
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(chunk_cache, "_write_checkpoint", original)
    with pytest.raises(
        ExternalMemoryDBSCANError,
        match="COMMITTED_PREFIX_MISMATCH|CHECKPOINT_CURSOR_MISMATCH",
    ):
        chunk_cache.materialize_cartesian_chunk_vector_cache(
            **kwargs, resume=True
        )


def test_cartesian_chunk_cache_recovers_rename_before_terminal_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    original_replace = chunk_cache.os.replace
    injected = False

    def replace_then_crash(source: object, target: object) -> None:
        nonlocal injected
        original_replace(source, target)
        if str(target).endswith("recourse_vectors.npy") and not injected:
            injected = True
            raise RuntimeError("rename window")

    monkeypatch.setattr(chunk_cache.os, "replace", replace_then_crash)
    with pytest.raises(RuntimeError, match="rename window"):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)
    monkeypatch.setattr(chunk_cache.os, "replace", original_replace)
    result = chunk_cache.materialize_cartesian_chunk_vector_cache(
        **kwargs, resume=True
    )
    assert result.manifest_path.is_file()


def test_cartesian_chunk_cache_rejects_source_mutation_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _source_fixture(tmp_path, monkeypatch)
    original_validate = chunk_cache._validate_chunk_source
    calls = 0

    def mutate_between_validations(**inner: object) -> dict[str, object]:
        nonlocal calls
        result = original_validate(**inner)
        calls += 1
        if calls == 2:
            vector_path = Path(result["chunks"][0]["vectors_path"])
            with vector_path.open("r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                value = handle.read(1)
                handle.seek(-1, os.SEEK_END)
                handle.write(bytes([value[0] ^ 1]))
        return result

    monkeypatch.setattr(chunk_cache, "_validate_chunk_source", mutate_between_validations)
    with pytest.raises(
        ExternalMemoryDBSCANError,
        match="CHANGED_DURING_CACHE_BUILD|TERMINAL_HASH_MISMATCH",
    ):
        chunk_cache.materialize_cartesian_chunk_vector_cache(**kwargs)


def test_cartesian_pair_index_view_numpy_indexing() -> None:
    pairs = chunk_cache.CartesianPairIndexView(
        parent_count=3, candidate_count=2, logical_npy_sha256="0" * 64
    )
    assert pairs.shape == (6, 2)
    assert pairs[-1].tolist() == [2, 1]
    assert pairs[1:6:2].tolist() == [[1, 0], [0, 1], [2, 1]]
    assert pairs[np.asarray([True, False, False, True, False, False])].tolist() == [
        [0, 0],
        [0, 1],
    ]
    assert pairs[[0, -1], 1].tolist() == [0, 1]


def test_implicit_cartesian_pairs_preserve_one_cluster_summary_exactly(
    tmp_path: Path,
) -> None:
    import sklearn
    import torch

    from src.baselines.comrecgc.external_memory_dbscan import (
        ALL_CORE_ONE_COMPONENT_SHORTCUT,
        ExternalDBSCANContract,
        _rss_bytes,
        fit_external_memory_dbscan,
    )

    vectors_path = tmp_path / "vectors.npy"
    pairs_path = tmp_path / "pairs.npy"
    vectors = np.asarray(
        [[0.010, 0.000], [0.011, 0.000], [0.012, 0.000], [0.013, 0.000]],
        dtype=np.float32,
    )
    pairs = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    with vectors_path.open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    with pairs_path.open("wb") as handle:
        np.save(handle, pairs, allow_pickle=False)
    vector_map = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    pair_map = np.load(pairs_path, mmap_mode="r", allow_pickle=False)
    dbscan = fit_external_memory_dbscan(
        vectors_path=vectors_path,
        work_dir=tmp_path / "dbscan",
        contract=ExternalDBSCANContract(
            eps=0.01,
            min_samples=2,
            query_block_size=2,
            checkpoint_interval_blocks=1,
            max_rss_bytes=_rss_bytes() + 512 * 1024**2,
            expected_sklearn_version=sklearn.__version__,
            shortcut_mode=ALL_CORE_ONE_COMPONENT_SHORTCUT,
            shortcut_anchor_count=4,
            shortcut_query_block_size=2,
            exact_fallback_max_samples=0,
        ),
    )

    def greedy(*, counterfactual_covering: object, graphs_covered_by: object, k: int):
        assert k == 1
        covering = counterfactual_covering  # type: ignore[assignment]
        return {1: (0, len(covering[0]))}

    pair_sha = chunk_cache._sha256_file(pairs_path)
    physical = summarize_proven_one_cluster_external(
        work_dir=tmp_path / "physical-summary",
        dbscan_manifest_path=dbscan.manifest_path,
        dbscan_manifest_sha256=dbscan.manifest_sha256,
        recourse_vectors=vector_map,
        pair_indices=pair_map,
        pairs_sha256=pair_sha,
        radius=0.1,
        theta=0.1,
        recourse_size=1,
        official_greedy=greedy,
        torch_module=torch,
        max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        block_size=2,
    )
    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps({"run_complete": True}, sort_keys=True), encoding="utf-8"
    )
    implicit = summarize_proven_one_cluster_external(
        work_dir=tmp_path / "implicit-summary",
        dbscan_manifest_path=dbscan.manifest_path,
        dbscan_manifest_sha256=dbscan.manifest_sha256,
        recourse_vectors=vector_map,
        pair_indices=chunk_cache.CartesianPairIndexView(
            parent_count=2,
            candidate_count=2,
            logical_npy_sha256=pair_sha,
        ),
        pairs_sha256=pair_sha,
        pair_authority_manifest_path=authority,
        pair_authority_manifest_sha256=chunk_cache._sha256_file(authority),
        radius=0.1,
        theta=0.1,
        recourse_size=1,
        official_greedy=greedy,
        torch_module=torch,
        max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        block_size=2,
    )
    assert implicit.official_result == physical.official_result
    assert implicit.selected == physical.selected
