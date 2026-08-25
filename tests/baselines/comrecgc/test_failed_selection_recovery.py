from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.baselines.comrecgc import external_memory_dbscan as external
from src.baselines.comrecgc import failed_selection_recovery as recovery
from src.baselines.comrecgc.failed_selection_recovery import (
    FailedSelectionRecoveryError,
    FailedSelectionRecoverySource,
    fit_promoted_failed_selection_component_recovery,
    promote_failed_adaptive_selection_for_component_recovery,
)


sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import DBSCAN  # noqa: E402


def _save(path: Path, values: np.ndarray) -> Path:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    return path


def _values() -> np.ndarray:
    values = np.zeros((11, 64), dtype=np.float64)
    values[:, 0] = np.asarray(
        [0.0, 0.0, 0.001, 0.019, 0.030, 0.031, 0.032,
         -0.019, -0.030, -0.031, -0.032],
        dtype=np.float64,
    )
    return values


def _contract(*, block: int = 2) -> external.ExternalDBSCANContract:
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
        exact_fallback_max_samples=0,
    )


def _failed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, external.ExternalDBSCANContract, FailedSelectionRecoverySource]:
    values = _values()
    vectors = _save(tmp_path / "vectors.npy", values)
    contract = _contract()
    source_root = tmp_path / "failed-source"
    original = external._fit_adaptive_disconnected_component_recovery

    def stop_as_c766(**kwargs: object) -> None:
        root = Path(kwargs["root"])
        identity = kwargs["identity"]
        state_path = Path(kwargs["state_path"])
        ledgers = external._load_progress_ledgers(
            external._load_checkpoint(state_path),
            identity=identity,
            num_samples=len(values),
        )
        failure = external._shortcut_failure(
            root=root,
            identity=identity,
            reason="anchor_epsilon_graph_disconnected",
            num_samples=len(values),
            fallback_limit=0,
            details={
                "anchor_count": len(kwargs["anchor_indices"]),
                "anchor_component_reached_count": 1,
                "anchor_edge_count": len(kwargs["anchor_edges"]),
            },
        )
        extra = external._progress_checkpoint_extra(ledgers, identity=identity)
        extra.update(external._adaptive_selection_checkpoint_fields(root))
        extra.update(
            {
                "shortcut_failure_path": failure["path"],
                "shortcut_failure_sha256": failure["sha256"],
                "shortcut_approximation_used": False,
            }
        )
        external._checkpoint(
            state_path,
            identity=identity,
            phase="shortcut_blocked",
            next_offset=0,
            peak_rss_bytes=external._rss_bytes(),
            extra=extra,
        )
        raise external.ExternalMemoryDBSCANError(
            "EXACT_DBSCAN_COMPLEXITY_BLOCKED:fixture-c766"
        )

    monkeypatch.setattr(
        external, "_fit_adaptive_disconnected_component_recovery", stop_as_c766
    )
    with pytest.raises(external.ExternalMemoryDBSCANError, match="fixture-c766"):
        external.fit_external_memory_dbscan(
            vectors_path=vectors, work_dir=source_root, contract=contract
        )
    monkeypatch.setattr(
        external, "_fit_adaptive_disconnected_component_recovery", original
    )
    selection = json.loads(
        (source_root / "adaptive_anchor_selection.json").read_text()
    )["selection_identity"]
    source = FailedSelectionRecoverySource(
        checkpoint_path=source_root / "checkpoint.json",
        checkpoint_sha256=external._sha256_file(source_root / "checkpoint.json"),
        selection_manifest_path=source_root / "adaptive_anchor_selection.json",
        selection_manifest_sha256=external._sha256_file(
            source_root / "adaptive_anchor_selection.json"
        ),
        failure_artifact_path=source_root / "shortcut_failure.json",
        failure_artifact_sha256=external._sha256_file(
            source_root / "shortcut_failure.json"
        ),
        failure_indices_sha256=selection["failure_indices_sha256"],
        anchor_indices_sha256=selection["anchor_indices_sha256"],
        anchor_rows_sha256=selection["anchor_rows_sha256"],
    )
    return vectors, source_root, contract, source


def _promote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, external.ExternalDBSCANContract, FailedSelectionRecoverySource]:
    vectors, source_root, contract, source = _failed_source(tmp_path, monkeypatch)
    receipt = tmp_path / "failed_selection_adoption_receipt.json"
    receipt.write_text('{"status":"RECOVERY_ONLY_READY"}\n', encoding="utf-8")
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    fresh = stage_root / "dbscan"
    promote_failed_adaptive_selection_for_component_recovery(
        vectors_path=vectors,
        work_dir=fresh,
        source=source,
        contract=contract,
        expected_vectors_sha256=external._sha256_file(vectors),
        adoption_receipt_path=receipt,
        adoption_receipt_sha256=external._sha256_file(receipt),
        source_authority_sha256="a" * 64,
    )
    return vectors, source_root, contract, source


def test_fresh_promotion_rebinds_paths_and_never_replays_seed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vectors, source_root, contract, _source = _promote(tmp_path, monkeypatch)
    fresh = tmp_path / "stage/dbscan"
    selection = json.loads((fresh / "adaptive_anchor_selection.json").read_text())[
        "selection_identity"
    ]
    for field in ("failure_indices_path", "anchor_indices_path", "anchor_rows_path"):
        assert Path(selection[field]).parent == fresh
    checkpoint = external._load_checkpoint(fresh / "checkpoint.json")
    assert checkpoint["phase"] == "shortcut_anchor_scan"
    assert checkpoint["next_offset"] == 0
    assert checkpoint["progress_ledgers"]["adaptive_seed_scan"]["committed_offset"] == len(
        _values()
    )
    assert checkpoint["progress_ledgers"]["adaptive_failure_scan"]["committed_offset"] == len(
        _values()
    )
    source_before = {
        path.name: (path.stat().st_mtime_ns, external._sha256_file(path))
        for path in source_root.iterdir()
        if path.is_file()
    }

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("adopted seed/failure scan was reexecuted")

    monkeypatch.setattr(external, "_adaptive_seed_block_candidates", forbidden)
    monkeypatch.setattr(external, "_adaptive_failure_block", forbidden)
    result = fit_promoted_failed_selection_component_recovery(
        vectors_path=vectors,
        work_dir=fresh,
        contract=contract,
        expected_vectors_sha256=external._sha256_file(vectors),
    )
    expected = DBSCAN(eps=contract.eps, min_samples=contract.min_samples).fit(
        _values()
    )
    assert np.array_equal(np.load(result.labels_path), expected.labels_)
    assert np.array_equal(
        np.flatnonzero(np.load(result.core_mask_path)), expected.core_sample_indices_
    )
    assert source_before == {
        path.name: (path.stat().st_mtime_ns, external._sha256_file(path))
        for path in source_root.iterdir()
        if path.is_file()
    }


def test_promoted_terminal_reopen_does_not_replay_any_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vectors, _source_root, contract, source = _promote(tmp_path, monkeypatch)
    fresh = tmp_path / "stage/dbscan"
    first = fit_promoted_failed_selection_component_recovery(
        vectors_path=vectors,
        work_dir=fresh,
        contract=contract,
        expected_vectors_sha256=external._sha256_file(vectors),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("terminal reopen ran a scan")

    monkeypatch.setattr(external, "_adaptive_seed_block_candidates", forbidden)
    monkeypatch.setattr(external, "_adaptive_failure_block", forbidden)
    monkeypatch.setattr(external, "_recovery_scan_block", forbidden)
    receipt = tmp_path / "failed_selection_adoption_receipt.json"
    promote_failed_adaptive_selection_for_component_recovery(
        vectors_path=vectors,
        work_dir=fresh,
        source=source,
        contract=contract,
        expected_vectors_sha256=external._sha256_file(vectors),
        adoption_receipt_path=receipt,
        adoption_receipt_sha256=external._sha256_file(receipt),
        source_authority_sha256="a" * 64,
        resume=True,
    )
    reopened = fit_promoted_failed_selection_component_recovery(
        vectors_path=vectors,
        work_dir=fresh,
        contract=contract,
        expected_vectors_sha256=external._sha256_file(vectors),
    )
    assert reopened.manifest_sha256 == first.manifest_sha256


def test_promotion_rejects_contract_drift_and_nonempty_fresh_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vectors, _source_root, contract, source = _failed_source(tmp_path, monkeypatch)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    parent = tmp_path / "stage"
    parent.mkdir()
    occupied = parent / "occupied"
    occupied.mkdir()
    with pytest.raises(FailedSelectionRecoveryError, match="already exists"):
        promote_failed_adaptive_selection_for_component_recovery(
            vectors_path=vectors,
            work_dir=occupied,
            source=source,
            contract=contract,
            expected_vectors_sha256=external._sha256_file(vectors),
            adoption_receipt_path=receipt,
            adoption_receipt_sha256=external._sha256_file(receipt),
            source_authority_sha256="a" * 64,
        )
    drift = external.ExternalDBSCANContract(
        **{**contract.__dict__, "shortcut_query_block_size": 3}
    )
    with pytest.raises(FailedSelectionRecoveryError, match="contract changed"):
        promote_failed_adaptive_selection_for_component_recovery(
            vectors_path=vectors,
            work_dir=parent / "drift",
            source=source,
            contract=drift,
            expected_vectors_sha256=external._sha256_file(vectors),
            adoption_receipt_path=receipt,
            adoption_receipt_sha256=external._sha256_file(receipt),
            source_authority_sha256="a" * 64,
        )


def test_preterminal_promotion_crash_resumes_without_full_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vectors, _source_root, contract, source = _failed_source(tmp_path, monkeypatch)
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"status":"RECOVERY_ONLY_READY"}\n', encoding="utf-8")
    parent = tmp_path / "stage"
    parent.mkdir()
    fresh = parent / "dbscan"
    original = recovery._ensure_exact_npy
    calls = 0

    def interrupt_after_first_array(*args: object, **kwargs: object) -> str:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            raise RuntimeError("fixture promotion crash")
        return result

    monkeypatch.setattr(recovery, "_ensure_exact_npy", interrupt_after_first_array)
    with pytest.raises(RuntimeError, match="fixture promotion crash"):
        promote_failed_adaptive_selection_for_component_recovery(
            vectors_path=vectors,
            work_dir=fresh,
            source=source,
            contract=contract,
            expected_vectors_sha256=external._sha256_file(vectors),
            adoption_receipt_path=receipt,
            adoption_receipt_sha256=external._sha256_file(receipt),
            source_authority_sha256="a" * 64,
        )
    assert (fresh / recovery.PROMOTION_CLAIM_NAME).is_file()
    assert (fresh / "adaptive_first_pass_failure_indices.npy").is_file()
    assert not (fresh / recovery.PROMOTION_MANIFEST_NAME).exists()

    monkeypatch.setattr(recovery, "_ensure_exact_npy", original)

    def forbidden_full_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("promotion refit all 91.9M rows")

    monkeypatch.setattr(external, "_fit_neighbors", forbidden_full_fit)
    result = promote_failed_adaptive_selection_for_component_recovery(
        vectors_path=vectors,
        work_dir=fresh,
        source=source,
        contract=contract,
        expected_vectors_sha256=external._sha256_file(vectors),
        adoption_receipt_path=receipt,
        adoption_receipt_sha256=external._sha256_file(receipt),
        source_authority_sha256="a" * 64,
        resume=True,
    )
    assert result.promotion_manifest_path.is_file()


def test_preterminal_promotion_resume_rejects_array_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vectors, _source_root, contract, source = _failed_source(tmp_path, monkeypatch)
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{}\n', encoding="utf-8")
    parent = tmp_path / "stage"
    parent.mkdir()
    fresh = parent / "dbscan"
    original = recovery._ensure_exact_npy

    def interrupt(*args: object, **kwargs: object) -> str:
        result = original(*args, **kwargs)
        raise RuntimeError("fixture promotion crash")

    monkeypatch.setattr(recovery, "_ensure_exact_npy", interrupt)
    with pytest.raises(RuntimeError, match="fixture promotion crash"):
        promote_failed_adaptive_selection_for_component_recovery(
            vectors_path=vectors,
            work_dir=fresh,
            source=source,
            contract=contract,
            expected_vectors_sha256=external._sha256_file(vectors),
            adoption_receipt_path=receipt,
            adoption_receipt_sha256=external._sha256_file(receipt),
            source_authority_sha256="a" * 64,
        )
    target = fresh / "adaptive_first_pass_failure_indices.npy"
    values = np.load(target, allow_pickle=False)
    values[0] = int(values[0]) + 1
    _save(target, values)
    monkeypatch.setattr(recovery, "_ensure_exact_npy", original)
    with pytest.raises(FailedSelectionRecoveryError, match="SHA256 mismatch"):
        promote_failed_adaptive_selection_for_component_recovery(
            vectors_path=vectors,
            work_dir=fresh,
            source=source,
            contract=contract,
            expected_vectors_sha256=external._sha256_file(vectors),
            adoption_receipt_path=receipt,
            adoption_receipt_sha256=external._sha256_file(receipt),
            source_authority_sha256="a" * 64,
            resume=True,
        )


def test_promotion_has_one_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vectors, _source_root, contract, source = _failed_source(tmp_path, monkeypatch)
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{}\n', encoding="utf-8")
    parent = tmp_path / "stage"
    parent.mkdir()
    fresh = parent / "dbscan"
    with recovery._promotion_writer(fresh):
        with pytest.raises(
            FailedSelectionRecoveryError,
            match="another failed-selection promotion writer",
        ):
            promote_failed_adaptive_selection_for_component_recovery(
                vectors_path=vectors,
                work_dir=fresh,
                source=source,
                contract=contract,
                expected_vectors_sha256=external._sha256_file(vectors),
                adoption_receipt_path=receipt,
                adoption_receipt_sha256=external._sha256_file(receipt),
                source_authority_sha256="a" * 64,
            )
