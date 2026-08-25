from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.cluster import DBSCAN
import sklearn

from src.baselines.comrecgc.close_pair_view import (
    ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
    FILTER_OPERATOR,
    NORMALIZED_DISTANCE_CONTRACT,
    PAIR_ORDER,
    PAIR_ORIENTATION,
    SCALE_CONTRACT,
    CartesianThetaClosePairs,
    ThetaClosePairContract,
    materialize_theta_close_pair_view,
    validate_theta_close_pair_view,
)
from src.baselines.comrecgc.contracts import RecourseParameters
from src.baselines.comrecgc.external_memory_dbscan import (
    ExternalDBSCANContract,
    ExternalMemoryDBSCANError,
    fit_external_memory_dbscan,
)
from src.baselines.comrecgc.recourse import run_common_recourse


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(*, parents: int, candidates: int, theta: float = 0.1) -> ThetaClosePairContract:
    return ThetaClosePairContract(
        theta=theta,
        parent_count=parents,
        candidate_count=candidates,
        distance_checkpoint_sha256="a" * 64,
        embedding_checkpoint_sha256="b" * 64,
        scale_contract=SCALE_CONTRACT,
        normalized_distance_contract=NORMALIZED_DISTANCE_CONTRACT,
    )


def test_cartesian_pair_indexing_allocates_only_the_requested_rows() -> None:
    pairs = CartesianThetaClosePairs(
        physical_rows=None,
        logical_count=91_916_686,
        parent_count=1_283,
    )
    assert pairs[np.asarray([0, 1_283], dtype=np.int64)].tolist() == [
        [0, 0],
        [0, 1],
    ]
    assert pairs[1_282:1_285].tolist() == [
        [1_282, 0],
        [0, 1],
        [1, 1],
    ]
    assert pairs[-1].tolist() == [1_282, 71_641]


def test_comrecgc_official_close_pair_filter_and_pair_axis_contract(
    tmp_path: Path,
) -> None:
    parents, candidates = 3, 4
    vectors = np.arange(parents * candidates * 2, dtype=np.float32).reshape(-1, 2)
    distances = np.asarray(
        [0.1, 0.10001, 0.0, 0.2, 0.09, 0.11, 0.02, 0.3, 0.1, 0.4, 0.05, 0.6],
        dtype=np.float32,
    )
    vectors_path = tmp_path / "physical.npy"
    distances_path = tmp_path / "distances.npy"
    np.save(vectors_path, vectors, allow_pickle=False)
    np.save(distances_path, distances, allow_pickle=False)

    result = materialize_theta_close_pair_view(
        physical_vectors_path=vectors_path,
        normalized_distances_path=distances_path,
        output_dir=tmp_path / "view",
        contract=_contract(parents=parents, candidates=candidates),
        max_compact_bytes=1024**2,
        block_size=4,
    )
    selected = np.flatnonzero(
        distances <= np.asarray(0.1, dtype=distances.dtype)
    ).astype(np.int64)
    pairs = np.load(result.pairs_path, allow_pickle=False)
    close_vectors = np.load(result.vectors_path, allow_pickle=False)
    bitmap = np.load(result.bitmap_path, allow_pickle=False)
    assert result.physical_store_rows == parents * candidates
    assert result.logical_close_rows == len(selected)
    assert np.array_equal(bitmap, distances <= np.float32(0.1))
    assert np.array_equal(
        pairs,
        np.column_stack((selected % parents, selected // parents)),
    )
    assert np.array_equal(close_vectors, vectors[selected])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["filter_operator"] == "<="
    assert manifest["pair_axis"] == "col0=parent;col1=candidate"
    assert manifest["dbscan_input_count"] == len(selected)
    assert manifest["dbscan_input"] == "theta_close_recourse_only"
    assert manifest["count_distance_eq_theta"] == 2
    assert manifest["recourse_vectors_recomputed"] is False


def test_close_pair_view_replay_rejects_tampered_bitmap(tmp_path: Path) -> None:
    vectors_path = tmp_path / "physical.npy"
    distances_path = tmp_path / "distances.npy"
    np.save(vectors_path, np.arange(12, dtype=np.float32).reshape(6, 2))
    np.save(
        distances_path,
        np.asarray([0.0, 0.2, 0.1, 0.3, 0.05, 0.4], dtype=np.float32),
    )
    result = materialize_theta_close_pair_view(
        physical_vectors_path=vectors_path,
        normalized_distances_path=distances_path,
        output_dir=tmp_path / "view",
        contract=_contract(parents=2, candidates=3),
        block_size=2,
    )
    bitmap = np.load(result.bitmap_path, mmap_mode="r+")
    bitmap[1] = True
    bitmap.flush()
    with pytest.raises(ExternalMemoryDBSCANError, match="artifact closure"):
        validate_theta_close_pair_view(result.manifest_path)


def test_partial_close_defaults_to_zero_copy_index_and_blocks_path_dbscan(
    tmp_path: Path,
) -> None:
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    distances = np.asarray([0.01, 0.2, 0.03, 0.4, 0.1, 0.5], dtype=np.float32)
    vectors_path = tmp_path / "physical.npy"
    distances_path = tmp_path / "distances.npy"
    np.save(vectors_path, vectors)
    np.save(distances_path, distances)
    result = materialize_theta_close_pair_view(
        physical_vectors_path=vectors_path,
        normalized_distances_path=distances_path,
        output_dir=tmp_path / "indexed-view",
        contract=_contract(parents=2, candidates=3),
        block_size=2,
    )
    selected = np.flatnonzero(distances <= np.float32(0.1))
    assert result.view_storage == "bitmap_index_zero_copy"
    assert result.eligible_for_dbscan is False
    assert result.blocking_reason == "BLOCKED_STORAGE_INDEXED_DBSCAN_ENGINE_REQUIRED"
    assert result.vectors_path == vectors_path.resolve()
    assert result.pairs_path is None
    assert not (tmp_path / "indexed-view/recourse_vectors.npy").exists()
    assert not (tmp_path / "indexed-view/pair_indices.npy").exists()
    assert np.array_equal(result.open_vectors()[:], vectors[selected])
    assert np.array_equal(
        result.open_pairs()[:],
        np.column_stack((selected % 2, selected // 2)),
    )
    with pytest.raises(
        ExternalMemoryDBSCANError,
        match="BLOCKED_STORAGE_INDEXED_DBSCAN_ENGINE_REQUIRED",
    ):
        validate_theta_close_pair_view(
            result.manifest_path, require_dbscan_eligible=True
        )


def test_all_pairs_close_requires_review_then_allows_certified_zero_copy(
    tmp_path: Path,
) -> None:
    parents, candidates = 2, 3
    vectors_path = tmp_path / "physical.npy"
    distances_path = tmp_path / "distances.npy"
    np.save(vectors_path, np.arange(18, dtype=np.float32).reshape(6, 3))
    np.save(distances_path, np.full(6, 0.05, dtype=np.float32))
    output = tmp_path / "view"
    contract = _contract(parents=parents, candidates=candidates)
    with pytest.raises(
        ExternalMemoryDBSCANError, match="ALL_PAIRS_CLOSE_REVIEW_REQUIRED"
    ):
        materialize_theta_close_pair_view(
            physical_vectors_path=vectors_path,
            normalized_distances_path=distances_path,
            output_dir=output,
            contract=contract,
            block_size=2,
        )
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    identity = checkpoint["scientific_identity"]
    certificate = {
        "schema_version": ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
        "status": "PASS",
        "all_pairs_close_proven": True,
        "full_distance_scan_complete": True,
        "official_sample_comparison_pass": True,
        "normalization_audit_pass": True,
        "physical_store_rows": parents * candidates,
        "count_distance_le_theta": parents * candidates,
        "count_distance_gt_theta": 0,
        "count_distance_eq_theta": 0,
        "theta": contract.theta,
        "filter_operator": FILTER_OPERATOR,
        "pair_orientation": PAIR_ORIENTATION,
        "pair_order": PAIR_ORDER,
        "physical_vectors_sha256": identity["physical_vectors_sha256"],
        "normalized_distances_sha256": identity["normalized_distances_sha256"],
        "distance_checkpoint_sha256": contract.distance_checkpoint_sha256,
        "embedding_checkpoint_sha256": contract.embedding_checkpoint_sha256,
        "scale_contract": contract.scale_contract,
        "normalized_distance_contract": contract.normalized_distance_contract,
        "approximation_used": False,
    }
    certificate_path = tmp_path / "all_pairs_close_certificate.json"
    certificate_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = materialize_theta_close_pair_view(
        physical_vectors_path=vectors_path,
        normalized_distances_path=distances_path,
        output_dir=output,
        contract=contract,
        all_pairs_close_certificate_path=certificate_path,
        block_size=2,
        resume=True,
    )
    assert result.all_pairs_close is True
    assert result.view_storage == "zero_copy_full_cartesian"
    assert result.vectors_path == vectors_path.resolve()
    assert result.pairs_path is None
    assert result.physical_row_indices_path is None
    assert not (output / "recourse_vectors.npy").exists()
    assert not (output / "pair_indices.npy").exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["all_pairs_close_certificate_sha256"] == _sha(certificate_path)


def test_exact_dbscan_receives_only_logical_close_vectors(tmp_path: Path) -> None:
    # Physical rows 7 and 8 would bridge the two close components if the full
    # Cartesian store were clustered.  They are deliberately > theta.
    vectors = np.asarray(
        [
            [0.000],
            [0.005],
            [0.010],
            [0.100],
            [0.105],
            [0.110],
            [0.300],
            [0.050],
            [0.060],
        ],
        dtype=np.float32,
    )
    distances = np.asarray(
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2],
        dtype=np.float32,
    )
    vectors_path = tmp_path / "physical.npy"
    distances_path = tmp_path / "distances.npy"
    np.save(vectors_path, vectors)
    np.save(distances_path, distances)
    close = materialize_theta_close_pair_view(
        physical_vectors_path=vectors_path,
        normalized_distances_path=distances_path,
        output_dir=tmp_path / "view",
        contract=_contract(parents=3, candidates=3),
        max_compact_bytes=1024**2,
        block_size=3,
    )
    exact = fit_external_memory_dbscan(
        vectors_path=close.vectors_path,
        work_dir=tmp_path / "dbscan",
        contract=ExternalDBSCANContract(
            eps=0.02,
            min_samples=3,
            query_block_size=2,
            checkpoint_interval_blocks=1,
            max_rss_bytes=1024**3,
            expected_sklearn_version=sklearn.__version__,
        ),
    )
    expected = DBSCAN(eps=0.02, min_samples=3, metric="euclidean").fit_predict(
        vectors[distances <= np.float32(0.1)]
    )
    actual = np.load(exact.labels_path, allow_pickle=False)
    assert exact.num_samples == close.logical_close_rows == 7
    assert np.array_equal(actual, expected)
    assert exact.cluster_count == 2
    assert exact.noise_count == 1


def test_cartesian_snapshot_without_logical_view_fails_before_runtime_io(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="UNPROVEN_CARTESIAN_DBSCAN_INPUT"):
        run_common_recourse(
            upstream_root=tmp_path / "missing-upstream",
            dataset="aids",
            dataset_dir=tmp_path / "missing-dataset",
            source_csv=tmp_path / "missing.csv",
            generation_dir=tmp_path / "missing-generation",
            distance_checkpoint=tmp_path / "missing-checkpoint",
            output_dir=tmp_path / "output",
            mode="full",
            parent_limit=1283,
            parameters=RecourseParameters.for_mode("full"),
            device="cpu",
            engine="external_memory_exact_v1",
            external_pair_store_source_checkpoint=tmp_path / "physical-checkpoint.json",
            external_pair_store_source_owner_root=tmp_path / "old-root",
            external_vector_cache_root=tmp_path / "cache",
            external_vector_cache_lock=tmp_path / "cache.lock",
            external_vector_cache_route_lock=tmp_path / "route.lock",
        )


def test_close_pair_view_cli_writes_pass_last_only_after_eligible_closure(
    tmp_path: Path,
) -> None:
    parents, candidates = 2, 2
    vectors_path = tmp_path / "physical.npy"
    distances_path = tmp_path / "normalized.npy"
    np.save(vectors_path, np.zeros((4, 2), dtype=np.float32))
    np.save(distances_path, np.full(4, 0.01, dtype=np.float32))
    source_contract = tmp_path / "pair-semantics.json"
    source_contract.write_text(
        json.dumps(
            {
                "theta": 0.1,
                "parent_count": parents,
                "candidate_count": candidates,
                "distance_checkpoint_sha256": "a" * 64,
                "embedding_checkpoint_sha256": "b" * 64,
                "scale_contract": SCALE_CONTRACT,
                "normalized_distance_contract": NORMALIZED_DISTANCE_CONTRACT,
                "pair_orientation": PAIR_ORIENTATION,
                "filter_operator": FILTER_OPERATOR,
                "physical_vectors_sha256": _sha(vectors_path),
                "normalized_distances_sha256": _sha(distances_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts/baselines/comrecgc/build_close_pair_view.py"
    output = tmp_path / "view"
    base = [
        sys.executable,
        str(script),
        "--pair-semantics-contract",
        str(source_contract),
        "--physical-vectors",
        str(vectors_path),
        "--normalized-distances",
        str(distances_path),
        "--output-dir",
        str(output),
        "--block-size",
        "2",
    ]
    first = subprocess.run(base, cwd=root, text=True, capture_output=True)
    assert first.returncode != 0
    assert not (output / "PASS").exists()
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    identity = checkpoint["scientific_identity"]
    certificate = {
        "schema_version": ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
        "status": "PASS",
        "all_pairs_close_proven": True,
        "full_distance_scan_complete": True,
        "official_sample_comparison_pass": True,
        "normalization_audit_pass": True,
        "physical_store_rows": 4,
        "count_distance_le_theta": 4,
        "count_distance_gt_theta": 0,
        "count_distance_eq_theta": 0,
        "theta": 0.1,
        "filter_operator": FILTER_OPERATOR,
        "pair_orientation": PAIR_ORIENTATION,
        "pair_order": PAIR_ORDER,
        "physical_vectors_sha256": identity["physical_vectors_sha256"],
        "normalized_distances_sha256": identity["normalized_distances_sha256"],
        "distance_checkpoint_sha256": "a" * 64,
        "embedding_checkpoint_sha256": "b" * 64,
        "scale_contract": SCALE_CONTRACT,
        "normalized_distance_contract": NORMALIZED_DISTANCE_CONTRACT,
        "approximation_used": False,
    }
    certificate_path = tmp_path / "all_pairs_close_certificate.json"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    completed = subprocess.run(
        base
        + [
            "--all-pairs-close-certificate",
            str(certificate_path),
            "--resume",
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == (
        "[COMRECGC_CLOSE_PAIR_VIEW_PASS]"
    )
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"
    manifest = json.loads((output / "close_pair_contract.json").read_text())
    identity = manifest["scientific_identity"]
    assert identity["pair_semantics_contract_path"] == str(
        source_contract.resolve()
    )
    assert identity["pair_semantics_contract_sha256"] == _sha(source_contract)
