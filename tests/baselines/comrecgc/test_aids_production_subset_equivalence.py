from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from scripts.baselines.comrecgc import audit_aids_production_subsets as audit_cli  # noqa: E402
from src.baselines.comrecgc.close_pair_view import (  # noqa: E402
    NORMALIZED_DISTANCE_CONTRACT,
    SCALE_CONTRACT,
    ThetaClosePairContract,
    materialize_theta_close_pair_view,
)
from src.baselines.comrecgc.external_memory_dbscan import (  # noqa: E402
    ExternalMemoryDBSCANError,
    _rss_bytes,
)
from src.baselines.comrecgc.production_subset_audit import (  # noqa: E402
    ProductionSubsetAuditContract,
    _partition_summary,
    _production_downstream_summary,
    run_production_subset_equivalence_audit,
    sha256_file,
)
from src.baselines.comrecgc import production_subset_audit as subset_audit  # noqa: E402


def _save(path: Path, values: np.ndarray) -> Path:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    return path


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, str, Path, str]:
    parent_count = 6
    candidate_count = 4
    count = parent_count * candidate_count
    vectors = np.zeros((count, 2), dtype=np.float32)
    vectors[:8, 0] = np.asarray(
        [0.0, 0.0, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04], dtype=np.float32
    )
    vectors[8:16, 0] = np.asarray(
        [0.07, 0.07, 0.08, 0.09, 0.09, 0.1, 0.11, 0.11], dtype=np.float32
    )
    vectors[16:, 0] = np.asarray(
        [0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28], dtype=np.float32
    )
    pairs = np.column_stack(
        (
            np.arange(count, dtype=np.int64) % parent_count,
            np.arange(count, dtype=np.int64) // parent_count,
        )
    )
    distances = np.asarray(
        [0.005 + index * 0.003 for index in range(count)], dtype=np.float32
    )
    distances[-2:] = np.asarray([0.11, 0.12], dtype=np.float32)
    vectors_path = _save(tmp_path / "physical_vectors.npy", vectors)
    pairs_path = _save(tmp_path / "physical_pairs.npy", pairs)
    distances_path = _save(tmp_path / "normalized_distances.npy", distances)
    pair_sha = sha256_file(pairs_path)
    pair_manifest = _write_json(
        tmp_path / "pair_store_manifest.json",
        {
            "pairs_path": str(pairs_path.resolve()),
            "pairs_sha256": pair_sha,
            "recourse_vectors_path": str(vectors_path.resolve()),
            "recourse_vectors_sha256": sha256_file(vectors_path),
        },
    )
    pair_authority = _write_json(
        tmp_path / "pair_semantics_contract.json",
        {
            "pair_orientation": "col0_parent_col1_candidate",
            "pair_order": "candidate_major_parent_minor",
            "filter_operator": "<=",
            "physical_pair_count": count,
            "physical_vectors_sha256": sha256_file(vectors_path),
            "normalized_distances_sha256": sha256_file(distances_path),
            "source_pair_store_manifest": str(pair_manifest.resolve()),
            "source_pair_store_manifest_sha256": sha256_file(pair_manifest),
            "source_hashes": {"pair_indices_direct_sha256": pair_sha},
        },
    )
    view = materialize_theta_close_pair_view(
        physical_vectors_path=vectors_path,
        normalized_distances_path=distances_path,
        output_dir=tmp_path / "close_view",
        contract=ThetaClosePairContract(
            theta=0.1,
            parent_count=parent_count,
            candidate_count=candidate_count,
            distance_checkpoint_sha256="a" * 64,
            embedding_checkpoint_sha256="b" * 64,
            scale_contract=SCALE_CONTRACT,
            normalized_distance_contract=NORMALIZED_DISTANCE_CONTRACT,
        ),
        expected_physical_vectors_sha256=sha256_file(vectors_path),
        expected_normalized_distances_sha256=sha256_file(distances_path),
        pair_semantics_contract_path=pair_authority,
        expected_pair_semantics_contract_sha256=sha256_file(pair_authority),
        max_compact_bytes=1024**2,
        block_size=7,
    )
    return view.manifest_path, view.manifest_sha256, pairs_path, pair_sha


def _contract() -> ProductionSubsetAuditContract:
    return ProductionSubsetAuditContract(
        eps=0.02,
        min_samples=3,
        radius=0.02,
        recourse_size=100,
        subset_size=8,
        seed=19,
        scan_block_size=5,
        query_block_size=3,
        max_rss_bytes=_rss_bytes() + 4 * 1024**3,
        working_rss_margin_bytes=1024**3,
        expected_sklearn_version=sklearn.__version__,
        expected_parent_count=6,
        expected_candidate_count=4,
        expected_physical_pair_count=24,
    )


def test_aids_production_subset_equivalence(tmp_path: Path) -> None:
    close_path, close_sha, pairs_path, pair_sha = _fixture(tmp_path)
    output = tmp_path / "audit"
    result = run_production_subset_equivalence_audit(
        close_pair_contract_path=close_path,
        expected_close_pair_contract_sha256=close_sha,
        physical_pairs_path=pairs_path,
        expected_physical_pairs_sha256=pair_sha,
        output_dir=output,
        contract=_contract(),
    )
    assert result["status"] == "PASS"
    assert result["all_subsets_pass"] is True
    assert result["full_production_dbscan_equivalence_claimed"] is False
    assert result["rss_budget"]["mode"] == (
        "measured_pre_dbscan_peak_plus_bounded_working_margin"
    )
    assert set(result["subsets"]) == {
        "first",
        "random",
        "dense",
        "sparse",
        "theta_boundary",
    }
    assert (output / "PASS").read_text() == "PASS\n"
    for name, entry in result["subsets"].items():
        audit = json.loads(Path(entry["audit_path"]).read_text())
        assert audit["status"] == "PASS"
        assert audit["subset_scope_only"] is True
        assert audit["sklearn_result"] == audit["external_result"]
        assert len(np.load(output / name / "logical_indices.npy")) == 8
        assert audit["rss_budget"] == result["rss_budget"]


def test_production_subset_cli_defaults_match_bounded_rss_policy() -> None:
    args = audit_cli.build_parser().parse_args(
        [
            "--close-pair-contract",
            "/authority/close.json",
            "--expected-close-pair-contract-sha256",
            "a" * 64,
            "--physical-pairs",
            "/authority/pairs.npy",
            "--expected-physical-pairs-sha256",
            "b" * 64,
            "--output-dir",
            "/fresh/output",
            "--expected-sklearn-version",
            sklearn.__version__,
        ]
    )
    assert args.max_rss_gb == 32.0
    assert args.working_rss_margin_gb == 8.0


def test_subset_rss_budget_uses_measured_peak_plus_bounded_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_production_peak = 24_293_740_544
    monkeypatch.setattr(
        subset_audit, "_rss_bytes", lambda: observed_production_peak
    )
    contract = replace(
        _contract(),
        max_rss_bytes=32 * 1024**3,
        working_rss_margin_bytes=8 * 1024**3,
    )
    budget = subset_audit._derive_subset_rss_budget(contract)
    assert budget == {
        "measurement": "max_of_process_VmRSS_and_VmHWM_with_ru_maxrss_fallback",
        "mode": "measured_pre_dbscan_peak_plus_bounded_working_margin",
        "measured_after_full_source_selection_scans": True,
        "baseline_rss_bytes": observed_production_peak,
        "working_rss_margin_bytes": 8 * 1024**3,
        "effective_max_rss_bytes": observed_production_peak + 8 * 1024**3,
        "authorized_max_rss_bytes": 32 * 1024**3,
    }


def test_subset_rss_budget_rejects_peak_plus_margin_above_authorized_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subset_audit, "_rss_bytes", lambda: 25 * 1024**3)
    contract = replace(
        _contract(),
        max_rss_bytes=32 * 1024**3,
        working_rss_margin_bytes=8 * 1024**3,
    )
    with pytest.raises(
        ExternalMemoryDBSCANError,
        match="SUBSET_RSS_AUTHORIZED_CEILING_EXCEEDED",
    ):
        subset_audit._derive_subset_rss_budget(contract)


@pytest.mark.parametrize("target", ["close", "pairs"])
def test_production_subset_audit_rejects_tampered_authority(
    tmp_path: Path, target: str
) -> None:
    close_path, close_sha, pairs_path, pair_sha = _fixture(tmp_path)
    if target == "close":
        payload = json.loads(close_path.read_text())
        payload["theta"] = 0.099
        close_path.write_text(json.dumps(payload) + "\n")
    else:
        pairs = np.load(pairs_path, allow_pickle=False)
        pairs[[0, 1]] = pairs[[1, 0]]
        _save(pairs_path, pairs)
    with pytest.raises(
        ExternalMemoryDBSCANError,
        match="close-pair contract SHA256 mismatch|physical pair SHA256 mismatch",
    ):
        run_production_subset_equivalence_audit(
            close_pair_contract_path=close_path,
            expected_close_pair_contract_sha256=close_sha,
            physical_pairs_path=pairs_path,
            expected_physical_pairs_sha256=pair_sha,
            output_dir=tmp_path / f"audit_{target}",
            contract=_contract(),
        )


def test_subset_summary_preserves_exact_eps_and_strict_post_filters() -> None:
    vectors = np.asarray([[-0.02], [0.02]], dtype=np.float64)
    pairs = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
    summary = _partition_summary(
        labels=np.asarray([0, 0], dtype=np.int64),
        core_mask=np.asarray([True, True]),
        vectors=vectors,
        pairs=pairs,
        radius=0.02,
        theta=0.0,
        recourse_size=100,
    )
    cluster = summary["clusters"][0]
    assert cluster["count_exactly_at_delta"] == 2
    assert cluster["within_centroid_radius_count"] == 0
    assert cluster["outside_centroid_radius_count"] == 2
    assert cluster["count_exactly_at_theta"] == 1
    assert cluster["centroid_norm_lt_theta"] is False
    assert summary["selected_common_recourse_count"] == 0
    production = _production_downstream_summary(
        labels=np.asarray([0, 0], dtype=np.int64),
        vectors=vectors,
        pairs=pairs,
        radius=0.02,
        theta=0.0,
        recourse_size=100,
        max_rss_bytes=_rss_bytes() + 1024**3,
    )
    assert production["selected_common_recourse_count"] == 0


def test_subset_greedy_uses_stable_canonical_cluster_tie_break() -> None:
    summary = _partition_summary(
        labels=np.asarray([5, 5, 9, 9], dtype=np.int64),
        core_mask=np.ones(4, dtype=np.bool_),
        vectors=np.asarray([[0.0], [0.0], [0.01], [0.01]], dtype=np.float32),
        pairs=np.asarray([[0, 0], [1, 1], [0, 2], [1, 3]], dtype=np.int64),
        radius=0.02,
        theta=0.1,
        recourse_size=2,
    )
    assert [row["cluster_id"] for row in summary["selected"]] == [0, 1]
    assert [row["cumulative_covered_count"] for row in summary["selected"]] == [2, 2]
