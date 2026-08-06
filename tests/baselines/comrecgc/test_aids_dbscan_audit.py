from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baselines.comrecgc.aids_dbscan_audit import (
    DBSCANContract,
    _resolve_project_input,
    _working_directory,
    audit_geometry,
    distribution_summary,
)


def test_project_relative_trusted_payload_can_be_resolved_before_upstream_chdir(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    upstream = project / "external/COMRECGC"
    payload = project / "outputs/trusted_dataset.pt"
    upstream.mkdir(parents=True)
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"trusted")

    resolved = _resolve_project_input(project, "outputs/trusted_dataset.pt")
    with _working_directory(upstream):
        assert resolved == payload.resolve()
        assert resolved.read_bytes() == b"trusted"


def test_upstream_working_directory_is_scoped_and_restored(tmp_path: Path) -> None:
    before = Path.cwd()
    with _working_directory(tmp_path):
        assert Path.cwd() == tmp_path.resolve()
    assert Path.cwd() == before


def test_distribution_summary_uses_float64_without_rounding() -> None:
    result = distribution_summary(np.asarray([0.0, 0.123456789012345, 0.5], dtype=np.float64))
    assert result["count"] == 3
    assert result["max"] == 0.5
    assert result["num_zero"] == 1
    assert result["mean"] == np.mean(np.asarray([0.0, 0.123456789012345, 0.5]))


def test_dbscan_geometry_records_core_and_components() -> None:
    recourses = np.asarray([[0.0, 0.0], [0.005, 0.0], [0.009, 0.0], [1.0, 1.0]])
    result = audit_geometry(
        recourse_vectors=recourses,
        pair_indices=[(0, 0), (1, 0), (2, 1), (3, 1)],
        parent_ids=["p0", "p1", "p2", "p3"],
        candidate_ids=["c0", "c1"],
        contract=DBSCANContract(0.1, 0.02, 3, 100, 100000),
    )
    assert result["dbscan_core_points"] == 3
    assert result["dbscan_noise_points"] == 1
    assert result["dbscan_non_noise_clusters"] == 1
    assert result["largest_component_size"] == 3
    assert result["postfilter_cluster_count"] == 1


def test_empty_cluster_is_valid_scientific_geometry_result() -> None:
    result = audit_geometry(
        recourse_vectors=np.asarray([[0.0], [0.5], [1.0]]),
        pair_indices=[(0, 0), (1, 0), (2, 0)],
        parent_ids=["p0", "p1", "p2"],
        candidate_ids=["c0"],
        contract=DBSCANContract(0.1, 0.02, 3, 100, 100000),
    )
    assert result["dbscan_non_noise_clusters"] == 0
    assert result["postfilter_cluster_count"] == 0
    assert result["dbscan_noise_points"] == 3
