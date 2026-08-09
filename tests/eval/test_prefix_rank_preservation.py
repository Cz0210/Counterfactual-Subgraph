from __future__ import annotations

from src.eval.mutagenicity_wnode_selector import (
    build_candidate_chemistry,
    build_coverage_redundancy_matrix,
    compute_prefix_metrics,
)
from src.eval.wnode_prefix_selector import threshold_bundle_from_manifest
from tests.eval.wnode_v2_test_utils import matrix_data, threshold_manifest


def test_nested_prefix_rank_and_coverage_are_stable(tmp_path) -> None:
    matrix = matrix_data(tmp_path)
    thresholds = threshold_bundle_from_manifest(
        threshold_manifest(tmp_path / "thresholds.json"),
        finite_distance_count=len(matrix.full_finite_distances),
    )
    chemistry = build_candidate_chemistry(matrix.candidate_rows)
    sequence = list(range(20))
    metrics, _ = compute_prefix_metrics(
        sequence,
        matrix=matrix,
        thresholds=thresholds,
        coverage_redundancy_matrix=build_coverage_redundancy_matrix(
            matrix.distances, thresholds.levels
        ),
        structural_similarity_matrix=chemistry.structural_similarity,
    )
    assert [row["k"] for row in metrics] == list(range(1, 21))
    coverage = [row["ccrcov_theta_star"] for row in metrics]
    assert coverage == sorted(coverage)
