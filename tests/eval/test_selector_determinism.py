from __future__ import annotations

from src.eval.wnode_prefix_selector import build_calibration_folds
from tests.eval.wnode_v2_test_utils import matrix_data


def test_calibration_fold_assignment_is_deterministic(tmp_path) -> None:
    matrix = matrix_data(tmp_path, parents=60)
    assert build_calibration_folds(matrix) == build_calibration_folds(matrix)
    assert sorted(index for fold in build_calibration_folds(matrix) for index in fold) == list(range(60))
