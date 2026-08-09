from __future__ import annotations

import pytest

from src.eval.wnode_prefix_selector import assert_calibration_selector_inputs


def test_explicit_test_split_path_fails_closed() -> None:
    with pytest.raises(ValueError):
        assert_calibration_selector_inputs("data/processed/BACE/test.csv")


def test_gcf_result_path_fails_closed() -> None:
    with pytest.raises(ValueError):
        assert_calibration_selector_inputs("outputs/hpc/eval/gcfexplainer")
