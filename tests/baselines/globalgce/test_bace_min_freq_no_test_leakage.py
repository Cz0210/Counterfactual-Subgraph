from __future__ import annotations

import pytest

from src.baselines.globalgce_min_freq import (
    GlobalGCEMinFreqConfigurationError,
    select_bace_min_freq,
)


@pytest.mark.parametrize(
    ("split", "test_loaded"),
    [("test", False), ("calibration", True)],
)
def test_min_freq_selection_rejects_test_metrics(split: str, test_loaded: bool) -> None:
    with pytest.raises(GlobalGCEMinFreqConfigurationError):
        select_bace_min_freq(
            [
                {
                    "min_freq": 2,
                    "selection_split": split,
                    "test_loaded": test_loaded,
                    "prefix_auc_k1_k10": 0.1,
                    "multi_threshold_prefix_auc": 0.1,
                    "cost": 0.1,
                    "coverage_redundancy": 0.0,
                    "rule_count": 20,
                }
            ]
        )
