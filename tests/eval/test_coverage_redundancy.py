from __future__ import annotations

import numpy as np

from src.eval.mutagenicity_wnode_selector import (
    ThresholdLevel,
    build_coverage_redundancy_matrix,
    morgan_tanimoto,
)


def test_low_tanimoto_can_still_have_high_coverage_redundancy() -> None:
    distances = np.asarray([[0.1, 0.1], [0.1, 0.1], [np.inf, np.inf]])
    covred = build_coverage_redundancy_matrix(
        distances, (ThresholdLevel("q", 0.2, 1.0, (0.2,), ("q",)),)
    )
    structural = morgan_tanimoto("CC", "c1ccccc1")
    assert covred[0, 1] == 1.0
    assert structural < covred[0, 1]
