from __future__ import annotations

import json
from pathlib import Path

from scripts.autodl.build_aids_mut_matched_expectations import build


def test_shared_matched_expectations_are_identical_and_test_free():
    source = Path("configs/autodl/mutagenicity_matched_protocol_v1.json")
    result = build(source)
    aids = result["datasets"]["AIDS"]
    mut = result["datasets"]["Mutagenicity"]
    assert aids["thresholds"] == mut["thresholds"]
    assert len(aids["thresholds"]) == 601
    assert aids["theta_star"] == mut["theta_star"] == 0.05
    assert aids["cost_cap"] == mut["cost_cap"] == 0.0535
    assert aids["threshold_config_hash"] == mut["threshold_config_hash"]
    assert aids["test_used_for_selection"] is False
