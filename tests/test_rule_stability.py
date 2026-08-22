from __future__ import annotations

import json

import pytest

from src.eval.rule_stability import compare_frozen_rule_selections


def test_rule_stability_reports_all_registered_axes(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({"ordered_rules": [
        {"rule_id": "a", "fragment": "CC", "covered_parent_ids": ["p1", "p2"], "destination_distribution": {"0": 3, "2": 1}},
        {"rule_id": "b", "fragment": "CO", "covered_parent_ids": ["p3"]},
    ]}))
    right.write_text(json.dumps({"ordered_rules": [
        {"rule_id": "a", "fragment": "CC", "covered_parent_ids": ["p2"], "destination_distribution": {"0": 1, "2": 1}},
        {"rule_id": "c", "fragment": "CN", "covered_parent_ids": ["p4"]},
    ]}))
    result = compare_frozen_rule_selections(left, right)
    assert result["exact_rule_jaccard"] == pytest.approx(1 / 3)
    assert result["coverage_set_overlap"]["jaccard"] == pytest.approx(1 / 4)
    assert result["destination_distribution_similarity"]["status"] == "PASS"
    assert 0.0 <= result["destination_distribution_similarity"]["similarity"] <= 1.0
    assert "bidirectional_mean_max" in result["morgan_mean_max_similarity"]


def test_rule_stability_marks_optional_evidence_unavailable(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({"ordered_rules": [{"rule_id": "a", "fragment": "CC"}]}))
    right.write_text(json.dumps({"ordered_rules": [{"rule_id": "a", "fragment": "CC"}]}))
    result = compare_frozen_rule_selections(left, right)
    assert result["exact_rule_jaccard"] == 1.0
    assert result["coverage_set_overlap"]["status"] == "NOT_AVAILABLE"
    assert result["destination_distribution_similarity"]["status"] == "NOT_AVAILABLE"
