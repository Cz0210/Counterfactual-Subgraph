from __future__ import annotations

import json

import pytest

from scripts.autodl.verify_frozen_threshold_contract import verify
from src.eval.four_by_four_registry import build_threshold_contracts


def _source(tmp_path, dataset="AIDS"):
    expectation = json.loads(
        open("configs/autodl/mutagenicity_matched_protocol_v1.json", encoding="utf-8").read()
    )["datasets"]["Mutagenicity"]
    source_config = build_threshold_contracts(
        {"datasets": {dataset: expectation}}
    )[dataset]
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(source_config))
    return path


def test_threshold_adoption_is_explicit_selector_freeze(tmp_path):
    source = _source(tmp_path)
    audit = verify(dataset="aids", source=source, output=tmp_path / "out")
    frozen = json.loads((tmp_path / "out/frozen_threshold_contract.json").read_text())
    assert audit["status"] == "PASS"
    assert frozen["shared_across_methods"] is True
    assert frozen["distance_line"] == "MolCLR-Node-Wasserstein"
    assert frozen["threshold_fitted_on_test"] is False
    assert len(frozen["thresholds"]) == 601


def test_threshold_adoption_rejects_test_selected_contract(tmp_path):
    source = _source(tmp_path)
    payload = json.loads(source.read_text())
    payload["test_used_for_selection"] = True
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="test_used_for_selection"):
        verify(dataset="aids", source=source, output=tmp_path / "out")
