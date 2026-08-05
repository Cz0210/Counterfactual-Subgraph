from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines.comrecgc.audit import validate_final_manifest, validate_monotonic
from src.baselines.comrecgc.contracts import (
    ADAPTATION_MODE,
    CF_MODE,
    DISTANCE_LINE,
    GenerationParameters,
    RecourseParameters,
    ContractError,
    ordered_ids_sha256,
    write_json,
)
from src.baselines.comrecgc.project_dataset import project_label_to_internal
from src.baselines.comrecgc.runtime import validate_counterfactual_payload


def test_generation_profiles_are_frozen() -> None:
    GenerationParameters.for_mode("smoke").validate("smoke")
    GenerationParameters.for_mode("full").validate("full")
    invalid = GenerationParameters.for_mode("smoke")
    with pytest.raises(ContractError):
        invalid.validate("full")


def test_common_recourse_profiles_are_frozen() -> None:
    assert RecourseParameters.for_mode("smoke").recourse_size == 5
    assert RecourseParameters.for_mode("full").cf_size == 100_000
    RecourseParameters.for_mode("full").validate("full")


def test_order_hash_is_order_sensitive() -> None:
    assert ordered_ids_sha256(["a", "b"]) != ordered_ids_sha256(["b", "a"])


def test_atomic_json_write(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_json(path, {"value": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3}
    assert not list(tmp_path.glob("*.tmp"))


def test_monotonicity_gate() -> None:
    validate_monotonic([0.0, 0.2, 0.2, 1.0], field="coverage")
    with pytest.raises(ContractError):
        validate_monotonic([0.0, 0.3, 0.2], field="coverage")


def test_final_semantic_gate() -> None:
    validate_final_manifest(
        {
            "method": "COMRECGC",
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "adaptation_mode": ADAPTATION_MODE,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "calibration_loaded": False,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
        }
    )


def test_project_label_mapping_is_explicit() -> None:
    assert project_label_to_internal(1) == 0
    assert project_label_to_internal(0) == 1
    with pytest.raises(ContractError):
        project_label_to_internal(2)


def test_upstream_payload_contract() -> None:
    graph_map, candidates = validate_counterfactual_payload(
        {"graph_map": {"hash": [object()]}, "counterfactual_candidates": [{"graph_hash": "hash"}]}
    )
    assert list(graph_map) == ["hash"]
    assert candidates[0]["graph_hash"] == "hash"
    with pytest.raises(RuntimeError):
        validate_counterfactual_payload({"graph_map": {}, "counterfactual_candidates": []})
