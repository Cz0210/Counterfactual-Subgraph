from __future__ import annotations

import pytest

from src.baselines.tastemolnet_multiclass_adapters import (
    CF_MODE,
    TasteMulticlassContractError,
    adapt_comrecgc_transition,
    adapt_gcf_fullgraph_score,
    authorize_split_access,
    gcf_candidate_condition,
    is_taste_strict_flip,
    merge_globalgce_target_branches,
    multiclass_extension_manifest,
    taste_destination_distribution,
    validate_frozen_gine_manifest,
)


CHECKPOINT_HASH = "a" * 64
TEMPERATURE_HASH = "b" * 64
FEATURE_HASH = "c" * 64


def _gine_manifest(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset": "tastemolnet",
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "num_classes": 3,
        "source_label": 1,
        "oracle_checkpoint_hash": CHECKPOINT_HASH,
        "temperature_calibration_hash": TEMPERATURE_HASH,
        "feature_schema_hash": FEATURE_HASH,
        "test_loaded": False,
        "test_used_for_selection": False,
    }
    payload.update(updates)
    return payload


def _rule(target: int, *, rule_hash: str = "d" * 64) -> dict[str, object]:
    return {
        "rule_hash": rule_hash,
        "lhs_hash": "e" * 64,
        "rhs_hash": "f" * 64,
        "attachment_map_hash": "1" * 64,
        "action_kind": "lhs_rhs_graph_transformation_rule",
        "target_label": target,
        "source_label": 1,
        "data_split_used": "train",
        "calibration_loaded": False,
        "test_loaded": False,
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": CHECKPOINT_HASH,
    }


def _comrecgc() -> dict[str, object]:
    return {
        "transition_id": "transition-001",
        "transition_uniqueness_enforced": True,
        "lineage_unique": True,
        "upstream_identity_matches": True,
        "downstream_hash_matches": True,
        "graph_content_identity": "canonical_global_graph_hash",
        "parent_metadata_is_graph_identity": False,
        "single_edit_count": 1,
        "true_transition_count": 1,
        "graph_hash_collision_or_corruption": False,
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "num_classes": 3,
        "source_label": 1,
    }


def test_multiclass_strict_flip_accepts_both_non_sweet_destinations() -> None:
    assert is_taste_strict_flip(pred_before=1, pred_after=0)
    assert is_taste_strict_flip(pred_before=1, pred_after=2)
    assert not is_taste_strict_flip(pred_before=1, pred_after=1)
    assert not is_taste_strict_flip(pred_before=0, pred_after=2)


def test_gcf_uses_native_untargeted_fullgraph_condition() -> None:
    assert gcf_candidate_condition(0)
    assert gcf_candidate_condition(2)
    assert not gcf_candidate_condition(1)
    score = adapt_gcf_fullgraph_score(
        pred_before=1,
        pred_candidate=2,
        probabilities_before=(0.1, 0.8, 0.1),
        probabilities_candidate=(0.2, 0.1, 0.7),
        candidate_id="fullgraph-2",
    )
    assert score["cf_flip"] is True
    assert score["destination_label"] == 2
    assert score["action_kind"] == "full_counterfactual_graph"
    assert score["native_action_preserved"] is True


def test_globalgce_merges_target_zero_and_two_before_calibration() -> None:
    merged = merge_globalgce_target_branches(
        {0: [_rule(0)], 2: [_rule(2)]},
        oracle_checkpoint_hash=CHECKPOINT_HASH,
    )
    assert len(merged) == 1
    assert merged[0]["target_branches"] == [0, 2]
    assert "target_label" not in merged[0]
    assert merged[0]["branch_merge_stage"] == "before_calibration"
    assert merged[0]["action_kind"] == "lhs_rhs_graph_transformation_rule"


def test_globalgce_requires_both_targets_and_fails_on_rule_hash_collision() -> None:
    with pytest.raises(TasteMulticlassContractError, match="exactly target branches"):
        merge_globalgce_target_branches(
            {0: [_rule(0)]}, oracle_checkpoint_hash=CHECKPOINT_HASH
        )
    corrupt = _rule(2)
    corrupt["rhs_hash"] = "2" * 64
    with pytest.raises(
        TasteMulticlassContractError,
        match="GLOBALGCE_RULE_HASH_COLLISION_OR_CORRUPTION",
    ):
        merge_globalgce_target_branches(
            {0: [_rule(0)], 2: [corrupt]},
            oracle_checkpoint_hash=CHECKPOINT_HASH,
        )


def test_globalgce_rejects_calibration_or_test_in_target_branch_merge() -> None:
    leaked = _rule(2)
    leaked["calibration_loaded"] = True
    with pytest.raises(TasteMulticlassContractError, match="before calibration/test"):
        merge_globalgce_target_branches(
            {0: [_rule(0)], 2: [leaked]},
            oracle_checkpoint_hash=CHECKPOINT_HASH,
        )


def test_comrecgc_accepts_destination_two_without_binary_target_hardcode() -> None:
    result = adapt_comrecgc_transition(
        _comrecgc(),
        pred_before=1,
        pred_after=2,
        probabilities_before=(0.1, 0.8, 0.1),
        probabilities_after=(0.1, 0.2, 0.7),
    )
    assert result["cf_flip"] is True
    assert result["destination_label"] == 2
    assert result["graph_content_identity"] == "canonical_global_graph_hash"
    assert result["parent_metadata_is_graph_identity"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lineage_unique", False),
        ("single_edit_count", 2),
        ("true_transition_count", 2),
        ("graph_hash_collision_or_corruption", True),
        ("parent_metadata_is_graph_identity", True),
    ],
)
def test_comrecgc_lineage_remains_fail_closed(field: str, value: object) -> None:
    row = _comrecgc()
    row[field] = value
    with pytest.raises(TasteMulticlassContractError, match="global-lineage gate"):
        adapt_comrecgc_transition(
            row,
            pred_before=1,
            pred_after=0,
            probabilities_before=(0.1, 0.8, 0.1),
            probabilities_after=(0.8, 0.1, 0.1),
        )


def test_rf_guard_requires_one_frozen_three_class_gine() -> None:
    identity = validate_frozen_gine_manifest(_gine_manifest())
    assert identity.checkpoint_hash == CHECKPOINT_HASH
    assert identity.to_dict()["rf_oracle_used"] is False
    with pytest.raises(ValueError, match="prohibited"):
        validate_frozen_gine_manifest(
            _gine_manifest(oracle_backend="rf", rf_oracle_used=True)
        )
    with pytest.raises(TasteMulticlassContractError, match="num_classes"):
        validate_frozen_gine_manifest(_gine_manifest(num_classes=2))
    with pytest.raises(TasteMulticlassContractError, match="test_loaded"):
        validate_frozen_gine_manifest(_gine_manifest(test_loaded=True))


def test_test_is_unavailable_until_calibration_selector_is_frozen() -> None:
    with pytest.raises(TasteMulticlassContractError, match="before selector freeze"):
        authorize_split_access(
            split="test",
            oracle_checkpoint_hash=CHECKPOINT_HASH,
        )
    selector = {
        "selection_frozen": True,
        "selector_fitted_on_calibration": True,
        "calibration_loaded": True,
        "test_loaded": False,
        "source_label": 1,
        "num_classes": 3,
        "cf_mode": CF_MODE,
        "oracle_checkpoint_hash": CHECKPOINT_HASH,
        "ordered_rule_ids": ["rule-1", "rule-2"],
    }
    authorize_split_access(
        split="test",
        selector_manifest=selector,
        oracle_checkpoint_hash=CHECKPOINT_HASH,
    )
    with pytest.raises(TasteMulticlassContractError, match="complete calibration freeze"):
        authorize_split_access(
            split="test",
            selector_manifest={**selector, "test_loaded": True},
            oracle_checkpoint_hash=CHECKPOINT_HASH,
        )


def test_destination_distribution_keeps_bitter_and_tasteless_per_rule() -> None:
    rows = [
        adapt_gcf_fullgraph_score(
            pred_before=1,
            pred_candidate=destination,
            probabilities_before=(0.1, 0.8, 0.1),
            probabilities_candidate=(
                (0.8, 0.1, 0.1) if destination == 0 else (0.1, 0.2, 0.7)
            ),
            candidate_id=rule,
        )
        for destination, rule in ((0, "rule-a"), (2, "rule-a"), (2, "rule-b"))
    ]
    summary = taste_destination_distribution(rows)
    assert summary["overall"]["transitions"]["1->0"]["count"] == 1
    assert summary["overall"]["transitions"]["1->2"]["count"] == 2
    assert summary["by_rule"]["rule-a"]["total_strict_flips"] == 2


def test_method_manifests_forbid_separate_binary_explainees() -> None:
    gcf = multiclass_extension_manifest("GCFExplainer")
    globalgce = multiclass_extension_manifest("GlobalGCE")
    comrecgc = multiclass_extension_manifest("ComRecGC")
    assert gcf["candidate_condition"] == "pred_candidate != source_label"
    assert globalgce["target_branches"] == [0, 2]
    assert comrecgc["candidate_condition"] == "pred_after != source_label"
    assert all(
        contract["separate_binary_explainee_forbidden"] is True
        for contract in (gcf, globalgce, comrecgc)
    )
