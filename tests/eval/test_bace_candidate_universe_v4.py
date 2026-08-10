from __future__ import annotations

from src.eval.bace_candidate_universe import (
    CONNECTED_FEASIBLE_V4_POLICY,
    build_connected_feasible_candidate_universe,
)
from src.eval.wnode_action_matrix import ActionMatrixConfig, _build_bace_universe


def _row(fragment: str, *, flip: bool, drop: float, index: int) -> dict:
    return {
        "molecule_id": f"source-{index}",
        "label": 1,
        "parent_smiles": "CCCO",
        "final_fragment": fragment,
        "parse_ok": True,
        "valid": True,
        "connected": True,
        "final_substructure": True,
        "direct_substructure": True,
        "oracle_ok": True,
        "cf_flip": flip,
        "cf_drop": drop,
        "atom_ratio": 0.25,
        "candidate_lineage_source": "/frozen/pool.jsonl",
        "candidate_lineage_source_index": index,
    }


def _valid_delete(_parent: str, _fragment: str) -> list[dict]:
    return [
        {
            "delete_valid": True,
            "residual_smiles": "CCO",
            "sanitize_ok": True,
            "residual_connected": True,
            "residual_num_components": 1,
            "boundary_bond_count": 1,
        }
    ]


def test_v4_source_flip_and_cfdrop_are_features_not_hard_gates() -> None:
    rows = [
        _row("C", flip=False, drop=0.01, index=0),
        _row("N", flip=True, drop=0.4, index=1),
    ]
    candidates, statistics, decisions = build_connected_feasible_candidate_universe(
        rows, deletion_fn=_valid_delete
    )
    assert len(candidates) == 2
    assert all(row["entered_connected_feasible_universe"] for row in decisions)
    assert statistics["source_filter_contract"]["source_cf_flip_is_feature_not_gate"] is True
    assert statistics["source_filter_contract"]["source_cf_drop_is_feature_not_gate"] is True
    by_fragment = {row["canonical_fragment"]: row for row in candidates}
    assert by_fragment["C"]["source_cf_flip_rate"] == 0.0
    assert by_fragment["C"]["source_cf_drop_max"] == 0.01


def test_legacy_policy_is_unchanged_while_v4_expands_universe() -> None:
    rows = [
        _row("C", flip=False, drop=0.01, index=0),
        _row("N", flip=True, drop=0.4, index=1),
    ]
    legacy, legacy_stats = _build_bace_universe(rows, ActionMatrixConfig())
    expanded, expanded_stats = _build_bace_universe(
        rows,
        ActionMatrixConfig(
            candidate_universe_policy=CONNECTED_FEASIBLE_V4_POLICY,
            require_candidate_lineage=True,
        ),
        deletion_fn=_valid_delete,
    )
    assert [row["canonical_fragment"] for row in legacy] == ["N"]
    assert {row["canonical_fragment"] for row in expanded} == {"C", "N"}
    assert legacy_stats["source_filter_contract"]["require_cf_flip"] is True
    assert expanded_stats["source_filter_contract"]["source_cf_flip_is_feature_not_gate"] is True


def test_v4_fails_closed_on_disconnected_source_residual() -> None:
    rows = [_row("C", flip=True, drop=0.4, index=0)]

    def disconnected(_parent: str, _fragment: str) -> list[dict]:
        return [
            {
                "delete_valid": False,
                "invalid_reason": "disconnected_residual",
                "residual_num_components": 2,
            }
        ]

    candidates, statistics, decisions = build_connected_feasible_candidate_universe(
        rows, deletion_fn=disconnected
    )
    assert candidates == []
    assert decisions[0]["matrix_exclusion_reason"] == "excluded_source_disconnected"
    assert statistics["source_filter_counts"] == {"excluded_source_disconnected": 1}


def test_v4_requires_complete_lineage() -> None:
    row = _row("C", flip=True, drop=0.4, index=0)
    del row["candidate_lineage_source"]
    candidates, _statistics, decisions = build_connected_feasible_candidate_universe(
        [row], deletion_fn=_valid_delete
    )
    assert candidates == []
    assert decisions[0]["matrix_exclusion_reason"] == "excluded_missing_lineage"
