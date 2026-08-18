from __future__ import annotations

import pytest

from src.baselines.globalgce_bace_action_adapter import (
    GlobalGCEActionAdapterError,
    assert_nonzero_fullgraph_applicability,
)


def test_all_zero_matrix_fails_closed() -> None:
    with pytest.raises(GlobalGCEActionAdapterError, match="SCHEMA_OR_ACTION_ADAPTER_FAILURE"):
        assert_nonzero_fullgraph_applicability(
            [{"match": False, "teacher_strict_flip": False}]
        )


def test_fullgraph_pairs_are_counted_without_substructure_semantics() -> None:
    audit = assert_nonzero_fullgraph_applicability(
        [{"match": True, "delete_valid": True, "teacher_strict_flip": True}]
    )
    assert audit == {
        "pair_count": 1,
        "applicable_count": 1,
        "valid_count": 1,
        "strict_flip_count": 1,
    }
