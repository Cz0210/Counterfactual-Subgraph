from __future__ import annotations

import pytest

from src.eval.bace_frozen_gnn_contracts import fixed_parent_shard_map


def test_fixed_parent_shards_use_sorted_position_mod_four() -> None:
    mapping = fixed_parent_shard_map(["p09", "p01", "p07", "p03", "p05"])
    assert mapping == {
        "p01": 0,
        "p03": 1,
        "p05": 2,
        "p07": 3,
        "p09": 0,
    }


def test_fixed_parent_shards_do_not_change_with_input_order() -> None:
    left = fixed_parent_shard_map(["d", "b", "a", "c"])
    right = fixed_parent_shard_map(["a", "b", "c", "d"])
    assert left == right == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_fixed_parent_shards_fail_on_duplicates_or_dynamic_shard_count() -> None:
    with pytest.raises(ValueError, match="unique"):
        fixed_parent_shard_map(["same", "same"])
    with pytest.raises(ValueError, match="exactly 4"):
        fixed_parent_shard_map(["p"], num_shards=2)
