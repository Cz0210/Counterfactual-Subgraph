"""Nested candidate-budget prefixes for controlled scaling experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


BUDGETS = (1, 2, 4, 8)


def nested_candidate_budget_prefixes(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_field: str = "parent_id",
    rank_field: str = "generation_rank",
    budgets: Sequence[int] = BUDGETS,
) -> dict[int, list[dict[str, Any]]]:
    requested = tuple(int(value) for value in budgets)
    if tuple(sorted(set(requested))) != requested or requested[-1] <= 0:
        raise ValueError("Candidate budgets must be unique, positive, and increasing.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        parent_id = str(row.get(parent_field) or "").strip()
        if not parent_id:
            raise ValueError(f"Candidate budget row lacks {parent_field!r}.")
        try:
            int(row[rank_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Candidate budget row lacks integer {rank_field!r}.") from exc
        grouped[parent_id].append(row)
    ordered: dict[str, list[dict[str, Any]]] = {}
    for parent_id, parent_rows in grouped.items():
        values = sorted(parent_rows, key=lambda row: int(row[rank_field]))
        ranks = [int(row[rank_field]) for row in values]
        if ranks != list(range(1, len(values) + 1)):
            raise ValueError(f"Generation ranks are not contiguous for parent={parent_id}.")
        if len(values) < requested[-1]:
            raise ValueError(
                f"Parent={parent_id} has {len(values)} candidates; "
                f"max budget requires {requested[-1]}."
            )
        ordered[parent_id] = values
    result: dict[int, list[dict[str, Any]]] = {}
    for budget in requested:
        result[budget] = [
            row
            for parent_id in sorted(ordered)
            for row in ordered[parent_id][:budget]
        ]
    for left, right in zip(requested, requested[1:]):
        left_ids = [str(row.get("candidate_id")) for row in result[left]]
        right_by_parent = {
            parent_id: [str(row.get("candidate_id")) for row in ordered[parent_id][:right]]
            for parent_id in sorted(ordered)
        }
        expected = [
            candidate_id
            for parent_id in sorted(ordered)
            for candidate_id in right_by_parent[parent_id][:left]
        ]
        if left_ids != expected:
            raise AssertionError("Candidate budget prefixes are not nested.")
    return result


__all__ = ["BUDGETS", "nested_candidate_budget_prefixes"]
