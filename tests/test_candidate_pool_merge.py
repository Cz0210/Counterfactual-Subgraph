from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.candidate_pool_merge import MergeConfig, merge_candidate_pools


DEDUP_KEY = ("final_fragment", "parent_smiles")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(str(row.get(field) or "").strip() for field in DEDUP_KEY)  # type: ignore[return-value]


def _merge(
    tmp_path: Path,
    pool_a_rows: list[dict[str, Any]],
    pool_b_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_a = tmp_path / "pool_a.jsonl"
    pool_b = tmp_path / "pool_b.jsonl"
    output = tmp_path / "merged.jsonl"
    summary_path = tmp_path / "merge_summary.json"
    _write_jsonl(pool_a, pool_a_rows)
    _write_jsonl(pool_b, pool_b_rows)
    summary = merge_candidate_pools(
        [pool_a, pool_b],
        out_jsonl=output,
        out_summary_json=summary_path,
        config=MergeConfig(
            dedup_key=DEDUP_KEY,
            keep_best_by="reward_total",
        ),
    )
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    return _read_jsonl(output), summary


def test_ordinary_dedup_keeps_largest_reward_total(tmp_path: Path) -> None:
    rows, summary = _merge(
        tmp_path,
        [
            {
                "final_fragment": "CC",
                "parent_smiles": "CCO",
                "reward_total": 1.0,
                "source": "base",
            },
            {
                "final_fragment": "CN",
                "parent_smiles": "CCN",
                "reward_total": 2.0,
            },
        ],
        [
            {
                "final_fragment": "CC",
                "parent_smiles": "CCO",
                "reward_total": 4.0,
                "source": "high_temp",
            }
        ],
    )

    assert len(rows) == 2
    assert next(row for row in rows if _key(row) == ("CC", "CCO"))[
        "source"
    ] == "high_temp"
    assert summary["dedup_removed_count"] == 1


def test_negative_reward_and_selector_ineligible_rows_are_retained(
    tmp_path: Path,
) -> None:
    invalid_rows = [
        {
            "final_fragment": f"fragment_{index}",
            "parent_smiles": f"parent_{index}",
            "reward_total": -6.45,
            "parse_ok": True,
            "final_substructure": False,
            "oracle_ok": False,
            "cf_flip": False,
        }
        for index in range(3)
    ]

    rows, summary = _merge(tmp_path, invalid_rows, [])

    assert {_key(row) for row in rows} == {_key(row) for row in invalid_rows}
    assert len(rows) == 3
    assert summary["eligible_unique_key_count"] == 3
    assert summary["missing_eligible_key_count"] == 0


def test_equal_reward_stably_keeps_first_input_row(tmp_path: Path) -> None:
    rows, _summary = _merge(
        tmp_path,
        [
            {
                "final_fragment": "CC",
                "parent_smiles": "CCO",
                "reward_total": 2.0,
                "cf_drop": 0.1,
                "source": "first",
            }
        ],
        [
            {
                "final_fragment": "CC",
                "parent_smiles": "CCO",
                "reward_total": 2.0,
                "cf_drop": 0.9,
                "source": "second",
            }
        ],
    )

    assert rows == [
        {
            "final_fragment": "CC",
            "parent_smiles": "CCO",
            "reward_total": 2.0,
            "cf_drop": 0.1,
            "source": "first",
        }
    ]


def test_missing_and_nonfinite_reward_keep_first_row_for_key(
    tmp_path: Path,
) -> None:
    rows, summary = _merge(
        tmp_path,
        [
            {
                "final_fragment": "CC",
                "parent_smiles": "CCO",
                "source": "missing-first",
            }
        ],
        [
            {
                "final_fragment": "CC",
                "parent_smiles": "CCO",
                "reward_total": float("nan"),
                "source": "nan-second",
            }
        ],
    )

    assert len(rows) == 1
    assert rows[0]["source"] == "missing-first"
    assert summary["non_numeric_keep_best_rows"] == 2


def test_finite_reward_replaces_non_numeric_first_row(tmp_path: Path) -> None:
    rows, summary = _merge(
        tmp_path,
        [
            {
                "final_fragment": "CC",
                "parent_smiles": "CCO",
                "reward_total": "not-a-number",
                "source": "bad",
            }
        ],
        [
            {
                "final_fragment": "CC",
                "parent_smiles": "CCO",
                "reward_total": -9.0,
                "source": "finite-negative",
            }
        ],
    )

    assert rows[0]["source"] == "finite-negative"
    assert summary["non_numeric_keep_best_rows"] == 1


def test_empty_key_components_are_skipped_and_counted(tmp_path: Path) -> None:
    rows, summary = _merge(
        tmp_path,
        [
            {
                "final_fragment": "   ",
                "parent_smiles": "CCO",
                "reward_total": 1.0,
            },
            {
                "final_fragment": "CN",
                "parent_smiles": "",
                "reward_total": 1.0,
            },
        ],
        [
            {
                "final_fragment": " CO ",
                "parent_smiles": " CCO ",
                "reward_total": -1.0,
            }
        ],
    )

    assert len(rows) == 1
    assert _key(rows[0]) == ("CO", "CCO")
    assert summary["skipped_empty_key_rows"] == 2
    assert summary["raw_unique_key_count"] == 3
    assert summary["eligible_unique_key_count"] == 1


def test_output_key_set_exactly_matches_eligible_input_key_set(
    tmp_path: Path,
) -> None:
    pool_a = [
        {"final_fragment": "A", "parent_smiles": "P1", "reward_total": -3.0},
        {"final_fragment": "", "parent_smiles": "P2", "reward_total": 9.0},
        {"final_fragment": "B", "parent_smiles": "P2", "reward_total": 1.0},
    ]
    pool_b = [
        {"final_fragment": "A", "parent_smiles": "P1", "reward_total": -2.0},
        {"final_fragment": "C", "parent_smiles": "P1", "reward_total": 0.0},
        {"final_fragment": "D", "parent_smiles": None, "reward_total": 1.0},
    ]
    rows, summary = _merge(tmp_path, pool_a, pool_b)
    eligible_input_keys = {
        key
        for row in [*pool_a, *pool_b]
        if all(key := _key(row))
    }

    assert {_key(row) for row in rows} == eligible_input_keys
    assert summary["missing_eligible_key_count"] == 0
    assert summary["unexpected_key_count"] == 0


def test_summary_counts_are_complete_and_backward_compatible(
    tmp_path: Path,
) -> None:
    rows, summary = _merge(
        tmp_path,
        [
            {"final_fragment": "A", "parent_smiles": "P1", "reward_total": 1.0},
            {"final_fragment": "B", "parent_smiles": "P2", "reward_total": -1.0},
            {"final_fragment": "", "parent_smiles": "P3", "reward_total": 0.0},
        ],
        [
            {"final_fragment": "A", "parent_smiles": "P1", "reward_total": 2.0},
            {"final_fragment": "C", "parent_smiles": "P1"},
            {"final_fragment": "D", "parent_smiles": "", "reward_total": 0.0},
        ],
    )

    assert len(rows) == 3
    assert [item["count"] for item in summary["input_counts"]] == [3, 3]
    assert summary["input_rows"] == 6
    assert summary["merged_count_before_dedup"] == 6
    assert summary["raw_unique_key_count"] == 5
    assert summary["eligible_unique_key_count"] == 3
    assert summary["eligible_input_rows"] == 4
    assert summary["skipped_empty_key_rows"] == 2
    assert summary["non_numeric_keep_best_rows"] == 1
    assert summary["merged_count_after_dedup"] == 3
    assert summary["dedup_removed_count"] == 1
    assert summary["missing_eligible_key_count"] == 0
    assert summary["unexpected_key_count"] == 0
    assert summary["unique_parent_count"] == 2
    assert summary["unique_final_fragment_count"] == 3
    assert summary["dedup_key"] == ["final_fragment", "parent_smiles"]
    assert summary["keep_best_by"] == "reward_total"
    assert summary["out_jsonl"].endswith("merged.jsonl")
