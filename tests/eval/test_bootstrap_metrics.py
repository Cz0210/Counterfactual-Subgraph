from __future__ import annotations

import pytest

from src.eval.bootstrap_metrics import (
    parent_level_bootstrap,
    parent_level_curve_bootstrap,
)


def _rows(count: int = 12) -> list[dict[str, object]]:
    return [
        {
            "parent_id": f"p{index}",
            "coverage": float(index % 2),
            "cost": float(index + 1) / 100,
            "cf_drop": 0.1 + index / 100,
            "flip_rate": float(index % 3 == 0),
            "valid_rate": 1.0,
            "structural_redundancy": 0.2,
            "coverage_redundancy": 0.3,
        }
        for index in range(count)
    ]


def test_parent_bootstrap_is_deterministic_and_has_ci() -> None:
    first = parent_level_bootstrap(_rows(), num_samples=100, seed=7)
    second = parent_level_bootstrap(_rows(), num_samples=100, seed=7)
    assert first == second
    assert first["resampling_unit"] == "parent_id"
    assert first["pair_row_bootstrap"] is False
    coverage = first["metrics"]["coverage"]
    assert coverage["ci_2_5"] <= coverage["median"] <= coverage["ci_97_5"]


def test_pair_rows_are_rejected_instead_of_bootstrapped_independently() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="one aggregated row per parent"):
        parent_level_bootstrap(rows, num_samples=10)


def test_curve_bootstrap_emits_parent_level_confidence_band_schema() -> None:
    rows = [
        {
            "parent_id": f"p{parent_index}",
            "method": "Ours",
            "k": k,
            "coverage": float(parent_index % (k + 1) == 0),
        }
        for k in (1, 2)
        for parent_index in range(12)
    ]
    result = parent_level_curve_bootstrap(
        rows,
        group_fields=("method", "k"),
        value_field="coverage",
        num_samples=50,
        seed=5,
    )
    assert result["schema_version"] == "parent_level_curve_confidence_band_v1"
    assert result["resampling_unit"] == "parent_id"
    assert result["pair_row_bootstrap"] is False
    assert result["num_parents"] == 12
    assert [row["k"] for row in result["rows"]] == ["1", "2"]
    for row in result["rows"]:
        assert row["coverage_ci_2_5"] <= row["coverage_mean"] <= row["coverage_ci_97_5"]


def test_curve_bootstrap_rejects_parent_universe_drift() -> None:
    rows = [
        {"parent_id": "p1", "k": 1, "coverage": 0.0},
        {"parent_id": "p2", "k": 1, "coverage": 1.0},
        {"parent_id": "p1", "k": 2, "coverage": 1.0},
    ]
    with pytest.raises(ValueError, match="different parent universes"):
        parent_level_curve_bootstrap(
            rows,
            group_fields=("k",),
            value_field="coverage",
            num_samples=10,
        )
