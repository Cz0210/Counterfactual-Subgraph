from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.eval.comparison_protocol_audit import (
    CLAIMS,
    ComparisonAuditError,
    classify_claim,
    load_frozen_contract,
    load_parent_observations,
    paired_parent_bootstrap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_contract_forbids_outcome_shaping() -> None:
    contract = load_frozen_contract(
        PROJECT_ROOT / "configs/autodl/final_paper_evaluation_v1.json"
    )
    assert contract["figure3"] == {"k_values": list(range(1, 21)), "theta": 0.05}
    assert contract["figure4"] == {
        "k": 10,
        "threshold_count": 601,
        "threshold_start": 0.0,
        "threshold_stop": 0.0535,
    }
    assert contract["table2"] == {
        "k": 10,
        "theta": 0.05,
        "undefined_conditional_cost": "N/A",
    }
    assert contract["forbid_test_tuning"] is True
    assert contract["forbid_method_specific_threshold"] is True
    assert contract["forbid_method_specific_parent_cohort"] is True


def test_parent_loader_uses_exact_k_and_preserves_na(tmp_path: Path) -> None:
    path = tmp_path / "parents.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "parent_id",
                "k",
                "best_distance",
                "strict_recourse_available",
                "theta_star_covered",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "parent_id": "p0",
                    "k": 10,
                    "best_distance": 0.03,
                    "strict_recourse_available": True,
                    "theta_star_covered": True,
                },
                {
                    "parent_id": "p1",
                    "k": 10,
                    "best_distance": "N/A",
                    "strict_recourse_available": False,
                    "theta_star_covered": False,
                },
                {
                    "parent_id": "ignored",
                    "k": 9,
                    "best_distance": 0.01,
                    "strict_recourse_available": True,
                    "theta_star_covered": True,
                },
            ]
        )
    assert load_parent_observations(path, k=10, theta=0.05) == {
        "p0": (True, 0.03),
        "p1": (False, None),
    }


def test_bootstrap_resamples_paired_test_parents_and_rejects_cohort_change() -> None:
    ours = {
        "p0": (True, 0.01),
        "p1": (True, 0.02),
        "p2": (True, 0.03),
        "p3": (False, None),
    }
    baseline = {
        "p0": (True, 0.03),
        "p1": (False, None),
        "p2": (False, None),
        "p3": (False, None),
    }
    result = paired_parent_bootstrap(ours, baseline, samples=1000, seed=0)
    assert result["bootstrap_unit"] == "test_parent"
    assert result["parent_count"] == 4
    assert result["coverage_difference"] == 0.5
    assert result["cost_difference"] == pytest.approx(0.01)
    assert result["samples"] == 1000
    with pytest.raises(ComparisonAuditError, match="parent population"):
        paired_parent_bootstrap(
            ours,
            {key: value for key, value in baseline.items() if key != "p3"},
            samples=10,
            seed=0,
        )


def test_claim_marker_is_derived_from_results() -> None:
    universal = [
        {
            "dataset": dataset,
            "baseline": baseline,
            "coverage_difference": 0.1,
            "coverage_ci_low": 0.01,
            "cost_difference": 0.02,
            "cost_ci_low": 0.001,
        }
        for dataset in ("AIDS", "Mutagenicity", "BACE", "TasteMolNet")
        for baseline in ("GCFExplainer", "GlobalGCE", "ComRecGC")
    ]
    assert classify_claim(universal, []) == CLAIMS[0]

    unresolved = [
        {
            **row,
            "coverage_difference": 0.0,
            "coverage_ci_low": -0.1,
            "cost_difference": None,
            "cost_ci_low": None,
        }
        for row in universal
    ]
    assert classify_claim(unresolved, []) == CLAIMS[3]
