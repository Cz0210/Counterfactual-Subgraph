from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pytest

from scripts import audit_gcf_prefix_cost_and_order as audit


def _candidates() -> tuple[audit.PrefixCandidate, ...]:
    return (
        audit.PrefixCandidate(rank=1, candidate_id="gcf_2", canonical_smiles="CC"),
        audit.PrefixCandidate(rank=2, candidate_id="gcf_10", canonical_smiles="CN"),
    )


def _pair_rows() -> list[dict[str, object]]:
    return [
        {
            "method": "GCFExplainer",
            "parent_id": "p1",
            "candidate_id": "gcf_2",
            "candidate_smiles": "CC",
            "label": 1,
            "pred_before": 1,
            "pred_after": 0,
            "cf_flip": True,
            "distance": 0.01,
        },
        {
            "method": "GCFExplainer",
            "parent_id": "p2",
            "candidate_id": "gcf_2",
            "candidate_smiles": "CC",
            "label": 1,
            "pred_before": 1,
            "pred_after": 1,
            "cf_flip": False,
            "distance": 0.005,
        },
        {
            "method": "GCFExplainer",
            "parent_id": "p1",
            "candidate_id": "gcf_10",
            "candidate_smiles": "CN",
            "label": 1,
            "pred_before": 1,
            "pred_after": 0,
            "cf_flip": True,
            "distance": 0.02,
        },
        {
            "method": "GCFExplainer",
            "parent_id": "p2",
            "candidate_id": "gcf_10",
            "candidate_smiles": "CN",
            "label": 1,
            "pred_before": 1,
            "pred_after": 0,
            "cf_flip": True,
            "distance": 0.03,
        },
    ]


def test_nested_prefix_best_and_unconditional_median_are_nonincreasing() -> None:
    pair_audit = audit.aggregate_pair_details(_pair_rows(), _candidates())
    rows = audit.compute_prefix_metrics(pair_audit, theta=0.04, max_k=2)
    assert math.isinf(rows[0]["unconditional_median_best_distance_all_parents"])
    assert rows[1]["unconditional_median_best_distance_all_parents"] == pytest.approx(0.02)
    assert rows[1]["unconditional_median_best_distance_all_parents"] <= rows[0][
        "unconditional_median_best_distance_all_parents"
    ]
    assert rows[1]["coverage"] >= rows[0]["coverage"]


def test_theta_covered_conditional_median_can_increase() -> None:
    pair_audit = audit.aggregate_pair_details(_pair_rows(), _candidates())
    rows = audit.compute_prefix_metrics(pair_audit, theta=0.04, max_k=2)
    assert rows[0]["conditional_median_best_distance_theta_covered"] == pytest.approx(0.01)
    assert rows[1]["conditional_median_best_distance_theta_covered"] == pytest.approx(0.02)


def test_raw_distance_and_strict_flip_semantics_are_separate() -> None:
    pair_audit = audit.aggregate_pair_details(_pair_rows(), _candidates())
    rows = audit.compute_prefix_metrics(pair_audit, theta=0.04, max_k=1)
    assert rows[0]["num_strict_flip_parents"] == 1
    assert rows[0]["num_missing_best_distance"] == 1
    assert rows[0]["raw_distance_num_missing"] == 0
    assert rows[0]["raw_distance_unconditional_median_all_parents"] == pytest.approx(0.0075)


def test_candidate_rank_is_numeric_and_never_candidate_id_lexical_order() -> None:
    candidate_rows = [
        {"rank": "2", "candidate_id": "gcf_10", "candidate_smiles": "CN"},
        {"rank": "1", "candidate_id": "gcf_2", "candidate_smiles": "CC"},
    ]
    selected_rows = [
        {"rank": "4", "candidate_id": "gcf_2"},
        {"rank": "9", "candidate_id": "gcf_10"},
    ]
    converted_rows = [
        {"candidate_id": "invalid", "smiles": "", "sanitize_ok": "false"},
        {"candidate_id": "gcf_2", "smiles": "CC", "sanitize_ok": "true"},
        {"candidate_id": "gcf_10", "smiles": "CN", "sanitize_ok": "true"},
    ]
    rows, summary = audit.build_candidate_order_audit(
        candidate_rows,
        selected_rows,
        converted_rows,
        _pair_rows(),
        max_k=2,
    )
    assert [row["candidate_id"] for row in rows] == ["gcf_2", "gcf_10"]
    assert summary["selected_metadata_relative_order_preserved"] is True
    assert summary["candidate_order_exact_match"] is True


def test_sanitize_filter_preserves_relative_order() -> None:
    rows = [
        {"candidate_id": "c2", "smiles": "CC", "sanitize_ok": "true"},
        {"candidate_id": "bad", "smiles": "", "sanitize_ok": "false"},
        {"candidate_id": "c10", "smiles": "CN", "sanitize_ok": "true"},
    ]
    assert [row["candidate_id"] for row in audit.valid_rows_preserving_order(rows)] == [
        "c2",
        "c10",
    ]


def test_duplicate_candidate_smiles_are_detected() -> None:
    candidates = [
        {"rank": "1", "candidate_id": "c1", "candidate_smiles": "CC"},
        {"rank": "2", "candidate_id": "c2", "candidate_smiles": "CC"},
    ]
    selected = [
        {"rank": "1", "candidate_id": "c1"},
        {"rank": "2", "candidate_id": "c2"},
    ]
    converted = [
        {"candidate_id": "c1", "smiles": "CC", "sanitize_ok": "true"},
        {"candidate_id": "c2", "smiles": "CC", "sanitize_ok": "true"},
    ]
    pair_rows = [
        {"parent_id": "p", "candidate_id": "c1", "candidate_smiles": "CC"},
        {"parent_id": "p", "candidate_id": "c2", "candidate_smiles": "CC"},
    ]
    _, summary = audit.build_candidate_order_audit(
        candidates, selected, converted, pair_rows, max_k=2
    )
    assert summary["canonical_smiles_duplicate_count"] == 1


def test_min_distance_seen_zero_is_not_replaced_by_default() -> None:
    assert audit.finite_float_or_default(0.0, 999.0) == 0.0
    assert audit.finite_float_or_default("0.0", 999.0) == 0.0
    assert audit.finite_float_or_default(float("nan"), 999.0) == 999.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_end_to_end_audit_is_read_only_and_writes_required_outputs(tmp_path: Path) -> None:
    pair = tmp_path / "pair.csv"
    candidate = tmp_path / "candidate.csv"
    selected = tmp_path / "selected.csv"
    converted = tmp_path / "converted.csv"
    output = tmp_path / "audit"
    _write_csv(pair, _pair_rows())
    _write_csv(
        candidate,
        [
            {"rank": 1, "candidate_id": "gcf_2", "candidate_smiles": "CC"},
            {"rank": 2, "candidate_id": "gcf_10", "candidate_smiles": "CN"},
        ],
    )
    _write_csv(
        selected,
        [
            {"rank": 1, "candidate_id": "gcf_2", "graph_hash": "h2"},
            {"rank": 2, "candidate_id": "gcf_10", "graph_hash": "h10"},
        ],
    )
    _write_csv(
        converted,
        [
            {"candidate_id": "gcf_2", "smiles": "CC", "sanitize_ok": True},
            {"candidate_id": "gcf_10", "smiles": "CN", "sanitize_ok": True},
        ],
    )
    args = argparse.Namespace(
        pair_details=str(pair),
        candidate_csv=str(candidate),
        selected_metadata=str(selected),
        converted_smiles=str(converted),
        theta=0.04,
        max_k=2,
        method="GCFExplainer",
        figure3_csv=None,
        output_dir=str(output),
    )
    summary = audit.run_audit(args)
    assert summary["distance_recomputed"] is False
    assert summary["num_parents"] == 2
    assert summary["candidate_order_exact_match"] is True
    for name in (
        "prefix_metrics_audit.csv",
        "candidate_order_audit.csv",
        "audit_summary.json",
        "audit_report.txt",
    ):
        assert (output / name).is_file()
    saved = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert saved["distance_recomputed"] is False
