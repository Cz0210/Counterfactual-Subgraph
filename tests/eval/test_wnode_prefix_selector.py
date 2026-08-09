from __future__ import annotations

import pytest
import csv
import tempfile
from pathlib import Path

from src.eval.mutagenicity_wnode_selector import build_candidate_chemistry
from src.eval.wnode_prefix_selector import (
    PrefixVariant,
    assert_calibration_selector_inputs,
    select_sequence,
    threshold_bundle_from_manifest,
    run_bace_wnode_prefix_selector,
)
from tests.eval.wnode_v2_test_utils import matrix_data, threshold_manifest


def test_frozen_threshold_manifest_is_used_verbatim(tmp_path) -> None:
    matrix = matrix_data(tmp_path)
    source = threshold_manifest(tmp_path / "thresholds.json")
    bundle = threshold_bundle_from_manifest(
        source, finite_distance_count=len(matrix.full_finite_distances)
    )
    assert bundle.theta_star == 0.02
    assert bundle.raw_thresholds == (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)


@pytest.mark.parametrize("path", ["split/test.csv", "outputs/gcf/results.json", "GCFExplainer.csv"])
def test_selector_rejects_test_and_gcf_inputs(path: str) -> None:
    with pytest.raises(ValueError, match="test or GCF"):
        assert_calibration_selector_inputs(path)


def test_deterministic_selection_and_swap_non_decrease(tmp_path) -> None:
    matrix = matrix_data(tmp_path)
    thresholds = threshold_bundle_from_manifest(
        threshold_manifest(tmp_path / "thresholds.json"),
        finite_distance_count=len(matrix.full_finite_distances),
    )
    chemistry = build_candidate_chemistry(matrix.candidate_rows)
    variant = PrefixVariant(
        "A4_prefix_covred_swap",
        "prefix",
        lambda_table2=1.0,
        lambda_covred=0.1,
        lambda_struct=0.05,
        lambda_size=0.05,
        lambda_cost=0.1,
        insertion_reorder=True,
        local_swap=True,
    )
    first, trace = select_sequence(matrix, thresholds, chemistry, variant, local_swap_passes=1)
    second, _ = select_sequence(matrix, thresholds, chemistry, variant, local_swap_passes=1)
    assert first == second
    assert len(first) == len(set(first)) == 20
    assert trace["objective_after_swap"] >= trace["objective_before_swap"] - 1e-12


def test_end_to_end_selector_freezes_ranked_calibration_sequence() -> None:
    with tempfile.TemporaryDirectory(prefix="bace_calibration_selector_") as temporary:
        root = Path(temporary)
        matrix = matrix_data(root, parents=60, candidates=20)
        thresholds = threshold_manifest(root / "thresholds.json")
        current = root / "current_selected.csv"
        with current.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rank", "fragment"])
            writer.writeheader()
            for rank, row in enumerate(matrix.candidate_rows, start=1):
                writer.writerow({"rank": rank, "fragment": row["canonical_fragment"]})
        output = root / "selector_output"
        summary = run_bace_wnode_prefix_selector(
            matrix_run_dir=matrix.matrix_run_dir,
            thresholds_json=thresholds,
            current_selected_csv=current,
            output_dir=output,
            local_swap_passes=1,
            fold_count=5,
        )
        assert summary["run_complete"] is True
        assert summary["test_used"] is False
        frozen = (output / "frozen_selection.json").read_text()
        assert '"selection_frozen": true' in frozen
        with (output / "selected_top20.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["rank"]) for row in rows] == list(range(1, 21))
