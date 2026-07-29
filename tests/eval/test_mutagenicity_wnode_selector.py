from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.mutagenicity_wnode_selector import (
    ChemistryData,
    MatrixData,
    ThresholdLevel,
    VariantConfig,
    audit_mutagenicity_wnode_selector,
    build_coverage_redundancy_matrix,
    choose_variant,
    compute_prefix_metrics,
    derive_thresholds,
    fixed_denominator_capped_cost,
    load_calibration_matrix,
    local_swap_search,
    morgan_tanimoto,
    optimize_insertion_order,
    run_mutagenicity_wnode_selector,
    single_threshold_coverage,
    weighted_coverage_jaccard,
    weighted_multi_threshold_utility,
)


def _matrix_data() -> MatrixData:
    distances = np.asarray(
        [
            [0.10, np.inf, 0.30],
            [np.inf, 0.15, 0.25],
            [0.40, 0.20, np.inf],
            [np.inf, np.inf, 0.35],
        ],
        dtype=np.float64,
    )
    cf_drops = np.where(np.isfinite(distances), 0.7, np.nan)
    applicable = np.asarray(
        [
            [True, True, True],
            [False, True, True],
            [True, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    candidates = tuple(
        {
            "candidate_id": f"c{index + 1}",
            "canonical_fragment": fragment,
            "source_parent_count": 3 - index,
            "source_cf_drop_mean": 0.7 - index * 0.1,
            "source_reward_mean": 1.0 - index * 0.1,
        }
        for index, fragment in enumerate(("C", "N", "O"))
    )
    finite = distances[np.isfinite(distances)]
    return MatrixData(
        matrix_run_dir=Path("/calibration_matrix"),
        parent_ids=("p1", "p2", "p3", "p4"),
        candidate_rows=candidates,
        distances=distances,
        cf_drops=cf_drops,
        applicable=applicable,
        full_finite_distances=np.asarray(finite, dtype=np.float64),
        full_parent_count=4,
        full_candidate_count=3,
        full_pair_count=12,
        full_strict_flip_pair_count=int(finite.size),
        summary={"test_loaded": False},
        manifest={
            "test_loaded": False,
            "inputs": {"cohort_name": "calibration"},
        },
    )


def _levels() -> tuple[ThresholdLevel, ...]:
    return (
        ThresholdLevel("q25", 0.20, 2.0, (0.25,), ("q25",)),
        ThresholdLevel("q50", 0.30, 1.0, (0.50,), ("q50",)),
    )


def test_single_threshold_coverage_and_strict_distance_semantics() -> None:
    best = np.asarray([0.1, np.inf, 0.2, np.inf], dtype=np.float64)

    assert single_threshold_coverage(best, 0.2) == pytest.approx(0.5)
    assert single_threshold_coverage(best, 0.15) == pytest.approx(0.25)


def test_multi_threshold_weighted_utility() -> None:
    best = np.asarray([0.1, 0.25, np.inf, np.inf], dtype=np.float64)

    value = weighted_multi_threshold_utility(best, _levels())

    assert value == pytest.approx((2.0 * 0.25 + 1.0 * 0.50) / 3.0)


def test_strict_false_and_null_wnode_do_not_cover(tmp_path: Path) -> None:
    run_dir = _write_fake_matrix(tmp_path / "calibration_matrix")
    pair_path = run_dir / "pair_matrix.jsonl"
    rows = [
        json.loads(line)
        for line in pair_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["pair_strict_flip"] = False
    rows[0]["wnode_distance"] = 0.0
    rows[1]["pair_strict_flip"] = True
    rows[1]["wnode_distance"] = None
    pair_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["strict_flip_pair_count"] = sum(
        bool(row["pair_strict_flip"]) for row in rows
    )
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="finite non-negative"):
        load_calibration_matrix(run_dir)

    rows[1]["pair_strict_flip"] = False
    pair_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary["strict_flip_pair_count"] = sum(
        bool(row["pair_strict_flip"]) for row in rows
    )
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    loaded = load_calibration_matrix(run_dir)
    assert np.isinf(loaded.distances[0, 0])
    assert np.isinf(loaded.distances[0, 1])


def test_fixed_denominator_capped_cost_includes_unavailable_parents() -> None:
    best = np.asarray([0.1, np.inf, 0.9, np.inf], dtype=np.float64)

    mean, median, capped = fixed_denominator_capped_cost(best, 0.5)

    assert capped.tolist() == pytest.approx([0.1, 0.5, 0.5, 0.5])
    assert mean == pytest.approx(0.4)
    assert median == pytest.approx(0.5)


def test_nested_prefix_metrics_are_monotonic() -> None:
    matrix = _matrix_data()
    thresholds = derive_thresholds(
        matrix.full_finite_distances,
        quantiles=(0.2, 0.5, 0.9),
        weights=(2, 1, 1),
        theta_star_quantile=0.3,
        cost_cap_quantile=0.9,
    )
    covred = build_coverage_redundancy_matrix(
        matrix.distances,
        thresholds.levels,
    )
    structural = np.eye(3, dtype=np.float64)

    metrics, _ = compute_prefix_metrics(
        [0, 1, 2],
        matrix=matrix,
        thresholds=thresholds,
        coverage_redundancy_matrix=covred,
        structural_similarity_matrix=structural,
    )

    assert [row["k"] for row in metrics] == [1, 2, 3]
    coverage = [row["ccrcov_theta_star"] for row in metrics]
    costs = [row["fixed_capped_mean_cost"] for row in metrics]
    assert coverage == sorted(coverage)
    assert all(right <= left + 1e-12 for left, right in zip(costs, costs[1:]))


def test_theta_star_is_q30_and_duplicate_threshold_weights_merge() -> None:
    values = np.asarray([0.1, 0.1, 0.1, 0.4], dtype=np.float64)

    thresholds = derive_thresholds(
        values,
        quantiles=(0.05, 0.10, 0.20, 0.30),
        weights=(4, 4, 3, 3),
    )

    assert thresholds.theta_star == pytest.approx(
        np.quantile(values, 0.30, method="linear")
    )
    assert len(thresholds.levels) < 4
    first = thresholds.levels[0]
    assert first.quantiles == (0.05, 0.10, 0.20, 0.30)
    assert first.weight == pytest.approx(14.0)


def test_weighted_covred_jaccard() -> None:
    left = np.asarray([0.1, 0.2, np.inf, np.inf])
    right = np.asarray([0.1, np.inf, 0.2, np.inf])
    levels = (ThresholdLevel("q", 0.2, 1.0, (0.2,), ("q",)),)

    assert weighted_coverage_jaccard(left, right, levels) == pytest.approx(1 / 3)


def test_morgan_tanimoto_is_finite_and_identity_is_one() -> None:
    assert morgan_tanimoto("CCO", "CCO") == pytest.approx(1.0)
    different = morgan_tanimoto("CCO", "c1ccccc1")
    assert 0.0 <= different <= 1.0
    with pytest.raises(ValueError, match="Invalid fragment"):
        morgan_tanimoto("not-smiles", "CCO")


def test_a3_insertion_reordering_never_lowers_objective() -> None:
    candidate_ids = ("a", "b", "c")

    def objective(sequence: list[int]) -> float:
        weights = (3.0, 2.0, 1.0)
        values = (1.0, 3.0, 2.0)
        return sum(
            weights[index] * values[candidate]
            for index, candidate in enumerate(sequence)
        )

    original = [0, 1, 2]
    reordered, trace = optimize_insertion_order(
        original,
        objective_fn=objective,
        candidate_ids=candidate_ids,
    )

    assert objective(reordered) >= objective(original)
    assert reordered == [1, 2, 0]
    assert trace[-1]["objective"] >= trace[0]["objective"]


def test_a4_local_swap_never_lowers_objective() -> None:
    values = (1.0, 2.0, 5.0, 4.0)
    candidate_ids = ("a", "b", "c", "d")

    def objective(sequence: list[int]) -> float:
        return sum(values[index] for index in sequence)

    selected, trace = local_swap_search(
        [0, 1],
        all_candidate_indices=[0, 1, 2, 3],
        objective_fn=objective,
        candidate_ids=candidate_ids,
        max_passes=2,
    )

    assert objective(selected) >= objective([0, 1])
    assert set(selected) == {2, 3}
    accepted = [row for row in trace if row["operation"] == "swap_accept"]
    assert all(row["objective_after"] > row["objective_before"] for row in accepted)


def test_variant_decision_uses_preregistered_lexicographic_rule() -> None:
    rows = []
    for name in (
        "A1_SingleTheta",
        "A2_MultiThreshold",
        "A3_MultiThresholdPrefix",
        "A4_MultiThresholdPrefixCovRedSwap",
    ):
        rows.append(
            {
                "variant": name,
                "prefix_weighted_multi_threshold_utility": 0.5,
                "k10_ccrcov_theta_star": 0.4,
                "k10_fixed_capped_mean_cost": 0.2,
                "k20_weighted_multi_threshold_utility": 0.6,
                "k20_coverage_redundancy": 0.1,
                "k20_structural_redundancy": 0.1,
            }
        )
    rows[2]["prefix_weighted_multi_threshold_utility"] = 0.6
    rows[3]["prefix_weighted_multi_threshold_utility"] = 0.6
    rows[3]["k10_ccrcov_theta_star"] = 0.5

    assert choose_variant(rows)["variant"] == "A4_MultiThresholdPrefixCovRedSwap"


def _write_fake_matrix(path: Path) -> Path:
    path.mkdir(parents=True)
    candidate_fragments = ("C", "N", "O", "CC", "CN", "CO")
    candidate_rows = [
        {
            "candidate_id": f"c{index + 1}",
            "canonical_fragment": fragment,
            "source_parent_count": 6 - index,
            "source_cf_drop_mean": 0.8 - index * 0.05,
            "source_reward_mean": 1.0 - index * 0.05,
        }
        for index, fragment in enumerate(candidate_fragments)
    ]
    (path / "selected_candidate_universe.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidate_rows),
        encoding="utf-8",
    )
    distances = np.asarray(
        [
            [0.10, 0.20, np.inf, 0.40, np.inf, 0.05],
            [np.inf, 0.30, 0.10, np.inf, 0.25, np.inf],
            [0.40, np.inf, 0.20, 0.10, 0.30, np.inf],
            [np.inf, np.inf, 0.50, 0.20, 0.35, 0.45],
        ],
        dtype=np.float64,
    )
    pair_rows = []
    for parent_index in range(distances.shape[0]):
        for candidate_index in range(distances.shape[1]):
            distance = float(distances[parent_index, candidate_index])
            strict = math_isfinite(distance)
            pair_rows.append(
                {
                    "parent_id": f"p{parent_index + 1}",
                    "candidate_id": f"c{candidate_index + 1}",
                    "applicable": bool(
                        strict or (parent_index + candidate_index) % 2 == 0
                    ),
                    "pair_strict_flip": strict,
                    "wnode_distance": distance if strict else None,
                    "cf_drop": 0.8 - 0.05 * candidate_index if strict else None,
                    "pred_before": 1 if strict else None,
                    "pred_after": 0 if strict else None,
                }
            )
    (path / "pair_matrix.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    strict_count = sum(bool(row["pair_strict_flip"]) for row in pair_rows)
    summary = {
        "parent_count": 4,
        "selected_candidate_count": 6,
        "actual_pair_rows": 24,
        "strict_flip_pair_count": strict_count,
        "test_loaded": False,
        "run_complete": True,
    }
    manifest = {
        "inputs": {
            "cohort_name": "calibration",
            "calibration_csv": {
                "path": str(path.parent / "calibration.csv"),
            },
        },
        "test_loaded": False,
        "run_complete": True,
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))


def test_full_fake_selector_writes_nested_top_prefix_and_passes_audit(
    tmp_path: Path,
) -> None:
    matrix_dir = _write_fake_matrix(tmp_path / "calibration_matrix")
    output_dir = tmp_path / "selector"
    prefix_weights = (1.0, 1.0, 0.5)

    summary = run_mutagenicity_wnode_selector(
        matrix_run_dir=matrix_dir,
        output_dir=output_dir,
        top_k=3,
        table_k=2,
        threshold_quantiles=(0.05, 0.20, 0.30, 0.50, 0.90),
        threshold_weights=(4, 3, 3, 2, 1),
        prefix_weights=prefix_weights,
        local_swap_passes=2,
        seed=13,
        forbid_test=True,
    )

    assert summary["run_complete"] is True
    for variant in (
        "A1_SingleTheta",
        "A2_MultiThreshold",
        "A3_MultiThresholdPrefix",
        "A4_MultiThresholdPrefixCovRedSwap",
    ):
        variant_dir = output_dir / "variants" / variant
        top2 = json.loads((variant_dir / "selected_top10.json").read_text())
        top3 = json.loads((variant_dir / "selected_top20.json").read_text())
        assert top2["candidate_ids"] == top3["candidate_ids"][:2]

    audit = audit_mutagenicity_wnode_selector(
        run_dir=output_dir,
        matrix_run_dir=matrix_dir,
        expected_parent_count=4,
        expected_candidate_count=6,
        expected_top_k=3,
        expected_table_k=2,
        require_all_variants=True,
        require_nested_prefix=True,
        require_monotonic_coverage=True,
        require_nonincreasing_capped_cost=True,
        forbid_test=True,
    )
    assert audit["audit_passed"] is True


def test_test_path_or_test_cohort_is_rejected(tmp_path: Path) -> None:
    forbidden_path = _write_fake_matrix(tmp_path / "test_matrix")
    with pytest.raises(ValueError, match="Test matrix path"):
        load_calibration_matrix(forbidden_path)

    calibration_path = _write_fake_matrix(tmp_path / "calibration_matrix")
    manifest_path = calibration_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["cohort_name"] = "test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort must be calibration"):
        load_calibration_matrix(calibration_path)
