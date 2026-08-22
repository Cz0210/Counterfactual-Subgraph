from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_ccrcov_with_molclr_node_wasserstein import summarize_method
from src.eval.fullgraph_wnode_artifacts import (
    OFFICIAL_FIELDS,
    TABLE_REQUIRED_FIELDS,
    _method_slug,
    audit_final_artifacts,
    compute_prefix_artifacts,
    export_final_artifacts,
    load_ranked_candidates,
    reconstruct_official_summary,
    summarize_wnode_thresholds,
    validate_complete_cartesian,
)
from src.eval.mutagenicity_wnode_selector import morgan_tanimoto


FRAGMENTS = (
    "C",
    "N",
    "O",
    "F",
    "Cl",
    "Br",
    "CC",
    "CN",
    "CO",
    "C=C",
    "C#N",
    "CCC",
    "CCN",
    "CCO",
    "CCF",
    "CCCl",
    "CCBr",
    "CNC",
    "COC",
    "N#N",
)
THRESHOLDS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10)
THETA_STAR = 0.04
COST_CAP = 0.10
HISTORICAL_MUT_THETA_STAR = 0.05
HISTORICAL_MUT_COST_CAP = 0.0535
HISTORICAL_MUT_THRESHOLDS = tuple(
    HISTORICAL_MUT_COST_CAP * index / 600 for index in range(601)
)


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _candidate_rows() -> list[dict]:
    return [
        {
            "rank": rank,
            "candidate_id": f"c{rank:02d}",
            "canonical_smiles": fragment,
        }
        for rank, fragment in enumerate(FRAGMENTS, start=1)
    ]


def _pair_rows() -> list[dict]:
    rows: list[dict] = []
    for parent_id in ("p1", "p2", "p3"):
        for rank in range(20, 0, -1):
            candidate_id = f"c{rank:02d}"
            distance = 0.2 + rank / 1000.0
            pred_after = 1
            cf_drop = 0.0
            if parent_id == "p1" and rank == 1:
                distance, pred_after, cf_drop = 0.03, 0, 0.50
            elif parent_id == "p1" and rank == 2:
                distance, pred_after, cf_drop = 0.02, 0, 0.60
            elif parent_id == "p2" and rank == 11:
                distance, pred_after, cf_drop = 0.04, 0, 0.40
            rows.append(
                {
                    "method": "fake_fullgraph",
                    "parent_id": parent_id,
                    "parent_smiles": "CCO",
                    "label": 1,
                    "candidate_id": candidate_id,
                    "candidate_smiles": FRAGMENTS[rank - 1],
                    "match": True,
                    "delete_valid": True,
                    "applicable": True,
                    "pred_before": 1,
                    "pred_after": pred_after,
                    "cf_flip": pred_after == 0,
                    "teacher_strict_flip": pred_after == 0,
                    "cf_drop": cf_drop,
                    "distance": distance,
                }
            )
    return list(reversed(rows))


def _threshold_rows() -> list[dict]:
    return [
        {
            "threshold": threshold,
            "threshold_source": "explicit",
            "quantile": quantile,
        }
        for threshold, quantile in zip(
            THRESHOLDS, (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
        )
    ]


def _make_inputs(root: Path) -> dict[str, Path]:
    candidates_path = root / "frozen" / "selected_top20.csv"
    # Deliberately reverse file rows: rank, not CSV row order, is authoritative.
    _write_csv(candidates_path, list(reversed(_candidate_rows())))
    test_run = root / "test_run"
    details = _pair_rows()
    _write_csv(test_run / "details" / "pair_details.csv", details)
    official = summarize_wnode_thresholds(
        method="fake_fullgraph",
        details=details,
        threshold_rows=_threshold_rows(),
        total_parents=3,
        total_candidates=20,
        source_label=1,
        target_label=0,
    )
    _write_csv(
        test_run / "combined" / "combined_threshold_summary.csv",
        official,
    )
    (test_run / "run_config.json").write_text(
        json.dumps(
            {
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "threshold_source": "explicit",
                "teacher_path": "missing_teacher.pkl",
                "teacher_hash": "teacher-hash",
                "molclr_checkpoint": "missing_molclr.pt",
                "molclr_checkpoint_hash": "molclr-hash",
            }
        ),
        encoding="utf-8",
    )
    calibration = root / "calibration"
    calibration.mkdir()
    (calibration / "summary.json").write_text(
        json.dumps({"cohort": "calibration"}), encoding="utf-8"
    )
    ours = root / "ours"
    ours.mkdir()
    (ours / "thresholds.json").write_text(
        json.dumps(
            {
                "theta_star": THETA_STAR,
                "cost_cap": COST_CAP,
                "raw_quantile_thresholds": [
                    {"threshold": value} for value in THRESHOLDS
                ],
            }
        ),
        encoding="utf-8",
    )
    fields = list(TABLE_REQUIRED_FIELDS)
    for k in (10, 20):
        _write_csv(
            ours / f"table2_ours_k{k}.csv",
            [{field: "" for field in fields}],
            fields,
        )
    return {
        "candidates": candidates_path,
        "test_run": test_run,
        "calibration": calibration,
        "ours": ours,
    }


def _export(root: Path, *, output_name: str = "output") -> tuple[Path, dict]:
    inputs = _make_inputs(root)
    output = root / output_name
    summary = export_final_artifacts(
        test_run_dir=inputs["test_run"],
        calibration_run_dir=inputs["calibration"],
        frozen_candidates_csv=inputs["candidates"],
        ours_schema_root=inputs["ours"],
        output_dir=output,
        method_name="FakeFullgraph",
        dataset="Mutagenicity",
        source_label=1,
        target_label=0,
        test_job_id="123",
        theta_star=THETA_STAR,
        cost_cap=COST_CAP,
        thresholds=THRESHOLDS,
        k_values=range(1, 21),
        expected_parent_count=3,
        expected_candidate_count=20,
        expected_pair_count=60,
        forbid_selection=True,
        forbid_fitting=True,
    )
    return output, summary


def test_candidate_rank_controls_prefix_and_cartesian_validation(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    candidates, _ = load_ranked_candidates(inputs["candidates"], expected_count=20)
    assert [row["candidate_id"] for row in candidates[:2]] == ["c01", "c02"]
    details = _pair_rows()
    parents, _ = validate_complete_cartesian(
        details,
        candidates,
        expected_parent_count=3,
        expected_pair_count=60,
    )
    prefix, _, _ = compute_prefix_artifacts(
        details=details,
        candidates=candidates,
        parent_ids=parents,
        thresholds=THRESHOLDS,
        theta_star=THETA_STAR,
        cost_cap=COST_CAP,
        source_label=1,
        target_label=0,
        method_name="FakeFullgraph",
    )
    assert prefix[0]["num_close_cf_covered"] == 1
    assert prefix[9]["num_any_strict_flip_parents"] == 1
    assert prefix[19]["num_any_strict_flip_parents"] == 2
    assert [row["candidate_id"] for row in candidates[:10]] == [
        f"c{rank:02d}" for rank in range(1, 11)
    ]


def test_costs_use_correct_parent_sets_and_fixed_denominator(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    candidates, _ = load_ranked_candidates(inputs["candidates"], expected_count=20)
    details = _pair_rows()
    parents, _ = validate_complete_cartesian(
        details, candidates, expected_parent_count=3, expected_pair_count=60
    )
    prefix, _, parent_rows = compute_prefix_artifacts(
        details=details,
        candidates=candidates,
        parent_ids=parents,
        thresholds=THRESHOLDS,
        theta_star=THETA_STAR,
        cost_cap=COST_CAP,
        source_label=1,
        target_label=0,
        method_name="FakeFullgraph",
    )
    k1 = prefix[0]
    assert k1["conditional_mean_cost"] == pytest.approx(0.03)
    assert k1["conditional_median_cost"] == pytest.approx(0.03)
    assert k1["fixed_capped_mean_cost"] == pytest.approx((0.03 + 0.10 + 0.10) / 3)
    assert k1["fixed_capped_median_cost"] == pytest.approx(0.10)
    p3 = next(row for row in parent_rows if row["k"] == 20 and row["parent_id"] == "p3")
    assert p3["best_distance"] is None
    assert p3["capped_distance"] == COST_CAP


def test_redundancy_reuses_project_helpers(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    candidates, _ = load_ranked_candidates(inputs["candidates"], expected_count=20)
    details = _pair_rows()
    parents, _ = validate_complete_cartesian(
        details, candidates, expected_parent_count=3, expected_pair_count=60
    )
    prefix, _, _ = compute_prefix_artifacts(
        details=details,
        candidates=candidates,
        parent_ids=parents,
        thresholds=THRESHOLDS,
        theta_star=THETA_STAR,
        cost_cap=COST_CAP,
        source_label=1,
        target_label=0,
        method_name="FakeFullgraph",
    )
    assert prefix[1]["coverage_redundancy"] == pytest.approx(1.0)
    assert prefix[1]["structural_redundancy"] == pytest.approx(
        morgan_tanimoto(FRAGMENTS[0], FRAGMENTS[1])
    )


def test_export_writes_figure_table_provenance_and_hashes(tmp_path: Path) -> None:
    output, summary = _export(tmp_path)
    assert summary["k10_ccrcov_theta_star"] == pytest.approx(1 / 3)
    assert summary["k20_ccrcov_theta_star"] == pytest.approx(2 / 3)
    assert len(_read_csv(output / "figure3_coverage_vs_k.csv")) == 20
    assert (output / "selected_sequence.jsonl").is_file()
    assert (output / "audit.json").is_file()
    assert (output / "_RUN_COMPLETE.json").is_file()
    figure4 = _read_csv(output / "figure4_coverage_vs_threshold.csv")
    assert len(figure4) == 14
    assert {int(row["k"]) for row in figure4} == {10, 20}
    for k in (10, 20):
        table_path = output / f"table2_fakefullgraph_k{k}.csv"
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            assert csv.DictReader(handle).fieldnames == list(TABLE_REQUIRED_FIELDS)
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        actual = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        assert actual == expected
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["selection_used_test"] is False
    assert run_manifest["threshold_fitted_on_test"] is False
    assert run_manifest["test_job_id"] == "123"
    result = audit_final_artifacts(
        run_dir=output,
        frozen_candidates_csv=tmp_path / "frozen" / "selected_top20.csv",
        ours_schema_root=tmp_path / "ours",
        expected_parent_count=3,
        expected_candidate_count=20,
        expected_pair_count=60,
        theta_star=THETA_STAR,
        cost_cap=COST_CAP,
        thresholds=THRESHOLDS,
    )
    assert result["manifest_hashes_verified"] is True


def test_export_historical_mut_grid_without_theta_is_schema_compatible(
    tmp_path: Path,
) -> None:
    inputs = _make_inputs(tmp_path)
    details = _pair_rows()
    official = summarize_wnode_thresholds(
        method="fake_fullgraph",
        details=details,
        threshold_rows=[
            {
                "threshold": threshold,
                "threshold_source": "frozen_calibration",
                "quantile": None,
            }
            for threshold in HISTORICAL_MUT_THRESHOLDS
        ],
        total_parents=3,
        total_candidates=20,
        source_label=1,
        target_label=0,
    )
    _write_csv(
        inputs["test_run"] / "combined" / "combined_threshold_summary.csv",
        official,
    )
    (inputs["ours"] / "thresholds.json").write_text(
        json.dumps(
            {
                "theta_star": HISTORICAL_MUT_THETA_STAR,
                "cost_cap": HISTORICAL_MUT_COST_CAP,
                "raw_quantile_thresholds": [
                    {"threshold": value} for value in HISTORICAL_MUT_THRESHOLDS
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "historical-mut-export"
    export_final_artifacts(
        test_run_dir=inputs["test_run"],
        calibration_run_dir=inputs["calibration"],
        frozen_candidates_csv=inputs["candidates"],
        ours_schema_root=inputs["ours"],
        output_dir=output,
        method_name="GCFExplainer-Top20",
        dataset="Mutagenicity",
        source_label=1,
        target_label=0,
        test_job_id="production-like-fixture",
        theta_star=HISTORICAL_MUT_THETA_STAR,
        cost_cap=HISTORICAL_MUT_COST_CAP,
        thresholds=HISTORICAL_MUT_THRESHOLDS,
        k_values=range(1, 21),
        expected_parent_count=3,
        expected_candidate_count=20,
        expected_pair_count=60,
        forbid_selection=True,
        forbid_fitting=True,
    )
    reconstruction = json.loads(
        (output / "official_summary_reconstruction_audit.json").read_text()
    )
    assert reconstruction["threshold_count"] == 601
    assert reconstruction["theta_star_row_source"] == "recomputed_prefix_theta_star"
    assert len(_read_csv(output / "test_threshold_summary.csv")) == 601
    assert len(_read_csv(output / "figure4_coverage_vs_threshold.csv")) == 1202
    final_audit = audit_final_artifacts(
        run_dir=output,
        frozen_candidates_csv=inputs["candidates"],
        ours_schema_root=inputs["ours"],
        expected_parent_count=3,
        expected_candidate_count=20,
        expected_pair_count=60,
        theta_star=HISTORICAL_MUT_THETA_STAR,
        cost_cap=HISTORICAL_MUT_COST_CAP,
        thresholds=HISTORICAL_MUT_THRESHOLDS,
    )
    assert final_audit["manifest_hashes_verified"] is True


def test_gcfexplainer_table_slug_is_stable() -> None:
    assert _method_slug("GCFExplainer-Top20") == "gcfexplainer"


def test_official_mismatch_rejects_without_output(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    official_path = (
        inputs["test_run"] / "combined" / "combined_threshold_summary.csv"
    )
    rows = _read_csv(official_path)
    rows[0]["num_valid_pairs"] = "59"
    _write_csv(official_path, rows)
    output = tmp_path / "rejected"
    with pytest.raises(RuntimeError, match="reconstruction failed"):
        export_final_artifacts(
            test_run_dir=inputs["test_run"],
            calibration_run_dir=inputs["calibration"],
            frozen_candidates_csv=inputs["candidates"],
            ours_schema_root=inputs["ours"],
            output_dir=output,
            method_name="FakeFullgraph",
            dataset="Mutagenicity",
            source_label=1,
            target_label=0,
            test_job_id="123",
            theta_star=THETA_STAR,
            cost_cap=COST_CAP,
            thresholds=THRESHOLDS,
            k_values=range(1, 21),
            expected_parent_count=3,
            expected_candidate_count=20,
            expected_pair_count=60,
            forbid_selection=True,
            forbid_fitting=True,
        )
    assert not output.exists()


def test_official_integer_and_float_reconstruction() -> None:
    details = _pair_rows()
    rows = summarize_wnode_thresholds(
        method="fake",
        details=details,
        threshold_rows=_threshold_rows(),
        total_parents=3,
        total_candidates=20,
        source_label=1,
        target_label=0,
    )
    result = reconstruct_official_summary(
        recomputed_k20=rows,
        official_rows=rows,
        thresholds=THRESHOLDS,
        theta_star=THETA_STAR,
        expected_theta_star_covered=2,
    )
    assert result["official_summary_reconstruction_passed"] is True
    assert result["theta_star_close_cf_coverage"] == pytest.approx(2 / 3)
    changed = [dict(row) for row in rows]
    changed[0]["close_cf_coverage"] += 2e-12
    with pytest.raises(RuntimeError):
        reconstruct_official_summary(
            recomputed_k20=rows,
            official_rows=changed,
            thresholds=THRESHOLDS,
            theta_star=THETA_STAR,
        )


def test_historical_mut_gcf_grid_uses_exact_separate_theta_row(
    tmp_path: Path,
) -> None:
    """The frozen 601-point grid brackets 0.05 but does not contain it."""

    assert not any(
        abs(threshold - HISTORICAL_MUT_THETA_STAR) <= 1e-12
        for threshold in HISTORICAL_MUT_THRESHOLDS
    )

    candidates_path = tmp_path / "selected_top20.csv"
    _write_csv(candidates_path, _candidate_rows())
    candidates, _ = load_ranked_candidates(candidates_path, expected_count=20)
    details = _pair_rows()
    parent_ids, _ = validate_complete_cartesian(
        details,
        candidates,
        expected_parent_count=3,
        expected_pair_count=60,
    )
    prefix, threshold_rows, _ = compute_prefix_artifacts(
        details=details,
        candidates=candidates,
        parent_ids=parent_ids,
        thresholds=HISTORICAL_MUT_THRESHOLDS,
        theta_star=HISTORICAL_MUT_THETA_STAR,
        cost_cap=HISTORICAL_MUT_COST_CAP,
        source_label=1,
        target_label=0,
        method_name="GCFExplainer-Top20",
    )
    k20_rows = [row for row in threshold_rows if int(row["k"]) == 20]
    theta_row = prefix[-1]
    assert theta_row["threshold"] == HISTORICAL_MUT_THETA_STAR
    assert theta_row["threshold_source"] == "frozen_calibration_theta_star"

    result = reconstruct_official_summary(
        recomputed_k20=k20_rows,
        official_rows=k20_rows,
        thresholds=HISTORICAL_MUT_THRESHOLDS,
        theta_star=HISTORICAL_MUT_THETA_STAR,
        recomputed_theta_star_row=theta_row,
    )
    assert result["official_summary_reconstruction_passed"] is True
    assert result["theta_star_row_source"] == "recomputed_prefix_theta_star"
    assert result["theta_star_num_close_cf_covered"] == 2

    with pytest.raises(RuntimeError, match="explicit recomputed theta-star row"):
        reconstruct_official_summary(
            recomputed_k20=k20_rows,
            official_rows=k20_rows,
            thresholds=HISTORICAL_MUT_THRESHOLDS,
            theta_star=HISTORICAL_MUT_THETA_STAR,
        )

    nearest_grid_row = min(
        k20_rows,
        key=lambda row: abs(
            float(row["threshold"]) - HISTORICAL_MUT_THETA_STAR
        ),
    )
    with pytest.raises(RuntimeError, match="threshold differs"):
        reconstruct_official_summary(
            recomputed_k20=k20_rows,
            official_rows=k20_rows,
            thresholds=HISTORICAL_MUT_THRESHOLDS,
            theta_star=HISTORICAL_MUT_THETA_STAR,
            recomputed_theta_star_row=nearest_grid_row,
        )

    wrong_provenance = dict(theta_row)
    wrong_provenance["threshold_source"] = "frozen_calibration"
    with pytest.raises(RuntimeError, match="lacks the historical"):
        reconstruct_official_summary(
            recomputed_k20=k20_rows,
            official_rows=k20_rows,
            thresholds=HISTORICAL_MUT_THRESHOLDS,
            theta_star=HISTORICAL_MUT_THETA_STAR,
            recomputed_theta_star_row=wrong_provenance,
        )

    incomplete = dict(theta_row)
    incomplete.pop("num_valid_pairs")
    with pytest.raises(RuntimeError, match="is incomplete"):
        reconstruct_official_summary(
            recomputed_k20=k20_rows,
            official_rows=k20_rows,
            thresholds=HISTORICAL_MUT_THRESHOLDS,
            theta_star=HISTORICAL_MUT_THETA_STAR,
            recomputed_theta_star_row=incomplete,
        )

    identity_drift = dict(theta_row)
    identity_drift["k"] = 19
    identity_drift["num_candidates"] = 19
    with pytest.raises(RuntimeError, match="identity field num_candidates"):
        reconstruct_official_summary(
            recomputed_k20=k20_rows,
            official_rows=k20_rows,
            thresholds=HISTORICAL_MUT_THRESHOLDS,
            theta_star=HISTORICAL_MUT_THETA_STAR,
            recomputed_theta_star_row=identity_drift,
        )


def test_shared_helper_preserves_evaluator_summary() -> None:
    details = _pair_rows()
    shared = summarize_wnode_thresholds(
        method="fake",
        details=details,
        threshold_rows=_threshold_rows(),
        total_parents=3,
        total_candidates=20,
    )
    evaluator = summarize_method(
        method="fake",
        details=details,
        threshold_rows=_threshold_rows(),
        total_parents=3,
        total_candidates=20,
        config=SimpleNamespace(
            feature_cost="cosine",
            node_mass="uniform",
            size_penalty_beta=0.0,
        ),
        cf_mode="strict_flip",
        cache_hit_rate=0.0,
        node_embedding_cache_hit_rate=0.0,
        skip_redundancy=True,
        group_audit={},
    )
    for shared_row, evaluator_row in zip(shared, evaluator):
        assert {
            field: shared_row[field] for field in OFFICIAL_FIELDS
        } == {
            field: evaluator_row[field] for field in OFFICIAL_FIELDS
        }


def test_duplicate_or_missing_cartesian_pair_is_rejected(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    candidates, _ = load_ranked_candidates(inputs["candidates"], expected_count=20)
    details = _pair_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_complete_cartesian(
            details + [details[0]],
            candidates,
            expected_parent_count=3,
            expected_pair_count=60,
        )
    with pytest.raises(ValueError, match="Cartesian"):
        validate_complete_cartesian(
            details[:-1],
            candidates,
            expected_parent_count=3,
            expected_pair_count=60,
        )


def test_existing_output_directory_is_rejected(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_final_artifacts(
            test_run_dir=inputs["test_run"],
            calibration_run_dir=inputs["calibration"],
            frozen_candidates_csv=inputs["candidates"],
            ours_schema_root=inputs["ours"],
            output_dir=output,
            method_name="FakeFullgraph",
            dataset="Mutagenicity",
            source_label=1,
            target_label=0,
            test_job_id="123",
            theta_star=THETA_STAR,
            cost_cap=COST_CAP,
            thresholds=THRESHOLDS,
            k_values=range(1, 21),
            expected_parent_count=3,
            expected_candidate_count=20,
            expected_pair_count=60,
            forbid_selection=True,
            forbid_fitting=True,
        )
