"""Fixture-only record export checks; final PDF visual QA belongs to integration."""
import csv
import json

import numpy as np
import pytest

from src.baselines.cm_crem_export import export_results, main, replot, _table
from src.baselines.cm_crem_selection import evaluate_frozen_test, select_calibration


def fixture_evaluation(empty=False):
    matrix = np.empty((2, 0)) if empty else np.array([[.1], [.4]])
    candidates = [] if empty else ["synthetic-candidate"]
    frozen = select_calibration(matrix, pair_status=np.full(matrix.shape, "OK"),
        parent_ids=["cal-a", "cal-b"], candidate_ids=candidates, source_mask=np.array([True, True]),
        theta=.2, cap=.5, contract_sha256="a"*64, frozen_pool_sha256="b"*64)
    test = np.empty((3, 0)) if empty else np.array([[.1], [2.], [np.inf]])
    statuses = np.where(np.isfinite(test), "OK", "NON_SOURCE")
    return evaluate_frozen_test(frozen, test, pair_status=statuses,
        parent_ids=["test-a", "test-b", "test-c"], candidate_ids=candidates,
        source_mask=np.array([True, True, False]), contract_sha256="a"*64)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("empty", [False, True])
def test_exports_full_fixture_records_without_science_pass(tmp_path, empty):
    report = export_results(fixture_evaluation(empty), tmp_path, dataset="bace", fixture=True, make_figures=False)
    source = tmp_path/"results/source_csv"
    prefix = read_csv(source/"prefix_metrics.csv")
    assert len(prefix) == 20 and all(r["fixture"] == "true" for r in prefix)
    assert read_csv(source/"table2_k10.csv") == [prefix[9]]
    assert read_csv(source/"table2_k20.csv") == [prefix[19]]
    assert len(read_csv(source/"parent_best_distances.csv")) == 60
    ecdf = read_csv(source/"figure4_k10_exact.csv")
    if empty:
        assert prefix[0]["coverage"] == "0.0" and prefix[0]["conditional_median_cost"] == "N/A"
        assert ecdf == [dict(ecdf[0], distance="0.0", coverage="0.0", finite_recourse_count="0")]
    else:
        assert [row["distance"] for row in ecdf] == ["0.0", "0.1", "2.0"]
        assert ecdf[-1]["coverage"] == repr(2/3)
    assert report["scientific_pass_claimed"] is False and report["status"] == "EXPORTED_RECORDS"
    assert not (tmp_path/"results/final_audit.json").exists()
    assert not list(tmp_path.rglob("PASS"))
    with pytest.raises(FileExistsError):
        export_results(fixture_evaluation(empty), tmp_path, dataset="bace", fixture=True, make_figures=False)


def test_replot_rejects_csv_mutation_before_plot_import(tmp_path):
    export_results(fixture_evaluation(), tmp_path, dataset="bace", fixture=True, make_figures=False)
    source = tmp_path/"results/source_csv"
    with (source/"prefix_metrics.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="CSV changed"):
        replot(source, tmp_path/"replot", dataset="bace")
    assert not (tmp_path/"replot").exists()


def test_replot_rejects_incompatible_oracle_or_dataset(tmp_path):
    export_results(fixture_evaluation(), tmp_path, dataset="bace", fixture=True, make_figures=False)
    with pytest.raises(ValueError, match="original-GINE panel"):
        replot(tmp_path/"results/source_csv", tmp_path/"replot", dataset="tastemolnet")
    with pytest.raises(ValueError, match="original frozen GINE"):
        replot(tmp_path/"results/source_csv", tmp_path/"replot", dataset="bace", oracle="gin_aplus")


def test_table_records_undefined_without_invented_zero(tmp_path):
    _table(tmp_path/"table.tex", {"effective_k": "0", "coverage": "0.0",
        "fixed_capped_mean_cost": ".5", "conditional_median_cost": "N/A"}, fixture=True)
    content = (tmp_path/"table.tex").read_text()
    assert "N/A" in content and "synthetic fixture" in content and "\\begin{tabular}" in content
    assert content.count("\\\\") == 2


def test_module_replot_help_is_executable(capsys):
    with pytest.raises(SystemExit) as result:
        main(["replot", "--help"])
    assert result.value.code == 0
    assert "--source-csv" in capsys.readouterr().out


def test_export_accepts_authenticated_saved_driver_json(tmp_path):
    payload = {"science_hash": "a"*64, **fixture_evaluation().to_dict()}
    result = export_results(payload, tmp_path, dataset="bace", fixture=True, make_figures=False)
    assert result["status"] == "EXPORTED_RECORDS"
