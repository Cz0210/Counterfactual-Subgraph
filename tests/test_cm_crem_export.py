"""Fixture-only record export checks; final PDF visual QA belongs to integration."""
import csv
import json

import numpy as np
import pytest

from src.baselines.cm_crem_export import export_diagnostics, export_results, main, replot, _table
from src.baselines.cm_crem_selection import PrefixEvaluation, evaluate_frozen_test, select_calibration


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


def diagnostic_fixture(tmp_path, one_target=False):
    # Reuse the existing explicitly synthetic closed-receipt fixture, not a
    # final-PASS directory or injected real scientific output.
    from test_cm_crem_audit import full_zero_fixture, fixture_audit, write_json
    spec, _ = full_zero_fixture(tmp_path, one_target=one_target)
    write_json(tmp_path/"spec.json", spec)
    provenance = fixture_audit(spec, tmp_path)
    write_json(tmp_path/"audit/provenance_review.json", provenance)
    return PrefixEvaluation.from_dict(provenance["test_evaluation"])


@pytest.mark.parametrize("one_target", [False, True])
def test_diagnostic_sidecars_use_actual_closed_fixture_receipts(tmp_path, one_target):
    evaluation = diagnostic_fixture(tmp_path, one_target)
    report = export_diagnostics(tmp_path, evaluation, fixture=True)
    funnel = read_csv(tmp_path/"candidate_funnel.csv")
    provenance = read_csv(tmp_path/"candidate_provenance.csv")
    budget = json.loads((tmp_path/"budget_and_timing.json").read_text())
    assert len(funnel) == 386
    assert sum(r["predicted_source"] == "true" for r in funnel) == 32
    assert all(r["retained_raw_count"] == "N/A" for r in funnel if r["predicted_source"] == "false")
    assert len(provenance) == int(one_target)
    assert report["candidate_provenance_rows"] == int(one_target)
    if one_target:
        assert provenance[0]["selected"] == "true" and provenance[0]["selection_rank"] == "1"
        assert provenance[0]["raw_smiles"] == "FIXTURE-TARGET"
        assert sum(int(r["retained_raw_count"]) for r in funnel if r["predicted_source"] == "true") == 1
    else:
        assert (tmp_path/"candidate_provenance.csv").read_text().startswith("dataset,oracle,method,fixture,")
    assert budget["generation_timing_missing_count"] == 32
    assert budget["observed_generation_unit_wall_seconds_sum"] is None
    assert budget["final_campaign_wall_seconds"] is None
    assert budget["pilot_filter_timing_receipt"] is None
    assert budget["generation_budget"]["parent_wall_limit_seconds"] == 900
    assert budget["pool_budget"]["max_candidates"] == 2000
    assert budget["summary_budget"]["k_max"] == 20
    assert budget["scientific_pass_claimed"] is False and budget["fixture"] is True
    assert report == export_diagnostics(tmp_path, evaluation, fixture=True)


def test_production_export_refuses_missing_receipts_before_creating_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="provenance_review"):
        export_results(fixture_evaluation(), tmp_path, dataset="bace", make_figures=False)
    assert not (tmp_path/"results").exists()


def test_export_diagnostic_manifest_does_not_change_existing_plot_csv_inventory(tmp_path):
    evaluation = diagnostic_fixture(tmp_path, True)
    report = export_results(evaluation, tmp_path, dataset="bace", fixture=True, make_figures=False)
    assert len(report["source_files"]) == 7
    assert set(report["diagnostic_files"]) == {"candidate_funnel.csv", "candidate_provenance.csv", "budget_and_timing.json"}
    assert report["diagnostic_status"] == "EXPORTED_RECORDED_DIAGNOSTICS"


def test_diagnostics_reject_source_mutation_and_do_not_write_outputs(tmp_path):
    evaluation = diagnostic_fixture(tmp_path, True)
    generation = next((tmp_path/"generation_units").glob("*.json"))
    row = json.loads(generation.read_text())
    row["parent_wall_seconds"] = 42
    generation.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Diagnostic source changed"):
        export_diagnostics(tmp_path, evaluation, fixture=True)
    assert not (tmp_path/"candidate_funnel.csv").exists()


def test_diagnostics_never_overwrite_conflicting_sidecar(tmp_path):
    evaluation = diagnostic_fixture(tmp_path)
    (tmp_path/"candidate_provenance.csv").write_text("unrelated existing user file\n")
    with pytest.raises(ValueError, match="Existing diagnostic sidecar differs"):
        export_diagnostics(tmp_path, evaluation, fixture=True)
    assert not (tmp_path/"candidate_funnel.csv").exists()
    assert (tmp_path/"candidate_provenance.csv").read_text() == "unrelated existing user file\n"


def test_diagnostics_refuse_fixture_as_production(tmp_path):
    evaluation = diagnostic_fixture(tmp_path)
    with pytest.raises(ValueError, match="matching completed"):
        export_diagnostics(tmp_path, evaluation)
