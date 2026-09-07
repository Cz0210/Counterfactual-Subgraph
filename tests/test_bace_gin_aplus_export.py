import pytest

from src.experiments import bace_gin_reach_v2 as driver
from src.eval.bace_frozen_gnn_contracts import stable_sha256


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "validate", lambda s: None)
    spec = dict(output_root=str(tmp_path), execution_commit="actual-science", main_matrix_write=False)
    sha = stable_sha256(spec)
    order = [f"r{i}" for i in range(20)]
    frozen = driver.seal(tmp_path / "selection_freeze.json", dict(
        state="CALIBRATION_SELECTOR_FROZEN", spec_sha256=sha, policy=driver.POLICY,
        test_loaded=False, controls=dict(old66_old_selector=order, old66_new_selector=order,
                                         adopted2607_new_selector=order)))
    metric = driver.seal(tmp_path / "metrics.json", dict(spec_sha256=sha,
        freeze_sha256=frozen["self_sha256"]))
    driver.seal(tmp_path / "contract.json", dict(spec_sha256=sha))
    driver.seal(tmp_path / "audit/final_audit.json", dict(
        state="SAVED_RECORD_AND_METRIC_CONSISTENCY_PASS", spec_sha256=sha,
        freeze_sha256=frozen["self_sha256"], metrics_sha256=metric["self_sha256"],
        audit_scope="SAVED_RECORDS_NOT_MODEL_REEXECUTION"))
    driver.seal(tmp_path / "adopted2607/test/terminal.json", dict(
        state="PARENT_EVALUATION_COMPLETE", parent_count=141, spec_sha256=sha))
    for name in ("ours_variant_comparison.csv", "figure3_k1_20.csv", "figure4_exact_ecdf.csv",
                 "parent_best_distances.csv"):
        driver.atomic_csv(tmp_path / "source_csv" / name, [dict(value="fixture")])
    return spec


def test_export_is_per_result_not_matrix_or_four_method_pass(tmp_path, monkeypatch):
    spec = fixture(tmp_path, monkeypatch)
    receipt = driver.export(spec)
    assert receipt["state"] == "VALIDATED_OURS_INCREMENTAL_COMPONENT"
    assert receipt["main_matrix_write"] is False
    assert receipt["global_registry_authority_created"] is False
    assert receipt["science_execution_commit"] == "actual-science"
    assert receipt["audit_scope"] == "SAVED_RECORDS_NOT_MODEL_REEXECUTION"
    assert len(receipt["artifacts"]) == 8
    assert driver.export(spec) == receipt


def test_export_refuses_missing_real_audit(tmp_path, monkeypatch):
    spec = fixture(tmp_path, monkeypatch)
    (tmp_path / "audit/final_audit.json").unlink()
    with pytest.raises(FileNotFoundError):
        driver.export(spec)
    assert not (tmp_path / "experiment_registry.json").exists()


def test_export_refuses_incomplete_test_terminal(tmp_path, monkeypatch):
    spec = fixture(tmp_path, monkeypatch)
    path = tmp_path / "adopted2607/test/terminal.json"
    path.unlink()
    driver.seal(path, dict(state="PARENT_EVALUATION_COMPLETE", parent_count=140,
                          spec_sha256=stable_sha256(spec)))
    with pytest.raises(ValueError, match="ACTUAL_TEST_AND_BOUND_AUDIT"):
        driver.export(spec)


def test_export_keeps_scope_and_refuses_changed_source_identity(tmp_path, monkeypatch):
    spec = fixture(tmp_path, monkeypatch)
    driver.export(spec)
    with pytest.raises(ValueError):
        driver.export(dict(spec, execution_commit="different"))
