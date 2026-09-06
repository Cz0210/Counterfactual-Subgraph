"""Pure summary replay tests; no oracle, chemistry, OT, or dataset execution."""
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

import pytest

from src.eval import mut_registry_k10_projection as projection
from src.eval import non_taste_matrix_append as append
from src.eval import user_approved_frozen_v4 as approved
from src.eval.four_by_four_registry import audit_explicit_candidate, CellStatus


def put_json(path, value):
    path.write_bytes(projection._json_bytes(value))


def put_rows(path, rows):
    path.write_bytes(projection._csv_bytes(rows))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source, reference = tmp_path / "source", tmp_path / "reference"
    standard = source / "standardized"
    standard.mkdir(parents=True)
    reference.mkdir()
    ids = list(map(str, range(217)))
    grid = [i * .0535 / 600 for i in range(601)]
    strings = list(map(str, grid))
    monkeypatch.setattr(projection, "REFERENCE_GRID_SHA", projection._grid_hash(strings))
    monkeypatch.setattr(approved, "validate_adopted_cell", lambda root: (
        True, (), {"dataset": "Mutagenicity", "method": "Ours"}))
    monkeypatch.setattr(append, "_writer_audit", lambda *a, **kw: {
        "procfs_verified": True, "writers": [], "writable_fd_count": 0})
    parents, prefixes = [], []
    for k in range(1, 21):
        ds = [] if k < 4 else ([.06] * 217 if k < 20 else [.03] * 217)
        for i in range(217):
            d = ds[i] if ds else None
            parents.append({"k": k, "requested_k": k, "parent_id": str(i),
                "best_distance": d, "strict_recourse_available": d is not None,
                "theta_star_covered": d is not None and d <= .05})
        covered = sum(d <= .05 for d in ds)
        prefixes.append({"method": "COMRECGC-Adapted-DeterministicChemRepair", "k": k,
            "num_parents": 217, "num_any_strict_flip_parents": len(ds),
            "num_close_cf_covered": covered, "close_cf_coverage": covered / 217,
            "conditional_median_cost": statistics.median(ds) if ds else None,
            "conditional_mean_cost": statistics.mean(ds) if ds else None})
    put_rows(standard / "parent_best_distances.csv", parents)
    put_rows(standard / "figure3_coverage_vs_k.csv", prefixes)
    put_rows(standard / "prefix_metrics.csv", prefixes)
    put_json(standard / "prefix_metrics.json", {"rows": prefixes})
    put_rows(standard / "table2_comrecgc_k10.csv", [{"dataset": "Mutagenicity", "method": "ComRecGC",
        "k": 10, "coverage": 0, "conditional_median_cost": .06}])
    put_rows(standard / "figure4_coverage_vs_threshold.csv", [{"method": "ComRecGC", "k": 20,
        "threshold": t, "num_close_cf_covered": 217 if t >= .03 else 0,
        "close_cf_coverage": 1.0 if t >= .03 else 0.0} for t in grid])
    put_rows(reference / "figure4_coverage_vs_threshold.csv", [
        {"method": "Ours", "k": 10, "threshold": t, "coverage": 0} for t in strings])
    put_json(reference / "evaluation_manifest.json", {"figure4_k": 10, "figure4_points": 601,
        "threshold_config_hash": projection.REFERENCE_GRID_SHA})
    for name in ("run_manifest.json", "registry_exception.json"):
        put_json(reference / name, {})
    upstream = source / "original_threshold_contract.json"
    contract = {"schema_version": "four_by_four_frozen_threshold_contract_v1", "dataset": "Mutagenicity",
        "status": "PASS", "cf_mode": "strict_flip", "distance_line": "MolCLR-Node-Wasserstein",
        "theta_star": .05, "cost_cap": .0535, "shared_across_methods": True,
        "test_used_for_selection": False, "selection_used_test": False,
        "threshold_fitted_on_test": False, "threshold_source_split": "existing_frozen_protocol",
        "thresholds": grid, "threshold_config_hash": "legacy label preserved"}
    put_json(upstream, contract)
    contract.update(source_contract=str(upstream), source_contract_sha256=projection._sha(upstream))
    frozen = source / "threshold_contract.json"
    put_json(frozen, contract)
    native = {"project_commit": projection.SCIENCE_COMMIT, "source_label": 1, "target_label": 0,
        "thresholds_path": str(frozen),
        "thresholds_sha256": projection._sha(frozen), "threshold_grid": grid,
        "parent_count": 217, "parent_ids_sha256": hashlib.sha256(json.dumps(ids,
            sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest(),
        "teacher_path": str(source / "teacher.pkl"), "teacher_sha256": "1" * 64,
        "dataset_csv_sha256": "2" * 64, "molclr_checkpoint_sha256": "3" * 64}
    put_json(standard / "run_manifest.json", native)
    for name in ("summary.json", "final_artifact_audit.json"):
        put_json(standard / name, {"status": "PASS"})
    freeze = {"files": {p.name: {"sha256": projection._sha(p), "bytes": p.stat().st_size}
                         for p in standard.iterdir()}}
    put_json(standard / "freeze_manifest.json", freeze)
    outer = {"project_commit": projection.SCIENCE_COMMIT, "dataset": "mutagenicity", "method": "COMRECGC",
        "oracle_backend": "rf", "classifier_family": "random_forest", "rf_oracle_used": True,
        "generation_adopted": True, "independent_scientific_adoption_authorized": True, "status": "PASS",
        "standardized_run_manifest_sha256": projection._sha(standard / "run_manifest.json"),
        "freeze_manifest_sha256": projection._sha(standard / "freeze_manifest.json")}
    put_json(source / "run_manifest.json", outer)
    put_json(source / "final_gate.json", outer)
    put_json(source / "_RUN_COMPLETE.json", {**outer, "run_complete": True})
    put_json(source / "FAILED.json", {"historical": "preserved"})
    repair = source / "control_receipt.json"
    put_json(repair, {"control": "original startup repair receipt"})
    terminal = {"terminal_kind": "MUT_FAST_ACCURATE_STANDARDIZATION_FINAL", "root": str(source),
        "startup_failure_supersession": {"status": "PASS"},
        "run_manifest_sha256": projection._sha(source / "run_manifest.json"),
        "final_gate_sha256": projection._sha(source / "final_gate.json"),
        "run_complete_sha256": projection._sha(source / "_RUN_COMPLETE.json")}
    monkeypatch.setattr(append, "_validate_mut_fast_accurate_terminal", lambda *a, **kw: terminal)
    return {"source": source, "reference": reference, "repair": repair,
            "terminal": terminal, "output": tmp_path / "projection"}


def create(bundle):
    return projection.create_registry_projection(terminal_root=bundle["source"],
        reference_standardized_root=bundle["reference"], startup_repair_receipt=bundle["repair"],
        output_root=bundle["output"])


def validate(bundle):
    return projection.validate_registry_projection(bundle["output"], terminal=bundle["terminal"],
        reference={"dataset": "Mutagenicity", "method": "Ours",
            "standardized_output_root": str(bundle["reference"]),
            "threshold_config_hash": projection.REFERENCE_GRID_SHA},
        startup_repair_receipt=bundle["repair"])


def test_real_ordinary_registry_pass_with_k10_not_native_k20(bundle):
    before = {str(p): p.read_bytes() for p in bundle["source"].rglob("*") if p.is_file()}
    result = create(bundle)
    assert result["ordinary_registry_status"] == "FROZEN_PASS"
    assert result["undefined_conditional_cost_prefixes"] == [1, 2, 3]
    assert all(result[key] is value for key, value in projection.FLAGS.items())
    root = bundle["output"] / "standardized"
    for name in projection.COPIED:
        assert (root / name).read_bytes() == (bundle["source"] / "standardized" / name).read_bytes()
    assert {int(row["k"]) for row in projection._rows(root / "figure4_coverage_vs_threshold.csv")} == {10}
    assert projection._rows(root / "figure4_coverage_vs_threshold.csv")[-1]["coverage"] == "0.0"
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    assert validate(bundle)["status"] == "PASS"


@pytest.mark.parametrize("name", ["parent_best_distances.csv", "figure3_coverage_vs_k.csv", "table2_comrecgc_k10.csv"])
def test_native_source_byte_change_rejected(bundle, name):
    create(bundle)
    path = bundle["source"] / "standardized" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="sealed native bytes changed"):
        validate(bundle)


@pytest.mark.parametrize("change", ["k", "coverage", "threshold"])
def test_projected_scientific_change_rejected(bundle, change):
    create(bundle)
    path = bundle["output"] / "standardized" / "figure4_coverage_vs_threshold.csv"
    rows = projection._rows(path)
    rows[0][change] = {"k": "20", "coverage": "0.5", "threshold": "0.0001"}[change]
    put_rows(path, rows)
    with pytest.raises(ValueError, match="non-deterministic projected artifact"):
        validate(bundle)


def test_unknown_additional_projection_file_rejected(bundle):
    create(bundle)
    (bundle["output"] / "standardized" / "FAILED.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected or missing projected files"):
        validate(bundle)


def test_exact_reference_grid_not_arbitrary_grid(bundle):
    p = bundle["reference"] / "figure4_coverage_vs_threshold.csv"
    rows = projection._rows(p)
    rows[1]["threshold"] = "0.00009"
    put_rows(p, rows)
    with pytest.raises(ValueError, match="exact preregistered reference threshold strings"):
        create(bundle)


def test_positive_strict_flip_count_cannot_have_undefined_cost(bundle):
    inputs, _ = projection._source_inputs(bundle["source"], bundle["reference"])
    p = bundle["source"] / "standardized" / "figure3_coverage_vs_k.csv"
    rows = projection._rows(p)
    rows[0]["num_any_strict_flip_parents"] = "1"
    put_rows(p, rows)
    with pytest.raises(ValueError, match="prefix/parent aggregate mismatch"):
        projection._aggregate(bundle["source"], inputs)


def test_per_parent_threshold_crossing_is_rejected_even_with_nominal_grid(bundle):
    inputs, _ = projection._source_inputs(bundle["source"], bundle["reference"])
    i = 400
    crossing = math.nextafter(float(inputs["grid"][i]), math.inf)
    inputs["native_grid"][i] = crossing
    standard = bundle["source"] / "standardized"
    parents = projection._rows(standard / "parent_best_distances.csv")
    changed = next(row for row in parents if row["k"] == "10")
    changed.update(best_distance=str(crossing), theta_star_covered="True")
    put_rows(standard / "parent_best_distances.csv", parents)
    rows = projection._rows(standard / "figure3_coverage_vs_k.csv")
    rows[9].update(num_close_cf_covered="1", close_cf_coverage=str(1 / 217))
    put_rows(standard / "figure3_coverage_vs_k.csv", rows)
    table = projection._rows(standard / "table2_comrecgc_k10.csv")
    table[0]["coverage"] = str(1 / 217)
    put_rows(standard / "table2_comrecgc_k10.csv", table)
    old = projection._rows(standard / "figure4_coverage_vs_threshold.csv")
    old[i]["threshold"] = str(crossing)
    put_rows(standard / "figure4_coverage_vs_threshold.csv", old)
    with pytest.raises(ValueError, match="K10 threshold crossing"):
        projection._aggregate(bundle["source"], inputs)


def test_untyped_native_missing_cost_still_rejected(bundle):
    audit = audit_explicit_candidate(bundle["source"] / "standardized", dataset="Mutagenicity", method="ComRecGC")
    assert audit.status != CellStatus.FROZEN_PASS
    assert "FIGURE3_INVALID:ValueError" in audit.reason_codes


def test_driver_identity_and_full_terminal_cannot_be_faked(bundle):
    create(bundle)
    receipt_path = bundle["output"] / "projection.json"
    receipt = projection._json(receipt_path)
    receipt["publication_driver_commit"] = projection.SCIENCE_COMMIT
    put_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="actual publication driver/science identity"):
        validate(bundle)


def test_new_consumer_head_does_not_invalidate_preserved_producer(bundle, monkeypatch):
    create(bundle)
    producer = projection._driver_identity()
    monkeypatch.setattr(projection, "_driver_identity", lambda: {
        **producer, "publication_driver_commit": "f" * 40})
    assert validate(bundle)["status"] == "PASS"


def test_full_original_terminal_failure_prevents_creation(bundle, monkeypatch):
    def rejected(*a, **kw):
        raise ValueError("original scientific terminal invalid")
    monkeypatch.setattr(append, "_validate_mut_fast_accurate_terminal", rejected)
    with pytest.raises(ValueError, match="original scientific terminal invalid"):
        create(bundle)
    assert not bundle["output"].exists()
