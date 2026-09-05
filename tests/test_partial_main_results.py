from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.eval.four_by_four_main_results import MainResultsError
from src.eval.four_by_four_registry import SCHEMA_VERSION, sha256_file
from src.eval.partial_main_results import export_partial_results


def _csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp):
    authority = tmp / "authority"
    authority.mkdir()
    rows = []
    for dataset in ("AIDS", "Mutagenicity", "BACE", "TasteMolNet"):
        for method in ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"):
            root = tmp / "cells" / dataset / method
            root.mkdir(parents=True)
            zero = dataset == "AIDS" and method == "ComRecGC"
            common = {"dataset": dataset, "method": method}
            f3 = root / "figure3_coverage_vs_k.csv"
            f4 = root / "figure4_coverage_vs_threshold.csv"
            table = root / f"table2_{method.lower()}_k10.csv"
            _csv(f3, [{**common, "k": k, "coverage": "0.0" if zero else ".5", "cost": "" if zero else ".123"} for k in range(1, 21)])
            _csv(f4, [{**common, "threshold": str(t), "coverage": "0.0" if zero else ".5"} for t in ([.001, .002, .007] if dataset == "BACE" else [0, .05, .0535])])
            _csv(table, [{**common, "k": 10, "coverage": "0.0" if zero else ".5", "cost": "" if zero else ".123"}])
            audit = {**common, "audit_passed": True, "schema_version": "legacy", "audited_files": {p.name: {"sha256": sha256_file(p)} for p in (f3, f4, table)}}
            (root / "final_artifact_audit.json").write_text(json.dumps(audit))
            (root / "run_manifest.json").write_text(json.dumps(common))
            status = "FROZEN_PASS" if dataset in ("AIDS", "BACE") or method != "ComRecGC" and dataset == "Mutagenicity" or dataset == "TasteMolNet" and method == "Ours" else "PENDING"
            rows.append({**common, "status": status, "standardized_output_root": str(root)})
    matrix = {"schema_version": SCHEMA_VERSION, "audit_complete": True, "no_numeric_imputation": True, "cells": rows, "matrix_complete_cells": 12, "matrix_total_cells": 16, "all_cells_complete": False}
    (authority / "matrix_status.json").write_text(json.dumps(matrix))
    (tmp / "control").mkdir()
    state = tmp / "control/state.json"
    state.write_text(json.dumps({"latest_authority_root": str(authority)}))
    return state, matrix


def _run(tmp, state, **kwargs):
    return export_partial_results(matrix_authority_state=state, output_root=tmp / "output", project_root=tmp / "project", renderer=lambda *_: None, **kwargs)


def test_partial_keeps_real_grid_zero_and_missing_pending(tmp_path):
    state, _ = _fixture(tmp_path)
    before = sha256_file(state), sha256_file(tmp_path / "authority/matrix_status.json")
    manifest = _run(tmp_path, state)
    assert manifest["status"] == "PARTIAL" and manifest["rendered_cells"] == 8
    assert manifest["matrix_complete_cells"] == 12 and not manifest["final_16_of_16_claimed"]
    assert before == (sha256_file(state), sha256_file(tmp_path / "authority/matrix_status.json"))
    assert len(list(csv.DictReader((tmp_path / "output/bace/figure4_coverage_vs_threshold.csv").open()))) == 12
    table = list(csv.DictReader((tmp_path / "output/aids/table2_k10.csv").open()))
    zero = next(row for row in table if row["method"] == "ComRecGC")
    assert zero["coverage"] == "0.0" and zero["cost"] == ""
    assert sum(row["partial_state"] == "PENDING" for row in manifest["cell_status"]) == 4
    assert "PARTIAL" in (tmp_path / "output/aids/table2.tex").read_text()


def test_hash_mismatch_cell_pending_not_zero(tmp_path):
    state, matrix = _fixture(tmp_path)
    source = Path(matrix["cells"][0]["standardized_output_root"]) / "figure3_coverage_vs_k.csv"
    source.write_text(source.read_text().replace(".5", ".4"))
    manifest = _run(tmp_path, state)
    assert manifest["rendered_cells"] == 7
    assert manifest["source_cells"][0]["state"] == "PENDING_WITH_PROVENANCE_REASON"
    assert "SOURCE_HASH_CLOSURE_MISMATCH" in manifest["source_cells"][0]["reason"]
    assert "Ours" not in {r["method"] for r in csv.DictReader((tmp_path / "output/aids/table2_k10.csv").open())}


@pytest.mark.parametrize("target", ("authority/new", "project/paper/new", "cells/AIDS/Ours/new"))
def test_no_authority_science_or_paper_write(tmp_path, target):
    state, _ = _fixture(tmp_path)
    with pytest.raises(MainResultsError, match="OUTPUT_MUST_BE_FRESH"):
        export_partial_results(matrix_authority_state=state, output_root=tmp_path / target, project_root=tmp_path / "project", renderer=lambda *_: None)
    assert not (tmp_path / target).exists()


def test_no_overwrite_and_no_science_calls(tmp_path):
    state, _ = _fixture(tmp_path)
    _run(tmp_path, state)
    with pytest.raises(MainResultsError, match="OUTPUT_MUST_BE_FRESH"):
        _run(tmp_path, state)
    text = (Path(__file__).parents[1] / "src/eval/partial_main_results.py").read_text()
    assert "subprocess" not in text and "torch" not in text and "sqlite" not in text


def test_source_audit_failure_is_pending(tmp_path):
    state, matrix = _fixture(tmp_path)
    root = Path(matrix["cells"][0]["standardized_output_root"])
    audit = json.loads((root / "final_artifact_audit.json").read_text())
    audit["audit_passed"] = False
    (root / "final_artifact_audit.json").write_text(json.dumps(audit))
    manifest = _run(tmp_path, state)
    assert manifest["source_cells"][0]["reason"] == "SOURCE_FINAL_AUDIT_NOT_PASS"


def test_authority_changed_during_export_never_seals_manifest(tmp_path):
    state, _ = _fixture(tmp_path)
    def changing_renderer(*_):
        state.write_text(state.read_text() + " ")
    with pytest.raises(MainResultsError, match="AUTHORITY_CHANGED"):
        export_partial_results(matrix_authority_state=state, output_root=tmp_path / "output", project_root=tmp_path / "project", renderer=changing_renderer)
    assert not (tmp_path / "output/partial_manifest.json").exists()


def test_partial_plot_renderer_writes_only_staging(tmp_path):
    pytest.importorskip("matplotlib")
    state, _ = _fixture(tmp_path)
    result = export_partial_results(matrix_authority_state=state, output_root=tmp_path / "output", project_root=tmp_path / "project")
    assert result["rendered_cells"] == 8
    assert (tmp_path / "output/aids/figure3_PARTIAL.pdf").stat().st_size > 0
    assert (tmp_path / "output/bace/figure4_PARTIAL.png").stat().st_size > 0
