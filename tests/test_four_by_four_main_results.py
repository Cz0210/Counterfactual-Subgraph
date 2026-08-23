from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.autodl.export_four_by_four_main_results import main as cli_main
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from src.eval.four_by_four_export_tasks import (
    EXPORT_LOG_MARKER,
    EXPORT_TASK_ID,
    build_export_task_fragment,
)
from src.eval.four_by_four_main_results import (
    DATASET_ORDER,
    METHOD_ORDER,
    MainResultsError,
    export_main_results,
)
from src.eval.four_by_four_registry import SCHEMA_VERSION, sha256_file
from src.eval.three_dataset_main_results import export_three_dataset_results


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _cell(tmp_path: Path, dataset: str, method: str) -> tuple[Path, dict[str, object]]:
    slug = method.lower()
    root = tmp_path / "cells" / dataset.lower() / slug
    root.mkdir(parents=True)
    backend = "rf" if dataset in {"AIDS", "Mutagenicity"} else "gnn"
    identities = {
        "dataset": dataset,
        "method": method,
        "oracle_backend": backend,
        "rf_oracle_used": backend == "rf",
        "oracle_hash": _hash(f"{dataset}:oracle"),
        "dataset_hash": _hash(f"{dataset}:dataset"),
        "test_cohort_hash": _hash(f"{dataset}:split"),
        "molclr_checkpoint_hash": _hash(f"{dataset}:molclr"),
        "threshold_config_hash": _hash(f"{dataset}:threshold"),
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "selector_fitted_on_calibration": True,
        "selector_frozen_before_test": True,
    }
    figure3 = [
        {
            "method": method,
            "k": k,
            "coverage": f"{0.01 * k:.17g}",
            "cost": f"{0.02 + 0.001 * k:.17g}",
        }
        for k in range(1, 21)
    ]
    raw_thresholds = ("0.000000", "0.0123456789012345", "0.0535000000000000")
    figure4 = [
        {"method": method, "threshold": threshold, "coverage": str(index / 10)}
        for index, threshold in enumerate(raw_thresholds)
    ]
    table2 = [
        {
            "method": method,
            "k": 10,
            "coverage": "0.1",
            "cost": "0.03",
            "flip_rate": "0.2",
            "cf_drop": "0.4",
            "valid_rate": "0.9",
        }
    ]
    _write_csv(root / "figure3_coverage_vs_k.csv", figure3)
    _write_csv(root / "figure4_coverage_vs_threshold.csv", figure4)
    _write_csv(root / "prefix_metrics.csv", figure3)
    _write_csv(root / "parent_best_distances.csv", [{"parent_id": "p0", "distance": "0.03"}])
    if dataset == "TasteMolNet":
        _write_csv(
            root / "destination_distribution.csv",
            [
                {
                    "destination_label": 0,
                    "Sweet_to_Bitter_count": 1,
                    "Sweet_to_Bitter_rate": "0.5",
                    "Sweet_to_Tasteless_count": 1,
                    "Sweet_to_Tasteless_rate": "0.5",
                    "per_rule_destination_distribution": '{"0":1,"2":1}',
                },
                {
                    "destination_label": 2,
                    "Sweet_to_Bitter_count": 1,
                    "Sweet_to_Bitter_rate": "0.5",
                    "Sweet_to_Tasteless_count": 1,
                    "Sweet_to_Tasteless_rate": "0.5",
                    "per_rule_destination_distribution": '{"0":1,"2":1}',
                },
            ],
        )
    else:
        _write_csv(
            root / "destination_distribution.csv",
            [{"applicable": "N/A", "reason": "binary task"}],
        )
    for name in ("prefix_metrics.json", "summary.json", "run_manifest.json", "oracle_manifest.json", "evaluation_manifest.json"):
        _write_json(root / name, {**identities, "file_role": name})
    table_name = f"table2_{slug}_k10.csv"
    _write_csv(root / table_name, table2)
    closure_names = (
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        table_name,
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "destination_distribution.csv",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
    )
    _write_json(
        root / "final_artifact_audit.json",
        {
            **identities,
            "audit_passed": True,
            "file_sha256": {name: sha256_file(root / name) for name in closure_names},
        },
    )
    row = {
        "dataset": dataset,
        "method": method,
        "standardized_output_root": str(root),
        "status": "FROZEN_PASS",
        "oracle_backend": backend,
        "oracle_hash": identities["oracle_hash"],
        "dataset_hash": identities["dataset_hash"],
        "split_hash": identities["test_cohort_hash"],
        "distance_line": identities["distance_line"],
        "molclr_checkpoint_hash": identities["molclr_checkpoint_hash"],
        "cf_mode": identities["cf_mode"],
        "threshold_config_hash": identities["threshold_config_hash"],
    }
    return root, row


def _matrix(tmp_path: Path, *, incomplete: tuple[str, str] | None = None) -> Path:
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            _, row = _cell(tmp_path, dataset, method)
            if incomplete == (dataset, method):
                row["status"] = "BLOCKED_LICENSE"
                row["rerun_reason"] = "license review"
            rows.append(row)
    complete = sum(row["status"] in {"FROZEN_PASS", "ADOPTABLE_PASS"} for row in rows)
    path = tmp_path / "matrix" / "matrix_status.json"
    path.parent.mkdir(parents=True)
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "audit_complete": True,
            "matrix_complete_cells": complete,
            "matrix_total_cells": 16,
            "all_cells_complete": complete == 16,
            "no_numeric_imputation": True,
            "cells": rows,
        },
    )
    return path


def _fake_renderer(root: Path, figure3: object, figure4: object) -> None:
    del figure3, figure4
    for dataset in ("aids", "mutagenicity", "bace", "tastemolnet"):
        combined = root / dataset / "combined"
        for name in (
            "figure3_coverage_vs_k.png",
            "figure3_coverage_vs_k.pdf",
            "figure4_coverage_vs_threshold.png",
            "figure4_coverage_vs_threshold.pdf",
        ):
            (combined / name).write_bytes(b"verified-render-fixture\n")
    (root / "paper_figure3_four_datasets.pdf").write_bytes(b"panel3\n")
    (root / "paper_figure4_four_datasets.pdf").write_bytes(b"panel4\n")


def _three_dataset_matrix(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            _, row = _cell(tmp_path, dataset, method)
            if dataset == "TasteMolNet":
                row["status"] = "BLOCKED_LICENSE"
                row["rerun_reason"] = "BLOCKED_LICENSE_REVIEW"
                row["standardized_output_root"] = ""
            rows.append(row)
    path = tmp_path / "matrix-three" / "matrix_status.json"
    path.parent.mkdir(parents=True)
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "audit_complete": True,
            "matrix_complete_cells": 12,
            "matrix_total_cells": 16,
            "all_cells_complete": False,
            "no_numeric_imputation": True,
            "cells": rows,
        },
    )
    return path


def _fake_three_renderer(root: Path, figure3: object, figure4: object) -> None:
    del figure3, figure4
    for dataset in ("aids", "mutagenicity", "bace"):
        combined = root / dataset / "combined"
        for name in (
            "figure3_coverage_vs_k.png",
            "figure3_coverage_vs_k.pdf",
            "figure4_coverage_vs_threshold.png",
            "figure4_coverage_vs_threshold.pdf",
        ):
            (combined / name).write_bytes(b"verified-three-dataset-render\n")
    (root / "paper_figure3_three_datasets.pdf").write_bytes(b"panel3\n")
    (root / "paper_figure4_three_datasets.pdf").write_bytes(b"panel4\n")


def test_complete_export_preserves_raw_thresholds_and_taste_destinations(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "paper").mkdir(parents=True)
    sentinel = project / "paper" / "user-owned.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    matrix = _matrix(tmp_path)
    output = tmp_path / "outputs" / "final"
    result = export_main_results(
        matrix_status=matrix,
        output_root=output,
        project_root=project,
        renderer=_fake_renderer,
    )
    assert result.complete is True
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"
    with (output / "aids/combined/figure3_coverage_vs_k.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 80
    assert [rows[index * 20]["method"] for index in range(4)] == list(METHOD_ORDER)
    with (output / "aids/combined/figure4_coverage_vs_threshold.csv").open(newline="") as handle:
        thresholds = [row["threshold"] for row in csv.DictReader(handle) if row["method"] == "Ours"]
    assert thresholds == ["0.000000", "0.0123456789012345", "0.0535000000000000"]
    taste_text = (output / "tastemolnet/combined/destination_distribution.csv").read_text(encoding="utf-8")
    assert "Sweet_to_Bitter_count" in taste_text
    assert '"{""0"":1,""2"":1}"' in taste_text
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    audit = json.loads((output / "final_export_audit.json").read_text(encoding="utf-8"))
    assert audit["zero_fill_used"] is False
    assert audit["paper_directory_written"] is False


def test_three_dataset_staging_requires_exact_12_and_preserves_taste_block(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project-three"
    (project / "paper").mkdir(parents=True)
    sentinel = project / "paper" / "user-owned.tex"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    matrix = _three_dataset_matrix(tmp_path)
    output = tmp_path / "three_datasets_complete_v1"
    staging = tmp_path / "paper_staging" / "three_datasets_complete_v1"
    result = export_three_dataset_results(
        matrix_status=matrix,
        output_root=output,
        paper_staging_root=staging,
        project_root=project,
        renderer=_fake_three_renderer,
    )
    assert result.complete is True
    assert result.matrix_complete_cells == 12
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"
    assert (output / "paper_figure3_three_datasets.pdf").is_file()
    assert (output / "paper_figure4_three_datasets.pdf").is_file()
    assert (output / "paper_table2_three_datasets.tex").is_file()
    assert not (output / "tastemolnet").exists()
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    for relative in (
        "paper_figure3_three_datasets.pdf",
        "paper_figure4_three_datasets.pdf",
        "paper_table2_three_datasets.tex",
        "three_dataset_export_manifest.json",
        "three_dataset_export_audit.json",
        "PASS",
    ):
        assert sha256_file(output / relative) == sha256_file(staging / relative)
    manifest = json.loads(
        (output / "three_dataset_export_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["matrix_complete_cells"] == 12
    assert manifest["matrix_total_cells"] == 16
    assert manifest["final_four_dataset_export"] is False
    assert manifest["paper_status"] == "PAPER_FROZEN_PARTIAL"
    assert {row["status"] for row in manifest["taste_cells"]} == {
        "BLOCKED_LICENSE"
    }


@pytest.mark.parametrize("failure", ("eleven_pass", "taste_not_license"))
def test_three_dataset_staging_fails_closed_before_writing_numeric_outputs(
    tmp_path: Path, failure: str
) -> None:
    project = tmp_path / "project-blocked"
    (project / "paper").mkdir(parents=True)
    matrix = _three_dataset_matrix(tmp_path)
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    if failure == "eleven_pass":
        target = next(
            row
            for row in payload["cells"]
            if row["dataset"] == "BACE" and row["method"] == "ComRecGC"
        )
        target["status"] = "INCOMPLETE"
        payload["matrix_complete_cells"] = 11
    else:
        target = next(
            row
            for row in payload["cells"]
            if row["dataset"] == "TasteMolNet" and row["method"] == "Ours"
        )
        target["status"] = "MISSING"
        target["rerun_reason"] = ""
    _write_json(matrix, payload)
    output = tmp_path / f"blocked-{failure}"
    with pytest.raises(MainResultsError):
        export_three_dataset_results(
            matrix_status=matrix,
            output_root=output,
            project_root=project,
            renderer=lambda *_: pytest.fail("renderer must not be called"),
        )
    assert not output.exists()


def test_three_dataset_staging_rejects_paper_tree_and_tampered_cell(tmp_path: Path) -> None:
    project = tmp_path / "project-tamper"
    (project / "paper").mkdir(parents=True)
    matrix = _three_dataset_matrix(tmp_path)
    with pytest.raises(MainResultsError, match="may not write into paper"):
        export_three_dataset_results(
            matrix_status=matrix,
            output_root=project / "paper" / "three",
            project_root=project,
            renderer=_fake_three_renderer,
        )
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    root = Path(
        next(
            row["standardized_output_root"]
            for row in payload["cells"]
            if row["dataset"] == "AIDS" and row["method"] == "Ours"
        )
    )
    with (root / "figure4_coverage_vs_threshold.csv").open("a", encoding="utf-8") as handle:
        handle.write("Ours,0.06,0.99\n")
    output = tmp_path / "tampered-three"
    with pytest.raises(MainResultsError):
        export_three_dataset_results(
            matrix_status=matrix,
            output_root=output,
            project_root=project,
            renderer=lambda *_: pytest.fail("renderer must not be called"),
        )
    assert not output.exists()


def test_incomplete_matrix_writes_only_partial_audit_and_never_zero_fills(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "paper").mkdir(parents=True)
    matrix = _matrix(tmp_path, incomplete=("TasteMolNet", "ComRecGC"))
    output = tmp_path / "partial"
    result = export_main_results(
        matrix_status=matrix,
        output_root=output,
        project_root=project,
        renderer=lambda *_: pytest.fail("renderer must not be called"),
    )
    assert result.complete is False
    assert sorted(path.name for path in output.iterdir()) == [
        "BLOCKED_INCOMPLETE_MATRIX",
        "partial_staging_audit.json",
    ]
    audit = json.loads((output / "partial_staging_audit.json").read_text(encoding="utf-8"))
    assert audit["numeric_outputs_generated"] is False
    assert audit["zero_fill_used"] is False
    assert not list(output.rglob("*.png"))
    assert not list(output.rglob("table2.csv"))
    assert cli_main(
        [
            "export",
            "--matrix-status",
            str(matrix),
            "--output-root",
            str(tmp_path / "partial-cli"),
            "--project-root",
            str(project),
            "--require-complete",
        ]
    ) == 3


def test_complete_flag_with_tampered_cell_closure_still_emits_no_final_outputs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "paper").mkdir(parents=True)
    matrix = _matrix(tmp_path)
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    root = Path(payload["cells"][0]["standardized_output_root"])
    with (root / "figure3_coverage_vs_k.csv").open("a", encoding="utf-8") as handle:
        handle.write("Ours,21,0.21,0.04\n")
    output = tmp_path / "tampered"
    result = export_main_results(
        matrix_status=matrix,
        output_root=output,
        project_root=project,
        renderer=lambda *_: pytest.fail("renderer must not be called"),
    )
    assert result.complete is False
    assert not list(output.rglob("*.pdf"))
    assert "closure" in (output / "partial_staging_audit.json").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    ("row_field", "manifest_field"),
    (
        ("oracle_hash", "oracle_hash"),
        ("split_hash", "test_cohort_hash"),
        ("threshold_config_hash", "threshold_config_hash"),
    ),
)
def test_cross_method_oracle_split_and_threshold_mismatch_fail_closed(
    tmp_path: Path, row_field: str, manifest_field: str
) -> None:
    project = tmp_path / "project"
    (project / "paper").mkdir(parents=True)
    matrix = _matrix(tmp_path)
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    target = next(
        row
        for row in payload["cells"]
        if row["dataset"] == "BACE" and row["method"] == "ComRecGC"
    )
    root = Path(target["standardized_output_root"])
    replacement = _hash(f"replacement:{row_field}")
    target[row_field] = replacement
    for name in (
        "prefix_metrics.json",
        "summary.json",
        "run_manifest.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "final_artifact_audit.json",
    ):
        path = root / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data[manifest_field] = replacement
        _write_json(path, data)
    audit_path = root / "final_artifact_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for name in (
        "prefix_metrics.json",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
    ):
        audit["file_sha256"][name] = sha256_file(root / name)
    _write_json(audit_path, audit)
    _write_json(matrix, payload)
    output = tmp_path / f"mismatch-{row_field}"
    result = export_main_results(
        matrix_status=matrix,
        output_root=output,
        project_root=project,
        renderer=lambda *_: pytest.fail("renderer must not be called"),
    )
    assert result.complete is False
    audit_text = (output / "partial_staging_audit.json").read_text(encoding="utf-8")
    assert row_field in audit_text
    assert not list(output.rglob("*.pdf"))


def test_clear_is_rejected_and_paper_output_root_is_rejected(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path)
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    payload["cells"][-1]["method"] = "CLEAR"
    _write_json(matrix, payload)
    project = tmp_path / "project"
    (project / "paper").mkdir(parents=True)
    with pytest.raises(MainResultsError, match="CLEAR is not ComRecGC"):
        export_main_results(
            matrix_status=matrix,
            output_root=tmp_path / "clear",
            project_root=project,
            renderer=_fake_renderer,
        )
    clean = _matrix(tmp_path / "clean")
    with pytest.raises(MainResultsError, match="may not write into paper"):
        export_main_results(
            matrix_status=clean,
            output_root=project / "paper" / "generated",
            project_root=project,
            renderer=_fake_renderer,
        )


def _dependency_contract(tmp_path: Path) -> tuple[Path, list[str]]:
    cells: dict[str, str] = {}
    task_ids: list[str] = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            task_id = f"cell_{dataset.lower()}_{method.lower()}_pass"
            cells[f"{dataset}/{method}"] = task_id
            task_ids.append(task_id)
    path = tmp_path / "dependencies.json"
    _write_json(
        path,
        {
            "matrix_task_id": "final_matrix_audit",
            "matrix_status": "{dep_final_matrix_audit_output}/matrix_status.json",
            "cells": cells,
        },
    )
    return path, task_ids


def test_generic_export_task_has_exact_16_cell_and_matrix_dependencies(tmp_path: Path) -> None:
    contract, task_ids = _dependency_contract(tmp_path)
    fragment = build_export_task_fragment(
        controller_id="four-by-four-test",
        dependency_contract=contract,
        output_root=Path("/persistent") / _hash(str(tmp_path))[:8] / "final",
    )
    task = fragment["tasks"][0]
    assert task["id"] == EXPORT_TASK_ID
    assert task["depends_on"] == [*task_ids, "final_matrix_audit"]
    assert task["required_log_marker"] == EXPORT_LOG_MARKER
    assert task["resource"] == "cpu"
    assert task["manifest_only"] is True
    assert "--require-complete" in task["command"]

    stub_tasks = [
        {
            "id": task_id,
            "dataset": "adopted-cell",
            "stage": "ADOPTED_CELL_TERMINAL",
            "depends_on": [],
            "resource": "cpu",
            "command": None,
            "blocked_reason": "BLOCKED_TEST_FIXTURE",
            "data_splits": [],
            "manifest_only": True,
        }
        for task_id in [*task_ids, "final_matrix_audit"]
    ]
    manifest_path = tmp_path / "controller.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "controller_id": "four-by-four-test",
            "paper_frozen": True,
            "runtime": {
                "max_gpus": 4,
                "stable_idle_seconds": 60,
                "sample_interval_seconds": 5,
                "poll_seconds": 60,
                "max_transient_retries": 1,
            },
            "resource_gates": {},
            "tasks": [*stub_tasks, task],
        },
    )
    loaded = load_controller_manifest(manifest_path)
    assert loaded.by_id[EXPORT_TASK_ID].depends_on == tuple([*task_ids, "final_matrix_audit"])


def test_task_fragment_cli_rejects_missing_cell_and_writes_fresh_fragment(tmp_path: Path) -> None:
    contract, _ = _dependency_contract(tmp_path)
    fragment = tmp_path / "fragment.json"
    assert cli_main(
        [
            "task-fragment",
            "--controller-id",
            "four-by-four-test",
            "--dependency-contract",
            str(contract),
            "--output-root",
            str(Path("/persistent") / _hash(str(tmp_path))[:8]),
            "--fragment-output",
            str(fragment),
        ]
    ) == 0
    assert json.loads(fragment.read_text(encoding="utf-8"))["tasks"][0]["id"] == EXPORT_TASK_ID
    broken = json.loads(contract.read_text(encoding="utf-8"))
    broken["cells"].pop("TasteMolNet/ComRecGC")
    broken_path = tmp_path / "broken.json"
    _write_json(broken_path, broken)
    assert cli_main(
        [
            "task-fragment",
            "--controller-id",
            "four-by-four-test",
            "--dependency-contract",
            str(broken_path),
            "--output-root",
            str(Path("/persistent") / "broken"),
            "--fragment-output",
            str(tmp_path / "broken-fragment.json"),
        ]
    ) == 2
