from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from src.eval import three_dataset_release_supervisor as release_module
from src.eval.three_dataset_release_supervisor import (
    EXPECTED_CELLS,
    FROZEN_V4_CELLS,
    POLL_INTERVAL_SECONDS,
    ReleaseBlocked,
    ReleaseSpecError,
    ReleaseSupervisor,
    build_release_spec,
    load_cell_catalog,
    load_release_spec,
    probe_cells,
    write_release_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = PROJECT_ROOT / "configs/autodl/three_dataset_release_cells_v1.template.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _make_inputs(tmp_path: Path) -> dict[str, Any]:
    runtime = tmp_path / "runtime"
    matrix_root = runtime / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1"
    runtime.mkdir(parents=True)
    adoption = runtime / "adoption_manifest.json"
    expectations = runtime / "expectations.json"
    taste = runtime / "taste_license_gate.json"
    owner_manifest = runtime / "controller_manifest.json"
    _write_json(adoption, {"status": "USER_APPROVED_FROZEN_V4"})
    _write_json(expectations, {})
    _write_json(
        taste,
        {
            "status": "BLOCKED_LICENSE_REVIEW",
            "passed": False,
            "reason": "exact-data license evidence required",
        },
    )

    roots: dict[tuple[str, str], Path] = {}
    tasks: list[dict[str, str]] = []
    cells: list[dict[str, Any]] = []
    for dataset, method in EXPECTED_CELLS:
        root = matrix_root / "fixture_cells" / _slug(dataset) / _slug(method) / "attempt-0"
        roots[(dataset, method)] = root
        cell: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "root_kind": "fixed",
            "standardized_root": str(root),
        }
        if (dataset, method) in FROZEN_V4_CELLS:
            cell["owner_kind"] = "user_approved_frozen_v4"
        else:
            task_id = f"standardize_{_slug(dataset)}_{_slug(method)}"
            cell.update(
                {
                    "owner_task_id": task_id,
                    "owner_manifest_hint": str(owner_manifest),
                }
            )
            if (dataset, method) == ("BACE", "GlobalGCE"):
                cell["required_owner_task_id"] = task_id
            tasks.append(
                {
                    "id": task_id,
                    "expected_output": str(root).replace("attempt-0", "attempt-{attempt}"),
                }
            )
        cells.append(cell)
    _write_json(owner_manifest, {"controller_id": "fixture", "tasks": tasks})
    catalog = runtime / "catalog.json"
    _write_json(
        catalog,
        {
            "schema_version": "three_dataset_release_cell_catalog_v1",
            "matrix_root": str(matrix_root),
            "adoption_manifest": str(adoption),
            "cells": cells,
        },
    )
    return {
        "runtime": runtime,
        "matrix_root": matrix_root,
        "adoption": adoption,
        "expectations": expectations,
        "taste": taste,
        "owner_manifest": owner_manifest,
        "catalog": catalog,
        "roots": roots,
    }


def _build_spec(
    tmp_path: Path,
    *,
    nested_cells: set[tuple[str, str]] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    inputs = _make_inputs(tmp_path)
    if nested_cells:
        catalog = json.loads(inputs["catalog"].read_text(encoding="utf-8"))
        for cell in catalog["cells"]:
            if (cell["dataset"], cell["method"]) in nested_cells:
                cell["root_layout"] = "nested_standardized"
        _write_json(inputs["catalog"], catalog)
    runtime = inputs["runtime"]
    matrix_root = inputs["matrix_root"]
    spec = build_release_spec(
        catalog_path=inputs["catalog"],
        controller_id="three_dataset_release_fixture",
        project_root=PROJECT_ROOT,
        runtime_root=runtime,
        python=Path(sys.executable),
        state_root=runtime / "control/three_dataset_release_fixture",
        registry_root=matrix_root / "release_registry_v1",
        output_root=matrix_root / "three_datasets_complete_v1",
        paper_staging_root=runtime / "outputs/autodl/paper_staging/three_datasets_complete_v1",
        expectations_json=inputs["expectations"],
        taste_license_gate_json=inputs["taste"],
        require_runnable=True,
    )
    spec_path = runtime / "specs/release.json"
    write_release_spec(spec_path, spec)
    return spec_path, spec, inputs


def _write_probe_closure(root: Path, method: str) -> None:
    required = (
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "destination_distribution.csv",
        "summary.json",
        "run_manifest.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "final_artifact_audit.json",
        f"table2_{_slug(method)}_k10.csv",
        "_FINALIZED.json",
        "PASS",
    )
    root.mkdir(parents=True)
    for name in required:
        value = "PASS\n" if name == "PASS" else "{}\n"
        (root / name).write_text(value, encoding="utf-8")


def _fake_process_identity(pid: int = 987_654_321) -> dict[str, Any]:
    return {
        "pid": pid,
        "boot_id": "fixture-boot",
        "start_time_ticks": 1,
        "cmdline_sha256": "0" * 64,
    }


def test_catalog_keeps_only_unsettled_routes_as_fail_closed_placeholders() -> None:
    catalog = load_cell_catalog(TEMPLATE)
    cells = {(cell["dataset"], cell["method"]): cell for cell in catalog["cells"]}

    assert len(cells) == 12
    assert {
        key for key, cell in cells.items() if cell["root_kind"] == "placeholder"
    } == {
        ("AIDS", "ComRecGC"),
        ("Mutagenicity", "ComRecGC"),
        ("BACE", "GlobalGCE"),
    }
    globalgce = cells[("BACE", "GlobalGCE")]
    assert globalgce["owner_task_id"] == "bace_globalgce_standardized"
    assert globalgce["required_owner_task_id"] == "bace_globalgce_standardized"
    assert (
        globalgce["upstream_scientific_final_role"]
        == "scientific_final_not_a_standardized_cell"
    )
    assert "/final/attempt-0" in globalgce["upstream_scientific_final_hint"]
    assert "owner_manifest_hint" not in globalgce
    assert "fresh external standardization manifest" in globalgce["owner_manifest_requirement"]


def test_builder_binds_external_owner_manifest_task_and_output(tmp_path: Path) -> None:
    spec_path, spec, inputs = _build_spec(tmp_path)
    assert spec["runnable"] is True
    assert spec["unresolved_bindings"] == []
    cell = next(
        item
        for item in spec["cells"]
        if item["dataset"] == "BACE" and item["method"] == "GlobalGCE"
    )
    assert cell["owner"]["manifest"]["status"] == "FIXED"
    assert len(cell["owner"]["manifest"]["sha256"]) == 64
    assert cell["owner"]["task_id"] == "standardize_bace_globalgce"
    assert cell["owner"]["binding"]["kind"] == "task_output_template"
    assert cell["standardized_root"] == str(inputs["roots"][("BACE", "GlobalGCE")])
    _, _, loaded = load_release_spec(spec_path)
    assert loaded == spec


def test_nested_controller_output_is_not_mistaken_for_standardized_cell(
    tmp_path: Path,
) -> None:
    key = ("AIDS", "ComRecGC")
    _spec_path, spec, inputs = _build_spec(tmp_path, nested_cells={key})
    binding = next(
        cell
        for cell in spec["cells"]
        if (cell["dataset"], cell["method"]) == key
    )
    owner_root = inputs["roots"][key]
    assert binding["owner_output_root"] == str(owner_root)
    assert binding["standardized_root"] == str(owner_root / "standardized")

    _write_probe_closure(owner_root / "standardized", "ComRecGC")
    (owner_root / "standardized/PASS").unlink()
    probe = next(item for item in probe_cells(spec) if (item.dataset, item.method) == key)
    assert probe.state == "WAITING_CLOSURE_FILES"
    assert "owner:PASS" in probe.missing_files
    for name in ("run_manifest.json", "final_gate.json", "_RUN_COMPLETE.json"):
        _write_json(owner_root / name, {"status": "PASS"})
    (owner_root / "PASS").write_text("PASS\n", encoding="utf-8")
    probe = next(item for item in probe_cells(spec) if (item.dataset, item.method) == key)
    assert probe.state == "READY_FOR_FULL_AUDIT"


def test_seven_of_twelve_waits_without_numeric_or_registry_outputs(tmp_path: Path) -> None:
    spec_path, spec, inputs = _build_spec(tmp_path)
    ready = set(FROZEN_V4_CELLS) | {("BACE", "Ours")}
    for dataset, method in ready:
        _write_probe_closure(inputs["roots"][(dataset, method)], method)

    probes = probe_cells(spec)
    assert sum(probe.state == "READY_FOR_FULL_AUDIT" for probe in probes) == 7
    with ReleaseSupervisor(
        spec_path, process_identity=_fake_process_identity()
    ) as supervisor:
        with pytest.raises(ReleaseBlocked, match="control lock"):
            with ReleaseSupervisor(
                spec_path, process_identity=_fake_process_identity(987_654_322)
            ):
                pass
        result = supervisor.tick()

    assert result.state == "WAITING_DEPENDENCY"
    assert result.complete_cells == 7
    state_root = Path(spec["paths"]["state_root"])
    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    heartbeat = json.loads((state_root / "heartbeat.json").read_text(encoding="utf-8"))
    assert state["state"] == "WAITING_DEPENDENCY"
    assert state["reason"] == "No matrix or numeric release output is created before 12/16"
    assert heartbeat["state"] == "WAITING_DEPENDENCY"
    assert heartbeat["sequence"] == 1
    assert POLL_INTERVAL_SECONDS == 60
    assert not Path(spec["paths"]["registry_root"]).exists()
    assert not Path(spec["paths"]["output_root"]).exists()
    assert not Path(spec["paths"]["paper_staging_root"]).exists()

    with ReleaseSupervisor(
        spec_path, process_identity=_fake_process_identity(987_654_323)
    ) as restarted:
        restarted.tick()
    controller = json.loads((state_root / "controller.json").read_text(encoding="utf-8"))
    heartbeat = json.loads((state_root / "heartbeat.json").read_text(encoding="utf-8"))
    assert controller["restart_count"] == 1
    assert heartbeat["sequence"] == 2


def test_owner_manifest_drift_blocks_even_before_cells_are_ready(tmp_path: Path) -> None:
    spec_path, spec, inputs = _build_spec(tmp_path)
    manifest = json.loads(inputs["owner_manifest"].read_text(encoding="utf-8"))
    manifest["unexpected_mutation"] = True
    _write_json(inputs["owner_manifest"], manifest)

    with ReleaseSupervisor(
        spec_path, process_identity=_fake_process_identity()
    ) as supervisor:
        result = supervisor.tick()

    assert result.state == "BLOCKED_BOUND_INPUT_DRIFT"
    assert "identity drift" in result.reason
    assert not Path(spec["paths"]["registry_root"]).exists()
    assert not Path(spec["paths"]["output_root"]).exists()
    assert not Path(spec["paths"]["paper_staging_root"]).exists()


def test_builder_does_not_activate_root_when_owner_task_does_not_bind_it(
    tmp_path: Path,
) -> None:
    inputs = _make_inputs(tmp_path)
    manifest = json.loads(inputs["owner_manifest"].read_text(encoding="utf-8"))
    target = next(task for task in manifest["tasks"] if task["id"] == "standardize_bace_globalgce")
    target["expected_output"] = str(inputs["runtime"] / "wrong/attempt-{attempt}")
    _write_json(inputs["owner_manifest"], manifest)
    matrix_root = inputs["matrix_root"]

    spec = build_release_spec(
        catalog_path=inputs["catalog"],
        controller_id="owner_binding_failure",
        project_root=PROJECT_ROOT,
        runtime_root=inputs["runtime"],
        python=Path(sys.executable),
        state_root=inputs["runtime"] / "control/owner_binding_failure",
        registry_root=matrix_root / "release_registry_v1",
        output_root=matrix_root / "three_datasets_complete_v1",
        paper_staging_root=(
            inputs["runtime"]
            / "outputs/autodl/paper_staging/three_datasets_complete_v1"
        ),
        expectations_json=inputs["expectations"],
        taste_license_gate_json=inputs["taste"],
    )
    globalgce = next(
        cell
        for cell in spec["cells"]
        if cell["dataset"] == "BACE" and cell["method"] == "GlobalGCE"
    )
    assert spec["runnable"] is False
    assert globalgce["binding_state"] == "PLACEHOLDER"
    assert globalgce["standardized_root"] is None
    assert globalgce["candidate_root_hint"] == str(inputs["roots"][("BACE", "GlobalGCE")])
    assert "BACE/GlobalGCE:owner_manifest_binding_unresolved" in spec["unresolved_bindings"]


def test_globalgce_raw_final_task_cannot_override_required_standardizer(
    tmp_path: Path,
) -> None:
    inputs = _make_inputs(tmp_path)
    matrix_root = inputs["matrix_root"]
    with pytest.raises(ReleaseSpecError, match="requires owner task"):
        build_release_spec(
            catalog_path=inputs["catalog"],
            controller_id="raw_globalgce_rejected",
            project_root=PROJECT_ROOT,
            runtime_root=inputs["runtime"],
            python=Path(sys.executable),
            state_root=inputs["runtime"] / "control/raw_globalgce_rejected",
            registry_root=matrix_root / "release_registry_v1",
            output_root=matrix_root / "three_datasets_complete_v1",
            paper_staging_root=(
                inputs["runtime"]
                / "outputs/autodl/paper_staging/three_datasets_complete_v1"
            ),
            expectations_json=inputs["expectations"],
            taste_license_gate_json=inputs["taste"],
            owner_task_overrides=("BACE/GlobalGCE=bace_globalgce_final_freeze",),
        )


def test_release_destinations_cannot_enter_paper_or_escape_runtime(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    matrix_root = inputs["matrix_root"]
    with pytest.raises(ReleaseSpecError, match="paper/|runtime_root"):
        build_release_spec(
            catalog_path=inputs["catalog"],
            controller_id="unsafe_destination",
            project_root=PROJECT_ROOT,
            runtime_root=inputs["runtime"],
            python=Path(sys.executable),
            state_root=inputs["runtime"] / "control/unsafe_destination",
            registry_root=matrix_root / "release_registry_v1",
            output_root=matrix_root / "three_datasets_complete_v1",
            paper_staging_root=PROJECT_ROOT / "paper/three_datasets_complete_v1",
            expectations_json=inputs["expectations"],
            taste_license_gate_json=inputs["taste"],
        )


def _write_export_closure(root: Path, matrix_sha256: str) -> None:
    root.mkdir(parents=True)
    (root / "PASS").write_text("PASS\n", encoding="utf-8")
    _write_json(root / "THREE_DATASET_EXPORT_PASS.json", {"status": "PASS"})
    _write_json(
        root / "three_dataset_export_manifest.json",
        {
            "status": "PASS",
            "matrix_complete_cells": 12,
            "matrix_total_cells": 16,
            "matrix_status_sha256": matrix_sha256,
            "paper_status": "PAPER_FROZEN_PARTIAL",
        },
    )
    _write_json(
        root / "three_dataset_export_audit.json",
        {
            "passed": True,
            "all_12_three_dataset_cells_verified": True,
            "taste_license_block_preserved": True,
        },
    )
    (root / "paper_figure3_three_datasets.pdf").write_bytes(b"figure-3")
    (root / "paper_figure4_three_datasets.pdf").write_bytes(b"figure-4")
    (root / "paper_table2_three_datasets.tex").write_text("table-2\n", encoding="utf-8")


def test_restart_promotes_complete_owned_export_transaction(tmp_path: Path) -> None:
    _spec_path, spec, _inputs = _build_spec(tmp_path)
    state_root = Path(spec["paths"]["state_root"])
    state_root.mkdir(parents=True)
    matrix_sha256 = "a" * 64
    runtime_spec = {**spec, "_runtime_spec_sha256": "b" * 64}
    transaction_id = "fixture-a1"
    temporary_output, temporary_staging = release_module._transaction_paths(
        spec, transaction_id
    )
    _write_export_closure(temporary_output, matrix_sha256)
    _write_export_closure(temporary_staging, matrix_sha256)
    _write_json(
        state_root / "release_transaction.json",
        {
            "schema_version": release_module.TRANSACTION_SCHEMA_VERSION,
            "status": "EXPORTING",
            "spec_sha256": "b" * 64,
            "temporary_output": str(temporary_output),
            "temporary_staging": str(temporary_staging),
            "exporter_process": _fake_process_identity(),
        },
    )

    assert release_module._reconcile_export(
        spec=runtime_spec,
        state_root=state_root,
        matrix_sha256=matrix_sha256,
    )
    assert Path(spec["paths"]["output_root"]).is_dir()
    assert Path(spec["paths"]["paper_staging_root"]).is_dir()
    transaction = json.loads(
        (state_root / "release_transaction.json").read_text(encoding="utf-8")
    )
    assert transaction["status"] == "PASS"
