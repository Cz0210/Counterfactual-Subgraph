from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl.build_bace_cell_standardization_tasks import (
    CONTROLLER_TASKS,
    build_fragment,
)
from scripts.autodl.standardize_bace_frozen_cell import build_parser
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.bace_frozen_cell_standardization import (
    BACECellStandardizationError,
    sha256_file,
    standardize_bace_frozen_cell,
)
from src.eval.bace_frozen_gnn_contracts import stable_sha256
from src.eval.four_by_four_main_results import audit_cell
from src.eval.four_by_four_registry import AuditConfig, audit_registry


HISTORICAL_B13_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "bace_ours_b13_verification_v1_structure.json"
)


def test_standardization_cli_accepts_globalgce() -> None:
    args = build_parser().parse_args(
        [
            "--method",
            "GlobalGCE",
            "--source-final-root",
            "/source",
            "--gnn-checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
        ]
    )
    assert args.method == "GlobalGCE"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _checkpoint(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "checkpoint"
    root.mkdir()
    model = root / "model.pt"
    model.write_bytes(b"frozen-gine")
    checkpoint_id = sha256_file(model)
    test_hash = "a" * 64
    test_path = "/raw/test/is/intentionally/not/opened/bace_test.csv"
    _write_json(
        root / "model_card.json",
        {
            "dataset": "bace",
            "backbone": "gine",
            "num_classes": 2,
            "source_label": 1,
            "oracle_backend": "gnn",
            "classifier_type": "gnn",
            "rf_oracle_used": False,
            "checkpoint_id": checkpoint_id,
        },
    )
    _write_json(
        root / "split_manifest.json",
        {
            "dataset": "bace",
            "files": {"test": {"path": test_path, "sha256": test_hash}},
            "test_loaded_for_training": False,
        },
    )
    _write_json(
        root / "test_evaluation_status.json",
        {
            "status": "NOT_EVALUATED",
            "test_loaded": False,
            "path": test_path,
            "sha256": test_hash,
        },
    )
    _write_json(root / "temperature_scaling.json", {"test_used_for_fit": False})
    _write_json(root / "feature_schema.json", {"schema": "unit"})
    (root / "sha256sums.txt").write_text(
        f"{checkpoint_id}  model.pt\n", encoding="utf-8"
    )
    return root, checkpoint_id, test_hash


def _thresholds() -> dict[str, object]:
    values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25]
    return {
        "merged_thresholds": [
            {"threshold_id": f"q{index}", "threshold": value}
            for index, value in enumerate(values)
        ],
        "theta_star": 0.08,
        "cost_cap": 0.25,
        "threshold_source": "calibration_all_finite_strict_flip_pairs",
        "test_used": False,
    }


def _pair_rows(method: str, checkpoint_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parent in ("p0", "p1"):
        for rank in range(1, 21):
            strict = parent == "p0" or rank >= 2
            distance = rank * 0.01 + (0.005 if parent == "p1" else 0.0)
            row: dict[str, object] = {
                "dataset": "bace",
                "method": method,
                "parent_id": parent,
                "candidate_id": f"c{rank:02d}",
                "applicable": True,
                "pair_strict_flip": strict,
                "wnode_distance": distance if strict else None,
                "pred_before": 1,
                "pred_after": 0 if strict else None,
                "cf_drop": 0.5 if strict else None,
                "oracle_backend": "gnn",
                "classifier_type": "gnn" if method == "Ours" else "gine",
                "rf_oracle_used": False,
                "oracle_checkpoint_hash": checkpoint_id,
                "cf_mode": "strict_flip",
            }
            if method == "Ours":
                row.update(
                    {
                        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
                    }
                )
            rows.append(row)
    return rows


def _frozen_prefix() -> tuple[list[dict[str, object]], list[float]]:
    rows: list[dict[str, object]] = []
    curve: list[float] = []
    for k in range(1, 21):
        supp = 0.5 if k == 1 else 1.0
        coverage = 0.5 if k == 1 else 1.0
        values = [0.01] + ([] if k == 1 else [0.025])
        rows.append(
            {
                "K": k,
                "strict_flip_any_rate": supp,
                "SuppCov": supp,
                "CCRCov": coverage,
                "avg_cost": sum(values) / len(values),
                "median_cost": sorted(values)[len(values) // 2]
                if len(values) % 2
                else sum(values) / 2,
            }
        )
        curve.append(coverage)
    return rows, curve


def _source(tmp_path: Path, *, method: str, checkpoint_id: str, test_hash: str) -> Path:
    slug = method.lower() if method != "GCFExplainer" else "gcfexplainer"
    root = tmp_path / f"source_{slug}"
    root.mkdir()
    ordered = [f"c{rank:02d}" for rank in range(1, 21)]
    molclr = "b" * 64
    selection = {
        "dataset": "bace",
        "method": method,
        "method_id": slug,
        "stage": "B12_SELECTOR" if method == "Ours" else "BASELINE_CALIBRATION_SELECTOR",
        "status": "FROZEN",
        "selection_frozen": True,
        "selector_fitted_on_calibration": True,
        "calibration_loaded": True,
        "test_loaded": False,
        "test_used": False,
        "oracle_backend": "gnn",
        "classifier_type": "gnn" if method == "Ours" else "gine",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "source_label": 1,
        "cf_mode": "strict_flip",
        "oracle_checkpoint_hash": checkpoint_id,
        "molclr_checkpoint_hash": molclr,
        "ordered_rule_ids": ordered,
        "thresholds": _thresholds(),
    }
    selection_path = root / "frozen_selection_manifest.json"
    _write_json(selection_path, selection)

    parent_hash = stable_sha256(["p0", "p1"])
    shard_identities = []
    for index in range(4):
        shard = root / f"shard-{index}" / "verification_manifest.json"
        _write_json(
            shard,
            {
                "dataset": "bace",
                "method": method,
                "stage": "B13_FINAL_EVAL" if method == "Ours" else "BASELINE_TEST_EVAL",
                "status": "PASS",
                "test_loaded": True,
                "selection_frozen_before_test": True,
                "oracle_backend": "gnn",
                "classifier_type": "gnn" if method == "Ours" else "gine",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "cf_mode": "strict_flip",
                "oracle_checkpoint_hash": checkpoint_id,
                "all_parent_ids_sha256": parent_hash,
                "shard_index": index,
                "split_identity": {
                    "path": "/raw/test/is/intentionally/not/opened/bace_test.csv",
                    "size": 987,
                    "sha256": test_hash,
                },
            },
        )
        shard_identities.append(_identity(shard))
    merge = root / "matrix_manifest.json"
    merge_payload = {
        "dataset": "bace",
        "method": method,
        "method_id": slug,
        "stage": "B13_FINAL_EVAL" if method == "Ours" else "BASELINE_TEST_EVAL",
        "status": "PASS",
        "test_loaded": True,
        "selection_frozen_before_test": True,
        "oracle_backend": "gnn",
        "classifier_type": "gnn" if method == "Ours" else "gine",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "cf_mode": "strict_flip",
        "oracle_checkpoint_hash": checkpoint_id,
        "molclr_checkpoint_hash": molclr,
        "inputs": {"shard_manifests": shard_identities},
    }
    _write_json(merge, merge_payload)
    pair_path = root / "pair_matrix.jsonl"
    _write_jsonl(pair_path, _pair_rows(method, checkpoint_id))
    frozen_rows, curve = _frozen_prefix()
    metrics = root / "final_metrics.json"
    _write_json(
        metrics,
        {
            "dataset": "bace",
            "method": method,
            "stage": "B13_FINAL_EVAL" if method == "Ours" else "BASELINE_TEST_EVAL",
            "status": "PASS",
            "parent_count": 2,
            "ordered_rule_ids": ordered,
            "prefix_metrics": frozen_rows,
            "ccrcov_theta_star_by_k": curve,
        },
    )
    if method == "Ours":
        test = root / "test_evaluation_manifest.json"
        _write_json(
            test,
            {
                "dataset": "bace",
                "stage": "B13_FINAL_EVAL",
                "status": "PASS",
                "selection_frozen_before_test": True,
                "test_used_only_after_freeze": True,
                "selector_refit_on_test": False,
                "threshold_refit_on_test": False,
                "oracle_backend": "gnn",
                "classifier_type": "gnn",
                "rf_oracle_used": False,
                "cf_mode": "strict_flip",
                "oracle_checkpoint_hash": checkpoint_id,
                "molclr_checkpoint_hash": molclr,
                "frozen_selection_manifest_identity": _identity(selection_path),
                "verification_manifest_identity": _identity(merge),
                "pair_matrix_identity": _identity(pair_path),
                "final_metrics_identity": _identity(metrics),
            },
        )
        _write_json(
            root / "FINAL_PASS.json",
            {
                "dataset": "bace",
                "stage": "B14_FROZEN",
                "status": "PASS",
                "final_gate_pass": True,
                "all_hashes_frozen": True,
                "oracle_backend": "gnn",
                "classifier_type": "gnn",
                "rf_oracle_used": False,
                "source_label": 1,
                "cf_mode": "strict_flip",
                "oracle_checkpoint_hash": checkpoint_id,
                "molclr_checkpoint_hash": molclr,
                "selector_fitted_on_calibration": True,
                "test_used_only_after_freeze": True,
                "b12_manifest_identity": _identity(selection_path),
                "b13_manifest_identity": _identity(test),
                "final_metrics_identity": _identity(metrics),
            },
        )
    else:
        action = {
            "GCFExplainer": (
                "full_counterfactual_graph",
                "official_vrrw_neurosed_greedy_fullgraph_v1",
            ),
            "GlobalGCE": (
                "lhs_rhs_graph_transformation_rule",
                "native_lhs_to_rhs_attachment_aware_v1",
            ),
            "ComRecGC": (
                "native_common_recourse_fullgraph",
                "official_comrecgc_lineage_unique_transition_medoid_v1",
            ),
        }[method]
        _write_json(
            root / "FINAL_PASS.json",
            {
                "dataset": "bace",
                "method": method,
                "method_id": slug,
                "stage": "BASELINE_FINAL_FREEZE",
                "status": "PASS",
                "run_complete": True,
                "all_hashes_frozen": True,
                "action_kind": action[0],
                "action_semantics": action[1],
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "source_label": 1,
                "cf_mode": "strict_flip",
                "oracle_checkpoint_hash": checkpoint_id,
                "molclr_checkpoint_hash": molclr,
                "selector_fitted_on_calibration": True,
                "selection_frozen_before_test": True,
                "test_used_only_after_freeze": True,
                "selection_manifest_identity": _identity(selection_path),
                "test_manifest_identity": _identity(merge),
                "test_pair_matrix_identity": _identity(pair_path),
                "final_metrics_identity": _identity(metrics),
            },
        )
    (root / "PASS").write_text("PASS\n", encoding="utf-8")
    return root


def _refresh_ours_b13_identities(root: Path) -> None:
    selection_path = root / "frozen_selection_manifest.json"
    merge_path = root / "matrix_manifest.json"
    test_path = root / "test_evaluation_manifest.json"
    final_path = root / "FINAL_PASS.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    merge["inputs"]["shard_manifests"] = [
        _identity(root / f"shard-{index}" / "verification_manifest.json")
        for index in range(4)
    ]
    _write_json(merge_path, merge)
    test = json.loads(test_path.read_text(encoding="utf-8"))
    test["frozen_selection_manifest_identity"] = _identity(selection_path)
    test["verification_manifest_identity"] = _identity(merge_path)
    _write_json(test_path, test)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["b12_manifest_identity"] = _identity(selection_path)
    final["b13_manifest_identity"] = _identity(test_path)
    _write_json(final_path, final)


def _apply_historical_ours_b13_structure(root: Path) -> dict[str, object]:
    fixture = json.loads(HISTORICAL_B13_FIXTURE.read_text(encoding="utf-8"))
    selection_path = root / "frozen_selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection.update(fixture["b12_selection"])
    ordered = [str(value) for value in selection["ordered_rule_ids"]]
    selection["ordered_rule_ids_sha256"] = stable_sha256(ordered)
    selection["selected_top20_hash"] = "c" * 64
    selection["policy_checkpoint_hash"] = "d" * 64
    _write_json(selection_path, selection)

    shard_fixture = dict(fixture["b13_shard"])
    absent = [str(value) for value in shard_fixture.pop("absent_fields")]
    for index in range(4):
        path = root / f"shard-{index}" / "verification_manifest.json"
        shard = json.loads(path.read_text(encoding="utf-8"))
        for field in absent:
            shard.pop(field, None)
        shard.update(shard_fixture)
        shard.update(
            {
                "candidate_ids": ordered,
                "candidate_ids_sha256": stable_sha256(ordered),
                "candidate_count": len(ordered),
                "candidate_source_hash": selection["selected_top20_hash"],
                "policy_checkpoint_hash": selection["policy_checkpoint_hash"],
                "oracle_checkpoint_hash": selection["oracle_checkpoint_hash"],
                "molclr_checkpoint_hash": selection["molclr_checkpoint_hash"],
            }
        )
        _write_json(path, shard)

    merge_path = root / "matrix_manifest.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    merge.pop("selection_frozen_before_test", None)
    merge.update(
        {
            key: value
            for key, value in fixture["b13_merge"].items()
            if key != "inputs"
        }
    )
    merge["inputs"].update(fixture["b13_merge"]["inputs"])
    merge["inputs"]["predecessor_manifest"] = str(selection_path.parent.resolve())
    _write_json(merge_path, merge)

    test_path = root / "test_evaluation_manifest.json"
    test = json.loads(test_path.read_text(encoding="utf-8"))
    test.update(fixture["b13_test_evaluation"])
    test["ordered_rule_ids"] = ordered
    test["ordered_rule_ids_sha256"] = stable_sha256(ordered)
    _write_json(test_path, test)
    _refresh_ours_b13_identities(root)
    return fixture


@pytest.mark.parametrize(
    "method", ["Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"]
)
def test_standardization_is_artifact_only_and_registry_complete(
    tmp_path: Path, method: str
) -> None:
    checkpoint, checkpoint_id, test_hash = _checkpoint(tmp_path)
    source = _source(tmp_path, method=method, checkpoint_id=checkpoint_id, test_hash=test_hash)
    output = tmp_path / "standardized"
    result = standardize_bace_frozen_cell(
        method=method,
        source_final_root=source,
        gnn_checkpoint=checkpoint,
        output_dir=output,
    )
    assert result["raw_test_opened"] is False
    assert (output / "PASS").read_text(encoding="utf-8").strip() == "PASS"
    assert (output / "freeze_manifest.json").is_file()
    assert json.loads((output / "_FINALIZED.json").read_text())["gate_passed"] is True
    assert not Path("/raw/test/is/intentionally/not/opened/bace_test.csv").exists()

    registry = audit_registry(
        AuditConfig(
            scan_roots=(output,),
            output_root=tmp_path / "registry",
            expectations={
                "datasets": {"BACE": {"oracle_checkpoint": str(checkpoint.resolve())}}
            },
            explicit_cells={f"BACE/{method}": str(output)},
        )
    )
    row = next(
        row
        for row in registry.matrix_rows
        if row["dataset"] == "BACE" and row["method"] == method
    )
    assert row["status"] == "FROZEN_PASS", row["rerun_reason"]
    cell = audit_cell(row)
    assert cell.method == method
    assert len(cell.figure3) == 20


def test_resource_capped_comrecgc_standardization_plateaus_after_effective_k(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_id, test_hash = _checkpoint(tmp_path)
    source = _source(
        tmp_path,
        method="ComRecGC",
        checkpoint_id=checkpoint_id,
        test_hash=test_hash,
    )
    ordered = [f"c{rank:02d}" for rank in range(1, 11)]
    selection_path = source / "frozen_selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection.update(
        {
            "ordered_rule_ids": ordered,
            "K": 10,
            "K_MAX": 20,
            "effective_rule_count": 10,
        }
    )
    _write_json(selection_path, selection)
    pair_path = source / "pair_matrix.jsonl"
    _write_jsonl(
        pair_path,
        [
            row
            for row in _pair_rows("ComRecGC", checkpoint_id)
            if row["candidate_id"] in ordered
        ],
    )
    metrics_path = source / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "ordered_rule_ids": ordered,
            "effective_rule_count": 10,
            "K_MAX": 20,
        }
    )
    _write_json(metrics_path, metrics)
    final_path = source / "FINAL_PASS.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final.update(
        {
            "ordered_rule_ids": ordered,
            "effective_rule_count": 10,
            "K_MAX": 20,
            "selection_manifest_identity": _identity(selection_path),
            "test_pair_matrix_identity": _identity(pair_path),
            "final_metrics_identity": _identity(metrics_path),
        }
    )
    _write_json(final_path, final)
    output = tmp_path / "standardized"
    result = standardize_bace_frozen_cell(
        method="ComRecGC",
        source_final_root=source,
        gnn_checkpoint=checkpoint,
        output_dir=output,
    )
    assert result["effective_rule_count"] == 10
    prefix = json.loads((output / "prefix_metrics.json").read_text(encoding="utf-8"))[
        "prefix_metrics"
    ]
    assert len(prefix) == 20
    assert prefix[9]["CCRCov"] == prefix[19]["CCRCov"]
    assert prefix[19]["plateau_after_effective_k"] is True


def test_standardization_rejects_pair_matrix_tamper(tmp_path: Path) -> None:
    checkpoint, checkpoint_id, test_hash = _checkpoint(tmp_path)
    source = _source(tmp_path, method="Ours", checkpoint_id=checkpoint_id, test_hash=test_hash)
    with (source / "pair_matrix.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(BACECellStandardizationError, match="(size|SHA256) changed"):
        standardize_bace_frozen_cell(
            method="Ours",
            source_final_root=source,
            gnn_checkpoint=checkpoint,
            output_dir=tmp_path / "out",
        )


def test_standardization_rejects_preregistered_threshold_mismatch(tmp_path: Path) -> None:
    checkpoint, checkpoint_id, test_hash = _checkpoint(tmp_path)
    source = _source(tmp_path, method="Ours", checkpoint_id=checkpoint_id, test_hash=test_hash)
    with pytest.raises(BACECellStandardizationError, match="threshold_config_hash"):
        standardize_bace_frozen_cell(
            method="Ours",
            source_final_root=source,
            gnn_checkpoint=checkpoint,
            output_dir=tmp_path / "out",
            expected_threshold_hash="f" * 64,
        )


def test_ours_historical_b13_shard_schema_proves_freeze_boundary(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_id, test_hash = _checkpoint(tmp_path)
    source = _source(
        tmp_path,
        method="Ours",
        checkpoint_id=checkpoint_id,
        test_hash=test_hash,
    )
    fixture = _apply_historical_ours_b13_structure(source)
    result = standardize_bace_frozen_cell(
        method="Ours",
        source_final_root=source,
        gnn_checkpoint=checkpoint,
        output_dir=tmp_path / "historical-standardized",
    )
    assert fixture["b13_shard"]["selector_frozen_before_split_load"] is True
    assert "selection_frozen_before_test" in fixture["b13_shard"]["absent_fields"]
    summary = json.loads(
        (tmp_path / "historical-standardized" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["selection_frozen_before_test"] is True
    assert result["raw_test_opened"] is False


def test_ours_historical_b13_shard_schema_rejects_missing_freeze_evidence(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_id, test_hash = _checkpoint(tmp_path)
    source = _source(
        tmp_path,
        method="Ours",
        checkpoint_id=checkpoint_id,
        test_hash=test_hash,
    )
    _apply_historical_ours_b13_structure(source)
    shard_path = source / "shard-0" / "verification_manifest.json"
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["selector_frozen_before_split_load"] = False
    _write_json(shard_path, shard)
    _refresh_ours_b13_identities(source)
    with pytest.raises(
        BACECellStandardizationError,
        match="selector_frozen_before_split_load=False",
    ):
        standardize_bace_frozen_cell(
            method="Ours",
            source_final_root=source,
            gnn_checkpoint=checkpoint,
            output_dir=tmp_path / "must-not-exist",
        )


def test_fragment_binds_all_four_frozen_science_terminals(tmp_path: Path) -> None:
    checkpoint, _checkpoint_id, _test_hash = _checkpoint(tmp_path)
    fragment = build_fragment(
        controller_id="four_methods_four_datasets_continuation_v1",
        output_root=tmp_path / "runs",
        gnn_checkpoint=checkpoint,
    )
    assert len(fragment["tasks"]) == 4
    assert {
        task["id"]: tuple(task["depends_on"]) for task in fragment["tasks"]
    } == {
        task_id: (dependency,)
        for _method, (task_id, dependency, _priority) in CONTROLLER_TASKS.items()
    }
    assert all(task["resource"] == "cpu" for task in fragment["tasks"])
    assert all(task["manifest_only"] is True for task in fragment["tasks"])
    assert all(task["data_splits"] == [] for task in fragment["tasks"])
    assert all("freeze_manifest.json" in task["required_output_files"] for task in fragment["tasks"])
    assert all("_FINALIZED.json" in task["required_output_files"] for task in fragment["tasks"])
    for task in fragment["tasks"]:
        task["command"] = [
            "/autodl-fs/data/frozen_gine"
            if value == str(checkpoint.resolve())
            else value
            for value in task["command"]
        ]

    dependencies = []
    for _method, (_task_id, dependency, priority) in CONTROLLER_TASKS.items():
        dependencies.append(
            {
                "id": dependency,
                "dataset": "frozen-science-evidence",
                "stage": "STATIC_FROZEN_SCIENCE_EVIDENCE",
                "depends_on": [],
                "resource": "cpu",
                "priority": priority - 10,
                "data_splits": [],
                "manifest_only": True,
                "command": None,
                "blocked_reason": "BLOCKED_UNIT_DEPENDENCY",
            }
        )
    manifest_path = tmp_path / "controller.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "controller_id": "four_methods_four_datasets_continuation_v1",
            "paper_frozen": True,
            "runtime": {
                "max_gpus": 4,
                "stable_idle_seconds": 60,
                "sample_interval_seconds": 5,
                "poll_seconds": 60,
                "keep_alive_when_blocked": True,
            },
            "resource_gates": {},
            "tasks": [*dependencies, *fragment["tasks"]],
        },
    )
    loaded = load_controller_manifest(manifest_path)
    assert set(CONTROLLER_TASKS) == {
        "Ours",
        "GCFExplainer",
        "GlobalGCE",
        "ComRecGC",
    }
    assert len(loaded.tasks) == 8
