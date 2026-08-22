from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerError,
    load_controller_manifest,
)
from src.eval.am_legacy_standardization import (
    LegacyStandardizationError,
    MUT_MATCHED_REQUIRED_STANDARDIZED_FILES,
    MUT_OURS_REQUIRED_SOURCE_FILES,
    MUT_OURS_REQUIRED_STANDARDIZED_FILES,
    adopt_mutagenicity_ours,
    audit_legacy_inventory,
    freeze_mutagenicity_gcf_candidates,
    reexport_mutagenicity_ours_matched,
    _validate_matched_mut_ours,
    verify_adopted_mut_ours,
)
from src.eval.four_by_four_main_results import audit_cell
from src.eval.mutagenicity_wnode_frozen_test import (
    FrozenTestConfig,
    build_frozen_test_run,
)
from src.eval.fullgraph_wnode_artifacts import stable_json_sha256


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


class FakeTeacher:
    def score_smiles(self, smiles, label=None, **kwargs):
        del kwargs
        parent = smiles in {"CCCO", "CCCN"}
        probability = 0.9 if parent else 0.2
        prediction = 1 if parent else 0
        return {
            "teacher_result_ok": True,
            "teacher_reason": "ok",
            "teacher_label": prediction,
            "teacher_prob": probability if label == 1 else 1.0 - probability,
        }


class FakeDistance:
    def distance(self, left, right):
        if left == right:
            value = 0.0
        elif right == "CN":
            value = 0.02
        elif right == "CC":
            value = 0.03
        else:
            value = 0.05
        return {"distance": value, "ok": True, "cache_hit": False, "error": None}

    def stats_dict(self):
        return {
            "pair_distance_cache_hit_rate": 0.0,
            "node_embedding_cache_hit_rate": 0.0,
        }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _thresholds() -> dict:
    quantiles = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
    values = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
    weights = (4.0, 4.0, 3.0, 3.0, 2.0, 1.0, 1.0)
    labels = ("q05", "q10", "q20", "q30", "q50", "q70", "q90")
    return {
        "finite_strict_flip_distance_count": 1573,
        "quantile_method": "linear",
        "dtype": "float64",
        "requested_quantiles": list(quantiles),
        "requested_weights": list(weights),
        "raw_quantile_thresholds": [
            {
                "quantile": q,
                "quantile_label": label,
                "threshold": threshold,
                "weight": weight,
            }
            for q, label, threshold, weight in zip(
                quantiles, labels, values, weights
            )
        ],
        "merged_thresholds": [
            {
                "threshold_id": label,
                "threshold": threshold,
                "weight": weight,
                "quantiles": [q],
                "quantile_labels": [label],
            }
            for q, label, threshold, weight in zip(
                quantiles, labels, values, weights
            )
        ],
        "duplicate_thresholds_merged": False,
        "theta_star_quantile": 0.30,
        "theta_star": 0.04,
        "cost_cap_quantile": 0.90,
        "cost_cap": 0.07,
        "threshold_source": "calibration_all_finite_strict_flip_pairs",
        "test_used": False,
    }


def _frozen_selector(root: Path) -> Path:
    selected = root / "selected_variant"
    selected.mkdir(parents=True)
    rows = [
        {
            "rank": rank,
            "candidate_id": f"C{rank:02d}",
            "canonical_fragment": fragment,
            "source_parent_count": 100 - rank,
            "source_cf_drop_mean": 0.5,
            "source_reward_mean": 1.0,
        }
        for rank, fragment in enumerate(FRAGMENTS, start=1)
    ]
    (selected / "selected_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    ids = [row["candidate_id"] for row in rows]
    _json(selected / "selected_top10.json", {"candidate_ids": ids[:10], "candidates": rows[:10]})
    _json(selected / "selected_top20.json", {"candidate_ids": ids, "candidates": rows})
    common = {
        "frozen": True,
        "selected_variant": "A2_MultiThreshold",
        "top_k": 20,
        "table_k": 10,
        "test_used_for_selection": False,
    }
    _json(root / "_FROZEN.json", common)
    _json(
        root / "calibration_decision.json",
        {"selected_variant": "A2_MultiThreshold", "test_used_for_selection": False},
    )
    _json(root / "thresholds.json", _thresholds())
    required = (
        "_FROZEN.json",
        "thresholds.json",
        "calibration_decision.json",
        "selected_variant/selected_sequence.jsonl",
        "selected_variant/selected_top10.json",
        "selected_variant/selected_top20.json",
    )
    _json(
        root / "frozen_selector_manifest.json",
        {**common, "file_sha256": {name: _sha(root / name) for name in required}},
    )
    return root


def _test_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["molecule_id", "smiles", "label", "split"])
        writer.writeheader()
        writer.writerow({"molecule_id": "P1", "smiles": "CCCO", "label": 1, "split": "test"})
        writer.writerow({"molecule_id": "P2", "smiles": "CCCN", "label": 1, "split": "test"})
    return path


def _deletions(parent, fragment):
    del parent
    if fragment != "C":
        return []
    return [
        {"match_index": 0, "match_atoms": [0], "delete_valid": True, "residual_smiles": "CC", "error": None},
        {"match_index": 1, "match_atoms": [1], "delete_valid": True, "residual_smiles": "CN", "error": None},
    ]


def _legacy_source(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    selector = _frozen_selector(tmp_path / "selector")
    test_csv = _test_csv(tmp_path / "frozen_input.csv")
    teacher = tmp_path / "teacher.pkl"
    teacher.write_bytes(b"teacher")
    checkpoint = tmp_path / "molclr.pt"
    checkpoint.write_bytes(b"molclr")
    molclr_root = tmp_path / "molclr"
    molclr_root.mkdir()
    raw_test_run = tmp_path / "raw_test_run"
    build_frozen_test_run(
        frozen_selector_root=selector,
        test_csv=test_csv,
        teacher_path=teacher,
        molclr_root=molclr_root,
        molclr_checkpoint=checkpoint,
        output_dir=raw_test_run,
        wnode_cache_db=tmp_path / "cache.sqlite",
        teacher=FakeTeacher(),
        distance_provider=FakeDistance(),
        config=FrozenTestConfig(expected_parent_count=2, flush_every=5),
        deletion_fn=_deletions,
    )
    source = tmp_path / "legacy_result"
    source.mkdir()
    for path in raw_test_run.iterdir():
        if path.is_file() and path.name != "resume_checkpoint.json":
            shutil.copy2(path, source / path.name)
    _json(source / "_FINALIZED.json", {"finalized": True})
    _json(
        source / "manual_final_test_audit_v2.json",
        {
            "audit_marker_present": True,
            "can_finalize_ours_results": True,
            "experiment_success": True,
            "official_audit_passed": True,
            "failed_hard_checks": [],
            "candidate_count": 20,
            "pair_rows": 40,
            "checks": {
                "complete_cartesian": True,
                "strict_flip": True,
                "thresholds_exactly_frozen": False,
            },
            "audit_correction": {
                "replacement_check": "threshold_values_and_weights_exactly_frozen_with_provenance_metadata_allowed"
            },
        },
    )
    _json(
        source / "threshold_freeze_semantic_audit_v2.json",
        {
            "audit_passed": True,
            "scientific_core_equal": True,
            "scientific_mismatches": {},
            "unexpected_difference_keys": [],
            "missing_scientific_keys": [],
            "provenance_checks": {
                "selector_frozen": True,
                "selector_frozen_before_test": True,
                "test_threshold_fitting_false": True,
                "test_used_for_selection_false": True,
            },
        },
    )
    files = {
        path.relative_to(source).as_posix(): _sha(path)
        for path in source.iterdir()
        if path.is_file()
    }
    assert set(MUT_OURS_REQUIRED_SOURCE_FILES) <= set(files)
    _json(
        source / "final_result_manifest.json",
        {
            "finalized": True,
            "method": "Ours-ChemLLM-PPO-WNode-A2",
            "dataset": "Mutagenicity",
            "source_label": 1,
            "target_label": 0,
            "selected_variant": "A2_MultiThreshold",
            "file_sha256": files,
            "frozen_selector_root": "/unavailable/wnode_selector_calibrated_v2",
            "test_run_root": "/unavailable/wnode_frozen_a2_test_p217_k20_v3",
            "theta_star": 0.04,
            "cost_cap": 0.07,
            "test_used_for_selection": False,
            "test_parent_count": 2,
            "top_k": 20,
            "table_k": 10,
        },
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    return source, teacher, checkpoint, proc


def test_mut_ours_is_strictly_adopted_without_changing_source(tmp_path):
    source, teacher, checkpoint, proc = _legacy_source(tmp_path)
    before = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source.iterdir()}
    output = tmp_path / "fresh" / "mut_ours"
    result = adopt_mutagenicity_ours(
        source_root=source,
        output_root=output,
        remap_roots=[tmp_path],
        expected_teacher_sha256=_sha(teacher),
        expected_molclr_sha256=_sha(checkpoint),
        expected_parent_count=2,
        expected_candidate_count=20,
        expected_pair_count=40,
        proc_root=proc,
    )
    assert result["status"] == "STALE_METRIC"
    audit = json.loads(
        (output / "standardized/final_artifact_audit.json").read_text()
    )
    assert audit["final_artifact_audit_passed"] is True
    assert audit["oracle_backend"] == "rf"
    assert audit["selector_fitted_on_calibration"] is True
    assert audit["test_used_only_after_freeze"] is True
    manifest = json.loads((output / "standardized/run_manifest.json").read_text())
    assert manifest["generation_adopted"] is True
    assert manifest["ordering_adopted"] is True
    assert manifest["evaluation_adopted"] is True
    assert {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source.iterdir()} == before
    verification = verify_adopted_mut_ours(
        adopted_root=output,
        output_root=tmp_path / "fresh" / "verification",
    )
    assert verification["manifest_only"] is True
    proof = json.loads(
        (tmp_path / "fresh/verification/adoption_verification.json").read_text()
    )
    assert proof["raw_heldout_input_opened"] is False


def test_mut_ours_fails_closed_on_manifest_tampering(tmp_path):
    source, teacher, checkpoint, proc = _legacy_source(tmp_path)
    with (source / "figure3_coverage_vs_k.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(LegacyStandardizationError, match="SHA256 mismatch"):
        adopt_mutagenicity_ours(
            source_root=source,
            output_root=tmp_path / "output",
            remap_roots=[tmp_path],
            expected_teacher_sha256=_sha(teacher),
            expected_molclr_sha256=_sha(checkpoint),
            expected_parent_count=2,
            expected_candidate_count=20,
            expected_pair_count=40,
            proc_root=proc,
        )


def test_mut_ours_manifest_only_verification_rehashes_standardized_files(tmp_path):
    source, teacher, checkpoint, proc = _legacy_source(tmp_path)
    adopted = tmp_path / "adopted"
    adopt_mutagenicity_ours(
        source_root=source,
        output_root=adopted,
        remap_roots=[tmp_path],
        expected_teacher_sha256=_sha(teacher),
        expected_molclr_sha256=_sha(checkpoint),
        expected_parent_count=2,
        expected_candidate_count=20,
        expected_pair_count=40,
        proc_root=proc,
    )
    with (adopted / "standardized/figure3_coverage_vs_k.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("tampered\n")
    with pytest.raises(
        LegacyStandardizationError, match="standardized artifact SHA256 mismatch"
    ):
        verify_adopted_mut_ours(
            adopted_root=adopted,
            output_root=tmp_path / "verification",
        )


def _matched_reexport_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, teacher, checkpoint, proc = _legacy_source(tmp_path)
    adopted = tmp_path / "adopted-original-protocol"
    adopt_mutagenicity_ours(
        source_root=source,
        output_root=adopted,
        remap_roots=[tmp_path],
        expected_teacher_sha256=_sha(teacher),
        expected_molclr_sha256=_sha(checkpoint),
        expected_parent_count=2,
        expected_candidate_count=20,
        expected_pair_count=40,
        proc_root=proc,
    )
    matched = tmp_path / "matched-fresh"
    reexport_mutagenicity_ours_matched(
        adopted_root=adopted,
        matched_protocol=Path(
            "configs/autodl/mutagenicity_matched_protocol_v1.json"
        ),
        output_root=matched,
        proc_root=proc,
    )
    return source, adopted, matched


def test_mut_ours_matched_reexport_has_601_point_frozen_closure(tmp_path):
    _source, _adopted, matched = _matched_reexport_fixture(tmp_path)
    assert _validate_matched_mut_ours(matched) == matched.resolve()
    standardized = matched / "standardized"
    figure3 = list(csv.DictReader((standardized / "figure3_coverage_vs_k.csv").open()))
    figure4 = list(
        csv.DictReader((standardized / "figure4_coverage_vs_threshold.csv").open())
    )
    assert [int(row["k"]) for row in figure3] == list(range(1, 21))
    assert len(figure4) == 601
    assert float(figure4[0]["threshold"]) == 0.0
    assert float(figure4[-1]["threshold"]) == pytest.approx(0.0535, abs=1e-15)
    assert {int(row["k"]) for row in figure4} == {10}
    manifest = json.loads((standardized / "run_manifest.json").read_text())
    assert manifest["status"] == "FROZEN_PASS"
    assert manifest["test_used_for_selection"] is False
    assert manifest["threshold_fitted_on_test"] is False
    assert manifest["raw_test_split_opened"] is False
    assert manifest["distance_recomputed"] is False
    assert manifest["oracle_recomputed"] is False
    assert manifest["molclr_recomputed"] is False
    assert (matched / "PASS").read_text().strip() == "PASS"
    assert set(MUT_MATCHED_REQUIRED_STANDARDIZED_FILES) <= {
        path.name for path in standardized.iterdir() if path.is_file()
    }
    row = {
        key: manifest[key]
        for key in (
            "dataset",
            "method",
            "oracle_backend",
            "oracle_hash",
            "dataset_hash",
            "split_hash",
            "distance_line",
            "molclr_checkpoint_hash",
            "cf_mode",
            "threshold_config_hash",
            "status",
        )
    }
    row["standardized_output_root"] = str(standardized.resolve())
    audited = audit_cell(row)
    assert audited.dataset == "Mutagenicity"
    assert audited.method == "Ours"


def test_old_14_point_curve_cannot_pass_matched_validation(tmp_path):
    _source, _adopted, matched = _matched_reexport_fixture(tmp_path)
    path = matched / "standardized/figure4_coverage_vs_threshold.csv"
    rows = list(csv.DictReader(path.open()))[:14]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(LegacyStandardizationError, match="not 601 points"):
        _validate_matched_mut_ours(matched)


def test_matched_reexport_requires_fresh_root_and_rejects_pair_tamper(tmp_path):
    source, adopted, matched = _matched_reexport_fixture(tmp_path)
    proc = tmp_path / "proc"
    with pytest.raises(FileExistsError, match="must be fresh"):
        reexport_mutagenicity_ours_matched(
            adopted_root=adopted,
            matched_protocol=Path(
                "configs/autodl/mutagenicity_matched_protocol_v1.json"
            ),
            output_root=matched,
            proc_root=proc,
        )
    with (source / "pair_matrix.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"tampered": true}\n')
    with pytest.raises(LegacyStandardizationError, match="was tampered"):
        reexport_mutagenicity_ours_matched(
            adopted_root=adopted,
            matched_protocol=Path(
                "configs/autodl/mutagenicity_matched_protocol_v1.json"
            ),
            output_root=tmp_path / "tampered-output",
            proc_root=proc,
        )


def _inventory_spec(tmp_path: Path) -> Path:
    cells = []
    definitions = (
        ("Mutagenicity", "Ours", "STALE_METRIC", "STRICT_AUDIT_PENDING"),
        ("Mutagenicity", "GCFExplainer", "INCOMPLETE", "MISSING_CALIBRATION"),
        ("Mutagenicity", "GlobalGCE", "INCOMPLETE", "MISSING_HELDOUT_MATRIX"),
        ("AIDS", "Ours", "INCOMPLETE", "MISSING_SELECTOR_ORDER"),
        ("AIDS", "GCFExplainer", "INCOMPLETE", "MISSING_RF_WNODE_CLOSURE"),
        ("AIDS", "GlobalGCE", "BLOCKED_CODE", "BLOCKED_GLOBALGCE_LHS_RHS_ATTACHMENT_MAPPING_UNAVAILABLE"),
    )
    for index, (dataset, method, status, reason) in enumerate(definitions):
        root = tmp_path / f"raw-{index}"
        root.mkdir()
        (root / "summary.json").write_text("{}\n", encoding="utf-8")
        cells.append(
            {
                "dataset": dataset,
                "method": method,
                "status": status,
                "source_roots": [str(root)],
                "generation_adopted": True,
                "ordering_adopted": False,
                "evaluation_adopted": False,
                "reason": reason,
                "probe_basenames": ["summary.json"],
            }
        )
    spec = tmp_path / "sources.json"
    _json(spec, {"schema_version": "am_legacy_sources_v1", "path_remap_roots": [str(tmp_path)], "cells": cells})
    return spec


def test_inventory_is_precise_and_never_includes_clear(tmp_path):
    spec = _inventory_spec(tmp_path)
    adopted = tmp_path / "adopted"
    standardized = adopted / "standardized"
    for name in MUT_OURS_REQUIRED_STANDARDIZED_FILES:
        path = standardized / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            path.write_text("method,value\nOurs,1\n", encoding="utf-8")
        else:
            _json(path, {"status": "STALE_METRIC"})
    _json(standardized / "summary.json", {"status": "STALE_METRIC"})
    _json(
        standardized / "run_manifest.json",
        {
            "status": "STALE_METRIC",
            "dataset": "Mutagenicity",
            "method": "Ours",
        },
    )
    _json(standardized / "oracle_manifest.json", {"oracle_backend": "rf"})
    _json(standardized / "evaluation_manifest.json", {"cf_mode": "strict_flip"})
    artifact_files = {
        name: _sha(standardized / name)
        for name in MUT_OURS_REQUIRED_STANDARDIZED_FILES
    }
    _json(
        standardized / "artifact_manifest.json",
        {"files": artifact_files, "file_count": len(artifact_files)},
    )
    _json(
        standardized / "final_artifact_audit.json",
        {
            "final_artifact_audit_passed": True,
            "selector_fitted_on_calibration": True,
            "test_used_only_after_freeze": True,
            "artifact_manifest_sha256": _sha(standardized / "artifact_manifest.json"),
        },
    )
    _json(adopted / "_RUN_COMPLETE.json", {"run_complete": True})
    (adopted / "PASS").write_text("PASS\n", encoding="utf-8")
    output = tmp_path / "inventory"
    result = audit_legacy_inventory(
        source_spec=spec,
        output_root=output,
        adopted_mut_ours_root=adopted,
    )
    assert result["cell_count"] == 6
    payload = json.loads((output / "matrix_patch.json").read_text())
    statuses = {(row["dataset"], row["method"]): row["status"] for row in payload["cells"]}
    assert statuses[("Mutagenicity", "Ours")] == "STALE_METRIC"
    assert statuses[("AIDS", "GlobalGCE")] == "BLOCKED_CODE"
    assert not any(row["method"].lower() == "clear" for row in payload["cells"])
    assert all(row["rerun_generation"] is False for row in payload["cells"])


def test_inventory_promotes_only_valid_matched_mut_ours_root(tmp_path):
    spec = _inventory_spec(tmp_path)
    _source, _adopted, matched = _matched_reexport_fixture(tmp_path / "matched")
    output = tmp_path / "inventory-matched"
    audit_legacy_inventory(
        source_spec=spec,
        output_root=output,
        adopted_mut_ours_root=matched,
    )
    payload = json.loads((output / "matrix_patch.json").read_text())
    ours = next(
        row
        for row in payload["cells"]
        if row["dataset"] == "Mutagenicity" and row["method"] == "Ours"
    )
    assert ours["status"] == "FROZEN_PASS"
    assert ours["standardized_output_root"] == str(
        (matched / "standardized").resolve()
    )
    assert "601_POINT_PROTOCOL" in ours["reason"]


def test_inventory_rejects_clear_in_source_spec(tmp_path):
    spec = _inventory_spec(tmp_path)
    payload = json.loads(spec.read_text())
    payload["cells"].append(
        {
            "dataset": "AIDS",
            "method": "CLEAR",
            "status": "INCOMPLETE",
            "source_roots": [str(tmp_path)],
        }
    )
    _json(spec, payload)
    with pytest.raises(LegacyStandardizationError, match="CLEAR is excluded"):
        audit_legacy_inventory(source_spec=spec, output_root=tmp_path / "out")


def test_controller_task_fragment_is_merge_compatible(tmp_path):
    fragment = json.loads(
        Path("configs/autodl/am_legacy_standardization_v1.tasks.json").read_text()
    )
    merged = {
        "schema_version": 1,
        "controller_id": "am-legacy-test-controller",
        "paper_frozen": True,
        "runtime": {
            "max_gpus": 4,
            "stable_idle_seconds": 60,
            "sample_interval_seconds": 5,
            "poll_seconds": 60,
            "max_transient_retries": 1,
        },
        "resource_gates": {},
        "tasks": fragment["tasks"],
    }
    path = tmp_path / "controller.json"
    _json(path, merged)
    manifest = load_controller_manifest(path)
    assert tuple(manifest.by_id) == (
        "mut_ours_legacy_adoption_verify",
        "mut_ours_matched_protocol_reexport",
        "am_legacy_inventory",
        "mut_gcf_legacy_freeze",
        "mut_gcf_legacy_calibration",
        "mut_gcf_legacy_heldout",
        "mut_gcf_legacy_standardized",
        "mut_globalgce_legacy_cell_blocked",
        "aids_ours_legacy_cell_blocked",
        "aids_gcfexplainer_legacy_cell_blocked",
        "aids_globalgce_legacy_cell_blocked",
    )
    assert manifest.by_id["mut_gcf_legacy_freeze"].manifest_only is True
    assert manifest.by_id["mut_gcf_legacy_calibration"].resource == "gpu"
    assert manifest.by_id["mut_gcf_legacy_heldout"].resource == "gpu"
    matrix_prefix = (
        "{runtime_root}/outputs/autodl/paper_matrix/"
        "four_methods_four_datasets_v1/"
    )
    persistent_prefix = matrix_prefix + "am_legacy/"
    assert manifest.by_id[
        "mut_ours_legacy_adoption_verify"
    ].input_manifest.startswith(matrix_prefix)
    assert manifest.by_id["am_legacy_inventory"].input_manifest.startswith(
        matrix_prefix
    )
    assert all(
        (task.expected_output or "").startswith(persistent_prefix)
        for task in manifest.tasks
        if task.blocked_reason is None
    )
    assert "{artifact_root}/am_legacy" not in json.dumps(fragment)
    actions = {task.environment.get("ACTION") for task in manifest.tasks}
    assert "adopt-mut-ours" not in actions
    assert "verify-mut-ours-adoption" in actions
    assert "reexport-mut-ours-matched" in actions
    reexport = manifest.by_id["mut_ours_matched_protocol_reexport"]
    inventory = manifest.by_id["am_legacy_inventory"]
    assert reexport.depends_on == ("mut_ours_legacy_adoption_verify",)
    assert inventory.depends_on == (reexport.task_id,)
    assert (
        inventory.environment["ADOPTED_MUT_OURS_ROOT"]
        == "{dep_mut_ours_matched_protocol_reexport_output}"
    )
    freeze = manifest.by_id["mut_gcf_legacy_freeze"]
    calibration = manifest.by_id["mut_gcf_legacy_calibration"]
    heldout = manifest.by_id["mut_gcf_legacy_heldout"]
    assert calibration.depends_on == (freeze.task_id,)
    standardized = manifest.by_id["mut_gcf_legacy_standardized"]
    assert heldout.depends_on == (calibration.task_id, freeze.task_id)
    assert standardized.depends_on == (heldout.task_id, freeze.task_id)
    assert standardized.resource == "cpu"
    assert standardized.manifest_only is True
    assert calibration.freezes_selector is True
    assert calibration.data_splits == ("calibration",)
    assert heldout.selector_parameters_frozen is True
    assert heldout.read_only_test is True
    assert heldout.data_splits == ("test",)
    assert (
        calibration.environment["THRESHOLDS_JSON"]
        == "{dep_mut_gcf_legacy_freeze_output}/matched_thresholds.json"
    )
    blocked_ids = {
        "mut_globalgce_legacy_cell_blocked",
        "aids_ours_legacy_cell_blocked",
        "aids_gcfexplainer_legacy_cell_blocked",
        "aids_globalgce_legacy_cell_blocked",
    }
    for task_id in blocked_ids:
        task = manifest.by_id[task_id]
        assert task.command is None
        assert task.resource == "cpu"
        assert task.blocked_reason


def test_mut_gcf_heldout_cannot_bypass_calibration_freeze(tmp_path):
    fragment = json.loads(
        Path("configs/autodl/am_legacy_standardization_v1.tasks.json").read_text()
    )
    for task in fragment["tasks"]:
        if task["id"] == "mut_gcf_legacy_heldout":
            task["depends_on"] = ["mut_gcf_legacy_freeze"]
    merged = {
        "schema_version": 1,
        "controller_id": "am-legacy-bypass-test",
        "paper_frozen": True,
        "runtime": {
            "max_gpus": 4,
            "stable_idle_seconds": 60,
            "sample_interval_seconds": 5,
            "poll_seconds": 60,
            "max_transient_retries": 1,
        },
        "resource_gates": {},
        "tasks": fragment["tasks"],
    }
    path = tmp_path / "controller.json"
    _json(path, merged)
    with pytest.raises(ControllerError, match="frozen B12/AM selector dependency"):
        load_controller_manifest(path)


def test_tracked_mut_matched_protocol_is_exact_and_self_identifying():
    payload = json.loads(
        Path("configs/autodl/mutagenicity_matched_protocol_v1.json").read_text()
    )["datasets"]["Mutagenicity"]
    values = [float(value) for value in payload["thresholds"]]
    assert len(values) == 601
    assert all(
        left < right for left, right in zip(values, values[1:])
    )
    assert all(
        abs(value - 0.0535 * index / 600) <= 1e-12
        for index, value in enumerate(values)
    )
    assert payload["theta_star"] == 0.05
    assert payload["cost_cap"] == 0.0535
    assert payload["threshold_source_split"] == "existing_frozen_protocol"
    assert payload["test_used_for_selection"] is False
    identity = {
        key: payload[key]
        for key in (
            "thresholds",
            "theta_star",
            "cost_cap",
            "threshold_source",
            "threshold_source_split",
            "test_used_for_selection",
        )
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert digest == payload["threshold_config_hash"]


def test_mut_gcf_foreground_wrapper_cannot_run_generation():
    source = Path(
        "scripts/autodl/run_mut_gcf_legacy_evaluation.sh"
    ).read_text(encoding="utf-8")
    assert "build_mutagenicity_wnode_calibration_matrix.sh" in source
    assert "evaluate_mutagenicity_wnode_frozen_test.sh" in source
    assert "export PROJECT_ROOT" in source
    for forbidden in (
        "run_mutagenicity_vrrw.py",
        "build_mutagenicity_train_pool.py",
        "reproduce_mutagenicity",
        "sbatch",
        "ssh ",
    ):
        assert forbidden not in source


def test_mut_gcf_raw_export_is_frozen_without_generation(tmp_path):
    source = tmp_path / "legacy-gcf-export"
    source.mkdir()
    rows = [
        {
            "candidate_id": f"gcf-{index:02d}",
            "native_rank": index * 3,
            "smiles": fragment,
            "canonical_smiles": fragment,
            "rdkit_valid": True,
            "rf_pred": 0,
            "rf_prob_0": 0.9,
            "rf_prob_1": 0.1,
            "source_method": "official_gcfexplainer_mutagenicity",
            "selection_method": "native_gcf_summary_rank_filtered_by_validity",
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "projected_new_edge_count": 0,
            "retained_edge_count": 1,
            "source_parent_id": "train-parent",
        }
        for index, fragment in enumerate(FRAGMENTS, start=1)
    ]
    with (source / "selected_top20.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidate_ids = [row["candidate_id"] for row in rows]
    native_ranks = [row["native_rank"] for row in rows]
    order_sha = stable_json_sha256(candidate_ids)
    teacher_sha = "a" * 64
    manifest = {
        "dataset": "Mutagenicity",
        "profile": "full",
        "parent_limit": 1448,
        "selected_count": 20,
        "selected_top20_rows": 20,
        "candidate_filter_audit_rows": 20,
        "candidate_yield_gate_passed": True,
        "full_result_ready": True,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "rf_reranking_performed": False,
        "wnode_reranking_performed": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "teacher_used_only_for_target_validation": True,
        "teacher_sha256": teacher_sha,
        "selection_method": "native_gcf_summary_rank_filtered_by_validity",
        "selected_candidate_order_sha256": order_sha,
        "run_complete": True,
    }
    _json(source / "run_manifest.json", manifest)
    _json(source / "_RUN_COMPLETE.json", manifest)
    _json(
        source / "filter_summary.json",
        {
            "selected_count": 20,
            "audit_complete": True,
            "all_candidates_terminal": True,
            "native_order_preserved": True,
            "candidate_yield_gate_passed": True,
            "rf_reranking_performed": False,
            "wnode_reranking_performed": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "selected_native_ranks": native_ranks,
        },
    )
    (source / "candidate_filter_audit.jsonl").write_text(
        "".join(
            json.dumps(
                {"native_rank": rank, "selected": True, "rejection_stage": "selected"}
            )
            + "\n"
            for rank in native_ranks
        ),
        encoding="utf-8",
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    matched = tmp_path / "mutagenicity.json"
    grid = [0.0535 * index / 600 for index in range(601)]
    _json(
        matched,
        {
            "schema_version": "four_by_four_frozen_threshold_contract_v1",
            "status": "PASS",
            "dataset": "Mutagenicity",
            "distance_line": "MolCLR-Node-Wasserstein",
            "cf_mode": "strict_flip",
            "thresholds": grid,
            "theta_star": 0.05,
            "cost_cap": 0.0535,
            "threshold_source": "matched existing frozen protocol",
            "threshold_source_split": "existing_frozen_protocol",
            "threshold_config_hash": "b" * 64,
            "test_used_for_selection": False,
        },
    )
    ours_schema = tmp_path / "ours-schema"
    ours_schema.mkdir()
    for k in (10, 20):
        (ours_schema / f"table2_ours_k{k}.csv").write_text(
            "dataset,method,k\nMutagenicity,Ours,%d\n" % k,
            encoding="utf-8",
        )
    output = tmp_path / "frozen"
    result = freeze_mutagenicity_gcf_candidates(
        source_export_root=source,
        output_root=output,
        expected_csv_sha256=_sha(source / "selected_top20.csv"),
        expected_order_sha256=order_sha,
        expected_native_ranks=native_ranks,
        expected_teacher_sha256=teacher_sha,
        matched_threshold_contract=matched,
        ours_schema_source_root=ours_schema,
        proc_root=proc,
    )
    assert result["status"] == "PASS"
    assert result["generation_rerun"] is False
    frozen = json.loads((output / "frozen_candidate_manifest.json").read_text())
    assert frozen["schema_version"] == "mut_gcf_frozen_top20_v1"
    assert frozen["selected_native_ranks"] == native_ranks
    thresholds = json.loads((output / "matched_thresholds.json").read_text())
    assert len(thresholds["thresholds"]) == 601
    assert thresholds["theta_star"] == 0.05
    assert thresholds["cost_cap"] == 0.0535
