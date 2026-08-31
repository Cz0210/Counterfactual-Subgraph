from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.autodl.run_comrecgc_standardized_continuation import ContinuationInputs
from src.baselines.comrecgc.contracts import sha256_file
from src.eval.four_by_four_registry import CellStatus
from src.utils import autodl_mut_comrecgc_exact_postprocess_v1 as postprocess


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file(path: Path, value: bytes = b"payload\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _inputs(tmp_path: Path) -> ContinuationInputs:
    source = tmp_path / "generation"
    source.mkdir(parents=True, exist_ok=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    molclr = tmp_path / "molclr"
    molclr.mkdir()
    return ContinuationInputs(
        dataset="mutagenicity",
        source_generation_root=source,
        upstream_root=upstream,
        dataset_dir=dataset,
        source_csv=None,
        distance_checkpoint=_file(tmp_path / "distance.pt"),
        dataset_csv=_file(tmp_path / "test.csv"),
        teacher_path=_file(tmp_path / "teacher.pkl"),
        molclr_root=molclr,
        molclr_checkpoint=_file(molclr / "model.pth"),
        thresholds_path=_json(tmp_path / "threshold.json", {"status": "PASS"}),
        output_root=tmp_path / "postprocess",
        device="cpu",
        theta_star=None,
        cost_cap=None,
        common_recourse_engine="adopted_exact_read_only_v1",
    )


def _exact_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    generation = tmp_path / "generation"
    generation.mkdir(parents=True, exist_ok=True)
    generation_manifest = _json(
        generation / "run_manifest.json",
        {"counterfactuals_sha256": postprocess.SOURCE_PAYLOAD_SHA256},
    )
    common = tmp_path / "exact/full"
    terminal = _json(
        common / "_RUN_COMPLETE.json",
        {
            "schema_version": "comrecgc_common_recourse_terminal_v2",
            "run_complete": True,
            "common_recourse_engine": "external_memory_exact_v1",
        },
    )
    dbscan_identity = {
        "schema_version": "comrecgc_external_memory_dbscan_v3",
        "vectors_path": str((common / "external_memory/recourse_vectors.npy").resolve()),
        "vectors_sha256": "v" * 64,
        "vectors_stat_identity": {"size": 1},
        "vectors_dtype": "float64",
        "vectors_shape": [813_595, 3],
        "contract": postprocess.EXPECTED_DBSCAN_CONTRACT,
        "sklearn_version": "1.7.2",
        "nearest_neighbors_fit_method": "brute",
        "nearest_neighbors_metric": "euclidean",
        "nearest_neighbors_algorithm": "brute",
        "border_assignment": "minimum_cluster_label_of_adjacent_core_component",
        "shortcut_contract": postprocess.EXPECTED_DBSCAN_SHORTCUT_CONTRACT,
        "distance_reference_dtype": "float64",
        "exact_worker_count": 4,
    }
    labels = _file(common / "external_memory/dbscan/labels.npy")
    dbscan_manifest = _json(
        common / "external_memory/dbscan/run_manifest.json",
        {
            "schema_version": "comrecgc_external_memory_dbscan_v3",
            "run_complete": True,
            "scientific_identity": dbscan_identity,
            "scientific_identity_sha256": postprocess.stable_json_sha256(
                dbscan_identity
            ),
            "num_samples": 813_595,
            "labels_path": str(labels.resolve()),
            "labels_sha256": sha256_file(labels),
            "neighbor_counts_available": True,
            "all_neighborhoods_materialized_simultaneously": False,
            "passes": ["neighbor_counts", "core_union", "border_assignment"],
            "max_rss_bytes": postprocess.EXPECTED_DBSCAN_CONTRACT["max_rss_bytes"],
            "clustering_path": "sklearn_float64_exact_multi_component_v1",
            "distance_reference_dtype": "float64",
            "exact_worker_count": 4,
            "single_component_shortcut_used": False,
            "failure_cap_used": False,
            "approximation_used": False,
            "sklearn_dbscan_label_semantics_preserved": True,
        },
    )
    manifest_payload = {
        "dataset": "mutagenicity",
        "mode": "full",
        "method": "COMRECGC",
        "cf_mode": "strict_flip",
        "test_loaded": False,
        "calibration_loaded": False,
        "run_complete": True,
        "common_recourse_engine": "external_memory_exact_v1",
        "common_recourse_count": 100,
        "parameters": postprocess.EXPECTED_RECOURSE_PARAMETERS,
        "generation_manifest_path": str(generation_manifest.resolve()),
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "counterfactuals_sha256": postprocess.SOURCE_PAYLOAD_SHA256,
        "external_memory_artifacts": {
            "engine": "external_memory_exact_v1",
            "dbscan_shortcut_mode": "sklearn_float64_exact_multi_component_v1",
            "dbscan_manifest": str(dbscan_manifest.resolve()),
            "dbscan_manifest_sha256": sha256_file(dbscan_manifest),
        },
    }
    manifest = _json(
        common / "run_manifest.json",
        manifest_payload,
    )
    artifacts = {
        "_RUN_COMPLETE.json": terminal,
        "run_manifest.json": manifest,
        "representative_counterfactuals.pt": _file(
            common / "representative_counterfactuals.pt"
        ),
        "selected_common_recourses.csv": _file(
            common / "selected_common_recourses.csv"
        ),
        "selected_common_recourses.json": _json(
            common / "selected_common_recourses.json",
            [{"rank": index + 1} for index in range(100)],
        ),
        "dbscan/run_manifest.json": dbscan_manifest,
        "dbscan/labels.npy": labels,
    }
    controller = _json(
        tmp_path / "control/state.json",
        {
            "schema_version": 1,
            "state": "PASS",
            "dataset": "mutagenicity",
            "stage": postprocess.EXPECTED_CONTROLLER_STAGE,
            "exit_code": 0,
            "failures": [],
            "child_pid": 470_261,
            "pid": 470_260,
            "run_id": "mut-exact-multicomponent-test",
            "completed_at": "2026-08-31T09:56:31+00:00",
        },
    )
    receipt = _json(
        tmp_path / "control/adoption.json",
        {
            "schema_version": postprocess.ADOPTION_SCHEMA,
            "status": "PASS",
            "state": "ADOPTED_COMPLETED_SCIENCE",
            "full_root": str(common.resolve()),
            "source_root": str(common.parent.resolve()),
            "source_worker_pid": 470_261,
            "source_worker_active": False,
            "source_worker_exit_code": 0,
            "exactly_zero_active_exact_writers": True,
            "active_exact_writer_pids": [],
            "second_writer_started": False,
            "labels_partition_centroid_radius_coverage_greedy_complete": True,
            "remaining_stages": postprocess.EXPECTED_REMAINING_STAGES,
            "source_controller_state": str(controller.resolve()),
            "source_controller_state_sha256": sha256_file(controller),
            "dbscan_phase": "complete",
            "dbscan_next_offset": 813_595,
            "dbscan_run_manifest_sha256": sha256_file(dbscan_manifest),
            "artifact_sha256": {
                name: sha256_file(path) for name, path in artifacts.items()
            },
        },
    )
    return receipt, common, generation


def _refresh_exact_manifest_hashes(receipt: Path, common: Path) -> None:
    dbscan = common / "external_memory/dbscan/run_manifest.json"
    manifest_path = common / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_memory_artifacts"]["dbscan_manifest_sha256"] = sha256_file(
        dbscan
    )
    _json(manifest_path, manifest)
    adoption = json.loads(receipt.read_text(encoding="utf-8"))
    adoption["dbscan_run_manifest_sha256"] = sha256_file(dbscan)
    adoption["artifact_sha256"]["dbscan/run_manifest.json"] = sha256_file(dbscan)
    adoption["artifact_sha256"]["run_manifest.json"] = sha256_file(manifest_path)
    _json(receipt, adoption)


def _refresh_controller_hash(receipt: Path) -> None:
    adoption = json.loads(receipt.read_text(encoding="utf-8"))
    controller = Path(adoption["source_controller_state"])
    adoption["source_controller_state_sha256"] = sha256_file(controller)
    _json(receipt, adoption)


def test_exact_adoption_is_read_only_and_hash_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, common, generation = _exact_fixture(tmp_path)
    monkeypatch.setattr(
        postprocess, "_validate_common_recourse_completion", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        postprocess,
        "_scan_live_source_writers",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "writable_fd_count": 0,
            "writers": [],
        },
    )
    result = postprocess.validate_exact_adoption(
        adoption_receipt_path=receipt,
        common_root=common,
        source_generation_root=generation,
        proc_root=tmp_path,
    )
    assert result["status"] == "PASS"
    assert result["common_recourse_count"] == 100
    assert result["common_recourse_parameters"] == postprocess.EXPECTED_RECOURSE_PARAMETERS
    assert result["controller_terminal"][
        "controller_and_worker_absent_from_procfs"
    ] is True
    assert result["common_recourse_rerun"] is False
    assert result["dbscan_rerun"] is False
    assert result["pair_store_rerun"] is False
    assert result["second_exact_writer_started"] is False


def test_exact_adoption_rejects_any_second_writer_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, common, generation = _exact_fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["second_writer_started"] = True
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        postprocess, "_validate_common_recourse_completion", lambda **_kwargs: None
    )
    with pytest.raises(postprocess.MutExactPostprocessError, match="second_writer"):
        postprocess.validate_exact_adoption(
            adoption_receipt_path=receipt,
            common_root=common,
            source_generation_root=generation,
            proc_root=tmp_path,
        )


def test_exact_adoption_rejects_parameter_or_dbscan_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        postprocess, "_validate_common_recourse_completion", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        postprocess,
        "_scan_live_source_writers",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "writable_fd_count": 0,
            "writers": [],
        },
    )
    receipt, common, generation = _exact_fixture(tmp_path / "parameters")
    manifest_path = common / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parameters"]["theta"] = 0.2
    _json(manifest_path, manifest)
    _refresh_exact_manifest_hashes(receipt, common)
    with pytest.raises(postprocess.MutExactPostprocessError, match="parameters"):
        postprocess.validate_exact_adoption(
            adoption_receipt_path=receipt,
            common_root=common,
            source_generation_root=generation,
            proc_root=tmp_path / "parameters",
        )

    receipt, common, generation = _exact_fixture(tmp_path / "dbscan")
    dbscan_path = common / "external_memory/dbscan/run_manifest.json"
    dbscan = json.loads(dbscan_path.read_text(encoding="utf-8"))
    dbscan["scientific_identity"]["contract"]["eps"] = 0.03
    dbscan["scientific_identity_sha256"] = postprocess.stable_json_sha256(
        dbscan["scientific_identity"]
    )
    _json(dbscan_path, dbscan)
    _refresh_exact_manifest_hashes(receipt, common)
    with pytest.raises(postprocess.MutExactPostprocessError, match="scientific identity"):
        postprocess.validate_exact_adoption(
            adoption_receipt_path=receipt,
            common_root=common,
            source_generation_root=generation,
            proc_root=tmp_path / "dbscan",
        )


def test_exact_adoption_requires_terminal_controller_and_absent_pids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        postprocess, "_validate_common_recourse_completion", lambda **_kwargs: None
    )
    receipt, common, generation = _exact_fixture(tmp_path / "bad-terminal")
    adoption = json.loads(receipt.read_text(encoding="utf-8"))
    controller_path = Path(adoption["source_controller_state"])
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    controller["state"] = "RUNNING"
    _json(controller_path, controller)
    _refresh_controller_hash(receipt)
    with pytest.raises(postprocess.MutExactPostprocessError, match="completed Mut exact"):
        postprocess.validate_exact_adoption(
            adoption_receipt_path=receipt,
            common_root=common,
            source_generation_root=generation,
            proc_root=tmp_path / "bad-terminal",
        )

    receipt, common, generation = _exact_fixture(tmp_path / "live-worker")
    (tmp_path / "live-worker" / "470261").mkdir()
    with pytest.raises(postprocess.MutExactPostprocessError, match="PID still exists"):
        postprocess.validate_exact_adoption(
            adoption_receipt_path=receipt,
            common_root=common,
            source_generation_root=generation,
            proc_root=tmp_path / "live-worker",
        )


def test_commands_start_at_chemistry_and_pin_all_100_recourses(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    common = tmp_path / "common"
    common.mkdir()
    parity = _json(tmp_path / "parity.json", {"status": "PASS"})
    commands = postprocess.build_postprocess_commands(
        inputs,
        common_root=common,
        parity_path=parity,
        project_commit="a" * 40,
        teacher_sha256="b" * 64,
    )
    assert [stage for stage, *_ in commands] == [
        "chemistry",
        "unified_eval",
        "full_gate",
        "freeze",
    ]
    chemistry = commands[0][1]
    offset = chemistry.index("--expected-medoid-count")
    assert chemistry[offset + 1] == "100"
    flattened = " ".join(arg for _, argv, _, _ in commands for arg in argv)
    assert "run_common_recourse.py" not in flattened
    assert "pair_store" not in flattened
    assert "run_external_dbscan" not in flattened


def test_missing_trace_parity_fails_before_fresh_root_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    parity = _json(tmp_path / "invalid-parity.json", {"status": "BLOCKED"})
    monkeypatch.setattr(
        postprocess,
        "_validate_parity_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Chemistry repair cannot be frozen before trace parity passes")
        ),
    )
    with pytest.raises(ValueError, match="trace parity"):
        postprocess.run_mut_exact_postprocess(
            inputs=inputs,
            exact_adoption_receipt=tmp_path / "not-reached.json",
            common_root=tmp_path / "not-reached",
            trace_parity_path=parity,
            prior_matrix_root=tmp_path / "not-reached-matrix",
            matrix_output_root=tmp_path / "matrix-new",
            resume=False,
            proc_root=tmp_path,
        )
    assert not inputs.output_root.exists()
    assert not (tmp_path / "matrix-new").exists()


def test_trace_parity_must_bind_the_adopted_generation_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    parity = _json(tmp_path / "parity.json", {"status": "PASS"})
    other_generation = tmp_path / "other-generation"
    other_generation.mkdir()
    monkeypatch.setattr(
        postprocess,
        "_validate_parity_gate",
        lambda *_args, **_kwargs: {
            "sha256": "p" * 64,
            "traced_source_root": str(other_generation),
        },
    )
    with pytest.raises(postprocess.MutExactPostprocessError, match="not bound"):
        postprocess.run_mut_exact_postprocess(
            inputs=inputs,
            exact_adoption_receipt=tmp_path / "not-reached.json",
            common_root=tmp_path / "not-reached",
            trace_parity_path=parity,
            prior_matrix_root=tmp_path / "not-reached-matrix",
            matrix_output_root=tmp_path / "matrix-new",
            resume=False,
            proc_root=tmp_path,
        )
    assert not inputs.output_root.exists()


def test_output_paths_must_be_disjoint_from_sources_and_each_other(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(
        postprocess.MutExactPostprocessError, match="OUTPUT_SOURCE_PATH_OVERLAP"
    ):
        postprocess._require_output_path_isolation(
            output_paths={"postprocess": source / "child"},
            protected_paths={"generation": source},
        )
    with pytest.raises(
        postprocess.MutExactPostprocessError, match="OUTPUT_PATH_OVERLAP"
    ):
        postprocess._require_output_path_isolation(
            output_paths={
                "postprocess": tmp_path / "outputs",
                "matrix": tmp_path / "outputs/matrix",
            },
            protected_paths={"generation": source},
        )


def test_resume_rejects_changed_scientific_contract_before_stage_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    inputs.output_root.mkdir()
    parity = _json(tmp_path / "parity.json", {"status": "PASS"})
    _json(inputs.output_root / "upstream_checkout_audit.json", {"actual_commit": "u"})
    _json(
        inputs.output_root / "continuation_resume_contract.json",
        {"schema_version": "frozen-contract"},
    )
    monkeypatch.setattr(
        postprocess,
        "_validate_parity_gate",
        lambda *_args, **_kwargs: {
            "sha256": "p" * 64,
            "traced_source_root": str(inputs.source_generation_root),
        },
    )
    exact_root = tmp_path / "exact/full"
    exact_root.mkdir(parents=True)
    adoption = _file(tmp_path / "control/adoption.json")
    controller = _file(tmp_path / "control/state.json")
    monkeypatch.setattr(
        postprocess,
        "validate_exact_adoption",
        lambda **_kwargs: {
            "common_root": str(exact_root),
            "adoption_receipt_path": str(adoption),
            "adoption_receipt_sha256": "a" * 64,
            "source_controller_state_path": str(controller),
            "source_controller_state_sha256": "z" * 64,
            "common_terminal_sha256": "c" * 64,
            "common_manifest_sha256": "m" * 64,
            "dbscan_scientific_identity_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        postprocess,
        "validate_adopted_generation",
        lambda _inputs: {
            "counterfactual_candidate_count": postprocess.SOURCE_CANDIDATE_COUNT,
            "counterfactuals_sha256_actual": postprocess.SOURCE_PAYLOAD_SHA256,
        },
    )
    monkeypatch.setattr(
        postprocess, "verify_checkout", lambda *_args, **_kwargs: {"actual_commit": "u"}
    )
    prior = {
        "root": tmp_path / "prior",
        "rows": {
            ("Mutagenicity", "ComRecGC"): {"status": "MISSING"},
        },
        "matrix_sha256": "x" * 64,
        "combined_sha256": "y" * 64,
    }
    (tmp_path / "prior").mkdir()
    monkeypatch.setattr(postprocess, "_verify_authority", lambda *_args, **_kwargs: prior)
    monkeypatch.setattr(postprocess, "_git_head", lambda: "g" * 40)
    monkeypatch.setattr(postprocess, "build_postprocess_commands", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        postprocess,
        "_resume_contract",
        lambda **_kwargs: {"schema_version": "changed-contract"},
    )
    launched: list[str] = []
    monkeypatch.setattr(
        postprocess, "_run_stage", lambda **kwargs: launched.append(kwargs["stage"])
    )
    with pytest.raises(
        postprocess.MutExactPostprocessError,
        match="RESUME_SCIENTIFIC_CONTRACT_MISMATCH",
    ):
        postprocess.run_mut_exact_postprocess(
            inputs=inputs,
            exact_adoption_receipt=adoption,
            common_root=exact_root,
            trace_parity_path=parity,
            prior_matrix_root=tmp_path / "prior",
            matrix_output_root=tmp_path / "matrix-new",
            resume=True,
            proc_root=tmp_path,
        )
    assert launched == []


def _row(dataset: str, method: str, *, status: str = "MISSING") -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "status": status,
        "standardized_output_root": None,
        "dataset_hash": None,
        "split_hash": None,
        "oracle_backend": None,
        "oracle_checkpoint": None,
        "oracle_hash": None,
        "molclr_checkpoint_hash": None,
        "distance_line": None,
        "cf_mode": None,
        "threshold_config_hash": None,
        "k_max": None,
        "table2_k": None,
    }


def test_matrix_append_changes_only_mut_comrecgc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = ("AIDS", "Mutagenicity", "BACE", "TasteMolNet")
    methods = ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC")
    rows = {(dataset, method): _row(dataset, method) for dataset in datasets for method in methods}
    shared = {
        "dataset_hash": "d" * 64,
        "split_hash": "s" * 64,
        "oracle_backend": "rf",
        "oracle_checkpoint": "/teacher.pkl",
        "oracle_hash": "o" * 64,
        "molclr_checkpoint_hash": "m" * 64,
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "threshold_config_hash": "t" * 64,
    }
    rows[("Mutagenicity", "Ours")].update(
        status="FROZEN_PASS",
        standardized_output_root=str(tmp_path / "ours"),
        k_max=20,
        table2_k=10,
        **shared,
    )
    (tmp_path / "ours").mkdir()
    prior = {
        "root": tmp_path / "prior",
        "rows": rows,
        "complete": 1,
        "matrix_sha256": "1" * 64,
        "combined_sha256": "2" * 64,
        "matrix": {"cells": list(rows.values())},
    }
    (tmp_path / "prior").mkdir()
    cell = tmp_path / "standardized"
    cell.mkdir()
    (cell / "PASS").write_bytes(b"PASS\n")
    new_rows = {key: dict(value) for key, value in rows.items()}
    new_rows[("Mutagenicity", "ComRecGC")].update(
        status=CellStatus.FROZEN_PASS.value,
        standardized_output_root=str(cell.resolve()),
        k_max=20,
        table2_k=10,
        **shared,
    )
    output = tmp_path / "new-matrix"
    reopened = {
        "root": output,
        "rows": new_rows,
        "complete": 2,
        "matrix_sha256": "3" * 64,
        "combined_sha256": "4" * 64,
        "matrix": {"cells": list(new_rows.values())},
    }

    def _authority(root: Path, expected_complete: int | None = None) -> dict:
        del expected_complete
        return (
            prior
            if Path(root).resolve(strict=False) == (tmp_path / "prior").resolve()
            else reopened
        )

    monkeypatch.setattr(postprocess, "_verify_authority", _authority)
    monkeypatch.setattr(
        postprocess,
        "audit_registry",
        lambda _config: SimpleNamespace(
            matrix_rows=tuple(new_rows.values()),
            matrix_complete_cells=2,
            matrix_total_cells=16,
        ),
    )

    publish_attempts: list[Path] = []

    def _publish(_result: object, destination: Path, *, supplemental_outputs: dict) -> None:
        publish_attempts.append(destination)
        destination.mkdir(parents=True)
        if len(publish_attempts) == 1:
            (destination / "partial.txt").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("simulated matrix writer crash")
        (destination / "append_authority.json").write_bytes(
            supplemental_outputs["append_authority.json"]
        )

    monkeypatch.setattr(postprocess, "write_registry_outputs", _publish)
    monkeypatch.setattr(
        postprocess,
        "atomic_rename_directory_noreplace",
        lambda source, target: Path(source).rename(target),
    )
    with pytest.raises(RuntimeError, match="simulated matrix writer crash"):
        postprocess._append_mut_matrix_authority(
            prior_authority_root=tmp_path / "prior",
            standardized_root=cell,
            output_root=output,
            require_writer_audit=False,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
        )
    assert not output.exists()

    result = postprocess._append_mut_matrix_authority(
        prior_authority_root=tmp_path / "prior",
        standardized_root=cell,
        output_root=output,
        require_writer_audit=False,
        git_identity={"commit": "a" * 40, "tree": "b" * 40},
    )
    assert result["matrix_complete_cells"] == 2
    assert result["appended_cell"] == "Mutagenicity/ComRecGC"
    assert len(publish_attempts) == 2
    assert publish_attempts[0] != publish_attempts[1]
    assert output.is_dir()
    for key, old in rows.items():
        if key != ("Mutagenicity", "ComRecGC"):
            assert new_rows[key] == old

    late = cell / "late-drift.txt"
    late.write_text("drift\n", encoding="utf-8")
    with pytest.raises(postprocess.MutExactPostprocessError, match="source_inventory"):
        postprocess._reopen_existing_matrix_append(
            prior_authority_root=tmp_path / "prior",
            standardized_root=cell,
            output_root=output,
            require_writer_audit=False,
        )
    late.unlink()
    new_rows[("AIDS", "Ours")]["status"] = "FAILED"
    with pytest.raises(postprocess.MutExactPostprocessError, match="non_target_row"):
        postprocess._reopen_existing_matrix_append(
            prior_authority_root=tmp_path / "prior",
            standardized_root=cell,
            output_root=output,
            require_writer_audit=False,
        )


def test_slurm_wrapper_keeps_hpc_contract_and_requires_parity() -> None:
    script = Path(
        "scripts/slurm/run_mut_comrecgc_exact_postprocess_v1.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
        "--proc-root /proc",
        "MUT_TRACE_PARITY:?",
    ):
        assert required in script
