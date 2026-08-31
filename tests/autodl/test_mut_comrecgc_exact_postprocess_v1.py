from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.autodl.run_comrecgc_standardized_continuation import ContinuationInputs
from src.baselines.comrecgc.contracts import sha256_file
from src.eval.four_by_four_registry import CellStatus
from src.utils import autodl_mut_comrecgc_exact_postprocess_v1 as postprocess


def _json(path: Path, value: dict) -> Path:
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
    dbscan_manifest = _json(
        common / "external_memory/dbscan/run_manifest.json",
        {
            "status": "PASS",
            "run_complete": True,
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
            common / "selected_common_recourses.json", {"selected": []}
        ),
        "dbscan/run_manifest.json": dbscan_manifest,
        "dbscan/labels.npy": _file(common / "external_memory/dbscan/labels.npy"),
    }
    controller = _json(tmp_path / "control/state.json", {"status": "PASS"})
    receipt = _json(
        tmp_path / "control/adoption.json",
        {
            "schema_version": postprocess.ADOPTION_SCHEMA,
            "status": "PASS",
            "state": "ADOPTED_COMPLETED_SCIENCE",
            "full_root": str(common.resolve()),
            "source_worker_active": False,
            "source_worker_exit_code": 0,
            "exactly_zero_active_exact_writers": True,
            "active_exact_writer_pids": [],
            "second_writer_started": False,
            "labels_partition_centroid_radius_coverage_greedy_complete": True,
            "remaining_stages": postprocess.EXPECTED_REMAINING_STAGES,
            "source_controller_state": str(controller.resolve()),
            "source_controller_state_sha256": sha256_file(controller),
            "artifact_sha256": {
                name: sha256_file(path) for name, path in artifacts.items()
            },
        },
    )
    return receipt, common, generation


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
        "_validate_parity",
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
        "_validate_parity",
        lambda *_args, **_kwargs: {"sha256": "p" * 64},
    )
    monkeypatch.setattr(
        postprocess,
        "validate_exact_adoption",
        lambda **_kwargs: {
            "common_root": str(tmp_path),
            "adoption_receipt_path": str(tmp_path / "adoption.json"),
            "adoption_receipt_sha256": "a" * 64,
            "common_terminal_sha256": "c" * 64,
            "common_manifest_sha256": "m" * 64,
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
    }
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
            exact_adoption_receipt=tmp_path / "adoption.json",
            common_root=tmp_path,
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
        return reopened if Path(root).resolve(strict=False) == output else prior

    monkeypatch.setattr(postprocess, "_verify_authority", _authority)
    monkeypatch.setattr(postprocess, "_inventory", lambda _root: {"PASS": {}})
    monkeypatch.setattr(
        postprocess,
        "audit_registry",
        lambda _config: SimpleNamespace(
            matrix_rows=tuple(new_rows.values()),
            matrix_complete_cells=2,
            matrix_total_cells=16,
        ),
    )

    def _publish(_result: object, destination: Path, *, supplemental_outputs: dict) -> None:
        destination.mkdir(parents=True)
        (destination / "append_authority.json").write_bytes(
            supplemental_outputs["append_authority.json"]
        )

    monkeypatch.setattr(postprocess, "write_registry_outputs", _publish)
    result = postprocess._append_mut_matrix_authority(
        prior_authority_root=tmp_path / "prior",
        standardized_root=cell,
        output_root=output,
        require_writer_audit=False,
        git_identity={"commit": "a" * 40, "tree": "b" * 40},
    )
    assert result["matrix_complete_cells"] == 2
    assert result["appended_cell"] == "Mutagenicity/ComRecGC"
    for key, old in rows.items():
        if key != ("Mutagenicity", "ComRecGC"):
            assert new_rows[key] == old


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
        "MUT_TRACE_PARITY:?",
    ):
        assert required in script
