from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

from scripts.autodl.build_four_by_four_repair_manifest import main as repair_cli
from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerError,
    load_controller_manifest,
)
from src.utils.autodl_four_by_four_repair import (
    MANIFEST_CONTROLLER_ID,
    RepairManifestError,
    SOURCE_TASK_IDS,
    build_repair_manifest,
    build_repair_payload,
    publish_source_adoption,
    sha256_file,
    validate_repair_payload,
    verify_comrecgc_generation_terminal,
    verify_controller_terminal,
)


HISTORICAL_RECOVERY_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "fixtures/autodl/comrecgc_recovery_terminal_contract.json"
)


def _json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _text(path: Path, value: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _generation_terminal(root: Path, dataset: str) -> None:
    root.mkdir(parents=True)
    payload = _text(root / "counterfactuals.pt", "historical-payload-fixture\n")
    payload_sha256 = sha256_file(payload)
    payload_bytes = payload.stat().st_size
    _json(
        root / "run_manifest.json",
        {
            "dataset": dataset,
            "mode": "full",
            "generation_mode": "adopted_read_only_cache",
            "run_complete": True,
            "freeze_only_recovery": True,
            "algorithm_rerun": False,
            "counterfactuals_path": str(payload),
            "counterfactuals_sha256": payload_sha256,
            "counterfactuals_bytes": payload_bytes,
        },
    )
    _json(
        root / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "freeze_only_recovery": True,
            "counterfactuals_sha256": payload_sha256,
        },
    )
    _json(
        root / "freeze_only_recovery.json",
        {
            "schema_version": "comrecgc_completed_generation_freeze_audit_v4",
            "dataset": dataset,
            "FREEZE_ONLY_RECOVERY_SAFE": True,
            "fresh_rerun_required": False,
            "recovery_completed": True,
            "algorithm_rerun": False,
            "output_dir": str(root),
            "counterfactuals_sha256": payload_sha256,
        },
    )
    _json(
        root / "frozen_payload_closure_audit.json",
        {
            "closure_complete": True,
            "post_write_reload_verified": True,
            "payload_checksum": payload_sha256,
            "payload_bytes": payload_bytes,
        },
    )
    _json(
        root / "adoption_manifest.json",
        {"generation_mode": "adopted_read_only_cache"},
    )


def _source_terminal(root: Path, name: str) -> None:
    root.mkdir(parents=True)
    if name == "bace_b14":
        _json(root / "FINAL_PASS.json", {"status": "PASS"})
    else:
        _json(root / "frozen_candidate_manifest.json", {"status": "PASS"})
        _json(root / "matched_thresholds.json", {"status": "PASS"})
        _text(root / "export/selected_top20.csv", "rank,smiles\n1,C\n")
        _text(root / "schema_reference/table2_ours_k10.csv", "method,k\nOurs,10\n")
    _text(root / "PASS", "PASS\n")


def _source_controller(base: Path, b14: Path, freeze: Path) -> tuple[Path, Path]:
    manifest_path = base / "source-namespace/manifests/source-v1.json"
    tasks = []
    for task_id, output in (
        ("bace_b14_frozen", b14),
        ("mut_gcf_legacy_freeze", freeze),
    ):
        tasks.append(
            {
                "id": task_id,
                "dataset": "source-audit",
                "stage": "SOURCE_TERMINAL",
                "depends_on": [],
                "resource": "cpu",
                "priority": 1,
                "data_splits": [],
                "manifest_only": True,
                "command": ["/usr/bin/true"],
                "input_manifest": str(output / "PASS"),
                "expected_output": str(output),
                "required_output_files": ["PASS"],
                "required_log_marker": "PASS",
                "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            }
        )
    _json(
        manifest_path,
        {
            "schema_version": 1,
            "controller_id": "source-v1",
            "paper_frozen": True,
            "runtime": {
                "max_gpus": 4,
                "stable_idle_seconds": 60,
                "sample_interval_seconds": 5,
                "poll_seconds": 60,
                "max_transient_retries": 0,
            },
            "resource_gates": {},
            "tasks": tasks,
        },
    )
    manifest_sha = sha256_file(manifest_path)
    controller_root = base / "source-namespace/source-v1"
    _json(
        controller_root / "controller_manifest.json",
        {
            "controller_id": "source-v1",
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": manifest_sha,
        },
    )
    for task in tasks:
        task_id = task["id"]
        output = b14 if task_id == "bace_b14_frozen" else freeze
        task_root = controller_root / "tasks" / task_id
        _json(
            task_root / "manifest.json",
            {
                "task_id": task_id,
                "controller_manifest_sha256": manifest_sha,
                "expected_output": str(output),
            },
        )
        _json(
            task_root / "state.json",
            {
                "task_id": task_id,
                "state": "PASS",
                "instances": {
                    "main": {
                        "instance_id": "main",
                        "state": "PASS",
                        "expected_output": str(output),
                    }
                },
            },
        )
        _json(task_root / "gate.json", {"task_id": task_id, "status": "PASS"})
    return manifest_path, controller_root


def _fixture() -> tuple[Path, Path, dict[str, Path]]:
    root = Path(tempfile.mkdtemp(prefix="repair-fixture-", dir="/private/tmp"))
    proc = root / "proc"
    proc.mkdir()
    runtime = root / "runtime"
    (runtime / "outputs/autodl").mkdir(parents=True)
    b14 = root / "old-output/bace-b14/attempt-0"
    freeze = root / "old-output/mut-gcf-freeze/attempt-0"
    _source_terminal(b14, "bace_b14")
    _source_terminal(freeze, "mut_gcf_freeze")
    source_manifest, source_controller_root = _source_controller(root, b14, freeze)
    mut_generation = root / "old-output/mut-comrec-generation"
    aids_generation = root / "old-output/aids-comrec-generation"
    _generation_terminal(mut_generation, "mutagenicity")
    _generation_terminal(aids_generation, "aids")

    paths: dict[str, Path] = {
        "runtime": runtime,
        "proc": proc,
        "b14": b14,
        "freeze": freeze,
        "source_manifest": source_manifest,
        "source_controller_root": source_controller_root,
        "mut_generation": mut_generation,
        "aids_generation": aids_generation,
    }
    for name in (
        "bace_dataset",
        "molclr",
        "official",
        "mut_dataset",
        "aids_dataset",
    ):
        paths[name] = root / name
        paths[name].mkdir()
    checkpoint = root / "bace-gine"
    checkpoint.mkdir()
    _json(checkpoint / "model_card.json", {"status": "PASS"})
    paths["checkpoint"] = checkpoint
    for name in (
        "bace_calibration.csv",
        "bace_test.csv",
        "molclr/model.pth",
        "neurosed.pt",
        "mut_calibration.csv",
        "mut_test.csv",
        "mut_rf.pkl",
        "aids_rf.pkl",
        "mut_distance.pt",
        "aids_distance.pt",
        "mut_dataset.csv",
        "aids_dataset.csv",
        "aids_source.csv",
        "mut_threshold.json",
        "aids_threshold.json",
    ):
        paths[name] = _text(root / name)

    spec = {
        "schema_version": "four_by_four_repair_spec_v1",
        "controller_id": MANIFEST_CONTROLLER_ID,
        "paper_frozen": True,
        "run_tastemolnet": 0,
        "runtime_root": str(runtime),
        "project_root": str(Path.cwd().resolve()),
        "python": str(Path(sys.executable).resolve()),
        "proc_root": str(proc),
        "fresh_output_root": str(runtime / "outputs/autodl/repairs/repair-v1"),
        "required_execution_commits": [],
        "source_controller": {
            "manifest": str(source_manifest),
            "root": str(source_controller_root),
        },
        "sources": {
            "bace_b14": {
                "task_id": "bace_b14_frozen",
                "output_root": str(b14),
            },
            "mut_gcf_freeze": {
                "task_id": "mut_gcf_legacy_freeze",
                "output_root": str(freeze),
            },
            "mut_comrec_generation": {"output_root": str(mut_generation)},
            "aids_comrec_generation": {"output_root": str(aids_generation)},
        },
        "bace": {
            "gnn_checkpoint": str(checkpoint),
            "dataset_dir": str(paths["bace_dataset"]),
            "calibration_split": str(paths["bace_calibration.csv"]),
            "test_split": str(paths["bace_test.csv"]),
            "molclr_root": str(paths["molclr"]),
            "molclr_checkpoint": str(paths["molclr/model.pth"]),
            "neurosed_checkpoint": str(paths["neurosed.pt"]),
            "comrecgc_official_root": str(paths["official"]),
            "omp_threads": 4,
            "expected_hashes": {
                "dataset": "a" * 64,
                "split": "b" * 64,
                "molclr": "c" * 64,
                "threshold": "d" * 64,
            },
        },
        "mut_gcf": {
            "calibration_csv": str(paths["mut_calibration.csv"]),
            "test_csv": str(paths["mut_test.csv"]),
            "teacher_path": str(paths["mut_rf.pkl"]),
            "molclr_root": str(paths["molclr"]),
            "molclr_checkpoint": str(paths["molclr/model.pth"]),
        },
        "am_comrecgc": {
            "shared": {
                "upstream_root": str(paths["official"]),
                "molclr_root": str(paths["molclr"]),
                "molclr_checkpoint": str(paths["molclr/model.pth"]),
            },
            "mutagenicity": {
                "dataset_dir": str(paths["mut_dataset"]),
                "dataset_csv": str(paths["mut_dataset.csv"]),
                "teacher_path": str(paths["mut_rf.pkl"]),
                "distance_checkpoint": str(paths["mut_distance.pt"]),
                "thresholds_source": str(paths["mut_threshold.json"]),
            },
            "aids": {
                "dataset_dir": str(paths["aids_dataset"]),
                "dataset_csv": str(paths["aids_dataset.csv"]),
                "source_csv": str(paths["aids_source.csv"]),
                "teacher_path": str(paths["aids_rf.pkl"]),
                "distance_checkpoint": str(paths["aids_distance.pt"]),
                "thresholds_source": str(paths["aids_threshold.json"]),
            },
        },
    }
    spec_path = _json(root / "repair-spec.json", spec)
    return root, spec_path, paths


@pytest.fixture
def repair_fixture():
    root, spec, paths = _fixture()
    try:
        yield root, spec, paths
    finally:
        shutil.rmtree(root)


def test_repair_payload_is_bounded_loader_valid_and_uses_shared_uuid_locks(repair_fixture):
    _root, spec, paths = repair_fixture
    payload, summary = build_repair_payload(spec_path=spec)
    validation = validate_repair_payload(payload)
    task_ids = set(validation["task_ids"])

    assert summary["controller_id"] == MANIFEST_CONTROLLER_ID
    assert payload["runtime"]["max_gpus"] == 4
    assert payload["runtime"]["max_cpu_tasks"] == 2
    assert payload["runtime"]["keep_alive_when_blocked"] is True
    assert "continuation" not in payload
    assert payload["repair_contract"]["old_v2_continuation_lock_inherited"] is False
    assert payload["repair_contract"]["shared_gpu_uuid_lock_root"] == str(
        paths["runtime"] / "locks"
    )

    assert {
        "bace_comrecgc_train_generation",
        "bace_comrecgc_train_common_recourse",
        "bace_comrecgc_train_candidates",
        "bace_comrecgc_selection",
        "bace_comrecgc_final_freeze",
        "bace_comrecgc_standardized",
        "bace_ours_standardized",
        "mut_gcf_legacy_calibration",
        "mut_gcf_legacy_heldout",
        "mut_gcf_legacy_standardized",
        "mutagenicity_comrecgc_threshold_freeze",
        "mutagenicity_comrecgc_standardized",
        "aids_comrecgc_threshold_freeze",
        "aids_comrecgc_standardized",
        *SOURCE_TASK_IDS.values(),
    }.issubset(task_ids)
    assert not any(task_id.startswith("tastemolnet") for task_id in task_ids)
    assert "four_by_four_main_results_export" not in task_ids
    assert "four_by_four_final_matrix_audit" not in task_ids
    assert not any(task_id.startswith("bace_b1") for task_id in task_ids)
    assert validation["test_boundary_validated"] is True


def test_repair_bace_ours_is_artifact_only_and_comrecgc_is_generic_native_chain(
    repair_fixture,
):
    _root, spec, paths = repair_fixture
    payload, _summary = build_repair_payload(spec_path=spec)
    tasks = {task["id"]: task for task in payload["tasks"]}
    ours = tasks["bace_ours_standardized"]
    assert ours["depends_on"] == [SOURCE_TASK_IDS["bace_b14"]]
    assert str(paths["b14"]) in ours["command"]
    assert ours["resource"] == "cpu"
    assert ours["manifest_only"] is True

    generation = tasks["bace_comrecgc_train_generation"]
    assert generation["resource"] == "gpu"
    assert "--route" in generation["command"]
    assert "project" in generation["command"]
    assert "{task_output}" in generation["command"]
    assert str(paths["checkpoint"]) in generation["command"]
    assert tasks["bace_comrecgc_standardized"]["depends_on"] == [
        "bace_comrecgc_final_freeze"
    ]


def test_mut_gcf_test_cannot_bypass_fresh_calibration_freeze(repair_fixture):
    _root, spec, _paths = repair_fixture
    payload, _summary = build_repair_payload(spec_path=spec)
    for task in payload["tasks"]:
        if task["id"] == "mut_gcf_legacy_heldout":
            task["depends_on"] = [SOURCE_TASK_IDS["mut_gcf_freeze"]]
    with pytest.raises(ControllerError, match="frozen B12/AM selector dependency"):
        validate_repair_payload(payload)


def test_controller_terminal_rejects_failed_state_and_output_mismatch(repair_fixture):
    _root, _spec, paths = repair_fixture
    kwargs = {
        "source_manifest": paths["source_manifest"],
        "source_controller_root": paths["source_controller_root"],
        "task_id": "bace_b14_frozen",
        "expected_output_root": paths["b14"],
        "required_files": ("FINAL_PASS.json", "PASS"),
        "proc_root": paths["proc"],
    }
    evidence = verify_controller_terminal(**kwargs)
    assert evidence["status"] == "PASS"

    state_path = (
        paths["source_controller_root"]
        / "tasks/bace_b14_frozen/state.json"
    )
    state = json.loads(state_path.read_text())
    state["state"] = "FAILED"
    _json(state_path, state)
    with pytest.raises(RepairManifestError, match="is not PASS"):
        verify_controller_terminal(**kwargs)

    state["state"] = "PASS"
    _json(state_path, state)
    other = paths["b14"].parent / "other-attempt"
    _source_terminal(other, "bace_b14")
    with pytest.raises(RepairManifestError, match="output mismatch"):
        verify_controller_terminal(**{**kwargs, "expected_output_root": other})


def test_source_adoption_rejects_live_writable_descriptor(repair_fixture):
    _root, _spec, paths = repair_fixture
    pid = paths["proc"] / "1234"
    (pid / "fd").mkdir(parents=True)
    (pid / "fdinfo").mkdir()
    os.symlink(paths["b14"] / "PASS", pid / "fd/7")
    _text(pid / "fdinfo/7", "flags:\t02\n")
    with pytest.raises(RepairManifestError, match="writer audit failed"):
        verify_controller_terminal(
            source_manifest=paths["source_manifest"],
            source_controller_root=paths["source_controller_root"],
            task_id="bace_b14_frozen",
            expected_output_root=paths["b14"],
            required_files=("FINAL_PASS.json", "PASS"),
            proc_root=paths["proc"],
        )


def test_generation_terminal_is_small_closure_only_and_publishes_fresh_adoption(
    repair_fixture,
):
    root, _spec, paths = repair_fixture
    evidence = verify_comrecgc_generation_terminal(
        dataset="mutagenicity",
        expected_output_root=paths["mut_generation"],
        proc_root=paths["proc"],
    )
    assert evidence["status"] == "PASS"
    assert evidence["large_payload_sha256_computed"] is False
    assert evidence["closure_member_count"] == 6
    assert evidence["closure_members"] == [
        "run_manifest.json",
        "_RUN_COMPLETE.json",
        "freeze_only_recovery.json",
        "frozen_payload_closure_audit.json",
        "adoption_manifest.json",
        "counterfactuals.pt",
    ]
    assert not (paths["mut_generation"] / "PASS").exists()
    assert not (paths["mut_generation"] / "fresh_recovery_audit.json").exists()
    output = root / "adoption-output"
    payload = publish_source_adoption(
        name="mut_comrec_generation", evidence=evidence, output_dir=output
    )
    assert payload["status"] == "PASS"
    assert (output / "source_adoption.json").is_file()
    assert (output / "PASS").read_text() == "PASS\n"
    with pytest.raises(FileExistsError, match="must be fresh"):
        publish_source_adoption(
            name="mut_comrec_generation", evidence=evidence, output_dir=output
        )


def test_historical_recovery_contract_accepts_no_bare_pass_or_registry(
    repair_fixture,
):
    _root, _spec, paths = repair_fixture
    fixture = json.loads(HISTORICAL_RECOVERY_CONTRACT.read_text(encoding="utf-8"))
    historical = fixture["historical_roots"]
    contract = fixture["terminal_contract"]

    assert fixture["schema_version"] == (
        "comrecgc_historical_recovery_terminal_contract_v1"
    )
    assert [row["run_id"] for row in historical] == [
        "20260822T025620Z-mut-lineage-v3-6ddd743",
        "20260822T020238Z-aids-lineage-v2-6ddd743",
    ]
    assert contract["terminal_registry_required_by_builder"] is False
    assert contract["forbidden_bare_terminal"] == "PASS"

    roots = {
        "mutagenicity": paths["mut_generation"],
        "aids": paths["aids_generation"],
    }
    for historical_root in historical:
        dataset = historical_root["dataset"]
        source = roots[dataset]
        assert not (source / "PASS").exists()
        assert not (source / "fresh_recovery_audit.json").exists()
        for dotted_field, expected in contract["required_values"].items():
            document, field = dotted_field.split(".", 1)
            payload = json.loads((source / f"{document}.json").read_text())
            assert payload[field] == expected
        evidence = verify_comrecgc_generation_terminal(
            dataset=dataset,
            expected_output_root=source,
            proc_root=paths["proc"],
        )
        assert set(evidence["required_files"]) == set(
            contract["required_small_manifests"]
        )
        assert evidence["payload_stat"]["path"] == str(
            (source / contract["large_payload"]).resolve()
        )
        assert evidence["large_payload_sha256_computed"] is False


def test_generation_terminal_rejects_cross_manifest_payload_claim_mismatch(
    repair_fixture,
):
    _root, _spec, paths = repair_fixture
    complete_path = paths["aids_generation"] / "_RUN_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["counterfactuals_sha256"] = "0" * 64
    _json(complete_path, complete)
    with pytest.raises(RepairManifestError, match="not closed"):
        verify_comrecgc_generation_terminal(
            dataset="aids",
            expected_output_root=paths["aids_generation"],
            proc_root=paths["proc"],
        )


def test_build_publishes_once_and_preserves_exact_source_anchors(repair_fixture):
    root, spec, paths = repair_fixture
    destination = root / "control/repair-v1.json"
    result = build_repair_manifest(spec_path=spec, output_path=destination)
    assert result["status"] == "PASS"
    manifest = load_controller_manifest(destination)
    assert manifest.controller_id == MANIFEST_CONTROLLER_ID
    source_task = manifest.by_id[SOURCE_TASK_IDS["bace_b14"]]
    assert str(paths["b14"]) in source_task.command
    assert manifest.runtime["max_cpu_tasks"] == 2
    with pytest.raises(FileExistsError, match="must be fresh"):
        build_repair_manifest(spec_path=spec, output_path=destination)


def test_build_fails_closed_if_fresh_root_exists(repair_fixture):
    _root, spec, _paths = repair_fixture
    payload = json.loads(spec.read_text())
    Path(payload["fresh_output_root"]).mkdir(parents=True)
    with pytest.raises(RepairManifestError, match="already exists"):
        build_repair_payload(spec_path=spec)


def test_thin_cli_validates_and_builds_the_same_manifest(repair_fixture, capsys):
    root, spec, _paths = repair_fixture
    assert repair_cli(["validate", "--spec", str(spec)]) == 0
    assert "[FOUR_BY_FOUR_REPAIR_MANIFEST_VALIDATE_PASS]" in capsys.readouterr().out
    destination = root / "control/cli-repair.json"
    assert repair_cli(
        ["build", "--spec", str(spec), "--output", str(destination)]
    ) == 0
    output = capsys.readouterr().out
    assert "[FOUR_BY_FOUR_REPAIR_MANIFEST_BUILD_PASS]" in output
    assert load_controller_manifest(destination).controller_id == MANIFEST_CONTROLLER_ID
