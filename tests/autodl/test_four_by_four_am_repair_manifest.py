from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.autodl.build_four_by_four_am_repair_manifest import main as am_cli
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
import src.utils.autodl_four_by_four_am_repair as am
from src.utils.autodl_four_by_four_am_repair import (
    MANIFEST_CONTROLLER_ID,
    SOURCE_CONTROLLER_ID,
    SOURCE_DEFINITIONS,
    VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
    build_am_repair_manifest,
    build_am_repair_payload,
    validate_am_repair_payload,
    verify_fix_ancestry,
)
from src.utils.autodl_four_by_four_repair import RepairManifestError, sha256_file


def _json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _text(path: Path, value: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _generation_adoption(output: Path, recovery: Path, dataset: str, name: str) -> None:
    recovery.mkdir(parents=True)
    for filename in (
        "run_manifest.json",
        "_RUN_COMPLETE.json",
        "freeze_only_recovery.json",
        "frozen_payload_closure_audit.json",
        "adoption_manifest.json",
    ):
        _json(recovery / filename, {"fixture": True})
    _text(recovery / "counterfactuals.pt", "large-payload-not-rehashed\n")
    payload_sha = "a" * 64 if dataset == "mutagenicity" else "b" * 64
    evidence = {
        "schema_version": "four_by_four_repair_generation_terminal_v1",
        "status": "PASS",
        "kind": "artifact_terminal",
        "dataset": dataset,
        "source_output_root": str(recovery),
        "closure_member_count": 6,
        "closure_members": [
            "run_manifest.json",
            "_RUN_COMPLETE.json",
            "freeze_only_recovery.json",
            "frozen_payload_closure_audit.json",
            "adoption_manifest.json",
            "counterfactuals.pt",
        ],
        "payload_claimed_sha256": payload_sha,
        "payload_claimed_sha256_cross_manifest_agreement": True,
        "large_payload_sha256_computed": False,
        "live_writer_audit": {
            "procfs_verified": True,
            "writable_fd_count": 0,
            "writers": [],
        },
    }
    output.mkdir(parents=True)
    _json(
        output / "source_adoption.json",
        {
            "schema_version": "four_by_four_repair_source_adoption_v1",
            "status": "PASS",
            "source_name": name,
            "source_evidence": evidence,
        },
    )
    _text(output / "PASS", "PASS\n")


def _threshold(output: Path, source: Path, dataset: str) -> None:
    output.mkdir(parents=True)
    source_hash = sha256_file(_json(source, {"frozen": True}))
    expected_dataset = "Mutagenicity" if dataset == "mutagenicity" else "AIDS"
    thresholds = [0.0535 * index / 600 for index in range(601)]
    _json(
        output / "frozen_threshold_contract.json",
        {
            "status": "PASS",
            "dataset": expected_dataset,
            "cf_mode": "strict_flip",
            "distance_line": "MolCLR-Node-Wasserstein",
            "threshold_source_split": "existing_frozen_protocol",
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "selection_used_test": False,
            "shared_across_methods": True,
            "thresholds": thresholds,
            "theta_star": 0.05,
            "cost_cap": 0.0535,
            "threshold_config_hash": "c" * 64,
            "source_contract": str(source),
            "source_contract_sha256": source_hash,
        },
    )
    _json(
        output / "threshold_adoption_audit.json",
        {
            "schema_version": "frozen_threshold_adoption_audit_v1",
            "status": "PASS",
            "dataset": expected_dataset,
            "source_contract": str(source),
            "source_contract_sha256": source_hash,
            "threshold_count": 601,
            "theta_star": 0.05,
            "cost_cap": 0.0535,
            "test_used_for_selection": False,
            "shared_across_methods": True,
            "failures": [],
        },
    )
    _text(output / "PASS", "PASS\n")


def _scientific_paths(root: Path, dataset: str) -> dict[str, str]:
    for name in ("recovery", "upstream", "dataset", "molclr"):
        (root / dataset / name).mkdir(parents=True, exist_ok=True)
    paths = {
        "SOURCE_GENERATION_ROOT": str(root / dataset / "recovery"),
        "COMRECGC_UPSTREAM_ROOT": str(root / dataset / "upstream"),
        "DATASET_DIR": str(root / dataset / "dataset"),
        "MOLCLR_ROOT": str(root / dataset / "molclr"),
    }
    for key, filename in (
        ("DATASET_CSV", "dataset.csv"),
        ("TEACHER_PATH", "teacher.pkl"),
        ("DISTANCE_CHECKPOINT", "distance.pt"),
        ("MOLCLR_CHECKPOINT", "molclr.pt"),
    ):
        paths[key] = str(_text(root / dataset / filename))
    if dataset == "aids":
        paths["SOURCE_CSV"] = str(_text(root / dataset / "source.csv"))
    return paths


def _task(
    *,
    task_id: str,
    dataset: str,
    stage: str,
    output: Path,
    depends_on: list[str] | None = None,
    environment: dict[str, str] | None = None,
    test: bool = False,
    freezes_selector: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": task_id,
        "dataset": dataset,
        "stage": stage,
        "depends_on": depends_on or [],
        "resource": "gpu" if test else "cpu",
        "priority": 1,
        "data_splits": ["test"] if test else [],
        "manifest_only": not test,
        "command": (
            [
                "bash",
                "{project_root}/scripts/autodl/run_comrecgc_standardized_continuation.sh",
            ]
            if test
            else ["/usr/bin/true"]
        ),
        # Keep source-audit paths free of pytest's own ``test_*`` temporary
        # directory spelling; production controller leakage checks correctly
        # interpret that token as held-out access.
        "input_manifest": (
            str(output / "PASS") if test else "/fixture/source/PASS"
        ),
        "expected_output": str(output.parent / "attempt-{attempt}"),
        "required_output_files": (
            list(am.STANDARDIZED_REQUIRED_FILES) if test else ["PASS"]
        ),
        "required_log_marker": "PASS",
        "environment": environment or {"PYTHONDONTWRITEBYTECODE": "1"},
    }
    if freezes_selector:
        result["freezes_selector"] = True
    if test:
        result["selector_parameters_frozen"] = True
        result["read_only_test"] = True
    return result


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    # Production's leakage guard treats any path component beginning with
    # ``test`` as held-out access.  Pytest's own temporary roots use exactly
    # that spelling, so construct controller paths in an isolated neutral root.
    temporary = tempfile.TemporaryDirectory(prefix="am-v2-fixture-", dir="/private/tmp")
    tmp_path = Path(temporary.name)
    runtime = tmp_path / "runtime"
    (runtime / "outputs/autodl").mkdir(parents=True)
    control = runtime / "control"
    namespace = control / am.SOURCE_NAMESPACE
    manifests = namespace / "manifests"
    manifests.mkdir(parents=True)
    proc = tmp_path / "proc"
    proc.mkdir()

    mut_generation = tmp_path / "repair-v1/mut-generation/attempt-0"
    aids_generation = tmp_path / "repair-v1/aids-generation/attempt-0"
    mut_recovery = tmp_path / "scientific/mutagenicity/recovery"
    aids_recovery = tmp_path / "scientific/aids/recovery"
    _generation_adoption(
        mut_generation, mut_recovery, "mutagenicity", "mut_comrec_generation"
    )
    _generation_adoption(
        aids_generation, aids_recovery, "aids", "aids_comrec_generation"
    )
    mut_threshold = tmp_path / "repair-v1/mut-threshold/attempt-0"
    aids_threshold = tmp_path / "repair-v1/aids-threshold/attempt-0"
    _threshold(mut_threshold, tmp_path / "mut-threshold-source.json", "mutagenicity")
    _threshold(aids_threshold, tmp_path / "aids-threshold-source.json", "aids")

    paths_by_dataset = {
        "mutagenicity": _scientific_paths(tmp_path / "scientific", "mutagenicity"),
        "aids": _scientific_paths(tmp_path / "scientific", "aids"),
    }
    paths_by_dataset["mutagenicity"]["SOURCE_GENERATION_ROOT"] = str(mut_recovery)
    paths_by_dataset["aids"]["SOURCE_GENERATION_ROOT"] = str(aids_recovery)

    source_outputs = {
        "repair_source_mut_comrec_generation": mut_generation,
        "mutagenicity_comrecgc_threshold_freeze": mut_threshold,
        "repair_source_aids_comrec_generation": aids_generation,
        "aids_comrecgc_threshold_freeze": aids_threshold,
    }
    source_tasks: list[dict[str, object]] = []
    for key, definition in SOURCE_DEFINITIONS.items():
        source_tasks.append(
            _task(
                task_id=definition.task_id,
                dataset=(
                    definition.dataset
                    if definition.kind == "threshold"
                    else "repair-source-audit"
                ),
                stage=(
                    "AM_COMRECGC_THRESHOLD_FREEZE"
                    if definition.kind == "threshold"
                    else "FOUR_BY_FOUR_REPAIR_SOURCE_ADOPTION"
                ),
                output=source_outputs[definition.task_id],
                freezes_selector=definition.kind == "threshold",
            )
        )
    for dataset, generation_key, threshold_key in (
        ("mutagenicity", "mut_generation", "mut_threshold"),
        ("aids", "aids_generation", "aids_threshold"),
    ):
        environment = {
            "AUTODL_PYTHON": "{python}",
            "DATASET": dataset,
            **paths_by_dataset[dataset],
            "THRESHOLDS_PATH": (
                "{dep_"
                + SOURCE_DEFINITIONS[threshold_key].task_id
                + "_output}/frozen_threshold_contract.json"
            ),
            "OUTPUT_ROOT": "{task_output}",
            "DEVICE": "cuda:0",
            "RUN_TASTEMOLNET": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        source_tasks.append(
            _task(
                task_id=f"{dataset}_comrecgc_standardized",
                dataset=dataset,
                stage="AM_COMRECGC_HELDOUT_EVAL",
                output=tmp_path / f"failed/{dataset}/attempt-0",
                depends_on=[
                    SOURCE_DEFINITIONS[generation_key].task_id,
                    SOURCE_DEFINITIONS[threshold_key].task_id,
                ],
                environment=environment,
                test=True,
            )
        )

    source_manifest = manifests / f"{SOURCE_CONTROLLER_ID}.json"
    _json(
        source_manifest,
        {
            "schema_version": 1,
            "controller_id": SOURCE_CONTROLLER_ID,
            "paper_frozen": True,
            "runtime": {
                "max_gpus": 4,
                "stable_idle_seconds": 60,
                "sample_interval_seconds": 5,
                "poll_seconds": 60,
                "max_transient_retries": 1,
            },
            "resource_gates": {},
            "tasks": source_tasks,
        },
    )
    source_manifest_sha = sha256_file(source_manifest)
    source_controller_root = namespace / SOURCE_CONTROLLER_ID
    _json(
        source_controller_root / "controller_manifest.json",
        {
            "controller_id": SOURCE_CONTROLLER_ID,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": source_manifest_sha,
        },
    )
    for definition in SOURCE_DEFINITIONS.values():
        output = source_outputs[definition.task_id]
        task_root = source_controller_root / "tasks" / definition.task_id
        _json(
            task_root / "manifest.json",
            {
                "task_id": definition.task_id,
                "controller_manifest_sha256": source_manifest_sha,
                "expected_output": str(output),
            },
        )
        _json(
            task_root / "state.json",
            {
                "task_id": definition.task_id,
                "state": "PASS",
                "instances": {
                    "main": {
                        "state": "PASS",
                        "expected_output": str(output),
                    }
                },
            },
        )
        _json(task_root / "gate.json", {"status": "PASS"})

    head = "e" * 40

    def passing_fix_gate(*, project_root: str | Path, required_fix_commit: str):
        assert Path(project_root).is_dir()
        assert required_fix_commit == VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT
        return {
            "required_fix_commit": required_fix_commit,
            "execution_head": head,
            "is_ancestor": "true",
        }

    monkeypatch.setattr(am, "verify_fix_ancestry", passing_fix_gate)
    spec = _json(
        tmp_path / "repair-spec.json",
        {
            "schema_version": "four_by_four_am_repair_spec_v2",
            "controller_id": MANIFEST_CONTROLLER_ID,
            "paper_frozen": True,
            "run_tastemolnet": 0,
            "runtime_root": str(runtime),
            "control_root": str(control),
            "project_root": str(Path.cwd().resolve()),
            "python": str(Path(os.sys.executable).resolve()),
            "proc_root": str(proc),
            "fresh_output_root": str(
                runtime
                / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs"
                / MANIFEST_CONTROLLER_ID
            ),
            "verify_comrecgc_checkout_safe_git_fix_commit": (
                VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT
            ),
            "source_controller": {
                "manifest": str(source_manifest),
                "root": str(source_controller_root),
            },
            "sources": {
                key: {
                    "task_id": definition.task_id,
                    "output_root": str(source_outputs[definition.task_id]),
                }
                for key, definition in SOURCE_DEFINITIONS.items()
            },
        },
    )
    return {
        "_temporary": temporary,  # keep the temporary root alive for the test
        "runtime": runtime,
        "control": control,
        "proc": proc,
        "spec": spec,
        "source_manifest": source_manifest,
        "source_controller_root": source_controller_root,
        "mut_generation": mut_generation,
        "mut_recovery": mut_recovery,
        "mut_threshold": mut_threshold,
    }


def test_payload_is_exact_six_task_am_graph_with_shared_uuid_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload, summary = build_am_repair_payload(spec_path=paths["spec"])
    validation = validate_am_repair_payload(payload)
    tasks = {task["id"]: task for task in payload["tasks"]}
    expected = {
        *(definition.gate_task_id for definition in SOURCE_DEFINITIONS.values()),
        "mutagenicity_comrecgc_standardized",
        "aids_comrecgc_standardized",
    }
    assert summary["controller_id"] == MANIFEST_CONTROLLER_ID
    assert set(validation["task_ids"]) == expected
    assert validation["task_count"] == 6
    assert payload["runtime"]["max_gpus"] == 4
    assert payload["runtime"]["max_cpu_tasks"] == 2
    assert payload["am_repair_contract"]["shared_gpu_uuid_lock_root"] == str(
        paths["runtime"] / "locks"
    )
    assert payload["am_repair_contract"]["old_continuation_guard_inherited"] is False
    assert "continuation" not in payload
    assert not any(
        token in task_id.lower()
        for task_id in tasks
        for token in ("bace", "gcf", "taste", "export")
    )


def test_standardized_jobs_use_repair_v1_science_and_exact_pass_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload, _summary = build_am_repair_payload(spec_path=paths["spec"])
    tasks = {task["id"]: task for task in payload["tasks"]}
    mut = tasks["mutagenicity_comrecgc_standardized"]
    assert mut["depends_on"] == [
        SOURCE_DEFINITIONS["mut_generation"].gate_task_id,
        SOURCE_DEFINITIONS["mut_threshold"].gate_task_id,
    ]
    assert mut["data_splits"] == ["test"]
    assert mut["selector_parameters_frozen"] is True
    assert mut["read_only_test"] is True
    assert mut["environment"]["SOURCE_GENERATION_ROOT"] == str(paths["mut_recovery"])
    assert mut["environment"]["THRESHOLDS_PATH"] == str(
        paths["mut_threshold"] / "frozen_threshold_contract.json"
    )
    assert mut["environment"]["TEACHER_PATH"].endswith("teacher.pkl")
    threshold_gate = tasks[SOURCE_DEFINITIONS["mut_threshold"].gate_task_id]
    assert threshold_gate["stage"] == "AM_COMRECGC_THRESHOLD_FREEZE"
    assert threshold_gate["freezes_selector"] is True


def test_source_controller_must_be_exact_repair_v1_pass_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    state_path = (
        paths["source_controller_root"]
        / "tasks/repair_source_mut_comrec_generation/state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state"] = "FAILED"
    _json(state_path, state)
    with pytest.raises(RepairManifestError, match="is not PASS"):
        build_am_repair_payload(spec_path=paths["spec"])


def test_generation_adoption_and_threshold_semantics_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    adoption_path = paths["mut_generation"] / "source_adoption.json"
    adoption = json.loads(adoption_path.read_text(encoding="utf-8"))
    adoption["source_evidence"]["dataset"] = "aids"
    _json(adoption_path, adoption)
    with pytest.raises(RepairManifestError, match="generation adoption"):
        build_am_repair_payload(spec_path=paths["spec"])

    paths = _fixture(tmp_path / "threshold", monkeypatch)
    contract_path = paths["mut_threshold"] / "frozen_threshold_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["test_used_for_selection"] = True
    _json(contract_path, contract)
    with pytest.raises(RepairManifestError, match="threshold terminal"):
        build_am_repair_payload(spec_path=paths["spec"])


def test_repair_v1_scientific_generation_root_must_match_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    source_manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    other = tmp_path / "other-recovery"
    other.mkdir()
    for task in source_manifest["tasks"]:
        if task["id"] == "mutagenicity_comrecgc_standardized":
            task["environment"]["SOURCE_GENERATION_ROOT"] = str(other)
    _json(paths["source_manifest"], source_manifest)
    changed_sha = sha256_file(paths["source_manifest"])
    snapshot_path = paths["source_controller_root"] / "controller_manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["source_manifest_sha256"] = changed_sha
    _json(snapshot_path, snapshot)
    for definition in SOURCE_DEFINITIONS.values():
        task_manifest_path = (
            paths["source_controller_root"]
            / "tasks"
            / definition.task_id
            / "manifest.json"
        )
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        task_manifest["controller_manifest_sha256"] = changed_sha
        _json(task_manifest_path, task_manifest)
    with pytest.raises(RepairManifestError, match="scientific contract is invalid"):
        build_am_repair_payload(spec_path=paths["spec"])


def test_live_writer_blocks_source_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    pid = paths["proc"] / "1234"
    (pid / "fd").mkdir(parents=True)
    (pid / "fdinfo").mkdir()
    os.symlink(paths["mut_generation"] / "PASS", pid / "fd/7")
    _text(pid / "fdinfo/7", "flags:\t02\n")
    with pytest.raises(RepairManifestError, match="writer audit failed"):
        build_am_repair_payload(spec_path=paths["spec"])


def test_fresh_output_and_controller_roots_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    Path(spec["fresh_output_root"]).mkdir(parents=True)
    with pytest.raises(RepairManifestError, match="fresh_output_root already exists"):
        build_am_repair_payload(spec_path=paths["spec"])

    paths = _fixture(tmp_path / "controller", monkeypatch)
    (
        paths["control"] / am.SOURCE_NAMESPACE / MANIFEST_CONTROLLER_ID
    ).mkdir(parents=True)
    with pytest.raises(RepairManifestError, match="controller root already exists"):
        build_am_repair_payload(spec_path=paths["spec"])


def test_fix_gate_requires_exact_reviewed_sha_and_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RepairManifestError, match="must equal the reviewed fix"):
        verify_fix_ancestry(project_root=Path.cwd(), required_fix_commit="0" * 40)

    monkeypatch.setattr(am, "_git_head", lambda _root: "f" * 40)
    monkeypatch.setattr(
        am.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(RepairManifestError, match="does not contain"):
        verify_fix_ancestry(
            project_root=Path.cwd(),
            required_fix_commit=VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
        )


def test_build_and_cli_publish_only_exact_fresh_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    destination = (
        paths["control"]
        / am.SOURCE_NAMESPACE
        / "manifests"
        / f"{MANIFEST_CONTROLLER_ID}.json"
    )
    result = build_am_repair_manifest(
        spec_path=paths["spec"], output_path=destination
    )
    assert result["status"] == "PASS"
    assert load_controller_manifest(destination).controller_id == MANIFEST_CONTROLLER_ID
    with pytest.raises(FileExistsError, match="must be fresh"):
        build_am_repair_manifest(spec_path=paths["spec"], output_path=destination)

    paths = _fixture(tmp_path / "cli", monkeypatch)
    assert am_cli(["validate", "--spec", str(paths["spec"])]) == 0
    assert "[FOUR_BY_FOUR_AM_REPAIR_MANIFEST_VALIDATE_PASS]" in capsys.readouterr().out
    destination = (
        paths["control"]
        / am.SOURCE_NAMESPACE
        / "manifests"
        / f"{MANIFEST_CONTROLLER_ID}.json"
    )
    assert am_cli(
        ["build", "--spec", str(paths["spec"]), "--output", str(destination)]
    ) == 0
    assert "[FOUR_BY_FOUR_AM_REPAIR_MANIFEST_BUILD_PASS]" in capsys.readouterr().out


def test_verify_source_cli_rechecks_ancestry_and_publishes_pass_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "runtime-source-gate"
    assert am_cli(
        [
            "verify-source",
            "--source-key",
            "mut_generation",
            "--source-manifest",
            str(paths["source_manifest"]),
            "--source-controller-root",
            str(paths["source_controller_root"]),
            "--control-root",
            str(paths["control"]),
            "--expected-output-root",
            str(paths["mut_generation"]),
            "--project-root",
            str(Path.cwd()),
            "--required-fix-commit",
            VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
            "--proc-root",
            str(paths["proc"]),
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert (output / "source_gate.json").is_file()
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"
    assert "[FOUR_BY_FOUR_AM_REPAIR_SOURCE_GATE_PASS]" in capsys.readouterr().out


def test_template_and_paired_slurm_are_autodl_safe() -> None:
    root = Path(__file__).resolve().parents[2]
    template = json.loads(
        (root / "configs/autodl/four_by_four_am_repair_v2.template.json").read_text()
    )
    assert template["controller_id"] == MANIFEST_CONTROLLER_ID
    assert template["verify_comrecgc_checkout_safe_git_fix_commit"] == (
        VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT
    )
    assert "bace" not in template
    assert "gcf" not in template
    assert "taste" not in template
    assert "continuation" not in template

    wrapper = (
        root / "scripts/slurm/build_four_by_four_am_repair_manifest.sh"
    ).read_text(encoding="utf-8")
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "do not submit",
    ):
        assert token in wrapper
