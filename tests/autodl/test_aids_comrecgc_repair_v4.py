from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.autodl import run_four_gpu_recovery_controller as controller
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from scripts.autodl.run_four_gpu_recovery_controller import _reconcile_instance
from src.utils import autodl_aids_comrecgc_repair_v4 as repair
from src.utils.autodl_runtime import build_runtime_layout
from src.utils.autodl_four_by_four_repair import RepairManifestError


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _file(path: Path, value: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    # The controller's no-test-before-freeze validator intentionally rejects any
    # scientific path containing ``test``.  Pytest's own tmp_path embeds the test
    # function name, so keep the synthetic runtime in an independently named
    # private temporary directory and retain its owner for the fixture lifetime.
    # Use a direct system-temporary child rather than pytest's ``tmp_path``.
    # The latter embeds the test function name (which the production
    # no-test-before-freeze validator correctly rejects), while an explicit
    # macOS-only ``/private/tmp`` would make this release test fail on AutoDL.
    temporary = tempfile.TemporaryDirectory(prefix="aids-v4-fixture-")
    fixture_root = Path(temporary.name)
    runtime = fixture_root / "runtime"
    (runtime / "outputs/autodl").mkdir(parents=True)
    (runtime / "locks").mkdir()
    control = runtime / "control"
    namespace = control / repair.SOURCE_NAMESPACE
    (namespace / "manifests").mkdir(parents=True)
    proc = fixture_root / "proc"
    proc.mkdir()
    cgroup = fixture_root / "cgroup"
    cgroup.mkdir()
    project = Path(__file__).resolve().parents[2]
    source_manifest = _json(namespace / "manifests/source-v2.json", {"source": True})
    source_root = namespace / "source-v2"
    source_root.mkdir()
    failed_manifest = _json(namespace / "manifests/source-v3.json", {"failed": True})
    failed_root = namespace / "source-v3"
    failed_root.mkdir()
    failed_output = fixture_root / "failed-output"
    failed_output.mkdir()
    generation = fixture_root / "generation"
    generation.mkdir()
    generation_payload = _file(generation / "counterfactuals.pt", "payload")
    generation_payload_sha = repair.sha256_file(generation_payload)
    generation_manifest = _json(
        generation / "run_manifest.json",
        {
            "counterfactuals_path": str(generation_payload),
            "counterfactuals_sha256": generation_payload_sha,
        },
    )
    threshold = _file(fixture_root / "threshold.json", "{}\n")
    source_outputs = {
        repair.GENERATION_SOURCE_KEY: fixture_root / "source-generation",
        repair.THRESHOLD_SOURCE_KEY: fixture_root / "source-threshold",
    }
    for path in source_outputs.values():
        path.mkdir()
    flock = _file(fixture_root / "bin/flock", "#!/bin/bash\nexit 0\n")
    flock.chmod(0o755)

    monkeypatch.setattr(
        repair,
        "verify_fix_ancestry",
        lambda **kwargs: {
            "required_fix_commit": kwargs["required_fix_commit"],
            "execution_head": "f" * 40,
            "is_ancestor": "true",
        },
    )

    def source_evidence(*, source_key, **_kwargs):
        semantic = (
            {
                "kind": "generation_adoption",
                "dataset": "aids",
                "generation_root": str(generation),
                "generation_payload_claimed_sha256": generation_payload_sha,
            }
            if source_key == repair.GENERATION_SOURCE_KEY
            else {
                "kind": "threshold",
                "dataset": "aids",
                "threshold_contract": str(threshold),
                "threshold_contract_sha256": "b" * 64,
                "threshold_count": 601,
                "theta_star": 0.05,
                "cost_cap": 0.0535,
                "test_used_for_selection": False,
            }
        )
        return {
            "status": "PASS",
            "source_key": source_key,
            "semantic": semantic,
            "external_memory_fix_gate": {"is_ancestor": "true"},
        }

    monkeypatch.setattr(repair, "verify_source", source_evidence)
    monkeypatch.setattr(
        repair,
        "_verify_v3_oom_failure",
        lambda **_kwargs: (
            SimpleNamespace(environment={}),
            {
                "status": "PASS",
                "source_signal": "SIGKILL",
                "source_exit_code": 1,
                "cgroup_oom_jointly_verified": True,
                "scientific_failure": False,
            },
        ),
    )
    science_root = fixture_root / "science"
    science_dirs = {
        key: science_root / key.lower()
        for key in ("COMRECGC_UPSTREAM_ROOT", "DATASET_DIR", "MOLCLR_ROOT")
    }
    for value in science_dirs.values():
        value.mkdir(parents=True)
    science_files = {
        key: _file(science_root / f"{key.lower()}.dat")
        for key in (
            "DATASET_CSV",
            "SOURCE_CSV",
            "TEACHER_PATH",
            "DISTANCE_CHECKPOINT",
            "MOLCLR_CHECKPOINT",
        )
    }
    monkeypatch.setattr(
        repair.v3,
        "_scientific_environment",
        lambda **_kwargs: {
            "AUTODL_PYTHON": "{python}",
            "DATASET": "aids",
            "SOURCE_GENERATION_ROOT": str(generation),
            "THRESHOLDS_PATH": str(threshold),
            "OUTPUT_ROOT": "{task_output}",
            "DEVICE": "cpu",
            "GPU_REQUIRED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "RUN_TASTEMOLNET": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "THETA_STAR": "0.05",
            "COST_CAP": "0.0535",
            **{key: str(value) for key, value in science_dirs.items()},
            **{key: str(value) for key, value in science_files.items()},
        },
    )
    smoke_root = runtime / "outputs/autodl/profiling/aids-real-equivalence"
    smoke_root.mkdir(parents=True)
    smoke_files = {
        "derived_counterfactuals_sha256": _file(
            smoke_root / "diagnostic_generation/counterfactuals.pt", "derived"
        ),
        "legacy_run_manifest_sha256": _file(
            smoke_root / "legacy/run_manifest.json", "{}\n"
        ),
        "external_run_manifest_sha256": _file(
            smoke_root / "external/run_manifest.json", "{}\n"
        ),
        "pair_order_sha256": _file(
            smoke_root / "legacy_intermediates/pairs.npy", "pairs"
        ),
        "recourse_vectors_sha256": _file(
            smoke_root / "legacy_intermediates/vectors.npy", "vectors"
        ),
        "labels_sha256": _file(
            smoke_root / "legacy_intermediates/labels.npy", "labels"
        ),
    }
    selected_rows = _file(
        smoke_root / "external/selected_common_recourses.json", "[]\n"
    )
    external_terminal = _json(
        smoke_root / "external/_RUN_COMPLETE.json",
        {
            "schema_version": "comrecgc_common_recourse_terminal_v2",
            "run_complete": True,
            "common_recourse_engine": "external_memory_exact_v1",
            "artifact_sha256": {"run_manifest.json": "a" * 64},
        },
    )
    smoke_script = _file(fixture_root / "smoke.py", "# diagnostic\n")
    smoke_gate_payload = {
        "schema_version": "aids_comrecgc_real_external_equivalence_v1",
        "status": "PASS",
        "project_commit": "f" * 40,
        "script_path": str(smoke_script),
        "script_sha256": repair.sha256_file(smoke_script),
        "diagnostic_only": True,
        "eligible_for_main_results": False,
        "formal_generation_budget_used": False,
        "full_recourse_parameters": {
            "theta": 0.1,
            "delta": 0.02,
            "recourse_size": 100,
            "cf_size": 100000,
            "cluster_size": 3,
            "seed": 0,
        },
        "parent_limit": 64,
        "candidate_limit": 31,
        "batch_size": 8,
        "source_generation_manifest_sha256": repair.sha256_file(
            generation_manifest
        ),
        "source_counterfactuals_sha256": generation_payload_sha,
        "checks": {key: True for key in repair.EQUIVALENCE_CHECKS},
        "pair_count": 1,
        "selected_rows_sha256": repair.stable_json_sha256([]),
        **{
            field: repair.sha256_file(path)
            for field, path in smoke_files.items()
        },
        "external_terminal_sha256": repair.sha256_file(external_terminal),
    }
    smoke_gate = _json(smoke_root / "equivalence_gate.json", smoke_gate_payload)
    _file(smoke_root / "PASS", "PASS\n")
    spec = _json(
        fixture_root / "spec.json",
        {
            "schema_version": repair.SPEC_SCHEMA,
            "controller_id": repair.CONTROLLER_ID,
            "paper_frozen": True,
            "run_tastemolnet": 0,
            "runtime_root": str(runtime),
            "control_root": str(control),
            "project_root": str(project),
            "python": str(Path(os.sys.executable).resolve()),
            "proc_root": str(proc),
            "cgroup_memory_root": str(cgroup),
            "min_cgroup_free_bytes": repair.MINIMUM_HEADROOM_BYTES,
            "external_max_rss_gb": repair.EXTERNAL_MAX_RSS_GB,
            "external_query_block_size": repair.EXTERNAL_QUERY_BLOCK_SIZE,
            "expected_sklearn_version": repair.EXPECTED_SKLEARN_VERSION,
            "highmem_lock_path": str(
                runtime / "locks/comrecgc_common_recourse_highmem.lock"
            ),
            "flock_bin": str(flock),
            "equivalence_smoke_gate": str(smoke_gate),
            "fresh_output_root": str(runtime / "outputs/autodl/repairs/aids-v4"),
            "source_controller": {
                "manifest": str(source_manifest),
                "root": str(source_root),
            },
            "source_outputs": {key: str(value) for key, value in source_outputs.items()},
            "failed_controller": {
                "manifest": str(failed_manifest),
                "root": str(failed_root),
            },
            "failed_aids_output_root": str(failed_output),
        },
    )
    return {
        "_temporary": temporary,
        "runtime": runtime,
        "control": control,
        "spec": spec,
    }


def test_payload_is_exact_cpu_external_memory_full_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload, summary = repair.build_payload(spec_path=paths["spec"])
    validation = repair.validate_payload(payload)
    assert summary["common_recourse_engine"] == "external_memory_exact_v1"
    assert validation["max_cpu_tasks"] == 1
    assert {task["id"] for task in payload["tasks"]} == {
        repair.GENERATION_GATE_TASK_ID,
        repair.THRESHOLD_GATE_TASK_ID,
        repair.STANDARDIZATION_TASK_ID,
    }
    standard = next(
        task for task in payload["tasks"] if task["id"] == repair.STANDARDIZATION_TASK_ID
    )
    assert standard["resource"] == "cpu"
    assert standard["command"] == [
        "bash",
        "{project_root}/scripts/autodl/run_aids_comrecgc_repair_v4_supervisor.sh",
    ]
    assert standard["environment"]["COMMON_RECOURSE_ENGINE"] == "external_memory_exact_v1"
    assert standard["environment"]["COMRECGC_EXTERNAL_MAX_RSS_GB"] == "96"
    assert standard["environment"]["OMP_NUM_THREADS"] == "1"
    assert standard["environment"]["COMRECGC_COMMON_RECOURSE_RESUME"] == "1"
    assert (
        standard["environment"]["AIDS_COMRECGC_V4_MAX_SAME_ROOT_RESUMES"]
        == "1"
    )
    assert standard["environment"]["AIDS_COMRECGC_V4_TEST_MODE"] == "0"
    contract = payload["aids_comrecgc_repair_v4_contract"]
    assert contract["parameters"] == {
        "theta": 0.1,
        "delta": 0.02,
        "recourse_size": 100,
        "cf_size": 100000,
        "cluster_size": 3,
        "seed": 0,
    }
    assert contract["scientific_budget_reduced"] is False
    assert contract["legacy_roots_mutated"] is False
    assert contract["failed_task_evidence"]["source_signal"] == "SIGKILL"
    assert contract["equivalence_smoke_evidence"]["status"] == "PASS"
    assert set(contract["equivalence_smoke_evidence"]["checks"]) == (
        repair.EQUIVALENCE_CHECKS
    )


def test_builder_rejects_budget_or_license_scope_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    spec = json.loads(paths["spec"].read_text())
    spec["external_max_rss_gb"] = 95
    _json(paths["spec"], spec)
    with pytest.raises(RepairManifestError, match="exactly 96"):
        repair.build_payload(spec_path=paths["spec"])


def test_controller_schema_rejects_test_hook_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload, _summary = repair.build_payload(spec_path=paths["spec"])
    standard = next(
        task
        for task in payload["tasks"]
        if task["id"] == repair.STANDARDIZATION_TASK_ID
    )
    standard["environment"]["AIDS_COMRECGC_V4_TEST_MODE"] = "1"
    with pytest.raises(RepairManifestError, match="environment is incomplete"):
        repair.validate_payload(payload)


def test_builder_rejects_missing_or_false_real_equivalence_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    gate_path = Path(spec["equivalence_smoke_gate"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["checks"]["pair_order_elementwise_exact"] = False
    _json(gate_path, gate)
    with pytest.raises(RepairManifestError, match="nine exact equivalence checks"):
        repair.build_payload(spec_path=paths["spec"])

    paths = _fixture(tmp_path / "missing", monkeypatch)
    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    spec["equivalence_smoke_gate"] = str(tmp_path / "absent-gate.json")
    _json(paths["spec"], spec)
    with pytest.raises(FileNotFoundError):
        repair.build_payload(spec_path=paths["spec"])


def test_builder_requires_complete_crash_resume_core_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    assert (
        repair.EXTERNAL_MEMORY_FIX_COMMIT
        == "d5c1d67339df4b9642beaf2b10908ed92bac30de"
    )

    def ancestry(*, required_fix_commit, **_kwargs):
        if required_fix_commit == repair.EXTERNAL_MEMORY_FIX_COMMIT:
            raise RepairManifestError("complete crash-resume core is not an ancestor")
        return {
            "required_fix_commit": required_fix_commit,
            "execution_head": "f" * 40,
            "is_ancestor": "true",
        }

    monkeypatch.setattr(repair, "verify_fix_ancestry", ancestry)
    with pytest.raises(RepairManifestError, match="crash-resume core"):
        repair.build_payload(spec_path=paths["spec"])

    paths = _fixture(tmp_path / "second", monkeypatch)
    spec = json.loads(paths["spec"].read_text())
    spec["taste"] = {"run": True}
    _json(paths["spec"], spec)
    with pytest.raises(RepairManifestError, match="forbidden"):
        repair.build_payload(spec_path=paths["spec"])


def test_manifest_is_fresh_and_published_at_exact_namespace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    destination = (
        paths["control"]
        / repair.SOURCE_NAMESPACE
        / "manifests"
        / f"{repair.CONTROLLER_ID}.json"
    )
    result = repair.build_manifest(spec_path=paths["spec"], output_path=destination)
    assert result["status"] == "PASS"
    assert load_controller_manifest(destination).controller_id == repair.CONTROLLER_ID
    with pytest.raises(FileExistsError, match="fresh"):
        repair.build_manifest(spec_path=paths["spec"], output_path=destination)


def test_template_and_static_slurm_are_autodl_only_and_cli_synced() -> None:
    root = Path(__file__).resolve().parents[2]
    template = json.loads(
        (root / "configs/autodl/aids_comrecgc_repair_v4.template.json").read_text()
    )
    assert template["controller_id"] == repair.CONTROLLER_ID
    assert template["external_max_rss_gb"] == 96
    assert template["expected_sklearn_version"] == "1.7.2"
    assert template["equivalence_smoke_gate"] == "__FRESH_EQUIVALENCE_GATE_JSON__"
    assert "mutagenicity" not in template
    wrapper = (
        root / "scripts/slurm/build_aids_comrecgc_repair_v4_manifest.sh"
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
    supervisor = (
        root / "scripts/autodl/run_aids_comrecgc_repair_v4_supervisor.sh"
    ).read_text(encoding="utf-8")
    assert "verify-resume-failure" in supervisor
    assert "resume_count >= 1" in supervisor
    assert "AIDS_COMRECGC_REPAIR_V4_SUPERVISOR_PASS" in supervisor
    static_supervisor = (
        root / "scripts/slurm/run_aids_comrecgc_repair_v4_supervisor.sh"
    ).read_text(encoding="utf-8")
    assert "do not submit" in static_supervisor
    assert "--config configs/hpc.yaml" in static_supervisor


def test_same_root_supervisor_gate_allows_only_process_signal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resume-root"
    root.mkdir()
    _json(
        root / "continuation_resume_contract.json",
        {
            "schema_version": "comrecgc_standardized_stage_resume_v1",
            "dataset": "aids",
            "output_root": str(root.resolve()),
            "common_recourse_engine": "external_memory_exact_v1",
            "external_query_block_size": 8,
            "external_max_rss_gb": 96.0,
            "expected_sklearn_version": "1.7.2",
        },
    )
    _json(
        root / "stage_checkpoints/common_recourse.json",
        {
            "schema_version": 2,
            "stage": "common_recourse",
            "status": "FAILED",
            "argv_sha256": "a" * 64,
        },
    )
    _json(
        root / "FAILED.json",
        {
            "status": "FAILED",
            "dataset": "aids",
            "message": "Command died with <Signals.SIGTERM: 15>",
        },
    )
    evidence = repair.verify_same_root_resume_failure(
        output_root=root, exit_code=1
    )
    assert evidence["status"] == "PASS"
    assert evidence["bounded_same_root_resume"] is True

    failure = json.loads((root / "FAILED.json").read_text())
    failure["message"] = "SKLEARN_VERSION_MISMATCH"
    _json(root / "FAILED.json", failure)
    with pytest.raises(RepairManifestError, match="semantic/non-process"):
        repair.verify_same_root_resume_failure(output_root=root, exit_code=1)


@pytest.mark.parametrize(
    ("verify_exit", "succeed_on_second", "expected_count", "expected_status"),
    ((0, True, 2, 0), (1, True, 1, 1), (0, False, 2, 1)),
)
def test_supervisor_is_bounded_same_root_and_pass_last(
    tmp_path: Path,
    verify_exit: int,
    succeed_on_second: bool,
    expected_count: int,
    expected_status: int,
) -> None:
    output = tmp_path / "same-root"
    output.mkdir()
    counter = tmp_path / "count"
    counter.write_text("0\n", encoding="utf-8")
    inner = tmp_path / "inner.sh"
    inner.write_text(
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        'n="$(cat "$AIDS_V4_TEST_COUNT")"\n'
        'n=$((n + 1))\n'
        'printf "%s\\n" "$n" > "$AIDS_V4_TEST_COUNT"\n'
        'if [[ "${AIDS_V4_TEST_SUCCEED_ON_SECOND:-0}" == "1" && "$n" == "2" ]]; then\n'
        '  printf "PASS\\n" > "$OUTPUT_ROOT/PASS"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    inner.chmod(0o755)
    verifier = tmp_path / "verify.py"
    verifier.write_text(
        "import os, sys\n"
        "sys.exit(int(os.environ['AIDS_V4_TEST_VERIFY_EXIT']))\n",
        encoding="utf-8",
    )
    verifier.chmod(0o755)
    supervisor = (
        Path(__file__).resolve().parents[2]
        / "scripts/autodl/run_aids_comrecgc_repair_v4_supervisor.sh"
    )
    completed = subprocess.run(
        ["bash", str(supervisor)],
        env={
            **os.environ,
            "AUTODL_PYTHON": sys.executable,
            "DATASET": "aids",
            "OUTPUT_ROOT": str(output),
            "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
            "COMRECGC_COMMON_RECOURSE_RESUME": "1",
            "AIDS_COMRECGC_V4_MAX_SAME_ROOT_RESUMES": "1",
            "AIDS_COMRECGC_V4_TEST_MODE": "1",
            "AIDS_COMRECGC_V4_TEST_INNER": str(inner),
            "AIDS_COMRECGC_V4_TEST_VERIFY": str(verifier),
            "AIDS_V4_TEST_COUNT": str(counter),
            "AIDS_V4_TEST_VERIFY_EXIT": str(verify_exit),
            "AIDS_V4_TEST_SUCCEED_ON_SECOND": "1" if succeed_on_second else "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == expected_status
    assert int(counter.read_text()) == expected_count
    assert (output / "PASS").exists() is (expected_status == 0)
    if expected_status == 0:
        assert "resumes=1" in completed.stdout
    elif verify_exit:
        assert "not a resumable process loss" in completed.stderr
    else:
        assert "bounded resume exhausted" in completed.stderr


def test_supervisor_rejects_ambient_invalid_test_mode(tmp_path: Path) -> None:
    output = tmp_path / "same-root"
    output.mkdir()
    supervisor = (
        Path(__file__).resolve().parents[2]
        / "scripts/autodl/run_aids_comrecgc_repair_v4_supervisor.sh"
    )
    completed = subprocess.run(
        ["bash", str(supervisor)],
        env={
            **os.environ,
            "AUTODL_PYTHON": sys.executable,
            "DATASET": "aids",
            "OUTPUT_ROOT": str(output),
            "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
            "COMRECGC_COMMON_RECOURSE_RESUME": "1",
            "AIDS_COMRECGC_V4_MAX_SAME_ROOT_RESUMES": "1",
            "AIDS_COMRECGC_V4_TEST_MODE": "unexpected",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 64
    assert "test mode must be exactly 0 or 1" in completed.stderr


def test_controller_restart_reconciles_live_supervisor_without_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload, _summary = repair.build_payload(spec_path=paths["spec"])
    manifest_path = tmp_path / "loaded-manifest.json"
    _json(manifest_path, payload)
    manifest = load_controller_manifest(manifest_path)
    task = manifest.by_id[repair.STANDARDIZATION_TASK_ID]
    project = tmp_path / "controller-project"
    data = tmp_path / "controller-data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    run_id = "aids-v4-live-supervisor"
    run_root = layout.runs_root / run_id
    run_root.mkdir(parents=True)
    _json(
        run_root / "state.json",
        {
            "state": "RUNNING",
            "pid": os.getpid(),
            "child_pid": None,
            "updated_at": "2026-08-23T00:00:00Z",
        },
    )
    worker_identity = {
        "pid": os.getpid(),
        "start_ticks": 424242,
        "command_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        controller,
        "read_process_identity",
        lambda pid: dict(worker_identity) if pid == os.getpid() else None,
    )
    instance = {
        "state": "RUNNING",
        "run_id": run_id,
        "attempt": 0,
        "started_at": "2026-08-23T00:00:00Z",
        "expected_output": task.expected_output,
        "worker_identity": dict(worker_identity),
    }
    _reconcile_instance(
        layout,
        task,
        instance,
        max_transient_retries=0,
    )
    assert instance["state"] == "RUNNING"
    assert instance["run_id"] == run_id
    assert instance["attempt"] == 0
