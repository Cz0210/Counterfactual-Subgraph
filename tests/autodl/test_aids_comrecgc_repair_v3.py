from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import pytest

from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
import src.utils.autodl_aids_comrecgc_repair_v3 as repair
from src.utils.autodl_aids_comrecgc_repair_v3 import (
    CONTROLLER_ID,
    GENERATION_GATE_TASK_ID,
    GENERATION_SOURCE_KEY,
    MINIMUM_HEADROOM_BYTES,
    SOURCE_CONTROLLER_ID,
    SOURCE_NAMESPACE,
    STANDARDIZATION_TASK_ID,
    THRESHOLD_GATE_TASK_ID,
    THRESHOLD_SOURCE_KEY,
    build_manifest,
    build_payload,
    validate_payload,
)
from src.utils.autodl_four_by_four_am_repair import (
    VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
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


def _source_task(
    task_id: str, output: Path, *, freezes_selector: bool = False
) -> dict[str, Any]:
    return {
        "id": task_id,
        "dataset": "aids" if freezes_selector else "repair-source-audit",
        "stage": (
            "AM_COMRECGC_THRESHOLD_FREEZE"
            if freezes_selector
            else "FOUR_BY_FOUR_AM_REPAIR_SOURCE_ADOPTION"
        ),
        "depends_on": [],
        "resource": "cpu",
        "priority": 1,
        "data_splits": [],
        "manifest_only": True,
        "freezes_selector": freezes_selector,
        "command": ["/usr/bin/true"],
        "input_manifest": str(output / "source_gate.json"),
        "expected_output": str(output),
        "required_output_files": ["source_gate.json", "PASS"],
        "required_log_marker": "PASS",
        "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
    }


def _scientific_paths(root: Path) -> dict[str, str]:
    for name in ("generation", "upstream", "dataset", "molclr"):
        (root / name).mkdir(parents=True, exist_ok=True)
    values = {
        "SOURCE_GENERATION_ROOT": str(root / "generation"),
        "COMRECGC_UPSTREAM_ROOT": str(root / "upstream"),
        "DATASET_DIR": str(root / "dataset"),
        "MOLCLR_ROOT": str(root / "molclr"),
        "DATASET": "aids",
        "RUN_TASTEMOLNET": "0",
        "DEVICE": "cuda:0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "THETA_STAR": "0.05",
        "COST_CAP": "0.0535",
    }
    for key, name in (
        ("DATASET_CSV", "dataset.csv"),
        ("SOURCE_CSV", "source.csv"),
        ("TEACHER_PATH", "teacher.pkl"),
        ("DISTANCE_CHECKPOINT", "distance.pt"),
        ("MOLCLR_CHECKPOINT", "molclr.pt"),
    ):
        values[key] = str(_text(root / name))
    return values


def _fixture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    temporary = tempfile.TemporaryDirectory(prefix="aids-v3-fixture-", dir="/private/tmp")
    root = Path(temporary.name)
    runtime = root / "runtime"
    (runtime / "outputs/autodl").mkdir(parents=True)
    (runtime / "locks").mkdir()
    control = runtime / "control"
    namespace = control / SOURCE_NAMESPACE
    manifests = namespace / "manifests"
    manifests.mkdir(parents=True)
    proc_root = root / "proc"
    proc_root.mkdir()
    cgroup = root / "cgroup"
    cgroup.mkdir()
    _text(cgroup / "memory.limit_in_bytes", str(480 * 1024**3) + "\n")
    _text(cgroup / "memory.max_usage_in_bytes", str(480 * 1024**3 + 4096) + "\n")
    _text(cgroup / "memory.failcnt", "1400\n")
    _text(cgroup / "memory.oom_control", "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n")
    _text(cgroup / "memory.usage_in_bytes", str(8 * 1024**3) + "\n")
    flock_bin = _text(root / "bin/flock", "#!/bin/bash\nexit 0\n")
    flock_bin.chmod(0o755)

    science = _scientific_paths(root / "science")
    generation_root = Path(science["SOURCE_GENERATION_ROOT"])
    threshold_contract = _json(
        root / "source-threshold/frozen_threshold_contract.json",
        {
            "status": "PASS",
            "dataset": "AIDS",
            "distance_line": "MolCLR-Node-Wasserstein",
        },
    )
    source_outputs = {
        GENERATION_SOURCE_KEY: root / "repair-v2/aids-generation/attempt-0",
        THRESHOLD_SOURCE_KEY: root / "repair-v2/aids-threshold/attempt-0",
    }
    generation_gate = {
        "schema_version": "four_by_four_am_repair_source_gate_v2",
        "status": "PASS",
        "source_key": GENERATION_SOURCE_KEY,
        "evidence": {
            "status": "PASS",
            "dataset": "aids",
            "semantic": {
                "kind": "generation_adoption",
                "dataset": "aids",
                "generation_root": str(generation_root),
                "generation_payload_claimed_sha256": "a" * 64,
            },
        },
    }
    threshold_gate = {
        "schema_version": "four_by_four_am_repair_source_gate_v2",
        "status": "PASS",
        "source_key": THRESHOLD_SOURCE_KEY,
        "evidence": {
            "status": "PASS",
            "dataset": "aids",
            "semantic": {
                "kind": "threshold",
                "dataset": "aids",
                "threshold_contract": str(threshold_contract),
                "threshold_count": 601,
                "theta_star": 0.05,
                "cost_cap": 0.0535,
                "test_used_for_selection": False,
            },
        },
    }
    for key, payload in (
        (GENERATION_SOURCE_KEY, generation_gate),
        (THRESHOLD_SOURCE_KEY, threshold_gate),
    ):
        output = source_outputs[key]
        _json(output / "source_gate.json", payload)
        _text(output / "PASS", "PASS\n")

    failed_root = root / "repair-v2/aids-standardized/attempt-0"
    _json(
        failed_root / "FAILED.json",
        {
            "status": "FAILED",
            "dataset": "aids",
            "message": "run_common_recourse.py died with <Signals.SIGKILL: 9>",
        },
    )
    source_tasks = [
        _source_task("am_v2_source_aids_comrec_generation", source_outputs[GENERATION_SOURCE_KEY]),
        _source_task(
            "am_v2_source_aids_comrec_threshold",
            source_outputs[THRESHOLD_SOURCE_KEY],
            freezes_selector=True,
        ),
        {
            "id": "aids_comrecgc_standardized",
            "dataset": "aids",
            "stage": "AM_COMRECGC_HELDOUT_EVAL",
            "depends_on": [
                "am_v2_source_aids_comrec_generation",
                "am_v2_source_aids_comrec_threshold",
            ],
            "resource": "gpu",
            "priority": 20,
            "data_splits": ["test"],
            "manifest_only": False,
            "selector_parameters_frozen": True,
            "read_only_test": True,
            "command": [
                "bash",
                "{project_root}/scripts/autodl/run_comrecgc_standardized_continuation.sh",
            ],
            "input_manifest": str(source_outputs[GENERATION_SOURCE_KEY] / "source_gate.json"),
            "expected_output": str(failed_root),
            "required_output_files": ["PASS"],
            "required_log_marker": "PASS",
            "environment": {
                **science,
                "THRESHOLDS_PATH": str(threshold_contract),
                "OUTPUT_ROOT": str(failed_root),
            },
        },
    ]
    source_manifest = manifests / f"{SOURCE_CONTROLLER_ID}.json"
    _json(
        source_manifest,
        {
            "schema_version": 1,
            "controller_id": SOURCE_CONTROLLER_ID,
            "paper_frozen": True,
            "runtime": {
                "max_gpus": 4,
                "max_cpu_tasks": 2,
                "max_transient_retries": 0,
            },
            "resource_gates": {},
            "tasks": source_tasks,
        },
    )
    source_sha = sha256_file(source_manifest)
    source_controller_root = namespace / SOURCE_CONTROLLER_ID
    _json(
        source_controller_root / "controller_manifest.json",
        {
            "controller_id": SOURCE_CONTROLLER_ID,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": source_sha,
        },
    )
    for key, task_id in (
        (GENERATION_SOURCE_KEY, "am_v2_source_aids_comrec_generation"),
        (THRESHOLD_SOURCE_KEY, "am_v2_source_aids_comrec_threshold"),
    ):
        output = source_outputs[key]
        task_root = source_controller_root / "tasks" / task_id
        _json(
            task_root / "manifest.json",
            {
                "task_id": task_id,
                "controller_manifest_sha256": source_sha,
                "expected_output": str(output),
            },
        )
        _json(
            task_root / "state.json",
            {
                "task_id": task_id,
                "state": "PASS",
                "instances": {"main": {"state": "PASS", "expected_output": str(output)}},
            },
        )
        _json(task_root / "gate.json", {"status": "PASS"})
    _json(
        source_controller_root / "tasks/aids_comrecgc_standardized/state.json",
        {
            "task_id": "aids_comrecgc_standardized",
            "state": "FAILED",
            "instances": {
                "main": {
                    "state": "FAILED",
                    "exit_code": 1,
                    "expected_output": str(failed_root),
                }
            },
        },
    )
    monkeypatch.setattr(
        repair,
        "verify_fix_ancestry",
        lambda **_kwargs: {
            "required_fix_commit": VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
            "execution_head": "f" * 40,
            "is_ancestor": "true",
        },
    )
    spec = _json(
        root / "spec.json",
        {
            "schema_version": repair.SPEC_SCHEMA,
            "controller_id": CONTROLLER_ID,
            "paper_frozen": True,
            "run_tastemolnet": 0,
            "runtime_root": str(runtime),
            "control_root": str(control),
            "project_root": str(Path.cwd().resolve()),
            "python": str(Path(os.sys.executable).resolve()),
            "proc_root": str(proc_root),
            "cgroup_memory_root": str(cgroup),
            "min_cgroup_free_bytes": MINIMUM_HEADROOM_BYTES,
            "highmem_lock_path": str(runtime / "locks/comrecgc_common_recourse_highmem.lock"),
            "flock_bin": str(flock_bin),
            "fresh_output_root": str(runtime / "outputs/autodl/repairs/aids-v3"),
            "verify_comrecgc_checkout_safe_git_fix_commit": VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
            "source_controller": {
                "manifest": str(source_manifest),
                "root": str(source_controller_root),
            },
            "source_outputs": {key: str(value) for key, value in source_outputs.items()},
            "failed_aids_output_root": str(failed_root),
        },
    )
    return {
        "_temporary": temporary,
        "runtime": runtime,
        "control": control,
        "cgroup": cgroup,
        "spec": spec,
        "source_manifest": source_manifest,
        "source_controller_root": source_controller_root,
        "failed_root": failed_root,
    }


def test_payload_is_exact_cpu_only_serial_fresh_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(monkeypatch)
    payload, summary = build_payload(spec_path=paths["spec"])
    validation = validate_payload(payload)
    tasks = {task["id"]: task for task in payload["tasks"]}
    assert set(tasks) == {
        GENERATION_GATE_TASK_ID,
        THRESHOLD_GATE_TASK_ID,
        STANDARDIZATION_TASK_ID,
    }
    assert summary["gpu_required"] is False
    assert validation["max_cpu_tasks"] == 1
    assert all(task["resource"] == "cpu" for task in tasks.values())
    standard = tasks[STANDARDIZATION_TASK_ID]
    assert standard["environment"]["DEVICE"] == "cpu"
    assert standard["environment"]["GPU_REQUIRED"] == "0"
    assert standard["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "mutagenicity" not in json.dumps(payload["tasks"]).lower()
    assert (
        payload["aids_comrecgc_repair_v3_contract"][
            "common_recourse_colocation_forbidden"
        ]
        is True
    )


def test_cgroup_oom_and_sigkill_evidence_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(monkeypatch)
    _text(paths["cgroup"] / "memory.oom_control", "oom_kill 0\n")
    with pytest.raises(RepairManifestError, match="does not prove"):
        build_payload(spec_path=paths["spec"])

    paths = _fixture(monkeypatch)
    failed = _json(
        paths["failed_root"] / "FAILED.json",
        {"status": "FAILED", "dataset": "aids", "message": "ordinary ValueError"},
    )
    assert failed.is_file()
    with pytest.raises(RepairManifestError, match="reviewed OOM"):
        build_payload(spec_path=paths["spec"])


def test_manifest_build_is_fresh_and_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(monkeypatch)
    destination = (
        paths["control"]
        / SOURCE_NAMESPACE
        / "manifests"
        / f"{CONTROLLER_ID}.json"
    )
    result = build_manifest(spec_path=paths["spec"], output_path=destination)
    assert result["status"] == "PASS"
    assert load_controller_manifest(destination).controller_id == CONTROLLER_ID
    with pytest.raises(FileExistsError, match="fresh"):
        build_manifest(spec_path=paths["spec"], output_path=destination)


def test_highmem_wrapper_enforces_cpu_and_runs_under_lock(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    wrapper_source = root / "scripts/autodl/run_comrecgc_standardized_continuation_cpu_highmem.sh"
    wrapper_root = tmp_path / "wrapper"
    wrapper_root.mkdir()
    wrapper = wrapper_root / wrapper_source.name
    shutil.copy2(wrapper_source, wrapper)
    wrapper.chmod(0o755)
    _text(
        wrapper_root / "run_comrecgc_standardized_continuation.sh",
        "#!/bin/bash\nset -euo pipefail\necho inner-pass\n",
    ).chmod(0o755)
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    _text(cgroup / "memory.limit_in_bytes", "1000\n")
    _text(cgroup / "memory.usage_in_bytes", "100\n")
    proc = tmp_path / "proc"
    proc.mkdir()
    environment = {
        **os.environ,
        "DATASET": "aids",
        "OUTPUT_ROOT": str(tmp_path / "fresh-output"),
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "COMRECGC_HIGHMEM_LOCK_PATH": str(tmp_path / "lock/highmem.lock"),
        "COMRECGC_CGROUP_MEMORY_ROOT": str(cgroup),
        "COMRECGC_MIN_CGROUP_FREE_BYTES": "800",
        "COMRECGC_PROC_ROOT": str(proc),
    }
    flock_bin = _text(tmp_path / "bin/flock", "#!/bin/bash\nexit 0\n")
    flock_bin.chmod(0o755)
    environment["COMRECGC_FLOCK_BIN"] = str(flock_bin)
    completed = subprocess.run(
        ["bash", str(wrapper)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[COMRECGC_HIGHMEM_GATE_PASS]" in completed.stdout
    assert "gpu_required=false" in completed.stdout
    assert "inner-pass" in completed.stdout

    command_file = proc / "123/cmdline"
    _text(
        command_file,
        "python\0scripts/baselines/comrecgc/run_common_recourse.py\0",
    )
    environment["OUTPUT_ROOT"] = str(tmp_path / "fresh-output-concurrent")
    concurrent = subprocess.run(
        ["bash", str(wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert concurrent.returncode == 75
    assert "another common-recourse process is active" in concurrent.stderr

    _text(command_file, "python\0unrelated.py\0")
    _text(cgroup / "memory.usage_in_bytes", "500\n")
    environment["OUTPUT_ROOT"] = str(tmp_path / "fresh-output-low-memory")
    low_memory = subprocess.run(
        ["bash", str(wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert low_memory.returncode == 75
    assert "insufficient cgroup headroom" in low_memory.stderr


def test_template_slurm_and_docs_keep_route_autodl_only() -> None:
    root = Path(__file__).resolve().parents[2]
    template = json.loads(
        (root / "configs/autodl/aids_comrecgc_repair_v3.template.json").read_text()
    )
    assert template["controller_id"] == CONTROLLER_ID
    assert template["min_cgroup_free_bytes"] >= MINIMUM_HEADROOM_BYTES
    assert "mutagenicity" not in template
    for relative in (
        "scripts/slurm/build_aids_comrecgc_repair_v3_manifest.sh",
        "scripts/slurm/run_comrecgc_standardized_continuation_cpu_highmem.sh",
    ):
        value = (root / relative).read_text(encoding="utf-8")
        for token in (
            "#SBATCH --partition=A800",
            "#SBATCH --gres=gpu:a800:1",
            "#SBATCH --output=logs/%j.out",
            "#SBATCH --error=logs/%j.err",
            "source ~/.bashrc",
            "conda activate smiles_pip118",
            "cd /share/home/u20526/czx/counterfactual-subgraph",
            "export PYTHONPATH=$PWD",
            "do not submit",
        ):
            assert token in value
