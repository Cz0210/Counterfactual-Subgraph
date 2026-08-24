from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest

from scripts.autodl import run_four_gpu_recovery_controller as controller
from scripts.autodl.verify_aids_comrecgc_v5_process_set import verify_process_set
from scripts.autodl import write_aids_comrecgc_v5_selector_gate as selector_gate
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from src.baselines.comrecgc.external_memory_recourse import (
    PAIR_STORE_SCHEMA,
    _stable_hash,
)
from src.utils import autodl_aids_comrecgc_exact_route_v5 as v5
from src.utils.autodl_aids_comrecgc_repair_v4 import (
    CONTROLLER_ID as V4_CONTROLLER_ID,
    STANDARDIZATION_TASK_ID as V4_TASK_ID,
)
from src.utils.autodl_four_by_four_repair import RepairManifestError, sha256_file


def _json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    monkeypatch.setattr(v5, "EXPECTED_PARENT_COUNT", 2)
    monkeypatch.setattr(controller, "TEST_PATH", re.compile(r"a^"))
    monkeypatch.setattr(v5, "EXPECTED_CANDIDATE_COUNT", 3)
    monkeypatch.setattr(v5, "EXPECTED_PAIR_COUNT", 6)
    monkeypatch.setattr(v5, "EXPECTED_VECTOR_DIM", 4)
    monkeypatch.setattr(v5, "_git_head", lambda _root: "f" * 40)
    monkeypatch.setattr(
        v5,
        "_require_ancestor",
        lambda _root, commit: {
            "required_commit": commit,
            "execution_head": "f" * 40,
            "is_ancestor": "true",
        },
    )
    runtime = tmp_path / "runtime"
    control = runtime / "control"
    (control / v5.SOURCE_NAMESPACE / "manifests").mkdir(parents=True)
    project = Path(__file__).resolve().parents[2]
    proc = tmp_path / "proc"
    proc.mkdir()
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    _file(cgroup / "memory.limit_in_bytes", str(512 * 1024**3))
    _file(cgroup / "memory.usage_in_bytes", str(64 * 1024**3))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    flock = _file(tmp_path / "bin/flock", "#!/bin/bash\nexit 0\n")
    flock.chmod(0o755)
    science = tmp_path / "science"
    directories = {
        key: science / key.lower()
        for key in (
            "SOURCE_GENERATION_ROOT",
            "COMRECGC_UPSTREAM_ROOT",
            "DATASET_DIR",
            "MOLCLR_ROOT",
        )
    }
    for value in directories.values():
        value.mkdir(parents=True)
    files = {
        key: _file(science / f"{key.lower()}.dat")
        for key in (
            "SOURCE_CSV",
            "DISTANCE_CHECKPOINT",
            "DATASET_CSV",
            "TEACHER_PATH",
            "MOLCLR_CHECKPOINT",
            "THRESHOLDS_PATH",
        )
    }
    generation_manifest = _json(
        directories["SOURCE_GENERATION_ROOT"] / "run_manifest.json",
        {"run_complete": True},
    )
    environment = {
        "AUTODL_PYTHON": "{python}",
        "DATASET": "aids",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_COMMON_RECOURSE_RESUME": "1",
        "THETA_STAR": "0.05",
        "COST_CAP": "0.0535",
        "OUTPUT_ROOT": "{task_output}",
        **{key: str(value) for key, value in directories.items()},
        **{key: str(value) for key, value in files.items()},
    }
    base_manifest = _json(
        control / v5.SOURCE_NAMESPACE / "manifests/base-v4.json",
        {
            "schema_version": 1,
            "controller_id": V4_CONTROLLER_ID,
            "paper_frozen": True,
            "runtime": {"max_gpus": 4, "max_cpu_tasks": 1},
            "tasks": [
                {
                    "id": "selector-freeze",
                    "dataset": "aids",
                    "stage": "AM_COMRECGC_THRESHOLD_FREEZE",
                    "resource": "cpu",
                    "manifest_only": True,
                    "freezes_selector": True,
                    "skip_reason": "fixture-only frozen selector",
                },
                {
                    "id": V4_TASK_ID,
                    "dataset": "aids",
                    "stage": "AM_COMRECGC_HELDOUT_EVAL",
                    "resource": "cpu",
                    "depends_on": ["selector-freeze"],
                    "data_splits": ["test"],
                    "selector_parameters_frozen": True,
                    "read_only_test": True,
                    "command": ["bash", "old.sh"],
                    "input_manifest": str(generation_manifest),
                    "config_files": [str(files["THRESHOLDS_PATH"])],
                    "expected_output": str(tmp_path / "old/attempt-{attempt}"),
                    "required_output_files": ["PASS"],
                    "required_log_marker": "PASS",
                    "environment": environment,
                }
            ],
        },
    )
    old_output_root = tmp_path / "old"
    old_project_root = tmp_path / "old-project"
    old_project_root.mkdir()
    pair_root = old_output_root / "pair_store"
    pair_root.mkdir(parents=True)
    pairs_path = pair_root / "pair_indices.npy"
    vectors_path = pair_root / "recourse_vectors.npy"
    with pairs_path.open("wb") as handle:
        np.save(
            handle,
            np.asarray(
                [[parent, candidate] for candidate in range(3) for parent in range(2)],
                dtype=np.int64,
            ),
            allow_pickle=False,
        )
    with vectors_path.open("wb") as handle:
        np.save(handle, np.zeros((6, 4), dtype=np.float32), allow_pickle=False)
    identity = {
        "dataset": "aids",
        "mode": "full",
        "parameters": dict(v5.EXPECTED_PARAMETERS),
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "distance_checkpoint_sha256": sha256_file(files["DISTANCE_CHECKPOINT"]),
        "candidate_count": 3,
        "parent_count": 2,
        "pair_order": "candidate_major_parent_minor",
        "device": "cpu",
    }
    pair_manifest = _json(
        pair_root / "run_manifest.json",
        {
            "schema_version": PAIR_STORE_SCHEMA,
            "run_complete": True,
            "scientific_identity": identity,
            "scientific_identity_sha256": _stable_hash(identity),
            "row_count": 6,
            "vector_dim": 4,
            "vectors_dtype": "float32",
            "pairs_path": str(pairs_path),
            "pairs_sha256": sha256_file(pairs_path),
            "vectors_path": str(vectors_path),
            "vectors_sha256": sha256_file(vectors_path),
            "candidate_major_parent_minor_order": True,
        },
    )
    old_pid = 273939
    old_start_ticks = 687141119
    old_proc = proc / str(old_pid)
    old_proc.mkdir()
    old_script = old_project_root / "scripts/baselines/comrecgc/run_common_recourse.py"
    old_script.parent.mkdir(parents=True)
    old_script.write_text("# fixture\n", encoding="utf-8")
    old_cmdline = (
        str(Path(os.sys.executable).resolve()).encode()
        + b"\0"
        + str(old_script).encode()
        + b"\0--output-dir\0"
        + str(old_output_root).encode()
        + b"\0"
    )
    (old_proc / "cmdline").write_bytes(old_cmdline)
    (old_proc / "stat").write_text(
        f"{old_pid} (python) "
        + " ".join(["S", *("0" for _ in range(18)), str(old_start_ticks)])
        + "\n",
        encoding="utf-8",
    )
    (old_proc / "cwd").symlink_to(old_project_root, target_is_directory=True)
    fresh = runtime / "outputs/autodl/repairs/aids-v5"
    spec = _json(
        tmp_path / "aids-v5-spec.json",
        {
            "schema_version": v5.SPEC_SCHEMA,
            "controller_id": v5.CONTROLLER_ID,
            "paper_frozen": True,
            "run_tastemolnet": 0,
            "runtime_root": str(runtime),
            "control_root": str(control),
            "project_root": str(project),
            "execution_commit": "f" * 40,
            "python": str(Path(os.sys.executable).resolve()),
            "proc_root": str(proc),
            "cgroup_memory_root": str(cgroup),
            "min_cgroup_free_bytes": v5.MINIMUM_CGROUP_FREE_BYTES,
            "flock_bin": str(flock),
            "local_scratch_root": str(scratch),
            "route_lock_path": str(scratch / "locks/aids-v5.lock"),
            "base_v4_manifest": str(base_manifest),
            "base_v4_manifest_sha256": sha256_file(base_manifest),
            "terminal_pair_store_root": str(pair_root),
            "terminal_pair_store_manifest_sha256": sha256_file(pair_manifest),
            "allowed_old_read_only_process": {
                "pid": old_pid,
                "start_ticks": old_start_ticks,
                "cmdline_sha256": hashlib.sha256(old_cmdline).hexdigest(),
                "output_root": str(old_output_root),
                "project_root": str(old_project_root),
            },
            "fresh_output_root": str(fresh),
        },
    )
    return {
        "runtime": runtime,
        "control": control,
        "spec": spec,
        "pair_root": pair_root,
        "pair_manifest": pair_manifest,
        "fresh": fresh,
        "cgroup": cgroup,
        "proc": proc,
        "old_proc": old_proc,
        "old_output_root": old_output_root,
        "old_project_root": old_project_root,
        "old_cmdline": old_cmdline,
    }


def test_v5_payload_is_terminal_only_cpu_and_freezes_mut_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload, summary = v5.build_payload(spec_path=paths["spec"])
    assert v5.validate_payload(payload)["status"] == "PASS"
    task = next(task for task in payload["tasks"] if task["id"] == v5.TASK_ID)
    assert task["id"] == v5.TASK_ID
    assert task["resource"] == "cpu"
    assert task["command"] == [
        "bash",
        "{project_root}/scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh",
    ]
    environment = task["environment"]
    assert environment["COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL"] == "1"
    assert environment["COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT"] == str(
        paths["pair_root"]
    )
    assert environment["COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT"] == str(
        paths["pair_root"]
    )
    assert environment["COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES"] == "0"
    assert environment["AIDS_COMRECGC_V5_ALLOWED_OLD_PID"] == "273939"
    assert environment["AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS"] == "687141119"
    assert not any("SOURCE_CHECKPOINT" in key or "CACHE_ROOT" in key for key in environment)
    dependency = payload["aids_comrecgc_exact_route_v5_contract"]["mut_dependency"]
    assert dependency["controller_id"] == v5.CONTROLLER_ID
    assert dependency["task_id"] == v5.TASK_ID
    assert dependency["expected_output"].endswith("/attempt-0")
    assert summary["gpu_required"] is False


def _process_gate(paths: dict[str, Path]) -> dict[str, Any]:
    return verify_process_set(
        proc_root=paths["proc"],
        allowed_pid=273939,
        allowed_start_ticks=687141119,
        allowed_cmdline_sha256=hashlib.sha256(paths["old_cmdline"]).hexdigest(),
        allowed_output_root=paths["old_output_root"],
        allowed_project_root=paths["old_project_root"],
    )


def test_v5_process_gate_accepts_exact_old_generation_and_natural_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    present = _process_gate(paths)
    assert present["process_set_status"] == "ALLOWED_OLD_READ_ONLY_PROCESS_PRESENT"
    for child in paths["old_proc"].iterdir():
        child.unlink()
    paths["old_proc"].rmdir()
    exited = _process_gate(paths)
    assert exited["process_set_status"] == "ALLOWED_OLD_PROCESS_NATURALLY_EXITED"
    assert exited["active_common_recourse_count"] == 0


@pytest.mark.parametrize("mutation", ["start_ticks", "cmdline", "cwd"])
def test_v5_process_gate_rejects_pid_reuse_or_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    if mutation == "start_ticks":
        (paths["old_proc"] / "stat").write_text(
            "273939 (python) "
            + " ".join(["S", *("0" for _ in range(18)), "687141120"])
            + "\n",
            encoding="utf-8",
        )
    elif mutation == "cmdline":
        (paths["old_proc"] / "cmdline").write_bytes(
            paths["old_cmdline"].replace(b"--output-dir", b"--changed-output")
        )
    else:
        (paths["old_proc"] / "cwd").unlink()
        other = tmp_path / "other-project"
        other.mkdir()
        (paths["old_proc"] / "cwd").symlink_to(other, target_is_directory=True)
    with pytest.raises(RuntimeError, match="IDENTITY_MISMATCH"):
        _process_gate(paths)


def test_v5_process_gate_rejects_any_second_common_recourse_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    second = paths["proc"] / "274000"
    second.mkdir()
    (second / "cmdline").write_bytes(paths["old_cmdline"])
    (second / "stat").write_text(
        "274000 (python) "
        + " ".join(["S", *("0" for _ in range(18)), "687141121"])
        + "\n",
        encoding="utf-8",
    )
    (second / "cwd").symlink_to(paths["old_project_root"], target_is_directory=True)
    with pytest.raises(RuntimeError, match="UNEXPECTED_COMMON_RECOURSE_PROCESS_SET"):
        _process_gate(paths)


def test_v5_terminal_source_gate_rejects_partial_and_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _file(paths["pair_root"] / "unexpected.partial", "partial")
    with pytest.raises(Exception, match="PARTIAL"):
        v5.build_payload(spec_path=paths["spec"])
    (paths["pair_root"] / "unexpected.partial").unlink()
    pair_manifest = json.loads(paths["pair_manifest"].read_text())
    pair_manifest["row_count"] += 1
    _json(paths["pair_manifest"], pair_manifest)
    with pytest.raises(RepairManifestError, match="manifest SHA256 mismatch"):
        v5.build_payload(spec_path=paths["spec"])


def test_v5_rejects_cgroup_headroom_or_chunk_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _file(paths["cgroup"] / "memory.usage_in_bytes", str(500 * 1024**3))
    with pytest.raises(RepairManifestError, match="headroom gate"):
        v5.build_payload(spec_path=paths["spec"])
    _file(paths["cgroup"] / "memory.usage_in_bytes", str(64 * 1024**3))
    payload, _summary = v5.build_payload(spec_path=paths["spec"])
    task = next(task for task in payload["tasks"] if task["id"] == v5.TASK_ID)
    task["environment"][
        "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT"
    ] = "/forbidden"
    with pytest.raises(RepairManifestError, match="fallback bypass"):
        v5.validate_payload(payload)
    task["environment"].pop("COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT")
    task["environment"]["AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS"] = "687141120"
    with pytest.raises(RepairManifestError, match="old-process environment drifted"):
        v5.validate_payload(payload)


def test_v5_manifest_publishes_only_at_exact_fresh_namespace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    destination = (
        paths["control"]
        / v5.SOURCE_NAMESPACE
        / "manifests"
        / f"{v5.CONTROLLER_ID}.json"
    )
    result = v5.build_manifest(spec_path=paths["spec"], output_path=destination)
    assert result["status"] == "PASS"
    assert load_controller_manifest(destination).controller_id == v5.CONTROLLER_ID
    assert result["manifest_sha256"] == sha256_file(destination)
    with pytest.raises(FileExistsError, match="fresh"):
        v5.build_manifest(spec_path=paths["spec"], output_path=destination)


def test_v5_release_pins_reviewed_core_and_has_static_paired_slurm() -> None:
    assert v5.REVIEWED_CORE_COMMIT == "645c6e51b7abcdc5dd4a9e0a1226d71d020880da"
    assert v5.ROUTE_RELEASE_COMMIT == "e75b6e8160e07c869c558080259b4b05695f76d7"
    root = Path(__file__).resolve().parents[2]
    wrapper = (
        root / "scripts/slurm/build_aids_comrecgc_exact_route_v5_manifest.sh"
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
        "exit 78",
    ):
        assert token in wrapper
    selector_wrapper = (
        root / "scripts/slurm/write_aids_comrecgc_v5_selector_gate.sh"
    ).read_text(encoding="utf-8")
    assert "do not submit" in selector_wrapper
    assert "--config configs/hpc.yaml" in selector_wrapper
    assert "exit 78" in selector_wrapper


def test_v5_selector_adoption_gate_is_hash_bound_and_fresh(
    tmp_path: Path,
) -> None:
    threshold = _json(
        tmp_path / "threshold.json", {"test_used_for_selection": False}
    )
    output = tmp_path / "selector"
    assert selector_gate.main(
        [
            "--thresholds",
            str(threshold),
            "--expected-sha256",
            sha256_file(threshold),
            "--output-dir",
            str(output),
        ]
    ) == 0
    gate = json.loads((output / "selector_gate.json").read_text())
    assert gate["selector_fitted_on_calibration"] is True
    assert gate["test_used_for_selection"] is False
    assert (output / "PASS").read_text() == "PASS\n"
    with pytest.raises(SystemExit, match="SHA256 mismatch"):
        selector_gate.main(
            [
                "--thresholds",
                str(threshold),
                "--expected-sha256",
                "0" * 64,
                "--output-dir",
                str(tmp_path / "bad"),
            ]
        )
