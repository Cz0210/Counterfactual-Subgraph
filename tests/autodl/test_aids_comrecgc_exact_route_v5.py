from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
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
from src.utils.aids_comrecgc_v5_snapshot import (
    create_promoted_pair_store_snapshot,
)
from src.utils.aids_comrecgc_v5_snapshot_adoption import (
    SOURCE_CONTROLLER_ID as SNAPSHOT_OWNER_CONTROLLER_ID,
    SOURCE_SNAPSHOT_TASK_ID as SNAPSHOT_OWNER_TASK_ID,
)
from src.utils import aids_comrecgc_v5_snapshot_adoption as adoption
from src.utils import aids_comrecgc_v5_science_exec as science_exec
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mock_git: bool = True,
) -> dict[str, Path]:
    monkeypatch.setattr(v5, "EXPECTED_PARENT_COUNT", 2)
    monkeypatch.setattr(controller, "TEST_PATH", re.compile(r"a^"))
    monkeypatch.setattr(v5, "EXPECTED_CANDIDATE_COUNT", 3)
    monkeypatch.setattr(v5, "EXPECTED_PAIR_COUNT", 6)
    monkeypatch.setattr(v5, "EXPECTED_VECTOR_DIM", 4)
    monkeypatch.setattr(v5, "SNAPSHOT_EXPECTED_PARENT_COUNT", 2)
    monkeypatch.setattr(v5, "SNAPSHOT_EXPECTED_CANDIDATE_COUNT", 3)
    monkeypatch.setattr(v5, "SNAPSHOT_EXPECTED_ROWS", 6)
    monkeypatch.setattr(v5, "SNAPSHOT_EXPECTED_VECTOR_DIM", 4)
    monkeypatch.setattr(adoption, "EXPECTED_PARENT_COUNT", 2)
    monkeypatch.setattr(adoption, "EXPECTED_CANDIDATE_COUNT", 3)
    monkeypatch.setattr(adoption, "EXPECTED_ROWS", 6)
    monkeypatch.setattr(adoption, "EXPECTED_VECTOR_DIM", 4)
    if mock_git:
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
    highmem_lock = _file(
        runtime / "locks/comrecgc_common_recourse_highmem.lock", ""
    )
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
        "COMRECGC_HIGHMEM_LOCK_PATH": str(highmem_lock),
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
            "runtime": {
                "max_gpus": 4,
                "max_cpu_tasks": 1,
                "max_transient_retries": 0,
            },
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
    snapshot_root = tmp_path / "approved-snapshot"
    create_promoted_pair_store_snapshot(
        source_root=pair_root,
        expected_source_manifest_sha256=sha256_file(pair_manifest),
        output_dir=snapshot_root,
        proc_root=proc,
        allowed_pid=old_pid,
        allowed_start_ticks=old_start_ticks,
        allowed_cmdline_sha256=hashlib.sha256(old_cmdline).hexdigest(),
        allowed_output_root=old_output_root,
        allowed_project_root=old_project_root,
        min_free_after_bytes=0,
        expected_row_count=6,
        expected_vector_dim=4,
        expected_parent_count=2,
        expected_candidate_count=3,
    )
    owner_manifest = _json(
        control
        / v5.SOURCE_NAMESPACE
        / "manifests"
        / f"{SNAPSHOT_OWNER_CONTROLLER_ID}.json",
        {
            "schema_version": 1,
            "controller_id": SNAPSHOT_OWNER_CONTROLLER_ID,
            "paper_frozen": True,
            "runtime": {
                "max_gpus": 4,
                "max_cpu_tasks": 1,
                "max_transient_retries": 0,
            },
            "tasks": [
                {
                    "id": "owner-selector-freeze",
                    "dataset": "aids",
                    "stage": "AM_COMRECGC_THRESHOLD_FREEZE",
                    "resource": "cpu",
                    "manifest_only": True,
                    "freezes_selector": True,
                    "skip_reason": "fixture-only owner selector",
                },
                {
                    "id": SNAPSHOT_OWNER_TASK_ID,
                    "dataset": "aids",
                    "stage": "AM_COMRECGC_HELDOUT_EVAL",
                    "resource": "cpu",
                    "depends_on": ["owner-selector-freeze"],
                    "data_splits": ["test"],
                    "selector_parameters_frozen": True,
                    "read_only_test": True,
                    "command": ["bash", "snapshot.sh"],
                    "input_manifest": str(pair_manifest),
                    "expected_output": str(snapshot_root),
                    "required_output_files": ["snapshot_manifest.json", "PASS"],
                    "required_log_marker": "PASS",
                }
            ],
        },
    )
    owner_gate = _json(
        control
        / v5.SOURCE_NAMESPACE
        / SNAPSHOT_OWNER_CONTROLLER_ID
        / "tasks"
        / SNAPSHOT_OWNER_TASK_ID
        / "gate.json",
        {
            "schema_version": 1,
            "task_id": SNAPSHOT_OWNER_TASK_ID,
            "status": "PASS",
            "reason": None,
            "runs": [
                {
                    "attempt": 0,
                    "expected_output": str(snapshot_root),
                    "instance_id": "main",
                    "state": "PASS",
                }
            ],
        },
    )
    fresh = runtime / "outputs/autodl/repairs/aids-v5"
    spec = _json(
        tmp_path / "aids-v5-spec.json",
        {
            "schema_version": v5.SPEC_SCHEMA,
            "controller_id": v5.CONTROLLER_ID,
            "paper_frozen": True,
            "run_tastemolnet": 0,
            "physical_snapshot_required": True,
            "snapshot_adoption_required": True,
            "runtime_root": str(runtime),
            "control_root": str(control),
            "project_root": str(project),
            "execution_commit": "f" * 40 if mock_git else v5._git_head(project),
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
            "adopted_snapshot": {
                "owner_namespace_root": str(control / v5.SOURCE_NAMESPACE),
                "owner_manifest": str(owner_manifest),
                "owner_manifest_sha256": sha256_file(owner_manifest),
                "owner_task_gate": str(owner_gate),
                "owner_task_gate_sha256": sha256_file(owner_gate),
                "snapshot_root": str(snapshot_root),
                "snapshot_manifest_sha256": sha256_file(
                    snapshot_root / "snapshot_manifest.json"
                ),
                "dbscan_contract_sha256": sha256_file(
                    snapshot_root / "dbscan_contract.json"
                ),
                "pair_store_manifest_sha256": sha256_file(
                    snapshot_root / "pair_store/run_manifest.json"
                ),
                "pairs_sha256": sha256_file(
                    snapshot_root / "pair_store/pair_indices.npy"
                ),
                "vectors_sha256": sha256_file(
                    snapshot_root / "pair_store/recourse_vectors.npy"
                ),
            },
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
        "snapshot_root": snapshot_root,
        "owner_manifest": owner_manifest,
        "owner_gate": owner_gate,
        "owner_namespace_root": control / v5.SOURCE_NAMESPACE,
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
    snapshot_dependency = "{dep_" + v5.SNAPSHOT_TASK_ID + "_output}"
    adopted_root = str(paths["snapshot_root"])
    assert environment["COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT"] == (
        adopted_root + "/pair_store"
    )
    assert environment["COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT"] == (
        adopted_root + "/pair_store"
    )
    assert environment["AIDS_COMRECGC_V5_SNAPSHOT_ROOT"] == adopted_root
    assert (
        environment["AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_ROOT"]
        == snapshot_dependency
    )
    snapshot_task = next(
        task for task in payload["tasks"] if task["id"] == v5.SNAPSHOT_TASK_ID
    )
    assert snapshot_task["resource"] == "cpu"
    assert snapshot_task["command"][:2] == [
        "{python}",
        "{project_root}/scripts/autodl/adopt_aids_comrecgc_v5_snapshot.py",
    ]
    assert snapshot_task["required_output_files"] == [
        "snapshot_adoption_manifest.json",
        "PASS",
    ]
    assert task["depends_on"] == [v5.SELECTOR_TASK_ID, v5.SNAPSHOT_TASK_ID]
    assert environment["COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES"] == "0"
    assert environment["AIDS_COMRECGC_V5_ALLOWED_OLD_PID"] == "273939"
    assert environment["AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS"] == "687141119"
    assert environment["COMRECGC_HIGHMEM_LOCK_PATH"].endswith(
        "/locks/comrecgc_common_recourse_highmem.lock"
    )
    assert not any("SOURCE_CHECKPOINT" in key or "CACHE_ROOT" in key for key in environment)
    dependency = payload["aids_comrecgc_exact_route_v5_contract"]["mut_dependency"]
    assert dependency["controller_id"] == v5.CONTROLLER_ID
    assert dependency["task_id"] == v5.TASK_ID
    assert dependency["expected_output"].endswith("/attempt-0")
    headroom = payload["aids_comrecgc_exact_route_v5_contract"][
        "highmem_exclusion"
    ]["cgroup_headroom_gate"]
    assert headroom["limit_path"].endswith("/memory.limit_in_bytes")
    assert headroom["usage_path"].endswith("/memory.usage_in_bytes")
    assert headroom["free_bytes_at_build"] == 448 * 1024**3
    assert headroom["host_memfree_used"] is False
    assert summary["gpu_required"] is False


def _adoption_kwargs(paths: dict[str, Path], output: Path) -> dict[str, Any]:
    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    frozen = spec["adopted_snapshot"]
    return {
        "output_dir": output,
        "proc_root": paths["proc"],
        "owner_namespace_root": frozen["owner_namespace_root"],
        "owner_manifest": frozen["owner_manifest"],
        "owner_manifest_sha256": frozen["owner_manifest_sha256"],
        "owner_task_gate": frozen["owner_task_gate"],
        "owner_task_gate_sha256": frozen["owner_task_gate_sha256"],
        "snapshot_root": frozen["snapshot_root"],
        "snapshot_manifest_sha256": frozen["snapshot_manifest_sha256"],
        "dbscan_contract_sha256": frozen["dbscan_contract_sha256"],
        "pair_store_manifest_sha256": frozen["pair_store_manifest_sha256"],
        "pairs_sha256": frozen["pairs_sha256"],
        "vectors_sha256": frozen["vectors_sha256"],
        "source_root": paths["pair_root"],
        "source_manifest_sha256": sha256_file(paths["pair_manifest"]),
        "allowed_pid": 273939,
        "allowed_start_ticks": 687141119,
        "allowed_cmdline_sha256": hashlib.sha256(paths["old_cmdline"]).hexdigest(),
        "allowed_output_root": paths["old_output_root"],
        "allowed_project_root": paths["old_project_root"],
        "expected_row_count": 6,
        "expected_vector_dim": 4,
        "expected_parent_count": 2,
        "expected_candidate_count": 3,
    }


def test_snapshot_adoption_reopens_full_closure_without_copy_or_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "adoption"
    terminal = adoption.create_snapshot_adoption(**_adoption_kwargs(paths, output))
    assert terminal["status"] == "PASS"
    assert terminal["copy_or_hardlink_performed"] is False
    assert terminal["old_snapshot_mutated"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "PASS",
        "snapshot_adoption_manifest.json",
    ]
    assert not list(output.rglob("*.npy"))
    validated = adoption.validate_snapshot_adoption(
        output_dir=output,
        proc_root=paths["proc"],
        identity=terminal["identity"],
    )
    assert validated == terminal


def test_snapshot_adoption_rejects_identical_gate_from_non_authority_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    copied_gate = _file(tmp_path / "copied/gate.json", paths["owner_gate"].read_text())
    kwargs = _adoption_kwargs(paths, tmp_path / "adoption-wrong-gate")
    kwargs["owner_task_gate"] = copied_gate
    kwargs["owner_task_gate_sha256"] = sha256_file(copied_gate)
    with pytest.raises(adoption.SnapshotAdoptionError, match="authority path"):
        adoption.create_snapshot_adoption(**kwargs)


def test_builder_rejects_copied_owner_manifest_and_gate_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    forged_namespace = tmp_path / "forged-control/namespace"
    copied_manifest = _file(
        forged_namespace
        / "manifests"
        / f"{SNAPSHOT_OWNER_CONTROLLER_ID}.json",
        paths["owner_manifest"].read_text(encoding="utf-8"),
    )
    copied_gate = _file(
        forged_namespace
        / SNAPSHOT_OWNER_CONTROLLER_ID
        / "tasks"
        / SNAPSHOT_OWNER_TASK_ID
        / "gate.json",
        paths["owner_gate"].read_text(encoding="utf-8"),
    )
    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    spec["adopted_snapshot"].update(
        {
            "owner_namespace_root": str(forged_namespace),
            "owner_manifest": str(copied_manifest),
            "owner_manifest_sha256": sha256_file(copied_manifest),
            "owner_task_gate": str(copied_gate),
            "owner_task_gate_sha256": sha256_file(copied_gate),
        }
    )
    paths["spec"].write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(v5.RepairManifestError, match="control authority"):
        v5.build_payload(spec_path=paths["spec"])


def test_snapshot_adoption_rejects_copied_tree_outside_bound_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    forged_namespace = tmp_path / "forged-control/namespace"
    copied_manifest = _file(
        forged_namespace
        / "manifests"
        / f"{SNAPSHOT_OWNER_CONTROLLER_ID}.json",
        paths["owner_manifest"].read_text(encoding="utf-8"),
    )
    copied_gate = _file(
        forged_namespace
        / SNAPSHOT_OWNER_CONTROLLER_ID
        / "tasks"
        / SNAPSHOT_OWNER_TASK_ID
        / "gate.json",
        paths["owner_gate"].read_text(encoding="utf-8"),
    )
    kwargs = _adoption_kwargs(paths, tmp_path / "adoption-forged-tree")
    kwargs.update(
        {
            "owner_manifest": copied_manifest,
            "owner_manifest_sha256": sha256_file(copied_manifest),
            "owner_task_gate": copied_gate,
            "owner_task_gate_sha256": sha256_file(copied_gate),
        }
    )
    with pytest.raises(adoption.SnapshotAdoptionError, match="authority path"):
        adoption.create_snapshot_adoption(**kwargs)


def test_snapshot_adoption_requires_main_owner_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    gate = json.loads(paths["owner_gate"].read_text(encoding="utf-8"))
    gate["runs"][0]["instance_id"] = "not-main"
    paths["owner_gate"].write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    kwargs = _adoption_kwargs(paths, tmp_path / "adoption-non-main")
    kwargs["owner_task_gate_sha256"] = sha256_file(paths["owner_gate"])
    with pytest.raises(adoption.SnapshotAdoptionError, match="exact PASS attempt-0"):
        adoption.create_snapshot_adoption(**kwargs)


@pytest.mark.parametrize("tamper", ["owner_gate", "snapshot_manifest", "missing_pass"])
def test_snapshot_adoption_revalidation_fails_closed_before_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "adoption-tamper"
    terminal = adoption.create_snapshot_adoption(**_adoption_kwargs(paths, output))
    if tamper == "owner_gate":
        paths["owner_gate"].write_text('{"status":"FAILED"}\n', encoding="utf-8")
    elif tamper == "snapshot_manifest":
        manifest = paths["snapshot_root"] / "snapshot_manifest.json"
        manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    else:
        (output / "PASS").unlink()
    with pytest.raises(adoption.SnapshotAdoptionError):
        adoption.validate_snapshot_adoption(
            output_dir=output,
            proc_root=paths["proc"],
            identity=terminal["identity"],
        )


def test_snapshot_adoption_real_cli_create_and_validate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "adoption-cli"
    kwargs = _adoption_kwargs(paths, output)
    root = Path(__file__).resolve().parents[2]
    cli = root / "scripts/autodl/adopt_aids_comrecgc_v5_snapshot.py"
    args = [
        sys.executable,
        str(cli),
        "--config",
        "configs/hpc.yaml",
    ]
    option_names = {
        "output_dir": "output-dir",
        "proc_root": "proc-root",
        "owner_namespace_root": "owner-namespace-root",
        "owner_manifest": "owner-manifest",
        "owner_manifest_sha256": "owner-manifest-sha256",
        "owner_task_gate": "owner-task-gate",
        "owner_task_gate_sha256": "owner-task-gate-sha256",
        "snapshot_root": "snapshot-root",
        "snapshot_manifest_sha256": "snapshot-manifest-sha256",
        "dbscan_contract_sha256": "dbscan-contract-sha256",
        "pair_store_manifest_sha256": "pair-store-manifest-sha256",
        "pairs_sha256": "pairs-sha256",
        "vectors_sha256": "vectors-sha256",
        "source_root": "source-root",
        "source_manifest_sha256": "source-manifest-sha256",
        "allowed_pid": "allowed-pid",
        "allowed_start_ticks": "allowed-start-ticks",
        "allowed_cmdline_sha256": "allowed-cmdline-sha256",
        "allowed_output_root": "allowed-output-root",
        "allowed_project_root": "allowed-project-root",
        "expected_row_count": "expected-row-count",
        "expected_vector_dim": "expected-vector-dim",
        "expected_parent_count": "expected-parent-count",
        "expected_candidate_count": "expected-candidate-count",
    }
    for key, option in option_names.items():
        args.extend([f"--{option}", str(kwargs[key])])
    created = subprocess.run(args, cwd=root, check=True, capture_output=True, text=True)
    assert "[AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_PASS]" in created.stdout
    validated = subprocess.run(
        [*args, "--validate-only"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_VALIDATE_PASS]" in validated.stdout


def test_science_supervisor_runs_adoption_validator_before_any_science_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    adoption_root = tmp_path / "adoption-preflight"
    terminal = adoption.create_snapshot_adoption(
        **_adoption_kwargs(paths, adoption_root)
    )
    trace = tmp_path / "python-trace.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$*\" > {trace!s}\n"
        "exit 75\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    identity = terminal["identity"]
    old = identity["allowed_old_generation"]
    root = Path(__file__).resolve().parents[2]
    supervisor = root / "scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh"
    environment = {
        **os.environ,
        "AUTODL_PYTHON": str(fake_python),
        "OUTPUT_ROOT": str(tmp_path / "fresh-science"),
        "DATASET": "aids",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_COMMON_RECOURSE_RESUME": "1",
        "AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES": "1",
        "COMRECGC_EXTERNAL_MAX_RSS_GB": "96",
        "COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE": "8",
        "COMRECGC_EXTERNAL_CHECKPOINT_INTERVAL_BLOCKS": "1",
        "COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE": (
            "all_core_one_component_adaptive_anchor_v1"
        ),
        "COMRECGC_EXTERNAL_SHORTCUT_SEED_COUNT": "3",
        "COMRECGC_EXTERNAL_SHORTCUT_FAILURE_CAP": "4096",
        "COMRECGC_EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE": "65536",
        "COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES": "0",
        "COMRECGC_EXTERNAL_SUMMARY_BLOCK_SIZE": "65536",
        "COMRECGC_EXPECTED_SKLEARN_VERSION": "1.7.2",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB": "3",
        "COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL": "1",
        "COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT": str(
            paths["snapshot_root"] / "pair_store"
        ),
        "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT": str(
            paths["snapshot_root"] / "pair_store"
        ),
        "COMRECGC_EXTERNAL_ROUTE_LOCK": "/root/autodl-tmp/aids-v5-test.lock",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT": str(paths["proc"]),
        "COMRECGC_CGROUP_MEMORY_ROOT": str(paths["cgroup"]),
        "AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES": str(128 * 1024**3),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_PID": str(old["pid"]),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS": str(old["start_ticks"]),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256": str(
            old["cmdline_sha256"]
        ),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT": str(old["output_root"]),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT": str(old["project_root"]),
        "COMRECGC_HIGHMEM_LOCK_PATH": str(
            paths["runtime"] / "locks/comrecgc_common_recourse_highmem.lock"
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_ROOT": str(paths["snapshot_root"]),
        "AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_ROOT": str(adoption_root),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_NAMESPACE_ROOT": str(
            identity["owner_namespace_root"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST": str(
            identity["owner_manifest"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST_SHA256": str(
            identity["owner_manifest_sha256"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE": str(
            identity["owner_task_gate"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE_SHA256": str(
            identity["owner_task_gate_sha256"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_MANIFEST_SHA256": str(
            identity["snapshot_manifest_sha256"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_DBSCAN_SHA256": str(
            identity["dbscan_contract_sha256"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_PAIR_MANIFEST_SHA256": str(
            identity["pair_store_manifest_sha256"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_PAIRS_SHA256": str(identity["pairs_sha256"]),
        "AIDS_COMRECGC_V5_SNAPSHOT_VECTORS_SHA256": str(
            identity["vectors_sha256"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_ROOT": str(paths["pair_root"]),
        "AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256": str(
            identity["source_manifest_sha256"]
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT": str(paths["proc"]),
        "AIDS_COMRECGC_V5_SNAPSHOT_MIN_FREE_AFTER_BYTES": "42949672960",
        "AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_ROWS": "91916686",
        "AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_VECTOR_DIM": "64",
        "AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_PARENT_COUNT": "1283",
        "AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_CANDIDATE_COUNT": "71642",
    }
    completed = subprocess.run(
        ["bash", str(supervisor)],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 75
    assert "snapshot adoption closure validation failed" in completed.stderr
    traced = trace.read_text(encoding="utf-8")
    assert "adopt_aids_comrecgc_v5_snapshot.py" in traced
    assert "--validate-only" in traced
    assert str(adoption_root) in traced
    assert not (tmp_path / "fresh-science/common_recourse").exists()


def test_v5_real_head_builds_and_validates_release_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, mock_git=False)
    payload, _summary = v5.build_payload(spec_path=paths["spec"])
    assert v5.validate_payload(payload)["status"] == "PASS"
    contract = payload["aids_comrecgc_exact_route_v5_contract"]
    assert contract["reviewed_core_gate"]["integrated_commit_is_ancestor"] is True
    assert contract["route_release_gate"]["required_commit"] == (
        v5.ROUTE_RELEASE_COMMIT
    )
    assert contract["route_release_gate"]["is_ancestor"] == "true"
    assert contract["snapshot_release_gate"]["release_commit"] == (
        v5.SNAPSHOT_RELEASE_COMMIT
    )
    assert contract["snapshot_adoption_release_gate"]["required_commit"] == (
        "daa9c89e21ec7cf2a812aea082619b2a3e731a65"
    )
    assert contract["snapshot_adoption_release_gate"]["is_ancestor"] == "true"


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
    expected_error = (
        "ALLOWED_OLD_PROCESS_PID_REUSED"
        if mutation == "start_ticks"
        else "UNEXPECTED_COMMON_RECOURSE_PROCESS_SET"
    )
    with pytest.raises(RuntimeError, match=expected_error):
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


@pytest.mark.parametrize(
    "argv",
    [
        [
            "/bin/bash",
            "-c",
            "grep -R run_common_recourse.py /proc/*/cmdline",
        ],
        [
            "/usr/bin/grep",
            "run_common_recourse.py",
            "/tmp/diagnostic.txt",
        ],
        [
            "/bin/bash",
            "-c",
            "python regex='.*run_common_recourse.py.*' monitor-only",
        ],
        [
            str(Path(os.sys.executable).resolve()),
            "/tmp/not_run_common_recourse.py.txt",
            "--output-dir",
            "/tmp/not-science",
        ],
        [
            str(Path(os.sys.executable).resolve()),
            "-c",
            "print('run_common_recourse.py monitor literal')",
        ],
        [
            str(Path(os.sys.executable).resolve()),
            "-m",
            "monitor",
            "run_common_recourse.py",
        ],
    ],
)
def test_v5_process_gate_ignores_literal_diagnostic_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    diagnostic = paths["proc"] / "274001"
    diagnostic.mkdir()
    (diagnostic / "cmdline").write_bytes(
        b"\0".join(token.encode() for token in argv) + b"\0"
    )
    (diagnostic / "stat").write_text(
        "274001 (diagnostic) "
        + " ".join(["S", *(
            "0" for _ in range(18)
        ), "687141122"])
        + "\n",
        encoding="utf-8",
    )
    (diagnostic / "cwd").symlink_to(
        paths["old_project_root"], target_is_directory=True
    )
    result = _process_gate(paths)
    assert result["active_common_recourse_count"] == 1
    assert result["allowed_old_process_count"] == 1


@pytest.mark.parametrize(
    "launch",
    [
        "python",
        "python-u",
        "python-O",
        "python-X",
        "python-combined",
        "direct",
        "python-relative",
        "direct-relative",
    ],
)
def test_v5_process_gate_detects_real_entrypoint_even_when_output_is_rogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    rogue_project = tmp_path / "rogue-project"
    rogue_script = rogue_project / "scripts/baselines/comrecgc/run_common_recourse.py"
    rogue_script.parent.mkdir(parents=True)
    rogue_script.write_text("# real rogue entrypoint fixture\n", encoding="utf-8")
    rogue = paths["proc"] / "274002"
    rogue.mkdir()
    relative_script = "scripts/baselines/comrecgc/run_common_recourse.py"
    if launch == "direct":
        argv = [str(rogue_script)]
    elif launch == "direct-relative":
        argv = [relative_script]
    else:
        argv = [str(Path(os.sys.executable).resolve())]
        if launch == "python-u":
            argv.append("-u")
        elif launch == "python-O":
            argv.append("-O")
        elif launch == "python-X":
            argv.extend(["-X", "dev"])
        elif launch == "python-combined":
            argv.extend(["-u", "-bb", "-i", "-R", "--"])
        argv.append(relative_script if launch == "python-relative" else str(rogue_script))
    argv.extend(["--output-dir", str(tmp_path / "rogue-output")])
    (rogue / "cmdline").write_bytes(
        b"\0".join(token.encode() for token in argv) + b"\0"
    )
    (rogue / "stat").write_text(
        "274002 (python) "
        + " ".join(["S", *("0" for _ in range(18)), "687141123"])
        + "\n",
        encoding="utf-8",
    )
    (rogue / "cwd").symlink_to(rogue_project, target_is_directory=True)
    with pytest.raises(RuntimeError, match="UNEXPECTED_COMMON_RECOURSE_PROCESS_SET"):
        _process_gate(paths)


@pytest.mark.parametrize("entrypoint_state", ["missing", "symlink"])
def test_v5_process_gate_fails_closed_for_unreadable_real_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint_state: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    rogue_project = tmp_path / "rogue-unreadable-project"
    script = rogue_project / "scripts/baselines/comrecgc/run_common_recourse.py"
    script.parent.mkdir(parents=True)
    if entrypoint_state == "symlink":
        target = tmp_path / "physical.py"
        target.write_text("# physical target\n", encoding="utf-8")
        script.symlink_to(target)
    rogue = paths["proc"] / "274003"
    rogue.mkdir()
    (rogue / "cmdline").write_bytes(
        str(Path(os.sys.executable).resolve()).encode()
        + b"\0scripts/baselines/comrecgc/run_common_recourse.py\0"
    )
    (rogue / "stat").write_text(
        "274003 (python) "
        + " ".join(["S", *("0" for _ in range(18)), "687141124"])
        + "\n",
        encoding="utf-8",
    )
    (rogue / "cwd").symlink_to(rogue_project, target_is_directory=True)
    with pytest.raises(
        RuntimeError, match="COMMON_RECOURSE_ENTRYPOINT_IDENTITY_UNREADABLE"
    ):
        _process_gate(paths)


def test_v5_process_gate_rejects_old_pid_reused_by_non_common_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["old_proc"] / "cmdline").write_bytes(b"/usr/bin/sleep\0infinity\0")
    (paths["old_proc"] / "stat").write_text(
        "273939 (sleep) "
        + " ".join(["S", *("0" for _ in range(18)), "687141120"])
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="ALLOWED_OLD_PROCESS_PID_REUSED"):
        _process_gate(paths)


def test_v5_process_gate_rejects_same_generation_non_common_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["old_proc"] / "cmdline").write_bytes(b"/usr/bin/sleep\0infinity\0")
    with pytest.raises(RuntimeError, match="UNEXPECTED_COMMON_RECOURSE_PROCESS_SET"):
        _process_gate(paths)


def test_v5_midrun_process_gate_allows_exact_route_child_and_rejects_rogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    route_project = tmp_path / "route-project"
    route_script = route_project / "scripts/baselines/comrecgc/run_common_recourse.py"
    route_script.parent.mkdir(parents=True)
    route_script.write_text("# route fixture\n", encoding="utf-8")
    route_output = tmp_path / "fresh/common_recourse"
    route_output.mkdir(parents=True)
    root_pid = 5000
    root_ticks = 700001
    route_root = paths["proc"] / str(root_pid)
    route_root.mkdir()
    (route_root / "stat").write_text(
        f"{root_pid} (continuation) "
        + " ".join(["S", "1", *("0" for _ in range(17)), str(root_ticks)])
        + "\n",
        encoding="utf-8",
    )
    child_pid = 5001
    route_child = paths["proc"] / str(child_pid)
    route_child.mkdir()
    (route_child / "stat").write_text(
        f"{child_pid} (python) "
        + " ".join(["S", str(root_pid), *("0" for _ in range(17)), "700002"])
        + "\n",
        encoding="utf-8",
    )
    route_cmdline = (
        str(Path(os.sys.executable).resolve()).encode()
        + b"\0"
        + str(route_script).encode()
        + b"\0--output-dir\0"
        + str(route_output).encode()
        + b"\0"
    )
    (route_child / "cmdline").write_bytes(route_cmdline)
    (route_child / "cwd").symlink_to(route_project, target_is_directory=True)

    allowed = verify_process_set(
        proc_root=paths["proc"],
        allowed_pid=273939,
        allowed_start_ticks=687141119,
        allowed_cmdline_sha256=hashlib.sha256(paths["old_cmdline"]).hexdigest(),
        allowed_output_root=paths["old_output_root"],
        allowed_project_root=paths["old_project_root"],
        allowed_route_root_pid=root_pid,
        allowed_route_root_start_ticks=root_ticks,
        allowed_route_output_root=route_output,
        allowed_route_project_root=route_project,
    )
    assert allowed["active_common_recourse_count"] == 2
    assert allowed["allowed_old_process_count"] == 1
    assert allowed["allowed_route_process_count"] == 1

    rogue_pid = 5002
    rogue = paths["proc"] / str(rogue_pid)
    rogue.mkdir()
    (rogue / "stat").write_text(
        f"{rogue_pid} (python) "
        + " ".join(["S", "1", *("0" for _ in range(17)), "700003"])
        + "\n",
        encoding="utf-8",
    )
    (rogue / "cmdline").write_bytes(route_cmdline)
    (rogue / "cwd").symlink_to(route_project, target_is_directory=True)
    with pytest.raises(RuntimeError, match="UNEXPECTED_COMMON_RECOURSE_PROCESS_SET"):
        verify_process_set(
            proc_root=paths["proc"],
            allowed_pid=273939,
            allowed_start_ticks=687141119,
            allowed_cmdline_sha256=hashlib.sha256(paths["old_cmdline"]).hexdigest(),
            allowed_output_root=paths["old_output_root"],
            allowed_project_root=paths["old_project_root"],
            allowed_route_root_pid=root_pid,
            allowed_route_root_start_ticks=root_ticks,
            allowed_route_output_root=route_output,
            allowed_route_project_root=route_project,
        )


def test_v5_midrun_process_gate_rejects_route_root_pid_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    route_project = tmp_path / "route-project"
    route_script = route_project / "scripts/baselines/comrecgc/run_common_recourse.py"
    route_script.parent.mkdir(parents=True)
    route_script.write_text("# route fixture\n", encoding="utf-8")
    route_output = tmp_path / "fresh/common_recourse"
    route_output.mkdir(parents=True)
    root_pid = 5010
    route_root = paths["proc"] / str(root_pid)
    route_root.mkdir()
    (route_root / "stat").write_text(
        f"{root_pid} (reused) "
        + " ".join(["S", "1", *("0" for _ in range(17)), "900002"])
        + "\n",
        encoding="utf-8",
    )
    child_pid = 5011
    child = paths["proc"] / str(child_pid)
    child.mkdir()
    (child / "stat").write_text(
        f"{child_pid} (python) "
        + " ".join(["S", str(root_pid), *("0" for _ in range(17)), "900003"])
        + "\n",
        encoding="utf-8",
    )
    (child / "cmdline").write_bytes(
        str(Path(os.sys.executable).resolve()).encode()
        + b"\0"
        + str(route_script).encode()
        + b"\0--output-dir\0"
        + str(route_output).encode()
        + b"\0"
    )
    (child / "cwd").symlink_to(route_project, target_is_directory=True)
    with pytest.raises(RuntimeError, match="UNEXPECTED_COMMON_RECOURSE_PROCESS_SET"):
        verify_process_set(
            proc_root=paths["proc"],
            allowed_pid=273939,
            allowed_start_ticks=687141119,
            allowed_cmdline_sha256=hashlib.sha256(paths["old_cmdline"]).hexdigest(),
            allowed_output_root=paths["old_output_root"],
            allowed_project_root=paths["old_project_root"],
            allowed_route_root_pid=root_pid,
            allowed_route_root_start_ticks=900001,
            allowed_route_output_root=route_output,
            allowed_route_project_root=route_project,
        )


def test_v5_highmem_handover_queues_acquires_retains_and_releases(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    supervisor_pid = 4242
    supervisor = proc / str(supervisor_pid)
    supervisor.mkdir(parents=True)
    supervisor_start_ticks = 998877
    (supervisor / "stat").write_text(
        f"{supervisor_pid} (supervisor) "
        + " ".join(["S", *("0" for _ in range(18)), str(supervisor_start_ticks)])
        + "\n",
        encoding="utf-8",
    )
    lock = tmp_path / "highmem.lock"
    lock.touch()
    state = tmp_path / "handover.json"
    old_fd = lock.open("r+")
    fcntl.flock(old_fd.fileno(), fcntl.LOCK_EX)
    root = Path(__file__).resolve().parents[2]
    helper = subprocess.Popen(
        [
            str(Path(os.sys.executable).resolve()),
            "-m",
            "src.utils.aids_comrecgc_v5_lock_handover",
            "--lock-path",
            str(lock),
            "--state-path",
            str(state),
            "--supervisor-pid",
            str(supervisor_pid),
            "--proc-root",
            str(proc),
            "--poll-seconds",
            "0.01",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
    )

    def wait_status(expected: str) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if state.is_file():
                try:
                    if json.loads(state.read_text())["status"] == expected:
                        return
                except (json.JSONDecodeError, KeyError):
                    pass
            time.sleep(0.01)
        raise AssertionError(f"handover did not reach {expected}")

    third_fd = lock.open("r+")
    try:
        wait_status("QUEUED")
        with pytest.raises(BlockingIOError):
            fcntl.flock(third_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(old_fd.fileno(), fcntl.LOCK_UN)
        wait_status("ACQUIRED")
        with pytest.raises(BlockingIOError):
            fcntl.flock(third_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (supervisor / "stat").unlink()
        supervisor.rmdir()
        assert helper.wait(timeout=5) == 0
        fcntl.flock(third_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if helper.poll() is None:
            helper.terminate()
            helper.wait(timeout=5)
        fcntl.flock(third_fd.fileno(), fcntl.LOCK_UN)
        third_fd.close()
        old_fd.close()


@pytest.mark.parametrize("replace_phase", ["QUEUED", "ACQUIRED"])
def test_v5_highmem_handover_rejects_lock_path_replacement(
    tmp_path: Path, replace_phase: str
) -> None:
    proc = tmp_path / "proc"
    supervisor_pid = 4243
    supervisor = proc / str(supervisor_pid)
    supervisor.mkdir(parents=True)
    (supervisor / "stat").write_text(
        f"{supervisor_pid} (supervisor) "
        + " ".join(["S", *("0" for _ in range(18)), "998878"])
        + "\n",
        encoding="utf-8",
    )
    lock = tmp_path / "highmem.lock"
    lock.touch()
    state = tmp_path / "handover.json"
    old_fd = lock.open("r+")
    fcntl.flock(old_fd.fileno(), fcntl.LOCK_EX)
    root = Path(__file__).resolve().parents[2]
    helper = subprocess.Popen(
        [
            str(Path(os.sys.executable).resolve()),
            "-m",
            "src.utils.aids_comrecgc_v5_lock_handover",
            "--lock-path",
            str(lock),
            "--state-path",
            str(state),
            "--supervisor-pid",
            str(supervisor_pid),
            "--proc-root",
            str(proc),
            "--poll-seconds",
            "0.01",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if state.is_file() and json.loads(state.read_text())["status"] == "QUEUED":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("handover did not queue")
        if replace_phase == "ACQUIRED":
            fcntl.flock(old_fd.fileno(), fcntl.LOCK_UN)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if json.loads(state.read_text())["status"] == "ACQUIRED":
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("handover did not acquire")
        lock.unlink()
        lock.touch()
        assert helper.wait(timeout=5) == 75
        failed = json.loads(state.read_text())
        assert failed["status"] == "FAILED"
        assert "HIGHMEM_LOCK_PATH_IDENTITY_CHANGED" in failed["error"]
        replacement_fd = lock.open("r+")
        try:
            fcntl.flock(replacement_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            fcntl.flock(replacement_fd.fileno(), fcntl.LOCK_UN)
            replacement_fd.close()
    finally:
        if helper.poll() is None:
            helper.terminate()
            helper.wait(timeout=5)
        fcntl.flock(old_fd.fileno(), fcntl.LOCK_UN)
        old_fd.close()


def test_v5_science_exec_establishes_session_before_fixed_script_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/autodl/run_comrecgc_standardized_continuation.sh"
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(science_exec.os, "setsid", lambda: calls.append(("setsid", None)))

    def fake_exec(path: str, argv: list[str]) -> None:
        calls.append((path, argv))
        raise RuntimeError("EXEC_CAPTURED")

    monkeypatch.setattr(science_exec.os, "execv", fake_exec)
    with pytest.raises(RuntimeError, match="EXEC_CAPTURED"):
        science_exec.main(
            ["--project-root", str(root), "--script", str(script)]
        )
    assert calls == [
        ("setsid", None),
        ("/bin/bash", ["bash", str(script.resolve(strict=True))]),
    ]
    with pytest.raises(RuntimeError, match="identity changed"):
        science_exec.main(
            ["--project-root", str(root), "--script", str(root / "README.md")]
        )


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


def test_v5_rejects_replaced_global_highmem_lock_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    lock = paths["runtime"] / "locks/comrecgc_common_recourse_highmem.lock"
    lock.unlink()
    replacement = _file(tmp_path / "replacement-highmem.lock", "")
    lock.symlink_to(replacement)
    with pytest.raises(RepairManifestError, match="must be physical"):
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


def test_v5_rejects_snapshot_bypass_or_production_test_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload, _summary = v5.build_payload(spec_path=paths["spec"])
    tasks = {task["id"]: task for task in payload["tasks"]}
    snapshot_task = tasks[v5.SNAPSHOT_TASK_ID]
    science_task = tasks[v5.TASK_ID]

    snapshot_task["environment"]["GPU_REQUIRED"] = "1"
    with pytest.raises(RepairManifestError, match="snapshot adoption command"):
        v5.validate_payload(payload)
    snapshot_task["environment"]["GPU_REQUIRED"] = "0"

    snapshot_task["required_output_files"].remove("snapshot_adoption_manifest.json")
    with pytest.raises(RepairManifestError, match="snapshot adoption artifact"):
        v5.validate_payload(payload)
    snapshot_task["required_output_files"].insert(
        0, "snapshot_adoption_manifest.json"
    )

    science_task["environment"]["COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT"] = str(
        paths["pair_root"]
    )
    with pytest.raises(RepairManifestError, match="exact snapshot"):
        v5.validate_payload(payload)


def test_v5_rejects_missing_snapshot_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload, _summary = v5.build_payload(spec_path=paths["spec"])
    science_task = next(task for task in payload["tasks"] if task["id"] == v5.TASK_ID)
    science_task["depends_on"] = [v5.SELECTOR_TASK_ID]
    with pytest.raises(RepairManifestError, match="selector-freeze dependency"):
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
    assert v5.CONTROLLER_ID == (
        "four_methods_four_datasets_aids_comrecgc_exact_route_v5_snapshot_adopt_v1"
    )
    assert v5.SNAPSHOT_TASK_ID.endswith("_snapshot_adoption_v5_v1")
    assert v5.TASK_ID.endswith("_snapshot_adopt_v1")
    assert v5.REVIEWED_SOURCE_CORE_COMMIT == (
        "645c6e51b7abcdc5dd4a9e0a1226d71d020880da"
    )
    assert v5.INTEGRATED_REVIEWED_CORE_COMMIT == (
        "8c371b1c8ee1d8188555581c4f8e8b6060ae42eb"
    )
    assert v5.REVIEWED_CORE_COMMIT == v5.INTEGRATED_REVIEWED_CORE_COMMIT
    assert v5.ROUTE_RELEASE_COMMIT == "a6cdfd51d19af7f390d1cbc9d00827c97baee150"
    assert v5.SNAPSHOT_RELEASE_COMMIT == (
        "8b99498a0c1beab11a6844ddf5f6f9d7c2c4458f"
    )
    root = Path(__file__).resolve().parents[2]
    template = json.loads(
        (
            root / "configs/autodl/aids_comrecgc_exact_route_v5.template.json"
        ).read_text(encoding="utf-8")
    )
    assert template["allowed_old_read_only_process"]["cmdline_sha256"] == (
        "792679fed417737f85462d940243153e5081d8b80c7dab663591131c5bbd51b8"
    )
    assert template["python"] == "/root/miniconda3/envs/smiles_pip118/bin/python3.10"
    assert template["snapshot_adoption_required"] is True
    assert template["adopted_snapshot"]["owner_manifest_sha256"] == (
        "6c6bea5157d86f2323681bc00310c1b89a2929ee635065965296885aed6524e6"
    )
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
    adoption_slurm = (
        root / "scripts/slurm/adopt_aids_comrecgc_v5_snapshot.sh"
    ).read_text(encoding="utf-8")
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "exit 78",
    ):
        assert token in adoption_slurm


def test_v5_real_head_accepts_integrated_reviewed_core_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = v5._require_reviewed_core_equivalence(root)
    assert evidence["reviewed_source_commit"] == v5.REVIEWED_SOURCE_CORE_COMMIT
    assert evidence["integrated_equivalent_commit"] == (
        v5.INTEGRATED_REVIEWED_CORE_COMMIT
    )
    assert evidence["integrated_commit_is_ancestor"] is True
    assert evidence["equivalence_basis"] == (
        "exact-git-blob-and-current-content-sha256"
    )
    assert set(evidence["files"]) == set(v5.REVIEWED_CORE_FILE_IDENTITIES)


def test_v5_real_head_accepts_snapshot_release_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = v5._require_snapshot_release(root)
    assert evidence["release_commit"] == v5.SNAPSHOT_RELEASE_COMMIT
    assert evidence["release_commit_is_ancestor"] is True
    assert set(evidence["files"]) == set(v5.SNAPSHOT_RELEASE_FILE_IDENTITIES)


def test_v5_snapshot_release_gate_rejects_working_content_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    original = v5.sha256_file

    def changed(path: str | Path) -> str:
        if Path(path).name == "aids_comrecgc_v5_snapshot.py":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(v5, "sha256_file", changed)
    with pytest.raises(RepairManifestError, match="snapshot release content changed"):
        v5._require_snapshot_release(root)


def test_v5_reviewed_core_gate_rejects_integrated_blob_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    identities = {
        path: dict(identity)
        for path, identity in v5.REVIEWED_CORE_FILE_IDENTITIES.items()
    }
    first = next(iter(identities))
    identities[first]["git_blob"] = "0" * 40
    monkeypatch.setattr(v5, "REVIEWED_CORE_FILE_IDENTITIES", identities)
    with pytest.raises(RepairManifestError, match="integrated reviewed core blob changed"):
        v5._require_reviewed_core_equivalence(root)
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
