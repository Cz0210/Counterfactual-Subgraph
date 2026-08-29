from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import threading
import subprocess
import time
from types import SimpleNamespace

import pytest

from src.utils import autodl_aids_comrecgc_exact_recovery_controller_v1 as recovery


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source_authority(tmp_path: Path) -> dict[str, object]:
    sha_fields = (
        "source_controller_manifest_sha256",
        "close_pass_gate_sha256",
        "failed_final_gate_sha256",
        "failed_shortcut_artifact_sha256",
        "failed_checkpoint_sha256",
        "adaptive_selection_sha256",
        "anchor_indices_sha256",
        "anchor_rows_sha256",
        "failure_indices_sha256",
        "anchor_edges_sha256",
        "close_pair_manifest_sha256",
        "pair_semantics_receipt_sha256",
        "pair_store_manifest_sha256",
        "physical_pairs_sha256",
        "normalized_distances_sha256",
        "close_bitmap_sha256",
        "source_vectors_sha256",
    )
    value: dict[str, object] = {
        "physical_pair_count": recovery.EXPECTED_ROWS,
        "pair_store_regenerated": False,
        "seed_failure_scan_reexecuted": False,
        "source_pair_store_access": "read_only_zero_copy",
        "failed_final_gate_status": "FAILED",
        "failed_final_reason": "anchor_epsilon_graph_disconnected",
        "failed_final_gate_ordinary_pass_eligible": False,
    }
    for index, field in enumerate(sha_fields, 1):
        value[field] = f"{index:064x}"
        value[field[: -len("_sha256")] + "_path"] = str(
            tmp_path / "source" / f"{index}.artifact"
        )
    return value


def _spec(
    tmp_path: Path,
    *,
    pins_ready: bool = False,
    deployment_authorized: bool = False,
) -> tuple[Path, dict[str, object]]:
    project_root = Path(__file__).resolve().parents[2]
    adoption_entrypoint = (
        project_root / "scripts/autodl/status_aids_comrecgc_exact_recovery.py"
    )
    stage_entrypoint = (
        project_root / "scripts/autodl/run_aids_comrecgc_exact_recovery_stage.py"
    )
    python = Path(sys.executable).resolve(strict=True)
    authority_parent = tmp_path / "adoption-authority"
    authority_parent.mkdir()
    cid = "aids_comrecgc_exact_recovery_v1_20260825T010203Z_deadbeef"
    controller_root = tmp_path / cid
    controller_manifest_path = tmp_path / "controller-manifest.json"
    source_authority = _source_authority(tmp_path)
    science_root = controller_root / "science"
    output_by_stage = {
        recovery.ADOPTION_STAGE: authority_parent / "cid-adoption",
        recovery.SUBSET_STAGE: controller_root / "subset_preflight",
        recovery.EXACT_STAGE: science_root / "common_recourse/external_memory",
        recovery.DOWNSTREAM_STAGE: science_root
        / "common_recourse/external_memory/all_core_component_summary",
        recovery.FINAL_STAGE: science_root,
    }
    terminal_by_stage = {
        recovery.ADOPTION_STAGE: output_by_stage[recovery.ADOPTION_STAGE]
        / "failed_selection_adoption_receipt.json",
        recovery.SUBSET_STAGE: output_by_stage[recovery.SUBSET_STAGE]
        / "subset_stage_receipt.json",
        recovery.EXACT_STAGE: output_by_stage[recovery.EXACT_STAGE]
        / "exact_recovery_receipt.json",
        recovery.DOWNSTREAM_STAGE: output_by_stage[recovery.DOWNSTREAM_STAGE]
        / "run_manifest.json",
        recovery.FINAL_STAGE: output_by_stage[recovery.FINAL_STAGE]
        / "exact_recovery_freeze_receipt.json",
    }
    stages = []
    for stage_id in recovery.STAGE_ORDER:
        binding_values: dict[str, tuple[str, str]]
        if stage_id == recovery.ADOPTION_STAGE:
            binding_values = {
                "output": ("--output-dir", str(output_by_stage[stage_id])),
            }
            entrypoint = adoption_entrypoint
            argv_prefix = [str(python), str(entrypoint)]
        elif stage_id == recovery.SUBSET_STAGE:
            binding_values = {
                "output": ("--output-dir", str(output_by_stage[stage_id])),
                "controller_manifest": (
                    "--controller-manifest",
                    str(controller_manifest_path),
                ),
                "adoption_gate": (
                    "--adoption-gate",
                    str(controller_root / "gates/01_failed_selection_adoption.json"),
                ),
            }
            entrypoint = stage_entrypoint
            argv_prefix = [str(python), str(entrypoint), "subset"]
        elif stage_id == recovery.EXACT_STAGE:
            binding_values = {
                "output": ("--output-dir", str(output_by_stage[stage_id])),
                "controller_manifest": (
                    "--controller-manifest",
                    str(controller_manifest_path),
                ),
                "adoption_gate": (
                    "--adoption-gate",
                    str(controller_root / "gates/01_failed_selection_adoption.json"),
                ),
                "subset_gate": (
                    "--subset-gate",
                    str(controller_root / "gates/02_production_subset_equivalence.json"),
                ),
            }
            entrypoint = stage_entrypoint
            argv_prefix = [str(python), str(entrypoint), "exact"]
        elif stage_id == recovery.DOWNSTREAM_STAGE:
            binding_values = {
                "output": ("--output-dir", str(output_by_stage[stage_id])),
                "controller_manifest": (
                    "--controller-manifest",
                    str(controller_manifest_path),
                ),
                "exact_gate": (
                    "--exact-gate",
                    str(controller_root / "gates/03_exact_component_recovery.json"),
                ),
            }
            entrypoint = stage_entrypoint
            argv_prefix = [str(python), str(entrypoint), "downstream"]
        else:
            binding_values = {
                "output": ("--output-dir", str(output_by_stage[stage_id])),
                "controller_manifest": (
                    "--controller-manifest",
                    str(controller_manifest_path),
                ),
                "adoption_gate": (
                    "--adoption-gate",
                    str(controller_root / "gates/01_failed_selection_adoption.json"),
                ),
                "subset_gate": (
                    "--subset-gate",
                    str(controller_root / "gates/02_production_subset_equivalence.json"),
                ),
                "exact_gate": (
                    "--exact-gate",
                    str(controller_root / "gates/03_exact_component_recovery.json"),
                ),
                "downstream_gate": (
                    "--downstream-gate",
                    str(controller_root / "gates/04_component_downstream_radius_ab.json"),
                ),
            }
            entrypoint = stage_entrypoint
            argv_prefix = [str(python), str(entrypoint), "final"]
        argv = list(argv_prefix)
        for flag, bound_value in binding_values.values():
            argv.extend([flag, bound_value])
        resume_argv = (
            [*argv, "--resume"]
            if stage_id
            in {
                recovery.SUBSET_STAGE,
                recovery.EXACT_STAGE,
                recovery.DOWNSTREAM_STAGE,
                recovery.FINAL_STAGE,
            }
            else None
        )
        stages.append(
            {
                "stage_id": stage_id,
                "kind": recovery.STAGE_KINDS[stage_id],
                "dependencies": list(recovery.DEPENDENCIES[stage_id]),
                "output_dir": str(output_by_stage[stage_id]),
                "terminal_path": str(terminal_by_stage[stage_id]),
                "terminal_schema": f"fixture-{stage_id}",
                "entrypoint_sha256": recovery.sha256_file(entrypoint),
                "commands": {
                    "fresh": argv,
                    "resume": resume_argv,
                },
                "argv_bindings": {
                    role: {"flag": flag, "value": bound_value}
                    for role, (flag, bound_value) in binding_values.items()
                },
                "progress_checkpoint_path": (
                    str(output_by_stage[stage_id] / "dbscan/checkpoint.json")
                    if stage_id == recovery.EXACT_STAGE
                    else None
                ),
                "progress_field": (
                    recovery.EXACT_MONOTONIC_PROGRESS_FIELD
                    if stage_id == recovery.EXACT_STAGE
                    else None
                ),
            }
        )
    budget = recovery.derive_output_budget(
        row_count=recovery.EXPECTED_ROWS,
        vector_dim=recovery.EXPECTED_VECTOR_DIM,
        subset_size=recovery.DEFAULT_SUBSET_SIZE,
        subset_count=5,
        block_size=recovery.DEFAULT_BLOCK_SIZE,
    )
    head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    pins = {
        field: (head if pins_ready else None)
        for field in recovery.REQUIRED_RELEASE_PINS
    }
    pins["science_commit"] = recovery.SCIENCE_RELEASE_COMMIT if pins_ready else None
    runtime_root = tmp_path / "runtime-inputs"
    runtime_directories = {
        field: runtime_root / field
        for field in (
            "source_generation_root",
            "upstream_root",
            "dataset_dir",
            "molclr_root",
            "pair_store_owner_root",
        )
    }
    for directory in runtime_directories.values():
        directory.mkdir(parents=True)
    runtime_files = {
        field: runtime_root / f"{field}.bin"
        for field in (
            "source_csv",
            "distance_checkpoint",
            "dataset_csv",
            "teacher_path",
            "molclr_checkpoint",
            "thresholds_path",
        )
    }
    for artifact in runtime_files.values():
        artifact.write_bytes(b"fixture\n")
    value: dict[str, object] = {
        "schema_version": recovery.SPEC_SCHEMA,
        "controller_id": recovery.CONTROLLER_ID,
        "cid": cid,
        "project_root": str(project_root),
        "controller_root": str(controller_root),
        "controller_manifest_path": str(controller_manifest_path),
        "adoption_authority_parent": str(authority_parent),
        "production_deployment_authorized": deployment_authorized,
        "stages": stages,
        "adoption_contract": {
            "receipt_schema": "aids_comrecgc_c766_failed_selection_adoption_v2",
            "receipt_name": "failed_selection_adoption_receipt.json",
            "ready_marker_name": "RECOVERY_EVIDENCE_READY",
            "receipt_status": "RECOVERY_ONLY_READY",
            "artifact_kind": "aids_c766_failed_selection_recovery_evidence_v2",
            "validator_module": "src.baselines.comrecgc.failed_selection_adoption",
            "validator_callable": (
                "verify_aids_c766_failed_selection_recovery_evidence"
            ),
            "validator_module_sha256": ("d" * 64 if pins_ready else None),
            "validator_api": recovery.ADOPTION_VALIDATOR_API,
            "authority_profile_sha256": "a" * 64,
            "expected_task_state_projection_sha256": dict(
                recovery.EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256
            ),
            "recovery_only": True,
            "ordinary_pass_dependency_eligible": False,
            "dbscan_partition_proven": False,
            "authority_parent_allowed_entries": [],
        },
        "source_authority": source_authority,
        "runtime_inputs": {
            **{key: str(value) for key, value in runtime_directories.items()},
            **{key: str(value) for key, value in runtime_files.items()},
            "expected_sklearn_version": "1.7.2",
            "theta_star": 0.1,
            "cost_cap": None,
        },
        "resources": {
            "row_count": recovery.EXPECTED_ROWS,
            "vector_dim": recovery.EXPECTED_VECTOR_DIM,
            "subset_size": recovery.DEFAULT_SUBSET_SIZE,
            "subset_max_attempts": recovery.SUBSET_MAX_ATTEMPTS,
            "block_size": recovery.DEFAULT_BLOCK_SIZE,
            "partial_stage_archive_count": recovery.PARTIAL_STAGE_ARCHIVE_COUNT,
            "partial_stage_archive_max_bytes_each": (
                recovery.PARTIAL_STAGE_ARCHIVE_MAX_BYTES
            ),
            "startup_barrier_max_generations": (
                recovery.STARTUP_BARRIER_MAX_GENERATIONS
            ),
            "startup_barrier_record_max_bytes": (
                recovery.STARTUP_BARRIER_RECORD_MAX_BYTES
            ),
            "startup_barrier_publication_file_multiplier": (
                recovery.STARTUP_BARRIER_PUBLICATION_FILE_MULTIPLIER
            ),
            "safety_floor_bytes": recovery.DEFAULT_SAFETY_FLOOR_BYTES,
            "budget": budget,
            "max_rss_bytes": recovery.DEFAULT_MAX_RSS_BYTES,
            "max_rss_scope": (
                "exact_dbscan_process_with_native_peak_certificate"
            ),
            "thread_count": recovery.DEFAULT_THREAD_COUNT,
            "cpu_only": True,
            "gpu_lock_required": False,
            "proc_root": "/proc",
            "old_brute_handover": recovery.handover_contract(),
            "coexistence_probe": {
                "min_progress_rows": recovery.DEFAULT_BLOCK_SIZE,
                "max_load_per_cpu": 0.8,
                "max_iowait_fraction": 0.35,
                "timeout_seconds": 1800,
            },
        },
        "release_pins": pins,
    }
    path = tmp_path / "spec.json"
    _write_json(path, value)
    return path, value


def _built_manifest(tmp_path: Path, **kwargs: object) -> tuple[Path, dict[str, object]]:
    spec_path, _ = _spec(tmp_path, **kwargs)
    output = tmp_path / "controller-manifest.json"
    value = recovery.build_controller_manifest(spec_path=spec_path, output_path=output)
    return output, value


def _source_artifacts(manifest: dict[str, object]) -> list[dict[str, str]]:
    source = manifest["source_authority"]
    assert isinstance(source, dict)
    return [
        {
            "path": str(source[field[: -len("_sha256")] + "_path"]),
            "sha256": str(value),
        }
        for field, value in source.items()
        if field.endswith("_sha256")
    ]


def test_dag_order_subset_is_a_preflight_before_full_recovery(tmp_path: Path) -> None:
    _path, manifest = _built_manifest(tmp_path)
    assert manifest["stage_order"] == [
        recovery.ADOPTION_STAGE,
        recovery.SUBSET_STAGE,
        recovery.EXACT_STAGE,
        recovery.DOWNSTREAM_STAGE,
        recovery.FINAL_STAGE,
    ]
    assert manifest["stages"][1]["dependencies"] == [recovery.ADOPTION_STAGE]
    assert manifest["stages"][2]["dependencies"] == [
        recovery.ADOPTION_STAGE,
        recovery.SUBSET_STAGE,
    ]


def test_resource_budget_is_derived_and_below_old_100gib_floor(tmp_path: Path) -> None:
    _path, manifest = _built_manifest(tmp_path)
    budget = manifest["resources"]["budget"]
    assert budget["zero_copy_source_bytes_excluded"] is True
    assert budget["source_pair_store_regenerated"] is False
    assert budget["subset_max_attempts"] == 8
    assert budget["fixed_bounds"][
        "all_subset_attempts_dense_edge_upper_bound"
    ] == 8 * 5 * recovery.DEFAULT_SUBSET_SIZE**2 * 8
    assert budget["fixed_bounds"][
        "startup_barrier_record_and_temp_upper_bound"
    ] == 2 * (4 + 5) * 32 * 64 * 1024
    assert budget["partial_stage_archive_max_bytes_each"] == 1024**3
    assert budget["max_output_bytes"] == sum(budget["arrays"].values()) + sum(
        budget["fixed_bounds"].values()
    )
    assert budget["max_output_bytes"] < 10 * 1024**3
    assert budget["minimum_free_bytes_before_launch"] < 20 * 1024**3
    assert manifest["resources"]["max_rss_bytes"] == 96 * 1024**3
    assert manifest["resources"]["max_rss_scope"] == (
        "exact_dbscan_process_with_native_peak_certificate"
    )
    assert manifest["resources"]["thread_count"] == 8


@pytest.mark.parametrize("thread_count", [0, 7, 13, 16])
def test_controller_rejects_worker_count_outside_fast_release_bound(
    tmp_path: Path, thread_count: int
) -> None:
    spec_path, value = _spec(tmp_path)
    value["resources"]["thread_count"] = thread_count
    _write_json(spec_path, value)
    with pytest.raises(
        recovery.RecoveryControllerError, match="between 8 and 12"
    ):
        recovery.build_controller_payload(spec_path)


def test_disk_preflight_enforces_maximum_output_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "controller"
    root.mkdir()
    payload = root / "payload.bin"
    manifest = {
        "controller_root": str(root),
        "resources": {
            "budget": {"max_output_bytes": 10, "safety_floor_bytes": 5}
        },
    }
    monkeypatch.setattr(
        recovery.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=10_000),
    )
    payload.write_bytes(b"123456789")
    assert recovery._disk_preflight(manifest)["existing_output_bytes"] == 9
    payload.write_bytes(b"1234567890")
    assert recovery._disk_preflight(manifest)["remaining_output_budget_bytes"] == 0
    payload.write_bytes(b"12345678901")
    with pytest.raises(recovery.RecoveryControllerError, match="OUTPUT_BUDGET_EXCEEDED"):
        recovery._disk_preflight(manifest)


def test_state_growth_is_reserved_before_atomic_replace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "controller"
    root.mkdir()
    state_path = root / "state.json"
    old_state = {"status": "RUNNING"}
    new_state = {"status": "RUNNING", "payload": "x" * 512}
    _write_json(state_path, old_state)
    old_bytes = state_path.read_bytes()
    new_size = len(recovery._json_payload_bytes(new_state))
    manifest = {
        "controller_root": str(root),
        "resources": {
            "budget": {
                "max_output_bytes": new_size - 1,
                "safety_floor_bytes": 0,
            }
        },
    }
    with pytest.raises(
        recovery.RecoveryControllerError,
        match="OUTPUT_BUDGET_RESERVATION_EXCEEDED",
    ):
        recovery._save_state(
            manifest,
            root,
            new_state,
            lambda: None,
            refresh_timestamp=False,
        )
    assert state_path.read_bytes() == old_bytes

    manifest["resources"]["budget"]["max_output_bytes"] = new_size
    recovery._save_state(
        manifest,
        root,
        new_state,
        lambda: None,
        refresh_timestamp=False,
    )
    assert json.loads(state_path.read_text(encoding="utf-8")) == new_state


def test_final_terminal_reserves_terminal_state_and_pass_before_any_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "controller"
    root.mkdir()
    state = {"schema_version": "fixture", "status": "RUNNING"}
    _write_json(root / "state.json", state)
    existing = (root / "state.json").stat().st_size
    manifest = {
        "manifest_path": str(tmp_path / "manifest.json"),
        "manifest_sha256": "a" * 64,
        "controller_root": str(root),
        "resources": {
            "budget": {
                "max_output_bytes": existing + 1,
                "safety_floor_bytes": 0,
            }
        },
    }
    monkeypatch.setattr(
        recovery,
        "_open_gate",
        lambda value, stage: {"stage_id": stage, "gate_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        recovery,
        "_validate_controller_owner_claim",
        lambda value: {
            "owner_claim_path": str(root / "owner.json"),
            "owner_claim_sha256": "c" * 64,
            "root_preclaim": {"inode": 1},
        },
    )
    held = SimpleNamespace(identity={"inode": 2}, verify=lambda: None)
    with pytest.raises(
        recovery.RecoveryControllerError,
        match="OUTPUT_BUDGET_RESERVATION_EXCEEDED",
    ):
        recovery._publish_final_terminal(manifest, root, held, state)
    assert not (root / "terminal.json").exists()
    assert not (root / "PASS").exists()
    assert json.loads((root / "state.json").read_text(encoding="utf-8")) == {
        "schema_version": "fixture",
        "status": "RUNNING",
    }


def test_live_bound_worker_gets_graceful_sigterm_on_initial_budget_breach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "proc").mkdir()
    argv = ["python", "exact-worker.py"]
    argv_sha = recovery.stable_json_sha256(argv)
    stage = {
        "stage_id": recovery.EXACT_STAGE,
        "terminal_path": str(tmp_path / "terminal.json"),
        "commands": {"fresh_sha256": argv_sha, "resume_sha256": "b" * 64},
    }
    manifest = {"resources": {"proc_root": str(tmp_path / "proc")}}
    state = {
        "worker": {
            "stage_id": recovery.EXACT_STAGE,
            "pid": 1234,
            "start_ticks": 5678,
            "argv_sha256": argv_sha,
        }
    }
    monkeypatch.setattr(
        recovery,
        "_disk_preflight",
        lambda value: (_ for _ in ()).throw(
            recovery.RecoveryControllerError("RECOVERY_OUTPUT_BUDGET_EXCEEDED")
        ),
    )
    monkeypatch.setattr(recovery, "_pid_alive", lambda *args, **kwargs: True)
    monkeypatch.setattr(recovery, "_proc_argv", lambda *args, **kwargs: argv)
    monkeypatch.setattr(
        recovery,
        "_worker_actual_argv_is_bound",
        lambda **kwargs: (
            kwargs["worker"].get("stage_id") == recovery.EXACT_STAGE
            and kwargs["worker"].get("argv_sha256") == argv_sha
            and list(kwargs["actual_argv"] or ()) == argv
        ),
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        recovery.os, "killpg", lambda pid, value: signals.append((pid, value))
    )
    with pytest.raises(recovery.RecoveryControllerError, match="OUTPUT_BUDGET"):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state=state,
            guard=lambda: None,
            poll_seconds=0.1,
        )
    assert signals == [(1234, recovery.signal.SIGTERM)]

    state["worker"] = {
        **state["worker"],
        "stage_id": recovery.SUBSET_STAGE,
    }
    signals.clear()
    with pytest.raises(recovery.RecoveryControllerError, match="OUTPUT_BUDGET"):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state=state,
            guard=lambda: None,
            poll_seconds=0.1,
        )
    assert signals == []

    state["worker"] = {
        **state["worker"],
        "stage_id": recovery.EXACT_STAGE,
        "argv_sha256": "f" * 64,
    }
    with pytest.raises(recovery.RecoveryControllerError, match="OUTPUT_BUDGET"):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state=state,
            guard=lambda: None,
            poll_seconds=0.1,
        )
    assert signals == []

    state["worker"] = {
        **state["worker"],
        "argv_sha256": argv_sha,
    }
    observed_argv = iter((argv, ["different-program"]))
    monkeypatch.setattr(
        recovery, "_proc_argv", lambda *args, **kwargs: next(observed_argv)
    )
    with pytest.raises(recovery.RecoveryControllerError, match="OUTPUT_BUDGET"):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state=state,
            guard=lambda: None,
            poll_seconds=0.1,
        )
    assert signals == []

    state["worker"] = None
    signals.clear()
    with pytest.raises(recovery.RecoveryControllerError, match="OUTPUT_BUDGET"):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state=state,
            guard=lambda: None,
            poll_seconds=0.1,
        )
    assert signals == []


class _SpawnLifecycleProcess:
    def __init__(
        self,
        *,
        pid: int = 4242,
        poll_result: int | None = None,
        wait_result: int = 0,
        wait_error: BaseException | None = None,
        on_wait: object | None = None,
    ) -> None:
        self.pid = pid
        self._poll_result = poll_result
        self._wait_result = wait_result
        self._wait_error = wait_error
        self._on_wait = on_wait

    def poll(self) -> int | None:
        return self._poll_result

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if callable(self._on_wait):
            self._on_wait()
        if self._wait_error is not None:
            raise self._wait_error
        return self._wait_result


def _spawn_lifecycle_contract(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], list[str]]:
    (tmp_path / "proc").mkdir(exist_ok=True)
    argv = [sys.executable, "subset-worker.py"]
    argv_sha = recovery.stable_json_sha256(argv)
    stage: dict[str, object] = {
        "stage_id": recovery.SUBSET_STAGE,
        "output_dir": str(tmp_path / "stage-output"),
        "terminal_path": str(tmp_path / "stage-output/terminal.json"),
        "commands": {
            "fresh": argv,
            "resume": [*argv, "--resume"],
            "fresh_sha256": argv_sha,
            "resume_sha256": recovery.stable_json_sha256([*argv, "--resume"]),
        },
    }
    manifest: dict[str, object] = {
        "project_root": str(tmp_path),
        "resources": {"proc_root": str(tmp_path / "proc"), "thread_count": 12},
    }
    return manifest, stage, argv


@pytest.mark.parametrize("window", ("temp_only", "final_and_temp"))
def test_controller_prearm_reconciles_barrier_record_publication_crash(
    tmp_path: Path, window: str
) -> None:
    _manifest, stage, argv = _spawn_lifecycle_contract(tmp_path)
    root = tmp_path / "controller"
    (root / "logs").mkdir(parents=True)
    lock_path, record_path = recovery._startup_barrier_paths(
        root, stage_id=recovery.SUBSET_STAGE, generation=0
    )
    barrier = recovery.arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=argv,
    )
    barrier.abort()
    temporary = Path(f"{record_path}.tmp.fixture")
    if window == "temp_only":
        record_path.rename(temporary)
    else:
        os.link(record_path, temporary, follow_symlinks=False)
    binding = {
        "schema_version": recovery.STARTUP_BARRIER_BINDING_SCHEMA,
        "stage_id": recovery.SUBSET_STAGE,
        "generation": 0,
        "phase": "PRE_ARM",
        "record_path": str(record_path),
        "lock_path": str(lock_path),
        "target_argv_sha256": recovery.stable_json_sha256(argv),
    }
    target, record = recovery._validate_startup_barrier_binding(
        root=root,
        stage=stage,
        binding=binding,
        allowed_phases={"PRE_ARM"},
    )
    assert target == argv
    assert (record is not None) is (window == "final_and_temp")
    assert not temporary.exists()


def test_fresh_spawn_save_state_failure_terminates_owned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, stage, argv = _spawn_lifecycle_contract(tmp_path)
    process = _SpawnLifecycleProcess()
    monkeypatch.setattr(recovery, "_disk_preflight", lambda value: {})
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(recovery, "_read_proc_start_ticks", lambda *args, **kwargs: 99)
    state: dict[str, object] = {"worker": None, "startup_barrier": None}

    def observed_launcher(*_args: object, **_kwargs: object) -> list[str]:
        binding = state["startup_barrier"]
        assert isinstance(binding, dict)
        record = recovery.validate_startup_barrier_record(binding["record_path"])
        return list(record.launcher_argv)

    monkeypatch.setattr(recovery, "_proc_argv", observed_launcher)
    monkeypatch.setattr(
        recovery,
        "_host_sample",
        lambda **kwargs: {
            "load_per_cpu": 0.0,
            "cpu_total_ticks": 100,
            "cpu_iowait_ticks": 0,
        },
    )
    def fail_bound_state(
        _manifest: object,
        _root: Path,
        value: dict[str, object],
        _guard: object,
    ) -> None:
        if value.get("worker") is not None:
            raise OSError("state fsync failed")

    monkeypatch.setattr(recovery, "_save_state", fail_bound_state)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        recovery.os, "killpg", lambda pid, value: signals.append((pid, value))
    )

    with pytest.raises(OSError, match="state fsync failed"):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state=state,
            guard=lambda: None,
            poll_seconds=0.1,
        )
    # The target has not been released, so EOF abort is sufficient and no
    # scientific process is ever signalled or executed.
    assert signals == []


def test_process_group_quiescence_ignores_zombie_but_not_live_descendant(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    _file_stat = lambda pid, state, pgrp: (
        proc / str(pid) / "stat",
        f"{pid} (worker) {state} 1 {pgrp} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
    )
    zombie_path, zombie = _file_stat(4242, "Z", 4242)
    live_path, live = _file_stat(4243, "S", 4242)
    zombie_path.parent.mkdir(parents=True)
    zombie_path.write_text(zombie, encoding="utf-8")
    live_path.parent.mkdir(parents=True)
    live_path.write_text(live, encoding="utf-8")
    assert recovery._process_group_member_pids(4242, proc_root=proc) == (4243,)
    live_path.unlink()
    assert recovery._process_group_member_pids(4242, proc_root=proc) == ()
    recovery._wait_for_process_group_quiescence(4242, proc_root=proc)


def test_fresh_spawn_pid_bind_timeout_terminates_owned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, stage, _argv = _spawn_lifecycle_contract(tmp_path)
    process = _SpawnLifecycleProcess(
        wait_error=subprocess.TimeoutExpired(cmd="subset-worker.py", timeout=30)
    )
    monotonic_counter = itertools.count(step=5.0)
    monkeypatch.setattr(recovery, "_disk_preflight", lambda value: {})
    monkeypatch.setattr(recovery, "_reserve_output_growth", lambda *args: {})
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(recovery, "_read_proc_start_ticks", lambda *args, **kwargs: None)
    monkeypatch.setattr(recovery.time, "monotonic", lambda: next(monotonic_counter))
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        recovery.os, "killpg", lambda pid, value: signals.append((pid, value))
    )

    with pytest.raises(subprocess.TimeoutExpired):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state={"worker": None},
            guard=lambda: None,
            poll_seconds=0.1,
        )
    assert signals == []


def test_popen_failure_does_not_signal_an_unowned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, stage, _argv = _spawn_lifecycle_contract(tmp_path)
    monkeypatch.setattr(recovery, "_disk_preflight", lambda value: {})
    monkeypatch.setattr(recovery, "_reserve_output_growth", lambda *args: {})
    monkeypatch.setattr(
        recovery.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("popen failed")),
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        recovery.os, "killpg", lambda pid, value: signals.append((pid, value))
    )

    with pytest.raises(OSError, match="popen failed"):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state={"worker": None},
            guard=lambda: None,
            poll_seconds=0.1,
        )
    assert signals == []


def test_terminal_cannot_bypass_durable_pid_binding_before_barrier_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, stage, _argv = _spawn_lifecycle_contract(tmp_path)
    terminal = Path(str(stage["terminal_path"]))

    def publish_terminal() -> None:
        _write_json(terminal, {"status": "PASS"})

    process = _SpawnLifecycleProcess(poll_result=0, on_wait=publish_terminal)
    monotonic_counter = itertools.count(step=5.0)
    monkeypatch.setattr(recovery, "_disk_preflight", lambda value: {})
    monkeypatch.setattr(recovery, "_reserve_output_growth", lambda *args: {})
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(recovery, "_read_proc_start_ticks", lambda *args, **kwargs: None)
    monkeypatch.setattr(recovery.time, "monotonic", lambda: next(monotonic_counter))
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        recovery.os, "killpg", lambda pid, value: signals.append((pid, value))
    )

    with pytest.raises(recovery.RecoveryControllerError, match="cannot bind worker"):
        recovery._run_or_attach_stage(
            manifest=manifest,
            stage=stage,
            root=tmp_path,
            state={"worker": None, "startup_barrier": None},
            guard=lambda: None,
            poll_seconds=0.1,
        )
    assert terminal.is_file()
    assert signals == []


def test_exact_progress_is_monotonic_across_primary_to_expansion(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    _write_json(
        checkpoint,
        {
            "next_offset": 7,
            "progress_ledgers": {
                "adaptive_seed_scan": {
                    "committed_offset": recovery.EXPECTED_ROWS,
                },
                "adaptive_failure_scan": {
                    "committed_offset": recovery.EXPECTED_ROWS,
                },
                "shortcut_anchor_scan": {
                    "committed_offset": recovery.EXPECTED_ROWS,
                },
                "adaptive_component_expansion_scan": {
                    "committed_offset": recovery.DEFAULT_BLOCK_SIZE,
                },
            },
        },
    )
    stage = {
        "progress_checkpoint_path": str(checkpoint),
        "progress_field": recovery.EXACT_MONOTONIC_PROGRESS_FIELD,
    }
    assert recovery._progress_value(stage) == (
        recovery.EXPECTED_ROWS + recovery.DEFAULT_BLOCK_SIZE
    )


def test_exact_coexistence_baseline_survives_worker_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    _write_json(
        checkpoint,
        {
            "progress_ledgers": {
                "shortcut_anchor_scan": {"committed_offset": 11},
            }
        },
    )
    stage = {
        "stage_id": recovery.EXACT_STAGE,
        "progress_checkpoint_path": str(checkpoint),
        "progress_field": recovery.EXACT_MONOTONIC_PROGRESS_FIELD,
        "commands": {
            "fresh_sha256": "a" * 64,
            "resume_sha256": "b" * 64,
        },
    }
    root = tmp_path / "controller"
    root.mkdir()
    manifest = {
        "manifest_sha256": "c" * 64,
        "controller_root": str(root),
        "resources": {
            "budget": {"max_output_bytes": 1024**2, "safety_floor_bytes": 0}
        },
    }
    state = recovery._initial_state(manifest)
    monkeypatch.setattr(
        recovery,
        "_host_sample",
        lambda **kwargs: {
            "load_per_cpu": 0.1,
            "cpu_total_ticks": 100,
            "cpu_iowait_ticks": 10,
        },
    )
    first = recovery._ensure_exact_coexistence_baseline(
        manifest=manifest,
        stage=stage,
        state=state,
        worker_argv_sha256="a" * 64,
        root=root,
        guard=lambda: None,
        proc_root=Path("/proc"),
    )
    _write_json(
        checkpoint,
        {
            "progress_ledgers": {
                "shortcut_anchor_scan": {
                    "committed_offset": recovery.EXPECTED_ROWS,
                },
                "adaptive_component_expansion_scan": {
                    "committed_offset": recovery.EXPECTED_ROWS - 1,
                },
            }
        },
    )
    second = recovery._ensure_exact_coexistence_baseline(
        manifest=manifest,
        stage=stage,
        state=state,
        worker_argv_sha256="b" * 64,
        root=root,
        guard=lambda: None,
        proc_root=Path("/proc"),
    )
    assert second == first
    assert second["start_progress"] == 11
    assert second["worker_argv_sha256"] == "a" * 64
    assert recovery._load_state(root, manifest)["exact_coexistence_baseline"] == first


def test_fast_exact_terminal_closes_coexistence_from_persisted_peak_rss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = tmp_path / "exact"
    checkpoint = exact / "dbscan/checkpoint.json"
    _write_json(
        checkpoint,
        {
            "progress_ledgers": {
                "shortcut_anchor_scan": {
                    "committed_offset": recovery.DEFAULT_BLOCK_SIZE - 1,
                }
            }
        },
    )
    dbscan = exact / "dbscan/run_manifest.json"
    _write_json(dbscan, {"peak_rss_bytes_observed": 123456})
    terminal = exact / "exact_recovery_receipt.json"
    _write_json(
        terminal,
        {
            "dbscan_manifest_path": str(dbscan),
            "dbscan_manifest_sha256": recovery.sha256_file(dbscan),
        },
    )
    argv_sha = "a" * 64
    stage = {
        "stage_id": recovery.EXACT_STAGE,
        "terminal_path": str(terminal),
        "progress_checkpoint_path": str(checkpoint),
        "progress_field": recovery.EXACT_MONOTONIC_PROGRESS_FIELD,
        "commands": {
            "fresh_sha256": argv_sha,
            "resume_sha256": "b" * 64,
        },
    }
    contract = {
        "min_progress_rows": recovery.DEFAULT_BLOCK_SIZE,
        "max_load_per_cpu": 0.8,
        "max_iowait_fraction": 0.35,
        "timeout_seconds": 1800,
    }
    manifest = {
        "manifest_sha256": "c" * 64,
        "controller_root": str(tmp_path),
        "stages": [stage],
        "resources": {
            "thread_count": 12,
            "max_rss_bytes": recovery.DEFAULT_MAX_RSS_BYTES,
            "max_rss_scope": (
                "exact_dbscan_process_with_native_peak_certificate"
            ),
            "proc_root": "/proc",
            "coexistence_probe": contract,
            "budget": {"max_output_bytes": 1024**2, "safety_floor_bytes": 0},
        },
    }
    state = {
        "exact_coexistence_baseline": {
            "stage_id": recovery.EXACT_STAGE,
            "controller_manifest_sha256": "c" * 64,
            "worker_argv_sha256": argv_sha,
            "start_progress": 0,
            "start_host": {
                "load_per_cpu": 0.0,
                "cpu_total_ticks": 100,
                "cpu_iowait_ticks": 10,
            },
        }
    }
    monkeypatch.setattr(
        recovery,
        "_host_sample",
        lambda **kwargs: {
            "load_per_cpu": 0.1,
            "cpu_total_ticks": 200,
            "cpu_iowait_ticks": 12,
        },
    )
    path = tmp_path / "coexistence_probe.json"
    result = recovery._publish_terminal_exact_coexistence_probe(
        manifest=manifest,
        stage=stage,
        state=state,
        path=path,
    )
    assert result["status"] == "PASS"
    assert result["end_progress"] == recovery.DEFAULT_BLOCK_SIZE - 1
    assert result["worker_rss_bytes"] == 123456
    assert result["terminal_fast_completion_reconciled"] is True
    assert result["terminal_completed_before_min_progress"] is True


@pytest.mark.parametrize(
    ("monitored", "current", "age", "checkpoint_age", "expected"),
    (
        (recovery.DEFAULT_BLOCK_SIZE, recovery.DEFAULT_BLOCK_SIZE, 5.0, 5.0, "RUNNING_PROGRESSING"),
        (recovery.DEFAULT_BLOCK_SIZE, recovery.DEFAULT_BLOCK_SIZE, 1801.0, 5.0, "RUNNING_STALLED"),
        (recovery.DEFAULT_BLOCK_SIZE, 2 * recovery.DEFAULT_BLOCK_SIZE, 1801.0, 5.0, "RUNNING_PROGRESSING"),
        (recovery.DEFAULT_BLOCK_SIZE, 2 * recovery.DEFAULT_BLOCK_SIZE, 1801.0, 1801.0, "RUNNING_STALLED"),
    ),
)
def test_status_uses_progress_freshness_not_historical_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    monitored: int,
    current: int,
    age: float,
    checkpoint_age: float,
    expected: str,
) -> None:
    manifest_path, manifest = _built_manifest(tmp_path)
    root = Path(manifest["controller_root"])
    root.mkdir()
    exact_stage = next(
        stage
        for stage in manifest["stages"]
        if stage["stage_id"] == recovery.EXACT_STAGE
    )
    checkpoint = Path(exact_stage["progress_checkpoint_path"])
    _write_json(
        checkpoint,
        {
            "progress_ledgers": {
                "shortcut_anchor_scan": {"committed_offset": current},
            }
        },
    )
    manifest_sha = recovery.sha256_file(manifest_path)
    state = recovery._initial_state({"manifest_sha256": manifest_sha})
    state["status"] = "RUNNING"
    state["current_stage"] = recovery.EXACT_STAGE
    state["stages"][recovery.EXACT_STAGE] = "RUNNING"
    state["worker"] = {
        "stage_id": recovery.EXACT_STAGE,
        "pid": 4242,
        "start_ticks": 99,
        "argv_sha256": exact_stage["commands"]["fresh_sha256"],
        "start_progress": 0,
        "elapsed_seconds": 1.0,
    }
    now = 10_000.0
    state["exact_progress_monitor"] = {
        "schema_version": recovery.EXACT_PROGRESS_MONITOR_SCHEMA,
        "controller_manifest_sha256": manifest_sha,
        "stage_id": recovery.EXACT_STAGE,
        "progress": monitored,
        "last_change_epoch": now - age,
        "continuous_progress_since_epoch": None,
        "continuous_start_progress": None,
        "baseline_progress": 0,
        "observed_epoch": now - min(age, 1.0),
        "observed_at": "fixture",
    }
    state["exact_progress_monitor"]["monitor_sha256"] = (
        recovery.stable_json_sha256(state["exact_progress_monitor"])
    )
    _write_json(root / "state.json", state)
    os.utime(checkpoint, (now - checkpoint_age, now - checkpoint_age))
    monkeypatch.setattr(recovery, "_pid_alive", lambda *args, **kwargs: True)
    monkeypatch.setattr(recovery.time, "time", lambda: now)

    status = recovery.controller_status(manifest_path)
    assert status["scientific_progress_state"] == expected
    assert status["route_viability"] == expected
    assert status["scientific_progress_age_seconds"] == age


def test_unset_release_pins_and_authorization_refuse_launch(tmp_path: Path) -> None:
    path, manifest = _built_manifest(tmp_path)
    assert manifest["release_ready"] is False
    with pytest.raises(recovery.RecoveryControllerError, match="RELEASE_PINS_UNSET"):
        recovery.run_controller(path, resume=False)
    assert not Path(manifest["controller_root"]).exists()


def test_ready_pins_still_require_explicit_deployment_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "_execution_tree_clean", lambda root: True)
    path, manifest = _built_manifest(tmp_path, pins_ready=True)
    assert manifest["release_ready"] is True
    with pytest.raises(
        recovery.RecoveryControllerError, match="PRODUCTION_DEPLOYMENT_NOT_AUTHORIZED"
    ):
        recovery.run_controller(path, resume=False)
    assert not Path(manifest["controller_root"]).exists()


def test_adoption_output_must_be_unique_direct_child(tmp_path: Path) -> None:
    spec_path, value = _spec(tmp_path)
    value["stages"][0]["output_dir"] = str(
        tmp_path / "adoption-authority/nested/cid-adoption"
    )
    value["stages"][0]["terminal_path"] = str(
        tmp_path / "adoption-authority/nested/cid-adoption/recovery_receipt.json"
    )
    old_output = value["stages"][0]["argv_bindings"]["output"]["value"]
    new_output = value["stages"][0]["output_dir"]
    value["stages"][0]["argv_bindings"]["output"]["value"] = new_output
    value["stages"][0]["commands"]["fresh"] = [
        new_output if token == old_output else token
        for token in value["stages"][0]["commands"]["fresh"]
    ]
    _write_json(spec_path, value)
    with pytest.raises(recovery.RecoveryControllerError, match="direct authority-parent child"):
        recovery.build_controller_payload(spec_path)


def test_typed_adoption_uses_canonical_projection_and_is_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_projections = {
        "close": recovery.stable_json_sha256({"state": "PASS"}),
        "final": recovery.stable_json_sha256({"state": "FAILED"}),
    }
    monkeypatch.setattr(
        recovery,
        "EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256",
        fixture_projections,
    )
    _path, manifest = _built_manifest(tmp_path)
    stage = manifest["stages"][0]
    output = Path(stage["output_dir"])
    output.mkdir()
    close_projection = {"state": "PASS"}
    final_projection = {"state": "FAILED"}
    projection_shas = {
        "close": recovery.stable_json_sha256(close_projection),
        "final": recovery.stable_json_sha256(final_projection),
    }
    observations = {
        name: [
            {
                "observed_sha256": character * 64,
                "projection_sha256": projection_shas[name],
                "projection": projection,
            }
            for _ in range(2)
        ]
        for name, projection, character in (
            ("close", close_projection, "c"),
            ("final", final_projection, "f"),
        )
    }
    receipt = {
        "schema_version": "aids_comrecgc_c766_failed_selection_adoption_v2",
        "status": "RECOVERY_ONLY_READY",
        "artifact_kind": "aids_c766_failed_selection_recovery_evidence_v2",
        "terminal_marker": "RECOVERY_EVIDENCE_READY",
        "failed_evidence_adopted_for_recovery_only": True,
        "ordinary_pass_dependency_eligible": False,
        "generic_pass_marker_created": False,
        "scientific_result_pass": False,
        "dbscan_partition_pass": False,
        "source_final_status": "FAILED",
        "source_recomputed": False,
        "source_copied": False,
        "large_payload_copied": False,
        "authority_profile_sha256": "a" * 64,
        "task_state_authority": {
            "close": close_projection,
            "close_projection_sha256": projection_shas["close"],
            "final": final_projection,
            "final_projection_sha256": projection_shas["final"],
        },
        "task_state_observations": observations,
        "terminal_reopen_task_state_observations": observations,
        "failed_selection": {"dbscan_partition_proven": False},
        "source_artifacts": _source_artifacts(manifest),
    }
    _write_json(Path(stage["terminal_path"]), receipt)
    (output / "RECOVERY_EVIDENCE_READY").write_text("READY\n", encoding="utf-8")

    def validator(
        *, output_dir: Path
    ) -> dict[str, object]:
        assert output_dir == output
        return receipt

    validated = recovery.validate_typed_adoption_receipt(
        manifest=manifest, validator=validator
    )
    assert validated["receipt_sha256"] == recovery.sha256_file(stage["terminal_path"])
    with pytest.raises(
        recovery.RecoveryControllerError,
        match="RECOVERY_ONLY_EVIDENCE_IS_NOT_ORDINARY_PASS",
    ):
        recovery.validate_ordinary_pass_dependency(stage["terminal_path"])


def test_typed_adoption_fails_if_authority_parent_gains_second_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_projections = {
        "close": recovery.stable_json_sha256({"state": "PASS"}),
        "final": recovery.stable_json_sha256({"state": "FAILED"}),
    }
    monkeypatch.setattr(
        recovery,
        "EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256",
        fixture_projections,
    )
    _path, manifest = _built_manifest(tmp_path)
    stage = manifest["stages"][0]
    output = Path(stage["output_dir"])
    output.mkdir()
    close_projection = {"state": "PASS"}
    final_projection = {"state": "FAILED"}
    projection_shas = manifest["adoption_contract"][
        "expected_task_state_projection_sha256"
    ]
    observations = {
        name: [
            {
                "observed_sha256": character * 64,
                "projection_sha256": projection_shas[name],
                "projection": projection,
            }
            for _ in range(2)
        ]
        for name, projection, character in (
            ("close", close_projection, "c"),
            ("final", final_projection, "f"),
        )
    }
    receipt = {
        "schema_version": "aids_comrecgc_c766_failed_selection_adoption_v2",
        "status": "RECOVERY_ONLY_READY",
        "artifact_kind": "aids_c766_failed_selection_recovery_evidence_v2",
        "terminal_marker": "RECOVERY_EVIDENCE_READY",
        "failed_evidence_adopted_for_recovery_only": True,
        "ordinary_pass_dependency_eligible": False,
        "generic_pass_marker_created": False,
        "scientific_result_pass": False,
        "dbscan_partition_pass": False,
        "source_final_status": "FAILED",
        "source_recomputed": False,
        "source_copied": False,
        "large_payload_copied": False,
        "authority_profile_sha256": "a" * 64,
        "task_state_authority": {
            "close": close_projection,
            "close_projection_sha256": projection_shas["close"],
            "final": final_projection,
            "final_projection_sha256": projection_shas["final"],
        },
        "task_state_observations": observations,
        "terminal_reopen_task_state_observations": observations,
        "failed_selection": {"dbscan_partition_proven": False},
        "source_artifacts": _source_artifacts(manifest),
    }
    _write_json(Path(stage["terminal_path"]), receipt)
    (output / "RECOVERY_EVIDENCE_READY").write_text("READY\n", encoding="utf-8")
    (output.parent / "second-child").mkdir()

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        return receipt

    with pytest.raises(recovery.RecoveryControllerError, match="unique-child"):
        recovery.validate_typed_adoption_receipt(
            manifest=manifest, validator=validator
        )


def test_manifest_publication_is_no_clobber(tmp_path: Path) -> None:
    spec_path, _ = _spec(tmp_path)
    output = tmp_path / "controller-manifest.json"
    recovery.build_controller_manifest(spec_path=spec_path, output_path=output)
    original = output.read_bytes()
    with pytest.raises(recovery.RecoveryControllerError, match="already exists"):
        recovery.build_controller_manifest(spec_path=spec_path, output_path=output)
    assert output.read_bytes() == original
    assert not (tmp_path / ".controller-manifest.json.publish.tmp").exists()


def test_mutable_state_fixed_temp_recovers_pre_replace_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state.json"
    real_replace = recovery.os.replace

    def crash_before_replace(source: object, target: object) -> None:
        if Path(str(target)) == state:
            raise RuntimeError("crash-before-state-replace")
        real_replace(source, target)

    monkeypatch.setattr(recovery.os, "replace", crash_before_replace)
    with pytest.raises(RuntimeError, match="crash-before-state-replace"):
        recovery._atomic_state(state, {"generation": 1})
    temporary = tmp_path / ".state.json.replace.tmp"
    assert temporary.is_file()
    assert not state.exists()
    monkeypatch.setattr(recovery.os, "replace", real_replace)
    recovery._atomic_state(state, {"generation": 2})
    assert json.loads(state.read_text(encoding="utf-8")) == {"generation": 2}
    assert not temporary.exists()


def test_read_only_immutable_inspector_rejects_permission_drift(
    tmp_path: Path,
) -> None:
    terminal = tmp_path / "terminal.json"
    recovery._write_new_bytes(terminal, b"{}\n")
    assert recovery._inspect_immutable_publication(terminal) is False
    terminal.chmod(0o644)
    with pytest.raises(recovery.RecoveryControllerError, match="identity changed"):
        recovery._inspect_immutable_publication(terminal)


def test_controller_root_has_one_writer(tmp_path: Path) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = dict(manifest)
    runtime["manifest_sha256"] = "c" * 64
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def claim() -> None:
        barrier.wait()
        try:
            recovery._claim_controller_root(runtime, resume=False)
        except recovery.RecoveryControllerError:
            outcomes.append("blocked")
        else:
            outcomes.append("claimed")

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["blocked", "claimed"]


@pytest.mark.parametrize("failure_point", ("root", "gates", "logs", "owner"))
def test_controller_root_publication_crash_resumes_same_cid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = dict(manifest)
    runtime["manifest_sha256"] = "c" * 64
    original = recovery._write_new_json
    original_mkdir = Path.mkdir
    injected = False

    def fail_owner(path: Path, payload: dict[str, object]) -> None:
        nonlocal injected
        if failure_point == "owner" and path.name == "owner_claim.json" and not injected:
            injected = True
            raise RuntimeError("root publication crash")
        original(path, payload)

    root = Path(runtime["controller_root"])

    def fail_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal injected
        targets = {
            "root": root,
            "gates": root / "gates",
            "logs": root / "logs",
        }
        if failure_point in targets and path == targets[failure_point] and not injected:
            injected = True
            raise RuntimeError("root publication crash")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(recovery, "_write_new_json", fail_owner)
    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    with pytest.raises(RuntimeError, match="root publication crash"):
        recovery._claim_controller_root(runtime, resume=False)
    recovered = recovery._claim_controller_root(runtime, resume=True)
    assert recovered == root
    assert (root / "gates").is_dir()
    assert (root / "logs").is_dir()
    owner = json.loads((root / "owner_claim.json").read_text(encoding="utf-8"))
    assert owner["schema_version"] == recovery.OWNER_SCHEMA
    assert owner["root_preclaim"]["path"] == str(recovery._root_claim_path(runtime))
    with pytest.raises(recovery.RecoveryControllerError, match="fresh controller root"):
        recovery._claim_controller_root(runtime, resume=False)


@pytest.mark.parametrize("window", ("temp_only", "final_and_temp"))
def test_owner_claim_immutable_publication_windows_resume_same_cid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str
) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = {**manifest, "manifest_sha256": "c" * 64}
    root = Path(runtime["controller_root"])
    owner = root / "owner_claim.json"
    temporary = root / ".owner_claim.json.publish.tmp"
    if window == "temp_only":
        real_link = recovery.os.link

        def crash_before_link(source: object, target: object, **kwargs: object) -> None:
            if Path(str(target)) == owner:
                raise RuntimeError("crash-before-owner-link")
            real_link(source, target, **kwargs)

        monkeypatch.setattr(recovery.os, "link", crash_before_link)
        with pytest.raises(RuntimeError, match="crash-before-owner-link"):
            recovery._claim_controller_root(runtime, resume=False)
        assert temporary.is_file()
        assert not owner.exists()
        monkeypatch.setattr(recovery.os, "link", real_link)
    else:
        recovery._claim_controller_root(runtime, resume=False)
        os.link(owner, temporary, follow_symlinks=False)
        assert owner.stat().st_nlink == 2
    assert recovery._claim_controller_root(runtime, resume=True) == root
    assert owner.is_file()
    assert owner.stat().st_nlink == 1
    assert not temporary.exists()


def test_controller_root_preclaim_replacement_or_conflict_is_rejected(
    tmp_path: Path,
) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = {**manifest, "manifest_sha256": "c" * 64}
    root = recovery._claim_controller_root(runtime, resume=False)
    claim = recovery._root_claim_path(runtime)
    claim.unlink()
    claim.write_bytes(b"")
    claim.chmod(0o600)
    with pytest.raises(recovery.RecoveryControllerError, match="preclaim identity"):
        recovery._claim_controller_root(runtime, resume=True)
    conflicting = {**runtime, "manifest_sha256": "d" * 64}
    with pytest.raises(recovery.RecoveryControllerError, match="conflicting root claim"):
        recovery._claim_controller_root(conflicting, resume=True)
    assert root.is_dir()


def test_root_preclaim_receipt_binds_nonce_content_and_stat(tmp_path: Path) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = {**manifest, "manifest_sha256": "c" * 64}
    root = recovery._claim_controller_root(runtime, resume=False)
    claim = recovery._root_claim_path(runtime)
    owner = json.loads((root / "owner_claim.json").read_text(encoding="utf-8"))
    binding = owner["root_preclaim"]
    payload_bytes = claim.read_bytes()
    payload = json.loads(payload_bytes)
    observed = claim.stat()

    assert payload["schema_version"] == recovery.ROOT_CLAIM_SCHEMA
    assert payload["controller_id"] == recovery.CONTROLLER_ID
    assert payload["controller_manifest_sha256"] == runtime["manifest_sha256"]
    assert payload["controller_root"] == runtime["controller_root"]
    assert len(payload["claim_attempt_id"]) == 32
    assert len(payload["claim_nonce"]) == 64
    assert binding["size"] == len(payload_bytes) == observed.st_size
    assert binding["mtime_ns"] == observed.st_mtime_ns
    assert binding["ctime_ns"] == observed.st_ctime_ns
    assert binding["nlink"] == observed.st_nlink == 1
    assert binding["content_sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    assert binding["claim_attempt_id"] == payload["claim_attempt_id"]
    assert binding["claim_nonce_sha256"] == hashlib.sha256(
        payload["claim_nonce"].encode("ascii")
    ).hexdigest()
    assert recovery._claim_controller_root(runtime, resume=True) == root


def test_root_preclaim_copy_content_replacement_fails_closed(tmp_path: Path) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = {**manifest, "manifest_sha256": "c" * 64}
    recovery._claim_controller_root(runtime, resume=False)
    claim = recovery._root_claim_path(runtime)
    payload = claim.read_bytes()
    claim.unlink()
    claim.write_bytes(payload)
    claim.chmod(0o600)

    with pytest.raises(recovery.RecoveryControllerError, match="owner claim mismatch"):
        recovery._claim_controller_root(runtime, resume=True)


def test_root_preclaim_same_inode_aba_fails_closed(tmp_path: Path) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = {**manifest, "manifest_sha256": "c" * 64}
    recovery._claim_controller_root(runtime, resume=False)
    claim = recovery._root_claim_path(runtime)
    original = claim.stat()
    backup = claim.parent / f".{claim.name}.aba-backup"

    os.link(claim, backup, follow_symlinks=False)
    claim.unlink()
    os.link(backup, claim, follow_symlinks=False)
    backup.unlink()
    restored = claim.stat()
    assert (restored.st_dev, restored.st_ino) == (original.st_dev, original.st_ino)
    assert restored.st_nlink == 1

    with pytest.raises(recovery.RecoveryControllerError, match="owner claim mismatch"):
        recovery._claim_controller_root(runtime, resume=True)


def test_root_preclaim_same_inode_content_tamper_fails_closed(tmp_path: Path) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = {**manifest, "manifest_sha256": "c" * 64}
    recovery._claim_controller_root(runtime, resume=False)
    claim = recovery._root_claim_path(runtime)
    original_inode = claim.stat().st_ino
    payload = json.loads(claim.read_text(encoding="utf-8"))
    payload["claim_nonce"] = "0" * 64
    claim.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim.chmod(0o600)
    assert claim.stat().st_ino == original_inode

    with pytest.raises(recovery.RecoveryControllerError, match="owner claim mismatch"):
        recovery._claim_controller_root(runtime, resume=True)


def test_controller_root_preclaim_symlink_or_truncation_is_rejected(
    tmp_path: Path,
) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = {**manifest, "manifest_sha256": "c" * 64}
    recovery._claim_controller_root(runtime, resume=False)
    claim = recovery._root_claim_path(runtime)
    claim.write_bytes(b"tamper")
    with pytest.raises(recovery.RecoveryControllerError, match="preclaim identity"):
        recovery._claim_controller_root(runtime, resume=True)

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_path, other_manifest = _built_manifest(other_root)
    del other_path
    other = {**other_manifest, "manifest_sha256": "e" * 64}
    other_claim = recovery._root_claim_path(other)
    target = tmp_path / "target"
    target.write_bytes(b"")
    other_claim.symlink_to(target)
    with pytest.raises(recovery.RecoveryControllerError, match="not physical"):
        recovery._claim_controller_root(other, resume=True)


def test_concurrent_ownerless_root_resume_has_one_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, manifest = _built_manifest(tmp_path)
    runtime = {**manifest, "manifest_sha256": "c" * 64}
    original = recovery._write_new_json

    def initial_crash(path: Path, payload: dict[str, object]) -> None:
        if path.name == "owner_claim.json":
            raise RuntimeError("initial owner crash")
        original(path, payload)

    monkeypatch.setattr(recovery, "_write_new_json", initial_crash)
    with pytest.raises(RuntimeError, match="initial owner crash"):
        recovery._claim_controller_root(runtime, resume=False)

    owner_writes = 0
    owner_writes_lock = threading.Lock()

    def slow_owner(path: Path, payload: dict[str, object]) -> None:
        nonlocal owner_writes
        if path.name == "owner_claim.json":
            with owner_writes_lock:
                owner_writes += 1
            time.sleep(0.05)
        original(path, payload)

    monkeypatch.setattr(recovery, "_write_new_json", slow_owner)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def resume() -> None:
        barrier.wait()
        try:
            recovery._claim_controller_root(runtime, resume=True)
        except recovery.RecoveryControllerError:
            outcomes.append("blocked")
        else:
            outcomes.append("claimed")

    threads = [threading.Thread(target=resume) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert owner_writes == 1
    assert sorted(outcomes) == ["blocked", "claimed"]
    assert (Path(runtime["controller_root"]) / "owner_claim.json").is_file()


def test_prelaunch_claim_is_cid_local_no_clobber_and_forces_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "_execution_tree_clean", lambda root: True)
    monkeypatch.setattr(
        recovery.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=100 * 1024**3),
    )
    path, manifest = _built_manifest(
        tmp_path, pins_ready=True, deployment_authorized=True
    )
    prepared = recovery.prepare_controller_launch(path, resume=False)
    root = Path(manifest["controller_root"])
    assert prepared["controller_root"] == str(root)
    assert prepared["controller_invocation_requires_resume"] is True
    assert prepared["thread_count"] == 8
    assert Path(prepared["log_path"]).parent == root / "logs"
    assert Path(prepared["pid_path"]).parent == root / "logs"
    assert Path(prepared["prelaunch_receipt_path"]).parent == root / "logs"
    assert Path(prepared["prelaunch_receipt_path"]).is_file()
    with pytest.raises(recovery.RecoveryControllerError, match="fresh controller root"):
        recovery.prepare_controller_launch(path, resume=False)
    resumed = recovery.prepare_controller_launch(path, resume=True)
    assert resumed["requested_mode"] == "resume"
    assert resumed["launch_id"] != prepared["launch_id"]


def test_controller_lock_detects_concurrent_writer_and_replacement(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "gates").mkdir()
    (root / "logs").mkdir()
    with pytest.raises(recovery.RecoveryControllerError, match="identity changed"):
        with recovery._controller_lock(root) as held:
            with pytest.raises(recovery.RecoveryControllerError, match="another recovery"):
                with recovery._controller_lock(root):
                    pass
            held.path.unlink()
            held.path.write_bytes(b"replacement")
            held.verify()


def test_stage_closure_rejects_extra_files_but_allows_owned_future_subtree(
    tmp_path: Path,
) -> None:
    _path, manifest = _built_manifest(tmp_path)
    subset = next(
        row for row in manifest["stages"] if row["stage_id"] == recovery.SUBSET_STAGE
    )
    subset_terminal = Path(subset["terminal_path"])
    _write_json(subset_terminal, {"status": "PASS"})
    (Path(subset["output_dir"]) / "attempt-0").mkdir()
    (Path(subset["output_dir"]) / "attempt-0/.interrupted.tmp").write_bytes(
        b"diagnostic partial\n"
    )
    subset_inventory = recovery._build_stage_closure_inventory(
        manifest, recovery.SUBSET_STAGE
    )
    recovery._validate_stage_closure_inventory(
        manifest, recovery.SUBSET_STAGE, subset_inventory
    )
    _write_json(Path(subset["output_dir"]) / "unexpected.json", {"extra": True})
    with pytest.raises(recovery.RecoveryControllerError, match="file set changed"):
        recovery._validate_stage_closure_inventory(
            manifest, recovery.SUBSET_STAGE, subset_inventory
        )

    exact = next(
        row for row in manifest["stages"] if row["stage_id"] == recovery.EXACT_STAGE
    )
    downstream = next(
        row
        for row in manifest["stages"]
        if row["stage_id"] == recovery.DOWNSTREAM_STAGE
    )
    exact_terminal = Path(exact["terminal_path"])
    _write_json(exact_terminal, {"status": "PASS"})
    exact_inventory = recovery._build_stage_closure_inventory(
        manifest, recovery.EXACT_STAGE
    )
    _write_json(Path(downstream["output_dir"]) / "run_manifest.json", {"status": "PASS"})
    recovery._validate_stage_closure_inventory(
        manifest, recovery.EXACT_STAGE, exact_inventory
    )
    _write_json(Path(exact["output_dir"]) / "unowned-extra.json", {"extra": True})
    with pytest.raises(recovery.RecoveryControllerError, match="file set changed"):
        recovery._validate_stage_closure_inventory(
            manifest, recovery.EXACT_STAGE, exact_inventory
        )


def test_controller_stage_gates_are_typed_restartable_and_final_is_ordinary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "_execution_tree_clean", lambda root: True)
    monkeypatch.setattr(
        recovery.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=100 * 1024**3),
    )
    path, manifest = _built_manifest(
        tmp_path, pins_ready=True, deployment_authorized=True
    )
    observed: list[str] = []

    def fake_run(
        *,
        manifest: dict[str, object],
        stage: dict[str, object],
        root: Path,
        state: dict[str, object],
        guard: object,
        poll_seconds: float,
    ) -> None:
        del root, state, poll_seconds
        guard()
        stage_id = str(stage["stage_id"])
        observed.append(stage_id)
        stage_terminal = Path(str(stage["terminal_path"]))
        if stage_id == recovery.SUBSET_STAGE:
            stage_terminal.parent.mkdir(parents=True, exist_ok=True)
            recovery._write_new_bytes(
                stage_terminal,
                (json.dumps({"stage_id": stage_id}) + "\n").encode("utf-8"),
            )
            os.link(
                stage_terminal,
                stage_terminal.parent / f".{stage_terminal.name}.publish.tmp",
                follow_symlinks=False,
            )
        else:
            _write_json(stage_terminal, {"stage_id": stage_id})
        if stage_id == recovery.EXACT_STAGE:
            _write_json(
                Path(str(stage["output_dir"])) / "dbscan/nested-certificate.json",
                {"status": "PASS"},
            )
            contract = manifest["resources"]["coexistence_probe"]
            recovery._write_new_json(
                Path(manifest["controller_root"]) / "coexistence_probe.json",
                {
                    "schema_version": recovery.COEXISTENCE_SCHEMA,
                    "status": "PASS",
                    "controller_id": recovery.CONTROLLER_ID,
                    "controller_manifest_sha256": manifest["manifest_sha256"],
                    "worker_argv_sha256": stage["commands"]["fresh_sha256"],
                    "thread_count": manifest["resources"]["thread_count"],
                    "cuda_visible_devices": "",
                    "gpu_lock_acquired": False,
                    "start_progress": 0,
                    "end_progress": contract["min_progress_rows"],
                    "start_host": {
                        "load_per_cpu": 0.0,
                        "cpu_total_ticks": 100,
                        "cpu_iowait_ticks": 10,
                    },
                    "end_host": {
                        "load_per_cpu": 0.0,
                        "cpu_total_ticks": 200,
                        "cpu_iowait_ticks": 12,
                    },
                    "iowait_fraction": 0.0,
                    "worker_rss_bytes": 0,
                    "contract": contract,
                },
            )

    def fake_validate(
        manifest: dict[str, object],
        *,
        stage_id: str,
        adoption_validator: object = None,
    ) -> dict[str, object]:
        del adoption_validator
        stage = next(row for row in manifest["stages"] if row["stage_id"] == stage_id)
        terminal = Path(stage["terminal_path"])
        return {
            "path": str(terminal.resolve(strict=True)),
            "sha256": recovery.sha256_file(terminal),
            "stage_id": stage_id,
        }

    monkeypatch.setattr(recovery, "_run_or_attach_stage", fake_run)
    monkeypatch.setattr(recovery, "validate_stage_terminal", fake_validate)
    terminal = recovery.run_controller(path, resume=False, poll_seconds=0.1)
    assert observed == list(recovery.STAGE_ORDER)
    assert terminal["status"] == "PASS"
    subset_terminal = Path(
        next(
            row["terminal_path"]
            for row in manifest["stages"]
            if row["stage_id"] == recovery.SUBSET_STAGE
        )
    )
    assert subset_terminal.stat().st_nlink == 1
    assert not (
        subset_terminal.parent / f".{subset_terminal.name}.publish.tmp"
    ).exists()
    runtime_manifest = dict(manifest)
    runtime_manifest["manifest_path"] = str(path)
    runtime_manifest["manifest_sha256"] = recovery.sha256_file(path)
    for stage_id in recovery.STAGE_ORDER:
        gate = recovery._open_gate(runtime_manifest, stage_id)
        assert gate["ordinary_pass_dependency_eligible"] is False
        expected = [
            recovery._open_gate(runtime_manifest, dep)["gate_sha256"]
            for dep in recovery.DEPENDENCIES[stage_id]
        ]
        assert gate["dependency_gate_sha256"] == expected
    root = Path(manifest["controller_root"])
    crash_window_paths = [
        recovery._gate_path(runtime_manifest, recovery.SUBSET_STAGE),
        root / "terminal.json",
        root / "PASS",
    ]
    crash_window_temps = []
    for immutable in crash_window_paths:
        temporary = immutable.parent / f".{immutable.name}.publish.tmp"
        os.link(immutable, temporary, follow_symlinks=False)
        crash_window_temps.append(temporary)
    def filesystem_snapshot() -> dict[str, tuple[int, int, int, int, int]]:
        return {
            str(entry.relative_to(root)): (
                int(entry.lstat().st_dev),
                int(entry.lstat().st_ino),
                int(entry.lstat().st_nlink),
                int(entry.lstat().st_size),
                int(entry.lstat().st_mtime_ns),
            )
            for entry in root.rglob("*")
        }

    before_status = filesystem_snapshot()
    pending_status = recovery.controller_status(path)
    assert pending_status["status"] == "FILESYSTEM_RECONCILIATION_REQUIRED"
    assert pending_status["route_viability"] == "BLOCKED_RECOVERABLE"
    assert (
        pending_status["stages"][recovery.SUBSET_STAGE]
        == "RECONCILIATION_REQUIRED"
    )
    assert filesystem_snapshot() == before_status
    # A terminal restart reopens immutable typed gates and launches no worker.
    observed.clear()
    reopened = recovery.run_controller(path, resume=True, poll_seconds=0.1)
    assert reopened["status"] == "PASS"
    assert observed == []
    assert all(not temporary.exists() for temporary in crash_window_temps)
    assert all(immutable.stat().st_nlink == 1 for immutable in crash_window_paths)
    status = recovery.controller_status(path)
    assert status["status"] == "PASS"
    assert set(status["stages"].values()) == {"PASS"}
    final_path = Path(manifest["controller_root"]) / "terminal.json"
    pass_path = Path(manifest["controller_root"]) / "PASS"
    pass_path.unlink()
    with pytest.raises(recovery.RecoveryControllerError, match="PASS-last"):
        recovery.validate_ordinary_pass_dependency(final_path)
    pending = recovery.controller_status(path)
    assert pending["status"] == "TERMINAL_RECONCILIATION_REQUIRED"
    assert pending["scientific_progress_state"] == "BLOCKED_RECOVERABLE"
    assert pending["route_viability"] == "BLOCKED_RECOVERABLE"
    reconciled = recovery.run_controller(path, resume=True, poll_seconds=0.1)
    assert reconciled["status"] == "PASS"
    assert pass_path.read_bytes() == b"PASS\n"
    ordinary = recovery.validate_ordinary_pass_dependency(final_path)
    assert ordinary["ordinary_pass_dependency_eligible"] is True
    nested = (
        Path(next(
            row["output_dir"]
            for row in manifest["stages"]
            if row["stage_id"] == recovery.EXACT_STAGE
        ))
        / "dbscan/nested-certificate.json"
    )
    _write_json(nested, {"status": "TAMPERED"})
    with pytest.raises(recovery.RecoveryControllerError, match="closure stat changed"):
        recovery.validate_controller_terminal(runtime_manifest)


def test_cpu_environment_cannot_see_gpu_and_freezes_libraries(tmp_path: Path) -> None:
    _path, manifest = _built_manifest(tmp_path)
    env = recovery._stage_environment(manifest)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["DEVICE"] == "cpu"
    assert env["GPU_REQUIRED"] == "0"
    assert {env[name] for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )} == {"8"}


def test_source_authority_rejects_failed_gate_relabelled_pass(tmp_path: Path) -> None:
    spec_path, value = _spec(tmp_path)
    value["source_authority"]["failed_final_gate_status"] = "PASS"
    _write_json(spec_path, value)
    with pytest.raises(recovery.RecoveryControllerError, match="rerun/copy or bless"):
        recovery.build_controller_payload(spec_path)


def test_resume_smoke_requires_a_second_controller_and_live_exact_reattachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "controller"
    (root / "gates").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / ".controller.lock").write_bytes(b"")
    checkpoint_path = tmp_path / "exact/dbscan/checkpoint.json"
    exact_stage = {
        "stage_id": recovery.EXACT_STAGE,
        "progress_checkpoint_path": str(checkpoint_path),
    }
    vector_sha = "a" * 64
    manifest = {
        "manifest_sha256": "b" * 64,
        "controller_root": str(root),
        "resources": {"proc_root": "/proc"},
        "source_authority": {"source_vectors_sha256": vector_sha},
        "stages": [exact_stage],
    }
    held = SimpleNamespace(identity=recovery._current_lock_identity(root))
    state = {
        "controller_process": None,
        "worker": {
            "stage_id": recovery.EXACT_STAGE,
            "pid": 300,
            "start_ticks": 33,
            "argv_sha256": "c" * 64,
        },
        "resume_smoke": None,
    }
    current = {"pid": 200, "start_ticks": 22, "argv_sha256": "d" * 64}
    assert recovery._record_resume_smoke_if_reattached(
        manifest=manifest,
        root=root,
        state=state,
        current_controller=current,
        held=held,
    ) is None
    assert state["resume_smoke"] is None

    state["controller_process"] = {
        "pid": 100,
        "start_ticks": 11,
        "argv_sha256": "e" * 64,
    }
    monkeypatch.setattr(recovery, "_pid_alive", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        recovery,
        "_validated_bound_worker_pid_for_signal",
        lambda **kwargs: 300,
    )
    monkeypatch.setattr(
        recovery,
        "_validated_exact_checkpoint_snapshot",
        lambda stage: {
            "path": str(checkpoint_path),
            "sha256_at_observation": "f" * 64,
            "checkpoint_payload_sha256": "0" * 64,
            "identity_sha256": "1" * 64,
            "progress_ledgers_sha256": "2" * 64,
            "progress_rows": recovery.DEFAULT_BLOCK_SIZE,
            "vectors_sha256": vector_sha,
        },
    )
    receipt = recovery._record_resume_smoke_if_reattached(
        manifest=manifest,
        root=root,
        state=state,
        current_controller=current,
        held=held,
    )
    assert receipt is not None
    assert receipt["status"] == "PASS"
    assert receipt["science_worker_reattached"] is True
    assert receipt["signals_sent"] == []
    assert receipt["previous_controller"]["pid"] == 100
    assert receipt["resumed_controller"]["pid"] == 200


def test_handover_checkpoint_reopens_authenticated_hash_chain(tmp_path: Path) -> None:
    from src.baselines.comrecgc import external_memory_dbscan as dbscan

    path = tmp_path / "checkpoint.json"
    identity = {"vectors_sha256": "a" * 64}
    ledger = dbscan._new_progress_ledger(
        phase="shortcut_anchor_scan", identity=identity
    )
    dbscan._append_progress_entry(
        ledger,
        start=0,
        stop=7,
        payload={"fixture": True},
    )
    ledgers = {"shortcut_anchor_scan": ledger}
    dbscan._checkpoint(
        path,
        identity=identity,
        phase="shortcut_anchor_scan",
        next_offset=7,
        peak_rss_bytes=1,
        extra=dbscan._progress_checkpoint_extra(ledgers, identity=identity),
    )
    snapshot = recovery._validated_exact_checkpoint_snapshot(
        {"progress_checkpoint_path": str(path)}
    )
    assert snapshot["progress_rows"] == 7
    assert snapshot["vectors_sha256"] == "a" * 64
    assert snapshot["sha256_at_observation"] == recovery.sha256_file(path)
    assert len(snapshot["checkpoint_payload_sha256"]) == 64

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["peak_rss_bytes"] = 2
    _write_json(path, tampered)
    with pytest.raises(
        recovery.RecoveryControllerError,
        match="checkpoint authentication changed",
    ):
        recovery._validated_exact_checkpoint_snapshot(
            {"progress_checkpoint_path": str(path)}
        )


def test_handover_contract_binds_exact_old_pid_generation(tmp_path: Path) -> None:
    spec_path, value = _spec(tmp_path)
    contract = value["resources"]["old_brute_handover"]
    assert contract["old_brute_process"] == {
        "pid": 273939,
        "start_ticks": 687141119,
        "cmdline_sha256": recovery.OLD_BRUTE_CMDLINE_SHA256,
    }
    contract["old_brute_process"]["start_ticks"] += 1
    _write_json(spec_path, value)
    with pytest.raises(
        recovery.RecoveryControllerError,
        match="old-brute handover contract changed",
    ):
        recovery.build_controller_payload(spec_path)


def test_old_brute_live_check_reopens_only_the_frozen_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations: list[tuple[int, int, Path]] = []

    def cmdline_sha(pid: int, ticks: int, *, proc_root: Path) -> str:
        observations.append((pid, ticks, proc_root))
        return recovery.OLD_BRUTE_CMDLINE_SHA256

    monkeypatch.setattr(recovery, "_proc_cmdline_sha256", cmdline_sha)
    manifest = {
        "resources": {
            "proc_root": "/proc",
            "old_brute_handover": recovery.handover_contract(),
        }
    }
    assert recovery._old_brute_generation_alive(manifest) is True
    assert observations == [
        (273939, 687141119, Path("/proc")),
    ]
    monkeypatch.setattr(
        recovery,
        "_proc_cmdline_sha256",
        lambda *args, **kwargs: "0" * 64,
    )
    assert recovery._old_brute_generation_alive(manifest) is False


def test_aids_old_brute_handover_requires_typed_evidence_and_sends_no_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 100_000.0
    source = _source_authority(tmp_path)
    vector_sha = str(source["source_vectors_sha256"])
    checkpoint_path = tmp_path / "checkpoint.json"
    exact_stage = {
        "stage_id": recovery.EXACT_STAGE,
        "progress_checkpoint_path": str(checkpoint_path),
    }
    manifest = {
        "manifest_sha256": "f" * 64,
        "project_root": str(Path(__file__).resolve().parents[2]),
        "controller_root": str(tmp_path / "controller"),
        "execution_commit": "a" * 40,
        "release_pins": {"controller_commit": "b" * 40},
        "release_ready": True,
        "production_deployment_authorized": True,
        "source_authority": source,
        "resources": {
            "proc_root": "/proc",
            "old_brute_handover": recovery.handover_contract(),
        },
        "stages": [exact_stage],
    }
    result = {
        "status": "RUNNING",
        "route_viability": "RUNNING_PROGRESSING",
        "current_stage": recovery.EXACT_STAGE,
        "scientific_worker_alive": True,
        "stages": {recovery.ADOPTION_STAGE: "PASS"},
    }
    progress = 12_000_000
    monitor = {
        "schema_version": recovery.EXACT_PROGRESS_MONITOR_SCHEMA,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "stage_id": recovery.EXACT_STAGE,
        "progress": progress,
        "baseline_progress": 0,
        "continuous_start_progress": 0,
        "continuous_progress_since_epoch": now - 601.0,
        "last_change_epoch": now - 1.0,
        "observed_epoch": now - 1.0,
        "observed_at": "fixture",
    }
    monitor["monitor_sha256"] = recovery.stable_json_sha256(monitor)
    resume_checkpoint = {
        "path": str(checkpoint_path),
        "sha256_at_observation": "c" * 64,
        "checkpoint_payload_sha256": "0" * 64,
        "identity_sha256": "d" * 64,
        "progress_ledgers_sha256": "e" * 64,
        "progress_rows": recovery.DEFAULT_BLOCK_SIZE,
        "vectors_sha256": vector_sha,
    }
    state = {
        "resume_smoke": {"checkpoint_snapshot": resume_checkpoint},
        "exact_progress_monitor": monitor,
        "worker": {
            "stage_id": recovery.EXACT_STAGE,
            "pid": 300,
            "start_ticks": 33,
            "argv_sha256": "9" * 64,
        },
    }
    signals: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        recovery.os,
        "killpg",
        lambda *args: signals.append(tuple(args)),
    )
    monkeypatch.setattr(
        recovery.os,
        "kill",
        lambda *args: signals.append(tuple(args)),
    )
    monkeypatch.setattr(recovery.time, "time", lambda: now)
    monkeypatch.setattr(recovery, "_execution_tree_clean", lambda root: True)
    monkeypatch.setattr(
        recovery, "_release_commits_are_ancestors", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        recovery, "_validate_resume_smoke", lambda manifest, value: dict(value)
    )
    monkeypatch.setattr(recovery, "_old_brute_generation_alive", lambda manifest: True)
    monkeypatch.setattr(
        recovery,
        "_validated_bound_worker_pid_for_signal",
        lambda **kwargs: 300,
    )
    monkeypatch.setattr(
        recovery,
        "_validated_exact_checkpoint_snapshot",
        lambda stage: {
            **resume_checkpoint,
            "progress_rows": progress,
        },
    )

    handover = recovery._old_brute_handover_status(
        manifest=manifest, result=result, state=state
    )
    assert handover["status"] == "ELIGIBLE"
    assert handover["eligible_to_request_old_brute_stop"] is True
    assert handover["old_brute_process"] == {
        "pid": 273939,
        "start_ticks": 687141119,
        "cmdline_sha256": recovery.OLD_BRUTE_CMDLINE_SHA256,
    }
    assert handover["conditions"]["resume_smoke_pass"] is True
    assert handover["conditions"]["first_durable_checkpoint_pass"] is True
    assert handover["conditions"]["continuous_progress_10m_pass"] is True
    assert handover["conditions"]["positive_throughput_pass"] is True
    assert handover["conditions"]["eta_within_48h_pass"] is True
    assert handover["conditions"]["relative_speedup_100x_pass"] is False
    assert handover["conditions"]["eta_or_100x_pass"] is True
    assert handover["conditions"]["old_brute_exact_generation_alive_pass"] is True
    assert handover["conditions"]["new_exact_worker_generation_bound_pass"] is True
    assert handover["old_route_signal_authorized_here"] is False
    assert handover["old_route_signal_sent"] is False
    assert signals == []

    tampered_state = json.loads(json.dumps(state))
    tampered_state["exact_progress_monitor"]["progress"] += 1
    tampered = recovery._old_brute_handover_status(
        manifest=manifest, result=result, state=tampered_state
    )
    assert tampered["eligible_to_request_old_brute_stop"] is False
    assert "continuous_progress" in tampered["errors"]
    assert signals == []

    monkeypatch.setattr(
        recovery,
        "_validated_bound_worker_pid_for_signal",
        lambda **kwargs: None,
    )
    unbound_new_worker = recovery._old_brute_handover_status(
        manifest=manifest, result=result, state=state
    )
    assert unbound_new_worker["eligible_to_request_old_brute_stop"] is False
    assert (
        unbound_new_worker["conditions"]["new_exact_worker_generation_bound_pass"]
        is False
    )
    assert signals == []

    monkeypatch.setattr(
        recovery,
        "_validated_bound_worker_pid_for_signal",
        lambda **kwargs: 300,
    )
    monkeypatch.setattr(recovery, "_old_brute_generation_alive", lambda manifest: False)
    refused = recovery._old_brute_handover_status(
        manifest=manifest, result=result, state=state
    )
    assert refused["eligible_to_request_old_brute_stop"] is False
    assert refused["conditions"]["old_brute_exact_generation_alive_pass"] is False
    assert signals == []
