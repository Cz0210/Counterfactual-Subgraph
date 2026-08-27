from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable

import pytest

from src.utils import tastemolnet_gine_pass_adoption_v1 as adoption


def _write_json(path: Path, payload: object, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_snapshot(path: Path) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for entry in [path, *sorted(path.rglob("*"))]:
        info = entry.lstat()
        relative = "." if entry == path else entry.relative_to(path).as_posix()
        if entry.is_dir():
            rows.append(
                (
                    relative,
                    "directory",
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_uid,
                )
            )
        else:
            rows.append(
                (
                    relative,
                    "file",
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_uid,
                    info.st_nlink,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    _sha(entry),
                )
            )
    return rows


def _snapshot(
    pid: int,
    ppid: int,
    argv: list[str],
    *,
    cwd: str = "/frozen/project",
    exe: str = "/python",
) -> dict[str, object]:
    return {
        "pid": pid,
        "linux_start_ticks": pid * 10,
        "ppid": ppid,
        "argv": argv,
        "argv_sha256": adoption._stable_sha256(argv),
        "cmdline_sha256": f"{pid:064x}"[-64:],
        "cwd": cwd,
        "exe": exe,
        "exe_identity": {
            "device": 1,
            "inode": 9001,
            "mode": 0o100755,
            "uid": os.getuid(),
            "nlink": 1,
            "size": 1,
            "mtime_ns": 1,
            "ctime_ns": 1,
        },
    }


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[adoption.T2PassAdoptionSources, dict[str, Path]]:
    runtime = tmp_path / "runtime"
    control = runtime / "control"
    controller = control / "tastemolnet-gine-training-v2" / adoption.SOURCE_CID
    training = control / "tastemolnet-gine-state-v2" / adoption.SOURCE_CID
    output = runtime / adoption.SOURCE_OUTPUT_RELATIVE
    run_state = (
        control / "experiment_registry" / "run_state" / adoption.SOURCE_RUN_ID
    )
    execution = tmp_path / "execution-583bf"
    identity_fix = tmp_path / "identity-fix-3a90"
    proc = tmp_path / "proc"
    for directory in (
        controller,
        training,
        output,
        run_state,
        execution,
        identity_fix,
        proc,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    training.chmod(0o700)

    source_files = {
        "scripts/autodl/run_tastemolnet_gnn_full.sh": b"#!/bin/bash\n",
        "src/utils/autodl_tastemolnet_gine_controller_v1.py": b"# deployed controller\n",
        "configs/gnn/gine.yaml": b"gnn:\n  backbone: gine\n",
        "configs/hpc.yaml": b"runtime: {}\n",
        "configs/autodl/tastemolnet_gine_research_v1.yaml": b"autodl: {}\n",
    }
    for relative, data in source_files.items():
        path = execution / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    fix_module = identity_fix / "src/utils/autodl_tastemolnet_gine_controller_v1.py"
    fix_module.parent.mkdir(parents=True)
    fix_module.write_bytes(b"# identity fix\n")

    model = output / "model.pt"
    last = output / "last.pt"
    model.write_bytes(b"frozen model")
    last.write_bytes(b"epoch 42 checkpoint")
    model_sha = _sha(model)
    last_sha = _sha(last)
    health = {"status": "PASS", "predicted_classes": [0, 1, 2], "failures": []}
    last_receipt = {
        "schema_version": "tastemolnet_last_training_checkpoint_v1",
        "checkpoint_file": "last.pt",
        "same_bytes_as_latest_epoch_checkpoint": True,
        "completed_epoch": 42,
        "checkpoint_sha256": last_sha,
        "source_checkpoint_sha256": last_sha,
    }
    output_json = {
        "config.yaml": {
            "gnn": {"backbone": "gine", "num_classes": 3},
            "training": {
                "primary_seed": 7,
                "class_weighted_loss": True,
                "weighted_sampler": False,
            },
        },
        "model_card.json": {
            "dataset": "tastemolnet",
            "profile": "full",
            "backbone": "gine",
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "num_classes": 3,
            "source_label": 1,
            "seed": 7,
            "training_commit": adoption.SOURCE_EXECUTION_COMMIT,
            "selection_split": "validation",
            "selection_metric": "macro_ovr_roc_auc",
            "selection_tiebreak_metric": "macro_f1",
            "test_loaded_during_training": False,
            "test_used_for_model_fit_or_selection": False,
            "checkpoint_id": model_sha,
            "training_resume_contract_sha256": "pending",
            "health_gate": health,
            "paper_result_reporting_allowed": True,
            "dataset_redistributed": False,
            "upstream_license_not_explicit": True,
            "license_pass_claimed": False,
        },
        "feature_schema.json": {"schema": "molecular"},
        "label_map.json": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
        "split_manifest.json": {"dataset": "tastemolnet"},
        "training_metrics.json": {
            "selection_metric": "macro_ovr_roc_auc",
            "selection_tiebreak_metric": "macro_f1",
            "class_weighted_loss": True,
            "weighted_sampler": False,
            "final_validation": {
                "macro_ovr_roc_auc": 0.91,
                "macro_f1": 0.72,
                "per_class": {
                    "0": {"recall": 0.5},
                    "1": {"recall": 0.6},
                    "2": {"recall": 0.7},
                }
            },
            "health_gate": health,
        },
        "test_evaluation_status.json": {
            "status": "NOT_EVALUATED",
            "test_loaded": False,
            "reason": "held out",
        },
        "temperature_scaling.json": {"status": "validation_only"},
        "environment.json": {"cuda": True},
        "git_state.json": {"commit": adoption.SOURCE_EXECUTION_COMMIT},
        "data_use_policy_binding.json": {
            "schema_version": "tastemolnet_training_policy_binding_v1",
            "dataset": "tastemolnet",
            "status": "NOT_EXPLICITLY_STATED",
            "authorization_status": "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION",
            "policy": {
                "policy_version": 2,
                "authorization_state": "ACTIVE_SCOPED_AUTHORIZATION",
                "authorization_status": "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION",
                "research_execution_allowed": True,
                "paper_reporting_allowed": True,
                "research_compute_allowed": True,
                "paper_result_reporting_allowed": True,
                "data_redistribution_allowed": False,
                "dataset_redistributed": False,
                "main_route_state": "READY_FOR_MAIN_ROUTE",
                "license_conclusion": "NOT_GRANTED_OR_INFERRED",
            },
            "paper_result_reporting_allowed": True,
            "paper_results_reporting_allowed_by_project_policy": True,
            "data_redistribution_allowed": False,
            "dataset_redistributed": False,
            "upstream_license_not_explicit": True,
            "upstream_license_status": "NOT_EXPLICITLY_STATED",
            "upstream_license_claimed_resolved": False,
            "license_pass_claimed": False,
            "public_artifact_audit_required": True,
            "hpc_execution_authorized": False,
        },
        "graph_cache_usage.json": {
            "schema_version": "tastemolnet_graph_cache_usage_v1",
            "dataset": "tastemolnet",
            "mode": "read_only_existing_cache",
            "loaded_splits": ["train", "validation"],
            "calibration_loaded": False,
            "test_loaded": False,
            "test_metadata_hash_only": True,
            "graph_cache_rebuilt": False,
            "data_reprepared": False,
        },
        "oracle_manifest.json": {
            "schema_version": "tastemolnet_three_class_gine_oracle_manifest_v1",
            "dataset": "tastemolnet",
            "status": "PASS",
            "checkpoint_id": model_sha,
            "classifier_family": "gine",
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "num_classes": 3,
            "label_map": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
            "source_label": 1,
            "source_label_name": "Sweet",
            "selection_split": "validation",
            "selection_metric": "macro_ovr_roc_auc",
            "selection_tiebreak_metric": "macro_f1",
            "temperature_calibration_split": "validation",
            "test_loaded": False,
            "test_evaluated": False,
            "paper_result_reporting_allowed": True,
            "dataset_redistributed": False,
            "upstream_license_not_explicit": True,
            "health_gate": health,
        },
        "last_checkpoint.json": last_receipt,
        "checkpoint_reload.json": {
            "schema_version": "tastemolnet_gine_checkpoint_reload_v1",
            "status": "PASS",
            "checkpoint_reload_pass": True,
            "batch_single_probability_equivalence": True,
            "all_probabilities_finite": True,
            "num_classes": 3,
            "source_label": 1,
            "checkpoint_id": model_sha,
            "last_checkpoint": last_receipt,
        },
    }
    for name, payload in output_json.items():
        _write_json(output / name, payload)
    (output / "validation_predictions.csv").write_text(
        "molecule_id,label,predicted_label\n1,1,1\n", encoding="utf-8"
    )
    hash_lines = [
        f"{_sha(output / name)}  {name}"
        for name in sorted(adoption.HASHED_CHECKPOINT_FILES)
    ]
    (output / "sha256sums.txt").write_text("\n".join(hash_lines) + "\n")

    contract = {
        "schema_version": "molecular_gnn_training_resume_contract_v1",
        "dataset": "tastemolnet",
        "profile": "full",
        "output_dir": str(output),
        "source_identity": {"commit": adoption.SOURCE_EXECUTION_COMMIT},
        "model_config": {
            "backbone": "gine",
            "num_classes": 3,
            "num_layers": 5,
            "hidden_dim": 256,
            "dropout": 0.2,
            "pooling": "mean",
            "readout_layers": 2,
            "normalization": "batch_norm",
            "residual": True,
            "edge_feature_mode": "native_edge_conditioned_message",
        },
        "training": {
            "max_epochs": 200,
            "early_stopping_patience": 20,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "seed": 7,
            "class_weighted_loss": True,
            "weighted_sampler": False,
            "selection_metric": "macro_ovr_roc_auc",
            "selection_tiebreak_metric": "macro_f1",
            "gradient_clip_norm": 5.0,
        },
        "tastemolnet_scoped_authority": {
            "policy": {
                "policy_version": 2,
                "authorization_state": "ACTIVE_SCOPED_AUTHORIZATION",
                "research_compute_allowed": True,
                "paper_result_reporting_allowed": True,
                "data_redistribution_allowed": False,
            }
        },
    }
    contract_sha = adoption._stable_sha256(contract)
    model_card = json.loads((output / "model_card.json").read_text())
    model_card["training_resume_contract_sha256"] = contract_sha
    _write_json(output / "model_card.json", model_card)
    # Refresh the 18-file hash closure after binding the contract.
    hash_lines = [
        f"{_sha(output / name)}  {name}"
        for name in sorted(adoption.HASHED_CHECKPOINT_FILES)
    ]
    (output / "sha256sums.txt").write_text("\n".join(hash_lines) + "\n")
    latest_name = "checkpoint-000042.pt"
    (training / latest_name).write_bytes(last.read_bytes())
    (training / ".root_identity").write_bytes(b"training identity\n")
    (training / ".root_identity").chmod(0o600)
    (training / ".writer.lock").write_bytes(b"")
    (training / ".writer.lock").chmod(0o600)
    training_claim = {
        "schema_version": adoption.TRAINING_STATE_SCHEMA,
        "artifact_kind": "molecular_gnn_training_state_root_claim",
        "root": str(training),
        "root_identity": adoption._file_identity(os.stat(training)),
        "sentinel": {
            "name": ".root_identity",
            "sha256": _sha(training / ".root_identity"),
            "identity": adoption._file_identity(os.stat(training / ".root_identity")),
        },
        "lock": {
            "name": ".writer.lock",
            "identity": adoption._file_identity(os.stat(training / ".writer.lock")),
        },
        "claim_nonce": "fixture",
    }
    _write_json(training / "root_claim.json", training_claim)
    contract_payload = {
        "schema_version": adoption.TRAINING_STATE_SCHEMA,
        "artifact_kind": "molecular_gnn_training_contract",
        "contract": contract,
        "contract_sha256": contract_sha,
        "root_claim_sha256": _sha(training / "root_claim.json"),
    }
    _write_json(
        training / "training_contract.json",
        contract_payload,
    )
    contract_evidence = {
        "schema_version": "molecular_gnn_training_contract_physical_v1",
        "name": "training_contract.json",
        "identity": adoption._file_identity(os.stat(training / "training_contract.json")),
        "file_sha256": _sha(training / "training_contract.json"),
        "canonical_sha256": contract_sha,
        "content": contract_payload,
    }
    _write_json(
        training / "latest_checkpoint.json",
        {
            "schema_version": adoption.TRAINING_STATE_SCHEMA,
            "status": "CHECKPOINT_COMPLETE",
            "contract_sha256": contract_sha,
            "training_contract_evidence": contract_evidence,
            "completed_epoch": 42,
            "next_epoch": 43,
            "checkpoint_file": latest_name,
            "checkpoint_sha256": last_sha,
            "checkpoint_bytes": len(last.read_bytes()),
        },
    )
    _write_json(
        training / "training_complete.json",
        {
            "schema_version": adoption.TRAINING_STATE_SCHEMA,
            "artifact_kind": "molecular_gnn_training_complete",
            "status": "PASS",
            "contract_sha256": contract_sha,
            "training_contract_evidence": contract_evidence,
            "output_dir": str(output),
            "output_identity": {
                "model_sha256": model_sha,
                "model_card_sha256": _sha(output / "model_card.json"),
                "sha256s_sha256": _sha(output / "sha256sums.txt"),
                "checkpoint_id": model_sha,
                "training_resume_contract_sha256": contract_sha,
            },
        },
    )

    parent_pid = 101
    child_pid = 102
    trainer_command = [
        "/python-link",
        "scripts/train_molecular_gnn.py",
        "--backbone",
        "gine",
    ]
    runtime_log = runtime / "logs" / f"{adoption.SOURCE_RUN_ID}.log"
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    runtime_log.write_text(
        "epoch=42\n[TASTE_GINE_THREE_CLASS_PASS]\n"
        "[MOLECULAR_GNN_TRAIN_OK]\n"
        "[AUTODL_RUN_EXIT] exit_code=0 timestamp=fixture\n",
        encoding="utf-8",
    )
    authority_path = run_state / "trainer_child_authority.json"
    barrier_lock = run_state / "trainer-startup.lock"
    barrier_record_path = run_state / "trainer-startup.json"
    barrier_lock.write_bytes(b"")
    barrier_lock.chmod(0o600)
    launcher_argv = [
        "/python-link",
        "-S",
        "-m",
        "src.utils.autodl_exec_startup_barrier",
        "--record",
        str(barrier_record_path),
        "--release-read-fd",
        "3",
        "--lock-fd",
        "4",
        "--",
        *trainer_command,
    ]
    expected_config_paths = [
        execution / "configs/hpc.yaml",
        execution / "configs/gnn/gine.yaml",
        execution / "configs/autodl/tastemolnet_gine_research_v1.yaml",
    ]
    parent_argv = [
        "/python-link",
        str(execution / "scripts/autodl/exp_run.py"),
        "--project-root",
        str(execution),
        "--data-root",
        str(tmp_path),
        "launch",
        "--dataset",
        "tastemolnet",
        "--stage",
        adoption.SOURCE_STAGE,
        "--gpu-index",
        "1",
        "--gpu-uuid",
        "GPU-fixture-0001",
        "--gpu-required",
        "--heavy",
        "--max-gpus",
        "4",
        "--gpu-hard-limit",
        "4",
        "--foreground",
    ]
    for config_path in expected_config_paths:
        parent_argv.extend(("--config-file", str(config_path)))
    parent_argv.extend(("--expected-output", str(output)))
    for required_name in adoption.CONTROLLER_REQUIRED_OUTPUT_FILES:
        parent_argv.extend(("--required-output-file", required_name))
    parent_argv.extend(
        (
            "--required-log-marker",
            "[TASTE_GINE_THREE_CLASS_PASS]",
            "--",
            *trainer_command,
        )
    )
    barrier = {
        "schema": "autodl_exec_startup_barrier_v1",
        "kind": "durable_exec_startup_barrier",
        "state": "ARMED_UNRELEASED",
        "lock_path": str(barrier_lock),
        "lock_dev": os.stat(barrier_lock).st_dev,
        "lock_inode": os.stat(barrier_lock).st_ino,
        "lock_mode": os.stat(barrier_lock).st_mode & 0o7777,
        "lock_uid": os.stat(barrier_lock).st_uid,
        "lock_nlink": os.stat(barrier_lock).st_nlink,
        "record_path": str(barrier_record_path),
        "python_executable": "/python-link",
        "release_read_fd": 3,
        "lock_fd": 4,
        "release_token_bytes": 32,
        "release_token_sha256": "a" * 64,
        "target_argv": trainer_command,
        "target_argv_sha256": adoption._stable_sha256(trainer_command),
        "launcher_argv": launcher_argv,
        "launcher_argv_sha256": adoption._stable_sha256(launcher_argv),
    }
    _write_json(barrier_record_path, barrier)
    authority = {
        "schema_version": adoption.TRAINER_AUTHORITY_SCHEMA,
        "status": "RELEASE_AUTHORIZED",
        "run_id": adoption.SOURCE_RUN_ID,
        "dataset": "tastemolnet",
        "stage": adoption.SOURCE_STAGE,
        "controller_cid": adoption.SOURCE_CID,
        "controller_root": str(controller),
        "project_root": str(execution),
        "authority_path": str(authority_path),
        "parent_exp_run": _snapshot(
            parent_pid,
            1,
            parent_argv,
            cwd=str(execution),
        ),
        "child_registered": _snapshot(
            child_pid,
            parent_pid,
            launcher_argv,
            cwd=str(execution),
        ),
        "trainer_command": trainer_command,
        "trainer_command_sha256": adoption._stable_sha256(trainer_command),
        "barrier_record": barrier,
    }
    _write_json(authority_path, authority)
    _write_json(
        run_state / "launch_spec.json",
        {
            "schema_version": 1,
            "run_id": adoption.SOURCE_RUN_ID,
            "project_root": str(execution),
            "data_root": str(tmp_path),
            "control_root": str(control),
            "python_executable": "/python",
            "dataset": "tastemolnet",
            "stage": adoption.SOURCE_STAGE,
            "command": trainer_command,
            "gpu_index": 1,
            "gpu_uuid": "GPU-fixture-0001",
            "max_gpus": 4,
            "gpu_hard_limit": 4,
            "git_commit": adoption.SOURCE_EXECUTION_COMMIT,
            "config_files": [str(path) for path in expected_config_paths],
            "input_manifest": None,
            "expected_output": str(output),
            "resume_published_output_receipt": None,
            "resume_published_output_receipt_sha256": None,
            "required_output_files": list(adoption.CONTROLLER_REQUIRED_OUTPUT_FILES),
            "required_output_any": [],
            "required_absolute_output_files": [],
            "required_log_marker": "[TASTE_GINE_THREE_CLASS_PASS]",
            "log_path": str(runtime_log),
            "heavy": True,
        },
    )
    _write_json(
        run_state / "state.json",
        {
            "schema_version": 1,
            "run_id": adoption.SOURCE_RUN_ID,
            "dataset": "tastemolnet",
            "stage": adoption.SOURCE_STAGE,
            "state": "PASS",
            "exit_code": 0,
            "pid": parent_pid,
            "child_pid": child_pid,
            "gpu_index": 1,
            "gpu_uuid": "GPU-fixture-0001",
            "failures": [],
            "log_path": str(runtime_log),
        },
    )
    registry = control / "experiment_registry/runs.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": adoption.SOURCE_RUN_ID,
                "dataset": "tastemolnet",
                "stage": adoption.SOURCE_STAGE,
                "state": "PASS",
                "exit_code": 0,
                "pid": parent_pid,
                "gpu_index": 1,
                "gpu_uuid": "GPU-fixture-0001",
                "command": trainer_command,
                "expected_output": str(output),
                "git_commit": adoption.SOURCE_EXECUTION_COMMIT,
                "backend": "autodl",
                "log_path": str(runtime_log),
            },
            sort_keys=True,
        )
        + "\n"
    )

    source_identity = {
        "commit": adoption.SOURCE_EXECUTION_COMMIT,
        "tree": "e" * 40,
        "worker_wrapper_path": str(
            execution / "scripts/autodl/run_tastemolnet_gnn_full.sh"
        ),
        "worker_wrapper_sha256": _sha(
            execution / "scripts/autodl/run_tastemolnet_gnn_full.sh"
        ),
        "controller_module_sha256": _sha(
            execution / "src/utils/autodl_tastemolnet_gine_controller_v1.py"
        ),
        "worker_program_path": "/bin/bash",
        "worker_program_sha256": "b" * 64,
        "python_executable": "/python",
        "python_executable_sha256": "c" * 64,
        "verified_backbone_config_path": str(execution / "configs/gnn/gine.yaml"),
        "verified_backbone_config_sha256": _sha(execution / "configs/gnn/gine.yaml"),
    }
    environment = {
        "PRIMARY_GNN_BACKBONE": "gine",
        "PRIMARY_SEED": "7",
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "TASTE_UPSTREAM_LICENSE_STATUS": "NOT_EXPLICITLY_STATED",
        "AUTODL_MAX_GPUS": "4",
        "TASTEMOLNET_GPU_INDEX": "1",
        "AUTODL_DATA_ROOT": str(tmp_path),
        "AUTODL_CONTROL_ROOT": str(control),
        "TASTEMOLNET_GNN_FULL_OUTPUT": str(output),
        "TASTEMOLNET_GNN_TRAINING_STATE_ROOT": str(training),
        "TASTEMOLNET_GINE_CONTROLLER_CID": adoption.SOURCE_CID,
        "TASTEMOLNET_GINE_CONTROLLER_ROOT": str(controller),
        "AUTODL_PYTHON": "/python-link",
    }
    config_files = [
        {"path": str(execution / relative), "sha256": _sha(execution / relative)}
        for relative in (
            "configs/hpc.yaml",
            "configs/gnn/gine.yaml",
            "configs/autodl/tastemolnet_gine_research_v1.yaml",
        )
    ]
    spec = {
        "schema_version": adoption.CONTROLLER_SCHEMA,
        "cid": adoption.SOURCE_CID,
        "controller_root": str(controller),
        "project_root": str(execution),
        "output_dir": str(output),
        "training_state_root": str(training),
        "source_identity": source_identity,
        "environment_authority": environment,
        "environment_authority_sha256": adoption._stable_sha256(environment),
        "config_files": config_files,
        "verified_model_route": {"backbone": "gine", "seed": 7},
        "physical_gpu_index": 1,
        "terminal_marker": "[TASTE_GINE_THREE_CLASS_PASS]",
        "required_output_files": list(adoption.CONTROLLER_REQUIRED_OUTPUT_FILES),
        "resource_wait_deadline_seconds": 21_600,
        "worker_argv": ["/bin/bash", str(execution / "scripts/autodl/run_tastemolnet_gnn_full.sh")],
    }
    spec["worker_argv_sha256"] = adoption._stable_sha256(spec["worker_argv"])
    _write_json(controller / "controller_spec.json", spec)
    (controller / ".controller-root-identity").write_bytes(b"controller identity\n")
    (controller / ".controller-root-identity").chmod(0o600)
    (controller / ".controller.lock").write_bytes(b"")
    (controller / ".controller.lock").chmod(0o600)
    deadline = {
        "schema_version": "autodl_tastemolnet_gine_resource_deadline_v1",
        "cid": adoption.SOURCE_CID,
        "spec_sha256": adoption._stable_sha256(spec),
        "duration_seconds": spec["resource_wait_deadline_seconds"],
        "started_epoch_seconds": 1_799_978_400,
        "deadline_epoch_seconds": 1_800_000_000,
    }
    _write_json(controller / "resource_wait_deadline.json", deadline)
    claim = {
        "schema_version": adoption.CONTROLLER_CLAIM_SCHEMA,
        "cid": adoption.SOURCE_CID,
        "root": str(controller),
        "root_identity": adoption._directory_identity(os.stat(controller)),
        "spec_sha256": adoption._stable_sha256(spec),
        "sentinel": {
            "sha256": _sha(controller / ".controller-root-identity"),
            "identity": adoption._file_identity(
                os.stat(controller / ".controller-root-identity")
            ),
        },
        "lock": {
            "identity": adoption._file_identity(os.stat(controller / ".controller.lock")),
        },
    }
    _write_json(controller / "controller_root_claim.json", claim)
    _write_json(
        controller / "controller_state.json",
        {
            "schema_version": adoption.CONTROLLER_STATE_SCHEMA,
            "cid": adoption.SOURCE_CID,
            "spec_sha256": adoption._stable_sha256(spec),
            "root_claim_sha256": _sha(controller / "controller_root_claim.json"),
            "updated_at": "2026-08-27T16:08:00Z",
            "resource_deadline_sha256": _sha(
                controller / "resource_wait_deadline.json"
            ),
            "resource_deadline_epoch_seconds": deadline["deadline_epoch_seconds"],
            "phase": "FAILED",
            "reason": adoption.SOURCE_FAILED_REASON,
            "attempt": 0,
            "launch_index": 0,
            "retries_used": 0,
        },
    )

    def fake_git(
        path: Path, *, critical_paths: tuple[str, ...] = ()
    ) -> dict[str, object]:
        common: dict[str, object] = {
            "status_porcelain": "",
            "critical_blobs": {
                relative: _sha(Path(adoption.__file__).resolve().parents[2] / relative)
                for relative in critical_paths
            },
            "parents": ["d" * 40],
            "parent_tree": "c" * 40,
            "changed_from_parent": ["many-files"],
            "parent_critical_blobs": {},
            "git_binary_path": "/usr/bin/git",
            "git_binary_identity": {"device": 1},
            "git_binary_sha256": "f" * 64,
            "git_environment_policy": "fixed_allowlist_no_replace_no_config_hooks",
        }
        if path == execution:
            return {
                **common,
                "commit": adoption.SOURCE_EXECUTION_COMMIT,
                "tree": "e" * 40,
            }
        if path == identity_fix:
            return {
                **common,
                "commit": adoption.SOURCE_IDENTITY_FIX_COMMIT,
                "tree": "f" * 40,
            }
        if path == Path(adoption.__file__).resolve().parents[2]:
            return {
                **common,
                "commit": "a" * 40,
                "tree": "b" * 40,
            }
        raise AssertionError(path)

    monkeypatch.setattr(adoption, "_git_identity", fake_git)
    sources = adoption.T2PassAdoptionSources._build_for_tests(
        control_root=control,
        controller_root=controller,
        output_root=output,
        training_state_root=training,
        execution_project_root=execution,
        identity_fix_project_root=identity_fix,
        proc_root=proc,
    )
    return sources, {
        "controller": controller,
        "output": output,
        "training": training,
        "registry": registry,
        "run_state": run_state,
    }


def _authorize_from_evidence(
    monkeypatch: pytest.MonkeyPatch, evidence: dict[str, object]
) -> None:
    expected = adoption._observed_release_values(evidence)
    monkeypatch.setattr(
        adoption,
        "_require_release",
        lambda _hold, observed: (
            dict(expected)
            if adoption._observed_release_values(observed) == expected
            else (_ for _ in ()).throw(
                adoption.T2PassAdoptionReleaseDisabled("test evidence drift")
            )
        ),
    )


def test_preflight_is_read_only_and_default_publish_is_stage_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    before = {name: _sha(path) for name, path in {
        "registry": paths["registry"],
        "controller": paths["controller"] / "controller_state.json",
        "completion": paths["training"] / "training_complete.json",
    }.items()}

    evidence = adoption.preflight_t2_gine_pass_adoption(sources)

    assert evidence["source_result"]["num_classes"] == 3
    assert evidence["source_result"]["source_label"] == 1
    assert evidence["failed_controller"]["reason"] == "WORKER_PROCESS_IDENTITY_DRIFT"
    assert evidence["run_authority"]["all_declared_pids_dead"] is True
    assert evidence["run_authority"]["trainer_python_raw_argv_token"] == "/python-link"
    assert evidence["run_authority"]["trainer_python_physical_executable"] == "/python"
    assert evidence["scientific_output"]["hash_count"] == 18
    candidate = adoption.reviewed_release_candidate(evidence)
    assert candidate["authorization"] is False
    assert candidate["status"] == "UNREVIEWED_RELEASE_CANDIDATE"
    assert candidate["source_pins"]["controller_root"] == str(paths["controller"])
    assert candidate["source_pins"]["training_state_root"] == str(paths["training"])
    assert candidate["source_pins"]["output_inventory_sha256"] == evidence["scientific_output"][
        "inventory_sha256"
    ]
    assert not sources.adoption_root.exists()
    assert before == {name: _sha(path) for name, path in {
        "registry": paths["registry"],
        "controller": paths["controller"] / "controller_state.json",
        "completion": paths["training"] / "training_complete.json",
    }.items()}
    with pytest.raises(adoption.T2PassAdoptionReleaseDisabled, match="stage-frozen"):
        adoption.publish_t2_gine_pass_adoption(sources)
    assert not sources.adoption_root.exists()


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("controller", "exact failed identity-drift"),
        ("registry", "registry terminal event"),
        ("runtime", "runtime PASS marker"),
        ("completion", "training-complete/latest checkpoint"),
        ("log", "lacks required marker"),
    ),
)
def test_each_independent_pass_authority_is_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    if case == "controller":
        target = paths["controller"] / "controller_state.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["reason"] = "SOME_OTHER_FAILURE"
        _write_json(target, payload)
    elif case == "registry":
        payload = json.loads(paths["registry"].read_text(encoding="utf-8"))
        payload["state"] = "FAILED"
        paths["registry"].write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif case == "runtime":
        target = paths["run_state"] / "state.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["state"] = "FAILED"
        _write_json(target, payload)
    elif case == "completion":
        target = paths["training"] / "training_complete.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["status"] = "FAILED"
        _write_json(target, payload)
    else:
        runtime_log = sources.runtime_root / "logs" / f"{adoption.SOURCE_RUN_ID}.log"
        runtime_log.write_text("[AUTODL_RUN_EXIT] exit_code=0\n", encoding="utf-8")

    with pytest.raises(adoption.T2PassAdoptionError, match=message):
        adoption.preflight_t2_gine_pass_adoption(sources)
    assert not sources.adoption_root.exists()


def test_reviewed_publish_is_exact_five_file_receipt_and_status_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    main_root = sources.control_root / "tastemolnet-main-v1"
    matrix_root = (
        sources.runtime_root
        / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1"
    )
    _write_json(main_root / "controller_state.json", {"state": "UNCHANGED"})
    _write_json(matrix_root / "matrix_status.json", {"state": "UNCHANGED"})
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    _authorize_from_evidence(monkeypatch, evidence)
    protected_roots = {
        "old_controller": paths["controller"],
        "old_scientific_output": paths["output"],
        "old_training_state": paths["training"],
        "old_run_state": paths["run_state"],
        "main_controller": main_root,
        "main_matrix": matrix_root,
    }
    before_protected = {
        name: _tree_snapshot(path) for name, path in protected_roots.items()
    }
    before_registry = _tree_snapshot(paths["registry"])
    write_order: list[str] = []
    original_write_new_at = adoption._write_new_at

    def recording_write_new_at(
        directory_fd: int, name: str, data: bytes
    ) -> adoption.HeldFile:
        write_order.append(name)
        return original_write_new_at(directory_fd, name, data)

    monkeypatch.setattr(adoption, "_write_new_at", recording_write_new_at)
    original_gate_commit = adoption._rename_gate_noreplace

    def recording_gate_commit(
        directory_fd: int, *, retained_closure: Callable[[], None]
    ) -> None:
        write_order.append("gate.json")
        original_gate_commit(
            directory_fd,
            retained_closure=retained_closure,
        )

    monkeypatch.setattr(adoption, "_rename_gate_noreplace", recording_gate_commit)

    result = adoption.publish_t2_gine_pass_adoption(sources)

    assert result["state"] == adoption.ADOPTION_MARKER
    assert write_order == list(adoption.FIVE_FILE_SET)
    assert sorted(path.name for path in sources.adoption_root.iterdir()) == sorted(
        adoption.FIVE_FILE_SET
    )
    assert not (sources.adoption_root / "PASS").exists()
    receipt = sources.adoption_root / "manifest.json"
    assert result["receipt_sha256"] == _sha(receipt)
    manifest = json.loads(receipt.read_text(encoding="utf-8"))
    state = json.loads(
        (sources.adoption_root / "state.json").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (sources.adoption_root / "gate.json").read_text(encoding="utf-8")
    )
    completion = manifest["completion_semantics"]
    assert completion["old_controller_state"] == "FAILED"
    assert completion["old_controller_reason"] == adoption.SOURCE_FAILED_REASON
    assert completion["old_controller_is_scientific_false_negative"] is True
    assert completion["old_controller_record_retained_as_control_plane_truth"] is True
    assert completion["scientific_bundle_status"] == "PASS"
    assert completion["run_registry_status"] == "PASS"
    assert completion["runtime_status"] == "PASS"
    assert completion["training_complete_status"] == "PASS"
    assert state["completion_semantics"] == completion
    assert manifest["publication_boundary"] == {
        "only_write_root": str(sources.adoption_root),
        "old_controller": "RETAINED_READ_ONLY",
        "old_scientific_output": "RETAINED_READ_ONLY",
        "old_training_state": "RETAINED_READ_ONLY",
        "main_controller": "NOT_OPENED_NOT_WRITTEN",
        "main_matrix": "NOT_OPENED_NOT_WRITTEN",
        "source_identity_revalidated_through_gate_commit": True,
    }
    assert manifest["t3_dependency_contract"] == {
        "required_t2_gate_file": "gate.json",
        "required_t2_receipt_file": "manifest.json",
        "receipt_binding": "gate.receipt_sha256 == SHA256(manifest.json)",
        "required_formal_bundle_root": str(paths["output"]),
        "required_formal_bundle_inventory_sha256": evidence["scientific_output"][
            "inventory_sha256"
        ],
        "required_model_sha256": evidence["scientific_output"]["model_sha256"],
        "other_t2_authorities_allowed": False,
    }
    assert gate["gate_published_last"] is True
    assert gate["no_fallible_operation_after_gate_publication"] is True
    assert set(gate["physical_binding"]["documents"]) == set(
        adoption.FIVE_FILE_SET[:-1]
    )
    assert set(
        gate["physical_binding"]["documents"]["manifest.json"]["identity"]
    ) == {"device", "inode", "mode", "uid", "gid", "nlink", "size"}
    assert set(gate["physical_binding"]["adoption_root_identity"]) == {
        "device",
        "inode",
        "mode",
        "uid",
        "gid",
    }
    assert gate["receipt_sha256"] == _sha(receipt)
    assert before_protected == {
        name: _tree_snapshot(path) for name, path in protected_roots.items()
    }
    assert before_registry == _tree_snapshot(paths["registry"])
    assert evidence == adoption.preflight_t2_gine_pass_adoption(sources)
    before = {path.name: _sha(path) for path in sources.adoption_root.iterdir()}
    status = adoption.validate_t2_gine_pass_adoption(sources)
    assert status["receipt_sha256"] == result["receipt_sha256"]
    assert status["read_only_validation"] is True
    assert before == {path.name: _sha(path) for path in sources.adoption_root.iterdir()}
    with pytest.raises(adoption.T2PassAdoptionError, match="already exists"):
        adoption.publish_t2_gine_pass_adoption(sources)


def test_source_or_receipt_tamper_fails_closed_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    _authorize_from_evidence(monkeypatch, evidence)
    adoption.publish_t2_gine_pass_adoption(sources)
    gate = sources.adoption_root / "gate.json"
    gate.write_text(gate.read_text() + " ", encoding="utf-8")
    with pytest.raises(adoption.T2PassAdoptionError):
        adoption.validate_t2_gine_pass_adoption(sources)
    assert gate.exists()

    paths["output"].joinpath("model.pt").write_bytes(b"tampered")
    with pytest.raises(adoption.T2PassAdoptionError):
        adoption.validate_t2_gine_pass_adoption(sources)
    assert gate.exists()


def test_live_pid_and_symlinked_source_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    (sources.proc_root / "101").mkdir()
    with pytest.raises(adoption.T2PassAdoptionError, match="still present"):
        adoption.preflight_t2_gine_pass_adoption(sources)
    (sources.proc_root / "101").rmdir()

    real = paths["output"] / "model.pt"
    replacement = tmp_path / "replacement-model"
    replacement.write_bytes(real.read_bytes())
    real.unlink()
    real.symlink_to(replacement)
    with pytest.raises(adoption.T2PassAdoptionError, match="symlinked or special"):
        adoption.preflight_t2_gine_pass_adoption(sources)


def test_release_config_and_source_pins_require_exact_native_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _paths = _fixture(tmp_path, monkeypatch)
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    disabled = {
        "schema_version": adoption.RELEASE_CONFIG_SCHEMA,
        "stage": "T2_GINE_FULL",
        "dataset": "tastemolnet",
        "authorization": False,
        "external_authority_path": None,
        "external_authority_sha256": None,
    }
    adoption._SourceHold._validate_release_config(disabled)
    for foreign in (1, 0, "true", None):
        changed = {**disabled, "authorization": foreign}
        with pytest.raises(adoption.T2PassAdoptionError, match="schema/type"):
            adoption._SourceHold._validate_release_config(changed)
    enabled = {
        **disabled,
        "authorization": True,
        "external_authority_path": "/reviewed/t2-release.json",
        "external_authority_sha256": "a" * 64,
    }
    adoption._SourceHold._validate_release_config(enabled)
    for key, foreign in (
        ("external_authority_path", Path("/reviewed/t2-release.json")),
        ("external_authority_path", "relative.json"),
        ("external_authority_sha256", True),
        ("external_authority_sha256", "A" * 64),
    ):
        with pytest.raises(adoption.T2PassAdoptionError):
            adoption._SourceHold._validate_release_config(
                {**enabled, key: foreign}
            )

    pins = adoption._observed_release_values(evidence)
    assert adoption._validate_source_release_pins(pins) == pins
    for key, foreign in (
        ("control_root", Path(pins["control_root"])),
        ("training_complete_sha256", True),
        ("execution_tree", "e" * 39),
    ):
        changed_pins = {**pins, key: foreign}
        with pytest.raises(adoption.T2PassAdoptionReleaseDisabled):
            adoption._validate_source_release_pins(changed_pins)


def test_existing_partial_root_is_never_reconciled_or_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _paths = _fixture(tmp_path, monkeypatch)
    sources.adoption_root.mkdir(parents=True)
    partial = sources.adoption_root / "state.json"
    partial.write_bytes(b"interrupted\n")

    with pytest.raises(adoption.T2PassAdoptionError, match="never resumes"):
        adoption.publish_t2_gine_pass_adoption(sources)

    assert partial.read_bytes() == b"interrupted\n"


def test_destination_namespace_replacement_fails_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _paths = _fixture(tmp_path, monkeypatch)
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    _authorize_from_evidence(monkeypatch, evidence)
    original_write_new_at = adoption._write_new_at
    moved_namespace = sources.control_root / f"{adoption.ADOPTION_NAMESPACE}.moved"

    def replace_after_first_write(
        directory_fd: int, name: str, data: bytes
    ) -> adoption.HeldFile:
        held = original_write_new_at(directory_fd, name, data)
        if name == "input_hashes.json":
            (sources.control_root / adoption.ADOPTION_NAMESPACE).rename(
                moved_namespace
            )
            (sources.control_root / adoption.ADOPTION_NAMESPACE).mkdir(mode=0o700)
        return held

    monkeypatch.setattr(adoption, "_write_new_at", replace_after_first_write)
    with pytest.raises(adoption.T2PassAdoptionError, match="destination authority"):
        adoption.publish_t2_gine_pass_adoption(sources)

    partial_root = moved_namespace / adoption.SOURCE_CID
    assert (partial_root / "input_hashes.json").is_file()
    assert not (partial_root / "gate.json").exists()


def test_exact_checkpoint_file_and_hash_count_closure_is_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    (paths["output"] / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(adoption.T2PassAdoptionError, match="exact 19-file"):
        adoption.preflight_t2_gine_pass_adoption(sources)


@pytest.mark.parametrize(
    ("field", "foreign_pid"),
    (
        ("pid", True),
        ("pid", 101.0),
        ("pid", "101"),
        ("pid", None),
        ("linux_start_ticks", True),
        ("ppid", False),
    ),
)
def test_process_snapshot_pid_requires_exact_native_int(
    field: str, foreign_pid: object
) -> None:
    snapshot = _snapshot(101, 1, ["/python", "worker.py"])
    snapshot[field] = foreign_pid
    with pytest.raises(adoption.T2PassAdoptionError, match="PID|parent"):
        adoption._validate_process_snapshot(snapshot, label="hostile snapshot")


def test_failed_controller_state_rejects_unreviewed_generation_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    target = paths["controller"] / "controller_state.json"
    state = json.loads(target.read_text(encoding="utf-8"))
    state["worker_generation"] = {"pid": 87809, "linux_start_ticks": 728135306}
    _write_json(target, state)
    with pytest.raises(adoption.T2PassAdoptionError, match="exact failed"):
        adoption.preflight_t2_gine_pass_adoption(sources)


@pytest.mark.parametrize("authority", ("runtime", "registry"))
def test_runtime_and_registry_pid_fields_reject_bool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    target = (
        paths["run_state"] / "state.json"
        if authority == "runtime"
        else paths["registry"]
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["pid"] = True
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(adoption.T2PassAdoptionError, match="runtime PASS|registry"):
        adoption.preflight_t2_gine_pass_adoption(sources)


def test_production_sources_fix_procfs_and_cli_has_no_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _paths = _fixture(tmp_path, monkeypatch)
    production = adoption.T2PassAdoptionSources.build(
        control_root=sources.control_root,
        controller_root=sources.controller_root,
        output_root=sources.output_root,
        training_state_root=sources.training_state_root,
        execution_project_root=sources.execution_project_root,
        identity_fix_project_root=sources.identity_fix_project_root,
    )
    assert production.proc_root == Path("/proc")
    assert production._test_proc_override is False
    with pytest.raises(TypeError):
        adoption.T2PassAdoptionSources.build(
            control_root=sources.control_root,
            controller_root=sources.controller_root,
            output_root=sources.output_root,
            training_state_root=sources.training_state_root,
            execution_project_root=sources.execution_project_root,
            identity_fix_project_root=sources.identity_fix_project_root,
            proc_root=tmp_path / "fake-proc",  # type: ignore[call-arg]
        )


def test_status_rejects_equal_byte_leaf_and_root_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _paths = _fixture(tmp_path, monkeypatch)
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    _authorize_from_evidence(monkeypatch, evidence)
    adoption.publish_t2_gine_pass_adoption(sources)

    manifest = sources.adoption_root / "manifest.json"
    replacement = sources.adoption_root / ".manifest.copy"
    shutil.copy2(manifest, replacement)
    os.replace(replacement, manifest)
    with pytest.raises(adoption.T2PassAdoptionError, match="closure changed"):
        adoption.validate_t2_gine_pass_adoption(sources)

    # Restore a clean fixture for the directory replacement case.
    sources2, _paths2 = _fixture(tmp_path / "second", monkeypatch)
    evidence2 = adoption.preflight_t2_gine_pass_adoption(sources2)
    _authorize_from_evidence(monkeypatch, evidence2)
    adoption.publish_t2_gine_pass_adoption(sources2)
    moved = sources2.adoption_root.with_name(sources2.adoption_root.name + ".held")
    sources2.adoption_root.rename(moved)
    shutil.copytree(moved, sources2.adoption_root, copy_function=shutil.copy2)
    with pytest.raises(adoption.T2PassAdoptionError, match="closure changed"):
        adoption.validate_t2_gine_pass_adoption(sources2)


@pytest.mark.parametrize(
    "replacement_site",
    ("destination_root", "destination_leaf", "source_leaf"),
)
def test_terminal_primitive_revalidates_retained_closure_after_entry_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_site: str,
) -> None:
    sources, paths = _fixture(tmp_path, monkeypatch)
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    _authorize_from_evidence(monkeypatch, evidence)
    original_commit = adoption._rename_gate_noreplace
    detached_root = sources.adoption_root.with_name(
        sources.adoption_root.name + ".detached"
    )

    def replace_at_terminal_primitive_entry(
        directory_fd: int, *, retained_closure: Callable[[], None]
    ) -> None:
        if replacement_site == "destination_root":
            sources.adoption_root.rename(detached_root)
            shutil.copytree(
                detached_root,
                sources.adoption_root,
                copy_function=shutil.copy2,
            )
        else:
            target = (
                sources.adoption_root / "manifest.json"
                if replacement_site == "destination_leaf"
                else paths["output"] / "model.pt"
            )
            replacement = target.with_name(target.name + ".equal-byte-copy")
            shutil.copy2(target, replacement)
            os.replace(replacement, target)
        original_commit(
            directory_fd,
            retained_closure=retained_closure,
        )

    monkeypatch.setattr(
        adoption,
        "_rename_gate_noreplace",
        replace_at_terminal_primitive_entry,
    )

    with pytest.raises(adoption.T2PassAdoptionError):
        adoption.publish_t2_gine_pass_adoption(sources)

    assert not (sources.adoption_root / "gate.json").exists()
    if detached_root.exists():
        assert not (detached_root / "gate.json").exists()


@pytest.mark.parametrize("failure_site", ("prepare", "commit"))
def test_gate_prepare_or_commit_failure_never_publishes_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    sources, _paths = _fixture(tmp_path, monkeypatch)
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    _authorize_from_evidence(monkeypatch, evidence)
    if failure_site == "prepare":
        original = adoption._write_prepared_gate

        def fail_after_prepared(directory_fd: int, data: bytes) -> adoption.HeldFile:
            held = original(directory_fd, data)
            held.close()
            raise OSError("injected prepared-gate fsync/close failure")

        monkeypatch.setattr(adoption, "_write_prepared_gate", fail_after_prepared)
    else:
        monkeypatch.setattr(
            adoption,
            "_rename_gate_noreplace",
            lambda _directory_fd, *, retained_closure: (_ for _ in ()).throw(
                OSError("injected terminal rename failure")
            ),
        )
    with pytest.raises(OSError, match="injected"):
        adoption.publish_t2_gine_pass_adoption(sources)
    assert not (sources.adoption_root / "gate.json").exists()
    assert (sources.adoption_root / ".gate.json.prepared").exists()


def test_gate_commit_has_no_fallible_postcommit_fsync_or_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _paths = _fixture(tmp_path, monkeypatch)
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    _authorize_from_evidence(monkeypatch, evidence)
    committed = False
    original_commit = adoption._rename_gate_noreplace
    original_fsync = adoption.os.fsync
    original_close = adoption.os.close

    def commit(
        directory_fd: int, *, retained_closure: Callable[[], None]
    ) -> None:
        nonlocal committed
        original_commit(
            directory_fd,
            retained_closure=retained_closure,
        )
        committed = True

    def hostile_fsync(descriptor: int) -> None:
        if committed:
            raise OSError("postcommit fsync attempted")
        original_fsync(descriptor)

    def hostile_close(descriptor: int) -> None:
        original_close(descriptor)
        if committed:
            raise OSError("postcommit close failure")

    monkeypatch.setattr(adoption, "_rename_gate_noreplace", commit)
    monkeypatch.setattr(adoption.os, "fsync", hostile_fsync)
    monkeypatch.setattr(adoption.os, "close", hostile_close)

    result = adoption.publish_t2_gine_pass_adoption(sources)

    assert result["status"] == "PASS"
    assert committed is True
    assert (sources.adoption_root / "gate.json").is_file()


def test_hardened_git_ignores_hostile_config_and_rejects_hidden_index_or_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    tracked = root / "tracked.txt"
    tracked.write_text("reviewed\n", encoding="utf-8")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(adoption.GIT_BINARY), *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )

    git("init", "-q")
    git("add", ".gitignore", "tracked.txt")
    git(
        "-c",
        "user.name=T2 Test",
        "-c",
        "user.email=t2@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    sentinel = tmp_path / "FS_MONITOR_EXECUTED"
    monitor = tmp_path / "monitor.sh"
    monitor.write_text(f"#!/bin/sh\ntouch {sentinel}\n", encoding="utf-8")
    monitor.chmod(0o755)
    git("config", "core.fsmonitor", str(monitor))
    git("config", "core.worktree", str(tmp_path / "foreign-worktree"))

    identity = adoption._git_identity(root, critical_paths=("tracked.txt",))

    assert identity["critical_blobs"]["tracked.txt"] == _sha(tracked)
    assert identity["git_binary_path"] == "/usr/bin/git"
    assert not sentinel.exists()

    # Explicit work-tree arguments are required after the hostile local config.
    subprocess.run(
        [
            str(adoption.GIT_BINARY),
            f"--git-dir={root / '.git'}",
            f"--work-tree={root}",
            "update-index",
            "--assume-unchanged",
            "tracked.txt",
        ],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    with pytest.raises(adoption.T2PassAdoptionError, match="index contains"):
        adoption._git_identity(root, critical_paths=("tracked.txt",))
    subprocess.run(
        [
            str(adoption.GIT_BINARY),
            f"--git-dir={root / '.git'}",
            f"--work-tree={root}",
            "update-index",
            "--no-assume-unchanged",
            "tracked.txt",
        ],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "hostile.pyc").write_bytes(b"cache")
    with pytest.raises(adoption.T2PassAdoptionError, match="not fully clean"):
        adoption._git_identity(root, critical_paths=("tracked.txt",))


def test_external_release_binds_parent_commit_tree_and_critical_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _paths = _fixture(tmp_path, monkeypatch)
    evidence = adoption.preflight_t2_gine_pass_adoption(sources)
    implementation_commit = "1" * 40
    implementation_tree = "2" * 40
    blobs = dict(evidence["source_code"]["adoption_git"]["critical_blobs"])
    release_evidence = json.loads(json.dumps(evidence))
    release_git = release_evidence["source_code"]["adoption_git"]
    release_git.update(
        {
            "parents": [implementation_commit],
            "parent_tree": implementation_tree,
            "changed_from_parent": [adoption.RELEASE_CONFIG_RELATIVE],
            "parent_critical_blobs": blobs,
        }
    )
    receipt = {
        "schema_version": adoption.EXTERNAL_RELEASE_SCHEMA,
        "stage": "T2_GINE_FULL",
        "dataset": "tastemolnet",
        "status": "REVIEWED_RELEASE_AUTHORIZED",
        "authorization": True,
        "implementation": {
            "commit": implementation_commit,
            "tree": implementation_tree,
            "critical_blobs": blobs,
        },
        "source_pins": adoption._observed_release_values(release_evidence),
    }
    assert adoption._validate_external_release(
        receipt, evidence=release_evidence
    ) == receipt["source_pins"]
    for mutation in (
        {"authorization": 1},
        {"implementation": {**receipt["implementation"], "tree": "3" * 40}},
        {"source_pins": {**receipt["source_pins"], "control_root": True}},
    ):
        with pytest.raises(adoption.T2PassAdoptionReleaseDisabled):
            adoption._validate_external_release(
                {**receipt, **mutation}, evidence=release_evidence
            )


def test_formula_release_defaults_cli_and_slurm_are_static() -> None:
    root = Path(__file__).resolve().parents[2]
    control = Path("/autodl-fs/data/counterfactual-subgraph-runtime/control")
    assert adoption.adoption_output_root(control) == (
        control / adoption.ADOPTION_NAMESPACE / adoption.SOURCE_CID
    )
    release_config = json.loads(
        (root / adoption.RELEASE_CONFIG_RELATIVE).read_text(encoding="utf-8")
    )
    assert release_config["authorization"] is False
    assert release_config["external_authority_path"] is None
    assert release_config["external_authority_sha256"] is None
    cli = (root / "scripts/autodl/adopt_tastemolnet_gine_pass_v1.py").read_text()
    slurm = (root / "scripts/slurm/adopt_tastemolnet_gine_pass_v1.sh").read_text()
    assert {"preflight", "status", "publish"} <= set(cli.split('"'))
    assert "--proc-root" not in cli
    assert "#SBATCH --partition=A800" in slurm
    assert "#SBATCH --gres=gpu:a800:1" in slurm
    assert "#SBATCH --output=logs/%j.out" in slurm
    assert "#SBATCH --error=logs/%j.err" in slurm
    assert "export PYTHONPATH=$PWD" in slurm
    assert "--config configs/hpc.yaml" in slurm
    assert "exit 78" in slurm
    assert "python -I -B scripts/autodl/adopt_tastemolnet_gine_pass_v1.py" in slurm
    assert slurm.index("exit 78") < slurm.index(
        "python -I -B scripts/autodl/adopt_tastemolnet_gine_pass_v1.py"
    )
