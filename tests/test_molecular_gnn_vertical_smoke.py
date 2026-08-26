from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.calibrate_gnn_classifier import main as calibrate_main
from scripts.autodl.exp_run import (
    SCHEMA_VERSION as EXP_RUN_SCHEMA_VERSION,
    TRAINER_CHILD_AUTHORITY_NAME,
    TRAINER_CHILD_AUTHORITY_SCHEMA,
    run_worker,
)
from scripts.evaluate_molecular_gnn import main as evaluate_main
from scripts.train_molecular_gnn import (
    _runtime_identity,
    _set_seed,
    main as train_main,
)
from src.data.molecular_graph_dataset import MolecularGraphDataset
from src.oracles.gnn_oracle import GNNOracle, sha256_file, verify_checkpoint_bundle
from src.utils.autodl_runtime import (
    GPU_LOCK_EXCLUSIVE,
    build_runtime_layout,
    sanitized_environment,
)
from src.utils import autodl_tastemolnet_gine_controller_v1 as controller_module
from src.utils.autodl_tastemolnet_gine_controller_v1 import (
    PUBLISHED_ADOPTION_NAME,
    TasteGINEControllerSpec,
    TasteGINEPersistentController,
)
from src.train.molecular_gnn_resume import MolecularGNNResumeError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_split(path: Path, split: str) -> None:
    smiles_by_label = {
        1: ("CC", "CCC", "CCCC", "CCCCC"),
        0: ("CN", "CCN", "CCCN", "CCCCN"),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("molecule_id", "smiles", "label", "split")
        )
        writer.writeheader()
        # Deliberately group labels to regress the real BACE prefix ordering.
        for label in (1, 0):
            for index, smiles in enumerate(smiles_by_label[label]):
                writer.writerow(
                    {
                        "molecule_id": f"{split}-{label}-{index}",
                        "smiles": smiles,
                        "label": label,
                        "split": split,
                    }
                )


def _write_tiny_resume_config(path: Path) -> Path:
    path.write_text(
        "gnn:\n"
        "  backbone: gine\n"
        "  num_layers: 1\n"
        "  hidden_dim: 16\n"
        "  dropout: 0.0\n"
        "  pooling: mean\n"
        "  readout_layers: 1\n"
        "  normalization: layer_norm\n"
        "  residual: true\n"
        "training:\n"
        "  optimizer: adamw\n"
        "  learning_rate: 0.001\n"
        "  weight_decay: 0.0\n"
        "  max_epochs: 1\n"
        "  early_stopping_patience: 1\n"
        "  batch_size: 4\n"
        "  primary_seed: 7\n"
        "  selection_metric: roc_auc\n"
        "  class_weighted_loss: true\n"
        "  weighted_sampler: false\n"
        "  gradient_clip_norm: 5.0\n"
        "runtime:\n"
        "  device: cpu\n"
        "  num_workers: 0\n",
        encoding="utf-8",
    )
    return path


def test_taste_determinism_is_error_mode_and_no_gpu_runtime_fails_closed(
    monkeypatch,
) -> None:
    import torch

    required = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "7",
        "NVIDIA_TF32_OVERRIDE": "0",
        "CUDNN_DETERMINISTIC": "1",
    }
    for key, value in required.items():
        monkeypatch.setenv(key, value)
    _set_seed(7, exact_cuda=True)
    assert torch.are_deterministic_algorithms_enabled() is True
    if hasattr(torch, "get_deterministic_debug_mode"):
        assert torch.get_deterministic_debug_mode() == 2
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(MolecularGNNResumeError, match="frozen deterministic"):
        _set_seed(7, exact_cuda=True)

    if not torch.cuda.is_available():
        monkeypatch.setenv("AUTODL_PHYSICAL_GPU_INDEX", "2")
        monkeypatch.setenv("AUTODL_PHYSICAL_GPU_UUID", "GPU-fixture")
        with pytest.raises(
            MolecularGNNResumeError, match="exactly one masked CUDA device"
        ):
            _runtime_identity(torch=torch, device="cuda:0", taste_full=True)


def test_train_calibrate_evaluate_vertical_smoke(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "splits"
    data.mkdir()
    _write_split(data / "train.csv", "train")
    _write_split(data / "val.csv", "val")
    _write_split(data / "calibration.csv", "calibration")
    _write_split(data / "test.csv", "test")
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "gnn:\n"
        "  backbone: gine\n"
        "  num_layers: 1\n"
        "  hidden_dim: 16\n"
        "  dropout: 0.0\n"
        "  pooling: mean\n"
        "  readout_layers: 1\n"
        "  normalization: layer_norm\n"
        "  residual: true\n"
        "training:\n"
        "  optimizer: adamw\n"
        "  learning_rate: 0.001\n"
        "  weight_decay: 0.0\n"
        "  max_epochs: 1\n"
        "  early_stopping_patience: 1\n"
        "  batch_size: 4\n"
        "  primary_seed: 7\n"
        "  selection_metric: roc_auc\n"
        "  class_weighted_loss: true\n"
        "  weighted_sampler: false\n"
        "  gradient_clip_norm: 5.0\n"
        "runtime:\n"
        "  device: cpu\n"
        "  num_workers: 0\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    loaded_split_paths: list[Path] = []
    original_from_csv = MolecularGraphDataset.from_csv.__func__

    def tracked_from_csv(cls, csv_path, *args, **kwargs):
        resolved = Path(csv_path).resolve()
        loaded_split_paths.append(resolved)
        if resolved == (data / "test.csv").resolve():
            raise AssertionError("training must not load the held-out test split")
        return original_from_csv(cls, csv_path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            MolecularGraphDataset, "from_csv", classmethod(tracked_from_csv)
        )
        assert (
            train_main(
                [
                    "--config",
                    str(config),
                    "--dataset",
                    "bace",
                    "--data-dir",
                    str(data),
                    "--output-dir",
                    str(checkpoint),
                    "--profile",
                    "smoke",
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )
    assert loaded_split_paths == [
        (data / "train.csv").resolve(),
        (data / "val.csv").resolve(),
    ]
    assert not (checkpoint / "test_predictions.csv").exists()
    test_status = json.loads(
        (checkpoint / "test_evaluation_status.json").read_text(encoding="utf-8")
    )
    assert test_status == {
        "path": str((data / "test.csv").resolve()),
        "reason": "held_out_until_frozen_final_evaluation",
        "schema_version": "molecular_gnn_test_evaluation_status_v1",
        "sha256": sha256_file(data / "test.csv"),
        "status": "NOT_EVALUATED",
        "test_loaded": False,
    }
    assert len(test_status["sha256"]) == 64
    assert calibrate_main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--validation-csv",
            str(data / "val.csv"),
        ]
    ) == 0
    evaluation = tmp_path / "evaluation"
    assert evaluate_main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--dataset-csv",
            str(data / "test.csv"),
            "--dataset",
            "bace",
            "--split",
            "test",
            "--output-dir",
            str(evaluation),
            "--device",
            "cpu",
        ]
    ) == 0

    audit = verify_checkpoint_bundle(checkpoint)
    oracle = GNNOracle.from_checkpoint(checkpoint, device="cpu")
    temperature = json.loads(
        (checkpoint / "temperature_scaling.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    assert audit["model_card"]["oracle_backend"] == "gnn"
    assert temperature["status"] == "fit"
    assert oracle.temperature == temperature["temperature"]
    assert metrics["selection_performed"] is False
    assert metrics["temperature_fitted"] is False


def test_train_epoch_resume_state_and_terminal_reopen(tmp_path: Path) -> None:
    data = tmp_path / "splits"
    data.mkdir()
    _write_split(data / "train.csv", "train")
    _write_split(data / "val.csv", "val")
    _write_split(data / "calibration.csv", "calibration")
    _write_split(data / "test.csv", "test")
    config = tmp_path / "tiny-resume.yaml"
    config.write_text(
        "gnn:\n"
        "  backbone: gine\n"
        "  num_layers: 1\n"
        "  hidden_dim: 16\n"
        "  dropout: 0.0\n"
        "  pooling: mean\n"
        "  readout_layers: 1\n"
        "  normalization: layer_norm\n"
        "  residual: true\n"
        "training:\n"
        "  optimizer: adamw\n"
        "  learning_rate: 0.001\n"
        "  weight_decay: 0.0\n"
        "  max_epochs: 1\n"
        "  early_stopping_patience: 1\n"
        "  batch_size: 4\n"
        "  primary_seed: 7\n"
        "  selection_metric: roc_auc\n"
        "  class_weighted_loss: true\n"
        "  weighted_sampler: false\n"
        "  gradient_clip_norm: 5.0\n"
        "runtime:\n"
        "  device: cpu\n"
        "  num_workers: 0\n",
        encoding="utf-8",
    )
    output = tmp_path / "checkpoint"
    state = tmp_path / "training-state"
    base_args = [
        "--config",
        str(config),
        "--dataset",
        "bace",
        "--data-dir",
        str(data),
        "--output-dir",
        str(output),
        "--profile",
        "smoke",
        "--device",
        "cpu",
        "--training-state-dir",
        str(state),
    ]
    assert train_main(base_args) == 0
    assert output.is_dir()
    assert (state / "checkpoint-000001.pt").is_file()
    completion = json.loads(
        (state / "training_complete.json").read_text(encoding="utf-8")
    )
    resume_contract = json.loads(
        (state / "training_contract.json").read_text(encoding="utf-8")
    )["contract"]
    assert resume_contract["configuration"]["merged_canonical_sha256"]
    assert resume_contract["configuration"]["config_files"] == [
        {"path": str(config.resolve()), "sha256": sha256_file(config)}
    ]
    assert resume_contract["configuration"]["dotlist_overrides"] == []
    assert resume_contract["source_identity"]["commit"]
    assert resume_contract["source_identity"]["tree"]
    assert resume_contract["source_identity"]["tracked_source_files"]
    assert resume_contract["runtime_identity"]["python_executable"]
    runtime = resume_contract["runtime_identity"]
    assert runtime["numpy"]["version"]
    assert "rdkit" in runtime
    assert "torch_geometric" in runtime
    assert "cudnn_version" in runtime
    assert "cuda_driver" in runtime
    assert runtime["environment_manifest_sha256"]
    assert "CUDA_VISIBLE_DEVICES" in runtime["environment_manifest"]
    card = json.loads((output / "model_card.json").read_text(encoding="utf-8"))
    assert completion["status"] == "PASS"
    assert completion["contract_sha256"] == card[
        "training_resume_contract_sha256"
    ]
    # A controller restart adopts only the fully verified terminal bundle and
    # does not execute another epoch or overwrite the immutable output.
    before = sha256_file(output / "model.pt")
    assert train_main([*base_args, "--resume-training"]) == 0
    assert sha256_file(output / "model.pt") == before


def test_published_bundle_crash_resumes_only_through_controller_exp_run_and_trainer(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise the real producer -> exp_run -> trainer completion-only path.

    The first trainer is a real CPU smoke run whose completion-manifest write is
    deliberately interrupted after immutable finalization.  The controller then
    issues its held-source receipt, and exp_run starts the real trainer CLI behind
    the durable child startup barrier to write only the missing completion.
    """

    data = tmp_path / "splits"
    data.mkdir()
    _write_split(data / "train.csv", "train")
    _write_split(data / "val.csv", "val")
    _write_split(data / "calibration.csv", "calibration")
    _write_split(data / "test.csv", "test")
    config = _write_tiny_resume_config(tmp_path / "tiny-adoption.yaml")
    output = tmp_path / "checkpoint"
    state = tmp_path / "training-state"
    controller_root = tmp_path / "controller"
    runtime_data = tmp_path / "runtime-data"
    runtime_data.mkdir()
    control_root = runtime_data / "control"
    cid = "tastemolnet_gine_v1_20260825T000000Z_c0ffee00"
    receipt_path = controller_root / PUBLISHED_ADOPTION_NAME
    train_script = (PROJECT_ROOT / "scripts/train_molecular_gnn.py").resolve(
        strict=True
    )
    base_args = [
        "--config",
        str(config),
        "--dataset",
        "bace",
        "--data-dir",
        str(data),
        "--output-dir",
        str(output),
        "--profile",
        "smoke",
        "--device",
        "cpu",
        "--training-state-dir",
        str(state),
    ]

    clean_environment = sanitized_environment()
    inherited_pythonpath = clean_environment.get("PYTHONPATH")
    trainer_pythonpath = str(PROJECT_ROOT) + (
        f":{inherited_pythonpath}" if inherited_pythonpath else ""
    )
    controller_environment = {
        "PATH": clean_environment.get("PATH", ""),
        "HOME": clean_environment.get("HOME", ""),
        "PYTHONPATH": trainer_pythonpath,
        "AUTODL_CONTROL_ROOT": str(control_root),
        "TASTEMOLNET_GNN_FULL_OUTPUT": str(output),
        "TASTEMOLNET_GNN_TRAINING_STATE_ROOT": str(state),
        "TASTEMOLNET_GINE_CONTROLLER_CID": cid,
        "TASTEMOLNET_GINE_CONTROLLER_ROOT": str(controller_root),
        "TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT": str(receipt_path),
    }
    first_environment = dict(clean_environment)
    first_environment.update(controller_environment)

    crash_wrapper = tmp_path / "crash_after_publish.py"
    crash_wrapper.write_text(
        "from scripts.train_molecular_gnn import main\n"
        "from src.train.molecular_gnn_resume import MolecularGNNResumeStore\n"
        "def crash_after_publish(self, *, output_dir, output_identity):\n"
        "    raise RuntimeError('forced crash after finalization publication')\n"
        "MolecularGNNResumeStore.mark_complete = crash_after_publish\n"
        "raise SystemExit(main())\n",
        encoding="utf-8",
    )
    crashed = subprocess.run(
        [sys.executable, str(crash_wrapper), *base_args],
        cwd=PROJECT_ROOT,
        env=first_environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert crashed.returncode != 0
    assert "forced crash after finalization publication" in crashed.stderr
    verify_checkpoint_bundle(output)
    assert not (state / "training_complete.json").exists()

    controller_spec = TasteGINEControllerSpec(
        cid=cid,
        controller_root=controller_root,
        project_root=PROJECT_ROOT,
        output_dir=output,
        training_state_root=state,
        worker_argv=(sys.executable, str(train_script)),
        source_identity={},
        environment_authority=controller_environment,
        config_files=(),
        poll_seconds=0.01,
        terminal_stability_seconds=0.0,
        resource_wait_deadline_seconds=60,
    )
    monkeypatch.setattr(
        TasteGINEPersistentController,
        "_validate_spec_sources",
        lambda self: None,
    )
    with TasteGINEPersistentController(
        controller_spec, resume=False
    ) as controller:
        real_write_state = controller._write_state
        crashed_once = False

        def crash_after_receipt(phase, **fields):
            nonlocal crashed_once
            if phase == "PUBLISHED_OUTPUT_ADOPTION_PENDING" and not crashed_once:
                crashed_once = True
                raise RuntimeError("forced crash after adoption receipt publication")
            return real_write_state(phase, **fields)

        with monkeypatch.context() as patch:
            patch.setattr(controller, "_write_state", crash_after_receipt)
            with pytest.raises(RuntimeError, match="after adoption receipt"):
                controller._issue_published_output_adoption(
                    attempt=0, launch_index=1
                )
        receipt_inode = os.lstat(receipt_path).st_ino
        receipt = controller._issue_published_output_adoption(
            attempt=0, launch_index=1
        )
        assert receipt == receipt_path
        assert os.lstat(receipt_path).st_ino == receipt_inode
        assert sha256_file(receipt) == sha256_file(receipt_path)
        # A completion-only worker can itself be lost.  Reuse the immutable
        # receipt under the same science attempt while advancing only the
        # bounded launch generation; never replace or reissue the receipt.
        assert controller._issue_published_output_adoption(
            attempt=0, launch_index=2
        ) == receipt
        assert os.lstat(receipt_path).st_ino == receipt_inode

        original_evidence = controller_module._published_output_adoption_evidence

        def evidence_with_controller_heartbeat(*, output_dir, training_state_root):
            evidence = original_evidence(
                output_dir=output_dir,
                training_state_root=training_state_root,
            )
            controller._write_state(
                "RUNNING",
                attempt=0,
                launch_index=2,
                retries_used=0,
                heartbeat="validator-race-regression",
            )
            return evidence

        # A legitimate controller heartbeat may atomically replace state.json
        # while exp_run performs its read-only scan.  Stable authority fields,
        # not byte equality of the heartbeat payload, define the closure.
        with monkeypatch.context() as patch:
            patch.setattr(
                controller_module,
                "_published_output_adoption_evidence",
                evidence_with_controller_heartbeat,
            )
            controller_module.validate_tastemolnet_published_output_adoption_readonly(
                receipt,
                expected_output_dir=output,
                expected_training_state_root=state,
            )

        layout = build_runtime_layout(
            project_root=PROJECT_ROOT,
            data_root=runtime_data,
            control_root=control_root,
        ).ensure()
        run_id = "completion-adoption-vertical"
        run_root = layout.runs_root / run_id
        run_root.mkdir(parents=True)
        log_path = layout.logs_dir / f"{run_id}.log"
        trainer_command = [
            sys.executable,
            str(train_script),
            *base_args,
            "--resume-training",
            "--resume-published-output-receipt",
            str(receipt),
        ]
        exp_environment = {
            key: value
            for key, value in controller_environment.items()
            if key != "PYTHONPATH"
        }
        launch_spec = {
            "schema_version": EXP_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": "2026-08-25T00:00:00Z",
            "project_root": str(PROJECT_ROOT),
            "data_root": str(runtime_data),
            "control_root": str(control_root),
            "python_executable": str(Path(sys.executable).resolve(strict=True)),
            "dataset": "completion-adoption-fixture",
            "stage": "producer-exp-run-trainer",
            "command": trainer_command,
            "environment": exp_environment,
            "gpu_index": None,
            "gpu_uuid": None,
            "gpu_lock_mode": GPU_LOCK_EXCLUSIVE,
            "gpu_memory_reservation_mb": 0,
            "gpu_shared_workload_class": None,
            "gpu_colocation_gate": None,
            "gpu_colocation_gate_sha256": None,
            "min_free_memory_mb": 0,
            "idle_util_threshold": 100,
            "max_gpus": 1,
            "gpu_hard_limit": 4,
            "git_commit": None,
            "config_files": [str(config)],
            "config_hash": sha256_file(config),
            "input_manifest": None,
            "input_hash": None,
            "expected_output": str(output),
            "resume_published_output_receipt": str(receipt),
            "resume_published_output_receipt_sha256": sha256_file(receipt),
            "required_output_files": [
                "model.pt",
                "model_card.json",
                "sha256sums.txt",
            ],
            "required_output_any": [],
            "required_absolute_output_files": [],
            "required_log_marker": "[MOLECULAR_GNN_TRAIN_OK]",
            "log_path": str(log_path),
            "launcher": "foreground",
            "tmux_session": None,
            "heavy": False,
        }
        spec_path = run_root / "launch_spec.json"
        spec_path.write_text(
            json.dumps(launch_spec, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        assert run_worker(spec_path) == 0

    completion = json.loads(
        (state / "training_complete.json").read_text(encoding="utf-8")
    )
    authority = json.loads(
        (run_root / TRAINER_CHILD_AUTHORITY_NAME).read_text(encoding="utf-8")
    )
    assert completion["status"] == "PASS"
    assert authority["schema_version"] == TRAINER_CHILD_AUTHORITY_SCHEMA
    assert authority["status"] == "RELEASE_AUTHORIZED"
    assert authority["trainer_command"] == trainer_command
    assert authority["child_registered"]["pid"] > 0
    assert authority["parent_exp_run"]["pid"] == os.getpid()
    assert "[MOLECULAR_GNN_TRAIN_OK]" in log_path.read_text(encoding="utf-8")
