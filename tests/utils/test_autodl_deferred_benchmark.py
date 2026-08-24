from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils.autodl_deferred_benchmark import (
    DeferredObservation,
    REQUIRED_THREAD_ENVIRONMENT,
    _validate_input_manifest,
    classify_readiness,
    parse_args,
    validate_benchmark_result,
)


def _observation(**updates: object) -> DeferredObservation:
    payload = {
        "launch_spec_valid": True,
        "launch_spec_reason": None,
        "pair_state": "PASS",
        "pair_worker_identity": "GONE",
        "uuid_lock_available": True,
        "project_slot_available": True,
        "gpu_identity_valid": True,
        "gpu_process_count": 0,
        "gpu_free_memory_mb": 80000,
        "gpu_utilization_percent": 0,
        "minimum_free_memory_mb": 16000,
        "maximum_utilization_percent": 10,
        "benchmark_output_exists": False,
        "benchmark_run_state_exists": False,
        "immutable_checkout_valid": True,
        "input_manifest_valid": True,
    }
    payload.update(updates)
    return DeferredObservation(**payload)


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        (
            {"benchmark_output_exists": True},
            "PARTIAL_OR_PREEXISTING_BENCHMARK_OUTPUT_ROOT",
        ),
        (
            {"benchmark_run_state_exists": True},
            "PARTIAL_OR_PREEXISTING_BENCHMARK_RUN_STATE",
        ),
        (
            {
                "launch_spec_valid": False,
                "launch_spec_reason": "PAIR_LAUNCH_SPEC_SHA256_MISMATCH",
            },
            "PAIR_LAUNCH_SPEC_SHA256_MISMATCH",
        ),
        (
            {"immutable_checkout_valid": False},
            "IMMUTABLE_EXECUTION_CHECKOUT_INVALID",
        ),
        ({"input_manifest_valid": False}, "BENCHMARK_INPUT_MANIFEST_INVALID"),
        (
            {"pair_state": "RUNNING", "pair_worker_identity": "PID_REUSED"},
            "PAIR_WORKER_PID_REUSED_BEFORE_REGISTRY_TERMINAL",
        ),
        ({"gpu_identity_valid": False}, "GPU_INDEX_UUID_IDENTITY_MISMATCH"),
    ],
)
def test_deferred_controller_blocks_provenance_or_partial_targets(
    updates: dict[str, object], reason: str
) -> None:
    assert classify_readiness(_observation(**updates)) == ("BLOCKED", reason)


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        (
            {"pair_state": "RUNNING", "pair_worker_identity": "MATCHING_ALIVE"},
            "PAIR_REGISTRY_RUNNING",
        ),
        (
            {"pair_state": "PASS", "pair_worker_identity": "MATCHING_ALIVE"},
            "PAIR_WORKER_STILL_ALIVE_AFTER_TERMINAL",
        ),
        ({"uuid_lock_available": False}, "GPU_UUID_EXCLUSIVE_LOCK_STILL_HELD"),
        ({"project_slot_available": False}, "NO_PROJECT_GPU_SLOT_AVAILABLE"),
        ({"gpu_process_count": 1}, "GPU_UUID_HAS_COMPUTE_PROCESS"),
        ({"gpu_free_memory_mb": 1000}, "GPU_FREE_MEMORY_BELOW_GATE"),
        ({"gpu_utilization_percent": 11}, "GPU_UTILIZATION_ABOVE_GATE"),
    ],
)
def test_deferred_controller_keeps_resource_contention_waiting(
    updates: dict[str, object], reason: str
) -> None:
    assert classify_readiness(_observation(**updates)) == (
        "WAITING_RESOURCE",
        reason,
    )


def test_deferred_controller_does_not_bind_transient_science_pid() -> None:
    assert classify_readiness(
        _observation(pair_state="FAILED", pair_worker_identity="PID_REUSED")
    ) == ("READY", "ALL_DEPENDENCY_AND_RESOURCE_GATES_READY")
    assert classify_readiness(_observation()) == (
        "READY",
        "ALL_DEPENDENCY_AND_RESOURCE_GATES_READY",
    )


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--config",
        str(tmp_path / "config.yaml"),
        "--set",
        "inference.fallback_to_heuristic=false",
        "--controller-id",
        "bace-gine-deferred-test",
        "--state-root",
        str(tmp_path / "controller"),
        "--project-root",
        str(tmp_path / "project"),
        "--execution-commit",
        "a" * 40,
        "--python",
        str(tmp_path / "python"),
        "--pair-run-id",
        "pair-run",
        "--pair-run-state-root",
        str(tmp_path / "pair"),
        "--pair-launch-spec-sha256",
        "b" * 64,
        "--pair-worker-pid",
        "456429",
        "--pair-worker-start-ticks",
        "698595621",
        "--gpu-index",
        "2",
        "--gpu-uuid",
        "GPU-uuid",
        "--lock-root",
        str(tmp_path / "locks"),
        "--registry-path",
        str(tmp_path / "runs.jsonl"),
        "--benchmark-run-id",
        "benchmark-run",
        "--benchmark-run-state-root",
        str(tmp_path / "benchmark-state"),
        "--benchmark-output-root",
        str(tmp_path / "benchmark-output"),
        "--benchmark-input-manifest",
        str(tmp_path / "input.json"),
        "--benchmark-input-manifest-sha256",
        "c" * 64,
        "--benchmark-log-path",
        str(tmp_path / "benchmark.log"),
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--checkpoint-dir",
        str(tmp_path / "checkpoint"),
    ]


def test_deferred_controller_cli_enforces_sixty_second_stability(
    tmp_path: Path,
) -> None:
    args = parse_args(_required_args(tmp_path))
    assert args.poll_seconds == 60.0
    assert args.stable_ready_seconds == 60.0
    assert args.pair_worker_pid == 456429
    assert args.pair_worker_start_ticks == 698595621
    with pytest.raises(SystemExit):
        parse_args(_required_args(tmp_path) + ["--stable-ready-seconds", "59"])


def test_deferred_controller_cli_and_paired_slurm_are_synchronized() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cli = (
        project_root
        / "scripts/autodl/run_deferred_bace_gnn_inference_benchmark.py"
    ).read_text(encoding="utf-8")
    assert "from src.utils.autodl_deferred_benchmark import main" in cli
    source = (project_root / "src/utils/autodl_deferred_benchmark.py").read_text(
        encoding="utf-8"
    )
    assert "time.sleep(args.poll_seconds)" in source
    assert "pass_fds=lock_fds" in source
    assert "456442" not in source
    assert "PARTIAL_OR_PREEXISTING_BENCHMARK_OUTPUT_ROOT" in source

    slurm = (
        project_root
        / "scripts/slurm/run_deferred_bace_gnn_inference_benchmark.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export TOKENIZERS_PARALLELISM=false",
    ):
        assert required in slurm


def test_validate_benchmark_result_requires_pass_last_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    log = tmp_path / "benchmark.log"
    log.write_text("[BACE_GNN_INFERENCE_MATRIX_BENCHMARK_PASS]\n", encoding="utf-8")
    result = {
        "status": "PASS",
        "batch_sizes": [1, 8, 32, 128, 512],
        "all_argmax_and_allclose_checks_pass": True,
        "all_calibrated_probability_checks_pass": True,
        "cpu_raw_byte_repeat_exact_all_batches": True,
        "authorizes_vrrw_replacement": False,
        "cohort": {"test_loaded": False},
        "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        "best_end_to_end": {
            "overall": {
                "device": "gpu",
                "batch_size": 512,
                "median_rows_per_second": 1000.0,
            }
        },
    }
    result_path = root / "bace_gnn_inference_benchmark.json"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (root / "_BENCHMARK_COMPLETE.json").write_text(
        json.dumps({"status": "PASS", "result_sha256": digest}) + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        benchmark_output_root=root,
        benchmark_log_path=log,
    )
    assert validate_benchmark_result(args, 0) == []
    result["batch_sizes"] = [1, 8]
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    assert "batch_matrix_mismatch" in validate_benchmark_result(args, 0)


def test_input_manifest_binds_run_provenance_and_thread_environment(
    tmp_path: Path,
) -> None:
    args = parse_args(_required_args(tmp_path))
    payload = {
        "schema_version": "bace_gnn_inference_deferred_input_v1",
        "controller_id": args.controller_id,
        "benchmark_run_id": args.benchmark_run_id,
        "execution_commit": args.execution_commit,
        "project_root": str(args.project_root),
        "benchmark_output_root": str(args.benchmark_output_root),
        "benchmark_run_state_root": str(args.benchmark_run_state_root),
        "dataset_dir": str(args.dataset_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "pair_run_id": args.pair_run_id,
        "pair_launch_spec_sha256": args.pair_launch_spec_sha256,
        "pair_worker_pid": args.pair_worker_pid,
        "pair_worker_start_ticks": args.pair_worker_start_ticks,
        "batch_sizes": [1, 8, 32, 128, 512],
        "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
    }
    args.benchmark_input_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_input_manifest.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.benchmark_input_manifest_sha256 = hashlib.sha256(
        args.benchmark_input_manifest.read_bytes()
    ).hexdigest()
    assert _validate_input_manifest(args) is True
    payload["thread_environment"]["OPENBLAS_NUM_THREADS"] = "8"
    args.benchmark_input_manifest.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.benchmark_input_manifest_sha256 = hashlib.sha256(
        args.benchmark_input_manifest.read_bytes()
    ).hexdigest()
    assert _validate_input_manifest(args) is False
