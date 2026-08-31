from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.tastemolnet_gcf_full_resume import TasteGCFFullResumeError
from src.baselines.tastemolnet_gcf_replay_canary import (
    THRESHOLD_AUTHORITY_MARKER,
    THRESHOLD_AUTHORITY_SCHEMA,
    THRESHOLD_INPUT_SCHEMA,
    THRESHOLD_RECEIPT_SCHEMA,
    THRESHOLD_SELECTOR_MARKER,
    configure_exact_cuda_replay,
    load_threshold_authority,
    require_real_a800,
    run_replay_canary_phase,
    _BoundedNeuroSEDCoverage,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _threshold_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    roots = {
        name: tmp_path / name
        for name in ("neurosed", "t3", "official", "cache", "t4", "molclr")
    }
    for root in roots.values():
        root.mkdir()
    molclr_checkpoint = roots["molclr"] / "model.pth"
    molclr_checkpoint.write_bytes(b"model")
    input_authority: dict[str, object] = {
        "schema_version": THRESHOLD_INPUT_SCHEMA,
        "t3_root": str(roots["t3"]),
        "t3_gate_sha256": "1" * 64,
        "t3_verification_sha256": "2" * 64,
        "t3_checkpoint_sha256": "3" * 64,
        "t3_temperature_scaling_sha256": "4" * 64,
        "graph_cache_root": str(roots["cache"]),
        "graph_cache_manifest_sha256": "5" * 64,
        "calibration_cache_sha256": "6" * 64,
        "t4_root": str(roots["t4"]),
        "t4_verification_sha256": "7" * 64,
        "t4_oracle_smoke_sha256": "8" * 64,
        "t4_terminal_round": 5,
        "t4_selected_count": 64,
        "t4_valid_deletion_count": 733,
        "t4_strict_flip_count": 38,
        "managed_neurosed_root": str(roots["neurosed"]),
        "neurosed_checkpoint_sha256": "9" * 64,
        "neurosed_feature_schema_sha256": "a" * 64,
        "official_gcf_root": str(roots["official"]),
        "official_gcf_inventory_sha256": "b" * 64,
        "molclr_root": str(roots["molclr"]),
        "molclr_checkpoint": str(molclr_checkpoint),
        "molclr_checkpoint_sha256": "c" * 64,
        "opened_payload_splits": ["calibration"],
        "train_payload_loaded": False,
        "validation_payload_loaded": False,
        "test_payload_loaded": False,
    }
    input_digest = _digest(_canonical(input_authority))
    input_authority["input_authority_sha256"] = input_digest
    pair_digest = "d" * 64
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    authority = {
        "schema_version": THRESHOLD_AUTHORITY_SCHEMA,
        "status": "PASS",
        "marker": THRESHOLD_AUTHORITY_MARKER,
        "dataset": "tastemolnet",
        "method_consumer": "GCFExplainer",
        "distance_line": "official_normged_generated_query_to_original_target_v1",
        "inference_direction": "generated_query_to_original_target",
        "distance_normalization": "divide_by_sum_graph_element_counts",
        "selection_split": "calibration",
        "threshold_source_split": "calibration",
        "threshold_source": "tastemolnet_t4_strict_flip_neurosed_q30_v1",
        "objective": (
            "method_independent_empirical_distance_quantiles_over_all_finite_"
            "t4_calibration_strict_flip_residual_to_parent_pairs"
        ),
        "quantile_method": "linear",
        "dtype": "float64",
        "requested_quantiles": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
        "raw_quantile_thresholds": [
            {"quantile": q, "threshold": value}
            for q, value in zip(
                [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
                thresholds,
                strict=True,
            )
        ],
        "theta_star_quantile": 0.30,
        "neurosed_distance_threshold": 0.04,
        "finite_strict_flip_distance_count": 38,
        "tie_break": (
            "numpy_float64_linear_interpolation; equal_adjacent_order_"
            "statistics_retain_the_identical_smaller_threshold"
        ),
        "shared_across_t7_training_and_evaluation": True,
        "threshold_fitted_on_test": False,
        "selection_used_test": False,
        "test_used_for_selection": False,
        "train_payload_loaded": False,
        "validation_payload_loaded": False,
        "test_payload_loaded": False,
        "cf_mode": "strict_flip",
        "pair_inventory_sha256": pair_digest,
        "input_authority_sha256": input_digest,
        "selected_at": "2026-09-01T01:02:03+00:00",
    }
    output = tmp_path / "authority"
    output.mkdir()
    _write(output / "input_authority.json", input_authority)
    _write(output / "t7_neurosed_threshold_authority.json", authority)
    (output / "calibration_distance_rows.jsonl").write_bytes(b'{"pair_id":"p"}\n')
    _write(output / "tastemolnet.json", {"schema_version": "wnode"})
    authority_sha = _digest((output / "t7_neurosed_threshold_authority.json").read_bytes())
    receipt = {
        "schema_version": THRESHOLD_RECEIPT_SCHEMA,
        "status": "PASS",
        "marker": THRESHOLD_SELECTOR_MARKER,
        "dataset": "tastemolnet",
        "selection_split": "calibration",
        "opened_payload_splits": ["calibration"],
        "train_payload_loaded": False,
        "validation_payload_loaded": False,
        "test_payload_loaded": False,
        "test_used_for_selection": False,
        "strict_flip_pair_count": 38,
        "pair_inventory_sha256": pair_digest,
        "neurosed_authority_sha256": authority_sha,
        "wnode_contract_sha256": _digest((output / "tastemolnet.json").read_bytes()),
        "distance_rows_sha256": _digest(
            (output / "calibration_distance_rows.jsonl").read_bytes()
        ),
        "input_authority_sha256": input_digest,
        "wnode_runtime_stats": {},
        "paper_cell_published": False,
        "selected_at": "2026-09-01T01:02:04+00:00",
    }
    _write(output / "selection_receipt.json", receipt)
    names = (
        "calibration_distance_rows.jsonl",
        "input_authority.json",
        "selection_receipt.json",
        "t7_neurosed_threshold_authority.json",
        "tastemolnet.json",
    )
    (output / "sha256sums.txt").write_text(
        "".join(f"{_digest((output / name).read_bytes())}  {name}\n" for name in names),
        encoding="ascii",
    )
    (output / "PASS").write_text(THRESHOLD_SELECTOR_MARKER + "\n", encoding="ascii")
    expected = {
        "expected_neurosed_checkpoint_sha256": "9" * 64,
        "expected_neurosed_feature_schema_sha256": "a" * 64,
        "expected_t3_checkpoint_id": "3" * 64,
        "expected_t3_temperature_sha256": "4" * 64,
        "expected_t3_gate_sha256": "1" * 64,
        "expected_t3_verification_sha256": "2" * 64,
        "expected_official_inventory_sha256": "b" * 64,
        "expected_managed_neurosed_root": roots["neurosed"],
        "expected_t3_root": roots["t3"],
        "expected_official_root": roots["official"],
    }
    return output / "t7_neurosed_threshold_authority.json", expected


def test_threshold_loader_reopens_real_selector_contract(tmp_path: Path) -> None:
    path, expected = _threshold_fixture(tmp_path)
    authority, digest = load_threshold_authority(path, **expected)
    assert authority["neurosed_distance_threshold"] == 0.04
    assert digest == _digest(path.read_bytes())
    assert authority["test_payload_loaded"] is False


def test_threshold_loader_rejects_test_selection(tmp_path: Path) -> None:
    path, expected = _threshold_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["test_payload_loaded"] = True
    _write(path, payload)
    with pytest.raises(TasteGCFFullResumeError):
        load_threshold_authority(path, **expected)


def test_require_real_a800_binds_one_physical_uuid(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(
        "src.baselines.tastemolnet_gcf_replay_canary.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="3, GPU-abcd, NVIDIA A800 80GB PCIe, 81920\n",
            stderr="",
        ),
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda index: SimpleNamespace(
            name="NVIDIA A800 80GB PCIe", total_memory=80 * 1024**3
        ),
    )
    torch = SimpleNamespace(
        cuda=cuda,
        __version__="2.1.0",
        version=SimpleNamespace(cuda="11.8"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 8900)),
    )
    result = require_real_a800(gpu_uuid="GPU-abcd", torch=torch)
    assert result["physical_index"] == 3
    assert result["gpu_uuid"] == "GPU-abcd"
    assert result["cuda_used"] is True


def test_exact_cuda_controls_are_pinned_before_science(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    state = {
        "algorithms": False,
        "debug": 0,
        "precision": "highest",
    }
    matmul = SimpleNamespace(
        allow_tf32=True,
        allow_fp16_reduced_precision_reduction=True,
        allow_bf16_reduced_precision_reduction=True,
    )
    cudnn = SimpleNamespace(deterministic=False, benchmark=True, allow_tf32=True)
    torch = SimpleNamespace(
        use_deterministic_algorithms=lambda enabled, warn_only: state.update(
            algorithms=enabled, warn_only=warn_only
        ),
        are_deterministic_algorithms_enabled=lambda: state["algorithms"],
        set_deterministic_debug_mode=lambda value: state.update(
            debug=2 if value == "error" else -1
        ),
        get_deterministic_debug_mode=lambda: state["debug"],
        set_float32_matmul_precision=lambda value: state.update(precision=value),
        get_float32_matmul_precision=lambda: state["precision"],
        backends=SimpleNamespace(
            cudnn=cudnn, cuda=SimpleNamespace(matmul=matmul)
        ),
    )
    result = configure_exact_cuda_replay(torch=torch)
    assert result["deterministic_algorithms"] is True
    assert result["deterministic_debug_mode"] == 2
    assert result["cudnn_benchmark"] is False
    assert result["cuda_matmul_allow_tf32"] is False
    assert all(
        value is False
        for value in result["optional_reduced_precision_reductions"].values()
    )


def test_neurosed_retry_does_not_swallow_non_oom_runtime_error() -> None:
    import torch

    importance = SimpleNamespace(
        torch=torch,
        util=SimpleNamespace(
            graph_element_counts=lambda dataset: torch.ones(len(dataset))
        ),
    )
    runtime = _BoundedNeuroSEDCoverage(importance)

    class Model:
        calls = 0

        def predict_outer_with_queries(self, dataset, *, batch_size):
            self.calls += 1
            raise RuntimeError("deterministic implementation is unavailable")

    model = Model()
    with pytest.raises(RuntimeError, match="deterministic implementation"):
        runtime(model, [1, 2, 3, 4], torch.ones(2), 0.5)
    assert model.calls == 1
    assert runtime.calls == []


def test_neurosed_cuda_oom_retry_is_bounded_and_checkpointed() -> None:
    import torch

    importance = SimpleNamespace(
        torch=torch,
        util=SimpleNamespace(
            graph_element_counts=lambda dataset: torch.ones(len(dataset))
        ),
    )
    runtime = _BoundedNeuroSEDCoverage(importance)

    class Model:
        attempted: list[int] = []

        def predict_outer_with_queries(self, dataset, *, batch_size):
            self.attempted.append(batch_size)
            if batch_size > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return torch.ones((len(dataset), 2))

    model = Model()
    selected = runtime(model, [1, 2, 3, 4], torch.ones(2), 0.5)
    assert model.attempted == [4, 2, 1]
    assert selected.shape == (4, 2)
    assert runtime.calls[0]["cuda_oom_retry_count"] == 2
    restored = _BoundedNeuroSEDCoverage(importance)
    restored.restore_checkpoint_state(runtime.checkpoint_state())
    assert restored.checkpoint_state() == runtime.checkpoint_state()

    class PersistentOOM:
        attempted: list[int] = []

        def predict_outer_with_queries(self, dataset, *, batch_size):
            self.attempted.append(batch_size)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    failed = PersistentOOM()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        runtime(failed, [1, 2, 3, 4], torch.ones(2), 0.5)
    assert failed.attempted == [4, 2, 1]


def test_phase_rejects_invalid_mode_before_cuda() -> None:
    with pytest.raises(TasteGCFFullResumeError, match="mode"):
        run_replay_canary_phase(
            mode="production",
            output_root="/tmp/not-used",
            observation_path=None,
            checkpoint_manifest=None,
            attempt_id="not-used",
            generation_token=SHA,
            gpu_uuid="GPU-not-used",
            managed_neurosed_root="/tmp/not-used",
            t3_root="/tmp/not-used",
            official_root="/tmp/not-used",
            threshold_authority_path="/tmp/not-used",
        )


def _load_worker_module():
    script = REPO_ROOT / "scripts/run_tastemolnet_gcf_replay_canary_worker.py"
    spec = importlib.util.spec_from_file_location("t12_canary_worker", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_cli_forwards_one_terminal_phase(tmp_path: Path, monkeypatch) -> None:
    worker = _load_worker_module()
    observed: dict[str, object] = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "OBSERVATION_COMMITTED"}

    monkeypatch.setattr(worker, "run_replay_canary_phase", fake_run)
    root = tmp_path / "fresh"
    observation = tmp_path / "observation.json"
    args = [
        "--config",
        str(REPO_ROOT / "configs/hpc.yaml"),
        "--set",
        "inference.fallback_to_heuristic=false",
        "--mode",
        "uninterrupted",
        "--output-root",
        str(root),
        "--observation",
        str(observation),
        "--attempt-id",
        "9ec18261-d8bc-4d83-8c69-8e8b57ce4e4a",
        "--generation-token",
        SHA,
        "--gpu-uuid",
        "GPU-abcd",
        "--managed-neurosed-root",
        str(tmp_path),
        "--t3-root",
        str(tmp_path),
        "--official-root",
        str(tmp_path),
        "--neurosed-threshold-authority",
        str(tmp_path / "threshold.json"),
    ]
    assert worker.main(args) == 0
    assert observed["mode"] == "uninterrupted"
    assert observed["observation_path"] == observation
    assert observed["checkpoint_manifest"] is None


def test_worker_cli_and_slurm_are_fail_closed() -> None:
    worker = (
        REPO_ROOT / "scripts/run_tastemolnet_gcf_replay_canary_worker.py"
    ).read_text(encoding="utf-8")
    slurm = (
        REPO_ROOT / "scripts/slurm/run_tastemolnet_gcf_replay_canary_worker.sh"
    ).read_text(encoding="utf-8")
    assert "--neurosed-threshold-authority" in worker
    assert "inference.fallback_to_heuristic=false" in worker
    assert "#SBATCH --partition=A800" in slurm
    assert "#SBATCH --gres=gpu:a800:1" in slurm
    assert "#SBATCH --output=logs/%j.out" in slurm
    assert "#SBATCH --error=logs/%j.err" in slurm
    assert "conda activate smiles_pip118" in slurm
    assert "cd /share/home/u20526/czx/counterfactual-subgraph" in slurm
    assert "export PYTHONPATH=$PWD" in slurm
    assert "--config configs/hpc.yaml" in slurm
    assert "--set inference.fallback_to_heuristic=false" in slurm
    assert "TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY" in slurm
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in slurm
    assert "export PYTHONHASHSEED=7" in slurm
    assert "[TASTE_GCF_PASS]" not in worker + slurm


def test_sequence_uses_one_gpu_allocation_and_three_science_processes() -> None:
    sequence = (
        REPO_ROOT
        / "scripts/slurm/run_tastemolnet_gcf_replay_canary_sequence.sh"
    ).read_text(encoding="utf-8")
    assert sequence.count(
        "python scripts/run_tastemolnet_gcf_replay_canary_worker.py"
    ) == 3
    assert "--mode uninterrupted" in sequence
    assert "--mode checkpoint" in sequence
    assert "--mode resume" in sequence
    assert "--checkpoint-prefix-receipt" in sequence
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in sequence
    assert "export PYTHONHASHSEED=7" in sequence
    assert "#SBATCH --gres=gpu:a800:1" in sequence
    assert "[TASTE_GCF_PASS]" not in sequence
