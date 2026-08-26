from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTODL = PROJECT_ROOT / "scripts" / "autodl"


def test_tastemolnet_full_script_has_scoped_compute_and_no_redistribution_guards() -> None:
    script = (AUTODL / "run_tastemolnet_gnn_full.sh").read_text(encoding="utf-8")
    assert '[[ "$RUN_TASTEMOLNET" == "1" ]]' in script
    assert '[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-}" == "1" ]]' in script
    assert '[[ "${TASTE_PAPER_RESULTS_ALLOWED:-}" == "1" ]]' in script
    assert '[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-}" == "0" ]]' in script
    assert "NOT_EXPLICITLY_STATED" in script
    assert "--heavy" in script
    assert "--training-state-dir" in script
    assert "--resume-training" in script
    assert "--max-gpus 4" in script
    assert "--gpu-hard-limit 4" in script
    assert "--foreground" in script
    assert "paths_overlap" in script
    assert "[TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW]" not in script
    inventory = (AUTODL / "gpu_inventory.py").read_text(encoding="utf-8")
    assert '"--gpu-hard-limit"' in inventory
    assert "validate_max_gpus(args.max_gpus, hard_limit=args.gpu_hard_limit)" in inventory
    assert "hard_limit=args.gpu_hard_limit" in inventory


def test_tastemolnet_full_requires_direct_scoped_compute_authority_without_launch(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    environment = {
        **os.environ,
        "AUTODL_PYTHON": sys.executable,
        "AUTODL_DATA_ROOT": str(data_root),
        "AUTODL_RUNTIME_ROOT": str(data_root / "runtime"),
        "RUN_TASTEMOLNET": "0",
    }
    environment.pop("TASTEMOLNET_UPSTREAM_COMMIT", None)
    result = subprocess.run(
        ["bash", str(AUTODL / "run_tastemolnet_gnn_full.sh")],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert result.stdout == ""
    assert result.stderr.strip() == "RUN_TASTEMOLNET must be 1"


def test_gpu_inventory_four_requires_explicit_hard_limit_real_cli(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "AUTODL_DATA_ROOT": str(tmp_path / "data"),
        "AUTODL_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "AUTODL_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "AUTODL_CONTROL_ROOT": str(tmp_path / "control"),
    }
    base = [
        sys.executable,
        str(AUTODL / "gpu_inventory.py"),
        "--project-root",
        str(PROJECT_ROOT),
        "--data-root",
        str(tmp_path / "data"),
        "--max-gpus",
        "4",
        "--once",
    ]
    rejected = subprocess.run(
        base,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "must be in [1, 2], got 4" in rejected.stderr

    explicit = subprocess.run(
        [*base, "--gpu-hard-limit", "4"],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    # This host may have no nvidia-smi, but the explicit four-GPU hard-limit
    # gate must have been accepted before physical inventory is attempted.
    assert "must be in [1, 2], got 4" not in explicit.stderr


def test_bace_scripts_use_stage_order_and_task_specific_gnn() -> None:
    smoke = (AUTODL / "run_bace_gnn_smoke.sh").read_text(encoding="utf-8")
    full = (AUTODL / "run_bace_gnn_full.sh").read_text(encoding="utf-8")
    assert "B2_GNN_SMOKE" in smoke
    assert "B3_GNN_FULL" in full
    for script in (smoke, full):
        assert "--dataset bace" in script
        assert "train_molecular_gnn.py" in script
        assert "configs/datasets/bace_gnn.yaml" in script
        assert "test_evaluation_status.json" in script
        assert "test_predictions.csv" not in script
        assert "random_forest" not in script.lower()
        assert "sbatch" not in script.lower()


def test_b4_b5_launchers_use_frozen_predecessors_and_scientific_splits() -> None:
    calibration = (AUTODL / "run_bace_gnn_calibration.sh").read_text(
        encoding="utf-8"
    )
    oracle = (AUTODL / "run_bace_gnn_oracle_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "B3_GNN_FULL" in calibration
    assert "B4_GNN_CALIBRATED" in calibration
    assert "val.csv" in calibration
    assert "[BACE_GNN_CALIBRATION_PASS]" in calibration
    assert "split_manifest.json" in calibration
    assert "B4_GNN_CALIBRATED" in oracle
    assert "B5_ORACLE_SMOKE" in oracle
    assert "calibration.csv" in oracle
    assert "--source-count 16" in oracle
    assert "val.csv" not in oracle
    assert "test.csv" not in oracle
    assert "[BACE_GNN_ORACLE_SMOKE_PASS]" in oracle
    assert "deletion_records.jsonl" in oracle


def test_waiting_launcher_never_starts_tastemolnet() -> None:
    script = (AUTODL / "launch_bace_when_idle.sh").read_text(encoding="utf-8")
    assert "export RUN_TASTEMOLNET=0" in script
    assert "run_bace_gnn_smoke.sh" in script
    assert "run_bace_gnn_full.sh" in script
    assert "run_bace_gnn_calibration.sh" in script
    assert "run_bace_gnn_oracle_smoke.sh" in script
    assert "run_tastemolnet" not in script


def test_every_python_entrypoint_has_paired_slurm_status_wrapper() -> None:
    for name in (
        "detect_runtime",
        "gpu_inventory",
        "gpu_lock",
        "exp_run",
        "status",
        "bace_gnn_stage",
        "bace_frozen_gnn_route",
        "run_four_gpu_recovery_controller",
        "status_four_gpu_recovery",
        "run_tastemolnet_gine_controller",
    ):
        wrapper = PROJECT_ROOT / "scripts" / "slurm" / f"{name}.sh"
        assert wrapper.is_file(), name
        text = wrapper.read_text(encoding="utf-8")
        assert "#SBATCH --partition=A800" in text
        assert "#SBATCH --gres=gpu:a800:1" in text
        assert "--config configs/hpc.yaml" in text


def test_autodl_shells_pin_one_explicit_python_interpreter() -> None:
    common = (AUTODL / "common.sh").read_text(encoding="utf-8")
    assert "/root/miniconda3/envs/smiles_pip118/bin/python" in common
    assert "export PYTHONDONTWRITEBYTECODE=1" in common
    assert '[[ "$AUTODL_PYTHON" != /* || ! -x "$AUTODL_PYTHON" ]]' in common
    bare_python = re.compile(r"(^|[;&|()]|\s)python(?:\s|$)")
    for script in AUTODL.glob("*.sh"):
        text = script.read_text(encoding="utf-8")
        executable_lines = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        assert bare_python.search(executable_lines) is None, script.name
    for name in (
        "run_bace_gnn_smoke.sh",
        "run_bace_gnn_full.sh",
        "run_bace_gnn_calibration.sh",
        "run_bace_gnn_oracle_smoke.sh",
        "run_bace_frozen_gnn_stage.sh",
    ):
        text = (AUTODL / name).read_text(encoding="utf-8")
        assert 'exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py"' in text
        assert '"$AUTODL_PYTHON"' in text


def test_tastemolnet_defaults_use_fixed_persistent_foundation_paths() -> None:
    common = (AUTODL / "common.sh").read_text(encoding="utf-8")
    commit = "16af8ead8a17b6bd3941d9eb5879c5be75c14114"
    assert commit in common
    assert (
        "$AUTODL_RUNTIME_ROOT/data/tastemolnet/prepared/"
        "$TASTEMOLNET_UPSTREAM_COMMIT/splits"
    ) in common
    assert (
        "$AUTODL_RUNTIME_ROOT/cache/tastemolnet/"
        "$TASTEMOLNET_UPSTREAM_COMMIT/molecular_graph_v1"
    ) in common
    assert "$PROJECT_ROOT/data/processed/tastemolnet/splits" not in common


def test_detached_worker_freezes_python_and_control_root_in_launch_spec() -> None:
    text = (AUTODL / "exp_run.py").read_text(encoding="utf-8")
    assert 'spec["python_executable"]' in text
    assert '"python_executable": str(Path(sys.executable).resolve(strict=True))' in text
    assert 'spec["control_root"]' in text
    assert '"control_root": str(layout.control_root)' in text


def test_exp_run_rejects_preexisting_output_without_controller_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ordinary-output"
    output.mkdir()
    (output / "existing.txt").write_text("published\n", encoding="utf-8")
    data_root = tmp_path / "runtime-data"
    data_root.mkdir()
    environment = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "AUTODL_CONTROL_ROOT": str(data_root / "control"),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(AUTODL / "exp_run.py"),
            "--project-root",
            str(PROJECT_ROOT),
            "--data-root",
            str(data_root),
            "launch",
            "--dataset",
            "ordinary-fixture",
            "--stage",
            "fresh-gate",
            "--expected-output",
            str(output),
            "--foreground",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2
    assert "Expected output must be fresh/absent" in result.stderr
