from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTODL = PROJECT_ROOT / "scripts" / "autodl"


def test_tastemolnet_full_script_has_two_independent_heavy_guards() -> None:
    script = (AUTODL / "run_tastemolnet_gnn_full.sh").read_text(encoding="utf-8")
    assert 'RUN_TASTEMOLNET" != "1"' in script
    assert "--heavy" in script
    assert "[TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW]" in script
    assert "[TASTEMOLNET_FOUNDATION_READY_NOT_RUN]" not in script


def test_tastemolnet_full_license_marker_blocks_without_launch(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    marker = tmp_path / "LICENSE_REVIEW_REQUIRED"
    marker.write_text("review required\n", encoding="utf-8")
    environment = {
        **os.environ,
        "AUTODL_PYTHON": sys.executable,
        "AUTODL_DATA_ROOT": str(data_root),
        "AUTODL_RUNTIME_ROOT": str(data_root / "runtime"),
        "TASTEMOLNET_LICENSE_MARKER": str(marker),
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
    assert result.returncode == 65
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "[TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW]"
    )


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
