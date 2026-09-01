from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/autodl/run_tastemolnet_t14_postprocess_relay_v1.sh"
LAUNCHER = ROOT / "scripts/autodl/launch_tastemolnet_t14_postprocess_relay_v1.sh"
SLURM = ROOT / "scripts/slurm/run_tastemolnet_t14_postprocess_relay_v1.sh"


def test_t14_relay_binds_exact_generation_without_signalling_it() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        "T14_GENERATION_LAUNCHER_JSON",
        "T14_GENERATION_EXECUTION_COMMIT",
        "generation_launcher_sha256",
        "launcher_pid",
        "start_ticks",
        "run_tastemolnet_comrecgc_full.py",
        "tastemolnet_t14_progress_v1",
        "scan_live_writers",
        "[TASTE_T14_COMRECGC_FULL_GENERATION_PASS]",
    ):
        assert token in text
    for forbidden in ("kill -TERM", "kill -KILL", "kill -9", "pkill", "killall", "SIGKILL"):
        assert forbidden not in text


def test_t14_relay_uses_fresh_postprocess_and_exact_terminal_locator() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert text.count("uuid.uuid4()") == 2
    assert "science-attempt-$POSTPROCESS_ID" in text
    assert "final-attempt-$FINAL_ID" in text
    assert "run_tastemolnet_t14_comrecgc_postprocess.sh" in text
    assert "TASTEMOLNET_T14_POSTPROCESS_RESUME=0" in text
    assert "[TASTE_COMRECGC_PASS]" in text
    assert '"method":"ComRecGC"' in text
    assert "fast16_matrix_cell_root_locator_v1" in text
    assert "os.replace(temporary, destination)" in text
    assert "os.fsync" in text


def test_t14_relay_is_gpu2_persistent_and_ablation_safe() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "T14_POSTPROCESS_GPU_INDEX:-2" in runner
    assert '[[ "$GPU_INDEX" == "2" ]]' in runner
    assert "RUN_GNN_ABLATION" in runner
    assert "heartbeat.json" in runner
    assert "controller.pid" in runner
    assert "flock -n 9" in runner
    assert "nohup bash" in launcher
    assert "launcher.pid" in launcher
    assert "cell_root_locator.json" in launcher


def test_t14_relay_has_paired_hpc_refusal() -> None:
    text = SLURM.read_text(encoding="utf-8")
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "direct Slurm execution is disabled",
    ):
        assert token in text
