from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/autodl/run_tastemolnet_t12_release_relay_v1.sh"
LAUNCHER = ROOT / "scripts/autodl/launch_tastemolnet_t12_release_relay_v1.sh"


def test_release_relay_has_only_explicit_science_dependencies() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "T11" not in text
    for token in (
        "TASTE_T3_ROOT",
        "TASTE_T7_PASS_ROOT",
        "TASTE_MANAGED_NEUROSED_ROOT",
        "T12_MANAGED_RELEASE_ROOT",
        "T12_RELEASE_VALIDATOR_ROOT",
        "tastemolnet_t7_typed_release_v1.py",
        "[TASTE_T12_DEPENDENCY_DECOUPLED]",
    ):
        assert token in text


def test_release_relay_is_fresh_gpu3_science() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'GPU_INDEX=${T12_GPU_INDEX:-3}' in text
    assert '[[ "$GPU_INDEX" == "3" ]]' in text
    assert "uuid.uuid4()" in text
    assert '[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]' in text
    assert "--mode fresh" in text
    assert "--mode resume" in text
    assert "checkpoint-00010000.manifest.json" in text
    assert "checkpoint-00020000.manifest.json" in text
    assert "[TASTE_T12_GCF_FULL_LAUNCHED]" in text
    assert "RUN_GNN_ABLATION" in text
    assert "export PYTHONDONTWRITEBYTECODE=1" in text
    assert 'PYTHONPATH="$T12_RELEASE_VALIDATOR_ROOT"' in text


def test_launcher_persists_controller_and_heartbeat_root() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "nohup bash" in text
    assert "run_tastemolnet_t12_release_relay_v1.sh" in text
    assert "launcher.pid" in text
    assert "T12_GPU_INDEX=3" in text
