from __future__ import annotations

from pathlib import Path


def test_cap_postprocess_sidecar_is_scoped_and_fail_closed() -> None:
    project = Path(__file__).resolve().parents[2]
    script = (
        project / "scripts/autodl/wait_launch_bace_comrecgc_cap_postprocess.sh"
    ).read_text(encoding="utf-8")

    assert "BACE_CAP_SOURCE_FRAGMENT" in script
    assert "BACE_CAP_QUEUE_ROOT" in script
    assert "prepare_bace_comrecgc_resource_cap_postprocess.py" in script
    assert "launch_four_by_four.sh" in script
    assert "heartbeat.json" in script
    assert "RUN_TASTEMOLNET=0" in script
    assert '"gnn_ablation_started": False' in script
    assert "kill " not in script
    assert "pkill" not in script
    assert "SIGKILL" not in script
