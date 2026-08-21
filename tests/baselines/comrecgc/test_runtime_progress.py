from __future__ import annotations

import os
import socket
import time

from src.baselines.comrecgc.runtime import _progress_payload


def test_generation_progress_exposes_complete_resume_heartbeat_contract() -> None:
    payload = _progress_payload(
        current_step=500,
        max_steps=50_000,
        config_sha256="c" * 64,
        run_complete=False,
        checkpoint_dir="/persistent/checkpoints/step-000000000500",
        last_checkpoint_step=500,
        started_monotonic=time.monotonic() - 2.0,
        process_start_step=250,
        gpu_id="0",
        code_commit="a" * 40,
    )

    required = {
        "completed_step",
        "next_step",
        "total_steps",
        "steps_per_hour",
        "elapsed_seconds",
        "last_checkpoint_step",
        "latest_checkpoint",
        "heartbeat_at",
        "pid",
        "hostname",
        "gpu_id",
        "code_commit",
    }
    assert required <= set(payload)
    assert payload["completed_step"] == 500
    assert payload["next_step"] == 501
    assert payload["total_steps"] == 50_000
    assert payload["last_checkpoint_step"] == 500
    assert payload["latest_checkpoint"].endswith("step-000000000500")
    assert payload["elapsed_seconds"] >= 2.0
    assert payload["steps_per_hour"] > 0.0
    assert payload["pid"] == os.getpid()
    assert payload["hostname"] == socket.gethostname()
    assert payload["gpu_id"] == "0"
    assert payload["code_commit"] == "a" * 40
