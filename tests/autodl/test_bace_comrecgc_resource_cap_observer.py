from __future__ import annotations

import json
from pathlib import Path

from src.utils.autodl_bace_comrecgc_resource_cap_observer import ResourceCapObserver


def _write_hook(root: Path, step: int, *, unique: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "result": {
            "status": "CONTINUE",
            "evaluation_step": step,
            "checkpoint_summaries": [
                {
                    "step": step,
                    "valid_unique_count": unique,
                    "lineage_error_count": 0,
                }
            ],
            "checkpoint_evidence": [
                {"step": step, "checkpoint_digest": "b" * 64}
            ],
        }
    }
    (root / f"step-{step}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_observer_persists_read_only_handover_request(
    tmp_path: Path, monkeypatch
) -> None:
    hooks = tmp_path / "hooks"
    state = tmp_path / "state"
    _write_hook(hooks, 20_000, unique=10)
    monkeypatch.setattr(
        "src.utils.autodl_bace_comrecgc_resource_cap_observer._process_generation_matches",
        lambda _pid, _ticks: True,
    )
    observer = ResourceCapObserver(
        convergence_hook_root=hooks,
        state_root=state,
        science_pid=123,
        science_start_ticks=456,
    )
    result = observer.tick()
    request = json.loads((state / "handover_request.json").read_text(encoding="utf-8"))
    assert result["state"] == "HANDOVER_ELIGIBLE"
    assert request["m_effective"] == 20_000
    assert request["signal_sent"] is False
    assert request["postprocess_started"] is False
    assert request["manual_or_separately_authorized_executor_required"] is True


def test_observer_waits_when_hook_root_has_not_been_created(tmp_path: Path) -> None:
    observer = ResourceCapObserver(
        convergence_hook_root=tmp_path / "future-hooks",
        state_root=tmp_path / "state",
        science_pid=123,
        science_start_ticks=456,
    )
    result = observer.tick()
    assert result["state"] == "WAITING_COMMITTED_AUDIT"
    assert not (tmp_path / "state/handover_request.json").exists()
