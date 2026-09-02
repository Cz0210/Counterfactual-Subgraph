from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval import tastemolnet_globalgce_valid_zero_relay as relay


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source(tmp_path: Path, *, rules0: int, rules2: int) -> Path:
    source = tmp_path / "source"
    _json(source / "checkpoint.json", {"phase": "TARGET_2_COMPLETE"})
    for target, rules in ((0, rules0), (2, rules2)):
        _json(
            source / "raw" / f"target_{target}" / "branch_manifest.json",
            {"status": "PASS", "valid_native_rule_count": rules},
        )
        _json(
            source / "raw" / f"target_{target}"
            / "globalgce_training_checkpoints" / "gspan" / "support_1"
            / "heartbeat.json",
            {
                "schema_version": relay.GSPAN_SCHEMA,
                "root_count": 5,
                "completed_root_count": 5,
                "frequent_subgraph_count": 17,
                "heartbeat_epoch_seconds": 1_800_000_000.0 + target,
                "stage": "complete",
                "current_root_index": 5,
                "peak_rss_mib": 1024,
                "sqlite_path": "/must/not/be/opened.sqlite",
            },
        )
    return source


def test_observer_routes_nonzero_to_normal_path_without_sqlite(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source(tmp_path, rules0=1, rules2=0)
    monkeypatch.setattr(
        relay, "_process_sample", lambda *args, **kwargs: {
            "alive": True, "pid": 7, "start_ticks": 11, "rss_bytes": 2,
            "cpu_percent": 3.0, "cpu_ticks": 4, "sampled_at_unix": 5.0,
        }
    )
    result = relay.observe_once(
        source_root=source, proc_root=tmp_path / "proc", science_pid=7,
        science_start_ticks=11, runtime=relay.RelayRuntime(), now=5.0,
    )
    assert result["state"] == "NORMAL_PATH"
    assert result["branches"]["valid_rule_count"] == 1
    assert result["gspan_progress"]["sqlite_opened"] is False


def test_zero_observation_requires_two_completed_gspan_authorities(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, rules0=0, rules2=0)
    heartbeat = {
        "state": "ZERO_CANDIDATE",
        "gspan_progress": relay.read_gspan_progress(source),
        "last_live_process": {"rss_bytes": 10, "cpu_percent": 2.0},
        "patterns_delta": 0,
        "patterns_per_minute": 0.0,
    }
    observation = relay.build_terminal_observation(
        heartbeat, attempt_id="attempt", source_root=source, output_bytes=100
    )
    assert observation["training_complete"] is True
    assert observation["active_database_opened"] is False
    assert observation["root_completed_count"] == observation["root_total_count"]


def test_relay_has_no_signal_sqlite_or_training_entrypoint() -> None:
    source = Path(relay.__file__).read_text(encoding="utf-8")
    assert "sqlite3" not in source
    assert "os.kill" not in source
    assert "subprocess" not in source
    assert '"training_started": False' in source

