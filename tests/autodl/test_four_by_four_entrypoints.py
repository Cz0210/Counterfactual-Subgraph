from __future__ import annotations

from pathlib import Path

from scripts.autodl import run_four_by_four_controller
from scripts.autodl import run_four_gpu_recovery_controller as engine
from scripts.autodl import status_four_by_four
from scripts.autodl import status_four_gpu_recovery as status_engine


def test_controller_delegates_under_isolated_namespace(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_main(argv):
        observed["argv"] = argv
        observed["namespace"] = engine.CONTROLLER_NAME
        return 17

    monkeypatch.setattr(engine, "main", fake_main)
    assert run_four_by_four_controller.main(["validate"]) == 17
    assert observed == {
        "argv": ["validate"],
        "namespace": "four_methods_four_datasets_continuation",
    }


def test_status_delegates_under_same_isolated_namespace(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_main(argv):
        observed["argv"] = argv
        observed["status_namespace"] = status_engine.CONTROLLER_NAME
        observed["engine_namespace"] = engine.CONTROLLER_NAME
        return 19

    monkeypatch.setattr(status_engine, "main", fake_main)
    assert status_four_by_four.main(["--format", "json"]) == 19
    assert observed == {
        "argv": ["--format", "json"],
        "status_namespace": "four_methods_four_datasets_continuation",
        "engine_namespace": "four_methods_four_datasets_continuation",
    }


def test_launcher_uses_new_namespace_and_never_stops_old_controller() -> None:
    launcher = Path("scripts/autodl/launch_four_by_four.sh").read_text(encoding="utf-8")
    assert "four_methods_four_datasets_continuation/$CONTROLLER_ID" in launcher
    assert "run_four_by_four_controller.py" in launcher
    assert "status_four_by_four.py" in launcher
    assert "kill" not in launcher
    assert "run_three_lines" not in launcher
