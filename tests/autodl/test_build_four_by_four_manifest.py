from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl import build_four_by_four_manifest as builder


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _blocked_task(task_id: str) -> dict:
    return {
        "id": task_id,
        "dataset": "tastemolnet",
        "stage": task_id.upper(),
        "depends_on": [],
        "resource": "gpu",
        "priority": 100,
        "enabled": True,
        "blocked_reason": "BLOCKED_LICENSE_REVIEW",
        "data_splits": ["train"],
        "command": None,
    }


def test_composes_fragments_with_persistent_blocked_heartbeat(tmp_path: Path) -> None:
    first = _write(tmp_path / "one.json", [_blocked_task("taste_ours")])
    second = _write(
        tmp_path / "two.json",
        {"tasks": [_blocked_task("taste_gcfexplainer")]},
    )
    output = tmp_path / "manifest.json"
    result = builder.compose_manifest(
        controller_id="four_methods_four_datasets_continuation_v1",
        fragments=[first, second],
        output=output,
    )
    assert result["status"] == "PASS"
    assert result["task_count"] == 2
    payload = json.loads(output.read_text())
    assert payload["runtime"]["keep_alive_when_blocked"] is True
    assert payload["paper_frozen"] is True
    assert [task["id"] for task in payload["tasks"]] == [
        "taste_ours",
        "taste_gcfexplainer",
    ]


def test_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    first = _write(tmp_path / "one.json", [_blocked_task("duplicate")])
    second = _write(tmp_path / "two.json", [_blocked_task("duplicate")])
    with pytest.raises(ValueError, match="Duplicate task id"):
        builder.compose_manifest(
            controller_id="four_methods_four_datasets_continuation_v1",
            fragments=[first, second],
            output=tmp_path / "manifest.json",
        )


def test_rejects_placeholders_even_for_blocked_tasks(tmp_path: Path) -> None:
    task = _blocked_task("taste_ours")
    task["environment"] = {"INPUT": "__CONFIGURE_PATH__"}
    fragment = _write(tmp_path / "fragment.json", [task])
    with pytest.raises(ValueError, match="placeholder"):
        builder.compose_manifest(
            controller_id="four_methods_four_datasets_continuation_v1",
            fragments=[fragment],
            output=tmp_path / "manifest.json",
        )


def test_failed_validation_removes_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fragment = _write(tmp_path / "fragment.json", [_blocked_task("taste_ours")])
    output = tmp_path / "manifest.json"

    def fail(_path: Path):
        raise ValueError("bad manifest")

    monkeypatch.setattr(builder, "load_controller_manifest", fail)
    with pytest.raises(ValueError, match="bad manifest"):
        builder.compose_manifest(
            controller_id="four_methods_four_datasets_continuation_v1",
            fragments=[fragment],
            output=output,
        )
    assert not output.exists()
