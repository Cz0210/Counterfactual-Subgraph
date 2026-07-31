from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.report import write_blocked_report, write_final_report


def state():
    return {
        "task_id": "task",
        "run_id": "run",
        "status": "COMPLETED",
        "local_commit": "abc",
        "remote_commit": None,
        "stages": {"one": {"status": "PASSED", "job_id": None}},
    }


def test_final_report_is_short_and_structured(tmp_path: Path) -> None:
    md, result_json = write_final_report(
        tmp_path,
        state=state(),
        gate_summary="passed",
        output_roots=["outputs/a"],
        provenance={"test_used": False},
        next_allowed_stage=None,
        stop_reason="done",
    )
    assert len(md.read_text(encoding="utf-8").splitlines()) < 40
    assert json.loads(result_json.read_text(encoding="utf-8"))["status"] == (
        "COMPLETED"
    )


def test_blocked_report_stderr_is_limited_to_80_lines(tmp_path: Path) -> None:
    md, result_json = write_blocked_report(
        tmp_path,
        state=state(),
        failed_stage="gate",
        error_class="GateFailure",
        return_code=2,
        stderr="\n".join(f"line {index}" for index in range(100)),
        artifacts=[],
        retry_count=1,
        recommended_action="inspect",
        scientific_semantics_risk=False,
    )
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert len(payload["stderr_tail"]) == 80
    assert "line 0" not in md.read_text(encoding="utf-8")
