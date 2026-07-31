"""Short Markdown and JSON automation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def tail_lines(text: str, maximum: int = 80) -> list[str]:
    return text.splitlines()[-maximum:]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_final_report(
    run_dir: Path,
    *,
    state: dict[str, Any],
    gate_summary: str,
    output_roots: Iterable[str],
    provenance: dict[str, Any],
    next_allowed_stage: str | None,
    stop_reason: str,
    details: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    stages = state.get("stages", {})
    passed = [
        stage_id
        for stage_id, record in stages.items()
        if record.get("status") in {"PASSED", "ADOPTED_EXISTING"}
    ]
    jobs = {
        stage_id: record["job_id"]
        for stage_id, record in stages.items()
        if record.get("job_id")
    }
    payload = {
        "task": state["task_id"],
        "run_id": state["run_id"],
        "status": state["status"],
        "local_commit": state.get("local_commit"),
        "remote_commit": state.get("remote_commit"),
        "slurm_jobs": jobs,
        "passed_stages": passed,
        "failed_stage": None,
        "gate_summary": gate_summary,
        "output_roots": list(output_roots),
        "scientific_provenance": provenance,
        "next_allowed_stage": next_allowed_stage,
        "stop_reason": stop_reason,
        "detailed_logs": str(run_dir),
        "details": dict(details or {}),
    }
    markdown = "\n".join(
        [
            "# Automation Result",
            "",
            f"- Task: {payload['task']}",
            f"- Run ID: {payload['run_id']}",
            f"- Status: {payload['status']}",
            f"- Local commit: {payload['local_commit'] or 'not created'}",
            f"- Remote commit: {payload['remote_commit'] or 'not deployed'}",
            f"- Slurm jobs: {jobs or 'none'}",
            f"- Passed stages: {', '.join(passed) or 'none'}",
            "- Failed stage: none",
            f"- Gate summary: {gate_summary}",
            f"- Output roots: {', '.join(payload['output_roots']) or 'none'}",
            f"- Scientific provenance: {json.dumps(provenance, sort_keys=True)}",
            f"- Next allowed stage: {next_allowed_stage or 'none'}",
            f"- Stop reason: {stop_reason}",
            f"- Detailed logs: {run_dir}",
            f"- Details: {json.dumps(payload['details'], sort_keys=True)}",
            "",
        ]
    )
    md_path = run_dir / "FINAL_REPORT.md"
    json_path = run_dir / "FINAL_REPORT.json"
    md_path.write_text(markdown, encoding="utf-8")
    _write_json(json_path, payload)
    return md_path, json_path


def write_blocked_report(
    run_dir: Path,
    *,
    state: dict[str, Any],
    failed_stage: str,
    error_class: str,
    return_code: int | None,
    stderr: str,
    artifacts: Iterable[str],
    retry_count: int,
    recommended_action: str,
    scientific_semantics_risk: bool,
    details: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    excerpt = tail_lines(stderr, 80)
    payload = {
        "task": state["task_id"],
        "run_id": state["run_id"],
        "status": state["status"],
        "failed_stage": failed_stage,
        "error_class": error_class,
        "return_code": return_code,
        "stderr_tail": excerpt,
        "artifacts": list(artifacts),
        "retry_count": retry_count,
        "recommended_action": recommended_action,
        "scientific_semantics_risk": scientific_semantics_risk,
        "details": dict(details or {}),
    }
    lines = [
        "# Automation Blocked",
        "",
        f"- Failed stage: {failed_stage}",
        f"- Error class: {error_class}",
        f"- Return code: {return_code}",
        f"- Artifact paths: {', '.join(payload['artifacts']) or 'none'}",
        f"- Retried: {retry_count}",
        f"- Recommended action: {recommended_action}",
        f"- Scientific semantics may change: {scientific_semantics_risk}",
        f"- Details: {json.dumps(payload['details'], sort_keys=True)}",
        "",
        "## Stderr Tail",
        "",
        "```text",
        *excerpt,
        "```",
        "",
    ]
    md_path = run_dir / "BLOCKED_REPORT.md"
    json_path = run_dir / "BLOCKED_REPORT.json"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    _write_json(json_path, payload)
    return md_path, json_path
