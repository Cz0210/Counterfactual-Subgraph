from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.gate_runner import evaluate_gate, sha256_file


def gate(path: str | None, **values):
    return {
        "json_path": path,
        "required_marker": values.get("marker"),
        "required_fields": values.get("required", {}),
        "forbidden_fields": values.get("forbidden", {}),
        "float_tolerance": values.get("tolerance", 1e-12),
        "sha256": values.get("sha256", {}),
    }


def test_gate_json_and_artifacts_pass(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload\n", encoding="utf-8")
    payload = {
        "audit_passed": True,
        "run_complete": True,
        "failed_hard_checks": [],
        "checks": {"rows": 64, "distance": 0.1},
        "provenance": {"test_used": False},
    }
    (tmp_path / "gate.json").write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_gate(
        task_id="task",
        run_id="run",
        stage_id="audit",
        gate_spec=gate(
            "gate.json",
            marker="[PASS]",
            required={"checks.rows": 64, "checks.distance": 0.1},
            forbidden={"provenance.test_used": True},
            sha256={"artifact.txt": sha256_file(artifact)},
        ),
        expected_artifacts=["artifact.txt"],
        root=tmp_path,
        stdout="[PASS]",
        slurm_exit_code="0:0",
    )
    assert result.passed


def test_missing_artifact_blocks(tmp_path: Path) -> None:
    result = evaluate_gate(
        task_id="task",
        run_id="run",
        stage_id="audit",
        gate_spec=gate(None),
        expected_artifacts=["missing.json"],
        root=tmp_path,
    )
    assert not result.passed
    assert any("artifact_missing" in item for item in result.failed_hard_checks)


def test_failed_scientific_gate_blocks(tmp_path: Path) -> None:
    payload = {
        "audit_passed": False,
        "run_complete": True,
        "failed_hard_checks": ["chemistry"],
    }
    (tmp_path / "gate.json").write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_gate(
        task_id="task",
        run_id="run",
        stage_id="audit",
        gate_spec=gate("gate.json"),
        expected_artifacts=[],
        root=tmp_path,
        slurm_exit_code="0:0",
    )
    assert not result.passed
    assert "audit_passed_not_true" in result.failed_hard_checks
    assert "failed_hard_checks_not_empty" in result.failed_hard_checks


def test_nonzero_slurm_exit_blocks_even_when_json_passes(tmp_path: Path) -> None:
    payload = {
        "audit_passed": True,
        "run_complete": True,
        "failed_hard_checks": [],
    }
    (tmp_path / "gate.json").write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_gate(
        task_id="task",
        run_id="run",
        stage_id="audit",
        gate_spec=gate("gate.json"),
        expected_artifacts=[],
        root=tmp_path,
        slurm_exit_code="1:0",
    )
    assert not result.passed
