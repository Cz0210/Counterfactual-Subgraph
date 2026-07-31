from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from scripts.ops.state import RunStatus, RunStore


def test_state_and_events_are_persistent(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path, "task", run_id="run1", spec_path="x.yaml")
    store.transition(RunStatus.VALIDATED)
    store.record_stage("one", {"status": "PASSED", "attempt": 1})
    reopened = RunStore.open(store.run_dir)
    assert reopened.load()["status"] == "VALIDATED"
    assert reopened.stage_succeeded("one")
    events = [
        json.loads(line)
        for line in reopened.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )


def test_approval_records_identity_and_reason(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path, "task", run_id="run2")
    store.approve("freeze", "Reviewed selector provenance.", "researcher")
    approval = store.load()["approvals"]["freeze"]
    assert approval["username"] == "researcher"
    assert approval["reason"] == "Reviewed selector provenance."
    assert approval["hostname"]


def test_successful_stage_can_be_skipped_on_resume(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path, "task", run_id="run3")
    store.record_stage("done", {"status": "PASSED", "attempt": 1})
    assert store.stage_succeeded("done")
    store.append_event("stage_skipped_resume", stage_id="done")
    assert store.load()["stages"]["done"]["attempt"] == 1


def test_persisted_state_matches_repository_schema(tmp_path: Path) -> None:
    store = RunStore.create(
        tmp_path, "task", run_id="schema", spec_path="spec.yaml"
    )
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "ops/schemas/run_state.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(store.load())


def test_deploy_preflight_states_are_schema_compatible(tmp_path: Path) -> None:
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "ops/schemas/run_state.schema.json"
        ).read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(schema)
    for index, status in enumerate(
        (
            RunStatus.DRY_RUN_COMPLETED,
            RunStatus.ADOPT_EXISTING_DRY_RUN,
            RunStatus.REMOTE_PREFLIGHT_RUNNING,
            RunStatus.REMOTE_PREFLIGHT_PASSED,
            RunStatus.REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS,
            RunStatus.REMOTE_PREFLIGHT_BLOCKED,
            RunStatus.NEEDS_DEPLOY,
            RunStatus.NEEDS_PROXY_SETUP,
            RunStatus.ADOPT_EXISTING_VERIFYING,
            RunStatus.ADOPTED_EXISTING,
            RunStatus.STOPPED_BEFORE_APPROVAL,
        )
    ):
        store = RunStore.create(
            tmp_path / str(index),
            "task",
            run_id=f"state_{index}",
            spec_path="spec.yaml",
        )
        store.transition(status)
        validator.validate(store.load())
