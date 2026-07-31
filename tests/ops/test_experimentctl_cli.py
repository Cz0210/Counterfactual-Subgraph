from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from scripts.ops import experimentctl
from scripts.ops.state import RunStore


ROOT = Path(__file__).resolve().parents[2]


def test_validate_and_plan_example(capsys) -> None:
    spec = ROOT / "ops/specs/example_smoke.yaml"
    assert experimentctl.main(["validate-spec", str(spec)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    assert experimentctl.main(["plan", str(spec)]) == 0
    output = capsys.readouterr().out
    assert '"submission_entrypoint": "scripts/exp_sbatch.sh"' in output
    assert '"long_polling": false' in output


def test_run_local_dry_run_has_no_report_side_effect(capsys) -> None:
    spec_path = ROOT / "ops/specs/example_smoke.yaml"
    reports = ROOT / "ops/reports/example_smoke"
    before = set(reports.iterdir()) if reports.exists() else set()
    assert (
        experimentctl.main(["run-local", str(spec_path), "--dry-run"]) == 0
    )
    after = set(reports.iterdir()) if reports.exists() else set()
    assert after == before
    assert '"side_effects": false' in capsys.readouterr().out


def test_resume_skips_successful_stage(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    spec_path = write_spec(base_spec)
    spec = experimentctl.load_task_spec(spec_path)
    store = RunStore.create(
        tmp_path / "reports",
        "unit_task",
        run_id="resume",
        spec_path=str(spec_path),
    )
    experimentctl.dump_spec_snapshot(spec, store.run_dir / "spec.snapshot.yaml")
    store.record_stage("local_gate", {"status": "PASSED", "attempt": 1})
    monkeypatch.setattr(
        experimentctl, "_run_local_stage", lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("successful stage reran")
        )
    )
    resumed = experimentctl.run_local(spec, existing_store=store)
    assert resumed.load()["stages"]["local_gate"]["attempt"] == 1


def test_deploy_and_submit_permissions_block(base_spec, write_spec) -> None:
    path = write_spec(base_spec)
    assert experimentctl.main(["deploy", str(path)]) == 2
    assert experimentctl.main(["submit", str(path)]) == 2


def test_deploy_dry_run_only_constructs_commands(
    base_spec, write_spec, monkeypatch
) -> None:
    path = write_spec(base_spec)
    monkeypatch.setattr(experimentctl, "head_commit", lambda runner, root: "abc")
    monkeypatch.setattr(
        experimentctl,
        "inspect_status",
        lambda runner, root, allowed: SimpleNamespace(
            allowed_modified_paths=(), staged_paths=()
        ),
    )
    monkeypatch.setattr(
        experimentctl,
        "commits_changed_paths",
        lambda runner, root, branch: [],
    )
    result = experimentctl.deploy(
        experimentctl.load_task_spec(path), dry_run=True
    )
    assert result["dry_run"] is True
    assert result["commit"] == "abc"
    assert result["preflight_argv"][0] == "ssh"


def test_source_contains_no_long_poll_or_direct_sbatch() -> None:
    source = (
        ROOT / "scripts/ops/experimentctl.py"
    ).read_text(encoding="utf-8")
    assert "while True" not in source
    assert '["sbatch"' not in source


def test_local_retry_limit_is_applied(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    payload = deepcopy(base_spec)
    payload["execution"]["max_auto_retries"] = 1
    spec = experimentctl.load_task_spec(write_spec(payload))
    store = experimentctl._new_store(spec, run_id="retry")
    outcomes = iter([False, True])

    def fake_stage(spec_arg, store_arg, stage_arg, runner_arg):
        previous = store_arg.load()["stages"].get(stage_arg["id"], {})
        store_arg.record_stage(
            stage_arg["id"],
            {
                "status": "PASSED" if (outcome := next(outcomes)) else "FAILED",
                "attempt": int(previous.get("attempt", 0)) + 1,
            },
        )
        return outcome

    monkeypatch.setattr(experimentctl, "_run_local_stage", fake_stage)
    result = experimentctl.run_local(spec, existing_store=store)
    assert result.load()["status"] == "COMPLETED"
    assert result.load()["stages"]["local_gate"]["attempt"] == 2


def test_nonlocal_stage_stops_local_runner(
    base_spec, write_spec, monkeypatch
) -> None:
    payload = deepcopy(base_spec)
    remote = deepcopy(payload["stages"][0])
    remote["id"] = "remote_next"
    remote["kind"] = "remote_command"
    remote["dependencies"] = ["local_gate"]
    payload["stages"].append(remote)
    payload["execution"]["auto_until"] = "remote_next"
    spec = experimentctl.load_task_spec(write_spec(payload))
    store = experimentctl._new_store(spec, run_id="local_stop")

    def pass_stage(spec_arg, store_arg, stage_arg, runner_arg):
        store_arg.record_stage(
            stage_arg["id"], {"status": "PASSED", "attempt": 1}
        )
        return True

    monkeypatch.setattr(experimentctl, "_run_local_stage", pass_stage)
    result = experimentctl.run_local(spec, existing_store=store)
    assert result.load()["status"] == "WAITING_APPROVAL"
    assert "remote_next" not in result.load()["stages"]


def test_scientific_boundaries_require_approval() -> None:
    stage = {
        "id": "first_test_evaluation",
        "kind": "slurm_job",
        "resources": {"tags": "evaluation", "notes": ""},
    }
    assert experimentctl._requires_approval(stage)
