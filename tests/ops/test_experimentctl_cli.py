from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import json
import pytest

from scripts.ops import experimentctl
from scripts.ops.state import RunStore
from scripts.ops.subprocess_utils import CommandResult


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
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    runner = ExplodingRunner()
    run_dir = tmp_path / "dry_run"
    result = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=True,
        run_dir=run_dir,
        runner=runner,
    )
    assert result["dry_run"] is True
    assert result["commit"] == "abc"
    assert result["preflight_argv"][0] == "ssh"
    assert runner.calls == []
    assert result["status"] == "DRY_RUN_COMPLETED"
    for name in (
        "state.json",
        "events.jsonl",
        "plan.json",
        "commands.jsonl",
        "FINAL_REPORT.md",
    ):
        assert (run_dir / name).is_file()
    assert (run_dir / "commands.jsonl").stat().st_size > 0


def test_deploy_reuses_caller_provided_empty_run_dir(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    run_dir = tmp_path / "provided"
    run_dir.mkdir()
    result = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=True,
        run_dir=run_dir,
        runner=ExplodingRunner(),
    )
    assert Path(result["run_dir"]) == run_dir
    assert (run_dir / "state.json").is_file()


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


class ExplodingRunner:
    def __init__(self) -> None:
        self.calls = []

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        raise AssertionError("unexpected external command")


class FakePreflightRunner:
    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.calls: list[list[str]] = []

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        return self.result


def _mock_deploy_git(monkeypatch, commit: str = "abc") -> None:
    monkeypatch.setattr(
        experimentctl, "head_commit", lambda runner, root: commit
    )
    monkeypatch.setattr(
        experimentctl,
        "inspect_status",
        lambda runner, root, allowed: SimpleNamespace(
            allowed_modified_paths=(),
            staged_paths=(),
            unrelated_modified_paths=(),
        ),
    )
    monkeypatch.setattr(
        experimentctl,
        "commits_changed_paths",
        lambda runner, root, branch: [],
    )


def _preflight_stdout(
    *, commit: str = "abc", dirty: tuple[str, ...] = ()
) -> str:
    return "\n".join(
        [
            "[PREFLIGHT_CONDA_READY] true",
            "[PREFLIGHT_HOSTNAME] logini02",
            "[PREFLIGHT_PWD] /share/home/u20526/czx/counterfactual-subgraph",
            "[PREFLIGHT_BRANCH] main",
            f"[PREFLIGHT_COMMIT] {commit}",
            "[PREFLIGHT_DIRTY_BEGIN]",
            *dirty,
            "[PREFLIGHT_DIRTY_END]",
            "[PREFLIGHT_PYTHON] Python 3.10.18",
            "[PREFLIGHT_SBATCH_READY] true",
            "[PREFLIGHT_SACCT_READY] true",
            "[PREFLIGHT_PROXY_http_proxy] false",
            "[PREFLIGHT_PROXY_https_proxy] true",
            "[PREFLIGHT_PROXY_HTTP_PROXY] false",
            "[PREFLIGHT_PROXY_HTTPS_PROXY] false",
            "[PREFLIGHT_PROXY_all_proxy] false",
            "[PREFLIGHT_PROXY_ALL_PROXY] false",
            "[PREFLIGHT_FINALIZED_BLOCKED] false",
            "",
        ]
    )


def _command_result(
    tmp_path: Path,
    *,
    stdout: str | None = None,
    stderr: str = "",
    returncode: int = 0,
) -> CommandResult:
    return CommandResult(
        argv=["ssh"],
        cwd=str(tmp_path),
        returncode=returncode,
        stdout=stdout if stdout is not None else _preflight_stdout(),
        stderr=stderr,
    )


def test_deploy_parser_accepts_preflight_only() -> None:
    args = experimentctl._parser().parse_args(
        ["deploy", "task.yaml", "--preflight-only"]
    )
    assert args.preflight_only is True
    assert args.dry_run is False


def test_deploy_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        experimentctl._parser().parse_args(
            ["deploy", "task.yaml", "--dry-run", "--preflight-only"]
        )


def test_preflight_only_executes_exactly_one_read_only_ssh_command(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    runner = FakePreflightRunner(_command_result(tmp_path))
    run_dir = tmp_path / "preflight"
    result = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=False,
        preflight_only=True,
        run_dir=run_dir,
        runner=runner,
    )
    assert result["status"] == "REMOTE_PREFLIGHT_PASSED"
    assert result["return_code"] == 0
    assert len(runner.calls) == 1
    command = runner.calls[0]
    assert command[0] == "ssh"
    remote_script = command[-1]
    assert "git fetch" not in remote_script
    assert "git pull" not in remote_script
    assert "exp_sbatch" not in remote_script
    assert "bash -lc" not in remote_script
    assert (run_dir / "FINAL_REPORT.json").is_file()
    report = json.loads(
        (run_dir / "FINAL_REPORT.json").read_text(encoding="utf-8")
    )
    assert report["details"]["commits_equal"] is True
    assert report["details"]["next_action"] == (
        "remote_write_approval_required"
    )
    assert report["details"]["remote_write_performed"] is False


def test_preflight_commit_mismatch_never_runs_pull(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    runner = FakePreflightRunner(
        _command_result(tmp_path, stdout=_preflight_stdout(commit="older"))
    )
    result = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=False,
        preflight_only=True,
        run_dir=tmp_path / "mismatch",
        runner=runner,
    )
    assert result["status"] == "NEEDS_DEPLOY"
    assert result["return_code"] == 0
    assert len(runner.calls) == 1
    assert "git pull" not in runner.calls[0][-1]
    report = json.loads(
        (Path(result["report"]).parent / "FINAL_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["details"]["next_action"] == "deploy"
    assert report["details"]["commits_equal"] is False


def test_preflight_remote_failure_propagates_and_writes_blocked_report(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    runner = FakePreflightRunner(
        _command_result(
            tmp_path,
            stdout="",
            stderr="connection failed",
            returncode=7,
        )
    )
    result = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=False,
        preflight_only=True,
        run_dir=tmp_path / "failed",
        runner=runner,
    )
    assert result["return_code"] == 7
    assert result["status"] == "BLOCKED"
    assert Path(result["report"]).name == "BLOCKED_REPORT.md"
    assert (tmp_path / "failed/BLOCKED_REPORT.json").is_file()
    assert json.loads(
        (tmp_path / "failed/state.json").read_text(encoding="utf-8")
    )["status"] == "BLOCKED"


def test_finalized_marker_blocks_preflight(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    stdout = _preflight_stdout().replace(
        "[PREFLIGHT_FINALIZED_BLOCKED] false", ""
    )
    runner = FakePreflightRunner(
        _command_result(
            tmp_path,
            stdout=stdout,
            stderr=(
                "[PREFLIGHT_FINALIZED] "
                "/remote/outputs/final/_FINALIZED.json"
            ),
            returncode=42,
        )
    )
    result = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=False,
        preflight_only=True,
        run_dir=tmp_path / "finalized",
        runner=runner,
    )
    assert result["status"] == "BLOCKED"
    report = json.loads(
        (tmp_path / "finalized/BLOCKED_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["details"]["finalized_output_blocked"] is True
    assert "finalized_output_blocked" in report["details"]["failed_checks"]


def test_preflight_checks_finalized_even_when_overwrite_permission_is_true(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    base_spec["permissions"]["allow_overwrite"] = True
    base_spec["stages"][0]["resources"]["expected_output_root"] = (
        "outputs/final"
    )
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    runner = FakePreflightRunner(_command_result(tmp_path))
    experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=False,
        preflight_only=True,
        run_dir=tmp_path / "overwrite",
        runner=runner,
    )
    assert "outputs/final/_FINALIZED.json" in runner.calls[0][-1]


def test_successful_preflight_resume_does_not_repeat_ssh(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    run_dir = tmp_path / "resume_preflight"
    first_runner = FakePreflightRunner(_command_result(tmp_path))
    first = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=False,
        preflight_only=True,
        run_dir=run_dir,
        runner=first_runner,
    )
    second = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=False,
        preflight_only=True,
        run_dir=run_dir,
        runner=ExplodingRunner(),
    )
    assert first["status"] == second["status"]
    assert second["resumed"] is True
    commands = (run_dir / "commands.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(commands) == 1


def test_preflight_report_never_contains_proxy_value(
    base_spec, write_spec, monkeypatch, tmp_path
) -> None:
    path = write_spec(base_spec)
    _mock_deploy_git(monkeypatch)
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:39393/secret")
    runner = FakePreflightRunner(_command_result(tmp_path))
    result = experimentctl.deploy(
        experimentctl.load_task_spec(path),
        dry_run=False,
        preflight_only=True,
        run_dir=tmp_path / "proxy",
        runner=runner,
    )
    report_text = Path(result["report"]).read_text(encoding="utf-8")
    environment_text = (tmp_path / "proxy/environment_audit.json").read_text(
        encoding="utf-8"
    )
    assert "39393" not in report_text
    assert "39393" not in environment_text
    report = json.loads(
        (tmp_path / "proxy/FINAL_REPORT.json").read_text(encoding="utf-8")
    )
    assert report["details"]["proxy_variables_present"]["https_proxy"] is True
