from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.utils.main_ready_task_specs import MainReadyTaskSpecError
from src.utils import mut_clean_successor_v2 as successor


def _prior(tmp_path: Path) -> dict:
    root = tmp_path.resolve()
    return {
        "schema_version": "main_ready_task_spec_v1",
        "task_id": "mut-prior",
        "task_kind": "MUT_TRACE_EQUIVALENCE_AND_ADOPTION",
        "attempt_uuid": str(uuid4()),
        "repo_root": str(root / "old-repo"),
        "execution_commit": "1" * 40,
        "python": "/usr/bin/python3",
        "entrypoint": str(root / "old-repo/owner.py"),
        "config_path": str(root / "old-repo/config.yaml"),
        "config_sha256": "2" * 64,
        "manifest_path": str(root / "legacy-spec.json"),
        "manifest_sha256": "3" * 64,
        "input_roots": {"candidate_universe": str(root / "input")},
        "input_hashes": {"candidate_universe": "4" * 64},
        "output_root": str(root / "prior-output"),
        "gpu_request": {"index": 0},
        "cpu_request": {"workers": 2},
        "memory_request": {"headroom": 64},
        "required_environment": {"PYTHONHASHSEED": "0"},
        "matrix_authority_root": str(root / "matrix"),
        "expected_owner_command_sha256": "5" * 64,
        "expected_heartbeat_path": str(root / "prior-owner/heartbeat.json"),
        "expected_pid_file": str(root / "prior-owner/pid.json"),
        "expected_terminal_path": str(root / "prior-owner/terminal.json"),
        "resume_policy": "fresh",
        "single_writer_policy": "fail_if_live_owner_or_output_writer",
        "created_at": "2026-09-04T00:00:00+00:00",
        "spec_sha256": "6" * 64,
        "arguments": ["--task-spec", str(root / "prior.json")],
        "science_contract": {
            "trace_on_root": str(root / "old-on"),
            "trace_off_root": str(root / "old-off"),
        },
    }


def test_operationally_failed_predecessor_allows_one_fresh_successor(
    tmp_path: Path,
) -> None:
    prior = _prior(tmp_path)
    terminal = tmp_path / "terminal.json"
    terminal.write_text(
        json.dumps(
            {
                "task_id": "mut-prior",
                "status": "FAILED_REVIEWED_WORKER_EXIT_1",
                "owner_pid": 161697,
                "owner_start_ticks": 18437020,
                "output_root": prior["output_root"],
            }
        ),
        encoding="utf-8",
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    result = successor.validate_predecessor_terminal(
        prior_spec=prior, terminal_path=terminal.resolve(), proc_root=proc
    )
    assert result["status"] == "PASS_PREDECESSOR_OPERATIONAL_FAILURE_NO_LIVE_WRITER"
    assert result["science_failure_claimed"] is False


def test_live_predecessor_is_never_restarted(tmp_path: Path) -> None:
    prior = _prior(tmp_path)
    terminal = tmp_path / "terminal.json"
    terminal.write_text(
        json.dumps(
            {
                "task_id": "mut-prior",
                "status": "FAILED_REVIEWED_WORKER_EXIT_1",
                "owner_pid": 10,
                "owner_start_ticks": 77,
                "output_root": prior["output_root"],
            }
        ),
        encoding="utf-8",
    )
    proc = tmp_path / "proc"
    process = proc / "10"
    process.mkdir(parents=True)
    # start_ticks is field 22; the helper parses the 20th field after state.
    (process / "stat").write_text(
        "10 (python) S 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 77 0\n",
        encoding="utf-8",
    )
    (process / "cmdline").write_bytes(b"python\0owner.py\0")
    (process / "cwd").symlink_to(tmp_path)
    with pytest.raises(MainReadyTaskSpecError, match="still alive"):
        successor.validate_predecessor_terminal(
            prior_spec=prior, terminal_path=terminal.resolve(), proc_root=proc
        )


def test_successor_is_fresh_vs_fresh_and_preserves_science_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _prior(tmp_path)
    repo = tmp_path / "repo"
    (repo / "scripts/autodl").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "scripts/autodl/run_mut_clean_trace_equivalence_v1.py").write_text(
        "# owner\n", encoding="utf-8"
    )
    (repo / "configs/hpc.yaml").write_text("runtime: {}\n", encoding="utf-8")
    monkeypatch.setattr(successor, "_git_head", lambda _root: "a" * 40)
    attempt = str(uuid4())
    spec_root = tmp_path / "specs"
    spec_path = spec_root / "mut-successor.json"
    result = successor.build_successor_spec(
        prior_spec=prior,
        repo_root=repo.resolve(),
        task_id="mut-successor",
        attempt_uuid=attempt,
        spec_path=spec_path.resolve(),
        output_root=(tmp_path / "new-output").resolve(),
        owner_runtime_root=(tmp_path / "new-owner").resolve(),
        gpu_index=0,
        gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
        lease_path=(tmp_path / "leases/mut.lock").resolve(),
    )
    assert result["input_roots"] == prior["input_roots"]
    assert result["input_hashes"] == prior["input_hashes"]
    assert result["matrix_authority_root"] == prior["matrix_authority_root"]
    assert result["science_contract"]["fresh_vs_fresh"] is True
    assert result["science_contract"]["predecessor_result_adopted"] is False
    assert result["required_environment"]["MUT_PROTECTED_BASELINE_MAX_WAIT_SECONDS"] == "1800"
    assert result["cpu_request"]["workers"] == 2


def test_launcher_uses_robust_profile_without_old_binder() -> None:
    project = Path(__file__).resolve().parents[2]
    owner = (project / "scripts/autodl/run_mut_clean_trace_equivalence_v1.py").read_text(
        encoding="utf-8"
    )
    launcher = (project / "scripts/autodl/launch_mut_clean_successor_v2.sh").read_text(
        encoding="utf-8"
    )
    assert '"--throttle-profile", "robust-v2"' in owner
    assert "nice -n 10" in launcher
    assert "ionice -c 2 -n 7" in launcher
    assert 'taskset -c "$MUT_CPUSET"' in launcher
    assert "hot_bind_main_ready_task_specs.py" not in launcher
    assert "MUT_COMPLETED_A_ARM_ROOT" not in launcher

