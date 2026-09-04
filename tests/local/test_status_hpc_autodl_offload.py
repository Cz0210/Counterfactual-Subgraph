from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_SCRIPT = REPO_ROOT / "scripts/local/status_hpc_autodl_offload.sh"


def _write_fake_ssh(path: Path, *, fail: bool = False) -> None:
    if fail:
        body = "#!/bin/sh\nexit 255\n"
    else:
        body = r"""#!/bin/sh
case "$*" in
  *test-hpc*)
    printf '%s\n' \
      'hpc_ssh_state=PASS' \
      'hpc_hostname=hpc-test' \
      'hpc_execution_worktree_state=AVAILABLE' \
      'hpc_execution_commit=0123456789abcdef' \
      'hpc_python_state=AVAILABLE' \
      'hpc_python_version=Python_3.10.0' \
      'hpc_runtime_free_kib=123456' \
      'hpc_slurm_state=AVAILABLE' \
      'hpc_t8_jobs=41,t8-hpc-canary,RUNNING,00:02,node01' \
      'hpc_t8_current_pointer_state=PASS' \
      'hpc_t8_chain_state=REFINEMENT_CANARY_SUBMITTED' \
      'hpc_t8_chain_stage=REFINEMENT_CANARY' \
      'hpc_t8_chain_refinement_depth=6' \
      'hpc_t8_chain_job_ids=41,42' \
      'hpc_t8_chain_dependency=afterany:41' \
      'hpc_t8_chain_jobs=41,42' \
      'hpc_t8_chain_slurm=41,RUNNING,00:02,node01,(null);42,PENDING,0:00,(Dependency),afterany:41(unfulfilled)'
    ;;
  *test-autodl*)
    printf '%s\n' \
      'autodl_ssh_state=PASS' \
      'autodl_hostname=autodl-test' \
      'autodl_gpu_summary=index_usedMiB_freeMiB_utilPct:0, 100, 80000, 4' \
      'autodl_matrix_state=READABLE' \
      'autodl_matrix_complete_cells=12' \
      'autodl_matrix_total_cells=16' \
      'autodl_t8_state=RUNNING' \
      'autodl_t12_state=WAITING' \
      'autodl_t14_state=RUNNING' \
      'autodl_mut_state=RUNNING'
    ;;
  *) exit 255 ;;
esac
"""
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _environment(tmp_path: Path, fake_ssh: Path) -> dict[str, str]:
    local_repo = tmp_path / "repo"
    transfer = tmp_path / "transfer"
    local_repo.mkdir()
    transfer.mkdir()
    return {
        **os.environ,
        "STATUS_SSH_BIN": str(fake_ssh),
        "HPC_ALIAS": "test-hpc",
        "AUTODL_ALIAS": "test-autodl",
        "HPC_CONTROL_SOCKET": str(tmp_path / "absent.sock"),
        "LOCAL_REPO_ROOT": str(local_repo),
        "LOCAL_TRANSFER_ROOT": str(transfer),
        "HPC_EXECUTION_WORKTREE": "/share/test/worktree",
        "HPC_RUNTIME_ROOT": "/share/test/runtime",
        "HPC_T8_CURRENT_POINTER": "/share/test/runtime/control/t8-hpc-current-chain/current.json",
        "HPC_PYTHON": "/share/test/python",
        "AUTODL_MATRIX_AUTHORITY": "/autodl/test/matrix",
        "AUTODL_PYTHON": "/autodl/test/python",
    }


def test_status_is_redacted_and_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    fake_ssh = tmp_path / "fake-ssh"
    _write_fake_ssh(fake_ssh)
    env = _environment(tmp_path, fake_ssh)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = subprocess.run(
        ["bash", str(STATUS_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert "hpc_ssh_state=PASS" in result.stdout
    assert "autodl_ssh_state=PASS" in result.stdout
    assert "autodl_matrix_complete_cells=12" in result.stdout
    assert "hpc_t8_current_pointer_state=PASS" in result.stdout
    assert "hpc_t8_chain_dependency=afterany:41" in result.stdout
    assert "hpc_t8_chain_slurm=41,RUNNING" in result.stdout
    assert "status_side_effects=NONE" in result.stdout
    lowered = result.stdout.lower()
    assert "password" not in lowered
    assert "private key" not in lowered
    assert "identityfile" not in lowered
    assert "authorization:" not in lowered


def test_status_reports_unreachable_without_failing_or_retrying(tmp_path: Path) -> None:
    fake_ssh = tmp_path / "fake-ssh"
    _write_fake_ssh(fake_ssh, fail=True)
    env = _environment(tmp_path, fake_ssh)

    result = subprocess.run(
        ["bash", str(STATUS_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "hpc_ssh_state=UNREACHABLE" in result.stdout
    assert "autodl_ssh_state=UNREACHABLE" in result.stdout
    assert "status_side_effects=NONE" in result.stdout


def test_status_script_has_no_mutating_or_secret_dump_commands() -> None:
    source = STATUS_SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "StrictHostKeyChecking=no",
        "ForwardAgent=yes",
        "ssh -vv",
        "ssh -G",
        "printenv",
        "env |",
        "ps -",
        "rm -",
        "kill ",
        "pkill",
        "killall",
        "sbatch",
        "scancel",
        "mkdir",
        "touch ",
        "tee ",
    )
    for token in forbidden:
        assert token not in source


def test_status_defaults_to_pinned_science_worktree_and_dynamic_pointer() -> None:
    source = STATUS_SCRIPT.read_text(encoding="utf-8")
    assert "/share/home/u20526/czx/worktrees/t8-hpc-481475c3" in source
    assert "$HPC_RUNTIME_ROOT/control/t8-production-chain/current.json" in source
    assert "/tmp/tongji-codex.sock" not in source
    assert "hpc_t8_chain_dependency" in source
    assert "hpc_t8_chain_slurm" in source
    for stale_job_id in ("2535373", "2536033", "2536034", "2536148", "2536149"):
        assert stale_job_id not in source
