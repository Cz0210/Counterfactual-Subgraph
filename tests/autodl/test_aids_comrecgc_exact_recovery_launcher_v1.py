from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/autodl/launch_aids_comrecgc_exact_recovery_v1.sh"


def _executable(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o755)
    return path


def _run_launcher(tmp_path: Path, *, thread_count: int) -> subprocess.CompletedProcess[str]:
    controller = tmp_path / "controller"
    controller.mkdir()
    project = tmp_path / "project"
    (project / "scripts/autodl").mkdir(parents=True)
    fake_python = _executable(
        tmp_path / "bin/fake-python",
        """#!/bin/bash
set -euo pipefail
[[ "$*" == *"--prepare-only"* ]]
printf '%s\n' \
  "$TEST_CONTROLLER_ROOT" \
  exact-cid \
  launch-id \
  "$TEST_CONTROLLER_ROOT/controller.log" \
  "$TEST_CONTROLLER_ROOT/controller.pid" \
  exact-session \
  "$TEST_CONTROLLER_ROOT/prelaunch.json" \
  "$TEST_THREAD_COUNT"
""",
    )
    _executable(
        tmp_path / "bin/tmux",
        """#!/bin/bash
set -euo pipefail
case "$1" in
  has-session) exit 1 ;;
  new-session) exit 0 ;;
  list-panes) printf '4242\n' ;;
  *) exit 64 ;;
esac
""",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    # macOS ships Bash 3 without readarray; AutoDL uses modern Bash.  Supply a
    # test-only compatibility function so the real launcher can be exercised
    # through the worker-count gate on both hosts.
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        """readarray() {
  local option=$1
  local array_name=$2
  local line
  local index=0
  local escaped
  [[ "$option" == "-t" ]] || return 64
  eval "$array_name=()"
  while IFS= read -r line; do
    printf -v escaped '%q' "$line"
    eval "$array_name[$index]=$escaped"
    index=$((index + 1))
  done
}
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "AUTODL_PROJECT_ROOT": str(project),
            "AUTODL_PYTHON": str(fake_python),
            "BASH_ENV": str(bash_env),
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            "TEST_CONTROLLER_ROOT": str(controller),
            "TEST_THREAD_COUNT": str(thread_count),
        }
    )
    return subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(manifest), "fresh"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_launcher_accepts_twelve_worker_prelaunch_context(tmp_path: Path) -> None:
    completed = _run_launcher(tmp_path, thread_count=12)

    assert completed.returncode == 0, completed.stderr
    assert "launched cid=exact-cid launch_id=launch-id" in completed.stdout
    assert (tmp_path / "controller/controller.pid").read_text(encoding="utf-8") == "4242\n"
    assert (tmp_path / "controller/controller.log").is_file()


@pytest.mark.parametrize("thread_count", [0, 8, 16])
def test_launcher_rejects_any_non_twelve_worker_context(
    tmp_path: Path, thread_count: int
) -> None:
    completed = _run_launcher(tmp_path, thread_count=thread_count)

    assert completed.returncode == 70
    assert not (tmp_path / "controller/controller.log").exists()
    assert not (tmp_path / "controller/controller.pid").exists()
