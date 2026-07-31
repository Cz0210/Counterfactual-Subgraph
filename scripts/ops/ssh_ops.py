"""Non-interactive SSH command construction and remote safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import shlex
from typing import Sequence


class SSHSafetyError(RuntimeError):
    """An unsafe SSH command or remote state was detected."""


@dataclass(frozen=True, slots=True)
class SSHConfig:
    host: str
    port: int
    user: str
    remote_root: str
    conda_env: str
    control_socket: str | None = None

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"


def shell_join(argv: Sequence[str]) -> str:
    if not argv:
        raise ValueError("Remote argv must not be empty.")
    return " ".join(shlex.quote(str(value)) for value in argv)


def build_ssh_argv(config: SSHConfig, remote_argv: Sequence[str]) -> list[str]:
    if any("password" in str(value).lower() for value in remote_argv):
        raise SSHSafetyError("Password automation is forbidden.")
    argv = ["ssh"]
    if config.control_socket:
        argv.extend(["-S", config.control_socket])
    argv.extend(
        [
            "-p",
            str(config.port),
            "-o",
            "BatchMode=yes",
            config.destination,
            "--",
            "bash",
            "-lc",
            shell_join(remote_argv),
        ]
    )
    return argv


def _activation_script(config: SSHConfig, body: str) -> list[str]:
    script = "\n".join(
        [
            "set -eo pipefail",
            "set +u",
            "source ~/.bashrc",
            f"conda activate {shlex.quote(config.conda_env)}",
            "set -u",
            body,
        ]
    )
    return ["bash", "-lc", script]


def build_preflight_argv(
    config: SSHConfig,
    *,
    protected_output_roots: Sequence[str] = (),
    allow_overwrite: bool = False,
) -> list[str]:
    root = shlex.quote(str(PurePosixPath(config.remote_root)))
    lines = [
        "hostname",
        f"test -d {root}",
        f"cd {root}",
        "pwd",
        "git branch --show-current",
        "git rev-parse HEAD",
        "python --version",
        "command -v sbatch",
        "command -v sacct",
    ]
    if not allow_overwrite:
        for output_root in protected_output_roots:
            output = PurePosixPath(output_root)
            if not output.is_absolute():
                output = PurePosixPath(config.remote_root) / output
            finalized = shlex.quote(str(output / "_FINALIZED.json"))
            lines.append(
                f"test ! -e {finalized} || "
                f"{{ echo '[FINALIZED_OUTPUT_BLOCKED] {finalized}' >&2; exit 42; }}"
            )
    body = "\n".join(lines)
    return build_ssh_argv(config, _activation_script(config, body))


def build_deploy_argv(
    config: SSHConfig,
    *,
    branch: str,
    expected_commit: str,
    affected_paths: Sequence[str] = (),
    dynamic_paths: Sequence[str] = (),
) -> list[str]:
    root = shlex.quote(str(PurePosixPath(config.remote_root)))
    branch_q = shlex.quote(branch)
    commit_q = shlex.quote(expected_commit)
    status_paths = " ".join(shlex.quote(value) for value in affected_paths)
    dynamic_cases = "|".join(shlex.quote(value) for value in dynamic_paths)
    dirty_gate: list[str] = ["git status --short"]
    if affected_paths:
        dirty_gate.extend(
            [
                f"while IFS= read -r line; do",
                '  path="${line:3}"',
                (
                    f"  case \"$path\" in {dynamic_cases}) ;; "
                    "*) echo \"[REMOTE_DIRTY_BLOCKED] $line\" >&2; exit 41 ;; esac"
                    if dynamic_cases
                    else '  echo "[REMOTE_DIRTY_BLOCKED] $line" >&2; exit 41'
                ),
                f"done < <(git status --porcelain -- {status_paths})",
            ]
        )
    body = "\n".join(
        [
            f"cd {root}",
            *dirty_gate,
            f"git fetch origin {branch_q}",
            f"git merge-base --is-ancestor HEAD origin/{branch_q}",
            f"git pull --ff-only origin {branch_q}",
            f'test "$(git rev-parse HEAD)" = {commit_q}',
            "git rev-parse HEAD",
        ]
    )
    return build_ssh_argv(config, ["bash", "-lc", body])


def build_status_argv(config: SSHConfig, job_ids: Sequence[str]) -> list[str]:
    if not job_ids:
        raise ValueError("At least one Slurm job ID is required.")
    if any(not str(value).isdigit() for value in job_ids):
        raise SSHSafetyError("Slurm job IDs must be numeric.")
    root = shlex.quote(str(PurePosixPath(config.remote_root)))
    lines = [f"cd {root}"]
    for job_id in job_ids:
        quoted = shlex.quote(str(job_id))
        lines.append(
            "sacct -n -P -j "
            f"{quoted} --format=JobIDRaw,State,ExitCode "
            f"|| squeue -h -j {quoted} -o '%i|%T|N/A'"
        )
    return build_ssh_argv(
        config, _activation_script(config, "\n".join(lines))
    )
