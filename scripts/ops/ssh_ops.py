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


@dataclass(frozen=True, slots=True)
class RemotePreflight:
    hostname: str | None
    pwd: str | None
    branch: str | None
    commit: str | None
    python_version: str | None
    dirty_lines: tuple[str, ...]
    conda_ready: bool
    sbatch_ready: bool
    sacct_ready: bool
    finalized_output_blocked: bool
    finalized_paths: tuple[str, ...]
    proxy_present: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "hostname": self.hostname,
            "pwd": self.pwd,
            "branch": self.branch,
            "commit": self.commit,
            "python_version": self.python_version,
            "remote_dirty_summary": list(self.dirty_lines),
            "conda_ready": self.conda_ready,
            "sbatch_ready": self.sbatch_ready,
            "sacct_ready": self.sacct_ready,
            "finalized_output_blocked": self.finalized_output_blocked,
            "finalized_paths": list(self.finalized_paths),
            "proxy_variables_present": dict(self.proxy_present),
        }


def shell_join(argv: Sequence[str]) -> str:
    if not argv:
        raise ValueError("Remote argv must not be empty.")
    return " ".join(shlex.quote(str(value)) for value in argv)


def build_ssh_argv(
    config: SSHConfig, remote_command: str | Sequence[str]
) -> list[str]:
    script = (
        remote_command
        if isinstance(remote_command, str)
        else shell_join(remote_command)
    )
    if "password" in script.lower():
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
            script,
        ]
    )
    return argv


def _activation_script(config: SSHConfig, body: str) -> str:
    return "\n".join(
        [
            "set -eo pipefail",
            "set +u",
            "source ~/.bashrc",
            f"conda activate {shlex.quote(config.conda_env)}",
            "echo '[PREFLIGHT_CONDA_READY] true'",
            body,
        ]
    )


def build_preflight_argv(
    config: SSHConfig,
    *,
    protected_output_roots: Sequence[str] = (),
    allow_overwrite: bool = False,
) -> list[str]:
    root = shlex.quote(str(PurePosixPath(config.remote_root)))
    lines = [
        "printf '[PREFLIGHT_HOSTNAME] '; hostname",
        f"test -d {root}",
        f"cd {root}",
        "printf '[PREFLIGHT_PWD] '; pwd",
        "printf '[PREFLIGHT_BRANCH] '; git branch --show-current",
        "printf '[PREFLIGHT_COMMIT] '; git rev-parse HEAD",
        "echo '[PREFLIGHT_DIRTY_BEGIN]'",
        "git status --short",
        "echo '[PREFLIGHT_DIRTY_END]'",
        "printf '[PREFLIGHT_PYTHON] '; python --version 2>&1",
        (
            "if command -v sbatch >/dev/null 2>&1; then "
            "echo '[PREFLIGHT_SBATCH_READY] true'; else "
            "echo '[PREFLIGHT_SBATCH_READY] false'; exit 43; fi"
        ),
        (
            "if command -v sacct >/dev/null 2>&1; then "
            "echo '[PREFLIGHT_SACCT_READY] true'; else "
            "echo '[PREFLIGHT_SACCT_READY] false'; exit 44; fi"
        ),
    ]
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        lines.append(
            f"if [[ -n ${{{key}+x}} ]]; then "
            f"echo '[PREFLIGHT_PROXY_{key}] true'; else "
            f"echo '[PREFLIGHT_PROXY_{key}] false'; fi"
        )
    if not allow_overwrite:
        for output_root in protected_output_roots:
            output = PurePosixPath(output_root)
            if not output.is_absolute():
                output = PurePosixPath(config.remote_root) / output
            finalized = shlex.quote(str(output / "_FINALIZED.json"))
            lines.append(
                f"test ! -e {finalized} || "
                f"{{ echo '[PREFLIGHT_FINALIZED] {finalized}' >&2; exit 42; }}"
            )
    lines.append("echo '[PREFLIGHT_FINALIZED_BLOCKED] false'")
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
    return build_ssh_argv(config, body)


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
    return build_ssh_argv(config, _activation_script(config, "\n".join(lines)))


def parse_preflight_output(stdout: str, stderr: str = "") -> RemotePreflight:
    scalar_markers = {
        "[PREFLIGHT_HOSTNAME]": "hostname",
        "[PREFLIGHT_PWD]": "pwd",
        "[PREFLIGHT_BRANCH]": "branch",
        "[PREFLIGHT_COMMIT]": "commit",
        "[PREFLIGHT_PYTHON]": "python_version",
    }
    values: dict[str, str | None] = {
        field: None for field in scalar_markers.values()
    }
    booleans = {
        "conda_ready": False,
        "sbatch_ready": False,
        "sacct_ready": False,
        "finalized_output_blocked": False,
    }
    proxy_present: dict[str, bool] = {}
    dirty_lines: list[str] = []
    finalized_paths: list[str] = []
    in_dirty = False
    combined_lines = [*stdout.splitlines(), *stderr.splitlines()]
    for line in combined_lines:
        if line == "[PREFLIGHT_DIRTY_BEGIN]":
            in_dirty = True
            continue
        if line == "[PREFLIGHT_DIRTY_END]":
            in_dirty = False
            continue
        if in_dirty:
            if line.strip():
                dirty_lines.append(line)
            continue
        matched_scalar = False
        for marker, field in scalar_markers.items():
            prefix = marker + " "
            if line.startswith(prefix):
                values[field] = line[len(prefix) :].strip()
                matched_scalar = True
                break
        if matched_scalar:
            continue
        if line == "[PREFLIGHT_CONDA_READY] true":
            booleans["conda_ready"] = True
        elif line == "[PREFLIGHT_SBATCH_READY] true":
            booleans["sbatch_ready"] = True
        elif line == "[PREFLIGHT_SACCT_READY] true":
            booleans["sacct_ready"] = True
        elif line.startswith("[PREFLIGHT_PROXY_"):
            marker, _, raw_value = line.partition("] ")
            key = marker.removeprefix("[PREFLIGHT_PROXY_")
            proxy_present[key] = raw_value.strip().lower() == "true"
        elif line.startswith("[PREFLIGHT_FINALIZED] "):
            booleans["finalized_output_blocked"] = True
            finalized_paths.append(
                line.removeprefix("[PREFLIGHT_FINALIZED] ").strip()
            )
    return RemotePreflight(
        hostname=values["hostname"],
        pwd=values["pwd"],
        branch=values["branch"],
        commit=values["commit"],
        python_version=values["python_version"],
        dirty_lines=tuple(dirty_lines),
        conda_ready=booleans["conda_ready"],
        sbatch_ready=booleans["sbatch_ready"],
        sacct_ready=booleans["sacct_ready"],
        finalized_output_blocked=booleans["finalized_output_blocked"],
        finalized_paths=tuple(finalized_paths),
        proxy_present=proxy_present,
    )
