"""Safe subprocess execution with bounded output and audit metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


SAFE_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    "PROJECT_ROOT",
    "CONDA_DEFAULT_ENV",
    "SSH_AUTH_SOCK",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "NO_PROXY",
    "no_proxy",
)
PROXY_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inherited_environment(
    *,
    extra: Mapping[str, str] | None = None,
    preserve_proxy_environment: bool = True,
) -> dict[str, str]:
    """Return a child environment without logging or mutating proxy values."""

    environment = dict(os.environ)
    if not preserve_proxy_environment:
        for key in PROXY_KEYS:
            environment.pop(key, None)
    if extra:
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def environment_audit(environment: Mapping[str, str]) -> dict[str, object]:
    """Expose allowlisted values; proxy values are represented only by presence."""

    values = {
        key: environment[key]
        for key in SAFE_ENV_KEYS
        if key in environment and key not in PROXY_KEYS
    }
    return {
        "allowlisted_values": values,
        "proxy_present": {
            key: key in environment for key in PROXY_KEYS
        },
    }


class CommandRunner:
    def __init__(self, *, default_timeout_seconds: int = 600) -> None:
        self.default_timeout_seconds = int(default_timeout_seconds)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path,
        timeout_seconds: int | None = None,
        environment: Mapping[str, str] | None = None,
        dry_run: bool = False,
    ) -> CommandResult:
        command = [str(value) for value in argv]
        if not command or any("\x00" in value for value in command):
            raise ValueError("Command argv must be a non-empty NUL-free array.")
        resolved_cwd = Path(cwd).expanduser().resolve()
        if not resolved_cwd.is_dir():
            raise FileNotFoundError(f"Command cwd does not exist: {resolved_cwd}")
        if dry_run:
            return CommandResult(
                argv=command,
                cwd=str(resolved_cwd),
                returncode=0,
                stdout="",
                stderr="",
                dry_run=True,
            )
        try:
            completed = subprocess.run(
                command,
                cwd=str(resolved_cwd),
                env=dict(environment) if environment is not None else None,
                check=False,
                capture_output=True,
                text=True,
                timeout=(
                    int(timeout_seconds)
                    if timeout_seconds is not None
                    else self.default_timeout_seconds
                ),
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                argv=command,
                cwd=str(resolved_cwd),
                returncode=124,
                stdout=str(exc.stdout or ""),
                stderr=str(exc.stderr or "") or f"Timed out: {exc}",
                timed_out=True,
            )
        return CommandResult(
            argv=command,
            cwd=str(resolved_cwd),
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def write_command_streams(
    result: CommandResult,
    *,
    stdout_path: str | Path,
    stderr_path: str | Path,
) -> None:
    stdout = Path(stdout_path).expanduser().resolve()
    stderr = Path(stderr_path).expanduser().resolve()
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
