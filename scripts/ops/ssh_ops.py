"""Non-interactive SSH command construction and remote safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import shlex
from typing import Any, Mapping, Sequence


class SSHSafetyError(RuntimeError):
    """An unsafe SSH command or remote state was detected."""


PROXY_VARIABLES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)


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
class RemoteSubmoduleStatus:
    path: str
    status_lines: tuple[str, ...]
    modified_paths: tuple[str, ...]
    staged_paths: tuple[str, ...]
    marker_results: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "status_lines": list(self.status_lines),
            "modified_paths": list(self.modified_paths),
            "staged_paths": list(self.staged_paths),
            "marker_results": dict(self.marker_results),
        }


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
    submodules: tuple[RemoteSubmoduleStatus, ...] = ()
    untracked_paths: tuple[str, ...] = ()
    untracked_scan_complete: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "hostname": self.hostname,
            "pwd": self.pwd,
            "branch": self.branch,
            "commit": self.commit,
            "python_version": self.python_version,
            "remote_dirty_summary": list(self.dirty_lines),
            "remote_untracked_paths": list(self.untracked_paths),
            "untracked_scan_complete": self.untracked_scan_complete,
            "conda_ready": self.conda_ready,
            "sbatch_ready": self.sbatch_ready,
            "sacct_ready": self.sacct_ready,
            "finalized_output_blocked": self.finalized_output_blocked,
            "finalized_paths": list(self.finalized_paths),
            "proxy_variables_present": dict(self.proxy_present),
            "patched_submodule_evidence": [
                item.to_dict() for item in self.submodules
            ],
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
            "-o",
            "ClearAllForwardings=yes",
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


def _proxy_presence_lines(variable: str) -> list[str]:
    if variable not in PROXY_VARIABLES:
        raise ValueError(f"Unsupported proxy variable: {variable!r}")
    marker = f"PREFLIGHT_PROXY_{variable}"
    return [
        f"if [[ -n ${{{variable}+x}} ]]; then",
        f"  echo '[{marker}] true'",
        "else",
        f"  echo '[{marker}] false'",
        "fi",
    ]


def build_preflight_argv(
    config: SSHConfig,
    *,
    protected_output_roots: Sequence[str] = (),
    allow_overwrite: bool = False,
    patched_submodules: Sequence[Mapping[str, Any]] = (),
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
        "echo '[PREFLIGHT_UNTRACKED_BEGIN]'",
        "git ls-files --others --exclude-standard",
        "echo '[PREFLIGHT_UNTRACKED_END]'",
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
    for variable in PROXY_VARIABLES:
        lines.extend(_proxy_presence_lines(variable))
    for policy in patched_submodules:
        path = str(policy["path"])
        path_q = shlex.quote(path)
        lines.extend(
            [
                "echo "
                + shlex.quote(f"[PREFLIGHT_SUBMODULE_BEGIN] {path}"),
                "echo "
                + shlex.quote(f"[PREFLIGHT_SUBMODULE_STATUS_BEGIN] {path}"),
                f"git -C {path_q} status --porcelain=v1",
                "echo "
                + shlex.quote(f"[PREFLIGHT_SUBMODULE_STATUS_END] {path}"),
                "echo "
                + shlex.quote(
                    f"[PREFLIGHT_SUBMODULE_MODIFIED_BEGIN] {path}"
                ),
                f"git -C {path_q} diff --name-only",
                "echo "
                + shlex.quote(
                    f"[PREFLIGHT_SUBMODULE_MODIFIED_END] {path}"
                ),
                "echo "
                + shlex.quote(f"[PREFLIGHT_SUBMODULE_STAGED_BEGIN] {path}"),
                f"git -C {path_q} diff --cached --name-only",
                "echo "
                + shlex.quote(f"[PREFLIGHT_SUBMODULE_STAGED_END] {path}"),
            ]
        )
        for marker in policy.get("required_markers") or []:
            marker_file = str(marker["file"])
            marker_text = str(marker["contains"])
            marker_path = shlex.quote(str(PurePosixPath(path) / marker_file))
            marker_q = shlex.quote(marker_text)
            marker_prefix = f"{path}|{marker_file}|"
            lines.extend(
                [
                    f"if grep -F -- {marker_q} {marker_path} >/dev/null 2>&1; then",
                    "  echo "
                    + shlex.quote(
                        "[PREFLIGHT_SUBMODULE_MARKER] "
                        f"{marker_prefix}true"
                    ),
                    "else",
                    "  echo "
                    + shlex.quote(
                        "[PREFLIGHT_SUBMODULE_MARKER] "
                        f"{marker_prefix}false"
                    ),
                    "fi",
                ]
            )
        lines.append(
            "echo " + shlex.quote(f"[PREFLIGHT_SUBMODULE_END] {path}")
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
        lines.extend(
            [
                "status_output=\"$(sacct -n -X -P -j "
                f"{quoted} --format=JobIDRaw,State,ExitCode 2>/dev/null || true)\"",
                "if [[ -n \"$status_output\" ]]; then",
                "  printf '%s\\n' \"$status_output\"",
                "else",
                f"  squeue -h -j {quoted} -o '%i|%T|N/A'",
                "fi",
            ]
        )
    return build_ssh_argv(config, _activation_script(config, "\n".join(lines)))


def build_remote_project_command_argv(
    config: SSHConfig, command: Sequence[str]
) -> list[str]:
    """Run one audited argv command from the remote project root."""

    root = shlex.quote(str(PurePosixPath(config.remote_root)))
    body = "\n".join([f"cd {root}", f"exec {shell_join(command)}"])
    return build_ssh_argv(config, _activation_script(config, body))


def build_remote_submit_argv(
    config: SSHConfig,
    command: Sequence[str],
    *,
    expected_commit: str,
    output_root: str,
) -> list[str]:
    """Build a guarded remote exp_sbatch invocation.

    The preconditions are read-only. The final command is expected to be the
    repository-owned ``scripts/exp_sbatch.sh`` submission wrapper.
    """

    if not command or str(command[0]) != "scripts/exp_sbatch.sh":
        raise SSHSafetyError("Remote Slurm submission must use exp_sbatch.sh.")
    output = PurePosixPath(output_root)
    if output.is_absolute() or ".." in output.parts or str(output) in {"", "."}:
        raise SSHSafetyError(f"Unsafe remote output root: {output_root!r}")
    root_path = PurePosixPath(config.remote_root)
    root = shlex.quote(str(root_path))
    output_absolute = shlex.quote(str(root_path / output))
    expected = shlex.quote(expected_commit)
    body = "\n".join(
        [
            f"cd {root}",
            f'if [[ "$(git rev-parse HEAD)" != {expected} ]]; then',
            "  echo '[AUTOMATION_SUBMIT_BLOCKED] remote commit mismatch' >&2",
            "  exit 45",
            "fi",
            f"if [[ -e {output_absolute} ]]; then",
            "  echo '[AUTOMATION_SUBMIT_BLOCKED] output root already exists' >&2",
            "  exit 46",
            "fi",
            f"exec {shell_join(command)}",
        ]
    )
    return build_ssh_argv(config, _activation_script(config, body))


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
    untracked_paths: list[str] = []
    finalized_paths: list[str] = []
    in_dirty = False
    in_untracked = False
    untracked_scan_complete = False
    submodule_sections: dict[str, dict[str, list[str]]] = {}
    submodule_markers: dict[str, dict[str, bool]] = {}
    active_submodule: str | None = None
    active_section: str | None = None
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
        if line == "[PREFLIGHT_UNTRACKED_BEGIN]":
            in_untracked = True
            untracked_scan_complete = False
            continue
        if line == "[PREFLIGHT_UNTRACKED_END]":
            in_untracked = False
            untracked_scan_complete = True
            continue
        if in_untracked:
            if line:
                untracked_paths.append(line)
            continue
        section_markers = {
            "[PREFLIGHT_SUBMODULE_STATUS_BEGIN] ": "status",
            "[PREFLIGHT_SUBMODULE_MODIFIED_BEGIN] ": "modified",
            "[PREFLIGHT_SUBMODULE_STAGED_BEGIN] ": "staged",
        }
        section_ends = {
            "[PREFLIGHT_SUBMODULE_STATUS_END] ",
            "[PREFLIGHT_SUBMODULE_MODIFIED_END] ",
            "[PREFLIGHT_SUBMODULE_STAGED_END] ",
        }
        matched_section = False
        for prefix, section in section_markers.items():
            if line.startswith(prefix):
                active_submodule = line.removeprefix(prefix).strip()
                active_section = section
                submodule_sections.setdefault(
                    active_submodule,
                    {"status": [], "modified": [], "staged": []},
                )
                matched_section = True
                break
        if matched_section:
            continue
        if any(line.startswith(prefix) for prefix in section_ends):
            active_submodule = None
            active_section = None
            continue
        if active_submodule is not None and active_section is not None:
            if line.strip():
                submodule_sections[active_submodule][active_section].append(
                    line
                )
            continue
        if line.startswith("[PREFLIGHT_SUBMODULE_MARKER] "):
            payload = line.removeprefix(
                "[PREFLIGHT_SUBMODULE_MARKER] "
            )
            path, separator, remainder = payload.partition("|")
            marker_file, second_separator, raw_value = remainder.rpartition("|")
            if separator and second_separator:
                submodule_markers.setdefault(path, {})[marker_file] = (
                    raw_value.strip().lower() == "true"
                )
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
    submodule_paths = list(submodule_sections)
    for path in submodule_markers:
        if path not in submodule_paths:
            submodule_paths.append(path)
    submodules = tuple(
        RemoteSubmoduleStatus(
            path=path,
            status_lines=tuple(
                submodule_sections.get(path, {}).get("status", [])
            ),
            modified_paths=tuple(
                submodule_sections.get(path, {}).get("modified", [])
            ),
            staged_paths=tuple(
                submodule_sections.get(path, {}).get("staged", [])
            ),
            marker_results=dict(submodule_markers.get(path, {})),
        )
        for path in submodule_paths
    )
    return RemotePreflight(
        hostname=values["hostname"],
        pwd=values["pwd"],
        branch=values["branch"],
        commit=values["commit"],
        python_version=values["python_version"],
        dirty_lines=tuple(dirty_lines),
        untracked_paths=tuple(untracked_paths),
        untracked_scan_complete=untracked_scan_complete,
        conda_ready=booleans["conda_ready"],
        sbatch_ready=booleans["sbatch_ready"],
        sacct_ready=booleans["sacct_ready"],
        finalized_output_blocked=booleans["finalized_output_blocked"],
        finalized_paths=tuple(finalized_paths),
        proxy_present=proxy_present,
        submodules=submodules,
    )
