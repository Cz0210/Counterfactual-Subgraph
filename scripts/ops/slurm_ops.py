"""Slurm dependency planning through the existing exp_sbatch wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


class SlurmSubmissionError(RuntimeError):
    """Submission output did not satisfy the exp_sbatch protocol."""


JOB_ID_PATTERN = re.compile(r"^job_id=(\d+)$", flags=re.MULTILINE)


def parse_exp_sbatch_job_id(stdout: str) -> str:
    if "[EXP_SUBMIT_OK]" not in stdout:
        raise SlurmSubmissionError("Missing [EXP_SUBMIT_OK] marker.")
    match = JOB_ID_PATTERN.search(stdout)
    if not match:
        raise SlurmSubmissionError("Missing numeric job_id in exp_sbatch output.")
    return match.group(1)


def dependency_mode(
    stage: Mapping[str, Any], stage_by_id: Mapping[str, Mapping[str, Any]]
) -> str:
    kind = str(stage["kind"])
    stage_id = str(stage["id"])
    if kind == "audit" or "final" in stage_id.split("_"):
        return "afterany"
    dependencies = [stage_by_id[value] for value in stage["dependencies"]]
    if any(str(item["kind"]) == "audit" for item in dependencies):
        return "afterok"
    return "afterok"


def dependency_argument(
    stage: Mapping[str, Any],
    stage_by_id: Mapping[str, Mapping[str, Any]],
    job_ids: Mapping[str, str],
) -> str | None:
    dependencies = [
        str(value)
        for value in stage["dependencies"]
        if str(stage_by_id[str(value)]["kind"]) in {"slurm_job", "audit"}
        and stage_by_id[str(value)].get("script")
    ]
    if not dependencies:
        return None
    missing = [value for value in dependencies if value not in job_ids]
    if missing:
        raise SlurmSubmissionError(
            f"Missing dependency job IDs for {stage['id']}: {missing}"
        )
    mode = dependency_mode(stage, stage_by_id)
    return f"--dependency={mode}:{':'.join(job_ids[value] for value in dependencies)}"


def build_exp_sbatch_argv(
    stage: Mapping[str, Any],
    stage_by_id: Mapping[str, Mapping[str, Any]],
    job_ids: Mapping[str, str],
    *,
    automation_environment: Mapping[str, str] | None = None,
) -> list[str]:
    script = stage.get("script")
    if not script:
        raise SlurmSubmissionError(f"Stage {stage['id']} has no script.")
    resources = stage.get("resources") or {}
    argv = [
        "scripts/exp_sbatch.sh",
        "--name",
        str(resources.get("name") or stage["id"]),
        "--tags",
        str(resources.get("tags") or ""),
        "--notes",
        str(resources.get("notes") or ""),
        "--expected-output-root",
        str(resources.get("expected_output_root") or ""),
        "--",
    ]
    dependency = dependency_argument(stage, stage_by_id, job_ids)
    if dependency:
        argv.append(dependency)
    if automation_environment:
        values = ["ALL"]
        for key, value in sorted(automation_environment.items()):
            if "," in str(value) or "\n" in str(value):
                raise SlurmSubmissionError(
                    f"Unsafe Slurm export value for {key}."
                )
            values.append(f"{key}={value}")
        argv.append(f"--export={','.join(values)}")
    argv.append(str(script))
    command = stage.get("command") or []
    argv.extend(str(value) for value in command)
    return argv


@dataclass(frozen=True, slots=True)
class SlurmStatus:
    job_id: str
    state: str
    exit_code: str


def parse_sacct_line(line: str) -> SlurmStatus:
    columns = line.strip().split("|")
    if len(columns) < 3:
        raise ValueError(f"Invalid sacct line: {line!r}")
    return SlurmStatus(
        job_id=columns[0].strip(),
        state=columns[1].strip(),
        exit_code=columns[2].strip(),
    )


ACTIVE_STATES = {
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "SUSPENDED",
}
FAILURE_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OOM",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
}


def normalize_slurm_state(value: str) -> str:
    return value.strip().split(maxsplit=1)[0].rstrip("+").upper()


def parse_slurm_status_output(stdout: str, job_id: str) -> SlurmStatus:
    """Select the top-level job record from sacct or squeue output."""

    candidates: list[SlurmStatus] = []
    for line in stdout.splitlines():
        if not line.strip() or line.lstrip().startswith("["):
            continue
        try:
            parsed = parse_sacct_line(line)
        except ValueError:
            continue
        if parsed.job_id == str(job_id):
            candidates.append(
                SlurmStatus(
                    job_id=parsed.job_id,
                    state=normalize_slurm_state(parsed.state),
                    exit_code=parsed.exit_code or "N/A",
                )
            )
    if not candidates:
        raise ValueError(f"No top-level Slurm status found for job {job_id}.")
    return candidates[-1]


def is_active_status(status: SlurmStatus) -> bool:
    return status.state in ACTIVE_STATES


def is_success_status(status: SlurmStatus) -> bool:
    return status.state == "COMPLETED" and status.exit_code == "0:0"


def is_failure_status(status: SlurmStatus) -> bool:
    return status.state in FAILURE_STATES or (
        status.state == "COMPLETED" and status.exit_code != "0:0"
    )
