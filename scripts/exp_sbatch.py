#!/usr/bin/env python3
"""Submit Slurm jobs and record a lightweight experiment registry entry."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_JSONL = "outputs/hpc/experiment_registry/jobs.jsonl"
DEFAULT_MARKDOWN_LOG = "docs/EXPERIMENT_LOG.md"

ENV_KEYS = [
    "PROJECT_ROOT",
    "DATASET_CSV",
    "HIV_CSV",
    "AIDS_CSV",
    "SOURCE_INPUT_CSV",
    "TEACHER_PATH",
    "GCF_CANDIDATES_PATH",
    "OURS_SELECTED_PATH",
    "OUTPUT_ROOT",
    "LABEL",
    "TARGET_LABEL",
    "SMILES_COL",
    "LABEL_COL",
    "GED_THRESHOLDS",
    "EMBEDDING_THRESHOLDS",
    "GEN_TEMPERATURE",
    "GEN_TOP_P",
    "NUM_RETURN_SEQUENCES",
    "CUDA_VISIBLE_DEVICES",
    "CONDA_DEFAULT_ENV",
    "PYTHONPATH",
]


SBATCH_OPTIONS_WITH_VALUE = {
    "-A",
    "--account",
    "-a",
    "--array",
    "-C",
    "--constraint",
    "-c",
    "--cpus-per-task",
    "-D",
    "--chdir",
    "-d",
    "--dependency",
    "-e",
    "--error",
    "--export",
    "-G",
    "--gpus",
    "--gres",
    "-J",
    "--job-name",
    "-L",
    "--licenses",
    "-M",
    "--clusters",
    "--mail-type",
    "--mail-user",
    "--mem",
    "--mem-per-cpu",
    "-N",
    "--nodes",
    "-n",
    "--ntasks",
    "-o",
    "--output",
    "-p",
    "--partition",
    "-q",
    "--qos",
    "-t",
    "--time",
    "-w",
    "--nodelist",
    "-x",
    "--exclude",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a Slurm job with automatic experiment logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default=None, help="Human-readable experiment name.")
    parser.add_argument("--tags", default="", help="Comma-separated experiment tags.")
    parser.add_argument("--notes", default="", help="Short experiment notes.")
    parser.add_argument("--expected-output-root", default=None, help="Expected output directory.")
    parser.add_argument("--dataset", default="", help="Dataset name or path.")
    parser.add_argument("--label", default="", help="Target label or class subset.")
    parser.add_argument("--method", default="", help="Method name.")
    parser.add_argument("--metric", default="", help="Primary metric.")
    parser.add_argument("--dry-run", action="store_true", help="Print the entry without calling sbatch.")
    parser.add_argument(
        "--write-dry-run",
        action="store_true",
        help="When used with --dry-run, append the dry-run entry to logs.",
    )
    parser.add_argument("--registry-jsonl", default=DEFAULT_REGISTRY_JSONL, help="JSONL registry path.")
    parser.add_argument("--markdown-log", default=DEFAULT_MARKDOWN_LOG, help="Markdown experiment log path.")
    parser.add_argument(
        "sbatch_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to sbatch. Put them after --, for example: -- scripts/slurm/job.sh",
    )
    args = parser.parse_args(argv)
    args.sbatch_args = strip_remainder_separator(args.sbatch_args)
    if not args.sbatch_args:
        parser.error("missing sbatch arguments after --")
    return args


def strip_remainder_separator(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def now_values() -> tuple[datetime, str, str, str]:
    local_dt = datetime.now().astimezone()
    utc_dt = datetime.now(timezone.utc)
    display = local_dt.strftime("%Y-%m-%d %H:%M:%S")
    return local_dt, local_dt.isoformat(timespec="seconds"), utc_dt.isoformat(timespec="seconds"), display


def run_git(args: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - defensive for unusual HPC shells.
        return f"ERROR: {exc}"
    output = completed.stdout.strip()
    if completed.returncode != 0:
        error = completed.stderr.strip()
        return error or output or f"git {' '.join(args)} failed with code {completed.returncode}"
    return output


def parse_job_id(stdout: str) -> str:
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if match:
        return match.group(1)
    return "UNKNOWN"


def split_tags(tags: str) -> list[str]:
    return [part.strip() for part in tags.split(",") if part.strip()]


def infer_sbatch_script(sbatch_args: list[str]) -> str:
    for token in sbatch_args:
        if token.endswith(".sh") or token.endswith(".sbatch") or token.endswith(".slurm"):
            return token

    skip_next = False
    for token in sbatch_args:
        if skip_next:
            skip_next = False
            continue
        if token in SBATCH_OPTIONS_WITH_VALUE:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return token
    return ""


def parse_sbatch_directives(script_path: str, cwd: Path) -> dict[str, str]:
    if not script_path:
        return {}
    path = Path(script_path)
    if not path.is_absolute():
        path = cwd / path
    if not path.exists() or not path.is_file():
        return {}

    directives: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        body = stripped[len("#SBATCH") :].strip()
        if body.startswith("--output="):
            directives["output"] = body.split("=", 1)[1].strip()
        elif body.startswith("--error="):
            directives["error"] = body.split("=", 1)[1].strip()
        elif body.startswith("-o "):
            directives["output"] = body[3:].strip()
        elif body.startswith("-e "):
            directives["error"] = body[3:].strip()
    return directives


def build_slurm_log_hint(script_path: str, cwd: Path, job_id: str) -> str:
    directives = parse_sbatch_directives(script_path, cwd)
    if not directives:
        return ""

    parts: list[str] = []
    for key in ("output", "error"):
        value = directives.get(key)
        if not value:
            continue
        rendered = value.replace("%j", job_id).replace("%A", job_id)
        parts.append(f"{key}={rendered}")
    return "; ".join(parts)


def environment_snapshot() -> dict[str, str]:
    return {key: os.environ[key] for key in ENV_KEYS if key in os.environ}


def ensure_markdown_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Experiment Log\n\n"
            "This file is append-only. Each Slurm experiment should be submitted through "
            "`scripts/exp_sbatch.py` or `scripts/exp_sbatch.sh` so that job id, command, "
            "git commit, environment snapshot, and expected output paths are preserved.\n\n"
            "Recommended command:\n\n"
            "```bash\n"
            "scripts/exp_sbatch.sh \\\n"
            "  --name \"short experiment name\" \\\n"
            "  --tags \"label1,selector,ccrcov\" \\\n"
            "  --notes \"short note\" \\\n"
            "  --expected-output-root \"outputs/hpc/...\" \\\n"
            "  -- scripts/slurm/xxx.sh\n"
            "```\n",
            encoding="utf-8",
        )


def format_env(snapshot: dict[str, str]) -> str:
    if not snapshot:
        return "(empty)"
    return "\n".join(f"{key}={value}" for key, value in sorted(snapshot.items()))


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def append_markdown(path: Path, entry: dict[str, Any]) -> None:
    ensure_markdown_log(path)
    command = "sbatch " + shlex.join(entry["sbatch_args"])
    env_text = format_env(entry.get("environment_snapshot", {}))
    sbatch_output = entry.get("sbatch_stdout", "")
    if entry.get("sbatch_stderr"):
        sbatch_output = f"{sbatch_output}\nSTDERR:\n{entry['sbatch_stderr']}".strip()
    if not sbatch_output:
        sbatch_output = "(empty)"

    text = (
        f"\n\n## {entry['timestamp_display']} | Job {entry['job_id']} | {entry['experiment_name']}\n\n"
        f"* Status: {entry['status']}\n"
        f"* Tags: {entry['tags_raw']}\n"
        f"* Dataset: {entry['dataset']}\n"
        f"* Label: {entry['label']}\n"
        f"* Method: {entry['method']}\n"
        f"* Metric: {entry['metric']}\n"
        f"* Expected output root: {entry['expected_output_root']}\n"
        f"* Slurm script: {entry['sbatch_script']}\n"
        f"* Slurm log hint: {entry['slurm_log_hint']}\n"
        f"* Working directory: {entry['cwd']}\n"
        f"* Git branch: {entry['git_branch']}\n"
        f"* Git commit: {entry['git_commit']}\n"
        f"* Notes: {entry['notes']}\n\n"
        "Command:\n\n"
        "```bash\n"
        f"{command}\n"
        "```\n\n"
        "Environment snapshot:\n\n"
        "```text\n"
        f"{env_text}\n"
        "```\n\n"
        "Sbatch output:\n\n"
        "```text\n"
        f"{sbatch_output}\n"
        "```\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def build_entry(
    args: argparse.Namespace,
    *,
    status: str,
    job_id: str,
    sbatch_stdout: str,
    sbatch_stderr: str,
    return_code: int | None,
) -> dict[str, Any]:
    cwd = Path.cwd()
    _, timestamp_local, timestamp_utc, timestamp_display = now_values()
    sbatch_script = infer_sbatch_script(args.sbatch_args)
    experiment_name = args.name or (Path(sbatch_script).stem if sbatch_script else "unnamed experiment")
    expected_output_root = args.expected_output_root or os.environ.get("OUTPUT_ROOT", "")

    return {
        "timestamp_local": timestamp_local,
        "timestamp_utc": timestamp_utc,
        "timestamp_display": timestamp_display,
        "status": status,
        "job_id": job_id,
        "experiment_name": experiment_name,
        "tags": split_tags(args.tags),
        "tags_raw": args.tags,
        "notes": args.notes,
        "dataset": args.dataset,
        "label": args.label,
        "method": args.method,
        "metric": args.metric,
        "expected_output_root": expected_output_root,
        "cwd": str(cwd),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "git_branch": run_git(["branch", "--show-current"], cwd),
        "git_commit": run_git(["rev-parse", "HEAD"], cwd),
        "git_status_short": run_git(["status", "--short"], cwd),
        "sbatch_args": args.sbatch_args,
        "sbatch_script": sbatch_script,
        "sbatch_stdout": sbatch_stdout.strip(),
        "sbatch_stderr": sbatch_stderr.strip(),
        "sbatch_return_code": return_code,
        "slurm_log_hint": build_slurm_log_hint(sbatch_script, cwd, job_id),
        "environment_snapshot": environment_snapshot(),
    }


def submit_sbatch(sbatch_args: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["sbatch", *sbatch_args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except Exception as exc:  # pragma: no cover - defensive.
        return 1, "", str(exc)


def print_dry_run(entry: dict[str, Any]) -> None:
    print("[EXP_DRY_RUN]")
    print("This is a dry run. sbatch was not called.")
    print(json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry_path = Path(args.registry_jsonl)
    markdown_path = Path(args.markdown_log)

    if args.dry_run:
        entry = build_entry(
            args,
            status="DRY_RUN",
            job_id="DRY_RUN",
            sbatch_stdout="DRY RUN: sbatch was not called.",
            sbatch_stderr="",
            return_code=None,
        )
        print_dry_run(entry)
        if args.write_dry_run:
            append_jsonl(registry_path, entry)
            append_markdown(markdown_path, entry)
            print(f"registry_jsonl={registry_path}")
            print(f"markdown_log={markdown_path}")
        return 0

    return_code, stdout, stderr = submit_sbatch(args.sbatch_args)
    if return_code == 0:
        job_id = parse_job_id(stdout)
        status = "SUBMITTED"
    else:
        job_id = "SUBMIT_FAILED"
        status = "SUBMIT_FAILED"

    entry = build_entry(
        args,
        status=status,
        job_id=job_id,
        sbatch_stdout=stdout,
        sbatch_stderr=stderr,
        return_code=return_code,
    )
    append_jsonl(registry_path, entry)
    append_markdown(markdown_path, entry)

    if return_code == 0:
        print("[EXP_SUBMIT_OK]")
        print(f"job_id={job_id}")
        print(f"markdown_log={markdown_path}")
        print(f"registry_jsonl={registry_path}")
        print(f"expected_output_root={entry['expected_output_root']}")
        return 0

    if stderr.strip():
        print(stderr.strip(), file=sys.stderr)
    print("[EXP_SUBMIT_FAILED]", file=sys.stderr)
    print(f"markdown_log={markdown_path}", file=sys.stderr)
    print(f"registry_jsonl={registry_path}", file=sys.stderr)
    return return_code or 1


if __name__ == "__main__":
    raise SystemExit(main())
