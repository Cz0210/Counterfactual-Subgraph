#!/usr/bin/env python3
"""Append Slurm status snapshots for jobs registered by exp_sbatch.py."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_JSONL = "outputs/hpc/experiment_registry/jobs.jsonl"
DEFAULT_OUT_JSONL = "outputs/hpc/experiment_registry/job_status_updates.jsonl"
DEFAULT_MARKDOWN_LOG = "docs/EXPERIMENT_LOG.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Slurm job status and append experiment status updates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--registry-jsonl", default=DEFAULT_REGISTRY_JSONL, help="Experiment registry JSONL.")
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL, help="Status update JSONL.")
    parser.add_argument("--markdown-log", default=DEFAULT_MARKDOWN_LOG, help="Markdown experiment log.")
    parser.add_argument("--job-id", default=None, help="Optional single Slurm job id to query.")
    parser.add_argument("--days", type=int, default=30, help="Only sync registry entries from the last N days.")
    return parser.parse_args()


def now_display() -> tuple[str, str]:
    local_dt = datetime.now().astimezone()
    return local_dt.isoformat(timespec="seconds"), local_dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def read_registry(path: Path, days: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    cutoff = datetime.now().astimezone() - timedelta(days=days)
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            job_id = str(entry.get("job_id", ""))
            if not job_id or not job_id.isdigit():
                continue
            timestamp = parse_timestamp(str(entry.get("timestamp_local", "")))
            if timestamp is not None and timestamp < cutoff:
                continue
            entries.append(entry)
    return entries


def unique_job_ids(entries: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    job_ids: list[str] = []
    for entry in entries:
        job_id = str(entry.get("job_id", ""))
        if job_id and job_id.isdigit() and job_id not in seen:
            seen.add(job_id)
            job_ids.append(job_id)
    return job_ids


def run_command(command: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        return completed.returncode, completed.stdout, completed.stderr
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except Exception as exc:  # pragma: no cover - defensive.
        return 1, "", str(exc)


def parse_sacct(job_id: str, raw: str) -> dict[str, str]:
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    header = lines[0].split("|")
    rows = [line.split("|") for line in lines[1:]]
    selected = rows[0]
    for row in rows:
        if row and row[0] == job_id:
            selected = row
            break
    data = {header[index]: selected[index] if index < len(selected) else "" for index in range(len(header))}
    return {
        "job_id": data.get("JobID", job_id),
        "state": data.get("State", ""),
        "exit_code": data.get("ExitCode", ""),
        "elapsed": data.get("Elapsed", ""),
        "start": data.get("Start", ""),
        "end": data.get("End", ""),
        "nodelist": data.get("NodeList", ""),
    }


def parse_squeue(job_id: str, raw: str) -> dict[str, str]:
    line = next((item for item in raw.splitlines() if item.strip()), "")
    if not line:
        return {}
    parts = line.split("|")
    return {
        "job_id": parts[0] if len(parts) > 0 else job_id,
        "state": parts[2] if len(parts) > 2 else "",
        "exit_code": "",
        "elapsed": parts[3] if len(parts) > 3 else "",
        "start": parts[4] if len(parts) > 4 else "",
        "end": "",
        "nodelist": parts[5] if len(parts) > 5 else "",
    }


def query_job(job_id: str) -> dict[str, Any]:
    timestamp, _ = now_display()
    sacct_cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList%30",
        "-P",
    ]
    code, stdout, stderr = run_command(sacct_cmd)
    if code == 0 and stdout.strip():
        parsed = parse_sacct(job_id, stdout)
        if parsed:
            return {
                "timestamp": timestamp,
                "job_id": job_id,
                **parsed,
                "query_source": "sacct",
                "raw_output": stdout.strip(),
            }

    squeue_cmd = ["squeue", "-j", job_id, "-h", "-o", "%i|%j|%T|%M|%S|%R"]
    sq_code, sq_stdout, sq_stderr = run_command(squeue_cmd)
    if sq_code == 0 and sq_stdout.strip():
        parsed = parse_squeue(job_id, sq_stdout)
        return {
            "timestamp": timestamp,
            "job_id": job_id,
            **parsed,
            "query_source": "squeue",
            "raw_output": sq_stdout.strip(),
        }

    return {
        "timestamp": timestamp,
        "job_id": job_id,
        "state": "STATUS_QUERY_FAILED",
        "exit_code": "",
        "elapsed": "",
        "start": "",
        "end": "",
        "nodelist": "",
        "query_source": "none",
        "raw_output": "\n".join(
            part
            for part in [
                "sacct stdout:",
                stdout.strip(),
                "sacct stderr:",
                stderr.strip(),
                "squeue stdout:",
                sq_stdout.strip(),
                "squeue stderr:",
                sq_stderr.strip(),
            ]
            if part
        ),
    }


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ensure_markdown_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Experiment Log\n", encoding="utf-8")


def append_markdown_status(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_markdown_log(path)
    _, display = now_display()
    lines = [
        "",
        "",
        f"## Status Sync | {display}",
        "",
        "| JobID | State | ExitCode | Elapsed | Start | End |",
        "| ----- | ----- | -------- | ------- | ----- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {job_id} | {state} | {exit_code} | {elapsed} | {start} | {end} |".format(
                job_id=row.get("job_id", ""),
                state=row.get("state", ""),
                exit_code=row.get("exit_code", ""),
                elapsed=row.get("elapsed", ""),
                start=row.get("start", ""),
                end=row.get("end", ""),
            )
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if args.job_id:
        job_ids = [args.job_id]
    else:
        entries = read_registry(Path(args.registry_jsonl), args.days)
        job_ids = unique_job_ids(entries)

    if not job_ids:
        print("[EXP_STATUS_SYNC]")
        print("No registered Slurm job ids found to sync.")
        return 0

    rows = [query_job(job_id) for job_id in job_ids]
    append_jsonl(Path(args.out_jsonl), rows)
    append_markdown_status(Path(args.markdown_log), rows)

    print("[EXP_STATUS_SYNC]")
    print(f"jobs={len(rows)}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"markdown_log={args.markdown_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
