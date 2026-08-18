#!/usr/bin/env python3
"""Audit active and pending Slurm GPU requests for the MUT/BACE project lanes."""

from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROJECT_PREFIXES = (
    "/share/home/u20526/czx/counterfactual-subgraph",
    "/share/home/u20526/czx/worktrees/",
)
PROTECTED_NAMES = ("pi05", "goal-l4", "long-norm", "long-dense-distill", "libero")


def _run(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=60
    ).stdout


def _scontrol_fields(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=", text.strip()))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[match.group(1)] = text[start:end].strip()
    return result


def _gpu_count(record: dict[str, str]) -> int:
    values = " ".join(
        str(record.get(field) or "")
        for field in ("AllocTRES", "ReqTRES", "TresPerNode", "TRESPerNode")
    )
    matches = re.findall(r"(?:gres/gpu(?::[^=, ]+)?|gres:gpu(?::[^:, ]+)?)[=:](\d+)", values)
    if not matches:
        matches = re.findall(r"gres:gpu:[^: ,]+:(\d+)", values)
    return max((int(value) for value in matches), default=0)


def _project_owned(record: dict[str, str]) -> bool:
    workdir = str(record.get("WorkDir") or "")
    command = str(record.get("Command") or "")
    return any(workdir.startswith(prefix) for prefix in PROJECT_PREFIXES) and (
        "counterfactual-subgraph" in command
        or any(command.startswith(prefix) for prefix in PROJECT_PREFIXES)
    )


def _lane(record: dict[str, str]) -> str | None:
    text = " ".join(
        str(record.get(field) or "")
        for field in ("JobName", "Command", "WorkDir", "StdOut", "StdErr")
    ).lower()
    if "mutagenicity" in text or re.search(r"(?:^|[_/.-])mut(?:[_/.-]|$)", text):
        return "mut"
    if "bace" in text:
        return "bace"
    if "aids" in text:
        return "aids"
    return None


def audit_records(records: Iterable[dict[str, str]]) -> dict[str, Any]:
    active_project: list[dict[str, Any]] = []
    pending_mut: list[dict[str, Any]] = []
    pending_bace: list[dict[str, Any]] = []
    protected_other: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        state = str(record.get("JobState") or record.get("State") or "").upper()
        record["gpus"] = _gpu_count(record)
        record["gpu_lane"] = _lane(record)
        record["project_owned"] = _project_owned(record)
        name = str(record.get("JobName") or "").lower()
        protected = any(token in name for token in PROTECTED_NAMES)
        concise = {
            key: record.get(key)
            for key in (
                "JobId",
                "JobName",
                "JobState",
                "Reason",
                "Dependency",
                "WorkDir",
                "Command",
                "AllocTRES",
                "ReqTRES",
                "TresPerNode",
                "gpus",
                "gpu_lane",
                "project_owned",
            )
        }
        if state in {"RUNNING", "COMPLETING"} and record["project_owned"]:
            active_project.append(concise)
        elif state in {"RUNNING", "COMPLETING"} and protected:
            protected_other.append(concise)
        elif state in {"PENDING", "CONFIGURING"} and record["project_owned"]:
            if record["gpu_lane"] == "mut" and record["gpus"]:
                pending_mut.append(concise)
            if record["gpu_lane"] == "bace" and record["gpus"]:
                pending_bace.append(concise)
    active_mut = [
        row
        for row in active_project
        if row["gpu_lane"] == "mut" and int(row["gpus"] or 0) > 0
    ]
    active_bace = [
        row
        for row in active_project
        if row["gpu_lane"] == "bace" and int(row["gpus"] or 0) > 0
    ]
    active_aids = [
        row
        for row in active_project
        if row["gpu_lane"] == "aids" and int(row["gpus"] or 0) > 0
    ]
    result = {
        "schema_version": "project_gpu_lane_usage_v1",
        "active_project_jobs": active_project,
        "active_mut_gpu_jobs": active_mut,
        "active_bace_gpu_jobs": active_bace,
        "active_aids_gpu_jobs": active_aids,
        "active_mut_gpus": sum(int(row["gpus"] or 0) for row in active_mut),
        "active_bace_gpus": sum(int(row["gpus"] or 0) for row in active_bace),
        "active_aids_gpus": sum(int(row["gpus"] or 0) for row in active_aids),
        "pending_mut_gpu_requests": pending_mut,
        "pending_bace_gpu_requests": pending_bace,
        "protected_other_project_gpus": sum(
            int(row["gpus"] or 0) for row in protected_other
        ),
        "protected_other_project_jobs": protected_other,
        "total_account_gpus_visible": None,
    }
    result["active_mut_plus_bace_gpus"] = (
        result["active_mut_gpus"] + result["active_bace_gpus"]
    )
    result["active_mut_aids_gpus"] = (
        result["active_mut_gpus"] + result["active_aids_gpus"]
    )
    result["active_project_gpus"] = (
        result["active_mut_aids_gpus"] + result["active_bace_gpus"]
    )
    result["limits_pass"] = (
        result["active_mut_aids_gpus"] <= 1
        and result["active_bace_gpus"] <= 1
        and result["active_project_gpus"] <= 2
    )
    return result


def collect_scontrol_records(user: str) -> list[dict[str, str]]:
    rows = _run(["squeue", "-h", "-u", user, "-r", "-o", "%i|%T"]).splitlines()
    records: list[dict[str, str]] = []
    for row in rows:
        if not row.strip():
            continue
        job_id = row.split("|", 1)[0].strip()
        try:
            fields = _scontrol_fields(_run(["scontrol", "show", "job", "-dd", "-o", job_id]))
        except subprocess.CalledProcessError:
            continue
        records.append(fields)
    return records


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-json", required=True)
    parser.add_argument("--user", default=getpass.getuser())
    parser.add_argument("--records-json", help="Test/offline scontrol record list.")
    args = parser.parse_args(argv)
    if args.records_json:
        records = json.loads(Path(args.records_json).read_text(encoding="utf-8"))
    else:
        records = collect_scontrol_records(args.user)
    result = audit_records(records)
    _atomic_json(Path(args.emit_json).expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["limits_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
