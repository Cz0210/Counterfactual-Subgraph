#!/usr/bin/env python3
"""Consume one sealed Mut next action and execute exactly one successor lane."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_first_divergence_v1 import file_sha256  # noqa: E402
from src.utils.autodl_mut_next_stage_executor_v1 import (  # noqa: E402
    TERMINAL_SCHEMA,
    MutNextStageError,
    acquire_lease,
    atomic_json,
    consume_next_action_once,
    run_stage,
    validate_successor_spec,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_executor(
    *, task_spec: Path, poll_seconds: int = 60, once: bool = False
) -> dict[str, Any]:
    spec = validate_successor_spec(_json(task_spec), check_files=True)
    runtime = Path(str(spec["runtime_root"]))
    if runtime.exists() or runtime.is_symlink():
        raise FileExistsError(f"Mut next-stage runtime must be fresh: {runtime}")
    runtime.mkdir(parents=True, exist_ok=False)
    lease = acquire_lease(Path(str(spec["lease_path"])))
    pid = os.getpid()
    stopped = False
    state: dict[str, Any] = {
        "state": "WAITING_FOR_NEXT_ACTION",
        "stage": None,
        "science_pid": None,
        "lane": None,
        "route_b_started": False,
        "fresh_50k_started": False,
    }

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    previous = {
        signum: signal.signal(signum, _stop)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    sequence = 0

    def heartbeat(stage: str | None = None, science_pid: int | None = None) -> None:
        nonlocal sequence
        sequence += 1
        if stage is not None:
            state["stage"] = stage
        state["science_pid"] = science_pid
        atomic_json(
            runtime / "heartbeat.json",
            {
                "schema_version": "mut_next_stage_executor_heartbeat_v1",
                "task_id": spec["task_id"],
                "pid": pid,
                "state": state["state"],
                "lane": state["lane"],
                "stage": state["stage"],
                "science_pid": state["science_pid"],
                "task_spec": str(task_spec),
                "task_spec_sha256": spec["spec_sha256"],
                "sequence": sequence,
                "route_b_started": state["route_b_started"],
                "fresh_50k_started": state["fresh_50k_started"],
                "written_at": _now(),
            },
        )

    atomic_json(
        runtime / "owner_pid.json",
        {
            "schema_version": "mut_next_stage_executor_owner_v1",
            "task_id": spec["task_id"],
            "pid": pid,
            "task_spec": str(task_spec),
            "task_spec_sha256": spec["spec_sha256"],
            "started_at": _now(),
        },
    )
    heartbeat()
    try:
        action_path = Path(str(spec["next_action_path"]))
        predecessor_terminal = Path(str(spec["predecessor_terminal"]))
        while not (action_path.is_file() and predecessor_terminal.is_file()):
            heartbeat()
            if stopped or once:
                return {
                    "status": "WAITING_FOR_NEXT_ACTION",
                    "runtime_root": str(runtime),
                }
            time.sleep(poll_seconds)
        lane, _action, consumed, consumption = consume_next_action_once(
            action_path=action_path,
            predecessor_terminal=predecessor_terminal,
            task_spec_sha256=str(spec["spec_sha256"]),
            expected_task_id=str(spec["predecessor_task_id"]),
            expected_task_spec=Path(str(spec["predecessor_task_spec"])),
        )
        state["lane"] = lane
        atomic_json(runtime / "next_action_consumption.json", consumption)
        if lane == "ENGINEERING_REPAIR":
            state["state"] = "BLOCKED_ENGINEERING_REPAIR"
            terminal = {
                "schema_version": TERMINAL_SCHEMA,
                "status": state["state"],
                "lane": lane,
                "consumed_next_action": str(consumed),
                "fresh_50k_started": False,
                "route_b_started": False,
                "completed_stages": [],
                "written_at": _now(),
            }
            atomic_json(runtime / "terminal.json", terminal)
            heartbeat()
            return terminal
        pipeline = (
            spec["adoption_pipeline"] if lane == "ADOPTION" else spec["route_b_pipeline"]
        )
        state["state"] = "RUNNING_SUCCESSOR_PIPELINE"
        receipts: list[dict[str, Any]] = []
        for index, stage in enumerate(pipeline):
            if stopped:
                raise MutNextStageError("executor received graceful stop between stages")
            heartbeat(str(stage["stage"]), None)
            runtime_stage = dict(stage)
            if lane == "ROUTE_B":
                runtime_stage["environment"] = {
                    **dict(stage["environment"]),
                    "MUT_NEXT_ACTION_CONSUMED_PATH": str(consumed),
                    "MUT_NEXT_ACTION_CONSUMPTION_RECEIPT": str(
                        runtime / "next_action_consumption.json"
                    ),
                }
            receipt = run_stage(
                runtime_stage,
                log_path=runtime / "logs" / f"{index:02d}-{stage['stage']}.log",
                progress=heartbeat,
            )
            receipts.append(receipt)
            atomic_json(runtime / "stages" / f"{index:02d}-{stage['stage']}.json", receipt)
            state["route_b_started"] = (
                state["route_b_started"] or receipt.get("route_b_started") is True
            )
            state["fresh_50k_started"] = (
                state["fresh_50k_started"]
                or receipt.get("fresh_50k_started") is True
            )
            if str(receipt["terminal_status"]).startswith("BLOCKED_"):
                raise MutNextStageError(
                    f"{stage['stage']} reported {receipt['terminal_status']}"
                )
        if lane == "ROUTE_B" and not (
            state["route_b_started"] and state["fresh_50k_started"]
        ):
            raise MutNextStageError(
                "Route-B pipeline returned without proving fresh generation started"
            )
        if lane == "ADOPTION":
            locator = Path(str(spec["publisher_locator"]))
            if locator.is_symlink() or not locator.is_file():
                raise MutNextStageError("canonical Mut publisher locator was not produced")
            locator_sha = file_sha256(locator)
        else:
            locator_sha = None
        state["state"] = "PASS"
        terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "status": "PASS",
            "lane": lane,
            "consumed_next_action": str(consumed),
            "consumption_receipt_sha256": consumption["receipt_sha256"],
            "completed_stages": receipts,
            "publisher_id": spec["publisher_id"],
            "publisher_locator": spec["publisher_locator"],
            "publisher_locator_sha256": locator_sha,
            "fresh_50k_started": state["fresh_50k_started"],
            "route_b_started": state["route_b_started"],
            "pair_store_recomputed": False if lane == "ADOPTION" else None,
            "dbscan_recomputed": False if lane == "ADOPTION" else None,
            "completed_at": _now(),
        }
        atomic_json(runtime / "terminal.json", terminal)
        heartbeat()
        return terminal
    except BaseException as exc:
        state["state"] = "BLOCKED"
        terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "status": "BLOCKED",
            "lane": state["lane"],
            "stage": state["stage"],
            "error": f"{type(exc).__name__}: {exc}",
            "route_b_started": state["route_b_started"],
            "fresh_50k_started": state["fresh_50k_started"],
            "automatic_route_b_after_engineering_failure": False,
            "failed_at": _now(),
        }
        atomic_json(runtime / "terminal.json", terminal)
        heartbeat()
        raise
    finally:
        lease.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--task-spec", type=_absolute, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not 30 <= args.poll_seconds <= 3600:
        raise ValueError("poll interval must be in [30, 3600]")
    result = run_executor(
        task_spec=args.task_spec, poll_seconds=args.poll_seconds, once=args.once
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] == "PASS":
        print("[MUT_SUCCESSOR_CHAIN_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
