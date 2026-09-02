#!/usr/bin/env python3
"""Minimal priority scheduler for the remaining main cells and ablations.

The scheduler deliberately delegates scientific work to dataset-specific
launchers.  Its only responsibilities are single-owner dispatch, durable
heartbeat/status, and enforcing MAIN > LLM > GNN admission gates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
from uuid import uuid4


SCHEMA = "main_and_ablations_controller_v1"
HEARTBEAT_SCHEMA = "main_and_ablations_heartbeat_v1"
DEFAULT_POLL_SECONDS = 30
COMPONENTS = (
    "mut_continuation",
    "t14_convergence_auditor",
    "t8_valid_zero_finalizer",
    "llm_ablation",
    "gnn_ablation",
)
DEFAULT_LAUNCHERS = {
    "mut_continuation": "scripts/autodl/launch_mut_throttled_continuation_v1.sh",
    "t14_convergence_auditor": "scripts/autodl/launch_t14_external_convergence_auditor_v1.sh",
    "t8_valid_zero_finalizer": "scripts/autodl/launch_tastemolnet_globalgce_valid_zero_finalizer_v1.sh",
    "llm_ablation": "scripts/autodl/launch_llm_ablation_core_v1.sh",
    "gnn_ablation": "scripts/autodl/launch_gnn_five_backbone_ablation_v1.sh",
}
LAUNCHER_ENV = {
    "mut_continuation": "MUT_CONTINUATION_LAUNCHER",
    "t14_convergence_auditor": "T14_AUDITOR_LAUNCHER",
    "t8_valid_zero_finalizer": "T8_ZERO_FINALIZER_LAUNCHER",
    "llm_ablation": "LLM_ABLATION_LAUNCHER",
    "gnn_ablation": "GNN_ABLATION_LAUNCHER",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _required_env(name: str, expected: str) -> None:
    observed = os.environ.get(name, expected)
    if observed != expected:
        raise ValueError(f"{name} must equal {expected!r}, observed {observed!r}")


def validate_policy() -> dict[str, Any]:
    for name, expected in (
        ("RUN_MAIN_TABLE", "1"),
        ("RUN_LLM_ABLATION", "1"),
        ("RUN_GNN_ABLATION", "1"),
        ("MAIN_TABLE_PRIORITY", "100"),
        ("LLM_ABLATION_PRIORITY", "50"),
        ("GNN_ABLATION_PRIORITY", "20"),
        ("ALLOW_MUT_CONTINUATION_RECOVERY", "1"),
        ("MUT_CPU_WORKERS", "2"),
        ("MUT_BASELINE_WINDOW_SECONDS", "1800"),
        ("MUT_SLOWDOWN_THRESHOLD", "0.15"),
        ("MUT_SLOWDOWN_SUSTAIN_SECONDS", "1200"),
        ("ALLOW_T14_EXTERNAL_CONVERGENCE_AUDITOR", "1"),
        ("ALLOW_TASTE_GLOBALGCE_VALID_ZERO_RULE_RESULT", "1"),
        ("T8_RECOVERY_MAX_ATTEMPTS", "1"),
        ("LLM_EARLY_START_MIN_MATRIX", "13"),
        ("LLM_EARLY_START_IDLE_SECONDS", "1200"),
        ("LLM_MAX_EARLY_GPUS", "1"),
        ("RUN_MATCHED_SFT_AUXILIARY_STUDY", "0"),
        ("GNN_START_AFTER_MATRIX", "16"),
        ("GNN_PRIMARY_SEEDS", "7"),
        ("GNN_MAX_CONCURRENT_GPUS", "2"),
        ("RUN_GRAPH_MAMBA", "0"),
    ):
        _required_env(name, expected)
    llm_variants = os.environ.get(
        "LLM_CORE_VARIANTS",
        "BRICS_FIXED,CHEMLLM_7B_OFF_THE_SHELF,CHEMLLM_7B_PPO_MAIN,CHEMLLM_2B_OFF_THE_SHELF",
    )
    gnn_backbones = os.environ.get("GNN_BACKBONES", "gine,gin,gcn,gatv2,gps")
    if llm_variants.split(",") != [
        "BRICS_FIXED",
        "CHEMLLM_7B_OFF_THE_SHELF",
        "CHEMLLM_7B_PPO_MAIN",
        "CHEMLLM_2B_OFF_THE_SHELF",
    ]:
        raise ValueError("LLM_CORE_VARIANTS differs from the authorized core rows")
    if gnn_backbones.split(",") != ["gine", "gin", "gcn", "gatv2", "gps"]:
        raise ValueError("GNN_BACKBONES differs from the authorized five rows")
    return {
        "main_priority": 100,
        "llm_priority": 50,
        "gnn_priority": 20,
        "llm_variants": llm_variants.split(","),
        "gnn_backbones": gnn_backbones.split(","),
        "graph_mamba_run_enabled": False,
        "matched_sft_auxiliary_run_enabled": False,
    }


def _matrix(path: Path) -> dict[str, Any]:
    pointer = _load_json(path)
    if pointer.get("schema_version") != "fast16_matrix_authority_pointer_v1":
        raise ValueError("matrix authority pointer schema changed")
    count = pointer.get("latest_count")
    cells = pointer.get("applied_cells")
    if not isinstance(count, int) or not 0 <= count <= 16:
        raise ValueError("matrix latest_count is invalid")
    if not isinstance(cells, list) or len(cells) != count or len(set(cells)) != count:
        raise ValueError("matrix applied_cells do not match latest_count")
    return {
        "count": count,
        "cells": tuple(str(item) for item in cells),
        "pointer": pointer,
    }


def _cell_present(matrix: Mapping[str, Any], dataset: str, method: str) -> bool:
    wanted = f"{dataset}/{method}".casefold()
    return any(str(item).casefold() == wanted for item in matrix["cells"])


def _process_identity(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        return {"pid": pid, "alive": False}
    try:
        raw = (proc / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        ticks = int(fields[19])
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (OSError, ValueError, IndexError):
        return {"pid": pid, "alive": False, "unreadable": True}
    return {"pid": pid, "alive": True, "start_ticks": ticks, "command": command}


def _launcher(project_root: Path, component: str) -> Path:
    configured = os.environ.get(LAUNCHER_ENV[component], DEFAULT_LAUNCHERS[component])
    candidate = Path(configured)
    path = candidate if candidate.is_absolute() else project_root / candidate
    if path.is_symlink():
        raise ValueError(f"{component} launcher must not be a symlink")
    return path


def _latest_launch(state_root: Path, component: str) -> dict[str, Any] | None:
    path = state_root / "components" / component / "launch.json"
    if not path.is_file():
        return None
    value = _load_json(path)
    pid = value.get("launcher_pid")
    if isinstance(pid, int):
        value["launcher_process"] = _process_identity(pid)
    return value


def _dispatch(
    state_root: Path,
    project_root: Path,
    component: str,
    *,
    reason: str,
    dry_run: bool,
) -> dict[str, Any]:
    previous = _latest_launch(state_root, component)
    if previous is not None:
        return {"state": "ALREADY_DISPATCHED", "launch": previous}
    launcher = _launcher(project_root, component)
    if not launcher.is_file():
        return {"state": "BLOCKED_MISSING_LAUNCHER", "launcher": str(launcher)}
    if dry_run:
        return {"state": "WOULD_DISPATCH", "launcher": str(launcher), "reason": reason}
    log_root = state_root / "components" / component
    log_root.mkdir(parents=True, exist_ok=True)
    log_handle = (log_root / "launcher.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            ["bash", str(launcher)],
            cwd=project_root,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    receipt = {
        "schema_version": "main_and_ablations_component_launch_v1",
        "component": component,
        "launcher": str(launcher.resolve(strict=True)),
        "launcher_pid": process.pid,
        "launcher_start_ticks": _process_identity(process.pid).get("start_ticks"),
        "reason": reason,
        "dispatched_at": _utc_now(),
    }
    _atomic_json(log_root / "launch.json", receipt)
    return {"state": "DISPATCHED", "launch": receipt}


def _explicit_pass(path_text: str | None) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        return False
    try:
        value = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return value.get("status") == "PASS" or value.get("state") == "PASS"


def _main_ready_waiting() -> tuple[bool, str]:
    configured = os.environ.get("MAIN_READY_QUEUE")
    if not configured:
        return True, "MAIN_READY_QUEUE_UNBOUND"
    path = Path(configured)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        return True, "MAIN_READY_QUEUE_UNAVAILABLE"
    value = _load_json(path)
    rows = value.get("ready_waiting_gpu", value.get("tasks", []))
    if isinstance(rows, int):
        return rows > 0, f"ready_waiting_gpu={rows}"
    if isinstance(rows, list):
        waiting = [row for row in rows if not isinstance(row, Mapping) or row.get("state") == "READY_WAITING_GPU"]
        return bool(waiting), f"ready_waiting_gpu={len(waiting)}"
    return True, "MAIN_READY_QUEUE_SCHEMA_UNKNOWN"


def observe_and_dispatch(
    *,
    state_root: Path,
    project_root: Path,
    matrix_path: Path,
    policy: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    matrix = _matrix(matrix_path)
    main_ready, main_ready_reason = _main_ready_waiting()
    components: dict[str, Any] = {}

    if _cell_present(matrix, "Mutagenicity", "ComRecGC"):
        components["mut_continuation"] = {"state": "MAIN_CELL_PASS"}
    else:
        components["mut_continuation"] = _dispatch(
            state_root,
            project_root,
            "mut_continuation",
            reason="P0_MUT_CELL_MISSING",
            dry_run=dry_run,
        )
    if _cell_present(matrix, "TasteMolNet", "ComRecGC"):
        components["t14_convergence_auditor"] = {"state": "MAIN_CELL_PASS"}
    else:
        components["t14_convergence_auditor"] = _dispatch(
            state_root,
            project_root,
            "t14_convergence_auditor",
            reason="P0_TASTE_COMRECGC_CELL_MISSING",
            dry_run=dry_run,
        )
    if _cell_present(matrix, "TasteMolNet", "GlobalGCE"):
        components["t8_valid_zero_finalizer"] = {"state": "MAIN_CELL_PASS"}
    else:
        components["t8_valid_zero_finalizer"] = _dispatch(
            state_root,
            project_root,
            "t8_valid_zero_finalizer",
            reason="P0_TASTE_GLOBALGCE_CELL_MISSING",
            dry_run=dry_run,
        )

    llm_blockers: list[str] = []
    if matrix["count"] < 13:
        llm_blockers.append("MATRIX_LT_13")
    if not _cell_present(matrix, "Mutagenicity", "ComRecGC"):
        llm_blockers.append("MUT_STILL_REQUIRES_PRIORITY")
    if main_ready:
        llm_blockers.append(main_ready_reason)
    llm_gate_receipt = os.environ.get("LLM_EARLY_GATE_RECEIPT")
    if not _explicit_pass(llm_gate_receipt):
        llm_blockers.append("LLM_EARLY_GATE_RECEIPT_NOT_PASS")
    if llm_blockers:
        components["llm_ablation"] = {
            "state": "BLOCKED_MAIN_PRIORITY",
            "blockers": llm_blockers,
        }
    else:
        components["llm_ablation"] = _dispatch(
            state_root,
            project_root,
            "llm_ablation",
            reason="P1_EARLY_GATE_PASS",
            dry_run=dry_run,
        )

    gnn_blockers: list[str] = []
    if matrix["count"] != 16:
        gnn_blockers.append("MATRIX_NOT_16")
    for label, variable in (
        ("FINAL_MATRIX_AUDIT", "FINAL_MATRIX_AUDIT_RECEIPT"),
        ("FINAL_FIGURE3", "FINAL_FIGURE3_RECEIPT"),
        ("FINAL_FIGURE4", "FINAL_FIGURE4_RECEIPT"),
        ("FINAL_TABLE2", "FINAL_TABLE2_RECEIPT"),
    ):
        if not _explicit_pass(os.environ.get(variable)):
            gnn_blockers.append(f"{label}_NOT_PASS")
    if gnn_blockers:
        components["gnn_ablation"] = {
            "state": "BLOCKED_WAITING_FINAL_MAIN",
            "blockers": gnn_blockers,
        }
    else:
        components["gnn_ablation"] = _dispatch(
            state_root,
            project_root,
            "gnn_ablation",
            reason="P2_MAIN_16_AND_FINAL_ARTIFACTS_PASS",
            dry_run=dry_run,
        )

    return {
        "schema_version": HEARTBEAT_SCHEMA,
        "written_at": _utc_now(),
        "controller_pid": os.getpid(),
        "matrix_authority": str(matrix_path),
        "matrix_complete_cells": matrix["count"],
        "matrix_total_cells": 16,
        "matrix_cells": list(matrix["cells"]),
        "priorities": dict(policy),
        "main_ready_waiting_gpu": main_ready,
        "main_ready_queue_reason": main_ready_reason,
        "components": components,
        "dry_run": dry_run,
    }


def run(args: argparse.Namespace) -> int:
    policy = validate_policy()
    project_root = Path(__file__).resolve().parents[2]
    expected_config = project_root / "configs/hpc.yaml"
    if args.config.resolve(strict=True) != expected_config.resolve(strict=True):
        raise ValueError("--config must bind this checkout's configs/hpc.yaml")
    state_root = args.state_root.absolute()
    matrix_path = args.matrix_authority.absolute()
    if args.poll_seconds != DEFAULT_POLL_SECONDS:
        raise ValueError("scheduler poll interval must remain 30 seconds")
    state_root.mkdir(parents=True, exist_ok=True)
    lock = (state_root / "controller.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another main-and-ablations controller owns this root") from exc
    controller_id = os.environ.get(
        "MAIN_AND_ABLATIONS_CONTROLLER_ID", f"main_and_ablations_v1_{uuid4()}"
    )
    receipt_path = state_root / "controller_receipt.json"
    if not receipt_path.exists():
        _atomic_json(
            receipt_path,
            {
                "schema_version": SCHEMA,
                "controller_id": controller_id,
                "controller_pid": os.getpid(),
                "execution_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "created_at": _utc_now(),
                "matrix_authority": str(matrix_path),
                "policy": policy,
            },
        )
    else:
        controller_id = str(_load_json(receipt_path)["controller_id"])

    stopped = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    sequence = 0
    try:
        while True:
            sequence += 1
            payload = observe_and_dispatch(
                state_root=state_root,
                project_root=project_root,
                matrix_path=matrix_path,
                policy=policy,
                dry_run=args.dry_run,
            )
            payload.update({"controller_id": controller_id, "sequence": sequence})
            _atomic_json(state_root / "heartbeat.json", payload)
            if args.once or stopped:
                return 0
            time.sleep(args.poll_seconds)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--matrix-authority", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"[MAIN_AND_ABLATIONS_BLOCKED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
