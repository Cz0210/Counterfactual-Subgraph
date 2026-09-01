#!/usr/bin/env python3
"""Bridge one known T8 recovery chain to a standard T13 matrix locator.

This dataset-specific follower owns no scientific process and sends no signal.
It waits for the exact T8 dual-branch controller to persist its T13 relay,
reopens that relay's PASS terminal, and atomically creates the locator consumed
by the existing fast16 matrix publisher queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid


LOCATOR_SCHEMA = "fast16_matrix_cell_root_locator_v1"
HEARTBEAT_SCHEMA = "tastemolnet_t13_external_locator_heartbeat_v1"
RUN_MANIFEST_SCHEMA = "tastemolnet_t13_run_manifest_v1"
AUDIT_SCHEMA = "tastemolnet_t13_terminal_verification_v1"
CHECKPOINT_SCHEMA = "tastemolnet_t13_checkpoint_v1"
T13_STAGE = "T13_GLOBALGCE_FULL"
DATASET = "TasteMolNet"
METHOD = "GlobalGCE"
T8_PASS_STATE = "PASS_AND_T13_RELAY_PERSISTED"
T13_PASS_STATE = "PASS"


class TasteT13ExternalLocatorError(RuntimeError):
    """The fixed T8-to-T13 chain or its final terminal is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(value: str | Path, *, label: str, exists: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise TasteT13ExternalLocatorError(
            f"{label} must be an absolute non-symlink path"
        )
    try:
        return path.resolve(strict=exists)
    except OSError as exc:
        raise TasteT13ExternalLocatorError(f"{label} is absent: {path}") from exc


def _physical_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise TasteT13ExternalLocatorError(
            f"{label} is not a physical nonempty file: {path}"
        )
    return path


def _physical_dir(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise TasteT13ExternalLocatorError(
            f"{label} is not a physical directory: {path}"
        )
    return path.resolve(strict=True)


def _json(path: Path, *, label: str) -> dict[str, Any]:
    _physical_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TasteT13ExternalLocatorError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TasteT13ExternalLocatorError(f"{label} must be one JSON object")
    return dict(value)


def _one_line(path: Path, *, label: str) -> str:
    _physical_file(path, label=label)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TasteT13ExternalLocatorError(f"Cannot read {label}: {path}") from exc
    if len(lines) != 1 or not lines[0].strip():
        raise TasteT13ExternalLocatorError(f"{label} must contain one nonempty line")
    return lines[0].strip()


def _key_values(path: Path, *, label: str, expected: set[str]) -> dict[str, str]:
    _physical_file(path, label=label)
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            raise TasteT13ExternalLocatorError(f"{label} contains a malformed line")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            raise TasteT13ExternalLocatorError(f"{label} contains a duplicate/empty field")
        result[key] = value
    if set(result) != expected:
        raise TasteT13ExternalLocatorError(
            f"{label} fields changed: {sorted(result)}"
        )
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TasteT13ExternalLocatorError(f"Output may not be a symlink: {path}")
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            if path.exists():
                existing = _json(path, label="existing T13 locator")
                if existing != dict(payload):
                    raise TasteT13ExternalLocatorError(
                        "Existing T13 locator conflicts with the verified root"
                    )
            else:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    existing = _json(path, label="raced T13 locator")
                    if existing != dict(payload):
                        raise TasteT13ExternalLocatorError(
                            "Raced T13 locator conflicts with the verified root"
                        )
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _wait(state: str, **evidence: Any) -> dict[str, Any]:
    return {"state": state, **evidence}


def _live_controller(
    root: Path,
    *,
    label: str,
    proc_root: Path,
    expected_pid: int | None = None,
    require_controller_pid: bool = True,
) -> dict[str, Any]:
    launcher_pid_raw = _one_line(root / "launcher.pid", label=f"{label} launcher PID")
    if not launcher_pid_raw.isdigit() or int(launcher_pid_raw) <= 1:
        raise TasteT13ExternalLocatorError(f"{label} launcher PID is invalid")
    pid = int(launcher_pid_raw)
    if expected_pid is not None and pid != expected_pid:
        raise TasteT13ExternalLocatorError(f"{label} launcher PID binding changed")
    controller_pid_path = root / "controller.pid"
    if controller_pid_path.exists():
        controller_pid_raw = _one_line(
            controller_pid_path, label=f"{label} controller PID"
        )
        if not controller_pid_raw.isdigit() or int(controller_pid_raw) != pid:
            raise TasteT13ExternalLocatorError(f"{label} controller PID binding changed")
    elif require_controller_pid:
        raise TasteT13ExternalLocatorError(f"{label} controller PID is absent")
    stat_path = proc_root / str(pid) / "stat"
    try:
        stat_text = stat_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TasteT13ExternalLocatorError(
            f"{label} PID {pid} is not live while its state is nonterminal"
        ) from exc
    close = stat_text.rfind(")")
    if close <= 0:
        raise TasteT13ExternalLocatorError(f"{label} proc stat is malformed")
    try:
        stat_pid = int(stat_text[: stat_text.find(" ")])
        fields = stat_text[close + 2 :].split()
        process_state = fields[0]
        start_ticks = int(fields[19])
    except (ValueError, IndexError) as exc:
        raise TasteT13ExternalLocatorError(f"{label} proc stat is malformed") from exc
    if stat_pid != pid or process_state in {"Z", "X", "x"} or start_ticks <= 0:
        raise TasteT13ExternalLocatorError(
            f"{label} PID {pid} is not a live bound controller"
        )
    return {
        f"{label}_pid": pid,
        f"{label}_start_ticks": start_ticks,
        f"{label}_process_state": process_state,
    }


def _uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise TasteT13ExternalLocatorError(f"{label} is not a UUID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise TasteT13ExternalLocatorError(f"{label} is not a canonical UUIDv4")
    return value


def _validate_t13_terminal(root: Path) -> dict[str, Any]:
    if (root / "FAILED").exists():
        raise TasteT13ExternalLocatorError("T13 terminal contains FAILED")
    if _physical_file(root / "PASS", label="T13 PASS").read_bytes() != b"PASS\n":
        raise TasteT13ExternalLocatorError("T13 PASS marker bytes changed")
    if _physical_file(root / "SEALED", label="T13 SEALED").read_bytes() != b"SEALED\n":
        raise TasteT13ExternalLocatorError("T13 SEALED marker bytes changed")
    run_manifest_path = root / "run_manifest.json"
    audit_path = root / "final_artifact_audit.json"
    checkpoint_path = root / "checkpoint.json"
    run_manifest = _json(run_manifest_path, label="T13 run manifest")
    audit = _json(audit_path, label="T13 final audit")
    checkpoint = _json(checkpoint_path, label="T13 checkpoint")
    if (
        run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or run_manifest.get("dataset") != DATASET
        or run_manifest.get("method") != METHOD
        or run_manifest.get("stage") != T13_STAGE
        or run_manifest.get("status") != "PASS"
        or run_manifest.get("state") != "PASS"
        or run_manifest.get("run_complete") is not True
        or run_manifest.get("finalized") is not True
        or run_manifest.get("frozen") is not True
        or run_manifest.get("test_used_for_selection") is not False
        or run_manifest.get("threshold_fitted_on_test") is not False
        or run_manifest.get("independent_terminal_verification_required") is not False
        or run_manifest.get("worker_wrote_pass") is not False
        or run_manifest.get("terminal_verifier")
        != "separate_verify_only_invocation"
    ):
        raise TasteT13ExternalLocatorError("T13 run manifest is not exact PASS")
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("dataset") != DATASET
        or audit.get("method") != METHOD
        or audit.get("stage") != T13_STAGE
        or audit.get("status") != "PASS"
        or audit.get("passed") is not True
        or audit.get("audit_passed") is not True
        or audit.get("frozen") is not True
        or audit.get("registry_status") not in {"FROZEN_PASS", "ADOPTABLE_PASS"}
        or audit.get("registry_reason_codes") not in ([], None)
    ):
        raise TasteT13ExternalLocatorError("T13 final audit is not exact PASS")
    audit_sha = _sha256(audit_path)
    if run_manifest.get("final_artifact_audit_sha256") != audit_sha:
        raise TasteT13ExternalLocatorError("T13 final audit hash binding changed")
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("stage") != T13_STAGE
        or checkpoint.get("phase") != "PASS"
        or (checkpoint.get("detail") or {}).get("final_artifact_audit_sha256")
        != audit_sha
    ):
        raise TasteT13ExternalLocatorError("T13 final checkpoint is not PASS-bound")
    return {
        "terminal_root": str(root),
        "run_manifest_sha256": _sha256(run_manifest_path),
        "final_artifact_audit_sha256": audit_sha,
        "checkpoint_sha256": _sha256(checkpoint_path),
    }


def inspect_chain(
    *,
    t8_dual_controller_root: str | Path,
    control_root: str | Path,
    t13_output_base: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Return WAITING or one hash-closed READY observation without writing."""

    dual = _physical_dir(
        _absolute(t8_dual_controller_root, label="T8 dual controller", exists=True),
        label="T8 dual controller",
    )
    control = _physical_dir(
        _absolute(control_root, label="control root", exists=True),
        label="control root",
    )
    output_base = _physical_dir(
        _absolute(t13_output_base, label="T13 output base", exists=True),
        label="T13 output base",
    )
    if dual.parent != control:
        raise TasteT13ExternalLocatorError("T8 controller escaped the bound control root")
    proc = _absolute(proc_root, label="proc root")
    dual_state_path = dual / "state"
    if not dual_state_path.exists():
        dual_state = "STARTING"
    else:
        dual_state = _one_line(dual_state_path, label="T8 dual state")
    if "FAILED" in dual_state:
        raise TasteT13ExternalLocatorError(f"T8 dual chain failed: {dual_state}")
    dual_live = {}
    if dual_state != T8_PASS_STATE:
        dual_live = _live_controller(
            dual, label="t8_controller", proc_root=proc
        )
    if not dual_state_path.exists():
        return _wait(
            "WAITING_T8_CHAIN_STATE",
            t8_dual_controller_root=str(dual),
            **dual_live,
        )

    downstream = dual / "downstream-salvage"
    launch_path = downstream / "t13-relay-launch.txt"
    if not launch_path.exists():
        return _wait(
            "WAITING_T13_RELAY_LAUNCH",
            t8_dual_controller_root=str(dual),
            t8_dual_state=dual_state,
            **dual_live,
        )
    launch = _key_values(
        launch_path,
        label="T13 relay launch receipt",
        expected={"controller_id", "controller_pid", "controller_root"},
    )
    if not launch["controller_pid"].isdigit() or int(launch["controller_pid"]) <= 1:
        raise TasteT13ExternalLocatorError("T13 launcher PID is invalid")
    t13_root = _physical_dir(
        _absolute(launch["controller_root"], label="T13 controller root", exists=True),
        label="T13 controller root",
    )
    if (
        t13_root.parent != control
        or t13_root.name != launch["controller_id"]
        or not t13_root.name.startswith("tastemolnet-t13-after-t8-salvage-")
    ):
        raise TasteT13ExternalLocatorError("T13 relay escaped its launcher identity")
    launcher_pid = _one_line(t13_root / "launcher.pid", label="T13 launcher PID")
    if launcher_pid != launch["controller_pid"]:
        raise TasteT13ExternalLocatorError("T13 launcher PID binding changed")
    if not (t13_root / "controller.pid").exists():
        t13_live = _live_controller(
            t13_root,
            label="t13_controller",
            proc_root=proc,
            expected_pid=int(launch["controller_pid"]),
            require_controller_pid=False,
        )
        return _wait(
            "WAITING_T13_CONTROLLER_START",
            t8_dual_controller_root=str(dual),
            t13_controller_root=str(t13_root),
            **dual_live,
            **t13_live,
        )
    controller_pid = _one_line(t13_root / "controller.pid", label="T13 controller PID")
    if controller_pid != launch["controller_pid"]:
        raise TasteT13ExternalLocatorError("T13 controller PID binding changed")
    launch_env_path = t13_root / "launch.env"
    if not launch_env_path.exists():
        t13_live = _live_controller(
            t13_root,
            label="t13_controller",
            proc_root=proc,
            expected_pid=int(controller_pid),
        )
        return _wait(
            "WAITING_T13_GPU_ADMISSION",
            t8_dual_controller_root=str(dual),
            t13_controller_root=str(t13_root),
            **dual_live,
            **t13_live,
        )
    launch_env = _key_values(
        launch_env_path,
        label="T13 launch env",
        expected={"attempt_id", "output_root", "gpu_index", "gpu_uuid", "t8_pass_root"},
    )
    attempt_id = _uuid4(launch_env["attempt_id"], label="T13 attempt ID")
    if launch_env["gpu_index"] != "1" or not launch_env["gpu_uuid"].startswith("GPU-"):
        raise TasteT13ExternalLocatorError("T13 GPU binding changed")
    output_root = _absolute(launch_env["output_root"], label="T13 output root")
    if output_root.parent != output_base or output_root.name != f"attempt-{attempt_id}":
        raise TasteT13ExternalLocatorError("T13 output escaped the fixed output base/attempt")
    t8_pass_root = _absolute(launch_env["t8_pass_root"], label="T8 managed PASS root")

    t13_state_path = t13_root / "state"
    if t13_state_path.exists():
        t13_state = _one_line(t13_state_path, label="T13 state")
        if "FAILED" in t13_state:
            raise TasteT13ExternalLocatorError(f"T13 relay failed: {t13_state}")
    else:
        t13_state = "STARTING"
    completed_path = t13_root / "completed_output_root"
    if not completed_path.exists():
        t13_live = _live_controller(
            t13_root,
            label="t13_controller",
            proc_root=proc,
            expected_pid=int(controller_pid),
        )
        return _wait(
            "WAITING_T13_PASS",
            t8_dual_controller_root=str(dual),
            t8_dual_state=dual_state,
            t13_controller_root=str(t13_root),
            t13_state=t13_state,
            t13_output_root=str(output_root),
            **dual_live,
            **t13_live,
        )

    completed_output = _absolute(
        _one_line(completed_path, label="T13 completed output root"),
        label="T13 completed output root",
        exists=True,
    )
    if completed_output != output_root:
        raise TasteT13ExternalLocatorError("T13 completed output differs from launch env")
    if dual_state != T8_PASS_STATE:
        return _wait(
            "WAITING_T8_CHAIN_FINAL_RECEIPT",
            t8_dual_controller_root=str(dual),
            t8_dual_state=dual_state,
            t13_controller_root=str(t13_root),
            t13_output_root=str(output_root),
            **dual_live,
        )
    dual_t8_root = _physical_dir(
        _absolute(
            _one_line(dual / "completed_t8_root", label="T8 completed managed root"),
            label="T8 completed managed root",
            exists=True,
        ),
        label="T8 completed managed root",
    )
    downstream_t8_root = _physical_dir(
        _absolute(
            _one_line(
                downstream / "completed_t8_root",
                label="downstream T8 managed root",
            ),
            label="downstream T8 managed root",
            exists=True,
        ),
        label="downstream T8 managed root",
    )
    if dual_t8_root != downstream_t8_root or dual_t8_root != t8_pass_root:
        raise TasteT13ExternalLocatorError("T8 managed PASS root binding changed")
    if t13_state != T13_PASS_STATE:
        # The relay intentionally persists completed_output_root before its
        # final atomic PASS heartbeat/state update.  Observing that tiny
        # publication window is not a scientific failure; keep following the
        # exact bound controller until PASS becomes visible.  Explicit FAILED
        # states were rejected above.
        t13_live = _live_controller(
            t13_root,
            label="t13_controller",
            proc_root=proc,
            expected_pid=int(controller_pid),
        )
        return _wait(
            "WAITING_T13_CONTROLLER_PASS",
            t8_dual_controller_root=str(dual),
            t8_dual_state=dual_state,
            t13_controller_root=str(t13_root),
            t13_state=t13_state,
            t13_output_root=str(output_root),
            completed_output_root=str(completed_output),
            **t13_live,
        )
    heartbeat_path = t13_root / "heartbeat.json"
    if not heartbeat_path.exists():
        t13_live = _live_controller(
            t13_root,
            label="t13_controller",
            proc_root=proc,
            expected_pid=int(controller_pid),
        )
        return _wait(
            "WAITING_T13_PASS_HEARTBEAT",
            t8_dual_controller_root=str(dual),
            t8_dual_state=dual_state,
            t13_controller_root=str(t13_root),
            t13_state=t13_state,
            t13_output_root=str(output_root),
            **t13_live,
        )
    heartbeat = _json(heartbeat_path, label="T13 heartbeat")
    if (
        heartbeat.get("controller_pid") != int(controller_pid)
        or heartbeat.get("phase") != T13_PASS_STATE
        or heartbeat.get("science_pid") != 0
    ):
        raise TasteT13ExternalLocatorError("T13 terminal heartbeat is not exact PASS")
    terminal = _physical_dir(completed_output, label="T13 completed output root")
    evidence = _validate_t13_terminal(terminal)
    return {
        "state": "READY",
        "t8_dual_controller_root": str(dual),
        "t8_dual_state": dual_state,
        "t8_managed_pass_root": str(dual_t8_root),
        "t13_controller_id": launch["controller_id"],
        "t13_controller_pid": int(controller_pid),
        "t13_controller_root": str(t13_root),
        "t13_attempt_id": attempt_id,
        "t13_gpu_uuid": launch_env["gpu_uuid"],
        "t13_launch_receipt_sha256": _sha256(launch_path),
        "t13_launch_env_sha256": _sha256(launch_env_path),
        **evidence,
    }


def run_follower(
    *,
    t8_dual_controller_root: str | Path,
    control_root: str | Path,
    t13_output_base: str | Path,
    locator_path: str | Path,
    heartbeat_path: str | Path,
    poll_seconds: int,
    once: bool = False,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    if not 5 <= poll_seconds <= 3600:
        raise TasteT13ExternalLocatorError("poll_seconds must be in [5, 3600]")
    locator = _absolute(locator_path, label="locator path")
    heartbeat = _absolute(heartbeat_path, label="heartbeat path")
    while True:
        observed = inspect_chain(
            t8_dual_controller_root=t8_dual_controller_root,
            control_root=control_root,
            t13_output_base=t13_output_base,
            proc_root=proc_root,
        )
        heartbeat_payload = {
            "schema_version": HEARTBEAT_SCHEMA,
            "pid": os.getpid(),
            "locator_path": str(locator),
            **observed,
            "updated_epoch": time.time(),
        }
        _atomic_json(heartbeat, heartbeat_payload, replace=True)
        if observed["state"] == "READY":
            locator_payload = {
                "schema_version": LOCATOR_SCHEMA,
                "status": "READY",
                "dataset": DATASET,
                "method": METHOD,
                "terminal_root": observed["terminal_root"],
            }
            _atomic_json(locator, locator_payload, replace=False)
            final = dict(heartbeat_payload)
            final["state"] = "PASS"
            final["locator_sha256"] = _sha256(locator)
            final["updated_epoch"] = time.time()
            _atomic_json(heartbeat, final, replace=True)
            return final
        if once:
            return heartbeat_payload
        time.sleep(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--t8-dual-controller-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--t13-output-base", required=True)
    parser.add_argument("--locator-path", required=True)
    parser.add_argument("--heartbeat-path", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.config != "configs/hpc.yaml":
        raise SystemExit("--config must be configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise SystemExit(
            "T13 locator requires exactly --set inference.fallback_to_heuristic=false"
        )
    result = run_follower(
        t8_dual_controller_root=args.t8_dual_controller_root,
        control_root=args.control_root,
        t13_output_base=args.t13_output_base,
        locator_path=args.locator_path,
        heartbeat_path=args.heartbeat_path,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["state"] == "PASS":
        print("[TASTE_T13_EXTERNAL_LOCATOR_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
