#!/usr/bin/env python3
"""One-shot owner for Taste T14 Route C parity, promotion, and full science."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_t14_route_c_fresh import (  # noqa: E402
    EARLY_CHECKPOINT_STEPS,
    M_MAX,
    T14RouteCFreshError,
    atomic_json,
    audit_route_c_convergence,
    audit_route_c_matrix_cell_absent,
    build_spec,
    compare_step_ledgers,
    load_spec,
    validate_checkpoint_boundary,
    validate_promotion_receipt,
    validate_route_c_convergence_receipt,
    promotion_receipt_path,
    write_route_c_convergence_receipt,
    write_spec,
)
from src.baselines.tastemolnet_t14_route_c_continuation import (  # noqa: E402
    launch_continuation_owner,
    load_continuation_spec,
    publish_generation_handoff,
)


OWNER_SCHEMA = "tastemolnet_t14_route_c_owner_v1"
WAITING_EXIT = 75


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _counter(path: Path) -> int:
    value = path.read_text(encoding="utf-8").strip()
    if value == "max":
        return 2**63 - 1
    return int(value)


def _rss_bytes(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    try:
        rows = (proc_root / str(pid) / "status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for row in rows:
        if row.startswith("VmRSS:"):
            return int(row.split()[1]) * 1024
    return 0


def _proc_identity(pid: int, *, proc_root: Path = Path("/proc")) -> tuple[int, int]:
    """Return (parent PID, start ticks), parsing comm parentheses safely."""

    raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise ValueError(f"malformed /proc stat for PID {pid}")
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        raise ValueError(f"short /proc stat for PID {pid}")
    return int(fields[1]), int(fields[19])


def _start_ticks(pid: int) -> int:
    return _proc_identity(pid)[1]


def _process_tree_snapshot(
    root_pid: int, *, proc_root: Path = Path("/proc")
) -> list[dict[str, int]]:
    """Snapshot one exact descendant tree without fuzzy command matching."""

    identities: dict[int, tuple[int, int]] = {}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        try:
            identities[pid] = _proc_identity(pid, proc_root=proc_root)
        except (OSError, ValueError, IndexError):
            continue
    if root_pid not in identities:
        return []
    children: dict[int, list[int]] = {}
    for pid, (parent, _ticks) in identities.items():
        children.setdefault(parent, []).append(pid)
    rows: list[dict[str, int]] = []
    queue = [(int(root_pid), 0)]
    observed: set[int] = set()
    while queue:
        pid, depth = queue.pop(0)
        if pid in observed or pid not in identities:
            continue
        observed.add(pid)
        parent, ticks = identities[pid]
        rows.append(
            {
                "pid": pid,
                "ppid": parent,
                "start_ticks": ticks,
                "depth": depth,
                "rss_bytes": _rss_bytes(pid, proc_root=proc_root),
            }
        )
        queue.extend((child, depth + 1) for child in sorted(children.get(pid, ())))
    return rows


def _process_tree_rss_bytes(root_pid: int) -> tuple[int, list[dict[str, int]]]:
    rows = _process_tree_snapshot(root_pid)
    return sum(row["rss_bytes"] for row in rows), rows


def _terminate_process_tree(root_pid: int, *, expected_root_ticks: int) -> None:
    """SIGTERM only the snapshotted exact descendants, deepest first."""

    rows = _process_tree_snapshot(root_pid)
    root = next((row for row in rows if row["pid"] == root_pid), None)
    if root is None:
        return
    if root["start_ticks"] != int(expected_root_ticks):
        raise T14RouteCFreshError("Route C science owner PID identity changed")
    for row in sorted(rows, key=lambda value: (value["depth"], value["pid"]), reverse=True):
        try:
            if _start_ticks(row["pid"]) != row["start_ticks"]:
                raise T14RouteCFreshError("Route C descendant PID identity changed")
            os.kill(row["pid"], signal.SIGTERM)
        except ProcessLookupError:
            continue


def _phase(owner_root: Path, **fields: Any) -> None:
    atomic_json(
        owner_root / "heartbeat.json",
        {
            "schema_version": OWNER_SCHEMA,
            "owner_pid": os.getpid(),
            "owner_start_ticks": _start_ticks(os.getpid()),
            "updated_at": _utc_now(),
            **fields,
        },
    )


def _child_spec(
    master: Mapping[str, Any],
    *,
    owner_root: Path,
    role: str,
    storage_mode: str,
) -> tuple[dict[str, Any], Path]:
    attempt = str(uuid4())
    child_owner = owner_root / "canaries" / role.lower() / attempt
    output = child_owner / f"science-{attempt}"
    path = child_owner / "route_c_spec.json"
    memory = master["memory"]
    spec = build_spec(
        attempt_uuid=attempt,
        execution_commit=str(master["execution_commit"]),
        python=Path(master["python"]),
        science_wrapper=Path(master["science_wrapper"]),
        owner_entrypoint=Path(master["owner_entrypoint"]),
        output_root=output,
        owner_root=child_owner,
        cgroup_limit_path=Path(memory["cgroup_limit_path"]),
        cgroup_current_path=Path(memory["cgroup_current_path"]),
        cgroup_failcnt_path=Path(memory["cgroup_failcnt_path"]),
        forbidden_legacy_root=Path(master["forbidden_legacy_root"]),
        science_environment=master["science_environment"],
        storage_mode=storage_mode,
        canary_role=role,
        max_process_rss_bytes=int(memory["max_process_rss_bytes"]),
        launch_headroom_bytes=int(memory["launch_headroom_bytes"]),
        runtime_headroom_bytes=int(memory["runtime_headroom_bytes"]),
        sample_seconds=float(memory["sample_seconds"]),
        launch_samples_required=int(memory.get("launch_samples_required", 1)),
        runtime_low_headroom_samples=int(
            memory.get("runtime_low_headroom_samples", 1)
        ),
    )
    write_spec(path, spec)
    return spec, path


def _run_science(
    spec: Mapping[str, Any],
    spec_path: Path,
    *,
    owner_root: Path,
    label: str,
    resume: bool,
    stop_step: int | None,
    convergence_receipt: Path | None = None,
) -> None:
    environment = dict(os.environ)
    environment.update(spec["science_environment"])
    environment.update(
        {
            "TASTEMOLNET_T14_OUTPUT": str(spec["output_root"]),
            "TASTEMOLNET_T14_RUN_ID": str(spec["run_id"]),
            "TASTEMOLNET_T14_GPU_INDEX": "2",
            "TASTEMOLNET_T14_RESUME": "1" if resume else "0",
            "TASTEMOLNET_T14_ROUTE_C_SPEC": str(spec_path),
            "TASTEMOLNET_T14_ROUTE_C_STORAGE": str(spec["storage_mode"]),
            "TASTEMOLNET_T14_CHECKPOINT_ONLY_STEP": (
                "" if stop_step is None else str(stop_step)
            ),
            "TASTEMOLNET_T14_CONVERGENCE_RECEIPT": (
                "" if convergence_receipt is None else str(convergence_receipt)
            ),
        }
    )
    stdout_path = owner_root / "logs" / f"{label}.out"
    stderr_path = owner_root / "logs" / f"{label}.err"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    launch_streak = 0
    launch_samples_required = int(
        spec["memory"].get("launch_samples_required", 1)
    )
    runtime_low_samples_required = int(
        spec["memory"].get("runtime_low_headroom_samples", 1)
    )
    while True:
        limit = _counter(Path(spec["memory"]["cgroup_limit_path"]))
        current = _counter(Path(spec["memory"]["cgroup_current_path"]))
        headroom = max(0, limit - current)
        if headroom < int(spec["memory"]["launch_headroom_bytes"]):
            launch_streak = 0
            _phase(
                owner_root,
                phase="WAITING_MEMORY_HEADROOM",
                task=label,
                headroom_bytes=headroom,
                other_science_rss_bytes=max(0, current - _rss_bytes(os.getpid())),
                launch_admission_samples=launch_streak,
                launch_admission_samples_required=launch_samples_required,
                science_pid=None,
            )
            time.sleep(float(spec["memory"]["sample_seconds"]))
            continue
        launch_streak += 1
        if launch_streak < launch_samples_required:
            _phase(
                owner_root,
                phase="SAMPLING_MEMORY_HEADROOM",
                task=label,
                headroom_bytes=headroom,
                other_science_rss_bytes=max(0, current - _rss_bytes(os.getpid())),
                launch_admission_samples=launch_streak,
                launch_admission_samples_required=launch_samples_required,
                science_pid=None,
            )
            time.sleep(float(spec["memory"]["sample_seconds"]))
            continue
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
            "ab", buffering=0
        ) as stderr:
            child = subprocess.Popen(
                [str(spec["science_wrapper"])],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=False,
            )
            child_start_ticks = _start_ticks(child.pid)
            stop_requested = False

            def request_stop(_signum: int, _frame: object) -> None:
                nonlocal stop_requested
                stop_requested = True
                if child.poll() is None:
                    _terminate_process_tree(
                        child.pid, expected_root_ticks=child_start_ticks
                    )

            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            failcnt_start = _counter(Path(spec["memory"]["cgroup_failcnt_path"]))
            peak_rss = 0
            low_headroom_streak = 0
            while child.poll() is None:
                rss, process_tree = _process_tree_rss_bytes(child.pid)
                peak_rss = max(peak_rss, rss)
                limit = _counter(Path(spec["memory"]["cgroup_limit_path"]))
                current = _counter(Path(spec["memory"]["cgroup_current_path"]))
                failcnt = _counter(Path(spec["memory"]["cgroup_failcnt_path"]))
                headroom = max(0, limit - current)
                if headroom < int(spec["memory"]["runtime_headroom_bytes"]):
                    low_headroom_streak += 1
                else:
                    low_headroom_streak = 0
                _phase(
                    owner_root,
                    phase="SCIENCE_RUNNING",
                    task=label,
                    science_pid=child.pid,
                    science_start_ticks=child_start_ticks,
                    process_rss_bytes=rss,
                    process_peak_rss_bytes=peak_rss,
                    science_process_tree=process_tree,
                    cgroup_headroom_bytes=headroom,
                    other_science_rss_bytes=max(
                        0, current - rss - _rss_bytes(os.getpid())
                    ),
                    low_headroom_samples=low_headroom_streak,
                    low_headroom_samples_required=runtime_low_samples_required,
                    stop_requested=stop_requested,
                )
                resource_reason = None
                if rss > int(spec["memory"]["max_process_rss_bytes"]):
                    resource_reason = "PROCESS_RSS_LIMIT"
                elif failcnt > failcnt_start:
                    resource_reason = "CGROUP_FAILCNT_INCREASED"
                elif low_headroom_streak >= runtime_low_samples_required:
                    resource_reason = "RUNTIME_HEADROOM_BELOW_RESERVE_THREE_SAMPLES"
                if resource_reason is not None:
                    latest = _latest_step(spec) if Path(spec["output_root"]).exists() else None
                    atomic_json(
                        owner_root / "resource_watchdog_stop_request.json",
                        {
                            "schema_version": "tastemolnet_t14_resource_stop_request_v1",
                            "reason": resource_reason,
                            "science_pid": child.pid,
                            "science_start_ticks": child_start_ticks,
                            "cgroup_headroom_bytes": headroom,
                            "low_headroom_samples": low_headroom_streak,
                            "low_headroom_samples_required": runtime_low_samples_required,
                            "latest_promoted_checkpoint": latest,
                            "signal": "SIGTERM",
                            "sigkill_allowed": False,
                            "requested_at": _utc_now(),
                        },
                    )
                    _terminate_process_tree(
                        child.pid, expected_root_ticks=child_start_ticks
                    )
                    stop_requested = True
                    child.wait()
                    raise T14RouteCFreshError(
                        "Route C resource watchdog stopped "
                        f"{label} at a safe request: {resource_reason}"
                    )
                time.sleep(float(spec["memory"]["sample_seconds"]))
            return_code = int(child.wait())
        if return_code == WAITING_EXIT:
            _phase(
                owner_root,
                phase="WAITING_GPU2",
                task=label,
                science_pid=None,
            )
            time.sleep(30)
            continue
        if return_code != 0:
            raise T14RouteCFreshError(
                f"Route C science phase failed: {label}, exit={return_code}"
            )
        return


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14RouteCFreshError(f"Route C owner evidence is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise T14RouteCFreshError(f"Route C owner evidence is not an object: {path}")
    return value


def _load_or_create_plan(
    master: Mapping[str, Any], *, master_path: Path, owner_root: Path
) -> dict[str, Any]:
    plan_path = owner_root / "owner_plan.json"
    if plan_path.exists():
        if plan_path.is_symlink() or not plan_path.is_file():
            raise T14RouteCFreshError("Route C owner plan is indirect")
        plan = _json_object(plan_path)
        if (
            plan.get("schema_version") != "tastemolnet_t14_route_c_owner_plan_v1"
            or plan.get("master_spec") != str(master_path)
            or plan.get("master_spec_sha256") != master["spec_sha256"]
            or not isinstance(plan.get("children"), dict)
        ):
            raise T14RouteCFreshError("Route C sealed owner plan changed")
        for role, expected_mode in (
            ("REFERENCE_500", "reference"),
            ("LOW_MEMORY_CONTINUOUS_510", "lowmemory"),
            ("LOW_MEMORY_RELOAD_510", "lowmemory"),
        ):
            row = plan["children"].get(role)
            if not isinstance(row, dict):
                raise T14RouteCFreshError("Route C sealed owner child is absent")
            child_path = Path(str(row.get("spec_path")))
            child = load_spec(child_path)
            if (
                child.get("spec_sha256") != row.get("spec_sha256")
                or child.get("canary_role") != role
                or child.get("storage_mode") != expected_mode
                or Path(child["owner_root"]).parent.parent != owner_root / "canaries"
            ):
                raise T14RouteCFreshError("Route C sealed owner child changed")
        return plan
    if plan_path.is_symlink():
        raise T14RouteCFreshError("Route C owner plan path is indirect")
    children: dict[str, dict[str, Any]] = {}
    for role, mode in (
        ("REFERENCE_500", "reference"),
        ("LOW_MEMORY_CONTINUOUS_510", "lowmemory"),
        ("LOW_MEMORY_RELOAD_510", "lowmemory"),
    ):
        child, child_path = _child_spec(
            master, owner_root=owner_root, role=role, storage_mode=mode
        )
        children[role] = {
            "spec_path": str(child_path),
            "spec_sha256": child["spec_sha256"],
            "output_root": child["output_root"],
        }
    plan = {
        "schema_version": "tastemolnet_t14_route_c_owner_plan_v1",
        "master_spec": str(master_path),
        "master_spec_sha256": master["spec_sha256"],
        "children": children,
        "created_at": _utc_now(),
    }
    atomic_json(plan_path, plan)
    return plan


def _planned_child(plan: Mapping[str, Any], role: str) -> tuple[dict[str, Any], Path]:
    row = plan["children"][role]
    path = Path(row["spec_path"])
    return load_spec(path), path


def _small_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*1024**2:
        raise T14RouteCFreshError(f"T14 replacement requires bounded physical evidence: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replacement_commit() -> str:
    commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
    if dirty or len(commit) != 40:
        raise T14RouteCFreshError("T14 replacement requires clean immutable driver")
    return commit


def prepare_failed_stage_replacement(
    master: Mapping[str, Any], *, master_path: Path, authorization_path: Path,
    output: Path, proc_root: Path = Path('/proc'),
) -> dict[str, Any]:
    """Bind two fresh storage-repaired canaries within retry2; never retry3."""
    owner = Path(master['owner_root'])
    if output != owner/'stage_replacement_retry2'/'receipt.json' or output.exists():
        raise T14RouteCFreshError('T14 single fixed replacement binding already used or wrong path')
    auth = _json_object(authorization_path)
    expected = dict(schema_version='t14_retry2_failed_stage_replacement_authorization_v1',
        authorized_by='user_project_owner',same_retry_index=2,max_stage_replacements=1,
        stage='LOW_MEMORY_CONTINUOUS_510',allow_storage_only_driver_repair=True,
        reference_rerun_allowed=False,old_failed_root_mutation_allowed=False,
        retry3_allowed=False,formal_execution_rebind_required=True)
    if any(auth.get(k) != v for k,v in expected.items()):
        raise T14RouteCFreshError('T14 exact replacement authorization required')
    if (master.get('fresh_retry',{}).get('retry_index') != 2
            or auth.get('master_spec') != {'path':str(master_path),'sha256':_small_sha(master_path)}
            or auth.get('driver_commit') != _replacement_commit()):
        raise T14RouteCFreshError('T14 replacement master/retry/driver binding changed')
    terminal_path=owner/'terminal.json'
    terminal=_json_object(terminal_path)
    if (terminal.get('status')!='FAILED' or
            terminal.get('error')!='Route C science phase failed: lowmemory-continuous-510, exit=1'
            or auth.get('failed_terminal') != {'path':str(terminal_path),'sha256':_small_sha(terminal_path)}):
        raise T14RouteCFreshError('T14 replacement requires exact failed stage terminal')
    if (proc_root/str(terminal['owner_pid'])).exists():
        raise T14RouteCFreshError('T14 prior owner PID requires exact live review')
    _small_sha(owner/'owner_plan.json')
    plan=_load_or_create_plan(master,master_path=master_path,owner_root=owner)
    reference,_=_planned_child(plan,'REFERENCE_500')
    failed,_=_planned_child(plan,'LOW_MEMORY_CONTINUOUS_510')
    unused,_=_planned_child(plan,'LOW_MEMORY_RELOAD_510')
    # Only metadata and the short scientific ledger; never deserialize reference
    # or forbidden legacy checkpoints just to replace an empty failed stage.
    boundary=validate_checkpoint_boundary(reference,step=500,validate_envelope=False)
    if not _ledger_has(Path(reference['output_root']),500):
        raise T14RouteCFreshError('T14 completed reference500 ledger required')
    failed_root=Path(failed['output_root'])
    progress=_json_object(failed_root/'progress.json')
    if int(progress.get('completed_step',progress.get('step',-1))) != 0:
        raise T14RouteCFreshError('T14 failed-stage replacement requires exact step0')
    checkpoints=failed_root/'checkpoints'
    if (not checkpoints.is_dir() or any(checkpoints.iterdir())
            or _ledger_has(failed_root,1) or Path(unused['output_root']).exists()
            or Path(master['output_root']).exists()):
        raise T14RouteCFreshError('T14 replacement cannot discard scientific progress')
    old_tokens={str(master_path),str(failed_root),plan['children']['LOW_MEMORY_CONTINUOUS_510']['spec_path']}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():continue
        try:tokens=set((entry/'cmdline').read_bytes().decode().split('\0'))
        except (OSError,UnicodeError):continue
        if int(entry.name)!=os.getpid() and tokens & old_tokens:
            raise T14RouteCFreshError('T14 replacement has an exact active writer')
    output.parent.mkdir(parents=True,exist_ok=False)
    runtime_master=dict(master,execution_commit=auth['driver_commit'],
        science_wrapper=str(REPO_ROOT/'scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh'),
        owner_entrypoint=str(REPO_ROOT/'scripts/autodl/run_t14_route_c_owner.py'))
    children={}
    for role in ('LOW_MEMORY_CONTINUOUS_510','LOW_MEMORY_RELOAD_510'):
        child,path=_child_spec(runtime_master,owner_root=owner,role=role,storage_mode='lowmemory')
        children[role]={'spec_path':str(path),'spec_sha256':child['spec_sha256'],'output_root':child['output_root']}
    receipt=dict(schema_version='t14_retry2_failed_stage_replacement_v1',retry_index=2,
        max_stage_replacements=1,created_at=_utc_now(),authorization={'path':str(authorization_path),'sha256':_small_sha(authorization_path)},
        master_spec={'path':str(master_path),'sha256':_small_sha(master_path)},
        original_plan={'path':str(owner/'owner_plan.json'),'sha256':_small_sha(owner/'owner_plan.json')},
        source_science_commit=master['execution_commit'],driver_commit=auth['driver_commit'],
        driver_wrapper_sha256=_small_sha(Path(runtime_master['science_wrapper'])),
        driver_owner_sha256=_small_sha(Path(runtime_master['owner_entrypoint'])),
        failed_terminal=terminal,failed_terminal_sha256=_small_sha(terminal_path),
        failed_child=plan['children']['LOW_MEMORY_CONTINUOUS_510'],
        preserved_reference=plan['children']['REFERENCE_500'],reference_checkpoint_digest=boundary['boundary']['checkpoint_digest'],
        children=children,reference_rerun=False,failed_checkpoint_loaded=False,
        old_plan_modified=False,retry3_created=False,formal_execution_rebind_required=True)
    receipt['receipt_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    atomic_json(output,receipt)
    return receipt


def load_failed_stage_replacement(master: Mapping[str,Any],master_path: Path,plan: dict[str,Any],path: Path) -> dict[str,Any]:
    receipt=_json_object(path)
    unsigned={k:v for k,v in receipt.items() if k!='receipt_sha256'}
    if receipt.get('receipt_sha256')!=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest():
        raise T14RouteCFreshError('T14 replacement self hash changed')
    if (path!=Path(master['owner_root'])/'stage_replacement_retry2/receipt.json'
            or receipt.get('schema_version')!='t14_retry2_failed_stage_replacement_v1'
            or receipt.get('retry_index')!=2 or receipt.get('max_stage_replacements')!=1
            or receipt.get('source_science_commit')!=master['execution_commit']
            or receipt.get('driver_commit')!=_replacement_commit()
            or receipt.get('master_spec')!={'path':str(master_path),'sha256':_small_sha(master_path)}
            or receipt.get('preserved_reference')!=plan['children']['REFERENCE_500']
            or receipt.get('failed_child')!=plan['children']['LOW_MEMORY_CONTINUOUS_510']):
        raise T14RouteCFreshError('T14 replacement immutable binding changed')
    for key in ('authorization','original_plan'):
        row=receipt[key]
        if _small_sha(Path(row['path']))!=row['sha256']:
            raise T14RouteCFreshError('T14 replacement source changed:'+key)
    for name,key in (('run_t14_route_c_owner.py','driver_owner_sha256'),('run_tastemolnet_t14_comrecgc_full.sh','driver_wrapper_sha256')):
        if _small_sha(REPO_ROOT/'scripts/autodl'/name)!=receipt[key]:
            raise T14RouteCFreshError('T14 replacement driver bytes changed')
    children=dict(plan['children'])
    for role in ('LOW_MEMORY_CONTINUOUS_510','LOW_MEMORY_RELOAD_510'):
        row=receipt['children'][role];child=load_spec(Path(row['spec_path']))
        if (child['spec_sha256']!=row['spec_sha256'] or child['output_root']!=row['output_root']
                or child['canary_role']!=role or child['storage_mode']!='lowmemory'
                or child['execution_commit']!=receipt['driver_commit']
                or Path(child['owner_root']).parent.parent!=Path(master['owner_root'])/'canaries'):
            raise T14RouteCFreshError('T14 replacement child changed')
        for key in ('science_environment','memory','route_c_state','m_configured_max',
                    'm_fallback_max','forbidden_legacy_root','production_checkpoint_steps'):
            if child[key]!=master[key]:
                raise T14RouteCFreshError('T14 replacement scientific/resource contract changed:'+key)
        children[role]=row
    return dict(plan,children=children,failed_stage_replacement=receipt)


def _ledger_has(output_root: Path, step: int) -> bool:
    path = output_root / "route_c_step_states.jsonl"
    if not path.is_file() or path.is_symlink():
        return False
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip()) >= step


def _replay_boundary_valid(spec: Mapping[str, Any], *, step: int = 510) -> bool:
    path = Path(spec["output_root"]) / f"route_c_boundary_{step:06d}.json"
    if not path.is_file() or path.is_symlink() or not _ledger_has(Path(spec["output_root"]), step):
        return False
    row = _json_object(path)
    return (
        row.get("status") == "REPLAY_BOUNDARY_REACHED"
        and row.get("completed_step") == step
        and row.get("spec_sha256") == spec["spec_sha256"]
        and row.get("output_root") == spec["output_root"]
        and row.get("legacy_checkpoint_loaded") is False
    )


def _latest_step(spec: Mapping[str, Any]) -> int | None:
    path = Path(spec["output_root"]) / "checkpoints" / "LATEST"
    if not path.is_file() or path.is_symlink():
        return None
    row = _json_object(path)
    step = int(row.get("completed_step", -1))
    if row.get("checkpoint_dir") != f"step-{step:012d}":
        raise T14RouteCFreshError("Route C LATEST pointer changed")
    return step


def _promote_checkpoint_fresh_process(
    spec: Mapping[str, Any],
    spec_path: Path,
    *,
    step: int,
    owner_root: Path,
    label: str,
) -> None:
    """Reload and promote one boundary in a process fresh from science state."""

    stdout_path = owner_root / "logs" / f"{label}-reload.out"
    stderr_path = owner_root / "logs" / f"{label}-reload.err"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
        "ab", buffering=0
    ) as stderr:
        completed = subprocess.run(
            [
                str(spec["python"]),
                "-I",
                "-B",
                str(REPO_ROOT / "scripts/autodl/promote_t14_route_c_checkpoint.py"),
                "--config",
                str(REPO_ROOT / "configs/hpc.yaml"),
                "--route-c-spec",
                str(spec_path),
                "--step",
                str(step),
            ],
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise T14RouteCFreshError(
            f"Route C fresh-process checkpoint reload failed: step={step}"
        )
    validate_promotion_receipt(
        promotion_receipt_path(spec, step), spec=spec, expected_step=step
    )


def _promote_pending(
    spec: Mapping[str, Any],
    spec_path: Path,
    *,
    owner_root: Path,
    label: str,
) -> int | None:
    root = Path(spec["output_root"])
    pending_path = root / "checkpoints" / "PENDING_LATEST.json"
    if not pending_path.is_file():
        return _latest_step(spec)
    if pending_path.is_symlink():
        raise T14RouteCFreshError("Route C pending pointer is indirect")
    pending = _json_object(pending_path)
    step = int(pending.get("completed_step", -1))
    validate_checkpoint_boundary(spec, step=step, validate_envelope=True)
    _promote_checkpoint_fresh_process(
        spec,
        spec_path,
        step=step,
        owner_root=owner_root,
        label=label,
    )
    return step


def _ensure_promoted_boundary(
    spec: Mapping[str, Any],
    spec_path: Path,
    *,
    owner_root: Path,
    target: int,
    label: str,
) -> None:
    root = Path(spec["output_root"])
    latest = (
        _promote_pending(
            spec,
            spec_path,
            owner_root=owner_root,
            label=f"{label}-pending",
        )
        if root.exists()
        else None
    )
    if latest is not None and latest >= target:
        return
    if root.exists() and latest is None:
        raise T14RouteCFreshError(
            "Route C same-root run crashed before its first promotable checkpoint"
        )
    _run_science(
        spec,
        spec_path,
        owner_root=owner_root,
        label=label,
        resume=latest is not None,
        stop_step=target,
    )
    _promote_checkpoint_fresh_process(
        spec,
        spec_path,
        step=target,
        owner_root=owner_root,
        label=label,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--task-spec", type=_absolute, required=True)
    parser.add_argument("--continuation-spec", type=_absolute, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-failed-stage-replacement", type=_absolute)
    parser.add_argument("--replacement-authorization", type=_absolute)
    parser.add_argument("--failed-stage-replacement", type=_absolute)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file() or args.config.is_symlink():
        raise ValueError("T14 Route C owner requires one physical config")
    master = load_spec(args.task_spec)
    continuation = load_continuation_spec(args.continuation_spec)
    if (
        continuation["route_c_spec"] != str(args.task_spec)
        or continuation["generation_root"] != master["output_root"]
        or continuation["generation_owner_root"] != master["owner_root"]
    ):
        raise T14RouteCFreshError("T14 Route C continuation binding changed")
    if master["storage_mode"] != "lowmemory" or master["canary_role"] != (
        "PROMOTABLE_LOW_MEMORY"
    ):
        raise T14RouteCFreshError("T14 Route C owner spec is not promotable")
    owner_root = Path(master["owner_root"])
    owner_root.mkdir(parents=True, exist_ok=True)
    terminal_path = owner_root / "terminal.json"
    lock = (owner_root / "owner.lock").open("a+b")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise T14RouteCFreshError("T14 Route C already has one owner") from exc
        # Preparation only adds one fresh binding/child specs; never archives the
        # failure, launches science or edits the immutable prior owner plan.
        if args.prepare_failed_stage_replacement:
            if not args.replacement_authorization or args.failed_stage_replacement or args.dry_run:
                raise T14RouteCFreshError('T14 replacement preparation flags conflict')
            receipt=prepare_failed_stage_replacement(master,master_path=args.task_spec,
                authorization_path=args.replacement_authorization,output=args.prepare_failed_stage_replacement)
            print(json.dumps(receipt,sort_keys=True),flush=True)
            return 0
        if args.replacement_authorization:
            raise T14RouteCFreshError('T14 replacement authorization is preparation-only')
        owner_path = owner_root / "owner.json"
        previous_owner = _json_object(owner_path) if owner_path.is_file() else None
        if previous_owner is not None and (
            previous_owner.get("schema_version") != OWNER_SCHEMA
            or previous_owner.get("task_spec") != str(args.task_spec)
            or previous_owner.get("task_spec_sha256") != master["spec_sha256"]
        ):
            raise T14RouteCFreshError("T14 Route C prior owner evidence changed")
        terminal_path = owner_root / "terminal.json"
        if terminal_path.is_file() and not terminal_path.is_symlink():
            prior_terminal = _json_object(terminal_path)
            if prior_terminal.get("status") == "FAILED":
                history = owner_root / "terminal_history"
                history.mkdir(parents=True, exist_ok=True)
                destination = history / (
                    f"failed-owner-{int(prior_terminal.get('owner_pid', 0)):010d}-"
                    f"restart-{int((previous_owner or {}).get('restart_count', 0)) + 1:03d}.json"
                )
                if destination.exists() or destination.is_symlink():
                    raise T14RouteCFreshError("T14 Route C terminal history collision")
                os.replace(terminal_path, destination)
                _fsync_directory(owner_root)
                _fsync_directory(history)
            elif prior_terminal.get("status") != "GENERATION_PASS_PENDING_POSTPROCESS":
                raise T14RouteCFreshError("T14 Route C prior terminal state changed")
        elif terminal_path.exists() or terminal_path.is_symlink():
            raise T14RouteCFreshError("T14 Route C owner terminal is indirect")
        atomic_json(
            owner_path,
            {
                "schema_version": OWNER_SCHEMA,
                "status": "OWNER_CONFIRMED",
                "owner_pid": os.getpid(),
                "owner_start_ticks": _start_ticks(os.getpid()),
                "task_spec": str(args.task_spec),
                "task_spec_sha256": master["spec_sha256"],
                "forbidden_legacy_root": master["forbidden_legacy_root"],
                "legacy_checkpoint_loaded": False,
                "restart_count": (
                    0
                    if previous_owner is None
                    else int(previous_owner.get("restart_count", 0)) + 1
                ),
                "created_at": (
                    _utc_now()
                    if previous_owner is None
                    else previous_owner.get("created_at")
                ),
                "reconfirmed_at": _utc_now(),
            },
        )
        matrix = continuation["matrix"]
        matrix_receipt = owner_root / "matrix_launch_gate_latest.json"
        try:
            audit_route_c_matrix_cell_absent(
                state_path=Path(matrix["authority_state_path"]),
                lock_path=Path(matrix["authority_lock_path"]),
                receipt_path=matrix_receipt,
            )
        except T14RouteCFreshError:
            observed = _json_object(matrix_receipt) if matrix_receipt.is_file() else {}
            if observed.get("status") == "ALREADY_PUBLISHED":
                _phase(
                    owner_root,
                    phase="ALREADY_PUBLISHED_NO_SCIENCE",
                    science_pid=None,
                )
                return 0
            raise
        if args.dry_run:
            _phase(owner_root, phase="DRY_RUN_PASS", science_pid=None)
            return 0

        plan = _load_or_create_plan(
            master, master_path=args.task_spec, owner_root=owner_root
        )
        if args.failed_stage_replacement:
            plan=load_failed_stage_replacement(master,args.task_spec,plan,args.failed_stage_replacement)
        reference, reference_path = _planned_child(plan, "REFERENCE_500")
        continuous, continuous_path = _planned_child(
            plan, "LOW_MEMORY_CONTINUOUS_510"
        )
        reload_spec, reload_path = _planned_child(plan, "LOW_MEMORY_RELOAD_510")

        reference_root = Path(reference["output_root"])
        if args.failed_stage_replacement and not reference_root.is_dir():
            raise T14RouteCFreshError('T14 replacement reference500 vanished; rerun forbidden')
        if reference_root.exists():
            validate_checkpoint_boundary(reference, step=500,
                validate_envelope=not bool(args.failed_stage_replacement))
            if not _ledger_has(reference_root, 500):
                raise T14RouteCFreshError("Route C reference ledger is incomplete")
        else:
            _run_science(
                reference,
                reference_path,
                owner_root=owner_root,
                label="reference-500",
                resume=False,
                stop_step=500,
            )

        continuous_root = Path(continuous["output_root"])
        if continuous_root.exists():
            if not _replay_boundary_valid(continuous):
                raise T14RouteCFreshError(
                    "Route C continuous canary crashed before its sealed 510 boundary"
                )
        else:
            _run_science(
                continuous,
                continuous_path,
                owner_root=owner_root,
                label="lowmemory-continuous-510",
                resume=False,
                stop_step=510,
            )

        reload_root = Path(reload_spec["output_root"])
        if not reload_root.exists():
            _run_science(
                reload_spec,
                reload_path,
                owner_root=owner_root,
                label="lowmemory-reload-250",
                resume=False,
                stop_step=250,
            )
            _promote_checkpoint_fresh_process(
                reload_spec,
                reload_path,
                step=250,
                owner_root=owner_root,
                label="lowmemory-reload-250",
            )
        elif not _replay_boundary_valid(reload_spec):
            latest = _promote_pending(
                reload_spec,
                reload_path,
                owner_root=owner_root,
                label="lowmemory-reload-pending",
            )
            if latest is None:
                raise T14RouteCFreshError(
                    "Route C reload canary crashed before a sealed checkpoint"
                )
        if not _replay_boundary_valid(reload_spec):
            _run_science(
                reload_spec,
                reload_path,
                owner_root=owner_root,
                label="lowmemory-reload-510",
                resume=True,
                stop_step=510,
            )
        reference_ledger = Path(reference["output_root"]) / "route_c_step_states.jsonl"
        continuous_ledger = Path(continuous["output_root"]) / "route_c_step_states.jsonl"
        reload_ledger = Path(reload_spec["output_root"]) / "route_c_step_states.jsonl"
        receipts = {
            "reference_vs_lowmemory_1_500": compare_step_ledgers(
                reference_ledger, continuous_ledger, start_step=1, end_step=500
            ),
            "continuous_vs_reload_1_500": compare_step_ledgers(
                continuous_ledger, reload_ledger, start_step=1, end_step=500
            ),
            "continuous_vs_reload_501_510": compare_step_ledgers(
                continuous_ledger, reload_ledger, start_step=501, end_step=510
            ),
        }
        if any(value["status"] != "PASS" for value in receipts.values()):
            raise T14RouteCFreshError("T14 Route C semantic parity failed")

        if args.failed_stage_replacement:
            replacement=plan['failed_stage_replacement']
            # The old retry2 master and authorization/cadence pin the original
            # execution. Do not relabel new-driver full outputs as old science.
            hold=dict(schema_version='t14_retry2_storage_repair_canary_result_v1',
                status='CANARY_PARITY_PASS_FORMAL_EXECUTION_REBIND_REQUIRED',
                retry_index=2,retry3_created=False,reference_rerun=False,
                replacement_receipt_sha256=replacement['receipt_sha256'],receipts=receipts,
                source_science_commit=master['execution_commit'],driver_commit=replacement['driver_commit'],
                full_started=False,blocked_fields=['master.execution_commit',
                    'fresh_retry.authorization_receipt.corrected_execution_commit',
                    'fresh_retry.formal_cadence_contract.execution_commit'],written_at=_utc_now())
            atomic_json(owner_root/'stage_replacement_retry2/canary_verification.json',hold)
            _phase(owner_root,phase=hold['status'],science_pid=None)
            return WAITING_EXIT

        # The promotable root is the formal Route-C run.  Stop at every early
        # boundary so this owner independently reloads and promotes each one.
        # After parity, the same step-500 state continues to full; steps 1-500
        # are not generated again.
        for early_checkpoint in (*EARLY_CHECKPOINT_STEPS, 500):
            _ensure_promoted_boundary(
                master,
                args.task_spec,
                owner_root=owner_root,
                target=early_checkpoint,
                label=f"promotable-lowmemory-{early_checkpoint}",
            )
        promotable_ledger = Path(master["output_root"]) / "route_c_step_states.jsonl"
        receipts["continuous_vs_promotable_1_500"] = compare_step_ledgers(
            continuous_ledger, promotable_ledger, start_step=1, end_step=500
        )
        if receipts["continuous_vs_promotable_1_500"]["status"] != "PASS":
            raise T14RouteCFreshError("T14 Route C promotable canary diverged")
        parity = {
            "schema_version": "tastemolnet_t14_route_c_parity_gate_v1",
            "status": "PASS",
            "receipts": receipts,
            "checkpoint_250_reload_pass": True,
            "steps_501_510_exact": True,
            "promotable_checkpoint_step": 500,
            "durable_early_checkpoints": [50, 100, 250, 500],
            "formal_route_c_500_promoted_to_full_without_replay": True,
            "legacy_checkpoint_loaded": False,
            "written_at": _utc_now(),
        }
        atomic_json(owner_root / "parity.json", parity)

        convergence_receipt: Path | None = None
        convergence_root = owner_root / "convergence"
        convergence_root.mkdir(parents=True, exist_ok=True)
        existing_convergence = convergence_root / "early_stop_receipt.json"
        if existing_convergence.is_file() and not existing_convergence.is_symlink():
            raw_convergence = _json_object(existing_convergence)
            convergence_step = int(raw_convergence.get("m_effective", -1))
            promotion = validate_promotion_receipt(
                promotion_receipt_path(master, convergence_step),
                spec=master,
                expected_step=convergence_step,
            )
            validate_route_c_convergence_receipt(
                existing_convergence,
                spec=master,
                expected_step=convergence_step,
                expected_checkpoint_digest=str(promotion["checkpoint_digest"]),
            )
            convergence_receipt = existing_convergence
        checkpoints = (
            ()
            if convergence_receipt is not None
            else (2_500, 5_000, 7_500, 10_000, 12_500, 15_000, 17_500)
        )
        for checkpoint in checkpoints:
            if (_latest_step(master) or 0) >= M_MAX:
                break
            _ensure_promoted_boundary(
                master,
                args.task_spec,
                owner_root=owner_root,
                target=checkpoint,
                label=f"full-to-{checkpoint}",
            )
            if checkpoint < 10_000:
                continue
            convergence = audit_route_c_convergence(master)
            convergence["audited_at"] = _utc_now()
            convergence["owner_pid"] = os.getpid()
            convergence["audit_sha256"] = hashlib.sha256(
                json.dumps(
                    convergence, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest()
            atomic_json(
                convergence_root / f"audit-{checkpoint:06d}.json", convergence
            )
            if convergence.get("converged") is True:
                convergence_receipt = convergence_root / "early_stop_receipt.json"
                write_route_c_convergence_receipt(
                    convergence_receipt, spec=master, audit=convergence
                )
                break

        generation_root = Path(master["output_root"])
        if (generation_root / "GENERATION_PASS").is_file():
            from src.baselines.tastemolnet_comrecgc_full import validate_t14_full_output

            generation_verification = validate_t14_full_output(generation_root)
        else:
            _run_science(
                master,
                args.task_spec,
                owner_root=owner_root,
                label="convergence-or-resource-cap-postprocess",
                resume=True,
                stop_step=None,
                convergence_receipt=convergence_receipt,
            )
            from src.baselines.tastemolnet_comrecgc_full import validate_t14_full_output

            generation_verification = validate_t14_full_output(generation_root)
        publish_generation_handoff(
            continuation_spec_path=args.continuation_spec,
            generation_verification=generation_verification,
            owner_pid=os.getpid(),
            owner_start_ticks=_start_ticks(os.getpid()),
        )
        launch_continuation_owner(args.continuation_spec)
        return 0
    except Exception as exc:
        if args.prepare_failed_stage_replacement:
            # A failed preparation cannot rewrite the prior scientific failure.
            raise
        existing_terminal = (
            _json_object(terminal_path) if terminal_path.is_file() else {}
        )
        if existing_terminal.get("status") != "GENERATION_PASS_PENDING_POSTPROCESS":
            atomic_json(
                terminal_path,
                {
                    "schema_version": OWNER_SCHEMA,
                    "status": "FAILED",
                    "owner_pid": os.getpid(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "legacy_checkpoint_loaded": False,
                    "written_at": _utc_now(),
                },
            )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
