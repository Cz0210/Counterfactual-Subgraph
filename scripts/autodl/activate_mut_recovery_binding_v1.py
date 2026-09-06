#!/usr/bin/env python3
"""One-shot activation of the sealed Mut recovery through its existing owners.

No daemon/poller, new lock, new registry, or second publisher is introduced.
Without --execute this is read-only; failed resource admission performs no
signal or subprocess launch even when --execute is present.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.autodl_mut_recovery_binding_v1 import read_json, resource_status, sealed
from src.utils.autodl_mut_first_divergence_v1 import atomic_json, stable_sha256
from src.utils.autodl_mut_first_divergence_v1 import file_sha256
from src.utils.autodl_mut_same_contract_ab_v1 import validate_same_contract_ab_spec
from src.utils.autodl_mut_next_stage_executor_v1 import validate_successor_spec
from src.utils.final16_owner_registry_v1 import (
    validate_owner_registry, build_owner_registry, atomic_write_owner_registry, process_start_ticks,
)
from src.utils.autodl_runtime import query_gpu_inventory
from src.ablations.llm.existing_gpu_owner import validate_resource_config


def identity(pid: int) -> dict[str, Any]:
    ticks = process_start_ticks("/proc", pid)
    if ticks is None:
        raise RuntimeError("Expected executor PID is absent or reused")
    root = Path(f"/proc/{pid}")
    return {"pid": pid, "start_ticks": ticks,
            "argv": [x.decode() for x in (root/"cmdline").read_bytes().split(b"\0") if x],
            "cwd": os.readlink(root/"cwd"),
            "children": [int(x) for x in (root/"task"/str(pid)/"children").read_text().split()]}


def idle_executor(binding: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    spec_path = Path(binding["old_executor_spec"])
    spec = read_json(spec_path)
    rows = [r for r in registry["tasks"] if r["task_id"] == spec["task_id"]]
    if len(rows) != 1:
        raise RuntimeError("Old executor canonical registry row is not unique")
    row = rows[0]
    observed = identity(int(row["owner_pid"]))
    if observed["start_ticks"] != row["owner_start_ticks"]:
        raise RuntimeError("Old executor start ticks changed")
    argv = observed["argv"]
    expected_script = str(Path(spec["adoption_pipeline"][0]["cwd"]) / "scripts/autodl/run_mut_next_stage_executor_v1.py")
    if (expected_script not in argv or "--task-spec" not in argv or
            argv[argv.index("--task-spec")+1] != str(spec_path) or
            observed["cwd"] != spec["adoption_pipeline"][0]["cwd"] or observed["children"]):
        raise RuntimeError("Old executor command/cwd/child set is not safely idle")
    runtime = Path(spec["runtime_root"])
    heartbeat = read_json(runtime/"heartbeat.json")
    age = time.time() - datetime.fromisoformat(heartbeat["written_at"]).timestamp()
    if (not 0 <= age <= 120 or heartbeat.get("pid") != observed["pid"] or
            heartbeat.get("state") != "WAITING_FOR_NEXT_ACTION" or heartbeat.get("science_pid") is not None or
            heartbeat.get("stage") is not None or heartbeat.get("lane") is not None):
        raise RuntimeError("Old executor heartbeat is not a fresh empty WAITING state")
    action = Path(spec["next_action_path"])
    if (action.exists() or list(action.parent.glob("next_action.consumed-*.json")) or
            (runtime/"next_action_consumption.json").exists() or
            ((runtime/"stages").exists() and any((runtime/"stages").iterdir()))):
        raise RuntimeError("Old executor has an action or active/completed claim")
    publishers = [r for r in registry["publishers"] if r["publisher_id"] == spec["publisher_id"]]
    if len(publishers) != 1 or publishers[0]["active_writer_count"] != 0 or Path(spec["publisher_locator"]).exists():
        raise RuntimeError("Mut canonical publisher is no longer idle/unpublished")
    observed["heartbeat"] = heartbeat
    observed["task_id"] = spec["task_id"]
    return observed


def read_cgroup(root: Path) -> dict[str, int]:
    limit = int((root/"memory.limit_in_bytes").read_text())
    current = int((root/"memory.usage_in_bytes").read_text())
    return {"limit_bytes": limit, "usage_bytes": current, "headroom_bytes": max(0, limit-current)}


def admission(ab: dict[str, Any], registry_path: Path, resource_path: Path) -> dict[str, Any]:
    """Reuse the existing joint resource config, not only Mut's lower bound."""
    cfg = validate_resource_config(read_json(resource_path))
    if cfg["main_registry_path"] != str(registry_path) or cfg["gpu_lock_root"] != ab["gpu_lock_root"]:
        raise RuntimeError("Joint resource config uses another registry or GPU namespace")
    resource = resource_status(Path(ab["run_root"]))
    if os.stat(resource["path"]).st_dev != os.stat(cfg["persistent_root"]).st_dev:
        raise RuntimeError("Joint inode budget and Mut output are different filesystems")
    if cfg.get("minimum_free_inodes", 0) < 100000 or cfg.get("reserved_new_inodes", 0) < 160:
        raise RuntimeError("Existing joint inode guard/reservation cannot be lowered")
    #160 is already covered by the existing joint reservation; do not add it
    #again. Unknown transient peaks remain an explicit limitation, never zero.
    joint_required = max(resource["fixed_required_free"],
                         cfg["minimum_free_inodes"] + cfg["reserved_new_inodes"])
    cgroup = read_cgroup(Path(cfg["cgroup_memory_root"]))
    memory_required = max(64*1024**3, cfg["minimum_memory_headroom_bytes"])
    bytes_required = max(resource["minimum_free_bytes"], cfg["minimum_persistent_free_bytes"])
    admitted = (resource["state"] == "ADMISSION_PASS" and resource["free_inodes"] >= joint_required
                and resource["free_bytes"] >= bytes_required and cgroup["headroom_bytes"] >= memory_required)
    return {"state": "KNOWN_RESOURCE_THRESHOLDS_PASS" if admitted else "SEALED_WAITING_RESOURCE",
        "mut_resource": resource, "cgroup": cgroup, "resource_config": str(resource_path),
        "resource_config_sha256": stable_sha256(cfg), "joint_required_free_inodes": joint_required,
        "joint_inode_shortfall": max(0, joint_required-resource["free_inodes"]),
        "memory_required_bytes": memory_required, "persistent_required_bytes": bytes_required,
        "unknown_dynamic_peak": "UNKNOWN_NOT_ZERO", "complete_peak_admission_claimed": False}


def complete_peak_admission(resource: dict[str, Any], registry: dict[str, Any],
                            peak_path: Path | None) -> dict[str, Any]:
    """Consume a bounded engineering peak plan; absence is not zero demand.

    The live config remains untouched. A future plan is data for this existing
    preflight, not another resource registry/authority or a scientific gate.
    Only small task-spec/engineering evidence is read, never model payloads.
    """
    missing = {"state": "RESOURCE_PEAK_PLAN_INCOMPLETE",
               "first_missing_field": "task_peak_reservations",
               "science_started": False, "old_executor_signaled": False}
    if peak_path is None:
        return missing
    plan = read_json(peak_path)
    if (plan.get("scope") != "MUT_RESOURCE_REPLAY_WITH_CURRENT_MAIN_RESERVATIONS" or
            plan.get("resource_config_sha256") != resource["resource_config_sha256"]):
        raise RuntimeError("Peak plan scope/current resource config binding differs")
    rows = plan.get("task_peak_reservations")
    required = {row["task_id"] for row in registry["gpu_leases"] if row["state"] == "HELD"}
    if not isinstance(rows, list) or len(rows) != len(required) or {r.get("task_id") for r in rows} != required:
        return {**missing, "first_missing_field": "task_peak_reservations.coverage_of_current_held_main_tasks"}
    extra_inodes = extra_memory = 0
    for row in rows:
        for key in ("additional_new_inode_upper_bound", "additional_memory_headroom_upper_bound_bytes"):
            if type(row.get(key)) is not int or row[key] < 0:
                return {**missing, "first_missing_field": row["task_id"]+"."+key}
        # Additional means above the existing10761/64GiB known reservations,
        #not a second charge for existing files/current cgroup resident memory.
        evidence = Path(row.get("peak_evidence_path", ""))
        if (not evidence.is_absolute() or evidence.is_symlink() or not evidence.is_file()
                or evidence.stat().st_size > 2*1024**2 or
                file_sha256(evidence) != row.get("peak_evidence_sha256")):
            return {**missing, "first_missing_field": row["task_id"]+".bound_small_peak_evidence"}
        extra_inodes += row["additional_new_inode_upper_bound"]
        extra_memory += row["additional_memory_headroom_upper_bound_bytes"]
    need_inodes = resource["joint_required_free_inodes"]+extra_inodes
    need_memory = resource["memory_required_bytes"]+extra_memory
    ok = (resource["mut_resource"]["free_inodes"] >= need_inodes and
          resource["cgroup"]["headroom_bytes"] >= need_memory)
    return {"state": "RESOURCE_PEAK_PLAN_PASS" if ok else "SEALED_WAITING_RESOURCE",
        "peak_plan": str(peak_path), "peak_plan_sha256": stable_sha256(plan),
        "required_free_inodes": need_inodes, "required_headroom_bytes": need_memory,
        "inode_shortfall": max(0, need_inodes-resource["mut_resource"]["free_inodes"]),
        "memory_shortfall_bytes": max(0, need_memory-resource["cgroup"]["headroom_bytes"])}


def _wait_heartbeat(path: Path, child: subprocess.Popen[Any], task_id: str | None, seconds: int = 120) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"New waiting owner exited: PID={child.pid}, rc={child.returncode}")
        if path.is_file():
            data = read_json(path)
            if data.get("pid", data.get("owner_pid")) == child.pid:
                if task_id is not None and data.get("task_id") not in (None, task_id):
                    raise RuntimeError("New heartbeat task differs")
                return data
        time.sleep(1)
    raise RuntimeError(f"New owner heartbeat timeout: {path}")


def _launch(command: list[str], *, root: Path, log: Path, env: dict[str, str]) -> subprocess.Popen[Any]:
    with log.open("xb") as stream:
        return subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)


def activate(binding_path: Path, registry_path: Path, *, resource_path: Path,
             peak_path: Path | None = None, execute: bool) -> dict[str, Any]:
    binding = read_json(binding_path)
    if sealed(binding, "binding_sha256") != binding:
        raise RuntimeError("Sealed recovery binding changed")
    root = binding_path.parent
    ab = validate_same_contract_ab_spec(read_json(Path(binding["ab_task_spec"])), check_files=False)
    executor = validate_successor_spec(read_json(Path(binding["executor_task_spec"])), check_files=False)
    resource = admission(ab, registry_path, resource_path)
    if resource["state"] != "KNOWN_RESOURCE_THRESHOLDS_PASS":
        return {"state": "SEALED_WAITING_RESOURCE", "resource": resource,
                "old_executor_signaled": False, "science_started": False}
    registry = validate_owner_registry(read_json(registry_path), check_processes=False)
    peak = complete_peak_admission(resource, registry, peak_path)
    if peak["state"] != "RESOURCE_PEAK_PLAN_PASS":
        return {**peak, "resource": resource, "science_started": False, "old_executor_signaled": False}
    if registry["matrix_authority_root"] != binding["matrix_authority"]:
        raise RuntimeError("Wrong matrix authority")
    old = idle_executor(binding, registry)
    legacy_ab = read_json(Path(read_json(Path(binding["replay_contract"]))["original_ab_spec"]))
    related = [r for r in registry["tasks"] if r["task_id"] == legacy_ab["task_id"] or
               (r["stage"] == "POST_AB_DECISION" and r["input_root"] == legacy_ab["control_root"])]
    if len(related) != 2:
        raise RuntimeError("Legacy A/B and postAB canonical rows are not uniquely bound")
    for row in related:
        pid = row.get("owner_pid")
        if pid is not None and process_start_ticks("/proc", int(pid)) is not None:
            raise RuntimeError("Legacy A/B or postAB still alive; do not duplicate")
    gpu = [g for g in query_gpu_inventory() if g.index == 0]
    leases = [r for r in registry["gpu_leases"] if r["gpu"] == 0 and r["state"] == "HELD"]
    if (len(gpu) != 1 or gpu[0].uuid != ab["gpu_uuid"] or gpu[0].processes or len(leases) != 1
            or leases[0]["task_id"] != legacy_ab["task_id"] or leases[0]["lease_path"] != ab["lease_path"]):
        raise RuntimeError("Mut GPU0 reservation/device/process contract is not ready")
    for key in ("control_root", "run_root", "output_dir"):
        if Path(ab[key]).exists():
            raise RuntimeError("Recovery A/B already activated; no duplicate successor")
    if Path(executor["runtime_root"]).exists() or (root/"post-ab").exists() or (root/"activation.json").exists():
        raise RuntimeError("Recovery successor state exists; inspect instead of activate again")
    preflight = {"state": "READY_FOR_EXACT_IDLE_EXECUTOR_REPLACEMENT", "resource": resource,
                 "peak_plan": peak, "old_executor": old, "science_started": False,
                 "registry_sha256": registry["self_sha256"]}
    if not execute:
        return preflight
    # Full original input binding is rechecked once for this real activation.
    validate_same_contract_ab_spec(ab, check_files=True)
    validate_successor_spec(executor, check_files=True)
    lock_path = Path(binding["matrix_authority"])/"publish.lock"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise RuntimeError("Existing publish lock absent; never manufacture another lock")
    children: dict[str, int] = {}
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        latest = validate_owner_registry(read_json(registry_path), check_processes=False)
        if latest["self_sha256"] != registry["self_sha256"]:
            raise RuntimeError("Registry changed during activation preflight; safe retry required")
        # Recheck all mutable admission facts immediately before the only signal.
        again = idle_executor(binding, latest)
        if again["start_ticks"] != old["start_ticks"]:
            raise RuntimeError("Idle executor identity changed")
        new_resource = admission(ab, registry_path, resource_path)
        if (new_resource["state"] != "KNOWN_RESOURCE_THRESHOLDS_PASS" or
                new_resource["resource_config_sha256"] != resource["resource_config_sha256"]):
            raise RuntimeError("Resource admission lost before executor handover")
        new_peak = complete_peak_admission(new_resource, latest, peak_path)
        if (new_peak["state"] != "RESOURCE_PEAK_PLAN_PASS" or
                new_peak["peak_plan_sha256"] != peak["peak_plan_sha256"]):
            raise RuntimeError("Peak plan admission changed before executor handover")
        atomic_json(root/"activation-preflight.json", preflight)
        atomic_json(root/"registry-before-activation.json", latest)
        os.kill(old["pid"], signal.SIGTERM)
        deadline = time.monotonic()+90
        while process_start_ticks("/proc", old["pid"]) == old["start_ticks"]:
            if time.monotonic() > deadline:
                raise RuntimeError("Idle executor did not exit after SIGTERM; no SIGKILL")
            time.sleep(1)
        env = {**os.environ, **ab["required_environment"], "CUDA_VISIBLE_DEVICES": "0"}
        try:
            ex = _launch(binding["executor_command"], root=Path(binding["driver_root"]),
                         log=root/"executor-launch.log", env=env)
            children["executor"] = ex.pid
            _wait_heartbeat(Path(executor["runtime_root"])/"heartbeat.json", ex, executor["task_id"])
            post = _launch(binding["post_ab_command"], root=Path(binding["driver_root"]),
                           log=root/"post-ab-launch.log", env=env)
            children["post_ab"] = post.pid
            _wait_heartbeat(root/"post-ab/heartbeat.json", post, None)
            owner = _launch(binding["owner_command"], root=Path(binding["driver_root"]),
                            log=root/"ab-owner-launch.log", env=env)
            children["ab_owner"] = owner.pid
            _wait_heartbeat(Path(ab["control_root"])/"heartbeat.json", owner, ab["task_id"])
            value = copy.deepcopy(latest)
            old_ids = {old["task_id"], *(r["task_id"] for r in related)}
            post_id = ab["task_id"]+"-post-ab"
            for row in value["tasks"]:
                if row["task_id"] in old_ids:
                    row.update(owner_state="TERMINAL_FAILED_ENGINEERING", owner_pid=None, owner_start_ticks=None,
                               successor_task_id=executor["task_id"] if row["task_id"] == old["task_id"] else
                               post_id if row["stage"] == "POST_AB_DECISION" else ab["task_id"])
            def task(task_id, stage, pid, heartbeat, input_root, output_root, digest, gpu_index, successor, publisher):
                return {"task_id": task_id, "dataset": "Mutagenicity", "method": "ComRecGC", "stage": stage,
                    "execution_commit": binding["driver_commit"], "task_spec_sha": digest, "gpu": gpu_index,
                    "owner_state": "RUNNING", "owner_pid": pid, "owner_start_ticks": process_start_ticks("/proc",pid),
                    "heartbeat": str(heartbeat), "input_root": str(input_root), "output_root": str(output_root),
                    "successor_task_id": successor, "publisher_id": publisher}
            value["tasks"].extend([
                task(ab["task_id"], "TRACE_ON_OFF_SAME_CONTRACT_AB", owner.pid, Path(ab["control_root"])/"heartbeat.json",
                     ab["historical_artifact_root"], ab["run_root"], ab["spec_sha256"], 0, post_id, None),
                task(post_id, "POST_AB_DECISION", post.pid, root/"post-ab/heartbeat.json", ab["control_root"], root/"post-ab",
                     ab["spec_sha256"], None, executor["task_id"], None),
                task(executor["task_id"], "ADOPTION_TO_PUBLICATION_SUCCESSOR", ex.pid,
                     Path(executor["runtime_root"])/"heartbeat.json", root/"post-ab", executor["runtime_root"],
                     executor["spec_sha256"], None, None, executor["publisher_id"]),
            ])
            for lease in value["gpu_leases"]:
                if lease["gpu"] == 0 and lease["state"] == "HELD":
                    lease["task_id"] = ab["task_id"]
            for publisher in value["publishers"]:
                if publisher["publisher_id"] == executor["publisher_id"]:
                    publisher["execution_commit"] = binding["driver_commit"]
            updated = build_owner_registry(registry_id=value["registry_id"], matrix_authority_root=value["matrix_authority_root"],
                tasks=value["tasks"], publishers=value["publishers"], gpu_leases=value["gpu_leases"], check_processes=False)
            atomic_write_owner_registry(registry_path, updated)
            result = {"state": "EXISTING_OWNERS_ACTIVATED", "pids": children, "old_executor_signaled": True,
                      "old_executor_identity": old, "new_registry_sha256": updated["self_sha256"],
                      "matrix_scientific_state_written": False, "new_matrix_authority": False,
                      "science_final_pass": False, "activated_at": datetime.now(timezone.utc).isoformat()}
            atomic_json(root/"activation.json", result)
            return result
        except BaseException as exc:
            atomic_json(root/"activation-failed.json", {"error": repr(exc), "started_owner_pids": children,
                "automatic_retry_allowed": False, "no_broad_signal_sent": True})
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--owner-registry", type=Path, required=True)
    parser.add_argument("--resource-config", type=Path, required=True,
                        help="Current existing joint main/LLM resource config; no synthetic values")
    parser.add_argument("--peak-plan", type=Path,
                        help="Bound existing-task peak evidence; absence remains RESOURCE_PEAK_PLAN_INCOMPLETE")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (args.config != "configs/hpc.yaml" or not args.binding.is_absolute() or
            not args.owner_registry.is_absolute() or not args.resource_config.is_absolute() or
            (args.peak_plan is not None and not args.peak_plan.is_absolute())):
        raise ValueError("Bound AutoDL absolute inputs and configs/hpc.yaml required")
    print(json.dumps(activate(args.binding, args.owner_registry,
        resource_path=args.resource_config, peak_path=args.peak_plan, execute=args.execute), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
