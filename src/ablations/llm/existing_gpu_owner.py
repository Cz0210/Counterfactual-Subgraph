"""BACE LLM adapter for the existing AutoDL UUID/project-slot locks.

This is not another scheduler or reservation authority. Source paths are frozen
by the dispatch specification; every admission/runtime observation reopens the
actual registry, heartbeats, Linux counters and GPU inventory. Missing coverage
is a blocker, not evidence that the main queue is empty.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from src.ablations.gnn.early_policy import gpu_allowed
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file
from src.utils.autodl_runtime import GPUFileLock, ProjectGPUSlotLock, query_gpu_inventory
from src.utils.final16_owner_registry_v1 import process_start_ticks, validate_owner_registry
from src.utils.stage_file_policy import config_file_admission, load_stage_policy

RESOURCE_SCHEMA = "bace_llm_gpu_owner_resource_v1"
DISPATCH_SCHEMA = "bace_llm_existing_owner_dispatch_v1"
READY = {"READY", "READY_WAITING_GPU", "WAITING_GPU", "WAITING_FOR_GPU", "READY_WAITING_RESOURCE"}
FAILED = {"FAILED", "BLOCKED", "TERMINAL_FAILED_ENGINEERING", "MISSING"}


def validate_resource_config(config):
    required = {"main_registry_path", "main_ready_sources", "proc_root", "cgroup_memory_root",
                "persistent_root", "gpu_lock_root", "minimum_gpu_free_mb", "maximum_idle_utilization_percent",
                "minimum_memory_headroom_bytes", "minimum_persistent_free_bytes", "checkpoint_resume_pass"}
    optional = {"minimum_free_inodes", "reserved_new_inodes", "stage_file_policy"}
    if not required <= set(config) or set(config) - required - optional:
        raise ValueError("LLM_RESOURCE_SOURCE_CONFIG_FIELDS_CHANGED")
    for field in ("main_registry_path", "proc_root", "cgroup_memory_root", "persistent_root", "gpu_lock_root"):
        if not Path(config[field]).is_absolute():
            raise ValueError("RESOURCE_SOURCE_PATH_MUST_BE_ABSOLUTE:" + field)
    if not isinstance(config["main_ready_sources"], list) or any(not Path(p).is_absolute() for p in config["main_ready_sources"]):
        raise ValueError("MAIN_READY_SOURCE_PATHS_REQUIRED")
    for field in ("minimum_gpu_free_mb", "minimum_memory_headroom_bytes", "minimum_persistent_free_bytes"):
        if isinstance(config[field], bool) or not isinstance(config[field], int) or config[field] <= 0:
            raise ValueError("POSITIVE_RESOURCE_ADMISSION_THRESHOLD_REQUIRED:" + field)
    if not isinstance(config["maximum_idle_utilization_percent"], int) or not 0 <= config["maximum_idle_utilization_percent"] <= 10:
        raise ValueError("INVALID_IDLE_UTILIZATION_THRESHOLD")
    if config["checkpoint_resume_pass"] is not True:
        raise ValueError("REAL_CHECKPOINT_RESUME_REQUIRED")
    for field in optional - {"stage_file_policy"}:
        if field in config and (type(config[field]) is not int or config[field] < 0):
            raise ValueError("INVALID_INODE_BUDGET:" + field)
    if "stage_file_policy" in config:
        descriptor = config["stage_file_policy"]
        if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
                or not Path(descriptor["path"]).is_absolute() or len(descriptor["sha256"]) != 64):
            raise ValueError("INVALID_STAGE_FILE_POLICY_DESCRIPTOR")
    return config


def read_small(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise ValueError(f"MISSING_OR_UNSAFE_LIVE_SOURCE:{path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"LIVE_SOURCE_NOT_OBJECT:{path}")
    return payload, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                     "mtime_ns": path.stat().st_mtime_ns}


def observation_time(payload):
    for field in ("observed_at", "updated_at", "written_at", "timestamp", "updated_epoch"):
        value = payload.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("LIVE_SOURCE_TIMESTAMP_HAS_NO_TIMEZONE")
            return stamp.timestamp()
    raise ValueError("LIVE_SOURCE_TIMESTAMP_UNAVAILABLE")


def fresh(payload, *, now):
    age = now - observation_time(payload)
    if not 0 <= age <= 120:
        raise ValueError(f"LIVE_SOURCE_STALE:age_seconds={age}")


def states(payload):
    """Inspect only status fields, not arbitrary paths or diagnostic prose."""
    found = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"state", "status", "phase", "owner_state"} and isinstance(value, str):
                found.add(value)
            elif isinstance(value, (dict, list)):
                found.update(states(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(states(value))
    return found


def memory_headroom(proc_root, cgroup_root):
    fields = dict(line.split(":", 1) for line in (proc_root / "meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].strip().split()[0]) * 1024
    if (cgroup_root / "memory.max").is_file():
        raw = (cgroup_root / "memory.max").read_text().strip()
        current = int((cgroup_root / "memory.current").read_text())
        limit = None if raw == "max" else int(raw)
    else:
        limit = int((cgroup_root / "memory.limit_in_bytes").read_text())
        current = int((cgroup_root / "memory.usage_in_bytes").read_text())
    return min(available, max(0, limit - current)) if limit is not None else available


def bounded_gpu_inventory():
    # Two driver queries must not hang the owner past its freshness budget.
    def run(*args, **kwargs):
        return subprocess.run(*args, timeout=15, **kwargs)
    return query_gpu_inventory(runner=run)


class ResourceSampler:
    """Refresh source values; idle history is accumulated only in this owner."""
    def __init__(self, config, gpu_index, gpu_uuid, *, inventory=bounded_gpu_inventory,
                 clock=time.time, monotonic=time.monotonic, task_family="llm", reach_contract=None):
        validate_resource_config(config)
        if task_family not in ("llm", "ours_reach"):
            raise ValueError("EXISTING_OWNER_FAMILY_NOT_SUPPORTED")
        self.task_family, self.reach_contract = task_family, reach_contract
        if task_family == "ours_reach":
            from src.eval.bace_reach_v2 import unseal
            value = unseal(Path(reach_contract["path"]))
            if (value.get("schema") != "bace_ours_reach_v2_20260907"
                or value.get("proposal_source") != "OURS_MAIN_PPO_66"
                or value.get("test_opened") is not False
                or sha256_file(reach_contract["path"]) != reach_contract["sha256"]
                or int(gpu_index) != 0):
                raise ValueError("OURS_REACH_EXACT_TRAIN_SCOPE_REQUIRED")
        self.config, self.index, self.uuid = config, int(gpu_index), gpu_uuid
        self.inventory, self.clock, self.monotonic = inventory, clock, monotonic
        self.idle_since = None
        self.last_sample = None
        self.admitted_idle_seconds = None
        self.file_policy = (load_stage_policy(config["stage_file_policy"], config["persistent_root"])
                            if "stage_file_policy" in config else None)
        self.initial_free_slots = None

    def sample(self, *, child_pid=None, child_start_ticks=None):
        now, tick = self.clock(), self.monotonic()
        cfg, blockers, sources = self.config, [], []
        proc = Path(cfg["proc_root"])
        registry, source = read_small(cfg["main_registry_path"])
        # Registry is a declarative ownership/reservation map. Do not stamp it
        # fresh or believe stale owner_state: live PIDs/HBs are reopened below.
        registry = validate_owner_registry(registry, check_processes=False)
        sources.append({**source, "source_updated_at": registry.get("updated_at"),
                        "role": "declarative_registry_reopened"})
        reserved = any(row["gpu"] == self.index and row["state"] != "RELEASED"
                       for row in registry["gpu_leases"])
        pending = []
        heartbeat_paths = set(cfg["main_ready_sources"])
        identities = {}
        task_by_id = {row["task_id"]: row for row in registry["tasks"]}
        retired = set()
        for row in registry["tasks"]:
            successor = task_by_id.get(row.get("successor_task_id"))
            terminal = row["owner_state"] in {"TERMINAL_FAILED_ENGINEERING", "SUPERSEDED", "RETIRED"} or (
                row["owner_state"] == "BLOCKED" and successor is not None)
            held = any(lease["task_id"] == row["task_id"] and lease["state"] != "RELEASED"
                       for lease in registry["gpu_leases"])
            same_cell_successor = successor and all(successor[k] == row[k] for k in ("dataset", "method"))
            alternate = any(other["task_id"] != row["task_id"]
                and all(other[k] == row[k] for k in ("dataset", "method"))
                and other["owner_state"] not in FAILED
                and process_start_ticks(proc, other.get("owner_pid")) == other.get("owner_start_ticks")
                and other.get("owner_pid") for other in registry["tasks"])
            if terminal and not held and (same_cell_successor or alternate):
                retired.add(row["task_id"])
                sources.append({"task_id": row["task_id"], "role": "EXPLICIT_TERMINAL_WITH_SAME_CELL_SUCCESSOR",
                                "reservation_released_by_this_sampler": False})
        if not heartbeat_paths:
            blockers.append("MAIN_READY_SOURCE_COVERAGE_UNAVAILABLE")
        # A configured predecessor source can be retired only through the
        # existing registry's explicit same-cell successor and released lease.
        # The successor's real PID/heartbeat is still checked below. This does
        # not suppress an arbitrary stale source or clear any reservation.
        heartbeat_paths.difference_update(row.get("heartbeat") for row in registry["tasks"]
                                         if row["task_id"] in retired)
        for row in registry["tasks"]:
            if row["task_id"] in retired:
                continue
            # Failed primary work keeps its declared GPU until the authority
            # explicitly releases it. An empty nvidia-smi list is not release.
            if row["gpu"] == self.index and row["owner_state"] != "PASS":
                reserved = True
            if row["owner_state"] in READY:
                pending.append(row["task_id"])
            if row["owner_state"] == "PASS":
                continue
            pid, expected = row.get("owner_pid"), row.get("owner_start_ticks")
            if not pid or process_start_ticks(proc, pid) != expected:
                blockers.append("MAIN_OWNER_NOT_LIVE:" + row["task_id"])
            if row.get("heartbeat"):
                heartbeat_paths.add(row["heartbeat"])
                identities[row["heartbeat"]] = (pid, expected)
            else:
                blockers.append("MAIN_OWNER_HEARTBEAT_UNAVAILABLE:" + row["task_id"])
        for row in registry["publishers"]:
            if row["owner_state"] in {"PASS", "SUPERSEDED_DUPLICATE_CLAIM"}:
                continue
            if (row["owner_state"] == "PREDEPLOYED" and row.get("owner_pid") is None
                    and any(task.get("publisher_id") == row["publisher_id"] and task.get("heartbeat")
                            and task.get("owner_pid") and process_start_ticks(proc, task["owner_pid"]) == task.get("owner_start_ticks")
                            for task in registry["tasks"] if task["task_id"] not in retired)):
                # This is an explicit conditional successor, not an empty READY
                # queue. Its live predecessor is sampled above and still blocks
                # launch when ready, stale or failed. No fabricated publisher PID.
                sources.append({"publisher_id": row["publisher_id"], "role": "PREDEPLOYED_BOUND_TO_LIVE_PREDECESSOR"})
                continue
            if row.get("heartbeat"):
                heartbeat_paths.add(row["heartbeat"])
                identities[row["heartbeat"]] = (row.get("owner_pid"), row.get("owner_start_ticks"))
            elif row["owner_state"] not in {"PASS", "SUPERSEDED_DUPLICATE_CLAIM"}:
                blockers.append("PUBLISHER_READY_SOURCE_UNAVAILABLE:" + row["publisher_id"])
        for path in sorted(heartbeat_paths):
            try:
                heartbeat, source = read_small(path)
                # A live writer may publish after sampling began. Compare with
                # the clock after this read, without changing source timestamps
                # or relaxing rejection of genuinely future/expired evidence.
                fresh(heartbeat, now=self.clock())
            except (ValueError, OSError) as exc:
                blockers.append(f"MAIN_SOURCE_NOT_CURRENT:{path}:{exc}")
                continue
            if path in identities:
                pid, ticks = identities[path]
                actual_pid = heartbeat.get("owner_pid", heartbeat.get("pid"))
                actual_ticks = heartbeat.get("owner_start_ticks", heartbeat.get("start_ticks"))
                if (not pid or actual_pid != pid or process_start_ticks(proc, pid) != ticks
                        or (actual_ticks is not None and actual_ticks != ticks)):
                    blockers.append("MAIN_HEARTBEAT_PROCESS_BINDING_CHANGED:" + path)
            source["source_observed_at"] = observation_time(heartbeat)
            sources.append(source)
            seen = states(heartbeat)
            if seen & READY or any("GPU" in value and ("READY" in value or "WAIT" in value) for value in seen):
                pending.append(path)
            if seen & FAILED:
                blockers.append("MAIN_HEARTBEAT_NOT_HEALTHY:" + path)
        observations = self.inventory()
        matched = [row for row in observations if row.uuid == self.uuid and row.index == self.index]
        if len(matched) != 1:
            raise ValueError("GPU_INDEX_UUID_MAPPING_CHANGED")
        gpu = matched[0]
        if child_pid is not None and process_start_ticks(proc, child_pid) != child_start_ticks:
            raise ValueError("LLM_CHILD_PROCESS_GENERATION_CHANGED")
        foreign = [p.pid for p in gpu.processes if p.pid != child_pid]
        headroom = memory_headroom(proc, Path(cfg["cgroup_memory_root"]))
        disk = os.statvfs(cfg["persistent_root"])
        free_bytes = disk.f_frsize * disk.f_bavail
        if self.initial_free_slots is None:
            self.initial_free_slots = disk.f_favail
        file_admission = config_file_admission(cfg, disk.f_favail, stage_id="llm_gpu_generation",
            policy=self.file_policy, baseline_available=self.initial_free_slots)
        required_inodes = file_admission["required_free_inodes"]
        clean_idle = (not gpu.processes and not reserved and not pending and not blockers
                      and gpu.memory_free_mb >= cfg["minimum_gpu_free_mb"]
                      and gpu.utilization_gpu_percent <= cfg["maximum_idle_utilization_percent"])
        if self.last_sample is not None and tick - self.last_sample > 120:
            self.idle_since = None
        self.last_sample = tick
        if child_pid is None:
            if not clean_idle:
                self.idle_since = None
            elif self.idle_since is None:
                self.idle_since = tick
            idle = 0 if self.idle_since is None else max(0, tick - self.idle_since)
        else:
            # This is the measured pre-admission idle duration, not a claim
            # that a GPU with our running CUDA process is currently idle.
            idle = self.admitted_idle_seconds or 0
        if foreign:
            blockers.append("FOREIGN_CUDA_PROCESS_ON_TARGET_GPU")
        if (self.task_family == "ours_reach" and child_pid is None
            and (gpu.memory_free_mb < cfg["minimum_gpu_free_mb"]
                 or gpu.utilization_gpu_percent > cfg["maximum_idle_utilization_percent"])):
            blockers.append("OURS_REACH_GPU_CAPACITY_NOT_ADMITTED")
        other_llm = 0
        lock_root = Path(cfg["gpu_lock_root"])
        for path in lock_root.glob("gpu-*.lock"):
            if path.is_symlink():
                raise ValueError("INDIRECT_GPU_LOCK_SOURCE")
            with path.open("r+") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    try:
                        metadata, _ = read_small(path)
                    except (ValueError, OSError) as exc:
                        # Existing main coordination leases may deliberately
                        # contain no JSON. Never interpret that as an idle GPU
                        # or rewrite the held file; retain it as a blocker.
                        declared = [row for row in registry['gpu_leases']
                                    if row['state'] == 'HELD' and Path(row['lease_path']) == path]
                        if len(declared) == 1:
                            # This is a known primary main lease, not another
                            # LLM. Its actual owner/heartbeat is already checked
                            # above and its target reservation remains intact.
                            sources.append({'path': str(path), 'role': 'HELD_DECLARED_MAIN_COORDINATION',
                                            'task_id': declared[0]['task_id'], 'gpu': declared[0]['gpu']})
                        else:
                            blockers.append(f"HELD_LEASE_METADATA_UNAVAILABLE:{path}:{type(exc).__name__}")
                        continue
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    continue
            if (metadata.get("state") == "LOCKED" and metadata.get("ablation_family") in ("llm", "ours_reach")
                    and metadata.get("pid") != os.getpid()):
                other_llm += 1
        return {"schema_version": RESOURCE_SCHEMA,
                "task_family": self.task_family, "reach_contract": self.reach_contract,
                "ours_reach_contract_verified": self.task_family == "ours_reach",
                "observed_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "source_observations": sources, "actual_gpu_observation": gpu.as_json(),
                "gpu_index": self.index, "gpu_uuid": self.uuid, "target_gpu_uuid": self.uuid,
                "gpu_lease_mode": "EXCLUSIVE_IDLE", "logical_device": "cuda:0",
                "gpu_idle_seconds": idle, "gpu_idle_duration_kind": "PRE_ADMISSION_MEASURED" if child_pid else "CONTINUOUS_OBSERVED",
                "gpu_main_reservation": reserved, "main_ready_waiting_gpu": bool(pending),
                "main_ready_sources": pending, "source_blockers": blockers,
                "owners_healthy": not blockers, "registry_healthy": True,
                "memory_headroom_bytes": headroom, "persistent_free_bytes": free_bytes,
                "memory_safe": headroom >= cfg["minimum_memory_headroom_bytes"],
                "free_inodes": disk.f_favail, "required_free_inodes": required_inodes,
                "stage_file_admission": file_admission,
                "stage_file_policy": cfg.get("stage_file_policy"),
                "pause_requested": file_admission["pause_requested"],
                "storage_safe": free_bytes >= cfg["minimum_persistent_free_bytes"] and file_admission["admitted"],
                "checkpoint_resume_pass": cfg["checkpoint_resume_pass"] is True,
                "active_early_ablation_gpus": other_llm,
                "active_gpu_count_semantics": "OTHER_OWNERS_EXCLUDING_VERIFIED_CURRENT_LEASE",
                "max_llm_gpus": 1, "borrow_enabled": False}


def validate_inherited_lease(evidence, held_fd, slot_fd=None):
    """Independent contention, process generation and descriptor identities."""
    fresh(evidence, now=time.time())
    if evidence.get("schema_version") != RESOURCE_SCHEMA or evidence.get("pause_requested"):
        raise ValueError("WAITING_RESOURCE_OR_INVALID_OWNER_EVIDENCE")
    if evidence.get("gpu_lease_mode") != "EXCLUSIVE_IDLE" or evidence.get("borrow_enabled") is not False:
        raise ValueError("GPU_BORROW_FORBIDDEN")
    if evidence.get("gpu_child_pid") != os.getpid() or evidence.get("gpu_owner_pid") != os.getppid():
        raise ValueError("OWNER_CHILD_PID_BINDING_MISMATCH")
    proc = Path(evidence["proc_root"])
    for role in ("owner", "child"):
        if process_start_ticks(proc, evidence[f"gpu_{role}_pid"]) != evidence[f"gpu_{role}_start_ticks"]:
            raise ValueError("OWNER_CHILD_START_TICKS_CHANGED")
    if slot_fd is None:
        raise ValueError("HELD_LLM_SINGLE_SLOT_FD_REQUIRED")
    for fd, field in ((held_fd, "gpu_lock_path"), (slot_fd, "project_slot_path")):
        path = Path(evidence[field])
        if path.is_symlink() or not path.is_file():
            raise ValueError("OWNER_LOCK_DISAPPEARED")
        a, b = os.fstat(fd), path.stat()
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            raise ValueError("INHERITED_FD_INODE_MISMATCH")
        with path.open("r+") as competitor:
            try:
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(competitor, fcntl.LOCK_UN)
                raise ValueError("INHERITED_DESCRIPTOR_IS_NOT_A_HELD_LOCK")
        metadata = json.loads(os.pread(fd, 65536, 0))
        for key in ("gpu_uuid", "gpu_index", "owner_nonce", "gpu_child_pid", "gpu_child_start_ticks"):
            if metadata.get(key) != evidence.get(key):
                raise ValueError("LOCK_METADATA_CHILD_UUID_BINDING_MISMATCH")
        if metadata.get("state") != "LOCKED" or metadata.get("pid") != evidence["gpu_owner_pid"]:
            raise ValueError("LOCK_METADATA_OWNER_MISMATCH")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != evidence["gpu_uuid"]
            or evidence.get("target_gpu_uuid") != evidence["gpu_uuid"]
            or evidence.get("logical_device") != "cuda:0"):
        raise ValueError("GPU_UUID_TO_CUDA0_MAPPING_MISMATCH")
    family = evidence.get("task_family", "llm")
    if family == "ours_reach":
        descriptor = evidence.get("reach_contract", {})
        if not descriptor.get("path") or sha256_file(descriptor["path"]) != descriptor.get("sha256"):
            raise ValueError("OURS_REACH_CHILD_CONTRACT_CHANGED")
    decision = gpu_allowed({**evidence, "gnn_core_seed7_audit": "PASS"}, family=family)
    if not decision["allowed"]:
        raise ValueError("WAITING_RESOURCE:" + ",".join(decision["blockers"]))
    return True


def receive_owner_binding():
    """One inherited startup pipe; never evaluate a command supplied in it."""
    fd = int(os.environ.pop("AUTODL_LLM_OWNER_BOOTSTRAP_FD"))
    with os.fdopen(fd, "rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError("OWNER_BOOTSTRAP_OVERSIZED")
    binding = json.loads(raw)
    if binding["gpu_child_pid"] != os.getpid() or binding["gpu_owner_pid"] != os.getppid():
        raise ValueError("OWNER_BOOTSTRAP_PID_MISMATCH")
    seal_held_descriptors(binding["held_gpu_lock_fd"], binding["held_project_slot_fd"])
    return binding


def seal_held_descriptors(*fds):
    """Retain leases here; prevent leakage through either fork or exec."""
    identities = []
    for fd in fds:
        os.set_inheritable(fd, False)
        stat = os.fstat(fd)
        identities.append((fd, stat.st_dev, stat.st_ino))
    def close_in_forked_child():
        for fd, device, inode in identities:
            try:
                stat = os.fstat(fd)
                if (stat.st_dev, stat.st_ino) == (device, inode):
                    os.close(fd)
            except OSError:
                pass
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=close_in_forked_child)


def _bind_child(lock, binding):
    handle = lock._handle
    handle.seek(0)
    payload = json.load(handle)
    payload.update({key: binding[key] for key in ("gpu_child_pid", "gpu_child_start_ticks")})
    handle.seek(0)
    handle.truncate()
    json.dump(payload, handle, sort_keys=True)
    handle.flush()
    os.fsync(handle.fileno())


def next_completed_evaluation(dispatch):
    """Select only a real complete pool; no future PID or completion inference."""
    for variant in dispatch["variant_order"]:
        row = dispatch["downstream_commands"][variant]
        candidate = Path(row["candidate_root"]) / "candidate_generation_receipt.json"
        if not candidate.is_file():
            if "parser_correction_overlay" in dispatch:
                raise ValueError("CORRECTION_COMPLETED_GENERATION_DISAPPEARED")
            return None
        generated, _ = read_small(candidate)
        if generated.get("status") != "CANDIDATE_POOL_PASS":
            if "parser_correction_overlay" in dispatch:
                raise ValueError("CORRECTION_COMPLETED_GENERATION_NOT_PASS")
            return None
        task, _ = read_small(row["task_spec"]["path"])
        if (sha256_file(row["task_spec"]["path"]) != row["task_spec"]["sha256"]
                or generated.get("spec_sha256") != canonical_sha(task)
                or generated.get("variant") != variant or generated.get("next_call") != len(task["calls"])):
            raise ValueError("DOWNSTREAM_COMPLETED_POOL_BINDING_CHANGED")
        output = Path(row["output_root"])
        final = output / "final_audit.json"
        if final.is_file():
            audit, _ = read_small(final)
            manifest, _ = read_small(output / "run_manifest.json")
            if (audit.get("state") != "PASS" or audit.get("main_matrix_write") is not False
                    or manifest.get("task_spec_sha256") != row["task_spec"]["sha256"]):
                raise ValueError("DOWNSTREAM_FINAL_BINDING_CHANGED")
            if "parser_correction_overlay" in dispatch:
                from src.eval.bace_frozen_gnn_contracts import stable_sha256
                correction, _ = read_small(output / "parser_correction_receipt.json")
                correction_body = {key: value for key, value in correction.items() if key != "self_sha256"}
                expected_source = dispatch["downstream_commands"][variant]["command"]
                source_sha = expected_source[expected_source.index("--parser-correction-source-audit-sha256") + 1]
                if (manifest.get("execution_commit") != dispatch["execution_commit"]
                        or manifest.get("binding_sha256") != audit.get("binding_sha256")
                        or correction.get("self_sha256") != stable_sha256(correction_body)
                        or manifest.get("pool", {}).get("parser_correction_sha256") != stable_sha256(correction_body)
                        or correction.get("parser_contract") != "structural_wrapper_extraction_v2"
                        or manifest.get("parser_distance_reuse", {}).get("audit_sha256") != source_sha
                        or audit.get("files", {}).get("parser_correction_receipt.json") != sha256_file(output / "parser_correction_receipt.json")):
                    raise ValueError("CORRECTIVE_FINAL_PARSER_BINDING_CHANGED")
            continue
        return {"variant": variant, **row, "resume": output.exists()}
    return None


def canonical_sha(payload):
    from src.ablations.llm.contracts import canonical_json_sha256
    return canonical_json_sha256(payload)


def run_owned_child(*, command, environment, sampler, output_root, lock_root, run_id,
                    interval=30, max_wait_seconds=0):
    """One owner lifecycle. Never unlock while its launched child is alive."""
    if not 0 < interval <= 60:
        raise ValueError("Owner refresh interval must be in (0,60]")
    if not 0 <= max_wait_seconds <= 86400:
        raise ValueError("Owner wait must be bounded to 0..86400 seconds")
    output = Path(output_root)
    family = getattr(sampler, "task_family", "llm")
    if family == "ours_reach":
        if (len(command) < 5 or Path(command[3]).name != "run_bace_reach_v2.py"
            or command.count("--action") != 1
            or command[command.index("--action") + 1] not in ("canary", "train-search")):
            raise ValueError("OURS_REACH_OWNER_CANNOT_DISPATCH_LLM_OR_PPO")
    output.mkdir(parents=True, exist_ok=False)
    pause = False
    def request_pause(*_):
        nonlocal pause
        pause = True
    handlers = {sig: signal.signal(sig, request_pause) for sig in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    child = None
    terminal = {"state": "WAITING_RESOURCE", "gpu_started": False}
    try:
        while True:
            try:
                evidence = sampler.sample()
                decision = gpu_allowed({**evidence, "gnn_core_seed7_audit": "PASS"}, family=family)
            except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                sampler.idle_since = None
                evidence = {"observed_at": datetime.now(timezone.utc).isoformat(), "source_error": str(exc)}
                decision = {"allowed": False, "blockers": [str(exc)]}
            atomic_json(output / "heartbeat.json", {"state": "WAITING_RESOURCE", "pid": os.getpid(),
                        "observed_at": datetime.now(timezone.utc).isoformat(), "decision": decision})
            atomic_json(output / "resource_evidence.json", evidence)
            if decision["allowed"] and not pause:
                break
            if pause or max_wait_seconds <= 0 or time.monotonic() - started >= max_wait_seconds:
                terminal.update(blockers=decision["blockers"])
                return 75
            time.sleep(max(0, min(interval, max_wait_seconds - (time.monotonic() - started))))
        sampler.admitted_idle_seconds = evidence["gpu_idle_seconds"]
        nonce = str(uuid.uuid4())
        owner = {"run_id": run_id, "ablation_family": family, "owner_nonce": nonce,
                 "gpu_uuid": sampler.uuid, "gpu_index": sampler.index, "command": command}
        with ExitStack() as stack:
            slot = stack.enter_context(ProjectGPUSlotLock(Path(lock_root) / "llm-ablation", max_slots=1, owner=owner))
            gpu = stack.enter_context(GPUFileLock(Path(lock_root), gpu_index=sampler.index, gpu_uuid=sampler.uuid, owner=owner))
            # Re-sample after both existing locks; an idle preflight is not a lease.
            evidence = sampler.sample()
            decision = gpu_allowed({**evidence, "gnn_core_seed7_audit": "PASS"}, family=family)
            if not decision["allowed"] or pause:
                terminal.update(blockers=decision["blockers"])
                return 75
            read_fd, write_fd = os.pipe()
            env = dict(environment, CUDA_VISIBLE_DEVICES=sampler.uuid,
                       AUTODL_PHYSICAL_GPU_INDEX=str(sampler.index), AUTODL_PHYSICAL_GPU_UUID=sampler.uuid,
                       AUTODL_LLM_OWNER_BOOTSTRAP_FD=str(read_fd))
            fds = (gpu._handle.fileno(), slot._handle.fileno())
            try:
                child = subprocess.Popen(command, env=env, close_fds=True, pass_fds=(*fds, read_fd))
            except BaseException:
                os.close(write_fd)
                raise
            finally:
                os.close(read_fd)
            binding = {"gpu_owner_pid": os.getpid(), "gpu_child_pid": child.pid,
                       "gpu_owner_start_ticks": process_start_ticks(sampler.config["proc_root"], os.getpid()),
                       "gpu_child_start_ticks": process_start_ticks(sampler.config["proc_root"], child.pid),
                       "owner_nonce": nonce, "proc_root": sampler.config["proc_root"],
                       "gpu_lock_path": str(gpu.path), "project_slot_path": str(slot.path)}
            try:
                if binding["gpu_owner_start_ticks"] is None or binding["gpu_child_start_ticks"] is None:
                    raise ValueError("OWNER_CHILD_PROCESS_IDENTITY_UNAVAILABLE")
                _bind_child(gpu, binding)
                _bind_child(slot, binding)
                evidence.update(binding)
                initial = output / "initial_resource_evidence.json"
                live = output / "resource_evidence.json"
                atomic_json(initial, evidence)
                atomic_json(live, evidence)
                pipe_fd, write_fd = write_fd, None
                with os.fdopen(pipe_fd, "w") as pipe:
                    json.dump({**binding, "resource_evidence": str(initial), "resource_evidence_sha256": sha256_file(initial),
                               "resource_live_evidence": str(live), "held_gpu_lock_fd": fds[0],
                               "held_project_slot_fd": fds[1]}, pipe)
                terminal.update(state="RUNNING", gpu_started=True, child_pid=child.pid)
                while child.poll() is None:
                    try:
                        evidence = sampler.sample(child_pid=child.pid, child_start_ticks=binding["gpu_child_start_ticks"])
                        evidence.update(binding, pause_requested=pause or evidence.get("pause_requested", False))
                    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                        # Explicit error, not a fabricated refresh of old values.
                        evidence = {"schema_version": RESOURCE_SCHEMA, "pause_requested": True,
                                    "source_error": str(exc), "observed_at": datetime.now(timezone.utc).isoformat()}
                    atomic_json(live, evidence)
                    atomic_json(output / "heartbeat.json", {"state": "PAUSE_REQUESTED" if pause or evidence.get("pause_requested") else "RUNNING",
                                "pid": os.getpid(), "child_pid": child.pid, "observed_at": datetime.now(timezone.utc).isoformat()})
                    try:
                        child.wait(timeout=interval)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if write_fd is not None:
                    os.close(write_fd)
                # Do not SIGKILL, unlock, or claim released while the child still
                # owns CUDA. If owner setup failed, EOF aborts its startup pipe.
                child.wait()
            terminal.update(state="COMPLETE" if child.returncode == 0 else "PAUSED" if child.returncode == 75 else "FAILED",
                            returncode=child.returncode, gpu_released_after_child_exit=True)
            return child.returncode
    except BaseException as exc:
        terminal.update(state="FAILED", error=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        atomic_json(output / "terminal.json", {**terminal, "owner_pid": os.getpid(),
                    "written_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started})
