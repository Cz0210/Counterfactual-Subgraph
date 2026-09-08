#!/usr/bin/env python3
"""Probe or hold a UUID-scoped AutoDL GPU advisory lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# Compact immutable deployments carry the same-commit source archive; no
# dependence on PYTHONPATH (which isolated children intentionally ignore).
if (PROJECT_ROOT / "source.zip").is_file():
    sys.path.insert(0, str(PROJECT_ROOT / "source.zip"))
from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    GPUFileLock,
    build_runtime_layout,
    gpu_lock_available,
    resolve_project_root,
    sanitized_environment,
    select_data_root,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--project-root", type=Path)
    root.add_argument("--data-root", type=Path)
    root.add_argument("--config", action="append", default=[])
    commands = root.add_subparsers(dest="action", required=True)
    commands.add_parser("list")
    for name in ("probe", "run"):
        command = commands.add_parser(name)
        command.add_argument("--gpu-index", type=int, required=True)
        command.add_argument("--gpu-uuid", required=True)
        command.add_argument("--run-id")
        if name == "run":
            command.add_argument("--t13-performance-spec", type=Path)
            command.add_argument("--t13-performance-spec-sha256")
            command.add_argument("--llm-dispatch-spec", type=Path)
            command.add_argument("--llm-dispatch-spec-sha256")
            command.add_argument("--owner-output-root", type=Path)
            command.add_argument("--wait-seconds", type=float, default=86400)
            command.add_argument("--refresh-seconds", type=float, default=30)
            command.add_argument("--llm-complete-chain", action="store_true", help="Use the existing owner sequentially for L1/L2/L3 and their CPU evaluator")
            command.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if (args.action == "run" and args.t13_performance_spec
                and (PROJECT_ROOT / "source.zip").is_file()):
            # The diagnostic preflight is read-only and validates its sealed
            # execution path below; do not fabricate .git for zip deployments.
            if args.project_root is None or args.project_root.resolve() != PROJECT_ROOT:
                raise AutoDLRuntimeError("Compact T13 preflight requires its explicit immutable source root")
            project_root = PROJECT_ROOT
        else:
            project_root = resolve_project_root(args.project_root)
        data_root = select_data_root(project_root, explicit=args.data_root)
        layout = build_runtime_layout(project_root=project_root, data_root=data_root).ensure()
        if args.action == "list":
            rows = []
            for path in sorted(layout.locks_dir.glob("gpu-*.lock")):
                try:
                    metadata = json.loads(path.read_text(encoding="utf-8") or "{}")
                except json.JSONDecodeError:
                    metadata = {"state": "CORRUPT"}
                rows.append({"path": str(path), "metadata": metadata})
            print(json.dumps({"locks": rows}, indent=2, sort_keys=True))
            return 0
        if args.action == "probe":
            available = gpu_lock_available(layout.locks_dir, args.gpu_uuid)
            print(
                json.dumps(
                    {
                        "gpu_index": args.gpu_index,
                        "gpu_uuid": args.gpu_uuid,
                        "lock_available": available,
                    },
                    sort_keys=True,
                )
            )
            return 0 if available else 3
        if args.t13_performance_spec:
            if args.llm_dispatch_spec or args.command or not args.t13_performance_spec_sha256 or not args.owner_output_root:
                raise AutoDLRuntimeError("T13 diagnostic requires sealed spec, fresh receipt root, and no injected command")
            from src.utils.t13_performance_dispatch import preflight
            return preflight(spec_descriptor={"path":str(args.t13_performance_spec),"sha256":args.t13_performance_spec_sha256},
                project_root=PROJECT_ROOT,gpu_index=args.gpu_index,gpu_uuid=args.gpu_uuid,
                lock_root=layout.locks_dir,output_root=args.owner_output_root)
        if args.llm_dispatch_spec:
            if args.command or not args.llm_dispatch_spec_sha256 or not args.owner_output_root:
                raise AutoDLRuntimeError("LLM dispatch requires sealed SHA, fresh owner root, and no injected command")
            from src.ablations.llm.bace_native_runtime import verified_file
            from src.ablations.llm.contracts import canonical_json_sha256
            from src.ablations.llm.existing_gpu_owner import DISPATCH_SCHEMA, ResourceSampler, run_owned_child, next_completed_evaluation, memory_headroom
            from src.utils.stage_file_policy import config_file_admission
            spec = json.loads(verified_file({"path": str(args.llm_dispatch_spec), "sha256": args.llm_dispatch_spec_sha256}).read_text())
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
            from src.ablations.llm.stage_dispatch_binding import validate_dispatch_runtime
            dispatch_binding = validate_dispatch_runtime(spec, commit, PROJECT_ROOT)
            if (dispatch_binding.get("parser_correction") and
                    (not args.llm_complete_chain or str(args.owner_output_root.resolve()) != dispatch_binding["owner_output_root"])):
                raise AutoDLRuntimeError("Parser correction requires its single bound CPU queue root")
            config = json.loads(verified_file(spec["resource_config"]).read_text())
            if Path(config["gpu_lock_root"]).resolve() != layout.locks_dir.resolve():
                raise AutoDLRuntimeError("LLM must use the existing AutoDL project UUID lock root")
            sampler = ResourceSampler(config, args.gpu_index, args.gpu_uuid)
            if args.llm_complete_chain:
                if set(spec.get("downstream_commands", {})) != set(spec["variant_order"]):
                    raise AutoDLRuntimeError("Complete chain requires sealed executable common evaluation commands")
                from src.eval.bace_frozen_gnn_contracts import atomic_json
                queue_root = args.owner_output_root
                queue_root.mkdir(parents=True, exist_ok=False)
                # At most three treatment generations and three evaluations;
                # pauses/failures stop this invocation, not automatic fresh retry.
                stage = 0
                while stage < 7:
                    pending = next_completed_evaluation(spec)
                    if pending:
                        disk = os.statvfs(config["persistent_root"])
                        file_admission = config_file_admission(config, disk.f_favail,
                            stage_id="llm_cpu_evaluation", policy=sampler.file_policy)
                        if (memory_headroom(Path(config["proc_root"]), Path(config["cgroup_memory_root"])) < config["minimum_memory_headroom_bytes"]
                                or not file_admission["admitted"]
                                or disk.f_bavail * disk.f_frsize < config["minimum_persistent_free_bytes"]):
                            atomic_json(queue_root / "queue_status.json", {"state": "WAITING_CPU_RESOURCE", "variant": pending["variant"], "science_started": False, "file_admission": file_admission})
                            time.sleep(min(60, args.refresh_seconds))
                            continue
                        command = list(pending["command"])
                        if pending["resume"]:
                            command.append("--resume")
                            pause_path = Path(pending["output_root"]) / "pause.request"
                            if pause_path.exists():
                                pause_receipt = json.loads(pause_path.read_text())
                                if (pause_receipt.get("reason") != "NEXT_CHECKPOINT_RESOURCE_GUARD"
                                        or pause_receipt.get("owner_control") != str(queue_root)
                                        or pause_receipt.get("variant") != pending["variant"]):
                                    raise AutoDLRuntimeError("Unbound pause request cannot be consumed")
                                os.replace(pause_path, queue_root / f"consumed-pause-{stage}.json")
                        env = dict(sanitized_environment(), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                        child = subprocess.Popen(command, env=env, close_fds=True)
                        atomic_json(queue_root / "queue_status.json", {"state": "CPU_COMMON_EVALUATION", "variant": pending["variant"], "science_pid": child.pid, "gpu_lease_held": False})
                        # Existing evaluator reads pause.request before each
                        # parent and preserves its hash-bound parent checkpoint.
                        # No signal or second resource controller is required.
                        while child.poll() is None:
                            disk = os.statvfs(config["persistent_root"])
                            sampled = config_file_admission(config, disk.f_favail,
                                stage_id="llm_cpu_evaluation", policy=sampler.file_policy)
                            memory_ok = memory_headroom(Path(config["proc_root"]), Path(config["cgroup_memory_root"])) >= config["minimum_memory_headroom_bytes"]
                            bytes_ok = disk.f_bavail * disk.f_frsize >= config["minimum_persistent_free_bytes"]
                            if ((sampled["pause_requested"] or not memory_ok or not bytes_ok)
                                    and Path(pending["output_root"]).is_dir()):
                                atomic_json(Path(pending["output_root"]) / "pause.request",
                                    {"reason": "NEXT_CHECKPOINT_RESOURCE_GUARD", "file_admission": sampled,
                                     "memory_safe": memory_ok, "bytes_safe": bytes_ok,
                                     "owner_control": str(queue_root), "variant": pending["variant"]})
                            atomic_json(queue_root / "cpu_resource_evidence.json",
                                {"observed_at": time.time(), "file_admission": sampled,
                                 "memory_safe": memory_ok, "bytes_safe": bytes_ok, "child_pid": child.pid})
                            try:
                                child.wait(timeout=min(60, args.refresh_seconds))
                            except subprocess.TimeoutExpired:
                                pass
                        result = child.returncode
                        if result == 75:
                            # Resource pause retains the same queue owner and
                            # committed parent state. Re-admit, then same-root
                            # resume; never relaunch generation or completed OT.
                            time.sleep(min(60, args.refresh_seconds))
                            continue
                        if result != 0 or not (Path(pending["output_root"]) / "final_audit.json").is_file():
                            atomic_json(queue_root / "queue_status.json", {"state": "FAILED_OR_PAUSED", "returncode": result, "variant": pending["variant"]})
                            return result or 75
                        stage += 1
                        continue
                    if "parser_correction_overlay" in spec:
                        # All three raw pools were independently bound at startup.
                        # This branch can never fall through to GPU generation.
                        atomic_json(queue_root / "queue_status.json", {
                            "state": "ALL_THREE_CORRECTIVE_CPU_EVALUATIONS_PASS", "main_matrix_write": False})
                        return 0
                    # A --plan-only read does not acquire a lease or start models.
                    plan = subprocess.run(spec["command"] + ["--plan-only"], check=True, capture_output=True, text=True)
                    decision = json.loads(plan.stdout.strip().splitlines()[-1])
                    if decision["state"] == "ALL_THREE_GENERATION_POOLS_COMPLETE":
                        atomic_json(queue_root / "queue_status.json", {"state": "ALL_THREE_VARIANTS_COMMON_EVALUATION_PASS", "main_matrix_write": False})
                        return 0
                    if decision["state"] != "READY_WAITING_EXISTING_IDLE_GPU_OWNER":
                        atomic_json(queue_root / "queue_status.json", decision)
                        return 75
                    atomic_json(queue_root / "queue_status.json", {"state": "WAITING_RESOURCE", "variant": decision["variant"], "owner_pid": os.getpid(), "science_started": False})
                    result = run_owned_child(command=spec["command"], environment=sanitized_environment(), sampler=sampler,
                        output_root=queue_root / f"generation-owner-{stage}", lock_root=layout.locks_dir, run_id=args.run_id,
                        interval=args.refresh_seconds, max_wait_seconds=args.wait_seconds)
                    if result != 0:
                        return result
                    stage += 1
                raise AutoDLRuntimeError("Bounded three-variant chain exceeded stage count")
            return run_owned_child(command=spec["command"], environment=sanitized_environment(), sampler=sampler,
                       output_root=args.owner_output_root, lock_root=layout.locks_dir, run_id=args.run_id,
                       interval=args.refresh_seconds, max_wait_seconds=args.wait_seconds)
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise AutoDLRuntimeError("gpu_lock.py run requires a command after --")
        owner = {"run_id": args.run_id, "command": command}
        with GPUFileLock(
            layout.locks_dir,
            gpu_index=args.gpu_index,
            gpu_uuid=args.gpu_uuid,
            owner=owner,
        ):
            environment = sanitized_environment()
            # Legacy main-table consumers require the physical numeric selector.
            # LLM dispatch has its own UUID/FD owner branch above and is unchanged.
            environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
            environment["AUTODL_PHYSICAL_GPU_INDEX"] = str(args.gpu_index)
            environment["AUTODL_PHYSICAL_GPU_UUID"] = str(args.gpu_uuid)
            completed = subprocess.run(command, env=environment, check=False)
            return int(completed.returncode)
    except (AutoDLRuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"AUTODL_GPU_LOCK_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
