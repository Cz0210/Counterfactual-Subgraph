#!/usr/bin/env python3
"""Run the five-batch/shared-rules Taste diagnostic through an existing owner.

Only the explicit v2 plan is accepted: two updates per arm (six total), with
the complete original validation loader whenever its epoch boundary is due.
Old two-train/one-validation-batch plans are not recovery evidence.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if (ROOT / "source.zip").is_file():
    sys.path.insert(0, str(ROOT / "source.zip"))
    # The pinned official provenance checker opens three project .py leaves.
    # Keep those exact same-commit leaves physical; other modules may remain
    # in the compact source archive. This does not weaken its source checks.
    import src
    src.__path__.insert(0, str(ROOT / 'src'))
    import src.baselines
    if str(ROOT / 'source.zip/src/baselines') not in src.baselines.__path__:
        src.baselines.__path__.append(str(ROOT / 'source.zip/src/baselines'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--action", choices=("inspect", "run", "status", "verify-checkpoint"), default="inspect")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-state-sha256")
    parser.add_argument("--held-gpu-fd", type=int)
    parser.add_argument("--owner-evidence", type=Path)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        parser.error("Use the pinned interpreter with -I -B")
    if not args.config.is_file() or args.set != ["inference.fallback_to_heuristic=false"]:
        parser.error("Existing config and explicit heuristic prohibition required")
    from src.baselines.t13_real_batch_performance import validate_plan
    if args.action == "verify-checkpoint":
        import torch
        from src.baselines.t13_indexed_canary import state_digest
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if state_digest(saved) != args.expected_state_sha256:
            raise ValueError("T13_INDEPENDENT_PROCESS_CHECKPOINT_DIGEST_MISMATCH")
        print(json.dumps(dict(state="INDEPENDENT_CHECKPOINT_CONTAINER_PASS", model_training=False)))
        return 0
    if args.plan is None:
        parser.error("--plan is required")
    plan = validate_plan(json.loads(args.plan.read_text()))
    output = Path(plan["output_root"])
    if args.action == "status":
        terminal = output / "performance.json"
        print(terminal.read_text() if terminal.is_file() else json.dumps(
            dict(state="NOT_STARTED", scope=plan["scope"], active_handover_ready=False)))
        return 0
    if args.action == "inspect":
        paths = {key: Path(plan[key]).exists() for key in (
            "source_checkpoint", "source_index_manifest", "source_cohort_manifest", "train_csv",
            "gnn_checkpoint", "official_root", "gspan_adoption_proof", "reference_source")}
        descriptor = plan.get('committed_compact_payload') or {}
        paths['committed_compact_payload'] = bool(descriptor.get('path')) and Path(descriptor['path']).is_file()
        print(json.dumps(dict(state="INPUT_PATHS_PRESENT" if all(paths.values()) else "MISSING_INPUT",
            paths=paths, synthetic=False, science_started=False, active_handover_ready=False), sort_keys=True))
        return 0 if all(paths.values()) else 2
    if not plan.get('committed_compact_payload'):
        parser.error('COMPACT_MASK_INDEX_PAYLOAD_MISSING_NOT_RESEEDABLE')
    use_owner_pipe = 'AUTODL_LLM_OWNER_BOOTSTRAP_FD' in os.environ
    if use_owner_pipe:
        if args.held_gpu_fd is not None or args.owner_evidence is not None:
            parser.error('Do not mix the inherited owner pipe with manual FD fields')
        from src.ablations.llm.existing_gpu_owner import receive_owner_binding
        binding = receive_owner_binding()
        args.held_gpu_fd = binding['held_gpu_lock_fd']
        args.owner_evidence = Path(binding['resource_live_evidence'])
    if args.held_gpu_fd is None or args.owner_evidence is None:
        parser.error("run requires the existing owner's actually inherited GPU FD and fresh evidence")
    if output.exists():
        raise ValueError("T13_REAL_CANARY_REQUIRES_FRESH_OUTPUT")

    def resources_and_lease():
        evidence = json.loads(args.owner_evidence.read_text())
        if use_owner_pipe:
            from src.utils.t13_performance_dispatch import child_evidence
            evidence = child_evidence(evidence, plan_sha=plan_sha)
        age = time.time() - float(evidence["observed_at_epoch_seconds"])
        if not 0 <= age <= 120 or evidence.get("resource_admission") != "PASS":
            raise ValueError("T13_CANARY_STALE_OR_FAILED_OWNER_EVIDENCE")
        if (evidence.get("owner_pid") != os.getppid() or evidence.get("child_pid") != os.getpid()
                or evidence.get("gpu_index") != 2 or evidence.get("borrow_enabled") is not False
                or evidence.get("target_gpu_uuid") != os.environ.get("CUDA_VISIBLE_DEVICES")
                or evidence.get("plan_sha256") != plan_sha):
            raise ValueError("T13_CANARY_OWNER_GPU_OR_PLAN_BINDING_CHANGED")
        from src.ablations.llm.existing_gpu_owner import process_start_ticks
        for role, pid in (("owner", os.getppid()), ("child", os.getpid())):
            if process_start_ticks(Path("/proc"), pid) != evidence[role + "_start_ticks"]:
                raise ValueError("T13_CANARY_OWNER_GENERATION_CHANGED")
        path = Path(evidence["gpu_lock_path"])
        if path.is_symlink():
            raise ValueError("T13_CANARY_LOCK_SYMLINK_FORBIDDEN")
        a, b = os.fstat(args.held_gpu_fd), path.stat()
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            raise ValueError("T13_CANARY_INHERITED_FD_IDENTITY_CHANGED")
        with path.open("r+") as competitor:
            try:
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(competitor, fcntl.LOCK_UN)
                raise ValueError("T13_CANARY_EXISTING_LEASE_NOT_EXCLUSIVE")
        meta = json.loads(os.pread(args.held_gpu_fd, 65536, 0))
        if meta.get("pid") != os.getppid() or meta.get("gpu_uuid") != evidence["target_gpu_uuid"]:
            raise ValueError("T13_CANARY_EXISTING_LOCK_OWNER_MISMATCH")
        from src.utils.t13_lazy_recovery_guard import resources
        actual = resources("/autodl-fs/data/counterfactual-subgraph-runtime")
        rss = 0
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
        needed = int(evidence["other_task_headroom_required_bytes"]) + max(0, plan["process_peak_budget_bytes"] - rss)
        if actual["headroom_bytes"] < needed:
            raise ValueError("T13_CANARY_OTHER_MAIN_HEADROOM_NOT_PRESERVED")
        if actual["free_bytes"] < plan["persistent_free_required_bytes"] or actual["free_inodes"] < plan["free_inodes_required"]:
            raise ValueError("T13_CANARY_PERSISTENT_ADMISSION_FAILED")
        return actual

    import hashlib
    plan_sha = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    resources_and_lease()
    os.set_inheritable(args.held_gpu_fd, False)
    output.mkdir(parents=True, exist_ok=False)
    from src.eval.bace_frozen_gnn_contracts import atomic_json
    from src.baselines.t13_real_batch_performance import (PerformanceComplete, make_interceptor,
        reference_module, snapshot_checkpoint)
    started = time.monotonic()
    samples = []
    monitor_stop = threading.Event()
    monitor_errors = []
    def monitor():
        while not monitor_stop.wait(60):
            try:
                observed = resources_and_lease()
                atomic_json(output / "resource_progress.json", dict(observed,
                    sampled_at_epoch_seconds=time.time(), owner_evidence_reread=True))
            except Exception as exc:
                monitor_errors.append(str(exc))
                return
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    def sample(phase):
        if monitor_errors:
            raise ValueError("T13_OWNER_RESOURCE_MONITOR_FAILED:" + monitor_errors[0])
        actual = resources_and_lease()
        from src.baselines.t13_indexed_canary import memory_snapshot
        row = {**memory_snapshot(phase), **actual, "elapsed_seconds": time.monotonic() - started}
        if row.get("VmHWM_bytes", 0) > plan["process_peak_budget_bytes"]:
            raise ValueError("T13_CANARY_PROCESS_PEAK_EXCEEDED")
        samples.append(row)
        atomic_json(output / "memory_boundaries.json", dict(samples=samples))
    def timeout(signum, frame):
        raise TimeoutError("T13_REAL_CANARY_1800_SECOND_WALL_LIMIT")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(plan["max_wall_seconds"])
    report = None
    try:
        # The owner establishes CUBLAS before creating this child, never afterwards.
        import torch
        from src.utils.t13_performance_dispatch import source_backend
        backend_helper = source_backend(plan)
        apply_backend, BACKEND = backend_helper.apply_backend, backend_helper.BACKEND
        source_backend = json.loads(Path(plan["source_runtime_backend_receipt"]).read_text())
        if (source_backend["observed_backend"] != BACKEND
                or source_backend["torch_num_threads"] != plan["torch_num_threads"]
                or source_backend["torch_num_interop_threads"] != plan["torch_num_interop_threads"]):
            raise ValueError("T13_REAL_PLAN_DIFFERS_FROM_SOURCE_RUNTIME_BACKEND")
        for name, value in source_backend["thread_environment"].items():
            if os.environ.get(name) != value:
                raise ValueError("T13_REAL_CHILD_THREAD_ENV_CHANGED:" + name)
        backend = apply_backend(torch, BACKEND)
        torch.set_num_threads(plan["torch_num_threads"])
        torch.set_num_interop_threads(plan["torch_num_interop_threads"])
        atomic_json(output / "runtime_backend_receipt.json", dict(backend=backend, pid=os.getpid(),
                    gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"], formal_start=False))
        copied = snapshot_checkpoint(plan["source_checkpoint"], output / "source_checkpoint_snapshot.pt")
        atomic_json(output / "checkpoint_snapshot.json", copied)
        checkpoint = torch.load(copied["snapshot"], map_location="cpu", weights_only=False)
        if checkpoint["resume_identity"]["target_label"] != plan["target_label"]:
            raise ValueError("T13_SOURCE_CHECKPOINT_TARGET_CHANGED")
        original = reference_module(plan["reference_source"], plan["reference_source_sha256"])
        from src.baselines.tastemolnet_globalgce_full import (_checkpoint_payloads, load_full_train_split,
            select_full_sweet_train_cohort, FrozenTasteGINEScorer)
        from src.baselines.globalgce_bace_native_rules import validate_official_globalgce_root
        from src.baselines import globalgce_mutagenicity_adapter as adapter
        official = validate_official_globalgce_root(Path(plan["official_root"]))
        payloads = _checkpoint_payloads(Path(plan["gnn_checkpoint"]))
        split = json.loads(payloads["split_manifest.json"])
        # Train file is newly consumed here; this is not a rescan of unrelated packages.
        digest = hashlib.sha256()
        with Path(plan["train_csv"]).open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        observed_train_sha = digest.hexdigest()
        if observed_train_sha != split["files"]["train"]["sha256"]:
            raise ValueError("T13_REAL_TRAIN_INPUT_CHANGED")
        train = load_full_train_split(SimpleNamespace(train_path=Path(plan["train_csv"]),
            train_count=split["train_manifest"]["num_records"], train_label_counts=split["train_manifest"]["label_counts"]))
        scorer = FrozenTasteGINEScorer(payloads, device="cuda:0", batch_size=256)
        selected, cohort = select_full_sweet_train_cohort(train, scorer=scorer, batch_size=256)
        if cohort != json.loads(Path(plan["source_cohort_manifest"]).read_text()):
            raise ValueError("T13_REAL_SOURCE_COHORT_CHANGED")
        atomic_json(output / "train_cohort_manifest.json", cohort)
        del scorer
        generator = adapter.OfficialGlobalGCEMutagenicityGenerator(Path(plan["official_root"]),
            native_train_csv=Path(plan["train_csv"]), dataset_name="TasteMolNet", min_freq=2,
            frozen_gine_checkpoint=Path(plan["gnn_checkpoint"]), source_label=1,
            target_label=plan["target_label"], num_classes=3,
            official_source_authority=official["runtime_source_authority"], require_isolated_imports=True,
            rules_only_min_valid_native_rules=0)
        generator.t13_indexed_options = dict(storage="t13_indexed_augmentation_v1", diagnostic_profile="deterministic")
        previous_train = adapter.train_globalgce_resumable
        adapter.train_globalgce_resumable = make_interceptor(plan, checkpoint, original, output, sample)
        try:
            generator.generate(selected, output_dir=output / "native_preparation", seed=7, epochs=100,
                top_k_native=20, learning_rate=0.1, dropout=0.5, device="cuda:0", resume=False,
                gspan_adoption_proof=Path(plan["gspan_adoption_proof"]), rules_only=True)
        except PerformanceComplete as done:
            report = done.report
        else:
            raise ValueError("T13_CANARY_DID_NOT_STOP_BEFORE_FORMAL_TRAINING")
        finally:
            adapter.train_globalgce_resumable = previous_train
        if report["state"] == "T13_REAL_BATCH_COMPONENTS_PASS_PENDING_INDEPENDENT_RELOAD":
            binding = json.loads((output / "reload_binding.json").read_text())
            result = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                "--config", str(args.config), "--set", "inference.fallback_to_heuristic=false",
                "--action", "verify-checkpoint", "--checkpoint", str(output / "diagnostic_checkpoint.pt"),
                "--expected-state-sha256", binding["state_sha256"]], check=True, capture_output=True,
                text=True, timeout=max(1, int(1800 - (time.monotonic() - started))))
            report["independent_process_reload_state"] = json.loads(result.stdout.splitlines()[-1])["state"]
            report["state"] = "T13_REAL_BATCH_PERFORMANCE_PASS"
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "performance.json", report)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["state"] == "T13_REAL_BATCH_PERFORMANCE_PASS" else 2
    except Exception as exc:
        atomic_json(output / "performance.json", dict(state="T13_REAL_BATCH_INCOMPLETE_OR_FAILED",
            error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic() - started,
            synthetic=False, formal_start=False, active_handover_ready=False,
            current_phase=samples[-1]["phase"] if samples else "BEFORE_FIRST_BATCH"))
        raise
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=2)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
