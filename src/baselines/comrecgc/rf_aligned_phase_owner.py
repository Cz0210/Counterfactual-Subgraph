"""Finite AIDS existing-pairs owner: wait → exact cluster → summary → release.

This extends the same recourse writer namespace and existing release CLI. It
does not have authority to start pair generation, RF search or another owner.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .rf_aligned_pool import atomic_json, file_sha
from .rf_aligned_cluster_phase import sealed_pairs, phase_memory_plan, resource_sample


def child_env(worktree, scratch=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(worktree),
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    if scratch is not None:
        scratch = Path(scratch); scratch.mkdir(parents=True,exist_ok=True)
        env.update(TMPDIR=str(scratch), XDG_CACHE_HOME=str(scratch/"cache"), MPLCONFIGDIR=str(scratch/"matplotlib"))
    return env


def phase_argv(*, python, worktree, manifest, recourse_root, pool_root, evidence, phase, fd, cli=None):
    if phase not in ("cluster-existing", "summary-existing"):
        raise ValueError("Existing-pair owner cannot dispatch pair generation")
    return ["nice", "-n", "10", python, "-B", str(cli or Path(worktree)/"scripts/continue_aids_rf_pairs.py"),
            "--config", str(Path(worktree)/"configs/hpc.yaml"), "--run-manifest", str(manifest),
            "--recourse-root", str(recourse_root), "--pool-root", str(pool_root),
            "--output-root", str(evidence), "--action", phase, "--writer-fd", str(fd)]


def validate_writer_fd(fd, root):
    expected = (Path(root)/"writer.lock").stat(); actual = os.fstat(fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise RuntimeError("Child inherited FD is not the existing recourse writer lock")
    # A separate open file description must lose the competition.
    with (Path(root)/"writer.lock").open("a+") as contender:
        try: fcntl.flock(contender, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return
        fcntl.flock(contender, fcntl.LOCK_UN)
        raise RuntimeError("No exclusive original recourse lease was inherited")


def wait_admission(config, plan, root, phase):
    while True:
        result = resource_sample(config, plan, running=False)
        atomic_json(root/"resource_admission.json", result)
        atomic_json(root/"heartbeat.json", {"state": "READY_TO_DISPATCH" if result["state"]=="PASS" else "WAITING_RESOURCE",
            "phase": phase, "owner_pid": os.getpid(), "science_pid": None, "sampled_at_unix": time.time(),
            "resource_shortfall_bytes": result["shortfall_bytes"], "gpu_requested": False})
        if result["state"] == "PASS": return result
        time.sleep(60)


def run_child(argv, *, worktree, root, phase, env, pass_fds=()):
    terminal = root/(phase+"_terminal.json")
    submission = root/(phase+"_submission.json")
    if terminal.exists():
        value = json.loads(terminal.read_text())
        if value["returncode"] != 0: raise RuntimeError("Failed exact stage requires diagnosis; no automatic retry")
        return value
    if submission.exists(): raise RuntimeError("Unreconciled prior submission; never duplicate science")
    atomic_json(root/(phase+"_intent.json"), {"argv": argv, "created_at_unix": time.time(), "phase": phase})
    with (root/(phase+".log")).open("ab") as log:
        child = subprocess.Popen(argv, cwd=worktree, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 close_fds=True, pass_fds=pass_fds)
        ticks = Path(f"/proc/{child.pid}/stat").read_text().split(") ",1)[1].split()[19]
        atomic_json(submission, {"argv": argv, "pid": child.pid, "start_ticks": ticks, "worktree": str(worktree),
                                "GPU_requested": False, "phase": phase})
        while child.poll() is None:
            atomic_json(root/"heartbeat.json", {"state": "SCIENCE_RUNNING", "phase": phase,
                "owner_pid": os.getpid(), "science_pid": child.pid, "science_start_ticks": ticks,
                "sampled_at_unix": time.time(), "gpu_requested": False})
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: pass
    value = {"returncode": child.returncode, "pid": child.pid, "start_ticks": ticks,
             "phase": phase, "finished_at_unix": time.time()}
    atomic_json(terminal, value)
    if child.returncode: raise RuntimeError("Exact stage failed; preserve checkpoint, no automatic retry: "+phase)
    return value


def run_owner(config, *, manifest, recourse_root, pool_root, output_root):
    root, recourse, pool = Path(output_root), Path(recourse_root), Path(pool_root)
    root.mkdir(parents=True, exist_ok=True)
    worktree = Path(config["phase_execution_worktree"])
    actual_commit = subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != config["phase_execution_commit"]:
        raise ValueError("Phase driver execution commit differs")
    with (root/"owner.lock").open("a+") as owner, (recourse/"writer.lock").open("a+") as writer:
        fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(writer, fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding = sealed_pairs(config, recourse)
        atomic_json(root/"source_pair_adoption.json", binding)
        for action, phase in (("cluster-existing", "CERTIFIED_EXACT_DBSCAN"), ("summary-existing", "NATIVE_SUMMARY")):
            evidence = root/action; evidence.mkdir(exist_ok=True)
            if (root/(action+"_terminal.json")).exists():
                if json.loads((root/(action+"_terminal.json")).read_text())["returncode"] != 0:
                    raise ValueError("Previous failed phase cannot be silently retried")
                continue
            plan = phase_memory_plan(binding, phase=phase)
            atomic_json(evidence/"phase_memory_plan.json", plan)
            wait_admission(config, plan, root, phase)
            command = phase_argv(python=sys.executable, worktree=worktree, manifest=manifest,
                recourse_root=recourse, pool_root=pool, evidence=evidence, phase=action, fd=writer.fileno(), cli=config.get("phase_cli_path"))
            run_child(command, worktree=worktree, root=root, phase=action, env=child_env(worktree,root/"scratch"), pass_fds=(writer.fileno(),))
        # Exact scientific predecessor must release the original lock before
        # the existing release-after-recourse entrypoint can adopt it.
        fcntl.flock(writer, fcntl.LOCK_UN)
        writer.close()  # an unlocked writable FD would still fail terminal writer audit
        release_source = Path(config["phase_release_manifest"])
        if file_sha(release_source) != config["phase_release_manifest_sha256"]:
            raise ValueError("Existing release successor input changed")
        release = json.loads(release_source.read_text())
        release.update(execution_commit=actual_commit, predecessor_owner_terminal=str(root/"summary-existing_terminal.json"))
        release["matrix_publication"].update(worktree=str(worktree), execution_commit=actual_commit)
        release_manifest = root/"release_run_manifest.json"; atomic_json(release_manifest, release)
        wait_admission(config, phase_memory_plan(binding, phase="RF_WNODE_RELEASE"), root, "RF_WNODE_RELEASE")
        command = ["nice", "-n", "10", sys.executable, "-B", str(worktree/"scripts/repair_aids_rf_aligned.py"),
                   "--config", str(worktree/"configs/hpc.yaml"), "--run-manifest", str(release_manifest),
                   "--pool-root", str(recourse), "--output-root", str(root/"shared_evaluation"), "--action", "release-after-recourse"]
        run_child(command, worktree=worktree, root=root, phase="release", env=child_env(worktree,root/"scratch"))
        atomic_json(root/"terminal.json", {"state": "EXISTING_PAIRS_TO_ORIGINAL_RELEASE_COMPLETE", "pair_rows": binding["rows"],
            "pairs_recomputed": False, "new_search_started": False, "old_failure_preserved": True})
