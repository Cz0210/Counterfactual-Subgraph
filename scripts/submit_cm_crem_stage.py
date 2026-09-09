#!/usr/bin/env python3
"""Submit one fixed CM stage using exp_sbatch; preserve uncertain intent, never duplicate."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cm_crem_experiment import Experiment, HPC_SCOPE
from src.baselines.cm_crem_runtime import atomic_json, checked_root, digest, read_json, utc_now

STAGES = ("pilot-oracle", "pilot-generate", "pilot-filter", "pilot-closeout", "attribution",
          "generate", "filter", "encode", "calibrate", "select", "test", "audit", "export")


def submit(spec_path: Path, root: Path, stage: str, dependency: str | None = None) -> dict:
    root = checked_root(root, HPC_SCOPE)
    experiment = Experiment(spec_path, root)
    spec = experiment.spec
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if code_commit != spec["execution"]["execution_commit"]:
        raise ValueError("Submission worktree is not the immutable configured execution commit")
    action = stage.replace("pilot-generate", "generate").replace("pilot-filter", "filter")
    if stage in {"generate", "attribution"}:
        experiment.require_pilot()
    if "generate" in stage and experiment.stage_preflight()["missing_assets"]:
        raise FileNotFoundError("ASSET_BLOCKED: no generation submission without official DB and isolated environment")
    root.joinpath("submissions").mkdir(exist_ok=True)
    root.joinpath("logs").mkdir(exist_ok=True)
    receipt_path = root / "submissions" / (stage + ".json")
    intent_path = root / "submissions" / (stage + ".intent.json")
    # This is a short submission critical section, not a second GPU lease.
    with (root / "submissions" / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            return read_json(receipt_path)
        if intent_path.exists():
            intent = read_json(intent_path)
            # Do not infer failure from an empty queue or redo uncertain sbatch.
            query = subprocess.run(["squeue", "--noheader", "--name", intent["job_name"], "-o", "%A|%j|%T"],
                                   text=True, capture_output=True, timeout=30)
            return {"status": "SUBMISSION_OUTCOME_UNCERTAIN", "intent": str(intent_path),
                    "job_name": intent["job_name"], "squeue": query.stdout,
                    "next_action": "RECONCILE_EXACT_INTENT_WITH_SACCT_AND_EXPERIMENT_REGISTRY_NO_RESUBMIT"}
        job_name = "cm-" + stage + "-" + digest([str(root), experiment.sha])[:8]
        env = os.environ.copy()
        env.update(CM_EXECUTION_ROOT=str(ROOT), CM_SPEC=str(spec_path.resolve()), CM_RUN_ROOT=str(root),
                   CM_ACTION=action, CM_PILOT_ONLY="1" if stage.startswith("pilot-") and stage in {"pilot-generate", "pilot-filter"} else "0",
                   CM_GENERATOR_PYTHON=spec["execution"].get("generator_python", ""), PYTHONDONTWRITEBYTECODE="1")
        argv = [sys.executable, "-I", "-B", str(ROOT/"scripts/exp_sbatch.py"), "--name", job_name,
                "--dataset", "bace", "--method", "CM-CReM-Global-Budgeted-v1", "--metric", "frozen-GINE-MolCLR-WNode",
                "--expected-output-root", str(root), "--registry-jsonl", str(root/"submissions/jobs.jsonl"),
                "--markdown-log", str(root/"submissions/EXPERIMENT_LOG.md"), "--", "--job-name", job_name,
                "--chdir", str(ROOT), "--output", str(root/"logs/%x-%j.out"), "--error", str(root/"logs/%x-%j.err")]
        if stage in {"generate", "calibrate", "test"}:
            argv += ["--array", "0-1%2"]
        if dependency:
            if not dependency.replace(":", "").isdigit():
                raise ValueError("Only resolved numeric afterok dependencies are allowed")
            argv += ["--dependency", "afterok:" + dependency]
        argv.append(str(ROOT/"scripts/slurm/run_cm_crem.sh"))
        intent = {"schema": "cm_crem_submission_intent_v1", "created_at": utc_now(), "stage": stage,
                  "job_name": job_name, "argv": argv, "science_hash": experiment.sha, "execution_commit": code_commit,
                  "submission_environment": {k: v for k, v in env.items() if k.startswith("CM_")}, "dependency": dependency}
        atomic_json(intent_path, intent, immutable=True)
        result = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
        registry = root / "submissions/jobs.jsonl"
        entries = [json.loads(line) for line in registry.read_text().splitlines()] if registry.exists() else []
        matching = [row for row in entries if row.get("name") == job_name or row.get("experiment_name") == job_name]
        # The reused exp_sbatch emits the resolved job ID in a stable line.
        ids = [line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("job_id=")]
        if result.returncode or len(ids) != 1 or not ids[0].isdigit():
            record = {"status": "SUBMISSION_OUTCOME_UNCERTAIN", "intent": str(intent_path),
                      "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        else:
            record = {"status": "SUBMITTED_NOT_SCIENCE_PASS", "job_id": ids[0], "stage": stage,
                      "intent": str(intent_path), "submitted_at": utc_now(), "execution_commit": code_commit,
                      "science_hash": experiment.sha, "dependency": dependency, "job_name": job_name,
                      "stdout": result.stdout, "stderr": result.stderr, "registry_matching_rows": len(matching)}
        atomic_json(receipt_path, record, immutable=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dependency")
    args = parser.parse_args()
    result = submit(args.spec, args.run_root, args.stage, args.dependency)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "SUBMITTED_NOT_SCIENCE_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
