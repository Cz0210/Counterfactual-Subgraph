#!/usr/bin/env python3
"""Run one fresh 500/1000-step BACE COMRECGC legacy/optimized A/B gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.bace_preprocessing import PREPROCESS_ENGINE  # noqa: E402
from src.baselines.comrecgc.contracts import require_empty_output, write_json  # noqa: E402
from src.baselines.comrecgc.equivalence import audit_generation_equivalence  # noqa: E402


def _command(
    args: argparse.Namespace,
    *,
    role: str,
    run_root: Path,
) -> list[str]:
    aux = run_root / "_native_aux"
    command = [
        str(Path(args.python).expanduser().resolve()),
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/run_generation.py"),
        "--config",
        args.config,
        "--set",
        "inference.fallback_to_heuristic=false",
        "--route",
        "project",
        "--dataset",
        "bace",
        "--mode",
        "full",
        "--diagnostic-equivalence-steps",
        str(args.steps),
        "--equivalence-gate-role",
        role,
        "--project-root",
        str(PROJECT_ROOT),
        "--upstream-root",
        str(Path(args.upstream_root).expanduser().resolve()),
        "--dataset-dir",
        str(Path(args.dataset_dir).expanduser().resolve()),
        "--gnn-checkpoint",
        str(Path(args.gnn_checkpoint).expanduser().resolve()),
        "--distance-checkpoint",
        str(Path(args.distance_checkpoint).expanduser().resolve()),
        "--output-dir",
        str(run_root),
        "--parent-limit",
        str(args.parent_limit),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--trace-output-dir",
        str(aux / "trace"),
        "--graph-state-dir",
        str(aux / "graph_state"),
        "--checkpoint-root",
        str(aux / "checkpoints"),
        "--checkpoint-mirror-root",
        str(aux / "checkpoint_mirror"),
        "--checkpoint-interval-steps",
        "500",
        "--checkpoint-keep-last",
        "2",
        "--progress-interval-steps",
        "25",
    ]
    if role == "legacy":
        command.extend(
            [
                "--bace-preprocess-engine",
                "legacy_sequential_rdkit_v1",
                "--bace-preprocess-workers",
                "0",
                "--bace-source-cache-capacity",
                "0",
                "--bace-candidate-cache-capacity",
                "0",
            ]
        )
    else:
        command.extend(
            [
                "--bace-preprocess-engine",
                PREPROCESS_ENGINE,
                "--bace-preprocess-workers",
                str(args.workers),
                "--bace-preprocess-max-inflight",
                str(args.max_inflight),
                "--bace-source-cache-capacity",
                str(args.source_cache_capacity),
                "--bace-candidate-cache-capacity",
                str(args.candidate_cache_capacity),
            ]
        )
    return command


def _run_one(
    args: argparse.Namespace,
    *,
    role: str,
    root: Path,
    gate_root: Path,
) -> None:
    complete = root / "_RUN_COMPLETE.json"
    if complete.is_file():
        return
    command = _command(args, role=role, run_root=root)
    if root.exists() and any(root.iterdir()):
        latest = root / "_native_aux/checkpoints/LATEST"
        if not args.continue_existing or not latest.is_file():
            raise FileExistsError(
                f"Incomplete {role} root requires a verified checkpoint and "
                f"--continue-existing: {root}"
            )
        command.append("--resume")
    log_path = gate_root / f"{role}.log"
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": str(args.omp_threads),
                "MKL_NUM_THREADS": str(args.omp_threads),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        failure = {
            "status": "FAIL",
            "role": role,
            "exit_code": int(result.returncode),
            "output_root": str(root),
            "log": str(log_path),
        }
        write_json(gate_root / f"{role}_FAIL.json", failure)
        raise RuntimeError(f"{role} equivalence prefix failed with {result.returncode}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--distance-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, choices=(500, 1000), required=True)
    parser.add_argument("--parent-limit", type=int, default=360)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--source-cache-capacity", type=int, default=1024)
    parser.add_argument("--candidate-cache-capacity", type=int, default=8192)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--continue-existing", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    gate_root = Path(args.output_dir).expanduser().resolve()
    if args.audit_only or args.continue_existing:
        if not gate_root.is_dir():
            raise FileNotFoundError(gate_root)
    else:
        require_empty_output(gate_root)
    legacy = gate_root / "legacy"
    optimized = gate_root / "optimized"
    if not args.audit_only:
        _run_one(args, role="legacy", root=legacy, gate_root=gate_root)
        _run_one(args, role="optimized", root=optimized, gate_root=gate_root)
    audit_root = gate_root / "audit"
    if audit_root.exists():
        raise FileExistsError(
            f"Equivalence audit root already exists; use a fresh gate root: {audit_root}"
        )
    result = audit_generation_equivalence(
        legacy_root=legacy,
        optimized_root=optimized,
        output_dir=audit_root,
        expected_steps=args.steps,
    )
    (gate_root / "PASS").write_text(
        f"BACE COMRECGC {args.steps}-step A/B equivalence passed.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
