#!/usr/bin/env python3
"""Run GlobalGCE official code from a copied source tree under outputs/."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy GlobalGCE official src into a run directory and execute main.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--globalgce-root", default="baselines/globalgce_official")
    parser.add_argument("--run-root", default="outputs/hpc/globalgce/aids_official_top30")
    parser.add_argument("--dataset", default="AIDS")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gnn-lr", type=float, default=0.01)
    parser.add_argument("--gnn-wd", type=float, default=0.0001)
    parser.add_argument("--train-gnn-epochs", type=int, default=300)
    parser.add_argument("--exp-num", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="0", help="GPU id or cpu.")
    parser.add_argument("--reuse-cache", action="store_true", help="Copy official saved_results cache too.")
    parser.add_argument(
        "--overwrite-run-src",
        action="store_true",
        help="Remove an existing run-root/GlobalGCE_src before copying.",
    )
    parser.add_argument("--config", default=None, help="Ignored compatibility hook for HPC wrappers.")
    parser.add_argument("--set", action="append", default=[], help="Ignored compatibility hook for HPC wrappers.")
    return parser.parse_args()


def run_git(path: Path, args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return f"unavailable:{exc}"
    if completed.returncode == 0:
        return completed.stdout.strip()
    return f"unavailable:{completed.stderr.strip() or completed.stdout.strip()}"


def copy_official_src(source: Path, destination: Path, *, reuse_cache: bool, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run source already exists: {destination}. Use --overwrite-run-src to replace it."
            )
        shutil.rmtree(destination)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache"}
        if not reuse_cache:
            ignored.add("saved_results")
        return ignored.intersection(names)

    shutil.copytree(source, destination, ignore=ignore)


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--dataset",
        str(args.dataset),
        "--epochs",
        str(args.epochs),
        "--topk",
        str(args.topk),
        "--lr",
        str(args.lr),
        "--gnn_lr",
        str(args.gnn_lr),
        "--gnn_wd",
        str(args.gnn_wd),
        "--train_gnn_epochs",
        str(args.train_gnn_epochs),
        "--exp_num",
        str(args.exp_num),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        str(args.device),
    ]


def main() -> int:
    args = parse_args()
    official_root = Path(args.globalgce_root).expanduser().resolve()
    official_src = official_root / "src"
    run_root = Path(args.run_root).expanduser().resolve()
    run_src = run_root / "GlobalGCE_src"
    run_root.mkdir(parents=True, exist_ok=True)

    if not (official_src / "main.py").exists():
        raise FileNotFoundError(f"GlobalGCE main.py not found: {official_src / 'main.py'}")

    source_commit = run_git(official_root, ["rev-parse", "HEAD"])
    (run_root / "globalgce_source_commit.txt").write_text(source_commit + "\n", encoding="utf-8")
    copy_official_src(
        official_src,
        run_src,
        reuse_cache=bool(args.reuse_cache),
        overwrite=bool(args.overwrite_run_src),
    )

    command = build_command(args)
    command_text = " ".join(command)
    (run_root / "run_command.sh").write_text(command_text + "\n", encoding="utf-8")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    completed = subprocess.run(
        command,
        cwd=str(run_src),
        check=False,
        capture_output=True,
        text=True,
    )
    ended = datetime.now(timezone.utc).isoformat(timespec="seconds")

    (run_root / "run_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_root / "run_stderr.log").write_text(completed.stderr, encoding="utf-8")
    metadata: dict[str, Any] = {
        "started_utc": started,
        "ended_utc": ended,
        "returncode": completed.returncode,
        "globalgce_root": str(official_root),
        "globalgce_source_commit": source_commit,
        "run_root": str(run_root),
        "run_src": str(run_src),
        "command": command,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "topk": args.topk,
        "lr": args.lr,
        "gnn_lr": args.gnn_lr,
        "gnn_wd": args.gnn_wd,
        "train_gnn_epochs": args.train_gnn_epochs,
        "exp_num": args.exp_num,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "reuse_cache": bool(args.reuse_cache),
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[GLOBALGCE_RUN] run_root={run_root}")
    print(f"[GLOBALGCE_RUN] command={command_text}")
    print(f"[GLOBALGCE_RUN] returncode={completed.returncode}")
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
