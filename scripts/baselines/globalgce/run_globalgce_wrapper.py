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
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Deprecated compatibility flag; official saved_results is never copied.",
    )
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


GLOBALGCE_RUNTIME_DIRS = [
    "saved_results",
    "saved_results/saved_models",
    "saved_results/saved_models/gnn_model",
    "saved_results/saved_exp_res",
    "saved_results/saved_exp_res/GlobalGCE",
    "saved_results/saved_rules",
    "saved_results/saved_rules/GlobalGCE",
    "saved_results/saved_rules/plots",
    "saved_results/saved_cfs",
    "saved_results/saved_cfs/GlobalGCE",
]

GLOBALGCE_TRAIN_GNN_PATCH_MARKER = (
    "# patched by counterfactual-subgraph wrapper: avoid None best_model_dict"
)
GLOBALGCE_TRAIN_GNN_PATCH_REASON = (
    "avoid torch.save(None, gnn_model_path) in GlobalGCE smoke/offical run"
)


def copy_official_src(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run source already exists: {destination}. Use --overwrite-run-src to replace it."
            )
        shutil.rmtree(destination)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache", "saved_results"}
        return ignored.intersection(names)

    shutil.copytree(source, destination, ignore=ignore)


def ensure_globalgce_runtime_dirs(work_src: Path) -> list[str]:
    """Create the clean saved_results directory skeleton required by GlobalGCE."""

    created_dirs: list[str] = []
    for relative in GLOBALGCE_RUNTIME_DIRS:
        path = work_src / relative
        path.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(path))
    return created_dirs


def patch_globalgce_train_gnn_best_model(work_src: Path) -> dict[str, Any]:
    """Patch the copied GlobalGCE runtime tree so train_gnn never saves None."""

    target_file = work_src / "models" / "models_utils.py"
    patch_info: dict[str, Any] = {
        "applied": False,
        "target_file": str(target_file),
        "reason": GLOBALGCE_TRAIN_GNN_PATCH_REASON,
        "changed": False,
        "import_copy_added": False,
        "initialization_patched": False,
        "fallback_patched": False,
        "error": None,
    }
    if not target_file.exists():
        patch_info["error"] = f"target file does not exist: {target_file}"
        return patch_info

    original_text = target_file.read_text(encoding="utf-8")
    text = original_text

    if "\nimport copy\n" not in f"\n{text}\n":
        text = "import copy\n" + text
        patch_info["import_copy_added"] = True

    init_old = "    best_loss, best_model_dict = 10e9, None"
    init_new = (
        f"    {GLOBALGCE_TRAIN_GNN_PATCH_MARKER}\n"
        "    best_loss, best_model_dict = 10e9, copy.deepcopy(gnn_model.state_dict())"
    )
    if init_new not in text and init_old in text:
        text = text.replace(init_old, init_new, 1)

    save_line = "    torch.save(best_model_dict, gnn_model.save_model_path)"
    fallback_block = (
        f"    {GLOBALGCE_TRAIN_GNN_PATCH_MARKER}\n"
        "    if best_model_dict is None:\n"
        "        best_model_dict = copy.deepcopy(gnn_model.state_dict())\n"
    )
    if fallback_block not in text and save_line in text:
        text = text.replace(save_line, fallback_block + save_line, 1)

    if text != original_text:
        target_file.write_text(text, encoding="utf-8")
        patch_info["changed"] = True

    patch_info["initialization_patched"] = (
        "best_loss, best_model_dict = 10e9, copy.deepcopy(gnn_model.state_dict())"
        in text
    )
    patch_info["fallback_patched"] = (
        "if best_model_dict is None:\n"
        "        best_model_dict = copy.deepcopy(gnn_model.state_dict())\n"
        "    torch.save(best_model_dict, gnn_model.save_model_path)"
        in text
    )
    patch_info["applied"] = bool(
        "\nimport copy\n" in f"\n{text}\n"
        and patch_info["initialization_patched"]
        and patch_info["fallback_patched"]
    )
    if not patch_info["applied"] and patch_info["error"] is None:
        patch_info["error"] = "expected train_gnn patch anchors were not found"
    return patch_info


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
        overwrite=bool(args.overwrite_run_src),
    )
    created_runtime_dirs = ensure_globalgce_runtime_dirs(run_src)
    train_gnn_patch = patch_globalgce_train_gnn_best_model(run_src)

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
        "reuse_cache_note": "official saved_results is intentionally not copied; clean runtime dirs are created",
        "created_runtime_dirs": created_runtime_dirs,
        "runtime_patches": {
            "train_gnn_best_model_dict": train_gnn_patch,
        },
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[GLOBALGCE_RUN] run_root={run_root}")
    print(f"[GLOBALGCE_RUN] created_runtime_dirs={len(created_runtime_dirs)}")
    print(f"[GLOBALGCE_RUN] train_gnn_best_model_patch_applied={train_gnn_patch['applied']}")
    print(f"[GLOBALGCE_RUN] command={command_text}")
    print(f"[GLOBALGCE_RUN] returncode={completed.returncode}")
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
