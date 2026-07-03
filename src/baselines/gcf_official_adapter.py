"""Project-owned adapter for the official GCFExplainer repository.

The upstream GCFExplainer code writes to fixed relative paths such as
``results/aids/runs/counterfactuals.pt``.  This adapter runs the official code
inside an isolated working directory so parallel alpha/theta jobs do not
overwrite each other.  The official repository itself is treated as a runtime
asset and is not modified.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ASSETS = [
    "data/aids/gnn/model_best.pth",
    "data/aids/gnn/preds.pt",
    "data/aids/gnn/logits.pt",
    "data/aids/gnn/graph_embeddings.pt",
    "data/aids/neurosed/best_model.pt",
    "vrrw.py",
    "summary.py",
    "data.py",
    "gnn.py",
    "importance.py",
    "distance.py",
]


@dataclass(frozen=True)
class OfficialRunConfig:
    official_repo: Path
    dataset: str
    alpha: float
    train_theta: float
    max_steps: int
    teleport: float
    sample: bool
    sample_size: int
    device1: str
    device2: str
    run_dir: Path


def _run_quiet(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def git_commit(path: Path) -> str:
    return _run_quiet(["git", "-C", str(path), "rev-parse", "HEAD"]) or "unknown"


def resolve_official_repo(value: str | Path | None = None) -> Path:
    """Resolve official GCFExplainer checkout.

    Preferred order:
    explicit CLI value -> ``GCF_OFFICIAL_REPO`` -> ``third_party/GCFExplainer`` ->
    legacy ``baselines/gcfexplainer_official``.
    """

    candidates: list[Path] = []
    if value not in (None, ""):
        candidates.append(Path(value))
    env_value = os.environ.get("GCF_OFFICIAL_REPO")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            REPO_ROOT / "third_party" / "GCFExplainer",
            REPO_ROOT / "baselines" / "gcfexplainer_official",
        ]
    )
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.exists():
            return path
    return candidates[0].expanduser().resolve() if candidates else (REPO_ROOT / "third_party" / "GCFExplainer")


def check_official_assets(official_repo: str | Path | None = None) -> dict[str, Any]:
    repo = resolve_official_repo(official_repo)
    detected: list[str] = []
    missing: list[str] = []
    for relative in REQUIRED_ASSETS:
        path = repo / relative
        if path.exists():
            detected.append(relative)
        else:
            missing.append(relative)
    return {
        "ok": repo.exists() and not missing,
        "official_repo": str(repo),
        "official_commit": git_commit(repo) if repo.exists() else "missing_repo",
        "required_files": REQUIRED_ASSETS,
        "detected_files": detected,
        "missing_files": missing,
    }


def write_asset_check(payload: dict[str, Any], out_dir: str | Path) -> Path:
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    out_json = destination / "gcf_official_asset_check.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_json


def _copy_or_link(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return "symlink"
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
            return "copytree"
        shutil.copy2(src, dst)
        return "copy2"


def prepare_isolated_workdir(official_repo: Path, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Create an isolated official-code workdir under ``run_dir``."""

    workdir = run_dir / "official_workdir"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for source in official_repo.glob("*.py"):
        shutil.copy2(source, workdir / source.name)
        copied_files.append(source.name)
    copied_dirs: dict[str, str] = {}
    for dirname in ("neurosed",):
        source = official_repo / dirname
        if source.exists():
            shutil.copytree(source, workdir / dirname)
            copied_dirs[dirname] = "copytree"
    data_source = official_repo / "data"
    data_mode = "missing"
    if data_source.exists():
        data_mode = _copy_or_link(data_source, workdir / "data")
    (workdir / "results").mkdir(parents=True, exist_ok=True)
    return workdir, {
        "workdir": str(workdir),
        "copied_files": copied_files,
        "copied_dirs": copied_dirs,
        "data_source": str(data_source),
        "data_link_mode": data_mode,
    }


def _metadata_path(run_dir: Path) -> Path:
    return run_dir / "gcf_official_run_config.json"


def run_official_vrrw(config: OfficialRunConfig) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    asset_payload = check_official_assets(config.official_repo)
    if not asset_payload["ok"]:
        write_asset_check(asset_payload, config.run_dir)
        raise FileNotFoundError(
            "Official GCFExplainer assets are incomplete. Missing: "
            + ", ".join(asset_payload.get("missing_files", []))
        )

    print("[GCF_OFFICIAL_CONFIG]", flush=True)
    print("GCF_MODE=official_native", flush=True)
    print(f"DATASET={config.dataset}", flush=True)
    print(f"ALPHA={config.alpha}", flush=True)
    print(f"TRAIN_THETA={config.train_theta}", flush=True)
    print(f"MAX_STEPS={config.max_steps}", flush=True)
    print("CF_MODE=strict_flip", flush=True)

    workdir, workdir_meta = prepare_isolated_workdir(config.official_repo, config.run_dir)
    command = [
        sys.executable,
        "vrrw.py",
        "--dataset",
        config.dataset,
        "--alpha",
        str(config.alpha),
        "--theta",
        str(config.train_theta),
        "--max_steps",
        str(config.max_steps),
        "--teleport",
        str(config.teleport),
        "--sample_size",
        str(config.sample_size),
        "--device1",
        str(config.device1),
        "--device2",
        str(config.device2),
    ]
    if config.sample:
        command.append("--sample")

    env = os.environ.copy()
    py_paths = [str(workdir), str(config.official_repo)]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    started = time.time()
    result = subprocess.run(
        command,
        cwd=str(workdir),
        text=True,
        capture_output=True,
        shell=False,
        env=env,
        check=False,
    )
    elapsed = time.time() - started

    stdout_path = config.run_dir / "vrrw_stdout.log"
    stderr_path = config.run_dir / "vrrw_stderr.log"
    stdout_path.write_text(result.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(result.stderr or "", encoding="utf-8", errors="replace")
    (config.run_dir / "vrrw_command.json").write_text(
        json.dumps({"command": command, "cwd": str(workdir)}, indent=2) + "\n",
        encoding="utf-8",
    )

    expected = workdir / "results" / config.dataset / "runs" / "counterfactuals.pt"
    copied_path = config.run_dir / "counterfactuals.pt"
    counterfactuals_found = expected.exists()
    if counterfactuals_found:
        shutil.copy2(expected, copied_path)

    metadata = {
        "GCF_MODE": "official_native",
        "CF_MODE": "strict_flip",
        "official_repo": str(config.official_repo),
        "official_commit": git_commit(config.official_repo),
        "dataset": config.dataset,
        "alpha": config.alpha,
        "train_theta": config.train_theta,
        "max_steps": config.max_steps,
        "teleport": config.teleport,
        "sample": config.sample,
        "sample_size": config.sample_size,
        "device1": config.device1,
        "device2": config.device2,
        "run_dir": str(config.run_dir),
        "isolated_workdir": workdir_meta,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "expected_counterfactuals_path": str(expected),
        "counterfactuals_path": str(copied_path) if counterfactuals_found else "",
        "counterfactuals_found": counterfactuals_found,
    }
    _metadata_path(config.run_dir).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"official vrrw.py failed with returncode={result.returncode}; stderr={stderr_path}")
    if not counterfactuals_found:
        raise FileNotFoundError(f"official vrrw.py did not produce {expected}")
    return metadata


def build_run_config_from_args(args: Any) -> OfficialRunConfig:
    return OfficialRunConfig(
        official_repo=resolve_official_repo(args.official_repo),
        dataset=str(args.dataset),
        alpha=float(args.alpha),
        train_theta=float(args.train_theta),
        max_steps=int(args.max_steps),
        teleport=float(args.teleport),
        sample=bool(args.sample),
        sample_size=int(args.sample_size),
        device1=str(args.device1),
        device2=str(args.device2),
        run_dir=Path(args.run_dir).expanduser().resolve(),
    )

