"""Path resolution helpers for local and HPC execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """Resolved path bundle for one local or HPC run."""

    repo_root: Path
    data_root: Path
    output_root: Path
    cache_root: Path
    run_dir: Path
    log_dir: Path
    model_path: Path | None = None
    tokenizer_path: Path | None = None
    train_file: Path | None = None
    valid_file: Path | None = None
    test_file: Path | None = None
    checkpoint_path: Path | None = None


def get_repo_root() -> Path:
    """Return the repository root inferred from this file location."""

    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    """Resolve an optional path relative to a chosen base directory."""

    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def default_run_name(config: Mapping[str, Any], *, stage_name: str) -> str:
    """Create a stable default run name when one is not configured."""

    configured_name = str(config.get("run", {}).get("name") or "").strip()
    if configured_name:
        return configured_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stage_name}_{timestamp}"


def build_runtime_paths(
    config: Mapping[str, Any],
    *,
    stage_name: str,
    create_dirs: bool = True,
) -> RuntimePaths:
    """Resolve the main directory layout for one run."""

    repo_root = get_repo_root()
    paths_cfg = config.get("paths", {})
    data_root = resolve_path(paths_cfg.get("data_root", "data"), base_dir=repo_root) or repo_root
    output_root = (
        resolve_path(paths_cfg.get("output_root", "outputs"), base_dir=repo_root) or repo_root
    )
    cache_root = (
        resolve_path(paths_cfg.get("cache_root", ".cache"), base_dir=repo_root) or repo_root
    )
    run_dir = output_root / stage_name / default_run_name(config, stage_name=stage_name)
    log_dir = run_dir / "logs"

    if create_dirs:
        ensure_directory(output_root)
        ensure_directory(cache_root)
        ensure_directory(run_dir)
        ensure_directory(log_dir)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})

    return RuntimePaths(
        repo_root=repo_root,
        data_root=data_root,
        output_root=output_root,
        cache_root=cache_root,
        run_dir=run_dir,
        log_dir=log_dir,
        model_path=resolve_path(
            model_cfg.get("model_path") or model_cfg.get("model_name_or_path"),
            base_dir=repo_root,
        ),
        tokenizer_path=resolve_path(model_cfg.get("tokenizer_path"), base_dir=repo_root),
        train_file=resolve_path(data_cfg.get("train_file"), base_dir=data_root),
        valid_file=resolve_path(data_cfg.get("valid_file"), base_dir=data_root),
        test_file=resolve_path(data_cfg.get("test_file"), base_dir=data_root),
        checkpoint_path=resolve_path(eval_cfg.get("checkpoint_path"), base_dir=repo_root),
    )


def inject_runtime_paths(
    config: Mapping[str, Any],
    runtime_paths: RuntimePaths,
) -> dict[str, Any]:
    """Attach resolved absolute paths back into a JSON-serializable config."""

    resolved = dict(config)
    resolved_paths = {
        "repo_root": str(runtime_paths.repo_root),
        "data_root": str(runtime_paths.data_root),
        "output_root": str(runtime_paths.output_root),
        "cache_root": str(runtime_paths.cache_root),
        "run_dir": str(runtime_paths.run_dir),
        "log_dir": str(runtime_paths.log_dir),
        "model_path": str(runtime_paths.model_path) if runtime_paths.model_path else None,
        "tokenizer_path": (
            str(runtime_paths.tokenizer_path) if runtime_paths.tokenizer_path else None
        ),
        "train_file": str(runtime_paths.train_file) if runtime_paths.train_file else None,
        "valid_file": str(runtime_paths.valid_file) if runtime_paths.valid_file else None,
        "test_file": str(runtime_paths.test_file) if runtime_paths.test_file else None,
        "checkpoint_path": (
            str(runtime_paths.checkpoint_path) if runtime_paths.checkpoint_path else None
        ),
    }
    resolved["resolved_paths"] = resolved_paths
    return resolved
