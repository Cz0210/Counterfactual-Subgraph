"""Logging helpers for local and HPC runs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class RunContext:
    """Minimal run metadata that can be serialized into logs or reports."""

    run_name: str
    output_dir: Path
    stage: str
    seed: int | None = None
    notes: tuple[str, ...] = ()


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def get_logger(name: str, *, level: str | int = logging.INFO) -> logging.Logger:
    """Create or reuse a process-local logger with a stable formatter."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False
    return logger


def configure_run_logger(
    name: str,
    *,
    context: RunContext,
    log_dir: Path,
    level: str | int = logging.INFO,
) -> logging.Logger:
    """Configure one logger with both stdout and file handlers."""

    ensure_directory(log_dir)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / f"{context.stage}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def write_runtime_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON manifest for one resolved run configuration."""

    manifest_path = Path(path)
    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
