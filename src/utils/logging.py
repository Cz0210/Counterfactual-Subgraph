"""Logging helpers for long-running local or HPC jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunContext:
    """Minimal run metadata that can be serialized into logs or reports."""

    run_name: str
    output_dir: Path
    stage: str
    seed: int | None = None
    notes: tuple[str, ...] = ()


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Create or reuse a process-local logger with a stable formatter."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
