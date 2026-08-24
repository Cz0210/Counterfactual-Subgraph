#!/usr/bin/env python3
"""Thin AutoDL entrypoint for the frozen-GREED AIDS pair-semantics audit."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.aids_pair_semantics import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
