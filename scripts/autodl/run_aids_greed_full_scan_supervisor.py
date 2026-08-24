#!/usr/bin/env python3
"""Thin CLI for the restart-safe AIDS GREED full-scan supervisor."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_aids_greed_full_scan_supervisor import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

