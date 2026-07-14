#!/usr/bin/env python3
"""Audit saved GlobalGCE Frequency-Top20 Node-FGW evaluation artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.globalgce_frequency_fgw_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
