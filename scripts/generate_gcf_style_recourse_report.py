#!/usr/bin/env python3
"""Generate GCFExplainer-style tables and curves from completed recourse runs."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.gcf_style_recourse_report import build_parser, generate_report, main  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
