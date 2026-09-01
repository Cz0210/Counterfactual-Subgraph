#!/usr/bin/env python3
"""Report the post-main LLM-ablation launch gate without starting science."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.status_cli import status_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(status_main("llm", __doc__ or "LLM ablation status"))
