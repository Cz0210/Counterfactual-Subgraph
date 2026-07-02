#!/usr/bin/env python3
"""Check the expected GlobalGCE official repository layout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_OUT = "outputs/hpc/globalgce/layout_check.json"


REQUIRED_RELATIVE_PATHS = [
    "src/main.py",
    "src/utils.py",
    "src/datasets/AIDS",
    "src/datasets/AIDS/AIDS/raw",
    "src/datasets/AIDS/AIDS/raw/AIDS_A.txt",
    "src/datasets/AIDS/AIDS/raw/AIDS_graph_labels.txt",
    "src/datasets/AIDS/AIDS/raw/AIDS_node_labels.txt",
    "src/datasets/AIDS/AIDS/raw/AIDS_edge_labels.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the GlobalGCE official baseline layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--globalgce-root", default="baselines/globalgce_official")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--config", default=None, help="Ignored compatibility hook for HPC wrappers.")
    parser.add_argument("--set", action="append", default=[], help="Ignored compatibility hook for HPC wrappers.")
    return parser.parse_args()


def git_commit(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return f"unavailable:{exc}"
    if completed.returncode == 0:
        return completed.stdout.strip()
    return f"unavailable:{completed.stderr.strip() or completed.stdout.strip()}"


def main() -> int:
    args = parse_args()
    root = Path(args.globalgce_root).expanduser().resolve()
    out_path = Path(args.out).expanduser()

    detected: dict[str, bool] = {}
    missing: list[str] = []
    for relative in REQUIRED_RELATIVE_PATHS:
        exists = (root / relative).exists()
        detected[relative] = bool(exists)
        if not exists:
            missing.append(relative)

    result: dict[str, Any] = {
        "ok": not missing,
        "missing_files": missing,
        "globalgce_root": str(root),
        "globalgce_commit": git_commit(root),
        "detected_files": detected,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
