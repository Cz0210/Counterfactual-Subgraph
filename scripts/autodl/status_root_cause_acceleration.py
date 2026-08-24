#!/usr/bin/env python3
"""Print one root-cause acceleration monitor snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.root.expanduser().resolve(strict=True) / "state.json").read_text())
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
