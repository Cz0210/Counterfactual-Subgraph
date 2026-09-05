#!/usr/bin/env python3
"""Package sealed BACE GNN seed-7 artifacts without any main-matrix writer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.cpu_evaluation import package_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--execution-commit", required=True)
    args = vars(parser.parse_args(argv))
    args.pop("config")
    print(json.dumps(package_evaluation(**args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
