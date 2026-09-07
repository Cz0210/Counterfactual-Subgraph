#!/usr/bin/env python3
"""CPU-only dataset-specific frozen pool screening entrypoint."""
import argparse
import json
from pathlib import Path
from src.baselines.comrecgc.rf_aligned_pool import screen_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Project config, recorded only; no inference fallback")
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--action", choices=["screen-pool", "status"], required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    if args.action == "status":
        path = root / "terminal.json"
        if not path.exists():
            path = root / "progress.json"
        print(path.read_text() if path.exists() else json.dumps({"state": "NOT_STARTED"}))
    else:
        config = json.loads(Path(args.run_manifest).read_text())
        print(json.dumps(screen_pool(config, root), sort_keys=True))


if __name__ == "__main__":
    main()
