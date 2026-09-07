#!/usr/bin/env python3
"""Train-only GIN128/+384 supplement; CPU compute node, never login science."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", required=True, help="Unmodified A+ source experiment spec")
    parser.add_argument("--output-root", required=True, help="Fresh supplement leaf, never old2607 root")
    parser.add_argument("--action", required=True, choices=("plan", "run", "status"))
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error("Existing runtime config required")
    from src.experiments import bace_gin_reach_supplement as leaf
    if args.action == "status":
        result = leaf.status(args.output_root)
    else:
        spec = json.loads(Path(args.spec).read_text())
        result = (leaf.prepare(spec, args.output_root)[0] if args.action == "plan"
                  else leaf.run(spec, args.output_root))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
