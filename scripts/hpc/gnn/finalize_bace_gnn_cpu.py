#!/usr/bin/env python3
"""Publish complete BACE CPU staging bundles without rerunning any science."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.staged_finalize import finalize_models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--model-roots-json", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = finalize_models(bundle_root=args.bundle_root,
        model_roots=json.loads(Path(args.model_roots_json).read_text()), output_root=args.output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
