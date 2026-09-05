#!/usr/bin/env python
"""Write a fresh corrective promotion receipt; never refit or modify science."""
import argparse
import json
from pathlib import Path

from src.ablations.gnn.scientific_verification import temperature_promotion_audit
from src.eval.bace_frozen_gnn_contracts import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Correction receipt must have a fresh filename")
    result = temperature_promotion_audit(args.archive, args.sha256)
    atomic_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
