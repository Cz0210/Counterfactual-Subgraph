#!/usr/bin/env python3
"""Lightweight RDKit-only debugger for nearest-parent-subgraph reward."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chem import compute_substructure_distance_reward


CASES = (
    {
        "name": "direct_match",
        "parent": "CCOc1ccc(N)cc1",
        "fragment": "c1ccc(N)cc1",
    },
    {
        "name": "similar_but_not_direct",
        "parent": "CCOc1ccc(N)cc1",
        "fragment": "c1ccc(Cl)cc1",
    },
    {
        "name": "unrelated_fragment",
        "parent": "CCOc1ccc(N)cc1",
        "fragment": "C1CCCCC1",
    },
)


def main() -> None:
    for case in CASES:
        result = compute_substructure_distance_reward(
            case["parent"],
            case["fragment"],
        )
        payload = {
            "case": case["name"],
            "parent": case["parent"],
            "fragment": case["fragment"],
            **result,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
