#!/usr/bin/env python3
"""Build the read-only four-method/four-dataset artifact registry on AutoDL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.four_by_four_registry import (  # noqa: E402
    AuditConfig,
    audit_registry,
    write_registry_outputs,
)


def _json_object(path: str | None, *, label: str) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {source}")
    return dict(payload)


def _default_scan_roots(runtime_root: Path) -> list[Path]:
    return [
        runtime_root / "outputs/hpc",
        runtime_root / "outputs/autodl",
        runtime_root / "outputs/final",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--runtime-root",
        default="/autodl-fs/data/counterfactual-subgraph-runtime",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="Read-only output tree to inventory; repeat for multiple roots.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--expectations-json",
        default=None,
        help="Optional exact dataset/oracle/split identity contract.",
    )
    parser.add_argument(
        "--explicit-cells-json",
        default=None,
        help="Optional mapping from '<dataset>/<method>' to candidate roots.",
    )
    parser.add_argument(
        "--taste-license-gate-json",
        default=None,
        help="Explicit TasteMolNet license PASS gate; absent means BLOCKED_LICENSE.",
    )
    parser.add_argument(
        "--max-hash-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Bounded SHA limit per inventory file; larger payloads are listed but not hashed.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return non-zero unless all 16 cells are passing/adoptable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    scan_roots = (
        [Path(value).expanduser().resolve() for value in args.scan_root]
        if args.scan_root
        else _default_scan_roots(runtime_root)
    )
    expectations = _json_object(args.expectations_json, label="expectations")
    explicit_payload = _json_object(args.explicit_cells_json, label="explicit cells")
    explicit_cells = dict(explicit_payload.get("cells") or explicit_payload)
    taste_gate = (
        _json_object(args.taste_license_gate_json, label="Taste license gate")
        if args.taste_license_gate_json
        else None
    )
    result = audit_registry(
        AuditConfig(
            scan_roots=tuple(scan_roots),
            output_root=Path(args.output_root).expanduser().resolve(),
            expectations=expectations,
            explicit_cells=explicit_cells,
            taste_license_gate=taste_gate,
            max_hash_bytes=args.max_hash_bytes,
        )
    )
    output_root = write_registry_outputs(result, args.output_root)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "matrix_complete_cells": result.matrix_complete_cells,
                "matrix_total_cells": result.matrix_total_cells,
                "scan_roots": [str(path) for path in scan_roots],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("[MATRIX_AUDIT_PASS]", flush=True)
    if args.require_complete and result.matrix_complete_cells != result.matrix_total_cells:
        print("[FOUR_BY_FOUR_MATRIX_INCOMPLETE]", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
