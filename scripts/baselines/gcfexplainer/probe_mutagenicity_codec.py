#!/usr/bin/env python3
"""Re-run the deterministic 64-row strict source codec probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_adapter import (  # noqa: E402
    load_dataset_artifacts,
    run_codec_probe,
    write_failure_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("Codec probe requires --forbid-calibration-test.")
    if int(args.seed) != 13:
        raise ValueError("Codec probe seed must be 13.")
    try:
        schema, train, val, _generation, _summary = load_dataset_artifacts(
            args.dataset_dir
        )
        summary = run_codec_probe(
            (*train, *val),
            schema,
            output_dir=args.output_dir,
            limit=int(args.limit),
            require_all=True,
        )
    except Exception as exc:
        write_failure_artifacts(args.output_dir, error=exc, resolved_config=vars(args))
        raise
    print("[MUTAGENICITY_GCFEXPLAINER_CODEC_PROBE_OK]", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
