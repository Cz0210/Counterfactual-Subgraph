#!/usr/bin/env python3
"""Hash-close the committed Taste T14 step-12,500 resume source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_t14_resume import (  # noqa: E402
    build_resume_spec,
    write_resume_spec,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--checkpoint-dir", type=_absolute, required=True)
    parser.add_argument("--resume-execution-commit", required=True)
    parser.add_argument("--historical-process-peak-bytes", type=int, required=True)
    parser.add_argument("--historical-checkpoint-peak-bytes", type=int, required=True)
    parser.add_argument("--spec-out", type=_absolute, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file() or args.config.is_symlink():
        raise ValueError("T14 resume spec requires one physical config file")
    payload = build_resume_spec(
        output_root=args.output_root,
        checkpoint_dir=args.checkpoint_dir,
        resume_execution_commit=args.resume_execution_commit,
        historical_process_peak_bytes=args.historical_process_peak_bytes,
        historical_checkpoint_peak_bytes=args.historical_checkpoint_peak_bytes,
    )
    path = write_resume_spec(args.spec_out, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "spec_path": str(path),
                "spec_sha256": payload["spec_sha256"],
                "checkpoint_digest": payload["checkpoint_digest"],
                "completed_step": payload["completed_step"],
                "historical_required_headroom_bytes": payload["memory"][
                    "historical_required_headroom_bytes"
                ],
                "optimized_canary_receipt_required": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
