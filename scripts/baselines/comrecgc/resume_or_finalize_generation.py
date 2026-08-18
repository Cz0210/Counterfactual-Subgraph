#!/usr/bin/env python3
"""Finalize a complete COMRECGC run or resume only a strict safe checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.continuation import (  # noqa: E402
    decide_resume_or_finalize,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=None, help=argparse.SUPPRESS)
    value.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    value.add_argument("--generation-dir", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--expected-steps", type=int, default=50_000)
    value.add_argument(
        "--resume-command-json",
        help="JSON file containing one argv list for a checkpoint-aware runtime.",
    )
    value.add_argument("--validate-only", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    decision = decide_resume_or_finalize(
        args.generation_dir, expected_steps=args.expected_steps
    )
    if args.validate_only:
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["status"] in {"ALREADY_COMPLETE", "RESUME_SAFE"} else 3

    if decision["status"] == "ALREADY_COMPLETE":
        decision["continuation_action"] = "no_op"
        _write(output / "continuation_manifest.json", decision)
        _write(output / "_RUN_COMPLETE.json", {"run_complete": True, **decision})
        print("[COMRECGC_CONTINUATION_ALREADY_COMPLETE]")
        return 0

    if decision["status"] != "RESUME_SAFE":
        decision["continuation_action"] = "fail_closed"
        _write(output / "continuation_manifest.json", decision)
        raise SystemExit(
            "[COMRECGC_CONTINUATION_UNSAFE] No complete atomic RNG/transition/trace "
            "checkpoint exists; refusing an implicit step-0 rerun."
        )
    if not args.resume_command_json:
        raise SystemExit(
            "[COMRECGC_CONTINUATION_COMMAND_MISSING] A safe checkpoint exists, but "
            "no explicitly audited checkpoint-aware command was supplied."
        )
    command_path = Path(args.resume_command_json).expanduser().resolve()
    command = json.loads(command_path.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not command or not all(
        isinstance(value, str) and value for value in command
    ):
        raise ValueError("--resume-command-json must contain a nonempty argv list.")
    environment = os.environ.copy()
    environment["COMRECGC_RESUME_CHECKPOINT_DIR"] = str(
        decision["selected_checkpoint"]["candidate_root"]
    )
    subprocess.run(command, check=True, env=environment)
    final = decide_resume_or_finalize(
        args.generation_dir, expected_steps=args.expected_steps
    )
    if final["status"] != "ALREADY_COMPLETE":
        raise RuntimeError("Checkpoint-aware continuation returned without completing generation.")
    final["continuation_action"] = "resumed_from_safe_checkpoint"
    final["source_checkpoint"] = decision["selected_checkpoint"]
    _write(output / "continuation_manifest.json", final)
    _write(output / "_RUN_COMPLETE.json", {"run_complete": True, **final})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
