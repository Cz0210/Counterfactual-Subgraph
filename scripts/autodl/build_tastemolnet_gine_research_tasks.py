#!/usr/bin/env python3
"""Build the fresh, typed TasteMolNet GINE research controller fragment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from src.baselines.tastemolnet_gine_research_tasks import (
    build_tastemolnet_gine_research_fragment,
    validate_tastemolnet_gine_research_fragment,
)
from src.utils.tastemolnet_research_policy import TasteResearchPolicyError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"fragment output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--expected-policy-sha256")
    parser.add_argument("--prepared-root", type=_absolute)
    parser.add_argument("--graph-cache-root", type=_absolute)
    parser.add_argument("--policy-receipt", type=_absolute)
    parser.add_argument("--expected-output-root", type=_absolute, required=True)
    parser.add_argument("--output", type=_absolute, required=True)
    parser.add_argument("--require-active", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (
            args.output == args.expected_output_root
            or args.expected_output_root in args.output.parents
        ):
            raise TasteResearchPolicyError(
                "fragment output must remain outside the future science root"
            )
        payload = build_tastemolnet_gine_research_fragment(
            policy_path=args.policy,
            expected_policy_sha256=args.expected_policy_sha256,
            prepared_root=args.prepared_root,
            graph_cache_root=args.graph_cache_root,
            policy_receipt=args.policy_receipt,
            expected_output_root=args.expected_output_root,
        )
        validate_tastemolnet_gine_research_fragment(
            payload, require_active=args.require_active
        )
        _write_new_json(args.output, payload)
    except (
        FileExistsError,
        OSError,
        TasteResearchPolicyError,
        ValueError,
    ) as exc:
        print(f"TASTEMOLNET_GINE_FRAGMENT_FAILED: {exc}", flush=True)
        return 65
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["status"],
                "policy_active": payload["policy_active"],
                "task_id": payload["tasks"][0]["id"],
                "enabled": payload["tasks"][0]["enabled"],
                "physical_gpu_index": payload["tasks"][0]["physical_gpu_index"],
                "gpu_lock_mode": payload["tasks"][0]["gpu_lock_mode"],
                "run_tastemolnet": payload["tasks"][0]["run_tastemolnet"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    marker = (
        "TASTEMOLNET_GINE_RESEARCH_FRAGMENT_ACTIVE"
        if payload["policy_active"]
        else "TASTEMOLNET_GINE_RESEARCH_FRAGMENT_DISABLED"
    )
    print(f"[{marker}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
