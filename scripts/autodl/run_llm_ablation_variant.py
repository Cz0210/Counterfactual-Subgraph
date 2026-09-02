#!/usr/bin/env python3
"""Execute/resume one evidence-bound core BACE LLM ablation variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.llm.core_execution import (  # noqa: E402
    load_authorized_launch_decision,
    load_core_run_spec,
    run_core_variant,
)
from src.ablations.llm.early_launch_gate import EarlyLaunchSnapshot  # noqa: E402


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip().lower()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--run-spec-sha256", required=True)
    parser.add_argument("--launch-decision", type=Path, required=True)
    parser.add_argument("--launch-decision-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    commit = _commit()
    spec = load_core_run_spec(args.run_spec, args.run_spec_sha256)
    if spec.execution_commit != commit:
        raise ValueError("run spec execution commit differs from deployed checkout")
    decision = load_authorized_launch_decision(
        args.launch_decision,
        args.launch_decision_sha256,
        spec=spec,
        execution_commit=commit,
    )
    snapshot_path = Path(decision["main_snapshot"]["path"])

    def live_snapshot() -> EarlyLaunchSnapshot:
        return EarlyLaunchSnapshot.from_mapping(_load_json(snapshot_path))

    result = run_core_variant(
        spec,
        resume=args.resume,
        live_snapshot_loader=live_snapshot,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("state") == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
