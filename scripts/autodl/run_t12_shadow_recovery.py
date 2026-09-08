#!/usr/bin/env python3
"""Bounded T12 plan/status, evidence comparison and inherited-owner activation.

This entrypoint does not launch an observer, acquire a GPU, or install a new
controller. Shadow execution remains blocked until live-state tail and raw-cache
evidence bindings are implemented and their observational regression passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # -I deliberately ignores PYTHONPATH.

from src.utils.main_ready_task_specs import atomic_json
from src.utils.final16_owner_registry_v1 import process_start_ticks
from src.utils.t12_shadow_recovery import (
    activate_inherited_owner, build_shadow_plan, compare_ledgers,
    read_ledger, require_natural_510, validate_plan,
)


def _read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--input", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    status = sub.add_parser("status")
    status.add_argument("--plan", required=True, type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("--left", required=True, type=Path)
    compare.add_argument("--right", required=True, type=Path)
    compare.add_argument("--binding-sha", required=True)
    compare.add_argument("--start", required=True, type=int)
    compare.add_argument("--end", required=True, type=int)
    compare.add_argument("--output", required=True, type=Path)
    activate = sub.add_parser("activate-inherited")
    activate.add_argument("--plan", required=True, type=Path)
    activate.add_argument("--parity", required=True, type=Path)
    activate.add_argument("--owner-binding", required=True, type=Path)
    activate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.config.is_file():
        raise ValueError("T12_CONFIG_ABSENT")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("T12_FAIL_CLOSED_INFERENCE_REQUIRED")
    if args.action == "plan":
        result = build_shadow_plan(**_read(args.input))
        if args.output.exists():
            raise FileExistsError("T12_FRESH_PLAN_REQUIRED")
        atomic_json(args.output, result)
    elif args.action == "status":
        plan = _read(args.plan)
        validate_plan(plan)
        try:
            require_natural_510(plan, process_alive=lambda pid, ticks:
                process_start_ticks("/proc", pid) == ticks)
            natural = "PASS"
        except (ValueError, OSError, KeyError) as exc:
            natural = str(exc)
        result = {"plan": str(args.plan), "natural_510": natural,
                  "transitions_budgeted": plan["transitions_budgeted"],
                  "science_started": False, "status": "BLOCKED_IMPLEMENTATION",
                  "first_unclosed_function": "selected_raw_cache_evidence_binding",
                  "remaining_interfaces": ["live_state_continuous_501_510_tail",
                      "observer_train_only_exact_regression", "canonical_owner_activation_binding"],
                  "active_reader_changed": False, "matrix_write": False}
    elif args.action == "compare":
        left = read_ledger(args.left, binding_sha=args.binding_sha, start=args.start, end=args.end)
        right = read_ledger(args.right, binding_sha=args.binding_sha, start=args.start, end=args.end)
        result = compare_ledgers(left, right)
        atomic_json(args.output, result)
    else:
        result = activate_inherited_owner(plan=_read(args.plan), parity=_read(args.parity),
            binding=_read(args.owner_binding), output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
