#!/usr/bin/env python3
"""Run one read-only external convergence audit of committed T14 checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Sequence
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import stable_json_sha256, write_json  # noqa: E402
from src.eval.tastemolnet_t14_external_convergence import (  # noqa: E402
    RECEIPT_SCHEMA,
    audit_t14_external_convergence,
)


def _physical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be one absolute physical directory: {path}")
    return path.resolve(strict=True)


def _fresh_output(path: Path, checkpoint_root: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise ValueError(f"output root must be fresh, absolute, and non-symlink: {path}")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(checkpoint_root)
    except ValueError:
        return resolved
    raise ValueError("external auditor output must not be inside the active T14 root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints = _physical_directory(args.checkpoint_root, "checkpoint root")
    output = _fresh_output(args.output_root, checkpoints)
    if re.fullmatch(r"[0-9a-f]{40}", str(args.execution_commit)) is None:
        raise ValueError("execution commit must be one exact lowercase Git SHA")
    output.mkdir(parents=True, exist_ok=False)
    audit = audit_t14_external_convergence(checkpoints)
    audit = {
        **audit,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "auditor_commit": str(args.execution_commit),
        "active_t14_root_modified": False,
        "active_sqlite_opened": False,
        "checkpoint_sqlite_opened": False,
        "signal_sent": False,
    }
    audit["audit_sha256"] = stable_json_sha256(audit)
    write_json(output / "t14_external_convergence_audit.json", audit)
    if audit.get("converged") is True:
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "PASS",
            "audit_path": str(output / "t14_external_convergence_audit.json"),
            "audit_sha256": audit["audit_sha256"],
            "stop_action_performed": False,
            "next_safe_checkpoint_exact_pid_handover_required": True,
            "calibration_loaded": False,
            "test_loaded": False,
        }
        receipt["receipt_sha256"] = stable_json_sha256(receipt)
        write_json(output / "t14_convergence_early_stop_receipt.json", receipt)
        print("[T14_EXTERNAL_CONVERGENCE_AUDITOR_PASS]", flush=True)
    else:
        write_json(
            output / "t14_convergence_waiting.json",
            {
                "status": audit["status"],
                "available_steps": audit["available_steps"],
                "converged": False,
                "stop_action_performed": False,
            },
        )
    print(f"state={audit['status']}")
    print(f"output_root={output}")
    print("sqlite_opened=false")
    print("signal_sent=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
