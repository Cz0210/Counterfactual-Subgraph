#!/usr/bin/env python3
"""Publish or revalidate one exact read-only AIDS v5 snapshot adoption gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.aids_comrecgc_v5_snapshot_adoption import (  # noqa: E402
    create_snapshot_adoption,
    validate_snapshot_adoption,
)
from src.utils.aids_comrecgc_v5_snapshot import (  # noqa: E402
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_PARENT_COUNT,
    EXPECTED_ROWS,
    EXPECTED_VECTOR_DIM,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, required=True)
    parser.add_argument("--owner-manifest", type=_absolute, required=True)
    parser.add_argument("--owner-manifest-sha256", required=True)
    parser.add_argument("--owner-namespace-root", type=_absolute, required=True)
    parser.add_argument("--owner-task-gate", type=_absolute, required=True)
    parser.add_argument("--owner-task-gate-sha256", required=True)
    parser.add_argument("--snapshot-root", type=_absolute, required=True)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--dbscan-contract-sha256", required=True)
    parser.add_argument("--pair-store-manifest-sha256", required=True)
    parser.add_argument("--pairs-sha256", required=True)
    parser.add_argument("--vectors-sha256", required=True)
    parser.add_argument("--source-root", type=_absolute, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--allowed-pid", type=int, required=True)
    parser.add_argument("--allowed-start-ticks", type=int, required=True)
    parser.add_argument("--allowed-cmdline-sha256", required=True)
    parser.add_argument("--allowed-output-root", type=_absolute, required=True)
    parser.add_argument("--allowed-project-root", type=_absolute, required=True)
    parser.add_argument("--expected-row-count", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--expected-vector-dim", type=int, default=EXPECTED_VECTOR_DIM)
    parser.add_argument("--expected-parent-count", type=int, default=EXPECTED_PARENT_COUNT)
    parser.add_argument(
        "--expected-candidate-count", type=int, default=EXPECTED_CANDIDATE_COUNT
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _common(args: argparse.Namespace) -> dict[str, object]:
    return {
        "output_dir": args.output_dir,
        "proc_root": args.proc_root,
        "owner_manifest": args.owner_manifest,
        "owner_manifest_sha256": args.owner_manifest_sha256,
        "owner_namespace_root": args.owner_namespace_root,
        "owner_task_gate": args.owner_task_gate,
        "owner_task_gate_sha256": args.owner_task_gate_sha256,
        "snapshot_root": args.snapshot_root,
        "snapshot_manifest_sha256": args.snapshot_manifest_sha256,
        "dbscan_contract_sha256": args.dbscan_contract_sha256,
        "pair_store_manifest_sha256": args.pair_store_manifest_sha256,
        "pairs_sha256": args.pairs_sha256,
        "vectors_sha256": args.vectors_sha256,
        "source_root": args.source_root,
        "source_manifest_sha256": args.source_manifest_sha256,
        "allowed_pid": args.allowed_pid,
        "allowed_start_ticks": args.allowed_start_ticks,
        "allowed_cmdline_sha256": args.allowed_cmdline_sha256,
        "allowed_output_root": args.allowed_output_root,
        "allowed_project_root": args.allowed_project_root,
        "expected_row_count": args.expected_row_count,
        "expected_vector_dim": args.expected_vector_dim,
        "expected_parent_count": args.expected_parent_count,
        "expected_candidate_count": args.expected_candidate_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = _common(args)
    if args.validate_only:
        # Recompute the exact identity through the same create API only when
        # publishing.  Validation reads the hash-bound identity from the
        # terminal adoption manifest, then reopens the owner and full snapshot
        # closure.  The caller-provided arguments are compared below so a
        # tampered terminal cannot redirect science to another snapshot.
        output = Path(args.output_dir)
        terminal_path = output / "snapshot_adoption_manifest.json"
        if (
            output.is_symlink()
            or terminal_path.is_symlink()
            or not output.resolve(strict=True).is_dir()
            or not terminal_path.resolve(strict=True).is_file()
        ):
            raise RuntimeError("snapshot adoption terminal path is not physical")
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        identity = terminal.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeError("snapshot adoption identity is missing")
        expected = {
            "owner_manifest": str(Path(args.owner_manifest).resolve(strict=True)),
            "owner_manifest_sha256": args.owner_manifest_sha256,
            "owner_namespace_root": str(
                Path(args.owner_namespace_root).resolve(strict=True)
            ),
            "owner_task_gate": str(Path(args.owner_task_gate).resolve(strict=True)),
            "owner_task_gate_sha256": args.owner_task_gate_sha256,
            "snapshot_root": str(Path(args.snapshot_root).resolve(strict=True)),
            "snapshot_manifest_sha256": args.snapshot_manifest_sha256,
            "dbscan_contract_sha256": args.dbscan_contract_sha256,
            "pair_store_manifest_sha256": args.pair_store_manifest_sha256,
            "pairs_sha256": args.pairs_sha256,
            "vectors_sha256": args.vectors_sha256,
            "source_root": str(Path(args.source_root).resolve(strict=True)),
            "source_manifest_sha256": args.source_manifest_sha256,
            "expected_row_count": args.expected_row_count,
            "expected_vector_dim": args.expected_vector_dim,
            "expected_parent_count": args.expected_parent_count,
            "expected_candidate_count": args.expected_candidate_count,
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise RuntimeError(f"snapshot adoption CLI identity mismatch: {key}")
        old = identity.get("allowed_old_generation")
        if not isinstance(old, dict) or old != {
            "pid": args.allowed_pid,
            "start_ticks": args.allowed_start_ticks,
            "cmdline_sha256": args.allowed_cmdline_sha256,
            "output_root": str(Path(args.allowed_output_root).resolve(strict=True)),
            "project_root": str(Path(args.allowed_project_root).resolve(strict=True)),
        }:
            raise RuntimeError("snapshot adoption CLI old-generation mismatch")
        result = validate_snapshot_adoption(
            output_dir=args.output_dir,
            proc_root=args.proc_root,
            identity=identity,
            require_pass=True,
        )
        marker = "[AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_VALIDATE_PASS]"
    else:
        result = create_snapshot_adoption(**common, resume=args.resume)
        marker = "[AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_PASS]"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
