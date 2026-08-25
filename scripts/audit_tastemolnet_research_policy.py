#!/usr/bin/env python3
"""Audit scoped TasteMolNet research/reporting policy and existing data only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from src.utils.tastemolnet_research_policy import (
    PENDING_STATUS,
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    validate_tastemolnet_local_authority,
)


RECEIPT_SCHEMA = "tastemolnet_research_reporting_policy_receipt_v1"
ACTIVE_MARKER = "TASTEMOLNET_SCOPED_RESEARCH_AUTHORIZED"
PENDING_MARKER = "TASTEMOLNET_POLICY_READY_EXECUTION_DISABLED"


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"policy audit output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
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


def audit_research_policy(
    *,
    policy_path: str | Path,
    prepared_root: str | Path,
    graph_cache_root: str | Path,
    output_dir: str | Path,
    expected_policy_sha256: str | None = None,
    require_active: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"policy audit output must be a fresh absent root: {output}")
    prepared = Path(prepared_root).expanduser().resolve(strict=True)
    cache = Path(graph_cache_root).expanduser().resolve(strict=True)
    for private_root in (prepared, cache):
        if (
            output == private_root
            or output in private_root.parents
            or private_root in output.parents
        ):
            raise TasteResearchPolicyError(
                "policy audit output must be disjoint from private data/cache"
            )
    policy = load_tastemolnet_research_policy(
        policy_path, expected_file_sha256=expected_policy_sha256
    )
    if require_active:
        policy.require_active()
    authority = validate_tastemolnet_local_authority(
        policy,
        prepared_root=prepared,
        graph_cache_root=cache,
    )
    marker = ACTIVE_MARKER if policy.active else PENDING_MARKER
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "tastemolnet",
        "status": policy.status,
        "authorization_state": policy.authorization_state,
        "policy": policy.evidence(),
        "private_data_authority": authority.evidence(),
        "run_tastemolnet": 1 if policy.active else 0,
        "heavy_route_authorized": policy.active,
        "paper_reporting_authorized": policy.active,
        "dataset_redistribution_authorized": False,
        "upstream_terms_status": "NOT_EXPLICITLY_STATED",
        "license_conclusion": "NOT_GRANTED_OR_INFERRED",
        "hpc_execution_authorized": False,
        "data_reprepared": False,
        "graph_cache_rebuilt": False,
        "terminal_marker": marker,
    }
    markdown = "\n".join(
        [
            "# TasteMolNet scoped data-use audit",
            "",
            f"- Policy status: `{policy.status}`",
            "- Upstream terms: `NOT_EXPLICITLY_STATED`",
            f"- Private research execution: `{str(policy.active).lower()}`",
            f"- Aggregate paper reporting: `{str(policy.active).lower()}`",
            "- Dataset redistribution: `false`",
            "- Upstream data licence was not inferred or declared passed.",
            "- Existing prepared data and graph caches were validated read-only; neither was rebuilt.",
            "- Public outputs require the independent no-dataset-redistribution audit.",
            "",
        ]
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_new(
        output / "tastemolnet_policy_receipt.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _write_new(output / "tastemolnet_policy_audit.md", markdown.encode("utf-8"))
    _write_new(output / marker, (marker + "\n").encode("utf-8"))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--policy", required=True, type=_absolute)
    parser.add_argument("--expected-policy-sha256")
    parser.add_argument("--prepared-root", required=True, type=_absolute)
    parser.add_argument("--graph-cache-root", required=True, type=_absolute)
    parser.add_argument("--output-dir", required=True, type=_absolute)
    parser.add_argument(
        "--require-active",
        action="store_true",
        help="Fail unless the independently reviewed policy activation is present.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = audit_research_policy(
            policy_path=args.policy,
            expected_policy_sha256=args.expected_policy_sha256,
            prepared_root=args.prepared_root,
            graph_cache_root=args.graph_cache_root,
            output_dir=args.output_dir,
            require_active=args.require_active,
        )
    except (FileExistsError, OSError, TasteResearchPolicyError, ValueError) as exc:
        print(f"TASTEMOLNET_RESEARCH_POLICY_AUDIT_FAILED: {exc}", flush=True)
        return 65
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"[{receipt['terminal_marker']}]", flush=True)
    if receipt["status"] == PENDING_STATUS:
        print("[TASTEMOLNET_EXECUTION_REMAINS_DISABLED]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
