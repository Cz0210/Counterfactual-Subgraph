#!/usr/bin/env python3
"""Package a sealed hierarchical T8 merge without rerunning the merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import globalgce_hpc_exact as exact  # noqa: E402
from src.baselines.globalgce_hpc_exact import validate_hpc_cli_contract  # noqa: E402
from src.baselines.globalgce_hpc_hierarchical import (  # noqa: E402
    FINAL_VERIFICATION_SCHEMA,
    publish_hierarchical_evidence,
)
from src.baselines.globalgce_hpc_storage_safe import (  # noqa: E402
    build_storage_safe_archive,
    publish_storage_safe_archive,
    stream_verify_storage_safe_bundle,
    write_source_shard_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--merge-root", required=True, type=Path)
    parser.add_argument("--group-plan", required=True, type=Path)
    parser.add_argument("--groups-root", required=True, type=Path)
    parser.add_argument("--parity-receipt", required=True, type=Path)
    parser.add_argument("--environment-manifest", required=True, type=Path)
    parser.add_argument("--slurm-inventory", required=True, type=Path)
    parser.add_argument("--resource-metrics", required=True, type=Path)
    parser.add_argument("--packaging-commit", required=True)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    manifest = exact.validate_partition_manifest(args.partition_manifest)
    merge = json.loads((args.merge_root / "merge_manifest.json").read_text())
    claimed_merge = merge.pop("result_sha256", None)
    if exact.canonical_sha256(merge) != claimed_merge:
        raise SystemExit("hierarchical merge self-hash is invalid")
    merge["result_sha256"] = claimed_merge
    hierarchical = json.loads(
        (args.merge_root / "hierarchical_verification.json").read_text(encoding="utf-8")
    )
    claimed = hierarchical.pop("verification_sha256", None)
    if (
        hierarchical.get("schema_version") != FINAL_VERIFICATION_SCHEMA
        or hierarchical.get("status") != "PASS"
        or hierarchical.get("merge_result_sha256") != merge["result_sha256"]
        or exact.canonical_sha256(hierarchical) != claimed
    ):
        raise SystemExit("hierarchical final verification is invalid")
    hierarchical["verification_sha256"] = claimed
    scratch = args.scratch_root.resolve(strict=True)
    source_inventory = scratch / "source_shard_inventory.json"
    write_source_shard_inventory(
        partition_manifest=args.partition_manifest,
        shards_root=args.shards_root,
        merge_manifest=merge,
        output=source_inventory,
    )
    archive = scratch / "t8_exact_result_bundle.tar.gz"
    built = build_storage_safe_archive(
        partition_manifest=args.partition_manifest,
        merge_root=args.merge_root,
        parity_receipt=args.parity_receipt,
        environment_manifest=args.environment_manifest,
        slurm_inventory=args.slurm_inventory,
        resource_metrics=args.resource_metrics,
        source_shard_inventory=source_inventory,
        packaging_commit=args.packaging_commit,
        output_archive=archive,
    )
    verified = stream_verify_storage_safe_bundle(archive)
    report = publish_storage_safe_archive(
        scratch_archive=archive,
        inner_manifest=built["inner_manifest"],
        prepublication_verification=verified,
        output_root=args.output_root,
    )
    hierarchy = publish_hierarchical_evidence(
        group_plan=args.group_plan,
        groups_root=args.groups_root,
        merge_root=args.merge_root,
        package_root=args.output_root,
        storage_safe_receipt=report,
        scratch_root=scratch,
    )
    report["hierarchical_package_ready_sha256"] = hierarchy[
        "package_ready_sha256"
    ]
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
