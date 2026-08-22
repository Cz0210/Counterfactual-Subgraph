#!/usr/bin/env python3
"""Foreground CLI for strict AIDS/Mutagenicity legacy-result handling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.eval.am_legacy_standardization import (
    adopt_mutagenicity_ours,
    audit_legacy_inventory,
    freeze_mutagenicity_gcf_candidates,
    mut_gcf_contract_from_spec,
    mut_ours_contract_from_spec,
    verify_adopted_mut_ours,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly adopt the frozen Mutagenicity Ours result or inventory "
            "legacy AIDS/Mutagenicity raw evidence. No generation is run."
        )
    )
    # Controller and paired Slurm entrypoints always provide the project config.
    # This post-processing tool intentionally does not consume training settings.
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="action", required=True)

    adopt = subparsers.add_parser("adopt-mut-ours")
    adopt.add_argument("--source-spec", type=Path, required=True)
    adopt.add_argument("--output-root", type=Path, required=True)
    adopt.add_argument("--proc-root", type=Path, default=Path("/proc"))

    freeze_gcf = subparsers.add_parser("freeze-mut-gcf-candidates")
    freeze_gcf.add_argument("--source-spec", type=Path, required=True)
    freeze_gcf.add_argument(
        "--matched-threshold-contract", type=Path, required=True
    )
    freeze_gcf.add_argument("--output-root", type=Path, required=True)
    freeze_gcf.add_argument("--proc-root", type=Path, default=Path("/proc"))

    verify = subparsers.add_parser("verify-mut-ours-adoption")
    verify.add_argument("--adopted-root", type=Path, required=True)
    verify.add_argument("--output-root", type=Path, required=True)

    inventory = subparsers.add_parser("audit-inventory")
    inventory.add_argument("--source-spec", type=Path, required=True)
    inventory.add_argument("--output-root", type=Path, required=True)
    inventory.add_argument("--adopted-mut-ours-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "adopt-mut-ours":
        contract = mut_ours_contract_from_spec(args.source_spec)
        result = adopt_mutagenicity_ours(
            **contract,
            output_root=args.output_root,
            proc_root=args.proc_root,
        )
        marker = "[MUT_OURS_LEGACY_ADOPTION_PASS]"
    elif args.action == "freeze-mut-gcf-candidates":
        contract = mut_gcf_contract_from_spec(args.source_spec)
        result = freeze_mutagenicity_gcf_candidates(
            **contract,
            output_root=args.output_root,
            matched_threshold_contract=args.matched_threshold_contract,
            proc_root=args.proc_root,
        )
        marker = "[MUT_GCF_LEGACY_FREEZE_PASS]"
    elif args.action == "verify-mut-ours-adoption":
        result = verify_adopted_mut_ours(
            adopted_root=args.adopted_root,
            output_root=args.output_root,
        )
        marker = "[MUT_OURS_ADOPTION_VERIFY_PASS]"
    else:
        result = audit_legacy_inventory(
            source_spec=args.source_spec,
            output_root=args.output_root,
            adopted_mut_ours_root=args.adopted_mut_ours_root,
        )
        marker = "[AM_LEGACY_INVENTORY_PASS]"
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
