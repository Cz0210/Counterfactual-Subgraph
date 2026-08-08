#!/usr/bin/env python3
"""Prepare a deterministic BBBP graph dataset without downloading data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bbbp_adapter import (  # noqa: E402
    DEFAULT_SPLIT_RATIOS,
    prepare_bbbp_dataset,
    validate_bbbp_source,
)


def _ratios(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parsed) != 4:
        raise argparse.ArgumentTypeError("Expected train,val,calibration,test ratios.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--raw-csv", default="data/raw/BBBP/bbbp.csv")
    parser.add_argument("--output-dir", default="data/processed/BBBP")
    parser.add_argument(
        "--smiles-col",
        "--raw-smiles-col",
        dest="raw_smiles_col",
        default=os.environ.get("RAW_SMILES_COL"),
    )
    parser.add_argument(
        "--label-col",
        "--raw-label-col",
        dest="raw_label_col",
        default=os.environ.get("RAW_LABEL_COL"),
    )
    parser.add_argument("--seed", "--split-seed", dest="split_seed", type=int, default=13)
    parser.add_argument(
        "--split-config",
        help="Optional JSON/YAML object with split_ratios and acyclic_policy.",
    )
    parser.add_argument(
        "--split-ratios",
        type=_ratios,
        default=DEFAULT_SPLIT_RATIOS,
        help="Comma-separated train,val,calibration,test ratios.",
    )
    parser.add_argument(
        "--acyclic-policy",
        choices=("canonical-smiles", "group"),
        default="canonical-smiles",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_split_config(path: str | None) -> dict[str, object]:
    if not path:
        return {}
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment gate.
            raise RuntimeError("YAML split configs require PyYAML.") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"BBBP split config must be an object: {source}")
    allowed = {"split_ratios", "acyclic_policy"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unsupported BBBP split config fields: {unknown}")
    return dict(payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split_config = _load_split_config(args.split_config)
    ratios = tuple(split_config.get("split_ratios", args.split_ratios))
    acyclic_policy = str(split_config.get("acyclic_policy", args.acyclic_policy))
    if args.validate_only or args.dry_run:
        result = validate_bbbp_source(
            args.raw_csv,
            raw_smiles_col=args.raw_smiles_col,
            raw_label_col=args.raw_label_col,
        )
        result.update(
            {
                "mode": "validate_only" if args.validate_only else "dry_run",
                "planned_output_dir": str(Path(args.output_dir).expanduser()),
                "split_seed": int(args.split_seed),
                "split_ratios": list(ratios),
                "acyclic_policy": acyclic_policy,
            }
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        print("[BBBP_DATASET_VALIDATE_OK]", flush=True)
        return 0
    summary = prepare_bbbp_dataset(
        raw_csv=args.raw_csv,
        output_dir=args.output_dir,
        raw_smiles_col=args.raw_smiles_col,
        raw_label_col=args.raw_label_col,
        split_seed=args.split_seed,
        split_ratios=ratios,
        acyclic_policy=acyclic_policy,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[BBBP_DATASET_PREPARE_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
