#!/usr/bin/env python3
"""Check runtime assets for the official GCFExplainer AIDS reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.gcf_official_adapter import check_official_assets, resolve_official_repo, write_asset_check  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--official-repo", default=None)
    parser.add_argument("--out-dir", default="outputs/hpc/gcfexplainer_official/asset_check")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    official_repo = resolve_official_repo(args.official_repo)
    payload = check_official_assets(official_repo)
    out_json = write_asset_check(payload, args.out_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[GCF_ASSET_CHECK] out_json={out_json}", flush=True)
    if payload.get("ok"):
        print("[GCF_ASSET_CHECK_OK]", flush=True)
        return 0
    print("[GCF_ASSET_CHECK_FAILED]", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

