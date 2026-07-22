#!/usr/bin/env python3
"""Verify the local Mutagenicity download against its SHA256 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity import verify_download, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--root", default="data/raw/Mutagenicity")
    parser.add_argument("--manifest", default="data/raw/Mutagenicity/SHA256SUMS")
    parser.add_argument("--out-json", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_download(args.root, args.manifest)
    if args.out_json:
        write_json(Path(args.out_json), result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if not result["passed"]:
        print("[MUTAGENICITY_DOWNLOAD_VERIFY_FAILED]", flush=True)
        return 1
    print("[MUTAGENICITY_DOWNLOAD_VERIFY_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
