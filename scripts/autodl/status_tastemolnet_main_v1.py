#!/usr/bin/env python3
"""Read the fresh TasteMolNet main-controller state without mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_tastemolnet_main_v1 import (  # noqa: E402
    TasteMainControllerError,
    inspect_tastemolnet_main,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = Path(args.config).expanduser().resolve(strict=True)
        if config != (PROJECT_ROOT / "configs/hpc.yaml").resolve(strict=True):
            raise TasteMainControllerError("Taste status freezes configs/hpc.yaml")
        result = inspect_tastemolnet_main(args.controller_root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"TASTEMOLNET_MAIN_STATUS_FAILED: {exc}", file=sys.stderr, flush=True)
        return 65
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
