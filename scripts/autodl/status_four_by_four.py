#!/usr/bin/env python3
"""Read-only status view for the four-method × four-dataset controller."""

from __future__ import annotations

from collections.abc import Sequence

from scripts.autodl import run_four_gpu_recovery_controller as engine
from scripts.autodl import status_four_gpu_recovery as status


CONTROLLER_NAME = "four_methods_four_datasets_continuation"


def main(argv: Sequence[str] | None = None) -> int:
    engine.CONTROLLER_NAME = CONTROLLER_NAME
    status.CONTROLLER_NAME = CONTROLLER_NAME
    return status.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
