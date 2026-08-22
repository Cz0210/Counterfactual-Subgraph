#!/usr/bin/env python3
"""Run the persistent four-method × four-dataset AutoDL controller.

The scheduler implementation is shared with the audited four-GPU recovery
engine.  This entry point changes only the persistent control namespace, so a
continuation can never overwrite an earlier recovery controller.
"""

from __future__ import annotations

from collections.abc import Sequence

from scripts.autodl import run_four_gpu_recovery_controller as engine


CONTROLLER_NAME = "four_methods_four_datasets_continuation"


def main(argv: Sequence[str] | None = None) -> int:
    engine.CONTROLLER_NAME = CONTROLLER_NAME
    return engine.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
