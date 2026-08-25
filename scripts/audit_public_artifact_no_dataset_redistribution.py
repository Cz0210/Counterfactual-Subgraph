#!/usr/bin/env python3
"""Public entrypoint for the TasteMolNet no-dataset-redistribution audit."""

from __future__ import annotations

from scripts.audit_tastemolnet_public_artifacts import build_parser, main


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
