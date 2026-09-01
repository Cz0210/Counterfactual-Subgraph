#!/usr/bin/env python3
"""Reconcile completed zero-flip AIDS science, then publish via fast16 authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.aids_comrecgc_terminal_reconciliation import (  # noqa: E402
    publish_reconciliation,
    science_terminal_projection,
    validate_missing_controller_terminal,
    validate_reconciliation_root,
    validate_zero_strict_flip_science,
)
from src.eval.fast16_matrix_authority_pointer import (  # noqa: E402
    DEFAULT_LOCK_PATH,
    DEFAULT_STATE_PATH,
    append_under_authority_pointer,
)
from src.eval.non_taste_matrix_append import (  # noqa: E402
    _validate_aids_science_terminal,
    append_non_taste_matrix_cell,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--controller-manifest", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    actions = parser.add_subparsers(dest="action", required=True)
    reconcile = actions.add_parser("reconcile")
    _common(reconcile)
    reconcile.add_argument("--source-science-root", type=_absolute, required=True)
    reconcile.add_argument("--output-root", type=_absolute, required=True)
    verify = actions.add_parser("verify")
    _common(verify)
    verify.add_argument("--reconciliation-root", type=_absolute, required=True)
    publish = actions.add_parser("publish")
    _common(publish)
    publish.add_argument("--reconciliation-root", type=_absolute, required=True)
    publish.add_argument("--matrix-output-root", type=_absolute, required=True)
    publish.add_argument("--authority-state-path", type=_absolute, default=DEFAULT_STATE_PATH)
    publish.add_argument("--authority-lock-path", type=_absolute, default=DEFAULT_LOCK_PATH)
    publish.add_argument("--initial-authority-root", type=_absolute)
    return parser


def _check_cli(args: argparse.Namespace) -> None:
    if args.config not in (None, "configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("unsupported --set override")


def _reconcile(args: argparse.Namespace) -> dict[str, object]:
    controller = validate_missing_controller_terminal(
        args.controller_manifest, proc_root=args.proc_root
    )
    science = _validate_aids_science_terminal(
        args.source_science_root,
        controller_manifest_path=args.controller_manifest,
        proc_root=args.proc_root,
        require_writer_audit=True,
        require_controller_terminal=False,
    )
    zero = validate_zero_strict_flip_science(args.source_science_root)
    receipt = publish_reconciliation(
        output_root=args.output_root,
        science_projection=science_terminal_projection(science),
        controller_evidence=controller,
        zero_evidence=zero,
        proc_root=args.proc_root,
    )
    print("[AIDS_COMRECGC_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION_PASS]", flush=True)
    return receipt


def _verify(args: argparse.Namespace) -> dict[str, object]:
    receipt = validate_reconciliation_root(
        args.reconciliation_root, proc_root=args.proc_root
    )
    if (
        Path(str(receipt["controller_terminal_reconciliation"]["controller_manifest_path"]))
        .resolve(strict=True)
        != args.controller_manifest.resolve(strict=True)
    ):
        raise ValueError("reconciliation/controller manifest identity changed")
    print("[AIDS_COMRECGC_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION_PASS]", flush=True)
    return receipt


def _publish(args: argparse.Namespace) -> dict[str, object]:
    _verify(args)

    def append(prior: Path) -> dict[str, object]:
        return append_non_taste_matrix_cell(
            prior_authority_root=prior,
            dataset="AIDS",
            method="ComRecGC",
            cell_terminal_root=args.reconciliation_root,
            aids_controller_manifest=args.controller_manifest,
            output_root=args.matrix_output_root,
            proc_root=args.proc_root,
        )

    result = append_under_authority_pointer(
        state_path=args.authority_state_path,
        lock_path=args.authority_lock_path,
        initial_authority_root=args.initial_authority_root,
        requested_cells=("AIDS/ComRecGC",),
        append=append,
    )
    print(result["marker"], flush=True)
    return dict(result)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _check_cli(args)
    result = {
        "reconcile": _reconcile,
        "verify": _verify,
        "publish": _publish,
    }[args.action](args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
