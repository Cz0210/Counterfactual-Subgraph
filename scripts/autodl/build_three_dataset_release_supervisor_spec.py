#!/usr/bin/env python3
"""Freeze one hash-bound, CPU-only three-dataset release specification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.four_by_four_registry import sha256_file  # noqa: E402
from src.eval.three_dataset_release_supervisor import (  # noqa: E402
    ReleaseSpecError,
    build_release_spec,
    write_release_spec,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"build audit must be fresh: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--catalog", type=_absolute, required=True)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--project-root", type=_absolute, default=PROJECT_ROOT)
    parser.add_argument("--runtime-root", type=_absolute, required=True)
    parser.add_argument("--python", type=_absolute, required=True)
    parser.add_argument("--state-root", type=_absolute, required=True)
    parser.add_argument("--registry-root", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--paper-staging-root", type=_absolute, required=True)
    parser.add_argument("--expectations-json", type=_absolute, required=True)
    parser.add_argument("--taste-license-gate-json", type=_absolute, required=True)
    parser.add_argument("--adoption-manifest", type=_absolute)
    parser.add_argument(
        "--cell-root",
        action="append",
        default=[],
        metavar="DATASET/METHOD=/ABS/ROOT",
    )
    parser.add_argument(
        "--cell-owner-manifest",
        action="append",
        default=[],
        metavar="DATASET/METHOD=/ABS/MANIFEST.json",
    )
    parser.add_argument(
        "--cell-owner-task",
        action="append",
        default=[],
        metavar="DATASET/METHOD=TASK_ID",
    )
    parser.add_argument("--spec-output", type=_absolute, required=True)
    parser.add_argument("--build-audit", type=_absolute, required=True)
    parser.add_argument("--require-runnable", action="store_true")
    return parser


def _validate_compatibility_args(args: argparse.Namespace) -> None:
    if args.config is not None and not Path(args.config).is_file():
        raise ReleaseSpecError(f"Missing config: {args.config}")
    unsupported = [
        value
        for value in args.set
        if value != "inference.fallback_to_heuristic=false"
    ]
    if unsupported:
        raise ReleaseSpecError(f"Unsupported --set values: {unsupported}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_compatibility_args(args)
        spec = build_release_spec(
            catalog_path=args.catalog,
            controller_id=args.controller_id,
            project_root=args.project_root,
            runtime_root=args.runtime_root,
            python=args.python,
            state_root=args.state_root,
            registry_root=args.registry_root,
            output_root=args.output_root,
            paper_staging_root=args.paper_staging_root,
            expectations_json=args.expectations_json,
            taste_license_gate_json=args.taste_license_gate_json,
            adoption_manifest=args.adoption_manifest,
            cell_root_overrides=args.cell_root,
            owner_manifest_overrides=args.cell_owner_manifest,
            owner_task_overrides=args.cell_owner_task,
            require_runnable=args.require_runnable,
        )
        destination = write_release_spec(args.spec_output, spec)
        audit = {
            "schema_version": "three_dataset_release_spec_build_audit_v1",
            "status": "PASS" if spec["runnable"] else "BLOCKED_PLACEHOLDERS",
            "controller_id": spec["controller_id"],
            "spec_path": str(destination),
            "spec_file_sha256": sha256_file(destination),
            "spec_content_sha256": spec["content_sha256"],
            "execution_commit": spec["execution_commit"],
            "resource": "cpu",
            "gpu_required": False,
            "cell_count": len(spec["cells"]),
            "resolved_cell_count": sum(
                cell["binding_state"] == "FIXED" for cell in spec["cells"]
            ),
            "unresolved_bindings": list(spec["unresolved_bindings"]),
            "numeric_outputs_created": False,
        }
        _fresh_json(args.build_audit, audit)
    except (FileExistsError, OSError, ReleaseSpecError, ValueError) as exc:
        print(
            f"[THREE_DATASET_RELEASE_SPEC_FAILED] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    print(json.dumps(audit, indent=2, sort_keys=True))
    if spec["runnable"]:
        print("[THREE_DATASET_RELEASE_SPEC_PASS]")
    else:
        print("[THREE_DATASET_RELEASE_SPEC_BLOCKED_PLACEHOLDERS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
