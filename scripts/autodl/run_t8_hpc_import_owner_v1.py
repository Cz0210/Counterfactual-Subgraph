#!/usr/bin/env python3
"""Wait for one exact relayed package, import it, and release sealed T13."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_autodl_import import (  # noqa: E402
    T8HPCAutoDLImportError,
    import_relayed_hpc_package,
    validate_relayed_hpc_package,
)
from src.utils.t8_hpc_t13_successor_v1 import (  # noqa: E402
    atomic_json,
    validate_spec_set,
    write_t13_release,
)


HEARTBEAT_SCHEMA = "t8_hpc_import_t13_owner_heartbeat_v1"


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def _heartbeat(path: Path, *, state: str, detail: Any = None) -> dict[str, Any]:
    payload = {
        "schema_version": HEARTBEAT_SCHEMA,
        "owner_pid": os.getpid(),
        "state": state,
        "detail": detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_write_enabled": False,
        "science_started": False,
    }
    atomic_json(path, payload)
    return payload


def _discover(spec: dict[str, Any]) -> tuple[Path | None, list[dict[str, str]]]:
    parent = Path(spec["relay_import_parent"])
    if not parent.is_dir():
        return None, []
    accepted: list[Path] = []
    rejected: list[dict[str, str]] = []
    for candidate in sorted(parent.iterdir(), key=lambda path: path.name):
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or not (candidate / spec["relay_ready_marker"]).is_file()
        ):
            continue
        try:
            validate_relayed_hpc_package(
                candidate,
                expected_execution_commit=spec["expected_hpc_execution_commit"],
                expected_scientific_input_sha256=spec[
                    "expected_scientific_input_sha256"
                ],
                expected_partition_manifest_sha256=spec[
                    "expected_partition_manifest_sha256"
                ],
                expected_official_globalgce_commit=spec[
                    "expected_official_globalgce_commit"
                ],
            )
            accepted.append(candidate.resolve(strict=True))
        except Exception as exc:
            rejected.append(
                {
                    "candidate": str(candidate),
                    "reason": f"{type(exc).__name__}: {exc}"[:1024],
                }
            )
    if len(accepted) > 1:
        raise T8HPCAutoDLImportError(
            "multiple relayed packages match the exact scientific identity"
        )
    return (accepted[0] if accepted else None), rejected


def run(
    *,
    spec_root: Path,
    heartbeat: Path,
    release: Path,
    poll_seconds: int,
    once: bool,
) -> dict[str, Any]:
    specs = validate_spec_set(spec_root, check_files=True)
    import_spec = specs["import"]
    import_root = Path(import_spec["output_root"])
    stopped = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    previous = {
        signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        while not stopped:
            if (import_root / "HPC_IMPORT_PASS").is_file():
                published = write_t13_release(
                    spec_root=spec_root,
                    import_root=import_root,
                    output=release,
                )
                result = _heartbeat(
                    heartbeat,
                    state="READY_WAITING_T13_GPU",
                    detail={
                        "import_root": str(import_root),
                        "release": str(release),
                        "release_sha256": published["release_sha256"],
                    },
                )
                if once:
                    return result
                time.sleep(poll_seconds)
                continue
            package, rejected = _discover(import_spec)
            if package is None:
                result = _heartbeat(
                    heartbeat,
                    state="WAITING_HPC_PACKAGE",
                    detail={
                        "relay_import_parent": import_spec["relay_import_parent"],
                        "rejected_candidates": rejected,
                    },
                )
                if once:
                    return result
                time.sleep(poll_seconds)
                continue
            _heartbeat(
                heartbeat,
                state="VERIFYING_HPC_PACKAGE",
                detail={"package_root": str(package)},
            )
            imported = import_relayed_hpc_package(
                package,
                import_root,
                expected_execution_commit=import_spec[
                    "expected_hpc_execution_commit"
                ],
                expected_scientific_input_sha256=import_spec[
                    "expected_scientific_input_sha256"
                ],
                expected_partition_manifest_sha256=import_spec[
                    "expected_partition_manifest_sha256"
                ],
                expected_official_globalgce_commit=import_spec[
                    "expected_official_globalgce_commit"
                ],
            )
            write_t13_release(
                spec_root=spec_root,
                import_root=import_root,
                output=release,
            )
            result = _heartbeat(
                heartbeat,
                state="READY_WAITING_T13_GPU",
                detail={
                    "package_root": str(package),
                    "import_root": str(import_root),
                    "import_manifest_sha256": imported["import_manifest_sha256"],
                    "release": str(release),
                },
            )
            if once:
                return result
            time.sleep(poll_seconds)
        return _heartbeat(heartbeat, state="STOPPED", detail=None)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default=None, help=argparse.SUPPRESS)
    result.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    result.add_argument("--spec-root", type=_absolute, required=True)
    result.add_argument("--heartbeat", type=_absolute, required=True)
    result.add_argument("--release", type=_absolute, required=True)
    result.add_argument("--poll-seconds", type=int, default=300)
    result.add_argument("--once", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml when supplied")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("unsupported --set override")
    if not 5 <= args.poll_seconds <= 3600:
        raise SystemExit("--poll-seconds must be in [5,3600]")
    result = run(
        spec_root=args.spec_root,
        heartbeat=args.heartbeat,
        release=args.release,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
