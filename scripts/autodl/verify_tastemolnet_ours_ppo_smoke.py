#!/usr/bin/env python3
"""Independently consume one terminal Taste T6 PPO smoke and publish a receipt."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.tastemolnet_gnn_ppo import (  # noqa: E402
    TASTE_PPO_MARKER,
    validate_taste_ppo_output,
)


SCHEMA = "tastemolnet_t6_independent_verification_v1"
MARKER = "[TASTE_T6_OURS_PPO_INDEPENDENT_VERIFIER_PASS]"


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    normalized = Path(os.path.abspath(path))
    if not path.is_absolute() or path != normalized:
        raise argparse.ArgumentTypeError("one normalized absolute path is required")
    return path


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("short receipt write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Taste T6 receipt publication requires Linux renameat2")
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = int(
        renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
    )
    if result != 0:
        observed = ctypes.get_errno()
        if observed in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"verification root exists: {destination}")
        raise OSError(observed, os.strerror(observed), str(destination))


def verify(science_root: Path, verification_root: Path) -> dict[str, Any]:
    if verification_root.exists() or verification_root.is_symlink():
        raise FileExistsError(f"verification root must be fresh: {verification_root}")
    if not verification_root.parent.is_dir():
        raise ValueError("verification parent must already exist")
    evidence = validate_taste_ppo_output(science_root)
    if (
        evidence.get("status") != "PASS"
        or evidence.get("stage") != "T6_OURS_SMOKE"
        or evidence.get("optimizer_step_count", 0) < 5
    ):
        raise ValueError("Taste T6 strict consumer did not return PASS")
    staging = verification_root.parent / (
        f".{verification_root.name}.staging-{os.getpid()}"
    )
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"verification staging root exists: {staging}")
    staging.mkdir(mode=0o700)
    terminal = False
    try:
        verification = {
            "schema_version": SCHEMA,
            "status": "PASS",
            "stage": "T6_OURS_SMOKE",
            "independent_verifier": True,
            "science_marker": TASTE_PPO_MARKER,
            "verifier_marker": MARKER,
            "science_root": str(science_root),
            "science_evidence": evidence,
        }
        data = (
            json.dumps(
                verification,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_new(staging / "verification.json", data)
        gate = {
            "schema_version": "tastemolnet_t6_independent_gate_v1",
            "status": "PASS",
            "stage": "T6_OURS_SMOKE",
            "independent_verifier": True,
            "marker": MARKER,
            "science_gate_sha256": evidence["gate_sha256"],
            "science_output_inventory_sha256": evidence[
                "output_inventory_sha256"
            ],
        }
        _write_new(
            staging / "gate.json",
            (
                json.dumps(gate, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode("utf-8"),
        )
        _write_new(staging / "PASS", (MARKER + "\n").encode("utf-8"))
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_noreplace(staging, verification_root)
        parent_fd = os.open(verification_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        terminal = True
        return verification
    finally:
        if not terminal and staging.is_dir() and not any(staging.iterdir()):
            staging.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--science-root", type=_absolute, required=True)
    parser.add_argument("--verification-root", type=_absolute, required=True)
    args = parser.parse_args(argv)
    if args.config.resolve(strict=True) != (REPO_ROOT / "configs/hpc.yaml"):
        raise ValueError("Taste T6 verifier requires configs/hpc.yaml")
    if args.overrides != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("Taste T6 verifier requires fail-closed inference override")
    result = verify(args.science_root, args.verification_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    print(MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
