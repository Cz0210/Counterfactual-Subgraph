"""Pinned COMRECGC checkout and import helpers.

The upstream repository has no clear redistribution license.  This module only
loads a separately fetched checkout and verifies its exact commit.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import PurePosixPath
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from .contracts import ContractError, UPSTREAM_COMMIT, sha256_file

UPSTREAM_MODULES = (
    "util",
    "data",
    "gnn",
    "distance",
    "comrecgc",
    "common_recourse",
)


def _git(root: Path, *args: str) -> str:
    resolved_root = root.expanduser().resolve()
    config_value = str(resolved_root)
    if any(ord(character) < 32 for character in config_value):
        raise ValueError("COMRECGC checkout path contains a control character")
    quoted_value = config_value.replace("\\", "\\\\").replace('"', '\\"')
    # The ownership check backported to AutoDL's Git 2.34.1 ignores `git -c`
    # for safe.directory.  Redirect the global-config *lookup* for this child
    # process to one private exact-path file instead of modifying ~/.gitconfig
    # or the immutable vendor checkout.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="comrecgc-safe-directory-",
        suffix=".gitconfig",
    ) as safe_config:
        safe_config.write(f'[safe]\n\tdirectory = "{quoted_value}"\n')
        safe_config.flush()
        environment = os.environ.copy()
        environment["GIT_CONFIG_GLOBAL"] = safe_config.name
        environment["GIT_CONFIG_NOSYSTEM"] = "1"
        result = subprocess.run(
            ["git", "-C", str(resolved_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    return result.stdout.strip()


def validate_upstream_checkout(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"COMRECGC checkout is missing: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ContractError(
            f"COMRECGC checkout commit mismatch: actual={commit}, expected={UPSTREAM_COMMIT}"
        )
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    blocked: list[str] = []
    for line in dirty.splitlines():
        status, relative = line[:2], line[3:]
        parts = PurePosixPath(relative).parts
        runtime_data = bool(
            status == "??"
            and len(parts) >= 4
            and parts[0] == "data"
            and parts[2] in {"tudataset", "processed"}
        )
        vendor_manifest = bool(status == "??" and relative == "vendor_manifest.json")
        if not runtime_data and not vendor_manifest:
            blocked.append(line)
    if blocked:
        raise ContractError(
            "COMRECGC source checkout is dirty outside allowed TU runtime data: "
            + "; ".join(blocked[:20])
        )
    for filename in ("comrecgc.py", "common_recourse.py", "data.py", "gnn.py"):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Pinned COMRECGC file is missing: {root / filename}")
    vendor_manifest = root / "vendor_manifest.json"
    if vendor_manifest.is_file():
        payload = json.loads(vendor_manifest.read_text(encoding="utf-8"))
        expected_files = {
            filename: sha256_file(root / filename)
            for filename in ("comrecgc.py", "common_recourse.py", "data.py", "gnn.py")
        }
        if (
            payload.get("commit") != commit
            or payload.get("key_file_sha256") != expected_files
            or payload.get("read_only_usage") is not True
        ):
            raise ContractError("COMRECGC vendor manifest integrity check failed.")
    return root


@contextmanager
def imported_upstream(path: str | Path) -> Iterator[dict[str, ModuleType]]:
    root = validate_upstream_checkout(path)
    old_path = list(sys.path)
    old_dont_write_bytecode = sys.dont_write_bytecode
    displaced = {name: sys.modules.get(name) for name in UPSTREAM_MODULES}
    for name in UPSTREAM_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    # The pinned upstream commit contains a tracked CPython cache file.  Never
    # let importing the external checkout mutate that file (or create other
    # caches), especially when several Slurm jobs start concurrently.
    sys.dont_write_bytecode = True
    try:
        modules = {name: importlib.import_module(name) for name in UPSTREAM_MODULES}
        yield modules
    finally:
        for name in UPSTREAM_MODULES:
            sys.modules.pop(name, None)
            previous = displaced[name]
            if previous is not None:
                sys.modules[name] = previous
        sys.dont_write_bytecode = old_dont_write_bytecode
        sys.path[:] = old_path
