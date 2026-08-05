"""Pinned COMRECGC checkout and import helpers.

The upstream repository has no clear redistribution license.  This module only
loads a separately fetched checkout and verifies its exact commit.
"""

from __future__ import annotations

import importlib
from pathlib import PurePosixPath
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from .contracts import ContractError, UPSTREAM_COMMIT

UPSTREAM_MODULES = (
    "util",
    "data",
    "gnn",
    "distance",
    "comrecgc",
    "common_recourse",
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
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
        if not runtime_data:
            blocked.append(line)
    if blocked:
        raise ContractError(
            "COMRECGC source checkout is dirty outside allowed TU runtime data: "
            + "; ".join(blocked[:20])
        )
    for filename in ("comrecgc.py", "common_recourse.py", "data.py", "gnn.py"):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Pinned COMRECGC file is missing: {root / filename}")
    return root


@contextmanager
def imported_upstream(path: str | Path) -> Iterator[dict[str, ModuleType]]:
    root = validate_upstream_checkout(path)
    old_path = list(sys.path)
    displaced = {name: sys.modules.get(name) for name in UPSTREAM_MODULES}
    for name in UPSTREAM_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    try:
        modules = {name: importlib.import_module(name) for name in UPSTREAM_MODULES}
        yield modules
    finally:
        for name in UPSTREAM_MODULES:
            sys.modules.pop(name, None)
            previous = displaced[name]
            if previous is not None:
                sys.modules[name] = previous
        sys.path[:] = old_path
