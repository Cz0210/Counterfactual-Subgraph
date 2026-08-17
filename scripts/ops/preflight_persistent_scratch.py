#!/usr/bin/env python3
"""Validate persistent scratch capacity, inode availability, and SQLite WAL locking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path


def audit(root: Path, *, min_free_bytes: int, min_free_inodes: int) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    stat = os.statvfs(root)
    with tempfile.TemporaryDirectory(prefix="sqlite-preflight-", dir=root) as temporary:
        database = Path(temporary) / "locking.sqlite3"
        first = sqlite3.connect(database, timeout=5)
        journal_mode = str(first.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        first.execute("CREATE TABLE values_test (id INTEGER PRIMARY KEY, value TEXT)")
        first.execute("INSERT INTO values_test(value) VALUES ('first')")
        first.commit()
        second = sqlite3.connect(database, timeout=5)
        observed = second.execute("SELECT value FROM values_test").fetchone()[0]
        second.execute("INSERT INTO values_test(value) VALUES ('second')")
        second.commit()
        integrity = str(second.execute("PRAGMA integrity_check").fetchone()[0])
        checkpoint = tuple(int(value) for value in second.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
        second.close()
        first.close()
    result = {
        "schema_version": "persistent_scratch_preflight_v1",
        "root": str(root),
        "writable": os.access(root, os.W_OK),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "free_inodes": int(stat.f_favail),
        "min_free_bytes": int(min_free_bytes),
        "min_free_inodes": int(min_free_inodes),
        "sqlite_journal_mode": journal_mode,
        "sqlite_second_connection_read": observed == "first",
        "sqlite_integrity_check": integrity,
        "sqlite_wal_checkpoint": checkpoint,
    }
    result["SQLITE_LOCK_TEST_PASS"] = bool(
        journal_mode == "wal" and observed == "first" and integrity == "ok"
    )
    result["STORAGE_PREFLIGHT_PASS"] = bool(
        result["writable"]
        and usage.free >= int(min_free_bytes)
        and stat.f_favail >= int(min_free_inodes)
        and result["SQLITE_LOCK_TEST_PASS"]
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-free-gib", type=float, default=50.0)
    parser.add_argument("--min-free-inodes", type=int, default=100_000)
    args = parser.parse_args(argv)
    result = audit(
        Path(args.root).expanduser().resolve(),
        min_free_bytes=int(args.min_free_gib * 1024**3),
        min_free_inodes=args.min_free_inodes,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["STORAGE_PREFLIGHT_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
