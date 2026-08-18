from __future__ import annotations

import sqlite3

from src.baselines.globalgce_resumable import _atomic_json


def test_sqlite_checkpoint_marks_only_committed_roots_complete(tmp_path) -> None:
    database = tmp_path / "patterns.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE roots(root_index INTEGER PRIMARY KEY, root_label TEXT, complete INTEGER, pattern_count INTEGER)"
    )
    connection.execute("INSERT INTO roots VALUES(0, 'a', 1, 20)")
    connection.execute("INSERT INTO roots VALUES(1, 'b', 0, 3)")
    connection.commit()
    rows = connection.execute(
        "SELECT root_index FROM roots WHERE complete=1 ORDER BY root_index"
    ).fetchall()
    connection.close()
    _atomic_json(tmp_path / "checkpoint.json", {"completed_root_count": len(rows)})
    assert rows == [(0,)]
