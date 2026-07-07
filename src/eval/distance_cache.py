"""Lightweight SQLite cache for pairwise graph distances."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DISTANCE_CACHE_PATH = "outputs/hpc/cache/distance_cache/molclr_node_fgw_v1.sqlite"


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_pair_key(
    *,
    distance_type: str,
    version: str,
    canonical_smiles_a: str,
    canonical_smiles_b: str,
    molclr_ckpt: str,
    fgw_lambda: float,
    structure_mode: str,
    feature_cost: str,
    atom_penalty: float,
) -> str:
    """Return the symmetric sha256 cache key for a molecule pair."""

    left, right = sorted([str(canonical_smiles_a), str(canonical_smiles_b)])
    payload = {
        "distance_type": str(distance_type),
        "version": str(version),
        "canonical_smiles_a": left,
        "canonical_smiles_b": right,
        "molclr_ckpt": str(molclr_ckpt),
        "fgw_lambda": float(fgw_lambda),
        "structure_mode": str(structure_mode),
        "feature_cost": str(feature_cost),
        "atom_penalty": float(atom_penalty),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


@dataclass
class DistanceCacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits / total) if total else 0.0


class SQLiteDistanceCache:
    """Small append/update SQLite distance cache with WAL enabled."""

    def __init__(self, path: str | Path = DEFAULT_DISTANCE_CACHE_PATH) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS distances (
                key TEXT PRIMARY KEY,
                distance_type TEXT NOT NULL,
                value REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()
        self.stats = DistanceCacheStats()

    def close(self) -> None:
        self.conn.close()

    def get_distance(self, key: str) -> tuple[float | None, dict[str, Any] | None]:
        cursor = self.conn.execute("SELECT value, metadata_json FROM distances WHERE key = ?", (str(key),))
        row = cursor.fetchone()
        if row is None:
            self.stats.misses += 1
            return None, None
        self.stats.hits += 1
        metadata: dict[str, Any] | None = None
        try:
            metadata = json.loads(row[1])
        except Exception:
            metadata = None
        return float(row[0]), metadata

    def set_distance(
        self,
        key: str,
        value: float,
        metadata: dict[str, Any] | None = None,
        *,
        distance_type: str = "molclr_node_fgw",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = dict(metadata or {})
        payload.setdefault("created_at", now)
        self.conn.execute(
            """
            INSERT INTO distances(key, distance_type, value, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (str(key), str(distance_type), float(value), _json_dumps(payload), now, now),
        )
        self.conn.commit()

    def stats_dict(self) -> dict[str, Any]:
        return {
            "pair_distance_cache_hits": self.stats.hits,
            "pair_distance_cache_misses": self.stats.misses,
            "pair_distance_cache_hit_rate": self.stats.hit_rate,
            "cache_path": str(self.path),
        }


__all__ = [
    "DEFAULT_DISTANCE_CACHE_PATH",
    "DistanceCacheStats",
    "SQLiteDistanceCache",
    "canonical_pair_key",
]
