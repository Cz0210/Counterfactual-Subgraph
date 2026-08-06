"""Read-only trust audit for the pinned upstream AIDS PyG cache."""

from __future__ import annotations

import grp
import os
import pwd
import stat
import subprocess
from pathlib import Path
from typing import Any

from .contracts import UPSTREAM_COMMIT, stable_json_sha256, write_json


def load_aids_tensor_payload(
    path: str | Path, *, expected_inventory_sha256: str
) -> tuple[list[Any], dict[str, Any]]:
    """Safely load the tensor-only derivative produced by the trusted child."""

    import torch
    from torch_geometric.data import Data

    source = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Trusted AIDS dataset payload must be a dictionary.")
    if payload.get("cache_inventory_sha256") != expected_inventory_sha256:
        raise ValueError("Trusted AIDS dataset payload cache lineage mismatch.")
    rows = payload.get("graphs")
    if not isinstance(rows, list) or len(rows) != 1837:
        raise ValueError("Trusted AIDS dataset payload must contain 1837 graphs.")
    graphs: list[Any] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or int(row.get("source_index", -1)) != index:
            raise ValueError("Trusted AIDS graph ordering differs from the frozen cache.")
        graphs.append(
            Data(
                x=row["x"],
                edge_index=row["edge_index"],
                edge_attr=row.get("edge_attr"),
                y=row["y"],
                num_nodes=int(row["num_nodes"]),
            )
        )
    return graphs, payload


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def audit_aids_pyg_cache(
    *,
    upstream_root: str | Path,
    output_path: str | Path,
    expected_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Inventory the exact pinned cache without loading or modifying it."""

    upstream = Path(upstream_root).expanduser().resolve()
    cache = upstream / "data/aids/processed"
    expected_cache = cache.resolve(strict=True)
    if cache.is_symlink():
        raise ValueError("Pinned AIDS cache directory must not be a symlink.")
    if expected_cache != cache.absolute():
        raise ValueError("Pinned AIDS cache realpath differs from its expected path.")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"Pinned COMRECGC commit mismatch: actual={commit}, expected={UPSTREAM_COMMIT}."
        )

    paths = sorted(cache.glob("*.pt"), key=lambda path: path.name)
    if not paths:
        raise ValueError(f"Pinned AIDS cache contains no .pt files: {cache}")
    rows: list[dict[str, Any]] = []
    any_symlink = False
    group_writable = False
    world_writable = False
    total_size = 0
    for path in paths:
        if path.is_symlink():
            any_symlink = True
            continue
        resolved = path.resolve(strict=True)
        if resolved.parent != expected_cache or not resolved.is_file():
            raise ValueError(f"Pinned AIDS cache file escapes its exact directory: {path}")
        metadata = resolved.stat()
        mode = stat.S_IMODE(metadata.st_mode)
        row = {
            "path": resolved.name,
            "bytes": int(metadata.st_size),
            "sha256": _sha256(resolved),
            "owner": pwd.getpwuid(metadata.st_uid).pw_name,
            "group": grp.getgrgid(metadata.st_gid).gr_name,
            "mode": f"{mode:04o}",
            "mtime_ns": int(metadata.st_mtime_ns),
            "is_symlink": False,
            "group_writable": bool(mode & stat.S_IWGRP),
            "world_writable": bool(mode & stat.S_IWOTH),
        }
        rows.append(row)
        total_size += int(metadata.st_size)
        group_writable = group_writable or bool(row["group_writable"])
        world_writable = world_writable or bool(row["world_writable"])

    inventory_sha256 = stable_json_sha256(rows)
    directory_stat = expected_cache.stat()
    directory_mode = stat.S_IMODE(directory_stat.st_mode)
    directory_group_writable = bool(directory_mode & stat.S_IWGRP)
    directory_world_writable = bool(directory_mode & stat.S_IWOTH)
    group_writable = group_writable or directory_group_writable
    world_writable = world_writable or directory_world_writable
    passed = bool(
        not any_symlink
        and not group_writable
        and not world_writable
        and (
            expected_inventory_sha256 is None
            or inventory_sha256 == expected_inventory_sha256
        )
    )
    result = {
        "cache_trust_schema_version": 1,
        "cache_trust_passed": passed,
        "cache_realpath": str(expected_cache),
        "cache_sha256": inventory_sha256,
        "cache_size": total_size,
        "cache_file_count": len(rows),
        "cache_owner": sorted({str(row["owner"]) for row in rows}),
        "cache_group": sorted({str(row["group"]) for row in rows}),
        "cache_mode": f"{directory_mode:04o}",
        "cache_directory_group_writable": directory_group_writable,
        "cache_directory_world_writable": directory_world_writable,
        "cache_file_modes": sorted({str(row["mode"]) for row in rows}),
        "cache_mtime_ns_min": min(int(row["mtime_ns"]) for row in rows),
        "cache_mtime_ns_max": max(int(row["mtime_ns"]) for row in rows),
        "cache_source": "pinned_upstream_processed_aids_cache",
        "upstream_commit": commit,
        "torch_version": _version("torch"),
        "torch_geometric_version": _version("torch_geometric"),
        "is_symlink": any_symlink,
        "group_writable": group_writable,
        "world_writable": world_writable,
        "expected_inventory_sha256": expected_inventory_sha256,
        "inventory_sha256_matches_expected": (
            expected_inventory_sha256 is None
            or inventory_sha256 == expected_inventory_sha256
        ),
        "environment_has_force_no_weights_only_load": (
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in os.environ
        ),
        "files": rows,
    }
    write_json(output_path, result)
    return result


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
