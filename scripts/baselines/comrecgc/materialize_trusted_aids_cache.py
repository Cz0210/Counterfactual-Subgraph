#!/usr/bin/env python3
"""Materialize the pinned AIDS PyG cache as a weights-only tensor payload."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import sha256_file  # noqa: E402


DATA_FILE_RE = re.compile(r"^data_(\d+)\.pt$")


def _torch_save_atomic(payload: object, destination: Path) -> None:
    import tempfile

    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-trust-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
        raise RuntimeError("Trusted AIDS cache loader requires its scoped compatibility flag.")
    trust_path = Path(args.cache_trust_json).expanduser().resolve()
    trust = json.loads(trust_path.read_text(encoding="utf-8"))
    if trust.get("cache_trust_passed") is not True:
        raise RuntimeError("Trusted AIDS cache preflight did not pass.")
    cache = Path(str(trust["cache_realpath"])).resolve(strict=True)
    rows = []
    for inventory_row in trust.get("files") or []:
        match = DATA_FILE_RE.fullmatch(str(inventory_row.get("path") or ""))
        if match is None:
            continue
        rows.append((int(match.group(1)), inventory_row))
    rows.sort(key=lambda value: value[0])
    if [index for index, _row in rows] != list(range(1837)):
        raise RuntimeError("Pinned AIDS cache data files are not the exact 0..1836 cohort.")

    import torch

    graphs = []
    for index, inventory_row in rows:
        source = cache / str(inventory_row["path"])
        if sha256_file(source) != inventory_row["sha256"]:
            raise RuntimeError(f"Trusted AIDS cache file changed before load: {source}")
        graph = torch.load(source, map_location="cpu", weights_only=False)
        graphs.append(
            {
                "x": graph.x.detach().cpu().clone(),
                "edge_index": graph.edge_index.detach().cpu().clone(),
                "edge_attr": (
                    graph.edge_attr.detach().cpu().clone()
                    if getattr(graph, "edge_attr", None) is not None
                    else None
                ),
                "y": graph.y.detach().cpu().clone(),
                "num_nodes": int(graph.num_nodes),
                "source_index": index,
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": "TU_AIDS",
        "graph_count": len(graphs),
        "cache_inventory_sha256": trust["cache_sha256"],
        "cache_realpath": str(cache),
        "graphs": graphs,
    }
    output = Path(args.output).expanduser().resolve()
    _torch_save_atomic(payload, output)
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    if reloaded.get("graph_count") != 1837 or len(reloaded.get("graphs") or []) != 1837:
        output.unlink(missing_ok=True)
        raise RuntimeError("Tensor-only AIDS cache payload failed safe reload.")
    print(
        json.dumps(
            {
                "cache_inventory_sha256": trust["cache_sha256"],
                "graph_count": 1837,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "scoped_non_weights_only_load": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
