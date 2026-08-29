"""Production I/O for fixed-budget TasteMolNet NeuroSED pairs and labels.

The pair sampler intentionally stores reconstruction metadata rather than one
large graph payload per pair.  This module reopens the self-hashed sampler
bundle, reconstructs each sampled query from its persisted seed, and provides
one pickle-free NumPy ``npz`` contract shared by the GED writer and trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from src.data.tastemolnet_neurosed_fixed_budget import (
    FixedBudgetGraph,
    FixedBudgetQuery,
    PAIR_SAMPLER_MANIFEST_SCHEMA,
    reserve_pair_count,
    sample_official_style_query,
)


COMPACT_LABEL_SCHEMA = "tastemolnet_neurosed_compact_ged_labels_v1"
COMPACT_LABEL_COLUMNS = (
    "pair_id",
    "query_graph_id",
    "target_graph_id",
    "query_hash",
    "target_hash",
    "split",
    "lower_bound",
    "upper_bound",
    "exact_bound",
    "label_contract",
    "backend",
    "backend_config_hash",
    "status",
    "elapsed_seconds",
    "cache_key",
    "cache_hit",
    "error",
)
LABEL_CONTRACT = "ordered_query_target_lower_upper_interval"
ATTEMPT_STATUSES = frozenset({"SUCCESS", "TIMEOUT", "GEDLIB_ERROR"})


class NeuroSEDProductionDataError(RuntimeError):
    """A fixed-budget pair/compact-label authority is malformed or drifted."""


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeuroSEDProductionDataError(f"invalid JSON authority: {path}") from exc
    if type(value) is not dict:
        raise NeuroSEDProductionDataError(f"JSON authority is not one object: {path}")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if type(value) is not dict:
                    raise NeuroSEDProductionDataError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeuroSEDProductionDataError(f"invalid JSONL authority: {path}") from exc
    return rows


@dataclass(frozen=True, slots=True)
class PreparedFixedBudgetPair:
    metadata: Mapping[str, Any]
    query: FixedBudgetQuery
    target: FixedBudgetGraph

    @property
    def pair_id(self) -> str:
        return str(self.metadata["pair_id"])


@dataclass(frozen=True, slots=True)
class FixedBudgetPairInventory:
    root: Path
    split: str
    requested_pair_count: int
    manifest: Mapping[str, Any]
    manifest_file_sha256: str
    pairs_file_sha256: str
    graph_inventory_file_sha256: str
    reserve_available_count: int
    pairs: tuple[PreparedFixedBudgetPair, ...]


def _graphs(rows: Iterable[Mapping[str, Any]], *, split: str) -> dict[str, FixedBudgetGraph]:
    result: dict[str, FixedBudgetGraph] = {}
    for index, row in enumerate(rows):
        try:
            graph = FixedBudgetGraph(
                graph_id=str(row["graph_id"]),
                split=str(row["split"]),
                node_labels=tuple(int(value) for value in row["node_labels"]),
                directed_edges=tuple(
                    (int(edge[0]), int(edge[1])) for edge in row["directed_edges"]
                ),
                scaffold=str(row["scaffold"]),
                class_label=int(row["class_label_sampling_diagnostic_only"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NeuroSEDProductionDataError(
                f"malformed graph inventory row {index}"
            ) from exc
        if (
            graph.split != split
            or graph.graph_sha256 != row.get("graph_sha256")
            or graph.canonical_graph_sha256 != row.get("canonical_graph_sha256")
            or graph.graph_id in result
        ):
            raise NeuroSEDProductionDataError("graph inventory identity changed")
        result[graph.graph_id] = graph
    if not result:
        raise NeuroSEDProductionDataError("graph inventory is empty")
    return result


def load_fixed_budget_pair_inventory(
    root: str | Path,
    *,
    split: str,
    requested_pair_count: int,
) -> FixedBudgetPairInventory:
    """Reopen one frozen exact-budget or explicitly materialized reserve bundle."""

    if split not in ("train", "validation"):
        raise NeuroSEDProductionDataError("pair inventory split is not admitted")
    requested = int(requested_pair_count)
    if requested <= 0:
        raise NeuroSEDProductionDataError("requested pair count must be positive")
    source = Path(root).absolute()
    manifest_path = source / "pair_sampler_manifest.json"
    pairs_path = source / "pairs.jsonl"
    graphs_path = source / "graph_inventory.jsonl"
    manifest = load_json(manifest_path)
    claimed_manifest_sha = str(manifest.get("manifest_sha256") or "")
    if claimed_manifest_sha != stable_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise NeuroSEDProductionDataError("pair sampler manifest self-hash changed")
    inventory_pair_count = manifest.get("pair_count")
    admitted_pair_counts = {requested, reserve_pair_count(requested)}
    if type(inventory_pair_count) is not int or inventory_pair_count not in admitted_pair_counts:
        raise NeuroSEDProductionDataError(
            "pair inventory must contain the exact budget or its frozen 10% reserve"
        )
    exact = {
        "schema_version": PAIR_SAMPLER_MANIFEST_SCHEMA,
        "dataset": "tastemolnet",
        "split": split,
        "pair_count": inventory_pair_count,
        "pair_sampling_seed": 7,
        "independent_query_target_pairs": True,
        "query_graph_id_differs_from_target_graph_id": True,
        "parent_own_subgraph_shortcut": False,
        "cartesian_product_materialized": False,
        "ged_labels_present": False,
        "class_label_used_as_supervision": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    if any(manifest.get(key) != value for key, value in exact.items()):
        raise NeuroSEDProductionDataError("pair sampler contract changed")
    pairs_sha = sha256_file(pairs_path)
    graphs_sha = sha256_file(graphs_path)
    if (
        pairs_sha != manifest.get("pairs_jsonl_sha256")
        or graphs_sha != manifest.get("graph_inventory_sha256")
    ):
        raise NeuroSEDProductionDataError("pair sampler payload hash changed")
    metadata_rows = load_jsonl(pairs_path)
    graph_map = _graphs(load_jsonl(graphs_path), split=split)
    if (
        len(metadata_rows) != inventory_pair_count
        or len({row.get("pair_id") for row in metadata_rows}) != inventory_pair_count
        or stable_sha256(metadata_rows) != manifest.get("metadata_rows_sha256")
    ):
        raise NeuroSEDProductionDataError("pair metadata inventory changed")
    query_config = manifest.get("query_generation")
    if type(query_config) is not dict:
        raise NeuroSEDProductionDataError("pair query-generation contract is absent")
    prepared: list[PreparedFixedBudgetPair] = []
    for index, row in enumerate(metadata_rows):
        try:
            query_graph_id = str(row["query_graph_id"])
            target_graph_id = str(row["target_graph_id"])
            query_source = graph_map[query_graph_id]
            target = graph_map[target_graph_id]
            query = sample_official_style_query(
                query_source,
                n_hops=int(query_config["n_hops_query"]),
                traversal_probability=float(
                    query_config["traversal_probability_query"]
                ),
                node_limit=query_config.get("node_limit_query"),
                sampling_seed=int(row["query_sampling_seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NeuroSEDProductionDataError(
                f"pair reconstruction failed at row {index}"
            ) from exc
        if (
            row.get("query_split") != split
            or row.get("target_split") != split
            or query_graph_id == target_graph_id
            or query.graph_sha256 != row.get("query_instance_sha256")
            or query.canonical_graph_sha256
            != row.get("query_canonical_graph_sha256")
            or target.graph_sha256 != row.get("target_graph_sha256")
            or target.canonical_graph_sha256
            != row.get("target_canonical_graph_sha256")
            or row.get("ged_direction") != "query_to_target"
            or row.get("ged_label_present") is not False
        ):
            raise NeuroSEDProductionDataError("reconstructed pair identity changed")
        prepared.append(
            PreparedFixedBudgetPair(metadata=dict(row), query=query, target=target)
        )
    return FixedBudgetPairInventory(
        root=source,
        split=split,
        requested_pair_count=requested,
        manifest=manifest,
        manifest_file_sha256=sha256_file(manifest_path),
        pairs_file_sha256=pairs_sha,
        graph_inventory_file_sha256=graphs_sha,
        reserve_available_count=inventory_pair_count - requested,
        pairs=tuple(prepared),
    )


def _unicode_array(values: Sequence[Any]) -> Any:
    import numpy as np

    strings = [str(value) for value in values]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype=f"<U{width}")


def _compact_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not rows:
        raise NeuroSEDProductionDataError("compact label rows are empty")
    statuses = [str(row.get("status") or "") for row in rows]
    if any(status not in ATTEMPT_STATUSES for status in statuses):
        raise NeuroSEDProductionDataError("compact label status changed")

    def optional_bound(row: Mapping[str, Any], key: str) -> float:
        value = row.get(key)
        return float("nan") if value is None else float(value)

    arrays: dict[str, Any] = {
        "schema_version": _unicode_array([COMPACT_LABEL_SCHEMA]),
        "pair_id": _unicode_array([row.get("pair_id", "") for row in rows]),
        "query_graph_id": _unicode_array(
            [row.get("query_graph_id", "") for row in rows]
        ),
        "target_graph_id": _unicode_array(
            [row.get("target_graph_id", "") for row in rows]
        ),
        "query_hash": _unicode_array([row.get("query_hash", "") for row in rows]),
        "target_hash": _unicode_array([row.get("target_hash", "") for row in rows]),
        "split": _unicode_array([row.get("split", "") for row in rows]),
        "lower_bound": np.asarray(
            [optional_bound(row, "lower_bound") for row in rows], dtype=np.float32
        ),
        "upper_bound": np.asarray(
            [optional_bound(row, "upper_bound") for row in rows], dtype=np.float32
        ),
        "exact_bound": np.asarray(
            [bool(row.get("exact_bound", False)) for row in rows], dtype=np.bool_
        ),
        "label_contract": _unicode_array(
            [row.get("label_contract", "") for row in rows]
        ),
        "backend": _unicode_array([row.get("backend", "") for row in rows]),
        "backend_config_hash": _unicode_array(
            [row.get("backend_config_hash", "") for row in rows]
        ),
        "status": _unicode_array(statuses),
        "elapsed_seconds": np.asarray(
            [row.get("elapsed_seconds", 0.0) for row in rows], dtype=np.float64
        ),
        "cache_key": _unicode_array([row.get("cache_key", "") for row in rows]),
        "cache_hit": np.asarray(
            [bool(row.get("cache_hit", False)) for row in rows], dtype=np.bool_
        ),
        "error": _unicode_array([row.get("error", "") for row in rows]),
    }
    total = len(rows)
    if any(array.shape[0] != total for key, array in arrays.items() if key != "schema_version"):
        raise NeuroSEDProductionDataError("compact label columns are misaligned")
    return arrays


def write_compact_npz(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Atomically write one pickle-free compressed NumPy label table."""

    import numpy as np

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **_compact_arrays(rows))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return sha256_file(destination)


def read_compact_npz(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate one compact table without permitting object arrays."""

    import numpy as np

    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            if set(payload.files) != {"schema_version", *COMPACT_LABEL_COLUMNS}:
                raise NeuroSEDProductionDataError("compact label columns changed")
            schema = payload["schema_version"]
            if schema.shape != (1,) or str(schema[0]) != COMPACT_LABEL_SCHEMA:
                raise NeuroSEDProductionDataError("compact label schema changed")
            arrays = {key: payload[key].copy() for key in COMPACT_LABEL_COLUMNS}
    except (OSError, ValueError) as exc:
        raise NeuroSEDProductionDataError("compact label table cannot be reopened") from exc
    lengths = {int(value.shape[0]) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise NeuroSEDProductionDataError("compact label columns are misaligned")
    rows: list[dict[str, Any]] = []
    for index in range(next(iter(lengths))):
        status = str(arrays["status"][index])
        lower = float(arrays["lower_bound"][index])
        upper = float(arrays["upper_bound"][index])
        elapsed = float(arrays["elapsed_seconds"][index])
        if status not in ATTEMPT_STATUSES or not math.isfinite(elapsed) or elapsed < 0:
            raise NeuroSEDProductionDataError("compact label row status/time changed")
        if status == "SUCCESS":
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise NeuroSEDProductionDataError("compact successful interval changed")
        elif not (math.isnan(lower) and math.isnan(upper)):
            raise NeuroSEDProductionDataError("failed compact row carries GED bounds")
        row = {
            "pair_id": str(arrays["pair_id"][index]),
            "query_graph_id": str(arrays["query_graph_id"][index]),
            "target_graph_id": str(arrays["target_graph_id"][index]),
            "query_hash": str(arrays["query_hash"][index]),
            "target_hash": str(arrays["target_hash"][index]),
            "split": str(arrays["split"][index]),
            "lower_bound": lower if status == "SUCCESS" else None,
            "upper_bound": upper if status == "SUCCESS" else None,
            "exact_bound": bool(arrays["exact_bound"][index]),
            "label_contract": str(arrays["label_contract"][index]),
            "backend": str(arrays["backend"][index]),
            "backend_config_hash": str(arrays["backend_config_hash"][index]),
            "status": status,
            "elapsed_seconds": elapsed,
            "cache_key": str(arrays["cache_key"][index]),
            "cache_hit": bool(arrays["cache_hit"][index]),
            "error": str(arrays["error"][index]),
        }
        if (
            not row["pair_id"]
            or not row["query_graph_id"]
            or not row["target_graph_id"]
            or row["query_graph_id"] == row["target_graph_id"]
            or row["split"] not in ("train", "validation")
            or row["label_contract"] != LABEL_CONTRACT
        ):
            raise NeuroSEDProductionDataError("compact label pair roles changed")
        rows.append(row)
    return rows


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    destination = Path(path)
    data = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short JSON artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "ATTEMPT_STATUSES",
    "COMPACT_LABEL_COLUMNS",
    "COMPACT_LABEL_SCHEMA",
    "FixedBudgetPairInventory",
    "LABEL_CONTRACT",
    "NeuroSEDProductionDataError",
    "PreparedFixedBudgetPair",
    "atomic_json",
    "load_fixed_budget_pair_inventory",
    "load_json",
    "load_jsonl",
    "read_compact_npz",
    "sha256_file",
    "stable_sha256",
    "write_compact_npz",
]
