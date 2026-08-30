#!/usr/bin/env python3
"""Verify the Mut pair store on deterministic sklearn-float64 exact subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.external_memory_dbscan import (  # noqa: E402
    ExternalDBSCANContract,
    SKLEARN_FLOAT64_EXACT_MULTI_COMPONENT,
    _rss_bytes,
    fit_external_memory_dbscan,
)
from src.baselines.comrecgc.external_memory_recourse import (  # noqa: E402
    PAIR_STORE_SCHEMA,
    _validate_pair_store_manifest,
)
from src.baselines.comrecgc.contracts import (  # noqa: E402
    require_empty_output,
    sha256_file,
    write_json,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _indices(total: int, size: int, subset_index: int) -> np.ndarray:
    # Each subset spans the complete production order, while the cyclic offset
    # keeps the subsets distinct and deterministic without an RNG contract.
    base = (np.arange(size, dtype=np.int64) * total) // size
    values = np.sort((base + int(subset_index)) % total)
    if len(np.unique(values)) != size:
        raise ValueError("deterministic subset indices are not unique")
    return values


def run_subset_gate(
    *,
    pair_store_manifest: Path,
    output_dir: Path,
    subset_count: int,
    subset_size: int,
    expected_sklearn_version: str,
) -> dict[str, Any]:
    if subset_count <= 0 or subset_size < 3:
        raise ValueError("subset_count must be positive and subset_size at least 3")
    source_path = pair_store_manifest.expanduser().resolve(strict=True)
    source = _load_object(source_path)
    if (
        source.get("schema_version") != PAIR_STORE_SCHEMA
        or source.get("run_complete") is not True
        or not isinstance(source.get("scientific_identity"), dict)
        or source["scientific_identity"].get("dataset") != "mutagenicity"
    ):
        raise ValueError("not a completed Mutagenicity pair store")
    _validate_pair_store_manifest(source_path, source)
    vectors_path = Path(str(source["vectors_path"])).resolve(strict=True)
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    total = int(source["row_count"])
    if vectors.shape != (total, int(source["vector_dim"])):
        raise ValueError("Mutagenicity pair-store vector schema changed")
    if subset_size > total:
        raise ValueError("subset_size exceeds the production pair store")
    root = require_empty_output(output_dir, resume=False)
    rows: list[dict[str, Any]] = []
    from sklearn.cluster import DBSCAN

    for subset_index in range(subset_count):
        indices = _indices(total, subset_size, subset_index)
        values = np.asarray(vectors[indices], dtype=np.float32)
        subset_root = root / f"subset-{subset_index:02d}"
        subset_root.mkdir(parents=True)
        vector_path = subset_root / "vectors.npy"
        with vector_path.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        contract = ExternalDBSCANContract(
            eps=0.02,
            min_samples=3,
            query_block_size=4,
            checkpoint_interval_blocks=1,
            max_rss_bytes=_rss_bytes() + 4 * 1024**3,
            expected_sklearn_version=expected_sklearn_version,
            shortcut_mode=SKLEARN_FLOAT64_EXACT_MULTI_COMPONENT,
        )
        result = fit_external_memory_dbscan(
            vectors_path=vector_path,
            work_dir=subset_root / "dbscan",
            contract=contract,
        )
        actual = np.load(result.labels_path, allow_pickle=False)
        expected = DBSCAN(
            eps=0.02,
            min_samples=3,
            algorithm="brute",
            n_jobs=4,
        ).fit_predict(values.astype(np.float64))
        if not np.array_equal(actual, expected):
            raise RuntimeError("Mutagenicity subset labels differ from sklearn float64")
        reopened = fit_external_memory_dbscan(
            vectors_path=vector_path,
            work_dir=subset_root / "dbscan",
            contract=contract,
            resume=True,
        )
        if reopened.manifest_sha256 != result.manifest_sha256:
            raise RuntimeError("Mutagenicity subset terminal reload changed")
        rows.append(
            {
                "subset_index": subset_index,
                "row_count": subset_size,
                "indices_sha256": hashlib.sha256(
                    np.ascontiguousarray(indices).tobytes(order="C")
                ).hexdigest(),
                "vectors_sha256": sha256_file(vector_path),
                "dbscan_manifest_path": str(result.manifest_path),
                "dbscan_manifest_sha256": result.manifest_sha256,
                "cluster_count": result.cluster_count,
                "noise_count": result.noise_count,
                "labels_equal_sklearn_float64": True,
                "terminal_reload_equal": True,
            }
        )
    manifest = {
        "schema_version": "mutagenicity_comrecgc_exact_multicomponent_subset_v1",
        "status": "PASS",
        "run_complete": True,
        "dataset": "mutagenicity",
        "source_pair_store_manifest": str(source_path),
        "source_pair_store_manifest_sha256": sha256_file(source_path),
        "source_vectors_sha256": str(source["vectors_sha256"]),
        "source_row_count": total,
        "subset_count": subset_count,
        "subset_size": subset_size,
        "route": SKLEARN_FLOAT64_EXACT_MULTI_COMPONENT,
        "reference_semantics": "SKLEARN_FLOAT64",
        "single_component_shortcut_used": False,
        "failure_cap_used": False,
        "exact_worker_count": 4,
        "subsets": rows,
    }
    write_json(root / "run_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--pair-store-manifest", type=_absolute, required=True)
    parser.add_argument("--output-dir", type=_absolute, required=True)
    parser.add_argument("--subset-count", type=int, default=3)
    parser.add_argument("--subset-size", type=int, default=2048)
    parser.add_argument("--expected-sklearn-version", default="1.7.2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_subset_gate(
        pair_store_manifest=args.pair_store_manifest,
        output_dir=args.output_dir,
        subset_count=args.subset_count,
        subset_size=args.subset_size,
        expected_sklearn_version=args.expected_sklearn_version,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[MUT_COMRECGC_EXACT_MULTICOMPONENT_SUBSET_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
