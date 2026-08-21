#!/usr/bin/env python3
"""Build safe, reusable tensor graph caches for BACE or TasteMolNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_registry import get_dataset_spec, normalize_dataset_id  # noqa: E402
from src.data.molecular_graph_dataset import (  # noqa: E402
    MOLECULAR_GRAPH_CACHE_SCHEMA_VERSION,
    MolecularGraphDataset,
    load_molecular_graph_cache,
    save_molecular_graph_cache,
)
from src.data.molecular_graph_featurizer import (  # noqa: E402
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)
from src.utils.env import load_and_merge_config_files  # noqa: E402


CACHE_MANIFEST_SCHEMA_VERSION = "molecular_graph_cache_manifest_v1"
SPLIT_CANDIDATES = {
    "train": ("train.csv",),
    "validation": ("validation.csv", "val.csv", "valid.csv"),
    "calibration": ("calibration.csv",),
    "test": ("test.csv",),
}
EXPECTED_SPLITS = {
    "train": "train",
    "validation": "val",
    "calibration": "calibration",
    "test": "test",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_split_files(data_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    missing: dict[str, tuple[str, ...]] = {}
    for split_name, candidates in SPLIT_CANDIDATES.items():
        match = next(
            (data_dir / name for name in candidates if (data_dir / name).is_file()),
            None,
        )
        if match is None:
            missing[split_name] = candidates
        else:
            resolved[split_name] = match.resolve()
    if missing:
        details = ", ".join(
            f"{split}={list(names)}" for split, names in sorted(missing.items())
        )
        raise FileNotFoundError(
            f"All four molecular splits are required under {data_dir}: {details}"
        )
    return resolved


def _resolve_config_files(paths: Sequence[str | Path]) -> list[Path]:
    if not paths:
        raise ValueError("At least one --config file is required.")
    resolved = [Path(path).expanduser().resolve() for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Config files do not exist: {missing}")
    # Parse once so malformed configs fail before graph construction begins.
    load_and_merge_config_files(resolved)
    return resolved


def build_molecular_graph_cache(
    *,
    config_files: Sequence[str | Path],
    dataset: str,
    data_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build all four cache splits in a fresh directory and verify round trips."""

    dataset_id = normalize_dataset_id(dataset, allow_historical=False)
    if dataset_id not in {"bace", "tastemolnet"}:
        raise ValueError("Graph-cache construction is scoped to BACE and TasteMolNet.")
    spec = get_dataset_spec(dataset_id, allow_historical=False)
    configs = _resolve_config_files(config_files)
    source_root = Path(data_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Molecular split directory does not exist: {source_root}")
    split_files = _resolve_split_files(source_root)

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Molecular graph-cache output must be fresh: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    schema = default_molecular_feature_schema()
    featurizer = MolecularGraphFeaturizer(schema)
    split_manifest: dict[str, dict[str, Any]] = {}
    total_graph_count = 0

    try:
        for split_name in ("train", "validation", "calibration", "test"):
            source_csv = split_files[split_name]
            dataset_split = MolecularGraphDataset.from_csv(
                source_csv,
                num_classes=spec.num_classes,
                featurizer=featurizer,
                expected_split=EXPECTED_SPLITS[split_name],
            )
            cache_path = staging / f"{split_name}.pt"
            saved = save_molecular_graph_cache(
                dataset_split,
                cache_path,
                split_name=EXPECTED_SPLITS[split_name],
            )
            reloaded = load_molecular_graph_cache(
                cache_path,
                expected_num_classes=spec.num_classes,
                expected_source_sha256=str(saved["source_csv_sha256"]),
                expected_feature_schema=schema,
            )
            if reloaded.dataset_fingerprint != dataset_split.dataset_fingerprint:
                raise RuntimeError(
                    f"Graph-cache round trip changed {split_name} dataset identity."
                )
            graph_count = len(dataset_split)
            total_graph_count += graph_count
            split_manifest[split_name] = {
                "cache_file": cache_path.name,
                "cache_sha256": str(saved["sha256"]),
                "source_csv": str(source_csv),
                "source_csv_sha256": str(saved["source_csv_sha256"]),
                "graph_count": graph_count,
                "dataset_fingerprint": dataset_split.dataset_fingerprint,
                "num_classes": spec.num_classes,
                "feature_schema_sha256": schema.to_dict()["schema_sha256"],
                "safe_load_verified": True,
            }

        manifest: dict[str, Any] = {
            "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
            "cache_schema_version": MOLECULAR_GRAPH_CACHE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_id,
            "dataset_spec": spec.to_dict(),
            "num_classes": spec.num_classes,
            "feature_schema": schema.to_dict(),
            "feature_schema_sha256": schema.to_dict()["schema_sha256"],
            "split_order": ["train", "validation", "calibration", "test"],
            "splits": split_manifest,
            "total_graph_count": total_graph_count,
            "serialization_contract": {
                "payload_types": "plain_tensors_and_python_primitives",
                "custom_pickled_objects": False,
                "torch_load_weights_only": True,
                "fresh_output_required": True,
            },
            "config_files": [
                {"path": str(path), "sha256": _sha256_file(path)} for path in configs
            ],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest["output_dir"] = str(target)
    manifest["manifest_path"] = str(target / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Config file to validate and record; may be supplied more than once.",
    )
    parser.add_argument("--dataset", required=True, help="bace or tastemolnet")
    parser.add_argument("--data-dir", required=True, help="Directory with four split CSVs")
    parser.add_argument("--output-dir", required=True, help="Fresh cache directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_molecular_graph_cache(
        config_files=args.config,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "graph_count": manifest["total_graph_count"],
                "manifest": manifest["manifest_path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("[MOLECULAR_GRAPH_CACHE_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
