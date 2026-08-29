#!/usr/bin/env python3
"""Derive the canonical Taste NeuroSED feature schema from train/validation."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tastemolnet_neurosed_pairs import (  # noqa: E402
    TasteNeuroSEDPairError,
    TasteSplitRow,
    derive_feature_schema,
    read_taste_split_rows,
)


HPC_CONFIG = PROJECT_ROOT / "configs/hpc.yaml"
HPC_CONFIG_SHA256 = "7d3fb9e5c42101ae4a2ee5c43f400710fad6227014c573b1550872c7005e0110"
FAIL_CLOSED_OVERRIDE = "inference.fallback_to_heuristic=false"
FEATURE_SCHEMA_VERSION = "tastemolnet_gcf_neurosed_feature_schema_v1"
FEATURE_SCHEMA_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "node_feature_semantics",
        "feature_atomic_numbers",
        "input_dim",
        "explicit_h_nodes",
        "native_adjacency_semantics",
        "edge_features_used",
        "validation_unseen_atomic_numbers",
        "train_derived_only",
        "maximum_train_or_validation_nodes",
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lower_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise TasteNeuroSEDPairError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _validate_config(values: Sequence[str]) -> Path:
    if len(values) != 1:
        raise TasteNeuroSEDPairError("exactly one --config configs/hpc.yaml is required")
    path = Path(os.path.abspath(Path(values[0]).expanduser()))
    if path != HPC_CONFIG:
        raise TasteNeuroSEDPairError("--config must be this checkout's configs/hpc.yaml")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise TasteNeuroSEDPairError("--config may not contain symlink components")
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise TasteNeuroSEDPairError("--config must be one physical single-link file")
    if _sha256(path.read_bytes()) != HPC_CONFIG_SHA256:
        raise TasteNeuroSEDPairError("tracked configs/hpc.yaml SHA-256 changed")
    return path


def _validate_split(
    rows: Sequence[TasteSplitRow],
    evidence: Mapping[str, Any],
    *,
    role: str,
    expected_sha256: str,
) -> None:
    if (
        not rows
        or evidence.get("split") != role
        or evidence.get("source_csv_sha256") != expected_sha256
        or evidence.get("row_count") != len(rows)
        or evidence.get("all_rows_declared_expected_split") is not True
        or evidence.get("labels_opened_but_not_consumed") is not True
        or any(row.split != role for row in rows)
    ):
        raise TasteNeuroSEDPairError(f"{role} split role/SHA evidence changed")


def _validate_feature_schema(schema: Mapping[str, Any]) -> None:
    vocabulary = schema.get("feature_atomic_numbers")
    if (
        set(schema) != FEATURE_SCHEMA_FIELDS
        or schema.get("schema_version") != FEATURE_SCHEMA_VERSION
        or schema.get("dataset") != "tastemolnet"
        or schema.get("node_feature_semantics") != "one_hot_atomic_number"
        or type(vocabulary) is not list
        or not vocabulary
        or any(type(value) is not int or value <= 0 for value in vocabulary)
        or vocabulary != sorted(set(vocabulary))
        or schema.get("input_dim") != len(vocabulary)
        or schema.get("explicit_h_nodes") is not True
        or schema.get("native_adjacency_semantics")
        != "binary_connectivity_directed_both_ways"
        or schema.get("edge_features_used") is not False
        or schema.get("validation_unseen_atomic_numbers") != []
        or schema.get("train_derived_only") is not True
        or type(schema.get("maximum_train_or_validation_nodes")) is not int
        or schema["maximum_train_or_validation_nodes"] <= 0
    ):
        raise TasteNeuroSEDPairError("derived Taste NeuroSED feature schema changed")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _normalized_absolute(path: Path, *, label: str) -> Path:
    normalized = Path(os.path.abspath(path.expanduser()))
    if not path.is_absolute() or normalized != path:
        raise TasteNeuroSEDPairError(f"{label} must be normalized absolute")
    return normalized


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise TasteNeuroSEDPairError(f"{label} contains a symlink component")


def _atomic_write_new(path: Path, data: bytes) -> None:
    destination = _normalized_absolute(path, label="feature-schema output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, label="feature-schema output parent")
    parent_info = os.lstat(destination.parent)
    if not stat.S_ISDIR(parent_info.st_mode):
        raise TasteNeuroSEDPairError("feature-schema output parent is not a directory")
    if destination.exists() or destination.is_symlink():
        raise TasteNeuroSEDPairError("feature-schema output must be fresh")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise TasteNeuroSEDPairError("feature-schema output must be fresh") from exc
            raise
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
            os.unlink(temporary)
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

    info = os.lstat(destination)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise TasteNeuroSEDPairError("published feature schema is not one physical file")
    if destination.read_bytes() != data:
        raise TasteNeuroSEDPairError("published feature schema failed exact reopen")


def build_feature_schema(
    *,
    train_csv: Path,
    expected_train_sha256: str,
    validation_csv: Path,
    expected_validation_sha256: str,
    output_json: Path,
) -> dict[str, Any]:
    """Load exactly train/validation, derive, and atomically publish one schema."""

    train_sha256 = _lower_sha256(expected_train_sha256, label="expected train SHA-256")
    validation_sha256 = _lower_sha256(
        expected_validation_sha256, label="expected validation SHA-256"
    )
    train_rows, train_evidence = read_taste_split_rows(
        train_csv, expected_split="train"
    )
    validation_rows, validation_evidence = read_taste_split_rows(
        validation_csv, expected_split="validation"
    )
    _validate_split(
        train_rows, train_evidence, role="train", expected_sha256=train_sha256
    )
    _validate_split(
        validation_rows,
        validation_evidence,
        role="validation",
        expected_sha256=validation_sha256,
    )
    if {row.molecule_id for row in train_rows}.intersection(
        row.molecule_id for row in validation_rows
    ):
        raise TasteNeuroSEDPairError("train/validation molecule IDs overlap")

    schema = derive_feature_schema(train_rows, validation_rows)
    _validate_feature_schema(schema)
    schema_bytes = _canonical_json_bytes(schema)
    _atomic_write_new(output_json, schema_bytes)

    return {
        "schema_version": "tastemolnet_gcf_neurosed_feature_schema_producer_receipt_v1",
        "status": "BUILT",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": _sha256(schema_bytes),
        "feature_schema_output": str(output_json),
        "feature_schema_atomic_no_replace": True,
        "opened_payload_splits": ["train", "validation"],
        "split_evidence": {
            "train": {
                "role": "train",
                "source_csv_sha256": train_sha256,
                "row_count": len(train_rows),
                "all_rows_declared_expected_split": True,
            },
            "validation": {
                "role": "validation",
                "source_csv_sha256": validation_sha256,
                "row_count": len(validation_rows),
                "all_rows_declared_expected_split": True,
            },
        },
        "train_validation_id_intersection_empty": True,
        "labels_opened_but_not_consumed": True,
        "labels_used": False,
        "classifier_used": False,
        "calibration_payload_opened": False,
        "test_payload_opened": False,
        "calibration_graph_ids_observed": False,
        "test_graph_ids_observed": False,
        "calibration_smiles_observed": False,
        "test_smiles_observed": False,
        "forbidden_payload_splits_opened": [],
        "no_calibration_or_test_payload_access_evidence": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        _validate_config(args.config)
        if args.set != [FAIL_CLOSED_OVERRIDE]:
            raise TasteNeuroSEDPairError(
                "--set must be exactly inference.fallback_to_heuristic=false"
            )
        receipt = build_feature_schema(
            train_csv=args.train_csv,
            expected_train_sha256=args.expected_train_sha256,
            validation_csv=args.validation_csv,
            expected_validation_sha256=args.expected_validation_sha256,
            output_json=args.output_json,
        )
        receipt["config_sha256"] = HPC_CONFIG_SHA256
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except (TasteNeuroSEDPairError, OSError, ValueError) as exc:
        print(f"TASTE_NEUROSED_FEATURE_SCHEMA_BLOCKED: {exc}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
