#!/usr/bin/env python3
"""Train and freeze one task-specific molecular GNN classifier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_registry import get_dataset_spec, normalize_dataset_id  # noqa: E402
from src.data.molecular_graph_dataset import (  # noqa: E402
    MolecularGraphData,
    MolecularGraphDataset,
    build_molecular_data_loader,
    load_molecular_graph_cache,
)
from src.data.molecular_graph_featurizer import (  # noqa: E402
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)
from src.chem.hard_deletion import enumerate_connected_hard_deletions  # noqa: E402
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig  # noqa: E402
from src.oracles.gnn_oracle import (  # noqa: E402
    GNNOracle,
    classification_metrics,
    fit_temperature_scaling,
    save_gnn_checkpoint_bundle,
    sha256_file,
    update_checkpoint_sha256sums,
    verify_checkpoint_bundle,
)
from src.train.molecular_gnn_resume import (  # noqa: E402
    FinalizationWorkspace,
    MolecularGNNResumeError,
    MolecularGNNResumeStore,
    OutputParentAuthority,
    assert_no_symlink_components,
    canonical_sha256,
    paths_overlap,
)
from src.utils.env import (  # noqa: E402
    apply_dotlist_overrides,
    load_and_merge_config_files,
)
from src.utils.tastemolnet_research_policy import (  # noqa: E402
    TasteLocalDataAuthority,
    TastePolicyReceipt,
    TasteResearchPolicy,
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    validate_tastemolnet_local_authority,
    validate_tastemolnet_policy_receipt,
)


def _physical_file_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "nlink": int(info.st_nlink),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _open_frozen_input_file(
    path: Path, *, expected_sha256: str, label: str
) -> tuple[int, dict[str, Any], bytes]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _physical_file_identity(before) != _physical_file_identity(named)
        ):
            raise MolecularGNNResumeError(
                f"{label} must be one named physical regular file"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        observed_sha256 = digest.hexdigest()
        if (
            _physical_file_identity(before) != _physical_file_identity(after)
            or _physical_file_identity(after)
            != _physical_file_identity(named_after)
            or observed_sha256 != expected_sha256
        ):
            raise MolecularGNNResumeError(
                f"{label} inode/content/hash differs from its frozen manifest"
            )
        return descriptor, {
            "path": str(path),
            "identity": _physical_file_identity(after),
            "sha256": observed_sha256,
        }, b"".join(chunks)
    except BaseException:
        os.close(descriptor)
        raise


def _verify_frozen_input_file(
    path: Path,
    descriptor: int,
    evidence: Mapping[str, Any],
    *,
    label: str,
) -> None:
    before = os.fstat(descriptor)
    named_before = os.stat(path, follow_symlinks=False)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        digest.update(chunk)
    after = os.fstat(descriptor)
    named_after = os.stat(path, follow_symlinks=False)
    if (
        evidence.get("path") != str(path)
        or evidence.get("identity") != _physical_file_identity(before)
        or _physical_file_identity(before) != _physical_file_identity(after)
        or _physical_file_identity(after)
        != _physical_file_identity(named_before)
        or _physical_file_identity(after)
        != _physical_file_identity(named_after)
        or digest.hexdigest() != evidence.get("sha256")
    ):
        raise MolecularGNNResumeError(
            f"{label} drifted across the graph-cache descriptor load window"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Merge config files in order; pass hpc.yaml before a GNN config.",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--train-csv")
    parser.add_argument("--validation-csv")
    parser.add_argument("--calibration-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--device")
    parser.add_argument("--backbone")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--source-label", type=int)
    parser.add_argument("--label-map-json")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument(
        "--graph-cache-root",
        help=(
            "Optional prebuilt molecular_graph_tensor_cache_v1 root. "
            "TasteMolNet full training requires this read-only cache."
        ),
    )
    parser.add_argument(
        "--taste-policy-file",
        help="Active scoped TasteMolNet research/no-redistribution policy.",
    )
    parser.add_argument("--taste-policy-sha256")
    parser.add_argument(
        "--taste-policy-receipt",
        help="Typed receipt from audit_tastemolnet_research_policy.py.",
    )
    parser.add_argument(
        "--taste-prepared-root",
        help="Frozen private TasteMolNet prepared root; never copied to output.",
    )
    parser.add_argument(
        "--training-state-dir",
        help=(
            "Separate persistent epoch-checkpoint root. TasteMolNet full training "
            "requires this root so its immutable output remains fresh until terminal."
        ),
    )
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume from the latest hash-bound epoch checkpoint in --training-state-dir.",
    )
    parser.add_argument(
        "--resume-published-output-receipt",
        help="Controller-issued completion-only adoption for a published Taste bundle.",
    )
    return parser


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL runtime dependency.
        raise RuntimeError("train_molecular_gnn.py requires PyTorch.") from exc
    return torch


def _config(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(path) for path in args.config]
    if not paths:
        paths = [PROJECT_ROOT / "configs" / "gnn" / "gine.yaml"]
    config = load_and_merge_config_files(paths)
    return apply_dotlist_overrides(config, args.set)


def _nested(config: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    return value.get(key, default) if isinstance(value, Mapping) else default


def _first_existing(root: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of the required split files exist: {names} under {root}")


def _resolve_split_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Molecular GNN data directory does not exist: {root}")

    def explicit_or(names: Sequence[str], explicit: str | None) -> Path:
        if explicit:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        return _first_existing(root, names)

    return {
        "train": explicit_or(("train.csv",), args.train_csv),
        "validation": explicit_or(("validation.csv", "val.csv", "valid.csv"), args.validation_csv),
        "calibration": explicit_or(("calibration.csv",), args.calibration_csv),
        "test": explicit_or(("test.csv",), args.test_csv),
    }


def _load_fit_datasets(
    *,
    dataset_id: str,
    profile: str,
    split_paths: Mapping[str, Path],
    graph_cache_root: str | None,
    num_classes: int,
    featurizer: MolecularGraphFeaturizer,
    train_limit: int | None,
    validation_limit: int | None,
    stratified_limit: bool,
    graph_cache_manifest_sha256: str | None = None,
) -> tuple[MolecularGraphDataset, MolecularGraphDataset, dict[str, Any]]:
    """Load only train/validation, preferring a frozen safe tensor cache."""

    if dataset_id == "tastemolnet" and profile == "full" and not graph_cache_root:
        raise ValueError("TasteMolNet full training requires --graph-cache-root.")
    if not graph_cache_root:
        train = MolecularGraphDataset.from_csv(
            split_paths["train"],
            num_classes=num_classes,
            featurizer=featurizer,
            expected_split="train",
            limit=train_limit,
            stratified_limit=stratified_limit,
        )
        validation = MolecularGraphDataset.from_csv(
            split_paths["validation"],
            num_classes=num_classes,
            featurizer=featurizer,
            expected_split="val",
            limit=validation_limit,
            stratified_limit=stratified_limit,
        )
        return train, validation, {
            "schema_version": "molecular_graph_training_input_v1",
            "mode": "csv_featurized",
            "graph_cache_used": False,
            "loaded_splits": ["train", "validation"],
            "calibration_loaded": False,
            "test_loaded": False,
        }

    if train_limit is not None or validation_limit is not None:
        raise ValueError("A frozen full graph cache cannot be combined with row limits.")
    unresolved_cache_root = Path(graph_cache_root).expanduser()
    unresolved_cache_root = Path(os.path.abspath(unresolved_cache_root))
    current = Path(unresolved_cache_root.anchor)
    for part in unresolved_cache_root.parts[1:]:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("graph-cache path may not contain symlink components")
    cache_root = unresolved_cache_root.resolve(strict=True)
    if not stat.S_ISDIR(os.lstat(cache_root).st_mode):
        raise ValueError("graph-cache root must be one physical directory")
    cache_files = {
        "train": cache_root / "train.pt",
        "validation": cache_root / "validation.pt",
    }
    manifest_path = cache_root / "manifest.json"
    if dataset_id == "tastemolnet" and profile == "full":
        if not isinstance(graph_cache_manifest_sha256, str):
            raise MolecularGNNResumeError(
                "Taste graph-cache manifest SHA authority is absent"
            )
        manifest_fd, manifest_evidence, manifest_bytes = _open_frozen_input_file(
            manifest_path,
            expected_sha256=graph_cache_manifest_sha256,
            label="Taste graph-cache manifest",
        )
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            os.close(manifest_fd)
            raise MolecularGNNResumeError(
                "Taste graph-cache manifest is not valid JSON"
            ) from exc
        raw_splits = manifest.get("splits") if isinstance(manifest, Mapping) else None
        if (
            not isinstance(raw_splits, Mapping)
            or manifest.get("schema_version")
            != "molecular_graph_cache_manifest_v1"
        ):
            os.close(manifest_fd)
            raise MolecularGNNResumeError(
                "Taste graph-cache manifest schema changed before load"
            )
    else:
        manifest_fd = None
        manifest_evidence = None
        raw_splits = None
    source_hashes = {
        split: sha256_file(split_paths[split]) for split in cache_files
    }
    held_cache: dict[str, tuple[int, dict[str, Any]]] = {}
    try:
        for split, path in cache_files.items():
            if raw_splits is not None:
                entry = raw_splits.get(split)
                if (
                    not isinstance(entry, Mapping)
                    or entry.get("cache_file") != path.name
                    or entry.get("source_csv_sha256") != source_hashes[split]
                    or not isinstance(entry.get("cache_sha256"), str)
                ):
                    raise MolecularGNNResumeError(
                        f"Taste graph-cache manifest binding changed for {split}"
                    )
                expected_cache_sha = str(entry["cache_sha256"])
            else:
                expected_cache_sha = sha256_file(path)
            descriptor, evidence, _ = _open_frozen_input_file(
                path,
                expected_sha256=expected_cache_sha,
                label=f"{split} graph cache",
            )
            held_cache[split] = (descriptor, evidence)
        if manifest_fd is not None and manifest_evidence is not None:
            _verify_frozen_input_file(
                manifest_path,
                manifest_fd,
                manifest_evidence,
                label="Taste graph-cache manifest before descriptor load",
            )
        with os.fdopen(os.dup(held_cache["train"][0]), "rb") as train_stream:
            train = load_molecular_graph_cache(
                train_stream,
                expected_num_classes=num_classes,
                expected_source_sha256=source_hashes["train"],
                expected_feature_schema=featurizer.schema,
            )
        _verify_frozen_input_file(
            cache_files["train"],
            held_cache["train"][0],
            held_cache["train"][1],
            label="train graph cache",
        )
        with os.fdopen(os.dup(held_cache["validation"][0]), "rb") as validation_stream:
            validation = load_molecular_graph_cache(
                validation_stream,
                expected_num_classes=num_classes,
                expected_source_sha256=source_hashes["validation"],
                expected_feature_schema=featurizer.schema,
            )
        for split, path in cache_files.items():
            _verify_frozen_input_file(
                path,
                held_cache[split][0],
                held_cache[split][1],
                label=f"{split} graph cache after descriptor load",
            )
        if manifest_fd is not None and manifest_evidence is not None:
            _verify_frozen_input_file(
                manifest_path,
                manifest_fd,
                manifest_evidence,
                label="Taste graph-cache manifest after descriptor load",
            )
    finally:
        for descriptor, _ in held_cache.values():
            os.close(descriptor)
        if manifest_fd is not None:
            os.close(manifest_fd)
    cache_contract = {
        "schema_version": "molecular_graph_training_cache_contract_v1",
        "manifest": manifest_evidence,
        "splits": {
            split: {
                **evidence,
                "source_csv_sha256": source_hashes[split],
            }
            for split, (_, evidence) in held_cache.items()
        },
        "loaded_splits": ["train", "validation"],
        "calibration_loaded": False,
        "test_loaded": False,
    }
    return train, validation, {
        "schema_version": "molecular_graph_training_input_v1",
        "mode": "frozen_safe_tensor_cache",
        "graph_cache_used": True,
        "graph_cache_root": str(cache_root),
        "loaded_splits": ["train", "validation"],
        "calibration_loaded": False,
        "test_loaded": False,
        "cache_contract": cache_contract,
        "cache_files": {
            split: {
                "path": str(path),
                "sha256": held_cache[split][1]["sha256"],
                "source_csv_sha256": source_hashes[split],
            }
            for split, path in cache_files.items()
        },
    }


def _taste_runtime_authority(
    args: argparse.Namespace,
    *,
    dataset_id: str,
    profile: str,
    split_paths: Mapping[str, Path],
) -> tuple[TasteResearchPolicy, TasteLocalDataAuthority, TastePolicyReceipt] | None:
    """Open the scoped Taste policy and frozen private inputs without loading rows."""

    values = (
        args.taste_policy_file,
        args.taste_policy_sha256,
        args.taste_policy_receipt,
        args.taste_prepared_root,
    )
    if dataset_id != "tastemolnet" or profile != "full":
        if any(value is not None for value in values):
            raise ValueError(
                "Taste policy authority flags are only valid for TasteMolNet full training."
            )
        return None
    if not all(values) or not args.graph_cache_root:
        raise ValueError(
            "TasteMolNet full training requires policy file/SHA/receipt, prepared root, "
            "and graph cache root."
        )
    policy = load_tastemolnet_research_policy(
        args.taste_policy_file,
        expected_file_sha256=args.taste_policy_sha256,
    )
    policy.require_main_route()
    authority = validate_tastemolnet_local_authority(
        policy,
        prepared_root=args.taste_prepared_root,
        graph_cache_root=args.graph_cache_root,
    )
    receipt = validate_tastemolnet_policy_receipt(
        args.taste_policy_receipt,
        policy=policy,
        authority=authority,
        require_active=True,
        require_policy_version=2,
    )
    expected_split_root = authority.prepared_root / "splits"
    for split, path in split_paths.items():
        expected = (expected_split_root / f"{split}.csv").resolve(strict=True)
        if path.resolve(strict=True) != expected:
            raise TasteResearchPolicyError(
                f"Taste {split} input escaped the frozen prepared authority"
            )
    return policy, authority, receipt


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a fresh JSON artifact without replacing an existing authority."""

    data = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_json_before_publication(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a private staging JSON before the bundle is published."""

    data = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_last_training_checkpoint(
    *, training_state_root: Path, bundle_dir: Path
) -> dict[str, Any]:
    """Copy the authenticated latest epoch state into the fresh final bundle."""

    latest_path = training_state_root / "latest_checkpoint.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if not isinstance(latest, dict):
        raise MolecularGNNResumeError("latest checkpoint authority is not an object")
    relative = latest.get("checkpoint_file")
    expected_sha256 = latest.get("checkpoint_sha256")
    completed_epoch = latest.get("completed_epoch")
    if (
        not isinstance(relative, str)
        or Path(relative).name != relative
        or not relative.startswith("checkpoint-")
        or not relative.endswith(".pt")
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or type(completed_epoch) is not int
        or completed_epoch < 1
    ):
        raise MolecularGNNResumeError("latest checkpoint authority is malformed")
    source = training_state_root / relative
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    target = bundle_dir / "last.pt"
    temporary = bundle_dir / ".last.pt.tmp"
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MolecularGNNResumeError("latest checkpoint is not one regular file")
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as input_handle:
            with temporary.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
                output_handle.flush()
                os.fsync(output_handle.fileno())
        after = os.fstat(descriptor)
        named = os.stat(source, follow_symlinks=False)
        identity = lambda value: (  # noqa: E731
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(named):
            raise MolecularGNNResumeError("latest checkpoint changed while copied")
        if sha256_file(source) != expected_sha256 or sha256_file(temporary) != expected_sha256:
            raise MolecularGNNResumeError("latest checkpoint SHA-256 changed")
        os.link(temporary, target)
        directory = os.open(bundle_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": "tastemolnet_last_training_checkpoint_v1",
        "checkpoint_file": "last.pt",
        "checkpoint_sha256": expected_sha256,
        "source_checkpoint_file": relative,
        "source_checkpoint_sha256": expected_sha256,
        "completed_epoch": completed_epoch,
        "same_bytes_as_latest_epoch_checkpoint": True,
        "training_state_root": str(training_state_root),
    }
    _write_new_json(bundle_dir / "last_checkpoint.json", receipt)
    return receipt


def _training_resume_contract(
    *,
    args: argparse.Namespace,
    dataset_id: str,
    profile: str,
    output_dir: Path,
    split_paths: Mapping[str, Path],
    taste_runtime: tuple[
        TasteResearchPolicy, TasteLocalDataAuthority, TastePolicyReceipt
    ]
    | None,
    model_config: MolecularGNNConfig,
    feature_schema: Mapping[str, Any],
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
    class_weighted: bool,
    weighted_sampler: bool,
    selection_metric: str,
    selection_tiebreak_metric: str | None,
    clip_norm: float,
    config: Mapping[str, Any],
    git_state: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    training_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable identity for an epoch-resumable training campaign."""

    contract: dict[str, Any] = {
        "schema_version": "molecular_gnn_training_resume_contract_v1",
        "dataset": dataset_id,
        "profile": profile,
        "output_dir": str(output_dir),
        "source_identity": dict(git_state),
        "configuration": {
            "merged_canonical": copy.deepcopy(dict(config)),
            "merged_canonical_sha256": canonical_sha256(config),
            "config_files": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in _resolved_config_paths(args)
            ],
            "dotlist_overrides": list(args.set),
            "cli_overrides": _resume_relevant_cli(args),
        },
        "runtime_identity": dict(runtime_identity),
        "training_input": copy.deepcopy(dict(training_input)),
        "splits": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(split_paths.items())
        },
        "graph_cache_root": (
            None
            if not args.graph_cache_root
            else str(Path(args.graph_cache_root).expanduser().resolve(strict=True))
        ),
        "model_config": model_config.to_dict(),
        "feature_schema": dict(feature_schema),
        "training": {
            "max_epochs": int(max_epochs),
            "early_stopping_patience": int(patience),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "num_workers": int(num_workers),
            "class_weighted_loss": bool(class_weighted),
            "weighted_sampler": bool(weighted_sampler),
            "selection_metric": selection_metric,
            "selection_tiebreak_metric": selection_tiebreak_metric,
            "gradient_clip_norm": float(clip_norm),
        },
    }
    if taste_runtime is not None:
        policy, authority, receipt = taste_runtime
        contract["tastemolnet_scoped_authority"] = {
            "policy": policy.evidence(),
            "private_data": authority.evidence(),
            "policy_receipt": {
                "path": str(receipt.path),
                "sha256": receipt.sha256,
            },
        }
    return contract


def _resolved_config_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path).expanduser().resolve(strict=True) for path in args.config]
    if not paths:
        paths = [(PROJECT_ROOT / "configs" / "gnn" / "gine.yaml").resolve(strict=True)]
    return paths


def _resume_relevant_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze every scientific CLI value while excluding only resume intent."""

    excluded = {
        "resume_training",
        "training_state_dir",
        "resume_published_output_receipt",
    }
    return {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }


def _runtime_identity(*, torch: Any, device: str, taste_full: bool) -> dict[str, Any]:
    def module_identity(name: str) -> dict[str, Any] | None:
        try:
            module = __import__(name)
        except ImportError:
            return None
        raw_path = getattr(module, "__file__", None)
        path = (
            None
            if raw_path is None
            else Path(raw_path).expanduser().resolve(strict=True)
        )
        return {
            "version": (
                None
                if getattr(module, "__version__", None) is None
                else str(module.__version__)
            ),
            "module_path": None if path is None else str(path),
            "module_sha256": None if path is None else sha256_file(path),
        }

    environment_keys = (
        "CUDA_VISIBLE_DEVICES",
        "AUTODL_PHYSICAL_GPU_INDEX",
        "AUTODL_PHYSICAL_GPU_UUID",
        "AUTODL_MAX_GPUS",
        "AUTODL_PYTHON",
        "RUN_TASTEMOLNET",
        "TASTE_RESEARCH_COMPUTE_ALLOWED",
        "TASTE_PAPER_RESULTS_ALLOWED",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED",
        "TASTE_UPSTREAM_LICENSE_STATUS",
        "TASTEMOLNET_POLICY_FILE",
        "TASTEMOLNET_POLICY_SHA256",
        "TASTEMOLNET_POLICY_RECEIPT",
        "TASTEMOLNET_PREPARED_ROOT",
        "TASTEMOLNET_SPLIT_ROOT",
        "TASTEMOLNET_GRAPH_CACHE_ROOT",
        "TASTEMOLNET_GNN_FULL_OUTPUT",
        "TASTEMOLNET_GNN_TRAINING_STATE_ROOT",
        "TASTEMOLNET_GINE_CONTROLLER_CID",
        "TASTEMOLNET_GINE_CONTROLLER_ROOT",
        "TASTEMOLNET_GPU_INDEX",
        "TASTEMOLNET_STORAGE_RESERVATION_GB",
        "MIN_PERSISTENT_FREE_GB",
        "MIN_FREE_AFTER_RESERVATIONS_GB",
        "TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT",
        "TASTEMOLNET_RESOURCE_WAIT_DEADLINE_EPOCH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    )
    environment_manifest = {
        key: os.environ.get(key) for key in environment_keys
    }
    physical_index = os.environ.get("AUTODL_PHYSICAL_GPU_INDEX")
    physical_uuid = os.environ.get("AUTODL_PHYSICAL_GPU_UUID")
    if taste_full:
        if device != "cuda:0":
            raise MolecularGNNResumeError(
                "TasteMolNet full training requires logical cuda:0 behind the GPU1 mask"
            )
        if physical_index != "1" or not physical_uuid or not physical_uuid.startswith("GPU-"):
            raise MolecularGNNResumeError(
                "TasteMolNet full training requires physical GPU1 UUID runtime authority"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise MolecularGNNResumeError(
                "TasteMolNet full training requires exactly one masked CUDA device"
            )
    numpy_identity = module_identity("numpy")
    rdkit_identity = module_identity("rdkit")
    pyg_identity = module_identity("torch_geometric")
    cudnn_version = (
        None
        if not hasattr(torch, "backends")
        or not hasattr(torch.backends, "cudnn")
        else torch.backends.cudnn.version()
    )
    driver_identity: dict[str, Any] | None = None
    if torch.cuda.is_available():
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is not None:
            observed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            versions = sorted(
                {
                    line.strip()
                    for line in observed.stdout.splitlines()
                    if line.strip()
                }
            )
            if observed.returncode == 0 and len(versions) == 1:
                nvidia_smi_path = Path(nvidia_smi).resolve(strict=True)
                driver_identity = {
                    "version": versions[0],
                    "nvidia_smi_path": str(nvidia_smi_path),
                    "nvidia_smi_sha256": sha256_file(nvidia_smi_path),
                }
    payload: dict[str, Any] = {
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "logical_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": physical_index,
        "physical_gpu_uuid": physical_uuid,
        "numpy": numpy_identity,
        "rdkit": rdkit_identity,
        "torch_geometric": pyg_identity,
        "cudnn_version": None if cudnn_version is None else int(cudnn_version),
        "cuda_driver": driver_identity,
        "environment_manifest": environment_manifest,
        "environment_manifest_sha256": canonical_sha256(environment_manifest),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        payload["visible_device"] = {
            "name": str(properties.name),
            "total_memory": int(properties.total_memory),
            "major": int(properties.major),
            "minor": int(properties.minor),
        }
    if taste_full and (
        numpy_identity is None
        or not numpy_identity.get("version")
        or rdkit_identity is None
        or not rdkit_identity.get("version")
        or pyg_identity is None
        or not pyg_identity.get("version")
        or cudnn_version is None
        or driver_identity is None
    ):
        raise MolecularGNNResumeError(
            "TasteMolNet resume runtime closure requires NumPy/RDKit/PyG/cuDNN/driver identities"
        )
    return payload


def _bundle_output_identity(output_dir: Path) -> dict[str, Any]:
    audit = verify_checkpoint_bundle(output_dir)
    result = {
        "model_sha256": sha256_file(output_dir / "model.pt"),
        "model_card_sha256": sha256_file(output_dir / "model_card.json"),
        "sha256s_sha256": sha256_file(output_dir / "sha256sums.txt"),
        "checkpoint_id": audit["model_card"]["checkpoint_id"],
        "training_resume_contract_sha256": audit["model_card"].get(
            "training_resume_contract_sha256"
        ),
    }
    contract_sha = result["training_resume_contract_sha256"]
    if isinstance(contract_sha, str) and len(contract_sha) == 64:
        prefix = f".{output_dir.name}.finalizing-{contract_sha}"
        claim = output_dir.parent / f"{prefix}.claim.json"
        ready = output_dir.parent / f"{prefix}.complete.json"
        result["finalization_claim_sha256"] = sha256_file(claim)
        result["finalization_completion_sha256"] = sha256_file(ready)
    return result


def _publish_staged_bundle(workspace: FinalizationWorkspace) -> Path:
    """Publish a verified deterministic stage through atomic no-replace."""

    verify_checkpoint_bundle(workspace.staging)
    workspace.mark_ready()
    workspace.publish()
    verify_checkpoint_bundle(workspace.output_dir)
    return workspace.output_dir


def _set_seed(seed: int, *, exact_cuda: bool = False) -> None:
    torch = _require_torch()
    if exact_cuda:
        required_environment = {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "7",
            "NVIDIA_TF32_OVERRIDE": "0",
            "CUDNN_DETERMINISTIC": "1",
        }
        if seed != 7 or any(
            os.environ.get(key) != value
            for key, value in required_environment.items()
        ):
            raise MolecularGNNResumeError(
                "Taste CUDA parity requires seed-7 and the frozen deterministic environment"
            )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not hasattr(torch, "use_deterministic_algorithms"):
        raise MolecularGNNResumeError(
            "PyTorch deterministic-algorithm enforcement is unavailable"
        )
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "are_deterministic_algorithms_enabled") and not (
        torch.are_deterministic_algorithms_enabled()
    ):
        raise MolecularGNNResumeError(
            "PyTorch deterministic algorithms did not become mandatory"
        )
    if hasattr(torch, "set_deterministic_debug_mode"):
        torch.set_deterministic_debug_mode("error")
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "allow_tf32")
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
    if exact_cuda:
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
        matmul = getattr(cuda_backend, "matmul", None)
        debug_mode = (
            torch.get_deterministic_debug_mode()
            if hasattr(torch, "get_deterministic_debug_mode")
            else None
        )
        if (
            cudnn is None
            or getattr(cudnn, "deterministic", None) is not True
            or getattr(cudnn, "benchmark", None) is not False
            or getattr(cudnn, "allow_tf32", None) is not False
            or matmul is None
            or getattr(matmul, "allow_tf32", None) is not False
            or (debug_mode is not None and debug_mode != 2)
        ):
            raise MolecularGNNResumeError(
                "Taste CUDA parity could not enforce cuDNN/TF32 error-mode determinism"
            )


def _resolve_device(requested: str | None, config: Mapping[str, Any]) -> str:
    torch = _require_torch()
    value = requested or str(_nested(config, "runtime", "device", "auto"))
    if value in {"auto", "cuda"}:
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return str(value)


def _class_weights(labels: Sequence[int], num_classes: int) -> list[float]:
    counts = Counter(int(label) for label in labels)
    if set(counts) != set(range(num_classes)):
        raise ValueError(
            f"Training split must contain every class: counts={dict(sorted(counts.items()))}"
        )
    total = len(labels)
    return [total / (num_classes * counts[label]) for label in range(num_classes)]


def _prediction_rows(
    dataset: MolecularGraphDataset,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, row_logits, row_probabilities in zip(
        dataset.records, logits, probabilities, strict=True
    ):
        rows.append(
            {
                "molecule_id": record.molecule_id,
                "smiles": record.graph.canonical_smiles,
                "split": record.split,
                "label": int(record.label),
                "predicted_label": int(row_probabilities.argmax()),
                "logits": json.dumps([float(value) for value in row_logits]),
                "probabilities": json.dumps(
                    [float(value) for value in row_probabilities]
                ),
                "source_graph_hash": record.graph.graph_sha256,
            }
        )
    return rows


def _numeric_values(payload: Any) -> list[float]:
    values: list[float] = []
    if isinstance(payload, Mapping):
        for value in payload.values():
            values.extend(_numeric_values(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            values.extend(_numeric_values(value))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        values.append(float(payload))
    return values


def _classifier_health_gate(
    *,
    metrics: Mapping[str, Any],
    probabilities: np.ndarray,
    source_label: int,
    profile: str,
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    raw_config = training_config.get("health_gate", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("training.health_gate must be a mapping.")
    enabled = bool(raw_config.get("enabled", False))
    apply_profile = str(raw_config.get("apply_profile", "full")).strip().lower()
    if not enabled or (apply_profile and profile != apply_profile):
        return {
            "status": "NOT_APPLIED",
            "profile": profile,
            "apply_profile": apply_profile,
            "failures": [],
        }

    failures: list[str] = []
    primary_metric = str(raw_config.get("primary_metric", "roc_auc"))
    minimum = float(raw_config.get("minimum_primary_metric", 0.0))
    observed = metrics.get(primary_metric)
    if observed is None or not math.isfinite(float(observed)):
        failures.append(f"{primary_metric}=unavailable_or_nonfinite")
    elif float(observed) < minimum:
        failures.append(
            f"{primary_metric}={float(observed):.6f}<minimum={minimum:.6f}"
        )

    predictions = np.asarray(probabilities, dtype=np.float64).argmax(axis=1)
    if bool(raw_config.get("require_multiple_predicted_classes", True)) and len(
        np.unique(predictions)
    ) < 2:
        failures.append("validation_predictions_are_single_class")
    if bool(raw_config.get("require_source_class_recall", True)):
        source_metrics = metrics.get("per_class", {}).get(str(int(source_label)), {})
        recall = source_metrics.get("recall") if isinstance(source_metrics, Mapping) else None
        if recall is None or not math.isfinite(float(recall)) or float(recall) <= 0.0:
            failures.append("source_class_recall_is_not_positive")
    if bool(raw_config.get("require_all_class_recall", False)):
        per_class = metrics.get("per_class", {})
        if not isinstance(per_class, Mapping):
            failures.append("per_class_metrics_are_unavailable")
        else:
            for class_index in range(int(probabilities.shape[1])):
                class_metrics = per_class.get(str(class_index), {})
                recall = (
                    class_metrics.get("recall")
                    if isinstance(class_metrics, Mapping)
                    else None
                )
                if (
                    recall is None
                    or not math.isfinite(float(recall))
                    or float(recall) <= 0.0
                ):
                    failures.append(f"class_{class_index}_recall_is_not_positive")
    if bool(raw_config.get("require_finite", True)):
        if not np.isfinite(probabilities).all() or not all(
            math.isfinite(value) for value in _numeric_values(metrics)
        ):
            failures.append("nonfinite_validation_output")

    return {
        "status": "PASS" if not failures else "FAIL",
        "profile": profile,
        "apply_profile": apply_profile,
        "primary_metric": primary_metric,
        "minimum_primary_metric": minimum,
        "observed_primary_metric": observed,
        "predicted_classes": sorted(int(value) for value in np.unique(predictions)),
        "failures": failures,
    }


def _selection_improves(
    *,
    primary: float,
    tiebreak: float | None,
    best_primary: float,
    best_tiebreak: float | None,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether one validation result wins the frozen lexicographic gate."""

    if primary > best_primary + tolerance:
        return True
    if abs(primary - best_primary) > tolerance or tiebreak is None:
        return False
    if best_tiebreak is None:
        return True
    return tiebreak > best_tiebreak + tolerance


def _reload_oracle_smoke(
    checkpoint_dir: Path,
    dataset: MolecularGraphDataset,
    *,
    device: str,
    require_taste_closure: bool = True,
) -> dict[str, Any]:
    """Exercise the persisted bundle and calibrated-probability API once."""

    oracle = GNNOracle.from_checkpoint(
        checkpoint_dir,
        device=device,
        batch_size=min(8, len(dataset)),
        require_taste_closure=require_taste_closure,
    )
    graphs = [dataset[index] for index in range(min(8, len(dataset)))]
    batched = oracle.predict_proba(graphs)
    singles = np.vstack([oracle.predict_proba([graph]) for graph in graphs])
    if not np.isfinite(batched).all() or not np.allclose(
        batched, singles, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            "Reloaded GNN oracle failed finite batch/single probability equivalence."
        )
    records = oracle.predict_records(graphs)
    if len(records) != len(graphs) or any(
        len(record["probabilities"]) != oracle.num_classes for record in records
    ):
        raise RuntimeError("Reloaded GNN oracle prediction-record contract failed.")

    deletion = next(
        (
            outcome
            for outcome in enumerate_connected_hard_deletions("CCO", "O")
            if outcome.valid and outcome.residual_smiles
        ),
        None,
    )
    empty_deletion = enumerate_connected_hard_deletions("CC", "CC")
    invalid_deletion = enumerate_connected_hard_deletions("CC", "not-a-smiles")
    if deletion is None:
        raise RuntimeError("Hard-deletion smoke did not produce a connected residual.")
    if not empty_deletion or any(outcome.valid for outcome in empty_deletion):
        raise RuntimeError("Empty-residual deletion did not fail closed.")
    if invalid_deletion:
        raise RuntimeError("Invalid-fragment deletion did not fail closed.")

    featurizer = MolecularGraphFeaturizer(dataset.feature_schema)

    def graph_from_smiles(smiles: str, molecule_id: str) -> MolecularGraphData:
        features = featurizer.featurize(smiles)
        return MolecularGraphData(
            x=features.node_features,
            edge_index=features.edge_index,
            edge_attr=features.edge_features,
            y=-1,
            molecule_id=molecule_id,
            smiles=features.canonical_smiles,
            split="smoke",
            graph_sha256=features.graph_sha256,
        )

    deletion_records = oracle.predict_records(
        [
            graph_from_smiles("CCO", "deletion-parent"),
            graph_from_smiles(str(deletion.residual_smiles), "deletion-residual"),
        ]
    )
    pred_before = int(deletion_records[0]["predicted_label"])
    pred_after = int(deletion_records[1]["predicted_label"])
    source_probability_before = float(deletion_records[0]["source_probability"])
    source_probability_after = float(deletion_records[1]["source_probability"])
    return {
        "checkpoint_id": oracle.checkpoint_id,
        "num_examples": len(graphs),
        "batch_single_max_abs_difference": float(np.max(np.abs(batched - singles))),
        "temperature": oracle.temperature,
        "deletion_valid": True,
        "deletion_residual_smiles": deletion.residual_smiles,
        "pred_before": pred_before,
        "pred_after": pred_after,
        "source_label": int(oracle.source_label),
        "destination_label": pred_after,
        "strict_flip": (
            pred_before == int(oracle.source_label)
            and pred_after != int(oracle.source_label)
        ),
        "cf_drop": source_probability_before - source_probability_after,
        "empty_deletion_failed_closed": True,
        "invalid_deletion_failed_closed": True,
    }


def _evaluate(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: str,
    num_classes: int,
) -> dict[str, Any]:
    torch = _require_torch()
    model.eval()
    logits_parts: list[Any] = []
    labels_parts: list[Any] = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y.long())
            count = int(batch.y.numel())
            total_loss += float(loss.item()) * count
            total_examples += count
            logits_parts.append(logits.detach().cpu())
            labels_parts.append(batch.y.detach().cpu())
    if not logits_parts:
        raise RuntimeError("Molecular GNN evaluation loader produced no examples.")
    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(labels_parts, dim=0)
    probabilities = torch.softmax(logits, dim=1)
    metrics = classification_metrics(
        labels.numpy(), probabilities.numpy(), num_classes=num_classes
    )
    metrics["loss"] = total_loss / total_examples
    return {
        "metrics": metrics,
        "logits": logits.numpy().astype(np.float64),
        "probabilities": probabilities.numpy().astype(np.float64),
        "labels": labels.numpy().astype(np.int64),
    }


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    tracked_sources = (
        PROJECT_ROOT / "scripts" / "train_molecular_gnn.py",
        PROJECT_ROOT / "src" / "train" / "molecular_gnn_resume.py",
        PROJECT_ROOT / "src" / "models" / "molecular_gnn.py",
        PROJECT_ROOT / "src" / "data" / "molecular_graph_dataset.py",
        PROJECT_ROOT / "src" / "oracles" / "gnn_oracle.py",
    )
    return {
        "commit": command("rev-parse", "HEAD"),
        "tree": command("rev-parse", "HEAD^{tree}"),
        "branch": command("branch", "--show-current"),
        "status_short": command("status", "--short").splitlines(),
        "tracked_source_files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in tracked_sources
        ],
    }


def _environment(device: str) -> dict[str, Any]:
    torch = _require_torch()
    try:
        import rdkit
    except ImportError:  # pragma: no cover
        rdkit_version = None
    else:
        rdkit_version = getattr(rdkit, "__version__", None)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": device,
        "rdkit": rdkit_version,
    }


def _label_map(args: argparse.Namespace, spec: Any) -> dict[str, str]:
    if not args.label_map_json:
        return {str(key): value for key, value in spec.label_map.items()}
    candidate = Path(args.label_map_json).expanduser()
    payload = (
        json.loads(candidate.read_text(encoding="utf-8"))
        if candidate.is_file()
        else json.loads(args.label_map_json)
    )
    return {str(int(key)): str(value) for key, value in payload.items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch = _require_torch()
    config = _config(args)
    dataset_id = normalize_dataset_id(args.dataset, allow_historical=False)
    spec = get_dataset_spec(dataset_id, allow_historical=False)
    num_classes = int(args.num_classes or spec.num_classes)
    source_label = spec.source_label if args.source_label is None else int(args.source_label)
    if num_classes != spec.num_classes or source_label != spec.source_label:
        raise ValueError("CLI class semantics conflict with the active dataset registry.")
    if dataset_id == "tastemolnet" and args.profile == "full":
        physical_cli_paths = {
            "Taste data directory": args.data_dir,
            "Taste output directory": args.output_dir,
            "Taste training-state directory": args.training_state_dir,
            "Taste prepared root": args.taste_prepared_root,
            "Taste graph-cache root": args.graph_cache_root,
            "Taste policy file": args.taste_policy_file,
            "Taste policy receipt": args.taste_policy_receipt,
        }
        for label, value in physical_cli_paths.items():
            if value is not None:
                assert_no_symlink_components(value, label=label)
        for value in args.config:
            assert_no_symlink_components(value, label="Taste config file")
    split_paths = _resolve_split_paths(args)
    taste_runtime = _taste_runtime_authority(
        args,
        dataset_id=dataset_id,
        profile=args.profile,
        split_paths=split_paths,
    )
    if taste_runtime is not None:
        policy, authority, _ = taste_runtime
        autodl_config = config.get("autodl")
        if not isinstance(autodl_config, Mapping) or (
            autodl_config.get("policy_file_sha256") != policy.file_sha256
            or autodl_config.get("physical_gpu_index") != 1
            or autodl_config.get("prepared_output_manifest_sha256")
            != authority.prepared_output_manifest_sha256
            or autodl_config.get("split_manifest_sha256")
            != authority.split_manifest_sha256
            or autodl_config.get("min_free_after_reservations_gb") != 100
        ):
            raise TasteResearchPolicyError(
                "Taste AutoDL config lost policy-v2/GPU1/data authority"
            )
    output_dir = Path(os.path.abspath(Path(args.output_dir).expanduser()))
    git_state = _git_state()
    if taste_runtime is not None:
        if git_state["status_short"]:
            raise MolecularGNNResumeError(
                "TasteMolNet full training requires a clean immutable Git worktree"
            )
        if len(str(git_state.get("commit", ""))) != 40 or len(
            str(git_state.get("tree", ""))
        ) != 40:
            raise MolecularGNNResumeError(
                "TasteMolNet full training requires commit/tree source identity"
            )
        _, authority, _ = taste_runtime
        for private_root in (authority.prepared_root, authority.graph_cache_root):
            if paths_overlap(output_dir, private_root):
                raise ValueError(
                    "immutable output must be disjoint from private Taste prepared/cache roots"
                )
    training_state_dir = (
        None
        if args.training_state_dir is None
        else Path(os.path.abspath(Path(args.training_state_dir).expanduser()))
    )
    if args.resume_training and training_state_dir is None:
        raise ValueError("--resume-training requires --training-state-dir")
    if dataset_id == "tastemolnet" and args.profile == "full" and training_state_dir is None:
        raise ValueError(
            "TasteMolNet full training requires a separate --training-state-dir"
        )
    if training_state_dir is not None:
        if (
            training_state_dir == output_dir
            or training_state_dir in output_dir.parents
            or output_dir in training_state_dir.parents
        ):
            raise ValueError("training-state root and immutable output must be disjoint")
        if taste_runtime is not None:
            _, authority, _ = taste_runtime
            for private_root in (authority.prepared_root, authority.graph_cache_root):
                if (
                    training_state_dir == private_root
                    or training_state_dir in private_root.parents
                    or private_root in training_state_dir.parents
                ):
                    raise ValueError(
                        "training-state root must be disjoint from private Taste inputs"
                    )
    output_preexisting = output_dir.exists() and any(output_dir.iterdir())
    if output_preexisting and not args.resume_training:
        raise FileExistsError(f"Molecular GNN output must be fresh: {output_dir}")
    if output_dir.exists() and not output_preexisting and training_state_dir is not None:
        raise FileExistsError(
            "resumable molecular GNN output must remain absent until terminal publication"
        )
    if output_preexisting and args.resume_published_output_receipt:
        if not args.resume_training:
            raise MolecularGNNResumeError(
                "Published-output adoption requires --resume-training"
            )
        from src.utils.autodl_tastemolnet_gine_controller_v1 import (
            validate_tastemolnet_published_output_adoption_readonly,
        )

        validate_tastemolnet_published_output_adoption_readonly(
            args.resume_published_output_receipt,
            expected_output_dir=output_dir,
            expected_training_state_root=training_state_dir,
        )
    elif taste_runtime is not None and output_preexisting:
        raise MolecularGNNResumeError(
            "Taste published output requires controller-issued completion-only adoption"
        )
    elif args.resume_published_output_receipt:
        raise MolecularGNNResumeError(
            "published-output adoption is forbidden without a published output"
        )

    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, Mapping):
        raise ValueError("training config must be a mapping.")
    seed = int(args.seed if args.seed is not None else training_cfg.get("primary_seed", 7))
    if taste_runtime is not None:
        resolved_configs = _resolved_config_paths(args)
        verified_gine = (PROJECT_ROOT / "configs/gnn/gine.yaml").resolve(strict=True)
        gnn_configs = [
            path for path in resolved_configs if path.parent == verified_gine.parent
        ]
        try:
            minimum_free_gb = int(os.environ.get("MIN_PERSISTENT_FREE_GB", "-1"))
            minimum_after_reservations_gb = int(
                os.environ.get("MIN_FREE_AFTER_RESERVATIONS_GB", "-1")
            )
            storage_reservation_gb = int(
                os.environ.get("TASTEMOLNET_STORAGE_RESERVATION_GB", "-1")
            )
        except ValueError as exc:
            raise MolecularGNNResumeError(
                "Taste minimum persistent free-space threshold is malformed"
            ) from exc
        if (
            args.backbone != "gine"
            or seed != 7
            or gnn_configs != [verified_gine]
            or minimum_free_gb < 100
            or minimum_after_reservations_gb != 100
            or storage_reservation_gb != 20
        ):
            raise MolecularGNNResumeError(
                "Taste full training must use GINE/seed-7 and the exact 20/100 GiB reservation gate"
            )
    _set_seed(seed, exact_cuda=taste_runtime is not None)
    device = _resolve_device(args.device, config)
    runtime_identity = _runtime_identity(
        torch=torch,
        device=device,
        taste_full=taste_runtime is not None,
    )
    profile = args.profile
    smoke = profile == "smoke"
    max_epochs = int(
        args.max_epochs
        if args.max_epochs is not None
        else min(int(training_cfg.get("max_epochs", 200)), 2) if smoke
        else int(training_cfg.get("max_epochs", 200))
    )
    patience = int(
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else training_cfg.get("early_stopping_patience", 20)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 64)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else training_cfg.get("learning_rate", 0.001)
    )
    weight_decay = float(
        args.weight_decay
        if args.weight_decay is not None
        else training_cfg.get("weight_decay", 0.00001)
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else _nested(config, "runtime", "num_workers", 0)
    )
    train_limit = args.train_limit if args.train_limit is not None else (64 if smoke else None)
    validation_limit = (
        args.validation_limit if args.validation_limit is not None else (32 if smoke else None)
    )
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("Epoch, patience, and batch size values must be positive.")
    class_weighted = bool(training_cfg.get("class_weighted_loss", True))
    weighted_sampler = bool(training_cfg.get("weighted_sampler", False))
    if class_weighted and weighted_sampler:
        raise ValueError(
            "class_weighted_loss and weighted_sampler are mutually exclusive."
        )

    schema = default_molecular_feature_schema()
    featurizer = MolecularGraphFeaturizer(schema)
    train_dataset, validation_dataset, training_input = _load_fit_datasets(
        dataset_id=dataset_id,
        profile=profile,
        split_paths=split_paths,
        graph_cache_root=args.graph_cache_root,
        num_classes=num_classes,
        featurizer=featurizer,
        train_limit=train_limit,
        validation_limit=validation_limit,
        stratified_limit=smoke,
        graph_cache_manifest_sha256=(
            None
            if taste_runtime is None
            else taste_runtime[1].graph_cache_manifest_sha256
        ),
    )
    # The held-out test split is never parsed or featurized by this training
    # entrypoint.  Only its path and streaming SHA-256 are frozen below.
    weights = _class_weights(train_dataset.labels, num_classes)
    sampler = None
    if weighted_sampler:
        sample_weights = torch.tensor(
            [weights[label] for label in train_dataset.labels], dtype=torch.double
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
    train_loader = build_molecular_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
    )
    validation_loader = build_molecular_data_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    gnn_values = dict(config.get("gnn", {}))
    if args.backbone:
        gnn_values["backbone"] = args.backbone
    gnn_values["num_classes"] = num_classes
    model_config = MolecularGNNConfig.from_mapping(gnn_values)
    if taste_runtime is not None and model_config.backbone != "gine":
        raise MolecularGNNResumeError(
            "verified Taste backbone and instantiated model config differ"
        )
    model = MolecularGNN(
        model_config,
        node_cardinalities=schema.node_cardinalities,
        edge_cardinalities=schema.edge_cardinalities,
    ).to(device)
    criterion_weights = (
        torch.tensor(weights, dtype=torch.float32, device=device)
        if class_weighted
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=criterion_weights)
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError("The frozen molecular GNN route currently requires optimizer=adamw.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    clip_norm = float(training_cfg.get("gradient_clip_norm", 5.0))
    selection_metric = str(training_cfg.get("selection_metric", "macro_f1"))
    raw_tiebreak_metric = training_cfg.get("selection_tiebreak_metric")
    selection_tiebreak_metric = (
        None
        if raw_tiebreak_metric is None or not str(raw_tiebreak_metric).strip()
        else str(raw_tiebreak_metric)
    )
    resume_store: MolecularGNNResumeStore | None = None
    output_authority: OutputParentAuthority | None = None
    if training_state_dir is not None:
        resume_contract = _training_resume_contract(
            args=args,
            dataset_id=dataset_id,
            profile=profile,
            output_dir=output_dir,
            split_paths=split_paths,
            taste_runtime=taste_runtime,
            model_config=model_config,
            feature_schema=schema.to_dict(),
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            num_workers=num_workers,
            class_weighted=class_weighted,
            weighted_sampler=weighted_sampler,
            selection_metric=selection_metric,
            selection_tiebreak_metric=selection_tiebreak_metric,
            clip_norm=clip_norm,
            config=config,
            git_state=git_state,
            runtime_identity=runtime_identity,
            training_input=training_input,
        )
        resume_store = MolecularGNNResumeStore(
            training_state_dir,
            resume=args.resume_training,
            contract=resume_contract,
            torch_module=torch,
        )
        output_authority = OutputParentAuthority(
            output_dir,
            contract_sha256=resume_store.contract_sha256,
            resume=args.resume_training,
        )
        output_authority.open()
        resume_store.open()
        completion = resume_store.completion()
        if output_preexisting:
            output_authority.verify()
            terminal_workspace = FinalizationWorkspace(
                output_dir,
                contract_sha256=resume_store.contract_sha256,
                resume=True,
                parent_authority=output_authority,
                training_state_root=training_state_dir,
            )
            terminal_workspace.verify_published()
            identity = _bundle_output_identity(output_dir)
            if identity.get("training_resume_contract_sha256") != (
                resume_store.contract_sha256
            ):
                raise MolecularGNNResumeError(
                    "existing terminal bundle belongs to another training contract"
                )
            if completion is None:
                completion = resume_store.mark_complete(
                    output_dir=output_dir, output_identity=identity
                )
            elif completion.get("output_identity") != identity:
                raise MolecularGNNResumeError(
                    "terminal bundle identity differs from training completion manifest"
                )
            resume_store.close()
            output_authority.close()
            terminal_workspace.close()
            print(json.dumps({"training_completion": completion}, sort_keys=True), flush=True)
            if dataset_id == "tastemolnet" and profile == "full":
                print("[TASTE_GINE_THREE_CLASS_PASS]", flush=True)
            print("[MOLECULAR_GNN_TRAIN_OK]", flush=True)
            return 0
        if completion is not None:
            raise MolecularGNNResumeError(
                "training completion manifest exists but immutable output is absent"
            )
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_value = -math.inf
    best_tiebreak_value: float | None = None
    epochs_without_improvement = 0
    first_epoch = 1
    if resume_store is not None:
        snapshot = resume_store.load(model=model, optimizer=optimizer)
        if snapshot is not None:
            first_epoch = snapshot.next_epoch
            history = snapshot.history
            best_state = (
                None if snapshot.best_state is None else dict(snapshot.best_state)
            )
            best_epoch = snapshot.best_epoch
            best_value = snapshot.best_primary
            best_tiebreak_value = snapshot.best_tiebreak
            epochs_without_improvement = snapshot.epochs_without_improvement
            if epochs_without_improvement >= patience:
                # The previous process committed the exact early-stop boundary
                # and then died during finalization.  Do not execute one extra
                # epoch on resume; continue directly to the frozen bundle.
                first_epoch = max_epochs + 1
            for optimizer_state in optimizer.state.values():
                for key, value in list(optimizer_state.items()):
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(device)

    for epoch in range(first_epoch, max_epochs + 1):
        if output_authority is not None:
            output_authority.verify()
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            count = int(batch.y.numel())
            total_loss += float(loss.item()) * count
            total_examples += count
            if taste_runtime is not None and (
                batch_index == 1 or batch_index % 50 == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "TASTE_GINE_BATCH_PROGRESS",
                            "epoch": epoch,
                            "batch": batch_index,
                            "examples_seen_in_epoch": total_examples,
                            "batch_loss": float(loss.item()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        validation = _evaluate(
            model,
            validation_loader,
            criterion,
            device=device,
            num_classes=num_classes,
        )
        metric_value = validation["metrics"].get(selection_metric)
        if metric_value is None or not math.isfinite(float(metric_value)):
            raise ValueError(
                f"Selection metric {selection_metric!r} is unavailable or nonfinite "
                "on validation."
            )
        tiebreak_value: float | None = None
        if selection_tiebreak_metric is not None:
            observed_tiebreak = validation["metrics"].get(selection_tiebreak_metric)
            if observed_tiebreak is None or not math.isfinite(
                float(observed_tiebreak)
            ):
                raise ValueError(
                    "Selection tie-break metric "
                    f"{selection_tiebreak_metric!r} is unavailable or nonfinite "
                    "on validation."
                )
            tiebreak_value = float(observed_tiebreak)
        epoch_row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_examples),
            "validation": validation["metrics"],
            "selection": {
                "primary_metric": selection_metric,
                "primary_value": float(metric_value),
                "tiebreak_metric": selection_tiebreak_metric,
                "tiebreak_value": tiebreak_value,
            },
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
        if _selection_improves(
            primary=float(metric_value),
            tiebreak=tiebreak_value,
            best_primary=best_value,
            best_tiebreak=best_tiebreak_value,
        ):
            best_value = float(metric_value)
            best_tiebreak_value = tiebreak_value
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if resume_store is not None:
            assert output_authority is not None
            output_authority.verify()
            resume_store.save(
                completed_epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_state=best_state,
                best_epoch=best_epoch,
                best_primary=best_value,
                best_tiebreak=best_tiebreak_value,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                metrics={
                    "train_loss": epoch_row["train_loss"],
                    "selection": epoch_row["selection"],
                    "early_stop_counter": epochs_without_improvement,
                },
            )
            output_authority.verify()
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Molecular GNN training did not produce a best checkpoint.")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    final_validation = _evaluate(
        model,
        validation_loader,
        criterion,
        device=device,
        num_classes=num_classes,
    )

    health_gate = _classifier_health_gate(
        metrics=final_validation["metrics"],
        probabilities=final_validation["probabilities"],
        source_label=source_label,
        profile=profile,
        training_config=training_cfg,
    )
    calibration_cfg = config.get("calibration", {})
    if not isinstance(calibration_cfg, Mapping):
        raise ValueError("calibration config must be a mapping.")
    fit_on_validation = bool(calibration_cfg.get("fit_on_validation", False))
    if fit_on_validation:
        if str(calibration_cfg.get("split", "validation")) != "validation":
            raise ValueError(
                "Frozen temperature scaling may use validation only."
            )
        temperature_scaling = fit_temperature_scaling(
            final_validation["logits"],
            final_validation["labels"],
            max_iter=int(calibration_cfg.get("max_iter", 100)),
        )
    else:
        temperature_scaling = None
    split_files = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in split_paths.items()
    }
    test_evaluation_status = {
        "schema_version": "molecular_gnn_test_evaluation_status_v1",
        "status": "NOT_EVALUATED",
        "test_loaded": False,
        "reason": "held_out_until_frozen_final_evaluation",
        "path": split_files["test"]["path"],
        "sha256": split_files["test"]["sha256"],
    }
    split_manifest = {
        "schema_version": "molecular_gnn_split_manifest_v1",
        "dataset": dataset_id,
        "roles": {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        },
        "files": split_files,
        "train_manifest": train_dataset.manifest(),
        "validation_manifest": validation_dataset.manifest(),
        "calibration_loaded_for_training": False,
        "test_loaded_for_training": False,
        "test_evaluated_during_training": False,
        "test_used_for_checkpoint_selection": False,
    }
    training_metrics = {
        "schema_version": "molecular_gnn_training_metrics_v1",
        "profile": profile,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_validation_selection_value": best_value,
        "selection_tiebreak_metric": selection_tiebreak_metric,
        "best_validation_tiebreak_value": best_tiebreak_value,
        "epochs_completed": len(history),
        "history": history,
        "final_validation": final_validation["metrics"],
        "test_evaluation": test_evaluation_status,
        "class_weights": weights,
        "class_weighted_loss": class_weighted,
        "weighted_sampler": weighted_sampler,
        "health_gate": health_gate,
        "temperature_scaling": temperature_scaling,
        "training_input": training_input,
    }
    resolved_config = copy.deepcopy(config)
    resolved_config["gnn"] = model_config.to_dict()
    resolved_config["training"] = {
        **dict(training_cfg),
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "primary_seed": seed,
        "class_weighted_loss": class_weighted,
        "weighted_sampler": weighted_sampler,
    }
    model_card = {
        "dataset": dataset_id,
        "backbone": model_config.backbone,
        "num_classes": num_classes,
        "source_label": source_label,
        "seed": seed,
        "training_commit": git_state["commit"],
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "selection_tiebreak_metric": selection_tiebreak_metric,
        "selection_split": "validation",
        "temperature_calibration_split": "validation",
        "temperature_calibration_fit_on_validation": fit_on_validation,
        "calibration_used_for_model_fit_or_selection": False,
        "test_used_for_model_fit_or_selection": False,
        "test_loaded_during_training": False,
        "test_evaluated_during_training": False,
        "profile": profile,
        "health_gate": health_gate,
        "graph_cache_used": training_input["graph_cache_used"],
        "training_input_mode": training_input["mode"],
    }
    if resume_store is not None:
        model_card.update(
            {
                "training_resume_schema": "molecular_gnn_epoch_resume_v1",
                "training_resume_contract_sha256": resume_store.contract_sha256,
                "training_state_separate_from_immutable_output": True,
            }
        )
    if taste_runtime is not None:
        taste_policy, taste_authority, taste_receipt = taste_runtime
        model_card.update(
            {
                "data_use_policy_file_sha256": taste_policy.file_sha256,
                "data_use_policy_canonical_sha256": taste_policy.canonical_sha256,
                "data_use_policy_receipt_sha256": taste_receipt.sha256,
                "paper_result_reporting_allowed": True,
                "dataset_redistributed": False,
                "upstream_license_not_explicit": True,
                "upstream_license_status": "NOT_EXPLICITLY_STATED",
                "license_pass_claimed": False,
                "graph_cache_manifest_sha256": (
                    taste_authority.graph_cache_manifest_sha256
                ),
            }
        )
    bundle_dir = output_dir
    finalization_workspace: FinalizationWorkspace | None = None
    if resume_store is not None:
        assert output_authority is not None
        output_authority.verify()
        resume_store.verify_writer_authority()
        finalization_workspace = FinalizationWorkspace(
            output_dir,
            contract_sha256=resume_store.contract_sha256,
            resume=args.resume_training,
            parent_authority=output_authority,
            training_state_root=training_state_dir,
        )
        bundle_dir, already_complete = finalization_workspace.prepare()
        resume_store.verify_writer_authority()
        if already_complete:
            verify_checkpoint_bundle(bundle_dir)
            finalization_workspace.publish()
            finalization_workspace.verify_published()
            identity = _bundle_output_identity(output_dir)
            completion = resume_store.mark_complete(
                output_dir=output_dir, output_identity=identity
            )
            output_authority.verify()
            resume_store.close()
            output_authority.close()
            finalization_workspace.close()
            print(json.dumps({"training_completion": completion}, sort_keys=True), flush=True)
            if dataset_id == "tastemolnet" and profile == "full":
                print("[TASTE_GINE_THREE_CLASS_PASS]", flush=True)
            print("[MOLECULAR_GNN_TRAIN_OK]", flush=True)
            return 0
    bundle = save_gnn_checkpoint_bundle(
        model=model,
        checkpoint_dir=bundle_dir,
        feature_schema=schema,
        config=resolved_config,
        model_card=model_card,
        label_map=_label_map(args, spec),
        split_manifest=split_manifest,
        training_metrics=training_metrics,
        test_evaluation_status=test_evaluation_status,
        validation_predictions=_prediction_rows(
            validation_dataset,
            final_validation["logits"],
            final_validation["probabilities"],
        ),
        temperature_scaling=temperature_scaling,
        environment=_environment(device),
        git_state=git_state,
        defer_tastemolnet_closure=taste_runtime is not None,
    )
    if health_gate["status"] == "FAIL":
        print(json.dumps(bundle, sort_keys=True), flush=True)
        print(json.dumps({"health_gate": health_gate}, sort_keys=True), flush=True)
        if dataset_id == "bace":
            print("[BACE_GNN_HEALTH_GATE_FAILED]", flush=True)
        if resume_store is not None:
            resume_store.close()
            assert output_authority is not None
            output_authority.close()
            if finalization_workspace is not None:
                finalization_workspace.close()
        return 3
    if taste_runtime is not None:
        start_policy, start_authority, start_receipt = taste_runtime
        terminal_runtime = _taste_runtime_authority(
            args,
            dataset_id=dataset_id,
            profile=profile,
            split_paths=split_paths,
        )
        assert terminal_runtime is not None
        terminal_policy, terminal_authority, terminal_receipt = terminal_runtime
        if (
            terminal_policy.file_sha256 != start_policy.file_sha256
            or terminal_policy.canonical_sha256 != start_policy.canonical_sha256
            or terminal_authority.evidence() != start_authority.evidence()
            or terminal_receipt.sha256 != start_receipt.sha256
            or terminal_receipt.payload != start_receipt.payload
        ):
            raise TasteResearchPolicyError(
                "Taste policy/private-data authority changed during training"
            )
        policy_binding = {
            "schema_version": "tastemolnet_training_policy_binding_v1",
            "dataset": "tastemolnet",
            "status": "NOT_EXPLICITLY_STATED",
            "authorization_status": "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION",
            "policy": terminal_policy.evidence(),
            "policy_receipt": {
                "path": str(terminal_receipt.path),
                "sha256": terminal_receipt.sha256,
            },
            "private_data_authority": terminal_authority.evidence(),
            "paper_result_reporting_allowed": True,
            "paper_results_reporting_allowed_by_project_policy": True,
            "dataset_redistributed": False,
            "data_redistribution_allowed": False,
            "upstream_license_not_explicit": True,
            "upstream_license_status": "NOT_EXPLICITLY_STATED",
            "upstream_license_claimed_resolved": False,
            "license_pass_claimed": False,
            "public_artifact_audit_required": True,
            "hpc_execution_authorized": False,
        }
        graph_cache_usage = {
            "schema_version": "tastemolnet_graph_cache_usage_v1",
            "dataset": "tastemolnet",
            "mode": "read_only_existing_cache",
            "graph_cache_used": training_input["graph_cache_used"],
            "graph_cache_root": str(terminal_authority.graph_cache_root),
            "graph_cache_manifest_sha256": (
                terminal_authority.graph_cache_manifest_sha256
            ),
            "cache_files": training_input.get("cache_files", {}),
            "cache_contract": training_input.get("cache_contract"),
            "loaded_splits": ["train", "validation"],
            "calibration_loaded": False,
            "test_loaded": False,
            "test_metadata_hash_only": True,
            "graph_cache_rebuilt": False,
            "data_reprepared": False,
        }
        oracle_manifest = {
            "schema_version": "tastemolnet_three_class_gine_oracle_manifest_v1",
            "dataset": "tastemolnet",
            "status": "PASS",
            "checkpoint_id": bundle["checkpoint_id"],
            "classifier_family": "gine",
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "num_classes": 3,
            "label_map": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
            "source_label": 1,
            "source_label_name": "Sweet",
            "selection_split": "validation",
            "selection_metric": selection_metric,
            "selection_tiebreak_metric": selection_tiebreak_metric,
            "temperature_calibration_split": "validation",
            "temperature_scaling": temperature_scaling,
            "health_gate": health_gate,
            "test_loaded": False,
            "test_evaluated": False,
            "test_path_sha256_only": split_files["test"],
            "paper_result_reporting_allowed": True,
            "dataset_redistributed": False,
            "upstream_license_not_explicit": True,
            "public_artifact_audit_required": True,
        }
        _write_new_json(bundle_dir / "data_use_policy_binding.json", policy_binding)
        _write_new_json(bundle_dir / "graph_cache_usage.json", graph_cache_usage)
        _write_new_json(bundle_dir / "oracle_manifest.json", oracle_manifest)
        if resume_store is None or training_state_dir is None:
            raise MolecularGNNResumeError(
                "Taste full training requires one persistent latest-checkpoint authority"
            )
        last_checkpoint = _publish_last_training_checkpoint(
            training_state_root=training_state_dir,
            bundle_dir=bundle_dir,
        )
        _write_new_json(
            bundle_dir / "checkpoint_reload.json",
            {
                "schema_version": "tastemolnet_gine_checkpoint_reload_v1",
                "status": "PENDING_PRIVATE_STAGING_RELOAD",
                "checkpoint_reload_pass": False,
                "last_checkpoint": last_checkpoint,
            },
        )
        update_checkpoint_sha256sums(bundle_dir)
        bundle["audit"] = verify_checkpoint_bundle(
            bundle_dir, require_taste_closure=False
        )
        reload_smoke = _reload_oracle_smoke(
            bundle_dir,
            validation_dataset,
            device=device,
            require_taste_closure=False,
        )
        checkpoint_reload = {
            "schema_version": "tastemolnet_gine_checkpoint_reload_v1",
            "status": "PASS",
            "checkpoint_id": reload_smoke["checkpoint_id"],
            "checkpoint_reload_pass": True,
            "batch_single_probability_equivalence": True,
            "all_probabilities_finite": True,
            "num_classes": 3,
            "source_label": 1,
            "last_checkpoint": last_checkpoint,
            "oracle_reload": reload_smoke,
        }
        _replace_json_before_publication(
            bundle_dir / "checkpoint_reload.json", checkpoint_reload
        )
        update_checkpoint_sha256sums(bundle_dir)
        bundle["audit"] = verify_checkpoint_bundle(bundle_dir)
        final_runtime = _taste_runtime_authority(
            args,
            dataset_id=dataset_id,
            profile=profile,
            split_paths=split_paths,
        )
        assert final_runtime is not None
        final_policy, final_authority, final_receipt = final_runtime
        if (
            final_policy.file_sha256 != terminal_policy.file_sha256
            or final_policy.canonical_sha256 != terminal_policy.canonical_sha256
            or final_authority.evidence() != terminal_authority.evidence()
            or final_receipt.sha256 != terminal_receipt.sha256
            or final_receipt.payload != terminal_receipt.payload
        ):
            raise TasteResearchPolicyError(
                "Taste policy/private-data authority drifted before terminal marker"
            )
    if resume_store is not None:
        assert finalization_workspace is not None
        assert output_authority is not None
        output_authority.verify()
        _publish_staged_bundle(finalization_workspace)
        finalization_workspace.verify_published()
        bundle["checkpoint_dir"] = str(output_dir)
        bundle["audit"] = verify_checkpoint_bundle(output_dir)
        output_identity = _bundle_output_identity(output_dir)
        if output_identity.get("training_resume_contract_sha256") != (
            resume_store.contract_sha256
        ):
            raise MolecularGNNResumeError(
                "published bundle lost its training resume contract"
            )
        resume_store.mark_complete(
            output_dir=output_dir,
            output_identity=output_identity,
        )
        output_authority.verify()
    if smoke:
        reload_smoke = _reload_oracle_smoke(
            output_dir, validation_dataset, device=device
        )
        print(json.dumps({"oracle_reload_smoke": reload_smoke}, sort_keys=True), flush=True)
    print(json.dumps(bundle, sort_keys=True), flush=True)
    if dataset_id == "bace" and profile == "smoke":
        print("[BACE_GNN_SMOKE_PASS]", flush=True)
    if dataset_id == "bace" and profile == "full":
        print("[BACE_GNN_TRAIN_PASS]", flush=True)
    if dataset_id == "tastemolnet" and profile == "full":
        print("[TASTE_GINE_THREE_CLASS_PASS]", flush=True)
    print("[MOLECULAR_GNN_TRAIN_OK]", flush=True)
    if resume_store is not None:
        resume_store.close()
        assert output_authority is not None
        output_authority.close()
        assert finalization_workspace is not None
        finalization_workspace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
