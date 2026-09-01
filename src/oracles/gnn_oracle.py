"""Frozen molecular-GNN oracle, checkpoint bundle, metrics, and calibration."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.molecular_graph_dataset import (
    MolecularGraphBatch,
    MolecularGraphDataset,
    collate_molecular_graphs,
)
from src.data.molecular_graph_featurizer import MolecularFeatureSchema
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig
from src.oracles.base_oracle import BaseOracle, OraclePredictionRecord


CHECKPOINT_BUNDLE_VERSION = "molecular_gnn_checkpoint_v2"
REQUIRED_CHECKPOINT_FILES = (
    "model.pt",
    "config.yaml",
    "model_card.json",
    "feature_schema.json",
    "label_map.json",
    "split_manifest.json",
    "training_metrics.json",
    "validation_predictions.csv",
    "test_evaluation_status.json",
    "temperature_scaling.json",
    "environment.json",
    "git_state.json",
    "sha256sums.txt",
)
TASTE_REQUIRED_CHECKPOINT_FILES = (
    "data_use_policy_binding.json",
    "graph_cache_usage.json",
    "oracle_manifest.json",
    "last.pt",
    "last_checkpoint.json",
    "checkpoint_reload.json",
)

EXPECTED_EMPTY_GRAPH_SEQUENCE = "NO_EVALUABLE_GRAPHS_AFTER_PRE_ORACLE_FILTERS"
UNEXPECTED_EMPTY_GRAPH_SEQUENCE = "UNEXPECTED_EMPTY_GRAPH_SEQUENCE"


def _descriptor_path_or_resolve(path_like: str | Path) -> Path:
    """Preserve an already-held Linux procfs directory authority."""

    path = Path(path_like).expanduser()
    parts = path.parts
    if (
        sys.platform.startswith("linux")
        and len(parts) >= 5
        and parts[0] == os.sep
        and parts[1:4] == ("proc", "self", "fd")
        and parts[4].isdigit()
    ):
        return path
    return path.resolve()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency.
        raise RuntimeError("The frozen GNN oracle requires PyTorch.") from exc
    return torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = [
            "molecule_id",
            "label",
            "predicted_label",
            "logits",
            "probabilities",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def update_checkpoint_sha256sums(checkpoint_dir: str | Path) -> Path:
    root = _descriptor_path_or_resolve(checkpoint_dir)
    lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "sha256sums.txt"
    ]
    target = root / "sha256sums.txt"
    _atomic_write_bytes(target, ("\n".join(lines) + "\n").encode("utf-8"))
    return target


def verify_checkpoint_bundle(
    checkpoint_dir: str | Path,
    *,
    verify_hashes: bool = True,
    require_taste_closure: bool = True,
) -> dict[str, Any]:
    root = _descriptor_path_or_resolve(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"GNN checkpoint directory does not exist: {root}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"GNN checkpoint bundle is missing required files: {missing}")
    checked = 0
    if verify_hashes:
        expected: dict[str, str] = {}
        for line in (root / "sha256sums.txt").read_text(encoding="utf-8").splitlines():
            digest, separator, relative = line.partition("  ")
            if not separator or not relative:
                raise ValueError(f"Malformed GNN checkpoint SHA line: {line!r}")
            if Path(relative).name != relative:
                raise ValueError(f"Nested paths are forbidden in checkpoint SHA file: {relative}")
            expected[relative] = digest
        required_hashed = set(REQUIRED_CHECKPOINT_FILES) - {"sha256sums.txt"}
        absent = sorted(required_hashed - set(expected))
        if absent:
            raise ValueError(f"GNN checkpoint SHA inventory omits required files: {absent}")
        for relative, digest in expected.items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"GNN checkpoint SHA mismatch: {relative}")
            checked += 1
    model_card = json.loads((root / "model_card.json").read_text(encoding="utf-8"))
    if str(model_card.get("oracle_backend")) != "gnn":
        raise ValueError("Frozen molecular classifier must declare oracle_backend=gnn.")
    if model_card.get("rf_oracle_used") is not False:
        raise ValueError("Frozen molecular classifier must declare rf_oracle_used=false.")
    if (
        require_taste_closure
        and str(model_card.get("dataset", "")).strip().lower() == "tastemolnet"
        and str(model_card.get("profile", "")).strip().lower() == "full"
    ):
        missing_taste = [
            name for name in TASTE_REQUIRED_CHECKPOINT_FILES if not (root / name).is_file()
        ]
        if missing_taste:
            raise ValueError(
                "TasteMolNet GINE bundle is missing scoped policy/cache closure: "
                f"{missing_taste}"
            )
        if verify_hashes:
            unhashed_taste = sorted(set(TASTE_REQUIRED_CHECKPOINT_FILES) - set(expected))
            if unhashed_taste:
                raise ValueError(
                    "TasteMolNet GINE SHA inventory omits scoped closure: "
                    f"{unhashed_taste}"
                )
        binding = json.loads(
            (root / "data_use_policy_binding.json").read_text(encoding="utf-8")
        )
        cache_usage = json.loads(
            (root / "graph_cache_usage.json").read_text(encoding="utf-8")
        )
        oracle_manifest = json.loads(
            (root / "oracle_manifest.json").read_text(encoding="utf-8")
        )
        last_checkpoint = json.loads(
            (root / "last_checkpoint.json").read_text(encoding="utf-8")
        )
        checkpoint_reload = json.loads(
            (root / "checkpoint_reload.json").read_text(encoding="utf-8")
        )
        if (
            binding.get("schema_version") != "tastemolnet_training_policy_binding_v1"
            or binding.get("dataset") != "tastemolnet"
            or binding.get("status") != "NOT_EXPLICITLY_STATED"
            or binding.get("authorization_status")
            != "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION"
            or binding.get("paper_result_reporting_allowed") is not True
            or binding.get("dataset_redistributed") is not False
            or binding.get("data_redistribution_allowed") is not False
            or binding.get("upstream_license_not_explicit") is not True
            or binding.get("upstream_license_status") != "NOT_EXPLICITLY_STATED"
            or binding.get("upstream_license_claimed_resolved") is not False
            or binding.get("license_pass_claimed") is not False
            or binding.get("hpc_execution_authorized") is not False
        ):
            raise ValueError("TasteMolNet scoped policy binding changed.")
        binding_policy = binding.get("policy", {})
        binding_receipt = binding.get("policy_receipt", {})
        if (
            not isinstance(binding_policy, Mapping)
            or not isinstance(binding_receipt, Mapping)
            or binding_policy.get("policy_file_sha256")
            != model_card.get("data_use_policy_file_sha256")
            or binding_policy.get("policy_canonical_sha256")
            != model_card.get("data_use_policy_canonical_sha256")
            or binding_receipt.get("sha256")
            != model_card.get("data_use_policy_receipt_sha256")
        ):
            raise ValueError("TasteMolNet policy hashes conflict with model_card.json.")
        if (
            cache_usage.get("schema_version") != "tastemolnet_graph_cache_usage_v1"
            or cache_usage.get("dataset") != "tastemolnet"
            or cache_usage.get("mode") != "read_only_existing_cache"
            or cache_usage.get("graph_cache_used") is not True
            or cache_usage.get("loaded_splits") != ["train", "validation"]
            or cache_usage.get("calibration_loaded") is not False
            or cache_usage.get("test_loaded") is not False
            or cache_usage.get("graph_cache_rebuilt") is not False
            or cache_usage.get("data_reprepared") is not False
            or cache_usage.get("graph_cache_manifest_sha256")
            != model_card.get("graph_cache_manifest_sha256")
        ):
            raise ValueError("TasteMolNet read-only graph-cache closure changed.")
        if (
            oracle_manifest.get("schema_version")
            != "tastemolnet_three_class_gine_oracle_manifest_v1"
            or oracle_manifest.get("dataset") != "tastemolnet"
            or oracle_manifest.get("status") != "PASS"
            or oracle_manifest.get("checkpoint_id") != model_card.get("checkpoint_id")
            or oracle_manifest.get("oracle_backend") != "gnn"
            or oracle_manifest.get("classifier_family") != "gine"
            or oracle_manifest.get("rf_oracle_used") is not False
            or oracle_manifest.get("num_classes") != 3
            or oracle_manifest.get("source_label") != 1
            or oracle_manifest.get("test_loaded") is not False
            or oracle_manifest.get("test_evaluated") is not False
            or oracle_manifest.get("paper_result_reporting_allowed") is not True
            or oracle_manifest.get("dataset_redistributed") is not False
            or oracle_manifest.get("upstream_license_not_explicit") is not True
            or oracle_manifest.get("health_gate", {}).get("status") != "PASS"
        ):
            raise ValueError("TasteMolNet three-class GINE oracle manifest changed.")
        if (
            last_checkpoint.get("schema_version")
            != "tastemolnet_last_training_checkpoint_v1"
            or last_checkpoint.get("checkpoint_file") != "last.pt"
            or last_checkpoint.get("same_bytes_as_latest_epoch_checkpoint") is not True
            or type(last_checkpoint.get("completed_epoch")) is not int
            or last_checkpoint.get("completed_epoch") < 1
            or last_checkpoint.get("checkpoint_sha256") != sha256_file(root / "last.pt")
            or last_checkpoint.get("source_checkpoint_sha256")
            != last_checkpoint.get("checkpoint_sha256")
        ):
            raise ValueError("TasteMolNet latest-epoch checkpoint closure changed.")
        if (
            checkpoint_reload.get("schema_version")
            != "tastemolnet_gine_checkpoint_reload_v1"
            or checkpoint_reload.get("status") != "PASS"
            or checkpoint_reload.get("checkpoint_reload_pass") is not True
            or checkpoint_reload.get("batch_single_probability_equivalence") is not True
            or checkpoint_reload.get("all_probabilities_finite") is not True
            or checkpoint_reload.get("num_classes") != 3
            or checkpoint_reload.get("source_label") != 1
            or checkpoint_reload.get("checkpoint_id") != sha256_file(root / "model.pt")
            or checkpoint_reload.get("last_checkpoint") != last_checkpoint
        ):
            raise ValueError("TasteMolNet checkpoint reload evidence changed.")
        serialized_taste = json.dumps(
            [
                binding,
                cache_usage,
                oracle_manifest,
                last_checkpoint,
                checkpoint_reload,
                model_card,
            ],
            sort_keys=True,
        )
        if "TASTE_LICENSE_PASS" in serialized_taste or "LICENSE_PASS" in serialized_taste:
            raise ValueError("TasteMolNet bundle may not claim an upstream license PASS.")
    test_status = json.loads(
        (root / "test_evaluation_status.json").read_text(encoding="utf-8")
    )
    if test_status.get("status") != "NOT_EVALUATED":
        raise ValueError(
            "Frozen training bundle must declare held-out test status NOT_EVALUATED."
        )
    if test_status.get("test_loaded") is not False:
        raise ValueError(
            "Frozen training bundle must declare held-out test_loaded=false."
        )
    if not str(test_status.get("reason", "")).strip():
        raise ValueError("Held-out test status must record a non-empty reason.")
    test_path = str(test_status.get("path", "")).strip()
    test_sha256 = str(test_status.get("sha256", "")).strip().lower()
    if not test_path or len(test_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in test_sha256
    ):
        raise ValueError("Held-out test status must record its path and SHA-256.")
    split_manifest = json.loads(
        (root / "split_manifest.json").read_text(encoding="utf-8")
    )
    manifest_test = split_manifest.get("files", {}).get("test", {})
    if manifest_test and (
        str(manifest_test.get("path")) != test_path
        or str(manifest_test.get("sha256", "")).lower() != test_sha256
    ):
        raise ValueError(
            "Held-out test status conflicts with split_manifest.json provenance."
        )
    return {
        "checkpoint_dir": str(root),
        "required_files": list(REQUIRED_CHECKPOINT_FILES),
        "hashes_verified": checked,
        "model_card": model_card,
    }


def save_gnn_checkpoint_bundle(
    *,
    model: MolecularGNN,
    checkpoint_dir: str | Path,
    feature_schema: MolecularFeatureSchema,
    config: Mapping[str, Any],
    model_card: Mapping[str, Any],
    label_map: Mapping[str | int, str],
    split_manifest: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    test_evaluation_status: Mapping[str, Any],
    validation_predictions: Sequence[Mapping[str, Any]] = (),
    temperature_scaling: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    git_state: Mapping[str, Any] | None = None,
    defer_tastemolnet_closure: bool = False,
) -> dict[str, Any]:
    """Write the complete immutable classifier bundle expected by downstream jobs."""

    torch = _require_torch()
    root = _descriptor_path_or_resolve(checkpoint_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"GNN checkpoint output must be fresh: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schema_payload = feature_schema.to_dict()
    payload = {
        "bundle_version": CHECKPOINT_BUNDLE_VERSION,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model.config.to_dict(),
        "node_cardinalities": list(model.node_cardinalities),
        "edge_cardinalities": list(model.edge_cardinalities),
        "feature_schema_sha256": schema_payload["schema_sha256"],
    }
    model_temporary = root / ".model.pt.tmp"
    torch.save(payload, model_temporary)
    with model_temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(model_temporary, root / "model.pt")
    checkpoint_id = sha256_file(root / "model.pt")

    card = dict(model_card)
    card.update(
        {
            "checkpoint_bundle_version": CHECKPOINT_BUNDLE_VERSION,
            "checkpoint_id": checkpoint_id,
            "oracle_backend": "gnn",
            "classifier_type": "gnn",
            "rf_oracle_used": False,
            "backbone": model.config.backbone,
            "num_classes": model.config.num_classes,
            "node_feature_schema": schema_payload["node_feature_schema"]["version"],
            "edge_feature_schema": schema_payload["edge_feature_schema"]["version"],
            "feature_schema_sha256": schema_payload["schema_sha256"],
            "edge_feature_mode": model.config.to_dict()["edge_feature_mode"],
        }
    )
    source_label = int(card.get("source_label", -1))
    if not 0 <= source_label < model.config.num_classes:
        raise ValueError("model_card source_label falls outside num_classes.")
    normalized_label_map = {str(int(key)): str(value) for key, value in label_map.items()}
    if set(normalized_label_map) != {
        str(index) for index in range(model.config.num_classes)
    }:
        raise ValueError("label_map must define every class index exactly once.")

    # JSON is a valid YAML 1.2 subset and remains readable without a YAML package.
    _atomic_json(root / "config.yaml", dict(config))
    _atomic_json(root / "model_card.json", card)
    _atomic_json(root / "feature_schema.json", schema_payload)
    _atomic_json(root / "label_map.json", normalized_label_map)
    _atomic_json(root / "split_manifest.json", dict(split_manifest))
    _atomic_json(root / "training_metrics.json", dict(training_metrics))
    _atomic_csv(root / "validation_predictions.csv", validation_predictions)
    _atomic_json(root / "test_evaluation_status.json", dict(test_evaluation_status))
    _atomic_json(
        root / "temperature_scaling.json",
        dict(
            temperature_scaling
            or {
                "schema_version": "temperature_scaling_v1",
                "status": "not_fit",
                "selection_split": "validation",
                "test_used_for_fit": False,
                "temperature": 1.0,
            }
        ),
    )
    _atomic_json(root / "environment.json", dict(environment or {}))
    _atomic_json(root / "git_state.json", dict(git_state or {}))
    update_checkpoint_sha256sums(root)
    audit = verify_checkpoint_bundle(
        root, require_taste_closure=not defer_tastemolnet_closure
    )
    return {
        "checkpoint_dir": str(root),
        "checkpoint_id": checkpoint_id,
        "bundle_version": CHECKPOINT_BUNDLE_VERSION,
        "audit": audit,
    }


def _torch_load(path: Path, *, map_location: Any) -> Mapping[str, Any]:
    torch = _require_torch()
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - older torch.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Molecular GNN model.pt must contain a mapping payload.")
    return payload


def _json_payload_bytes(
    payloads: Mapping[str, bytes],
    name: str,
) -> dict[str, Any]:
    data = payloads.get(name)
    if type(data) is not bytes or not data:
        raise ValueError(f"Frozen GNN payload {name} is missing")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Frozen GNN payload {name} is malformed") from exc
    if type(value) is not dict:
        raise ValueError(f"Frozen GNN payload {name} must contain one JSON object")
    return value


def load_gnn_checkpoint_payloads(
    payloads: Mapping[str, bytes],
    *,
    device: str | Any = "cpu",
) -> tuple[MolecularGNN, dict[str, Any]]:
    """Load a GNN only from already-authorized in-memory checkpoint bytes.

    A retained directory authority is responsible for proving the complete
    bundle inventory before supplying these payloads.  This loader performs no
    pathname reopen, which lets downstream jobs remain safe against a
    swap-load-restore race on an otherwise immutable checkpoint directory.
    """

    required = {
        "model.pt",
        "model_card.json",
        "feature_schema.json",
        "label_map.json",
        "split_manifest.json",
        "test_evaluation_status.json",
        "temperature_scaling.json",
    }
    if type(payloads) is not dict or set(payloads) != required:
        raise ValueError("Frozen GNN in-memory payload set differs from contract")
    model_bytes = payloads["model.pt"]
    if type(model_bytes) is not bytes or not model_bytes:
        raise ValueError("Frozen GNN model payload is empty")
    feature_schema_payload = _json_payload_bytes(payloads, "feature_schema.json")
    feature_schema = MolecularFeatureSchema.from_dict(feature_schema_payload)
    torch = _require_torch()
    try:
        try:
            model_payload = torch.load(
                io.BytesIO(model_bytes), map_location=device, weights_only=True
            )
        except TypeError:  # pragma: no cover - old torch.
            model_payload = torch.load(io.BytesIO(model_bytes), map_location=device)
    except Exception as exc:
        raise ValueError("Frozen GNN model payload could not be deserialized") from exc
    if type(model_payload) is not dict:
        raise ValueError("Frozen GNN model payload must contain one mapping")
    if model_payload.get("bundle_version") != CHECKPOINT_BUNDLE_VERSION:
        raise ValueError("Frozen GNN model bundle version changed")
    if (
        type(model_payload.get("feature_schema_sha256")) is not str
        or model_payload["feature_schema_sha256"]
        != feature_schema_payload.get("schema_sha256")
    ):
        raise ValueError("Frozen GNN model/feature fingerprints differ")
    state_dict = model_payload.get("state_dict")
    if type(state_dict) is not dict or not state_dict:
        raise ValueError("Frozen GNN model state is empty")
    config = MolecularGNNConfig.from_mapping(model_payload.get("model_config"))
    model = MolecularGNN(
        config,
        node_cardinalities=feature_schema.node_cardinalities,
        edge_cardinalities=feature_schema.edge_cardinalities,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    checkpoint_id = hashlib.sha256(model_bytes).hexdigest()
    model_card = _json_payload_bytes(payloads, "model_card.json")
    if (
        model_card.get("checkpoint_bundle_version") != CHECKPOINT_BUNDLE_VERSION
        or model_card.get("checkpoint_id") != checkpoint_id
        or model_card.get("feature_schema_sha256")
        != feature_schema_payload.get("schema_sha256")
        or type(model_card.get("backbone")) is not str
        or model_card.get("backbone") != config.backbone
        or type(model_card.get("num_classes")) is not int
        or model_card.get("num_classes") != config.num_classes
        or type(model_card.get("source_label")) is not int
        or not 0 <= model_card["source_label"] < config.num_classes
        or model_card.get("oracle_backend") != "gnn"
        or model_card.get("classifier_type") != "gnn"
        or model_card.get("rf_oracle_used") is not False
    ):
        raise ValueError("Frozen GNN model-card authority changed")
    temperature = _json_payload_bytes(payloads, "temperature_scaling.json")
    temperature_value = temperature.get("temperature")
    if (
        type(temperature_value) not in (int, float)
        or isinstance(temperature_value, bool)
        or not math.isfinite(float(temperature_value))
        or float(temperature_value) <= 0.0
    ):
        raise ValueError("Frozen GNN temperature authority changed")
    label_map = _json_payload_bytes(payloads, "label_map.json")
    if set(label_map) != {str(index) for index in range(config.num_classes)} or any(
        type(value) is not str or not value for value in label_map.values()
    ):
        raise ValueError("Frozen GNN label-map authority changed")
    split_manifest = _json_payload_bytes(payloads, "split_manifest.json")
    test_status = _json_payload_bytes(payloads, "test_evaluation_status.json")
    if (
        test_status.get("status") != "NOT_EVALUATED"
        or test_status.get("test_loaded") is not False
    ):
        raise ValueError("Frozen GNN test authority changed")
    return model, {
        "checkpoint_id": checkpoint_id,
        "model_card": model_card,
        "feature_schema": feature_schema,
        "temperature_scaling": temperature,
        "label_map": label_map,
        "split_manifest": split_manifest,
        "test_evaluation_status": test_status,
    }


def load_gnn_checkpoint_bundle(
    checkpoint_dir: str | Path,
    *,
    device: str | Any = "cpu",
    verify_hashes: bool = True,
    require_taste_closure: bool = True,
) -> tuple[MolecularGNN, dict[str, Any]]:
    root = _descriptor_path_or_resolve(checkpoint_dir)
    audit = verify_checkpoint_bundle(
        root,
        verify_hashes=verify_hashes,
        require_taste_closure=require_taste_closure,
    )
    feature_schema_payload = json.loads(
        (root / "feature_schema.json").read_text(encoding="utf-8")
    )
    feature_schema = MolecularFeatureSchema.from_dict(feature_schema_payload)
    payload = _torch_load(root / "model.pt", map_location=device)
    if payload.get("bundle_version") != CHECKPOINT_BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported molecular GNN checkpoint version: {payload.get('bundle_version')!r}"
        )
    if payload.get("feature_schema_sha256") != feature_schema_payload["schema_sha256"]:
        raise ValueError("model.pt and feature_schema.json fingerprints differ.")
    config = MolecularGNNConfig.from_mapping(payload["model_config"])
    model = MolecularGNN(
        config,
        node_cardinalities=feature_schema.node_cardinalities,
        edge_cardinalities=feature_schema.edge_cardinalities,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    temperature_payload = json.loads(
        (root / "temperature_scaling.json").read_text(encoding="utf-8")
    )
    metadata = {
        "checkpoint_dir": str(root),
        "checkpoint_id": sha256_file(root / "model.pt"),
        "model_card": audit["model_card"],
        "feature_schema": feature_schema,
        "temperature_scaling": temperature_payload,
        "label_map": json.loads((root / "label_map.json").read_text(encoding="utf-8")),
        "split_manifest": json.loads(
            (root / "split_manifest.json").read_text(encoding="utf-8")
        ),
        "test_evaluation_status": json.loads(
            (root / "test_evaluation_status.json").read_text(encoding="utf-8")
        ),
    }
    return model, metadata


class GNNOracle(BaseOracle):
    """Loaded-once frozen molecular graph classifier with calibrated probabilities."""

    def __init__(
        self,
        model: Any,
        *,
        device: str | Any = "cpu",
        checkpoint_id: str,
        backbone: str,
        num_classes: int,
        source_label: int,
        temperature: float = 1.0,
        edge_feature_dim: int,
        default_batch_size: int = 256,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.checkpoint_id = str(checkpoint_id)
        self.backbone = str(backbone)
        self.num_classes = int(num_classes)
        self.source_label = int(source_label)
        self.temperature = float(temperature)
        self.edge_feature_dim = int(edge_feature_dim)
        self.default_batch_size = int(default_batch_size)
        self.checkpoint_dir = (
            None
            if checkpoint_dir is None
            else _descriptor_path_or_resolve(checkpoint_dir)
        )
        self.validate_contract()
        if self.edge_feature_dim <= 0 or self.default_batch_size <= 0:
            raise ValueError("GNN oracle edge feature width and batch size must be positive.")
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str | Any = "cpu",
        batch_size: int = 256,
        verify_hashes: bool = True,
        require_taste_closure: bool = True,
    ) -> "GNNOracle":
        model, metadata = load_gnn_checkpoint_bundle(
            checkpoint_dir,
            device=device,
            verify_hashes=verify_hashes,
            require_taste_closure=require_taste_closure,
        )
        card = metadata["model_card"]
        temperature = float(metadata["temperature_scaling"].get("temperature", 1.0))
        feature_schema: MolecularFeatureSchema = metadata["feature_schema"]
        return cls(
            model,
            device=device,
            checkpoint_id=metadata["checkpoint_id"],
            backbone=str(card["backbone"]),
            num_classes=int(card["num_classes"]),
            source_label=int(card["source_label"]),
            temperature=temperature,
            edge_feature_dim=len(feature_schema.edge_fields),
            default_batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
        )

    @classmethod
    def from_payloads(
        cls,
        payloads: Mapping[str, bytes],
        *,
        device: str | Any = "cpu",
        batch_size: int = 256,
        checkpoint_dir: str | Path | None = None,
    ) -> "GNNOracle":
        """Build an oracle from descriptor-authorized in-memory payloads."""

        model, metadata = load_gnn_checkpoint_payloads(payloads, device=device)
        card = metadata["model_card"]
        temperature = metadata["temperature_scaling"]["temperature"]
        feature_schema: MolecularFeatureSchema = metadata["feature_schema"]
        return cls(
            model,
            device=device,
            checkpoint_id=metadata["checkpoint_id"],
            backbone=card["backbone"],
            num_classes=card["num_classes"],
            source_label=card["source_label"],
            temperature=float(temperature),
            edge_feature_dim=len(feature_schema.edge_fields),
            default_batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
        )

    def _batches(self, graphs: Any, batch_size: int | None) -> Iterable[Any]:
        size = int(batch_size or self.default_batch_size)
        if size <= 0:
            raise ValueError("Oracle batch_size must be positive.")
        if isinstance(graphs, MolecularGraphBatch):
            yield graphs
            return
        if isinstance(graphs, MolecularGraphDataset):
            # ``MolecularGraphDataset`` implements integer indexing but not
            # slice indexing.  Materialize its portable graph views once so
            # the chunking below works for full-dataset evaluation.
            values: Sequence[Any] = [graphs[index] for index in range(len(graphs))]
        elif hasattr(graphs, "x") and hasattr(graphs, "edge_index"):
            if hasattr(graphs, "batch") and getattr(graphs, "batch") is not None:
                yield graphs
                return
            values = [graphs]
        elif isinstance(graphs, Sequence) and not isinstance(
            graphs, (str, bytes, bytearray)
        ):
            values = graphs
        else:
            raise TypeError(
                "GNNOracle expects a graph, graph batch, dataset, or sequence of graphs."
            )
        if not values:
            raise ValueError("GNNOracle cannot predict an empty graph sequence.")
        for start in range(0, len(values), size):
            yield collate_molecular_graphs(
                values[start : start + size], edge_feature_dim=self.edge_feature_dim
            )

    def _known_graph_count(self, graphs: Any) -> int | None:
        """Return an input count when the public graph container exposes one."""

        if isinstance(graphs, MolecularGraphBatch):
            return int(graphs.num_graphs)
        if isinstance(graphs, MolecularGraphDataset):
            return len(graphs)
        if isinstance(graphs, Sequence) and not isinstance(
            graphs, (str, bytes, bytearray)
        ):
            return len(graphs)
        if hasattr(graphs, "x") and hasattr(graphs, "edge_index"):
            return None
        return None

    def _authorized_empty_logits(
        self,
        graphs: Any,
        *,
        allow_empty: bool,
        expected_count: int | None,
    ) -> np.ndarray | None:
        """Return an explicitly authorized empty batch or fail closed.

        Empty inference is never inferred from a truthy flag alone.  The
        caller must opt in with the exact pair ``allow_empty=True`` and
        ``expected_count=0``.  This keeps the default oracle contract strict
        while allowing a caller that has independently proved an expected
        empty application batch to preserve its typed array shapes.
        """

        if expected_count is not None and (
            type(expected_count) is not int or expected_count < 0
        ):
            raise ValueError("GNNOracle expected_count must be a non-negative int.")
        actual_count = self._known_graph_count(graphs)
        if actual_count is None:
            return None
        if expected_count is not None and actual_count != expected_count:
            if actual_count == 0 and expected_count > 0:
                raise ValueError(
                    f"{UNEXPECTED_EMPTY_GRAPH_SEQUENCE}: "
                    f"expected_count={expected_count}, actual_count=0"
                )
            raise ValueError(
                "GNNOracle graph sequence count mismatch: "
                f"expected_count={expected_count}, actual_count={actual_count}."
            )
        if actual_count != 0:
            return None
        if allow_empty is True and expected_count == 0:
            return np.empty((0, int(self.num_classes)), dtype=np.float64)
        raise ValueError("GNNOracle cannot predict an empty graph sequence.")

    def _predict_logits(
        self,
        graphs: Any,
        batch_size: int | None,
        *,
        allow_empty: bool = False,
        expected_count: int | None = None,
    ) -> np.ndarray:
        empty = self._authorized_empty_logits(
            graphs,
            allow_empty=allow_empty,
            expected_count=expected_count,
        )
        if empty is not None:
            return empty
        torch = _require_torch()
        outputs: list[Any] = []
        self.model.eval()
        with torch.no_grad():
            for batch in self._batches(graphs, batch_size):
                if hasattr(batch, "to"):
                    batch = batch.to(self.device)
                logits = self.model(batch)
                if isinstance(logits, Mapping):
                    logits = logits.get("logits")
                elif isinstance(logits, tuple):
                    logits = logits[-1]
                if logits is None or logits.ndim != 2:
                    raise RuntimeError("GNN classifier did not return rank-2 logits.")
                if int(logits.shape[1]) != self.num_classes:
                    raise RuntimeError(
                        "GNN classifier output width does not match checkpoint num_classes."
                    )
                outputs.append(logits.detach().cpu())
        if not outputs:
            raise RuntimeError("GNN oracle produced no logits.")
        result = torch.cat(outputs, dim=0).numpy().astype(np.float64, copy=False)
        if int(result.shape[0]) == 0:
            raise RuntimeError(
                f"{UNEXPECTED_EMPTY_GRAPH_SEQUENCE}: model returned zero rows"
            )
        known_count = self._known_graph_count(graphs)
        required_count = expected_count if expected_count is not None else known_count
        if required_count is not None and int(result.shape[0]) != required_count:
            raise RuntimeError(
                "GNNOracle prediction row count mismatch: "
                f"expected_count={required_count}, actual_count={result.shape[0]}."
            )
        return result

    def predict_logits(
        self,
        graphs: Any,
        *,
        batch_size: int | None = None,
        allow_empty: bool = False,
        expected_count: int | None = None,
    ) -> np.ndarray:
        return self._predict_logits(
            graphs,
            batch_size,
            allow_empty=allow_empty,
            expected_count=expected_count,
        )

    def predict_proba(
        self,
        graphs: Any,
        *,
        batch_size: int | None = None,
        allow_empty: bool = False,
        expected_count: int | None = None,
    ) -> np.ndarray:
        logits = self._predict_logits(
            graphs,
            batch_size,
            allow_empty=allow_empty,
            expected_count=expected_count,
        ) / self.temperature
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def predict_label(
        self,
        graphs: Any,
        *,
        batch_size: int | None = None,
        allow_empty: bool = False,
        expected_count: int | None = None,
    ) -> np.ndarray:
        # A positive scalar temperature cannot change argmax.
        return self._predict_logits(
            graphs,
            batch_size,
            allow_empty=allow_empty,
            expected_count=expected_count,
        ).argmax(axis=1)

    def predict_records(
        self,
        graphs: Any,
        *,
        batch_size: int | None = None,
        allow_empty: bool = False,
        expected_count: int | None = None,
    ) -> list[dict[str, Any]]:
        logits = self._predict_logits(
            graphs,
            batch_size,
            allow_empty=allow_empty,
            expected_count=expected_count,
        )
        scaled = logits / self.temperature
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        records: list[dict[str, Any]] = []
        for row_logits, row_probabilities in zip(logits, probabilities, strict=True):
            predicted = int(row_probabilities.argmax())
            records.append(
                OraclePredictionRecord(
                    predicted_label=predicted,
                    probabilities=tuple(float(value) for value in row_probabilities),
                    logits=tuple(float(value) for value in row_logits),
                    source_probability=float(row_probabilities[self.source_label]),
                    confidence=float(row_probabilities[predicted]),
                    checkpoint_id=self.checkpoint_id,
                    backbone=self.backbone,
                    num_classes=self.num_classes,
                    temperature=self.temperature,
                    source_label=self.source_label,
                ).to_dict()
            )
        return records


def _confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(labels, predictions, strict=True):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def _rank_auc(binary_labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(binary_labels.sum())
    negatives = int(binary_labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks[binary_labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _average_precision(binary_labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(binary_labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = binary_labels[order]
    cumulative = np.cumsum(sorted_labels)
    positions = np.arange(1, sorted_labels.size + 1)
    return float((cumulative[sorted_labels == 1] / positions[sorted_labels == 1]).sum() / positives)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    num_bins: int = 15,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    ece = 0.0
    for bin_index in range(int(num_bins)):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        selected = (confidences > lower) & (
            confidences <= upper if bin_index else confidences >= lower
        )
        if not selected.any():
            continue
        accuracy = float((predictions[selected] == targets[selected]).mean())
        confidence = float(confidences[selected].mean())
        ece += float(selected.mean()) * abs(accuracy - confidence)
    return float(ece)


def classification_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Compute binary or multiclass classifier metrics without test-time fitting."""

    targets = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (targets.size, int(num_classes)):
        raise ValueError(
            f"Probability shape mismatch: expected={(targets.size, num_classes)}, "
            f"observed={probs.shape}"
        )
    predictions = probs.argmax(axis=1)
    matrix = _confusion_matrix(targets, predictions, int(num_classes))
    per_class: dict[str, Any] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for label in range(int(num_classes)):
        tp = int(matrix[label, label])
        fp = int(matrix[:, label].sum() - tp)
        fn = int(matrix[label, :].sum() - tp)
        support = int(matrix[label, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        recalls.append(recall)
        f1s.append(f1)
    one_hot = np.eye(int(num_classes), dtype=np.float64)[targets]
    multiclass_brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    row_sums = matrix.sum(axis=1).astype(np.float64)
    col_sums = matrix.sum(axis=0).astype(np.float64)
    expected = float((row_sums * col_sums).sum() / (total * total)) if total else 0.0
    accuracy = correct / total if total else 0.0
    denominator = math.sqrt(
        max(0.0, float(total * total - np.square(col_sums).sum()))
        * max(0.0, float(total * total - np.square(row_sums).sum()))
    )
    mcc = (
        (total * correct - float((row_sums * col_sums).sum())) / denominator
        if denominator
        else 0.0
    )
    aucs: list[float] = []
    aps: list[float] = []
    for label in range(int(num_classes)):
        binary = (targets == label).astype(np.int64)
        auc = _rank_auc(binary, probs[:, label])
        ap = _average_precision(binary, probs[:, label])
        if auc is not None:
            aucs.append(float(auc))
        if ap is not None:
            aps.append(float(ap))
    result: dict[str, Any] = {
        "num_examples": total,
        "num_classes": int(num_classes),
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "brier_score": (
            float(np.mean((probs[:, 1] - targets) ** 2))
            if int(num_classes) == 2
            else multiclass_brier
        ),
        "multiclass_brier_score": multiclass_brier,
        "ece": expected_calibration_error(probs, targets),
        "macro_ovr_roc_auc": float(np.mean(aucs)) if len(aucs) == num_classes else None,
        "macro_ovr_pr_auc": float(np.mean(aps)) if len(aps) == num_classes else None,
        "mcc": float(mcc),
        "cohen_kappa": (accuracy - expected) / (1.0 - expected) if expected < 1.0 else 0.0,
    }
    if int(num_classes) == 2:
        result["roc_auc"] = _rank_auc((targets == 1).astype(np.int64), probs[:, 1])
        result["pr_auc"] = _average_precision(
            (targets == 1).astype(np.int64), probs[:, 1]
        )
    return result


def fit_temperature_scaling(
    logits: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Fit one positive scalar temperature on validation logits only."""

    torch = _require_torch()
    raw_logits = torch.tensor(np.asarray(logits), dtype=torch.float64)
    targets = torch.tensor(np.asarray(labels), dtype=torch.long)
    if raw_logits.ndim != 2 or int(raw_logits.shape[0]) != int(targets.shape[0]):
        raise ValueError("Temperature scaling logits/labels shapes differ.")
    if int(raw_logits.shape[0]) == 0:
        raise ValueError("Temperature scaling requires validation examples.")
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=int(max_iter), line_search_fn="strong_wolfe"
    )

    def closure() -> Any:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        loss = torch.nn.functional.cross_entropy(raw_logits / temperature, targets)
        loss.backward()
        return loss

    before_nll = float(torch.nn.functional.cross_entropy(raw_logits, targets).item())
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(1e-3, 1e3).item())
    after_nll = float(
        torch.nn.functional.cross_entropy(raw_logits / temperature, targets).item()
    )
    before_probs = torch.softmax(raw_logits, dim=1).numpy()
    after_probs = torch.softmax(raw_logits / temperature, dim=1).numpy()
    before_metrics = classification_metrics(
        targets.numpy(), before_probs, num_classes=int(raw_logits.shape[1])
    )
    after_metrics = classification_metrics(
        targets.numpy(), after_probs, num_classes=int(raw_logits.shape[1])
    )
    if not np.array_equal(before_probs.argmax(axis=1), after_probs.argmax(axis=1)):
        raise AssertionError("Positive temperature scaling unexpectedly changed argmax.")
    return {
        "schema_version": "temperature_scaling_v1",
        "status": "fit",
        "selection_split": "validation",
        "test_used_for_fit": False,
        "temperature": temperature,
        "num_examples": int(raw_logits.shape[0]),
        "num_classes": int(raw_logits.shape[1]),
        "nll_before": before_nll,
        "nll_after": after_nll,
        "ece_before": before_metrics["ece"],
        "ece_after": after_metrics["ece"],
        "brier_before": before_metrics["brier_score"],
        "brier_after": after_metrics["brier_score"],
        "argmax_invariant": True,
    }


__all__ = [
    "CHECKPOINT_BUNDLE_VERSION",
    "GNNOracle",
    "REQUIRED_CHECKPOINT_FILES",
    "TASTE_REQUIRED_CHECKPOINT_FILES",
    "classification_metrics",
    "expected_calibration_error",
    "fit_temperature_scaling",
    "load_gnn_checkpoint_bundle",
    "load_gnn_checkpoint_payloads",
    "save_gnn_checkpoint_bundle",
    "sha256_file",
    "update_checkpoint_sha256sums",
    "verify_checkpoint_bundle",
]
