"""Dataset-specific TasteMolNet T11 Ours full candidate/evaluation route.

The worker consumes a real 300-update T11 PPO checkpoint, generates two
train-only candidate pools, freezes a calibration-only prefix, and only then
opens held-out test.  It writes ``SEALED`` but never ``PASS``.  A separate
invocation replays the metrics from immutable pair rows and atomically
publishes the terminal result.
"""

from __future__ import annotations

import csv
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import errno
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
    enumerate_connected_hard_deletions,
)
from src.eval.full_candidate_pool import (
    CONNECTED_DELETION_PROMPT_MODE,
    FullPoolGenerationConfig,
    _build_lora_model,
    _build_tokenizer,
    build_generation_kwargs,
    generate_ids_with_sanitized_kwargs,
    render_generation_prompt,
    resolve_adapter_load_path,
    set_global_generation_seed,
)
from src.eval.frozen_threshold_manifest import load_shared_frozen_thresholds
from src.eval.molclr_node_embeddings import canonicalize_smiles
from src.eval.node_wasserstein_distance import (
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)
from src.models.llm_generator import clean_generated_smiles
from src.data.ppo_prompt_dataset import PPOPromptRecord
from src.data.tastemolnet_ppo import LABEL_MAP, TASTEMOLNET_PREPARED_FIELDS
from src.oracles.gnn_oracle import verify_checkpoint_bundle
from src.train.bace_gnn_ppo import validate_adapter_checkpoint_reload


DATASET = "TasteMolNet"
DATASET_ID = "tastemolnet"
METHOD = "Ours"
STAGE = "T11_OURS_FULL"
SOURCE_LABEL = 1
DESTINATIONS = (0, 2)
NUM_CLASSES = 3
K_MAX = 20
MIN_RULES = 10
TABLE_K = 10
PASS_MARKER = "[TASTE_OURS_PASS]"
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
DISTANCE_NAMESPACE = "tastemolnet_ours_full_wnode_v1"
GENERATION_CHUNK_RECEIPT_SCHEMA = "tastemolnet_t11_generation_chunk_receipt_v2"
GENERATION_MANIFEST_SCHEMA = "tastemolnet_t11_generation_manifest_v2"
PAIR_CHUNK_RECEIPT_SCHEMA = "tastemolnet_t11_pair_chunk_receipt_v2"
PAIR_MANIFEST_SCHEMA = "tastemolnet_t11_pair_manifest_v2"
GENERATION_REPLAY_ABS_TOL = 1e-7
GENERATION_ROW_FIELDS = frozenset(
    {
        "dataset",
        "method",
        "stage",
        "parent_id",
        "generation_parent_seed",
        "parent_smiles",
        "candidate_index",
        "candidate_id",
        "raw_output",
        "raw_fragment",
        "canonical_fragment",
        "parse_ok",
        "connected",
        "direct_substructure",
        "deletion_valid",
        "oracle_ok",
        "pred_before",
        "pred_after",
        "p_before",
        "p_after",
        "cf_drop",
        "cf_flip",
        "destination_label",
        "residual_smiles",
        "selected_match_index",
        "reward_total",
        "oracle_backend",
        "classifier_family",
        "rf_oracle_used",
        "oracle_checkpoint_hash",
        "split",
        "calibration_loaded",
        "test_loaded",
    }
)
PAIR_ROW_FIELDS = frozenset(
    {
        "dataset",
        "method",
        "stage",
        "split",
        "parent_id",
        "parent_smiles",
        "candidate_id",
        "canonical_fragment",
        "applicable",
        "num_matches",
        "num_valid_residuals",
        "pair_strict_flip",
        "best_match_index",
        "best_match_atom_indices",
        "residual_smiles",
        "pred_before",
        "pred_after",
        "p_before",
        "p_after",
        "p1_before",
        "p1_after",
        "cf_drop",
        "wnode_distance",
        "distance_for_selection",
        "destination_label",
        "failure_reason",
        "action_semantics_version",
        "match_selection_policy",
        "cf_mode",
        "source_label",
        "oracle_backend",
        "classifier_family",
        "rf_oracle_used",
        "oracle_checkpoint_hash",
        "temperature_calibration_hash",
        "feature_schema_hash",
        "molclr_checkpoint_hash",
        "distance_namespace",
    }
)
GENERATION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "status", "mode", "identity", "config",
        "parent_count", "candidate_count", "parent_inventory_sha256",
        "chunk_inventory", "chunk_inventory_sha256", "candidate_pool_sha256",
        "train_only", "calibration_loaded", "test_loaded", "rf_oracle_used",
        "row_schema_sha256", "resume_semantic_replay_required",
        "semantic_validation",
    }
)
PAIR_MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "status", "split", "evaluation_identity",
        "evaluation_identity_sha256", "parent_count", "candidate_count",
        "pair_count", "chunk_inventory", "chunk_inventory_sha256",
        "pair_details_sha256", "pair_details_bytes", "row_schema_sha256",
        "resume_semantic_replay_required", "semantic_validation",
    }
)
SELECTION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "dataset", "method", "stage", "status",
        "selection_frozen", "selector_fitted_on_calibration", "test_loaded",
        "test_used_for_selection", "frozen_at", "ordered_rule_ids",
        "ordered_rule_ids_sha256", "trace", "threshold_config_hash",
        "threshold_contract_file_sha256", "threshold_shared_across_methods",
        "theta_star", "cost_cap", "oracle_checkpoint_hash",
        "temperature_calibration_hash", "feature_schema_hash",
        "molclr_checkpoint_hash", "action_semantics_version",
        "distance_namespace", "selected_rules_sha256",
        "candidate_universe_sha256", "calibration_pair_details_sha256",
        "calibration_pair_manifest_sha256",
    }
)
TEST_ACCESS_FIELDS = frozenset(
    {
        "schema_version", "dataset", "method", "stage", "started_at",
        "declared_test_sha256", "selection_manifest_sha256",
        "selection_frozen_before_test", "test_used_for_selection",
    }
)
TEST_MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "status", "dataset", "method", "stage", "split",
        "started_at", "completed_at", "selection_manifest_sha256",
        "selection_frozen_before_test", "test_used_for_selection",
        "declared_test_sha256", "test_parent_ids_sha256",
        "oracle_checkpoint_hash", "temperature_calibration_hash",
        "feature_schema_hash", "molclr_checkpoint_hash",
        "threshold_config_hash", "threshold_contract_file_sha256",
        "parent_count", "candidate_count", "pair_count",
        "pair_details_sha256", "pair_manifest_sha256",
    }
)
GINE_PAYLOAD_FILES = frozenset(
    {
        "model.pt",
        "model_card.json",
        "feature_schema.json",
        "label_map.json",
        "split_manifest.json",
        "test_evaluation_status.json",
        "temperature_scaling.json",
    }
)


class TasteOursFullError(RuntimeError):
    """T11 violated a scientific, split, resume, or artifact contract."""


@dataclass(frozen=True, slots=True)
class TrainParent:
    parent_id: str
    smiles: str
    label: int
    split: str


@dataclass(frozen=True, slots=True)
class ThresholdContract:
    values: tuple[float, ...]
    theta_star: float
    cost_cap: float
    config_hash: str
    source: str
    source_split: str
    file_sha256: str


class TasteGINEScorer:
    """Loaded-once frozen three-class GINE scorer for T11."""

    def __init__(self, payloads: Mapping[str, bytes], *, device: str, batch_size: int) -> None:
        from src.data.molecular_graph_featurizer import MolecularFeatureSchema, MolecularGraphFeaturizer
        from src.oracles.gnn_oracle import GNNOracle

        if set(payloads) != GINE_PAYLOAD_FILES or any(not isinstance(value, bytes) or not value for value in payloads.values()):
            raise TasteOursFullError("T11 GINE payload inventory changed")
        card = json.loads(payloads["model_card.json"].decode("utf-8"))
        if (
            card.get("dataset") != DATASET_ID
            or card.get("oracle_backend") != "gnn"
            or card.get("rf_oracle_used") is not False
            or str(card.get("backbone") or "").lower() != "gine"
            or card.get("num_classes") != NUM_CLASSES
            or card.get("source_label") != SOURCE_LABEL
        ):
            raise TasteOursFullError("T11 loaded GINE contract changed")
        self._oracle = GNNOracle.from_payloads(dict(payloads), device=device, batch_size=int(batch_size))
        self._featurizer = MolecularGraphFeaturizer(
            MolecularFeatureSchema.from_dict(json.loads(payloads["feature_schema.json"].decode("utf-8")))
        )
        self.checkpoint_id = str(self._oracle.checkpoint_id)
        self.num_classes = int(self._oracle.num_classes)
        self.source_label = int(self._oracle.source_label)
        self.temperature = float(self._oracle.temperature)
        self._batch_size = int(batch_size)
        if self.num_classes != NUM_CLASSES or self.source_label != SOURCE_LABEL:
            raise TasteOursFullError("T11 loaded GINE class order changed")

    def score_smiles(self, values: Sequence[str]) -> list[dict[str, Any]]:
        from src.data.molecular_graph_dataset import MolecularGraphData

        if not values:
            raise TasteOursFullError("T11 GINE scorer cannot score an empty batch")
        graphs = []
        for position, value in enumerate(values):
            features = self._featurizer.featurize(str(value))
            graphs.append(
                MolecularGraphData(
                    x=features.node_features,
                    edge_index=features.edge_index,
                    edge_attr=features.edge_features,
                    y=SOURCE_LABEL,
                    molecule_id=f"private-t11-{position}",
                    smiles=features.canonical_smiles,
                    split="private_t11_evaluation",
                    graph_sha256=features.graph_sha256,
                )
            )
        return self._oracle.predict_records(graphs, batch_size=self._batch_size)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    name: str
    temperature: float
    top_p: float
    num_return_sequences: int
    seed: int
    max_new_tokens: int = 96

    @classmethod
    def base(cls) -> "GenerationConfig":
        return cls("base", 0.30, 0.90, 4, 7)

    @classmethod
    def high_temp(cls) -> "GenerationConfig":
        return cls("high_temp", 0.70, 0.90, 4, 13)


@dataclass(frozen=True, slots=True)
class OursAuthority:
    ppo_root: Path
    policy_root: Path
    base_model: Path
    checkpoint: Path
    train_path: Path
    calibration_path: Path
    test_path: Path
    molclr_root: Path
    molclr_checkpoint: Path
    threshold_path: Path
    policy_hash: str
    checkpoint_id: str
    dataset_hash: str
    temperature_calibration_hash: str
    feature_schema_hash: str
    feature_schema_file_sha256: str
    train_sha256: str
    calibration_sha256: str
    declared_test_sha256: str
    split_manifest_sha256: str
    molclr_checkpoint_sha256: str
    threshold: ThresholdContract

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": "tastemolnet_t11_full_identity_v1",
            "ppo_root": str(self.ppo_root),
            "policy_hash": self.policy_hash,
            "base_model": str(self.base_model),
            "checkpoint": str(self.checkpoint),
            "checkpoint_id": self.checkpoint_id,
            "dataset_hash": self.dataset_hash,
            "temperature_calibration_hash": self.temperature_calibration_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "feature_schema_file_sha256": self.feature_schema_file_sha256,
            "train_path": str(self.train_path),
            "train_sha256": self.train_sha256,
            "calibration_path": str(self.calibration_path),
            "calibration_sha256": self.calibration_sha256,
            "test_path": str(self.test_path),
            "declared_test_sha256": self.declared_test_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "molclr_checkpoint_sha256": self.molclr_checkpoint_sha256,
            "threshold_config_hash": self.threshold.config_hash,
            "threshold_contract_file_sha256": self.threshold.file_sha256,
            "theta_star": self.threshold.theta_star,
            "cost_cap": self.threshold.cost_cap,
            "threshold_source": self.threshold.source,
            "threshold_source_split": self.threshold.source_split,
            "threshold_fitted_on_test": False,
            "threshold_shared_across_methods": True,
            "test_used_for_selection": False,
            "cf_mode": "strict_flip",
            "generation": [asdict(GenerationConfig.base()), asdict(GenerationConfig.high_temp())],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path_like: str | Path) -> str:
    path = Path(path_like).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def read_json(path_like: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path_like).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TasteOursFullError(f"Expected one JSON object: {path_like}")
    return payload


def read_jsonl(path_like: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path_like).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TasteOursFullError(f"JSONL row {number} is not an object")
            rows.append(row)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a fresh terminal root without replacement on Linux."""

    if not sys.platform.startswith("linux"):
        if destination.exists():
            raise FileExistsError(f"terminal root exists: {destination}")
        os.rename(source, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise TasteOursFullError("T11 terminal publication requires renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if int(renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)) != 0:
        observed = ctypes.get_errno()
        if observed in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"terminal root exists: {destination}")
        raise OSError(observed, os.strerror(observed), str(destination))


def atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode())


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            dict(row), sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
        for row in rows
    ).encode()


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(path, jsonl_bytes(rows))


def _write_or_validate_json(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    expected = dict(payload)
    if path.is_file():
        if read_json(path) != expected:
            raise TasteOursFullError(f"T11 {label} changed across resume")
        return
    atomic_json(path, expected)


def _write_or_validate_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, label: str
) -> None:
    expected = jsonl_bytes(rows)
    if path.is_file():
        if path.read_bytes() != expected:
            raise TasteOursFullError(f"T11 {label} changed across resume")
        return
    _atomic_bytes(path, expected)


def _closed_chunk_receipt(
    *,
    schema_version: str,
    chunk: Path,
    rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "status": "PASS",
        "identity": dict(identity),
        "identity_sha256": stable_sha256(dict(identity)),
        "row_count": len(rows),
        "chunk_name": chunk.name,
        "chunk_sha256": sha256_file(chunk),
        "chunk_bytes": chunk.stat().st_size,
    }


def _commit_closed_chunk(
    *,
    chunk: Path,
    receipt_path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema_version: str,
    identity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if chunk.exists() or receipt_path.exists():
        raise TasteOursFullError(f"T11 {label} is only partially/unexpectedly present")
    atomic_jsonl(chunk, rows)
    receipt = _closed_chunk_receipt(
        schema_version=schema_version,
        chunk=chunk,
        rows=rows,
        identity=identity,
    )
    atomic_json(receipt_path, receipt)
    return receipt


def _load_closed_chunk(
    *,
    chunk: Path,
    receipt_path: Path,
    schema_version: str,
    identity: Mapping[str, Any],
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not chunk.is_file() or not receipt_path.is_file():
        raise TasteOursFullError(f"T11 {label} lacks its closed chunk/receipt pair")
    rows = read_jsonl(chunk)
    expected = _closed_chunk_receipt(
        schema_version=schema_version,
        chunk=chunk,
        rows=rows,
        identity=identity,
    )
    receipt = read_json(receipt_path)
    if receipt != expected:
        raise TasteOursFullError(f"T11 {label} receipt/hash changed")
    return rows, receipt


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise TasteOursFullError("Cannot serialize an empty standardized CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else ("" if value is None else value) for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(path, csv_bytes(rows))


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def load_threshold_contract(path_like: str | Path) -> ThresholdContract:
    path = Path(path_like).expanduser().resolve(strict=True)
    payload = read_json(path)
    try:
        shared = load_shared_frozen_thresholds(path)
    except (KeyError, TypeError, ValueError) as exc:
        raise TasteOursFullError(
            "threshold contract is not one shared, test-independent strict-flip authority"
        ) from exc
    if str(payload.get("dataset") or "").strip().lower() not in {"taste", DATASET_ID}:
        raise TasteOursFullError("threshold contract is not for TasteMolNet")
    raw = payload.get("thresholds")
    if not isinstance(raw, list) or not raw:
        raise TasteOursFullError("threshold contract lacks its frozen grid")
    try:
        values = tuple(float(value) for value in raw)
        theta_star = float(payload["theta_star"])
        cost_cap = float(payload["cost_cap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TasteOursFullError("threshold contract numeric fields are invalid") from exc
    if (
        any(not math.isfinite(value) or value < 0 for value in values)
        or any(right <= left for left, right in zip(values, values[1:]))
        or not math.isfinite(theta_star) or theta_star < 0 or theta_star not in values
        or not math.isfinite(cost_cap) or cost_cap < theta_star
    ):
        raise TasteOursFullError("threshold contract numeric gate failed")
    source_split = str(payload.get("threshold_source_split") or payload.get("selection_split") or "").strip().lower()
    if source_split not in {"calibration", "frozen_calibration", "legacy_frozen_calibration", "frozen_protocol", "existing_frozen_protocol", "legacy_frozen_protocol"}:
        raise TasteOursFullError("threshold contract is not calibration/frozen-protocol authority")
    if payload.get("test_used_for_selection") is not False:
        raise TasteOursFullError("threshold contract does not exclude test selection")
    if (
        payload.get("threshold_fitted_on_test") is not False
        or payload.get("selection_used_test") is True
        or payload.get("shared_across_methods") is not True
        or str(payload.get("cf_mode") or "strict_flip") != "strict_flip"
    ):
        raise TasteOursFullError(
            "threshold contract is not shared/test-independent strict-flip evidence"
        )
    config_hash = stable_sha256(list(values))
    if str(payload.get("threshold_config_hash") or "").lower() != config_hash:
        raise TasteOursFullError("threshold grid hash changed")
    source = str(payload.get("threshold_source") or "").strip()
    if not source:
        raise TasteOursFullError("threshold contract lacks its frozen source")
    if (
        tuple(float(value) for value in shared["thresholds"]) != values
        or float(shared["theta_star"]) != theta_star
        or float(shared["cost_cap"]) != cost_cap
        or str(shared["threshold_source"]) != source
        or str(shared["sha256"]) != sha256_file(path)
    ):
        raise TasteOursFullError("shared threshold loader and T11 contract differ")
    return ThresholdContract(values, theta_star, cost_cap, config_hash, source, source_split, sha256_file(path))


def load_prepared_split(
    path: Path, *, expected_split: str, expected_sha256: str
) -> list[TrainParent]:
    if expected_split not in {"train", "calibration", "test"}:
        raise TasteOursFullError("unsupported TasteMolNet split")
    resolved = path.expanduser().resolve(strict=True)
    if sha256_file(resolved) != expected_sha256:
        raise TasteOursFullError(f"{expected_split} split bytes changed")
    reader = csv.DictReader(io.StringIO(resolved.read_text(encoding="utf-8-sig"), newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != TASTEMOLNET_PREPARED_FIELDS:
        raise TasteOursFullError(f"{expected_split} split schema changed")
    parents: list[TrainParent] = []
    seen: set[str] = set()
    for number, row in enumerate(reader, start=2):
        parent_id = str(row.get("molecule_id") or "").strip()
        smiles = str(row.get("model_smiles") or "").strip()
        label_text = str(row.get("label") or "").strip()
        if (
            None in row or set(row) != set(TASTEMOLNET_PREPARED_FIELDS)
            or not parent_id or parent_id in seen or not smiles or label_text not in {"0", "1", "2"}
            or str(row.get("label_name") or "").strip() != LABEL_MAP[int(label_text)]
            or str(row.get("split") or "").strip() != expected_split
            or str(row.get("exclusion_reason") or "").strip()
        ):
            raise TasteOursFullError(f"{expected_split} row authority changed at {number}")
        seen.add(parent_id)
        if int(label_text) == SOURCE_LABEL:
            parents.append(TrainParent(parent_id, smiles, SOURCE_LABEL, expected_split))
    if not parents:
        raise TasteOursFullError(f"{expected_split} has no Sweet source cohort")
    return sorted(parents, key=lambda row: row.parent_id)


def load_authority(
    *, ppo_root: str | Path, gnn_checkpoint: str | Path, train_csv: str | Path,
    calibration_csv: str | Path, test_csv: str | Path, molclr_root: str | Path,
    molclr_checkpoint: str | Path, threshold_contract: str | Path,
) -> OursAuthority:
    ppo = Path(ppo_root).expanduser().resolve(strict=True)
    if (ppo / "PASS").read_text(encoding="utf-8").strip() != "[TASTE_T11_OURS_PPO_FULL_PASS]":
        raise TasteOursFullError("T11 downstream requires the exact full PPO PASS")
    manifest = read_json(ppo / "ppo_manifest.json")
    ppo_gate = read_json(ppo / "ppo_gate.json")
    ppo_oracle = read_json(ppo / "oracle_provenance.json")
    if (
        manifest.get("schema_version") != "tastemolnet_t11_ppo_manifest_v1"
        or manifest.get("status") != "PASS"
        or manifest.get("stage") != "T11_OURS_PPO_FULL"
        or manifest.get("optimizer_step_count") != 300
        or manifest.get("train_only") is not True
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
        or manifest.get("rf_oracle_used") is not False
        or ppo_gate.get("status") != "PASS"
        or ppo_gate.get("stage") != "T11_OURS_PPO_FULL"
        or manifest.get("ppo_gate_sha256") != sha256_file(ppo / "ppo_gate.json")
        or manifest.get("candidate_pool_sha256") != sha256_file(ppo / "candidate_pool.jsonl")
        or manifest.get("reward_manifest_sha256") != sha256_file(ppo / "reward_manifest.json")
        or manifest.get("oracle_manifest_sha256") != sha256_file(ppo / "oracle_provenance.json")
    ):
        raise TasteOursFullError("T11 PPO manifest is not a real train-only full run")
    if any(
        manifest.get(key) != value
        for key, value in ppo_gate.items()
        if key != "schema_version"
    ):
        raise TasteOursFullError("T11 PPO manifest differs from its terminal gate")
    resume_artifacts = manifest.get("resume_artifacts")
    checkpoints = resume_artifacts.get("checkpoints") if isinstance(resume_artifacts, dict) else None
    final_resume = resume_artifacts.get("final") if isinstance(resume_artifacts, dict) else None
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"50", "100", "150", "200", "250", "300"}:
        raise TasteOursFullError("T11 PPO lacks its complete periodic resume inventory")
    for step, evidence in checkpoints.items():
        checkpoint_root = ppo / f"checkpoint-{step}"
        if (
            not isinstance(evidence, dict)
            or Path(str(evidence.get("root") or "")).resolve(strict=True) != checkpoint_root
            or evidence.get("resume_manifest_sha256") != sha256_file(checkpoint_root / "stable_ppo_resume_manifest.json")
            or evidence.get("training_state_sha256") != sha256_file(checkpoint_root / "stable_ppo_training_state.pt")
            or evidence.get("candidate_pool_sha256") != sha256_file(checkpoint_root / "candidate_pool.jsonl")
        ):
            raise TasteOursFullError(f"T11 PPO checkpoint-{step} resume artifacts changed")
    if (
        not isinstance(final_resume, dict)
        or final_resume.get("completed_steps") != 300
        or final_resume.get("resume_manifest_sha256") != sha256_file(ppo / "stable_ppo_resume_manifest.json")
        or final_resume.get("training_state_sha256") != sha256_file(ppo / "stable_ppo_training_state.pt")
        or final_resume.get("candidate_pool_sha256") != sha256_file(ppo / "candidate_pool.jsonl")
    ):
        raise TasteOursFullError("T11 PPO final resumable state changed")
    policy = resolve_adapter_load_path(ppo)
    reload = validate_adapter_checkpoint_reload(policy)
    if reload.get("policy_checkpoint_hash") != manifest.get("policy_checkpoint_hash"):
        raise TasteOursFullError("T11 PPO adapter bytes changed after PASS")
    checkpoint = Path(gnn_checkpoint).expanduser().resolve(strict=True)
    verify_checkpoint_bundle(checkpoint, verify_hashes=True)
    card = read_json(checkpoint / "model_card.json")
    checkpoint_id = sha256_file(checkpoint / "model.pt")
    temperature_hash = sha256_file(checkpoint / "temperature_scaling.json")
    feature_file_hash = sha256_file(checkpoint / "feature_schema.json")
    feature_hash = str(read_json(checkpoint / "feature_schema.json").get("schema_sha256") or "")
    if not _is_sha256(feature_hash):
        raise TasteOursFullError("T11 GINE feature schema lacks its semantic hash")
    temperature = read_json(checkpoint / "temperature_scaling.json")
    if (
        temperature.get("status") != "fit"
        or not isinstance(temperature.get("temperature"), float)
        or not math.isfinite(float(temperature["temperature"]))
        or float(temperature["temperature"]) <= 0
        or temperature.get("selection_split") != "validation"
        or temperature.get("test_used_for_fit") is not False
        or temperature.get("argmax_invariant") is not True
    ):
        raise TasteOursFullError("T11 GINE temperature calibration contract changed")
    if read_json(checkpoint / "label_map.json") != {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}:
        raise TasteOursFullError("T11 GINE class order changed")
    test_status = read_json(checkpoint / "test_evaluation_status.json")
    if test_status.get("status") != "NOT_EVALUATED" or test_status.get("test_loaded") is not False:
        raise TasteOursFullError("T11 frozen GINE already consumed held-out test")
    if (
        card.get("dataset") != DATASET_ID or card.get("oracle_backend") != "gnn"
        or card.get("rf_oracle_used") is not False or str(card.get("backbone")).lower() != "gine"
        or card.get("num_classes") != NUM_CLASSES or card.get("source_label") != SOURCE_LABEL
        or card.get("checkpoint_id") != checkpoint_id or manifest.get("oracle_checkpoint_hash") != checkpoint_id
        or ppo_oracle.get("checkpoint_id") != checkpoint_id
        or ppo_oracle.get("temperature_calibration_hash") != temperature_hash
        or ppo_oracle.get("feature_schema_hash") != feature_hash
        or ppo_oracle.get("num_classes") != NUM_CLASSES
        or ppo_oracle.get("source_label") != SOURCE_LABEL
        or ppo_oracle.get("rf_oracle_used") is not False
    ):
        raise TasteOursFullError("T11 full and frozen GINE identities differ")
    train = Path(train_csv).expanduser().resolve(strict=True)
    calibration = Path(calibration_csv).expanduser().resolve(strict=True)
    test = Path(test_csv).expanduser().absolute()  # bytes intentionally unopened here
    split_path = checkpoint / "split_manifest.json"
    split = read_json(split_path)
    files = split.get("files")
    roles = split.get("roles")
    if (
        split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") != DATASET_ID
        or roles
        != {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        }
        or not isinstance(files, dict) or set(files) != {"train", "validation", "calibration", "test"}
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_evaluated_during_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
    ):
        raise TasteOursFullError("T11 split roles changed")
    declared = {role: str(files[role].get("sha256") or "").lower() for role in ("train", "calibration", "test")}
    if any(not _is_sha256(value) for value in declared.values()):
        raise TasteOursFullError("T11 split manifest lacks exact hashes")
    if sha256_file(train) != declared["train"] or sha256_file(calibration) != declared["calibration"]:
        raise TasteOursFullError("T11 train/calibration bytes differ from the GINE split authority")
    dataset_hash = str((split.get("train_manifest") or {}).get("dataset_fingerprint") or "").lower()
    if not _is_sha256(dataset_hash):
        raise TasteOursFullError("T11 split authority lacks its dataset fingerprint")
    base_model = Path(str(manifest.get("model_path") or "")).expanduser().resolve(strict=True)
    if manifest.get("train_split_sha256") != declared["train"]:
        raise TasteOursFullError("T11 PPO/full train identity differs")
    molclr_source = Path(molclr_root).expanduser().resolve(strict=True)
    molclr_ckpt = Path(molclr_checkpoint).expanduser().resolve(strict=True)
    threshold_path = Path(threshold_contract).expanduser().resolve(strict=True)
    return OursAuthority(
        ppo_root=ppo, policy_root=policy, base_model=base_model, checkpoint=checkpoint,
        train_path=train, calibration_path=calibration, test_path=test,
        molclr_root=molclr_source, molclr_checkpoint=molclr_ckpt, threshold_path=threshold_path,
        policy_hash=str(manifest["policy_checkpoint_hash"]), checkpoint_id=checkpoint_id,
        dataset_hash=dataset_hash,
        temperature_calibration_hash=temperature_hash,
        feature_schema_hash=feature_hash,
        feature_schema_file_sha256=feature_file_hash,
        train_sha256=declared["train"], calibration_sha256=declared["calibration"],
        declared_test_sha256=declared["test"], split_manifest_sha256=sha256_file(split_path),
        molclr_checkpoint_sha256=sha256_file(molclr_ckpt), threshold=load_threshold_contract(threshold_path),
    )


def _prompt(parent: TrainParent) -> PPOPromptRecord:
    return PPOPromptRecord(parent_index=0, parent_smiles=parent.smiles, label=SOURCE_LABEL, prompt="", raw_payload={"parent_id": parent.parent_id})


def _parent_generation_seed(config: GenerationConfig, parent_id: str) -> int:
    return (config.seed + int(stable_sha256({"mode": config.name, "parent_id": parent_id})[:8], 16)) % (2**32 - 1)


def _probabilities(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != NUM_CLASSES:
        return None
    if any(type(item) not in {int, float} for item in value):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if (
        any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in result)
        or not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-5)
    ):
        return None
    return result


def _predicted_label(probabilities: Sequence[float]) -> int:
    return max(range(len(probabilities)), key=lambda index: probabilities[index])


def _same_float(left: Any, right: float) -> bool:
    if type(left) not in {int, float}:
        return False
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _validate_generation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent: TrainParent,
    config: GenerationConfig,
    oracle_checkpoint_hash: str,
) -> None:
    if len(rows) != config.num_return_sequences:
        raise TasteOursFullError(f"{config.name} parent chunk candidate count changed")
    expected_seed = _parent_generation_seed(config, parent.parent_id)
    before_reference: tuple[float, ...] | None = None
    for index, row in enumerate(rows):
        if set(row) != GENERATION_ROW_FIELDS:
            raise TasteOursFullError(f"{config.name} generation row schema changed")
        raw = str(row.get("raw_output") or "")
        cleaned = clean_generated_smiles(raw)
        expected_fragment = canonicalize_smiles(cleaned) if cleaned else None
        candidate_id = "TASTEGEN_" + stable_sha256(
            {
                "mode": config.name,
                "parent": parent.parent_id,
                "index": index,
                "raw": raw,
            }
        )[:24].upper()
        outcomes = enumerate_connected_hard_deletions(
            parent.smiles,
            expected_fragment or cleaned,
            parent_id=parent.parent_id,
            candidate_id=candidate_id,
        )
        valid = [outcome for outcome in outcomes if outcome.valid and outcome.residual_smiles]
        before = _probabilities(row.get("p_before"))
        if (
            row.get("dataset") != DATASET
            or row.get("method") != METHOD
            or row.get("stage") != config.name
            or row.get("parent_id") != parent.parent_id
            or row.get("parent_smiles") != parent.smiles
            or type(row.get("generation_parent_seed")) is not int
            or row.get("generation_parent_seed") != expected_seed
            or type(row.get("candidate_index")) is not int
            or row.get("candidate_index") != index
            or row.get("candidate_id") != candidate_id
            or row.get("raw_fragment") != (cleaned or None)
            or row.get("canonical_fragment") != expected_fragment
            or row.get("parse_ok") is not (expected_fragment is not None)
            or row.get("connected") is not (
                expected_fragment is not None and "." not in expected_fragment
            )
            or row.get("direct_substructure") is not bool(outcomes)
            or row.get("deletion_valid") is not bool(valid)
            or row.get("oracle_ok") is not bool(valid)
            or before is None
            or type(row.get("pred_before")) is not int
            or row.get("pred_before") != _predicted_label(before)
            or row.get("pred_before") != SOURCE_LABEL
            or row.get("oracle_backend") != "gnn"
            or row.get("classifier_family") != "gine"
            or row.get("rf_oracle_used") is not False
            or row.get("oracle_checkpoint_hash") != oracle_checkpoint_hash
            or row.get("split") != "train"
            or row.get("calibration_loaded") is not False
            or row.get("test_loaded") is not False
        ):
            raise TasteOursFullError(f"{config.name} generation row authority changed")
        if before_reference is None:
            before_reference = before
        elif before != before_reference:
            raise TasteOursFullError(f"{config.name} parent prediction changed within chunk")
        if valid:
            after = _probabilities(row.get("p_after"))
            selected_index = row.get("selected_match_index")
            selected = next(
                (
                    outcome
                    for outcome in valid
                    if type(selected_index) is int and outcome.match_id == selected_index
                ),
                None,
            )
            if after is None or selected is None:
                raise TasteOursFullError(f"{config.name} selected deletion evidence changed")
            destination = _predicted_label(after)
            drop = before[SOURCE_LABEL] - after[SOURCE_LABEL]
            strict = destination in DESTINATIONS
            if (
                type(row.get("pred_after")) is not int
                or row.get("pred_after") != destination
                or not _same_float(row.get("cf_drop"), drop)
                or row.get("cf_flip") is not strict
                or (
                    strict
                    and type(row.get("destination_label")) is not int
                )
                or row.get("destination_label") != (destination if strict else None)
                or row.get("residual_smiles") != selected.residual_smiles
                or not _same_float(row.get("reward_total"), drop + (1.0 if strict else 0.0))
            ):
                raise TasteOursFullError(f"{config.name} generation score semantics changed")
        elif (
            row.get("p_after") != []
            or row.get("pred_after") is not None
            or row.get("cf_drop") is not None
            or row.get("cf_flip") is not False
            or row.get("destination_label") is not None
            or row.get("residual_smiles") is not None
            or row.get("selected_match_index") is not None
            or row.get("reward_total") is not None
        ):
            raise TasteOursFullError(f"{config.name} failed-generation semantics changed")


def _score_generation(
    *, parent: TrainParent, raw_outputs: Sequence[str], scorer: Any,
    config: GenerationConfig,
) -> list[dict[str, Any]]:
    parent_seed = _parent_generation_seed(config, parent.parent_id)
    before = scorer.score_smiles([parent.smiles])[0]
    if before.get("predicted_label") != SOURCE_LABEL:
        raise TasteOursFullError("generation cohort escaped frozen-GINE Sweet parents")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_outputs):
        cleaned = clean_generated_smiles(str(raw))
        fragment = canonicalize_smiles(cleaned) if cleaned else None
        candidate_id = "TASTEGEN_" + stable_sha256({"mode": config.name, "parent": parent.parent_id, "index": index, "raw": raw})[:24].upper()
        outcomes = enumerate_connected_hard_deletions(parent.smiles, fragment or cleaned, parent_id=parent.parent_id, candidate_id=candidate_id)
        valid = [outcome for outcome in outcomes if outcome.valid and outcome.residual_smiles]
        after_rows = scorer.score_smiles([str(outcome.residual_smiles) for outcome in valid]) if valid else []
        ranked: list[tuple[tuple[Any, ...], Any, Mapping[str, Any]]] = []
        for outcome, after in zip(valid, after_rows, strict=True):
            destination = int(after["predicted_label"])
            drop = float(before["probabilities"][SOURCE_LABEL]) - float(after["probabilities"][SOURCE_LABEL])
            strict = destination in DESTINATIONS
            ranked.append(((-int(strict), -drop, int(outcome.match_id)), outcome, after))
        ranked.sort(key=lambda item: item[0])
        best = ranked[0] if ranked else None
        outcome = best[1] if best else None
        after = best[2] if best else None
        destination = int(after["predicted_label"]) if after else None
        drop = float(before["probabilities"][1]) - float(after["probabilities"][1]) if after else None
        result.append({
            "dataset": DATASET, "method": METHOD, "stage": config.name, "parent_id": parent.parent_id,
            "generation_parent_seed": parent_seed,
            "parent_smiles": parent.smiles, "candidate_index": index, "candidate_id": candidate_id,
            "raw_output": str(raw), "raw_fragment": cleaned or None, "canonical_fragment": fragment,
            "parse_ok": fragment is not None, "connected": fragment is not None and "." not in fragment,
            "direct_substructure": bool(outcomes), "deletion_valid": bool(valid), "oracle_ok": after is not None,
            "pred_before": int(before["predicted_label"]), "pred_after": destination,
            "p_before": list(before["probabilities"]), "p_after": list(after["probabilities"]) if after else [],
            "cf_drop": drop, "cf_flip": destination in DESTINATIONS if destination is not None else False,
            "destination_label": destination if destination in DESTINATIONS else None,
            "residual_smiles": str(outcome.residual_smiles) if outcome else None,
            "selected_match_index": int(outcome.match_id) if outcome else None,
            "reward_total": (drop + (1.0 if destination in DESTINATIONS else 0.0)) if drop is not None else None,
            "oracle_backend": "gnn", "classifier_family": "gine", "rf_oracle_used": False,
            "oracle_checkpoint_hash": scorer.checkpoint_id, "split": "train",
            "calibration_loaded": False, "test_loaded": False,
        })
    return result


def _require_generation_semantic_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent: TrainParent,
    scorer: Any,
    config: GenerationConfig,
) -> None:
    replay_rows = _score_generation(
        parent=parent,
        raw_outputs=[str(row.get("raw_output") or "") for row in rows],
        scorer=scorer,
        config=config,
    )
    # CUDA kernels may change their final float32 reduction bits after a host
    # restart even when the frozen model, inputs, graph edits, predictions and
    # selected deletion are identical.  Keep every discrete scientific field
    # byte-exact and admit only a bounded probability/score round-off drift.
    float_fields = frozenset({"cf_drop", "reward_total"})
    vector_fields = frozenset({"p_before", "p_after"})

    def close_float(left: Any, right: Any) -> bool:
        if type(left) not in {int, float} or type(right) not in {int, float}:
            return False
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=GENERATION_REPLAY_ABS_TOL,
        )

    def symmetric_match_index_only(
        saved: Mapping[str, Any], replayed: Mapping[str, Any]
    ) -> bool:
        """Prove that two match IDs encode the same exact deletion result."""

        saved_index = saved.get("selected_match_index")
        replayed_index = replayed.get("selected_match_index")
        if saved_index == replayed_index:
            return True
        if type(saved_index) is not int or type(replayed_index) is not int:
            return False
        residual = saved.get("residual_smiles")
        fragment = saved.get("canonical_fragment")
        candidate_id = saved.get("candidate_id")
        if (
            type(residual) is not str
            or not residual
            or replayed.get("residual_smiles") != residual
            or type(fragment) is not str
            or not fragment
            or replayed.get("canonical_fragment") != fragment
            or type(candidate_id) is not str
            or not candidate_id
            or replayed.get("candidate_id") != candidate_id
            or saved.get("parent_id") != parent.parent_id
            or replayed.get("parent_id") != parent.parent_id
            or saved.get("parent_smiles") != parent.smiles
            or replayed.get("parent_smiles") != parent.smiles
        ):
            return False
        outcomes = enumerate_connected_hard_deletions(
            parent.smiles,
            fragment,
            parent_id=parent.parent_id,
            candidate_id=candidate_id,
        )
        residuals_by_match: dict[int, str] = {}
        for outcome in outcomes:
            if (
                not outcome.valid
                or not outcome.residual_smiles
                or type(outcome.match_id) is not int
                or outcome.match_id in residuals_by_match
            ):
                if type(outcome.match_id) is int and outcome.match_id in {
                    saved_index,
                    replayed_index,
                }:
                    return False
                continue
            residuals_by_match[outcome.match_id] = str(outcome.residual_smiles)
        return (
            residuals_by_match.get(saved_index) == residual
            and residuals_by_match.get(replayed_index) == residual
        )

    def row_matches(saved: Mapping[str, Any], replayed: Mapping[str, Any]) -> bool:
        if set(saved) != set(replayed):
            return False
        for field in vector_fields:
            left = saved.get(field)
            right = replayed.get(field)
            if not isinstance(left, list) or not isinstance(right, list):
                return False
            if len(left) != len(right) or any(
                not close_float(a, b) for a, b in zip(left, right, strict=True)
            ):
                return False
        for field in float_fields:
            left = saved.get(field)
            right = replayed.get(field)
            if left is None or right is None:
                if left is not right:
                    return False
            elif not close_float(left, right):
                return False
        numeric = float_fields | vector_fields
        for field in set(saved) - numeric:
            if saved[field] == replayed[field]:
                continue
            if field == "selected_match_index" and symmetric_match_index_only(
                saved, replayed
            ):
                continue
            return False
        return True

    if len(rows) != len(replay_rows) or any(
        not row_matches(saved, replayed)
        for saved, replayed in zip(rows, replay_rows, strict=True)
    ):
        raise TasteOursFullError(
            f"{config.name} resumed generation chunk fails frozen-GINE replay"
        )


def generate_mode_resumable(
    *, parents: Sequence[TrainParent], authority: OursAuthority, scorer: Any,
    output: Path, config: GenerationConfig,
) -> list[dict[str, Any]]:
    import torch

    root = output / "raw" / "generation" / config.name
    chunks = root / "parent_chunks"
    receipts = root / "parent_chunk_receipts"
    chunks.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    identity = stable_sha256({"authority": authority.identity(), "config": asdict(config), "parents": [(p.parent_id, p.smiles) for p in parents]})
    manifest_path = root / "generation_manifest.json"
    set_global_generation_seed(config.seed)
    tokenizer = _build_tokenizer(base_model_path=authority.base_model, trust_remote_code=True, local_files_only=True)
    model = _build_lora_model(base_model_path=authority.base_model, adapter_path=authority.policy_root, trust_remote_code=True, local_files_only=True)
    model_device = next(model.parameters()).device
    generation_config = FullPoolGenerationConfig(
        prompt_mode=CONNECTED_DELETION_PROMPT_MODE, num_return_sequences=config.num_return_sequences,
        generation_temperature=config.temperature, generation_top_p=config.top_p, generation_do_sample=True,
        max_new_tokens=config.max_new_tokens, batch_size=1, seed=config.seed,
        enable_parent_projection=False, enable_projected_cf_reward=False,
        enable_substructure_distance_reward=False,
    )
    all_rows: list[dict[str, Any]] = []
    chunk_inventory: list[dict[str, Any]] = []
    for position, parent in enumerate(parents):
        chunk = chunks / f"{position:08d}.jsonl"
        receipt_path = receipts / f"{position:08d}.json"
        chunk_identity = {
            "mode": config.name,
            "position": position,
            "parent_id": parent.parent_id,
            "parent_smiles": parent.smiles,
            "parent_split": parent.split,
            "generation_parent_seed": _parent_generation_seed(config, parent.parent_id),
            "generation_identity": identity,
            "oracle_checkpoint_hash": scorer.checkpoint_id,
            "row_schema_sha256": stable_sha256(sorted(GENERATION_ROW_FIELDS)),
        }
        if receipt_path.exists() and not chunk.is_file():
            raise TasteOursFullError(
                f"{config.name} parent-{position} receipt exists without its chunk"
            )
        if chunk.is_file() and not receipt_path.exists():
            rows = read_jsonl(chunk)
            _validate_generation_rows(
                rows,
                parent=parent,
                config=config,
                oracle_checkpoint_hash=scorer.checkpoint_id,
            )
            _require_generation_semantic_replay(
                rows,
                parent=parent,
                scorer=scorer,
                config=config,
            )
            receipt = _closed_chunk_receipt(
                schema_version=GENERATION_CHUNK_RECEIPT_SCHEMA,
                chunk=chunk,
                rows=rows,
                identity=chunk_identity,
            )
            atomic_json(receipt_path, receipt)
        elif chunk.is_file() or receipt_path.is_file():
            rows, receipt = _load_closed_chunk(
                chunk=chunk,
                receipt_path=receipt_path,
                schema_version=GENERATION_CHUNK_RECEIPT_SCHEMA,
                identity=chunk_identity,
                label=f"{config.name} parent-{position}",
            )
            _require_generation_semantic_replay(
                rows,
                parent=parent,
                scorer=scorer,
                config=config,
            )
        else:
            # Each parent owns a deterministic RNG stream.  Skipping completed
            # chunks on resume therefore produces byte-identical later chunks.
            set_global_generation_seed(_parent_generation_seed(config, parent.parent_id))
            prompt = render_generation_prompt(_prompt(parent), prompt_mode=CONNECTED_DELETION_PROMPT_MODE)
            encoded = tokenizer([prompt], return_tensors="pt", padding=True, truncation=False)
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            kwargs = build_generation_kwargs(encoded=encoded, tokenizer=tokenizer, config=generation_config)
            generated = generate_ids_with_sanitized_kwargs(model, kwargs, torch_module=torch)
            response = generated[:, encoded["input_ids"].shape[1]:]
            raw = tokenizer.batch_decode(response.detach().cpu().tolist(), skip_special_tokens=True)
            if len(raw) != config.num_return_sequences:
                raise TasteOursFullError(f"{config.name} generated the wrong candidate count")
            rows = _score_generation(parent=parent, raw_outputs=raw, scorer=scorer, config=config)
            _validate_generation_rows(
                rows,
                parent=parent,
                config=config,
                oracle_checkpoint_hash=scorer.checkpoint_id,
            )
            receipt = _commit_closed_chunk(
                chunk=chunk,
                receipt_path=receipt_path,
                rows=rows,
                schema_version=GENERATION_CHUNK_RECEIPT_SCHEMA,
                identity=chunk_identity,
                label=f"{config.name} parent-{position}",
            )
        _validate_generation_rows(
            rows,
            parent=parent,
            config=config,
            oracle_checkpoint_hash=scorer.checkpoint_id,
        )
        all_rows.extend(rows)
        chunk_inventory.append(
            {
                "position": position,
                "chunk": str(chunk.relative_to(root)),
                "receipt": str(receipt_path.relative_to(root)),
                "chunk_sha256": receipt["chunk_sha256"],
                "chunk_bytes": receipt["chunk_bytes"],
                "receipt_sha256": sha256_file(receipt_path),
            }
        )
        atomic_json(output / "checkpoint.json", {
            "schema_version": "tastemolnet_t11_stage_checkpoint_v1", "phase": f"GENERATION_{config.name.upper()}",
            "completed_parent_count": position + 1, "parent_count": len(parents), "identity": stable_sha256(authority.identity()),
        })
    candidate_pool = root / "candidate_pool.jsonl"
    _write_or_validate_jsonl(
        candidate_pool, all_rows, label=f"{config.name} aggregate candidate pool"
    )
    manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA, "status": "PASS", "mode": config.name,
        "identity": identity, "config": asdict(config), "parent_count": len(parents), "candidate_count": len(all_rows),
        "parent_inventory_sha256": stable_sha256(
            [(parent.parent_id, parent.smiles, parent.split) for parent in parents]
        ),
        "chunk_inventory": chunk_inventory,
        "chunk_inventory_sha256": stable_sha256(chunk_inventory),
        "candidate_pool_sha256": sha256_file(candidate_pool), "train_only": True,
        "calibration_loaded": False, "test_loaded": False, "rf_oracle_used": False,
        "row_schema_sha256": stable_sha256(sorted(GENERATION_ROW_FIELDS)),
        "resume_semantic_replay_required": True,
        "semantic_validation": "raw_canonical_deletion_frozen_gine_replay",
    }
    _write_or_validate_json(manifest_path, manifest, label=f"{config.name} manifest")
    return all_rows


def merge_candidate_modes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        fragment = canonicalize_smiles(str(row.get("canonical_fragment") or ""))
        if not fragment or not all(row.get(key) is True for key in ("parse_ok", "connected", "direct_substructure", "oracle_ok")):
            continue
        grouped.setdefault(fragment, []).append(row)
    universe: list[dict[str, Any]] = []
    for fragment in sorted(grouped):
        source = grouped[fragment]
        universe.append({
            "dataset": DATASET, "method": METHOD,
            "candidate_id": "TASTE_RULE_" + stable_sha256({"fragment": fragment})[:24].upper(),
            "canonical_fragment": fragment,
            "source_parent_ids": sorted({str(row["parent_id"]) for row in source}),
            "source_parent_count": len({str(row["parent_id"]) for row in source}),
            "source_modes": sorted({str(row["stage"]) for row in source}),
            "source_strict_flip_count": sum(row.get("cf_flip") is True for row in source),
            "source_cf_drop_mean": sum(float(row["cf_drop"]) for row in source) / len(source),
            "oracle_backend": "gnn", "classifier_family": "gine", "rf_oracle_used": False,
        })
    if len(universe) < K_MAX:
        raise TasteOursFullError(f"T11 needs at least 20 unique train candidates, found {len(universe)}")
    return universe


def build_pair_evaluation_identity(
    *,
    split: str,
    parents: Sequence[TrainParent],
    candidates: Sequence[Mapping[str, Any]],
    oracle_checkpoint_hash: str,
    temperature_calibration_hash: str,
    feature_schema_hash: str,
    molclr_checkpoint_hash: str,
    threshold_config_hash: str,
    threshold_contract_file_sha256: str,
) -> dict[str, Any]:
    if split not in {"calibration", "test"}:
        raise TasteOursFullError("T11 pair identity has an unsupported split")
    return {
        "schema_version": "tastemolnet_t11_pair_evaluation_identity_v2",
        "dataset": DATASET,
        "method": METHOD,
        "stage": STAGE,
        "split": split,
        "parent_inventory_sha256": stable_sha256(
            [(row.parent_id, row.smiles, row.split) for row in parents]
        ),
        "candidate_inventory_sha256": stable_sha256(
            [
                (str(row["candidate_id"]), str(row["canonical_fragment"]))
                for row in candidates
            ]
        ),
        "oracle_checkpoint_hash": oracle_checkpoint_hash,
        "temperature_calibration_hash": temperature_calibration_hash,
        "feature_schema_hash": feature_schema_hash,
        "molclr_checkpoint_hash": molclr_checkpoint_hash,
        "threshold_config_hash": threshold_config_hash,
        "threshold_contract_file_sha256": threshold_contract_file_sha256,
        "distance_line": DISTANCE_LINE,
        "distance_namespace": DISTANCE_NAMESPACE,
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        "cf_mode": "strict_flip",
        "pair_row_schema_sha256": stable_sha256(sorted(PAIR_ROW_FIELDS)),
    }


def evaluate_parent(
    *, parent: TrainParent, candidates: Sequence[Mapping[str, Any]], scorer: Any,
    distance: MolCLRNodeWassersteinDistance, split: str,
    evaluation_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    before = scorer.score_smiles([parent.smiles])[0]
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        fragment = str(candidate["canonical_fragment"])
        outcomes = enumerate_connected_hard_deletions(parent.smiles, fragment, parent_id=parent.parent_id, candidate_id=candidate_id)
        valid = [outcome for outcome in outcomes if outcome.valid and outcome.residual_smiles]
        after_rows = scorer.score_smiles([str(outcome.residual_smiles) for outcome in valid]) if valid else []
        finite: list[tuple[tuple[Any, ...], Any, Mapping[str, Any], float]] = []
        for outcome, after in zip(valid, after_rows, strict=True):
            destination = int(after["predicted_label"])
            if int(before["predicted_label"]) != SOURCE_LABEL or destination not in DESTINATIONS:
                continue
            drop = float(before["probabilities"][1]) - float(after["probabilities"][1])
            measured = distance.distance_for_action(parent.smiles, str(outcome.residual_smiles), action_context={
                "parent_id": parent.parent_id, "candidate_id": candidate_id, "match_index": int(outcome.match_id),
                "match_atom_indices": list(outcome.match_atom_indices), "teacher_sha256": scorer.checkpoint_id,
                "oracle_checkpoint_id": scorer.checkpoint_id, "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
            })
            value = measured.get("distance")
            if measured.get("ok") is True and value is not None and math.isfinite(float(value)) and float(value) >= 0:
                finite.append(((float(value), -drop, int(outcome.match_id)), outcome, after, float(value)))
        finite.sort(key=lambda item: item[0])
        best = finite[0] if finite else None
        outcome, after, value = (best[1], best[2], best[3]) if best else (None, None, None)
        destination = int(after["predicted_label"]) if after else None
        drop = float(before["probabilities"][1]) - float(after["probabilities"][1]) if after else None
        result.append({
            "dataset": DATASET, "method": METHOD, "stage": STAGE, "split": split,
            "parent_id": parent.parent_id, "parent_smiles": parent.smiles, "candidate_id": candidate_id,
            "canonical_fragment": fragment, "applicable": bool(outcomes), "num_matches": len(outcomes),
            "num_valid_residuals": len(valid), "pair_strict_flip": best is not None,
            "best_match_index": int(outcome.match_id) if outcome else None,
            "best_match_atom_indices": list(outcome.match_atom_indices) if outcome else [],
            "residual_smiles": str(outcome.residual_smiles) if outcome else None,
            "pred_before": int(before["predicted_label"]), "pred_after": destination,
            "p_before": list(before["probabilities"]),
            "p_after": list(after["probabilities"]) if after else [],
            "p1_before": float(before["probabilities"][1]), "p1_after": float(after["probabilities"][1]) if after else None,
            "cf_drop": drop, "wnode_distance": value, "distance_for_selection": value if value is not None else "+inf",
            "destination_label": destination if destination in DESTINATIONS else None,
            "failure_reason": None if best else ("no_substructure_match" if not outcomes else "no_valid_strict_flip_with_finite_wnode"),
            "action_semantics_version": CONNECTED_ACTION_SEMANTICS, "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
            "cf_mode": "strict_flip", "source_label": SOURCE_LABEL, "oracle_backend": "gnn",
            "classifier_family": "gine", "rf_oracle_used": False, "oracle_checkpoint_hash": scorer.checkpoint_id,
            "temperature_calibration_hash": evaluation_identity[
                "temperature_calibration_hash"
            ],
            "feature_schema_hash": evaluation_identity["feature_schema_hash"],
            "molclr_checkpoint_hash": evaluation_identity["molclr_checkpoint_hash"],
            "distance_namespace": evaluation_identity["distance_namespace"],
        })
    return result


def _require_pair_semantic_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent: TrainParent,
    candidates: Sequence[Mapping[str, Any]],
    scorer: Any,
    distance: MolCLRNodeWassersteinDistance,
    split: str,
    evaluation_identity: Mapping[str, Any],
) -> None:
    replay_rows = evaluate_parent(
        parent=parent,
        candidates=candidates,
        scorer=scorer,
        distance=distance,
        split=split,
        evaluation_identity=evaluation_identity,
    )
    if jsonl_bytes(replay_rows) != jsonl_bytes(rows):
        raise TasteOursFullError(
            f"T11 {split} resumed pair chunk fails GINE/WNode replay"
        )


def _validate_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent: TrainParent,
    candidates: Sequence[Mapping[str, Any]],
    split: str,
    evaluation_identity: Mapping[str, Any],
) -> None:
    if len(rows) != len(candidates):
        raise TasteOursFullError(f"T11 {split} pair chunk count changed")
    for candidate, row in zip(candidates, rows, strict=True):
        candidate_id = str(candidate["candidate_id"])
        fragment = str(candidate["canonical_fragment"])
        if set(row) != PAIR_ROW_FIELDS:
            raise TasteOursFullError(f"T11 {split} pair row schema changed")
        outcomes = enumerate_connected_hard_deletions(
            parent.smiles,
            fragment,
            parent_id=parent.parent_id,
            candidate_id=candidate_id,
        )
        valid = [outcome for outcome in outcomes if outcome.valid and outcome.residual_smiles]
        before = _probabilities(row.get("p_before"))
        if (
            row.get("dataset") != DATASET
            or row.get("method") != METHOD
            or row.get("stage") != STAGE
            or row.get("split") != split
            or row.get("parent_id") != parent.parent_id
            or row.get("parent_smiles") != parent.smiles
            or row.get("candidate_id") != candidate_id
            or row.get("canonical_fragment") != fragment
            or row.get("applicable") is not bool(outcomes)
            or type(row.get("num_matches")) is not int
            or row.get("num_matches") != len(outcomes)
            or type(row.get("num_valid_residuals")) is not int
            or row.get("num_valid_residuals") != len(valid)
            or type(row.get("pair_strict_flip")) is not bool
            or before is None
            or type(row.get("pred_before")) is not int
            or row.get("pred_before") != _predicted_label(before)
            or not _same_float(row.get("p1_before"), before[SOURCE_LABEL])
            or row.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS
            or row.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY
            or row.get("cf_mode") != "strict_flip"
            or type(row.get("source_label")) is not int
            or row.get("source_label") != SOURCE_LABEL
            or row.get("oracle_backend") != "gnn"
            or row.get("classifier_family") != "gine"
            or row.get("rf_oracle_used") is not False
            or row.get("oracle_checkpoint_hash")
            != evaluation_identity["oracle_checkpoint_hash"]
            or row.get("temperature_calibration_hash")
            != evaluation_identity["temperature_calibration_hash"]
            or row.get("feature_schema_hash")
            != evaluation_identity["feature_schema_hash"]
            or row.get("molclr_checkpoint_hash")
            != evaluation_identity["molclr_checkpoint_hash"]
            or row.get("distance_namespace") != DISTANCE_NAMESPACE
        ):
            raise TasteOursFullError(f"T11 {split} pair row authority changed")
        strict = row.get("pair_strict_flip") is True
        if strict:
            after = _probabilities(row.get("p_after"))
            selected_index = row.get("best_match_index")
            selected = next(
                (
                    outcome
                    for outcome in valid
                    if type(selected_index) is int and outcome.match_id == selected_index
                ),
                None,
            )
            if after is None or selected is None:
                raise TasteOursFullError(f"T11 {split} strict pair evidence changed")
            destination = _predicted_label(after)
            drop = before[SOURCE_LABEL] - after[SOURCE_LABEL]
            distance_value = row.get("wnode_distance")
            finite_distance = (
                float(distance_value)
                if type(distance_value) in {int, float}
                else math.nan
            )
            if (
                row.get("pred_before") != SOURCE_LABEL
                or destination not in DESTINATIONS
                or type(row.get("pred_after")) is not int
                or row.get("pred_after") != destination
                or type(row.get("destination_label")) is not int
                or row.get("destination_label") != destination
                or not _same_float(row.get("p1_after"), after[SOURCE_LABEL])
                or not _same_float(row.get("cf_drop"), drop)
                or not math.isfinite(finite_distance)
                or finite_distance < 0.0
                or not _same_float(row.get("distance_for_selection"), finite_distance)
                or row.get("best_match_atom_indices")
                != list(selected.match_atom_indices)
                or row.get("residual_smiles") != selected.residual_smiles
                or row.get("failure_reason") is not None
            ):
                raise TasteOursFullError(f"T11 {split} strict pair semantics changed")
        else:
            expected_failure = (
                "no_substructure_match"
                if not outcomes
                else "no_valid_strict_flip_with_finite_wnode"
            )
            if (
                row.get("p_after") != []
                or row.get("pred_after") is not None
                or row.get("p1_after") is not None
                or row.get("cf_drop") is not None
                or row.get("wnode_distance") is not None
                or row.get("distance_for_selection") != "+inf"
                or row.get("destination_label") is not None
                or row.get("best_match_index") is not None
                or row.get("best_match_atom_indices") != []
                or row.get("residual_smiles") is not None
                or row.get("failure_reason") != expected_failure
            ):
                raise TasteOursFullError(f"T11 {split} failed-pair semantics changed")


def evaluate_split_resumable(
    *, split: str, parents: Sequence[TrainParent], candidates: Sequence[Mapping[str, Any]],
    scorer: Any, distance: MolCLRNodeWassersteinDistance, output: Path,
    evaluation_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    chunks = output / "raw" / f"{split}_pair_chunks"
    receipts = output / "raw" / f"{split}_pair_chunk_receipts"
    chunks.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    ids = [str(row["candidate_id"]) for row in candidates]
    expected_parent_hash = stable_sha256(
        [(row.parent_id, row.smiles, row.split) for row in parents]
    )
    expected_candidate_hash = stable_sha256(
        [
            (str(row["candidate_id"]), str(row["canonical_fragment"]))
            for row in candidates
        ]
    )
    if (
        evaluation_identity.get("schema_version")
        != "tastemolnet_t11_pair_evaluation_identity_v2"
        or evaluation_identity.get("dataset") != DATASET
        or evaluation_identity.get("method") != METHOD
        or evaluation_identity.get("stage") != STAGE
        or evaluation_identity.get("split") != split
        or evaluation_identity.get("parent_inventory_sha256")
        != expected_parent_hash
        or evaluation_identity.get("candidate_inventory_sha256")
        != expected_candidate_hash
        or evaluation_identity.get("oracle_checkpoint_hash") != scorer.checkpoint_id
        or evaluation_identity.get("distance_line") != DISTANCE_LINE
        or evaluation_identity.get("distance_namespace") != DISTANCE_NAMESPACE
        or evaluation_identity.get("action_semantics_version")
        != CONNECTED_ACTION_SEMANTICS
        or evaluation_identity.get("match_selection_policy")
        != CONNECTED_MATCH_SELECTION_POLICY
        or evaluation_identity.get("cf_mode") != "strict_flip"
        or evaluation_identity.get("pair_row_schema_sha256")
        != stable_sha256(sorted(PAIR_ROW_FIELDS))
    ):
        raise TasteOursFullError(f"T11 {split} pair evaluation identity changed")
    result: list[dict[str, Any]] = []
    chunk_inventory: list[dict[str, Any]] = []
    for position, parent in enumerate(parents):
        chunk = chunks / f"{position:08d}.jsonl"
        receipt_path = receipts / f"{position:08d}.json"
        chunk_identity = {
            "split": split,
            "position": position,
            "parent_id": parent.parent_id,
            "parent_smiles": parent.smiles,
            "parent_split": parent.split,
            "candidate_ids": ids,
            "evaluation_identity_sha256": stable_sha256(dict(evaluation_identity)),
            "row_schema_sha256": stable_sha256(sorted(PAIR_ROW_FIELDS)),
        }
        if receipt_path.exists() and not chunk.is_file():
            raise TasteOursFullError(
                f"T11 {split} parent-{position} receipt exists without its chunk"
            )
        if chunk.is_file() and not receipt_path.exists():
            rows = read_jsonl(chunk)
            _validate_pair_rows(
                rows,
                parent=parent,
                candidates=candidates,
                split=split,
                evaluation_identity=evaluation_identity,
            )
            _require_pair_semantic_replay(
                rows,
                parent=parent,
                candidates=candidates,
                scorer=scorer,
                distance=distance,
                split=split,
                evaluation_identity=evaluation_identity,
            )
            receipt = _closed_chunk_receipt(
                schema_version=PAIR_CHUNK_RECEIPT_SCHEMA,
                chunk=chunk,
                rows=rows,
                identity=chunk_identity,
            )
            atomic_json(receipt_path, receipt)
        elif chunk.is_file() or receipt_path.is_file():
            rows, receipt = _load_closed_chunk(
                chunk=chunk,
                receipt_path=receipt_path,
                schema_version=PAIR_CHUNK_RECEIPT_SCHEMA,
                identity=chunk_identity,
                label=f"{split} parent-{position}",
            )
            _require_pair_semantic_replay(
                rows,
                parent=parent,
                candidates=candidates,
                scorer=scorer,
                distance=distance,
                split=split,
                evaluation_identity=evaluation_identity,
            )
        else:
            rows = evaluate_parent(
                parent=parent,
                candidates=candidates,
                scorer=scorer,
                distance=distance,
                split=split,
                evaluation_identity=evaluation_identity,
            )
            _validate_pair_rows(
                rows,
                parent=parent,
                candidates=candidates,
                split=split,
                evaluation_identity=evaluation_identity,
            )
            receipt = _commit_closed_chunk(
                chunk=chunk,
                receipt_path=receipt_path,
                rows=rows,
                schema_version=PAIR_CHUNK_RECEIPT_SCHEMA,
                identity=chunk_identity,
                label=f"{split} parent-{position}",
            )
        _validate_pair_rows(
            rows,
            parent=parent,
            candidates=candidates,
            split=split,
            evaluation_identity=evaluation_identity,
        )
        result.extend(rows)
        chunk_inventory.append(
            {
                "position": position,
                "chunk": str(chunk.relative_to(output / "raw")),
                "receipt": str(receipt_path.relative_to(output / "raw")),
                "chunk_sha256": receipt["chunk_sha256"],
                "chunk_bytes": receipt["chunk_bytes"],
                "receipt_sha256": sha256_file(receipt_path),
            }
        )
        atomic_json(output / "checkpoint.json", {
            "schema_version": "tastemolnet_t11_stage_checkpoint_v1", "phase": f"{split.upper()}_RUNNING",
            "completed_parent_count": position + 1, "parent_count": len(parents), "identity": stable_sha256(ids),
        })
    details_path = output / "raw" / f"{split}_pair_details.jsonl"
    _write_or_validate_jsonl(
        details_path, result, label=f"{split} aggregate pair details"
    )
    pair_manifest = {
        "schema_version": PAIR_MANIFEST_SCHEMA,
        "status": "PASS",
        "split": split,
        "evaluation_identity": dict(evaluation_identity),
        "evaluation_identity_sha256": stable_sha256(dict(evaluation_identity)),
        "parent_count": len(parents),
        "candidate_count": len(candidates),
        "pair_count": len(result),
        "chunk_inventory": chunk_inventory,
        "chunk_inventory_sha256": stable_sha256(chunk_inventory),
        "pair_details_sha256": sha256_file(details_path),
        "pair_details_bytes": details_path.stat().st_size,
        "row_schema_sha256": stable_sha256(sorted(PAIR_ROW_FIELDS)),
        "resume_semantic_replay_required": True,
        "semantic_validation": "connected_deletion_frozen_gine_molclr_replay",
    }
    _write_or_validate_json(
        output / "raw" / f"{split}_pair_manifest.json",
        pair_manifest,
        label=f"{split} pair manifest",
    )
    return result


def select_on_calibration(
    candidates: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], *, theta_star: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(candidate["candidate_id"]): [] for candidate in candidates}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id not in by_id or row.get("split") != "calibration":
            raise TasteOursFullError("calibration matrix escaped its candidate/split authority")
        by_id[candidate_id].append(row)
    candidate_by_id = {str(row["candidate_id"]): dict(row) for row in candidates}
    selected: list[str] = []
    covered_theta: set[str] = set()
    covered_strict: set[str] = set()
    trace: list[dict[str, Any]] = []
    remaining = set(by_id)
    while remaining and len(selected) < K_MAX:
        ranked: list[tuple[tuple[Any, ...], str, set[str], set[str], float]] = []
        for candidate_id in remaining:
            strict = {str(row["parent_id"]) for row in by_id[candidate_id] if row.get("pair_strict_flip") is True}
            theta = {str(row["parent_id"]) for row in by_id[candidate_id] if row.get("pair_strict_flip") is True and float(row["wnode_distance"]) <= theta_star}
            distances = [float(row["wnode_distance"]) for row in by_id[candidate_id] if row.get("pair_strict_flip") is True]
            mean = sum(distances) / len(distances) if distances else math.inf
            key = (-len(theta - covered_theta), -len(strict - covered_strict), -len(theta), -len(strict), mean, candidate_id)
            ranked.append((key, candidate_id, theta, strict, mean))
        _key, winner, theta, strict, mean = min(ranked)
        selected.append(winner)
        remaining.remove(winner)
        covered_theta.update(theta)
        covered_strict.update(strict)
        trace.append({"rank": len(selected), "candidate_id": winner, "cumulative_theta_coverage": len(covered_theta), "cumulative_strict_coverage": len(covered_strict), "mean_strict_distance": mean if math.isfinite(mean) else None})
    if len(selected) < MIN_RULES:
        raise TasteOursFullError("calibration selected fewer than ten rules")
    return [candidate_by_id[value] for value in selected], trace


def _median(values: Sequence[float]) -> float | str:
    if not values:
        return "N/A"
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def standardized_metrics(
    rows: Sequence[Mapping[str, Any]], ordered_ids: Sequence[str], threshold: ThresholdContract,
) -> dict[str, Any]:
    if not MIN_RULES <= len(ordered_ids) <= K_MAX or len(set(ordered_ids)) != len(ordered_ids):
        raise TasteOursFullError("frozen T11 prefix must contain 10..20 unique rules")
    by_parent: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        parent = str(row.get("parent_id") or "")
        candidate = str(row.get("candidate_id") or "")
        if row.get("split") != "test" or row.get("rf_oracle_used") is not False or candidate not in ordered_ids or not parent:
            raise TasteOursFullError("test matrix provenance escaped the frozen T11 prefix")
        if candidate in by_parent.setdefault(parent, {}):
            raise TasteOursFullError("duplicate T11 test pair")
        if row.get("pair_strict_flip") is True:
            value = row.get("wnode_distance")
            if value is None or not math.isfinite(float(value)) or float(value) < 0 or row.get("destination_label") not in DESTINATIONS:
                raise TasteOursFullError("strict T11 test pair lacks finite WNode/destination")
        elif row.get("wnode_distance") is not None:
            raise TasteOursFullError("non-strict T11 pair unexpectedly has WNode")
        by_parent[parent][candidate] = row
    if not by_parent or any(set(values) != set(ordered_ids) for values in by_parent.values()):
        raise TasteOursFullError("T11 test matrix is not the frozen Cartesian product")
    parents = sorted(by_parent)
    best: dict[str, tuple[float, float, str, int] | None] = {parent: None for parent in parents}
    applicable = {parent: False for parent in parents}
    prefix: list[dict[str, Any]] = []
    parent_best: list[dict[str, Any]] = []
    for k, candidate in enumerate(ordered_ids, start=1):
        for parent in parents:
            row = by_parent[parent][candidate]
            applicable[parent] |= bool(row.get("applicable"))
            if row.get("pair_strict_flip") is not True:
                continue
            value = (float(row["wnode_distance"]), -float(row.get("cf_drop") or 0), candidate, int(row["destination_label"]))
            if best[parent] is None or value[:3] < best[parent][:3]:
                best[parent] = value
        finite = [value for value in best.values() if value is not None]
        covered = [value for value in finite if value[0] <= threshold.theta_star]
        capped = [min(value[0], threshold.cost_cap) if value else threshold.cost_cap for value in best.values()]
        conditional = [value[0] for value in finite]
        drops = [-value[1] for value in covered]
        prefix.append({
            "dataset": DATASET, "method": METHOD, "k": k, "SuppCov": len(finite) / len(parents),
            "CCRCov": len(covered) / len(parents), "coverage": len(covered) / len(parents),
            "cost": sum(capped) / len(capped), "fixed_capped_mean_cost": sum(capped) / len(capped),
            "conditional_mean_cost": sum(conditional) / len(conditional) if conditional else "N/A",
            "conditional_median_cost": _median(conditional), "CFDrop": sum(drops) / len(drops) if drops else "N/A",
            "FlipRate": len(finite) / len(parents), "StructRed": "N/A", "CovRed": "N/A",
            "ValidRate": sum(applicable.values()) / len(parents), "AvgSize": "N/A",
            "applicable_rate": sum(applicable.values()) / len(parents), "effective_rule_count": len(ordered_ids),
        })
        for parent in parents:
            value = best[parent]
            parent_best.append({
                "dataset": DATASET, "method": METHOD, "k": k, "parent_id": parent,
                "best_distance": value[0] if value else "N/A", "capped_distance": min(value[0], threshold.cost_cap) if value else threshold.cost_cap,
                "best_candidate_id": value[2] if value else "N/A", "destination_label": value[3] if value else "N/A",
                "strict_recourse_available": value is not None, "theta_star_covered": value is not None and value[0] <= threshold.theta_star,
                "applicable": applicable[parent],
            })
    while len(prefix) < K_MAX:
        k = len(prefix) + 1
        prefix.append({**prefix[-1], "k": k, "plateau_after_effective_k": True})
        for parent in parents:
            previous = next(row for row in parent_best if row["parent_id"] == parent and row["k"] == len(ordered_ids))
            parent_best.append({**previous, "k": k, "plateau_after_effective_k": True})
    k10 = {row["parent_id"]: row for row in parent_best if row["k"] == TABLE_K}
    figure4 = []
    for value in threshold.values:
        coverage = sum(row["best_distance"] != "N/A" and float(row["best_distance"]) <= value for row in k10.values()) / len(k10)
        figure4.append({"dataset": DATASET, "method": METHOD, "k": TABLE_K, "threshold": value, "coverage": coverage, "CCRCov": coverage})
    final = {parent: next(row for row in reversed(parent_best) if row["parent_id"] == parent) for parent in parents}
    destinations = [row["destination_label"] for row in final.values() if row["destination_label"] != "N/A"]
    destination_rows = [{"dataset": DATASET, "method": METHOD, "destination_label": label, "count": destinations.count(label), "rate": destinations.count(label) / len(destinations) if destinations else "N/A", "denominator": len(destinations)} for label in DESTINATIONS]
    return {
        "prefix": prefix, "parent_best": parent_best,
        "figure3": [{"dataset": DATASET, "method": METHOD, "k": row["k"], "coverage": row["CCRCov"], "cost": row["cost"]} for row in prefix],
        "figure4": figure4, "table2": [dict(prefix[TABLE_K - 1])], "destination": destination_rows,
        "parent_count": len(parents), "pair_count": len(rows), "effective_rule_count": len(ordered_ids),
    }


def _artifact_rows(metrics: Mapping[str, Any]) -> dict[str, Sequence[Mapping[str, Any]]]:
    return {
        "figure3_coverage_vs_k.csv": metrics["figure3"],
        "figure4_coverage_vs_threshold.csv": metrics["figure4"],
        "prefix_metrics.csv": metrics["prefix"],
        "parent_best_distances.csv": metrics["parent_best"],
        "destination_distribution.csv": metrics["destination"],
        "table2_ours_k10.csv": metrics["table2"],
    }


def run_science(
    *, authority: OursAuthority, output_dir: str | Path, resume: bool, device: str,
    wnode_cache_db: str | Path, node_embedding_cache_dir: str | Path,
) -> dict[str, Any]:
    if device != "cuda:0":
        raise TasteOursFullError("T11 is bound to one visible logical cuda:0")
    output = Path(output_dir).expanduser().absolute()
    identity = authority.identity()
    identity_hash = stable_sha256(identity)
    if resume:
        if not output.is_dir() or not (output / "checkpoint.json").is_file():
            raise TasteOursFullError("--resume requires an existing T11 stage checkpoint")
        if read_json(output / "input_identity.json") != identity:
            raise TasteOursFullError("T11 resume identity changed")
        if (output / "SEALED").is_file():
            return read_json(output / "run_manifest.json")
    else:
        if output.exists():
            raise FileExistsError(f"fresh T11 science root exists: {output}")
        (output / "raw").mkdir(parents=True)
        atomic_json(output / "input_identity.json", identity)
        atomic_json(output / "checkpoint.json", {"schema_version": "tastemolnet_t11_stage_checkpoint_v1", "phase": "INITIALIZED", "identity": identity_hash})
    if (output / "PASS").exists():
        raise TasteOursFullError("T11 worker may not write PASS")
    payloads = {
        name: (authority.checkpoint / name).read_bytes()
        for name in GINE_PAYLOAD_FILES
    }
    scorer = TasteGINEScorer(payloads, device=device, batch_size=256)
    train = load_prepared_split(authority.train_path, expected_split="train", expected_sha256=authority.train_sha256)
    predictions = scorer.score_smiles([row.smiles for row in train])
    parents = [row for row, pred in zip(train, predictions, strict=True) if pred.get("predicted_label") == SOURCE_LABEL]
    if not parents:
        raise TasteOursFullError("T11 train generation cohort is empty")
    base = generate_mode_resumable(parents=parents, authority=authority, scorer=scorer, output=output, config=GenerationConfig.base())
    high = generate_mode_resumable(parents=parents, authority=authority, scorer=scorer, output=output, config=GenerationConfig.high_temp())
    universe = merge_candidate_modes([*base, *high])
    universe_path = output / "raw" / "candidate_universe.jsonl"
    _write_or_validate_jsonl(
        universe_path, universe, label="canonical candidate universe"
    )
    merge_manifest = {
        "schema_version": "tastemolnet_t11_merge_manifest_v2", "status": "PASS", "train_only": True,
        "base_count": len(base), "high_temp_count": len(high), "candidate_universe_count": len(universe),
        "base_candidate_pool_sha256": sha256_file(
            output / "raw" / "generation" / "base" / "candidate_pool.jsonl"
        ),
        "high_temp_candidate_pool_sha256": sha256_file(
            output / "raw" / "generation" / "high_temp" / "candidate_pool.jsonl"
        ),
        "base_generation_manifest_sha256": sha256_file(
            output / "raw" / "generation" / "base" / "generation_manifest.json"
        ),
        "high_temp_generation_manifest_sha256": sha256_file(
            output / "raw" / "generation" / "high_temp" / "generation_manifest.json"
        ),
        "candidate_universe_sha256": sha256_file(universe_path),
        "canonical_dedup_complete": True, "calibration_loaded": False, "test_loaded": False,
    }
    _write_or_validate_json(
        output / "raw" / "merge_manifest.json",
        merge_manifest,
        label="candidate merge manifest",
    )
    provider = MolCLRNodeWassersteinDistance(MolCLRNodeWassersteinConfig(
        molclr_root=authority.molclr_root, molclr_ckpt=authority.molclr_checkpoint,
        cache_db=Path(wnode_cache_db).expanduser().absolute(), node_emb_cache_dir=Path(node_embedding_cache_dir).expanduser().absolute(),
        device=device, distance_namespace=DISTANCE_NAMESPACE,
    ))
    try:
        selection_path = output / "raw" / "selection_manifest.json"
        selected_path = output / "raw" / "selected_rules.jsonl"
        calibration = load_prepared_split(
            authority.calibration_path,
            expected_split="calibration",
            expected_sha256=authority.calibration_sha256,
        )
        calibration_identity = build_pair_evaluation_identity(
            split="calibration",
            parents=calibration,
            candidates=universe,
            oracle_checkpoint_hash=authority.checkpoint_id,
            temperature_calibration_hash=authority.temperature_calibration_hash,
            feature_schema_hash=authority.feature_schema_hash,
            molclr_checkpoint_hash=authority.molclr_checkpoint_sha256,
            threshold_config_hash=authority.threshold.config_hash,
            threshold_contract_file_sha256=authority.threshold.file_sha256,
        )
        calibration_rows = evaluate_split_resumable(
            split="calibration",
            parents=calibration,
            candidates=universe,
            scorer=scorer,
            distance=provider,
            output=output,
            evaluation_identity=calibration_identity,
        )
        calibration_pair_manifest_sha256 = sha256_file(
            output / "raw" / "calibration_pair_manifest.json"
        )
        if selection_path.is_file():
            if not selected_path.is_file():
                raise TasteOursFullError("T11 committed selection lacks selected rules")
            selection = read_json(selection_path)
            selected = read_jsonl(selected_path)
            ordered = [str(row["candidate_id"]) for row in selected]
            if (
                set(selection) != SELECTION_MANIFEST_FIELDS
                or selection.get("schema_version") != "tastemolnet_t11_selection_v2"
                or selection.get("dataset") != DATASET
                or selection.get("method") != METHOD
                or selection.get("stage") != STAGE
                or selection.get("status") != "FROZEN" or selection.get("selection_frozen") is not True
                or selection.get("selector_fitted_on_calibration") is not True
                or selection.get("test_loaded") is not False or selection.get("test_used_for_selection") is not False
                or selection.get("threshold_config_hash") != authority.threshold.config_hash
                or selection.get("threshold_contract_file_sha256") != authority.threshold.file_sha256
                or selection.get("theta_star") != authority.threshold.theta_star
                or selection.get("cost_cap") != authority.threshold.cost_cap
                or selection.get("oracle_checkpoint_hash") != authority.checkpoint_id
                or selection.get("temperature_calibration_hash")
                != authority.temperature_calibration_hash
                or selection.get("feature_schema_hash") != authority.feature_schema_hash
                or selection.get("molclr_checkpoint_hash") != authority.molclr_checkpoint_sha256
                or selection.get("threshold_shared_across_methods") is not True
                or selection.get("action_semantics_version")
                != CONNECTED_ACTION_SEMANTICS
                or selection.get("distance_namespace") != DISTANCE_NAMESPACE
                or selection.get("ordered_rule_ids") != ordered or selection.get("selected_rules_sha256") != sha256_file(selected_path)
                or selection.get("candidate_universe_sha256") != sha256_file(output / "raw" / "candidate_universe.jsonl")
                or selection.get("calibration_pair_details_sha256") != sha256_file(output / "raw" / "calibration_pair_details.jsonl")
                or selection.get("calibration_pair_manifest_sha256")
                != calibration_pair_manifest_sha256
                or selection.get("ordered_rule_ids_sha256") != stable_sha256(ordered)
            ):
                raise TasteOursFullError("T11 frozen selection changed")
            replay_selected, replay_trace = select_on_calibration(
                universe,
                calibration_rows,
                theta_star=authority.threshold.theta_star,
            )
            if replay_selected != selected or replay_trace != selection.get("trace"):
                raise TasteOursFullError("T11 frozen selector replay changed")
        else:
            selected, trace = select_on_calibration(universe, calibration_rows, theta_star=authority.threshold.theta_star)
            ordered = [str(row["candidate_id"]) for row in selected]
            if selected_path.is_file():
                # A crash may occur after selected rules are durable but before
                # the selection manifest commit point.  Recompute on the same
                # calibration matrix and adopt only exact rows.
                if read_jsonl(selected_path) != selected:
                    raise TasteOursFullError("T11 uncommitted selected-rule bytes changed")
            else:
                atomic_jsonl(selected_path, selected)
            selection = {
                "schema_version": "tastemolnet_t11_selection_v2", "dataset": DATASET, "method": METHOD, "stage": STAGE,
                "status": "FROZEN", "selection_frozen": True, "selector_fitted_on_calibration": True,
                "test_loaded": False, "test_used_for_selection": False, "frozen_at": utc_now(),
                "ordered_rule_ids": ordered, "ordered_rule_ids_sha256": stable_sha256(ordered), "trace": trace,
                "threshold_config_hash": authority.threshold.config_hash, "oracle_checkpoint_hash": authority.checkpoint_id,
                "threshold_contract_file_sha256": authority.threshold.file_sha256,
                "threshold_shared_across_methods": True,
                "theta_star": authority.threshold.theta_star, "cost_cap": authority.threshold.cost_cap,
                "temperature_calibration_hash": authority.temperature_calibration_hash,
                "feature_schema_hash": authority.feature_schema_hash,
                "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
                "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                "distance_namespace": DISTANCE_NAMESPACE,
                "selected_rules_sha256": sha256_file(selected_path), "candidate_universe_sha256": sha256_file(output / "raw" / "candidate_universe.jsonl"),
                "calibration_pair_details_sha256": sha256_file(output / "raw" / "calibration_pair_details.jsonl"),
                "calibration_pair_manifest_sha256": calibration_pair_manifest_sha256,
            }
            atomic_json(selection_path, selection)
            atomic_json(output / "checkpoint.json", {"schema_version": "tastemolnet_t11_stage_checkpoint_v1", "phase": "CALIBRATION_SELECTION_FROZEN", "identity": identity_hash, "selection_manifest_sha256": sha256_file(selection_path)})
        selection_sha = sha256_file(selection_path)
        test_access_path = output / "raw" / "test_access_receipt.json"
        if test_access_path.is_file():
            test_access = read_json(test_access_path)
            if (
                set(test_access) != TEST_ACCESS_FIELDS
                or test_access.get("schema_version") != "tastemolnet_t11_test_access_v1"
                or test_access.get("dataset") != DATASET
                or test_access.get("method") != METHOD
                or test_access.get("stage") != STAGE
                or test_access.get("declared_test_sha256")
                != authority.declared_test_sha256
                or test_access.get("selection_manifest_sha256") != selection_sha
                or test_access.get("selection_frozen_before_test") is not True
                or test_access.get("test_used_for_selection") is not False
                or not str(test_access.get("started_at") or "")
            ):
                raise TasteOursFullError("T11 test-access resume receipt changed")
        else:
            test_access = {
                "schema_version": "tastemolnet_t11_test_access_v1",
                "dataset": DATASET,
                "method": METHOD,
                "stage": STAGE,
                "started_at": utc_now(),
                "declared_test_sha256": authority.declared_test_sha256,
                "selection_manifest_sha256": selection_sha,
                "selection_frozen_before_test": True,
                "test_used_for_selection": False,
            }
            # This fsynced receipt is committed before the only test loader.
            atomic_json(test_access_path, test_access)
        test_started = str(test_access["started_at"])
        test = load_prepared_split(authority.test_path, expected_split="test", expected_sha256=authority.declared_test_sha256)
        test_identity = build_pair_evaluation_identity(
            split="test",
            parents=test,
            candidates=selected,
            oracle_checkpoint_hash=authority.checkpoint_id,
            temperature_calibration_hash=authority.temperature_calibration_hash,
            feature_schema_hash=authority.feature_schema_hash,
            molclr_checkpoint_hash=authority.molclr_checkpoint_sha256,
            threshold_config_hash=authority.threshold.config_hash,
            threshold_contract_file_sha256=authority.threshold.file_sha256,
        )
        test_rows = evaluate_split_resumable(
            split="test",
            parents=test,
            candidates=selected,
            scorer=scorer,
            distance=provider,
            output=output,
            evaluation_identity=test_identity,
        )
        provider_stats = provider.stats_dict()
    finally:
        provider.close()
    test_manifest_path = output / "raw" / "test_evaluation_manifest.json"
    expected_test_manifest = {
        "schema_version": "tastemolnet_t11_test_manifest_v2", "status": "PASS",
        "dataset": DATASET, "method": METHOD, "stage": STAGE, "split": "test",
        "started_at": test_started,
        "selection_manifest_sha256": selection_sha,
        "selection_frozen_before_test": True, "test_used_for_selection": False,
        "declared_test_sha256": authority.declared_test_sha256,
        "test_parent_ids_sha256": stable_sha256(sorted(row.parent_id for row in test)),
        "oracle_checkpoint_hash": authority.checkpoint_id,
        "temperature_calibration_hash": authority.temperature_calibration_hash,
        "feature_schema_hash": authority.feature_schema_hash,
        "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
        "threshold_config_hash": authority.threshold.config_hash,
        "threshold_contract_file_sha256": authority.threshold.file_sha256,
        "parent_count": len(test), "candidate_count": len(selected), "pair_count": len(test_rows),
        "pair_details_sha256": sha256_file(output / "raw" / "test_pair_details.jsonl"),
        "pair_manifest_sha256": sha256_file(
            output / "raw" / "test_pair_manifest.json"
        ),
    }
    if test_manifest_path.is_file():
        test_manifest = read_json(test_manifest_path)
        if (
            set(test_manifest) != TEST_MANIFEST_FIELDS
            or {key: test_manifest.get(key) for key in expected_test_manifest}
            != expected_test_manifest
            or not str(test_manifest.get("completed_at") or "")
        ):
            raise TasteOursFullError("T11 test evaluation manifest changed")
    else:
        test_manifest = {**expected_test_manifest, "completed_at": utc_now()}
        atomic_json(test_manifest_path, test_manifest)
    metrics = standardized_metrics(test_rows, ordered, authority.threshold)
    for name, rows in _artifact_rows(metrics).items():
        atomic_csv(output / name, rows)
    atomic_json(output / "prefix_metrics.json", metrics["prefix"])
    test_parent_ids_hash = stable_sha256(sorted(row.parent_id for row in test))
    common = {
        "dataset": DATASET, "method": METHOD, "stage": STAGE, "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL, "oracle_backend": "gnn", "classifier_family": "gine",
        "rf_oracle_used": False, "oracle_checkpoint": str(authority.checkpoint),
        "oracle_hash": authority.checkpoint_id, "oracle_checkpoint_hash": authority.checkpoint_id,
        "dataset_hash": authority.dataset_hash, "test_parent_ids_sha256": test_parent_ids_hash,
        "temperature_calibration_hash": authority.temperature_calibration_hash,
        "feature_schema_hash": authority.feature_schema_hash,
        "feature_schema_file_sha256": authority.feature_schema_file_sha256,
        "policy_checkpoint_hash": authority.policy_hash, "distance_line": DISTANCE_LINE,
        "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256, "cf_mode": "strict_flip",
        "threshold_config_hash": authority.threshold.config_hash, "test_split_hash": authority.declared_test_sha256,
        "threshold_contract_file_sha256": authority.threshold.file_sha256,
        "theta_star": authority.threshold.theta_star, "cost_cap": authority.threshold.cost_cap,
        "test_used_for_selection": False, "threshold_fitted_on_test": False,
        "threshold_shared_across_methods": True,
        "raw_output_root": str(output),
    }
    summary = {
        "schema_version": "tastemolnet_t11_summary_v1", **common, "status": "SEALED",
        "frozen": True, "artifacts_frozen": True, "raw_output_complete": True,
        "selection_frozen_before_test": True, "calibration_loaded": True, "test_loaded": True,
        "effective_rule_count": metrics["effective_rule_count"], "parent_count": metrics["parent_count"],
        "pair_count": metrics["pair_count"], "destination_labels": list(DESTINATIONS),
        "base_generation": asdict(GenerationConfig.base()), "high_temp_generation": asdict(GenerationConfig.high_temp()),
        "ppo_updates": 300, "resume_supported": True, "distance_provider_stats": provider_stats,
    }
    oracle_manifest = {
        "schema_version": "tastemolnet_t11_oracle_manifest_v1", **common, "status": "SEALED",
        "same_frozen_gine_for_generation_calibration_test": True, "calibration_loaded_for_training": False,
        "test_loaded_for_training": False, "selection_frozen_before_test": True,
    }
    evaluation = {
        "schema_version": "tastemolnet_t11_evaluation_manifest_v1", **common, "status": "SEALED",
        "selection_manifest_sha256": selection_sha, "test_evaluation_manifest_sha256": sha256_file(output / "raw" / "test_evaluation_manifest.json"),
        "selection_frozen_before_test": True, "full_cartesian_test_pairs": True,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "oracle_manifest.json", oracle_manifest)
    atomic_json(output / "evaluation_manifest.json", evaluation)
    immutable = [
        *_artifact_rows(metrics),
        "input_identity.json",
        "prefix_metrics.json",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "raw/generation/base/candidate_pool.jsonl",
        "raw/generation/base/generation_manifest.json",
        "raw/generation/high_temp/candidate_pool.jsonl",
        "raw/generation/high_temp/generation_manifest.json",
        "raw/candidate_universe.jsonl",
        "raw/merge_manifest.json",
        "raw/calibration_pair_details.jsonl",
        "raw/calibration_pair_manifest.json",
        "raw/selected_rules.jsonl",
        "raw/selection_manifest.json",
        "raw/test_access_receipt.json",
        "raw/test_pair_details.jsonl",
        "raw/test_pair_manifest.json",
        "raw/test_evaluation_manifest.json",
    ]
    inventory = {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size} for name in immutable}
    source_evidence_names = sorted(
        str(path.relative_to(output))
        for directory in (
            output / "raw" / "generation" / "base" / "parent_chunks",
            output / "raw" / "generation" / "base" / "parent_chunk_receipts",
            output / "raw" / "generation" / "high_temp" / "parent_chunks",
            output / "raw" / "generation" / "high_temp" / "parent_chunk_receipts",
            output / "raw" / "calibration_pair_chunks",
            output / "raw" / "calibration_pair_chunk_receipts",
            output / "raw" / "test_pair_chunks",
            output / "raw" / "test_pair_chunk_receipts",
        )
        for path in directory.iterdir()
        if path.is_file()
    )
    source_evidence = {
        name: {
            "sha256": sha256_file(output / name),
            "bytes": (output / name).stat().st_size,
        }
        for name in source_evidence_names
    }
    atomic_json(output / "freeze_manifest.json", {
        "schema_version": "tastemolnet_t11_freeze_manifest_v2", **common,
        "status": "SEALED", "files": inventory,
        "inventory_sha256": stable_sha256(inventory),
        "source_evidence_files": source_evidence,
        "source_evidence_inventory_sha256": stable_sha256(source_evidence),
        "sealed_at": utc_now(),
    })
    run_manifest = {
        "schema_version": "tastemolnet_t11_run_manifest_v2", **common, "status": "SEALED", "state": "SEALED",
        "run_complete": False, "frozen": True, "raw_output_complete": True,
        "source_artifacts_complete": True, "selection_frozen_before_test": True,
        "independent_terminal_verification_required": True, "worker_wrote_pass": False,
        "freeze_manifest_sha256": sha256_file(output / "freeze_manifest.json"), "sealed_at": utc_now(),
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    _atomic_bytes(output / "SEALED", b"SEALED\n")
    atomic_json(output / "checkpoint.json", {"schema_version": "tastemolnet_t11_stage_checkpoint_v1", "phase": "SEALED", "identity": identity_hash})
    return run_manifest


def _relative_file(root: Path, value: Any, *, label: str) -> Path:
    if type(value) is not str:
        raise TasteOursFullError(f"T11 {label} path is not a string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise TasteOursFullError(f"T11 {label} path escaped its root")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise TasteOursFullError(f"T11 {label} is not one physical file")
    return path


def _verify_generation_mode(
    *,
    science: Path,
    mode: str,
    input_identity: Mapping[str, Any],
    authorized_train_parents: Sequence[TrainParent],
) -> tuple[list[dict[str, Any]], list[TrainParent], set[str]]:
    config = GenerationConfig.base() if mode == "base" else GenerationConfig.high_temp()
    root = science / "raw" / "generation" / mode
    manifest_path = root / "generation_manifest.json"
    manifest = read_json(manifest_path)
    inventory = manifest.get("chunk_inventory")
    if (
        set(manifest) != GENERATION_MANIFEST_FIELDS
        or manifest.get("schema_version") != GENERATION_MANIFEST_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("mode") != mode
        or manifest.get("config") != asdict(config)
        or manifest.get("train_only") is not True
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
        or manifest.get("rf_oracle_used") is not False
        or manifest.get("row_schema_sha256")
        != stable_sha256(sorted(GENERATION_ROW_FIELDS))
        or manifest.get("resume_semantic_replay_required") is not True
        or manifest.get("semantic_validation")
        != "raw_canonical_deletion_frozen_gine_replay"
        or not isinstance(inventory, list)
        or not inventory
        or manifest.get("chunk_inventory_sha256") != stable_sha256(inventory)
    ):
        raise TasteOursFullError(f"T11 {mode} generation manifest changed")
    provisional_parents: list[TrainParent] = []
    for position, item in enumerate(inventory):
        if not isinstance(item, dict) or item.get("position") != position:
            raise TasteOursFullError(f"T11 {mode} chunk inventory order changed")
        receipt_path = _relative_file(root, item.get("receipt"), label=f"{mode} receipt")
        receipt = read_json(receipt_path)
        chunk_identity = receipt.get("identity")
        if (
            receipt.get("schema_version") != GENERATION_CHUNK_RECEIPT_SCHEMA
            or receipt.get("status") != "PASS"
            or not isinstance(chunk_identity, dict)
            or chunk_identity.get("mode") != mode
            or chunk_identity.get("position") != position
            or type(chunk_identity.get("parent_id")) is not str
            or not chunk_identity.get("parent_id")
            or type(chunk_identity.get("parent_smiles")) is not str
            or not chunk_identity.get("parent_smiles")
            or chunk_identity.get("parent_split") != "train"
            or chunk_identity.get("oracle_checkpoint_hash")
            != input_identity.get("checkpoint_id")
            or chunk_identity.get("row_schema_sha256")
            != stable_sha256(sorted(GENERATION_ROW_FIELDS))
        ):
            raise TasteOursFullError(f"T11 {mode} chunk identity changed")
        provisional_parents.append(
            TrainParent(
                str(chunk_identity["parent_id"]),
                str(chunk_identity["parent_smiles"]),
                SOURCE_LABEL,
                "train",
            )
        )
    authorized = {
        (parent.parent_id, parent.smiles, parent.split)
        for parent in authorized_train_parents
    }
    observed = [
        (parent.parent_id, parent.smiles, parent.split)
        for parent in provisional_parents
    ]
    if len(set(observed)) != len(observed) or any(item not in authorized for item in observed):
        raise TasteOursFullError(
            f"T11 {mode} generation parents escaped the frozen train split"
        )
    generation_identity = stable_sha256(
        {
            "authority": dict(input_identity),
            "config": asdict(config),
            "parents": [(parent.parent_id, parent.smiles) for parent in provisional_parents],
        }
    )
    if (
        manifest.get("identity") != generation_identity
        or manifest.get("parent_count") != len(provisional_parents)
        or manifest.get("parent_inventory_sha256")
        != stable_sha256(
            [
                (parent.parent_id, parent.smiles, parent.split)
                for parent in provisional_parents
            ]
        )
    ):
        raise TasteOursFullError(f"T11 {mode} generation parent authority changed")
    result: list[dict[str, Any]] = []
    evidence_names: set[str] = set()
    replay_inventory: list[dict[str, Any]] = []
    for position, (item, parent) in enumerate(zip(inventory, provisional_parents, strict=True)):
        expected_chunk_name = f"parent_chunks/{position:08d}.jsonl"
        expected_receipt_name = f"parent_chunk_receipts/{position:08d}.json"
        if (
            item.get("chunk") != expected_chunk_name
            or item.get("receipt") != expected_receipt_name
        ):
            raise TasteOursFullError(f"T11 {mode} chunk path changed")
        chunk = _relative_file(root, expected_chunk_name, label=f"{mode} chunk")
        receipt_path = _relative_file(
            root, expected_receipt_name, label=f"{mode} receipt"
        )
        chunk_identity = {
            "mode": mode,
            "position": position,
            "parent_id": parent.parent_id,
            "parent_smiles": parent.smiles,
            "parent_split": "train",
            "generation_parent_seed": _parent_generation_seed(config, parent.parent_id),
            "generation_identity": generation_identity,
            "oracle_checkpoint_hash": input_identity["checkpoint_id"],
            "row_schema_sha256": stable_sha256(sorted(GENERATION_ROW_FIELDS)),
        }
        rows, receipt = _load_closed_chunk(
            chunk=chunk,
            receipt_path=receipt_path,
            schema_version=GENERATION_CHUNK_RECEIPT_SCHEMA,
            identity=chunk_identity,
            label=f"{mode} parent-{position}",
        )
        _validate_generation_rows(
            rows,
            parent=parent,
            config=config,
            oracle_checkpoint_hash=str(input_identity["checkpoint_id"]),
        )
        replay_item = {
            "position": position,
            "chunk": expected_chunk_name,
            "receipt": expected_receipt_name,
            "chunk_sha256": receipt["chunk_sha256"],
            "chunk_bytes": receipt["chunk_bytes"],
            "receipt_sha256": sha256_file(receipt_path),
        }
        replay_inventory.append(replay_item)
        result.extend(rows)
        evidence_names.update(
            {
                str(chunk.relative_to(science)),
                str(receipt_path.relative_to(science)),
            }
        )
    candidate_pool = root / "candidate_pool.jsonl"
    if (
        replay_inventory != inventory
        or candidate_pool.read_bytes() != jsonl_bytes(result)
        or manifest.get("candidate_count") != len(result)
        or manifest.get("candidate_pool_sha256") != sha256_file(candidate_pool)
    ):
        raise TasteOursFullError(f"T11 {mode} generation cannot be replayed")
    return result, provisional_parents, evidence_names


def _verify_pair_artifacts(
    *,
    science: Path,
    split: str,
    candidates: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    authorized_parents: Sequence[TrainParent],
) -> tuple[list[dict[str, Any]], list[TrainParent], set[str]]:
    raw = science / "raw"
    manifest_path = raw / f"{split}_pair_manifest.json"
    manifest = read_json(manifest_path)
    inventory = manifest.get("chunk_inventory")
    if (
        set(manifest) != PAIR_MANIFEST_FIELDS
        or manifest.get("schema_version") != PAIR_MANIFEST_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("split") != split
        or not isinstance(inventory, list)
        or not inventory
        or manifest.get("chunk_inventory_sha256") != stable_sha256(inventory)
        or manifest.get("row_schema_sha256")
        != stable_sha256(sorted(PAIR_ROW_FIELDS))
        or manifest.get("resume_semantic_replay_required") is not True
        or manifest.get("semantic_validation")
        != "connected_deletion_frozen_gine_molclr_replay"
    ):
        raise TasteOursFullError(f"T11 {split} pair manifest changed")
    parents: list[TrainParent] = []
    for position, item in enumerate(inventory):
        if not isinstance(item, dict) or item.get("position") != position:
            raise TasteOursFullError(f"T11 {split} pair inventory order changed")
        receipt_path = _relative_file(
            raw, item.get("receipt"), label=f"{split} pair receipt"
        )
        receipt = read_json(receipt_path)
        identity = receipt.get("identity")
        if (
            receipt.get("schema_version") != PAIR_CHUNK_RECEIPT_SCHEMA
            or receipt.get("status") != "PASS"
            or not isinstance(identity, dict)
            or identity.get("split") != split
            or identity.get("position") != position
            or identity.get("parent_split") != split
            or type(identity.get("parent_id")) is not str
            or not identity.get("parent_id")
            or type(identity.get("parent_smiles")) is not str
            or not identity.get("parent_smiles")
            or identity.get("row_schema_sha256")
            != stable_sha256(sorted(PAIR_ROW_FIELDS))
        ):
            raise TasteOursFullError(f"T11 {split} pair receipt identity changed")
        parents.append(
            TrainParent(
                str(identity["parent_id"]),
                str(identity["parent_smiles"]),
                SOURCE_LABEL,
                split,
            )
        )
    if len({parent.parent_id for parent in parents}) != len(parents):
        raise TasteOursFullError(f"T11 {split} pair manifest repeats a parent")
    if parents != list(authorized_parents):
        raise TasteOursFullError(
            f"T11 {split} pair parents differ from the frozen split"
        )
    evaluation_identity = build_pair_evaluation_identity(
        split=split,
        parents=parents,
        candidates=candidates,
        oracle_checkpoint_hash=str(run_manifest.get("oracle_checkpoint_hash") or ""),
        temperature_calibration_hash=str(
            run_manifest.get("temperature_calibration_hash") or ""
        ),
        feature_schema_hash=str(run_manifest.get("feature_schema_hash") or ""),
        molclr_checkpoint_hash=str(run_manifest.get("molclr_checkpoint_hash") or ""),
        threshold_config_hash=str(run_manifest.get("threshold_config_hash") or ""),
        threshold_contract_file_sha256=str(
            run_manifest.get("threshold_contract_file_sha256") or ""
        ),
    )
    if (
        manifest.get("evaluation_identity") != evaluation_identity
        or manifest.get("evaluation_identity_sha256")
        != stable_sha256(evaluation_identity)
        or evaluation_identity.get("pair_row_schema_sha256")
        != stable_sha256(sorted(PAIR_ROW_FIELDS))
    ):
        raise TasteOursFullError(f"T11 {split} pair evaluation identity changed")
    result: list[dict[str, Any]] = []
    evidence_names: set[str] = set()
    replay_inventory: list[dict[str, Any]] = []
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    for position, (item, parent) in enumerate(zip(inventory, parents, strict=True)):
        expected_chunk_name = f"{split}_pair_chunks/{position:08d}.jsonl"
        expected_receipt_name = f"{split}_pair_chunk_receipts/{position:08d}.json"
        if (
            item.get("chunk") != expected_chunk_name
            or item.get("receipt") != expected_receipt_name
        ):
            raise TasteOursFullError(f"T11 {split} pair chunk path changed")
        chunk = _relative_file(raw, expected_chunk_name, label=f"{split} pair chunk")
        receipt_path = _relative_file(
            raw, expected_receipt_name, label=f"{split} pair receipt"
        )
        chunk_identity = {
            "split": split,
            "position": position,
            "parent_id": parent.parent_id,
            "parent_smiles": parent.smiles,
            "parent_split": split,
            "candidate_ids": candidate_ids,
            "evaluation_identity_sha256": stable_sha256(evaluation_identity),
            "row_schema_sha256": stable_sha256(sorted(PAIR_ROW_FIELDS)),
        }
        rows, receipt = _load_closed_chunk(
            chunk=chunk,
            receipt_path=receipt_path,
            schema_version=PAIR_CHUNK_RECEIPT_SCHEMA,
            identity=chunk_identity,
            label=f"{split} parent-{position}",
        )
        _validate_pair_rows(
            rows,
            parent=parent,
            candidates=candidates,
            split=split,
            evaluation_identity=evaluation_identity,
        )
        replay_item = {
            "position": position,
            "chunk": expected_chunk_name,
            "receipt": expected_receipt_name,
            "chunk_sha256": receipt["chunk_sha256"],
            "chunk_bytes": receipt["chunk_bytes"],
            "receipt_sha256": sha256_file(receipt_path),
        }
        replay_inventory.append(replay_item)
        result.extend(rows)
        evidence_names.update(
            {
                str(chunk.relative_to(science)),
                str(receipt_path.relative_to(science)),
            }
        )
    details = raw / f"{split}_pair_details.jsonl"
    if (
        replay_inventory != inventory
        or details.read_bytes() != jsonl_bytes(result)
        or manifest.get("parent_count") != len(parents)
        or manifest.get("candidate_count") != len(candidates)
        or manifest.get("pair_count") != len(result)
        or manifest.get("pair_details_sha256") != sha256_file(details)
        or manifest.get("pair_details_bytes") != details.stat().st_size
    ):
        raise TasteOursFullError(f"T11 {split} pair artifacts cannot be replayed")
    return result, parents, evidence_names


def verify_and_publish(*, science_root: str | Path, final_root: str | Path, threshold_contract: str | Path) -> dict[str, Any]:
    science = Path(science_root).expanduser().resolve(strict=True)
    final = Path(final_root).expanduser().absolute()
    if final.exists():
        raise FileExistsError(f"T11 final root must be fresh: {final}")
    if (science / "SEALED").read_text(encoding="utf-8") != "SEALED\n" or (science / "PASS").exists():
        raise TasteOursFullError("T11 verifier requires SEALED worker output without PASS")
    run_manifest = read_json(science / "run_manifest.json")
    freeze = read_json(science / "freeze_manifest.json")
    if (
        run_manifest.get("schema_version") != "tastemolnet_t11_run_manifest_v2"
        or run_manifest.get("dataset") != DATASET
        or run_manifest.get("method") != METHOD
        or run_manifest.get("stage") != STAGE
        or run_manifest.get("status") != "SEALED"
        or run_manifest.get("state") != "SEALED"
        or run_manifest.get("rf_oracle_used") is not False
        or run_manifest.get("selection_frozen_before_test") is not True
        or run_manifest.get("test_used_for_selection") is not False
        or run_manifest.get("threshold_fitted_on_test") is not False
        or run_manifest.get("threshold_shared_across_methods") is not True
        or run_manifest.get("worker_wrote_pass") is not False
        or run_manifest.get("freeze_manifest_sha256")
        != sha256_file(science / "freeze_manifest.json")
    ):
        raise TasteOursFullError("T11 sealed manifest changed")
    files = freeze.get("files")
    source_files = freeze.get("source_evidence_files")
    required_files = set(_artifact_rows({
            "figure3": [], "figure4": [], "prefix": [], "parent_best": [],
            "destination": [], "table2": [],
        })) | {
        "input_identity.json",
        "prefix_metrics.json",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "raw/generation/base/candidate_pool.jsonl",
        "raw/generation/base/generation_manifest.json",
        "raw/generation/high_temp/candidate_pool.jsonl",
        "raw/generation/high_temp/generation_manifest.json",
        "raw/candidate_universe.jsonl",
        "raw/merge_manifest.json",
        "raw/calibration_pair_details.jsonl",
        "raw/calibration_pair_manifest.json",
        "raw/selected_rules.jsonl",
        "raw/selection_manifest.json",
        "raw/test_access_receipt.json",
        "raw/test_pair_details.jsonl",
        "raw/test_pair_manifest.json",
        "raw/test_evaluation_manifest.json",
    }
    if (
        freeze.get("schema_version") != "tastemolnet_t11_freeze_manifest_v2"
        or freeze.get("status") != "SEALED"
        or not isinstance(files, dict)
        or set(files) != required_files
        or freeze.get("inventory_sha256") != stable_sha256(files)
        or not isinstance(source_files, dict)
        or not source_files
        or freeze.get("source_evidence_inventory_sha256")
        != stable_sha256(source_files)
    ):
        raise TasteOursFullError("T11 immutable inventory changed")
    for label, inventory in (("publication", files), ("source evidence", source_files)):
        for name, row in inventory.items():
            path = _relative_file(science, name, label=label)
            if (
                not isinstance(row, dict)
                or row.get("sha256") != sha256_file(path)
                or row.get("bytes") != path.stat().st_size
            ):
                raise TasteOursFullError(f"T11 {label} inventory changed")
    input_identity = read_json(science / "input_identity.json")
    if (
        run_manifest.get("oracle_checkpoint_hash")
        != input_identity.get("checkpoint_id")
        or run_manifest.get("dataset_hash") != input_identity.get("dataset_hash")
        or run_manifest.get("temperature_calibration_hash")
        != input_identity.get("temperature_calibration_hash")
        or run_manifest.get("feature_schema_hash")
        != input_identity.get("feature_schema_hash")
        or run_manifest.get("feature_schema_file_sha256")
        != input_identity.get("feature_schema_file_sha256")
        or run_manifest.get("policy_checkpoint_hash")
        != input_identity.get("policy_hash")
        or run_manifest.get("oracle_checkpoint")
        != input_identity.get("checkpoint")
        or run_manifest.get("molclr_checkpoint_hash")
        != input_identity.get("molclr_checkpoint_sha256")
        or run_manifest.get("threshold_config_hash")
        != input_identity.get("threshold_config_hash")
        or run_manifest.get("threshold_contract_file_sha256")
        != input_identity.get("threshold_contract_file_sha256")
        or run_manifest.get("test_split_hash")
        != input_identity.get("declared_test_sha256")
        or run_manifest.get("theta_star") != input_identity.get("theta_star")
        or run_manifest.get("cost_cap") != input_identity.get("cost_cap")
        or input_identity.get("threshold_fitted_on_test") is not False
        or input_identity.get("threshold_shared_across_methods") is not True
        or input_identity.get("test_used_for_selection") is not False
        or input_identity.get("cf_mode") != "strict_flip"
    ):
        raise TasteOursFullError("T11 sealed input identity changed")
    try:
        train_authority = load_prepared_split(
            Path(str(input_identity["train_path"])),
            expected_split="train",
            expected_sha256=str(input_identity["train_sha256"]),
        )
        calibration_authority = load_prepared_split(
            Path(str(input_identity["calibration_path"])),
            expected_split="calibration",
            expected_sha256=str(input_identity["calibration_sha256"]),
        )
        test_authority = load_prepared_split(
            Path(str(input_identity["test_path"])),
            expected_split="test",
            expected_sha256=str(input_identity["declared_test_sha256"]),
        )
    except KeyError as exc:
        raise TasteOursFullError("T11 sealed split identity is incomplete") from exc
    base_rows, base_parents, base_evidence = _verify_generation_mode(
        science=science,
        mode="base",
        input_identity=input_identity,
        authorized_train_parents=train_authority,
    )
    high_rows, high_parents, high_evidence = _verify_generation_mode(
        science=science,
        mode="high_temp",
        input_identity=input_identity,
        authorized_train_parents=train_authority,
    )
    if base_parents != high_parents:
        raise TasteOursFullError("T11 base/high generation cohorts differ")
    replay_universe = merge_candidate_modes([*base_rows, *high_rows])
    universe_path = science / "raw" / "candidate_universe.jsonl"
    universe = read_jsonl(universe_path)
    merge = read_json(science / "raw" / "merge_manifest.json")
    expected_merge = {
        "schema_version": "tastemolnet_t11_merge_manifest_v2",
        "status": "PASS",
        "train_only": True,
        "base_count": len(base_rows),
        "high_temp_count": len(high_rows),
        "candidate_universe_count": len(replay_universe),
        "base_candidate_pool_sha256": sha256_file(
            science / "raw" / "generation" / "base" / "candidate_pool.jsonl"
        ),
        "high_temp_candidate_pool_sha256": sha256_file(
            science / "raw" / "generation" / "high_temp" / "candidate_pool.jsonl"
        ),
        "base_generation_manifest_sha256": sha256_file(
            science / "raw" / "generation" / "base" / "generation_manifest.json"
        ),
        "high_temp_generation_manifest_sha256": sha256_file(
            science / "raw" / "generation" / "high_temp" / "generation_manifest.json"
        ),
        "candidate_universe_sha256": sha256_file(universe_path),
        "canonical_dedup_complete": True,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    if merge != expected_merge or replay_universe != universe:
        raise TasteOursFullError("T11 base/high generation merge cannot be replayed")
    selection = read_json(science / "raw" / "selection_manifest.json")
    selected = read_jsonl(science / "raw" / "selected_rules.jsonl")
    ordered = [str(row["candidate_id"]) for row in selected]
    calibration_rows, _calibration_parents, calibration_evidence = (
        _verify_pair_artifacts(
            science=science,
            split="calibration",
            candidates=universe,
            run_manifest=run_manifest,
            authorized_parents=calibration_authority,
        )
    )
    test_access = read_json(science / "raw" / "test_access_receipt.json")
    test_manifest = read_json(science / "raw" / "test_evaluation_manifest.json")
    if (
        set(selection) != SELECTION_MANIFEST_FIELDS
        or selection.get("schema_version") != "tastemolnet_t11_selection_v2"
        or selection.get("dataset") != DATASET
        or selection.get("method") != METHOD
        or selection.get("stage") != STAGE
        or selection.get("status") != "FROZEN" or selection.get("selection_frozen") is not True
        or selection.get("selector_fitted_on_calibration") is not True
        or selection.get("test_loaded") is not False or selection.get("test_used_for_selection") is not False
        or selection.get("threshold_config_hash") != run_manifest.get("threshold_config_hash")
        or selection.get("threshold_contract_file_sha256") != run_manifest.get("threshold_contract_file_sha256")
        or selection.get("theta_star") != run_manifest.get("theta_star")
        or selection.get("cost_cap") != run_manifest.get("cost_cap")
        or selection.get("oracle_checkpoint_hash") != run_manifest.get("oracle_checkpoint_hash")
        or selection.get("temperature_calibration_hash")
        != run_manifest.get("temperature_calibration_hash")
        or selection.get("feature_schema_hash")
        != run_manifest.get("feature_schema_hash")
        or selection.get("molclr_checkpoint_hash") != run_manifest.get("molclr_checkpoint_hash")
        or selection.get("threshold_shared_across_methods") is not True
        or selection.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS
        or selection.get("distance_namespace") != DISTANCE_NAMESPACE
        or selection.get("ordered_rule_ids") != ordered or selection.get("selected_rules_sha256") != sha256_file(science / "raw" / "selected_rules.jsonl")
        or selection.get("ordered_rule_ids_sha256") != stable_sha256(ordered)
        or selection.get("candidate_universe_sha256") != sha256_file(science / "raw" / "candidate_universe.jsonl")
        or selection.get("calibration_pair_details_sha256") != sha256_file(science / "raw" / "calibration_pair_details.jsonl")
        or selection.get("calibration_pair_manifest_sha256")
        != sha256_file(science / "raw" / "calibration_pair_manifest.json")
        or set(test_access) != TEST_ACCESS_FIELDS
        or test_access.get("schema_version") != "tastemolnet_t11_test_access_v1"
        or test_access.get("dataset") != DATASET
        or test_access.get("method") != METHOD
        or test_access.get("stage") != STAGE
        or test_access.get("declared_test_sha256")
        != run_manifest.get("test_split_hash")
        or test_manifest.get("selection_manifest_sha256") != sha256_file(science / "raw" / "selection_manifest.json")
        or test_access.get("selection_manifest_sha256") != sha256_file(science / "raw" / "selection_manifest.json")
        or test_access.get("selection_frozen_before_test") is not True
        or test_access.get("test_used_for_selection") is not False
        or set(test_manifest) != TEST_MANIFEST_FIELDS
        or test_manifest.get("schema_version") != "tastemolnet_t11_test_manifest_v2"
        or test_manifest.get("status") != "PASS"
        or test_manifest.get("dataset") != DATASET
        or test_manifest.get("method") != METHOD
        or test_manifest.get("stage") != STAGE
        or test_manifest.get("split") != "test"
        or test_manifest.get("selection_frozen_before_test") is not True
        or test_manifest.get("test_used_for_selection") is not False
        or test_manifest.get("declared_test_sha256")
        != run_manifest.get("test_split_hash")
        or test_manifest.get("oracle_checkpoint_hash")
        != run_manifest.get("oracle_checkpoint_hash")
        or test_manifest.get("temperature_calibration_hash")
        != run_manifest.get("temperature_calibration_hash")
        or test_manifest.get("feature_schema_hash")
        != run_manifest.get("feature_schema_hash")
        or test_manifest.get("molclr_checkpoint_hash")
        != run_manifest.get("molclr_checkpoint_hash")
        or test_manifest.get("threshold_config_hash")
        != run_manifest.get("threshold_config_hash")
        or test_manifest.get("threshold_contract_file_sha256")
        != run_manifest.get("threshold_contract_file_sha256")
        or not str(selection.get("frozen_at") or "")
        or not str(test_manifest.get("started_at") or "")
        or not str(test_manifest.get("completed_at") or "")
        or str(test_manifest.get("completed_at") or "")
        < str(test_manifest.get("started_at") or "")
        or test_access.get("started_at") != test_manifest.get("started_at")
        or str(selection.get("frozen_at") or "") > str(test_manifest.get("started_at") or "")
    ):
        raise TasteOursFullError("T11 selector/test ordering cannot be independently proven")
    threshold = load_threshold_contract(threshold_contract)
    if (
        threshold.config_hash != run_manifest.get("threshold_config_hash")
        or threshold.file_sha256 != run_manifest.get("threshold_contract_file_sha256")
        or threshold.theta_star != run_manifest.get("theta_star")
        or threshold.cost_cap != run_manifest.get("cost_cap")
    ):
        raise TasteOursFullError("T11 verification threshold changed")
    replay_selected, replay_trace = select_on_calibration(
        universe, calibration_rows, theta_star=threshold.theta_star
    )
    if replay_selected != selected or replay_trace != selection.get("trace"):
        raise TasteOursFullError("T11 calibration selector cannot be independently replayed")
    test_rows, test_parents, test_evidence = _verify_pair_artifacts(
        science=science,
        split="test",
        candidates=selected,
        run_manifest=run_manifest,
        authorized_parents=test_authority,
    )
    if (
        test_manifest.get("parent_count") != len(test_parents)
        or test_manifest.get("candidate_count") != len(selected)
        or test_manifest.get("pair_count") != len(test_rows)
        or test_manifest.get("pair_details_sha256")
        != sha256_file(science / "raw" / "test_pair_details.jsonl")
        or test_manifest.get("pair_manifest_sha256")
        != sha256_file(science / "raw" / "test_pair_manifest.json")
        or run_manifest.get("test_parent_ids_sha256")
        != stable_sha256(sorted(parent.parent_id for parent in test_parents))
        or test_manifest.get("test_parent_ids_sha256")
        != stable_sha256(sorted(parent.parent_id for parent in test_parents))
    ):
        raise TasteOursFullError("T11 test evaluation manifest is incomplete")
    expected_source_files = {
        *base_evidence,
        *high_evidence,
        *calibration_evidence,
        *test_evidence,
    }
    if set(source_files) != expected_source_files:
        raise TasteOursFullError("T11 source evidence inventory is incomplete")
    metrics = standardized_metrics(test_rows, ordered, threshold)
    for name, rows in _artifact_rows(metrics).items():
        if (science / name).read_bytes() != csv_bytes(rows):
            raise TasteOursFullError(f"T11 standardized artifact cannot be replayed: {name}")
    expected_prefix = (json.dumps(metrics["prefix"], indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()
    if (science / "prefix_metrics.json").read_bytes() != expected_prefix:
        raise TasteOursFullError("T11 prefix JSON cannot be replayed")
    summary = read_json(science / "summary.json")
    oracle_manifest = read_json(science / "oracle_manifest.json")
    evaluation_manifest = read_json(science / "evaluation_manifest.json")
    common_checks = (
        ("dataset", DATASET),
        ("method", METHOD),
        ("stage", STAGE),
        ("oracle_checkpoint_hash", run_manifest.get("oracle_checkpoint_hash")),
        ("dataset_hash", run_manifest.get("dataset_hash")),
        (
            "temperature_calibration_hash",
            run_manifest.get("temperature_calibration_hash"),
        ),
        ("feature_schema_hash", run_manifest.get("feature_schema_hash")),
        ("molclr_checkpoint_hash", run_manifest.get("molclr_checkpoint_hash")),
        ("threshold_config_hash", run_manifest.get("threshold_config_hash")),
        (
            "threshold_contract_file_sha256",
            run_manifest.get("threshold_contract_file_sha256"),
        ),
        ("cf_mode", "strict_flip"),
        ("rf_oracle_used", False),
        ("test_used_for_selection", False),
        ("threshold_fitted_on_test", False),
        ("threshold_shared_across_methods", True),
    )
    if any(
        manifest.get(key) != expected
        for manifest in (summary, oracle_manifest, evaluation_manifest)
        for key, expected in common_checks
    ):
        raise TasteOursFullError("T11 publication manifest authority changed")
    if (
        summary.get("schema_version") != "tastemolnet_t11_summary_v1"
        or summary.get("status") != "SEALED"
        or summary.get("selection_frozen_before_test") is not True
        or summary.get("calibration_loaded") is not True
        or summary.get("test_loaded") is not True
        or summary.get("resume_supported") is not True
        or summary.get("base_generation") != asdict(GenerationConfig.base())
        or summary.get("high_temp_generation")
        != asdict(GenerationConfig.high_temp())
        or summary.get("effective_rule_count") != metrics["effective_rule_count"]
        or summary.get("parent_count") != metrics["parent_count"]
        or summary.get("pair_count") != metrics["pair_count"]
        or oracle_manifest.get("schema_version")
        != "tastemolnet_t11_oracle_manifest_v1"
        or oracle_manifest.get("status") != "SEALED"
        or oracle_manifest.get("same_frozen_gine_for_generation_calibration_test")
        is not True
        or oracle_manifest.get("calibration_loaded_for_training") is not False
        or oracle_manifest.get("test_loaded_for_training") is not False
        or evaluation_manifest.get("schema_version")
        != "tastemolnet_t11_evaluation_manifest_v1"
        or evaluation_manifest.get("status") != "SEALED"
        or evaluation_manifest.get("selection_manifest_sha256")
        != sha256_file(science / "raw" / "selection_manifest.json")
        or evaluation_manifest.get("test_evaluation_manifest_sha256")
        != sha256_file(science / "raw" / "test_evaluation_manifest.json")
        or evaluation_manifest.get("selection_frozen_before_test") is not True
        or evaluation_manifest.get("full_cartesian_test_pairs") is not True
    ):
        raise TasteOursFullError("T11 publication manifests cannot be replayed")
    staging = final.parent / f".{final.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"T11 verification staging root exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for name in files:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(science / name, destination)
        shutil.copyfile(science / "freeze_manifest.json", staging / "freeze_manifest.json")
        final_summary = read_json(staging / "summary.json")
        final_summary.update({"status": "PASS", "run_complete": True, "independent_terminal_verification_passed": True})
        atomic_json(staging / "summary.json", final_summary)
        final_manifest = {
            **run_manifest, "schema_version": "tastemolnet_t11_final_run_manifest_v2", "status": "PASS", "state": "PASS",
            "run_complete": True, "independent_terminal_verification_required": True,
            "independent_terminal_verification_passed": True, "worker_wrote_pass": False,
            "science_root": str(science),
            "science_freeze_manifest_sha256": sha256_file(
                science / "freeze_manifest.json"
            ),
            "source_evidence_inventory_sha256": freeze[
                "source_evidence_inventory_sha256"
            ],
            "base_high_merge_replayed": True,
            "calibration_pair_chunks_replayed": True,
            "test_pair_chunks_replayed": True,
            "verified_at": utc_now(),
        }
        atomic_json(staging / "run_manifest.json", final_manifest)
        final_inventory_names = [
            *files,
            "freeze_manifest.json",
            "run_manifest.json",
        ]
        audit = {
            "schema_version": "tastemolnet_t11_final_artifact_audit_v2", "dataset": DATASET, "method": METHOD,
            "stage": STAGE, "status": "PASS", "independent_verifier": True,
            "passed": True, "audit_passed": True,
            "science_freeze_manifest_sha256": sha256_file(science / "freeze_manifest.json"),
            "source_evidence_inventory_sha256": freeze[
                "source_evidence_inventory_sha256"
            ],
            "recomputed_metrics": True, "base_high_merge_replayed": True,
            "calibration_pair_chunks_replayed": True,
            "test_pair_chunks_replayed": True,
            "selection_frozen_before_test": True, "test_used_for_selection": False,
            "files": {name: {"sha256": sha256_file(staging / name), "bytes": (staging / name).stat().st_size} for name in final_inventory_names},
        }
        atomic_json(staging / "final_artifact_audit.json", audit)
        _atomic_bytes(staging / "PASS", (PASS_MARKER + "\n").encode())
        directory_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_noreplace(staging, final)
        parent_fd = os.open(final.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final_manifest


__all__ = [
    "GenerationConfig", "OursAuthority", "TasteOursFullError", "csv_bytes",
    "evaluate_parent", "load_authority", "merge_candidate_modes", "run_science",
    "select_on_calibration", "standardized_metrics", "verify_and_publish",
]
