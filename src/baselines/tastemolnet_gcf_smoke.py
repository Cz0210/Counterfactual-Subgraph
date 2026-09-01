"""Release-frozen TasteMolNet T7 native-GCF bounded smoke.

The official GCFExplainer full-graph edit neighbourhood, VRRW loop, and
NeuroSED threshold-coverage semantics are reused.  The BACE binary adapter and
BACE artifacts are forbidden.  At the bounded alpha=1 endpoint, a project
bridge supplies the calibrated three-class importance ``1 - p(Sweet)``, uses
the held Taste-specific NeuroSED predecessor for individual coverage, and
separately enforces the actual candidate predicate ``argmax != Sweet``.

Importing this module performs no scientific I/O and does not require the
AutoDL-only RDKit/PyTorch/PyG stack.
"""

from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import stat
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
import uuid

from src.baselines.tastemolnet_comrecgc_smoke import canonical_attributed_graph


STAGE = "T7_GCF_SMOKE"
DATASET = "tastemolnet"
METHOD = "GCFExplainer"
NUM_CLASSES = 3
SOURCE_LABEL = 1
LABEL_MAP = {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
PASS_MARKER = "[TASTE_T7_GCF_SMOKE_PASS]"

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_CONFIG_PATH = (
    REPO_ROOT / "configs/autodl/tastemolnet_t7_gcf_smoke_release_v1.json"
)
RELEASE_SCHEMA = "tastemolnet_t7_gcf_smoke_release_v1"
DISABLED_RELEASE_STATE = (
    "RELEASE_DISABLED_PENDING_INTEGRATION_COMMIT_AND_TYPED_AUTHORITY"
)

SMOKE_PARENT_COUNT = 8
SMOKE_SOURCE_POOL_LIMIT = 64
SMOKE_STEPS = 16
SMOKE_INTERRUPT_AFTER_STEPS = 8
SMOKE_SAMPLE_SIZE = 128
SMOKE_CANDIDATE_CAPACITY = 512
SMOKE_ALPHA = 1.0
SMOKE_TELEPORT = 0.1
SMOKE_SEED = 7
SMOKE_GPU_INDEX = 0
GRAPH_IDENTITY_CONTRACT = "canonical_parent_free_attributed_graph_sha256_v1"
GINE_CANONICAL_REUSE_RTOL = 1e-5
GINE_CANONICAL_REUSE_ATOL = 1e-7

RELEASE_KEYS = frozenset(
    {
        "schema_version",
        "release_enabled",
        "release_state",
        "implementation_commit",
        "implementation_tree",
        "external_authority_path",
        "external_authority_sha256",
        "t2_receipt_sha256",
        "t2_gate_sha256",
        "t2_source_evidence_sha256",
        "t3_gate_sha256",
        "t3_root_inventory_sha256",
        "t4_gate_sha256",
        "t4_root_inventory_sha256",
        "controller_receipt_sha256",
        "gpu_lease_receipt_sha256",
        "managed_execution_v2_pass_path",
        "managed_execution_v2_pass_sha256",
        "taste_gcf_neurosed_final_root",
        "taste_gcf_neurosed_pass_path",
        "taste_gcf_neurosed_pass_sha256",
        "taste_gcf_neurosed_gate_sha256",
        "taste_gcf_neurosed_verification_sha256",
        "taste_gcf_neurosed_checkpoint_path",
        "taste_gcf_neurosed_checkpoint_sha256",
        "taste_gcf_neurosed_feature_schema_sha256",
        "taste_gcf_neurosed_sha256s_sha256",
        "managed_stage_root",
        "gpu_index",
        "output_parent",
    }
)
RELEASE_PIN_FIELDS = tuple(
    sorted(
        RELEASE_KEYS
        - {"schema_version", "release_enabled", "release_state", "gpu_index"}
    )
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class TasteGCFSmokeError(RuntimeError):
    """A T7 scientific, authority, or retained-output contract failed."""


class TasteGCFSmokeReleaseDisabled(TasteGCFSmokeError):
    """The checked-in implementation is not an AutoDL execution release."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _HEX_64.fullmatch(value) is None:
        raise TasteGCFSmokeError(f"{field} must be one lowercase SHA-256")
    return value


def _native_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise TasteGCFSmokeError(f"{field} must be one bounded native JSON integer")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TasteGCFSmokeError(f"{field} must be one finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise TasteGCFSmokeError(f"{field} must be one finite JSON number")
    return result


def score_and_candidate(probabilities: Sequence[Any]) -> tuple[float, int, bool]:
    """Return exact importance and the separate multiclass CF predicate."""

    if len(probabilities) != NUM_CLASSES:
        raise TasteGCFSmokeError("Taste GCF probabilities must have three entries")
    values: list[float] = []
    for index, raw in enumerate(probabilities):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TasteGCFSmokeError(f"probabilities[{index}] is not numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise TasteGCFSmokeError(f"probabilities[{index}] is outside [0, 1]")
        values.append(value)
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise TasteGCFSmokeError("Taste GCF probabilities do not sum to one")
    prediction = max(range(NUM_CLASSES), key=lambda index: values[index])
    score = 1.0 - values[SOURCE_LABEL]
    return score, prediction, prediction != SOURCE_LABEL


@dataclass(frozen=True, slots=True)
class TasteGCFGraphSchema:
    """Train-only atom vocabulary for the official one-hot edit space."""

    atom_vocabulary: tuple[int, ...]
    feature_atomic_numbers: tuple[int, ...]
    formal_charge_vocabulary: tuple[int, ...]
    aromaticity_vocabulary: tuple[bool, ...]
    bond_type_vocabulary: tuple[str, ...]
    max_num_nodes: int
    atom_sidecar_schema_version: str = "tastemolnet_gcf_atom_sidecar_v1"
    explicit_h_nodes: bool = True

    @property
    def node_feature_dim(self) -> int:
        return len(self.feature_atomic_numbers)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "atom_vocabulary",
            "feature_atomic_numbers",
            "formal_charge_vocabulary",
            "aromaticity_vocabulary",
            "bond_type_vocabulary",
        ):
            payload[key] = list(payload[key])
        payload.update(
            {
                "schema_version": "tastemolnet_gcf_train_atom_schema_v1",
                "node_feature_dim": self.node_feature_dim,
                "data_split_used": "train",
                "native_num_classes": NUM_CLASSES,
                "native_label_map": dict(LABEL_MAP),
                "native_label_projection": "identity_0_1_2",
                "source_label_semantics": "record_native_label",
                "destination_label_semantics": "all_other_native_labels",
                "validation_loaded": False,
                "calibration_loaded": False,
                "test_loaded": False,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class TasteTrainRows:
    rows: tuple[Any, ...]
    sweet_rows: tuple[Any, ...]
    schema: TasteGCFGraphSchema
    evidence: Mapping[str, Any]


def load_train_rows(
    data: bytes,
    *,
    source_path: Path,
    expected_num_records: int,
    expected_label_counts: Mapping[str, Any],
) -> TasteTrainRows:
    """Parse the complete frozen train CSV without opening any other split."""

    from rdkit import Chem

    from src.baselines.gcfexplainer_mutagenicity_adapter import StrictMolecule
    from src.data.tastemolnet_ppo import TASTEMOLNET_PREPARED_FIELDS

    if type(data) is not bytes or not data:
        raise TasteGCFSmokeError("Taste GCF train bytes are empty")
    if (
        type(expected_label_counts) is not dict
        or set(expected_label_counts) != set(LABEL_MAP)
        or any(
            type(expected_label_counts[label]) is not int
            or expected_label_counts[label] <= 0
            for label in LABEL_MAP
        )
        or sum(expected_label_counts.values()) != expected_num_records
    ):
        raise TasteGCFSmokeError("Taste GCF train label-count authority changed")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TasteGCFSmokeError("Taste GCF train CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != TASTEMOLNET_PREPARED_FIELDS:
        raise TasteGCFSmokeError("Taste GCF train CSV columns changed")
    rows: list[Any] = []
    counts = {key: 0 for key in LABEL_MAP}
    identifiers: set[str] = set()
    atoms: set[int] = set()
    charges: set[int] = set()
    aromaticity: set[bool] = set()
    bonds: set[str] = set()
    max_nodes = 0
    bond_names = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
        Chem.BondType.AROMATIC: "AROMATIC",
    }
    for index, raw in enumerate(reader):
        molecule_id = str(raw.get("molecule_id") or "").strip()
        smiles = str(raw.get("model_smiles") or "").strip()
        label_text = str(raw.get("label") or "").strip()
        if (
            None in raw
            or set(raw) != set(TASTEMOLNET_PREPARED_FIELDS)
            or not molecule_id
            or molecule_id in identifiers
            or not smiles
            or label_text not in LABEL_MAP
            or str(raw.get("label_name") or "").strip() != LABEL_MAP[label_text]
            or str(raw.get("split") or "").strip() != "train"
            or str(raw.get("exclusion_reason") or "").strip()
        ):
            raise TasteGCFSmokeError("Taste GCF train row authority changed")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise TasteGCFSmokeError("Taste GCF train contains an invalid molecule")
        Chem.SanitizeMol(molecule)
        canonical = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        expanded = Chem.AddHs(molecule, addCoords=False)
        Chem.SanitizeMol(expanded)
        for atom in expanded.GetAtoms():
            atoms.add(int(atom.GetAtomicNum()))
            charges.add(int(atom.GetFormalCharge()))
            aromaticity.add(bool(atom.GetIsAromatic()))
        for bond in expanded.GetBonds():
            if bond.GetBondType() not in bond_names:
                raise TasteGCFSmokeError(
                    "Taste GCF train uses an unsupported bond type"
                )
            bonds.add(bond_names[bond.GetBondType()])
        max_nodes = max(max_nodes, int(expanded.GetNumAtoms()))
        identifiers.add(molecule_id)
        counts[label_text] += 1
        rows.append(
            StrictMolecule(
                molecule_id=molecule_id,
                smiles=smiles,
                canonical_smiles=canonical,
                label=int(label_text),
                split="train",
                semantic_label=LABEL_MAP[label_text],
                source_row_index=index,
                source_path=str(source_path),
            )
        )
    if len(rows) != expected_num_records or counts != dict(expected_label_counts):
        raise TasteGCFSmokeError("Taste GCF train counts differ from T2 authority")
    if not rows or not atoms or max_nodes <= 0:
        raise TasteGCFSmokeError("Taste GCF train graph schema is empty")
    schema = TasteGCFGraphSchema(
        atom_vocabulary=tuple(sorted(atoms)),
        feature_atomic_numbers=tuple(sorted(atoms)),
        formal_charge_vocabulary=tuple(sorted(charges)),
        aromaticity_vocabulary=tuple(sorted(aromaticity)),
        bond_type_vocabulary=tuple(
            value
            for value in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
            if value in bonds
        ),
        max_num_nodes=max_nodes,
    )
    sweet = tuple(
        sorted(
            (row for row in rows if row.label == SOURCE_LABEL),
            key=lambda row: row.molecule_id,
        )
    )
    if len(sweet) < SMOKE_SOURCE_POOL_LIMIT:
        raise TasteGCFSmokeError("Taste GCF train lacks the bounded Sweet source pool")
    evidence = {
        "schema_version": "tastemolnet_gcf_train_load_v1",
        "num_records": len(rows),
        "label_counts": counts,
        "sweet_records": len(sweet),
        "source_pool_limit": SMOKE_SOURCE_POOL_LIMIT,
        "graph_schema_sha256": _sha256_bytes(_canonical_bytes(schema.to_dict())),
        "train_loaded": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    return TasteTrainRows(tuple(rows), sweet, schema, evidence)


@dataclass(frozen=True, slots=True)
class NativeScoreBatch:
    probabilities: Any
    predictions: tuple[int, ...]
    scores: tuple[float, ...]
    candidate_flags: tuple[bool, ...]
    graph_embeddings: Any
    valid_fullgraphs: tuple[bool, ...]
    failure_reasons: tuple[str, ...]
    # Optional production evidence consumed by the Taste COMRECGC bridge.
    # Identity payloads are canonical chemistry only; model payloads preserve
    # every ordered node/edge tensor sent to the frozen GINE.  They must never
    # be used to reconstruct one another.
    identity_graph_payloads: tuple[Mapping[str, Any] | None, ...] = ()
    model_graph_payloads: tuple[Mapping[str, Any] | None, ...] = ()


class TasteFrozenGINENativeAdapter:
    """Decode native full graphs and score one frozen calibrated Taste GINE."""

    def __init__(
        self,
        checkpoint_payloads: Mapping[str, bytes],
        *,
        source_records: Sequence[Mapping[str, Any]],
        graph_schema: TasteGCFGraphSchema,
        device: str,
        canonical_replay_cache: bool = False,
    ) -> None:
        records = tuple(source_records)
        if not records or any(
            type(record.get("label")) is not int
            or record["label"] != SOURCE_LABEL
            or type(record.get("gnn_label")) is not int
            or record["gnn_label"] != SOURCE_LABEL
            or type(record.get("source_label")) is not int
            or record["source_label"] != SOURCE_LABEL
            or "target_label" in record
            or type(record.get("destination_labels")) is not list
            or any(
                type(label) is not int
                for label in record["destination_labels"]
            )
            or record["destination_labels"] != [0, 2]
            for record in records
        ):
            raise TasteGCFSmokeError(
                "Taste GCF adapter accepts only native Sweet source records"
            )
        import torch

        from src.baselines.frozen_gine_batch_scorer import FrozenGINEBatchScorer
        from src.data.molecular_graph_dataset import collate_molecular_graphs
        from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
        from src.oracles.gnn_oracle import load_gnn_checkpoint_payloads

        model, metadata = load_gnn_checkpoint_payloads(
            dict(checkpoint_payloads), device=device
        )
        card = metadata["model_card"]
        exact = {
            "dataset": DATASET,
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "backbone": "gine",
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "profile": "full",
        }
        if any(
            type(card.get(key)) is not type(value) or card.get(key) != value
            for key, value in exact.items()
        ):
            raise TasteGCFSmokeError("Taste GCF frozen GINE model-card drifted")
        if metadata["label_map"] != LABEL_MAP:
            raise TasteGCFSmokeError(
                "Taste GCF label map is not the frozen three-class map"
            )
        test_status = metadata["test_evaluation_status"]
        if (
            test_status.get("status") != "NOT_EVALUATED"
            or test_status.get("test_loaded") is not False
        ):
            raise TasteGCFSmokeError(
                "Taste GCF checkpoint opened an evaluated test route"
            )
        temperature = metadata["temperature_scaling"].get("temperature")
        if (
            type(temperature) is not float
            or not math.isfinite(temperature)
            or temperature <= 0.0
            or metadata["temperature_scaling"].get("selection_split")
            != "validation"
            or metadata["temperature_scaling"].get("test_used_for_fit") is not False
        ):
            raise TasteGCFSmokeError(
                "Taste GCF calibrated temperature authority drifted"
            )
        self._torch = torch
        self.model = model
        self.metadata = metadata
        self.source_records = records
        self.graph_schema = graph_schema
        self.device = device
        self.feature_schema = metadata["feature_schema"]
        self.featurizer = MolecularGraphFeaturizer(self.feature_schema)
        self.edge_feature_dim = len(self.feature_schema.edge_fields)
        self.hidden_dim = int(getattr(model.config, "hidden_dim"))
        self.parameter = next(model.parameters())
        self.scorer = FrozenGINEBatchScorer(
            model=model,
            device=device,
            temperature=float(temperature),
            checkpoint_id=metadata["checkpoint_id"],
            collate_fn=lambda rows: collate_molecular_graphs(
                rows, edge_feature_dim=self.edge_feature_dim
            ),
            cache_capacity=0,
            diagnostic_trace=False,
        )
        self.decode_failures: Counter[str] = Counter()
        self.decode_success_count = 0
        self.empty_valid_batch_count = 0
        self.call_count = 0
        # T14 can opt into an exact, model-input-keyed canonical replay layer.
        # The first byte-identical input is evaluated by the frozen GINE and
        # persisted; later occurrences are not rescored on CUDA.  This
        # preserves first-observation science and removes batch-history drift
        # without accepting a numerically different observation.
        self.canonical_replay_cache_enabled = bool(canonical_replay_cache)
        self._canonical_replay_cache: dict[
            str, tuple[bytes, Any, Any]
        ] = {}
        self.canonical_replay_cache_hits = 0
        self.canonical_replay_cache_misses = 0

    def enable_canonical_replay_cache(self) -> None:
        """Enable exact replay only at a generation bridge boundary.

        Source-cohort selection/replay must keep the historical uncached GINE
        batching semantics so a pre-existing T14 cohort remains byte-for-byte
        reproducible.  The generation adapter is switched on only after that
        source evidence has passed; a restored bridge may then prime this
        empty cache from its already validated checkpoint records.
        """

        if self.canonical_replay_cache_enabled:
            raise TasteGCFSmokeError(
                "Taste canonical replay cache was enabled before generation"
            )
        if self._canonical_replay_cache:
            raise TasteGCFSmokeError(
                "Taste canonical replay cache was nonempty before generation"
            )
        self.canonical_replay_cache_enabled = True

    @staticmethod
    def _model_input_cache_key(payload: Mapping[str, Any]) -> tuple[str, bytes]:
        try:
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TasteGCFSmokeError(
                "Taste canonical GINE model input is not serializable"
            ) from exc
        return hashlib.sha256(encoded).hexdigest(), encoded

    def _score_canonical_model_inputs(
        self,
        *,
        portable: Sequence[Any],
        valid_positions: Sequence[int],
        model_graph_payloads: Sequence[Mapping[str, Any] | None],
    ) -> tuple[Any, Any]:
        """Score each novel exact model input once, then replay its first row."""

        torch = self._torch
        probabilities = torch.empty(
            (len(portable), NUM_CLASSES),
            dtype=self.parameter.dtype,
            device=self.parameter.device,
        )
        embeddings = torch.empty(
            (len(portable), self.hidden_dim),
            dtype=self.parameter.dtype,
            device=self.parameter.device,
        )
        pending: dict[str, dict[str, Any]] = {}
        for row_index, graph_index in enumerate(valid_positions):
            payload = model_graph_payloads[graph_index]
            if type(payload) is not dict:
                raise TasteGCFSmokeError(
                    "Taste canonical replay lacks exact GINE model input"
                )
            key, encoded = self._model_input_cache_key(payload)
            cached = self._canonical_replay_cache.get(key)
            if cached is not None:
                cached_encoded, cached_probabilities, cached_embedding = cached
                if cached_encoded != encoded:
                    raise TasteGCFSmokeError(
                        "Taste canonical GINE model-input SHA-256 collision detected"
                    )
                probabilities[row_index].copy_(
                    cached_probabilities.to(
                        device=probabilities.device, dtype=probabilities.dtype
                    )
                )
                embeddings[row_index].copy_(
                    cached_embedding.to(
                        device=embeddings.device, dtype=embeddings.dtype
                    )
                )
                self.canonical_replay_cache_hits += 1
                continue
            queued = pending.get(key)
            if queued is None:
                pending[key] = {
                    "encoded": encoded,
                    "portable": portable[row_index],
                    "rows": [row_index],
                }
            elif queued["encoded"] == encoded:
                queued["rows"].append(row_index)
            else:
                raise TasteGCFSmokeError(
                    "Taste canonical GINE model-input SHA-256 collision detected"
                )
        if pending:
            ordered = list(pending.items())
            scored = self.scorer.score(
                [entry["portable"] for _key, entry in ordered]
            )
            novel_probabilities = torch.softmax(scored.project_logits, dim=-1)
            if (
                tuple(novel_probabilities.shape)
                != (len(ordered), NUM_CLASSES)
                or tuple(scored.graph_hidden.shape)
                != (len(ordered), self.hidden_dim)
            ):
                raise TasteGCFSmokeError(
                    "Taste canonical GINE scorer returned malformed novel rows"
                )
            for novel_index, (key, entry) in enumerate(ordered):
                probability_cpu = (
                    novel_probabilities[novel_index]
                    .detach()
                    .cpu()
                    .contiguous()
                    .clone()
                )
                embedding_cpu = (
                    scored.graph_hidden[novel_index]
                    .detach()
                    .cpu()
                    .contiguous()
                    .clone()
                )
                self._canonical_replay_cache[key] = (
                    entry["encoded"], probability_cpu, embedding_cpu
                )
                self.canonical_replay_cache_misses += 1
                for duplicate_index, row_index in enumerate(entry["rows"]):
                    probabilities[row_index].copy_(
                        probability_cpu.to(
                            device=probabilities.device,
                            dtype=probabilities.dtype,
                        )
                    )
                    embeddings[row_index].copy_(
                        embedding_cpu.to(
                            device=embeddings.device, dtype=embeddings.dtype
                        )
                    )
                    if duplicate_index > 0:
                        self.canonical_replay_cache_hits += 1
        return probabilities, embeddings

    def prime_canonical_replay_cache(self, records: Mapping[str, Any]) -> int:
        """Prime exact replay rows from an independently validated bridge state.

        This is used after T14 checkpoint restore.  Invalid/unscored graphs are
        excluded, and any duplicate model input with different canonical bytes
        or score bytes remains a hard scientific failure.
        """

        if not self.canonical_replay_cache_enabled:
            return 0
        import numpy as np

        primed = 0
        for record in records.values():
            valid = (
                record.get("valid_fullgraph")
                if isinstance(record, Mapping)
                else getattr(record, "valid_fullgraph", None)
            )
            if valid is not True:
                continue
            payload = (
                record.get("model_graph_payload")
                if isinstance(record, Mapping)
                else getattr(record, "model_graph_payload", None)
            )
            expected_sha = (
                record.get("model_graph_sha256")
                if isinstance(record, Mapping)
                else getattr(record, "model_graph_sha256", None)
            )
            if type(payload) is not dict:
                raise TasteGCFSmokeError(
                    "Taste checkpoint canonical replay lacks model input"
                )
            key, encoded = self._model_input_cache_key(payload)
            if expected_sha != key:
                raise TasteGCFSmokeError(
                    "Taste checkpoint canonical replay model SHA changed"
                )
            probabilities = (
                record.get("probabilities")
                if isinstance(record, Mapping)
                else getattr(record, "probabilities", None)
            )
            embedding_values = (
                record.get("embedding_values")
                if isinstance(record, Mapping)
                else getattr(record, "embedding_values", None)
            )
            embedding_dtype = (
                record.get("embedding_dtype")
                if isinstance(record, Mapping)
                else getattr(record, "embedding_dtype", None)
            )
            try:
                probability_row = self._torch.as_tensor(
                    probabilities,
                    dtype=self.parameter.dtype,
                ).detach().cpu().contiguous()
                embedding_array = np.asarray(
                    embedding_values, dtype=np.dtype(embedding_dtype)
                )
                embedding_row = self._torch.from_numpy(
                    np.ascontiguousarray(embedding_array)
                ).detach().cpu().contiguous()
            except (TypeError, ValueError) as exc:
                raise TasteGCFSmokeError(
                    "Taste checkpoint canonical replay row is malformed"
                ) from exc
            if (
                tuple(probability_row.shape) != (NUM_CLASSES,)
                or tuple(embedding_row.shape) != (self.hidden_dim,)
                or not bool(self._torch.isfinite(probability_row).all().item())
                or not bool(self._torch.isfinite(embedding_row).all().item())
            ):
                raise TasteGCFSmokeError(
                    "Taste checkpoint canonical replay row shape changed"
                )
            cached = self._canonical_replay_cache.get(key)
            if cached is not None:
                cached_encoded, cached_probabilities, cached_embedding = cached
                if (
                    cached_encoded != encoded
                    or not self._torch.equal(cached_probabilities, probability_row)
                    or not self._torch.equal(cached_embedding, embedding_row)
                ):
                    raise TasteGCFSmokeError(
                        "Taste checkpoint canonical replay has conflicting rows"
                    )
                continue
            self._canonical_replay_cache[key] = (
                encoded,
                probability_row.clone(),
                embedding_row.clone(),
            )
            primed += 1
        return primed

    def _portable(self, graph: Any, index: int) -> Any:
        origin = getattr(graph, "gcf_origin_index", None)
        if origin is None:
            raise ValueError("missing_source_index")
        if hasattr(origin, "detach"):
            origin = origin.detach().cpu().reshape(-1).tolist()
        if isinstance(origin, (list, tuple)):
            if len(origin) != 1:
                raise ValueError("invalid_source_index")
            origin = origin[0]
        source_index = _native_int(
            origin,
            field="Taste native source index",
            minimum=0,
        )
        if not 0 <= source_index < len(self.source_records):
            raise ValueError("source_index_out_of_range")
        source_label = _native_int(
            self.source_records[source_index].get("source_label"),
            field="Taste adapter source label",
            minimum=0,
            maximum=NUM_CLASSES - 1,
        )
        if source_label != SOURCE_LABEL:
            raise TasteGCFSmokeError(
                "Taste GCF adapter source changed from native Sweet"
            )
        from src.baselines.gcfexplainer_mutagenicity_adapter import (
            decode_generated_fullgraph,
        )
        from src.data.molecular_graph_dataset import MolecularGraphData

        decoded = decode_generated_fullgraph(
            graph,
            source_record=self.source_records[source_index],
            schema=self.graph_schema,
        )
        if not decoded.decode_ok:
            raise ValueError(decoded.failure_reason or "native_fullgraph_decode_failed")
        features = self.featurizer.featurize(decoded.canonical_smiles)
        return MolecularGraphData(
            x=features.node_features,
            edge_index=features.edge_index,
            edge_attr=features.edge_features,
            y=source_label,
            molecule_id=f"opaque-native-{index}",
            smiles=features.canonical_smiles,
            split="train_native_fullgraph_smoke",
            graph_sha256=features.graph_sha256,
        )

    def score(self, graphs: Sequence[Any]) -> NativeScoreBatch:
        torch = self._torch
        if not graphs:
            raise TasteGCFSmokeError(
                "Taste GCF native scorer received an empty batch"
            )
        self.call_count += 1
        valid_positions: list[int] = []
        portable: list[Any] = []
        identity_graph_payloads: list[Mapping[str, Any] | None] = [
            None
        ] * len(graphs)
        model_graph_payloads: list[Mapping[str, Any] | None] = [
            None
        ] * len(graphs)
        failures = [""] * len(graphs)
        for index, graph in enumerate(graphs):
            try:
                model_graph = self._portable(graph, index)
                graph_payload = {
                    "canonical_smiles": model_graph.smiles,
                    "node_features": [list(row) for row in model_graph.x],
                    "edge_index": [list(row) for row in model_graph.edge_index],
                    "edge_features": [list(row) for row in model_graph.edge_attr],
                    "schema_sha256": str(
                        self.feature_schema.to_dict()["schema_sha256"]
                    ),
                }
                encoded = json.dumps(
                    graph_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
                if hashlib.sha256(encoded).hexdigest() != model_graph.graph_sha256:
                    raise TasteGCFSmokeError(
                        "Taste GINE model graph differs from its feature hash"
                    )
                identity_graph_payloads[index] = {
                    "canonical_graph": model_graph.smiles,
                    "num_nodes": model_graph.num_nodes,
                    "num_edges": len(model_graph.edge_index[0]) // 2,
                }
                model_graph_payloads[index] = {
                    "schema_version": "tastemolnet_gine_model_graph_v1",
                    "canonical_smiles": model_graph.smiles,
                    "graph_sha256": model_graph.graph_sha256,
                    "feature_schema_sha256": graph_payload["schema_sha256"],
                    "node_features": {
                        "dtype": "int64",
                        "shape": [model_graph.num_nodes, len(model_graph.x[0])],
                        "values": graph_payload["node_features"],
                    },
                    "edge_index": {
                        "dtype": "int64",
                        "shape": [2, len(model_graph.edge_index[0])],
                        "values": graph_payload["edge_index"],
                    },
                    "edge_attr": {
                        "dtype": "int64",
                        "shape": [
                            len(model_graph.edge_attr),
                            self.edge_feature_dim,
                        ],
                        "values": graph_payload["edge_features"],
                    },
                }
                portable.append(model_graph)
                valid_positions.append(index)
            except TasteGCFSmokeError:
                raise
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                failures[index] = reason
                self.decode_failures[reason] += 1
        probabilities = torch.zeros(
            (len(graphs), NUM_CLASSES),
            dtype=self.parameter.dtype,
            device=self.parameter.device,
        )
        probabilities[:, SOURCE_LABEL] = 1.0
        embeddings = torch.zeros(
            (len(graphs), self.hidden_dim),
            dtype=self.parameter.dtype,
            device=self.parameter.device,
        )
        if portable:
            if self.canonical_replay_cache_enabled:
                valid_probabilities, valid_embeddings = (
                    self._score_canonical_model_inputs(
                        portable=portable,
                        valid_positions=valid_positions,
                        model_graph_payloads=model_graph_payloads,
                    )
                )
            else:
                scored = self.scorer.score(portable)
                valid_probabilities = torch.softmax(
                    scored.project_logits, dim=-1
                )
                valid_embeddings = scored.graph_hidden
            positions = torch.tensor(
                valid_positions, dtype=torch.long, device=self.parameter.device
            )
            probabilities.index_copy_(0, positions, valid_probabilities)
            embeddings.index_copy_(0, positions, valid_embeddings)
            self.decode_success_count += len(portable)
        else:
            self.empty_valid_batch_count += 1
        probability_rows = probabilities.detach().cpu().tolist()
        decisions = [score_and_candidate(row) for row in probability_rows]
        return NativeScoreBatch(
            probabilities=probabilities.detach().cpu().numpy(),
            predictions=tuple(row[1] for row in decisions),
            scores=tuple(row[0] for row in decisions),
            candidate_flags=tuple(row[2] for row in decisions),
            graph_embeddings=embeddings.detach().cpu().numpy(),
            valid_fullgraphs=tuple(not value for value in failures),
            # Exact failure categories are diagnostics only.  Colliding invalid
            # zero embeddings must have identical counterfactual semantics.
            failure_reasons=tuple("" if not value else "invalid_fullgraph" for value in failures),
            identity_graph_payloads=tuple(identity_graph_payloads),
            model_graph_payloads=tuple(model_graph_payloads),
        )

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "tastemolnet_gcf_native_gine_adapter_v1",
            "checkpoint_id": self.metadata["checkpoint_id"],
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "call_count": self.call_count,
            "decode_success_count": self.decode_success_count,
            "empty_valid_batch_count": self.empty_valid_batch_count,
            "decode_failures": dict(sorted(self.decode_failures.items())),
            "batch_scorer": self.scorer.report(),
            "rf_oracle_used": False,
        }


def _embedding_sha256(row: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(row)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_bytes(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def encode_taste_source_graph(
    row: Any, schema: TasteGCFGraphSchema
) -> Mapping[str, Any]:
    """Encode one source while preserving native three-class Taste labels.

    The shared lineage codec owns the atom/bond identity representation, but
    its Mutagenicity route deliberately remaps labels to a binary task.  Taste
    must remove that task-specific projection at this public boundary.
    """

    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        encode_source_graph,
    )

    label = _native_int(
        getattr(row, "label", None),
        field="Taste native source label",
        minimum=0,
        maximum=NUM_CLASSES - 1,
    )
    record = dict(encode_source_graph(row, schema))
    if type(record.get("label")) is not int or record["label"] != label:
        raise TasteGCFSmokeError("Taste lineage codec changed the native label")
    record["gnn_label"] = label
    record["source_label"] = label
    record.pop("target_label", None)
    record["destination_labels"] = [
        candidate for candidate in range(NUM_CLASSES) if candidate != label
    ]
    return record


def taste_record_to_pyg(
    record: Mapping[str, Any], *, origin_index: int
) -> Any:
    """Create one lineage-bound native three-class graph."""

    from src.baselines.gcfexplainer_mutagenicity_adapter import record_to_pyg

    _native_int(origin_index, field="Taste native origin index", minimum=0)
    label = _native_int(
        record.get("label"),
        field="Taste record label",
        minimum=0,
        maximum=NUM_CLASSES - 1,
    )
    destinations = record.get("destination_labels")
    if (
        "target_label" in record
        or type(record.get("gnn_label")) is not int
        or record["gnn_label"] != label
        or type(record.get("source_label")) is not int
        or record["source_label"] != label
        or type(destinations) is not list
        or any(type(candidate) is not int for candidate in destinations)
        or destinations
        != [candidate for candidate in range(NUM_CLASSES) if candidate != label]
    ):
        raise TasteGCFSmokeError(
            "Taste source record is not one native untargeted three-class graph"
        )
    graph = record_to_pyg(record, origin_index=origin_index)
    graph.gcf_source_label = graph.y.new_tensor([label])
    graph.gcf_destination_labels = graph.y.new_tensor(
        destinations
    )
    if graph.y.numel() != 1 or int(graph.y.item()) != label:
        raise TasteGCFSmokeError("Taste PyG graph label was binary-remapped")
    return graph


class TasteGCFImportanceBridge:
    """Patch official VRRW onto stable attributed-graph identities.

    The vendored implementation hashes raw embedding bytes with Python's
    salted built-in hash. Different edited graphs can share an embedding
    (invalid graphs use the same zero row), so that key can alias semantics.
    As in T12, ``call`` queues canonical parent-free attributed-graph SHA-256
    values in batch order. ``calculate_hash`` consumes the queue and uses the
    embedding SHA only to assert the official call order.
    """

    def __init__(
        self,
        *,
        adapter: TasteFrozenGINENativeAdapter,
        vrrw: Any,
        importance: Any,
        neurosed_model: Any,
        original_graph_element_counts: Any,
        distance_threshold: float,
        parent_count: int,
        feature_atomic_numbers: Sequence[Any],
    ) -> None:
        self.adapter = adapter
        self.vrrw = vrrw
        self.importance = importance
        self.neurosed_model = neurosed_model
        self.original_graph_element_counts = original_graph_element_counts
        self.distance_threshold = _finite_number(
            distance_threshold, field="Taste NeuroSED distance threshold"
        )
        if self.distance_threshold < 0:
            raise TasteGCFSmokeError(
                "Taste NeuroSED distance threshold must be nonnegative"
            )
        self.parent_count = _native_int(
            parent_count, field="Taste GCF parent count", minimum=1
        )
        self.feature_atomic_numbers = tuple(
            _native_int(
                value,
                field="Taste GCF feature atomic number",
                minimum=1,
            )
            for value in feature_atomic_numbers
        )
        if (
            not self.feature_atomic_numbers
            or len(set(self.feature_atomic_numbers))
            != len(self.feature_atomic_numbers)
        ):
            raise TasteGCFSmokeError(
                "Taste GCF feature atomic-number vocabulary must be unique"
            )
        self.records: dict[str, dict[str, Any]] = {}
        self._pending_hashes: deque[tuple[str, str]] = deque()
        self.call_count = 0
        self.evaluated_graph_count = 0
        self.calculate_hash_count = 0
        self.distance_call_count = 0
        self.distance_evaluated_graph_count = 0
        self.canonical_row_reuse_count = 0

    def _assert_idle(self) -> None:
        if self._pending_hashes:
            raise TasteGCFSmokeError(
                "official GCF did not consume every queued structural identity"
            )

    def call(
        self,
        graphs: Sequence[Any],
        _importance_args: Mapping[str, Any],
    ) -> tuple[Any, Any, Any]:
        import numpy as np

        self._assert_idle()
        values = list(graphs)
        if not values:
            raise TasteGCFSmokeError("Taste GCF received an empty graph batch")
        batch = self.adapter.score(values)
        probabilities = np.asarray(batch.probabilities)
        embeddings = np.asarray(batch.graph_embeddings)
        if probabilities.shape != (len(values), NUM_CLASSES):
            raise TasteGCFSmokeError(
                "Taste GCF GINE probabilities are not three-class"
            )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(values):
            raise TasteGCFSmokeError("Taste GCF GINE embeddings are unaligned")
        coverage = self.importance.neurosed_threshold_coverage_estimation(
            self.neurosed_model,
            values,
            self.original_graph_element_counts,
            self.distance_threshold,
        )
        self.distance_call_count += 1
        self.distance_evaluated_graph_count += len(graphs)
        if (
            tuple(coverage.shape) != (len(graphs), self.parent_count)
            or not bool(self.vrrw.torch.isfinite(coverage).all().item())
            or not bool(((coverage == 0) | (coverage == 1)).all().item())
        ):
            raise TasteGCFSmokeError(
                "official Taste NeuroSED coverage matrix changed"
            )
        coverage = coverage.cpu()
        coverage_ratios = (
            coverage.sum(dim=1) / float(self.parent_count)
        ).numpy()
        valid = tuple(batch.valid_fullgraphs)
        failures = tuple(batch.failure_reasons)
        if len(valid) != len(values) or len(failures) != len(values):
            raise TasteGCFSmokeError("Taste GCF adapter evidence is unaligned")

        canonical_parts: list[tuple[float, float]] = []
        for index, graph in enumerate(values):
            identity = canonical_attributed_graph(
                graph,
                feature_atomic_numbers=self.feature_atomic_numbers,
            )
            graph_hash = _sha256(
                identity.graph_identity_sha256,
                field="Taste GCF structural graph identity",
            )
            collision_payload = identity.collision_payload()
            row = tuple(float(value) for value in probabilities[index].tolist())
            score, prediction, candidate_condition = score_and_candidate(row)
            if (
                int(batch.predictions[index]) != prediction
                or not math.isclose(
                    float(batch.scores[index]),
                    score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or bool(batch.candidate_flags[index]) is not candidate_condition
            ):
                raise TasteGCFSmokeError(
                    "Taste GCF adapter multiclass semantics changed"
                )
            coverage_vector = tuple(
                int(value) for value in coverage[index].tolist()
            )
            if (
                len(coverage_vector) != self.parent_count
                or any(value not in (0, 1) for value in coverage_vector)
            ):
                raise TasteGCFSmokeError(
                    "Taste GCF NeuroSED coverage is not binary"
                )
            coverage_ratio = sum(coverage_vector) / float(self.parent_count)
            raw_embedding = np.ascontiguousarray(embeddings[index])
            if raw_embedding.ndim != 1 or raw_embedding.dtype.kind != "f":
                raise TasteGCFSmokeError(
                    "Taste GCF GINE embedding is not one float row"
                )
            observed = {
                "graph_identity_sha256": graph_hash,
                "collision_payload": collision_payload,
                "probabilities": list(row),
                "pred_candidate": prediction,
                "score": score,
                "covered_parent_count": sum(coverage_vector),
                "coverage_vector": list(coverage_vector),
                "coverage_ratio": coverage_ratio,
                "candidate_condition": candidate_condition,
                "valid_fullgraph": bool(valid[index]),
                "failure_reason": str(failures[index]),
                "canonical_embedding_dtype": raw_embedding.dtype.str,
                "canonical_embedding_values": [
                    float(value) for value in raw_embedding.tolist()
                ],
                "canonical_embedding_sha256": _embedding_sha256(raw_embedding),
            }
            previous = self.records.get(graph_hash)
            if previous is None:
                record = observed
                self.records[graph_hash] = record
            else:
                previous_embedding = np.asarray(
                    previous["canonical_embedding_values"],
                    dtype=np.dtype(previous["canonical_embedding_dtype"]),
                )
                if (
                    previous["collision_payload"] != collision_payload
                    or previous["pred_candidate"] != prediction
                    or previous["candidate_condition"] is not candidate_condition
                    or previous["valid_fullgraph"] is not bool(valid[index])
                    or previous["failure_reason"] != str(failures[index])
                    or previous["coverage_vector"] != list(coverage_vector)
                    or previous["canonical_embedding_dtype"]
                    != raw_embedding.dtype.str
                    or not np.allclose(
                        np.asarray(previous["probabilities"], dtype=np.float64),
                        np.asarray(row, dtype=np.float64),
                        rtol=GINE_CANONICAL_REUSE_RTOL,
                        atol=GINE_CANONICAL_REUSE_ATOL,
                    )
                    or not np.allclose(
                        previous_embedding,
                        raw_embedding,
                        rtol=GINE_CANONICAL_REUSE_RTOL,
                        atol=GINE_CANONICAL_REUSE_ATOL,
                    )
                ):
                    raise TasteGCFSmokeError(
                        "one T7 structural identity changed GINE/NeuroSED semantics"
                    )
                record = previous
                self.canonical_row_reuse_count += 1
            self._pending_hashes.append(
                (graph_hash, _embedding_sha256(raw_embedding))
            )
            canonical_parts.append(
                (float(record["score"]), float(record["coverage_ratio"]))
            )
        self.call_count += 1
        self.evaluated_graph_count += len(values)
        return np.asarray(canonical_parts, dtype=float), embeddings, coverage

    def calculate_hash(self, graph_embedding: Any) -> str:
        if not self._pending_hashes:
            raise TasteGCFSmokeError(
                "official GCF requested a hash without a scored graph"
            )
        graph_hash, expected_embedding_sha256 = self._pending_hashes.popleft()
        if _embedding_sha256(graph_embedding) != expected_embedding_sha256:
            raise TasteGCFSmokeError(
                "official GCF graph/embedding call order changed"
            )
        self.calculate_hash_count += 1
        return graph_hash

    def is_graph_counterfactual(self, graph_hash: Any) -> bool:
        if type(graph_hash) is not str or graph_hash not in self.records:
            raise TasteGCFSmokeError(
                "official VRRW queried an unknown structural identity"
            )
        row = self.records[graph_hash]
        return bool(row["valid_fullgraph"] and row["candidate_condition"])

    @contextmanager
    def installed(self, importance: Any) -> Iterator[None]:
        self._assert_idle()
        original_call = importance.call
        original_hash = self.vrrw.calculate_hash
        original_predicate = self.vrrw.is_graph_counterfactual
        importance.call = self.call
        self.vrrw.calculate_hash = self.calculate_hash
        self.vrrw.is_graph_counterfactual = self.is_graph_counterfactual
        try:
            yield
            self._assert_idle()
        finally:
            importance.call = original_call
            self.vrrw.calculate_hash = original_hash
            self.vrrw.is_graph_counterfactual = original_predicate


_MISSING_OFFICIAL_GLOBAL = object()


@contextmanager
def _installed_official_importance_args(
    vrrw: Any, importance_args: Mapping[str, Any]
) -> Iterator[None]:
    """Install the global read by official ``restart_randomwalk`` exactly.

    The vendored function accepts no importance argument and reads its module
    global instead.  The official top-level CLI happens to create that global;
    direct library callers must do so explicitly and must not leak it across
    runs in the persistent process.
    """

    original = getattr(vrrw, "importance_args", _MISSING_OFFICIAL_GLOBAL)
    vrrw.importance_args = importance_args
    try:
        yield
    finally:
        if original is _MISSING_OFFICIAL_GLOBAL:
            if hasattr(vrrw, "importance_args"):
                delattr(vrrw, "importance_args")
        else:
            vrrw.importance_args = original


_VRRW_PROGRESS_STATE_FIELDS = (
    "graph_map",
    "graph_index_map",
    "counterfactual_candidates",
    "input_graphs_covered",
    "covering_graphs",
    "transitions",
    "traversed_hashes",
    "MAX_COUNTERFACTUAL_SIZE",
    "starting_step",
    "dataset_name",
    "alpha",
    "sample_size",
    "is_sample",
    "importance_args",
)
_VRRW_PROGRESS_CHECKPOINT_SCHEMA = (
    "tastemolnet_t7_gcf_vrrw_progress_checkpoint_v2"
)
_VRRW_PROGRESS_EVIDENCE_SCHEMA = (
    "tastemolnet_t7_gcf_vrrw_progress_resume_v1"
)
_PROGRESS_CHECKPOINT_NAME = "vrrw_progress.pt"
_CHECKPOINT_PARENT_NAME = "checkpoints"
_GENERATION_TOKEN = re.compile(r"^[0-9a-f]{64}$")


def _require_uuid4(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise TasteGCFSmokeError(f"{field} must be a canonical UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise TasteGCFSmokeError(f"{field} must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise TasteGCFSmokeError(f"{field} must be a canonical UUIDv4")
    return value


def _semantic_snapshot(value: Any) -> Any:
    """Return an exact, JSON-canonicalizable scientific-state snapshot."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TasteGCFSmokeError("VRRW progress contains a non-finite float")
        return {"type": "float", "hex": value.hex()}
    if type(value) is bytes:
        return {
            "type": "bytes",
            "size": len(value),
            "sha256": _sha256_bytes(value),
        }

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - the real T7 runtime requires NumPy
        np = None
    if np is not None and isinstance(value, np.generic):
        return {
            "type": f"numpy_scalar:{value.dtype}",
            "value": _semantic_snapshot(value.item()),
        }
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "type": "numpy_array",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _sha256_bytes(array.tobytes(order="C")),
        }

    if hasattr(value, "detach") and hasattr(value, "dtype"):
        tensor = value.detach().cpu()
        if getattr(tensor, "is_sparse", False):
            tensor = tensor.coalesce()
            return {
                "type": "torch_sparse_coo",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "indices": _semantic_snapshot(tensor.indices()),
                "values": _semantic_snapshot(tensor.values()),
            }
        contiguous = tensor.contiguous()
        try:
            raw = contiguous.numpy().tobytes(order="C")
        except TypeError:  # bfloat16 and future dtypes without NumPy support
            import torch as torch_module

            raw = (
                contiguous.reshape(-1)
                .view(torch_module.uint8)
                .numpy()
                .tobytes(order="C")
            )
        return {
            "type": "torch_tensor",
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": _sha256_bytes(raw),
        }

    if isinstance(value, Mapping):
        items = [
            [_semantic_snapshot(key), _semantic_snapshot(item)]
            for key, item in value.items()
        ]
        items.sort(key=lambda row: _canonical_bytes(row[0]))
        return {"type": "mapping", "items": items}
    if isinstance(value, list):
        return {"type": "list", "items": [_semantic_snapshot(row) for row in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_semantic_snapshot(row) for row in value]}
    if isinstance(value, (set, frozenset)):
        items = [_semantic_snapshot(row) for row in value]
        items.sort(key=_canonical_bytes)
        return {"type": type(value).__name__, "items": items}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return {
            "type": f"object:{type(value).__module__}.{type(value).__qualname__}",
            "state": _semantic_snapshot(to_dict()),
        }
    if hasattr(value, "__dict__"):
        return {
            "type": f"object:{type(value).__module__}.{type(value).__qualname__}",
            "state": _semantic_snapshot(vars(value)),
        }
    raise TasteGCFSmokeError(
        f"unsupported VRRW progress value: {type(value).__module__}.{type(value).__qualname__}"
    )


def _semantic_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(_semantic_snapshot(value)))


def _capture_rng_state(*, np: Any, torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if cuda_available else [],
        "cuda_available": cuda_available,
    }


def _restore_rng_state(state: Mapping[str, Any], *, np: Any, torch: Any) -> None:
    expected = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "cuda_available",
    }
    if set(state) != expected or type(state.get("cuda_available")) is not bool:
        raise TasteGCFSmokeError("VRRW checkpoint RNG schema changed")
    if state["cuda_available"] is not bool(torch.cuda.is_available()):
        raise TasteGCFSmokeError("VRRW checkpoint CUDA RNG availability changed")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda_available"]:
        torch.cuda.set_rng_state_all(list(state["torch_cuda"]))


def _file_physical_identity(file_stat: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(file_stat.st_dev),
        "st_ino": int(file_stat.st_ino),
        "st_mode": int(file_stat.st_mode),
        "st_nlink": int(file_stat.st_nlink),
        "st_uid": int(file_stat.st_uid),
        "st_gid": int(file_stat.st_gid),
        "st_size": int(file_stat.st_size),
        "st_mtime_ns": int(file_stat.st_mtime_ns),
        "st_ctime_ns": int(file_stat.st_ctime_ns),
    }


def _directory_namespace_identity(directory_stat: os.stat_result) -> dict[str, int]:
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise TasteGCFSmokeError("VRRW runtime namespace node is not a directory")
    return {
        "st_dev": int(directory_stat.st_dev),
        "st_ino": int(directory_stat.st_ino),
        "st_mode": int(directory_stat.st_mode),
        "st_uid": int(directory_stat.st_uid),
        "st_gid": int(directory_stat.st_gid),
    }


def _read_descriptor_bytes(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise TasteGCFSmokeError("VRRW checkpoint descriptor truncated")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


@dataclass(slots=True)
class _HeldVRRWProgressCheckpoint:
    """One durable, inode-bound checkpoint inside the private runtime root."""

    runtime_root: Path
    runtime_ancestor_fds: list[int]
    runtime_ancestor_names: list[str]
    runtime_ancestor_identities: list[Mapping[str, int]]
    runtime_parent_physical_identity: Mapping[str, int]
    runtime_fd: int
    runtime_identity: Mapping[str, int]
    checkpoint_parent_fd: int
    checkpoint_parent_identity: Mapping[str, int]
    directory: Path
    directory_name: str
    checkpoint_uuid: str
    generation_token: str
    directory_fd: int
    directory_identity: Mapping[str, int]
    descriptor: int
    identity: Mapping[str, int]
    sha256: str

    @property
    def path(self) -> Path:
        return self.directory / _PROGRESS_CHECKPOINT_NAME

    @classmethod
    def write(
        cls,
        directory: Path,
        payload: Mapping[str, Any],
        *,
        torch: Any,
    ) -> "_HeldVRRWProgressCheckpoint":
        requested = Path(directory).expanduser()
        if (
            not requested.is_absolute()
            or Path(os.path.abspath(requested)) != requested
            or requested.name in {"", ".", ".."}
        ):
            raise TasteGCFSmokeError(
                "VRRW checkpoint directory must be normalized and absolute"
            )
        if requested.parent.name != _CHECKPOINT_PARENT_NAME:
            raise TasteGCFSmokeError(
                "VRRW checkpoint must use checkpoints/<UUIDv4>/"
            )
        checkpoint_uuid = _require_uuid4(
            requested.name, field="VRRW checkpoint UUID"
        )
        generation_token = payload.get("generation_token")
        if (
            payload.get("checkpoint_uuid") != checkpoint_uuid
            or type(generation_token) is not str
            or _GENERATION_TOKEN.fullmatch(generation_token) is None
        ):
            raise TasteGCFSmokeError(
                "VRRW checkpoint UUID/generation binding changed"
            )
        runtime_root = requested.parent.parent
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        ancestor_fds: list[int] = []
        ancestor_names: list[str] = []
        ancestor_identities: list[Mapping[str, int]] = []
        runtime_fd = -1
        checkpoint_parent_fd = -1
        directory_fd = -1
        descriptor = -1
        try:
            root_fd = os.open(os.sep, flags)
            ancestor_fds.append(root_fd)
            ancestor_identities.append(
                _directory_namespace_identity(os.fstat(root_fd))
            )
            for name in runtime_root.parts[1:-1]:
                named_identity = _directory_namespace_identity(
                    os.stat(
                        name,
                        dir_fd=ancestor_fds[-1],
                        follow_symlinks=False,
                    )
                )
                child_fd = os.open(name, flags, dir_fd=ancestor_fds[-1])
                held_identity = _directory_namespace_identity(
                    os.fstat(child_fd)
                )
                if held_identity != named_identity:
                    os.close(child_fd)
                    raise TasteGCFSmokeError(
                        "VRRW runtime ancestor changed while opening"
                    )
                ancestor_names.append(name)
                ancestor_fds.append(child_fd)
                ancestor_identities.append(held_identity)
            runtime_name = runtime_root.name
            runtime_named_identity = _directory_namespace_identity(
                os.stat(
                    runtime_name,
                    dir_fd=ancestor_fds[-1],
                    follow_symlinks=False,
                )
            )
            runtime_fd = os.open(
                runtime_name,
                flags,
                dir_fd=ancestor_fds[-1],
            )
            runtime_identity = _directory_namespace_identity(
                os.fstat(runtime_fd)
            )
            if (
                runtime_identity != runtime_named_identity
                or stat.S_IMODE(runtime_identity["st_mode"]) != 0o700
            ):
                raise TasteGCFSmokeError(
                    "VRRW runtime root changed or is not private"
                )

            # The checkpoint directory is an exact direct child.  Creation,
            # opening, and every later pathname check are relative to held
            # descriptors; the display Path is never an authority.  The fixed
            # ``checkpoints`` container exists only inside this fresh runtime,
            # while the actual checkpoint identity is an unreused UUIDv4.
            os.mkdir(_CHECKPOINT_PARENT_NAME, 0o700, dir_fd=runtime_fd)
            checkpoint_parent_fd = os.open(
                _CHECKPOINT_PARENT_NAME,
                flags,
                dir_fd=runtime_fd,
            )
            os.mkdir(requested.name, 0o700, dir_fd=checkpoint_parent_fd)
            directory_fd = os.open(
                requested.name,
                flags,
                dir_fd=checkpoint_parent_fd,
            )
            directory_stat = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(directory_stat.st_mode)
                or stat.S_IMODE(directory_stat.st_mode) != 0o700
            ):
                raise TasteGCFSmokeError(
                    "VRRW checkpoint directory is not private"
                )
            descriptor = os.open(
                _PROGRESS_CHECKPOINT_NAME,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(os.dup(descriptor), "wb", closefd=True) as stream:
                torch.save(dict(payload), stream)
                stream.flush()
                os.fsync(stream.fileno())
            identity = _file_physical_identity(os.fstat(descriptor))
            if (
                not stat.S_ISREG(identity["st_mode"])
                or stat.S_IMODE(identity["st_mode"]) != 0o600
                or identity["st_nlink"] != 1
                or identity["st_size"] <= 0
            ):
                raise TasteGCFSmokeError(
                    "VRRW checkpoint physical file contract changed"
                )
            payload_bytes = _read_descriptor_bytes(
                descriptor, identity["st_size"]
            )
            os.fsync(directory_fd)
            os.fsync(checkpoint_parent_fd)
            os.fsync(runtime_fd)
            held = cls(
                runtime_root=runtime_root,
                runtime_ancestor_fds=ancestor_fds,
                runtime_ancestor_names=ancestor_names,
                runtime_ancestor_identities=ancestor_identities,
                runtime_parent_physical_identity=_file_physical_identity(
                    os.fstat(ancestor_fds[-1])
                ),
                runtime_fd=runtime_fd,
                runtime_identity=runtime_identity,
                checkpoint_parent_fd=checkpoint_parent_fd,
                checkpoint_parent_identity=_file_physical_identity(
                    os.fstat(checkpoint_parent_fd)
                ),
                directory=directory,
                directory_name=requested.name,
                checkpoint_uuid=checkpoint_uuid,
                generation_token=generation_token,
                directory_fd=directory_fd,
                directory_identity=_file_physical_identity(
                    os.fstat(directory_fd)
                ),
                descriptor=descriptor,
                identity=identity,
                sha256=_sha256_bytes(payload_bytes),
            )
            held.revalidate()
            return held
        except BaseException:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                if directory_fd >= 0:
                    os.close(directory_fd)
            except OSError:
                pass
            try:
                if checkpoint_parent_fd >= 0:
                    os.close(checkpoint_parent_fd)
            except OSError:
                pass
            try:
                if runtime_fd >= 0:
                    os.close(runtime_fd)
            except OSError:
                pass
            for held_fd in reversed(ancestor_fds):
                try:
                    os.close(held_fd)
                except OSError:
                    pass
            raise

    def _revalidate_namespace(self) -> None:
        if (
            self.runtime_fd < 0
            or self.checkpoint_parent_fd < 0
            or self.directory_fd < 0
            or not self.runtime_ancestor_fds
        ):
            raise TasteGCFSmokeError("VRRW checkpoint namespace is no longer held")
        opened_checkpoint_parent_fd = -1
        opened_directory_fd = -1
        try:
            for index, (descriptor, expected) in enumerate(
                zip(
                    self.runtime_ancestor_fds,
                    self.runtime_ancestor_identities,
                    strict=True,
                )
            ):
                if _directory_namespace_identity(os.fstat(descriptor)) != dict(
                    expected
                ):
                    raise TasteGCFSmokeError(
                        "VRRW runtime ancestor identity drifted"
                    )
                if index:
                    named = os.stat(
                        self.runtime_ancestor_names[index - 1],
                        dir_fd=self.runtime_ancestor_fds[index - 1],
                        follow_symlinks=False,
                    )
                    if _directory_namespace_identity(named) != dict(expected):
                        raise TasteGCFSmokeError(
                            "VRRW runtime ancestor name was replaced"
                        )
            if _file_physical_identity(
                os.fstat(self.runtime_ancestor_fds[-1])
            ) != dict(self.runtime_parent_physical_identity):
                raise TasteGCFSmokeError(
                    "VRRW runtime temporary parent namespace drifted"
                )
            runtime_named = os.stat(
                self.runtime_root.name,
                dir_fd=self.runtime_ancestor_fds[-1],
                follow_symlinks=False,
            )
            if (
                _directory_namespace_identity(os.fstat(self.runtime_fd))
                != dict(self.runtime_identity)
                or _directory_namespace_identity(runtime_named)
                != dict(self.runtime_identity)
            ):
                raise TasteGCFSmokeError(
                    "VRRW runtime root name or inode drifted"
                )
            checkpoint_parent_named = os.stat(
                _CHECKPOINT_PARENT_NAME,
                dir_fd=self.runtime_fd,
                follow_symlinks=False,
            )
            opened_checkpoint_parent_fd = os.open(
                _CHECKPOINT_PARENT_NAME,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self.runtime_fd,
            )
            if (
                _file_physical_identity(os.fstat(self.checkpoint_parent_fd))
                != dict(self.checkpoint_parent_identity)
                or _file_physical_identity(checkpoint_parent_named)
                != dict(self.checkpoint_parent_identity)
                or _file_physical_identity(
                    os.fstat(opened_checkpoint_parent_fd)
                )
                != dict(self.checkpoint_parent_identity)
            ):
                raise TasteGCFSmokeError(
                    "VRRW checkpoint container identity drifted"
                )
            directory_named = os.stat(
                self.directory_name,
                dir_fd=self.checkpoint_parent_fd,
                follow_symlinks=False,
            )
            opened_directory_fd = os.open(
                self.directory_name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self.checkpoint_parent_fd,
            )
        except TasteGCFSmokeError:
            for descriptor in (
                opened_directory_fd,
                opened_checkpoint_parent_fd,
            ):
                if descriptor >= 0:
                    os.close(descriptor)
            raise
        except (OSError, ValueError) as exc:
            for descriptor in (
                opened_directory_fd,
                opened_checkpoint_parent_fd,
            ):
                if descriptor >= 0:
                    os.close(descriptor)
            raise TasteGCFSmokeError(
                "VRRW checkpoint runtime namespace is missing or replaced"
            ) from exc
        try:
            if (
                _file_physical_identity(os.fstat(self.directory_fd))
                != dict(self.directory_identity)
                or _file_physical_identity(directory_named)
                != dict(self.directory_identity)
                or _file_physical_identity(os.fstat(opened_directory_fd))
                != dict(self.directory_identity)
            ):
                raise TasteGCFSmokeError(
                    "VRRW checkpoint directory identity drifted or file is missing/replaced"
                )
        finally:
            os.close(opened_directory_fd)
            os.close(opened_checkpoint_parent_fd)

    def revalidate(self) -> bytes:
        if self.descriptor < 0 or self.directory_fd < 0:
            raise TasteGCFSmokeError("VRRW checkpoint is no longer held")
        self._revalidate_namespace()
        try:
            held_identity = _file_physical_identity(os.fstat(self.descriptor))
            named = os.stat(
                _PROGRESS_CHECKPOINT_NAME,
                dir_fd=self.directory_fd,
                follow_symlinks=False,
            )
            named_identity = _file_physical_identity(named)
            named_fd = os.open(
                _PROGRESS_CHECKPOINT_NAME,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=self.directory_fd,
            )
        except OSError as exc:
            raise TasteGCFSmokeError(
                "VRRW checkpoint is missing or physically replaced"
            ) from exc
        try:
            opened_identity = _file_physical_identity(os.fstat(named_fd))
            if (
                held_identity != dict(self.identity)
                or named_identity != dict(self.identity)
                or opened_identity != dict(self.identity)
            ):
                raise TasteGCFSmokeError(
                    "VRRW checkpoint physical identity drifted"
                )
            payload = _read_descriptor_bytes(named_fd, self.identity["st_size"])
            if _sha256_bytes(payload) != self.sha256:
                raise TasteGCFSmokeError("VRRW checkpoint SHA-256 drifted")
            self._revalidate_namespace()
            return payload
        finally:
            os.close(named_fd)

    def load(self, *, torch: Any) -> Mapping[str, Any]:
        payload = self.revalidate()
        stream = io.BytesIO(payload)
        try:
            loaded = torch.load(stream, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - old supported Torch
            stream.seek(0)
            loaded = torch.load(stream, map_location="cpu")
        self.revalidate()
        if type(loaded) is not dict:
            raise TasteGCFSmokeError("VRRW checkpoint payload is not one mapping")
        return loaded

    def close(self) -> None:
        """Close retained descriptors without unlinking any pathname.

        A same-user attacker can replace a name after an identity check but
        before ``unlink``.  Therefore T7 never performs checkpoint pathname
        deletion at this authority boundary.  The enclosing private
        ``TemporaryDirectory`` owns ordinary best-effort lifecycle cleanup.
        """

        for field in (
            "descriptor",
            "directory_fd",
            "checkpoint_parent_fd",
            "runtime_fd",
        ):
            descriptor = getattr(self, field)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                finally:
                    setattr(self, field, -1)
        for descriptor in reversed(self.runtime_ancestor_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.runtime_ancestor_fds = []


def _capture_adapter_progress(adapter: Any) -> dict[str, Any]:
    scorer = adapter.scorer
    if scorer.cache_capacity != 0 or len(scorer._cache) != 0:
        raise TasteGCFSmokeError(
            "Taste T7 progress checkpoint requires the exact no-cache scorer"
        )
    return {
        "decode_failures": dict(adapter.decode_failures),
        "decode_success_count": adapter.decode_success_count,
        "empty_valid_batch_count": adapter.empty_valid_batch_count,
        "call_count": adapter.call_count,
        "scorer_calls": scorer.calls,
        "scorer_cache_hits": scorer.cache_hits,
        "scorer_cache_misses": scorer.cache_misses,
        "scorer_scored_rows": scorer.scored_rows,
        "scorer_last_trace": scorer.last_trace,
    }


def _capture_progress_state(
    *,
    vrrw: Any,
    bridge: TasteGCFImportanceBridge,
    adapter: TasteFrozenGINENativeAdapter,
    action_counts: Counter[str],
    current_graph_hash: Any,
) -> dict[str, Any]:
    bridge._assert_idle()
    official = {
        field: getattr(vrrw, field) for field in _VRRW_PROGRESS_STATE_FIELDS
    }
    return {
        "official": official,
        "bridge": {
            "records": bridge.records,
            "call_count": bridge.call_count,
            "evaluated_graph_count": bridge.evaluated_graph_count,
            "calculate_hash_count": bridge.calculate_hash_count,
            "distance_call_count": bridge.distance_call_count,
            "distance_evaluated_graph_count": (
                bridge.distance_evaluated_graph_count
            ),
            "canonical_row_reuse_count": bridge.canonical_row_reuse_count,
            "graph_identity_contract": GRAPH_IDENTITY_CONTRACT,
            "python_builtin_hash_used": False,
            "embedding_identity_used": False,
        },
        "adapter": _capture_adapter_progress(adapter),
        "action_counts": dict(action_counts),
        "current_graph_hash": current_graph_hash,
    }


def _reset_progress_state(
    *,
    vrrw: Any,
    bridge: TasteGCFImportanceBridge,
    adapter: TasteFrozenGINENativeAdapter,
    action_counts: Counter[str],
    reset_official_vrrw: Any,
    np: Any,
    torch: Any,
) -> None:
    """Destroy every in-memory walk/bridge/scorer/RNG progress component."""

    bridge._assert_idle()
    reset_official_vrrw(vrrw)
    vrrw.MAX_COUNTERFACTUAL_SIZE = 0
    vrrw.dataset_name = ""
    vrrw.alpha = -1.0
    vrrw.sample_size = 0
    vrrw.is_sample = False
    vrrw.importance_args = {}
    bridge.records = {}
    bridge._pending_hashes.clear()
    bridge.call_count = 0
    bridge.evaluated_graph_count = 0
    bridge.calculate_hash_count = 0
    bridge.distance_call_count = 0
    bridge.distance_evaluated_graph_count = 0
    bridge.canonical_row_reuse_count = 0
    adapter.decode_failures = Counter()
    adapter.decode_success_count = 0
    adapter.empty_valid_batch_count = 0
    adapter.call_count = 0
    adapter.scorer._cache.clear()
    adapter.scorer.calls = 0
    adapter.scorer.cache_hits = 0
    adapter.scorer.cache_misses = 0
    adapter.scorer.scored_rows = 0
    adapter.scorer.last_trace = None
    action_counts.clear()
    reset_seed = SMOKE_SEED + 104_729
    random.seed(reset_seed)
    np.random.seed(reset_seed)
    torch.manual_seed(reset_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(reset_seed)


def _validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "stage",
        "checkpoint_uuid",
        "generation_token",
        "completed_steps",
        "total_steps",
        "resume_start_step",
        "progress",
        "rng",
        "progress_state_sha256",
        "rng_state_sha256",
        "trace_prefix_sha256",
        "resume_graph_identity_sha256",
    }
    if (
        set(payload) != expected
        or payload.get("schema_version") != _VRRW_PROGRESS_CHECKPOINT_SCHEMA
        or payload.get("stage") != STAGE
        or _require_uuid4(
            payload.get("checkpoint_uuid"), field="checkpoint.checkpoint_uuid"
        )
        != payload.get("checkpoint_uuid")
        or type(payload.get("generation_token")) is not str
        or _GENERATION_TOKEN.fullmatch(payload["generation_token"]) is None
        or type(payload.get("completed_steps")) is not int
        or payload["completed_steps"] != SMOKE_INTERRUPT_AFTER_STEPS
        or type(payload.get("total_steps")) is not int
        or payload["total_steps"] != SMOKE_STEPS
        or type(payload.get("resume_start_step")) is not int
        or payload["resume_start_step"] != SMOKE_INTERRUPT_AFTER_STEPS + 1
        or type(payload.get("progress")) is not dict
        or type(payload.get("rng")) is not dict
    ):
        raise TasteGCFSmokeError("VRRW progress checkpoint schema changed")
    for field in (
        "progress_state_sha256",
        "rng_state_sha256",
        "trace_prefix_sha256",
        "resume_graph_identity_sha256",
    ):
        _sha256(payload.get(field), field=f"checkpoint.{field}")
    progress = payload["progress"]
    if (
        set(progress)
        != {"official", "bridge", "adapter", "action_counts", "current_graph_hash"}
        or type(progress.get("official")) is not dict
        or set(progress["official"]) != set(_VRRW_PROGRESS_STATE_FIELDS)
        or type(progress.get("bridge")) is not dict
        or set(progress["bridge"])
        != {
            "records",
            "call_count",
            "evaluated_graph_count",
            "calculate_hash_count",
            "distance_call_count",
            "distance_evaluated_graph_count",
            "canonical_row_reuse_count",
            "graph_identity_contract",
            "python_builtin_hash_used",
            "embedding_identity_used",
        }
        or progress["bridge"].get("graph_identity_contract")
        != GRAPH_IDENTITY_CONTRACT
        or progress["bridge"].get("python_builtin_hash_used") is not False
        or progress["bridge"].get("embedding_identity_used") is not False
        or type(progress.get("adapter")) is not dict
        or set(progress["adapter"])
        != {
            "decode_failures",
            "decode_success_count",
            "empty_valid_batch_count",
            "call_count",
            "scorer_calls",
            "scorer_cache_hits",
            "scorer_cache_misses",
            "scorer_scored_rows",
            "scorer_last_trace",
        }
        or type(progress.get("action_counts")) is not dict
    ):
        raise TasteGCFSmokeError("VRRW progress checkpoint state schema changed")
    if _semantic_sha256(progress) != payload["progress_state_sha256"]:
        raise TasteGCFSmokeError("VRRW progress checkpoint state digest changed")
    if _semantic_sha256(payload["rng"]) != payload["rng_state_sha256"]:
        raise TasteGCFSmokeError("VRRW progress checkpoint RNG digest changed")


def _apply_progress_state(
    payload: Mapping[str, Any],
    *,
    vrrw: Any,
    bridge: TasteGCFImportanceBridge,
    adapter: TasteFrozenGINENativeAdapter,
    action_counts: Counter[str],
    np: Any,
    torch: Any,
) -> Any:
    """Restore an authenticated payload, then prove the live state matches it."""

    bridge._assert_idle()
    _validate_checkpoint_payload(payload)
    progress = payload["progress"]
    for field in _VRRW_PROGRESS_STATE_FIELDS:
        setattr(vrrw, field, progress["official"][field])
    bridge_state = progress["bridge"]
    bridge.records = bridge_state["records"]
    bridge._pending_hashes.clear()
    bridge.call_count = _native_int(
        bridge_state["call_count"], field="checkpoint bridge calls", minimum=0
    )
    bridge.evaluated_graph_count = _native_int(
        bridge_state["evaluated_graph_count"],
        field="checkpoint bridge graph count",
        minimum=0,
    )
    bridge.calculate_hash_count = _native_int(
        bridge_state["calculate_hash_count"],
        field="checkpoint bridge hash calls",
        minimum=0,
    )
    bridge.distance_call_count = _native_int(
        bridge_state["distance_call_count"],
        field="checkpoint bridge distance calls",
        minimum=0,
    )
    bridge.distance_evaluated_graph_count = _native_int(
        bridge_state["distance_evaluated_graph_count"],
        field="checkpoint bridge distance graph count",
        minimum=0,
    )
    bridge.canonical_row_reuse_count = _native_int(
        bridge_state["canonical_row_reuse_count"],
        field="checkpoint bridge canonical row reuse count",
        minimum=0,
    )
    adapter_state = progress["adapter"]
    adapter.decode_failures = Counter(adapter_state["decode_failures"])
    for field in (
        "decode_success_count",
        "empty_valid_batch_count",
        "call_count",
    ):
        setattr(
            adapter,
            field,
            _native_int(
                adapter_state[field], field=f"checkpoint adapter {field}", minimum=0
            ),
        )
    adapter.scorer._cache.clear()
    for target, source in (
        ("calls", "scorer_calls"),
        ("cache_hits", "scorer_cache_hits"),
        ("cache_misses", "scorer_cache_misses"),
        ("scored_rows", "scorer_scored_rows"),
    ):
        setattr(
            adapter.scorer,
            target,
            _native_int(
                adapter_state[source], field=f"checkpoint {source}", minimum=0
            ),
        )
    adapter.scorer.last_trace = adapter_state["scorer_last_trace"]
    action_counts.clear()
    for action, count in progress["action_counts"].items():
        if type(action) is not str or not action:
            raise TasteGCFSmokeError("VRRW checkpoint action key changed")
        action_counts[action] = _native_int(
            count, field=f"checkpoint action {action}", minimum=0
        )
    _restore_rng_state(payload["rng"], np=np, torch=torch)

    current_graph_hash = progress["current_graph_hash"]
    live = _capture_progress_state(
        vrrw=vrrw,
        bridge=bridge,
        adapter=adapter,
        action_counts=action_counts,
        current_graph_hash=current_graph_hash,
    )
    if _semantic_sha256(live) != payload["progress_state_sha256"]:
        raise TasteGCFSmokeError("VRRW checkpoint restore state mismatch")
    if _semantic_sha256(_capture_rng_state(np=np, torch=torch)) != payload[
        "rng_state_sha256"
    ]:
        raise TasteGCFSmokeError("VRRW checkpoint restore RNG mismatch")
    return current_graph_hash


def _graph_identity_for_hash(
    bridge: TasteGCFImportanceBridge, graph_hash: Any
) -> str:
    record = bridge.records.get(graph_hash)
    if type(record) is not dict:
        raise TasteGCFSmokeError("VRRW trace graph lacks bridge evidence")
    return _sha256(
        record.get("graph_identity_sha256"), field="VRRW trace graph identity"
    )


def _trace_identities(
    bridge: TasteGCFImportanceBridge, hashes: Sequence[Any]
) -> list[str]:
    return [_graph_identity_for_hash(bridge, value) for value in hashes]


@dataclass(frozen=True, slots=True)
class _OfficialWalkSegment:
    current_graph_hash: Any
    resume_entry_used_saved_graph: bool


def _run_official_walk_segment(
    *,
    vrrw: Any,
    input_graphs: Sequence[Any],
    importance_args: Mapping[str, Any],
    teleport_probability: float,
    start_step: int,
    end_step: int,
    resume_graph_hash: Any = _MISSING_OFFICIAL_GLOBAL,
) -> _OfficialWalkSegment:
    """Call the real official loop while binding its otherwise-unsaved cursor."""

    if (
        type(start_step) is not int
        or type(end_step) is not int
        or start_step < 1
        or end_step < start_step
        or len(vrrw.traversed_hashes) != start_step - 1
    ):
        raise TasteGCFSmokeError("official VRRW segment boundary changed")
    original_restart = vrrw.restart_randomwalk
    original_move = vrrw.move_to_next_graph
    original_starting_step = vrrw.starting_step
    current: dict[str, Any] = {"hash": None}
    resume_pending = resume_graph_hash is not _MISSING_OFFICIAL_GLOBAL
    resume_consumed = False

    def tracked_restart(graphs: Sequence[Any]) -> Any:
        nonlocal resume_pending, resume_consumed
        if resume_pending:
            value = resume_graph_hash
            resume_pending = False
            resume_consumed = True
        else:
            value = original_restart(graphs)
        current["hash"] = value
        return value

    def tracked_move(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
        next_hash, teleported = original_move(*args, **kwargs)
        if type(teleported) is not bool:
            raise TasteGCFSmokeError("official VRRW teleport flag changed type")
        current["hash"] = None if teleported else next_hash
        return next_hash, teleported

    vrrw.restart_randomwalk = tracked_restart
    vrrw.move_to_next_graph = tracked_move
    vrrw.starting_step = start_step
    try:
        vrrw.counterfactual_summary_with_randomwalk(
            input_graphs=input_graphs,
            importance_args=importance_args,
            teleport_probability=teleport_probability,
            max_steps=end_step,
        )
    finally:
        vrrw.restart_randomwalk = original_restart
        vrrw.move_to_next_graph = original_move
        vrrw.starting_step = original_starting_step
    if (
        resume_pending
        or current["hash"] is None
        or len(vrrw.traversed_hashes) != end_step
        or resume_consumed is not (resume_graph_hash is not _MISSING_OFFICIAL_GLOBAL)
    ):
        raise TasteGCFSmokeError("official VRRW segment cursor closure changed")
    return _OfficialWalkSegment(current["hash"], resume_consumed)


class _PlannedVRRWInterruption(RuntimeError):
    """Private control-flow boundary proving the bounded interruption occurred."""


def _checkpoint_binding_sha256(evidence: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": "tastemolnet_t7_gcf_checkpoint_binding_v1",
                "checkpoint_uuid": evidence["checkpoint_uuid"],
                "generation_token": evidence["generation_token"],
                "checkpoint_sha256": evidence["checkpoint_sha256"],
                "checkpoint_physical_identity_sha256": evidence[
                    "checkpoint_physical_identity_sha256"
                ],
                "saved_progress_state_sha256": evidence[
                    "saved_progress_state_sha256"
                ],
                "saved_rng_state_sha256": evidence["saved_rng_state_sha256"],
                "checkpoint_trace_prefix_sha256": evidence[
                    "checkpoint_trace_prefix_sha256"
                ],
                "resume_graph_identity_sha256": evidence[
                    "resume_graph_identity_sha256"
                ],
                "interruption_after_step": evidence["interruption_after_step"],
                "total_step_count": evidence["total_step_count"],
            }
        )
    )


def _trace_continuity_sha256(evidence: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": "tastemolnet_t7_gcf_trace_continuity_v1",
                "pre_resume_step_count": evidence["pre_resume_step_count"],
                "post_resume_step_count": evidence["post_resume_step_count"],
                "total_step_count": evidence["total_step_count"],
                "checkpoint_trace_prefix_sha256": evidence[
                    "checkpoint_trace_prefix_sha256"
                ],
                "final_trace_prefix_sha256": evidence[
                    "final_trace_prefix_sha256"
                ],
                "post_resume_trace_sha256": evidence[
                    "post_resume_trace_sha256"
                ],
                "full_trace_sha256": evidence["full_trace_sha256"],
                "resume_graph_identity_sha256": evidence[
                    "resume_graph_identity_sha256"
                ],
                "first_post_resume_graph_identity_sha256": evidence[
                    "first_post_resume_graph_identity_sha256"
                ],
            }
        )
    )


def _validate_progress_evidence(evidence: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "checkpoint_written",
        "checkpoint_durable",
        "planned_interruption_observed",
        "checkpoint_reloaded",
        "resumed",
        "resume_entry_used_saved_graph",
        "deterministic_restart_from_seed_used",
        "checkpoint_held_through_resume_evidence",
        "checkpoint_path_cleanup_delegated_to_temporary_runtime",
        "checkpoint_unlinked_by_t7_security_boundary",
        "checkpoint_payload_persisted_to_terminal_output",
        "checkpoint_uuid",
        "generation_token",
        "interruption_after_step",
        "resume_start_step",
        "pre_resume_step_count",
        "post_resume_step_count",
        "total_step_count",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "checkpoint_physical_identity",
        "checkpoint_physical_identity_sha256",
        "checkpoint_binding_sha256",
        "saved_progress_state_sha256",
        "reset_progress_state_sha256",
        "restored_progress_state_sha256",
        "saved_rng_state_sha256",
        "reset_rng_state_sha256",
        "restored_rng_state_sha256",
        "checkpoint_trace_prefix_sha256",
        "final_trace_prefix_sha256",
        "post_resume_trace_sha256",
        "full_trace_sha256",
        "resume_graph_identity_sha256",
        "first_post_resume_graph_identity_sha256",
        "trace_continuity_proven",
        "trace_continuity_sha256",
    }
    if (
        type(evidence) is not dict
        or set(evidence) != expected
        or evidence.get("schema_version") != _VRRW_PROGRESS_EVIDENCE_SCHEMA
    ):
        raise TasteGCFSmokeError("Taste T7 checkpoint/resume evidence schema changed")
    _require_uuid4(
        evidence.get("checkpoint_uuid"),
        field="progress_checkpoint.checkpoint_uuid",
    )
    if (
        type(evidence.get("generation_token")) is not str
        or _GENERATION_TOKEN.fullmatch(evidence["generation_token"]) is None
    ):
        raise TasteGCFSmokeError(
            "Taste T7 checkpoint generation token changed"
        )
    for field in (
        "checkpoint_written",
        "checkpoint_durable",
        "planned_interruption_observed",
        "checkpoint_reloaded",
        "resumed",
        "resume_entry_used_saved_graph",
        "checkpoint_held_through_resume_evidence",
        "checkpoint_path_cleanup_delegated_to_temporary_runtime",
        "trace_continuity_proven",
    ):
        if evidence.get(field) is not True:
            raise TasteGCFSmokeError(
                f"Taste T7 checkpoint/resume evidence did not prove {field}"
            )
    for field in (
        "deterministic_restart_from_seed_used",
        "checkpoint_unlinked_by_t7_security_boundary",
        "checkpoint_payload_persisted_to_terminal_output",
    ):
        if evidence.get(field) is not False:
            raise TasteGCFSmokeError(
                f"Taste T7 checkpoint/resume evidence changed {field}"
            )

    expected_counts = {
        "interruption_after_step": SMOKE_INTERRUPT_AFTER_STEPS,
        "resume_start_step": SMOKE_INTERRUPT_AFTER_STEPS + 1,
        "pre_resume_step_count": SMOKE_INTERRUPT_AFTER_STEPS,
        "post_resume_step_count": SMOKE_STEPS - SMOKE_INTERRUPT_AFTER_STEPS,
        "total_step_count": SMOKE_STEPS,
    }
    for field, expected_value in expected_counts.items():
        observed = _native_int(evidence.get(field), field=field, minimum=1)
        if observed != expected_value:
            raise TasteGCFSmokeError(
                "Taste T7 checkpoint/resume step split changed"
            )
    if (
        evidence["pre_resume_step_count"]
        + evidence["post_resume_step_count"]
        != evidence["total_step_count"]
        or evidence["resume_start_step"]
        != evidence["interruption_after_step"] + 1
    ):
        raise TasteGCFSmokeError("Taste T7 checkpoint/resume total is not exact")

    sha_fields = expected - {
        "schema_version",
        "checkpoint_written",
        "checkpoint_durable",
        "planned_interruption_observed",
        "checkpoint_reloaded",
        "resumed",
        "resume_entry_used_saved_graph",
        "deterministic_restart_from_seed_used",
        "checkpoint_held_through_resume_evidence",
        "checkpoint_path_cleanup_delegated_to_temporary_runtime",
        "checkpoint_unlinked_by_t7_security_boundary",
        "checkpoint_payload_persisted_to_terminal_output",
        "checkpoint_uuid",
        "generation_token",
        "interruption_after_step",
        "resume_start_step",
        "pre_resume_step_count",
        "post_resume_step_count",
        "total_step_count",
        "checkpoint_size_bytes",
        "checkpoint_physical_identity",
        "trace_continuity_proven",
    }
    for field in sha_fields:
        _sha256(evidence.get(field), field=f"progress_checkpoint.{field}")

    checkpoint_size = _native_int(
        evidence.get("checkpoint_size_bytes"),
        field="progress_checkpoint.checkpoint_size_bytes",
        minimum=1,
    )
    identity = evidence.get("checkpoint_physical_identity")
    identity_fields = {
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    }
    if type(identity) is not dict or set(identity) != identity_fields:
        raise TasteGCFSmokeError("Taste T7 checkpoint physical identity changed")
    for field in identity_fields:
        _native_int(identity[field], field=f"checkpoint identity {field}", minimum=0)
    if (
        not stat.S_ISREG(identity["st_mode"])
        or stat.S_IMODE(identity["st_mode"]) != 0o600
        or identity["st_nlink"] != 1
        or identity["st_size"] != checkpoint_size
        or _sha256_bytes(_canonical_bytes(identity))
        != evidence["checkpoint_physical_identity_sha256"]
        or _checkpoint_binding_sha256(evidence)
        != evidence["checkpoint_binding_sha256"]
    ):
        raise TasteGCFSmokeError("Taste T7 checkpoint physical binding changed")
    if (
        evidence["saved_progress_state_sha256"]
        != evidence["restored_progress_state_sha256"]
        or evidence["saved_progress_state_sha256"]
        == evidence["reset_progress_state_sha256"]
        or evidence["saved_rng_state_sha256"]
        != evidence["restored_rng_state_sha256"]
        or evidence["saved_rng_state_sha256"]
        == evidence["reset_rng_state_sha256"]
        or evidence["checkpoint_trace_prefix_sha256"]
        != evidence["final_trace_prefix_sha256"]
        or evidence["resume_graph_identity_sha256"]
        != evidence["first_post_resume_graph_identity_sha256"]
        or _trace_continuity_sha256(evidence)
        != evidence["trace_continuity_sha256"]
    ):
        raise TasteGCFSmokeError("Taste T7 checkpoint restore/trace proof changed")


def _execute_checkpointed_official_walk(
    *,
    vrrw: Any,
    bridge: TasteGCFImportanceBridge,
    adapter: TasteFrozenGINENativeAdapter,
    action_counts: Counter[str],
    input_graphs: Sequence[Any],
    importance_args: Mapping[str, Any],
    teleport_probability: float,
    runtime_root: Path,
    reset_official_vrrw: Any,
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    """Interrupt, checkpoint, erase, restore, and finish the real official loop."""

    checkpoint: _HeldVRRWProgressCheckpoint | None = None
    checkpoint_written = False
    planned_interruption_observed = False
    checkpoint_reloaded = False
    resumed = False
    try:
        first = _run_official_walk_segment(
            vrrw=vrrw,
            input_graphs=input_graphs,
            importance_args=importance_args,
            teleport_probability=teleport_probability,
            start_step=1,
            end_step=SMOKE_INTERRUPT_AFTER_STEPS,
        )
        prefix_identities = _trace_identities(
            bridge, list(vrrw.traversed_hashes)
        )
        if len(prefix_identities) != SMOKE_INTERRUPT_AFTER_STEPS:
            raise TasteGCFSmokeError("VRRW checkpoint trace prefix length changed")
        resume_identity = _graph_identity_for_hash(
            bridge, first.current_graph_hash
        )
        progress = _capture_progress_state(
            vrrw=vrrw,
            bridge=bridge,
            adapter=adapter,
            action_counts=action_counts,
            current_graph_hash=first.current_graph_hash,
        )
        rng_state = _capture_rng_state(np=np, torch=torch)
        saved_progress_sha256 = _semantic_sha256(progress)
        saved_rng_sha256 = _semantic_sha256(rng_state)
        trace_prefix_sha256 = _sha256_bytes(_canonical_bytes(prefix_identities))
        checkpoint_uuid = str(uuid.uuid4())
        generation_token = secrets.token_hex(32)
        checkpoint_payload = {
            "schema_version": _VRRW_PROGRESS_CHECKPOINT_SCHEMA,
            "stage": STAGE,
            "checkpoint_uuid": checkpoint_uuid,
            "generation_token": generation_token,
            "completed_steps": SMOKE_INTERRUPT_AFTER_STEPS,
            "total_steps": SMOKE_STEPS,
            "resume_start_step": SMOKE_INTERRUPT_AFTER_STEPS + 1,
            "progress": progress,
            "rng": rng_state,
            "progress_state_sha256": saved_progress_sha256,
            "rng_state_sha256": saved_rng_sha256,
            "trace_prefix_sha256": trace_prefix_sha256,
            "resume_graph_identity_sha256": resume_identity,
        }
        _validate_checkpoint_payload(checkpoint_payload)
        checkpoint = _HeldVRRWProgressCheckpoint.write(
            runtime_root
            / _CHECKPOINT_PARENT_NAME
            / checkpoint_uuid,
            checkpoint_payload,
            torch=torch,
        )
        checkpoint_written = True
        checkpoint_identity = dict(checkpoint.identity)
        checkpoint_sha256 = checkpoint.sha256
        checkpoint_physical_identity_sha256 = _sha256_bytes(
            _canonical_bytes(checkpoint_identity)
        )
        # Keep only aggregate commitments before the interruption.  The live
        # graph/bridge/RNG checkpoint payload must be recoverable exclusively
        # from the held physical file after the reset below.
        del checkpoint_payload, progress, rng_state, first

        try:
            raise _PlannedVRRWInterruption(
                f"planned T7 interruption after {SMOKE_INTERRUPT_AFTER_STEPS} steps"
            )
        except _PlannedVRRWInterruption:
            planned_interruption_observed = True

        _reset_progress_state(
            vrrw=vrrw,
            bridge=bridge,
            adapter=adapter,
            action_counts=action_counts,
            reset_official_vrrw=reset_official_vrrw,
            np=np,
            torch=torch,
        )
        reset_progress_sha256 = _semantic_sha256(
            _capture_progress_state(
                vrrw=vrrw,
                bridge=bridge,
                adapter=adapter,
                action_counts=action_counts,
                current_graph_hash=None,
            )
        )
        reset_rng_sha256 = _semantic_sha256(
            _capture_rng_state(np=np, torch=torch)
        )
        if (
            reset_progress_sha256 == saved_progress_sha256
            or reset_rng_sha256 == saved_rng_sha256
        ):
            raise TasteGCFSmokeError(
                "VRRW planned interruption did not erase in-memory progress"
            )

        loaded = checkpoint.load(torch=torch)
        checkpoint_reloaded = True
        restored_current_hash = _apply_progress_state(
            loaded,
            vrrw=vrrw,
            bridge=bridge,
            adapter=adapter,
            action_counts=action_counts,
            np=np,
            torch=torch,
        )
        restored_progress_sha256 = _semantic_sha256(
            _capture_progress_state(
                vrrw=vrrw,
                bridge=bridge,
                adapter=adapter,
                action_counts=action_counts,
                current_graph_hash=restored_current_hash,
            )
        )
        restored_rng_sha256 = _semantic_sha256(
            _capture_rng_state(np=np, torch=torch)
        )
        restored_prefix = _trace_identities(
            bridge, list(vrrw.traversed_hashes)
        )
        if (
            restored_progress_sha256 != saved_progress_sha256
            or restored_rng_sha256 != saved_rng_sha256
            or _sha256_bytes(_canonical_bytes(restored_prefix))
            != trace_prefix_sha256
            or _graph_identity_for_hash(bridge, restored_current_hash)
            != resume_identity
        ):
            raise TasteGCFSmokeError("VRRW checkpoint restore mismatch")

        second = _run_official_walk_segment(
            vrrw=vrrw,
            input_graphs=input_graphs,
            importance_args=importance_args,
            teleport_probability=teleport_probability,
            start_step=SMOKE_INTERRUPT_AFTER_STEPS + 1,
            end_step=SMOKE_STEPS,
            resume_graph_hash=restored_current_hash,
        )
        # The continuation may execute arbitrary official filesystem code.
        # Reclose the held runtime/checkpoint namespace immediately after it.
        checkpoint.revalidate()
        resume_entry_used_saved_graph = second.resume_entry_used_saved_graph
        resumed = True
        final_identities = _trace_identities(
            bridge, list(vrrw.traversed_hashes)
        )
        final_prefix = final_identities[:SMOKE_INTERRUPT_AFTER_STEPS]
        post_resume = final_identities[SMOKE_INTERRUPT_AFTER_STEPS:]
        if (
            len(final_identities) != SMOKE_STEPS
            or len(post_resume) != SMOKE_STEPS - SMOKE_INTERRUPT_AFTER_STEPS
            or final_prefix != prefix_identities
            or not post_resume
            or post_resume[0] != resume_identity
        ):
            raise TasteGCFSmokeError("VRRW post-resume trace continuity changed")

        # No terminal evidence is derived until the same runtime parent/name,
        # direct checkpoint directory, and checkpoint inode have all reclosed.
        checkpoint.revalidate()
        evidence: dict[str, Any] = {
            "schema_version": _VRRW_PROGRESS_EVIDENCE_SCHEMA,
            "checkpoint_written": checkpoint_written,
            "checkpoint_durable": True,
            "planned_interruption_observed": planned_interruption_observed,
            "checkpoint_reloaded": checkpoint_reloaded,
            "resumed": resumed,
            "resume_entry_used_saved_graph": resume_entry_used_saved_graph,
            "deterministic_restart_from_seed_used": False,
            "checkpoint_held_through_resume_evidence": True,
            "checkpoint_path_cleanup_delegated_to_temporary_runtime": True,
            "checkpoint_unlinked_by_t7_security_boundary": False,
            "checkpoint_payload_persisted_to_terminal_output": False,
            "checkpoint_uuid": checkpoint_uuid,
            "generation_token": generation_token,
            "interruption_after_step": SMOKE_INTERRUPT_AFTER_STEPS,
            "resume_start_step": SMOKE_INTERRUPT_AFTER_STEPS + 1,
            "pre_resume_step_count": len(prefix_identities),
            "post_resume_step_count": len(post_resume),
            "total_step_count": len(final_identities),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": checkpoint_identity["st_size"],
            "checkpoint_physical_identity": checkpoint_identity,
            "checkpoint_physical_identity_sha256": (
                checkpoint_physical_identity_sha256
            ),
            "checkpoint_binding_sha256": "",
            "saved_progress_state_sha256": saved_progress_sha256,
            "reset_progress_state_sha256": reset_progress_sha256,
            "restored_progress_state_sha256": restored_progress_sha256,
            "saved_rng_state_sha256": saved_rng_sha256,
            "reset_rng_state_sha256": reset_rng_sha256,
            "restored_rng_state_sha256": restored_rng_sha256,
            "checkpoint_trace_prefix_sha256": trace_prefix_sha256,
            "final_trace_prefix_sha256": _sha256_bytes(
                _canonical_bytes(final_prefix)
            ),
            "post_resume_trace_sha256": _sha256_bytes(
                _canonical_bytes(post_resume)
            ),
            "full_trace_sha256": _sha256_bytes(
                _canonical_bytes(final_identities)
            ),
            "resume_graph_identity_sha256": resume_identity,
            "first_post_resume_graph_identity_sha256": post_resume[0],
            "trace_continuity_proven": True,
            "trace_continuity_sha256": "",
        }
        evidence["checkpoint_binding_sha256"] = _checkpoint_binding_sha256(
            evidence
        )
        evidence["trace_continuity_sha256"] = _trace_continuity_sha256(evidence)
        checkpoint.revalidate()
        _validate_progress_evidence(evidence)
        return evidence
    finally:
        if checkpoint is not None:
            checkpoint.close()


def _official_modules(official_root: Path) -> Mapping[str, Any]:
    expected = REPO_ROOT / "baselines/gcfexplainer_official"
    if official_root != expected:
        raise TasteGCFSmokeError(
            "Taste GCF official root differs from the integrated source"
        )
    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        import_official_modules,
    )

    modules = import_official_modules(official_root)
    for name, module in modules.items():
        module_path = Path(module.__file__).resolve()
        if module_path.parent != official_root or module_path.name != f"{name}.py":
            raise TasteGCFSmokeError(
                "Taste GCF imported an unbound official module"
            )
    return modules


def _select_sources(
    *,
    adapter: TasteFrozenGINENativeAdapter,
    pool_graphs: Sequence[Any],
    pool_records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    batch = adapter.score(pool_graphs)
    selected_indices = [
        index
        for index, (valid, prediction) in enumerate(
            zip(batch.valid_fullgraphs, batch.predictions, strict=True)
        )
        if valid and prediction == SOURCE_LABEL
    ][:SMOKE_PARENT_COUNT]
    if len(selected_indices) != SMOKE_PARENT_COUNT:
        raise TasteGCFSmokeError(
            "bounded train pool lacks eight GINE-correct Sweet parents"
        )
    selected_records = [pool_records[index] for index in selected_indices]
    opaque = [str(row["source_graph_hash"]) for row in selected_records]
    return selected_records, {
        "source_pool_count": len(pool_graphs),
        "source_pool_gine_correct_sweet": sum(
            valid and prediction == SOURCE_LABEL
            for valid, prediction in zip(
                batch.valid_fullgraphs, batch.predictions, strict=True
            )
        ),
        "selected_parent_count": len(selected_records),
        "selected_parent_graph_hashes_sha256": _sha256_bytes(
            _canonical_bytes(opaque)
        ),
        "pred_before": SOURCE_LABEL,
    }


def execute_native_vrrw_smoke(
    *,
    checkpoint_payloads: Mapping[str, bytes],
    source_rows: Sequence[Any],
    graph_schema: TasteGCFGraphSchema,
    official_root: Path,
    neurosed_checkpoint_path: str | Path,
    neurosed_distance_threshold: float,
    neurosed_evidence: Mapping[str, Any],
    neurosed_revalidate: Any,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Execute the real official native full-graph random walk in a temp root."""

    import numpy as np
    import torch

    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        graph_lineage_neighbor_wrapper,
    )
    from src.baselines.gcfexplainer_mutagenicity_runtime import (
        _official_vrrw_alpha_endpoint_patch,
        _reset_official_vrrw,
        _torch_load_compat,
    )

    if len(source_rows) != SMOKE_SOURCE_POOL_LIMIT:
        raise TasteGCFSmokeError("Taste GCF source pool bound changed")
    modules = _official_modules(official_root)
    vrrw = modules["vrrw"]
    importance = modules["importance"]
    distance = modules["distance"]
    _reset_official_vrrw(vrrw)
    pool_records = [
        encode_taste_source_graph(row, graph_schema) for row in source_rows
    ]
    pool_graphs = [
        taste_record_to_pyg(record, origin_index=index)
        for index, record in enumerate(pool_records)
    ]
    pool_adapter = TasteFrozenGINENativeAdapter(
        checkpoint_payloads,
        source_records=pool_records,
        graph_schema=graph_schema,
        device=device,
    )
    selected_records, parent_evidence = _select_sources(
        adapter=pool_adapter,
        pool_graphs=pool_graphs,
        pool_records=pool_records,
    )
    # Re-index the exact selected cohort so lineage and coverage share one
    # bounded parent order throughout the official walk.
    selected_graphs = [
        taste_record_to_pyg(record, origin_index=index)
        for index, record in enumerate(selected_records)
    ]
    adapter = TasteFrozenGINENativeAdapter(
        checkpoint_payloads,
        source_records=selected_records,
        graph_schema=graph_schema,
        device=device,
    )
    source_check = adapter.score(selected_graphs)
    if any(
        not valid or prediction != SOURCE_LABEL
        for valid, prediction in zip(
            source_check.valid_fullgraphs,
            source_check.predictions,
            strict=True,
        )
    ):
        raise TasteGCFSmokeError(
            "selected Sweet parent changed under exact cohort replay"
        )
    expected_neurosed_evidence = {
        "schema_version",
        "status",
        "marker",
        "final_root",
        "attempt_id",
        "generation_token",
        "pass_path",
        "pass_sha256",
        "gate_path",
        "gate_sha256",
        "verification_path",
        "verification_sha256",
        "source_inventory_sha256",
        "published_inventory_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "feature_schema_path",
        "feature_schema_sha256",
        "sha256s_path",
        "sha256s_sha256",
        "t7_consumer",
    }
    consumer = (
        neurosed_evidence.get("t7_consumer")
        if type(neurosed_evidence) is dict
        else None
    )
    if (
        type(neurosed_evidence) is not dict
        or set(neurosed_evidence) != expected_neurosed_evidence
        or neurosed_evidence.get("schema_version")
        != "tastemolnet_gcf_neurosed_managed_final_v1"
        or neurosed_evidence.get("status") != "PASS"
        or neurosed_evidence.get("marker") != "MANAGED_EXECUTION_V2_PASS"
        or type(consumer) is not dict
        or consumer.get("schema_version")
        not in {
            "tastemolnet_gcf_neurosed_t7_consumer_v1",
            "tastemolnet_gcf_neurosed_t7_fixed_budget_consumer_v2",
        }
        or consumer.get("role")
        != "GCF_AUXILIARY_DISTANCE_MODEL"
        or consumer.get("classifier") is not False
        or consumer.get("source_label_independent") is not True
        or consumer.get("train_only_fit") is not True
        or consumer.get("validation_only_selection") is not True
        or consumer.get("calibration_loaded") is not False
        or consumer.get("test_loaded") is not False
        or consumer.get("health_gate_status") != "PASS"
        or consumer.get("checkpoint_sha256")
        != neurosed_evidence.get("checkpoint_sha256")
        or consumer.get("feature_schema_sha256")
        != neurosed_evidence.get("feature_schema_sha256")
        or consumer.get("sha256s_sha256")
        != neurosed_evidence.get("sha256s_sha256")
        or consumer.get("feature_atomic_numbers")
        != list(graph_schema.feature_atomic_numbers)
        or consumer.get("feature_input_dim") != graph_schema.node_feature_dim
    ):
        raise TasteGCFSmokeError(
            "Taste NeuroSED predecessor evidence is not an exact PASS"
        )
    for field in (
        "pass_sha256",
        "gate_sha256",
        "verification_sha256",
        "source_inventory_sha256",
        "published_inventory_sha256",
        "checkpoint_sha256",
        "feature_schema_sha256",
        "sha256s_sha256",
    ):
        _sha256(neurosed_evidence.get(field), field=f"NeuroSED {field}")
    if (
        consumer.get("schema_version")
        == "tastemolnet_gcf_neurosed_t7_consumer_v1"
    ):
        for field in (
            "neurosed_train_graph_ids_hash",
            "neurosed_validation_graph_ids_hash",
        ):
            _sha256(consumer.get(field), field=f"NeuroSED {field}")
    else:
        from src.eval.tastemolnet_neurosed_fixed_budget_adoption import (
            validate_t7_fixed_budget_consumer,
        )

        validate_t7_fixed_budget_consumer(
            consumer,
            checkpoint_sha256=str(neurosed_evidence["checkpoint_sha256"]),
            feature_schema_sha256=str(
                neurosed_evidence["feature_schema_sha256"]
            ),
            sha256s_sha256=str(neurosed_evidence["sha256s_sha256"]),
            feature_atomic_numbers=list(graph_schema.feature_atomic_numbers),
            feature_input_dim=graph_schema.node_feature_dim,
        )
    checkpoint_path = Path(neurosed_checkpoint_path)
    if (
        not checkpoint_path.is_absolute()
        or str(checkpoint_path).startswith("/proc/self/fd/") is False
    ):
        raise TasteGCFSmokeError(
            "Taste NeuroSED checkpoint must be a held descriptor path"
        )
    neurosed_revalidate()
    neurosed_model = distance.load_neurosed(
        selected_graphs,
        neurosed_model_path=str(checkpoint_path),
        device=device,
    )
    neurosed_revalidate()
    original_graph_element_counts = importance.util.graph_element_counts(
        selected_graphs
    )

    random.seed(SMOKE_SEED)
    np.random.seed(SMOKE_SEED)
    torch.manual_seed(SMOKE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SMOKE_SEED)
    original_neighbor = vrrw.neighbor_graph_access
    lineage_neighbor = graph_lineage_neighbor_wrapper(original_neighbor)
    action_counts: Counter[str] = Counter()

    def counted_neighbor(graph: Any, action: tuple[Any, ...]) -> Any:
        action_counts[str(action[0])] += 1
        return lineage_neighbor(graph, action)

    vrrw.neighbor_graph_access = counted_neighbor
    vrrw.dataset_name = DATASET
    vrrw.alpha = SMOKE_ALPHA
    vrrw.sample_size = SMOKE_SAMPLE_SIZE
    vrrw.is_sample = True
    vrrw.MAX_COUNTERFACTUAL_SIZE = SMOKE_CANDIDATE_CAPACITY
    vrrw.input_graphs_covered = torch.zeros(
        SMOKE_PARENT_COUNT, dtype=torch.float
    )
    bridge = TasteGCFImportanceBridge(
        adapter=adapter,
        vrrw=vrrw,
        importance=importance,
        neurosed_model=neurosed_model,
        original_graph_element_counts=original_graph_element_counts,
        distance_threshold=neurosed_distance_threshold,
        parent_count=SMOKE_PARENT_COUNT,
        feature_atomic_numbers=graph_schema.feature_atomic_numbers,
    )
    importance_args = {
        "schema_version": "tastemolnet_gcf_neurosed_importance_v1",
        "alpha": SMOKE_ALPHA,
        "distance_status": "EVALUATED",
        "distance_threshold": _finite_number(
            neurosed_distance_threshold,
            field="Taste NeuroSED distance threshold",
        ),
        "selector_status": "NOT_EVALUATED",
    }
    old_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(
            prefix="taste-t7-native-vrrw-"
        ) as temporary_parent:
            # A dedicated private envelope makes the runtime-root parent/name
            # a stable namespace authority.  Official output may add children
            # under ``runtime`` without mutating that sealed parent.
            runtime_root = Path(temporary_parent) / "runtime"
            runtime_root.mkdir(mode=0o700)
            os.chdir(runtime_root)
            with (
                bridge.installed(importance),
                _official_vrrw_alpha_endpoint_patch(vrrw),
                _installed_official_importance_args(vrrw, importance_args),
            ):
                progress_evidence = _execute_checkpointed_official_walk(
                    vrrw=vrrw,
                    bridge=bridge,
                    adapter=adapter,
                    action_counts=action_counts,
                    input_graphs=selected_graphs,
                    importance_args=importance_args,
                    teleport_probability=SMOKE_TELEPORT,
                    runtime_root=runtime_root,
                    reset_official_vrrw=_reset_official_vrrw,
                    np=np,
                    torch=torch,
                )
            neurosed_revalidate()
            result = (
                runtime_root
                / f"results/{DATASET}/runs/counterfactuals.pt"
            )
            if not result.is_file() or result.stat().st_size <= 0:
                raise TasteGCFSmokeError(
                    "official VRRW did not emit its native result"
                )
            payload = _torch_load_compat(result)
    finally:
        os.chdir(old_cwd)
        vrrw.neighbor_graph_access = original_neighbor
    if type(payload) is not dict:
        raise TasteGCFSmokeError("official VRRW result is not one mapping")
    candidates = list(payload.get("counterfactual_candidates", []))
    traversed = list(payload.get("traversed_hashes", []))
    graph_map = dict(payload.get("graph_map", {}))
    if (
        len(traversed) != SMOKE_STEPS
        or not candidates
        or set(graph_map)
        != {row.get("graph_hash") for row in candidates if type(row) is dict}
    ):
        raise TasteGCFSmokeError("official VRRW output closure changed")
    trace: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates):
        if type(candidate) is not dict:
            raise TasteGCFSmokeError("official VRRW candidate is untyped")
        official_hash = candidate.get("graph_hash")
        if official_hash not in bridge.records:
            raise TasteGCFSmokeError(
                "official candidate lacks multiclass score evidence"
            )
        evidence = bridge.records[official_hash]
        parts = candidate.get("importance_parts")
        if not isinstance(parts, (list, tuple, np.ndarray)) or len(parts) != 2:
            raise TasteGCFSmokeError(
                "official candidate importance parts changed"
            )
        if not math.isclose(
            float(parts[0]), evidence["score"], rel_tol=0.0, abs_tol=1e-12
        ):
            raise TasteGCFSmokeError(
                "official candidate score differs from 1-pSweet"
            )
        if not math.isclose(
            float(parts[1]),
            evidence["coverage_ratio"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise TasteGCFSmokeError(
                "Taste GCF official NeuroSED coverage changed"
            )
        trace.append(
            {
                "schema_version": "tastemolnet_gcf_candidate_trace_v1",
                "rank": rank,
                "graph_identity_sha256": evidence[
                    "graph_identity_sha256"
                ],
                "frequency": _native_int(
                    candidate.get("frequency"),
                    field="candidate.frequency",
                    minimum=1,
                ),
                "probabilities": evidence["probabilities"],
                "pred_before": SOURCE_LABEL,
                "pred_candidate": evidence["pred_candidate"],
                "source_label": SOURCE_LABEL,
                "score": evidence["score"],
                "covered_parent_count": evidence["covered_parent_count"],
                "coverage_ratio": evidence["coverage_ratio"],
                "score_definition": "1.0 - probabilities[source_label]",
                "candidate_condition": evidence["candidate_condition"],
                "candidate_condition_definition": (
                    "pred_candidate != source_label"
                ),
                "valid_fullgraph": evidence["valid_fullgraph"],
                "failure_reason": evidence["failure_reason"],
                "native_action_kind": "full_counterfactual_graph",
            }
        )
    counterfactual_count = sum(
        bool(row["candidate_condition"] and row["valid_fullgraph"])
        for row in trace
    )
    if counterfactual_count <= 0:
        raise TasteGCFSmokeError(
            "official bounded walk produced no strict non-Sweet candidate"
        )
    allowed_actions = {
        "NOTHING", "NLC", "NA", "INA", "NR", "INR", "ER", "ERR", "EA"
    }
    if set(action_counts) - allowed_actions or sum(
        value for key, value in action_counts.items() if key != "NOTHING"
    ) <= 0:
        raise TasteGCFSmokeError("official native edit action evidence changed")
    return {
        "trace": trace,
        "summary": {
            "schema_version": "tastemolnet_gcf_native_vrrw_smoke_v2",
            "stage": STAGE,
            "dataset": DATASET,
            "method": METHOD,
            "parent_evidence": dict(parent_evidence),
            "official_random_walk_steps": len(traversed),
            "progress_checkpoint": progress_evidence,
            "official_candidate_count": len(trace),
            "strict_counterfactual_candidate_count": counterfactual_count,
            "destination_prediction_counts": {
                str(label): sum(
                    row["candidate_condition"]
                    and row["pred_candidate"] == label
                    for row in trace
                )
                for label in (0, 2)
            },
            "native_action_invocation_counts": dict(
                sorted(action_counts.items())
            ),
            "importance_bridge_calls": bridge.call_count,
            "importance_bridge_evaluated_graphs": (
                bridge.evaluated_graph_count
            ),
            "neurosed_distance_calls": bridge.distance_call_count,
            "neurosed_distance_evaluated_graphs": (
                bridge.distance_evaluated_graph_count
            ),
            "adapter": adapter.report(),
            "alpha": SMOKE_ALPHA,
            "coverage_mode": "official_taste_neurosed_threshold_coverage",
            "neurosed_distance_threshold": bridge.distance_threshold,
            "neurosed_predecessor": dict(neurosed_evidence),
            "candidate_condition": "pred_candidate != source_label",
            "score_definition": "1.0 - p_source",
            "native_full_graph_semantics": True,
            "deletion_only_semantics": False,
            "neurosed_status": "PASS_INPUT_REVALIDATED",
            "distance_status": "EVALUATED",
            "selector_status": "NOT_EVALUATED",
            "full_route_status": "NOT_EVALUATED",
            "bace_artifacts_used": False,
            "rf_oracle_used": False,
            "train_loaded": True,
            "validation_payload_loaded": False,
            "calibration_payload_loaded": False,
            "test_payload_loaded": False,
            "native_graph_payload_persisted": False,
            "molecule_payload_persisted": False,
            "paper_result_eligible": False,
        },
    }


def build_worker_raw_evidence(
    *,
    inputs: Any,
    managed_worker: Any,
    train_evidence: Mapping[str, Any],
    science: Mapping[str, Any],
) -> dict[str, Any]:
    """Build aggregate-only worker evidence for an independent verifier."""

    trace = list(science["trace"])
    trace_bytes = _jsonl_bytes(trace)
    input_hashes = {
        "schema_version": "tastemolnet_t7_gcf_input_hashes_v1",
        "stage": STAGE,
        "implementation_commit": inputs.execution["implementation_commit"],
        "implementation_tree": inputs.execution["implementation_tree"],
        "execution_commit": inputs.execution["execution_commit"],
        "execution_tree": inputs.execution["execution_tree"],
        "external_authority_sha256": inputs.release[
            "external_authority_sha256"
        ],
        "controller_receipt_sha256": inputs.release[
            "controller_receipt_sha256"
        ],
        "gpu_lease_receipt_sha256": inputs.release[
            "gpu_lease_receipt_sha256"
        ],
        "managed_execution_v2_pass_sha256": inputs.release[
            "managed_execution_v2_pass_sha256"
        ],
        "taste_gcf_neurosed_pass_sha256": inputs.neurosed_evidence[
            "pass_sha256"
        ],
        "taste_gcf_neurosed_gate_sha256": inputs.neurosed_evidence[
            "gate_sha256"
        ],
        "taste_gcf_neurosed_verification_sha256": inputs.neurosed_evidence[
            "verification_sha256"
        ],
        "taste_gcf_neurosed_checkpoint_sha256": inputs.neurosed_evidence[
            "checkpoint_sha256"
        ],
        "taste_gcf_neurosed_feature_schema_sha256": inputs.neurosed_evidence[
            "feature_schema_sha256"
        ],
        "taste_gcf_neurosed_sha256s_sha256": inputs.neurosed_evidence[
            "sha256s_sha256"
        ],
        "t2_receipt_sha256": inputs.t2["receipt_sha256"],
        "t2_gate_sha256": inputs.t2["gate_sha256"],
        "t2_source_evidence_sha256": inputs.t2[
            "source_evidence_sha256"
        ],
        "t2_adoption_binding_sha256": _sha256_bytes(
            _canonical_bytes(inputs.t2)
        ),
        "t3_gate_sha256": inputs.stage_evidence["t3"]["gate_sha256"],
        "t3_root_inventory_sha256": inputs.stage_evidence["t3"][
            "root_inventory_sha256"
        ],
        "t4_gate_sha256": inputs.stage_evidence["t4"]["gate_sha256"],
        "t4_root_inventory_sha256": inputs.stage_evidence["t4"][
            "root_inventory_sha256"
        ],
        "checkpoint_id": inputs.stage_evidence["t3"]["checkpoint_id"],
        "train_csv_sha256": inputs.train_contract["sha256"],
        "feature_schema_sha256": _sha256_bytes(
            inputs.checkpoint_payloads["feature_schema.json"]
        ),
        "temperature_scaling_sha256": _sha256_bytes(
            inputs.checkpoint_payloads["temperature_scaling.json"]
        ),
        "official_vrrw_sha256": _sha256_bytes(
            (
                REPO_ROOT / "baselines/gcfexplainer_official/vrrw.py"
            ).read_bytes()
        ),
        "official_importance_sha256": _sha256_bytes(
            (
                REPO_ROOT / "baselines/gcfexplainer_official/importance.py"
            ).read_bytes()
        ),
    }
    smoke = {
        **dict(science["summary"]),
        "candidate_trace_sha256": _sha256_bytes(trace_bytes),
        "train_graph_schema_sha256": train_evidence[
            "graph_schema_sha256"
        ],
    }
    from src.utils.tastemolnet_t7_managed_v2 import (
        T7_RAW_EVIDENCE_SCHEMA,
    )

    return {
        "schema_version": T7_RAW_EVIDENCE_SCHEMA,
        "status": "SEALED_PENDING_INDEPENDENT_VERIFICATION",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "attempt_id": managed_worker.attempt_id,
        "generation_token": managed_worker.generation_token,
        "expected_final_path": str(managed_worker.expected_final_path),
        "predecessors": managed_worker.predecessor_evidence(),
        "managed_attempt_input_hashes": (
            managed_worker.attempt_input_hashes()
        ),
        "input_hashes": input_hashes,
        "input_hashes_sha256": _sha256_bytes(_canonical_bytes(input_hashes)),
        "scientific_summary": smoke,
        "scientific_summary_sha256": _sha256_bytes(_canonical_bytes(smoke)),
        "candidate_trace": trace,
        "candidate_trace_sha256": _sha256_bytes(trace_bytes),
        "candidate_count": len(trace),
        "strict_counterfactual_candidate_count": smoke[
            "strict_counterfactual_candidate_count"
        ],
        "run_id": inputs.controller["run_id"],
        "gpu_index": inputs.gpu["gpu_index"],
        "gpu_uuid": inputs.gpu["gpu_uuid"],
        "num_classes": NUM_CLASSES,
        "label_map": LABEL_MAP,
        "source_label": SOURCE_LABEL,
        "strict_flip": "pred_before == 1 and pred_candidate != 1",
        "research_compute_allowed": True,
        "paper_result_reporting_allowed": True,
        "data_redistribution_allowed": False,
        "train_loaded": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "smoke_only": True,
        "paper_result_eligible": False,
        "worker_terminal_authority": False,
        "independent_verification_required": True,
    }


def parse_candidate_trace(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_keys = {
        "schema_version",
        "rank",
        "graph_identity_sha256",
        "frequency",
        "probabilities",
        "pred_before",
        "pred_candidate",
        "source_label",
        "score",
        "covered_parent_count",
        "coverage_ratio",
        "score_definition",
        "candidate_condition",
        "candidate_condition_definition",
        "valid_fullgraph",
        "failure_reason",
        "native_action_kind",
    }
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line:
            raise TasteGCFSmokeError("candidate trace contains a blank row")
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TasteGCFSmokeError(
                f"candidate trace line {line_number} is malformed"
            ) from exc
        if type(row) is not dict or line != _canonical_bytes(row):
            raise TasteGCFSmokeError("candidate trace is not canonical JSONL")
        if set(row) != expected_keys:
            raise TasteGCFSmokeError("candidate trace row keys changed")
        rank = _native_int(row["rank"], field="candidate rank", minimum=0)
        if (
            row["schema_version"]
            != "tastemolnet_gcf_candidate_trace_v1"
            or rank != len(rows)
        ):
            raise TasteGCFSmokeError("candidate trace rank/schema changed")
        _sha256(
            row["graph_identity_sha256"], field="candidate graph identity"
        )
        _native_int(row["frequency"], field="candidate frequency", minimum=1)
        score, prediction, candidate = score_and_candidate(
            row["probabilities"]
        )
        pred_before = _native_int(
            row["pred_before"], field="candidate pred_before", minimum=0
        )
        pred_candidate = _native_int(
            row["pred_candidate"], field="candidate pred_candidate", minimum=0
        )
        source_label = _native_int(
            row["source_label"], field="candidate source_label", minimum=0
        )
        observed_score = _finite_number(row["score"], field="candidate score")
        covered_parent_count = _native_int(
            row["covered_parent_count"],
            field="candidate covered parent count",
            minimum=0,
            maximum=SMOKE_PARENT_COUNT,
        )
        coverage_ratio = _finite_number(
            row["coverage_ratio"], field="candidate coverage ratio"
        )
        if (
            pred_before != SOURCE_LABEL
            or source_label != SOURCE_LABEL
            or pred_candidate != prediction
            or type(row["candidate_condition"]) is not bool
            or row["candidate_condition"] is not candidate
            or not math.isclose(
                observed_score, score, rel_tol=0.0, abs_tol=1e-12
            )
            or not math.isclose(
                coverage_ratio,
                covered_parent_count / SMOKE_PARENT_COUNT,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or row["score_definition"]
            != "1.0 - probabilities[source_label]"
            or row["candidate_condition_definition"]
            != "pred_candidate != source_label"
            or row["native_action_kind"] != "full_counterfactual_graph"
            or type(row["valid_fullgraph"]) is not bool
            or type(row["failure_reason"]) is not str
            or (candidate and not row["valid_fullgraph"])
        ):
            raise TasteGCFSmokeError(
                "candidate trace multiclass semantics changed"
            )
        rows.append(row)
    if not rows:
        raise TasteGCFSmokeError("candidate trace is empty")
    return rows


def verify_t7_worker_raw_evidence(
    payload: Mapping[str, Any],
    *,
    expected_attempt_id: str,
    expected_generation_token: str,
    expected_final_path: str | Path,
    expected_predecessor: Mapping[str, str],
    expected_input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Independent scientific verification of one managed-v2 raw payload."""

    from src.utils.tastemolnet_t7_managed_v2 import (
        T7_RAW_EVIDENCE_SCHEMA,
    )

    expected_keys = {
        "schema_version",
        "status",
        "stage",
        "dataset",
        "method",
        "attempt_id",
        "generation_token",
        "expected_final_path",
        "predecessors",
        "managed_attempt_input_hashes",
        "input_hashes",
        "input_hashes_sha256",
        "scientific_summary",
        "scientific_summary_sha256",
        "candidate_trace",
        "candidate_trace_sha256",
        "candidate_count",
        "strict_counterfactual_candidate_count",
        "run_id",
        "gpu_index",
        "gpu_uuid",
        "num_classes",
        "label_map",
        "source_label",
        "strict_flip",
        "research_compute_allowed",
        "paper_result_reporting_allowed",
        "data_redistribution_allowed",
        "train_loaded",
        "validation_loaded",
        "calibration_loaded",
        "test_loaded",
        "rf_oracle_used",
        "smoke_only",
        "paper_result_eligible",
        "worker_terminal_authority",
        "independent_verification_required",
    }
    final_path = Path(expected_final_path)
    predecessor = dict(expected_predecessor)
    expected_managed_input_hashes = {
        "managed_execution_v2_pass": expected_input_hashes.get(
            "managed_execution_v2_pass_sha256"
        ),
        "taste_gcf_neurosed_pass": expected_input_hashes.get(
            "taste_gcf_neurosed_pass_sha256"
        ),
        "taste_gcf_neurosed_gate": expected_input_hashes.get(
            "taste_gcf_neurosed_gate_sha256"
        ),
        "taste_gcf_neurosed_verification": expected_input_hashes.get(
            "taste_gcf_neurosed_verification_sha256"
        ),
        "taste_gcf_neurosed_checkpoint": expected_input_hashes.get(
            "taste_gcf_neurosed_checkpoint_sha256"
        ),
        "taste_gcf_neurosed_feature_schema": expected_input_hashes.get(
            "taste_gcf_neurosed_feature_schema_sha256"
        ),
        "taste_gcf_neurosed_sha256s": expected_input_hashes.get(
            "taste_gcf_neurosed_sha256s_sha256"
        ),
        "taste_gine_t2_gate": expected_input_hashes.get("t2_gate_sha256"),
        "taste_gine_t3_gate": expected_input_hashes.get("t3_gate_sha256"),
        "taste_oracle_t4_gate": expected_input_hashes.get("t4_gate_sha256"),
        "taste_train_csv": expected_input_hashes.get("train_csv_sha256"),
    }
    if (
        type(payload) is not dict
        or set(payload) != expected_keys
        or payload.get("schema_version") != T7_RAW_EVIDENCE_SCHEMA
        or payload.get("status")
        != "SEALED_PENDING_INDEPENDENT_VERIFICATION"
        or payload.get("stage") != STAGE
        or payload.get("dataset") != DATASET
        or payload.get("method") != METHOD
        or payload.get("attempt_id") != expected_attempt_id
        or payload.get("generation_token") != expected_generation_token
        or payload.get("expected_final_path") != str(final_path)
        or payload.get("predecessors") != [predecessor]
        or payload.get("managed_attempt_input_hashes")
        != expected_managed_input_hashes
        or payload.get("input_hashes") != dict(expected_input_hashes)
        or payload.get("input_hashes_sha256")
        != _sha256_bytes(_canonical_bytes(dict(expected_input_hashes)))
        or predecessor.get("kind") != "TASTE_GCF_NEUROSED_PASS"
        or type(predecessor.get("path")) is not str
        or type(predecessor.get("sha256")) is not str
        or payload.get("num_classes") != NUM_CLASSES
        or payload.get("label_map") != LABEL_MAP
        or payload.get("source_label") != SOURCE_LABEL
        or payload.get("strict_flip")
        != "pred_before == 1 and pred_candidate != 1"
        or payload.get("gpu_index") != SMOKE_GPU_INDEX
        or type(payload.get("gpu_uuid")) is not str
        or _GPU_UUID.fullmatch(payload["gpu_uuid"]) is None
        or payload.get("research_compute_allowed") is not True
        or payload.get("paper_result_reporting_allowed") is not True
        or payload.get("data_redistribution_allowed") is not False
        or payload.get("train_loaded") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "validation_loaded",
                "calibration_loaded",
                "test_loaded",
                "rf_oracle_used",
                "paper_result_eligible",
                "worker_terminal_authority",
            )
        )
        or payload.get("smoke_only") is not True
        or payload.get("independent_verification_required") is not True
    ):
        raise TasteGCFSmokeError("Taste T7 worker raw evidence contract changed")
    _sha256(predecessor["sha256"], field="Taste NeuroSED predecessor SHA")
    if (
        expected_input_hashes.get("schema_version")
        != "tastemolnet_t7_gcf_input_hashes_v1"
        or expected_input_hashes.get("stage") != STAGE
    ):
        raise TasteGCFSmokeError("Taste T7 input hash schema changed")
    for key, digest in expected_input_hashes.items():
        if key in {"schema_version", "stage"}:
            continue
        if key in {
            "implementation_commit",
            "implementation_tree",
            "execution_commit",
            "execution_tree",
        }:
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{40}", digest) is None:
                raise TasteGCFSmokeError(f"T7 Git identity {key} changed")
            continue
        _sha256(digest, field=f"T7 input hash {key}")
    if expected_input_hashes.get("taste_gcf_neurosed_pass_sha256") != predecessor[
        "sha256"
    ]:
        raise TasteGCFSmokeError(
            "Taste T7 raw evidence does not cross-bind NeuroSED PASS"
        )

    trace_value = payload.get("candidate_trace")
    if type(trace_value) is not list:
        raise TasteGCFSmokeError("Taste T7 raw candidate trace is not a list")
    trace_bytes = _jsonl_bytes(trace_value)
    trace = parse_candidate_trace(trace_bytes)
    summary = payload.get("scientific_summary")
    if type(summary) is not dict:
        raise TasteGCFSmokeError("Taste T7 scientific summary is absent")
    progress = summary.get("progress_checkpoint")
    neurosed = summary.get("neurosed_predecessor")
    if (
        payload.get("candidate_trace_sha256") != _sha256_bytes(trace_bytes)
        or payload.get("candidate_count") != len(trace)
        or payload.get("strict_counterfactual_candidate_count")
        != sum(row["candidate_condition"] for row in trace)
        or payload.get("scientific_summary_sha256")
        != _sha256_bytes(_canonical_bytes(summary))
        or summary.get("schema_version")
        != "tastemolnet_gcf_native_vrrw_smoke_v2"
        or summary.get("native_full_graph_semantics") is not True
        or summary.get("deletion_only_semantics") is not False
        or summary.get("candidate_condition")
        != "pred_candidate != source_label"
        or summary.get("score_definition") != "1.0 - p_source"
        or summary.get("coverage_mode")
        != "official_taste_neurosed_threshold_coverage"
        or summary.get("neurosed_status") != "PASS_INPUT_REVALIDATED"
        or summary.get("distance_status") != "EVALUATED"
        or summary.get("selector_status") != "NOT_EVALUATED"
        or summary.get("train_loaded") is not True
        or any(
            summary.get(field) is not False
            for field in (
                "validation_payload_loaded",
                "calibration_payload_loaded",
                "test_payload_loaded",
                "bace_artifacts_used",
                "rf_oracle_used",
                "native_graph_payload_persisted",
                "molecule_payload_persisted",
                "paper_result_eligible",
            )
        )
        or type(neurosed) is not dict
        or neurosed.get("pass_path") != predecessor["path"]
        or neurosed.get("pass_sha256") != predecessor["sha256"]
        or neurosed.get("calibration_loaded") is not False
        or neurosed.get("test_loaded") is not False
        or neurosed.get("classifier") is not False
        or neurosed.get("role") != "GCF_AUXILIARY_DISTANCE_MODEL"
        or type(progress) is not dict
    ):
        raise TasteGCFSmokeError("Taste T7 scientific raw evidence changed")
    _validate_progress_evidence(progress)
    return {
        "schema_version": "tastemolnet_t7_gcf_independent_verification_v2",
        "status": "PASS",
        "marker": PASS_MARKER,
        "stage": STAGE,
        "attempt_id": expected_attempt_id,
        "generation_token": expected_generation_token,
        "expected_final_path": str(final_path),
        "predecessor": predecessor,
        "raw_evidence_sha256": _sha256_bytes(_canonical_bytes(payload)),
        "candidate_trace_sha256": payload["candidate_trace_sha256"],
        "candidate_count": len(trace),
        "strict_counterfactual_candidate_count": sum(
            row["candidate_condition"] for row in trace
        ),
        "native_full_graph_semantics": True,
        "same_calibrated_three_class_gine": True,
        "taste_neurosed_revalidated": True,
        "independent_verifier": True,
    }


def load_tastemolnet_gcf_verified_gate(
    path: str | Path,
) -> Mapping[str, Any]:
    """Consume only an independently verified managed-v2 final root."""

    from src.utils.managed_execution_v2 import load_verified_gate

    return load_verified_gate(path)


def run_tastemolnet_gcf_smoke(
    *,
    output_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Run T7 after every typed release and predecessor authority closes."""

    from src.utils.tastemolnet_t7_gcf_release import (
        hold_tastemolnet_t7_inputs,
    )
    from src.utils.tastemolnet_t7_managed_v2 import (
        create_t7_managed_worker,
    )

    inputs = hold_tastemolnet_t7_inputs(
        output_dir=output_dir,
        config_path=config_path,
    )
    managed_worker = None
    inputs_closed = False
    try:
        inputs.revalidate()
        managed_worker = create_t7_managed_worker(
            stage_root=inputs.managed_stage_root,
            expected_final_path=inputs.output_root,
            controller_id=inputs.controller["controller_cid"],
            task_id=f"{STAGE}:{inputs.controller['run_id']}",
            git_commit=inputs.execution["execution_commit"],
            config_hash=_sha256_bytes(RELEASE_CONFIG_PATH.read_bytes()),
            input_hashes=inputs.managed_input_hashes,
            neurosed_pass_path=inputs.neurosed_evidence["pass_path"],
            neurosed_pass_sha256=inputs.neurosed_evidence["pass_sha256"],
        )
        inputs.revalidate()
        loaded = load_train_rows(
            inputs.train_bytes,
            source_path=Path(inputs.train_contract["path"]),
            expected_num_records=inputs.train_contract["num_records"],
            expected_label_counts=inputs.train_contract["label_counts"],
        )
        science = execute_native_vrrw_smoke(
            checkpoint_payloads=inputs.checkpoint_payloads,
            source_rows=loaded.sweet_rows[:SMOKE_SOURCE_POOL_LIMIT],
            graph_schema=loaded.schema,
            official_root=Path(inputs.authority["official_root"]),
            neurosed_checkpoint_path=(
                f"/proc/self/fd/{inputs.neurosed_checkpoint.file_fd}"
            ),
            neurosed_distance_threshold=inputs.neurosed_distance_threshold,
            neurosed_evidence=inputs.neurosed_evidence,
            neurosed_revalidate=inputs.revalidate_neurosed,
            device="cuda:0",
        )
        raw_evidence = build_worker_raw_evidence(
            inputs=inputs,
            managed_worker=managed_worker,
            train_evidence=loaded.evidence,
            science=science,
        )
        inputs.revalidate()
        # The worker closes all mutable/scientific authorities before sealing.
        # It never invokes the independent verifier or terminal publisher.
        inputs.close()
        inputs_closed = True
        sealed = managed_worker.seal_raw_evidence(raw_evidence)
        return {
            "status": "SEALED_PENDING_INDEPENDENT_VERIFICATION",
            "stage": STAGE,
            "attempt_id": managed_worker.attempt_id,
            "generation_token": managed_worker.generation_token,
            "staging_path": str(sealed.staging_path),
            "seal_path": str(sealed.seal_path),
            "seal_sha256": sealed.seal_sha256,
            "inventory_sha256": sealed.inventory_sha256,
            "expected_final_path": str(managed_worker.expected_final_path),
        }
    except BaseException as exc:
        if managed_worker is not None:
            managed_worker.record_failure(exc)
        if not inputs_closed:
            inputs.close()
        raise
    finally:
        if managed_worker is not None:
            managed_worker.close()


__all__ = [
    "DATASET",
    "LABEL_MAP",
    "METHOD",
    "NUM_CLASSES",
    "PASS_MARKER",
    "RELEASE_CONFIG_PATH",
    "SOURCE_LABEL",
    "STAGE",
    "TasteGCFGraphSchema",
    "TasteGCFImportanceBridge",
    "TasteFrozenGINENativeAdapter",
    "TasteGCFSmokeError",
    "TasteGCFSmokeReleaseDisabled",
    "build_worker_raw_evidence",
    "execute_native_vrrw_smoke",
    "encode_taste_source_graph",
    "load_tastemolnet_gcf_verified_gate",
    "load_train_rows",
    "parse_candidate_trace",
    "run_tastemolnet_gcf_smoke",
    "score_and_candidate",
    "taste_record_to_pyg",
    "verify_t7_worker_raw_evidence",
]
